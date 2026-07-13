#!/usr/bin/env python3
"""Build a privacy-preserving real-world behavior layer for Driver 1003.

This script intentionally turns only defensible route evidence into plain
language. It uses cached local files: trip-level matched routes, enriched
road-network attributes, and local OSM-derived context. It does *not* call a
map API, scrape map services, publish coordinates/address/POI names, or make
claims that a person lives, works, studies, worships, shops, or receives care
at an identified location.

The output presents recurring activity areas and generic nearby category
context. It explicitly withholds home/work and other sensitive conclusions
when observation timing or privacy safeguards make them inappropriate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point

from build_driver_1003_research_insights import (
    COUNTY_DIRECTORY,
    PRIMARY_COUNTY,
    ROOT,
    TARGET_REPORT,
    dbscan_projected,
    nonempty_text,
    number,
    unique_fid_context,
)


OUTPUT_ROOT = ROOT / "deliverables" / "driver_1003" / "route_choice_change_index"
OUTPUT_DATA_DIR = OUTPUT_ROOT / "data"
TIMELINE_PATH = (
    ROOT
    / "sflorida_outputs"
    / "phase2"
    / "driver_timelines"
    / "driver_1_timeline.csv"
)
RCCI_SUMMARY_PATH = OUTPUT_DATA_DIR / "driver_1003_rcci_summary.csv"

COUNTY_GPS_PATH = {
    "Broward County": ROOT / "sflorida_outputs" / "Broward_County" / "Broward County_gps.csv",
    "Palm Beach County": ROOT / "sflorida_outputs" / "Palm_Beach_County" / "Palm Beach County_gps.csv",
    "Miami-Dade County": ROOT / "sflorida_outputs" / "Miami_Dade_County" / "Miami-Dade County_gps.csv",
}

# Land-use polygons are sparse and may span blocks. A 250 m local-context
# screen is intentionally conservative; it does not establish a visit.
LOCAL_POI_BUFFER_METERS = 250.0
# A 100 m DBSCAN neighborhood limits chaining across broad corridors.  The
# p95-radius screen below removes any residual broad cluster before it can be
# used as an activity area or OD anchor.
BEHAVIOR_CLUSTER_EPS_METERS = 100.0
BEHAVIOR_CLUSTER_MIN_SAMPLES = 8
MAX_CLUSTER_P95_RADIUS_METERS = 250.0
REPORT_CLUSTER_MIN_ENDPOINTS = 20
REPORT_CLUSTER_MIN_MONTHS = 3
COMMON_OD_MIN_TOTAL_TRIPS = 15
COMMON_OD_MIN_MONTHS = 3
COMMON_OD_MIN_MONTHLY_TRIPS = 3
MIN_OD_SEPARATION_METERS = 2_000.0
OD_CHANGE_SCORE_MIN = 55.0
PROMINENT_GLOBAL_RCCI_MIN = 70.0
STABLE_OD_OVERLAP_MIN = 0.75
STABLE_OD_MIN_MONTHLY_TRIPS = 5

BEHAVIOR_BEGIN = "<!-- BEGIN DRIVER 1003 REAL WORLD BEHAVIOR INSIGHTS -->"
BEHAVIOR_END = "<!-- END DRIVER 1003 REAL WORLD BEHAVIOR INSIGHTS -->"
BEHAVIOR_NAV_BEGIN = "<!-- BEGIN DRIVER 1003 REAL WORLD BEHAVIOR INSIGHTS NAV -->"
BEHAVIOR_NAV_END = "<!-- END DRIVER 1003 REAL WORLD BEHAVIOR INSIGHTS NAV -->"
BEHAVIOR_STYLE_MARKER = "/* DRIVER_1003_REAL_WORLD_BEHAVIOR_INSIGHTS_STYLE */"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DATA_DIR)
    parser.add_argument("--report", type=Path, default=TARGET_REPORT)
    parser.add_argument(
        "--skip-report-update",
        action="store_true",
        help="Write CSV/JSON outputs without modifying the HTML report.",
    )
    return parser.parse_args()


def generated_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def percentage(value: object, digits: int = 1) -> str:
    value_float = safe_float(value)
    return "—" if math.isnan(value_float) else f"{value_float * 100:.{digits}f}%"


def confidence_level(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.45:
        return "MEDIUM"
    return "LOW"


def parse_fid_sequence(value: object) -> list[int]:
    text = nonempty_text(value)
    if not text:
        return []
    sequence: list[int] = []
    for token in re.split(r"[|,;\s]+", text):
        try:
            fid = int(token)
        except ValueError:
            continue
        if fid >= 0:
            sequence.append(fid)
    return sequence


def canonical_highway(value: object) -> str:
    text = (nonempty_text(value) or "unknown").lower()
    return text.replace("_link", "_link")


def route_profile(sequence: Sequence[int], context: Mapping[tuple[str, int], Mapping[str, object]], county: str) -> dict[str, float]:
    counts: Counter[str] = Counter()
    named_corridors: Counter[str] = Counter()
    for fid in sequence:
        info = context.get((county, int(fid)), {})
        road_class = canonical_highway(info.get("highway"))
        counts[road_class] += 1
        road_name = nonempty_text(info.get("name"))
        if road_name:
            named_corridors[road_name] += 1
    total = max(sum(counts.values()), 1)
    controlled_classes = {
        "motorway",
        "trunk",
        "motorway_link",
        "trunk_link",
    }
    surface_classes = {
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "living_street",
    }
    local_classes = {"residential", "service", "living_street"}
    connector_classes = {
        "motorway_link",
        "trunk_link",
        "primary_link",
        "secondary_link",
        "tertiary_link",
    }
    return {
        "controlled_access_share": sum(counts[key] for key in controlled_classes) / total,
        "surface_street_share": sum(counts[key] for key in surface_classes) / total,
        "local_access_share": sum(counts[key] for key in local_classes) / total,
        "connector_share": sum(counts[key] for key in connector_classes) / total,
        "route_fid_count": float(total),
        "top_road_class": counts.most_common(1)[0][0] if counts else "unknown",
        # Public corridor labels are not exported in this behavior layer. The
        # count only supports a route-complexity quality check.
        "named_corridor_count": float(len(named_corridors)),
    }


def load_trip_table() -> tuple[pd.DataFrame, dict[tuple[str, int], dict[str, object]]]:
    columns = [
        "trip_id",
        "county",
        "trip_month",
        "trip_start_time",
        "duration_seconds",
        "fid_sequence",
        "route_signature",
        "start_fid",
        "end_fid",
        "matched_trip_id",
    ]
    trips = pd.read_csv(TIMELINE_PATH, usecols=columns)
    trips["start_time"] = pd.to_datetime(
        trips["trip_start_time"], errors="coerce", utc=True
    ).dt.tz_convert("America/New_York")
    trips["duration_seconds"] = pd.to_numeric(
        trips["duration_seconds"], errors="coerce"
    ).clip(lower=0)
    trips["matched_trip_id"] = pd.to_numeric(trips["matched_trip_id"], errors="coerce")
    trips = trips.dropna(
        subset=["trip_id", "county", "start_time", "duration_seconds", "matched_trip_id"]
    )
    trips["matched_trip_id"] = trips["matched_trip_id"].astype("int64")
    trips["end_time"] = trips["start_time"] + pd.to_timedelta(
        trips["duration_seconds"], unit="s"
    )
    trips["trip_date"] = trips["start_time"].dt.date.astype(str)
    trips["weekday_name"] = trips["start_time"].dt.day_name()
    trips["weekday_number"] = trips["start_time"].dt.dayofweek
    trips["is_weekday"] = trips["weekday_number"] < 5
    trips["start_hour"] = trips["start_time"].dt.hour + trips["start_time"].dt.minute / 60
    trips["end_hour"] = trips["end_time"].dt.hour + trips["end_time"].dt.minute / 60
    trips["fid_list"] = trips["fid_sequence"].map(parse_fid_sequence)

    context_frame = unique_fid_context()
    fields = [column for column in ("name", "highway", "landuse", "estimated_speed_limit") if column in context_frame]
    road_context = {
        (str(record.county), int(record.fid)): {
            field: getattr(record, field, None) for field in fields
        }
        for record in context_frame[["county", "fid", *fields]].itertuples(index=False)
    }
    profiles = [
        route_profile(sequence, road_context, str(county))
        for sequence, county in zip(trips["fid_list"], trips["county"], strict=False)
    ]
    profile_frame = pd.DataFrame(profiles, index=trips.index)
    trips = pd.concat([trips, profile_frame], axis=1)
    return trips.reset_index(drop=True), road_context


def friendly_cluster_mapping(clustered_endpoints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assigned = clustered_endpoints.loc[
        clustered_endpoints["cluster_internal"] >= 0
    ].copy()
    if assigned.empty:
        return assigned, pd.DataFrame()
    aggregates = (
        assigned.groupby(["county", "cluster_internal"], as_index=False)
        .agg(
            endpoint_records=("trip_id", "size"),
            months_visited=("trip_month", "nunique"),
            first_month=("trip_month", "min"),
            last_month=("trip_month", "max"),
            centroid_x=("x", "mean"),
            centroid_y=("y", "mean"),
        )
        .sort_values(["county", "endpoint_records"], ascending=[True, False])
        .reset_index(drop=True)
    )
    compactness_rows: list[dict[str, object]] = []
    for (county, cluster_internal), group in assigned.groupby(
        ["county", "cluster_internal"], sort=True
    ):
        dx = group["x"].to_numpy(dtype=float) - float(group["x"].mean())
        dy = group["y"].to_numpy(dtype=float) - float(group["y"].mean())
        radii = np.hypot(dx, dy)
        compactness_rows.append(
            {
                "county": county,
                "cluster_internal": cluster_internal,
                "endpoint_spread_p95_m": float(np.percentile(radii, 95)),
            }
        )
    compactness = pd.DataFrame(compactness_rows)
    aggregates = aggregates.merge(
        compactness,
        on=["county", "cluster_internal"],
        how="left",
        validate="one_to_one",
    )
    aggregates["passes_compactness_screen"] = (
        aggregates["endpoint_spread_p95_m"] <= MAX_CLUSTER_P95_RADIUS_METERS
    )
    maps: list[pd.DataFrame] = []
    for county, group in aggregates.groupby("county", sort=True):
        ranked = group.sort_values("endpoint_records", ascending=False).reset_index(drop=True)
        ranked["area_rank"] = np.arange(1, len(ranked) + 1)
        ranked["activity_area_id"] = [
            f"{str(county).split()[0].lower()}-area-{rank:02d}"
            for rank in ranked["area_rank"]
        ]
        ranked["activity_area_label"] = [
            f"recurring activity area {rank}" for rank in ranked["area_rank"]
        ]
        maps.append(ranked)
    mapping = pd.concat(maps, ignore_index=True)
    result = clustered_endpoints.merge(
        mapping[
            [
                "county",
                "cluster_internal",
                "activity_area_id",
                "activity_area_label",
                "endpoint_spread_p95_m",
                "passes_compactness_screen",
            ]
        ],
        on=["county", "cluster_internal"],
        how="left",
        validate="many_to_one",
    )
    result["activity_area_id"] = result["activity_area_id"].fillna("unclustered")
    result["activity_area_label"] = result["activity_area_label"].fillna("unclustered endpoint")
    noncompact = result["passes_compactness_screen"].eq(False)
    result.loc[noncompact, "activity_area_id"] = "unclustered"
    result.loc[noncompact, "activity_area_label"] = "unclustered endpoint"
    return result, mapping


def build_raw_gps_endpoints(trips: pd.DataFrame) -> pd.DataFrame:
    """Return first/last cached GPS points joined to Driver 1003 trips.

    These are observation endpoints, not verified real-world destinations. Raw
    coordinates improve spatial clustering relative to a road-network node,
    but are retained only in memory and never exported.
    """
    road_context = unique_fid_context()
    context_columns = [
        column
        for column in ("county", "fid", "name", "highway", "landuse", "estimated_speed_limit")
        if column in road_context
    ]
    parts: list[pd.DataFrame] = []
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:26917", always_xy=True)
    for county, county_trips in trips.groupby("county", sort=True):
        gps_path = COUNTY_GPS_PATH.get(str(county))
        if not gps_path or not gps_path.exists():
            continue
        ids = set(pd.to_numeric(county_trips["matched_trip_id"], errors="coerce").dropna().astype(int))
        if not ids:
            continue
        first_parts: list[pd.DataFrame] = []
        last_parts: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            gps_path,
            sep=";",
            usecols=["id", "lon", "lat", "timestamp", "point_idx"],
            chunksize=250_000,
        ):
            filtered = chunk.loc[chunk["id"].isin(ids)].copy()
            if filtered.empty:
                continue
            filtered = filtered.sort_values(["id", "point_idx"])
            first_parts.append(filtered.drop_duplicates("id", keep="first"))
            last_parts.append(filtered.drop_duplicates("id", keep="last"))
        if not first_parts or not last_parts:
            continue
        starts = (
            pd.concat(first_parts, ignore_index=True)
            .sort_values(["id", "point_idx"])
            .drop_duplicates("id", keep="first")
        )
        ends = (
            pd.concat(last_parts, ignore_index=True)
            .sort_values(["id", "point_idx"])
            .drop_duplicates("id", keep="last")
        )
        trip_lookup = county_trips.set_index("matched_trip_id")[
            ["trip_id", "county", "trip_month", "start_fid", "end_fid"]
        ]
        for role, points, fid_column in (
            ("start", starts, "start_fid"),
            ("end", ends, "end_fid"),
        ):
            endpoint = points.join(trip_lookup, on="id", how="inner")
            endpoint = endpoint.rename(columns={fid_column: "endpoint_fid"})
            endpoint["endpoint_role"] = role
            endpoint["endpoint_fid"] = pd.to_numeric(endpoint["endpoint_fid"], errors="coerce")
            endpoint = endpoint.dropna(subset=["endpoint_fid", "lon", "lat"])
            endpoint["endpoint_fid"] = endpoint["endpoint_fid"].astype("int64")
            endpoint["x"], endpoint["y"] = transformer.transform(
                endpoint["lon"].to_numpy(dtype=float), endpoint["lat"].to_numpy(dtype=float)
            )
            parts.append(
                endpoint[
                    [
                        "trip_id",
                        "county",
                        "trip_month",
                        "endpoint_role",
                        "endpoint_fid",
                        "x",
                        "y",
                    ]
                ]
            )
    if not parts:
        raise RuntimeError("No raw GPS endpoints could be joined to Driver 1003 trips")
    endpoints = pd.concat(parts, ignore_index=True)
    endpoint_context = road_context[context_columns].rename(columns={"fid": "endpoint_fid", "name": "road_name"})
    endpoints = endpoints.merge(
        endpoint_context,
        on=["county", "endpoint_fid"],
        how="left",
        validate="many_to_one",
    )
    return endpoints


def assign_behavior_endpoint_clusters(endpoints: pd.DataFrame) -> pd.DataFrame:
    """Apply the 100 m / 8-sample DBSCAN screen before a compactness check."""
    parts: list[pd.DataFrame] = []
    for county, group in endpoints.groupby("county", sort=True):
        part = group.copy().reset_index(drop=True)
        if len(part) < BEHAVIOR_CLUSTER_MIN_SAMPLES:
            part["cluster_internal"] = -1
        else:
            part["cluster_internal"] = dbscan_projected(
                part["x"].to_numpy(dtype=float),
                part["y"].to_numpy(dtype=float),
                eps=BEHAVIOR_CLUSTER_EPS_METERS,
                min_samples=BEHAVIOR_CLUSTER_MIN_SAMPLES,
            )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def attach_trip_activity_areas(trips: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoints = build_raw_gps_endpoints(trips)
    clustered = assign_behavior_endpoint_clusters(endpoints)
    mapped, mapping = friendly_cluster_mapping(clustered)
    endpoint_cols = ["trip_id", "endpoint_role", "activity_area_id", "activity_area_label"]
    starts = mapped.loc[mapped["endpoint_role"] == "start", endpoint_cols].rename(
        columns={
            "activity_area_id": "origin_area_id",
            "activity_area_label": "origin_area_label",
        }
    ).drop(columns=["endpoint_role"])
    ends = mapped.loc[mapped["endpoint_role"] == "end", endpoint_cols].rename(
        columns={
            "activity_area_id": "destination_area_id",
            "activity_area_label": "destination_area_label",
        }
    ).drop(columns=["endpoint_role"])
    output = trips.merge(starts, on="trip_id", how="left", validate="one_to_one")
    output = output.merge(ends, on="trip_id", how="left", validate="one_to_one")
    for column in (
        "origin_area_id",
        "origin_area_label",
        "destination_area_id",
        "destination_area_label",
    ):
        output[column] = output[column].fillna("unclustered")
    return output, mapped


def generic_poi_categories(nearby: gpd.GeoDataFrame) -> list[str]:
    """Classify local OSM tags into generic categories, never names."""
    found: set[str] = set()
    for _, row in nearby.iterrows():
        landuse = (nonempty_text(row.get("landuse")) or "").lower()
        amenity = (nonempty_text(row.get("amenity")) or "").lower()
        shop = (nonempty_text(row.get("shop")) or "").lower()
        office = nonempty_text(row.get("office"))
        aeroway = nonempty_text(row.get("aeroway"))
        leisure = (nonempty_text(row.get("leisure")) or "").lower()
        sport = nonempty_text(row.get("sport"))
        tourism = nonempty_text(row.get("tourism"))
        if amenity in {"hospital", "clinic", "doctors", "dentist", "pharmacy"} or shop == "chemist":
            found.add("healthcare-related category proximity")
        if landuse == "education" or amenity in {"school", "college", "university", "kindergarten", "childcare"}:
            found.add("educational category proximity")
        if amenity == "place_of_worship":
            found.add("place-of-worship category proximity")
        if amenity in {"restaurant", "cafe", "fast_food", "bar", "pub", "food_court"}:
            found.add("restaurant/food category proximity")
        if amenity in {"fuel", "bank", "atm", "car_wash"}:
            found.add("errand-service category proximity")
        if shop or landuse == "retail":
            found.add("shopping/retail category proximity")
        if office or landuse == "commercial":
            found.add("office/commercial category proximity")
        if aeroway in {"aerodrome", "terminal", "apron", "runway", "taxiway", "hangar", "helipad"}:
            found.add("airport category proximity")
        if leisure in {"park", "fitness_centre", "sports_centre", "stadium", "pitch", "playground"} or sport:
            found.add("recreation/fitness category proximity")
        if tourism:
            found.add("visitor/leisure category proximity")
        if landuse == "residential":
            found.add("residential land-use context")
        if landuse == "industrial":
            found.add("industrial land-use context")
    order = [
        "healthcare-related category proximity",
        "educational category proximity",
        "place-of-worship category proximity",
        "restaurant/food category proximity",
        "errand-service category proximity",
        "shopping/retail category proximity",
        "office/commercial category proximity",
        "airport category proximity",
        "recreation/fitness category proximity",
        "visitor/leisure category proximity",
        "residential land-use context",
        "industrial land-use context",
    ]
    return [category for category in order if category in found]


def local_context_for_clusters(cluster_mapping: pd.DataFrame) -> dict[str, str]:
    contexts: dict[str, str] = {}
    for county, group in cluster_mapping.groupby("county", sort=True):
        directory = COUNTY_DIRECTORY.get(str(county))
        poi_path = ROOT / "sflorida_outputs" / str(directory) / "osm_landuse.parquet"
        if not directory or not poi_path.exists():
            for record in group.itertuples(index=False):
                contexts[record.activity_area_id] = "unknown area"
            continue
        pois = gpd.read_parquet(poi_path)
        if pois.empty or pois.crs is None:
            for record in group.itertuples(index=False):
                contexts[record.activity_area_id] = "unknown area"
            continue
        if pois.crs.to_epsg() != 26917:
            pois = pois.to_crs("EPSG:26917")
        for record in group.itertuples(index=False):
            point = Point(float(record.centroid_x), float(record.centroid_y))
            nearby = pois.loc[pois.geometry.distance(point) <= LOCAL_POI_BUFFER_METERS]
            categories = generic_poi_categories(nearby)
            contexts[record.activity_area_id] = "; ".join(categories) if categories else "unknown area"
    return contexts


def median_hour(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else float("nan")


def build_inter_trip_interval_proxy(trips: pd.DataFrame) -> pd.DataFrame:
    """Return an observed same-area inter-trip interval, never a dwell estimate.

    A GPS trip ending in an area and the next logged trip beginning there does
    not establish that the traveler remained there.  This intentionally keeps
    the measure as a collection-window / inter-trip interval proxy.
    """
    ordered = trips.sort_values("start_time").reset_index(drop=True).copy()
    ordered["next_start_time"] = ordered["start_time"].shift(-1)
    ordered["next_origin_area_id"] = ordered["origin_area_id"].shift(-1)
    ordered["next_county"] = ordered["county"].shift(-1)
    ordered["gap_minutes"] = (
        (ordered["next_start_time"] - ordered["end_time"]).dt.total_seconds() / 60
    )
    eligible = ordered.loc[
        (ordered["destination_area_id"] != "unclustered")
        & (ordered["destination_area_id"] == ordered["next_origin_area_id"])
        & (ordered["county"] == ordered["next_county"])
        & (ordered["end_time"].dt.date == ordered["next_start_time"].dt.date)
        & ordered["gap_minutes"].between(10, 720, inclusive="both")
    ]
    return eligible[["destination_area_id", "gap_minutes"]].rename(
        columns={
            "destination_area_id": "activity_area_id",
            "gap_minutes": "inter_trip_gap_minutes",
        }
    )


def temporal_pattern(row: Mapping[str, object]) -> tuple[str, float, str]:
    """Return a generic pattern only; do not identify home/work/personal facts."""
    categories = str(row.get("poi_category_context", "unknown area"))
    endpoint_records = safe_float(row.get("endpoint_records"), 0)
    months = safe_float(row.get("months_visited"), 0)
    weekday_share = safe_float(row.get("weekday_endpoint_share"), 0)
    morning_arrivals = safe_float(row.get("weekday_morning_arrivals"), 0)
    afternoon_departures = safe_float(row.get("weekday_afternoon_departures"), 0)
    evening_weekend = safe_float(row.get("evening_or_weekend_endpoints"), 0)
    daytime_pattern = min(morning_arrivals, afternoon_departures)
    recurrence = min(endpoint_records / 120, 1.0) * 0.45 + min(months / 12, 1.0) * 0.30
    if "educational category proximity" in categories and morning_arrivals >= 5:
        score = min(recurrence + 0.20 + min(morning_arrivals / 25, 0.15), 1.0)
        return (
            "possible educational-area proximity pattern",
            score,
            "Recurring weekday arrivals occur near an educational category, but this does not establish a school/daycare visit or a child-related trip.",
        )
    if "healthcare-related category proximity" in categories and endpoint_records >= 5:
        score = min(recurrence + 0.15, 1.0)
        return (
            "possible healthcare-area proximity pattern",
            score,
            "Repeated endpoint activity occurs near a healthcare-related category; it does not establish a visit, treatment, or health condition.",
        )
    if "shopping/retail category proximity" in categories or "errand-service category proximity" in categories:
        score = min(recurrence + 0.10, 1.0)
        return (
            "recurring retail/errand-area proximity pattern",
            score,
            "Repeated endpoint activity occurs near retail or errand-service categories; the purpose of the trips is not confirmed.",
        )
    if "recreation/fitness category proximity" in categories or "restaurant/food category proximity" in categories:
        score = min(recurrence + 0.08 + min(evening_weekend / 40, 0.12), 1.0)
        return (
            "possible evening/weekend discretionary-area proximity pattern",
            score,
            "The time and generic category context may be consistent with discretionary activity, but no visit or preference is confirmed.",
        )
    if daytime_pattern >= 8 and weekday_share >= 0.60:
        score = min(recurrence + 0.08 + min(daytime_pattern / 35, 0.12), 1.0)
        return (
            "recurring weekday daytime activity pattern",
            score,
            "A repeated weekday arrival/departure pattern is present, but the data do not establish a workplace or trip purpose.",
        )
    if "residential land-use context" in categories:
        score = min(recurrence + 0.05, 1.0)
        return (
            "recurring private residential-context activity area",
            score,
            "This is a recurring activity area near residential context, not a home determination.",
        )
    score = min(recurrence, 1.0)
    return (
        "recurring activity area with unknown context",
        score,
        "The route data show recurrence, but available local context is insufficient to infer a place type or trip purpose.",
    )


def local_context_confidence(categories: str) -> float:
    """Score the local-only category context separately from recurrence.

    This is deliberately capped below high confidence: cached OSM land-use
    polygons inside a 250 m screen can support only generic *proximity*
    context, not a verified destination or visit.
    """
    if categories == "unknown area":
        return 0.15
    return 0.40


def interpretive_confidence(
    *, pattern_label: str, recurrence_score: float, context_score: float
) -> float:
    """Keep POI-type interpretations conditional on the weak local context."""
    if "proximity" in pattern_label:
        return min(recurrence_score, context_score)
    # Generic recurrence labels do not infer a POI type or personal fact.
    return recurrence_score


def build_location_clusters(trips: pd.DataFrame, endpoints: pd.DataFrame) -> pd.DataFrame:
    endpoint_time = endpoints.merge(
        trips[
            [
                "trip_id",
                "trip_month",
                "trip_date",
                "is_weekday",
                "start_hour",
                "end_hour",
            ]
        ],
        on=["trip_id", "trip_month"],
        how="left",
        validate="many_to_one",
    )
    reportable = endpoint_time.loc[
        endpoint_time["activity_area_id"] != "unclustered"
    ].copy()
    intervals = build_inter_trip_interval_proxy(trips)
    interval_stats = (
        intervals.groupby("activity_area_id", as_index=False)
        .agg(
            inter_trip_interval_observations=("inter_trip_gap_minutes", "size"),
            median_inter_trip_interval_minutes=("inter_trip_gap_minutes", "median"),
        )
        if not intervals.empty
        else pd.DataFrame(
            columns=[
                "activity_area_id",
                "inter_trip_interval_observations",
                "median_inter_trip_interval_minutes",
            ]
        )
    )
    grouped_rows: list[dict[str, object]] = []
    for (county, area_id), group in reportable.groupby(["county", "activity_area_id"], sort=True):
        endpoint_count = len(group)
        months = int(group["trip_month"].nunique())
        if endpoint_count < REPORT_CLUSTER_MIN_ENDPOINTS or months < REPORT_CLUSTER_MIN_MONTHS:
            continue
        starts = group.loc[group["endpoint_role"] == "start"]
        ends = group.loc[group["endpoint_role"] == "end"]
        road_classes = (
            group.assign(highway=group["highway"].map(nonempty_text).fillna("road"))
            .groupby("highway")
            .size()
            .sort_values(ascending=False)
        )
        road_context = "; ".join(
            f"{str(road).replace('_', ' ')}-road approaches ({count / endpoint_count:.1%})"
            for road, count in road_classes.head(2).items()
        )
        weekday_share = float(group["is_weekday"].mean())
        weekdays = int(group["is_weekday"].sum())
        weekend = int((~group["is_weekday"]).sum())
        morning_arrivals = int(
            ((ends["is_weekday"]) & ends["end_hour"].between(6, 10.5, inclusive="both")).sum()
        )
        afternoon_departures = int(
            ((starts["is_weekday"]) & starts["start_hour"].between(14.5, 19.5, inclusive="both")).sum()
        )
        evening_or_weekend = int(
            ((group["endpoint_role"] == "end") & (group["end_hour"] >= 19)).sum()
            + (~group["is_weekday"]).sum()
        )
        grouped_rows.append(
            {
                "activity_area_id": area_id,
                "activity_area_label": str(group["activity_area_label"].iloc[0]),
                "county": county,
                "endpoint_records": endpoint_count,
                "origin_records": int(len(starts)),
                "destination_records": int(len(ends)),
                "unique_trips": int(group["trip_id"].nunique()),
                "unique_days_visited": int(group["trip_date"].nunique()),
                "months_visited": months,
                "first_month": str(group["trip_month"].min()),
                "last_month": str(group["trip_month"].max()),
                "endpoint_spread_p95_m": float(group["endpoint_spread_p95_m"].iloc[0]),
                "cluster_quality": "compact endpoint cluster (p95 radius ≤ 250 m)",
                "weekday_endpoint_records": weekdays,
                "weekend_endpoint_records": weekend,
                "weekday_endpoint_share": weekday_share,
                "typical_arrival_hour": median_hour(ends["end_hour"]),
                "typical_departure_hour": median_hour(starts["start_hour"]),
                "weekday_morning_arrivals": morning_arrivals,
                "weekday_afternoon_departures": afternoon_departures,
                "evening_or_weekend_endpoints": evening_or_weekend,
                "road_context": road_context or "road context unavailable",
                "centroid_x": float(group["x"].mean()),
                "centroid_y": float(group["y"].mean()),
            }
        )
    summary = pd.DataFrame(grouped_rows)
    if summary.empty:
        return summary
    # Reuse the same internal mapping but make local POI context generic and
    # safe. Coordinates are used only in-memory and removed before export.
    mapping = summary[["activity_area_id", "county", "centroid_x", "centroid_y"]].copy()
    mapping["cluster_internal"] = 0
    poi_context = local_context_for_clusters(mapping)
    summary["poi_category_context"] = summary["activity_area_id"].map(poi_context).fillna("unknown area")
    summary["poi_source"] = (
        f"Local OSM-derived category tags within {int(LOCAL_POI_BUFFER_METERS)} m; "
        "no external map/POI lookup or reverse geocoding"
    )
    summary = summary.merge(interval_stats, on="activity_area_id", how="left")
    summary["inter_trip_interval_observations"] = (
        summary["inter_trip_interval_observations"].fillna(0).astype(int)
    )
    for column in ("median_inter_trip_interval_minutes",):
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
    patterns = [temporal_pattern(row) for row in summary.to_dict(orient="records")]
    summary["behavior_pattern"] = [pattern[0] for pattern in patterns]
    summary["recurrence_confidence_score"] = [round(pattern[1], 3) for pattern in patterns]
    summary["recurrence_confidence_level"] = [confidence_level(pattern[1]) for pattern in patterns]
    summary["context_confidence_score"] = [
        round(local_context_confidence(str(value)), 3)
        for value in summary["poi_category_context"]
    ]
    summary["context_confidence_level"] = [
        confidence_level(score) for score in summary["context_confidence_score"]
    ]
    summary["interpretive_confidence_score"] = [
        round(
            interpretive_confidence(
                pattern_label=pattern[0],
                recurrence_score=pattern[1],
                context_score=context_score,
            ),
            3,
        )
        for pattern, context_score in zip(
            patterns, summary["context_confidence_score"], strict=False
        )
    ]
    summary["interpretive_confidence_level"] = [
        confidence_level(score) for score in summary["interpretive_confidence_score"]
    ]
    summary["careful_interpretation"] = [pattern[2] for pattern in patterns]
    summary["privacy_note"] = (
        "No exact coordinate, address, POI name, home/work, school, healthcare, religion, family, or visit claim is reported."
    )
    return summary.drop(columns=["centroid_x", "centroid_y"]).sort_values(
        ["county", "endpoint_records"], ascending=[True, False]
    ).reset_index(drop=True)


def weighted_jaccard(left: Counter[int], right: Counter[int]) -> float:
    keys = set(left) | set(right)
    denominator = sum(max(left.get(key, 0), right.get(key, 0)) for key in keys)
    if not denominator:
        return float("nan")
    numerator = sum(min(left.get(key, 0), right.get(key, 0)) for key in keys)
    return numerator / denominator


def route_counter(frame: pd.DataFrame) -> Counter[int]:
    counter: Counter[int] = Counter()
    for sequence in frame["fid_list"]:
        counter.update(sequence)
    return counter


def od_month_profile(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "trip_count": float(len(frame)),
        "controlled_access_share": float(frame["controlled_access_share"].mean()),
        "surface_street_share": float(frame["surface_street_share"].mean()),
        "local_access_share": float(frame["local_access_share"].mean()),
        "connector_share": float(frame["connector_share"].mean()),
        "route_signature_concentration": float(
            frame["route_signature"].value_counts(normalize=True).iloc[0]
        )
        if not frame.empty
        else float("nan"),
    }


def route_change_label(a: Mapping[str, float], b: Mapping[str, float]) -> tuple[str, str]:
    controlled_delta = b["controlled_access_share"] - a["controlled_access_share"]
    surface_delta = b["surface_street_share"] - a["surface_street_share"]
    connector_delta = b["connector_share"] - a["connector_share"]
    consistency_delta = b["route_signature_concentration"] - a["route_signature_concentration"]
    if controlled_delta <= -0.12 and surface_delta >= 0.12:
        change = "route became more surface-street oriented"
    elif controlled_delta >= 0.12 and surface_delta <= -0.12:
        change = "route became more controlled-access/highway oriented"
    elif abs(connector_delta) >= 0.10:
        change = "connector/ramp use changed materially"
    elif (a["surface_street_share"] + b["surface_street_share"]) / 2 >= 0.75:
        change = "same broad area pair used a different surface-street mix"
    elif (a["controlled_access_share"] + b["controlled_access_share"]) / 2 >= 0.75:
        change = "same broad area pair used a different controlled-access corridor mix"
    else:
        change = "same broad area pair used a different road-corridor mix"
    if consistency_delta >= 0.15:
        consistency = "route choice became more consistent"
    elif consistency_delta <= -0.15:
        consistency = "route choice became more variable"
    else:
        consistency = "route consistency remained broadly similar"
    return change, consistency


def rcci_lookup() -> dict[tuple[str, str], dict[str, object]]:
    if not RCCI_SUMMARY_PATH.exists():
        return {}
    rcci = pd.read_csv(RCCI_SUMMARY_PATH)
    rcci = rcci.loc[rcci["county"] == PRIMARY_COUNTY].copy()
    return {
        (str(row.month_a), str(row.month_b)): {
            "global_rcci": safe_float(row.rcci_v1),
            "global_confidence": str(row.confidence_label),
        }
        for row in rcci.itertuples(index=False)
    }


def compact_activity_area_geometry(endpoints: pd.DataFrame) -> pd.DataFrame:
    """Return in-memory only centroids used to screen comparable OD pairs."""
    data = endpoints.loc[
        (endpoints["activity_area_id"] != "unclustered")
        & endpoints["passes_compactness_screen"].eq(True)
    ].copy()
    if data.empty:
        return pd.DataFrame(columns=["county", "activity_area_id", "x", "y"])
    return (
        data.groupby(["county", "activity_area_id"], as_index=False)
        .agg(x=("x", "mean"), y=("y", "mean"))
    )


def build_od_route_changes(trips: pd.DataFrame, endpoints: pd.DataFrame) -> pd.DataFrame:
    data = trips.loc[
        (trips["county"] == PRIMARY_COUNTY)
        & (trips["origin_area_id"] != "unclustered")
        & (trips["destination_area_id"] != "unclustered")
    ].copy()
    if data.empty:
        return pd.DataFrame()
    geometry = compact_activity_area_geometry(endpoints)
    geometry_lookup = {
        str(record.activity_area_id): (float(record.x), float(record.y))
        for record in geometry.loc[geometry["county"] == PRIMARY_COUNTY].itertuples(index=False)
    }
    rcci_by_pair = rcci_lookup()
    rows: list[dict[str, object]] = []
    od_groups = data.groupby(["origin_area_id", "destination_area_id"], sort=True)
    for (origin_id, destination_id), od in od_groups:
        origin_xy = geometry_lookup.get(str(origin_id))
        destination_xy = geometry_lookup.get(str(destination_id))
        if origin_xy is None or destination_xy is None:
            continue
        separation_m = math.dist(origin_xy, destination_xy)
        if separation_m < MIN_OD_SEPARATION_METERS:
            continue
        if len(od) < COMMON_OD_MIN_TOTAL_TRIPS or od["trip_month"].nunique() < COMMON_OD_MIN_MONTHS:
            continue
        monthly = {
            str(month): group.copy()
            for month, group in od.groupby("trip_month", sort=True)
            if len(group) >= COMMON_OD_MIN_MONTHLY_TRIPS
        }
        month_list = sorted(monthly)
        for month_a, month_b in zip(month_list, month_list[1:], strict=False):
            try:
                consecutive = pd.Period(month_b, freq="M") == pd.Period(month_a, freq="M") + 1
            except (TypeError, ValueError):
                consecutive = False
            if not consecutive:
                continue
            group_a = monthly[month_a]
            group_b = monthly[month_b]
            counts_a = route_counter(group_a)
            counts_b = route_counter(group_b)
            overlap = weighted_jaccard(counts_a, counts_b)
            if math.isnan(overlap):
                continue
            score = 100 * (1 - overlap)
            if score < OD_CHANGE_SCORE_MIN:
                continue
            profile_a = od_month_profile(group_a)
            profile_b = od_month_profile(group_b)
            route_change, consistency = route_change_label(profile_a, profile_b)
            lookup = rcci_by_pair.get((month_a, month_b), {})
            min_monthly_trips = min(len(group_a), len(group_b))
            sample_quality = min(min_monthly_trips / 5, 1.0)
            recurrence = min(od["trip_month"].nunique() / 12, 1.0)
            global_rcci = safe_float(lookup.get("global_rcci"))
            global_confidence = str(lookup.get("global_confidence", "not available"))
            rcci_support = (
                1.0
                if not math.isnan(global_rcci)
                and global_rcci >= PROMINENT_GLOBAL_RCCI_MIN
                and global_confidence in {"HIGH", "MEDIUM"}
                else 0.45
            )
            score_confidence = min(
                0.30 + sample_quality * 0.40 + recurrence * 0.15 + rcci_support * 0.15,
                1.0,
            )
            # Three/four-trip comparisons remain meaningful candidates, but
            # cannot receive high-confidence narrative treatment.
            if min_monthly_trips < 5 or rcci_support < 1.0:
                score_confidence = min(score_confidence, 0.70)
            report_priority = (
                "prominent"
                if rcci_support == 1.0 and min_monthly_trips >= COMMON_OD_MIN_MONTHLY_TRIPS
                else "supporting"
            )
            evidence = (
                f"Trips continued between the same privacy-safe recurring activity areas, while "
                f"frequency-weighted matched-segment overlap was {overlap:.1%}. The {route_change}; "
                f"{consistency}."
            )
            if not math.isnan(global_rcci):
                evidence += f" Overall Broward RCCI for this month pair was {global_rcci:.1f}."
            rows.append(
                {
                    "od_change_id": "",
                    "county": PRIMARY_COUNTY,
                    "origin_activity_area": str(group_a["origin_area_label"].iloc[0]),
                    "destination_activity_area": str(group_a["destination_area_label"].iloc[0]),
                    "month_a": month_a,
                    "month_b": month_b,
                    "trips_a": int(len(group_a)),
                    "trips_b": int(len(group_b)),
                    "od_pair_total_trips": int(len(od)),
                    "od_pair_months": int(od["trip_month"].nunique()),
                    "approximate_od_separation_screen": f"passed ≥{MIN_OD_SEPARATION_METERS / 1000:.0f} km screen",
                    "matched_segment_overlap_pct": overlap * 100,
                    "od_route_change_score": score,
                    "controlled_access_share_a_pct": profile_a["controlled_access_share"] * 100,
                    "controlled_access_share_b_pct": profile_b["controlled_access_share"] * 100,
                    "surface_street_share_a_pct": profile_a["surface_street_share"] * 100,
                    "surface_street_share_b_pct": profile_b["surface_street_share"] * 100,
                    "connector_share_a_pct": profile_a["connector_share"] * 100,
                    "connector_share_b_pct": profile_b["connector_share"] * 100,
                    "route_signature_concentration_a": profile_a["route_signature_concentration"],
                    "route_signature_concentration_b": profile_b["route_signature_concentration"],
                    "route_change_interpretation": route_change,
                    "consistency_interpretation": consistency,
                    "global_rcci": global_rcci,
                    "global_rcci_confidence": global_confidence,
                    "report_priority": report_priority,
                    "confidence_score": round(score_confidence, 3),
                    "confidence_level": confidence_level(score_confidence),
                    "careful_interpretation": evidence,
                    "privacy_note": "Activity-area IDs are generic; no address, destination name, or trip purpose is inferred.",
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(
        ["report_priority", "confidence_score", "od_route_change_score", "trips_a", "trips_b"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)
    result["od_change_id"] = [f"OD-{index:02d}" for index in range(1, len(result) + 1)]
    return result


def build_od_stability_insights(trips: pd.DataFrame, endpoints: pd.DataFrame) -> pd.DataFrame:
    """Find well-sampled repeated OD-area movements with stable route use.

    This is a deliberately narrow counterpart to the route-change screen: it
    is evidence of a stable *route pattern*, not a commute, home/work pair, or
    trip purpose.
    """
    data = trips.loc[
        (trips["county"] == PRIMARY_COUNTY)
        & (trips["origin_area_id"] != "unclustered")
        & (trips["destination_area_id"] != "unclustered")
    ].copy()
    if data.empty:
        return pd.DataFrame()
    geometry = compact_activity_area_geometry(endpoints)
    geometry_lookup = {
        str(record.activity_area_id): (float(record.x), float(record.y))
        for record in geometry.loc[geometry["county"] == PRIMARY_COUNTY].itertuples(index=False)
    }
    rcci_by_pair = rcci_lookup()
    rows: list[dict[str, object]] = []
    for (origin_id, destination_id), od in data.groupby(
        ["origin_area_id", "destination_area_id"], sort=True
    ):
        origin_xy = geometry_lookup.get(str(origin_id))
        destination_xy = geometry_lookup.get(str(destination_id))
        if origin_xy is None or destination_xy is None:
            continue
        if math.dist(origin_xy, destination_xy) < MIN_OD_SEPARATION_METERS:
            continue
        if len(od) < COMMON_OD_MIN_TOTAL_TRIPS or od["trip_month"].nunique() < COMMON_OD_MIN_MONTHS:
            continue
        monthly = {
            str(month): group.copy()
            for month, group in od.groupby("trip_month", sort=True)
            if len(group) >= STABLE_OD_MIN_MONTHLY_TRIPS
        }
        months = sorted(monthly)
        for month_a, month_b in zip(months, months[1:], strict=False):
            try:
                consecutive = pd.Period(month_b, freq="M") == pd.Period(month_a, freq="M") + 1
            except (TypeError, ValueError):
                consecutive = False
            if not consecutive:
                continue
            group_a = monthly[month_a]
            group_b = monthly[month_b]
            overlap = weighted_jaccard(route_counter(group_a), route_counter(group_b))
            if math.isnan(overlap) or overlap < STABLE_OD_OVERLAP_MIN:
                continue
            profile_a = od_month_profile(group_a)
            profile_b = od_month_profile(group_b)
            _, consistency = route_change_label(profile_a, profile_b)
            lookup = rcci_by_pair.get((month_a, month_b), {})
            global_rcci = safe_float(lookup.get("global_rcci"))
            min_monthly_trips = min(len(group_a), len(group_b))
            confidence = min(
                0.35
                + min(min_monthly_trips / STABLE_OD_MIN_MONTHLY_TRIPS, 1.0) * 0.35
                + min(od["trip_month"].nunique() / 12, 1.0) * 0.30,
                1.0,
            )
            rows.append(
                {
                    "stable_od_id": "",
                    "county": PRIMARY_COUNTY,
                    "origin_activity_area": str(group_a["origin_area_label"].iloc[0]),
                    "destination_activity_area": str(group_a["destination_area_label"].iloc[0]),
                    "month_a": month_a,
                    "month_b": month_b,
                    "trips_a": int(len(group_a)),
                    "trips_b": int(len(group_b)),
                    "od_pair_total_trips": int(len(od)),
                    "od_pair_months": int(od["trip_month"].nunique()),
                    "approximate_od_separation_screen": f"passed ≥{MIN_OD_SEPARATION_METERS / 1000:.0f} km screen",
                    "matched_segment_overlap_pct": overlap * 100,
                    "route_consistency_interpretation": consistency,
                    "global_rcci": global_rcci,
                    "global_rcci_confidence": str(lookup.get("global_confidence", "not available")),
                    "confidence_score": round(confidence, 3),
                    "confidence_level": confidence_level(confidence),
                    "careful_interpretation": (
                        "A repeated trip between the same privacy-safe activity areas retained "
                        f"{overlap:.1%} frequency-weighted matched-segment overlap across consecutive months. "
                        "This is consistent with a stable repeated route pattern; trip purpose and cause cannot be determined."
                    ),
                    "privacy_note": "Activity-area IDs are generic; no address, destination name, or trip purpose is inferred.",
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(
        ["confidence_score", "matched_segment_overlap_pct", "trips_a", "trips_b"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    result["stable_od_id"] = [f"STABLE-{index:02d}" for index in range(1, len(result) + 1)]
    return result


def output_safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    records: list[dict[str, Any]] = []
    for record in frame.replace({np.nan: None}).to_dict(orient="records"):
        safe: dict[str, Any] = {}
        for key, value in record.items():
            if key.lower() in {"centroid_x", "centroid_y", "latitude", "longitude", "x", "y"}:
                continue
            if isinstance(value, (np.integer,)):
                safe[key] = int(value)
            elif isinstance(value, (np.floating,)):
                safe[key] = float(value)
            else:
                safe[key] = value
        records.append(safe)
    return records


def behavior_json(
    *,
    trips: pd.DataFrame,
    clusters: pd.DataFrame,
    od_changes: pd.DataFrame,
    od_stability: pd.DataFrame,
) -> dict[str, Any]:
    broward = trips.loc[trips["county"] == PRIMARY_COUNTY]
    early_morning = int(broward["start_hour"].between(4, 8.99, inclusive="both").sum())
    weekday_commute_start = int(
        (broward["is_weekday"] & broward["start_hour"].between(6, 10, inclusive="both")).sum()
    )
    weekday_commute_end = int(
        (broward["is_weekday"] & broward["end_hour"].between(6, 10, inclusive="both")).sum()
    )
    location_findings = output_safe_records(clusters)
    route_findings = output_safe_records(od_changes)
    stable_route_findings = output_safe_records(od_stability)
    broward_cluster_count = int((clusters["county"] == PRIMARY_COUNTY).sum()) if not clusters.empty else 0
    broward_clusters = (
        clusters.loc[clusters["county"] == PRIMARY_COUNTY].copy()
        if not clusters.empty
        else clusters.copy()
    )
    clustered_endpoint_total = int(broward_clusters["endpoint_records"].sum()) if not broward_clusters.empty else 0
    top_three_endpoint_share = (
        float(broward_clusters.nlargest(3, "endpoint_records")["endpoint_records"].sum())
        / clustered_endpoint_total
        if clustered_endpoint_total
        else 0.0
    )
    education_context_count = int(
        broward_clusters["poi_category_context"].str.contains("educational category", na=False).sum()
    ) if not broward_clusters.empty else 0
    healthcare_context_count = int(
        broward_clusters["poi_category_context"].str.contains("healthcare-related category", na=False).sum()
    ) if not broward_clusters.empty else 0
    retail_context_count = int(
        broward_clusters["poi_category_context"].str.contains(
            "shopping/retail category|errand-service category", regex=True, na=False
        ).sum()
    ) if not broward_clusters.empty else 0
    prominent_route_count = (
        int(od_changes["report_priority"].eq("prominent").sum())
        if not od_changes.empty
        else 0
    )
    return {
        "generated_at": generated_at(),
        "driver_scope": "Driver 1003 / pseudonymous internal trip data",
        "method": {
            "endpoint_clustering": f"County-specific DBSCAN on first/last cached GPS endpoints transformed to EPSG:26917; epsilon {int(BEHAVIOR_CLUSTER_EPS_METERS)} m, min_samples {BEHAVIOR_CLUSTER_MIN_SAMPLES}; p95 endpoint radius must be at most {int(MAX_CLUSTER_P95_RADIUS_METERS)} m.",
            "poi_enrichment": f"Cached local OSM-derived category tags within {int(LOCAL_POI_BUFFER_METERS)} m. No external reverse geocoding, Google Maps, Overpass query, or map scraping was used.",
            "inter_trip_interval_measure": "Observed same-area inter-trip interval only when a trip ends and the next trip starts in the same area and county on the same date, within 10 minutes to 12 hours. It is not a dwell estimate.",
            "od_route_change": f"Frequency-weighted FID overlap between compact recurring privacy-safe origin/destination activity areas in consecutive months. OD pairs must pass an approximate {MIN_OD_SEPARATION_METERS / 1000:.0f} km separation screen; prominent report rows also have overall RCCI at least {PROMINENT_GLOBAL_RCCI_MIN:.0f} with HIGH/MEDIUM confidence.",
        },
        "privacy_safeguards": [
            "No exact coordinates, addresses, named POIs, or inferred home/work locations are exported.",
            "Recurring activity areas are route-network/recording clusters, not confirmed visits or destinations.",
            "Healthcare, educational, retail, recreation, and place-of-worship labels describe nearby generic category context only.",
            "No medical, employment, education, residence, religion, family, income, identity, or preference claim is made.",
        ],
        "data_quality": {
            "total_matched_trips": int(len(trips)),
            "broward_matched_trips": int(len(broward)),
            "broward_trip_share": float(len(broward) / len(trips)) if len(trips) else None,
            "observed_months": int(broward["trip_month"].nunique()),
            "early_morning_04_00_to_08_59_starts": early_morning,
            "weekday_06_00_to_10_00_starts": weekday_commute_start,
            "weekday_06_00_to_10_00_ends": weekday_commute_end,
            "top_three_compact_activity_area_endpoint_share": top_three_endpoint_share,
            "local_context_counts": {
                "retail_or_errand_proximity": retail_context_count,
                "educational_proximity": education_context_count,
                "healthcare_proximity": healthcare_context_count,
            },
            "home_work_inference_status": "Not established: temporal coverage lacks a conventional early-morning commute sample and such location conclusions are privacy-sensitive.",
        },
        "key_findings": [
            "The analysis identifies recurrent activity areas and route-pattern changes, not confirmed personal locations or trip purposes.",
            f"{broward_cluster_count} compact recurrent Broward activity areas meet the reportable recurrence screen.",
            f"{len(route_findings)} recurring OD-area month-pair candidates show substantial path change; {prominent_route_count} meet the stricter prominent-event screen.",
            f"{len(stable_route_findings)} well-sampled recurring OD-area month pairs retained a stable route pattern.",
            "Home, work, school/child, healthcare, religious, and family conclusions are intentionally not established from this dataset.",
        ],
        "location_clusters": location_findings,
        "od_route_change_insights": route_findings,
        "od_stability_insights": stable_route_findings,
        "limitations": [
            "RCCI and matched-route changes do not determine cause.",
            "Trip segmentation and collection windows can bias observed inter-trip intervals and time-of-day patterns.",
            "Local OSM categories can be incomplete or outdated.",
            "Sparse supplementary-county data are not used for behavioral interpretation.",
        ],
    }


def html_table(frame: pd.DataFrame, columns: Sequence[tuple[str, str]], limit: int | None = None) -> str:
    data = frame.head(limit).copy() if limit else frame.copy()
    if data.empty:
        return "<p class='empty'>No evidence met the documented screen.</p>"
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    rows = []
    for record in data.to_dict(orient="records"):
        cells = "".join(
            f"<td>{html.escape(str(record.get(key, '—')))}</td>" for key, _ in columns
        )
        rows.append(f"<tr>{cells}</tr>")
    return f"<div class='table-wrap behavior-table'><table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def behavior_css() -> str:
    return f"""
{BEHAVIOR_STYLE_MARKER}
.behavior-insights .behavior-lead{{font-size:16px;color:#334155}}
.behavior-insights .behavior-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:16px 0}}
.behavior-insights .behavior-card{{background:#f8fbff;border:1px solid var(--line);border-radius:13px;padding:14px}}
.behavior-insights .behavior-card strong{{display:block;color:var(--text);font-size:17px;margin-bottom:5px}}
.behavior-insights .behavior-card p{{margin:0;font-size:14px}}
.behavior-insights .behavior-privacy{{background:#fff1f0;border-left:5px solid var(--red);border-radius:12px;padding:15px;margin:16px 0;color:#6b1d15}}
.behavior-insights .behavior-method{{background:#edf8f3;border-left:5px solid var(--green);border-radius:12px;padding:15px;margin:16px 0;color:#24523a}}
.behavior-insights .behavior-table td{{white-space:normal;vertical-align:top;min-width:100px}}
"""


def render_behavior_html(
    *,
    trips: pd.DataFrame,
    clusters: pd.DataFrame,
    od_changes: pd.DataFrame,
    od_stability: pd.DataFrame,
) -> str:
    broward = trips.loc[trips["county"] == PRIMARY_COUNTY]
    early_morning = int(broward["start_hour"].between(4, 8.99, inclusive="both").sum())
    weekday_commute_starts = int(
        (broward["is_weekday"] & broward["start_hour"].between(6, 10, inclusive="both")).sum()
    )
    weekday_commute_ends = int(
        (broward["is_weekday"] & broward["end_hour"].between(6, 10, inclusive="both")).sum()
    )
    broward_cluster_count = int((clusters["county"] == PRIMARY_COUNTY).sum()) if not clusters.empty else 0
    broward_clusters = (
        clusters.loc[clusters["county"] == PRIMARY_COUNTY].copy()
        if not clusters.empty
        else clusters.copy()
    )
    clustered_endpoint_total = int(broward_clusters["endpoint_records"].sum()) if not broward_clusters.empty else 0
    top_three_endpoint_share = (
        float(broward_clusters.nlargest(3, "endpoint_records")["endpoint_records"].sum())
        / clustered_endpoint_total
        if clustered_endpoint_total
        else 0.0
    )
    education_context_count = int(
        broward_clusters["poi_category_context"].str.contains("educational category", na=False).sum()
    ) if not broward_clusters.empty else 0
    healthcare_context_count = int(
        broward_clusters["poi_category_context"].str.contains("healthcare-related category", na=False).sum()
    ) if not broward_clusters.empty else 0
    retail_context_count = int(
        broward_clusters["poi_category_context"].str.contains(
            "shopping/retail category|errand-service category", regex=True, na=False
        ).sum()
    ) if not broward_clusters.empty else 0
    prominent_od_changes = (
        od_changes.loc[od_changes["report_priority"] == "prominent"].copy()
        if not od_changes.empty
        else od_changes.copy()
    )
    stability_display = od_stability.copy()
    cluster_display = clusters.copy()
    if not cluster_display.empty:
        cluster_display["endpoint_records"] = cluster_display["endpoint_records"].map(number)
        cluster_display["unique_days_visited"] = cluster_display["unique_days_visited"].map(number)
        cluster_display["weekday_weekend"] = cluster_display.apply(
            lambda row: f"{number(row.weekday_endpoint_records)} / {number(row.weekend_endpoint_records)}", axis=1
        )
        cluster_display["confidence"] = cluster_display.apply(
            lambda row: (
                f"recurrence {row.recurrence_confidence_level} ({row.recurrence_confidence_score:.2f}); "
                f"context {row.context_confidence_level} ({row.context_confidence_score:.2f})"
            ),
            axis=1,
        )
        cluster_display["inter_trip_interval"] = cluster_display.apply(
            lambda row: (
                f"{row.median_inter_trip_interval_minutes:.0f} min ({int(row.inter_trip_interval_observations)} intervals)"
                if pd.notna(row.median_inter_trip_interval_minutes)
                else "insufficient same-area inter-trip intervals"
            ),
            axis=1,
        )
    cluster_columns = [
        ("activity_area_label", "Activity area"),
        ("endpoint_records", "Endpoints"),
        ("unique_days_visited", "Days"),
        ("inter_trip_interval", "Same-area interval"),
        ("weekday_weekend", "Weekday / weekend"),
        ("behavior_pattern", "Careful pattern"),
        ("poi_category_context", "Generic nearby context"),
        ("confidence", "Confidence"),
    ]
    context_display = clusters.loc[
        clusters["poi_category_context"].ne("unknown area")
    ].copy() if not clusters.empty else clusters.copy()
    if not context_display.empty:
        context_display["endpoint_records"] = context_display["endpoint_records"].map(number)
        context_display["context_confidence"] = context_display.apply(
            lambda row: f"{row.context_confidence_level} ({row.context_confidence_score:.2f})",
            axis=1,
        )
    context_columns = [
        ("activity_area_label", "Activity area"),
        ("endpoint_records", "Endpoints"),
        ("months_visited", "Months"),
        ("poi_category_context", "Generic nearby context"),
        ("behavior_pattern", "Careful pattern"),
        ("context_confidence", "Context confidence"),
    ]
    od_display = prominent_od_changes.copy()
    if not od_display.empty:
        od_display["period"] = od_display["month_a"].astype(str) + " → " + od_display["month_b"].astype(str)
        od_display["overlap"] = od_display["matched_segment_overlap_pct"].map(lambda value: f"{value:.1f}%")
        od_display["change_score"] = od_display["od_route_change_score"].map(lambda value: f"{value:.1f}")
        od_display["confidence"] = od_display.apply(
            lambda row: f"{row.confidence_level} ({row.confidence_score:.2f})", axis=1
        )
    od_columns = [
        ("period", "Month pair"),
        ("origin_activity_area", "Origin area"),
        ("destination_activity_area", "Destination area"),
        ("trips_a", "Trips A"),
        ("trips_b", "Trips B"),
        ("overlap", "Route overlap"),
        ("route_change_interpretation", "Plain-language change"),
        ("confidence", "Confidence"),
    ]
    if not stability_display.empty:
        stability_display["period"] = (
            stability_display["month_a"].astype(str)
            + " → "
            + stability_display["month_b"].astype(str)
        )
        stability_display["overlap"] = stability_display["matched_segment_overlap_pct"].map(
            lambda value: f"{value:.1f}%"
        )
        stability_display["confidence"] = stability_display.apply(
            lambda row: f"{row.confidence_level} ({row.confidence_score:.2f})", axis=1
        )
    stability_columns = [
        ("period", "Month pair"),
        ("origin_activity_area", "Origin area"),
        ("destination_activity_area", "Destination area"),
        ("trips_a", "Trips A"),
        ("trips_b", "Trips B"),
        ("overlap", "Route overlap"),
        ("careful_interpretation", "Careful interpretation"),
        ("confidence", "Confidence"),
    ]
    links = "".join(
        f"<li><a href='../data/{filename}'>{html.escape(label)}</a></li>"
        for filename, label in [
            ("driver_1003_location_clusters.csv", "Recurring activity-area evidence (CSV)"),
            ("driver_1003_od_route_change_insights.csv", "Stable-OD route-change evidence (CSV)"),
            ("driver_1003_behavior_insights.json", "Behavior-insights metadata and evidence (JSON)"),
        ]
    )
    return f"""{BEHAVIOR_BEGIN}
<section id="real-world-driver-behavior-insights" class="behavior-insights">
<h2>Real-World Driver Behavior Insights</h2>
<p class="behavior-lead">This layer translates recurring matched-route patterns into cautious, real-world language. It identifies generic recurring activity areas and route changes, while deliberately avoiding claims about where the driver lives, works, studies, worships, shops, or receives care.</p>

<h3>Key findings in plain English</h3>
<div class="behavior-grid">
  <div class="behavior-card"><strong>Recurring activity areas</strong><p>{broward_cluster_count} Broward activity areas meet the recurrence screen. They represent repeated route endpoints, not verified places or visits.</p></div>
  <div class="behavior-card"><strong>Home/work inference withheld</strong><p>{early_morning:,} Broward starts fall from 04:00–08:59, but there are {weekday_commute_starts:,} weekday starts and {weekday_commute_ends:,} weekday ends from 06:00–10:00. That does not support a conventional home-to-work or school-drop-off conclusion.</p></div>
  <div class="behavior-card"><strong>Same-area route changes</strong><p>{len(prominent_od_changes)} higher-priority repeated origin/destination-area month pairs pass the compact-area, 2 km separation, and overall-RCCI screens. These may reflect corridor, highway/surface-street, or connector choices—not a known reason for travel.</p></div>
  <div class="behavior-card"><strong>Stable repeated routes</strong><p>{len(od_stability)} well-sampled area-to-area month pairs retained high route overlap. This is consistent with a stable repeated route pattern, not a confirmed commute or trip purpose.</p></div>
  <div class="behavior-card"><strong>Generic local context only</strong><p>Nearby category labels come from cached local OSM data within {int(LOCAL_POI_BUFFER_METERS)} m. External maps, reverse geocoding, POI names, and addresses were not used.</p></div>
</div>

<h3>Plain-English synthesis</h3>
<ul>
  <li>The three most frequent compact activity areas account for {top_three_endpoint_share:.0%} of reportable clustered endpoints across {int(broward["trip_month"].nunique())} Broward months. This is consistent with a recurring travel geography in the pseudonymous trip data, not a confirmed personal routine.</li>
  <li>{retail_context_count} recurrent activity areas have low-confidence retail/errand category proximity from cached local OSM tags. That may help prioritize future POI verification, but it does not establish shopping or an errand.</li>
  <li>The report identifies route-pattern change and stability for repeated generic area pairs. It cannot determine whether traffic, preference, construction, a new obligation, or data collection caused a difference.</li>
</ul>

<div class="behavior-privacy"><strong>Privacy and uncertainty:</strong> A recurring residential-context cluster is not a home determination; a weekday daytime pattern is not a workplace determination; and education, healthcare, religious, retail, or recreation category proximity is not proof of a visit, affiliation, or personal circumstance.</div>

<h3>Likely home/work, school, and healthcare status</h3>
<div class="behavior-grid">
  <div class="behavior-card"><strong>Home area</strong><p><strong>Not established.</strong> The available time coverage does not contain a conventional weekday-morning departure sample, and a home inference would be privacy-sensitive. The report therefore uses only generic recurring activity-area labels.</p></div>
  <div class="behavior-card"><strong>Work area</strong><p><strong>Not established.</strong> Repeated weekday patterns can be described as activity patterns, but they do not prove employment or a workplace.</p></div>
  <div class="behavior-card"><strong>School/child stop</strong><p><strong>Not established.</strong> {education_context_count} reportable clusters have local educational-category proximity. Even if present, it is only a possible proximity pattern and does not establish a child, attendance, or drop-off.</p></div>
  <div class="behavior-card"><strong>Healthcare, recreation, and religious context</strong><p><strong>Not established.</strong> {healthcare_context_count} reportable clusters have local healthcare-category proximity. Any nearby category is context only; it does not confirm a healthcare visit, medical condition, religious affiliation, recreation, or a family visit.</p></div>
</div>

<h3>Recurring activity-area context</h3>
<p>Endpoint clusters use county-level DBSCAN on first/last cached GPS endpoints transformed to EPSG:26917 (ε={int(BEHAVIOR_CLUSTER_EPS_METERS)} m; minimum samples={BEHAVIOR_CLUSTER_MIN_SAMPLES}); clusters with a p95 radius above {int(MAX_CLUSTER_P95_RADIUS_METERS)} m are excluded. An observed same-area inter-trip interval is reported only when the next trip begins in the same county and area on the same date within 10 minutes to 12 hours; it is not a dwell estimate.</p>
{html_table(cluster_display, cluster_columns, limit=10)}
<div class="behavior-method"><strong>How to read this:</strong> “Confidence” separates recurrence evidence from weak local-context evidence. Cached local OSM context is capped at low/medium confidence and does not verify a category or trip purpose. “Unknown area” means local category data did not support a label.</div>

<h3>Major route changes with stable activity areas</h3>
<p>These higher-priority rows hold compact generic origin/destination activity areas constant, require an approximate {MIN_OD_SEPARATION_METERS / 1000:.0f} km separation screen, and compare the matched-route FID mix in consecutive months. They also have overall Broward RCCI of at least {PROMINENT_GLOBAL_RCCI_MIN:.0f} with HIGH/MEDIUM confidence. Route overlap is frequency-weighted; lower overlap means greater path change for the same broad area pair. Supporting candidates remain in the CSV.</p>
{html_table(od_display, od_columns, limit=12)}

<h3>Repeated routes that stayed consistent</h3>
<p>These rows use the same compact-area and 2 km separation safeguards, require at least {STABLE_OD_MIN_MONTHLY_TRIPS} trips in each month, and retain at least {STABLE_OD_OVERLAP_MIN:.0%} frequency-weighted matched-segment overlap. They support a stable route-pattern finding only.</p>
{html_table(stability_display, stability_columns, limit=6)}

<h3>Monthly recurring POI/context patterns</h3>
<p>The cluster table reports category proximity such as retail, healthcare, education, recreation, restaurant, office/commercial, or residential context only when local OSM tags support it. No named venue is reported, and the absence of a category is recorded as “unknown area.”</p>
{html_table(context_display, context_columns, limit=10)}

<h3>Limitations</h3>
<ul>
  <li>RCCI and route changes measure movement patterns, not causes, intent, or confirmed destinations.</li>
  <li>Trip start/end times reflect data-collection and segmentation windows; they are not a complete daily travel diary.</li>
  <li>Same-area inter-trip intervals are not confirmed stop or dwell durations.</li>
  <li>Local OSM context can be incomplete or outdated; no external POI lookup was used.</li>
  <li>No exact addresses, coordinates, named sensitive POIs, or claims about home, work, health, education, religion, family, income, or identity are included.</li>
</ul>

<h3>Evidence files</h3>
<ul>{links}</ul>
</section>
{BEHAVIOR_END}"""


def insert_behavior_section(report: Path, section: str) -> None:
    document = report.read_text(encoding="utf-8")
    document = re.sub(
        re.escape(BEHAVIOR_BEGIN) + r".*?" + re.escape(BEHAVIOR_END),
        "",
        document,
        flags=re.DOTALL,
    )
    document = re.sub(
        re.escape(BEHAVIOR_NAV_BEGIN) + r".*?" + re.escape(BEHAVIOR_NAV_END),
        "",
        document,
        flags=re.DOTALL,
    )
    if BEHAVIOR_STYLE_MARKER not in document:
        if "</style>" not in document:
            raise RuntimeError("Could not find report style block")
        document = document.replace("</style>", behavior_css() + "\n</style>", 1)
    nav = (
        f"{BEHAVIOR_NAV_BEGIN}\n"
        '<a href="#real-world-driver-behavior-insights">Real-World Behavior</a>\n'
        f"{BEHAVIOR_NAV_END}"
    )
    if "</nav>" not in document:
        raise RuntimeError("Could not find report navigation")
    document = document.replace("</nav>", nav + "\n</nav>", 1)
    research_marker = "<!-- END DRIVER 1003 RESEARCH INSIGHTS -->"
    if research_marker in document:
        document = document.replace(research_marker, research_marker + "\n\n" + section, 1)
    elif "</main>" in document:
        document = document.replace("</main>", section + "\n</main>", 1)
    else:
        raise RuntimeError("Could not find report insertion point")
    report.write_text(document, encoding="utf-8")


def validate(
    *,
    paths: Sequence[Path],
    report: Path,
    clusters: pd.DataFrame,
    od_changes: pd.DataFrame,
    od_stability: pd.DataFrame,
) -> None:
    missing = [str(path) for path in paths if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing behavior output(s): {missing}")
    for path in paths:
        if path.suffix.lower() == ".csv":
            header = path.read_text(encoding="utf-8").splitlines()[0].lower()
            forbidden = {"latitude", "longitude", "centroid_x", "centroid_y", "address", "poi_name"}
            if any(field in header.split(",") for field in forbidden):
                raise RuntimeError(f"Privacy-unsafe field in output: {path}")
    if clusters.empty:
        raise RuntimeError("No reportable activity clusters were generated")
    if not (clusters["endpoint_spread_p95_m"] <= MAX_CLUSTER_P95_RADIUS_METERS).all():
        raise RuntimeError("A noncompact activity cluster reached the exported output")
    if not od_changes.empty and not (od_changes["od_route_change_score"] >= OD_CHANGE_SCORE_MIN).all():
        raise RuntimeError("OD route-change table contains a row below the threshold")
    if not od_changes.empty and not od_changes["approximate_od_separation_screen"].eq(
        f"passed ≥{MIN_OD_SEPARATION_METERS / 1000:.0f} km screen"
    ).all():
        raise RuntimeError("OD route-change table contains a pair that failed the separation screen")
    if not od_stability.empty and not (
        od_stability["matched_segment_overlap_pct"] >= STABLE_OD_OVERLAP_MIN * 100
    ).all():
        raise RuntimeError("OD stability table contains a row below the overlap threshold")
    document = report.read_text(encoding="utf-8")
    for marker in (BEHAVIOR_BEGIN, BEHAVIOR_END, 'id="real-world-driver-behavior-insights"'):
        if marker not in document:
            raise RuntimeError(f"Behavior report marker missing: {marker}")
    if document.count(BEHAVIOR_BEGIN) != 1 or document.count(BEHAVIOR_END) != 1:
        raise RuntimeError("Behavior report contains duplicate section markers")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    report = args.report.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trips, _ = load_trip_table()
    trips, endpoints = attach_trip_activity_areas(trips)
    clusters = build_location_clusters(trips, endpoints)
    od_changes = build_od_route_changes(trips, endpoints)
    od_stability = build_od_stability_insights(trips, endpoints)

    paths = {
        "location_clusters": output_dir / "driver_1003_location_clusters.csv",
        "od_route_changes": output_dir / "driver_1003_od_route_change_insights.csv",
        "behavior_json": output_dir / "driver_1003_behavior_insights.json",
    }
    clusters.to_csv(paths["location_clusters"], index=False)
    od_changes.to_csv(paths["od_route_changes"], index=False)
    paths["behavior_json"].write_text(
        json.dumps(
            behavior_json(
                trips=trips,
                clusters=clusters,
                od_changes=od_changes,
                od_stability=od_stability,
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if not args.skip_report_update:
        insert_behavior_section(
            report,
            render_behavior_html(
                trips=trips,
                clusters=clusters,
                od_changes=od_changes,
                od_stability=od_stability,
            ),
        )
    validate(
        paths=list(paths.values()),
        report=report,
        clusters=clusters,
        od_changes=od_changes,
        od_stability=od_stability,
    )
    print("Driver 1003 real-world behavior insights complete")
    print(f"  location clusters: {paths['location_clusters']}")
    print(f"  OD route changes: {paths['od_route_changes']}")
    print(f"  behavior JSON: {paths['behavior_json']}")
    print(f"  report: {report}")
    print(f"  reportable activity areas: {len(clusters)}")
    print(f"  compact-OD route-change candidates: {len(od_changes)}")
    print(f"  stable repeated OD patterns: {len(od_stability)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
