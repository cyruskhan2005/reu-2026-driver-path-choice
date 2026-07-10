"""Real-world place and route-choice analysis for the Driver 1003 study.

The module consumes only the repository's explicit source-trip timeline, cached
county GPS observations, matched FID sequences, and enriched road attributes.
It keeps exact coordinates in private analysis tables while constructing a
separate sanitized representation for public HTML/JSON/map outputs.

External enrichment is deliberately isolated in :mod:`roadnet.google_places`.
No API credential is accepted as a function argument or serialized here.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
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

from .behavior_activity import (
    build_repeated_trip_chains,
    reconstruct_stays,
    summarize_cluster_stays,
    workplace_plausibility,
)
from .behavior_cluster_stability import analyze_cluster_stability
from .behavior_quality import quality_flag_counts, validate_trip_quality


ROOT = Path(__file__).resolve().parents[1]
DRIVER_ID = 1003
SUBJECT_INTERNAL_ID = "ca351c04cfabaae40cb77059ef799f4a"
SUBJECT_COLLECTION_ID = "1003 1004"
LOCAL_TIMEZONE = "America/New_York"
PROJECTED_CRS = "EPSG:26917"
WGS84 = "EPSG:4326"

TIMELINE_PATH = (
    ROOT / "sflorida_outputs" / "phase2" / "driver_timelines" / "driver_1_timeline.csv"
)
MONTHLY_NODE_PATH = (
    ROOT
    / "deliverables"
    / "google_drive_phase2"
    / "driver_1003_monthly_graphs"
    / "data"
    / "driver_1003_all_monthly_nodes.csv"
)
RCCI_SUMMARY_PATH = (
    ROOT
    / "deliverables"
    / "driver_1003"
    / "route_choice_change_index"
    / "data"
    / "driver_1003_rcci_summary.csv"
)
CURATED_REPORT_PATH = (
    ROOT
    / "deliverables"
    / "driver_1003"
    / "route_choice_change_index"
    / "visuals"
    / "driver_1003_route_choice_change_index_report.html"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_CACHE_DIR = ROOT / "cache" / "google_maps"

COUNTY_PATHS: dict[str, dict[str, Path]] = {
    "Broward County": {
        "gps": ROOT / "sflorida_outputs" / "Broward_County" / "Broward County_gps.csv",
        "network": ROOT / "sflorida_outputs" / "Broward_County" / "enriched_network.parquet",
        "landuse": ROOT / "sflorida_outputs" / "Broward_County" / "osm_landuse.parquet",
    },
    "Palm Beach County": {
        "gps": ROOT / "sflorida_outputs" / "Palm_Beach_County" / "Palm Beach County_gps.csv",
        "network": ROOT / "sflorida_outputs" / "Palm_Beach_County" / "enriched_network.parquet",
        "landuse": ROOT / "sflorida_outputs" / "Palm_Beach_County" / "osm_landuse.parquet",
    },
    "Miami-Dade County": {
        "gps": ROOT / "sflorida_outputs" / "Miami_Dade_County" / "Miami-Dade County_gps.csv",
        "network": ROOT / "sflorida_outputs" / "Miami_Dade_County" / "enriched_network.parquet",
        "landuse": ROOT / "sflorida_outputs" / "Miami_Dade_County" / "osm_landuse.parquet",
    },
}


def county_paths_for_output_root(output_root: Path) -> dict[str, dict[str, Path]]:
    """Return county GPS/network/land-use paths for a configured pipeline root."""
    root = Path(output_root)
    return {
        county: {
            "gps": root / slug / f"{county}_gps.csv",
            "network": root / slug / "enriched_network.parquet",
            "landuse": root / slug / "osm_landuse.parquet",
        }
        for county, slug in (
            ("Broward County", "Broward_County"),
            ("Palm Beach County", "Palm_Beach_County"),
            ("Miami-Dade County", "Miami_Dade_County"),
        )
    }

CANDIDATE_RADII_M = (50.0, 75.0, 100.0, 150.0, 250.0)
CLUSTER_MIN_SAMPLES = 4
MAX_BILLABLE_REQUESTS = 100
PLACES_RADII_M = (50.0, 100.0, 250.0, 500.0)

CONTROLLED_ACCESS_CLASSES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
}
ARTERIAL_CLASSES = {
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
}
LOCAL_CLASSES = {
    "residential",
    "service",
    "living_street",
    "unclassified",
}


class BehaviorAnalysisError(RuntimeError):
    """Raised when cached project data cannot support the requested analysis."""


@dataclass(frozen=True)
class ClusterSelection:
    """Selected DBSCAN radius and diagnostics for every evaluated radius."""

    radius_m: float
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class BuildResult:
    """Paths and validation metadata returned by the end-to-end build."""

    paths: Mapping[str, Path]
    source_trip_count: int
    county_fragment_count: int
    selected_cluster_radius_m: float
    google_requests: int
    cache_hits: int
    sources_used: tuple[str, ...]
    likely_home_cluster_id: str
    privacy_checks_passed: bool


def _require(path: Path) -> Path:
    if not path.exists():
        raise BehaviorAnalysisError(f"Required input is missing: {path}")
    return path


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise BehaviorAnalysisError(f"{label} is missing required columns: {missing}")


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _stable_slug(*parts: object) -> str:
    value = "_".join(_clean_text(part) for part in parts)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def parse_fid_sequence(value: object) -> list[int]:
    """Parse an ordered FID sequence while retaining nonconsecutive repeats."""
    text = _clean_text(value)
    if not text:
        return []
    result: list[int] = []
    for token in re.split(r"[|,;\s]+", text):
        try:
            fid = int(token)
        except (TypeError, ValueError):
            continue
        if fid >= 0 and (not result or result[-1] != fid):
            result.append(fid)
    return result


def collapse_adjacent(values: Iterable[str]) -> list[str]:
    """Remove blank and adjacent repeated strings without changing order."""
    result: list[str] = []
    for value in values:
        text = _clean_text(value)
        if text and (not result or result[-1] != text):
            result.append(text)
    return result


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in metres between two WGS84 points."""
    radius = 6_371_008.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return radius * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))


def _local_timestamp(epoch_seconds: object) -> pd.Timestamp:
    value = pd.to_numeric(pd.Series([epoch_seconds]), errors="coerce").iloc[0]
    if pd.isna(value):
        return pd.NaT
    return pd.Timestamp(value, unit="s", tz="UTC").tz_convert(LOCAL_TIMEZONE)


def load_driver_timeline(path: Path = TIMELINE_PATH) -> pd.DataFrame:
    """Load the canonical Driver 1003 county-fragment timeline."""
    timeline = pd.read_csv(
        _require(path),
        dtype={
            "trip_id": "string",
            "session_id": "string",
            "source_trip_id": "string",
            "fid_sequence": "string",
            "internal_driver_id": "string",
            "collection_id": "string",
        },
        low_memory=False,
    )
    required = {
        "internal_driver_id",
        "collection_id",
        "trip_id",
        "county",
        "trip_start_time",
        "duration_seconds",
        "fid_sequence",
        "matched_trip_id",
        "session_id",
        "source_trip_id",
    }
    missing = required - set(timeline.columns)
    if missing:
        raise BehaviorAnalysisError(f"Timeline is missing columns: {sorted(missing)}")
    timeline = timeline.loc[
        timeline["internal_driver_id"].eq(SUBJECT_INTERNAL_ID)
        & timeline["collection_id"].eq(SUBJECT_COLLECTION_ID)
    ].copy()
    if timeline.empty:
        raise BehaviorAnalysisError("Driver 1003 has no rows in the canonical timeline")
    timeline["matched_trip_id"] = pd.to_numeric(
        timeline["matched_trip_id"], errors="coerce"
    )
    timeline = timeline.dropna(subset=["matched_trip_id", "session_id", "source_trip_id"])
    timeline["matched_trip_id"] = timeline["matched_trip_id"].astype("int64")
    timeline["duration_seconds"] = pd.to_numeric(
        timeline["duration_seconds"], errors="coerce"
    ).clip(lower=0)
    timeline["fragment_start"] = pd.to_datetime(
        timeline["trip_start_time"], errors="coerce", utc=True
    ).dt.tz_convert(LOCAL_TIMEZONE)
    timeline["fragment_end"] = timeline["fragment_start"] + pd.to_timedelta(
        timeline["duration_seconds"], unit="s"
    )
    timeline["source_key"] = (
        timeline["session_id"].astype(str) + "::" + timeline["source_trip_id"].astype(str)
    )
    if timeline.duplicated(["county", "matched_trip_id"]).any():
        raise BehaviorAnalysisError("Timeline contains duplicate county/matched-trip IDs")
    return timeline.sort_values(["fragment_start", "county", "matched_trip_id"]).reset_index(
        drop=True
    )


def load_road_context(path: Path = MONTHLY_NODE_PATH) -> pd.DataFrame:
    """Return one enriched road row per county/FID used by Driver 1003."""
    header = pd.read_csv(_require(path), nrows=0).columns
    wanted = [
        "county",
        "fid",
        "u",
        "v",
        "name",
        "highway",
        "length",
        "road_length_m",
        "estimated_speed_limit",
        "FDOT_FUNCTIONAL_CLASS",
        "FDOT_ROADWAY",
        "landuse",
        "is_connector",
        "geometry_wkt",
    ]
    available = [column for column in wanted if column in header]
    context = pd.read_csv(path, usecols=available, low_memory=False)
    context["fid"] = pd.to_numeric(context["fid"], errors="coerce")
    context = context.dropna(subset=["county", "fid"]).copy()
    context["fid"] = context["fid"].astype("int64")
    if "road_length_m" not in context and "length" in context:
        context["road_length_m"] = context["length"]
    elif "road_length_m" in context and "length" in context:
        context["road_length_m"] = pd.to_numeric(
            context["road_length_m"], errors="coerce"
        ).fillna(pd.to_numeric(context["length"], errors="coerce"))
    context = context.sort_values(["county", "fid"]).drop_duplicates(
        ["county", "fid"], keep="first"
    )
    return context.reset_index(drop=True)


def add_local_toll_flags(
    context: pd.DataFrame,
    toll_path: Path = ROOT / "sflorida_outputs" / "fdot" / "toll_roads.parquet",
    *,
    tolerance_m: float = 30.0,
) -> pd.DataFrame:
    """Mark driver-used FIDs that overlap the cached FDOT toll-road layer."""
    output = context.copy()
    output["toll"] = False
    if not toll_path.exists() or "geometry_wkt" not in output:
        return output
    from shapely import wkt

    toll = gpd.read_parquet(toll_path)
    if toll.empty:
        return output
    if toll.crs is None:
        toll = toll.set_crs(PROJECTED_CRS)
    elif str(toll.crs) != PROJECTED_CRS:
        toll = toll.to_crs(PROJECTED_CRS)
    toll_area = toll.geometry.buffer(float(tolerance_m)).union_all()

    def overlaps(value: object) -> bool:
        text = _clean_text(value)
        if not text:
            return False
        try:
            geometry = wkt.loads(text)
        except Exception:
            return False
        return bool(geometry.intersects(toll_area))

    output["toll"] = output["geometry_wkt"].map(overlaps)
    return output


def _selected_gps_endpoints(
    timeline: pd.DataFrame,
    county_paths: Mapping[str, Mapping[str, Path]] = COUNTY_PATHS,
) -> pd.DataFrame:
    """Read first/last cached GPS observations for selected county fragments."""
    parts: list[pd.DataFrame] = []
    for county, county_timeline in timeline.groupby("county", sort=True):
        path = county_paths.get(str(county), {}).get("gps")
        if not path:
            raise BehaviorAnalysisError(f"No GPS source is configured for {county}")
        ids = set(county_timeline["matched_trip_id"].astype(int))
        first_rows: list[pd.DataFrame] = []
        last_rows: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            _require(path),
            sep=";",
            usecols=["id", "lon", "lat", "timestamp", "point_idx"],
            chunksize=250_000,
        ):
            selected = chunk.loc[chunk["id"].isin(ids)].copy()
            if selected.empty:
                continue
            selected = selected.sort_values(["id", "timestamp", "point_idx"])
            first_rows.append(selected.drop_duplicates("id", keep="first"))
            last_rows.append(selected.drop_duplicates("id", keep="last"))
        if not first_rows or not last_rows:
            raise BehaviorAnalysisError(f"No selected GPS points were read for {county}")
        first = (
            pd.concat(first_rows, ignore_index=True)
            .sort_values(["id", "timestamp", "point_idx"])
            .drop_duplicates("id", keep="first")
            .rename(
                columns={
                    "id": "matched_trip_id",
                    "lon": "start_longitude",
                    "lat": "start_latitude",
                    "timestamp": "start_epoch",
                    "point_idx": "start_point_idx",
                }
            )
        )
        last = (
            pd.concat(last_rows, ignore_index=True)
            .sort_values(["id", "timestamp", "point_idx"])
            .drop_duplicates("id", keep="last")
            .rename(
                columns={
                    "id": "matched_trip_id",
                    "lon": "end_longitude",
                    "lat": "end_latitude",
                    "timestamp": "end_epoch",
                    "point_idx": "end_point_idx",
                }
            )
        )
        endpoints = first.merge(last, on="matched_trip_id", validate="one_to_one")
        endpoints.insert(0, "county", county)
        parts.append(endpoints)
    result = pd.concat(parts, ignore_index=True)
    expected = timeline[["county", "matched_trip_id"]].drop_duplicates()
    joined = expected.merge(
        result[["county", "matched_trip_id"]],
        on=["county", "matched_trip_id"],
        how="left",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = int(joined["_merge"].ne("both").sum())
        raise BehaviorAnalysisError(f"{missing} timeline fragments lack GPS endpoints")
    return result


def _road_lookup(context: pd.DataFrame) -> dict[tuple[str, int], dict[str, object]]:
    fields = [
        column
        for column in (
            "name",
            "highway",
            "road_length_m",
            "estimated_speed_limit",
            "FDOT_FUNCTIONAL_CLASS",
            "FDOT_ROADWAY",
            "landuse",
            "is_connector",
            "toll",
            "u",
            "v",
            "geometry_wkt",
        )
        if column in context
    ]
    return {
        (str(row.county), int(row.fid)): {
            field: getattr(row, field, None) for field in fields
        }
        for row in context[["county", "fid", *fields]].itertuples(index=False)
    }


def _fragment_route_profile(
    county: str,
    fids: Sequence[int],
    roads: Mapping[tuple[str, int], Mapping[str, object]],
) -> dict[str, object]:
    total_distance = 0.0
    class_distance: Counter[str] = Counter()
    names: list[str] = []
    refs: list[str] = []
    missing = 0
    toll_distance = 0.0
    for fid in fids:
        row = roads.get((county, int(fid)), {})
        if not row:
            missing += 1
            continue
        length = max(_safe_float(row.get("road_length_m"), 0.0), 0.0)
        total_distance += length
        road_class = _clean_text(row.get("highway")).split("|")[0].lower() or "unknown"
        class_distance[road_class] += length
        if bool(row.get("toll", False)):
            toll_distance += length
        name = _clean_text(row.get("name")) or _clean_text(row.get("FDOT_ROADWAY"))
        if name:
            names.append(name)
        fdot = _clean_text(row.get("FDOT_ROADWAY"))
        if fdot:
            refs.append(fdot)
    controlled = sum(class_distance[name] for name in CONTROLLED_ACCESS_CLASSES)
    arterial = sum(class_distance[name] for name in ARTERIAL_CLASSES)
    local = sum(class_distance[name] for name in LOCAL_CLASSES)
    surface = max(total_distance - controlled, 0.0)
    return {
        "distance_m": total_distance,
        "controlled_distance_m": controlled,
        "arterial_distance_m": arterial,
        "local_distance_m": local,
        "surface_distance_m": surface,
        "toll_distance_m": toll_distance,
        "road_names": collapse_adjacent(names),
        "fdot_refs": collapse_adjacent(refs),
        "missing_fid_count": missing,
    }


def build_trip_summary(
    timeline: pd.DataFrame | None = None,
    road_context: pd.DataFrame | None = None,
    *,
    county_paths: Mapping[str, Mapping[str, Path]] = COUNTY_PATHS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one physical source-trip row and retain audited county fragments.

    Two source recordings cross county boundaries and therefore appear twice in
    the county-specific FMM timeline. They are stitched by explicit
    ``(session_id, source_trip_id)`` identity and endpoint timestamps, not by an
    inferred inactivity rule.
    """
    timeline = load_driver_timeline() if timeline is None else timeline.copy()
    road_context = load_road_context() if road_context is None else road_context.copy()
    endpoints = _selected_gps_endpoints(timeline, county_paths=county_paths)
    fragments = timeline.merge(
        endpoints,
        on=["county", "matched_trip_id"],
        how="left",
        validate="one_to_one",
    )
    roads = _road_lookup(road_context)
    fragments["fid_list"] = fragments["fid_sequence"].map(parse_fid_sequence)
    profiles = [
        _fragment_route_profile(str(county), fids, roads)
        for county, fids in zip(fragments["county"], fragments["fid_list"], strict=True)
    ]
    fragments = pd.concat(
        [fragments.reset_index(drop=True), pd.DataFrame(profiles)], axis=1
    )
    fragments["observed_start"] = fragments["start_epoch"].map(_local_timestamp)
    fragments["observed_end"] = fragments["end_epoch"].map(_local_timestamp)

    rows: list[dict[str, object]] = []
    for source_key, group in fragments.groupby("source_key", sort=False):
        ordered = group.sort_values(["observed_start", "county", "matched_trip_id"])
        first = ordered.sort_values("start_epoch").iloc[0]
        last = ordered.sort_values("end_epoch").iloc[-1]
        start_time = _local_timestamp(first["start_epoch"])
        end_time = _local_timestamp(last["end_epoch"])
        duration_seconds = max((end_time - start_time).total_seconds(), 0.0)
        distance_m = float(ordered["distance_m"].sum())
        controlled_m = float(ordered["controlled_distance_m"].sum())
        arterial_m = float(ordered["arterial_distance_m"].sum())
        local_m = float(ordered["local_distance_m"].sum())
        surface_m = float(ordered["surface_distance_m"].sum())
        toll_m = float(ordered["toll_distance_m"].sum())
        county_fragment_sequences = [
            {"county": str(record.county), "fids": list(record.fid_list)}
            for record in ordered.itertuples(index=False)
        ]
        scoped_fid_sequence = [
            {"county": fragment["county"], "fid": int(fid)}
            for fragment in county_fragment_sequences
            for fid in fragment["fids"]
        ]
        deduplicated_scoped_fids: list[dict[str, object]] = []
        for item in scoped_fid_sequence:
            if not deduplicated_scoped_fids or item != deduplicated_scoped_fids[-1]:
                deduplicated_scoped_fids.append(item)
        road_names = collapse_adjacent(
            road
            for names in ordered["road_names"]
            for road in (names if isinstance(names, list) else [])
        )
        major_road_names: list[str] = []
        for item in deduplicated_scoped_fids:
            info = roads.get((str(item["county"]), int(item["fid"])), {})
            road_class = _clean_text(info.get("highway")).split("|")[0].lower()
            if road_class not in CONTROLLED_ACCESS_CLASSES | ARTERIAL_CLASSES:
                continue
            name = _clean_text(info.get("name")) or _clean_text(
                info.get("FDOT_ROADWAY")
            )
            if name and (not major_road_names or major_road_names[-1] != name):
                major_road_names.append(name)
        session_id = str(first["session_id"])
        source_trip_id = str(first["source_trip_id"])
        trip_id = _stable_slug("driver_1003", session_id, source_trip_id)
        speed_mph = (
            distance_m / duration_seconds * 2.2369362920544
            if duration_seconds > 0
            else float("nan")
        )
        rows.append(
            {
                "trip_id": trip_id,
                "session_id": session_id,
                "driver_id": DRIVER_ID,
                "start_timestamp": start_time.isoformat(),
                "end_timestamp": end_time.isoformat(),
                "start_latitude": float(first["start_latitude"]),
                "start_longitude": float(first["start_longitude"]),
                "end_latitude": float(last["end_latitude"]),
                "end_longitude": float(last["end_longitude"]),
                "trip_duration_seconds": duration_seconds,
                "trip_duration_minutes": duration_seconds / 60.0,
                "day_of_week": start_time.day_name(),
                "month": start_time.strftime("%Y-%m"),
                "weekday_weekend": "weekday" if start_time.dayofweek < 5 else "weekend",
                "start_hour": start_time.hour + start_time.minute / 60.0,
                "end_hour": end_time.hour + end_time.minute / 60.0,
                "origin_county": str(first["county"]),
                "destination_county": str(last["county"]),
                "counties_traversed": _json_dumps(list(dict.fromkeys(ordered["county"].astype(str)))),
                "matched_fid_sequence": _json_dumps(scoped_fid_sequence),
                "deduplicated_fid_sequence": _json_dumps(deduplicated_scoped_fids),
                "county_fragment_fid_sequences": _json_dumps(county_fragment_sequences),
                "matched_road_name_sequence": _json_dumps(road_names),
                "major_road_name_sequence": _json_dumps(major_road_names),
                "route_distance_m": distance_m,
                "route_distance_km": distance_m / 1000.0,
                "highway_distance_m": controlled_m,
                "highway_share": controlled_m / distance_m if distance_m else float("nan"),
                "arterial_distance_m": arterial_m,
                "arterial_share": arterial_m / distance_m if distance_m else float("nan"),
                "local_road_distance_m": local_m,
                "local_road_share": local_m / distance_m if distance_m else float("nan"),
                "surface_street_distance_m": surface_m,
                "surface_street_share": surface_m / distance_m if distance_m else float("nan"),
                "toll_road_usage": bool(toll_m > 0),
                "toll_road_distance_m": toll_m,
                "average_speed_mph": speed_mph,
                "travel_direction": "ordered directed county/FID sequence (u-to-v network orientation)",
                "source_fragment_count": int(len(ordered)),
                "source_fragment_trip_ids": _json_dumps(ordered["trip_id"].astype(str).tolist()),
                "source_fragment_matched_ids": _json_dumps(
                    [
                        {"county": str(record.county), "matched_trip_id": int(record.matched_trip_id)}
                        for record in ordered.itertuples(index=False)
                    ]
                ),
                "source_file_references": _json_dumps(
                    sorted(
                        {
                            _clean_text(value)
                            for value in ordered.get(
                                "source_gps_path", pd.Series(dtype="object")
                            )
                            if _clean_text(value)
                        }
                    )
                ),
                "segmentation_method": "explicit source GPS recording; county fragments stitched by session_id/source_trip_id",
                "source_key": source_key,
            }
        )
    trips = pd.DataFrame(rows).sort_values(["start_timestamp", "trip_id"]).reset_index(
        drop=True
    )
    if trips["trip_id"].duplicated().any():
        raise BehaviorAnalysisError("Stitched source-trip IDs are not unique")
    if len(trips) != fragments["source_key"].nunique():
        raise BehaviorAnalysisError("Source-trip count does not reconcile after stitching")
    observed_months = {
        month: index + 1
        for index, month in enumerate(sorted(trips["month"].dropna().astype(str).unique()))
    }
    trips["observed_month_index"] = trips["month"].map(observed_months).astype("Int64")
    trips["moving_duration_seconds"] = float("nan")
    trips["stopped_duration_seconds"] = float("nan")
    trips["data_quality_flags"] = trips.apply(
        lambda row: _json_dumps(
            [
                flag
                for flag, active in (
                    ("nonpositive_duration", row["trip_duration_seconds"] <= 0),
                    ("no_matched_route", row["route_distance_m"] <= 0),
                    ("average_speed_unavailable", pd.isna(row["average_speed_mph"])),
                    ("cross_county_trip", row["source_fragment_count"] > 1),
                )
                if active
            ]
        ),
        axis=1,
    )
    return trips, fragments


def build_endpoint_events(trips: pd.DataFrame) -> pd.DataFrame:
    """Return one start and one end event per stitched source trip."""
    start = pd.DataFrame(
        {
            "trip_id": trips["trip_id"],
            "endpoint_role": "origin",
            "latitude": trips["start_latitude"],
            "longitude": trips["start_longitude"],
            "event_timestamp": trips["start_timestamp"],
            "county": trips["origin_county"],
            "month": trips["month"],
        }
    )
    end = pd.DataFrame(
        {
            "trip_id": trips["trip_id"],
            "endpoint_role": "destination",
            "latitude": trips["end_latitude"],
            "longitude": trips["end_longitude"],
            "event_timestamp": trips["end_timestamp"],
            "county": trips["destination_county"],
            "month": pd.to_datetime(trips["end_timestamp"], utc=True)
            .dt.tz_convert(LOCAL_TIMEZONE)
            .dt.strftime("%Y-%m"),
        }
    )
    endpoints = pd.concat([start, end], ignore_index=True)
    endpoints["event_timestamp"] = pd.to_datetime(
        endpoints["event_timestamp"], errors="coerce", utc=True
    ).dt.tz_convert(LOCAL_TIMEZONE)
    endpoints = endpoints.dropna(subset=["latitude", "longitude", "event_timestamp"])
    transformer = Transformer.from_crs(WGS84, PROJECTED_CRS, always_xy=True)
    endpoints["x"], endpoints["y"] = transformer.transform(
        endpoints["longitude"].to_numpy(dtype=float),
        endpoints["latitude"].to_numpy(dtype=float),
    )
    endpoints["event_date"] = endpoints["event_timestamp"].dt.date.astype(str)
    endpoints["event_week"] = endpoints["event_timestamp"].dt.strftime("%G-W%V")
    endpoints["hour"] = (
        endpoints["event_timestamp"].dt.hour
        + endpoints["event_timestamp"].dt.minute / 60.0
    )
    endpoints["is_weekday"] = endpoints["event_timestamp"].dt.dayofweek < 5
    return endpoints.reset_index(drop=True)


def dbscan_projected(
    x: Sequence[float],
    y: Sequence[float],
    *,
    eps_m: float,
    min_samples: int = CLUSTER_MIN_SAMPLES,
) -> list[int]:
    """Dependency-free DBSCAN using a metre-based spatial hash."""
    size = len(x)
    if size == 0:
        return []
    if eps_m <= 0 or min_samples <= 0:
        raise ValueError("eps_m and min_samples must be positive")
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (x_value, y_value) in enumerate(zip(x, y, strict=True)):
        cells[(math.floor(x_value / eps_m), math.floor(y_value / eps_m))].append(index)
    cache: dict[int, list[int]] = {}
    eps_squared = eps_m * eps_m

    def neighbors(index: int) -> list[int]:
        if index in cache:
            return cache[index]
        cell_x = math.floor(x[index] / eps_m)
        cell_y = math.floor(y[index] / eps_m)
        candidates: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates.extend(cells.get((cell_x + dx, cell_y + dy), []))
        result = [
            candidate
            for candidate in candidates
            if (x[index] - x[candidate]) ** 2 + (y[index] - y[candidate]) ** 2
            <= eps_squared
        ]
        cache[index] = result
        return result

    unvisited, noise = -99, -1
    labels = [unvisited] * size
    cluster_id = 0
    for index in range(size):
        if labels[index] != unvisited:
            continue
        nearby = neighbors(index)
        if len(nearby) < min_samples:
            labels[index] = noise
            continue
        labels[index] = cluster_id
        queue: deque[int] = deque(nearby)
        queued = set(nearby)
        while queue:
            candidate = queue.popleft()
            if labels[candidate] == noise:
                labels[candidate] = cluster_id
            if labels[candidate] != unvisited:
                continue
            labels[candidate] = cluster_id
            candidate_neighbors = neighbors(candidate)
            if len(candidate_neighbors) >= min_samples:
                for neighbor in candidate_neighbors:
                    if neighbor not in queued:
                        queued.add(neighbor)
                        queue.append(neighbor)
        cluster_id += 1
    return labels


def _cluster_diagnostics(
    endpoints: pd.DataFrame,
    labels: Sequence[int],
    radius_m: float,
) -> dict[str, float]:
    data = endpoints[["x", "y", "event_date"]].copy()
    data["label"] = labels
    assigned = data.loc[data["label"] >= 0]
    cluster_count = int(assigned["label"].nunique())
    if assigned.empty:
        return {
            "radius_m": radius_m,
            "cluster_count": 0,
            "assignment_share": 0.0,
            "recurring_assignment_share": 0.0,
            "median_p95_radius_m": float("nan"),
            "p90_p95_radius_m": float("nan"),
            "largest_cluster_share": 0.0,
            "overmerge_share": 1.0,
            "selection_score": -1.0,
        }
    spreads: list[float] = []
    recurring_points = 0
    overmerged_points = 0
    sizes: list[int] = []
    for _, group in assigned.groupby("label"):
        distances = np.hypot(
            group["x"].to_numpy() - float(group["x"].mean()),
            group["y"].to_numpy() - float(group["y"].mean()),
        )
        p95 = float(np.percentile(distances, 95))
        spreads.append(p95)
        size = len(group)
        sizes.append(size)
        if group["event_date"].nunique() >= 3:
            recurring_points += size
        if p95 > max(250.0, radius_m * 2.5):
            overmerged_points += size
    assignment = len(assigned) / len(data)
    recurring_assignment = recurring_points / len(data)
    largest_share = max(sizes) / len(data)
    overmerge_share = overmerged_points / max(len(assigned), 1)
    median_spread = float(np.median(spreads))
    p90_spread = float(np.percentile(spreads, 90))
    compactness = max(0.0, 1.0 - min(p90_spread / 300.0, 1.0))
    giant_penalty = max(0.0, largest_share - 0.35)
    score = (
        0.40 * recurring_assignment
        + 0.25 * assignment
        + 0.25 * compactness
        - 0.30 * overmerge_share
        - 0.50 * giant_penalty
    )
    return {
        "radius_m": radius_m,
        "cluster_count": float(cluster_count),
        "assignment_share": assignment,
        "recurring_assignment_share": recurring_assignment,
        "median_p95_radius_m": median_spread,
        "p90_p95_radius_m": p90_spread,
        "largest_cluster_share": largest_share,
        "overmerge_share": overmerge_share,
        "selection_score": score,
    }


def select_cluster_radius(
    endpoints: pd.DataFrame,
    radii_m: Sequence[float] = CANDIDATE_RADII_M,
    *,
    min_samples: int = CLUSTER_MIN_SAMPLES,
) -> ClusterSelection:
    """Evaluate several radii and select the best compact recurring solution."""
    if endpoints.empty:
        raise BehaviorAnalysisError("Cannot cluster an empty endpoint table")
    diagnostics: list[dict[str, float]] = []
    for radius in radii_m:
        labels = dbscan_projected(
            endpoints["x"].to_numpy(dtype=float),
            endpoints["y"].to_numpy(dtype=float),
            eps_m=float(radius),
            min_samples=min_samples,
        )
        diagnostics.append(_cluster_diagnostics(endpoints, labels, float(radius)))
    frame = pd.DataFrame(diagnostics).sort_values("radius_m").reset_index(drop=True)
    eligible = frame.loc[
        (frame["assignment_share"] >= 0.65)
        & (frame["p90_p95_radius_m"] <= 300.0)
        & (frame["overmerge_share"] <= 0.15)
    ]
    candidates = eligible if not eligible.empty else frame
    best_score = candidates["selection_score"].max()
    # Prefer the smallest radius within two score points of the best. This
    # explicitly resists merging adjacent tenants/campuses for trivial coverage.
    close = candidates.loc[candidates["selection_score"] >= best_score - 0.02]
    selected = float(close.sort_values("radius_m").iloc[0]["radius_m"])
    frame["selected"] = frame["radius_m"].eq(selected)
    return ClusterSelection(selected, frame)


def assign_location_clusters(
    endpoints: pd.DataFrame,
    radius_m: float,
    *,
    min_samples: int = CLUSTER_MIN_SAMPLES,
) -> pd.DataFrame:
    """Assign stable visit-ranked cluster IDs at the selected radius."""
    output = endpoints.copy()
    output["cluster_internal"] = dbscan_projected(
        output["x"].to_numpy(dtype=float),
        output["y"].to_numpy(dtype=float),
        eps_m=radius_m,
        min_samples=min_samples,
    )
    counts = (
        output.loc[output["cluster_internal"] >= 0, "cluster_internal"]
        .value_counts()
        .rename_axis("cluster_internal")
        .reset_index(name="endpoint_count")
        .sort_values(["endpoint_count", "cluster_internal"], ascending=[False, True])
        .reset_index(drop=True)
    )
    counts["cluster_id"] = [f"C{index:03d}" for index in range(1, len(counts) + 1)]
    mapping = dict(zip(counts["cluster_internal"], counts["cluster_id"], strict=True))
    output["cluster_id"] = output["cluster_internal"].map(mapping).fillna("UNCLUSTERED")
    return output


def add_cluster_stability_diagnostics(
    clusters: pd.DataFrame,
    endpoints: pd.DataFrame,
    selection: ClusterSelection,
    *,
    min_samples: int = CLUSTER_MIN_SAMPLES,
) -> pd.DataFrame:
    """Attach radius-sensitivity evidence and explicit cluster-quality flags."""
    stability = analyze_cluster_stability(
        endpoints,
        candidate_radii_m=tuple(selection.diagnostics["radius_m"].astype(float)),
        selected_radius_m=selection.radius_m,
        min_samples=min_samples,
        clusterer=dbscan_projected,
        important_endpoint_threshold=20,
    ).rename(
        columns={
            "stability_status": "cluster_stability_status",
            "stability_evidence_json": "cluster_stability_evidence_json",
            "data_quality_flags": "cluster_stability_flags_json",
        }
    )
    stability = stability.drop(columns=["selected_radius_m"], errors="ignore")
    output = clusters.merge(
        stability,
        on="cluster_id",
        how="left",
        validate="one_to_one",
    )

    def flags_for(row: Mapping[str, object]) -> str:
        flags: set[str] = set()
        try:
            decoded = json.loads(
                _clean_text(row.get("cluster_stability_flags_json")) or "[]"
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = []
        if isinstance(decoded, list):
            flags.update(str(flag) for flag in decoded if str(flag))
        if _safe_float(row.get("coordinate_spread_m"), 0.0) > 150:
            flags.add("wide_endpoint_spread")
        if int(_safe_float(row.get("destination_count"), 0.0)) < 3:
            flags.add("few_destination_arrivals")
        if int(_safe_float(row.get("valid_stay_count"), 0.0)) < 5:
            flags.add("insufficient_measured_stays_for_major_role")
        if bool(row.get("is_important_cluster")) and _clean_text(
            row.get("cluster_stability_status")
        ) not in {"", "stable"}:
            flags.add("important_cluster_radius_sensitive")
        return _json_dumps(sorted(flags))

    output["data_quality_flags"] = [
        flags_for(record) for record in output.to_dict(orient="records")
    ]
    return output


def attach_clusters_to_trips(
    trips: pd.DataFrame,
    endpoints: pd.DataFrame,
) -> pd.DataFrame:
    """Attach origin/destination cluster IDs without duplicating trips."""
    origin = endpoints.loc[
        endpoints["endpoint_role"].eq("origin"), ["trip_id", "cluster_id"]
    ].rename(columns={"cluster_id": "origin_cluster_id"})
    destination = endpoints.loc[
        endpoints["endpoint_role"].eq("destination"), ["trip_id", "cluster_id"]
    ].rename(columns={"cluster_id": "destination_cluster_id"})
    if origin["trip_id"].duplicated().any() or destination["trip_id"].duplicated().any():
        raise BehaviorAnalysisError("Endpoint clustering produced duplicate trip roles")
    result = trips.merge(origin, on="trip_id", how="left", validate="one_to_one").merge(
        destination, on="trip_id", how="left", validate="one_to_one"
    )
    result[["origin_cluster_id", "destination_cluster_id"]] = result[
        ["origin_cluster_id", "destination_cluster_id"]
    ].fillna("UNCLUSTERED")
    return result


def _time_bucket(hour: float) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "nighttime"


def _format_hour(hour: float) -> str:
    if not math.isfinite(hour):
        return "unknown"
    minutes = int(round((hour % 24) * 60)) % (24 * 60)
    value = pd.Timestamp("2000-01-01") + pd.Timedelta(minutes=minutes)
    return value.strftime("%-I:%M %p")


def _recurrence_pattern(
    total_events: int,
    unique_days: int,
    unique_weeks: int,
    months: int,
    span_days: int,
) -> str:
    if unique_weeks <= 0 or span_days <= 0:
        return "irregular"
    calendar_weeks = max(span_days / 7.0, 1.0)
    events_per_week = total_events / calendar_weeks
    observed_day_share = unique_days / max(span_days, 1)
    if events_per_week >= 5 and observed_day_share >= 0.45 and unique_days >= 20:
        return "daily"
    if events_per_week >= 2:
        return "several times per week"
    if events_per_week >= 0.70:
        return "weekly"
    if events_per_week >= 0.35:
        return "biweekly"
    if months >= 3 and total_events / months >= 0.70:
        return "monthly"
    return "irregular"


def _nearest_osm_context(
    cluster_rows: pd.DataFrame,
    county_paths: Mapping[str, Mapping[str, Path]] = COUNTY_PATHS,
) -> dict[str, dict[str, object]]:
    """Return local cached OSM context without making external requests."""
    contexts: dict[str, dict[str, object]] = {}
    for county, county_clusters in cluster_rows.groupby("county", sort=True):
        path = county_paths.get(str(county), {}).get("landuse")
        if not path or not path.exists():
            continue
        landuse = gpd.read_parquet(path)
        if landuse.crs is None:
            landuse = landuse.set_crs(PROJECTED_CRS)
        elif str(landuse.crs) != PROJECTED_CRS:
            landuse = landuse.to_crs(PROJECTED_CRS)
        tag_columns = [
            column
            for column in (
                "name",
                "landuse",
                "residential",
                "amenity",
                "shop",
                "office",
                "education",
                "leisure",
                "tourism",
                "addr:city",
                "addr:postcode",
                "addr:street",
            )
            if column in landuse
        ]
        for row in county_clusters.itertuples(index=False):
            point = Point(float(row.centroid_x), float(row.centroid_y))
            distances = landuse.geometry.distance(point)
            nearby = landuse.loc[distances <= 250.0, [*tag_columns, "geometry"]].copy()
            if nearby.empty:
                contexts[str(row.cluster_id)] = {
                    "osm_context": "no named local OSM feature within 250 m",
                    "osm_names": [],
                    "osm_categories": [],
                    "residential_context_score": 0.0,
                    "osm_source": "OpenStreetMap local cache",
                }
                continue
            nearby["distance_m"] = distances.loc[nearby.index]
            nearby = nearby.sort_values("distance_m")
            names = collapse_adjacent(nearby.get("name", pd.Series(dtype=str)).tolist())[:8]
            categories: list[str] = []
            residential = 0.0
            for record in nearby.itertuples(index=False):
                values = {
                    column: _clean_text(getattr(record, column.replace(":", "_"), ""))
                    for column in tag_columns
                }
                joined = " ".join(values.values()).lower()
                for category in (
                    "residential",
                    "retail",
                    "commercial",
                    "education",
                    "school",
                    "healthcare",
                    "hospital",
                    "clinic",
                    "recreation",
                    "park",
                    "industrial",
                    "restaurant",
                ):
                    if category in joined:
                        categories.append(category)
                if any(token in joined for token in ("residential", "apartments", "housing")):
                    residential = max(residential, 1.0)
            categories = collapse_adjacent(categories)
            contexts[str(row.cluster_id)] = {
                "osm_context": "; ".join(names or categories)
                or "unnamed local OSM land-use context",
                "osm_names": names,
                "osm_categories": categories,
                "residential_context_score": residential,
                "osm_source": "OpenStreetMap local cache",
            }
    return contexts


def summarize_location_clusters(
    trips: pd.DataFrame,
    endpoints: pd.DataFrame,
    *,
    selected_radius_m: float,
    county_paths: Mapping[str, Mapping[str, Path]] = COUNTY_PATHS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build required cluster evidence and transparent home scores."""
    assigned = endpoints.loc[endpoints["cluster_id"].ne("UNCLUSTERED")].copy()
    if assigned.empty:
        raise BehaviorAnalysisError("No endpoint cluster survived DBSCAN")
    endpoint_road_rows: list[dict[str, str]] = []
    for trip in trips[["trip_id", "matched_road_name_sequence"]].itertuples(index=False):
        try:
            road_names = json.loads(trip.matched_road_name_sequence)
        except (TypeError, ValueError, json.JSONDecodeError):
            road_names = []
        if road_names:
            endpoint_road_rows.extend(
                [
                    {
                        "trip_id": str(trip.trip_id),
                        "endpoint_role": "origin",
                        "endpoint_road_name": str(road_names[0]),
                    },
                    {
                        "trip_id": str(trip.trip_id),
                        "endpoint_role": "destination",
                        "endpoint_road_name": str(road_names[-1]),
                    },
                ]
            )
    if endpoint_road_rows:
        assigned = assigned.merge(
            pd.DataFrame(endpoint_road_rows),
            on=["trip_id", "endpoint_role"],
            how="left",
            validate="one_to_one",
        )
    else:
        assigned["endpoint_road_name"] = ""
    stays = reconstruct_stays(trips, timezone_name=LOCAL_TIMEZONE)
    stay_summary = summarize_cluster_stays(stays)
    stay_lookup = stay_summary.set_index("cluster_id").to_dict(orient="index")

    geometry_rows: list[dict[str, object]] = []
    for cluster_id, group in assigned.groupby("cluster_id", sort=True):
        centroid_x = float(group["x"].mean())
        centroid_y = float(group["y"].mean())
        squared = (group["x"] - centroid_x) ** 2 + (group["y"] - centroid_y) ** 2
        medoid = group.loc[squared.idxmin()]
        distances = np.sqrt(squared.to_numpy(dtype=float))
        geometry_rows.append(
            {
                "cluster_id": cluster_id,
                "county": group["county"].mode().iloc[0],
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "centroid_lat": float(group["latitude"].mean()),
                "centroid_lon": float(group["longitude"].mean()),
                "medoid_lat": float(medoid["latitude"]),
                "medoid_lon": float(medoid["longitude"]),
                "coordinate_spread_m": float(np.percentile(distances, 95)),
                "max_coordinate_spread_m": float(np.max(distances)),
            }
        )
    geometry = pd.DataFrame(geometry_rows)
    osm_context = _nearest_osm_context(geometry, county_paths=county_paths)

    observed_months = max(trips["month"].nunique(), 1)
    observed_weeks = max(
        pd.to_datetime(trips["start_timestamp"], utc=True)
        .dt.tz_convert(LOCAL_TIMEZONE)
        .dt.strftime("%G-W%V")
        .nunique(),
        1,
    )
    rows: list[dict[str, object]] = []
    for cluster_id, group in assigned.groupby("cluster_id", sort=True):
        origins = group.loc[group["endpoint_role"].eq("origin")]
        destinations = group.loc[group["endpoint_role"].eq("destination")]
        activity_events = destinations if not destinations.empty else origins
        arrival_hours = destinations["hour"]
        departure_hours = origins["hour"]
        arrival_buckets = Counter(_time_bucket(float(hour)) for hour in arrival_hours)
        departure_buckets = Counter(_time_bucket(float(hour)) for hour in departure_hours)
        total = len(group)
        unique_days = group["event_date"].nunique()
        unique_weeks = group["event_week"].nunique()
        months = sorted(group["month"].dropna().astype(str).unique())
        activity_first_date = pd.Timestamp(activity_events["event_date"].min())
        activity_last_date = pd.Timestamp(activity_events["event_date"].max())
        activity_span_days = max(
            int((activity_last_date - activity_first_date).days) + 1, 1
        )
        activity_unique_days = int(activity_events["event_date"].nunique())
        activity_unique_weeks = int(activity_events["event_week"].nunique())
        activity_months = int(activity_events["month"].nunique())
        stay_evidence = stay_lookup.get(str(cluster_id), {})
        connected_origins = (
            trips.loc[trips["destination_cluster_id"].eq(cluster_id), "origin_cluster_id"]
            .loc[lambda value: value.ne("UNCLUSTERED")]
            .value_counts()
            .head(5)
            .to_dict()
        )
        connected_destinations = (
            trips.loc[trips["origin_cluster_id"].eq(cluster_id), "destination_cluster_id"]
            .loc[lambda value: value.ne("UNCLUSTERED")]
            .value_counts()
            .head(5)
            .to_dict()
        )
        context = osm_context.get(
            str(cluster_id),
            {
                "osm_context": "no local OSM context",
                "osm_names": [],
                "osm_categories": [],
                "residential_context_score": 0.0,
                "osm_source": "OpenStreetMap local cache",
            },
        )
        top_endpoint_roads = (
            group["endpoint_road_name"]
            .dropna()
            .astype(str)
            .loc[lambda values: values.str.len() > 0]
            .value_counts()
            .head(8)
            .to_dict()
        )
        row = {
            "cluster_id": cluster_id,
            "selected_radius_m": selected_radius_m,
            "centroid_lat": float(group["latitude"].mean()),
            "centroid_lon": float(group["longitude"].mean()),
            "medoid_lat": float(
                geometry.loc[geometry["cluster_id"].eq(cluster_id), "medoid_lat"].iloc[0]
            ),
            "medoid_lon": float(
                geometry.loc[geometry["cluster_id"].eq(cluster_id), "medoid_lon"].iloc[0]
            ),
            "coordinate_spread_m": float(
                geometry.loc[
                    geometry["cluster_id"].eq(cluster_id), "coordinate_spread_m"
                ].iloc[0]
            ),
            "county": group["county"].mode().iloc[0],
            "origin_count": len(origins),
            "destination_count": len(destinations),
            "total_visit_count": total,
            "activity_visit_count": len(activity_events),
            "activity_unique_days": activity_unique_days,
            "activity_unique_weeks": activity_unique_weeks,
            "activity_months_visited": activity_months,
            "recurrence_basis": (
                "destination_arrivals"
                if not destinations.empty
                else "origin_only_fallback"
            ),
            "unique_days_visited": unique_days,
            "unique_weeks_visited": unique_weeks,
            "months_visited": len(months),
            "months_list_json": _json_dumps(months),
            "first_date_observed": group["event_date"].min(),
            "last_date_observed": group["event_date"].max(),
            "weekday_share": float(group["is_weekday"].mean()),
            "weekend_share": float((~group["is_weekday"]).mean()),
            "morning_arrival_count": int(arrival_buckets["morning"]),
            "afternoon_arrival_count": int(arrival_buckets["afternoon"]),
            "evening_arrival_count": int(arrival_buckets["evening"]),
            "nighttime_arrival_count": int(arrival_buckets["nighttime"]),
            "typical_arrival_time": _format_hour(
                float(arrival_hours.median()) if len(arrival_hours) else float("nan")
            ),
            "typical_departure_time": _format_hour(
                float(departure_hours.median()) if len(departure_hours) else float("nan")
            ),
            "typical_arrival_hour": float(arrival_hours.median())
            if len(arrival_hours)
            else float("nan"),
            "typical_departure_hour": float(departure_hours.median())
            if len(departure_hours)
            else float("nan"),
            "arrival_time_distribution": _json_dumps(dict(arrival_buckets)),
            "departure_time_distribution": _json_dumps(dict(departure_buckets)),
            "valid_stay_count": int(stay_evidence.get("valid_stay_count", 0)),
            "median_dwell_minutes": stay_evidence.get("median_dwell_minutes", float("nan")),
            "dwell_q25_minutes": stay_evidence.get("dwell_q25_minutes", float("nan")),
            "dwell_q75_minutes": stay_evidence.get("dwell_q75_minutes", float("nan")),
            "mean_dwell_minutes": stay_evidence.get("mean_dwell_minutes", float("nan")),
            "share_under_5_minutes": stay_evidence.get("share_under_5_minutes", 0.0),
            "share_5_to_20_minutes": stay_evidence.get("share_5_to_20_minutes", 0.0),
            "share_20_to_60_minutes": stay_evidence.get("share_20_to_60_minutes", 0.0),
            "share_1_to_3_hours": stay_evidence.get("share_1_to_3_hours", 0.0),
            "share_over_3_hours": stay_evidence.get("share_over_3_hours", 0.0),
            "measured_overnight_stay_count": int(
                stay_evidence.get("measured_overnight_stay_count", 0)
            ),
            "censored_continuity_count": int(
                stay_evidence.get("censored_continuity_count", 0)
            ),
            "censored_overnight_association_count": int(
                stay_evidence.get("censored_overnight_association_count", 0)
            ),
            "micro_stop_boundary_count": int(
                stay_evidence.get("micro_stop_boundary_count", 0)
            ),
            "weekday_median_dwell_minutes": stay_evidence.get(
                "weekday_median_dwell_minutes", float("nan")
            ),
            "weekend_median_dwell_minutes": stay_evidence.get(
                "weekend_median_dwell_minutes", float("nan")
            ),
            "dwell_observation_count": int(stay_evidence.get("valid_stay_count", 0)),
            "overnight_association_count": int(
                stay_evidence.get("measured_overnight_stay_count", 0)
                + stay_evidence.get("censored_overnight_association_count", 0)
            ),
            "recurring_frequency": _recurrence_pattern(
                len(activity_events),
                activity_unique_days,
                activity_unique_weeks,
                activity_months,
                activity_span_days,
            ),
            "top_origin_clusters_connected": _json_dumps(connected_origins),
            "top_destination_clusters_connected": _json_dumps(connected_destinations),
            "top_endpoint_roads_json": _json_dumps(top_endpoint_roads),
            "osm_context": context["osm_context"],
            "osm_names_json": _json_dumps(context["osm_names"]),
            "osm_categories_json": _json_dumps(context["osm_categories"]),
            "osm_context_source": context["osm_source"],
            "residential_context_score": float(context["residential_context_score"]),
            "privacy_flag": "NONE",
        }
        rows.append(row)
    clusters = pd.DataFrame(rows)

    max_endpoints = max(float(clusters["total_visit_count"].max()), 1.0)
    max_overnight = max(float(clusters["overnight_association_count"].max()), 1.0)
    morning_maximum = (
        assigned.loc[
            assigned["endpoint_role"].eq("origin")
            & assigned["hour"].between(5, 10, inclusive="left")
        ]
        .groupby("cluster_id")
        .size()
        .max()
    )
    max_morning = max(
        float(morning_maximum) if pd.notna(morning_maximum) else 0.0,
        1.0,
    )
    morning_counts = (
        assigned.loc[
            assigned["endpoint_role"].eq("origin")
            & assigned["hour"].between(5, 10, inclusive="left")
        ]
        .groupby("cluster_id")
        .size()
        .to_dict()
    )
    evening_counts = (
        assigned.loc[
            assigned["endpoint_role"].eq("destination")
            & ((assigned["hour"] >= 19) | (assigned["hour"] < 5))
        ]
        .groupby("cluster_id")
        .size()
        .to_dict()
    )
    scores: list[dict[str, float]] = []
    for row in clusters.itertuples(index=False):
        morning_count = float(morning_counts.get(row.cluster_id, 0))
        evening_count = float(evening_counts.get(row.cluster_id, 0))
        morning_score = min(morning_count / max_morning, 1.0)
        evening_score = min(evening_count / max(float(row.destination_count), 1.0), 1.0)
        overnight_score = min(float(row.overnight_association_count) / max_overnight, 1.0)
        recurrence_score = 0.5 * min(row.months_visited / observed_months, 1.0) + 0.5 * min(
            row.unique_weeks_visited / observed_weeks, 1.0
        )
        residential_score = float(row.residential_context_score)
        centrality_score = min(float(row.total_visit_count) / max_endpoints, 1.0)
        home_score = (
            0.15 * morning_score
            + 0.25 * evening_score
            + 0.25 * overnight_score
            + 0.15 * recurrence_score
            + 0.10 * residential_score
            + 0.10 * centrality_score
        )
        scores.append(
            {
                "morning_origin_score": morning_score,
                "evening_destination_score": evening_score,
                "overnight_score": overnight_score,
                "recurrence_score": recurrence_score,
                "network_centrality_score": centrality_score,
                "home_score": home_score,
            }
        )
    clusters = pd.concat([clusters.reset_index(drop=True), pd.DataFrame(scores)], axis=1)
    eligible = clusters.loc[
        (clusters["months_visited"] >= min(6, observed_months))
        & (clusters["total_visit_count"] >= 20)
    ]
    if eligible.empty:
        eligible = clusters
    home_index = eligible["home_score"].idxmax()
    clusters.loc[home_index, "privacy_flag"] = "HOME_SENSITIVE"
    clusters["home_candidate_rank"] = clusters["home_score"].rank(
        method="first", ascending=False
    ).astype(int)
    clusters["home_score_evidence"] = clusters.apply(
        lambda row: (
            f"morning-origin {row.morning_origin_score:.2f}; evening-destination "
            f"{row.evening_destination_score:.2f}; overnight {row.overnight_score:.2f}; "
            f"recurrence {row.recurrence_score:.2f}; residential context "
            f"{row.residential_context_score:.2f}; OD centrality {row.network_centrality_score:.2f}"
        ),
        axis=1,
    )
    clusters = clusters.sort_values(
        ["total_visit_count", "cluster_id"], ascending=[False, True]
    ).reset_index(drop=True)
    return clusters, stays


def generalized_home_point(lat: float, lon: float) -> tuple[float, float]:
    """Return a deterministic neighborhood-level point, never the true point."""
    # A 0.01-degree grid is roughly one kilometre north/south in South Florida.
    # The small deterministic offset prevents a raw point that happens to lie on
    # the grid from surviving unchanged.
    digest = hashlib.sha256(f"driver-1003-home-{lat:.3f}-{lon:.3f}".encode()).digest()
    lat_offset = 0.0025 if digest[0] % 2 else -0.0025
    lon_offset = 0.0025 if digest[1] % 2 else -0.0025
    return round(lat, 2) + lat_offset, round(lon, 2) + lon_offset


def _place_name(place: Mapping[str, Any]) -> str:
    display = place.get("displayName")
    if isinstance(display, Mapping):
        return _clean_text(display.get("text"))
    return _clean_text(display)


def _place_location(place: Mapping[str, Any]) -> tuple[float, float] | None:
    location = place.get("location")
    if not isinstance(location, Mapping):
        return None
    lat = _safe_float(location.get("latitude"))
    lon = _safe_float(location.get("longitude"))
    if not math.isfinite(lat) or not math.isfinite(lon):
        return None
    return lat, lon


def _geocode_context(payload: Mapping[str, Any]) -> dict[str, str]:
    results = payload.get("results", [])
    if not isinstance(results, list) or not results:
        return {
            "formatted_address": "",
            "neighborhood": "",
            "city": "",
            "postal_code": "",
            "route": "",
        }
    result = results[0] if isinstance(results[0], Mapping) else {}
    components = result.get("address_components", [])
    by_type: dict[str, str] = {}
    if isinstance(components, list):
        for component in components:
            if not isinstance(component, Mapping):
                continue
            name = _clean_text(component.get("long_name"))
            types = component.get("types", [])
            if not isinstance(types, list):
                continue
            for component_type in types:
                by_type.setdefault(str(component_type), name)
    neighborhood = next(
        (
            by_type.get(key, "")
            for key in (
                "neighborhood",
                "sublocality_level_1",
                "sublocality",
                "administrative_area_level_3",
            )
            if by_type.get(key)
        ),
        "",
    )
    city = next(
        (
            by_type.get(key, "")
            for key in ("locality", "postal_town", "administrative_area_level_2")
            if by_type.get(key)
        ),
        "",
    )
    return {
        "formatted_address": _clean_text(result.get("formatted_address")),
        "neighborhood": neighborhood,
        "city": city,
        "postal_code": by_type.get("postal_code", ""),
        "route": by_type.get("route", ""),
    }


def _candidate_record(
    place: Mapping[str, Any],
    *,
    cluster_lat: float,
    cluster_lon: float,
    source: str = "Google Places API (New)",
    search_radius_m: int | None = None,
    retrieved_at_utc: str = "",
    cache_hit: bool | None = None,
) -> dict[str, object]:
    location = _place_location(place)
    distance = (
        haversine_m(cluster_lat, cluster_lon, location[0], location[1])
        if location
        else float("nan")
    )
    types = place.get("types", [])
    types = [str(value) for value in types] if isinstance(types, list) else []
    record: dict[str, object] = {
        "name": _place_name(place),
        "primary_type": _clean_text(place.get("primaryType")),
        "types": types,
        "address": _clean_text(place.get("formattedAddress")),
        "latitude": location[0] if location else float("nan"),
        "longitude": location[1] if location else float("nan"),
        "distance_m": distance,
        "business_status": _clean_text(place.get("businessStatus")),
        "google_maps_uri": _clean_text(place.get("googleMapsUri")),
        "source": source,
        "search_radius_m": search_radius_m,
        "retrieved_at_utc": retrieved_at_utc or "unknown",
        "cache_hit": cache_hit,
    }
    record["match_quality_score"] = _candidate_match_score(record)
    return record


def _candidate_identity(place: Mapping[str, Any]) -> str:
    """Return a stable, non-secret identity for staged candidate provenance."""
    location = _place_location(place)
    coordinate = (
        f"{location[0]:.7f},{location[1]:.7f}" if location is not None else "unknown"
    )
    return "|".join(
        (
            _place_name(place).casefold(),
            _clean_text(place.get("formattedAddress")).casefold(),
            coordinate,
        )
    )


def _candidate_match_score(candidate: Mapping[str, Any]) -> float:
    """Score geographic/listing plausibility without treating it as a visit."""
    distance = _safe_float(candidate.get("distance_m"), 999_999.0)
    distance_score = max(0.0, 1.0 - min(distance, 500.0) / 500.0)
    primary = _clean_text(candidate.get("primary_type"))
    broad_types = {
        "shopping_mall",
        "hospital",
        "university",
        "school",
        "airport",
        "transit_station",
        "medical_center",
    }
    preferred_types = {
        "shopping_mall",
        "grocery_store",
        "supermarket",
        "hospital",
        "medical_center",
        "school",
        "university",
    }
    broad_bonus = 0.12 if primary in broad_types else 0.0
    preferred_bonus = 0.08 if primary in preferred_types else 0.0
    active_bonus = (
        0.03 if _clean_text(candidate.get("business_status")) == "OPERATIONAL" else 0.0
    )
    return round(min(distance_score + broad_bonus + preferred_bonus + active_bonus, 1.0), 4)


def _strong_place_match(
    candidates: Sequence[Mapping[str, Any]],
    radius_m: int,
    *,
    cluster_lat: float,
    cluster_lon: float,
) -> bool:
    broad_types = {
        "shopping_mall",
        "hospital",
        "university",
        "school",
        "airport",
        "transit_station",
    }
    excluded_types = {"bus_stop", "taxi_stand"}
    for place in candidates:
        record = _candidate_record(
            place, cluster_lat=cluster_lat, cluster_lon=cluster_lon
        )
        distance = _safe_float(record["distance_m"], 999_999.0)
        if record["primary_type"] in excluded_types:
            continue
        if distance <= 60:
            return True
        if record["primary_type"] in broad_types and distance <= 175:
            return True
    return radius_m >= 500


def _select_place_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, object] | None, str, str]:
    """Choose a defensible candidate without assuming closest means visited."""
    excluded_types = {"bus_stop", "taxi_stand"}
    usable = [
        candidate
        for candidate in candidates
        if candidate.get("primary_type") not in excluded_types
        and _clean_text(candidate.get("name"))
    ]
    if not usable:
        return None, "unresolved", "No nearby listed place was returned."
    broad_types = {
        "shopping_mall",
        "hospital",
        "university",
        "school",
        "airport",
        "transit_station",
        "medical_center",
    }
    scored: list[tuple[float, Mapping[str, Any]]] = []
    for candidate in usable:
        score = _safe_float(candidate.get("match_quality_score"))
        if not math.isfinite(score):
            score = _candidate_match_score(candidate)
        scored.append((score, candidate))
    scored.sort(key=lambda item: (-item[0], _safe_float(item[1].get("distance_m"), 999_999)))
    best = dict(scored[0][1])
    distance = _safe_float(best.get("distance_m"), 999_999.0)
    close_competitors = [
        candidate
        for score, candidate in scored[1:]
        if abs(score - scored[0][0]) <= 0.08
        and _safe_float(candidate.get("distance_m"), 999_999.0) <= 125
    ]
    medical_types = {"doctor", "medical_clinic", "dentist", "physiotherapist"}
    medical_candidates = [
        candidate
        for _, candidate in scored
        if candidate.get("primary_type") in medical_types
        and _safe_float(candidate.get("distance_m"), 999_999.0) <= 125
    ]
    if len(medical_candidates) >= 3:
        return (
            None,
            "ambiguous",
            "Several medical-office tenants at similar distances were returned; the area is retained as a possible multi-tenant medical destination without selecting one clinician.",
        )
    broad = best.get("primary_type") in broad_types
    second_distance = (
        _safe_float(scored[1][1].get("distance_m"), 999_999.0)
        if len(scored) > 1
        else 999_999.0
    )
    best_root = re.sub(r"[^a-z0-9]+", "", _clean_text(best.get("name")).lower())
    competitors_same_entity = bool(close_competitors) and all(
        (
            (root := re.sub(r"[^a-z0-9]+", "", _clean_text(candidate.get("name")).lower()))
            and (best_root.startswith(root[:4]) or root.startswith(best_root[:4]))
        )
        for candidate in close_competitors
    )
    decisive_nearest = distance <= 10 and second_distance >= distance + 5
    if distance <= 40 and (not close_competitors or broad or competitors_same_entity or decisive_nearest):
        quality = "high"
    elif distance <= 125:
        quality = "medium"
    elif distance <= 300:
        quality = "low"
    else:
        quality = "unresolved"
    if close_competitors and not broad and not competitors_same_entity and not decisive_nearest:
        quality = "low"
        reasoning = (
            "Several similarly plausible nearby tenants were returned; the closest listing "
            "is retained only as a low-confidence candidate."
        )
    elif quality == "unresolved":
        reasoning = "No listed place was geographically close enough for a defensible match."
    else:
        reasoning = (
            f"The listed place is approximately {distance:.0f} m from the endpoint cluster; "
            "timing and recurrence are evaluated separately before assigning an activity role."
        )
    return best if quality != "unresolved" else None, quality, reasoning


def _enrichment_priority(clusters: pd.DataFrame) -> pd.Series:
    visits = pd.to_numeric(clusters["total_visit_count"], errors="coerce").fillna(0)
    months = pd.to_numeric(clusters["months_visited"], errors="coerce").fillna(0)
    days = pd.to_numeric(clusters["unique_days_visited"], errors="coerce").fillna(0)
    monthly_like = ((months >= 4) & (visits / months.clip(lower=1) <= 3)).astype(float)
    weekday = pd.to_numeric(clusters["weekday_share"], errors="coerce").fillna(0)
    arrival = pd.to_numeric(clusters["typical_arrival_hour"], errors="coerce")
    dwell = pd.to_numeric(clusters["median_dwell_minutes"], errors="coerce")
    school_like = (
        (weekday >= 0.80)
        & arrival.between(13.0, 16.5, inclusive="both")
        & dwell.between(5, 75, inclusive="both")
    ).astype(float)
    healthcare_like = (
        (months >= 4)
        & (visits / months.clip(lower=1) <= 4)
        & dwell.between(20, 240, inclusive="both")
    ).astype(float)
    valid_stays = pd.to_numeric(
        clusters.get("valid_stay_count", pd.Series(0, index=clusters.index)),
        errors="coerce",
    ).fillna(0)
    long_share = pd.to_numeric(
        clusters.get("share_over_3_hours", pd.Series(0, index=clusters.index)),
        errors="coerce",
    ).fillna(0)
    long_duration_like = (
        (valid_stays >= 10)
        & dwell.ge(180)
        & (long_share >= 0.50)
    ).astype(float)
    return (
        np.log1p(visits)
        + 0.35 * np.log1p(days)
        + 0.08 * months
        + monthly_like
        + 2.5 * school_like
        + 1.5 * healthcare_like
        + 2.0 * long_duration_like
    )


def enrich_location_clusters(
    clusters: pd.DataFrame,
    *,
    google_client: Any | None,
    max_non_home_clusters: int = 20,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Reverse geocode and search Places for unique meaningful clusters.

    The home candidate is reverse geocoded for neighborhood/city context only;
    Places search is never used to identify the residence. Exact home address
    and Google URI fields are discarded before this function returns.
    """
    output = clusters.copy()
    home_mask = output["privacy_flag"].eq("HOME_SENSITIVE")
    if home_mask.sum() != 1:
        raise BehaviorAnalysisError("Exactly one HOME_SENSITIVE cluster is required")
    non_home = output.loc[~home_mask].copy()
    non_home["_priority"] = _enrichment_priority(non_home)
    meaningful = non_home.loc[
        (non_home["total_visit_count"] >= 10)
        & (non_home["unique_days_visited"] >= 5)
        & (non_home["months_visited"] >= 3)
    ].nlargest(max_non_home_clusters, "_priority")
    query_ids = set(meaningful["cluster_id"])
    query_ids.update(output.loc[home_mask, "cluster_id"])

    enrichment_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    private_home_exact_address = ""
    google_active = google_client is not None
    for row in output.itertuples(index=False):
        is_home = row.privacy_flag == "HOME_SENSITIVE"
        geocoding_attempted = False
        places_search_attempted = False
        places_search_completed = False
        places_search_radius_m: int | None = None
        places_search_stop_reason = "not_attempted"
        context = {
            "formatted_address": "",
            "neighborhood": "",
            "city": "",
            "postal_code": "",
            "route": "",
        }
        geocode_source = "OpenStreetMap local cache"
        reverse_error = ""
        if row.cluster_id in query_ids and google_active:
            geocoding_attempted = True
            try:
                response = google_client.reverse_geocode(row.medoid_lat, row.medoid_lon)
                context = _geocode_context(response.payload)
                geocode_source = "Google Geocoding API"
            except Exception as exc:  # GoogleAPIError is intentionally sanitized.
                reverse_error = str(exc)
                error = getattr(exc, "to_dict", lambda: {"message": str(exc)})()
                errors.append({"cluster_id": row.cluster_id, "stage": "geocoding", **error})
                if str(error.get("category", "")) in {
                    "authorization",
                    "api_disabled",
                    "billing",
                    "quota",
                    "missing_key",
                }:
                    google_active = False

        neighborhood_city = ", ".join(
            value for value in (context["neighborhood"], context["city"]) if value
        )
        generalized = neighborhood_city or context["city"] or str(row.county)
        selected: dict[str, object] | None = None
        alternatives: list[dict[str, object]] = []
        match_quality = "unresolved"
        match_reason = "No external Places lookup was attempted for this lower-priority cluster."
        places_source = "OpenStreetMap local cache"
        if not is_home and row.cluster_id in query_ids and google_active:
            places_search_attempted = True
            try:
                staged = google_client.search_nearby_staged(
                    row.medoid_lat,
                    row.medoid_lon,
                    radii_m=PLACES_RADII_M,
                    max_results=10,
                    strong_match=lambda candidates, radius: _strong_place_match(
                        candidates,
                        radius,
                        cluster_lat=row.medoid_lat,
                        cluster_lon=row.medoid_lon,
                    ),
                )
                places_search_completed = True
                places_search_radius_m = staged.stopped_radius_m or (
                    staged.stages[-1].radius_m if staged.stages else None
                )
                places_search_stop_reason = staged.stop_reason
                provenance_by_candidate: dict[str, dict[str, object]] = {}
                for stage in staged.stages:
                    stage_places = stage.response.payload.get("places", [])
                    if not isinstance(stage_places, list):
                        continue
                    for place in stage_places:
                        if not isinstance(place, Mapping):
                            continue
                        provenance_by_candidate.setdefault(
                            _candidate_identity(place),
                            {
                                "source": stage.response.source,
                                "search_radius_m": stage.radius_m,
                                "retrieved_at_utc": stage.response.retrieved_at_utc,
                                "cache_hit": stage.response.cache_hit,
                            },
                        )
                alternatives = [
                    _candidate_record(
                        place,
                        cluster_lat=row.medoid_lat,
                        cluster_lon=row.medoid_lon,
                        **provenance_by_candidate.get(
                            _candidate_identity(place),
                            {
                                "source": staged.source,
                                "search_radius_m": places_search_radius_m,
                                "retrieved_at_utc": "unknown",
                                "cache_hit": None,
                            },
                        ),
                    )
                    for place in staged.candidates
                ]
                alternatives.sort(
                    key=lambda candidate: _safe_float(candidate.get("distance_m"), 999_999)
                )
                selected, match_quality, match_reason = _select_place_candidate(alternatives)
                if match_quality == "ambiguous":
                    broad_medical = next(
                        (
                            candidate
                            for candidate in alternatives
                            if re.search(
                                r"\bmedical\b.*\b(suites?|plaza|center|centre)\b",
                                _clean_text(candidate.get("name")),
                                flags=re.IGNORECASE,
                            )
                            and _safe_float(candidate.get("distance_m"), 999_999.0)
                            <= 125
                        ),
                        None,
                    )
                    if broad_medical:
                        selected = dict(broad_medical)
                        selected["primary_type"] = "medical_center"
                        selected["types"] = sorted(
                            set(selected.get("types", [])) | {"medical_center"}
                        )
                        match_quality = "medium"
                        match_reason = (
                            "Several medical tenants were similarly close; the named multi-tenant "
                            "medical-suite location is selected instead of an individual clinician."
                        )
                places_source = staged.source
            except Exception as exc:
                error = getattr(exc, "to_dict", lambda: {"message": str(exc)})()
                errors.append({"cluster_id": row.cluster_id, "stage": "places", **error})
                match_reason = "Google Places lookup failed; local OSM context is retained."
                places_search_stop_reason = "request_failed"
                if str(error.get("category", "")) in {
                    "authorization",
                    "api_disabled",
                    "billing",
                    "quota",
                    "missing_key",
                }:
                    google_active = False

        if not is_home and selected is None:
            try:
                osm_names = json.loads(row.osm_names_json)
            except (TypeError, ValueError, json.JSONDecodeError):
                osm_names = []
            if osm_names:
                selected = {
                    "name": str(osm_names[0]),
                    "primary_type": "local_osm_context",
                    "types": json.loads(row.osm_categories_json),
                    "address": context["formatted_address"],
                    "latitude": row.medoid_lat,
                    "longitude": row.medoid_lon,
                    "distance_m": float("nan"),
                    "business_status": "",
                    "google_maps_uri": "",
                    "source": "OpenStreetMap local cache",
                    "search_radius_m": 250,
                    "retrieved_at_utc": "local_cache_timestamp_unavailable",
                    "cache_hit": True,
                    "match_quality_score": 0.25,
                }
                match_quality = "low"
                places_source = "OpenStreetMap local cache"
                match_reason = (
                    "A named local OSM feature supplies area context, but it does not prove "
                    "that the endpoint represents a visit to that feature."
                )

        if is_home:
            private_home_exact_address = context["formatted_address"]
            try:
                endpoint_roads = json.loads(row.top_endpoint_roads_json)
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                endpoint_roads = {}
            if any("wynmoor" in str(name).lower() for name in endpoint_roads):
                generalized = ", ".join(
                    value for value in ("Wynmoor area", context["city"]) if value
                )
            selected = None
            alternatives = []
            reverse_address = generalized
            generalized = generalized or f"Generalized residential area in {row.county}"
        else:
            reverse_address = context["formatted_address"]
            generalized = generalized or reverse_address or str(row.county)
        selected_address = selected.get("address", "") if selected else ""
        if (
            selected
            and not re.match(r"^\s*\d", _clean_text(selected_address))
            and re.match(r"^\s*\d", context["formatted_address"])
        ):
            selected_address = context["formatted_address"]
        if selected:
            candidate_types = selected.get("types", [])
        elif match_quality == "ambiguous" and "medical-office" in match_reason:
            candidate_types = ["medical_clinic"]
        elif match_quality == "ambiguous":
            candidate_types = sorted(
                {
                    str(place_type)
                    for alternative in alternatives
                    for place_type in alternative.get("types", [])
                }
            )
        else:
            candidate_types = []
        enrichment_rows.append(
            {
                "generalized_location": generalized,
                "reverse_geocoded_address": reverse_address,
                "reverse_geocode_source": geocode_source,
                "reverse_geocode_error": reverse_error,
                "selected_poi_name": selected.get("name", "") if selected else "",
                "selected_poi_category": selected.get("primary_type", "") if selected else "",
                "selected_poi_types": _json_dumps(candidate_types),
                "selected_poi_address": selected_address,
                "selected_poi_latitude": selected.get("latitude", float("nan")) if selected else float("nan"),
                "selected_poi_longitude": selected.get("longitude", float("nan")) if selected else float("nan"),
                "selected_poi_distance_m": selected.get("distance_m", float("nan")) if selected else float("nan"),
                "selected_poi_source": selected.get("source", places_source) if selected else "",
                "selected_poi_search_radius_m": selected.get("search_radius_m") if selected else None,
                "selected_poi_retrieved_at_utc": selected.get("retrieved_at_utc", "") if selected else "",
                "selected_poi_match_score": selected.get("match_quality_score", float("nan")) if selected else float("nan"),
                "selected_poi_google_maps_uri": selected.get("google_maps_uri", "") if selected else "",
                "alternative_pois_json": _json_dumps(alternatives[:10]),
                "poi_match_quality": match_quality,
                "poi_match_reasoning": match_reason,
                "enrichment_attempted": geocoding_attempted or places_search_attempted,
                "geocoding_attempted": geocoding_attempted,
                "places_search_attempted": places_search_attempted,
                "places_search_completed": places_search_completed,
                "places_search_radius_m": places_search_radius_m,
                "places_search_stop_reason": places_search_stop_reason,
            }
        )
    output = pd.concat([output.reset_index(drop=True), pd.DataFrame(enrichment_rows)], axis=1)
    home_mask = output["privacy_flag"].eq("HOME_SENSITIVE")
    for column in (
        "selected_poi_name",
        "selected_poi_category",
        "selected_poi_types",
        "selected_poi_address",
        "selected_poi_source",
        "selected_poi_retrieved_at_utc",
        "selected_poi_google_maps_uri",
        "alternative_pois_json",
    ):
        output.loc[home_mask, column] = "" if column != "alternative_pois_json" else "[]"
    output.loc[
        home_mask,
        [
            "selected_poi_latitude",
            "selected_poi_longitude",
            "selected_poi_distance_m",
            "selected_poi_search_radius_m",
            "selected_poi_match_score",
        ],
    ] = np.nan
    output.attrs["private_home_exact_address_for_validation"] = private_home_exact_address
    return output, errors


ROLE_TYPE_GROUPS: dict[str, set[str]] = {
    "possible school/daycare stop": {
        "school",
        "primary_school",
        "secondary_school",
        "preschool",
        "child_care_agency",
        "kindergarten",
    },
    "possible university/college destination": {"university", "college", "research_institute"},
    "possible healthcare destination": {
        "hospital",
        "medical_center",
        "medical_clinic",
        "doctor",
        "dentist",
        "physiotherapist",
        "pharmacy",
    },
    "grocery destination": {"grocery_store", "supermarket", "food_store"},
    "shopping/retail destination": {
        "shopping_mall",
        "department_store",
        "clothing_store",
        "store",
        "discount_store",
        "home_goods_store",
    },
    "restaurant/food destination": {
        "restaurant",
        "american_restaurant",
        "mexican_restaurant",
        "seafood_restaurant",
        "fast_food_restaurant",
        "cafe",
        "coffee_shop",
        "bakery",
        "bar",
    },
    "gym/fitness destination": {"gym", "fitness_center", "sports_club"},
    "religious destination": {"church", "place_of_worship", "mosque", "synagogue", "hindu_temple"},
    "park/recreation destination": {"park", "sports_complex", "golf_course", "community_center"},
    "airport/transit destination": {"airport", "transit_station", "train_station", "bus_station"},
    "gas station": {"gas_station"},
    "bank/financial errand": {"bank", "atm", "finance"},
    "likely workplace": {
        "corporate_office",
        "business_center",
        "government_office",
        "industrial_area",
    },
}


def _role_map_score(types: set[str], role: str) -> float:
    expected = ROLE_TYPE_GROUPS.get(role, set())
    return 1.0 if expected & types else 0.0


def _dwell_score(value: float, lower: float, upper: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if lower <= value <= upper:
        return 1.0
    if lower / 2 <= value <= upper * 1.5:
        return 0.5
    return 0.0


def _trip_chain_evidence_by_cluster(
    repeated_chains: pd.DataFrame | None,
    *,
    home_cluster_id: str,
) -> dict[str, dict[str, object]]:
    """Summarize repeated-chain positions without inferring activity purpose."""
    if repeated_chains is None or repeated_chains.empty:
        return {}
    evidence: dict[str, dict[str, object]] = {}
    for record in repeated_chains.to_dict(orient="records"):
        sequence = [
            str(value)
            for value in _json_loads_list(record.get("cluster_sequence_json"))
            if str(value) not in {"", "UNCLUSTERED"}
        ]
        if not sequence:
            continue
        occurrences = int(_safe_float(record.get("occurrence_count"), 0.0))
        if occurrences <= 0:
            continue
        try:
            stop_durations = json.loads(
                _clean_text(record.get("typical_stop_durations_json")) or "{}"
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            stop_durations = {}
        if not isinstance(stop_durations, Mapping):
            stop_durations = {}
        home_loop = (
            len(sequence) >= 3
            and sequence[0] == home_cluster_id
            and sequence[-1] == home_cluster_id
        )
        for cluster_id in dict.fromkeys(sequence):
            cluster = evidence.setdefault(
                cluster_id,
                {
                    "repeated_chain_pattern_count": 0,
                    "repeated_chain_occurrence_count": 0,
                    "home_loop_intermediate_occurrences": 0,
                    "preceded_by_home_occurrences": 0,
                    "followed_by_home_occurrences": 0,
                    "stop_duration_samples": [],
                    "top_chain": "",
                    "top_chain_occurrences": 0,
                },
            )
            cluster["repeated_chain_pattern_count"] = int(
                cluster["repeated_chain_pattern_count"]
            ) + 1
            cluster["repeated_chain_occurrence_count"] = int(
                cluster["repeated_chain_occurrence_count"]
            ) + occurrences
            positions = [
                index for index, value in enumerate(sequence) if value == cluster_id
            ]
            if home_loop and any(0 < index < len(sequence) - 1 for index in positions):
                cluster["home_loop_intermediate_occurrences"] = int(
                    cluster["home_loop_intermediate_occurrences"]
                ) + occurrences
            if any(index > 0 and sequence[index - 1] == home_cluster_id for index in positions):
                cluster["preceded_by_home_occurrences"] = int(
                    cluster["preceded_by_home_occurrences"]
                ) + occurrences
            if any(
                index < len(sequence) - 1 and sequence[index + 1] == home_cluster_id
                for index in positions
            ):
                cluster["followed_by_home_occurrences"] = int(
                    cluster["followed_by_home_occurrences"]
                ) + occurrences
            dwell = _safe_float(stop_durations.get(cluster_id))
            if math.isfinite(dwell):
                samples = cluster["stop_duration_samples"]
                if isinstance(samples, list):
                    samples.extend([dwell] * occurrences)
            if occurrences > int(cluster["top_chain_occurrences"]):
                cluster["top_chain_occurrences"] = occurrences
                cluster["top_chain"] = _clean_text(record.get("public_chain"))
    for cluster in evidence.values():
        samples = cluster.pop("stop_duration_samples", [])
        cluster["median_repeated_chain_stop_minutes"] = (
            float(pd.Series(samples, dtype=float).median()) if samples else float("nan")
        )
    return evidence


def classify_location_roles(
    clusters: pd.DataFrame,
    trips: pd.DataFrame | None = None,
    repeated_chains: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply common-sense activity gates to measured stays and map context.

    The nearest listing is context rather than proof.  In particular, a short
    commercial-area stop can never be promoted to a workplace merely because
    it is frequent.
    """
    output = clusters.copy()
    home_rows = output.loc[output["privacy_flag"].eq("HOME_SENSITIVE")]
    home_cluster_id = str(home_rows.iloc[0]["cluster_id"]) if not home_rows.empty else ""
    chain_evidence = _trip_chain_evidence_by_cluster(
        repeated_chains,
        home_cluster_id=home_cluster_id,
    )
    home_connections: Counter[str] = Counter()
    if trips is not None and home_cluster_id:
        for record in trips.to_dict(orient="records"):
            origin = str(record.get("origin_cluster_id"))
            destination = str(record.get("destination_cluster_id"))
            if origin == home_cluster_id and destination != home_cluster_id:
                home_connections[destination] += 1
            if destination == home_cluster_id and origin != home_cluster_id:
                home_connections[origin] += 1

    healthcare_types = {
        "doctor", "hospital", "medical_center", "medical_clinic", "dentist",
        "physiotherapist", "health", "pharmacy", "laboratory",
    }
    school_types = {"school", "preschool", "day_care_center", "child_care_agency"}
    university_types = {"university", "college", "school"}
    grocery_types = {"supermarket", "grocery_store"}
    gas_types = {"gas_station"}
    bank_types = {"bank", "atm", "finance"}
    retail_types = {
        "shopping_mall", "store", "department_store", "clothing_store",
        "home_goods_store", "electronics_store", "shopping_center",
    }
    restaurant_types = {"restaurant", "cafe", "coffee_shop", "meal_takeaway"}
    recreation_types = {"park", "gym", "fitness_center", "golf_course", "event_venue"}
    lodging_residential_types = {
        "lodging", "apartment_building", "apartment_complex", "housing_complex",
        "condominium_complex", "residential_building",
    }
    employment_types = {
        "corporate_office", "office", "government_office", "warehouse",
        "industrial", "hospital", "university", "college",
    }

    rows: list[dict[str, object]] = []
    for record in output.to_dict(orient="records"):
        if str(record.get("privacy_flag")) == "HOME_SENSITIVE":
            score = _safe_float(record.get("home_score"), 0.0)
            confidence = "high" if score >= 0.70 else "medium" if score >= 0.45 else "low"
            rows.append(
                {
                    "previous_label": "likely home area",
                    "inferred_role": "likely home area",
                    "selected_public_label": _clean_text(record.get("generalized_location")) or "Likely home area",
                    "multi_tenant_flag": False,
                    "role_confidence": confidence,
                    "role_evidence_score": score,
                    "behavioral_plausibility": "strong",
                    "map_plausibility": "residential context supports but does not prove the inference",
                    "trip_chain_patterns": "Repeated returns and departures anchor the recorded trip network.",
                    "behavioral_evidence": (
                        f"Observed across {int(record.get('months_visited', 0))} months with "
                        f"{int(record.get('censored_overnight_association_count', 0))} "
                        "cross-session overnight continuities."
                    ),
                    "map_evidence": "Residential context is reported only at neighborhood scale.",
                    "classification_reason": "This area has the strongest recurrence, nighttime-return, overnight-continuity, and network-anchor evidence.",
                    "competing_explanation": "A repeatedly used vehicle base or shared residence could produce a similar pattern.",
                    "uncertainty_statement": "This is a generalized likely home area, not a confirmed residence.",
                    "limitations": "Exact home address, coordinate, and map link are suppressed from public outputs.",
                    "workplace_gate_json": _json_dumps({"supported": False, "reason": "home candidate"}),
                }
            )
            continue

        try:
            types = {
                str(value).lower()
                for value in json.loads(_clean_text(record.get("selected_poi_types")) or "[]")
            }
        except (TypeError, ValueError, json.JSONDecodeError):
            types = set()
        primary = _clean_text(record.get("selected_poi_category")).lower()
        if primary:
            types.add(primary)
            if primary.endswith("_restaurant"):
                types.add("restaurant")
        try:
            alternatives = json.loads(_clean_text(record.get("alternative_pois_json")) or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            alternatives = []

        median = _safe_float(record.get("median_dwell_minutes"))
        valid_stays = int(_safe_float(record.get("valid_stay_count"), 0.0))
        long_share = _safe_float(record.get("share_over_3_hours"), 0.0)
        medium_share = _safe_float(record.get("share_20_to_60_minutes"), 0.0)
        one_to_three_share = _safe_float(record.get("share_1_to_3_hours"), 0.0)
        weekday = _safe_float(record.get("weekday_share"), 0.0)
        spread = _safe_float(record.get("coordinate_spread_m"), 0.0)
        months = int(_safe_float(record.get("months_visited"), 0.0))
        recurrence = str(record.get("recurring_frequency", "irregular"))
        home_count = int(home_connections.get(str(record.get("cluster_id")), 0))
        chain = chain_evidence.get(str(record.get("cluster_id")), {})
        repeated_chain_occurrences = int(
            _safe_float(chain.get("repeated_chain_occurrence_count"), 0.0)
        )
        home_loop_occurrences = int(
            _safe_float(chain.get("home_loop_intermediate_occurrences"), 0.0)
        )
        chain_stop_minutes = _safe_float(
            chain.get("median_repeated_chain_stop_minutes")
        )
        context_is_complex = bool(
            re.search(
                r"\b(?:center|centre|plaza|campus|mall|shoppes?|marketplace|suites)\b",
                _clean_text(record.get("osm_context")),
                flags=re.IGNORECASE,
            )
        )
        multi_tenant = bool(
            spread > 100
            or str(record.get("poi_match_quality")) in {"ambiguous", "low"}
            or len(alternatives) >= 4
            or (
                context_is_complex
                and str(record.get("poi_match_quality")) != "high"
            )
            or bool(types & {"shopping_mall", "university", "hospital", "medical_center"})
        )
        context = _clean_text(record.get("osm_context"))
        selected_name = _clean_text(record.get("selected_poi_name"))
        generic_contexts = {
            "retail", "residential", "commercial", "industrial", "institutional",
            "no local osm context", "no named local osm feature within 250 m",
        }
        broad_context = (
            context
            if context and context.lower() not in generic_contexts and not context.lower().startswith("no named")
            else ""
        )
        public_label = selected_name or _clean_text(record.get("generalized_location")) or "Recurring destination"
        if multi_tenant and broad_context:
            public_label = f"{broad_context} area"
        elif multi_tenant and selected_name:
            if re.search(r"\b(center|centre|plaza|campus|mall|suites|hospital|university|pomp)\b", selected_name, re.I):
                public_label = f"{selected_name} area"
            elif "cafe on the green" in selected_name.lower():
                public_label = "Wynmoor community destination near Cafe on the Green"
            else:
                try:
                    top_roads = json.loads(_clean_text(record.get("top_endpoint_roads_json")) or "{}")
                except (TypeError, ValueError, json.JSONDecodeError):
                    top_roads = {}
                road = next(iter(top_roads), "the recorded endpoint")
                public_label = f"Multi-tenant commercial area near {road}"

        workplace_gate = workplace_plausibility(
            {
                **record,
                "home_connection_count": home_count,
            }
        )
        employment_context = bool(types & employment_types)
        role = "unknown recurring destination"
        confidence = "low"
        score = 0.30
        competing = "A neighboring tenant, pickup/drop-off, parking-area endpoint, or recording boundary may explain the stop."
        plausibility = "limited"

        # Reconstruct the previous failure explicitly for the validation table.
        business_context = bool(types) and not bool(
            types & {"apartment_building", "apartment_complex", "housing_complex", "bus_stop"}
        )
        previous_label = (
            "likely workplace"
            if int(record.get("total_visit_count", 0)) >= 300
            and weekday >= 0.75
            and recurrence in {"daily", "several times per week"}
            and business_context
            else "map-proximity activity label"
        )

        if workplace_gate["supported"] and employment_context:
            role = "likely workplace"
            confidence = "medium" if multi_tenant else "high"
            score = 0.86 if not multi_tenant else 0.74
            plausibility = "repeated multi-hour weekday stays"
            competing = "The pattern could instead be another long-duration institutional or social activity."
        elif math.isfinite(median) and median >= 180 and long_share >= 0.50:
            if types & lodging_residential_types:
                role = "possible recurring lodging or residential destination"
                confidence = "medium" if valid_stays >= 15 else "low"
                score = 0.64 if valid_stays >= 15 else 0.46
                plausibility = (
                    "repeated multi-hour stays and lodging/residential map context, "
                    "with the specific purpose unresolved"
                )
                competing = (
                    "Lodging, a social visit, work, an endpoint near another tenant, or "
                    "another long-duration activity are plausible; no relationship is inferred."
                )
                public_label = (
                    f"{_clean_text(record.get('generalized_location')) or 'Generalized'} "
                    "lodging/residential area"
                )
            else:
                role = "unresolved recurring long-duration destination"
                confidence = "medium" if valid_stays >= 15 else "low"
                score = 0.63 if valid_stays >= 15 else 0.45
                plausibility = "repeated multi-hour stays, but purpose is unresolved"
                competing = "Work, recreation, a social visit, or another long-duration activity are all plausible."
        elif types & gas_types and math.isfinite(median) and median <= 20 and valid_stays >= 5:
            role = "gas station"
            confidence = "high" if not multi_tenant else "medium"
            score = 0.86
            plausibility = "repeated brief stops align with a fuel stop"
        elif types & grocery_types and math.isfinite(median) and 20 <= median <= 120:
            role = "grocery destination"
            confidence = "high" if not multi_tenant and valid_stays >= 10 else "medium"
            score = 0.82
            plausibility = "repeated medium-duration stops align with grocery shopping"
        elif types & bank_types and math.isfinite(median) and median < 30:
            role = "recurring short commercial/financial stop"
            confidence = "medium" if valid_stays >= 10 else "low"
            score = 0.68
            plausibility = "brief stays fit an errand or pickup/drop-off, not employment"
            public_label = f"{broad_context or selected_name or 'Commercial'} short-stop area"
        elif types & healthcare_types:
            appointment_fit = (
                valid_stays >= 5
                and math.isfinite(median)
                and 20 <= median <= 180
                and medium_share + one_to_three_share >= 0.35
                and recurrence not in {"daily", "several times per week"}
            )
            if appointment_fit:
                role = "possible healthcare-related destination"
                confidence = "medium" if not multi_tenant else "low"
                score = 0.66
                plausibility = "timing and stay lengths are compatible with appointments"
                competing = "A pharmacy pickup, passenger drop-off, or nearby office remains plausible."
            else:
                role = "recurring short stop near medical offices"
                confidence = "low"
                score = 0.42
                plausibility = "map context is medical, but measured stays are usually too short to establish purpose"
                competing = "A pickup/drop-off, pharmacy/lab errand, or neighboring business may explain the stops."
            public_label = f"{broad_context or selected_name or 'Multi-tenant'} medical-office area"
        elif types & school_types:
            arrival_hour = _safe_float(record.get("typical_arrival_hour"))
            school_time = math.isfinite(arrival_hour) and (
                6.5 <= arrival_hour <= 9.5 or 13.0 <= arrival_hour <= 16.5
            )
            school_chain_supported = (
                valid_stays >= 5
                and math.isfinite(median)
                and 5 <= median <= 60
                and weekday >= 0.75
                and school_time
                and repeated_chain_occurrences >= 5
                and home_loop_occurrences >= 3
            )
            if school_chain_supported:
                role = "possible school/daycare stop"
                confidence = "medium" if not multi_tenant else "low"
                score = 0.66
                plausibility = (
                    "brief weekday school-time stops recur inside repeated home-based chains"
                )
                competing = (
                    "An unrelated pickup/drop-off, nearby residence, or neighboring business "
                    "could produce a similar pattern; no child or relationship is inferred."
                )
            else:
                role = "school/daycare context without sufficient chain evidence"
                confidence = "low"
                score = 0.30
                plausibility = "a nearby listing alone is insufficient"
                competing = "A nearby residence, business, or unrelated pickup/drop-off may explain the endpoint."
        elif types & university_types and valid_stays >= 3 and math.isfinite(median) and 30 <= median <= 240:
            role = "possible university/college destination"
            confidence = "medium" if not multi_tenant else "low"
            score = 0.62
            plausibility = "repeated campus-area stays are plausible, but purpose is unknown"
            competing = "Employment, administration, an event, or pickup/drop-off could produce the same pattern."
        elif types & retail_types and (not math.isfinite(median) or median < 180):
            role = "shopping/retail area"
            confidence = "medium" if valid_stays >= 8 else "low"
            score = 0.67
            plausibility = "short-to-medium stays and retail context support a broad errand interpretation"
        elif types & restaurant_types and math.isfinite(median) and 10 <= median <= 150:
            role = "restaurant/food area"
            confidence = "medium" if not multi_tenant else "low"
            score = 0.60
            plausibility = "the stay distribution is compatible with food service"
        elif types & recreation_types and math.isfinite(median) and median >= 30:
            role = "possible recreation/entertainment destination"
            confidence = "low" if multi_tenant else "medium"
            score = 0.56
            plausibility = "longer stays are compatible with recreation, but purpose is unconfirmed"
        elif "residential" in context.lower() and int(record.get("total_visit_count", 0)) >= 10:
            role = "other recurring residential area"
            confidence = "low"
            score = 0.45
            plausibility = "repeated visits occur in residential map context"
            competing = "The location may be a vehicle base, social destination, or endpoint artifact; no relationship is inferred."
        elif int(record.get("total_visit_count", 0)) <= 2:
            role = "one-time destination"
            confidence = "low"
            score = 0.20

        if valid_stays < 5 and role not in {"one-time destination"}:
            confidence = "low"
        if multi_tenant and confidence == "high" and role not in {"gas station"}:
            confidence = "medium"

        distance = _safe_float(record.get("selected_poi_distance_m"))
        map_evidence = (
            f"Map context: {broad_context or selected_name or 'unresolved area'}; "
            f"nearest candidate category {primary or 'unresolved'}"
            + (f", approximately {distance:.0f} m away" if math.isfinite(distance) else "")
            + f"; source {record.get('selected_poi_source') or record.get('osm_context_source')}."
        )
        chain_text = (
            f"It appears in {repeated_chain_occurrences} occurrences across "
            f"{int(_safe_float(chain.get('repeated_chain_pattern_count'), 0.0))} "
            "repeated chain patterns"
            + (
                f", including {home_loop_occurrences} home-based loop occurrences"
                if home_loop_occurrences
                else ""
            )
            + (
                f"; the median reconstructed stop in those chains was "
                f"{chain_stop_minutes:.0f} minutes"
                if math.isfinite(chain_stop_minutes)
                else ""
            )
            + "."
            if repeated_chain_occurrences
            else "No repeated chain met the reporting threshold for this cluster."
        )
        behavioral = (
            f"{valid_stays} measured same-session stays across {months} months; "
            + (f"median {median:.0f} minutes; " if math.isfinite(median) else "median dwell unavailable; ")
            + f"{weekday:.0%} weekday activity; {home_count} recorded trips connected with the likely home area. "
            + chain_text
        )
        rows.append(
            {
                "previous_label": previous_label,
                "inferred_role": role,
                "selected_public_label": public_label,
                "multi_tenant_flag": multi_tenant,
                "role_confidence": confidence,
                "role_evidence_score": float(score),
                "behavioral_plausibility": plausibility,
                "map_plausibility": "broad complex/area context" if multi_tenant else "specific listing is geographically plausible",
                "trip_chain_patterns": chain_text,
                "behavioral_evidence": behavioral,
                "map_evidence": map_evidence,
                "classification_reason": plausibility.capitalize() + ".",
                "competing_explanation": competing,
                "uncertainty_statement": "The activity role is inferred from recorded mobility and map context; the actual purpose is not confirmed.",
                "limitations": "POI proximity does not prove a visit, and endpoints may fall on parking lots or access roads.",
                "workplace_gate_json": _json_dumps(workplace_gate),
            }
        )
    result = pd.concat([output.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    result["specific_candidate_poi_name"] = result["selected_poi_name"]
    result["specific_candidate_poi_address"] = result["selected_poi_address"]
    broad_categories = {"shopping_mall", "medical_center", "hospital", "university", "park"}
    broad_name = result["selected_poi_name"].fillna("").astype(str).str.contains(
        r"\b(?:shopping center|shopping centre|plaza|campus|mall|suites|hospital|university)\b",
        case=False,
        regex=True,
    )
    suppress_tenant = (
        result["multi_tenant_flag"].fillna(False).astype(bool)
        & ~result["selected_poi_category"].fillna("").astype(str).isin(broad_categories)
        & ~broad_name
        & result["privacy_flag"].ne("HOME_SENSITIVE")
    )
    for index in result.index[suppress_tenant]:
        context = _clean_text(result.at[index, "osm_context"])
        named_context = (
            context
            and context.lower() not in {
                "retail", "residential", "commercial", "industrial", "institutional",
                "no local osm context", "no named local osm feature within 250 m",
            }
            and not context.lower().startswith("no named")
        )
        replacement = context if named_context else _clean_text(
            result.at[index, "selected_public_label"]
        )
        if replacement:
            result.at[index, "selected_poi_name"] = replacement
            result.at[index, "selected_poi_category"] = "multi_tenant_complex"
            result.at[index, "selected_poi_address"] = _clean_text(
                result.at[index, "reverse_geocoded_address"]
            )
            result.at[index, "selected_poi_source"] = (
                "OpenStreetMap context + Google candidate search"
                if bool(result.at[index, "places_search_attempted"])
                else "OpenStreetMap context"
            )
            result.at[index, "selected_poi_google_maps_uri"] = ""
            result.at[index, "poi_match_quality"] = "ambiguous"
            result.at[index, "poi_match_reasoning"] = (
                "The endpoint cluster spans a shared commercial area; a complex-level "
                "label is used and the tenant-level Google result remains only an alternative."
            )
    residential_sensitive = result["inferred_role"].eq(
        "possible recurring lodging or residential destination"
    )
    for index in result.index[residential_sensitive]:
        broad_label = _clean_text(result.at[index, "selected_public_label"])
        result.at[index, "selected_poi_name"] = broad_label
        result.at[index, "selected_poi_category"] = "lodging_or_residential_area"
        result.at[index, "selected_poi_address"] = (
            f"{_clean_text(result.at[index, 'generalized_location'])}, Florida"
        )
        result.at[index, "selected_poi_google_maps_uri"] = ""
        result.at[index, "selected_poi_latitude"] = np.nan
        result.at[index, "selected_poi_longitude"] = np.nan
        result.at[index, "poi_match_reasoning"] = (
            "A city-level label is used because the nearby listing may represent a "
            "residential or lodging unit. The specific Google candidate remains in "
            "the private audit alternatives only."
        )
    return result


def build_recurring_patterns(
    trips: pd.DataFrame,
    enriched_clusters: pd.DataFrame,
    stays: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Describe monthly, weekly, new, disappearing, and day-type patterns."""
    observed_months = sorted(trips["month"].dropna().astype(str).unique())
    month_position = {month: index for index, month in enumerate(observed_months)}
    period_names = ("early", "middle", "late")
    period_by_month = {
        month: period_names[min(index * 3 // max(len(observed_months), 1), 2)]
        for index, month in enumerate(observed_months)
    }
    period_month_counts = Counter(period_by_month.values())
    cluster_lookup = enriched_clusters.set_index("cluster_id")
    rows: list[dict[str, object]] = []
    destinations = trips.loc[trips["destination_cluster_id"].ne("UNCLUSTERED")].copy()
    destinations["visit_date"] = (
        pd.to_datetime(destinations["end_timestamp"], utc=True)
        .dt.tz_convert(LOCAL_TIMEZONE)
        .dt.date.astype(str)
    )
    destinations["visit_timestamp"] = pd.to_datetime(
        destinations["end_timestamp"], utc=True
    ).dt.tz_convert(LOCAL_TIMEZONE)
    destinations["visit_week"] = destinations["visit_timestamp"].dt.strftime("%G-W%V")
    destinations["visit_is_weekday"] = destinations["visit_timestamp"].dt.dayofweek < 5
    destinations["period"] = destinations["month"].map(period_by_month)
    stay_data = pd.DataFrame()
    if stays is not None and not stays.empty:
        stay_data = stays.loc[stays["stay_status"].eq("MEASURED_STAY")].copy()
        stay_data["month"] = (
            pd.to_datetime(stay_data["arrival_timestamp"], errors="coerce", utc=True)
            .dt.tz_convert(LOCAL_TIMEZONE)
            .dt.strftime("%Y-%m")
        )
        stay_data["period"] = stay_data["month"].map(period_by_month)
    for cluster_id, visits in destinations.groupby("destination_cluster_id", sort=True):
        if cluster_id not in cluster_lookup.index:
            continue
        cluster = cluster_lookup.loc[cluster_id]
        if cluster["privacy_flag"] == "HOME_SENSITIVE" or len(visits) < 2:
            continue
        monthly = visits["month"].value_counts().sort_index()
        active_months = list(monthly.index.astype(str))
        first_arrival = visits["visit_timestamp"].min()
        last_arrival = visits["visit_timestamp"].max()
        arrival_span_days = max(int((last_arrival.date() - first_arrival.date()).days) + 1, 1)
        recurrence = _recurrence_pattern(
            len(visits),
            int(visits["visit_date"].nunique()),
            int(visits["visit_week"].nunique()),
            len(active_months),
            arrival_span_days,
        )
        origin_associations = trips.loc[
            trips["origin_cluster_id"].astype(str).eq(str(cluster_id))
        ].copy()
        origin_times = pd.to_datetime(
            origin_associations["start_timestamp"], errors="coerce", utc=True
        ).dt.tz_convert(LOCAL_TIMEZONE)
        association_times = pd.concat(
            [visits["visit_timestamp"], origin_times.dropna()], ignore_index=True
        ).dropna()
        first_association = association_times.min()
        last_association = association_times.max()
        first_activity_month = first_association.strftime("%Y-%m")
        last_activity_month = last_association.strftime("%Y-%m")
        patterns: list[str] = []
        if recurrence in {"daily", "several times per week", "weekly", "biweekly", "monthly"}:
            patterns.append(recurrence)
        arrival_weekday_share = float(visits["visit_is_weekday"].mean())
        if arrival_weekday_share >= 0.90:
            patterns.append("weekday-only or nearly weekday-only")
        if arrival_weekday_share <= 0.10:
            patterns.append("weekend-only or nearly weekend-only")
        first_pos = month_position.get(first_activity_month, 0)
        last_pos = month_position.get(last_activity_month, len(observed_months) - 1)
        if first_pos >= 3 and len(active_months) >= 6 and len(visits) >= 12:
            patterns.append("new recurring destination")
        if (
            last_pos <= len(observed_months) - 4
            and len(active_months) >= 6
            and len(visits) >= 12
        ):
            patterns.append("stopped appearing")
        month_numbers = [int(month.split("-")[1]) for month in active_months]
        recurring_calendar_months = [
            month
            for month, count in Counter(month_numbers).items()
            if count >= 2
        ]
        if recurring_calendar_months and len(set(month_numbers)) <= 5:
            patterns.append("possible seasonal pattern")
        if not patterns:
            patterns.append("irregular recurring destination")
        name = _clean_text(cluster.get("selected_public_label")) or _clean_text(cluster["selected_poi_name"]) or _clean_text(
            cluster["generalized_location"]
        )
        period_counts = {
            period: int(visits["period"].eq(period).sum()) for period in period_names
        }
        period_rates = {
            period: period_counts[period] / max(period_month_counts[period], 1)
            for period in period_names
        }
        if period_rates["late"] >= period_rates["early"] + 0.5 and period_rates["late"] >= period_rates["early"] * 1.35:
            trend = "increasing"
        elif period_rates["early"] >= period_rates["late"] + 0.5 and period_rates["early"] >= period_rates["late"] * 1.35:
            trend = "decreasing"
        elif last_activity_month != observed_months[-1] and last_pos <= len(observed_months) - 4:
            trend = "stopped appearing"
        elif first_activity_month != observed_months[0] and first_pos >= 3:
            trend = "newer destination"
        else:
            trend = "broadly stable or intermittent"
        typical_by_period: dict[str, str] = {}
        for period in period_names:
            subset = visits.loc[visits["period"].eq(period), "visit_timestamp"]
            if subset.empty:
                typical_by_period[period] = "not observed"
            else:
                hour = float((subset.dt.hour + subset.dt.minute / 60.0).median())
                typical_by_period[period] = _format_hour(hour)
        dwell_by_period: dict[str, float | None] = {}
        cluster_stays = (
            stay_data.loc[stay_data["cluster_id"].astype(str).eq(str(cluster_id))]
            if not stay_data.empty
            else pd.DataFrame()
        )
        for period in period_names:
            if cluster_stays.empty:
                dwell_by_period[period] = None
            else:
                value = pd.to_numeric(
                    cluster_stays.loc[cluster_stays["period"].eq(period), "dwell_minutes"],
                    errors="coerce",
                ).median()
                dwell_by_period[period] = None if pd.isna(value) else float(value)
        association_note = ""
        if last_association > last_arrival:
            association_note = (
                f" A later departure-only association was recorded in "
                f"{last_activity_month}; it does not establish an additional arrival."
            )
        narrative = (
            f"{name} was recorded in {len(active_months)} months from {active_months[0]} "
            f"through {active_months[-1]}. Arrival frequency was {recurrence}; the "
            f"longitudinal pattern was {trend}.{association_note} The activity "
            "interpretation remains inferred."
        )
        rows.append(
            {
                "cluster_id": cluster_id,
                "named_poi_or_generalized_location": name,
                "address": _clean_text(cluster["selected_poi_address"]),
                "pattern_type": "; ".join(patterns),
                "visit_dates_json": _json_dumps(sorted(visits["visit_date"].unique())),
                "month_counts_json": _json_dumps({str(month): int(count) for month, count in monthly.items()}),
                "period_visit_counts_json": _json_dumps(period_counts),
                "period_visit_rates_json": _json_dumps(period_rates),
                "typical_time_by_period_json": _json_dumps(typical_by_period),
                "dwell_by_period_json": _json_dumps(dwell_by_period),
                "first_month": active_months[0],
                "last_month": active_months[-1],
                "first_arrival_date": first_arrival.date().isoformat(),
                "last_arrival_date": last_arrival.date().isoformat(),
                "first_activity_association_date": first_association.date().isoformat(),
                "last_activity_association_date": last_association.date().isoformat(),
                "last_activity_association_month": last_activity_month,
                "activity_association_count": int(len(association_times)),
                "months_visited": len(active_months),
                "visit_count": len(visits),
                "visit_frequency": recurrence,
                "visit_frequency_basis": "destination_arrivals",
                "arrival_weekday_share": arrival_weekday_share,
                "typical_time": cluster["typical_arrival_time"],
                "median_dwell_minutes": cluster["median_dwell_minutes"],
                "inferred_activity": cluster["inferred_role"],
                "confidence": cluster["role_confidence"],
                "trend": trend,
                "plain_english_narrative": narrative,
                "evidence_score": cluster["role_evidence_score"],
                "behavioral_evidence": cluster["behavioral_evidence"],
                "map_evidence": cluster["map_evidence"],
                "alternative_interpretation": cluster["competing_explanation"],
                "uncertainty_statement": cluster["uncertainty_statement"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["visit_count", "months_visited"], ascending=[False, False]
    ).reset_index(drop=True)


def _cluster_public_label(row: Mapping[str, Any]) -> str:
    if str(row.get("privacy_flag")) == "HOME_SENSITIVE":
        return _clean_text(row.get("generalized_location")) or "Likely home area"
    revised = _clean_text(row.get("selected_public_label"))
    if revised:
        return revised
    selected = _clean_text(row.get("selected_poi_name"))
    if selected:
        return selected
    generalized = _clean_text(row.get("generalized_location"))
    county = _clean_text(row.get("county"))
    if not generalized or generalized == county:
        return f"Cluster {row.get('cluster_id', '')} ({county or 'unresolved area'})"
    return generalized


def attach_public_location_labels(
    trips: pd.DataFrame,
    enriched_clusters: pd.DataFrame,
) -> pd.DataFrame:
    labels = {
        str(row["cluster_id"]): _cluster_public_label(row)
        for row in enriched_clusters.to_dict(orient="records")
    }
    output = trips.copy()
    output["origin_label"] = output["origin_cluster_id"].map(labels).fillna(
        output["origin_cluster_id"]
    )
    output["destination_label"] = output["destination_cluster_id"].map(labels).fillna(
        output["destination_cluster_id"]
    )
    return output


def annotate_longitudinal_reporting(
    transitions: pd.DataFrame,
    od_summary: pd.DataFrame,
    enriched_clusters: pd.DataFrame,
) -> pd.DataFrame:
    """Mark transition rows that are suitable for public behavioral headlines.

    Route-family results remain intact for audit.  The reporting flag only
    suppresses changes that are better explained by access-road variation
    inside the generalized home complex than by a corridor choice.
    """
    if transitions.empty:
        return transitions.copy()
    output = transitions.copy()
    home_rows = enriched_clusters.loc[
        enriched_clusters["privacy_flag"].eq("HOME_SENSITIVE")
    ]
    home_id = str(home_rows.iloc[0]["cluster_id"]) if not home_rows.empty else ""
    separation_lookup = {
        (str(row.origin_cluster_id), str(row.destination_cluster_id)): float(
            row.median_od_separation_m
        )
        for row in od_summary.itertuples(index=False)
    }
    internal_road_pattern = re.compile(
        r"\b(?:wynmoor|portofino|antigua|eleuthera|aruba|bermuda|granada)\b",
        re.IGNORECASE,
    )
    eligibility: list[bool] = []
    reasons: list[str] = []
    for row in output.to_dict(orient="records"):
        origin = str(row.get("origin_cluster_id", ""))
        destination = str(row.get("destination_cluster_id", ""))
        separation = separation_lookup.get((origin, destination), float("nan"))
        family_text = " ".join(
            str(row.get(column, ""))
            for column in ("baseline_route_family", "later_route_family", "family_name")
        )
        road_words = [
            word.strip()
            for word in re.split(r"→|/|,", family_text)
            if word.strip()
        ]
        all_internal = bool(road_words) and all(
            internal_road_pattern.search(word) for word in road_words
        )
        home_internal = (
            home_id in {origin, destination}
            and math.isfinite(separation)
            and separation < 1_000.0
            and all_internal
        )
        if home_internal:
            eligibility.append(False)
            reasons.append(
                "Suppressed from headline reporting: the OD is within 1 km of the "
                "generalized home cluster and both families use only internal access roads."
            )
        else:
            eligibility.append(True)
            reasons.append("Eligible for public reporting after local-access screening.")
    output["public_reporting_eligible"] = eligibility
    output["reporting_screen_reason"] = reasons
    return output


def _route_latlon(
    sequence: Sequence[tuple[str, int]],
    context_lookup: Mapping[tuple[str, int], Mapping[str, object]],
) -> list[tuple[float, float]]:
    """Convert an ordered projected FID path into display-only WGS84 points."""
    projected: list[tuple[float, float]] = []
    for key in sequence:
        geometry = context_lookup.get(tuple(key), {}).get("geometry")
        if geometry is None:
            continue
        if getattr(geometry, "geom_type", "") == "MultiLineString":
            pieces = list(geometry.geoms)
            geometry = max(pieces, key=lambda item: item.length, default=None)
        if geometry is None or not hasattr(geometry, "coords"):
            continue
        coordinates = [(float(x), float(y)) for x, y, *_ in geometry.coords]
        if not coordinates:
            continue
        if projected:
            previous = projected[-1]
            forward = math.hypot(
                coordinates[0][0] - previous[0], coordinates[0][1] - previous[1]
            )
            reverse = math.hypot(
                coordinates[-1][0] - previous[0], coordinates[-1][1] - previous[1]
            )
            if reverse < forward:
                coordinates.reverse()
            if coordinates[0] == projected[-1]:
                coordinates = coordinates[1:]
        projected.extend(coordinates)
    if len(projected) < 2:
        return []
    transformer = Transformer.from_crs(PROJECTED_CRS, WGS84, always_xy=True)
    xs, ys = zip(*projected, strict=True)
    longitudes, latitudes = transformer.transform(xs, ys)
    return [(float(lat), float(lon)) for lat, lon in zip(latitudes, longitudes, strict=True)]


def build_route_map_frames(
    profiles: pd.DataFrame,
    changes: pd.DataFrame,
    road_context_lookup: Mapping[tuple[str, int], Mapping[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create bounded monthly/common/change route geometry frames for Folium."""
    if profiles.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    monthly = profiles.loc[profiles["trip_count"] >= 2].copy()
    monthly = (
        monthly.sort_values(["month", "trip_count"], ascending=[True, False])
        .groupby("month", as_index=False, group_keys=False)
        .head(5)
        .reset_index(drop=True)
    )
    monthly["latlon_sequence"] = monthly["dominant_county_fid_sequence"].map(
        lambda sequence: _route_latlon(sequence, road_context_lookup)
    )
    monthly["dominant_road_names"] = monthly["dominant_road_name_sequence"].map(
        lambda values: " → ".join(values) if isinstance(values, list) else _clean_text(values)
    )
    monthly["route_frequency"] = monthly["dominant_route_frequency"]

    pair_totals = (
        profiles.groupby(["origin_cluster_id", "destination_cluster_id"])["trip_count"]
        .sum()
        .rename("od_total_trips")
        .reset_index()
    )
    common = monthly.merge(
        pair_totals,
        on=["origin_cluster_id", "destination_cluster_id"],
        how="left",
        validate="many_to_one",
    )
    common = common.loc[common["od_total_trips"] >= 10].copy()

    changed = changes.copy()
    if not changed.empty:
        geometry_lookup = profiles.set_index(
            ["origin_cluster_id", "destination_cluster_id", "month"]
        )["dominant_county_fid_sequence"].to_dict()
        changed["latlon_sequence"] = changed.apply(
            lambda row: _route_latlon(
                geometry_lookup.get(
                    (
                        row["origin_cluster_id"],
                        row["destination_cluster_id"],
                        row["month_b"],
                    ),
                    [],
                ),
                road_context_lookup,
            ),
            axis=1,
        )
    return monthly, common, changed


def build_longitudinal_map_frames(
    route_family_representatives: pd.DataFrame,
    transitions: pd.DataFrame,
    temporary_deviations: pd.DataFrame,
    repeated_chains: pd.DataFrame,
    enriched_clusters: pd.DataFrame,
    road_context_lookup: Mapping[tuple[str, int], Mapping[str, object]],
    *,
    generalized_home_coordinates: tuple[float, float],
) -> dict[str, pd.DataFrame]:
    """Build privacy-safe display frames for longitudinal map layers."""
    representatives = route_family_representatives.copy()

    def fid_sequence(value: object) -> list[tuple[str, int]]:
        decoded = _json_loads_list(value)
        result: list[tuple[str, int]] = []
        for item in decoded:
            if not isinstance(item, Mapping):
                continue
            county = _clean_text(item.get("county"))
            try:
                fid = int(item.get("fid"))
            except (TypeError, ValueError):
                continue
            if county:
                result.append((county, fid))
        return result

    representatives["latlon_sequence"] = representatives[
        "county_fid_sequence_json"
    ].map(lambda value: _route_latlon(fid_sequence(value), road_context_lookup))
    representatives = representatives.loc[
        representatives["latlon_sequence"].map(len).ge(2)
    ].copy()
    representatives["route_label"] = representatives["family_name"]
    representatives["dominant_road_names"] = representatives["family_name"]
    safe_columns = [
        "route_family_id",
        "origin_cluster_id",
        "origin_label",
        "destination_cluster_id",
        "destination_label",
        "family_name",
        "route_label",
        "dominant_road_names",
        "trip_count",
        "overall_route_share",
        "latlon_sequence",
    ]
    representatives = representatives[safe_columns]
    families = representatives.loc[representatives["trip_count"].ge(5)].sort_values(
        "trip_count", ascending=False
    ).head(30)

    def transition_routes(family_column: str, *, later: bool) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for transition in transitions.loc[
            transitions.get("public_reporting_eligible", True) == True  # noqa: E712
        ].to_dict(orient="records"):
            family_id = str(transition.get(family_column, ""))
            match = representatives.loc[
                representatives["route_family_id"].eq(family_id)
            ]
            if match.empty:
                continue
            row = match.iloc[0].to_dict()
            row.update(
                {
                    "route_label": (
                        "Later route: " if later else "Earlier route: "
                    )
                    + str(row["family_name"]),
                    "month_a": transition.get(
                        "later_start" if later else "baseline_start"
                    ),
                    "month_b": transition.get(
                        "later_end" if later else "baseline_end"
                    ),
                    "plain_english_story": transition.get("plain_english_story"),
                    "confidence": transition.get("confidence"),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    early = transition_routes("baseline_route_family_id", later=False)
    later = transition_routes("later_route_family_id", later=True)
    sustained = later.copy()

    temporary_rows: list[dict[str, object]] = []
    for deviation in temporary_deviations.to_dict(orient="records"):
        match = representatives.loc[
            representatives["route_family_id"].eq(
                str(deviation.get("route_family_id", ""))
            )
        ]
        if match.empty:
            continue
        row = match.iloc[0].to_dict()
        row.update(
            {
                "route_label": "Temporary route: " + str(row["family_name"]),
                "month_a": deviation.get("episode_start_month"),
                "month_b": deviation.get("episode_end_month"),
                "plain_english_story": deviation.get("plain_english_story"),
                "confidence": deviation.get("confidence"),
            }
        )
        temporary_rows.append(row)
    temporary = pd.DataFrame(temporary_rows)

    home_rows = enriched_clusters.loc[
        enriched_clusters["privacy_flag"].eq("HOME_SENSITIVE")
    ]
    home_id = str(home_rows.iloc[0]["cluster_id"]) if not home_rows.empty else ""
    point_lookup = {
        str(row.cluster_id): (float(row.centroid_lat), float(row.centroid_lon))
        for row in enriched_clusters.loc[
            enriched_clusters["privacy_flag"].ne("HOME_SENSITIVE")
        ].itertuples(index=False)
    }
    point_lookup[home_id] = generalized_home_coordinates
    chain_rows: list[dict[str, object]] = []
    for chain in repeated_chains.loc[
        repeated_chains["occurrence_count"].ge(3)
    ].sort_values("occurrence_count", ascending=False).head(20).to_dict(orient="records"):
        cluster_ids = [str(value) for value in _json_loads_list(chain.get("cluster_sequence_json"))]
        points = [point_lookup[value] for value in cluster_ids if value in point_lookup]
        collapsed = [
            point for index, point in enumerate(points) if index == 0 or point != points[index - 1]
        ]
        if len(collapsed) < 2:
            continue
        chain_rows.append(
            {
                "route_label": chain.get("public_chain"),
                "public_chain": chain.get("public_chain"),
                "occurrence_count": chain.get("occurrence_count"),
                "trip_count": chain.get("occurrence_count"),
                "month_a": chain.get("first_observed_date"),
                "month_b": chain.get("last_observed_date"),
                "latlon_sequence": collapsed,
            }
        )
    return {
        "repeated_trip_chains": pd.DataFrame(chain_rows),
        "route_families": families.reset_index(drop=True),
        "early_preferred_routes": early,
        "later_preferred_routes": later,
        "sustained_route_changes": sustained,
        "temporary_alternatives": temporary,
    }


def _json_safe(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _json_loads_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return decoded if isinstance(decoded, list) else []


def _safe_records(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> list[dict[str, object]]:
    if frame.empty:
        return []
    selected = frame[[column for column in columns if column in frame]] if columns else frame
    return [_json_safe(record) for record in selected.to_dict(orient="records")]  # type: ignore[list-item]


def prepare_poi_output(enriched_clusters: pd.DataFrame) -> pd.DataFrame:
    """Return the requested POI-enriched cluster schema plus audit fields."""
    output = enriched_clusters.copy()
    output["total_visits"] = output["total_visit_count"]
    output["unique_days"] = output["unique_days_visited"]
    output["months_seen"] = output["months_visited"]
    output["recurrence_pattern"] = output["recurring_frequency"]
    required = [
        "cluster_id",
        "centroid_lat",
        "centroid_lon",
        "medoid_lat",
        "medoid_lon",
        "privacy_flag",
        "generalized_location",
        "reverse_geocoded_address",
        "inferred_role",
        "role_confidence",
        "total_visits",
        "unique_days",
        "months_seen",
        "recurrence_pattern",
        "typical_arrival_time",
        "typical_departure_time",
        "weekday_share",
        "weekend_share",
        "median_dwell_minutes",
        "selected_poi_name",
        "selected_poi_category",
        "selected_poi_types",
        "selected_poi_address",
        "selected_poi_distance_m",
        "selected_poi_source",
        "selected_poi_search_radius_m",
        "selected_poi_retrieved_at_utc",
        "selected_poi_match_score",
        "selected_poi_google_maps_uri",
        "alternative_pois_json",
        "behavioral_evidence",
        "map_evidence",
        "classification_reason",
        "limitations",
    ]
    audit = [
        column
        for column in (
            "county",
            "coordinate_spread_m",
            "origin_count",
            "destination_count",
            "activity_visit_count",
            "recurrence_basis",
            "data_quality_flags",
            "cluster_stability_status",
            "cluster_stability_evidence_json",
            "home_score",
            "home_candidate_rank",
            "home_score_evidence",
            "role_evidence_score",
            "competing_explanation",
            "uncertainty_statement",
            "poi_match_quality",
            "poi_match_reasoning",
            "reverse_geocode_source",
            "enrichment_attempted",
            "geocoding_attempted",
            "places_search_attempted",
            "places_search_completed",
            "places_search_radius_m",
            "places_search_stop_reason",
            "selected_poi_latitude",
            "selected_poi_longitude",
            "selected_public_label",
            "multi_tenant_flag",
            "previous_label",
            "valid_stay_count",
            "dwell_q25_minutes",
            "dwell_q75_minutes",
            "share_5_to_20_minutes",
            "share_20_to_60_minutes",
            "share_1_to_3_hours",
            "share_over_3_hours",
            "censored_continuity_count",
            "censored_overnight_association_count",
            "micro_stop_boundary_count",
            "behavioral_plausibility",
            "map_plausibility",
            "trip_chain_patterns",
            "workplace_gate_json",
            "specific_candidate_poi_name",
            "specific_candidate_poi_address",
        )
        if column in output
    ]
    return output[[*required, *audit]].copy()


def build_activity_role_validation(enriched_clusters: pd.DataFrame) -> pd.DataFrame:
    """Return a transparent before/after role audit without public home details."""
    rows: list[dict[str, object]] = []
    for record in enriched_clusters.to_dict(orient="records"):
        is_home = str(record.get("privacy_flag")) == "HOME_SENSITIVE"
        context = (
            _clean_text(record.get("generalized_location"))
            if is_home
            else _clean_text(record.get("selected_public_label"))
            or _clean_text(record.get("osm_context"))
            or "unresolved area"
        )
        dwell_distribution = {
            "5_to_20_minutes": _json_safe(record.get("share_5_to_20_minutes")),
            "20_to_60_minutes": _json_safe(record.get("share_20_to_60_minutes")),
            "1_to_3_hours": _json_safe(record.get("share_1_to_3_hours")),
            "over_3_hours": _json_safe(record.get("share_over_3_hours")),
            "micro_boundaries": int(_safe_float(record.get("micro_stop_boundary_count"), 0.0)),
        }
        rows.append(
            {
                "cluster_id": record.get("cluster_id"),
                "previous_label": record.get("previous_label"),
                "revised_label": record.get("inferred_role"),
                "selected_public_label": context,
                "poi_or_complex_context": context,
                "multi_tenant_flag": bool(record.get("multi_tenant_flag", False)),
                "valid_stay_count": int(_safe_float(record.get("valid_stay_count"), 0.0)),
                "median_dwell_minutes": _json_safe(record.get("median_dwell_minutes")),
                "dwell_distribution_json": _json_dumps(dwell_distribution),
                "typical_arrival": record.get("typical_arrival_time"),
                "typical_departure": record.get("typical_departure_time"),
                "weekday_share": _json_safe(record.get("weekday_share")),
                "recurrence": record.get("recurring_frequency"),
                "trip_chain_patterns": record.get("trip_chain_patterns"),
                "behavioral_plausibility": record.get("behavioral_plausibility"),
                "map_plausibility": record.get("map_plausibility"),
                "strongest_competing_explanation": record.get("competing_explanation"),
                "revised_confidence": record.get("role_confidence"),
                "revision_reason": record.get("classification_reason"),
            }
        )
    return pd.DataFrame(rows)


def _location_record(row: Mapping[str, Any]) -> dict[str, object]:
    return {
        "cluster_id": row.get("cluster_id"),
        "place_name": _clean_text(row.get("selected_public_label"))
        or _clean_text(row.get("selected_poi_name")),
        "address": _clean_text(row.get("selected_poi_address")),
        "generalized_location": _clean_text(row.get("generalized_location")),
        "inferred_role": row.get("inferred_role"),
        "confidence": row.get("role_confidence"),
        "evidence_score": _json_safe(row.get("role_evidence_score")),
        "visit_pattern": row.get("recurring_frequency"),
        "total_visits": int(_safe_float(row.get("total_visit_count"), 0.0)),
        "months_seen": int(_safe_float(row.get("months_visited"), 0.0)),
        "first_observed_date": row.get("first_date_observed"),
        "last_observed_date": row.get("last_date_observed"),
        "typical_arrival_time": row.get("typical_arrival_time"),
        "typical_departure_time": row.get("typical_departure_time"),
        "median_dwell_minutes": _json_safe(row.get("median_dwell_minutes")),
        "valid_stay_count": int(_safe_float(row.get("valid_stay_count"), 0.0)),
        "multi_tenant": bool(row.get("multi_tenant_flag", False)),
        "behavioral_evidence": row.get("behavioral_evidence"),
        "map_evidence": row.get("map_evidence"),
        "competing_explanation": row.get("competing_explanation"),
        "uncertainty_statement": row.get("uncertainty_statement"),
        "classification_reason": row.get("classification_reason"),
        "google_maps_uri": _clean_text(row.get("selected_poi_google_maps_uri")),
    }


def build_behavior_insights_document(
    *,
    trips: pd.DataFrame,
    clusters: pd.DataFrame,
    stays: pd.DataFrame,
    activity_validation: pd.DataFrame,
    repeated_chains: pd.DataFrame,
    recurring_patterns: pd.DataFrame,
    od_changes: pd.DataFrame,
    od_summary: pd.DataFrame,
    route_families: pd.DataFrame,
    route_family_monthly: pd.DataFrame,
    longitudinal_transitions: pd.DataFrame,
    temporary_deviations: pd.DataFrame,
    monthly_highway_trends: pd.DataFrame,
    clustering: ClusterSelection,
    api_usage: Mapping[str, object],
) -> dict[str, object]:
    """Build a public JSON document that contains no exact home data."""
    home = clusters.loc[clusters["privacy_flag"].eq("HOME_SENSITIVE")].iloc[0]
    non_home = clusters.loc[clusters["privacy_flag"].ne("HOME_SENSITIVE")]
    role_candidates = non_home.loc[
        non_home["enrichment_attempted"].eq(True)
        & ~non_home["inferred_role"].isin(
            ["unknown recurring destination", "one-time destination"]
        )
    ].copy()
    role_candidates = role_candidates.sort_values(
        ["role_evidence_score", "total_visit_count"], ascending=[False, False]
    )
    role_records = [
        _location_record(record) for record in role_candidates.to_dict(orient="records")
    ]
    finding_records = [
        record
        for record in role_records
        if record["confidence"] in {"high", "medium"}
        and (record["place_name"] or record["generalized_location"])
    ]

    def records_for(fragment: str) -> list[dict[str, object]]:
        return [
            record
            for record in role_records
            if fragment in str(record["inferred_role"]).lower()
        ]

    def records_for_any(*fragments: str) -> list[dict[str, object]]:
        return [
            record
            for record in role_records
            if any(
                fragment in str(record["inferred_role"]).lower()
                for fragment in fragments
            )
        ]

    pair_counts = (
        trips.loc[
            trips["origin_cluster_id"].ne("UNCLUSTERED")
            & trips["destination_cluster_id"].ne("UNCLUSTERED")
            & trips["origin_cluster_id"].ne(trips["destination_cluster_id"])
        ]
        .groupby(
            [
                "origin_cluster_id",
                "origin_label",
                "destination_cluster_id",
                "destination_label",
            ]
        )
        .agg(trips=("trip_id", "nunique"), months=("month", "nunique"))
        .reset_index()
        .sort_values(["trips", "months"], ascending=False)
    )
    routine: dict[str, object]
    closed_home_chains = repeated_chains.loc[
        repeated_chains["cluster_sequence_json"].astype(str).map(
            lambda value: (
                bool(decoded := _json_loads_list(value))
                and str(decoded[0]) == str(home.cluster_id)
                and str(decoded[-1]) == str(home.cluster_id)
                and len(decoded) >= 3
            )
        )
    ].sort_values(["occurrence_count", "months_visited"], ascending=False)
    if not closed_home_chains.empty:
        chain = closed_home_chains.iloc[0]
        routine = {
            "summary": (
                f"The strongest complete recorded loop was {chain.public_chain}, "
                f"observed {int(chain.occurrence_count)} times across "
                f"{int(chain.months_visited)} months, typically beginning around "
                f"{chain.typical_start_time}."
            ),
            "evidence": (
                "The chain begins and ends in the generalized home area and is "
                "reconstructed only from spatially continuous same-service-day trips."
            ),
            "confidence": "high" if chain.occurrence_count >= 15 else "medium",
            "competing_explanation": (
                "A repeated loop establishes a travel routine, not the purpose of its stop."
            ),
            "chain_id": chain.chain_id,
        }
    elif pair_counts.empty:
        routine = {
            "summary": "No stable origin-destination routine met the recurrence threshold.",
            "confidence": "low",
        }
    else:
        top_pair = pair_counts.iloc[0]
        routine = {
            "summary": (
                f"The strongest repeated movement was {top_pair.origin_label} to "
                f"{top_pair.destination_label}, observed in {int(top_pair.trips)} trips "
                f"across {int(top_pair.months)} months."
            ),
            "evidence": "Repeated source trips shared the same compact origin and destination clusters.",
            "confidence": "high" if top_pair.trips >= 20 and top_pair.months >= 6 else "medium",
            "competing_explanation": "A repeated activity-area connection does not by itself establish commuting or trip purpose.",
        }

    key_findings: list[dict[str, object]] = [
        {
            "title": "Likely home area",
            "finding": (
                f"{home.generalized_location} was the strongest home candidate: the most frequent "
                f"return/departure area with {int(home.overnight_association_count)} overnight "
                f"associations across {int(home.months_visited)} months."
            ),
            "confidence": home.role_confidence,
        }
    ]
    workplace_records = records_for("workplace")
    if workplace_records:
        workplace = workplace_records[0]
        key_findings.append(
            {
                "title": "Workplace candidate",
                "finding": (
                    f"{workplace['place_name'] or workplace['generalized_location']} "
                    "met the minimum repeated multi-hour-stay criteria for a workplace candidate; "
                    "employment remains unconfirmed."
                ),
                "confidence": workplace["confidence"],
            }
        )
    else:
        key_findings.append(
            {
                "title": "No workplace identified",
                "finding": (
                    "No destination combined sufficient reconstructed multi-hour weekday stays "
                    "with defensible workplace map context. The frequent Coconut Creek Plaza "
                    "cluster is a short-stop pattern and is not classified as work."
                ),
                "confidence": "high",
            }
        )
    finding_groups = (
        ("healthcare", "Possible healthcare pattern"),
        ("medical", "Possible healthcare-context pattern"),
        ("university", "Possible university/college pattern"),
        ("school", "Possible school/daycare pattern"),
        ("shopping", "Recurring shopping destination"),
        ("gas station", "Recurring fuel stop"),
    )
    used_clusters: set[str] = set()
    for fragment, title in finding_groups:
        matches = [
            record
            for record in role_records
            if fragment in str(record["inferred_role"]).lower()
            and str(record["cluster_id"]) not in used_clusters
        ]
        if not matches:
            continue
        confidence_order = {"high": 0, "medium": 1, "low": 2}
        matches.sort(
            key=lambda record: (
                confidence_order.get(str(record.get("confidence")), 3),
                -_safe_float(record.get("evidence_score"), 0.0),
            )
        )
        record = matches[0]
        used_clusters.add(str(record["cluster_id"]))
        key_findings.append(
            {
                "title": title,
                "finding": (
                    f"{record['place_name'] or record['generalized_location']} showed a "
                    f"{record['visit_pattern']} endpoint pattern; the role remains inferred."
                ),
                "confidence": record["confidence"],
            }
        )
        if len(key_findings) >= 5:
            break
    public_transitions = longitudinal_transitions.copy()
    if "public_reporting_eligible" in public_transitions:
        public_transitions = public_transitions.loc[
            public_transitions["public_reporting_eligible"].eq(True)
        ]
    if not public_transitions.empty:
        transition = public_transitions.sort_values(
            "route_share_change_percentage_points", ascending=False
        ).iloc[0]
        key_findings.append(
            {
                "title": "Sustained route-family change",
                "finding": transition["plain_english_story"],
                "confidence": transition["confidence"],
            }
        )
    if not temporary_deviations.empty:
        deviation = temporary_deviations.sort_values(
            ["episode_family_trips", "peak_route_share"], ascending=False
        ).iloc[0]
        key_findings.append(
            {
                "title": "Temporary route deviation",
                "finding": deviation["plain_english_story"],
                "confidence": deviation["confidence"],
            }
        )
    recurring_public_columns = [
        "cluster_id",
        "named_poi_or_generalized_location",
        "address",
        "pattern_type",
        "first_month",
        "last_month",
        "months_visited",
        "visit_count",
        "visit_frequency",
        "typical_time",
        "median_dwell_minutes",
        "inferred_activity",
        "confidence",
        "trend",
        "period_visit_counts_json",
        "period_visit_rates_json",
        "typical_time_by_period_json",
        "dwell_by_period_json",
        "plain_english_narrative",
        "alternative_interpretation",
        "uncertainty_statement",
    ]
    od_public_columns = [
        column
        for column in (
            "origin_cluster_id",
            "origin_label",
            "destination_cluster_id",
            "destination_label",
            "month_a",
            "month_b",
            "trip_count_a",
            "trip_count_b",
            "dominant_route_a",
            "dominant_route_b",
            "major_roads_removed",
            "major_roads_added",
            "highway_share_a",
            "highway_share_b",
            "surface_street_share_a",
            "surface_street_share_b",
            "distance_change",
            "travel_time_change",
            "RCCI",
            "plain_english_story",
            "confidence",
            "limitations",
        )
        if column in od_changes
    ]
    important_places = clusters.loc[
        clusters["privacy_flag"].ne("HOME_SENSITIVE")
        & (
            clusters["total_visit_count"].ge(10)
            | clusters["valid_stay_count"].ge(5)
        )
    ].sort_values(
        ["total_visit_count", "valid_stay_count"], ascending=False
    )
    important_place_records = [
        _location_record(record)
        for record in important_places.head(30).to_dict(orient="records")
    ]

    observed_months = sorted(trips["month"].dropna().astype(str).unique())
    early_months = set(observed_months[: len(observed_months) // 2])
    late_months = set(observed_months[len(observed_months) // 2 :])

    def weighted_share(frame: pd.DataFrame, numerator: str) -> float:
        denominator = pd.to_numeric(
            frame["route_distance_m"], errors="coerce"
        ).clip(lower=0).sum()
        return (
            float(
                pd.to_numeric(frame[numerator], errors="coerce").clip(lower=0).sum()
                / denominator
            )
            if denominator > 0
            else float("nan")
        )

    early_trips = trips.loc[trips["month"].isin(early_months)]
    late_trips = trips.loc[trips["month"].isin(late_months)]
    overall_highway_share = weighted_share(trips, "highway_distance_m")
    early_highway_share = weighted_share(early_trips, "highway_distance_m")
    late_highway_share = weighted_share(late_trips, "highway_distance_m")
    highway_change = late_highway_share - early_highway_share
    all_trip_monthly = monthly_highway_trends.loc[
        monthly_highway_trends["scope"].eq("all_trips")
        & monthly_highway_trends["trip_count"].gt(0)
    ].sort_values("observed_month_index")
    highway_monthly_values = pd.to_numeric(
        all_trip_monthly["distance_weighted_highway_share"], errors="coerce"
    ).dropna()
    monthly_differences = highway_monthly_values.diff().dropna()
    monotonic_monthly = bool(
        not monthly_differences.empty
        and (
            monthly_differences.ge(-1e-9).all()
            or monthly_differences.le(1e-9).all()
        )
    )
    highway_interpretation = (
        "No material full-period shift from highways to surface streets was detected."
        if math.isfinite(highway_change) and abs(highway_change) < 0.05
        else (
            "Highway distance share was higher in the late study window, but the monthly "
            "series was not monotonic; this is not evidence of a single permanent switch."
            if highway_change > 0 and not monotonic_monthly
            else "Highway distance share was higher in the late study window."
            if highway_change > 0
            else "Highway distance share was lower in the late study window, but the monthly "
            "series was not monotonic; this is not evidence of a single permanent switch."
            if not monotonic_monthly
            else "Highway distance share was lower in the late study window."
        )
    )

    status_counts = (
        stays["stay_status"].value_counts().sort_index().to_dict()
        if "stay_status" in stays
        else {}
    )
    data_quality = {
        "trip_segmentation": "Explicit source trip/session boundaries; two cross-county fragments were stitched by source identity.",
        "source_trip_count": int(len(trips)),
        "source_fragment_count": int(
            pd.to_numeric(trips["source_fragment_count"], errors="coerce").fillna(0).sum()
        ),
        "duplicate_trip_ids": int(trips["trip_id"].duplicated().sum()),
        "trip_quality_flag_counts": quality_flag_counts(trips),
        "flagged_trip_count": int(
            trips["data_quality_flags"].astype(str).ne("[]").sum()
        ),
        "quality_policy": (
            "Questionable trips are retained with deterministic flags; route-family "
            "eligibility applies additional OD-specific filtering."
        ),
        "observed_month_count": int(len(observed_months)),
        "timezone": "America/New_York (timestamps normalized with daylight-saving support)",
        "share_of_trips_starting_at_or_after_noon": float(
            pd.to_numeric(trips["start_hour"], errors="coerce").ge(12).mean()
        ),
        "stay_status_counts": _json_safe(status_counts),
        "measured_stay_count": int(status_counts.get("MEASURED_STAY", 0)),
        "censored_stays_excluded_from_dwell_medians": True,
        "recording_window_caution": (
            "The source is strongly afternoon/evening weighted; conventional morning-commute "
            "inference is therefore weak."
        ),
    }

    revised_roles = activity_validation.loc[
        activity_validation["previous_label"].fillna("").astype(str)
        != activity_validation["revised_label"].fillna("").astype(str)
    ].copy()
    revised_roles = revised_roles.sort_values(
        ["valid_stay_count", "cluster_id"], ascending=[False, True]
    )

    od_public = [
        column
        for column in (
            "origin_cluster_id",
            "origin_label",
            "destination_cluster_id",
            "destination_label",
            "total_trip_count",
            "eligible_direct_trip_count",
            "eligible_unique_days",
            "eligible_months",
            "median_od_separation_m",
            "median_direct_route_distance_m",
            "median_direct_circuity",
            "route_family_count",
            "dominant_route_family_id",
            "dominant_route_family_share",
            "early_late_total_variation",
            "assignment_agreement_similarity_0_65",
            "assignment_agreement_similarity_0_75",
        )
        if column in od_summary
    ]
    major_od = od_summary.loc[
        od_summary["eligible_direct_trip_count"].ge(10)
        & od_summary["eligible_months"].ge(3)
    ].sort_values("eligible_direct_trip_count", ascending=False)
    major_family_ids = set(
        route_families.sort_values("trip_count", ascending=False)
        .head(30)["route_family_id"]
        .astype(str)
    )
    family_public_columns = [
        column
        for column in (
            "route_family_id",
            "origin_cluster_id",
            "origin_label",
            "destination_cluster_id",
            "destination_label",
            "family_name",
            "backbone_roads_json",
            "trip_count",
            "unique_days",
            "months_seen",
            "first_month",
            "last_month",
            "overall_route_share",
            "median_route_distance_m",
            "median_duration_seconds",
            "similarity_threshold",
            "lcs_merge_threshold",
        )
        if column in route_families
    ]
    monthly_public_columns = [
        column
        for column in (
            "origin_cluster_id",
            "origin_label",
            "destination_cluster_id",
            "destination_label",
            "route_family_id",
            "family_name",
            "month",
            "observed_month_index",
            "family_trip_count",
            "eligible_od_trip_count",
            "route_share",
            "rolling_3_observed_month_share",
            "distance_weighted_highway_share",
            "distance_weighted_surface_street_share",
            "median_route_distance_m",
            "median_duration_seconds",
            "data_sufficiency",
            "confidence",
        )
        if column in route_family_monthly
    ]

    destination_changes: list[dict[str, object]] = []
    if observed_months:
        late_start_index = 2 * len(observed_months) // 3
        stopped_cutoff_index = max(len(observed_months) - 6, 0)
        month_index = {month: index for index, month in enumerate(observed_months)}
        for record in recurring_patterns.to_dict(orient="records"):
            first = str(record.get("first_month", ""))
            last = str(record.get("last_month", ""))
            visits = int(_safe_float(record.get("visit_count"), 0.0))
            months_seen = int(_safe_float(record.get("months_visited"), 0.0))
            change_type = ""
            if (
                first in month_index
                and month_index[first] >= late_start_index
                and visits >= 8
                and months_seen >= 5
                and str(record.get("confidence")) in {"high", "medium"}
            ):
                change_type = "new recurring destination"
            elif (
                last in month_index
                and month_index[last] < stopped_cutoff_index
                and visits >= 8
                and months_seen >= 5
                and str(record.get("confidence")) in {"high", "medium"}
            ):
                change_type = "stopped appearing"
            if change_type:
                destination_changes.append(
                    {
                        "cluster_id": record.get("cluster_id"),
                        "location": record.get("named_poi_or_generalized_location"),
                        "address": record.get("address"),
                        "change_type": change_type,
                        "first_month": first,
                        "last_month": last,
                        "months_visited": months_seen,
                        "visit_count": visits,
                        "confidence": record.get("confidence"),
                        "alternative_interpretation": record.get(
                            "alternative_interpretation"
                        ),
                    }
                )

    behavior_timeline: list[dict[str, object]] = [
        {
            "period": observed_months[0] if observed_months else "",
            "event": "Observation begins",
            "evidence": f"{len(trips)} explicit source trips are analyzed across the full window.",
            "confidence": "high",
        }
    ]
    for record in destination_changes:
        behavior_timeline.append(
            {
                "period": record["first_month"]
                if record["change_type"] == "new recurring destination"
                else record["last_month"],
                "event": record["change_type"],
                "evidence": (
                    f"{record['location']} was recorded {record['visit_count']} times across "
                    f"{record['months_visited']} months."
                ),
                "confidence": record["confidence"],
            }
        )
    for record in _safe_records(public_transitions):
        behavior_timeline.append(
            {
                "period": record.get("adoption_start"),
                "event": "Sustained route-family change",
                "evidence": record.get("plain_english_story"),
                "confidence": record.get("confidence"),
            }
        )
    for record in _safe_records(temporary_deviations):
        behavior_timeline.append(
            {
                "period": record.get("episode_start_month"),
                "event": "Temporary route deviation",
                "evidence": record.get("plain_english_story"),
                "confidence": record.get("confidence"),
            }
        )
    behavior_timeline.append(
        {
            "period": observed_months[-1] if observed_months else "",
            "event": "Observation ends",
            "evidence": "The final recorded month is an observation boundary, not proof that a routine ended.",
            "confidence": "high",
        }
    )

    limitations = [
        "Activity purposes are inferred and are not confirmed.",
        "POI proximity does not prove that the driver visited a place.",
        "GPS endpoints can fall in parking lots, entrance roads, or nearby streets.",
        "Google and OpenStreetMap listings may be incomplete, outdated, or changed.",
        "Household, employment, school, healthcare, religion, and family conclusions are not confirmed.",
        "The recording window contains little conventional weekday-morning travel, which weakens home/work timing inference.",
        "RCCI and route differences cannot establish congestion, construction, toll avoidance, or preference as a cause.",
        "Censored inter-trip intervals are retained as continuity evidence but excluded from measured dwell-time medians.",
        "Route-family persistence is counted over observed months; sparse or missing calendar months are flagged in the monthly table.",
    ]
    home_coverage_score = min(
        1.0,
        0.60 * _safe_float(home.recurrence_score, 0.0)
        + 0.20 * min(_safe_float(home.overnight_association_count, 0.0) / 20.0, 1.0)
        + 0.20 * min(_safe_float(home.total_visit_count, 0.0) / 100.0, 1.0),
    )
    home_alternatives: list[dict[str, object]] = []
    for candidate in clusters.loc[
        clusters["privacy_flag"].ne("HOME_SENSITIVE")
    ].sort_values("home_score", ascending=False).head(3).itertuples(index=False):
        score_gap = _safe_float(home.home_score, 0.0) - _safe_float(
            candidate.home_score, 0.0
        )
        home_alternatives.append(
            {
                "cluster_id": candidate.cluster_id,
                "generalized_location": f"Alternative recurring area in {candidate.county}",
                "home_score": _json_safe(candidate.home_score),
                "evidence": candidate.home_score_evidence,
                "why_not_selected": (
                    f"Its transparent home score was {score_gap:.2f} below the selected "
                    "cluster and it had weaker combined overnight/recurrence/network evidence."
                ),
            }
        )
    return {
        "driver_id": DRIVER_ID,
        "analysis_period": {
            "start": trips["start_timestamp"].min(),
            "end": trips["end_timestamp"].max(),
            "observed_months": int(trips["month"].nunique()),
            "source_trips": int(len(trips)),
        },
        "api_usage": _json_safe(api_usage),
        "data_quality": data_quality,
        "clustering": {
            "method": "projected-coordinate DBSCAN in EPSG:26917",
            "selected_radius_m": clustering.radius_m,
            "min_samples": CLUSTER_MIN_SAMPLES,
            "selection_rationale": (
                "The smallest radius within 0.02 of the best aggregate score was "
                "selected to preserve adjacent POIs while maintaining recurring-endpoint coverage."
            ),
            "evaluated_radii": _safe_records(clustering.diagnostics),
            "important_cluster_stability": _safe_records(
                clusters.loc[clusters.get("is_important_cluster", False).eq(True)],
                [
                    "cluster_id",
                    "selected_endpoint_count",
                    "cluster_stability_status",
                    "minimum_dominant_retention",
                    "minimum_best_jaccard",
                    "maximum_noise_share",
                    "maximum_merge_contamination_share",
                    "data_quality_flags",
                ],
            )
            if "is_important_cluster" in clusters
            else [],
        },
        "likely_home": {
            "cluster_id": home.cluster_id,
            "generalized_location": home.generalized_location,
            "confidence": home.role_confidence,
            "home_score": _json_safe(home.home_score),
            "data_coverage_score": home_coverage_score,
            "evidence": home.behavioral_evidence,
            "residential_context": home.map_evidence,
            "privacy_flag": "HOME_SENSITIVE",
            "uncertainty": home.uncertainty_statement,
            "selection_reason": home.classification_reason,
            "alternative_candidates": home_alternatives,
        },
        "likely_workplaces": workplace_records,
        "possible_school_or_daycare_locations": records_for("school"),
        "possible_healthcare_locations": records_for_any("healthcare", "medical"),
        "shopping_and_errand_locations": [
            record
            for record in role_records
            if any(
                token in str(record["inferred_role"]).lower()
                for token in ("grocery", "shopping", "retail", "restaurant", "gas", "bank")
            )
        ],
        "recreation_and_other_locations": [
            record
            for record in role_records
            if record not in records_for("workplace")
            and record not in records_for("school")
            and record not in records_for_any("healthcare", "medical")
        ],
        "important_places": important_place_records,
        "activity_role_revisions": _safe_records(revised_roles.head(30)),
        "repeated_trip_chains": _safe_records(
            repeated_chains.sort_values(
                ["occurrence_count", "months_visited"], ascending=False
            ).head(30)
        ),
        "recurring_patterns": _safe_records(recurring_patterns, recurring_public_columns),
        "recurring_destination_trends": _safe_records(
            recurring_patterns.sort_values(
                ["visit_count", "months_visited"], ascending=False
            ).head(40),
            recurring_public_columns,
        ),
        "major_od_pairs": _safe_records(major_od.head(40), od_public),
        "route_families": _safe_records(
            route_families.loc[
                route_families["route_family_id"].astype(str).isin(major_family_ids)
            ].sort_values("trip_count", ascending=False),
            family_public_columns,
        ),
        "route_family_monthly_shares": _safe_records(
            route_family_monthly.loc[
                route_family_monthly["route_family_id"].astype(str).isin(
                    major_family_ids
                )
            ],
            monthly_public_columns,
        ),
        "longitudinal_route_transitions": _safe_records(public_transitions),
        "longitudinal_route_transitions_audit": _safe_records(
            longitudinal_transitions
        ),
        "temporary_route_deviations": _safe_records(temporary_deviations),
        "highway_surface_street_summary": {
            "full_period_highway_distance_share": _json_safe(overall_highway_share),
            "full_period_surface_street_distance_share": _json_safe(
                1.0 - overall_highway_share
                if math.isfinite(overall_highway_share)
                else float("nan")
            ),
            "early_window_highway_distance_share": _json_safe(early_highway_share),
            "late_window_highway_distance_share": _json_safe(late_highway_share),
            "highway_share_change": _json_safe(highway_change),
            "monthly_series_monotonic": monotonic_monthly,
            "interpretation": highway_interpretation,
            "monthly_trends": _safe_records(monthly_highway_trends),
        },
        "behavior_timeline": behavior_timeline,
        "od_route_changes": _safe_records(od_changes, od_public_columns),
        "new_or_disappearing_destinations": destination_changes,
        "likely_routine": routine,
        "key_findings": key_findings,
        "limitations": limitations,
    }


def _write_json(data: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(data), indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _assert_public_json_privacy(
    path: Path,
    *,
    exact_home_address: str,
    exact_home_lat: float,
    exact_home_lon: float,
) -> None:
    document = path.read_text(encoding="utf-8")
    if exact_home_address and exact_home_address.casefold() in document.casefold():
        raise BehaviorAnalysisError("Public JSON contains the exact home address")
    coordinate_pairs = [
        (f"{exact_home_lat:.{digits}f}", f"{exact_home_lon:.{digits}f}")
        for digits in range(5, 9)
    ]
    if any(lat in document and lon in document for lat, lon in coordinate_pairs):
        raise BehaviorAnalysisError("Public JSON contains the exact home coordinate")
    if "GOOGLE_MAPS_API_KEY" in document or re.search(
        r"AIza[0-9A-Za-z_-]{20,}", document
    ):
        raise BehaviorAnalysisError("Public JSON contains API credential material")


def build_driver_1003_real_world_behavior(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    report_path: Path = CURATED_REPORT_PATH,
    update_report: bool = True,
    use_google: bool = True,
    google_cache_only: bool = False,
    max_non_home_clusters: int = 20,
    google_request_budget: int = 90,
    prior_google_requests: int = 0,
    access_test_requests: int = 0,
    pipeline_output_root: Path = ROOT / "sflorida_outputs",
    phase2_deliverable_root: Path = ROOT / "deliverables" / "google_drive_phase2",
) -> BuildResult:
    """Run the complete analysis, write all requested artifacts, and validate."""
    if prior_google_requests < 0 or prior_google_requests >= MAX_BILLABLE_REQUESTS:
        raise ValueError("prior_google_requests must be between 0 and 99")
    if access_test_requests < 0 or access_test_requests > prior_google_requests:
        raise ValueError("access_test_requests must be between 0 and prior_google_requests")
    if google_request_budget + prior_google_requests > MAX_BILLABLE_REQUESTS:
        google_request_budget = MAX_BILLABLE_REQUESTS - prior_google_requests
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    pipeline_output_root = Path(pipeline_output_root)
    phase2_deliverable_root = Path(phase2_deliverable_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    timeline_path = (
        pipeline_output_root / "phase2" / "driver_timelines" / "driver_1_timeline.csv"
    )
    monthly_node_path = (
        phase2_deliverable_root
        / "driver_1003_monthly_graphs"
        / "data"
        / "driver_1003_all_monthly_nodes.csv"
    )
    configured_county_paths = county_paths_for_output_root(pipeline_output_root)
    timeline = load_driver_timeline(timeline_path)
    local_road_context = add_local_toll_flags(
        load_road_context(monthly_node_path),
        toll_path=pipeline_output_root / "fdot" / "toll_roads.parquet",
    )
    trips, fragments = build_trip_summary(
        timeline,
        local_road_context,
        county_paths=configured_county_paths,
    )
    trips = validate_trip_quality(trips)
    endpoints = build_endpoint_events(trips)
    clustering = select_cluster_radius(endpoints)
    endpoints = assign_location_clusters(endpoints, clustering.radius_m)
    trips = attach_clusters_to_trips(trips, endpoints)
    trips = validate_trip_quality(trips)
    clusters, stays = summarize_location_clusters(
        trips,
        endpoints,
        selected_radius_m=clustering.radius_m,
        county_paths=configured_county_paths,
    )
    clusters = add_cluster_stability_diagnostics(
        clusters,
        endpoints,
        clustering,
    )

    google_client = None
    google_errors: list[dict[str, object]] = []
    if use_google or google_cache_only:
        from .google_places import GoogleMapsClient

        google_client = GoogleMapsClient(
            cache_dir=cache_dir,
            request_budget=google_request_budget,
            min_interval_seconds=0.12,
            max_retries=1,
            allow_network=not google_cache_only,
        )
    try:
        enriched, google_errors = enrich_location_clusters(
            clusters,
            google_client=google_client,
            max_non_home_clusters=max_non_home_clusters,
        )
        private_home_address = str(
            enriched.attrs.get("private_home_exact_address_for_validation", "")
        )
        # Build chains once with map/generalized labels so actual chain position
        # can strengthen or weaken role inference. Rebuild after classification
        # to carry the final public labels into researcher-facing output.
        trips = attach_public_location_labels(trips, enriched)
        preliminary_chains, _ = build_repeated_trip_chains(trips)
        enriched = classify_location_roles(
            enriched,
            trips=trips,
            repeated_chains=preliminary_chains,
        )
        trips = attach_public_location_labels(trips, enriched)
        repeated_chains, chain_occurrences = build_repeated_trip_chains(trips)
        activity_validation = build_activity_role_validation(enriched)

        from .behavior_routes import (
            build_road_context_lookup,
            compare_consecutive_od_months,
            compute_dominant_routes,
            load_unique_road_context,
        )

        route_context = load_unique_road_context(
            monthly_node_path, parse_geometry=True
        )
        route_context = route_context.drop(columns=["toll"], errors="ignore").merge(
            local_road_context[["county", "fid", "toll"]],
            on=["county", "fid"],
            how="left",
            validate="one_to_one",
        )
        route_context["toll"] = route_context["toll"].fillna(False).astype(bool)
        from .behavior_longitudinal import analyze_longitudinal_routes

        longitudinal = analyze_longitudinal_routes(trips, route_context)
        od_summary = longitudinal["od_summary"]
        route_families = longitudinal["route_families"]
        route_family_monthly = longitudinal["route_family_monthly_shares"]
        longitudinal_transitions = annotate_longitudinal_reporting(
            longitudinal["longitudinal_route_transitions"],
            od_summary,
            enriched,
        )
        temporary_deviations = longitudinal["temporary_route_deviations"]
        monthly_highway_trends = longitudinal["monthly_highway_surface_trends"]
        route_family_representatives = longitudinal[
            "route_family_map_representatives"
        ]
        profiles = compute_dominant_routes(
            trips.loc[
                trips["origin_cluster_id"].ne("UNCLUSTERED")
                & trips["destination_cluster_id"].ne("UNCLUSTERED")
                & trips["origin_cluster_id"].ne(trips["destination_cluster_id"])
            ],
            route_context,
            duration_col="trip_duration_seconds",
            county_col="origin_county",
        )
        od_changes = compare_consecutive_od_months(
            profiles,
            min_trips_per_month=3,
        )
        if not od_changes.empty:
            pair_stats = (
                profiles.groupby(["origin_cluster_id", "destination_cluster_id"])
                .agg(
                    od_pair_total_trips=("trip_count", "sum"),
                    od_pair_months=("month", "nunique"),
                )
                .reset_index()
            )
            od_changes = od_changes.merge(
                pair_stats,
                on=["origin_cluster_id", "destination_cluster_id"],
                how="left",
                validate="many_to_one",
            )
            od_changes = od_changes.loc[
                (od_changes["od_pair_total_trips"] >= 15)
                & (od_changes["od_pair_months"] >= 3)
            ].copy()
            location_confidence = enriched.set_index("cluster_id")[
                "role_confidence"
            ].to_dict()
            poi_match_quality = enriched.set_index("cluster_id")[
                "poi_match_quality"
            ].to_dict()
            od_changes["origin_location_confidence"] = od_changes[
                "origin_cluster_id"
            ].map(location_confidence)
            od_changes["destination_location_confidence"] = od_changes[
                "destination_cluster_id"
            ].map(location_confidence)
            od_changes["origin_poi_match_quality"] = od_changes[
                "origin_cluster_id"
            ].map(poi_match_quality)
            od_changes["destination_poi_match_quality"] = od_changes[
                "destination_cluster_id"
            ].map(poi_match_quality)
            od_changes = od_changes.sort_values(
                ["RCCI", "trip_count_a", "trip_count_b"],
                ascending=[False, False, False],
            ).reset_index(drop=True)
        recurring = build_recurring_patterns(trips, enriched, stays=stays)
        stats = google_client.request_stats if google_client is not None else None
        total_google_requests = prior_google_requests + (stats.google_requests if stats else 0)
        if total_google_requests >= MAX_BILLABLE_REQUESTS:
            raise BehaviorAnalysisError("Google request total reached or exceeded the hard stop")
        sources_used = {
            "repository matched GPS/FID data",
            "repository enriched road network",
            "OpenStreetMap local cache",
            "FDOT local toll-road layer",
        }
        if stats:
            sources_used.update(stats.sources_used)
        api_usage = {
            "google_requests": total_google_requests,
            "analysis_run_google_requests": stats.google_requests if stats else 0,
            "prior_google_requests": prior_google_requests,
            "access_test_requests_included": access_test_requests,
            "cache_hits": stats.cache_hits if stats else 0,
            "sources_used": sorted(sources_used),
            "request_budget": MAX_BILLABLE_REQUESTS,
            "errors": google_errors,
        }
        insights = build_behavior_insights_document(
            trips=trips,
            clusters=enriched,
            stays=stays,
            activity_validation=activity_validation,
            repeated_chains=repeated_chains,
            recurring_patterns=recurring,
            od_changes=od_changes,
            od_summary=od_summary,
            route_families=route_families,
            route_family_monthly=route_family_monthly,
            longitudinal_transitions=longitudinal_transitions,
            temporary_deviations=temporary_deviations,
            monthly_highway_trends=monthly_highway_trends,
            clustering=clustering,
            api_usage=api_usage,
        )

        route_lookup = build_road_context_lookup(route_context)
        monthly_routes, common_routes, changed_routes = build_route_map_frames(
            profiles, od_changes, route_lookup
        )
        home = enriched.loc[enriched["privacy_flag"].eq("HOME_SENSITIVE")].iloc[0]
        public_home_lat, public_home_lon = generalized_home_point(
            float(home.medoid_lat), float(home.medoid_lon)
        )
        longitudinal_map_frames = build_longitudinal_map_frames(
            route_family_representatives,
            longitudinal_transitions,
            temporary_deviations,
            repeated_chains,
            enriched,
            route_lookup,
            generalized_home_coordinates=(public_home_lat, public_home_lon),
        )
        from .behavior_report import (
            GeneralizedHomeArea,
            generate_verification_map,
            render_real_world_behavior_insights,
            update_report_html,
        )

        generalized_home = GeneralizedHomeArea(
            latitude=public_home_lat,
            longitude=public_home_lon,
            radius_m=800.0,
            label="Likely home area (generalized)",
            generalized_location=str(home.generalized_location),
            confidence=str(home.role_confidence),
            evidence=str(home.behavioral_evidence),
            generalization_method="0.01-degree grid rounding plus deterministic offset",
        )
        public_clusters = enriched.loc[
            enriched["privacy_flag"].ne("HOME_SENSITIVE")
        ].copy()
        map_path = output_dir / "driver_1003_poi_route_insights_map.html"
        import os

        api_key_for_scan = os.environ.get("GOOGLE_MAPS_API_KEY")
        generate_verification_map(
            public_clusters,
            generalized_home=generalized_home,
            poi_clusters=public_clusters.loc[
                public_clusters["selected_poi_name"].astype(str).str.len() > 0
            ],
            recurring_destinations=public_clusters.loc[
                public_clusters["total_visit_count"] >= 10
            ],
            monthly_routes=monthly_routes,
            common_od_routes=common_routes,
            major_route_changes=changed_routes.head(30),
            repeated_trip_chains=longitudinal_map_frames["repeated_trip_chains"],
            route_families=longitudinal_map_frames["route_families"],
            early_preferred_routes=longitudinal_map_frames[
                "early_preferred_routes"
            ],
            later_preferred_routes=longitudinal_map_frames[
                "later_preferred_routes"
            ],
            sustained_route_changes=longitudinal_map_frames[
                "sustained_route_changes"
            ],
            temporary_alternatives=longitudinal_map_frames[
                "temporary_alternatives"
            ],
            output_path=map_path,
            exact_home_address_for_validation=private_home_address,
            exact_home_coordinates_for_validation=(float(home.medoid_lat), float(home.medoid_lon)),
            exact_home_uri_for_validation=None,
            api_key_for_validation=api_key_for_scan,
        )

        paths = {
            "trip_summary": output_dir / "driver_1003_trip_summary.csv",
            "stays": output_dir / "driver_1003_stays.csv",
            "location_clusters": output_dir / "driver_1003_location_clusters.csv",
            "poi_enriched_clusters": output_dir / "driver_1003_poi_enriched_clusters.csv",
            "activity_role_validation": output_dir / "driver_1003_activity_role_validation.csv",
            "repeated_trip_chains": output_dir / "driver_1003_repeated_trip_chains.csv",
            "trip_chain_occurrences": output_dir / "driver_1003_trip_chain_occurrences.csv",
            "recurring_patterns": output_dir / "driver_1003_recurring_poi_patterns.csv",
            "od_route_changes": output_dir / "driver_1003_od_route_change_insights.csv",
            "od_summary": output_dir / "driver_1003_od_summary.csv",
            "route_families": output_dir / "driver_1003_route_families.csv",
            "route_family_monthly_shares": output_dir
            / "driver_1003_route_family_monthly_shares.csv",
            "longitudinal_route_transitions": output_dir
            / "driver_1003_longitudinal_route_transitions.csv",
            "temporary_route_deviations": output_dir
            / "driver_1003_temporary_route_deviations.csv",
            "monthly_highway_surface_trends": output_dir
            / "driver_1003_monthly_highway_surface_trends.csv",
            "route_family_map_representatives": output_dir
            / "driver_1003_route_family_map_representatives.csv",
            "map": map_path,
            "behavior_json": output_dir / "driver_1003_real_world_behavior_insights.json",
        }
        trips.drop(columns=["source_key"], errors="ignore").to_csv(
            paths["trip_summary"], index=False
        )
        stays.to_csv(paths["stays"], index=False)
        clusters.to_csv(paths["location_clusters"], index=False)
        prepare_poi_output(enriched).to_csv(paths["poi_enriched_clusters"], index=False)
        activity_validation.to_csv(paths["activity_role_validation"], index=False)
        repeated_chains.to_csv(paths["repeated_trip_chains"], index=False)
        chain_occurrences.to_csv(paths["trip_chain_occurrences"], index=False)
        recurring.to_csv(paths["recurring_patterns"], index=False)
        od_changes.to_csv(paths["od_route_changes"], index=False)
        od_summary.to_csv(paths["od_summary"], index=False)
        route_families.to_csv(paths["route_families"], index=False)
        route_family_monthly.to_csv(paths["route_family_monthly_shares"], index=False)
        longitudinal_transitions.to_csv(
            paths["longitudinal_route_transitions"], index=False
        )
        temporary_deviations.to_csv(paths["temporary_route_deviations"], index=False)
        monthly_highway_trends.to_csv(
            paths["monthly_highway_surface_trends"], index=False
        )
        route_family_representatives.to_csv(
            paths["route_family_map_representatives"], index=False
        )
        _write_json(insights, paths["behavior_json"])
        _assert_public_json_privacy(
            paths["behavior_json"],
            exact_home_address=private_home_address,
            exact_home_lat=float(home.medoid_lat),
            exact_home_lon=float(home.medoid_lon),
        )

        if update_report:
            report_pois = public_clusters.loc[
                public_clusters["poi_match_quality"].isin(["high", "medium"])
                & public_clusters["role_confidence"].isin(["high", "medium"])
            ].copy()
            report_od = od_changes.loc[
                od_changes["confidence"].isin(["high", "medium"])
                & od_changes["origin_location_confidence"].isin(["high", "medium"])
                & od_changes["destination_location_confidence"].isin(["high", "medium"])
            ].copy()
            section = render_real_world_behavior_insights(
                insights,
                poi_clusters=report_pois,
                recurring_patterns=recurring,
                od_route_changes=report_od,
                map_href="../../../../outputs/driver_1003_poi_route_insights_map.html",
                embed_map=False,
            )
            update_report_html(
                _require(report_path),
                section,
                exact_home_address=private_home_address,
                exact_home_coordinates=(float(home.medoid_lat), float(home.medoid_lon)),
                exact_home_uri=None,
                api_key=api_key_for_scan,
            )

        for name, path in paths.items():
            if not path.exists() or path.stat().st_size == 0:
                raise BehaviorAnalysisError(f"Required output is missing or empty: {name}")
        if len(trips) != timeline["source_key"].nunique() or len(fragments) != len(timeline):
            raise BehaviorAnalysisError("Trip/fragment reconciliation failed after output")
        if enriched["cluster_id"].duplicated().any():
            raise BehaviorAnalysisError("POI output contains duplicate cluster IDs")
        _require_columns(
            trips,
            {
                "trip_id",
                "session_id",
                "source_file_references",
                "data_quality_flags",
                "origin_cluster_id",
                "destination_cluster_id",
            },
            label="trip summary",
        )
        _require_columns(
            clusters,
            {
                "medoid_lat",
                "medoid_lon",
                "recurrence_basis",
                "cluster_stability_status",
                "cluster_stability_evidence_json",
                "data_quality_flags",
            },
            label="location clusters",
        )
        _require_columns(
            enriched,
            {
                "selected_poi_types",
                "selected_poi_search_radius_m",
                "selected_poi_retrieved_at_utc",
                "selected_poi_match_score",
                "places_search_attempted",
                "trip_chain_patterns",
                "competing_explanation",
            },
            label="POI-enriched clusters",
        )
        _require_columns(
            repeated_chains,
            {
                "median_intermediate_stop_minutes",
                "typical_stop_durations_json",
            },
            label="repeated trip chains",
        )
        _require_columns(
            recurring,
            {
                "visit_frequency_basis",
                "last_arrival_date",
                "last_activity_association_date",
                "alternative_interpretation",
            },
            label="recurring destination patterns",
        )
        return BuildResult(
            paths=paths,
            source_trip_count=len(trips),
            county_fragment_count=len(fragments),
            selected_cluster_radius_m=clustering.radius_m,
            google_requests=total_google_requests,
            cache_hits=stats.cache_hits if stats else 0,
            sources_used=tuple(sorted(sources_used)),
            likely_home_cluster_id=str(home.cluster_id),
            privacy_checks_passed=True,
        )
    finally:
        if google_client is not None:
            google_client.close()
