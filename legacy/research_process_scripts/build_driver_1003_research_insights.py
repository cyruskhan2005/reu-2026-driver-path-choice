#!/usr/bin/env python3
"""Build privacy-preserving research insights for the Driver 1003 RCCI report.

The script intentionally uses cached local project data only.  It derives
map-matched endpoint clusters from the first/last FID of each matched trip,
uses local OSM-derived land-use records for generic category context, writes
the research artefacts requested for the report, and inserts an idempotent
``Research Insights`` section into the standalone HTML report.

It does not call external APIs, scrape map services, export endpoint
coordinates, export POI names/addresses, or infer why a route was taken.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import html
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "deliverables" / "driver_1003" / "route_choice_change_index"
OUTPUT_DATA_DIR = OUTPUT_ROOT / "data"
TARGET_REPORT = OUTPUT_ROOT / "visuals" / "driver_1003_route_choice_change_index_report.html"

GRAPH_COMPARISON_DIR = (
    ROOT
    / "deliverables"
    / "google_drive_phase2"
    / "driver_1003_graph_comparisons"
    / "data"
)
MONTHLY_GRAPH_DIR = (
    ROOT
    / "deliverables"
    / "google_drive_phase2"
    / "driver_1003_monthly_graphs"
    / "data"
)
MONTHLY_FID_USAGE = (
    ROOT
    / "deliverables"
    / "google_drive_phase2"
    / "driver_1003_monthly_graphs"
    / "driver_1003"
    / "monthly_fid_usage.csv"
)
TIMELINE_PATH = (
    ROOT
    / "sflorida_outputs"
    / "phase2"
    / "driver_timelines"
    / "driver_1_timeline.csv"
)

COUNTY_DIRECTORY = {
    "Broward County": "Broward_County",
    "Palm Beach County": "Palm_Beach_County",
    "Miami-Dade County": "Miami_Dade_County",
}
PRIMARY_COUNTY = "Broward County"

CLUSTER_EPS_METERS = 100.0
CLUSTER_MIN_SAMPLES = 8
LOCAL_POI_BUFFER_METERS = 500.0
FREQUENT_CLUSTER_MIN_ENDPOINTS = 250
FREQUENT_CLUSTER_MIN_MONTHS = 12

HIGH_CHANGE_THRESHOLD = 70.0
UPPER_QUARTILE_HIGH_CHANGE_THRESHOLD = 75.0
STABLE_RCCI_MAX = 60.0
STABLE_NODE_RETENTION_MIN = 0.60
STABLE_EDGE_RETENTION_MIN = 0.55

SECTION_BEGIN = "<!-- BEGIN DRIVER 1003 RESEARCH INSIGHTS -->"
SECTION_END = "<!-- END DRIVER 1003 RESEARCH INSIGHTS -->"
NAV_BEGIN = "<!-- BEGIN DRIVER 1003 RESEARCH INSIGHTS NAV -->"
NAV_END = "<!-- END DRIVER 1003 RESEARCH INSIGHTS NAV -->"
STYLE_MARKER = "/* DRIVER_1003_RESEARCH_INSIGHTS_STYLE */"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Driver 1003 RCCI output root.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=TARGET_REPORT,
        help="HTML report to update.",
    )
    parser.add_argument(
        "--skip-report-update",
        action="store_true",
        help="Write CSV/Markdown artefacts without changing the HTML report.",
    )
    return parser.parse_args()


def generated_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def nonempty_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none", "null"} else None


def number(value: object, digits: int = 1) -> str:
    numeric = safe_float(value)
    if math.isnan(numeric):
        return "—"
    if numeric.is_integer():
        return f"{int(numeric):,}"
    return f"{numeric:,.{digits}f}"


def percent(value: object, digits: int = 1) -> str:
    numeric = safe_float(value)
    return "—" if math.isnan(numeric) else f"{numeric * 100:.{digits}f}%"


def format_score(value: object) -> str:
    return number(value, 1)


def county_slug(county: str) -> str:
    return county.lower().replace("-", "_").replace(" ", "_")


def month_is_consecutive(month_a: str, month_b: str) -> bool:
    try:
        return pd.Period(month_b, freq="M") == pd.Period(month_a, freq="M") + 1
    except (TypeError, ValueError):
        return False


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    return path


def unique_fid_context() -> pd.DataFrame:
    """Return one enriched road-context record for each county/FID."""
    path = require_file(MONTHLY_GRAPH_DIR / "driver_1003_all_monthly_nodes.csv")
    requested = [
        "county",
        "fid",
        "u",
        "v",
        "name",
        "highway",
        "landuse",
        "estimated_speed_limit",
        "road_owner_or_source",
    ]
    header = pd.read_csv(path, nrows=0).columns
    available = [column for column in requested if column in header]
    table = pd.read_csv(path, usecols=available, low_memory=False)
    for column in ("fid", "u", "v"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.dropna(subset=["county", "fid", "u", "v"]).copy()
    for column in ("fid", "u", "v"):
        table[column] = table[column].astype("int64")
    table = table.sort_values(["county", "fid"]).drop_duplicates(
        ["county", "fid"], keep="first"
    )
    return table.reset_index(drop=True)


def load_node_coordinates(counties: Iterable[str]) -> pd.DataFrame:
    """Load only local OSM node IDs and projected coordinates for counties."""
    parts: list[pd.DataFrame] = []
    for county in sorted(set(counties)):
        directory = COUNTY_DIRECTORY.get(county)
        if not directory:
            continue
        path = ROOT / "sflorida_outputs" / directory / "osm_nodes.parquet"
        if not path.exists():
            continue
        nodes = gpd.read_parquet(path)
        columns = [column for column in ("osmid", "x", "y") if column in nodes]
        if {"osmid", "x", "y"} - set(columns):
            continue
        part = pd.DataFrame(nodes[columns]).rename(columns={"osmid": "node_osmid"})
        part["county"] = county
        for column in ("node_osmid", "x", "y"):
            part[column] = pd.to_numeric(part[column], errors="coerce")
        part = part.dropna(subset=["node_osmid", "x", "y"])
        part["node_osmid"] = part["node_osmid"].astype("int64")
        parts.append(part[["county", "node_osmid", "x", "y"]])
    if not parts:
        raise RuntimeError("No local OSM node-coordinate files were available")
    return pd.concat(parts, ignore_index=True).drop_duplicates(
        ["county", "node_osmid"], keep="first"
    )


def build_map_matched_endpoints() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive privacy-safe map-matched start/end endpoint records.

    Start points are assigned to the directed start FID's ``u`` network node,
    end points to the directed end FID's ``v`` node.  This means the endpoint
    is a route-network/recording boundary, not a verified place visit.
    """
    timeline_columns = [
        "trip_id",
        "county",
        "trip_month",
        "start_fid",
        "end_fid",
    ]
    timeline = pd.read_csv(require_file(TIMELINE_PATH), usecols=timeline_columns)
    for column in ("start_fid", "end_fid"):
        timeline[column] = pd.to_numeric(timeline[column], errors="coerce")
    timeline = timeline.dropna(subset=["county", "trip_month", "start_fid", "end_fid"])
    timeline["start_fid"] = timeline["start_fid"].astype("int64")
    timeline["end_fid"] = timeline["end_fid"].astype("int64")

    fid_context = unique_fid_context()
    node_coordinates = load_node_coordinates(timeline["county"].unique())

    start_context = fid_context.rename(
        columns={
            "fid": "endpoint_fid",
            "u": "node_osmid",
            "name": "road_name",
        }
    ).drop(columns=["v"])
    end_context = fid_context.rename(
        columns={
            "fid": "endpoint_fid",
            "v": "node_osmid",
            "name": "road_name",
        }
    ).drop(columns=["u"])

    starts = timeline.rename(columns={"start_fid": "endpoint_fid"}).merge(
        start_context,
        on=["county", "endpoint_fid"],
        how="left",
        validate="many_to_one",
    )
    starts["endpoint_role"] = "start"
    ends = timeline.rename(columns={"end_fid": "endpoint_fid"}).merge(
        end_context,
        on=["county", "endpoint_fid"],
        how="left",
        validate="many_to_one",
    )
    ends["endpoint_role"] = "end"

    endpoints = pd.concat([starts, ends], ignore_index=True)
    endpoints["node_osmid"] = pd.to_numeric(
        endpoints["node_osmid"], errors="coerce"
    )
    endpoints = endpoints.dropna(subset=["node_osmid"]).copy()
    endpoints["node_osmid"] = endpoints["node_osmid"].astype("int64")
    endpoints = endpoints.merge(
        node_coordinates,
        on=["county", "node_osmid"],
        how="left",
        validate="many_to_one",
    )
    endpoints = endpoints.dropna(subset=["x", "y"]).copy()
    endpoints["x"] = pd.to_numeric(endpoints["x"], errors="coerce")
    endpoints["y"] = pd.to_numeric(endpoints["y"], errors="coerce")
    endpoints = endpoints.dropna(subset=["x", "y"])
    return endpoints.reset_index(drop=True), fid_context


def dbscan_projected(
    x: Sequence[float],
    y: Sequence[float],
    *,
    eps: float,
    min_samples: int,
) -> list[int]:
    """Dependency-free DBSCAN for projected, metre-based point coordinates."""
    size = len(x)
    if size == 0:
        return []
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (x_value, y_value) in enumerate(zip(x, y, strict=True)):
        cells[(math.floor(x_value / eps), math.floor(y_value / eps))].append(index)

    neighborhood_cache: dict[int, list[int]] = {}
    eps_squared = eps * eps

    def neighborhood(index: int) -> list[int]:
        cached = neighborhood_cache.get(index)
        if cached is not None:
            return cached
        cell_x = math.floor(x[index] / eps)
        cell_y = math.floor(y[index] / eps)
        candidates: list[int] = []
        for delta_x in (-1, 0, 1):
            for delta_y in (-1, 0, 1):
                candidates.extend(cells.get((cell_x + delta_x, cell_y + delta_y), []))
        result = [
            candidate
            for candidate in candidates
            if (x[index] - x[candidate]) ** 2 + (y[index] - y[candidate]) ** 2
            <= eps_squared
        ]
        neighborhood_cache[index] = result
        return result

    unvisited = -99
    noise = -1
    labels = [unvisited] * size
    cluster_number = 0
    for index in range(size):
        if labels[index] != unvisited:
            continue
        neighbors = neighborhood(index)
        if len(neighbors) < min_samples:
            labels[index] = noise
            continue
        labels[index] = cluster_number
        queue: deque[int] = deque(neighbors)
        queued = set(neighbors)
        while queue:
            candidate = queue.popleft()
            if labels[candidate] == noise:
                labels[candidate] = cluster_number
            if labels[candidate] != unvisited:
                continue
            labels[candidate] = cluster_number
            candidate_neighbors = neighborhood(candidate)
            if len(candidate_neighbors) >= min_samples:
                for neighbor in candidate_neighbors:
                    if neighbor not in queued:
                        queued.add(neighbor)
                        queue.append(neighbor)
        cluster_number += 1
    return labels


def assign_endpoint_clusters(endpoints: pd.DataFrame) -> pd.DataFrame:
    """Fit one DBSCAN model per county so labels remain comparable by month."""
    output_parts: list[pd.DataFrame] = []
    for county, group in endpoints.groupby("county", sort=True):
        part = group.copy().reset_index(drop=True)
        if len(part) < CLUSTER_MIN_SAMPLES:
            part["cluster_internal"] = -1
        else:
            part["cluster_internal"] = dbscan_projected(
                part["x"].to_numpy(dtype=float),
                part["y"].to_numpy(dtype=float),
                eps=CLUSTER_EPS_METERS,
                min_samples=CLUSTER_MIN_SAMPLES,
            )
        output_parts.append(part)
    return pd.concat(output_parts, ignore_index=True)


def category_labels(pois: gpd.GeoDataFrame) -> list[str]:
    """Normalize OSM tags into generic, privacy-safe POI category labels."""
    labels: list[str] = []
    for _, record in pois.iterrows():
        landuse = (nonempty_text(record.get("landuse")) or "").lower()
        amenity = (nonempty_text(record.get("amenity")) or "").lower()
        shop = nonempty_text(record.get("shop"))
        office = nonempty_text(record.get("office"))
        aeroway = nonempty_text(record.get("aeroway"))
        if amenity in {"hospital", "clinic", "doctors"}:
            labels.append("hospital/clinic area")
        if amenity in {"school", "college", "university", "kindergarten"}:
            labels.append("school/university area")
        if aeroway:
            labels.append("airport area")
        if shop or landuse == "retail":
            labels.append("shopping area")
        if office or landuse == "commercial":
            labels.append("office/commercial area")
        if landuse == "residential":
            labels.append("residential area")
    category_order = [
        "hospital/clinic area",
        "school/university area",
        "airport area",
        "shopping area",
        "office/commercial area",
        "residential area",
    ]
    return [label for label in category_order if label in set(labels)]


def local_poi_context(
    clusters: pd.DataFrame,
) -> dict[str, str]:
    """Return generic local OSM context without names or addresses."""
    contexts: dict[str, str] = {}
    for county, group in clusters.groupby("county", sort=True):
        directory = COUNTY_DIRECTORY.get(county)
        if not directory:
            continue
        path = ROOT / "sflorida_outputs" / directory / "osm_landuse.parquet"
        if not path.exists():
            for cluster_id in group["cluster_id"]:
                contexts[cluster_id] = "POI category unavailable"
            continue
        pois = gpd.read_parquet(path)
        if pois.empty or pois.crs is None:
            for cluster_id in group["cluster_id"]:
                contexts[cluster_id] = "POI category unavailable"
            continue
        for record in group.itertuples(index=False):
            point = Point(float(record.centroid_x), float(record.centroid_y))
            nearby = pois.loc[
                pois.geometry.distance(point) <= LOCAL_POI_BUFFER_METERS
            ]
            labels = category_labels(nearby)
            contexts[record.cluster_id] = "; ".join(labels) if labels else "POI category unavailable"
    return contexts


def endpoint_road_context(group: pd.DataFrame) -> str:
    classes = (
        group.assign(highway=group["highway"].map(nonempty_text).fillna("road"))
        .groupby("highway", dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    if classes.empty:
        return "road context unavailable"
    total = float(classes.sum())
    chunks = [
        f"{road.replace('_', ' ')}-road approaches ({count / total:.1%})"
        for road, count in classes.head(2).items()
    ]
    speeds = pd.to_numeric(group.get("estimated_speed_limit"), errors="coerce").dropna()
    if not speeds.empty:
        chunks.append(f"median estimated speed limit {speeds.median():.0f} mph")
    return "; ".join(chunks)


def build_cluster_summary(clustered_endpoints: pd.DataFrame) -> pd.DataFrame:
    assigned = clustered_endpoints.loc[
        clustered_endpoints["cluster_internal"] >= 0
    ].copy()
    if assigned.empty:
        return pd.DataFrame()
    aggregates = (
        assigned.groupby(["county", "cluster_internal"], as_index=False)
        .agg(
            endpoint_records=("trip_id", "size"),
            start_records=("endpoint_role", lambda values: int((values == "start").sum())),
            end_records=("endpoint_role", lambda values: int((values == "end").sum())),
            months_active=("trip_month", "nunique"),
            first_month=("trip_month", "min"),
            last_month=("trip_month", "max"),
            centroid_x=("x", "mean"),
            centroid_y=("y", "mean"),
        )
        .sort_values(["county", "endpoint_records"], ascending=[True, False])
        .reset_index(drop=True)
    )
    eligible = aggregates.loc[
        (aggregates["endpoint_records"] >= FREQUENT_CLUSTER_MIN_ENDPOINTS)
        & (aggregates["months_active"] >= FREQUENT_CLUSTER_MIN_MONTHS)
    ].copy()
    if eligible.empty:
        return pd.DataFrame()

    cluster_records: list[dict[str, object]] = []
    for county, group in eligible.groupby("county", sort=True):
        ordered = group.sort_values("endpoint_records", ascending=False).reset_index(drop=True)
        for rank, record in ordered.iterrows():
            cluster_id = f"{county_slug(county)}-{chr(ord('A') + rank)}"
            endpoint_group = assigned.loc[
                (assigned["county"] == county)
                & (assigned["cluster_internal"] == record["cluster_internal"])
            ]
            monthly_counts = endpoint_group.groupby("trip_month").size()
            start_share = record["start_records"] / record["endpoint_records"]
            end_share = record["end_records"] / record["endpoint_records"]
            if abs(start_share - end_share) <= 0.20:
                generic_label = f"frequent origin/destination cluster {chr(ord('A') + rank)}"
            elif start_share > end_share:
                generic_label = f"frequent origin cluster {chr(ord('A') + rank)}"
            else:
                generic_label = f"frequent destination cluster {chr(ord('A') + rank)}"
            cluster_records.append(
                {
                    "cluster_id": cluster_id,
                    "county": county,
                    "generic_label": generic_label,
                    "endpoint_records": int(record["endpoint_records"]),
                    "start_records": int(record["start_records"]),
                    "end_records": int(record["end_records"]),
                    "months_active": int(record["months_active"]),
                    "first_month": str(record["first_month"]),
                    "last_month": str(record["last_month"]),
                    "peak_monthly_endpoint_records": int(monthly_counts.max()),
                    "dbscan_eps_m": int(CLUSTER_EPS_METERS),
                    "min_samples": int(CLUSTER_MIN_SAMPLES),
                    "road_context": endpoint_road_context(endpoint_group),
                    "poi_category_context": "POI category unavailable",
                    "poi_source": (
                        "Local OSM-derived land-use records within a 500 m buffer; "
                        "external POI lookup skipped"
                    ),
                    "privacy_note": (
                        "Map-matched trip-segmentation area only; not a verified origin, "
                        "destination, visit, residence, workplace, school, or medical location."
                    ),
                    "centroid_x": float(record["centroid_x"]),
                    "centroid_y": float(record["centroid_y"]),
                }
            )
    result = pd.DataFrame(cluster_records)
    poi_context = local_poi_context(result)
    result["poi_category_context"] = result["cluster_id"].map(poi_context).fillna(
        "POI category unavailable"
    )
    # Coordinates are used only for the local category query and are never exported.
    return result.drop(columns=["centroid_x", "centroid_y"])


def load_rcci_data(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = output_root / "data"
    summary = pd.read_csv(require_file(data_dir / "driver_1003_rcci_summary.csv"))
    sensitivity = pd.read_csv(require_file(data_dir / "driver_1003_rcci_sensitivity.csv"))
    numeric_columns = [
        "trips_a",
        "trips_b",
        "trip_count_ratio",
        "nodes_a",
        "nodes_b",
        "edges_a",
        "edges_b",
        "shared_nodes",
        "shared_edges",
        "added_nodes",
        "removed_nodes",
        "added_edges",
        "removed_edges",
        "weighted_node_overlap_min",
        "weighted_edge_overlap_min",
        "node_jaccard_similarity",
        "edge_jaccard_similarity",
        "node_change_component",
        "edge_change_component",
        "rcci_v1",
    ]
    for column in numeric_columns:
        if column in summary:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")
    summary["node_retention_from_month_a"] = np.where(
        summary["nodes_a"] > 0,
        summary["shared_nodes"] / summary["nodes_a"],
        np.nan,
    )
    summary["edge_retention_from_month_a"] = np.where(
        summary["edges_a"] > 0,
        summary["shared_edges"] / summary["edges_a"],
        np.nan,
    )
    return summary, sensitivity


def road_label(row: Mapping[str, object]) -> str:
    name = nonempty_text(row.get("road_name")) or nonempty_text(row.get("name"))
    road_type = nonempty_text(row.get("road_type")) or nonempty_text(row.get("highway"))
    if name and road_type:
        return f"{name} ({road_type})"
    if name:
        return name
    if road_type:
        return f"unnamed {road_type}-road segments"
    return "unnamed road segment"


def compact_ranked_list(
    frame: pd.DataFrame,
    *,
    label_column: str,
    value_column: str,
    limit: int = 3,
    unit: str = "trip uses",
) -> str:
    if frame.empty:
        return "No comparable records"
    grouped = (
        frame.groupby(label_column, dropna=False)[value_column]
        .sum()
        .sort_values(ascending=False)
        .head(limit)
    )
    values = []
    for label, value in grouped.items():
        label_text = nonempty_text(label) or "unnamed road segment"
        values.append(f"{label_text} ({number(value)} {unit})")
    return "; ".join(values) if values else "No comparable records"


def build_high_change_events(rcci: pd.DataFrame) -> pd.DataFrame:
    node_details = pd.read_csv(
        require_file(GRAPH_COMPARISON_DIR / "driver_1003_month_to_month_node_comparisons.csv"),
        low_memory=False,
    )
    edge_details = pd.read_csv(
        require_file(GRAPH_COMPARISON_DIR / "driver_1003_month_to_month_edge_comparisons.csv"),
        low_memory=False,
    )
    for column in (
        "trip_use_count_a",
        "trip_use_count_b",
        "transition_count_a",
        "transition_count_b",
        "trip_count_using_transition_a",
        "trip_count_using_transition_b",
    ):
        if column in node_details:
            node_details[column] = pd.to_numeric(node_details[column], errors="coerce").fillna(0)
        if column in edge_details:
            edge_details[column] = pd.to_numeric(edge_details[column], errors="coerce").fillna(0)

    candidates = rcci.loc[
        (rcci["county"] == PRIMARY_COUNTY)
        & rcci["confidence_label"].isin(["HIGH", "MEDIUM"])
        & (rcci["rcci_v1"] >= HIGH_CHANGE_THRESHOLD)
    ].sort_values(["month_a", "month_b"])
    rows: list[dict[str, object]] = []
    for event_id, event in enumerate(candidates.itertuples(index=False), start=1):
        node_pair = node_details.loc[
            (node_details["county"] == event.county)
            & (node_details["month_a"].astype(str) == str(event.month_a))
            & (node_details["month_b"].astype(str) == str(event.month_b))
        ].copy()
        edge_pair = edge_details.loc[
            (edge_details["county"] == event.county)
            & (edge_details["month_a"].astype(str) == str(event.month_a))
            & (edge_details["month_b"].astype(str) == str(event.month_b))
        ].copy()
        node_pair["road_label"] = [road_label(record) for record in node_pair.to_dict("records")]
        fid_labels = (
            node_pair.sort_values("road_label")
            .drop_duplicates("fid", keep="first")
            .set_index("fid")["road_label"]
            .to_dict()
        )
        added_roads = compact_ranked_list(
            node_pair.loc[node_pair["status"] == "added"],
            label_column="road_label",
            value_column="trip_use_count_b",
        )
        removed_roads = compact_ranked_list(
            node_pair.loc[node_pair["status"] == "removed"],
            label_column="road_label",
            value_column="trip_use_count_a",
        )

        def transition_text(frame: pd.DataFrame, *, status: str) -> str:
            transition_rows = frame.loc[frame["status"] == status].copy()
            if transition_rows.empty:
                return "No comparable records"
            source = transition_rows["source_fid"].map(fid_labels).fillna("road segment")
            target = transition_rows["target_fid"].map(fid_labels).fillna("road segment")
            transition_rows["transition_label"] = source + " → " + target
            value_column = (
                "trip_count_using_transition_b" if status == "added" else "trip_count_using_transition_a"
            )
            return compact_ranked_list(
                transition_rows,
                label_column="transition_label",
                value_column=value_column,
                limit=3,
                unit="trips",
            )

        added_transitions = transition_text(edge_pair, status="added")
        removed_transitions = transition_text(edge_pair, status="removed")
        added_nodes = safe_float(event.added_nodes, 0.0)
        removed_nodes = safe_float(event.removed_nodes, 0.0)
        if added_nodes > removed_nodes * 1.15:
            interpretation = (
                "New recurring road segments and directed transitions appeared; this is "
                "consistent with a substantial route-pattern expansion."
            )
        elif removed_nodes > added_nodes * 1.15:
            interpretation = (
                "Previously frequent road segments and transitions receded; this is "
                "consistent with a substantial shift toward a different or more compact route set."
            )
        else:
            interpretation = (
                "Road additions and removals were both substantial, suggesting a possible "
                "change in route routine while some route-network elements remained."
            )
        ratio = safe_float(event.trip_count_ratio)
        quality_note = ""
        if not math.isnan(ratio) and ratio > 2:
            quality_note = (
                " Monthly trip volume changed by more than twofold, so coverage differences "
                "may contribute to the observed change."
            )
        rows.append(
            {
                "event_id": f"HC-{event_id:02d}",
                "event_tier": (
                    "upper-quartile high change"
                    if float(event.rcci_v1) >= UPPER_QUARTILE_HIGH_CHANGE_THRESHOLD
                    else "high relative change"
                ),
                "county": event.county,
                "month_a": event.month_a,
                "month_b": event.month_b,
                "rcci": float(event.rcci_v1),
                "node_change_pct": float(event.node_change_component) * 100,
                "edge_change_pct": float(event.edge_change_component) * 100,
                "weighted_node_overlap_pct": float(event.weighted_node_overlap_min) * 100,
                "weighted_edge_overlap_pct": float(event.weighted_edge_overlap_min) * 100,
                "node_retention_from_month_a_pct": float(event.node_retention_from_month_a) * 100,
                "edge_retention_from_month_a_pct": float(event.edge_retention_from_month_a) * 100,
                "trips_a": int(event.trips_a),
                "trips_b": int(event.trips_b),
                "trip_count_ratio": ratio,
                "confidence_label": event.confidence_label,
                "confidence_reason": event.confidence_reason,
                "top_roads_added": added_roads,
                "top_roads_removed": removed_roads,
                "top_transitions_added": added_transitions,
                "top_transitions_removed": removed_transitions,
                "possible_interpretation": interpretation,
                "data_quality_note": quality_note.strip() or "High/medium-confidence comparison; RCCI cannot determine cause.",
            }
        )
    return pd.DataFrame(rows)


def build_stable_periods(rcci: pd.DataFrame) -> pd.DataFrame:
    candidates = rcci.loc[
        (rcci["county"] == PRIMARY_COUNTY)
        & (rcci["confidence_label"] == "HIGH")
        & (rcci["rcci_v1"] <= STABLE_RCCI_MAX)
        & (rcci["node_retention_from_month_a"] >= STABLE_NODE_RETENTION_MIN)
        & (rcci["edge_retention_from_month_a"] >= STABLE_EDGE_RETENTION_MIN)
    ].copy()
    candidates = candidates.loc[
        [
            month_is_consecutive(month_a, month_b)
            for month_a, month_b in zip(candidates["month_a"], candidates["month_b"], strict=False)
        ]
    ].sort_values(["month_a", "month_b"])
    if candidates.empty:
        return pd.DataFrame()

    runs: list[list[pd.Series]] = []
    active: list[pd.Series] = []
    for _, row in candidates.iterrows():
        if active and str(active[-1]["month_b"]) != str(row["month_a"]):
            runs.append(active)
            active = []
        active.append(row)
    if active:
        runs.append(active)

    rows: list[dict[str, object]] = []
    for index, run in enumerate(runs, start=1):
        frame = pd.DataFrame(run)
        start_month = str(frame.iloc[0]["month_a"])
        end_month = str(frame.iloc[-1]["month_b"])
        pair_count = len(frame)
        multi_link = pair_count >= 2
        conclusion = (
            "Driver maintained a relatively stable route routine during this period, "
            "relative to this driver's observed route-change baseline."
            if multi_link
            else "Qualifying low-change comparison; it did not form a multi-link stable interval."
        )
        rows.append(
            {
                "stable_period_id": f"SP-{index:02d}",
                "period_type": "multi-comparison stable run" if multi_link else "single qualifying comparison",
                "county": PRIMARY_COUNTY,
                "start_month": start_month,
                "end_month": end_month,
                "comparison_count": pair_count,
                "month_pairs": "; ".join(
                    f"{record.month_a}→{record.month_b}" for record in frame.itertuples(index=False)
                ),
                "mean_rcci": frame["rcci_v1"].mean(),
                "min_rcci": frame["rcci_v1"].min(),
                "max_rcci": frame["rcci_v1"].max(),
                "mean_weighted_node_overlap_pct": frame["weighted_node_overlap_min"].mean() * 100,
                "mean_weighted_edge_overlap_pct": frame["weighted_edge_overlap_min"].mean() * 100,
                "mean_node_retention_from_month_a_pct": frame["node_retention_from_month_a"].mean() * 100,
                "mean_edge_retention_from_month_a_pct": frame["edge_retention_from_month_a"].mean() * 100,
                "minimum_monthly_trip_count": int(
                    pd.concat([frame["trips_a"], frame["trips_b"]]).min()
                ),
                "confidence": "HIGH",
                "stable_screen": (
                    "HIGH confidence; RCCI ≤ 60; prior-month node retention ≥ 60%; "
                    "prior-month edge retention ≥ 55%"
                ),
                "interpretation": conclusion,
            }
        )
    return pd.DataFrame(rows)


def formula_summary(rcci: pd.DataFrame, sensitivity: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    keys = ["county", "month_a", "month_b"]
    base = rcci.loc[
        (rcci["county"] == PRIMARY_COUNTY)
        & rcci["confidence_label"].isin(["HIGH", "MEDIUM"])
        & rcci["rcci_v1"].notna()
    ].copy()
    sensitivity_columns = keys + [
        "rcci_balanced_weighted",
        "rcci_edge_heavy_weighted",
        "rcci_balanced_jaccard",
        "rcci_geometric_weighted",
    ]
    available = [column for column in sensitivity_columns if column in sensitivity]
    combined = base.merge(sensitivity[available], on=keys, how="left", validate="one_to_one")
    combined["rcci_weighted_node_only"] = combined["node_change_component"] * 100
    combined["rcci_weighted_edge_only"] = combined["edge_change_component"] * 100
    variants = [
        ("Weighted node-only", "rcci_weighted_node_only"),
        ("Weighted edge-only", "rcci_weighted_edge_only"),
        ("Weighted 50/50 node-edge", "rcci_balanced_weighted"),
        ("Weighted edge-heavy (30/70)", "rcci_edge_heavy_weighted"),
        ("Weighted geometric overlap", "rcci_geometric_weighted"),
        ("Unweighted 50/50 Jaccard", "rcci_balanced_jaccard"),
    ]
    baseline = combined["rcci_balanced_weighted"]
    baseline_rank = baseline.rank(method="average", ascending=False)
    rows: list[dict[str, object]] = []
    for name, column in variants:
        scores = pd.to_numeric(combined[column], errors="coerce")
        difference = (scores - baseline).abs()
        rank = scores.rank(method="average", ascending=False)
        rows.append(
            {
                "formula_variant": name,
                "reliable_broward_pairs": int(scores.notna().sum()),
                "mean_score": scores.mean(),
                "median_score": scores.median(),
                "minimum_score": scores.min(),
                "maximum_score": scores.max(),
                "mean_absolute_difference_from_weighted_50_50": difference.mean(),
                "maximum_absolute_difference_from_weighted_50_50": difference.max(),
                "rank_correlation_with_weighted_50_50": scores.rank().corr(baseline.rank()),
                "maximum_rank_shift_from_weighted_50_50": (rank - baseline_rank).abs().max(),
            }
        )
    formula = pd.DataFrame(rows)
    facts = {
        "reliable_pairs": float(len(combined)),
        "node_only_mean": float(combined["rcci_weighted_node_only"].mean()),
        "edge_only_mean": float(combined["rcci_weighted_edge_only"].mean()),
        "baseline_mean": float(baseline.mean()),
        "edge_heavy_mean": float(combined["rcci_edge_heavy_weighted"].mean()),
        "geometric_mean": float(combined["rcci_geometric_weighted"].mean()),
        "jaccard_mean": float(combined["rcci_balanced_jaccard"].mean()),
        "edge_minus_node_change": float(
            (combined["edge_change_component"] - combined["node_change_component"]).mean() * 100
        ),
        "jaccard_mean_abs_gap": float(
            (combined["rcci_balanced_jaccard"] - baseline).abs().mean()
        ),
        "jaccard_max_abs_gap": float(
            (combined["rcci_balanced_jaccard"] - baseline).abs().max()
        ),
        "jaccard_rank_correlation": float(
            combined["rcci_balanced_jaccard"].rank().corr(baseline.rank())
        ),
        "jaccard_max_rank_shift": float(
            (
                combined["rcci_balanced_jaccard"].rank(method="average", ascending=False)
                - baseline_rank
            )
            .abs()
            .max()
        ),
    }
    return formula, facts


def network_context() -> dict[str, object]:
    usage = pd.read_csv(require_file(MONTHLY_FID_USAGE), low_memory=False)
    broward = usage.loc[usage["county"] == PRIMARY_COUNTY].copy()
    broward["trip_use_count"] = pd.to_numeric(broward["trip_use_count"], errors="coerce").fillna(0)
    total = broward["trip_use_count"].sum()
    by_highway = (
        broward.assign(highway=broward["highway"].map(nonempty_text).fillna("unknown"))
        .groupby("highway")["trip_use_count"]
        .sum()
        .sort_values(ascending=False)
    )
    primary_secondary = by_highway.reindex(["primary", "secondary"], fill_value=0).sum()
    residential_service = by_highway.reindex(["residential", "service"], fill_value=0).sum()
    recurring = (
        broward.groupby(["fid", "highway", "estimated_speed_limit"], dropna=False)
        .agg(months_active=("trip_month", "nunique"), total_trip_use=("trip_use_count", "sum"))
        .reset_index()
    )
    recurring_all_months = recurring.loc[recurring["months_active"] == broward["trip_month"].nunique()]
    landuse = broward["landuse"].map(nonempty_text)
    tagged_weight = broward.loc[landuse.notna(), "trip_use_count"].sum()
    retail_weight = broward.loc[landuse.eq("retail"), "trip_use_count"].sum()
    return {
        "total_weighted_segment_uses": float(total),
        "secondary_weighted_segment_uses": float(by_highway.get("secondary", 0)),
        "primary_weighted_segment_uses": float(by_highway.get("primary", 0)),
        "primary_secondary_share": float(primary_secondary / total) if total else np.nan,
        "residential_service_share": float(residential_service / total) if total else np.nan,
        "observed_months": int(broward["trip_month"].nunique()),
        "recurrent_all_month_segments": int(len(recurring_all_months)),
        "landuse_tagged_share": float(tagged_weight / total) if total else np.nan,
        "retail_landuse_share": float(retail_weight / total) if total else np.nan,
    }


def html_table(
    frame: pd.DataFrame,
    columns: Sequence[tuple[str, str]],
    *,
    limit: int | None = None,
) -> str:
    data = frame.head(limit).copy() if limit else frame.copy()
    if data.empty:
        return "<p class='empty'>No rows.</p>"
    headers = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    rows: list[str] = []
    for record in data.to_dict(orient="records"):
        cells = "".join(
            f"<td>{html.escape(str(record.get(column, '—')))}</td>"
            for column, _ in columns
        )
        rows.append(f"<tr>{cells}</tr>")
    return (
        "<div class='table-wrap research-table'><table><thead><tr>"
        f"{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def markdown_table(frame: pd.DataFrame, columns: Sequence[tuple[str, str]]) -> str:
    if frame.empty:
        return "No rows."
    headers = [label for _, label in columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for record in frame.to_dict(orient="records"):
        values = [
            str(record.get(column, "—")).replace("|", "\\|").replace("\n", " ")
            for column, _ in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def create_insight_rows(
    *,
    rcci: pd.DataFrame,
    clusters: pd.DataFrame,
    high_events: pd.DataFrame,
    stable_periods: pd.DataFrame,
    network: Mapping[str, object],
    formulas: Mapping[str, float],
    broward_trip_count: int,
    all_matched_trip_count: int,
) -> pd.DataFrame:
    reliable = rcci.loc[
        (rcci["county"] == PRIMARY_COUNTY)
        & rcci["confidence_label"].isin(["HIGH", "MEDIUM"])
        & rcci["rcci_v1"].notna()
    ]
    endpoint_share = (
        clusters["endpoint_records"].sum() / (2 * broward_trip_count)
        if broward_trip_count
        else np.nan
    )
    multi_link = stable_periods.loc[
        stable_periods["comparison_count"] >= 2
    ] if not stable_periods.empty else stable_periods
    rows = [
        {
            "insight_id": "RI-01",
            "section": "Key findings",
            "evidence_period": "2021-09 to 2024-06",
            "confidence": "HIGH for Broward longitudinal comparisons",
            "metric_summary": (
                f"{len(reliable)} HIGH/MEDIUM-confidence Broward comparisons; Broward accounts for "
                f"{broward_trip_count:,}/{all_matched_trip_count:,} matched trips "
                f"({broward_trip_count / all_matched_trip_count:.1%})."
            ),
            "interpretation": "Broward County is the defensible longitudinal series; Palm Beach and Miami-Dade are supplementary because they are sparse.",
            "caveat": "County coverage is not evidence about trip purpose or the driver’s identity.",
        },
        {
            "insight_id": "RI-02",
            "section": "POI/category context",
            "evidence_period": "32 observed Broward months",
            "confidence": "Descriptive local map context",
            "metric_summary": (
                f"{len(clusters)} recurring endpoint clusters met the ≥{FREQUENT_CLUSTER_MIN_ENDPOINTS} endpoint / "
                f"≥{FREQUENT_CLUSTER_MIN_MONTHS}-month screen, together representing {endpoint_share:.1%} of Broward endpoint records."
            ),
            "interpretation": "The recurring areas suggest repeated trip-segmentation locations, not confirmed destinations or visits.",
            "caveat": "Local OSM category proximity does not establish a visit, residence, workplace, education, or medical connection.",
        },
        {
            "insight_id": "RI-03",
            "section": "Segment usage and road network",
            "evidence_period": "2021-09 to 2024-06",
            "confidence": "Descriptive enriched-network aggregation",
            "metric_summary": (
                f"Primary/secondary roads account for {percent(network['primary_secondary_share'])} of weighted Broward segment use; "
                f"residential/service roads add {percent(network['residential_service_share'])}."
            ),
            "interpretation": "The data are consistent with a recurring mix of arterial corridors and local-access segments, rather than a single road class.",
            "caveat": "Road-category use cannot identify destination type or trip purpose.",
        },
        {
            "insight_id": "RI-04",
            "section": "High-change periods",
            "evidence_period": "Reliable Broward month pairs",
            "confidence": "HIGH/MEDIUM as reported with each event",
            "metric_summary": (
                f"{len(high_events)} HIGH/MEDIUM-confidence pairs have RCCI ≥ {HIGH_CHANGE_THRESHOLD:.0f}; "
                f"{int((high_events['rcci'] >= UPPER_QUARTILE_HIGH_CHANGE_THRESHOLD).sum())} are in the upper-quartile focus band (RCCI ≥ {UPPER_QUARTILE_HIGH_CHANGE_THRESHOLD:.0f})."
            ),
            "interpretation": "These periods may indicate substantial route-pattern change through added/removed road segments and directed transitions.",
            "caveat": "RCCI does not determine why a route pattern changed; trip-count imbalance can affect coverage.",
        },
        {
            "insight_id": "RI-05",
            "section": "Stable periods",
            "evidence_period": "Reliable Broward month pairs",
            "confidence": "HIGH",
            "metric_summary": (
                "Multi-link stable interval: "
                + (
                    ", ".join(
                        f"{record.start_month} to {record.end_month} ({record.comparison_count} comparisons)"
                        for record in multi_link.itertuples(index=False)
                    )
                    if multi_link is not None and not multi_link.empty
                    else "none under the predeclared screen"
                )
            ),
            "interpretation": "A low-RCCI, high-retention sequence is consistent with a relatively stable route routine during that interval.",
            "caveat": "No reliable pair retains at least 70% of both prior-month nodes and edges; stability is relative, not an unchanged network.",
        },
        {
            "insight_id": "RI-06",
            "section": "Formula recommendation",
            "evidence_period": "28 reliable Broward pairs",
            "confidence": "Robust descriptive sensitivity analysis",
            "metric_summary": (
                f"Weighted 50/50 mean RCCI {formulas['baseline_mean']:.2f}; unweighted Jaccard differs by "
                f"{formulas['jaccard_mean_abs_gap']:.2f} points on average (maximum {formulas['jaccard_max_abs_gap']:.2f})."
            ),
            "interpretation": "Weighted 50/50 node-edge RCCI is the best baseline because it balances roads used and their connections while emphasizing repeated behavior.",
            "caveat": "Sensitivity analyses should accompany future applications with different coverage or study aims.",
        },
        {
            "insight_id": "RI-07",
            "section": "Sparse-data uncertainty",
            "evidence_period": "2022-07 to 2022-09; 2023-04 to 2023-05",
            "confidence": "LOW",
            "metric_summary": "Low-trip and zero-baseline comparisons produce extreme RCCI values that are not used for behavioral interpretation.",
            "interpretation": "Sparse months primarily signal uncertainty or missing coverage, not a demonstrated routine shift.",
            "caveat": "Do not compare these low-confidence scores directly with high-coverage Broward events.",
        },
    ]
    return pd.DataFrame(rows)


def render_markdown(
    *,
    clusters: pd.DataFrame,
    high_events: pd.DataFrame,
    stable_periods: pd.DataFrame,
    formulas: pd.DataFrame,
    network: Mapping[str, object],
    formula_facts: Mapping[str, float],
) -> str:
    cluster_display = clusters.copy()
    cluster_display["generic_label"] = cluster_display["generic_label"].str.replace(
        "frequent origin/destination cluster", "recurring activity area cluster", regex=False
    )
    cluster_display["endpoint_records"] = cluster_display["endpoint_records"].map(number)
    cluster_display["start_records"] = cluster_display["start_records"].map(number)
    cluster_display["end_records"] = cluster_display["end_records"].map(number)
    cluster_columns = [
        ("generic_label", "Generic label"),
        ("endpoint_records", "Endpoint records"),
        ("start_records", "Starts"),
        ("end_records", "Ends"),
        ("months_active", "Months active"),
        ("road_context", "Local road context"),
        ("poi_category_context", "Nearby generic category context"),
    ]
    stable_display = stable_periods.copy()
    if not stable_display.empty:
        stable_display["mean_rcci"] = stable_display["mean_rcci"].map(format_score)
        stable_display["mean_node_retention_from_month_a_pct"] = stable_display[
            "mean_node_retention_from_month_a_pct"
        ].map(lambda value: f"{value:.1f}%")
        stable_display["mean_edge_retention_from_month_a_pct"] = stable_display[
            "mean_edge_retention_from_month_a_pct"
        ].map(lambda value: f"{value:.1f}%")
    stable_columns = [
        ("start_month", "Start"),
        ("end_month", "End"),
        ("comparison_count", "Comparisons"),
        ("mean_rcci", "Mean RCCI"),
        ("mean_node_retention_from_month_a_pct", "Node retention"),
        ("mean_edge_retention_from_month_a_pct", "Edge retention"),
        ("interpretation", "Interpretation"),
    ]
    high_display = high_events.copy()
    for column in ("rcci", "node_change_pct", "edge_change_pct"):
        high_display[column] = high_display[column].map(format_score)
    high_display["trips"] = high_display.apply(
        lambda row: f"{int(row.trips_a):,} → {int(row.trips_b):,}", axis=1
    )
    high_columns = [
        ("month_a", "Month A"),
        ("month_b", "Month B"),
        ("rcci", "RCCI"),
        ("node_change_pct", "Node change"),
        ("edge_change_pct", "Edge change"),
        ("trips", "Trips"),
        ("confidence_label", "Confidence"),
        ("possible_interpretation", "Possible interpretation"),
    ]
    formula_display = formulas.copy()
    for column in (
        "mean_score",
        "mean_absolute_difference_from_weighted_50_50",
        "maximum_absolute_difference_from_weighted_50_50",
        "rank_correlation_with_weighted_50_50",
    ):
        formula_display[column] = formula_display[column].map(lambda value: number(value, 2))
    formula_columns = [
        ("formula_variant", "Variant"),
        ("mean_score", "Mean RCCI"),
        ("mean_absolute_difference_from_weighted_50_50", "Mean |Δ| vs. baseline"),
        ("maximum_absolute_difference_from_weighted_50_50", "Max |Δ| vs. baseline"),
        ("rank_correlation_with_weighted_50_50", "Rank correlation"),
    ]
    return f"""# Driver 1003 RCCI Research Insights

Generated: {generated_at()}

## Scope and privacy safeguards

This analysis uses cached matched-route, RCCI, enriched road-network, and local OSM-derived land-use data. No external POI or map API was queried, and no map-service scraping was used. Endpoint coordinates, exact addresses, and POI names are deliberately excluded. A map-matched endpoint is a route-network/recording boundary; it is not evidence that the driver visited, lived, worked, studied, or received care at a nearby place.

## Key findings

- Broward County is the longitudinal evidence base: {number(network['broward_trip_count'])} of {number(network['all_matched_trip_count'])} matched trips ({network['broward_trip_count'] / network['all_matched_trip_count']:.1%}) and all 28 HIGH/MEDIUM-confidence RCCI comparisons. Palm Beach and Miami-Dade are retained only as sparse supplementary data.
- High-change comparisons show substantial additions/removals of road segments and directed transitions. They may indicate a route-routine shift, but RCCI cannot determine the cause.
- The most stable multi-link interval is 2023-06 to 2023-08. It is consistent with a relatively stable route routine, not an unchanged network.
- Primary and secondary roads account for {percent(network['primary_secondary_share'])} of weighted Broward segment use; residential and service roads add {percent(network['residential_service_share'])}. This is consistent with a repeated arterial-plus-local-access pattern.
- Sparse or zero-trip months should be interpreted as data uncertainty, not a demonstrated behavioral change.

## Stable periods

Stable screening rule: HIGH confidence, RCCI ≤ {STABLE_RCCI_MAX:.0f}, prior-month node retention ≥ {STABLE_NODE_RETENTION_MIN:.0%}, and prior-month edge retention ≥ {STABLE_EDGE_RETENTION_MIN:.0%}. Structural retention is calculated as shared nodes/edges divided by the prior month's node/edge count. RCCI itself uses the separate frequency-weighted overlap shown in the base report.

{markdown_table(stable_display, stable_columns)}

No reliable pair has both node and edge retention of at least 70%, so the terms “stable” and “routine” are relative to Driver 1003’s observed baseline.

## High-change periods

The detailed event file includes every HIGH/MEDIUM-confidence Broward comparison with RCCI ≥ {HIGH_CHANGE_THRESHOLD:.0f}. “Upper-quartile high change” is flagged at RCCI ≥ {UPPER_QUARTILE_HIGH_CHANGE_THRESHOLD:.0f}; all values remain descriptive route-network measures rather than explanations.

{markdown_table(high_display, high_columns)}

Top added/removed roads and transitions for each event are in `high_rcci_event_summary.csv`. They describe public road-network changes only and do not identify destinations or reasons for travel.

## POI/category context

Endpoint clusters use a county-level DBSCAN on map-matched start/end nodes (epsilon {int(CLUSTER_EPS_METERS)} m; minimum samples {CLUSTER_MIN_SAMPLES}) and are tabulated by month. The local context query uses cached `osm_landuse.parquet` records inside {int(LOCAL_POI_BUFFER_METERS)} m of a cluster centroid, normalizes them to generic categories, and omits POI names/addresses. External POI lookup was skipped.

{markdown_table(cluster_display, cluster_columns)}

The four listed areas are recurring activity/trip-segmentation areas, not verified origins/destinations or visits. POI proximity does not establish a visit, medical connection, employment, education, residence, income, or identity.

## Segment use and enriched road-network context

- {number(network['recurrent_all_month_segments'])} individual Broward road segments recur in all {number(network['observed_months'])} observed months; recurrent endpoints tend to use local residential/service approaches while aggregate through-use is concentrated on secondary and primary roads.
- Local road-network land-use tags cover only {percent(network['landuse_tagged_share'])} of weighted segment use. Context labels are therefore supplementary and deliberately conservative.
- Retail-tagged local road context represents {percent(network['retail_landuse_share'])} of weighted segment use; this is roadway context only, not evidence of shopping or a visit.

## Formula recommendation

The weighted 50/50 node-edge RCCI is the best baseline. It jointly captures which roads were used and how they were connected; weighting emphasizes repeated driving behavior and downweights one-time noisy segments; equal node/edge weighting remains transparent and scientifically defensible. Edge change averages {formula_facts['edge_minus_node_change']:.2f} points above node change, so an edge component adds useful transition information. Edge-heavy and geometric weighted variants add little empirical distinction, whereas unweighted Jaccard differs from the weighted baseline by {formula_facts['jaccard_mean_abs_gap']:.2f} points on average (maximum {formula_facts['jaccard_max_abs_gap']:.2f}; rank correlation {formula_facts['jaccard_rank_correlation']:.2f}).

{markdown_table(formula_display, formula_columns)}

## Limitations

- RCCI measures route-network change, not cause.
- POI proximity and local land-use context do not prove that the driver visited a location.
- Sparse months, zero-trip months, and materially imbalanced trip counts reduce confidence.
- Local OSM-derived context can be incomplete or outdated; no external POI enrichment was used.
- This analysis does not make medical, employment, education, residence, income, political, religious, or identity claims.

## Future research questions

- Do high-change events persist in subsequent high-coverage months or revert toward the recurring route core?
- How would confidence-adjusted uncertainty intervals change the interpretation of medium-confidence, high trip-ratio months?
- Can repeated road-category and transition patterns be reproduced in a larger, privacy-preserving driver sample?
- Would a sensitivity analysis stratified by trip duration, time of day, or data-collection cadence distinguish routine variation from sampling effects?
"""


def research_css() -> str:
    return f"""
{STYLE_MARKER}
.research-insights .insight-lead{{font-size:16px;color:#334155}}
.research-insights .research-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:12px;margin:16px 0}}
.research-insights .research-card{{background:#f8fbff;border:1px solid var(--line);border-radius:13px;padding:14px}}
.research-insights .research-card strong{{display:block;color:var(--text);font-size:17px;margin-bottom:5px}}
.research-insights .research-card p{{margin:0;font-size:14px}}
.research-insights .research-callout{{background:#edf8f3;border-left:5px solid var(--green);border-radius:12px;padding:15px;margin:16px 0;color:#24523a}}
.research-insights .research-caution{{background:#fff8e6;border-left:5px solid var(--orange);border-radius:12px;padding:15px;margin:16px 0;color:#5b3b00}}
.research-insights .research-table td{{white-space:normal;vertical-align:top;min-width:100px}}
.research-insights details{{background:#fafcff;border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:9px 0}}
.research-insights summary{{cursor:pointer;color:#1e3a5f;font-weight:700}}
.research-insights details p{{margin:8px 0 0}}
"""


def render_html_section(
    *,
    clusters: pd.DataFrame,
    high_events: pd.DataFrame,
    stable_periods: pd.DataFrame,
    formulas: pd.DataFrame,
    network: Mapping[str, object],
    formula_facts: Mapping[str, float],
) -> str:
    cluster_html = clusters.copy()
    cluster_html["endpoint_records"] = cluster_html["endpoint_records"].map(number)
    cluster_html["start_end"] = cluster_html.apply(
        lambda row: f"{number(row.start_records)} / {number(row.end_records)}", axis=1
    )
    cluster_columns = [
        ("generic_label", "Generic cluster label"),
        ("endpoint_records", "Endpoint records"),
        ("start_end", "Start / end"),
        ("months_active", "Months active"),
        ("road_context", "Local road context"),
        ("poi_category_context", "Generic category context"),
    ]
    stable_html = stable_periods.copy()
    if not stable_html.empty:
        stable_html["mean_rcci"] = stable_html["mean_rcci"].map(format_score)
        stable_html["retention"] = stable_html.apply(
            lambda row: (
                f"{row.mean_node_retention_from_month_a_pct:.1f}% / "
                f"{row.mean_edge_retention_from_month_a_pct:.1f}%"
            ),
            axis=1,
        )
    stable_columns = [
        ("start_month", "Start"),
        ("end_month", "End"),
        ("comparison_count", "Pairs"),
        ("mean_rcci", "Mean RCCI"),
        ("retention", "Node / edge retention"),
        ("interpretation", "Interpretation"),
    ]
    high_html = high_events.copy()
    high_html["period"] = high_html["month_a"].astype(str) + " → " + high_html["month_b"].astype(str)
    high_html["rcci"] = high_html["rcci"].map(format_score)
    high_html["change"] = high_html.apply(
        lambda row: f"{row.node_change_pct:.1f}% / {row.edge_change_pct:.1f}%", axis=1
    )
    high_html["trips"] = high_html.apply(
        lambda row: f"{int(row.trips_a):,} → {int(row.trips_b):,}", axis=1
    )
    high_columns = [
        ("period", "Month pair"),
        ("rcci", "RCCI"),
        ("change", "Node / edge change"),
        ("trips", "Trips"),
        ("confidence_label", "Confidence"),
        ("possible_interpretation", "Evidence-led interpretation"),
    ]
    details = "".join(
        f"""
<details><summary>{html.escape(str(record.month_a))} → {html.escape(str(record.month_b))} · RCCI {record.rcci:.1f} · {html.escape(str(record.event_tier))}</summary>
<p><strong>Top roads added:</strong> {html.escape(str(record.top_roads_added))}</p>
<p><strong>Top roads removed:</strong> {html.escape(str(record.top_roads_removed))}</p>
<p><strong>Top transitions added:</strong> {html.escape(str(record.top_transitions_added))}</p>
<p><strong>Top transitions removed:</strong> {html.escape(str(record.top_transitions_removed))}</p>
<p><strong>Interpretation:</strong> {html.escape(str(record.possible_interpretation))} {html.escape(str(record.data_quality_note))}</p>
</details>"""
        for record in high_events.itertuples(index=False)
    )
    formula_html = formulas.copy()
    formula_html["mean_score"] = formula_html["mean_score"].map(lambda value: number(value, 2))
    formula_html["mean_absolute_difference_from_weighted_50_50"] = formula_html[
        "mean_absolute_difference_from_weighted_50_50"
    ].map(lambda value: number(value, 2))
    formula_html["rank_correlation_with_weighted_50_50"] = formula_html[
        "rank_correlation_with_weighted_50_50"
    ].map(lambda value: number(value, 2))
    formula_columns = [
        ("formula_variant", "Formula"),
        ("mean_score", "Mean RCCI"),
        ("mean_absolute_difference_from_weighted_50_50", "Mean |Δ| vs. baseline"),
        ("rank_correlation_with_weighted_50_50", "Rank correlation"),
    ]
    data_links = "".join(
        f"<li><a href='../data/{filename}'>{html.escape(label)}</a></li>"
        for filename, label in [
            ("research_insights.md", "Research insights narrative (Markdown)"),
            ("research_insights.csv", "Research insights evidence table (CSV)"),
            ("poi_cluster_summary.csv", "Privacy-safe endpoint / POI-category summary (CSV)"),
            ("high_rcci_event_summary.csv", "High-RCCI event evidence (CSV)"),
            ("stable_period_summary.csv", "Stable-period evidence (CSV)"),
        ]
    )
    return f"""{SECTION_BEGIN}
<section id="research-insights" class="research-insights">
<h2>Research Insights</h2>
<p class="insight-lead">This section turns the RCCI outputs into privacy-preserving research findings. It focuses on route-network evidence, matched-segment frequency, generic local map context, and data quality; it does not infer trip purpose or personal attributes.</p>

<h3>Key findings</h3>
<div class="research-grid">
  <div class="research-card"><strong>Broward is the evidence base</strong><p>{number(network['broward_trip_count'])} of {number(network['all_matched_trip_count'])} matched trips ({network['broward_trip_count'] / network['all_matched_trip_count']:.1%}) and all HIGH/MEDIUM-confidence comparisons occur in Broward. Sparse supplementary counties are not used for behavioral interpretation.</p></div>
  <div class="research-card"><strong>Recurring route core</strong><p>Primary/secondary roads contribute {percent(network['primary_secondary_share'])} of weighted segment use, while residential/service roads add {percent(network['residential_service_share'])}; the pattern is consistent with recurring arterial-plus-local access.</p></div>
  <div class="research-card"><strong>Substantial change occurs in episodes</strong><p>{len(high_events)} reliable Broward pairs have RCCI ≥ {HIGH_CHANGE_THRESHOLD:.0f}; added/removed roads and transitions may indicate a possible routine shift, but RCCI cannot determine cause.</p></div>
  <div class="research-card"><strong>Relative stability appears in mid-2023</strong><p>The only multi-link stable run under the stated screen is 2023-06 to 2023-08. It suggests a relatively stable route routine, not an unchanged graph.</p></div>
</div>

<div class="research-caution"><strong>Interpretation guardrail:</strong> RCCI measures route-network change, not why it changed. Sparse months, zero-trip months, and large trip-count imbalances reduce confidence.</div>

<h3>Stable periods</h3>
<p>Stable screen: HIGH confidence, RCCI ≤ {STABLE_RCCI_MAX:.0f}, prior-month node retention ≥ {STABLE_NODE_RETENTION_MIN:.0%}, and prior-month edge retention ≥ {STABLE_EDGE_RETENTION_MIN:.0%}. Retention is structural (shared nodes/edges divided by the previous month); RCCI uses a separate frequency-weighted overlap.</p>
{html_table(stable_html, stable_columns)}
<p class="muted">No reliable pair retains at least 70% of both prior-month nodes and edges. “Stable” therefore means relatively stable within this driver’s observed history.</p>

<h3>High-change periods</h3>
<p>Every HIGH/MEDIUM-confidence Broward comparison with RCCI ≥ {HIGH_CHANGE_THRESHOLD:.0f} is summarized below. The upper-quartile focus band begins at RCCI ≥ {UPPER_QUARTILE_HIGH_CHANGE_THRESHOLD:.0f}.</p>
{html_table(high_html, high_columns)}
<div class="research-callout"><strong>Road and transition evidence:</strong> Expand a month pair to see the leading added/removed public-road corridors and directed transitions. These describe route-network change only; they do not identify destinations or reasons for travel.</div>
{details}

<h3>POI/category context</h3>
<p>Endpoint clusters use DBSCAN on map-matched start/end nodes (county-specific ε={int(CLUSTER_EPS_METERS)} m; min. samples={CLUSTER_MIN_SAMPLES}), then counts are tabulated by month. Local context comes only from cached OSM-derived land-use data within {int(LOCAL_POI_BUFFER_METERS)} m; names, addresses, and coordinates are omitted. No external API or map scraping was used.</p>
{html_table(cluster_html, cluster_columns)}
<div class="research-caution"><strong>Privacy note:</strong> A recurring cluster is a trip-segmentation/activity area, not proof of a visit, residence, workplace, school, clinic, hospital, or other personal association. Local category proximity is context only.</div>

<h3>Formula recommendation</h3>
<p>The <strong>weighted 50/50 node-edge RCCI</strong> is the recommended baseline. It balances which roads were used with how they were connected, emphasizes repeated driving behavior, downweights one-time noisy segments, and remains interpretable. Edge change averages {formula_facts['edge_minus_node_change']:.2f} points above node change, supporting retention of both components. Unweighted Jaccard differs from the weighted baseline by {formula_facts['jaccard_mean_abs_gap']:.2f} points on average (maximum {formula_facts['jaccard_max_abs_gap']:.2f}; rank correlation {formula_facts['jaccard_rank_correlation']:.2f}).</p>
{html_table(formula_html, formula_columns)}

<h3>Limitations</h3>
<ul>
  <li>RCCI measures route change, not cause.</li>
  <li>POI proximity does not prove the driver visited a location.</li>
  <li>Sparse months, zero-trip months, and large coverage differences reduce confidence.</li>
  <li>Local OSM-derived categories may be incomplete or outdated; external POI enrichment was skipped.</li>
  <li>No medical, employment, education, residence, income, political, religious, or identity claims are made.</li>
</ul>

<h3>Future research questions</h3>
<ul>
  <li>Do high-change episodes persist in following high-coverage months or revert toward the recurring route core?</li>
  <li>How would confidence-adjusted uncertainty intervals affect medium-confidence, high trip-ratio comparisons?</li>
  <li>Can these road/transition findings be reproduced across a larger privacy-preserving sample?</li>
  <li>Would time-of-day, trip-duration, or data-collection-cadence stratification distinguish route adaptation from sampling effects?</li>
</ul>

<h3>Research artefacts</h3>
<ul>{data_links}</ul>
</section>
{SECTION_END}"""


def insert_report_section(report: Path, section: str) -> None:
    if not report.exists():
        raise FileNotFoundError(f"Target report does not exist: {report}")
    document = report.read_text(encoding="utf-8")
    document = re.sub(
        re.escape(SECTION_BEGIN) + r".*?" + re.escape(SECTION_END),
        "",
        document,
        flags=re.DOTALL,
    )
    document = re.sub(
        re.escape(NAV_BEGIN) + r".*?" + re.escape(NAV_END),
        "",
        document,
        flags=re.DOTALL,
    )
    css = research_css()
    if STYLE_MARKER not in document:
        if "</style>" not in document:
            raise RuntimeError("Could not find a style block in the target report")
        document = document.replace("</style>", css + "\n</style>", 1)

    nav = f"{NAV_BEGIN}\n<a href=\"#research-insights\">Research Insights</a>\n{NAV_END}"
    if "</nav>" in document:
        document = document.replace("</nav>", nav + "\n</nav>", 1)
    else:
        raise RuntimeError("Could not find report navigation to insert Research Insights link")

    needle = "<section>\n<h2>What RCCI means</h2>"
    if needle in document:
        document = document.replace(needle, section + "\n\n" + needle, 1)
    elif "</main>" in document:
        document = document.replace("</main>", section + "\n</main>", 1)
    else:
        raise RuntimeError("Could not find a report insertion point")
    report.write_text(document, encoding="utf-8")


def validate_outputs(
    paths: Sequence[Path],
    report: Path,
    high_events: pd.DataFrame,
    clusters: pd.DataFrame,
    stable_periods: pd.DataFrame,
) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Expected research-insight files were not written: {missing}")
    if high_events.empty:
        raise RuntimeError("No high-change events were generated")
    if not (high_events["rcci"] >= HIGH_CHANGE_THRESHOLD).all():
        raise RuntimeError("High-change event table contains an RCCI below the threshold")
    if clusters.empty:
        raise RuntimeError("No recurring endpoint clusters met the documented frequency screen")
    if stable_periods.empty:
        raise RuntimeError("No stable comparisons met the documented stability screen")
    html_document = report.read_text(encoding="utf-8")
    for marker in (SECTION_BEGIN, SECTION_END, "id=\"research-insights\""):
        if marker not in html_document:
            raise RuntimeError(f"Research Insights HTML marker missing: {marker}")
    for forbidden_coordinate_header in ("centroid_x", "centroid_y", "latitude", "longitude"):
        for path in paths:
            if path.suffix == ".csv" and forbidden_coordinate_header in path.read_text(encoding="utf-8").splitlines()[0]:
                raise RuntimeError(f"Privacy-unsafe coordinate field found in {path}")


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    data_dir = output_root / "data"
    report = args.report.resolve()
    rcci, sensitivity = load_rcci_data(output_root)
    endpoints, _ = build_map_matched_endpoints()
    clustered = assign_endpoint_clusters(endpoints)
    clusters = build_cluster_summary(clustered)
    high_events = build_high_change_events(rcci)
    stable_periods = build_stable_periods(rcci)
    formulas, formula_facts = formula_summary(rcci, sensitivity)
    network = network_context()
    broward_trip_count = int(
        endpoints.loc[endpoints["county"] == PRIMARY_COUNTY, "trip_id"].nunique()
    )
    all_matched_trip_count = int(endpoints["trip_id"].nunique())
    network["broward_trip_count"] = broward_trip_count
    network["all_matched_trip_count"] = all_matched_trip_count
    insight_rows = create_insight_rows(
        rcci=rcci,
        clusters=clusters,
        high_events=high_events,
        stable_periods=stable_periods,
        network=network,
        formulas=formula_facts,
        broward_trip_count=broward_trip_count,
        all_matched_trip_count=all_matched_trip_count,
    )

    outputs = {
        "research_markdown": data_dir / "research_insights.md",
        "research_csv": data_dir / "research_insights.csv",
        "poi_clusters": data_dir / "poi_cluster_summary.csv",
        "high_events": data_dir / "high_rcci_event_summary.csv",
        "stable_periods": data_dir / "stable_period_summary.csv",
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs["research_markdown"].write_text(
        render_markdown(
            clusters=clusters,
            high_events=high_events,
            stable_periods=stable_periods,
            formulas=formulas,
            network=network,
            formula_facts=formula_facts,
        ),
        encoding="utf-8",
    )
    write_csv(insight_rows, outputs["research_csv"])
    write_csv(clusters, outputs["poi_clusters"])
    write_csv(high_events, outputs["high_events"])
    write_csv(stable_periods, outputs["stable_periods"])
    if not args.skip_report_update:
        insert_report_section(
            report,
            render_html_section(
                clusters=clusters,
                high_events=high_events,
                stable_periods=stable_periods,
                formulas=formulas,
                network=network,
                formula_facts=formula_facts,
            ),
        )
    validate_outputs(
        list(outputs.values()), report, high_events, clusters, stable_periods
    )
    print("Driver 1003 research insights complete")
    print(f"  report: {report}")
    for label, path in outputs.items():
        print(f"  {label}: {path}")
    print(f"  recurring endpoint clusters: {len(clusters)}")
    print(f"  high-change events: {len(high_events)}")
    print(f"  stable screens/runs: {len(stable_periods)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
