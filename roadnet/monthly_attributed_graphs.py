"""
Monthly attributed road graphs for the Driver 1003 longitudinal study.

The dynamic edge attribute is ``trip_use_count``: the number of distinct trips
in a calendar month whose matched route contains a given county/FID. Repeated
visits to the same FID within one trip count once.

County/FID is the edge identity because each county network has an independent
FID namespace. OSM ``u`` and ``v`` are the directed graph nodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Iterable, Sequence

import folium
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from branca.colormap import linear
from shapely.geometry import Point

from .driver_timeline import DriverTimelineError, parse_fid_sequence
from .html_assets import embed_local_html_assets


SUBJECT_LABEL = "Driver 1003"
SUBJECT_COLLECTION_ID = "1003 1004"
SUBJECT_INTERNAL_ID = "ca351c04cfabaae40cb77059ef799f4a"
SUBJECT_ALIAS = "Driver 1"
PRIMARY_COUNTY = "Broward County"
WGS84 = "EPSG:4326"

ROAD_ATTRIBUTE_COLUMNS = (
    "name",
    "highway",
    "length",
    "estimated_speed_limit",
    "speed_source",
    "speed_limit_confidence_score",
    "lanes",
    "oneway",
    "FDOT_AADT",
    "FDOT_TruckAADT",
    "FDOT_FUNCTIONAL_CLASS",
    "FDOT_LANE_COUNT",
    "landuse",
    "is_connector",
    "is_roundabout",
    "has_stop_sign_u",
    "has_stop_sign_v",
    "has_yield_u",
    "has_yield_v",
    "has_traffic_signal_u",
    "has_traffic_signal_v",
    "CUSTOM_OWNER",
    "FDOT_ROADWAY",
)

CANONICAL_NODE_ATTRIBUTE_COLUMNS = (
    "road_length_m",
    "oneway",
    "observed_avg_speed",
    "observed_median_speed",
    "road_owner_or_source",
)


@dataclass(frozen=True)
class SubjectDefinition:
    subject_label: str
    collection_id: str
    internal_driver_id: str
    timeline_path: Path
    primary_county: str = PRIMARY_COUNTY


@dataclass(frozen=True)
class MonthlyGraphBuildResult:
    subject_manifest_path: Path
    monthly_fid_usage_csv_path: Path
    monthly_fid_usage_parquet_path: Path
    monthly_graph_manifest_path: Path
    unmatched_fids_path: Path
    visual_overview_path: Path
    graph_root: Path
    graphml_count: int
    map_count: int
    observed_month_count: int
    calendar_month_count: int
    fid_node_dataset_count: int = 0
    fid_edge_dataset_count: int = 0
    total_monthly_fid_nodes: int = 0
    total_monthly_fid_edges: int = 0
    fid_graph_validation_path: Path | None = None
    proof_graph_path: Path | None = None
    upload_bundle_root: Path | None = None


def _generated_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _safe_scalar(value: object) -> object:
    """Convert pandas/numpy/list values into GraphML-safe scalar values."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (list, tuple, set, dict, np.ndarray)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def resolve_subject(
    output_dir: str | Path = "sflorida_outputs",
    subject: str = "1003",
) -> SubjectDefinition:
    """Resolve the professor-approved Driver 1003 study subject."""
    if subject not in {"1003", "driver_1003", "Driver 1003", SUBJECT_INTERNAL_ID}:
        raise DriverTimelineError(
            f"Unsupported subject {subject!r}; this phase is scoped to Driver 1003"
        )
    timeline_path = (
        Path(output_dir) / "phase2" / "driver_timelines" / "driver_1_timeline.csv"
    )
    if not timeline_path.exists():
        raise DriverTimelineError(
            f"Driver 1003 timeline not found: {timeline_path}. "
            "Run scripts/build_driver_timeline.py --driver auto first."
        )
    return SubjectDefinition(
        subject_label=SUBJECT_LABEL,
        collection_id=SUBJECT_COLLECTION_ID,
        internal_driver_id=SUBJECT_INTERNAL_ID,
        timeline_path=timeline_path,
    )


def load_subject_timeline(subject: SubjectDefinition) -> pd.DataFrame:
    """Load and validate the Phase 2A timeline for Driver 1003."""
    required = {
        "internal_driver_id",
        "collection_id",
        "trip_id",
        "county",
        "trip_month",
        "fid_sequence",
    }
    timeline = pd.read_csv(subject.timeline_path)
    missing = required - set(timeline.columns)
    if missing:
        raise DriverTimelineError(
            f"Timeline {subject.timeline_path} is missing columns: {sorted(missing)}"
        )
    timeline = timeline.loc[
        (timeline["internal_driver_id"].astype(str) == subject.internal_driver_id)
        & (timeline["collection_id"].astype(str) == subject.collection_id)
    ].copy()
    if timeline.empty:
        raise DriverTimelineError(
            "Driver 1003 timeline contains no rows matching the configured "
            "collection/internal ID"
        )
    return timeline


def build_trip_fid_membership(timeline: pd.DataFrame) -> pd.DataFrame:
    """Return one row per unique county/FID used by each trip."""
    rows: list[dict[str, object]] = []
    for trip in timeline.itertuples(index=False):
        unique_fids = sorted(set(parse_fid_sequence(trip.fid_sequence)))
        for fid in unique_fids:
            rows.append(
                {
                    "subject_label": SUBJECT_LABEL,
                    "internal_driver_id": str(trip.internal_driver_id),
                    "trip_month": str(trip.trip_month),
                    "county": str(trip.county),
                    "trip_id": str(trip.trip_id),
                    "fid": int(fid),
                }
            )
    membership = pd.DataFrame(rows)
    if membership.empty:
        raise DriverTimelineError("No unique trip/FID memberships were produced")
    if membership.duplicated(["trip_id", "county", "fid"]).any():
        raise DriverTimelineError("Trip/FID membership unexpectedly contains duplicates")
    return membership


def build_monthly_fid_usage(
    timeline: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Count distinct trip use for each month/county/FID."""
    trip_totals = (
        timeline.groupby(["trip_month", "county"])["trip_id"]
        .nunique()
        .rename("monthly_trip_count")
        .reset_index()
    )
    usage = (
        membership.groupby(["trip_month", "county", "fid"])
        .agg(
            trip_use_count=("trip_id", "nunique"),
        )
        .reset_index()
        .merge(trip_totals, on=["trip_month", "county"], how="left", validate="many_to_one")
    )
    usage.insert(0, "subject_label", SUBJECT_LABEL)
    usage.insert(1, "internal_driver_id", SUBJECT_INTERNAL_ID)
    usage["trip_use_share"] = (
        usage["trip_use_count"] / usage["monthly_trip_count"]
    )
    usage = usage.sort_values(
        ["trip_month", "county", "trip_use_count", "fid"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    return usage


def _gps_speed_mph(gps: pd.DataFrame) -> pd.Series:
    """Calculate point-to-point GPS speed in mph using the Phase 1 map method."""
    lat1 = gps["lat"].shift()
    lon1 = gps["lon"].shift()
    lat2 = gps["lat"]
    lon2 = gps["lon"]
    dt = gps["timestamp"].diff()
    rad = np.pi / 180.0
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2) ** 2
    )
    meters = 6_371_000.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    mph = (meters / dt.replace(0, np.nan)) * 2.236936
    return mph.where((mph >= 0) & (mph <= 120))


def build_monthly_observed_speeds(timeline: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Driver 1003 monthly/FID mean and median observed speed in mph.

    County GPS CSV points are aligned positionally with the matched ``opath``,
    matching the Phase 1 pipeline. This is driver/month-specific rather than a
    global network attribute.
    """
    observations: list[pd.DataFrame] = []
    for (county, gps_path, matched_path), trips in timeline.groupby(
        ["county", "gps_path", "matched_path"], sort=False
    ):
        gps = pd.read_csv(
            gps_path,
            sep=";",
            usecols=["id", "lon", "lat", "timestamp", "point_idx"],
        )
        matched = pd.read_csv(
            matched_path,
            sep=";",
            usecols=["id", "opath"],
            dtype={"opath": "string"},
        )
        # Resolve duplicated FMM fallback rows deterministically.
        matched["_present"] = matched["opath"].notna() & matched["opath"].str.len().gt(0)
        matched["_length"] = matched["opath"].str.len().fillna(-1)
        matched = (
            matched.sort_values(
                ["id", "_present", "_length"],
                ascending=[True, False, False],
            )
            .drop_duplicates("id", keep="first")
            .set_index("id")
        )
        month_by_trip = trips.set_index("matched_trip_id")["trip_month"].astype(str)
        selected_ids = set(month_by_trip.index.astype(int))
        gps = gps.loc[gps["id"].isin(selected_ids)].sort_values(["id", "point_idx"])

        for matched_trip_id, trip_gps in gps.groupby("id", sort=False):
            if matched_trip_id not in matched.index:
                continue
            opath = []
            for token in str(matched.at[matched_trip_id, "opath"]).split(","):
                try:
                    opath.append(int(token.strip()))
                except ValueError:
                    continue
            if not opath:
                continue
            trip_gps = trip_gps.reset_index(drop=True)
            length = min(len(trip_gps), len(opath))
            if length < 2:
                continue
            trip_gps = trip_gps.iloc[:length].copy()
            trip_gps["fid"] = opath[:length]
            trip_gps["observed_speed_mph"] = _gps_speed_mph(trip_gps)
            trip_gps = trip_gps.loc[trip_gps["fid"] >= 0]
            trip_gps["month"] = str(month_by_trip.loc[matched_trip_id])
            trip_gps["county"] = str(county)
            observations.append(
                trip_gps[["month", "county", "fid", "observed_speed_mph"]]
                .dropna(subset=["observed_speed_mph"])
            )

    if not observations:
        return pd.DataFrame(
            columns=[
                "month",
                "county",
                "fid",
                "observed_avg_speed",
                "observed_median_speed",
                "observed_speed_observation_count",
            ]
        )
    points = pd.concat(observations, ignore_index=True)
    return (
        points.groupby(["month", "county", "fid"])
        .agg(
            observed_avg_speed=("observed_speed_mph", "mean"),
            observed_median_speed=("observed_speed_mph", "median"),
            observed_speed_observation_count=("observed_speed_mph", "size"),
        )
        .reset_index()
    )


def collapse_consecutive_fids(value: object) -> list[int]:
    """Return the ordered FID path with only adjacent duplicates removed."""
    return parse_fid_sequence(value, collapse_consecutive=True)


def build_trip_fid_transitions(timeline: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row for every directed transition occurrence in every trip.

    A pair may occur more than once in the same trip after nonconsecutive route
    revisits. Those occurrences contribute to ``transition_count`` while the
    trip contributes only once to ``trip_count_using_transition``.
    """
    rows: list[dict[str, object]] = []
    for trip in timeline.itertuples(index=False):
        sequence = collapse_consecutive_fids(trip.fid_sequence)
        for index, (source_fid, target_fid) in enumerate(
            zip(sequence, sequence[1:])
        ):
            rows.append(
                {
                    "driver_id": str(trip.internal_driver_id),
                    "driver_label": SUBJECT_LABEL,
                    "driver_alias": SUBJECT_ALIAS,
                    "month": str(trip.trip_month),
                    "county": str(trip.county),
                    "trip_id": str(trip.trip_id),
                    "transition_index": index,
                    "source_fid": int(source_fid),
                    "target_fid": int(target_fid),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "driver_id",
            "driver_label",
            "driver_alias",
            "month",
            "county",
            "trip_id",
            "transition_index",
            "source_fid",
            "target_fid",
        ],
    )


def build_fid_node_table(attributed_edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert monthly road/FID usage rows into the FID-node representation."""
    nodes = attributed_edges.copy()
    nodes.insert(0, "driver_id", SUBJECT_INTERNAL_ID)
    nodes.insert(1, "driver_label", SUBJECT_LABEL)
    nodes.insert(2, "driver_alias", SUBJECT_ALIAS)
    nodes = nodes.rename(columns={"trip_month": "month"})
    nodes["count_rank"] = (
        nodes.groupby(["month", "county"])["trip_use_count"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    leading = [
        "driver_id",
        "driver_label",
        "driver_alias",
        "month",
        "county",
        "fid",
        "trip_use_count",
        "monthly_trip_count",
        "trip_use_share",
        "count_rank",
    ]
    remaining = [
        column for column in nodes.columns if column not in leading
    ]
    return gpd.GeoDataFrame(
        nodes[leading + remaining],
        geometry="geometry",
        crs=attributed_edges.crs,
    )


def add_canonical_node_attributes(
    fid_nodes: gpd.GeoDataFrame,
    observed_speeds: pd.DataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Add the stable public node-attribute schema without removing raw fields."""
    nodes = fid_nodes.copy()
    nodes["road_length_m"] = pd.to_numeric(
        nodes["length"] if "length" in nodes else pd.Series(pd.NA, index=nodes.index),
        errors="coerce",
    )
    if "oneway" not in nodes:
        nodes["oneway"] = pd.NA

    custom_owner = (
        nodes["CUSTOM_OWNER"]
        if "CUSTOM_OWNER" in nodes
        else pd.Series(pd.NA, index=nodes.index, dtype="object")
    )
    fdot_roadway = (
        nodes["FDOT_ROADWAY"]
        if "FDOT_ROADWAY" in nodes
        else pd.Series(pd.NA, index=nodes.index, dtype="object")
    )
    speed_source = (
        nodes["speed_source"]
        if "speed_source" in nodes
        else pd.Series(pd.NA, index=nodes.index, dtype="object")
    )
    owner = custom_owner.astype("string").str.strip()
    owner = owner.mask(owner.isin(["", "<NA>", "nan", "None"]))
    owner = owner.fillna(
        pd.Series(
            np.where(fdot_roadway.notna(), "FDOT", pd.NA),
            index=nodes.index,
            dtype="string",
        )
    )
    owner = owner.fillna(speed_source.astype("string").str.strip())
    owner = owner.mask(owner.isin(["", "<NA>", "nan", "None"]))
    nodes["road_owner_or_source"] = owner

    nodes["observed_avg_speed"] = np.nan
    nodes["observed_median_speed"] = np.nan
    nodes["observed_speed_observation_count"] = pd.Series(
        pd.NA, index=nodes.index, dtype="Int64"
    )
    if observed_speeds is not None and not observed_speeds.empty:
        nodes = nodes.merge(
            observed_speeds,
            on=["month", "county", "fid"],
            how="left",
            suffixes=("", "_calculated"),
            validate="one_to_one",
        )
        for column in (
            "observed_avg_speed",
            "observed_median_speed",
            "observed_speed_observation_count",
        ):
            calculated = f"{column}_calculated"
            if calculated in nodes:
                nodes[column] = nodes[calculated]
                nodes = nodes.drop(columns=calculated)

    for column in CANONICAL_NODE_ATTRIBUTE_COLUMNS:
        if column not in nodes:
            nodes[column] = pd.NA
    return gpd.GeoDataFrame(nodes, geometry="geometry", crs=fid_nodes.crs)


def add_canonical_attributes_to_map_edges(
    attributed_edges: gpd.GeoDataFrame,
    fid_nodes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Copy canonical node fields onto the road geometry rows used by maps."""
    lookup_columns = ["month", "county", "fid"] + list(
        CANONICAL_NODE_ATTRIBUTE_COLUMNS
    )
    lookup = fid_nodes[lookup_columns].rename(columns={"month": "trip_month"})
    base = attributed_edges.drop(
        columns=[
            column
            for column in CANONICAL_NODE_ATTRIBUTE_COLUMNS
            if column in attributed_edges.columns and column != "oneway"
        ],
        errors="ignore",
    )
    # ``oneway`` already exists as a raw enriched-network field.
    merge_columns = [
        column for column in lookup.columns if column != "oneway"
    ]
    merged = base.merge(
        lookup[merge_columns],
        on=["trip_month", "county", "fid"],
        how="left",
        validate="one_to_one",
    )
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=attributed_edges.crs)


def build_monthly_transition_edges(
    timeline: pd.DataFrame,
    fid_nodes: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate ordered trip transitions into monthly directed FID edges."""
    occurrences = build_trip_fid_transitions(timeline)
    columns = [
        "driver_id",
        "driver_label",
        "driver_alias",
        "month",
        "county",
        "source_fid",
        "target_fid",
        "transition_count",
        "trip_count_using_transition",
        "source_trip_use_count",
        "target_trip_use_count",
        "transition_share_of_month_trips",
        "source_count_rank",
        "target_count_rank",
    ]
    if occurrences.empty:
        return pd.DataFrame(columns=columns)

    monthly_trip_counts = (
        timeline.groupby(["trip_month", "county"])["trip_id"]
        .nunique()
        .rename("monthly_trip_count")
        .reset_index()
        .rename(columns={"trip_month": "month"})
    )
    edges = (
        occurrences.groupby(
            [
                "driver_id",
                "driver_label",
                "driver_alias",
                "month",
                "county",
                "source_fid",
                "target_fid",
            ],
            sort=False,
        )
        .agg(
            transition_count=("trip_id", "size"),
            trip_count_using_transition=("trip_id", "nunique"),
        )
        .reset_index()
        .merge(
            monthly_trip_counts,
            on=["month", "county"],
            how="left",
            validate="many_to_one",
        )
    )

    node_counts = fid_nodes[
        ["month", "county", "fid", "trip_use_count", "count_rank"]
    ].drop_duplicates(["month", "county", "fid"])
    source_counts = node_counts.rename(
        columns={
            "fid": "source_fid",
            "trip_use_count": "source_trip_use_count",
            "count_rank": "source_count_rank",
        }
    )
    target_counts = node_counts.rename(
        columns={
            "fid": "target_fid",
            "trip_use_count": "target_trip_use_count",
            "count_rank": "target_count_rank",
        }
    )
    edges = edges.merge(
        source_counts,
        on=["month", "county", "source_fid"],
        how="left",
        validate="many_to_one",
    ).merge(
        target_counts,
        on=["month", "county", "target_fid"],
        how="left",
        validate="many_to_one",
    )
    edges["transition_share_of_month_trips"] = (
        edges["trip_count_using_transition"] / edges["monthly_trip_count"]
    )
    edges = edges.sort_values(
        ["month", "county", "transition_count", "source_fid", "target_fid"],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)
    return edges[columns]


def discover_county_networks(
    output_dir: str | Path,
    counties: Iterable[str],
) -> dict[str, tuple[Path, Path]]:
    """Resolve enriched edge and OSM node caches for each county."""
    output_root = Path(output_dir)
    found: dict[str, tuple[Path, Path]] = {}
    for county in sorted(set(counties)):
        candidates = []
        county_key = _slug(county)
        for directory in output_root.glob("*_County"):
            edge_path = directory / "enriched_network.parquet"
            node_path = directory / "osm_nodes.parquet"
            if not edge_path.exists() or not node_path.exists():
                continue
            if _slug(directory.name) == county_key:
                candidates.append((edge_path, node_path))
        if len(candidates) != 1:
            raise DriverTimelineError(
                f"Expected one enriched edge/node cache for {county}; found {len(candidates)}"
            )
        found[county] = candidates[0]
    return found


def load_used_network_edges(
    usage: pd.DataFrame,
    networks: dict[str, tuple[Path, Path]],
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Join monthly FID usage to cached directed road edges."""
    joined_parts: list[gpd.GeoDataFrame] = []
    unmatched_parts: list[pd.DataFrame] = []

    for county, county_usage in usage.groupby("county", sort=True):
        edge_path, _ = networks[county]
        network = gpd.read_parquet(edge_path).reset_index()
        if "fid" not in network.columns:
            raise DriverTimelineError(f"Network {edge_path} has no fid column/index")
        network["fid"] = pd.to_numeric(network["fid"], errors="coerce").astype("Int64")
        wanted = set(county_usage["fid"].astype(int))
        network = network.loc[network["fid"].isin(wanted)].copy()
        available = set(network["fid"].dropna().astype(int))
        missing = county_usage.loc[~county_usage["fid"].isin(available)].copy()
        if not missing.empty:
            missing["reason"] = "fid_not_found_in_enriched_network"
            missing["network_path"] = str(edge_path)
            unmatched_parts.append(missing)

        keep = ["fid", "u", "v", "geometry"] + [
            column for column in ROAD_ATTRIBUTE_COLUMNS if column in network.columns
        ]
        network = network[keep].drop_duplicates("fid")
        merged = county_usage.merge(
            network,
            on="fid",
            how="inner",
            validate="many_to_one",
        )
        joined_parts.append(
            gpd.GeoDataFrame(merged, geometry="geometry", crs=network.crs)
        )

    if not joined_parts:
        raise DriverTimelineError("No monthly FID usage rows matched enriched networks")
    attributed = gpd.GeoDataFrame(
        pd.concat(joined_parts, ignore_index=True),
        geometry="geometry",
        crs=joined_parts[0].crs,
    )
    unmatched = (
        pd.concat(unmatched_parts, ignore_index=True)
        if unmatched_parts
        else pd.DataFrame(
            columns=list(usage.columns) + ["reason", "network_path"]
        )
    )
    return attributed, unmatched


def _csv_ready_nodes(nodes: gpd.GeoDataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(nodes.drop(columns="geometry")).copy()
    frame["geometry_wkt"] = nodes.geometry.map(
        lambda geometry: geometry.wkt
        if geometry is not None and not geometry.is_empty
        else ""
    )
    return frame


def export_fid_graph_tables(
    fid_nodes: gpd.GeoDataFrame,
    transition_edges: pd.DataFrame,
    data_dir: str | Path,
) -> dict[str, object]:
    """Export monthly and combined FID-node/transition-edge CSV and Parquet."""
    output = Path(data_dir)
    output.mkdir(parents=True, exist_ok=True)

    combined_node_csv = output / "driver_1003_all_monthly_nodes.csv"
    combined_node_parquet = output / "driver_1003_all_monthly_nodes.parquet"
    combined_edge_csv = output / "driver_1003_all_monthly_edges.csv"
    combined_edge_parquet = output / "driver_1003_all_monthly_edges.parquet"
    _csv_ready_nodes(fid_nodes).to_csv(combined_node_csv, index=False)
    fid_nodes.to_parquet(combined_node_parquet, index=False)
    transition_edges.to_csv(combined_edge_csv, index=False)
    transition_edges.to_parquet(combined_edge_parquet, index=False)

    node_paths: list[Path] = []
    edge_paths: list[Path] = []
    group_keys = sorted(
        {
            (str(row.month), str(row.county))
            for row in fid_nodes[["month", "county"]].itertuples(index=False)
        }
    )
    for month, county in group_keys:
        county_slug = _slug(county)
        prefix = output / f"driver_1003_{county_slug}_{month}"
        group_nodes = fid_nodes.loc[
            (fid_nodes["month"] == month) & (fid_nodes["county"] == county)
        ].copy()
        group_edges = transition_edges.loc[
            (transition_edges["month"] == month)
            & (transition_edges["county"] == county)
        ].copy()

        node_csv = Path(f"{prefix}_nodes.csv")
        node_parquet = Path(f"{prefix}_nodes.parquet")
        edge_csv = Path(f"{prefix}_edges.csv")
        edge_parquet = Path(f"{prefix}_edges.parquet")
        _csv_ready_nodes(group_nodes).to_csv(node_csv, index=False)
        group_nodes.to_parquet(node_parquet, index=False)
        group_edges.to_csv(edge_csv, index=False)
        group_edges.to_parquet(edge_parquet, index=False)
        node_paths.extend([node_csv, node_parquet])
        edge_paths.extend([edge_csv, edge_parquet])

    return {
        "monthly_node_dataset_count": len(group_keys),
        "monthly_edge_dataset_count": len(group_keys),
        "monthly_node_paths": tuple(node_paths),
        "monthly_edge_paths": tuple(edge_paths),
        "combined_node_paths": (combined_node_csv, combined_node_parquet),
        "combined_edge_paths": (combined_edge_csv, combined_edge_parquet),
    }


def validate_fid_transition_graphs(
    timeline: pd.DataFrame,
    fid_nodes: pd.DataFrame,
    transition_edges: pd.DataFrame,
    calendar_manifest: pd.DataFrame,
) -> dict[str, object]:
    """Validate the FID-node and transition-edge representation."""
    node_keys = set(
        zip(
            fid_nodes["month"].astype(str),
            fid_nodes["county"].astype(str),
            fid_nodes["fid"].astype(int),
        )
    )
    source_keys = set(
        zip(
            transition_edges["month"].astype(str),
            transition_edges["county"].astype(str),
            transition_edges["source_fid"].astype(int),
        )
    )
    target_keys = set(
        zip(
            transition_edges["month"].astype(str),
            transition_edges["county"].astype(str),
            transition_edges["target_fid"].astype(int),
        )
    )
    missing_source = sorted(source_keys - node_keys)
    missing_target = sorted(target_keys - node_keys)
    self_loops = transition_edges.loc[
        transition_edges["source_fid"] == transition_edges["target_fid"]
    ].copy()
    nonpositive_edges = transition_edges.loc[
        transition_edges["transition_count"] <= 0
    ].copy()

    node_groups = set(
        zip(fid_nodes["month"].astype(str), fid_nodes["county"].astype(str))
    )
    edge_groups = set(
        zip(
            transition_edges["month"].astype(str),
            transition_edges["county"].astype(str),
        )
    )
    nodes_without_edges = sorted(node_groups - edge_groups)

    expected_edge_groups: set[tuple[str, str]] = set()
    for trip in timeline.itertuples(index=False):
        if len(collapse_consecutive_fids(trip.fid_sequence)) >= 2:
            expected_edge_groups.add((str(trip.trip_month), str(trip.county)))
    missing_expected_edge_groups = sorted(expected_edge_groups - edge_groups)

    global_months = calendar_manifest.groupby("trip_month")["observed_month"].any()
    zero_trip_months = sorted(global_months.index[~global_months].astype(str))
    attribute_population_pct: dict[str, float] = {}
    for column in CANONICAL_NODE_ATTRIBUTE_COLUMNS:
        if column not in fid_nodes or len(fid_nodes) == 0:
            attribute_population_pct[column] = 0.0
            continue
        values = fid_nodes[column]
        populated = values.notna()
        if pd.api.types.is_string_dtype(values) or values.dtype == object:
            normalized = values.astype("string").str.strip().str.lower()
            populated &= ~normalized.isin(["", "unknown", "<na>", "nan", "none"])
        attribute_population_pct[column] = float(populated.mean() * 100.0)
    passed = not any(
        (
            missing_source,
            missing_target,
            len(nonpositive_edges),
            missing_expected_edge_groups,
        )
    )
    return {
        "monthly_node_dataset_count": len(node_groups),
        "monthly_edge_dataset_count": len(edge_groups),
        "total_monthly_nodes": int(len(fid_nodes)),
        "total_monthly_edges": int(len(transition_edges)),
        "zero_trip_months": zero_trip_months,
        "nodes_without_edges": nodes_without_edges,
        "self_loop_count": int(len(self_loops)),
        "self_loops": self_loops,
        "missing_source_references": missing_source,
        "missing_target_references": missing_target,
        "nonpositive_edge_count": int(len(nonpositive_edges)),
        "missing_expected_edge_groups": missing_expected_edge_groups,
        "attribute_population_pct": attribute_population_pct,
        "passed": passed,
    }


def write_fid_graph_validation(
    validation: dict[str, object],
    output_path: str | Path,
) -> Path:
    """Write the requested human-readable graph integrity report."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    def _group_list(values: Sequence[object]) -> str:
        return (
            ", ".join(
                f"{month} / {county}"
                for month, county in values
            )
            if values
            else "None"
        )

    zero_months = ", ".join(validation["zero_trip_months"]) or "None"
    missing_sources = len(validation["missing_source_references"])
    missing_targets = len(validation["missing_target_references"])
    coverage = validation["attribute_population_pct"]
    output.write_text(
        f"""# Driver 1003 monthly FID graph validation

Generated: {_generated_at()}

## Summary

- Monthly node datasets: {validation["monthly_node_dataset_count"]}
- Monthly edge datasets: {validation["monthly_edge_dataset_count"]}
- Total nodes across all months: {validation["total_monthly_nodes"]:,}
- Total directed edges across all months: {validation["total_monthly_edges"]:,}
- Months with zero trips: {zero_months}
- Month/county groups with nodes but no edges: {_group_list(validation["nodes_without_edges"])}
- Self-loop edges: {validation["self_loop_count"]}
- Missing source-FID node references: {missing_sources}
- Missing target-FID node references: {missing_targets}
- Nonpositive transition counts: {validation["nonpositive_edge_count"]}
- Eligible month/county groups missing edges: {_group_list(validation["missing_expected_edge_groups"])}

## Canonical node attribute population

- `road_length_m`: {coverage["road_length_m"]:.1f}%
- `oneway`: {coverage["oneway"]:.1f}%
- `observed_avg_speed`: {coverage["observed_avg_speed"]:.1f}%
- `observed_median_speed`: {coverage["observed_median_speed"]:.1f}%
- `road_owner_or_source`: {coverage["road_owner_or_source"]:.1f}%

Low or zero percentages indicate that the corresponding source attribute was
not available for those monthly FID rows; values are left null rather than
silently imputed.

## Result

**Validation {"PASSED" if validation["passed"] else "FAILED"}.**

Node identity is `(month, county, fid)`. Directed edge identity is
`(month, county, source_fid, target_fid)`. Consecutive duplicate FIDs are
collapsed before transitions are created. `transition_count` counts all
remaining transition occurrences, while `trip_count_using_transition` counts
distinct trips containing the transition.
""",
        encoding="utf-8",
    )
    return output


def load_graph_nodes(
    county: str,
    edges: gpd.GeoDataFrame,
    node_path: Path,
) -> gpd.GeoDataFrame:
    """Load endpoint nodes and fall back to edge endpoints when necessary."""
    needed = set(edges["u"]).union(set(edges["v"]))
    nodes = gpd.read_parquet(node_path)
    if "osmid" not in nodes.columns:
        raise DriverTimelineError(f"Node cache {node_path} has no osmid column")
    nodes = nodes.loc[nodes["osmid"].isin(needed)].copy()
    nodes = nodes[["osmid", "geometry"]].drop_duplicates("osmid")
    available = set(nodes["osmid"])

    fallback: dict[object, Point] = {}
    for edge in edges.itertuples(index=False):
        geometry = edge.geometry
        if geometry is None or geometry.is_empty:
            continue
        coords = list(geometry.geoms[0].coords) if geometry.geom_type == "MultiLineString" else list(geometry.coords)
        if edge.u not in available and edge.u not in fallback:
            fallback[edge.u] = Point(coords[0])
        if edge.v not in available and edge.v not in fallback:
            fallback[edge.v] = Point(coords[-1])

    if fallback:
        fallback_gdf = gpd.GeoDataFrame(
            {"osmid": list(fallback), "geometry": list(fallback.values())},
            geometry="geometry",
            crs=edges.crs,
        )
        nodes = pd.concat([nodes, fallback_gdf], ignore_index=True)
    nodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs=edges.crs)
    nodes["county"] = county
    nodes["coordinate_source"] = np.where(
        nodes["osmid"].isin(available), "osm_nodes", "edge_endpoint_fallback"
    )
    return nodes


def build_monthly_graph(
    edges_wgs84: gpd.GeoDataFrame,
    nodes_wgs84: gpd.GeoDataFrame,
    *,
    month: str,
    county: str,
) -> nx.MultiDiGraph:
    """Build a directed FID-keyed monthly graph."""
    graph = nx.MultiDiGraph(
        subject_label=SUBJECT_LABEL,
        internal_driver_id=SUBJECT_INTERNAL_ID,
        trip_month=month,
        county=county,
        crs=WGS84,
        dynamic_edge_attribute="trip_use_count",
    )
    for row in nodes_wgs84.itertuples(index=False):
        graph.add_node(
            str(row.osmid),
            x=float(row.geometry.x),
            y=float(row.geometry.y),
            county=county,
            coordinate_source=str(row.coordinate_source),
        )

    for row in edges_wgs84.itertuples(index=False):
        attributes = {
            column: _safe_scalar(getattr(row, column))
            for column in edges_wgs84.columns
            if column not in {"u", "v", "geometry"}
        }
        attributes["geometry_wkt"] = row.geometry.wkt
        graph.add_edge(
            str(row.u),
            str(row.v),
            key=str(int(row.fid)),
            **attributes,
        )
    return graph


def _format_popup_value(value: object) -> str:
    try:
        if pd.isna(value):
            return "unknown"
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return f"{value:,.2f}"
    return html.escape(str(value))


def write_month_map(
    edges: gpd.GeoDataFrame,
    output_path: str | Path,
    *,
    month: str,
    county: str,
    transitions: pd.DataFrame | None = None,
    top_edges: int = 250,
    presentation_title: str | None = None,
) -> Path:
    """Create a monthly FID-node map with a toggleable transition overlay."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    edges_wgs = edges.to_crs(WGS84).copy()
    bounds = edges_wgs.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    map_object = folium.Map(
        location=center,
        zoom_start=11,
        tiles="CartoDB positron",
        control_scale=True,
    )
    if presentation_title:
        map_object.get_root().header.add_child(
            folium.Element(
                f"<title>{html.escape(presentation_title)}</title>"
            )
        )

    maximum = max(int(edges_wgs["trip_use_count"].max()), 1)
    colormap = linear.YlOrRd_09.scale(1, maximum)
    colormap.caption = "Number of distinct trips using FID"
    colormap.add_to(map_object)
    fid_layer = folium.FeatureGroup(
        name="Road/FID usage count",
        show=True,
    ).add_to(map_object)

    popup_sections = [
        (
            "Basic",
            [
                ("FID", "fid"),
                ("Road name", "name"),
                ("Road type", "highway"),
            ],
        ),
        (
            "Usage",
            [
                ("Trips using this FID", "trip_use_count"),
                ("Monthly trip count", "monthly_trip_count"),
                ("Trip use share", "trip_use_share"),
            ],
        ),
        (
            "Road attributes",
            [
                ("Speed limit", "estimated_speed_limit"),
                ("Observed avg speed", "observed_avg_speed"),
                ("Observed median speed", "observed_median_speed"),
                ("Lanes", "lanes"),
                ("Road length", "road_length_m"),
                ("One-way", "oneway"),
                ("AADT", "FDOT_AADT"),
                ("Owner/source", "road_owner_or_source"),
            ],
        ),
    ]
    for row in edges_wgs.itertuples(index=False):
        count = int(row.trip_use_count)
        color = colormap(count)
        weight = 2.0 + 6.0 * math.sqrt(count / maximum)
        popup_parts = []
        for section, fields in popup_sections:
            popup_parts.append(
                "<tr><th colspan='2' style='text-align:left;padding:7px 0 3px;"
                "font-size:13px;color:#1e3a5f;border-bottom:1px solid #cbd5e1'>"
                f"{html.escape(section)}</th></tr>"
            )
            for label, field in fields:
                value = getattr(row, field, pd.NA)
                rendered = _format_popup_value(value)
                if field in {
                    "estimated_speed_limit",
                    "observed_avg_speed",
                    "observed_median_speed",
                } and rendered != "unknown":
                    rendered = f"{float(value):.1f} mph"
                elif field == "road_length_m" and rendered != "unknown":
                    rendered = f"{float(value):,.1f} m"
                elif field == "trip_use_share" and rendered != "unknown":
                    rendered = f"{float(value):.1%}"
                popup_parts.append(
                    "<tr>"
                    f"<th style='text-align:left;padding:3px 10px 3px 0;color:#475569'>{html.escape(label)}</th>"
                    f"<td style='padding:3px 0'>{rendered}</td>"
                    "</tr>"
                )
        popup_rows = "".join(popup_parts)
        popup = folium.Popup(
            f"<table style='font-size:12px'>{popup_rows}</table>",
            max_width=420,
        )
        geometry = row.geometry
        parts = list(geometry.geoms) if geometry.geom_type == "MultiLineString" else [geometry]
        for part in parts:
            folium.PolyLine(
                [(lat, lon) for lon, lat in part.coords],
                color=color,
                weight=weight,
                opacity=0.8,
                tooltip=(
                    f"FID {int(row.fid)} · {count} trip"
                    f"{'s' if count != 1 else ''}"
                ),
                popup=popup,
            ).add_to(fid_layer)

    transition_count = 0
    top_transition_label = "none"
    if transitions is not None and not transitions.empty and top_edges > 0:
        transition_layer = folium.FeatureGroup(
            name=f"Top FID transition count (max {top_edges})",
            show=False,
        ).add_to(map_object)
        selected = transitions.sort_values(
            ["transition_count", "trip_count_using_transition"],
            ascending=False,
        ).head(top_edges)
        transition_count = len(transitions)
        top_row = selected.iloc[0]
        top_transition_label = (
            f"{int(top_row.source_fid)} → {int(top_row.target_fid)} "
            f"({int(top_row.transition_count):,})"
        )
        geometry_by_fid = {
            int(row.fid): row.geometry
            for row in edges_wgs[["fid", "geometry"]].itertuples(index=False)
        }
        transition_max = max(int(selected["transition_count"].max()), 1)
        for transition in selected.itertuples(index=False):
            source_geometry = geometry_by_fid.get(int(transition.source_fid))
            target_geometry = geometry_by_fid.get(int(transition.target_fid))
            if source_geometry is None or target_geometry is None:
                continue
            source_point = source_geometry.interpolate(0.5, normalized=True)
            target_point = target_geometry.interpolate(0.5, normalized=True)
            weight = 1.5 + 6.0 * math.sqrt(
                int(transition.transition_count) / transition_max
            )
            transition_popup = folium.Popup(
                "<table style='font-size:12px'>"
                f"<tr><th>source_fid</th><td>{int(transition.source_fid)}</td></tr>"
                f"<tr><th>target_fid</th><td>{int(transition.target_fid)}</td></tr>"
                f"<tr><th>transition_count</th><td>{int(transition.transition_count):,}</td></tr>"
                "<tr><th>trip_count_using_transition</th>"
                f"<td>{int(transition.trip_count_using_transition):,}</td></tr>"
                "<tr><th>transition_share_of_month_trips</th>"
                f"<td>{float(transition.transition_share_of_month_trips):.3f}</td></tr>"
                "</table>",
                max_width=400,
            )
            folium.PolyLine(
                [
                    (source_point.y, source_point.x),
                    (target_point.y, target_point.x),
                ],
                color="#6d28d9",
                weight=weight,
                opacity=0.72,
                dash_array="6 5",
                tooltip=(
                    f"{int(transition.source_fid)} → "
                    f"{int(transition.target_fid)} · "
                    f"{int(transition.transition_count)} transitions"
                ),
                popup=transition_popup,
            ).add_to(transition_layer)

    monthly_trip_count = int(edges_wgs["monthly_trip_count"].iloc[0])
    title = presentation_title or (
        f"{SUBJECT_LABEL} · {month}"
    )
    legend = f"""
    <div style="position:fixed;top:18px;left:56px;z-index:9999;background:white;
      border:1px solid #cbd5e1;border-radius:9px;padding:13px 16px;
      box-shadow:0 3px 12px rgba(15,23,42,.18);font-family:Arial,sans-serif;
      max-width:620px;">
      <div style="font-size:16px;font-weight:700;">{html.escape(title)}</div>
      <div style="font-size:13px;line-height:1.5;color:#334155;">
        {html.escape(county)}<br>
        Trips: {monthly_trip_count:,}<br>
        Unique matched FIDs/nodes: {len(edges_wgs):,}<br>
        Directed transitions/edges: {transition_count:,}<br>
        <span style="color:#c2410c;font-weight:700;">━━</span>
        Road/FID usage count<br>
        <span style="color:#6d28d9;font-weight:700;">┄┄</span>
        Top FID transition count<br>
        Top transition: {html.escape(top_transition_label)}
      </div>
    </div>
    """
    map_object.get_root().html.add_child(folium.Element(legend))
    if presentation_title:
        top_fid = edges_wgs.sort_values(
            ["trip_use_count", "fid"], ascending=[False, True]
        ).iloc[0]
        summary_cards = f"""
        <div style="position:fixed;top:18px;right:18px;z-index:9998;
          display:grid;grid-template-columns:repeat(2,minmax(145px,1fr));
          gap:8px;font-family:Arial,sans-serif;max-width:390px;">
          <div style="background:white;padding:10px 12px;border-radius:8px;
            box-shadow:0 2px 9px rgba(15,23,42,.16);">
            <div style="font-size:10px;color:#64748b;text-transform:uppercase;">Trips</div>
            <div style="font-size:20px;font-weight:700;">{monthly_trip_count:,}</div>
          </div>
          <div style="background:white;padding:10px 12px;border-radius:8px;
            box-shadow:0 2px 9px rgba(15,23,42,.16);">
            <div style="font-size:10px;color:#64748b;text-transform:uppercase;">FID nodes</div>
            <div style="font-size:20px;font-weight:700;">{len(edges_wgs):,}</div>
          </div>
          <div style="background:white;padding:10px 12px;border-radius:8px;
            box-shadow:0 2px 9px rgba(15,23,42,.16);">
            <div style="font-size:10px;color:#64748b;text-transform:uppercase;">Directed edges</div>
            <div style="font-size:20px;font-weight:700;">{transition_count:,}</div>
          </div>
          <div style="background:white;padding:10px 12px;border-radius:8px;
            box-shadow:0 2px 9px rgba(15,23,42,.16);">
            <div style="font-size:10px;color:#64748b;text-transform:uppercase;">Most-used FID</div>
            <div style="font-size:15px;font-weight:700;">{int(top_fid.fid)}
              ({int(top_fid.trip_use_count):,})</div>
          </div>
          <div style="grid-column:1 / -1;background:white;padding:10px 12px;
            border-radius:8px;box-shadow:0 2px 9px rgba(15,23,42,.16);">
            <div style="font-size:10px;color:#64748b;text-transform:uppercase;">Most-used transition</div>
            <div style="font-size:15px;font-weight:700;">{html.escape(top_transition_label)}</div>
          </div>
        </div>
        """
        map_object.get_root().html.add_child(folium.Element(summary_cards))
    folium.LayerControl(collapsed=False).add_to(map_object)
    map_object.save(output)
    output.write_text(
        embed_local_html_assets(output.read_text(encoding="utf-8"), output.parent),
        encoding="utf-8",
    )
    return output


def build_calendar_manifest(
    timeline: pd.DataFrame,
    attributed_edges: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Create a complete month-by-county graph manifest."""
    first_month = str(timeline["trip_month"].min())
    last_month = str(timeline["trip_month"].max())
    months = pd.period_range(first_month, last_month, freq="M").astype(str)
    counties = sorted(timeline["county"].unique(), key=lambda value: (value != PRIMARY_COUNTY, value))
    trip_counts = (
        timeline.groupby(["trip_month", "county"])["trip_id"].nunique().to_dict()
    )
    fid_counts = (
        attributed_edges.groupby(["trip_month", "county"])["fid"].nunique().to_dict()
    )
    rows = []
    for month in months:
        for county in counties:
            trips = int(trip_counts.get((month, county), 0))
            rows.append(
                {
                    "subject_label": SUBJECT_LABEL,
                    "internal_driver_id": SUBJECT_INTERNAL_ID,
                    "trip_month": month,
                    "county": county,
                    "observed_month": trips > 0,
                    "trip_count": trips,
                    "unique_fid_count": int(fid_counts.get((month, county), 0)),
                    "is_primary_county": county == PRIMARY_COUNTY,
                    "edge_parquet_path": "",
                    "node_parquet_path": "",
                    "graphml_path": "",
                    "map_path": "",
                }
            )
    return pd.DataFrame(rows)


def write_subject_manifest(
    subject: SubjectDefinition,
    timeline: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Persist the mentor-approved study subject mapping."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "subject_label": subject.subject_label,
        "presentation_alias": SUBJECT_ALIAS,
        "collection_id": subject.collection_id,
        "internal_driver_id": subject.internal_driver_id,
        "primary_county": subject.primary_county,
        "timeline_path": str(subject.timeline_path),
        "usable_trip_count": int(len(timeline)),
        "first_month": str(timeline["trip_month"].min()),
        "last_month": str(timeline["trip_month"].max()),
        "interpretation": (
            "Professor Jang directed the project to use Driver 1003. The available "
            "data maps that study label to collection 1003 1004 and the stable "
            "internal pseudonymous ID listed above."
        ),
        "generated_at": _generated_at(),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def write_graph_overview(
    manifest: pd.DataFrame,
    fid_nodes: gpd.GeoDataFrame,
    transition_edges: pd.DataFrame,
    output_path: str | Path,
    *,
    data_dir: str | Path,
    map_dir: str | Path,
) -> Path:
    """Write the complete FID-node/transition-edge monthly overview."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)
    map_dir = Path(map_dir)
    top_fids = (
        fid_nodes.sort_values(
            ["month", "county", "trip_use_count", "fid"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["month", "county"])
        .set_index(["month", "county"])
    )
    top_transitions = (
        transition_edges.sort_values(
            [
                "month",
                "county",
                "transition_count",
                "trip_count_using_transition",
                "source_fid",
                "target_fid",
            ],
            ascending=[True, True, False, False, True, True],
        )
        .drop_duplicates(["month", "county"])
        .set_index(["month", "county"])
    )
    month_rows = []
    for month, month_data in manifest.groupby("trip_month", sort=True):
        observed = month_data.loc[month_data["observed_month"]]
        if observed.empty:
            month_rows.append(
                f"<tr class='zero'><td>{html.escape(month)}</td>"
                "<td colspan='8'>No observed trips</td></tr>"
            )
            continue
        for record in observed.itertuples(index=False):
            key = (str(month), str(record.county))
            top_fid = top_fids.loc[key]
            matching_edges = transition_edges.loc[
                (transition_edges["month"] == str(month))
                & (transition_edges["county"] == str(record.county))
            ]
            directed_edge_count = len(matching_edges)
            if key in top_transitions.index:
                top_edge = top_transitions.loc[key]
                top_transition = (
                    f"{int(top_edge.source_fid)} → {int(top_edge.target_fid)} "
                    f"({int(top_edge.transition_count):,})"
                )
            else:
                top_transition = "No transitions"
            county_slug = _slug(record.county)
            map_path = (
                map_dir
                / f"driver_1003_{county_slug}_{month}.html"
            )
            map_link = os.path.relpath(map_path, output.parent)
            month_rows.append(
                "<tr>"
                f"<td>{html.escape(month)}</td>"
                f"<td>{html.escape(str(record.county))}</td>"
                f"<td>{int(record.trip_count):,}</td>"
                f"<td>{int(record.unique_fid_count):,}</td>"
                f"<td>{directed_edge_count:,}</td>"
                f"<td>{int(top_fid.fid)} ({int(top_fid.trip_use_count):,} trips)</td>"
                f"<td>{html.escape(top_transition)}</td>"
                "<td>Generated separately</td>"
                f"<td><a href='{html.escape(map_link)}'>Open map</a></td>"
                "</tr>"
            )

    calendar_months = manifest["trip_month"].nunique()
    observed_months = manifest.loc[manifest["observed_month"], "trip_month"].nunique()
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{SUBJECT_LABEL} Monthly Attributed Graphs</title>
  <style>
    body {{ margin:0;background:#f1f5f9;color:#0f172a;
      font-family:Inter,ui-sans-serif,system-ui,sans-serif; }}
    main {{ max-width:1180px;margin:0 auto;padding:44px 28px 60px; }}
    h1 {{ margin:0;font-size:34px; }}
    p {{ color:#475569;line-height:1.65;max-width:920px; }}
    .cards {{ display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:28px 0; }}
    .card,.table-card {{ background:white;border:1px solid #dbe4ee;border-radius:14px;
      box-shadow:0 6px 22px rgba(15,23,42,.055); }}
    .card {{ padding:20px; }}
    .label {{ color:#64748b;font-size:12px;text-transform:uppercase;
      letter-spacing:.07em;font-weight:700; }}
    .value {{ font-size:24px;font-weight:750;margin-top:8px; }}
    .table-card {{ overflow-x:auto; }}
    table {{ width:100%;border-collapse:collapse; }}
    th,td {{ padding:14px 16px;border-bottom:1px solid #e2e8f0;text-align:left; }}
    th {{ background:#eaf1f8;font-size:12px;text-transform:uppercase;letter-spacing:.05em; }}
    tr.zero {{ color:#94a3b8;background:#f8fafc; }}
    a {{ color:#1d4ed8;font-weight:700;text-decoration:none; }}
  </style>
</head>
<body><main>
  <h1>{SUBJECT_LABEL} Monthly Attributed Graphs</h1>
  <p>
    Each graph uses matched FIDs as nodes and duplicate-collapsed consecutive
    FID pairs as directed edges. Node usage counts distinct trips; transition
    counts preserve repeated nonconsecutive transition occurrences.
    Broward County is primary; county FID namespaces remain separate.
  </p>
  <p>
    Node color and line width remain based on <code>trip_use_count</code>.
    Node popups include enriched road attributes and monthly Driver 1003
    observed average/median GPS speed where sufficient observations are available.
  </p>
  <div class="cards">
    <div class="card"><div class="label">Calendar months</div><div class="value">{calendar_months}</div></div>
    <div class="card"><div class="label">Observed months</div><div class="value">{observed_months}</div></div>
    <div class="card"><div class="label">Usable trips</div><div class="value">{int(manifest['trip_count'].sum()):,}</div></div>
    <div class="card"><div class="label">Monthly directed edges</div><div class="value">{len(transition_edges):,}</div></div>
  </div>
  <div class="table-card"><table>
    <thead><tr><th>Month</th><th>County</th><th>Trips</th>
      <th>Nodes</th><th>Directed edges</th><th>Most-used FID</th>
      <th>Top transition</th><th>Graph file</th><th>Map</th></tr></thead>
    <tbody>{''.join(month_rows)}</tbody>
  </table></div>
</main></body></html>
"""
    output.write_text(
        embed_local_html_assets(document, output.parent),
        encoding="utf-8",
    )
    return output


def export_monthly_graphs(
    attributed_edges: gpd.GeoDataFrame,
    transition_edges: pd.DataFrame,
    manifest: pd.DataFrame,
    networks: dict[str, tuple[Path, Path]],
    graph_root: str | Path,
    visual_root: str | Path,
    *,
    top_edges: int = 250,
) -> tuple[pd.DataFrame, int, int]:
    """Export GeoParquet, GraphML, and monthly maps for observed groups."""
    graph_root = Path(graph_root)
    visual_root = Path(visual_root)
    graphml_count = 0
    map_count = 0

    for (month, county), group in attributed_edges.groupby(
        ["trip_month", "county"], sort=True
    ):
        county_slug = _slug(county)
        month_dir = graph_root / county_slug
        month_dir.mkdir(parents=True, exist_ok=True)
        edges_path = month_dir / f"{month}_edges.parquet"
        nodes_path = month_dir / f"{month}_nodes.parquet"
        graphml_path = month_dir / f"{month}.graphml"
        map_path = visual_root / f"driver_1003_{county_slug}_{month}.html"

        edges = gpd.GeoDataFrame(group.copy(), geometry="geometry", crs=attributed_edges.crs)
        nodes = load_graph_nodes(county, edges, networks[county][1])
        edges_wgs = edges.to_crs(WGS84)
        nodes_wgs = nodes.to_crs(WGS84)
        edges_wgs.to_parquet(edges_path, index=False)
        nodes_wgs.to_parquet(nodes_path, index=False)

        graph = build_monthly_graph(
            edges_wgs,
            nodes_wgs,
            month=str(month),
            county=str(county),
        )
        nx.write_graphml(graph, graphml_path)
        graphml_count += 1
        write_month_map(
            edges,
            map_path,
            month=str(month),
            county=str(county),
            transitions=transition_edges.loc[
                (transition_edges["month"] == str(month))
                & (transition_edges["county"] == str(county))
            ],
            top_edges=top_edges,
        )
        map_count += 1

        selector = (manifest["trip_month"] == month) & (manifest["county"] == county)
        manifest.loc[selector, "edge_parquet_path"] = str(edges_path.resolve())
        manifest.loc[selector, "node_parquet_path"] = str(nodes_path.resolve())
        manifest.loc[selector, "graphml_path"] = str(graphml_path.resolve())
        manifest.loc[selector, "map_path"] = str(map_path.resolve())

    return manifest, graphml_count, map_count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_if_needed(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        return
    shutil.copy2(source, destination)


def _refresh_bundle_manifest(bundle_root: Path) -> None:
    """Recompute bundle inventory and SHA-256 checksums after all exports."""
    rows = []
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file() or path.name in {"MANIFEST.csv", "checksums.sha256"}:
            continue
        relative = path.relative_to(bundle_root)
        rows.append(
            {
                "category": relative.parts[0] if len(relative.parts) > 1 else "bundle_metadata",
                "relative_path": str(relative),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "source_path": "bundle_generated_or_copied",
                "contains_raw_gps": False,
                "contains_session_paths": False,
            }
        )
    manifest = pd.DataFrame(rows).sort_values(["category", "relative_path"])
    manifest.to_csv(bundle_root / "MANIFEST.csv", index=False)
    (bundle_root / "checksums.sha256").write_text(
        "\n".join(
            f"{row.sha256}  {row.relative_path}"
            for row in manifest.itertuples(index=False)
        )
        + "\n",
        encoding="utf-8",
    )


def prepare_google_drive_bundle(
    *,
    output_dir: str | Path,
    graph_root: Path,
    subject_manifest_path: Path,
    monthly_usage_paths: Sequence[Path],
    graph_manifest_path: Path,
    unmatched_fids_path: Path,
    overview_path: Path,
    visual_map_root: Path,
) -> Path:
    """Prepare a privacy-checked local upload bundle and checksums."""
    repo_root = Path.cwd()
    bundle_root = repo_root / "deliverables" / "google_drive_phase2"
    if bundle_root.exists():
        # Remove only the generated bundle, never Phase 1 source outputs.
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True)

    output_root = Path(output_dir)
    source_groups: list[tuple[str, Path]] = []
    for county_dir in sorted(output_root.glob("*_County")):
        for source in county_dir.glob("*_matched.csv"):
            source_groups.append(("matched_csv", source))
        for filename, category in (
            ("enriched_network.parquet", "enriched_network_parquet"),
            ("osm_nodes.parquet", "osm_nodes_parquet"),
        ):
            source = county_dir / filename
            if source.exists():
                source_groups.append((category, source))

    graph_sources = [
        path
        for path in graph_root.rglob("*")
        if path.is_file()
        and path.suffix in {".parquet", ".graphml", ".csv", ".json", ".md"}
    ]
    graph_sources.extend(
        [
            subject_manifest_path,
            *monthly_usage_paths,
            graph_manifest_path,
            unmatched_fids_path,
            overview_path,
        ]
    )
    graph_sources.extend(
        path for path in visual_map_root.rglob("*.html") if path.is_file()
    )
    for source in graph_sources:
        source_groups.append(("driver_1003_monthly_graphs", source))

    manifest_rows = []
    copied_destinations: set[Path] = set()
    for category, source in source_groups:
        source = Path(source)
        if not source.exists():
            continue
        if category == "driver_1003_monthly_graphs":
            if source == overview_path:
                relative = Path("visuals") / source.name
            elif source.is_relative_to(visual_map_root):
                relative = (
                    Path("visuals")
                    / visual_map_root.name
                    / source.relative_to(visual_map_root)
                )
            elif source.is_relative_to(graph_root / "data"):
                relative = Path("data") / source.relative_to(graph_root / "data")
            elif source.is_relative_to(graph_root):
                relative = Path("driver_1003") / source.relative_to(graph_root)
            else:
                relative = source.relative_to(graph_root.parent)
            destination = bundle_root / category / relative
        else:
            destination = bundle_root / category / source.parent.name / source.name
        if destination in copied_destinations:
            continue
        copied_destinations.add(destination)
        _copy_if_needed(source, destination)
        manifest_rows.append(
            {
                "category": category,
                "relative_path": str(destination.relative_to(bundle_root)),
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "source_path": str(source.resolve()),
                "contains_raw_gps": False,
                "contains_session_paths": False,
            }
        )

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["category", "relative_path"]
    )
    manifest.to_csv(bundle_root / "MANIFEST.csv", index=False)
    checksum_lines = [
        f"{row.sha256}  {row.relative_path}"
        for row in manifest.itertuples(index=False)
    ]
    (bundle_root / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )
    (bundle_root / "README.md").write_text(
        f"""# Phase 2 Google Drive upload bundle

Prepared: {_generated_at()}

This bundle contains cached matched route files, conflated/enriched road
networks, OSM node coordinates, and Driver 1003 monthly attributed graph
outputs requested by Professor Jang. The primary graph tables use FIDs as
nodes and duplicate-collapsed consecutive FID transitions as directed edges.

Privacy review:

- raw GPS JSONL files are excluded;
- accelerometer and OBD records are excluded;
- session directories and source GPS paths are excluded;
- the research subject is labeled `Driver 1003`;
- the internal pseudonymous ID is retained only in technical graph metadata and
  the subject manifest for reproducibility.

Use `MANIFEST.csv` for file descriptions/sizes and `checksums.sha256` to verify
the upload. This directory is prepared locally; uploading requires separate
authorization and a destination Google Drive folder.

Regeneration:

```bash
python scripts/build_driver_timeline.py --driver auto
python scripts/build_driver_1003_monthly_graphs.py --driver 1003 --top-edges 250
```
""",
        encoding="utf-8",
    )
    _refresh_bundle_manifest(bundle_root)
    return bundle_root


def add_fid_graph_manifest_fields(
    manifest: pd.DataFrame,
    fid_nodes: pd.DataFrame,
    transition_edges: pd.DataFrame,
    data_dir: str | Path,
) -> pd.DataFrame:
    """Add FID-node/transition counts and deliverable paths to the manifest."""
    updated = manifest.copy()
    updated["fid_node_count"] = 0
    updated["directed_edge_count"] = 0
    updated["top_transition"] = ""
    updated["fid_nodes_csv_path"] = ""
    updated["fid_edges_csv_path"] = ""
    data_dir = Path(data_dir)

    node_counts = fid_nodes.groupby(["month", "county"]).size().to_dict()
    edge_counts = transition_edges.groupby(["month", "county"]).size().to_dict()
    if transition_edges.empty:
        top_lookup: dict[tuple[str, str], object] = {}
    else:
        top_lookup = {
            (str(row.month), str(row.county)): row
            for row in transition_edges.sort_values(
                [
                    "month",
                    "county",
                    "transition_count",
                    "trip_count_using_transition",
                    "source_fid",
                    "target_fid",
                ],
                ascending=[True, True, False, False, True, True],
            )
            .drop_duplicates(["month", "county"])
            .itertuples(index=False)
        }

    for index, row in updated.iterrows():
        key = (str(row["trip_month"]), str(row["county"]))
        node_count = int(node_counts.get(key, 0))
        edge_count = int(edge_counts.get(key, 0))
        updated.at[index, "fid_node_count"] = node_count
        updated.at[index, "directed_edge_count"] = edge_count
        if node_count:
            county_slug = _slug(row["county"])
            prefix = data_dir / (
                f"driver_1003_{county_slug}_{row['trip_month']}"
            )
            updated.at[index, "fid_nodes_csv_path"] = str(
                Path(f"{prefix}_nodes.csv").resolve()
            )
            updated.at[index, "fid_edges_csv_path"] = str(
                Path(f"{prefix}_edges.csv").resolve()
            )
        top = top_lookup.get(key)
        if top is not None:
            updated.at[index, "top_transition"] = (
                f"{int(top.source_fid)}->{int(top.target_fid)} "
                f"({int(top.transition_count)})"
            )
    return updated


def build_driver_1003_monthly_graphs(
    *,
    subject: str = "1003",
    output_dir: str | Path = "sflorida_outputs",
    prepare_drive_bundle: bool = False,
    month: str | None = None,
    county: str | None = None,
    top_edges: int = 250,
) -> MonthlyGraphBuildResult:
    """Run the complete FID-node and transition-edge monthly graph workflow."""
    subject_definition = resolve_subject(output_dir, subject)
    timeline = load_subject_timeline(subject_definition)
    if month:
        timeline = timeline.loc[timeline["trip_month"].astype(str) == month].copy()
    if county:
        timeline = timeline.loc[timeline["county"].astype(str) == county].copy()
    if timeline.empty:
        raise DriverTimelineError(
            f"No Driver 1003 trips matched month={month!r}, county={county!r}"
        )

    membership = build_trip_fid_membership(timeline)
    usage = build_monthly_fid_usage(timeline, membership)
    networks = discover_county_networks(output_dir, timeline["county"].unique())
    attributed_edges, unmatched = load_used_network_edges(usage, networks)
    fid_nodes = build_fid_node_table(attributed_edges)
    observed_speeds = build_monthly_observed_speeds(timeline)
    fid_nodes = add_canonical_node_attributes(fid_nodes, observed_speeds)
    attributed_edges = add_canonical_attributes_to_map_edges(
        attributed_edges,
        fid_nodes,
    )
    transition_edges = build_monthly_transition_edges(timeline, fid_nodes)

    phase2_root = Path(output_dir) / "phase2"
    graph_root = phase2_root / "monthly_graphs" / "driver_1003"
    visual_root = phase2_root / "visuals" / "driver_1003_monthly_maps"
    data_root = graph_root / "data"
    graph_root.mkdir(parents=True, exist_ok=True)
    visual_root.mkdir(parents=True, exist_ok=True)

    subject_manifest_path = graph_root / "subject_manifest.json"
    write_subject_manifest(subject_definition, timeline, subject_manifest_path)

    usage_csv_path = graph_root / "monthly_fid_usage.csv"
    usage_parquet_path = graph_root / "monthly_fid_usage.parquet"
    attributed_edges.drop(columns="geometry").to_csv(usage_csv_path, index=False)
    attributed_edges.to_parquet(usage_parquet_path, index=False)

    unmatched_path = graph_root / "unmatched_fids.csv"
    unmatched.to_csv(unmatched_path, index=False)

    manifest = build_calendar_manifest(timeline, attributed_edges)
    table_outputs = export_fid_graph_tables(
        fid_nodes,
        transition_edges,
        data_root,
    )
    manifest, graphml_count, map_count = export_monthly_graphs(
        attributed_edges,
        transition_edges,
        manifest,
        networks,
        graph_root,
        visual_root,
        top_edges=top_edges,
    )
    manifest = add_fid_graph_manifest_fields(
        manifest,
        fid_nodes,
        transition_edges,
        data_root,
    )
    graph_manifest_path = graph_root / "monthly_graph_manifest.csv"
    manifest.to_csv(graph_manifest_path, index=False)

    validation = validate_fid_transition_graphs(
        timeline,
        fid_nodes,
        transition_edges,
        manifest,
    )
    validation_path = graph_root / "driver_1003_graph_validation.md"
    write_fid_graph_validation(validation, validation_path)

    overview_path = phase2_root / "visuals" / "driver_1003_monthly_graph_overview.html"
    write_graph_overview(
        manifest,
        fid_nodes,
        transition_edges,
        overview_path,
        data_dir=data_root,
        map_dir=visual_root,
    )

    proof_graph_path = (
        phase2_root
        / "visuals"
        / "driver_1003_broward_county_2023-08_graph.html"
    )
    proof_nodes = attributed_edges.loc[
        (attributed_edges["trip_month"] == "2023-08")
        & (attributed_edges["county"] == PRIMARY_COUNTY)
    ]
    proof_edges = transition_edges.loc[
        (transition_edges["month"] == "2023-08")
        & (transition_edges["county"] == PRIMARY_COUNTY)
    ]
    if not proof_nodes.empty:
        write_month_map(
            proof_nodes,
            proof_graph_path,
            month="2023-08",
            county=PRIMARY_COUNTY,
            transitions=proof_edges,
            top_edges=top_edges,
            presentation_title=(
                "Driver 1003 Monthly Attributed Graph: "
                "Broward County, 2023-08"
            ),
        )
    else:
        proof_graph_path = None

    bundle_root = None
    if prepare_drive_bundle:
        bundle_root = prepare_google_drive_bundle(
            output_dir=output_dir,
            graph_root=graph_root,
            subject_manifest_path=subject_manifest_path,
            monthly_usage_paths=[usage_csv_path, usage_parquet_path],
            graph_manifest_path=graph_manifest_path,
            unmatched_fids_path=unmatched_path,
            overview_path=overview_path,
            visual_map_root=visual_root,
        )
        bundle_graph_root = (
            bundle_root / "driver_1003_monthly_graphs"
        )
        bundle_data_root = bundle_graph_root / "data"
        # Ensure the exact requested flat data layout, independent of the
        # compatibility graph exports copied elsewhere in the bundle.
        export_fid_graph_tables(
            fid_nodes,
            transition_edges,
            bundle_data_root,
        )
        bundle_validation_path = (
            bundle_graph_root / "driver_1003_graph_validation.md"
        )
        shutil.copy2(validation_path, bundle_validation_path)
        bundle_overview_path = (
            bundle_graph_root / "driver_1003_monthly_graph_overview.html"
        )
        bundle_map_root = (
            bundle_graph_root
            / "visuals"
            / "driver_1003_monthly_maps"
        )
        write_graph_overview(
            manifest,
            fid_nodes,
            transition_edges,
            bundle_overview_path,
            data_dir=bundle_data_root,
            map_dir=bundle_map_root,
        )
        if proof_graph_path and proof_graph_path.exists():
            bundle_proof_path = (
                bundle_graph_root
                / "visuals"
                / proof_graph_path.name
            )
            _copy_if_needed(proof_graph_path, bundle_proof_path)
        _refresh_bundle_manifest(bundle_root)

    return MonthlyGraphBuildResult(
        subject_manifest_path=subject_manifest_path.resolve(),
        monthly_fid_usage_csv_path=usage_csv_path.resolve(),
        monthly_fid_usage_parquet_path=usage_parquet_path.resolve(),
        monthly_graph_manifest_path=graph_manifest_path.resolve(),
        unmatched_fids_path=unmatched_path.resolve(),
        visual_overview_path=overview_path.resolve(),
        graph_root=graph_root.resolve(),
        graphml_count=graphml_count,
        map_count=map_count,
        observed_month_count=manifest.loc[
            manifest["observed_month"], "trip_month"
        ].nunique(),
        calendar_month_count=manifest["trip_month"].nunique(),
        fid_node_dataset_count=int(
            table_outputs["monthly_node_dataset_count"]
        ),
        fid_edge_dataset_count=int(
            table_outputs["monthly_edge_dataset_count"]
        ),
        total_monthly_fid_nodes=int(len(fid_nodes)),
        total_monthly_fid_edges=int(len(transition_edges)),
        fid_graph_validation_path=validation_path.resolve(),
        proof_graph_path=(
            proof_graph_path.resolve()
            if proof_graph_path and proof_graph_path.exists()
            else None
        ),
        upload_bundle_root=bundle_root.resolve() if bundle_root else None,
    )
