"""Month-to-month attributed graph comparisons for Driver 1003.

The comparison layer consumes the cached Phase 2B node/edge tables. It never
reruns FMM or reconstructs monthly graphs. County is always part of node and
edge identity because FID namespaces are county-specific.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import html
import math
import os
from pathlib import Path
import re
from typing import Iterable, Sequence

import folium
import geopandas as gpd
import numpy as np
import pandas as pd

from .driver_timeline import DriverTimelineError


ALL_COUNTIES = "ALL_COUNTIES"
WGS84 = "EPSG:4326"
DEFAULT_GRAPH_ROOT = Path(
    "deliverables/google_drive_phase2/driver_1003_monthly_graphs"
)
DEFAULT_MANIFEST = Path(
    "sflorida_outputs/phase2/monthly_graphs/driver_1003/"
    "monthly_graph_manifest.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "deliverables/google_drive_phase2/driver_1003_graph_comparisons"
)

NODE_REQUIRED_COLUMNS = {
    "driver_id",
    "month",
    "county",
    "fid",
    "trip_use_count",
    "trip_use_share",
}
EDGE_REQUIRED_COLUMNS = {
    "driver_id",
    "month",
    "county",
    "source_fid",
    "target_fid",
    "transition_count",
    "trip_count_using_transition",
    "transition_share_of_month_trips",
}
MANIFEST_REQUIRED_COLUMNS = {
    "trip_month",
    "county",
    "observed_month",
    "trip_count",
}

NODE_DETAIL_COLUMNS = [
    "driver_id",
    "month_a",
    "month_b",
    "county",
    "fid",
    "status",
    "trip_use_count_a",
    "trip_use_count_b",
    "trip_use_count_delta",
    "trip_use_share_a",
    "trip_use_share_b",
    "trip_use_share_delta",
    "road_name",
    "road_type",
    "speed_limit",
    "lanes",
    "AADT",
    "road_length_m",
    "oneway",
    "road_owner_or_source",
    "observed_avg_speed_a",
    "observed_avg_speed_b",
    "observed_median_speed_a",
    "observed_median_speed_b",
]

EDGE_DETAIL_COLUMNS = [
    "driver_id",
    "month_a",
    "month_b",
    "county",
    "source_fid",
    "target_fid",
    "status",
    "transition_count_a",
    "transition_count_b",
    "transition_count_delta",
    "trip_count_using_transition_a",
    "trip_count_using_transition_b",
    "transition_share_a",
    "transition_share_b",
    "transition_share_delta",
]


@dataclass(frozen=True)
class GraphComparisonResult:
    node_comparison_csv: Path
    node_comparison_parquet: Path
    edge_comparison_csv: Path
    edge_comparison_parquet: Path
    summary_csv: Path
    summary_parquet: Path
    overview_html: Path
    detailed_html: Path | None
    detailed_map_html: Path | None
    comparison_map_htmls: tuple[Path, ...]
    validation_report: Path
    month_pair_count: int
    county_comparison_count: int
    node_comparison_rows: int
    edge_comparison_rows: int
    summary_rows: int
    validation_passed: bool


def _generated_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _require_columns(
    table: pd.DataFrame,
    required: set[str],
    description: str,
) -> None:
    missing = required - set(table.columns)
    if missing:
        raise DriverTimelineError(
            f"{description} is missing required columns: {sorted(missing)}"
        )


def _read_table(parquet_path: Path, csv_path: Path) -> pd.DataFrame:
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise DriverTimelineError(
        f"Neither cached input exists: {parquet_path} or {csv_path}"
    )


def load_graph_comparison_inputs(
    graph_root: str | Path = DEFAULT_GRAPH_ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load combined monthly graph tables and the zero-filled manifest."""
    graph_root = Path(graph_root)
    data_root = graph_root / "data"
    nodes = _read_table(
        data_root / "driver_1003_all_monthly_nodes.parquet",
        data_root / "driver_1003_all_monthly_nodes.csv",
    )
    edges = _read_table(
        data_root / "driver_1003_all_monthly_edges.parquet",
        data_root / "driver_1003_all_monthly_edges.csv",
    )
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise DriverTimelineError(f"Monthly graph manifest not found: {manifest_file}")
    manifest = pd.read_csv(manifest_file)
    _require_columns(nodes, NODE_REQUIRED_COLUMNS, "Combined node table")
    _require_columns(edges, EDGE_REQUIRED_COLUMNS, "Combined edge table")
    _require_columns(manifest, MANIFEST_REQUIRED_COLUMNS, "Monthly graph manifest")

    nodes = nodes.copy()
    edges = edges.copy()
    manifest = manifest.copy()
    nodes["month"] = nodes["month"].astype(str)
    edges["month"] = edges["month"].astype(str)
    manifest["trip_month"] = manifest["trip_month"].astype(str)
    manifest["observed_month"] = (
        manifest["observed_month"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(False)
        .astype(bool)
    )
    manifest["trip_count"] = (
        pd.to_numeric(manifest["trip_count"], errors="coerce").fillna(0).astype(int)
    )
    return nodes, edges, manifest


def build_consecutive_month_pairs(months: Iterable[str]) -> list[tuple[str, str]]:
    """Return every consecutive calendar pair between the minimum and maximum."""
    values = sorted({str(value) for value in months})
    if not values:
        return []
    periods = pd.period_range(values[0], values[-1], freq="M").astype(str).tolist()
    return list(zip(periods[:-1], periods[1:]))


def jaccard_similarity(shared_count: int, union_count: int) -> float:
    """Return set Jaccard, or NaN when both sets are empty."""
    return np.nan if union_count == 0 else shared_count / union_count


def weighted_overlap_min(
    weights_a: Sequence[float],
    weights_b: Sequence[float],
) -> float:
    """Generalized weighted Jaccard using min/max over aligned union rows."""
    a = np.asarray(weights_a, dtype=float)
    b = np.asarray(weights_b, dtype=float)
    denominator = np.maximum(a, b).sum()
    return np.nan if denominator == 0 else float(np.minimum(a, b).sum() / denominator)


def normalized_l1_change(
    weights_a: Sequence[float],
    weights_b: Sequence[float],
) -> float:
    """Normalize L1 change by the total weight across both months."""
    a = np.asarray(weights_a, dtype=float)
    b = np.asarray(weights_b, dtype=float)
    denominator = a.sum() + b.sum()
    return np.nan if denominator == 0 else float(np.abs(b - a).sum() / denominator)


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return np.nan if denominator == 0 else float(numerator / denominator)


def _coalesce_columns(
    merged: pd.DataFrame,
    source: str,
    destination: str,
) -> None:
    a = merged.get(f"{source}_a", pd.Series(pd.NA, index=merged.index))
    b = merged.get(f"{source}_b", pd.Series(pd.NA, index=merged.index))
    merged[destination] = b.combine_first(a)


def compare_node_tables(
    nodes_a: pd.DataFrame,
    nodes_b: pd.DataFrame,
    *,
    driver_id: str,
    month_a: str,
    month_b: str,
    county: str,
    key_columns: Sequence[str] = ("fid",),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare node membership and trip-use weights over the key union."""
    attribute_columns = [
        "name",
        "highway",
        "estimated_speed_limit",
        "lanes",
        "FDOT_AADT",
        "road_length_m",
        "oneway",
        "road_owner_or_source",
        "observed_avg_speed",
        "observed_median_speed",
    ]
    selected = list(key_columns) + [
        "trip_use_count",
        "trip_use_share",
        *[column for column in attribute_columns if column in nodes_a.columns or column in nodes_b.columns],
    ]

    def prepare(table: pd.DataFrame) -> pd.DataFrame:
        result = table.reindex(columns=selected).copy()
        if result.duplicated(list(key_columns)).any():
            raise DriverTimelineError(
                f"Duplicate node keys found for {county}, {month_a}->{month_b}"
            )
        return result

    merged = prepare(nodes_a).merge(
        prepare(nodes_b),
        on=list(key_columns),
        how="outer",
        suffixes=("_a", "_b"),
        indicator=True,
        validate="one_to_one",
    )
    merged["status"] = merged["_merge"].map(
        {"both": "shared", "left_only": "removed", "right_only": "added"}
    )
    for column in ("trip_use_count", "trip_use_share"):
        merged[f"{column}_a"] = pd.to_numeric(
            merged[f"{column}_a"], errors="coerce"
        ).fillna(0)
        merged[f"{column}_b"] = pd.to_numeric(
            merged[f"{column}_b"], errors="coerce"
        ).fillna(0)
        merged[f"{column}_delta"] = (
            merged[f"{column}_b"] - merged[f"{column}_a"]
        )

    static_mapping = {
        "name": "road_name",
        "highway": "road_type",
        "estimated_speed_limit": "speed_limit",
        "lanes": "lanes",
        "FDOT_AADT": "AADT",
        "road_length_m": "road_length_m",
        "oneway": "oneway",
        "road_owner_or_source": "road_owner_or_source",
    }
    for source, destination in static_mapping.items():
        _coalesce_columns(merged, source, destination)
    for speed in ("observed_avg_speed", "observed_median_speed"):
        for side in ("a", "b"):
            column = f"{speed}_{side}"
            if column not in merged:
                merged[column] = pd.NA

    merged.insert(0, "driver_id", driver_id)
    merged.insert(1, "month_a", month_a)
    merged.insert(2, "month_b", month_b)
    if "county" not in key_columns:
        merged.insert(3, "county", county)
    merged = merged.rename(
        columns={
            "trip_use_share_delta": "trip_use_share_delta",
        }
    )
    detail = merged.reindex(columns=NODE_DETAIL_COLUMNS)

    nodes_a_count = len(nodes_a)
    nodes_b_count = len(nodes_b)
    shared = int((merged["status"] == "shared").sum())
    added = int((merged["status"] == "added").sum())
    removed = int((merged["status"] == "removed").sum())
    union = len(merged)
    weights_a = merged["trip_use_count_a"].to_numpy(dtype=float)
    weights_b = merged["trip_use_count_b"].to_numpy(dtype=float)
    shared_mask = merged["status"] == "shared"
    summary = {
        "nodes_a": nodes_a_count,
        "nodes_b": nodes_b_count,
        "shared_nodes": shared,
        "added_nodes": added,
        "removed_nodes": removed,
        "union_nodes": union,
        "node_jaccard_similarity": jaccard_similarity(shared, union),
        "node_retention_rate": _safe_rate(shared, nodes_a_count),
        "node_new_rate": _safe_rate(added, nodes_b_count),
        "node_removed_rate": _safe_rate(removed, nodes_a_count),
        "total_node_weight_a": float(weights_a.sum()),
        "total_node_weight_b": float(weights_b.sum()),
        "shared_node_weight_a": float(weights_a[shared_mask].sum()),
        "shared_node_weight_b": float(weights_b[shared_mask].sum()),
        "weighted_node_overlap_min": weighted_overlap_min(weights_a, weights_b),
        "node_weight_change_l1": float(np.abs(weights_b - weights_a).sum()),
        "normalized_node_weight_change": normalized_l1_change(weights_a, weights_b),
    }
    return detail, summary


def compare_edge_tables(
    edges_a: pd.DataFrame,
    edges_b: pd.DataFrame,
    *,
    driver_id: str,
    month_a: str,
    month_b: str,
    county: str,
    key_columns: Sequence[str] = ("source_fid", "target_fid"),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare directed transition membership and transition weights."""
    value_columns = [
        "transition_count",
        "trip_count_using_transition",
        "transition_share_of_month_trips",
    ]
    selected = list(key_columns) + value_columns

    def prepare(table: pd.DataFrame) -> pd.DataFrame:
        result = table.reindex(columns=selected).copy()
        if result.duplicated(list(key_columns)).any():
            raise DriverTimelineError(
                f"Duplicate edge keys found for {county}, {month_a}->{month_b}"
            )
        return result

    merged = prepare(edges_a).merge(
        prepare(edges_b),
        on=list(key_columns),
        how="outer",
        suffixes=("_a", "_b"),
        indicator=True,
        validate="one_to_one",
    )
    merged["status"] = merged["_merge"].map(
        {"both": "shared", "left_only": "removed", "right_only": "added"}
    )
    for column in value_columns:
        merged[f"{column}_a"] = pd.to_numeric(
            merged[f"{column}_a"], errors="coerce"
        ).fillna(0)
        merged[f"{column}_b"] = pd.to_numeric(
            merged[f"{column}_b"], errors="coerce"
        ).fillna(0)
    merged["transition_count_delta"] = (
        merged["transition_count_b"] - merged["transition_count_a"]
    )
    merged["transition_share_delta"] = (
        merged["transition_share_of_month_trips_b"]
        - merged["transition_share_of_month_trips_a"]
    )
    merged = merged.rename(
        columns={
            "transition_share_of_month_trips_a": "transition_share_a",
            "transition_share_of_month_trips_b": "transition_share_b",
        }
    )
    merged.insert(0, "driver_id", driver_id)
    merged.insert(1, "month_a", month_a)
    merged.insert(2, "month_b", month_b)
    if "county" not in key_columns:
        merged.insert(3, "county", county)
    detail = merged.reindex(columns=EDGE_DETAIL_COLUMNS)

    edges_a_count = len(edges_a)
    edges_b_count = len(edges_b)
    shared = int((merged["status"] == "shared").sum())
    added = int((merged["status"] == "added").sum())
    removed = int((merged["status"] == "removed").sum())
    union = len(merged)
    weights_a = merged["transition_count_a"].to_numpy(dtype=float)
    weights_b = merged["transition_count_b"].to_numpy(dtype=float)
    summary = {
        "edges_a": edges_a_count,
        "edges_b": edges_b_count,
        "shared_edges": shared,
        "added_edges": added,
        "removed_edges": removed,
        "union_edges": union,
        "edge_jaccard_similarity": jaccard_similarity(shared, union),
        "edge_retention_rate": _safe_rate(shared, edges_a_count),
        "edge_new_rate": _safe_rate(added, edges_b_count),
        "edge_removed_rate": _safe_rate(removed, edges_a_count),
        "total_edge_weight_a": float(weights_a.sum()),
        "total_edge_weight_b": float(weights_b.sum()),
        "weighted_edge_overlap_min": weighted_overlap_min(weights_a, weights_b),
        "edge_weight_change_l1": float(np.abs(weights_b - weights_a).sum()),
        "normalized_edge_weight_change": normalized_l1_change(weights_a, weights_b),
    }
    return detail, summary


def data_quality_flag(
    *,
    trips_a: int,
    trips_b: int,
    nodes_a: int,
    nodes_b: int,
    edges_a: int,
    edges_b: int,
    missing_files: bool = False,
    low_trip_threshold: int = 10,
) -> str:
    """Return the primary quality flag using documented precedence."""
    if missing_files:
        return "missing_files"
    if trips_a == 0 and trips_b == 0:
        return "both_months_no_trips"
    if trips_a == 0:
        return "month_a_no_trips"
    if trips_b == 0:
        return "month_b_no_trips"
    if (nodes_a > 0 and edges_a == 0) or (nodes_b > 0 and edges_b == 0):
        return "nodes_but_no_edges"
    if (0 < trips_a < low_trip_threshold) or (0 < trips_b < low_trip_threshold):
        return "low_trip_count_month"
    return "ok"


def _manifest_trip_count(
    manifest: pd.DataFrame,
    month: str,
    county: str,
) -> int:
    selected = manifest.loc[
        (manifest["trip_month"] == month) & (manifest["county"] == county),
        "trip_count",
    ]
    return int(selected.iloc[0]) if not selected.empty else 0


def _expected_data_missing(
    manifest: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    month: str,
    counties: Sequence[str],
) -> bool:
    expected = manifest.loc[
        (manifest["trip_month"] == month)
        & (manifest["county"].isin(counties))
        & (manifest["trip_count"] > 0)
    ]
    if expected.empty:
        return False
    for row in expected.itertuples(index=False):
        month_nodes = nodes.loc[
            (nodes["month"] == month) & (nodes["county"] == row.county)
        ]
        if month_nodes.empty:
            return True
        expected_edges = int(getattr(row, "directed_edge_count", 0) or 0)
        if expected_edges > 0:
            month_edges = edges.loc[
                (edges["month"] == month) & (edges["county"] == row.county)
            ]
            if month_edges.empty:
                return True
    return False


def build_graph_comparisons(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    driver_id: str,
    counties: Sequence[str] | None = None,
    month_pairs: Sequence[tuple[str, str]] | None = None,
    include_combined_summary: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build detailed county rows and county/combined pair summaries."""
    selected_counties = list(counties or sorted(manifest["county"].unique()))
    pairs = list(
        month_pairs
        or build_consecutive_month_pairs(manifest["trip_month"].astype(str))
    )
    node_details: list[pd.DataFrame] = []
    edge_details: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    def compare_scope(
        month_a: str,
        month_b: str,
        county_label: str,
        scope_counties: Sequence[str],
        *,
        detailed: bool,
    ) -> None:
        nodes_a = nodes.loc[
            (nodes["month"] == month_a) & nodes["county"].isin(scope_counties)
        ]
        nodes_b = nodes.loc[
            (nodes["month"] == month_b) & nodes["county"].isin(scope_counties)
        ]
        edges_a = edges.loc[
            (edges["month"] == month_a) & edges["county"].isin(scope_counties)
        ]
        edges_b = edges.loc[
            (edges["month"] == month_b) & edges["county"].isin(scope_counties)
        ]
        combined = len(scope_counties) > 1
        node_keys = ("county", "fid") if combined else ("fid",)
        edge_keys = (
            ("county", "source_fid", "target_fid")
            if combined
            else ("source_fid", "target_fid")
        )
        node_detail, node_summary = compare_node_tables(
            nodes_a,
            nodes_b,
            driver_id=driver_id,
            month_a=month_a,
            month_b=month_b,
            county=county_label,
            key_columns=node_keys,
        )
        edge_detail, edge_summary = compare_edge_tables(
            edges_a,
            edges_b,
            driver_id=driver_id,
            month_a=month_a,
            month_b=month_b,
            county=county_label,
            key_columns=edge_keys,
        )
        if detailed:
            node_details.append(node_detail)
            edge_details.append(edge_detail)
        trips_a = sum(
            _manifest_trip_count(manifest, month_a, county)
            for county in scope_counties
        )
        trips_b = sum(
            _manifest_trip_count(manifest, month_b, county)
            for county in scope_counties
        )
        missing = _expected_data_missing(
            manifest, nodes, edges, month_a, scope_counties
        ) or _expected_data_missing(
            manifest, nodes, edges, month_b, scope_counties
        )
        summary = {
            "driver_id": driver_id,
            "month_a": month_a,
            "month_b": month_b,
            "county": county_label,
            "trips_a": trips_a,
            "trips_b": trips_b,
            **node_summary,
            **edge_summary,
        }
        summary["data_quality_flag"] = data_quality_flag(
            trips_a=trips_a,
            trips_b=trips_b,
            nodes_a=int(summary["nodes_a"]),
            nodes_b=int(summary["nodes_b"]),
            edges_a=int(summary["edges_a"]),
            edges_b=int(summary["edges_b"]),
            missing_files=missing,
        )
        summaries.append(summary)

    for month_a, month_b in pairs:
        for county in selected_counties:
            compare_scope(month_a, month_b, county, [county], detailed=True)
        if include_combined_summary:
            compare_scope(
                month_a,
                month_b,
                ALL_COUNTIES,
                selected_counties,
                detailed=False,
            )

    node_output = (
        pd.concat(node_details, ignore_index=True)
        if node_details
        else pd.DataFrame(columns=NODE_DETAIL_COLUMNS)
    )
    edge_output = (
        pd.concat(edge_details, ignore_index=True)
        if edge_details
        else pd.DataFrame(columns=EDGE_DETAIL_COLUMNS)
    )
    summary_output = pd.DataFrame(summaries)
    return node_output, edge_output, summary_output


def validate_graph_comparisons(
    node_details: pd.DataFrame,
    edge_details: pd.DataFrame,
    summary: pd.DataFrame,
    manifest: pd.DataFrame,
) -> dict[str, object]:
    """Validate row classifications, summary identities, and metric bounds."""
    errors: list[str] = []
    county_summary = summary.loc[summary["county"] != ALL_COUNTIES]
    for row in summary.itertuples(index=False):
        if row.shared_nodes + row.added_nodes != row.nodes_b:
            errors.append(
                f"{row.month_a}->{row.month_b} {row.county}: "
                "shared_nodes + added_nodes != nodes_b"
            )
        if row.shared_nodes + row.removed_nodes != row.nodes_a:
            errors.append(
                f"{row.month_a}->{row.month_b} {row.county}: "
                "shared_nodes + removed_nodes != nodes_a"
            )
        if row.shared_edges + row.added_edges != row.edges_b:
            errors.append(
                f"{row.month_a}->{row.month_b} {row.county}: "
                "shared_edges + added_edges != edges_b"
            )
        if row.shared_edges + row.removed_edges != row.edges_a:
            errors.append(
                f"{row.month_a}->{row.month_b} {row.county}: "
                "shared_edges + removed_edges != edges_a"
            )
    bounded = [
        "node_jaccard_similarity",
        "edge_jaccard_similarity",
        "weighted_node_overlap_min",
        "weighted_edge_overlap_min",
        "normalized_node_weight_change",
        "normalized_edge_weight_change",
    ]
    for column in bounded:
        invalid = summary[column].dropna().loc[
            ~summary[column].dropna().between(0, 1)
        ]
        if not invalid.empty:
            errors.append(f"{column} contains values outside [0, 1]")

    expected_node_rows = int(
        county_summary[
            ["union_nodes"]
        ].sum().iloc[0]
    )
    expected_edge_rows = int(
        county_summary[
            ["union_edges"]
        ].sum().iloc[0]
    )
    if len(node_details) != expected_node_rows:
        errors.append(
            f"Node detail rows {len(node_details)} != union total {expected_node_rows}"
        )
    if len(edge_details) != expected_edge_rows:
        errors.append(
            f"Edge detail rows {len(edge_details)} != union total {expected_edge_rows}"
        )
    zero_trip_months = sorted(
        manifest.groupby("trip_month")["trip_count"]
        .sum()
        .loc[lambda values: values == 0]
        .index.astype(str)
    )
    return {
        "passed": not errors,
        "errors": errors,
        "calendar_month_pairs": summary[["month_a", "month_b"]]
        .drop_duplicates()
        .shape[0],
        "county_comparisons": len(county_summary),
        "zero_trip_months": zero_trip_months,
        "missing_file_comparisons": int(
            (summary["data_quality_flag"] == "missing_files").sum()
        ),
        "low_trip_comparisons": int(
            (summary["data_quality_flag"] == "low_trip_count_month").sum()
        ),
        "node_comparison_rows": len(node_details),
        "edge_comparison_rows": len(edge_details),
        "summary_rows": len(summary),
    }


def write_validation_report(
    validation: dict[str, object],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    errors = validation["errors"]
    error_text = (
        "\n".join(f"- {error}" for error in errors)
        if errors
        else "- None"
    )
    zero_months = ", ".join(validation["zero_trip_months"]) or "None"
    output.write_text(
        f"""# Driver 1003 graph comparison validation

Generated: {_generated_at()}

- Calendar month pairs considered: {validation['calendar_month_pairs']}
- County-specific comparisons created: {validation['county_comparisons']}
- Months with no trips: {zero_months}
- Comparisons flagged for missing graph data: {validation['missing_file_comparisons']}
- Comparisons flagged for low trip count: {validation['low_trip_comparisons']}
- Node comparison rows created: {validation['node_comparison_rows']:,}
- Edge comparison rows created: {validation['edge_comparison_rows']:,}
- Summary rows created: {validation['summary_rows']:,}
- Added/shared/removed count identities: {'passed' if not errors else 'see errors'}
- Weighted metric bounds: {'passed' if not errors else 'see errors'}

## Errors

{error_text}

## Result

**Validation {'PASSED' if validation['passed'] else 'FAILED'}.**

Null similarity values represent comparisons where both graph sets are empty.
County remains part of every FID and transition key.
""",
        encoding="utf-8",
    )
    return output


def _format_number(value: object, decimals: int = 3) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):,.{decimals}f}"


def _html_table(
    frame: pd.DataFrame,
    columns: Sequence[tuple[str, str]],
    *,
    limit: int | None = None,
) -> str:
    selected = frame.head(limit) if limit else frame
    headers = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    rows = []
    for row in selected.itertuples(index=False):
        cells = []
        values = row._asdict()
        for column, _ in columns:
            value = values.get(column)
            if pd.isna(value):
                rendered = "—"
            elif column in {
                "fid",
                "source_fid",
                "target_fid",
                "trip_use_count_a",
                "trip_use_count_b",
                "trip_use_count_delta",
                "transition_count_a",
                "transition_count_b",
                "transition_count_delta",
            }:
                rendered = f"{int(value):,}"
            elif isinstance(value, (int, float, np.number)):
                rendered = _format_number(value)
            else:
                rendered = str(value)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f"<table><thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def write_comparison_map(
    nodes: gpd.GeoDataFrame,
    node_detail: pd.DataFrame,
    output_path: str | Path,
    *,
    county: str,
    month_a: str,
    month_b: str,
    summary_row: pd.Series | dict[str, object] | None = None,
) -> Path:
    """Write a three-layer added/shared/removed FID comparison map."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    print("[comparison-map] CRS before export:", nodes.crs)
    a = nodes.loc[
        (nodes["county"] == county) & (nodes["month"] == month_a)
    ].copy()
    b = nodes.loc[
        (nodes["county"] == county) & (nodes["month"] == month_b)
    ].copy()
    geometry = gpd.GeoDataFrame(
        pd.concat(
            [
                b[["county", "fid", "geometry"]].drop_duplicates(
                    ["county", "fid"]
                ),
                a[["county", "fid", "geometry"]].drop_duplicates(
                    ["county", "fid"]
                ),
            ],
            ignore_index=True,
        ).drop_duplicates(["county", "fid"]),
        geometry="geometry",
        crs=nodes.crs,
    )
    mapped = node_detail.merge(
        geometry,
        on=["county", "fid"],
        how="left",
        validate="one_to_one",
    )
    mapped = gpd.GeoDataFrame(mapped, geometry="geometry", crs=geometry.crs)
    print("[comparison-map] CRS after merge:", mapped.crs)
    missing_or_empty = mapped["geometry"].isna() | mapped.geometry.is_empty
    empty_geometry_count = int(missing_or_empty.sum())
    mapped = mapped.loc[~missing_or_empty].copy()
    if mapped.empty:
        raise DriverTimelineError("No node geometry available for comparison map")
    if mapped.crs is None:
        raise DriverTimelineError(
            "Comparison map geometry has no CRS; cannot safely pass coordinates "
            "to Leaflet."
        )
    if mapped.crs.to_epsg() != 4326:
        folium_nodes = mapped.to_crs(WGS84).copy()
    else:
        folium_nodes = mapped.copy()
    print("[comparison-map] CRS passed to Folium:", folium_nodes.crs)
    status_counts = {
        status: int((folium_nodes["status"] == status).sum())
        for status in ("shared", "added", "removed")
    }
    geometry_types = folium_nodes.geometry.geom_type.value_counts().to_dict()
    bounds = folium_nodes.total_bounds
    print("[comparison-map] total shared geometries:", status_counts["shared"])
    print("[comparison-map] total added geometries:", status_counts["added"])
    print("[comparison-map] total removed geometries:", status_counts["removed"])
    print("[comparison-map] number of empty geometries:", empty_geometry_count)
    print("[comparison-map] geometry types:", geometry_types)
    print("[comparison-map] map bounds before fit_bounds():", bounds.tolist())
    if summary_row is None:
        stats = {
            "trips_a": 0,
            "trips_b": 0,
            "nodes_a": int((folium_nodes["trip_use_count_a"] > 0).sum()),
            "nodes_b": int((folium_nodes["trip_use_count_b"] > 0).sum()),
            "shared_nodes": status_counts["shared"],
            "added_nodes": status_counts["added"],
            "removed_nodes": status_counts["removed"],
            "data_quality_flag": "unknown",
        }
    else:
        stats = dict(summary_row)

    trips_a = int(stats.get("trips_a", 0) or 0)
    trips_b = int(stats.get("trips_b", 0) or 0)
    nodes_a_count = int(stats.get("nodes_a", 0) or 0)
    nodes_b_count = int(stats.get("nodes_b", 0) or 0)
    shared_nodes = int(stats.get("shared_nodes", status_counts["shared"]) or 0)
    added_nodes = int(stats.get("added_nodes", status_counts["added"]) or 0)
    removed_nodes = int(stats.get("removed_nodes", status_counts["removed"]) or 0)
    quality_flag = str(stats.get("data_quality_flag", "unknown"))
    zero_note = ""
    if nodes_a_count == 0 and nodes_b_count > 0:
        zero_note = (
            f"No observations exist for this county in {html.escape(month_a)}. "
            f"All displayed FIDs are newly observed in {html.escape(month_b)}."
        )
    elif nodes_a_count > 0 and nodes_b_count == 0:
        zero_note = (
            f"No observations exist for this county in {html.escape(month_b)}. "
            f"All displayed FIDs were removed after {html.escape(month_a)}."
        )

    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    map_object = folium.Map(
        location=center,
        zoom_start=10,
        tiles="CartoDB positron",
        control_scale=True,
    )
    map_object.get_root().header.add_child(
        folium.Element(
            f"<title>Driver 1003 {html.escape(county)} "
            f"{html.escape(month_a)} to {html.escape(month_b)} "
            "FID Comparison</title>"
        )
    )
    colors = {"shared": "#64748b", "added": "#16a34a", "removed": "#dc2626"}
    labels = {
        "shared": "Shared FIDs",
        "added": f"Added in {month_b}",
        "removed": f"Removed after {month_a}",
    }
    maximum = max(
        float(folium_nodes[["trip_use_count_a", "trip_use_count_b"]].max().max()),
        1.0,
    )
    for status in ("shared", "added", "removed"):
        layer = folium.FeatureGroup(name=labels[status], show=True)
        for row in folium_nodes.loc[
            folium_nodes["status"] == status
        ].itertuples(index=False):
            relevant = max(float(row.trip_use_count_a), float(row.trip_use_count_b))
            weight = 2.0 + 6.0 * math.sqrt(relevant / maximum)
            popup = folium.Popup(
                "<table style='font-size:12px'>"
                f"<tr><th>FID</th><td>{int(row.fid)}</td></tr>"
                f"<tr><th>Status</th><td>{html.escape(status)}</td></tr>"
                f"<tr><th>{html.escape(month_a)} trips</th><td>{int(row.trip_use_count_a)}</td></tr>"
                f"<tr><th>{html.escape(month_b)} trips</th><td>{int(row.trip_use_count_b)}</td></tr>"
                f"<tr><th>Count delta</th><td>{int(row.trip_use_count_delta):+d}</td></tr>"
                f"<tr><th>Road</th><td>{html.escape(str(row.road_name) if not pd.isna(row.road_name) else 'unknown')}</td></tr>"
                f"<tr><th>Road type</th><td>{html.escape(str(row.road_type) if not pd.isna(row.road_type) else 'unknown')}</td></tr>"
                "</table>",
                max_width=380,
            )
            parts = (
                list(row.geometry.geoms)
                if row.geometry.geom_type == "MultiLineString"
                else [row.geometry]
            )
            for part in parts:
                folium.PolyLine(
                    [(lat, lon) for lon, lat in part.coords],
                    color=colors[status],
                    weight=weight,
                    opacity=0.78,
                    popup=popup,
                    tooltip=f"FID {int(row.fid)} · {status}",
                ).add_to(layer)
        layer.add_to(map_object)
    map_object.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    legend = f"""
    <div style="position:fixed;top:18px;left:55px;z-index:9999;background:white;
      border:1px solid #cbd5e1;border-radius:12px;padding:14px 16px;
      box-shadow:0 4px 16px rgba(15,23,42,.20);font-family:Arial,sans-serif;
      max-width:560px;">
      <div style="font-size:17px;font-weight:800;margin-bottom:4px;">
        Driver 1003 County-Specific Comparison
      </div>
      <div style="font-size:14px;font-weight:700;color:#334155;margin-bottom:8px;">
        {html.escape(county)} · {html.escape(month_a)} → {html.escape(month_b)}
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,minmax(92px,1fr));
        gap:8px;margin-bottom:10px;font-size:12px;">
        <div><span style="color:#64748b;">Trips</span><br><strong>{trips_a:,} → {trips_b:,}</strong></div>
        <div><span style="color:#64748b;">Unique FIDs</span><br><strong>{nodes_a_count:,} → {nodes_b_count:,}</strong></div>
        <div><span style="color:#64748b;">Quality</span><br><strong>{html.escape(quality_flag)}</strong></div>
        <div><span style="color:#64748b;">Shared</span><br><strong>{shared_nodes:,}</strong></div>
        <div><span style="color:#64748b;">Added</span><br><strong>{added_nodes:,}</strong></div>
        <div><span style="color:#64748b;">Removed</span><br><strong>{removed_nodes:,}</strong></div>
      </div>
      {f'<div style="background:#fff7ed;color:#7c2d12;border:1px solid #fed7aa;border-radius:8px;padding:8px;margin-bottom:9px;font-size:12px;line-height:1.4;">{zero_note}</div>' if zero_note else ''}
      <div style="font-size:13px;line-height:1.5;">
        <span style="color:#64748b">━━</span> Shared FIDs<br>
        <span style="color:#16a34a">━━</span> Added FIDs<br>
        <span style="color:#dc2626">━━</span> Removed FIDs
      </div>
    </div>
    """
    map_object.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(map_object)
    map_object.save(output)
    return output


def write_detailed_comparison_html(
    node_details: pd.DataFrame,
    edge_details: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: str | Path,
    *,
    county: str,
    month_a: str,
    month_b: str,
    map_path: Path | None = None,
) -> Path:
    """Write the proof-of-concept month-pair report."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pair_nodes = node_details.loc[
        (node_details["county"] == county)
        & (node_details["month_a"] == month_a)
        & (node_details["month_b"] == month_b)
    ]
    pair_edges = edge_details.loc[
        (edge_details["county"] == county)
        & (edge_details["month_a"] == month_a)
        & (edge_details["month_b"] == month_b)
    ]
    selected = summary.loc[
        (summary["county"] == county)
        & (summary["month_a"] == month_a)
        & (summary["month_b"] == month_b)
    ]
    if selected.empty:
        raise DriverTimelineError(
            f"No summary found for {county}, {month_a}->{month_b}"
        )
    row = selected.iloc[0]
    increased_nodes = pair_nodes.loc[
        (pair_nodes["status"] == "shared") & (pair_nodes["trip_use_count_delta"] > 0)
    ].sort_values("trip_use_count_delta", ascending=False)
    decreased_nodes = pair_nodes.loc[
        (pair_nodes["status"] == "shared") & (pair_nodes["trip_use_count_delta"] < 0)
    ].sort_values("trip_use_count_delta")
    added_nodes = pair_nodes.loc[pair_nodes["status"] == "added"].sort_values(
        "trip_use_count_b", ascending=False
    )
    removed_nodes = pair_nodes.loc[pair_nodes["status"] == "removed"].sort_values(
        "trip_use_count_a", ascending=False
    )
    increased_edges = pair_edges.loc[
        (pair_edges["status"] == "shared")
        & (pair_edges["transition_count_delta"] > 0)
    ].sort_values("transition_count_delta", ascending=False)
    decreased_edges = pair_edges.loc[
        (pair_edges["status"] == "shared")
        & (pair_edges["transition_count_delta"] < 0)
    ].sort_values("transition_count_delta")
    node_columns = [
        ("fid", "FID"),
        ("road_name", "Road"),
        ("road_type", "Type"),
        ("trip_use_count_a", month_a),
        ("trip_use_count_b", month_b),
        ("trip_use_count_delta", "Delta"),
    ]
    edge_columns = [
        ("source_fid", "Source"),
        ("target_fid", "Target"),
        ("transition_count_a", month_a),
        ("transition_count_b", month_b),
        ("transition_count_delta", "Delta"),
    ]
    cards = [
        ("Trips", f"{int(row.trips_a):,} → {int(row.trips_b):,}"),
        ("Nodes", f"{int(row.nodes_a):,} → {int(row.nodes_b):,}"),
        ("Edges", f"{int(row.edges_a):,} → {int(row.edges_b):,}"),
        ("Shared nodes", f"{int(row.shared_nodes):,}"),
        ("Added / removed nodes", f"{int(row.added_nodes):,} / {int(row.removed_nodes):,}"),
        ("Shared edges", f"{int(row.shared_edges):,}"),
        ("Added / removed edges", f"{int(row.added_edges):,} / {int(row.removed_edges):,}"),
        ("Node Jaccard", _format_number(row.node_jaccard_similarity)),
        ("Edge Jaccard", _format_number(row.edge_jaccard_similarity)),
    ]
    card_html = "".join(
        f"<div class='card'><div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{html.escape(value)}</div></div>"
        for label, value in cards
    )
    iframe = ""
    if map_path:
        relative = os.path.relpath(map_path, output.parent)
        iframe = (
            "<section><h2>FID membership map</h2>"
            "<p>Line width reflects the larger monthly trip-use count.</p>"
            f"<iframe src='{html.escape(relative)}' title='FID comparison map'></iframe>"
            "</section>"
        )
    sections = [
        ("Top increased FIDs", increased_nodes, node_columns),
        ("Top decreased FIDs", decreased_nodes, node_columns),
        ("Top added FIDs", added_nodes, node_columns),
        ("Top removed FIDs", removed_nodes, node_columns),
        ("Top increased transitions", increased_edges, edge_columns),
        ("Top decreased transitions", decreased_edges, edge_columns),
    ]
    table_html = "".join(
        f"<section><h2>{html.escape(title)}</h2><div class='table-card'>"
        f"{_html_table(frame, columns, limit=20)}</div></section>"
        for title, frame, columns in sections
    )
    output.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Driver 1003 {month_a} to {month_b} Graph Comparison</title>
<style>
body{{margin:0;background:#f1f5f9;color:#0f172a;font-family:Inter,system-ui,sans-serif}}
main{{max-width:1240px;margin:auto;padding:42px 28px 64px}} h1{{margin:0;font-size:34px}}
p{{color:#475569;line-height:1.6}} .cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:26px 0}}
.card,.table-card{{background:white;border:1px solid #dbe4ee;border-radius:13px;box-shadow:0 5px 18px rgba(15,23,42,.05)}}
.card{{padding:17px}} .label{{font-size:11px;text-transform:uppercase;color:#64748b;font-weight:750;letter-spacing:.06em}}
.value{{font-size:22px;font-weight:750;margin-top:7px}} section{{margin-top:30px}} .table-card{{overflow:auto}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:11px 13px;border-bottom:1px solid #e2e8f0;text-align:left}}
th{{background:#eaf1f8;font-size:12px}} iframe{{width:100%;height:690px;border:1px solid #cbd5e1;border-radius:13px;background:white}}
@media(max-width:800px){{.cards{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<h1>Driver 1003 Monthly Graph Comparison</h1>
<p><strong>{html.escape(county)} · {html.escape(month_a)} → {html.escape(month_b)}</strong></p>
<p>This report describes monthly route-network change using FIDs as nodes and
directed consecutive-FID transitions as edges. It does not constitute a final
driver-choice change metric or a clinical interpretation.</p>
<div class="cards">{card_html}</div>{iframe}{table_html}
</main></body></html>""",
        encoding="utf-8",
    )
    return output


def select_important_comparison_maps(summary: pd.DataFrame) -> pd.DataFrame:
    """Select a bounded set of county/month comparisons worth mapping.

    The comparison tables still cover every consecutive month pair. This helper
    only controls which interactive Leaflet maps are generated to avoid creating
    a large number of heavy HTML files.
    """
    county_rows = summary.loc[summary["county"] != ALL_COUNTIES].copy()
    if county_rows.empty:
        return county_rows

    selected_indices: set[int] = set()
    quality_ok = county_rows.loc[county_rows["data_quality_flag"] == "ok"]
    if not quality_ok.empty:
        highest_change = quality_ok.sort_values(
            ["edge_jaccard_similarity", "node_jaccard_similarity"],
            ascending=[True, True],
        ).index[0]
        most_stable = quality_ok.sort_values(
            ["edge_jaccard_similarity", "node_jaccard_similarity"],
            ascending=[False, False],
        ).index[0]
        selected_indices.update({int(highest_change), int(most_stable)})

    proof = county_rows.loc[
        (county_rows["county"] == "Broward County")
        & (county_rows["month_a"] == "2023-08")
        & (county_rows["month_b"] == "2023-09")
    ]
    selected_indices.update(int(index) for index in proof.index)

    zero_trip_adjacent = county_rows.loc[
        ((county_rows["trips_a"] == 0) | (county_rows["trips_b"] == 0))
        & ((county_rows["nodes_a"] > 0) | (county_rows["nodes_b"] > 0))
    ]
    selected_indices.update(int(index) for index in zero_trip_adjacent.index)

    if not selected_indices:
        return county_rows.iloc[0:0]
    return county_rows.loc[sorted(selected_indices)].copy()


def select_county_comparison_pages(summary: pd.DataFrame) -> pd.DataFrame:
    """Return every county/month comparison that has nodes in either month."""
    county_rows = summary.loc[summary["county"] != ALL_COUNTIES].copy()
    selected = county_rows.loc[
        (county_rows["nodes_a"] > 0) | (county_rows["nodes_b"] > 0)
    ].copy()
    if selected.empty:
        return selected
    selected["_zero_baseline_rank"] = (
        (selected["nodes_a"] == 0) | (selected["nodes_b"] == 0)
    ).astype(int)
    selected["_activity"] = (
        selected["nodes_a"]
        + selected["nodes_b"]
        + selected["trips_a"]
        + selected["trips_b"]
    )
    return (
        selected.sort_values(
            [
                "month_a",
                "month_b",
                "_zero_baseline_rank",
                "shared_nodes",
                "_activity",
                "county",
            ],
            ascending=[True, True, True, False, False, True],
        )
        .drop(columns=["_zero_baseline_rank", "_activity"])
        .copy()
    )


def comparison_page_path(
    visual_root: str | Path,
    *,
    county: str,
    month_a: str,
    month_b: str,
) -> Path:
    """Stable county-specific comparison page path."""
    visual_root = Path(visual_root)
    return (
        visual_root
        / "county_comparisons"
        / f"{month_a}_to_{month_b}"
        / f"driver_1003_{_slug(county)}_comparison.html"
    )


def write_overview_html(
    summary: pd.DataFrame,
    output_path: str | Path,
    *,
    detailed_path: Path | None = None,
    map_paths: dict[tuple[str, str, str], Path] | None = None,
) -> Path:
    """Write the presentation overview for all county/month comparisons."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    county_rows = summary.loc[summary["county"] != ALL_COUNTIES].copy()
    quality_ok = county_rows.loc[county_rows["data_quality_flag"] == "ok"]
    strongest = (
        quality_ok.sort_values(
            ["edge_jaccard_similarity", "node_jaccard_similarity"],
            ascending=[True, True],
        ).iloc[0]
        if not quality_ok.empty
        else None
    )
    stable = (
        quality_ok.sort_values(
            ["edge_jaccard_similarity", "node_jaccard_similarity"],
            ascending=[False, False],
        ).iloc[0]
        if not quality_ok.empty
        else None
    )
    observed_pairs = int(
        (
            (county_rows["trips_a"] > 0)
            & (county_rows["trips_b"] > 0)
        ).sum()
    )
    detailed_relative = (
        os.path.relpath(detailed_path, output.parent)
        if detailed_path
        else None
    )
    map_relatives = {
        key: os.path.relpath(path, output.parent)
        for key, path in (map_paths or {}).items()
    }
    mapped_rows = select_county_comparison_pages(summary)
    month_pair_sections = []
    month_pairs = (
        summary[["month_a", "month_b"]]
        .drop_duplicates()
        .sort_values(["month_a", "month_b"])
    )
    for pair in month_pairs.itertuples(index=False):
        pair_rows = mapped_rows.loc[
            (mapped_rows["month_a"] == pair.month_a)
            & (mapped_rows["month_b"] == pair.month_b)
        ].copy()
        if pair_rows.empty:
            county_cards = (
                "<div class='empty-pair'>No county has observed FID nodes in "
                f"{html.escape(str(pair.month_a))} or {html.escape(str(pair.month_b))}.</div>"
            )
        else:
            cards = []
            for record in pair_rows.itertuples(index=False):
                key = (str(record.county), str(record.month_a), str(record.month_b))
                map_link = map_relatives.get(key)
                open_link = (
                    f"<a href='{html.escape(map_link)}'>Open comparison</a>"
                    if map_link
                    else "<span class='missing-link'>Map not generated</span>"
                )
                details_link = ""
                if (
                    detailed_relative
                    and record.county == "Broward County"
                    and record.month_a == "2023-08"
                    and record.month_b == "2023-09"
                ):
                    details_link = (
                        f" · <a href='{html.escape(detailed_relative)}'>Details</a>"
                    )
                zero_note = ""
                if int(record.nodes_a) == 0 and int(record.nodes_b) > 0:
                    zero_note = (
                        f"<div class='zero-note'>No observations exist for this county in "
                        f"{html.escape(str(record.month_a))}. All displayed FIDs are newly "
                        f"observed in {html.escape(str(record.month_b))}.</div>"
                    )
                elif int(record.nodes_a) > 0 and int(record.nodes_b) == 0:
                    zero_note = (
                        f"<div class='zero-note'>No observations exist for this county in "
                        f"{html.escape(str(record.month_b))}. All displayed FIDs were "
                        f"removed after {html.escape(str(record.month_a))}.</div>"
                    )
                cards.append(
                    "<article class='county-card'>"
                    f"<h3>{html.escape(str(record.county))}</h3>"
                    "<div class='county-grid'>"
                    f"<div><span>Trips</span><strong>{int(record.trips_a):,} → {int(record.trips_b):,}</strong></div>"
                    f"<div><span>Unique FIDs</span><strong>{int(record.nodes_a):,} → {int(record.nodes_b):,}</strong></div>"
                    f"<div><span>Shared</span><strong>{int(record.shared_nodes):,}</strong></div>"
                    f"<div><span>Added</span><strong>{int(record.added_nodes):,}</strong></div>"
                    f"<div><span>Removed</span><strong>{int(record.removed_nodes):,}</strong></div>"
                    f"<div><span>Quality</span><strong>{html.escape(str(record.data_quality_flag))}</strong></div>"
                    "</div>"
                    f"{zero_note}"
                    "<div class='metrics'>"
                    f"Node Jaccard {_format_number(record.node_jaccard_similarity)} · "
                    f"Edge Jaccard {_format_number(record.edge_jaccard_similarity)} · "
                    f"Weighted node overlap {_format_number(record.weighted_node_overlap_min)} · "
                    f"Weighted edge overlap {_format_number(record.weighted_edge_overlap_min)}"
                    "</div>"
                    f"<div class='links'>{open_link}{details_link}</div>"
                    "</article>"
                )
            county_cards = f"<div class='county-cards'>{''.join(cards)}</div>"
        month_pair_sections.append(
            "<section class='month-pair'>"
            f"<h2>{html.escape(str(pair.month_a))} → {html.escape(str(pair.month_b))}</h2>"
            f"{county_cards}"
            "</section>"
        )

    def pair_label(row: pd.Series | None) -> str:
        if row is None:
            return "Not available"
        return (
            f"{row['month_a']} → {row['month_b']}<br>"
            f"<small>{html.escape(str(row['county']))}; edge Jaccard "
            f"{_format_number(row['edge_jaccard_similarity'])}</small>"
        )

    output.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Driver 1003 Month-to-Month Attributed Graph Comparison</title>
<style>
body{{margin:0;background:#f1f5f9;color:#0f172a;font-family:Inter,system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:42px 28px 64px}} h1{{margin:0;font-size:35px}}
p{{color:#475569;line-height:1.65;max-width:1050px}} .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:27px 0}}
.card,.table-card{{background:white;border:1px solid #dbe4ee;border-radius:14px;box-shadow:0 6px 20px rgba(15,23,42,.05)}}
.card{{padding:19px}} .label{{color:#64748b;font-size:11px;text-transform:uppercase;font-weight:750;letter-spacing:.06em}}
.value{{font-size:21px;font-weight:750;margin-top:8px}} small{{font-size:12px;color:#64748b}} .table-card{{overflow:auto}}
section{{margin-top:30px}} h2{{font-size:23px;margin:0 0 14px}} .section-note{{margin-top:0}}
.month-pair{{background:white;border:1px solid #dbe4ee;border-radius:16px;padding:22px;margin-top:22px;
box-shadow:0 6px 20px rgba(15,23,42,.045)}} .county-cards{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}
.county-card{{border:1px solid #dbe4ee;border-radius:13px;padding:16px;background:#f8fafc}} .county-card h3{{margin:0 0 12px;font-size:18px}}
.county-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}} .county-grid div{{background:white;border:1px solid #e2e8f0;border-radius:10px;padding:10px}}
.county-grid span{{display:block;color:#64748b;font-size:10px;text-transform:uppercase;font-weight:750;letter-spacing:.06em}}
.county-grid strong{{display:block;margin-top:4px;font-size:17px}} .metrics{{margin-top:11px;color:#475569;font-size:12px;line-height:1.5}}
.links{{margin-top:12px}} .zero-note{{margin-top:11px;background:#fff7ed;color:#7c2d12;border:1px solid #fed7aa;border-radius:9px;padding:9px;font-size:12px;line-height:1.45}}
.empty-pair{{background:#f8fafc;border:1px dashed #cbd5e1;border-radius:12px;padding:16px;color:#64748b}} .missing-link{{color:#94a3b8;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{padding:11px 12px;border-bottom:1px solid #e2e8f0;text-align:left;white-space:nowrap}}
th{{position:sticky;top:0;background:#eaf1f8;font-size:11px;text-transform:uppercase}} tr.quality-low{{background:#fff7ed;color:#7c2d12}}
.flag{{font-size:11px;font-weight:700}} a{{color:#1d4ed8;font-weight:700;text-decoration:none}}
@media(max-width:900px){{.cards,.county-cards{{grid-template-columns:1fr}} .county-grid{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<h1>Driver 1003 Month-to-Month Attributed Graph Comparison</h1>
<p>Each monthly graph uses FIDs as nodes and directed FID transitions as edges.
This report compares consecutive months to quantify how the driver/session route
graph changes over time. County-qualified keys prevent collisions between FID namespaces.</p>
<p>The strongest/stable labels below use edge Jaccard only as a descriptive
ordering among quality-OK comparisons; they are not the final path-choice change metric.</p>
<div class="cards">
<div class="card"><div class="label">Calendar month pairs</div><div class="value">{summary[['month_a','month_b']].drop_duplicates().shape[0]}</div></div>
<div class="card"><div class="label">Observed county pairs</div><div class="value">{observed_pairs}</div></div>
<div class="card"><div class="label">Strongest graph change</div><div class="value">{pair_label(strongest)}</div></div>
<div class="card"><div class="label">Most stable graph</div><div class="value">{pair_label(stable)}</div></div>
</div>
<section>
<h2>County-specific comparison pages by month pair</h2>
<p class="section-note">Each month pair can have multiple independent county
comparisons. Counties are ordered with meaningful shared overlap first, then
higher activity, with zero-baseline counties last. Counties with zero nodes in
both months are omitted.</p>
</section>
{''.join(month_pair_sections)}
</main></body></html>""",
        encoding="utf-8",
    )
    return output


def export_comparison_tables(
    node_details: pd.DataFrame,
    edge_details: pd.DataFrame,
    summary: pd.DataFrame,
    data_root: str | Path,
) -> dict[str, Path]:
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for key, frame, stem in (
        ("node", node_details, "driver_1003_month_to_month_node_comparisons"),
        ("edge", edge_details, "driver_1003_month_to_month_edge_comparisons"),
        ("summary", summary, "driver_1003_month_to_month_summary"),
    ):
        csv_path = data_root / f"{stem}.csv"
        parquet_path = data_root / f"{stem}.parquet"
        frame.to_csv(csv_path, index=False)
        frame.to_parquet(parquet_path, index=False)
        outputs[f"{key}_csv"] = csv_path
        outputs[f"{key}_parquet"] = parquet_path
    return outputs


def compare_driver_1003_monthly_graphs(
    *,
    driver: str = "1003",
    graph_root: str | Path = DEFAULT_GRAPH_ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_ROOT,
    county: str | None = None,
    month_a: str | None = None,
    month_b: str | None = None,
    compare_all: bool = True,
) -> GraphComparisonResult:
    """Run the complete cached month-to-month comparison workflow."""
    if driver not in {"1003", "driver_1003", "Driver 1003"}:
        raise DriverTimelineError("This phase is scoped to Driver 1003")
    nodes, edges, manifest = load_graph_comparison_inputs(graph_root, manifest_path)
    internal_driver_id = str(nodes["driver_id"].dropna().iloc[0])
    available_counties = sorted(manifest["county"].unique())
    if county and county not in available_counties:
        raise DriverTimelineError(
            f"Unknown county {county!r}; choose from {available_counties}"
        )
    counties = [county] if county else available_counties
    if month_a or month_b:
        if not (month_a and month_b):
            raise DriverTimelineError("--month-a and --month-b must be provided together")
        if pd.Period(month_a, freq="M") + 1 != pd.Period(month_b, freq="M"):
            raise DriverTimelineError("Month B must be the calendar month after month A")
        pairs = [(month_a, month_b)]
    elif compare_all:
        pairs = build_consecutive_month_pairs(manifest["trip_month"])
    else:
        raise DriverTimelineError("Use --all or provide --month-a and --month-b")

    node_details, edge_details, summary = build_graph_comparisons(
        nodes,
        edges,
        manifest,
        driver_id=internal_driver_id,
        counties=counties,
        month_pairs=pairs,
        include_combined_summary=len(counties) > 1,
    )
    output_root = Path(output_dir)
    data_root = output_root / "data"
    visual_root = output_root / "visuals"
    visual_root.mkdir(parents=True, exist_ok=True)
    outputs = export_comparison_tables(node_details, edge_details, summary, data_root)

    detailed_html: Path | None = None
    detailed_map: Path | None = None
    map_paths: dict[tuple[str, str, str], Path] = {}
    comparison_pages = select_county_comparison_pages(summary)
    if not comparison_pages.empty:
        geometry_nodes = gpd.read_parquet(
            Path(graph_root) / "data" / "driver_1003_all_monthly_nodes.parquet"
        )
        for record in comparison_pages.itertuples(index=False):
            key = (
                str(record.county),
                str(record.month_a),
                str(record.month_b),
            )
            pair_nodes = node_details.loc[
                (node_details["county"] == record.county)
                & (node_details["month_a"] == record.month_a)
                & (node_details["month_b"] == record.month_b)
            ]
            if pair_nodes.empty:
                continue
            map_path = comparison_page_path(
                visual_root,
                county=str(record.county),
                month_a=str(record.month_a),
                month_b=str(record.month_b),
            )
            write_comparison_map(
                geometry_nodes,
                pair_nodes,
                map_path,
                county=str(record.county),
                month_a=str(record.month_a),
                month_b=str(record.month_b),
                summary_row=record._asdict(),
            )
            map_paths[key] = map_path.resolve()

    proof_county = "Broward County"
    proof_a = "2023-08"
    proof_b = "2023-09"
    proof_exists = not summary.loc[
        (summary["county"] == proof_county)
        & (summary["month_a"] == proof_a)
        & (summary["month_b"] == proof_b)
    ].empty
    if proof_exists:
        detailed_map = map_paths.get((proof_county, proof_a, proof_b))
        detailed_html = visual_root / (
            "driver_1003_broward_2023-08_to_2023-09_comparison.html"
        )
        write_detailed_comparison_html(
            node_details,
            edge_details,
            summary,
            detailed_html,
            county=proof_county,
            month_a=proof_a,
            month_b=proof_b,
            map_path=detailed_map,
        )
    overview = visual_root / "driver_1003_graph_comparison_overview.html"
    write_overview_html(
        summary,
        overview,
        detailed_path=detailed_html,
        map_paths=map_paths,
    )
    validation = validate_graph_comparisons(
        node_details, edge_details, summary, manifest
    )
    validation_path = output_root / "driver_1003_graph_comparison_validation.md"
    write_validation_report(validation, validation_path)
    return GraphComparisonResult(
        node_comparison_csv=outputs["node_csv"].resolve(),
        node_comparison_parquet=outputs["node_parquet"].resolve(),
        edge_comparison_csv=outputs["edge_csv"].resolve(),
        edge_comparison_parquet=outputs["edge_parquet"].resolve(),
        summary_csv=outputs["summary_csv"].resolve(),
        summary_parquet=outputs["summary_parquet"].resolve(),
        overview_html=overview.resolve(),
        detailed_html=detailed_html.resolve() if detailed_html else None,
        detailed_map_html=detailed_map.resolve() if detailed_map else None,
        comparison_map_htmls=tuple(sorted(map_paths.values())),
        validation_report=validation_path.resolve(),
        month_pair_count=len(pairs),
        county_comparison_count=int(
            (summary["county"] != ALL_COUNTIES).sum()
        ),
        node_comparison_rows=len(node_details),
        edge_comparison_rows=len(edge_details),
        summary_rows=len(summary),
        validation_passed=bool(validation["passed"]),
    )
