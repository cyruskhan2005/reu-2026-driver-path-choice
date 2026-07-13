"""Route Choice Change Index (RCCI) for Driver 1003.

This Phase 3 layer consumes the completed Phase 2C graph comparison outputs.
It does not rerun FMM, rebuild monthly graphs, or change the graph comparison
logic.  RCCI v1 is an interpretable route-network change index, not a clinical
or diagnostic score.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import html
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .driver_timeline import DriverTimelineError
from .graph_comparisons import ALL_COUNTIES
from .html_assets import embed_local_html_assets


DEFAULT_INPUT_DIR = Path(
    "deliverables/google_drive_phase2/driver_1003_graph_comparisons/data"
)
DEFAULT_OUTPUT_DIR = Path(
    "deliverables/google_drive_phase2/driver_1003_route_choice_change_index"
)
DEFAULT_DRIVER_ID = "1003"
DEFAULT_REPORT_COUNTY = "Broward County"

SUMMARY_REQUIRED_COLUMNS = {
    "driver_id",
    "month_a",
    "month_b",
    "county",
    "trips_a",
    "trips_b",
    "nodes_a",
    "nodes_b",
    "edges_a",
    "edges_b",
    "shared_nodes",
    "added_nodes",
    "removed_nodes",
    "shared_edges",
    "added_edges",
    "removed_edges",
    "node_jaccard_similarity",
    "edge_jaccard_similarity",
    "weighted_node_overlap_min",
    "weighted_edge_overlap_min",
    "data_quality_flag",
}

RCCI_SUMMARY_COLUMNS = [
    "driver_id",
    "month_a",
    "month_b",
    "county",
    "trips_a",
    "trips_b",
    "trip_count_ratio",
    "nodes_a",
    "nodes_b",
    "edges_a",
    "edges_b",
    "weighted_node_overlap_min",
    "weighted_edge_overlap_min",
    "node_jaccard_similarity",
    "edge_jaccard_similarity",
    "node_change_component",
    "edge_change_component",
    "node_weight",
    "edge_weight",
    "rcci_v1",
    "confidence_label",
    "confidence_reason",
    "interpretation_label",
    "added_nodes",
    "removed_nodes",
    "added_edges",
    "removed_edges",
    "shared_nodes",
    "shared_edges",
    "data_quality_flag",
]

SENSITIVITY_COLUMNS = [
    "driver_id",
    "month_a",
    "month_b",
    "county",
    "trips_a",
    "trips_b",
    "trip_count_ratio",
    "rcci_v1",
    "rcci_balanced_weighted",
    "rcci_edge_heavy_weighted",
    "rcci_balanced_jaccard",
    "rcci_geometric_weighted",
    "confidence_label",
    "confidence_reason",
    "interpretation_label",
]


@dataclass(frozen=True)
class RCCIResult:
    summary_csv: Path
    summary_parquet: Path
    sensitivity_csv: Path
    sensitivity_parquet: Path
    report_html: Path
    validation_report: Path
    rows: int
    confidence_counts: dict[str, int]
    validation_passed: bool


def _generated_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_table(parquet_path: Path, csv_path: Path) -> pd.DataFrame:
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise DriverTimelineError(
        f"Neither comparison input exists: {parquet_path} or {csv_path}"
    )


def _require_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(table.columns)
    if missing:
        raise DriverTimelineError(f"{name} missing required columns: {sorted(missing)}")


def load_comparison_outputs(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Phase 2C graph comparison outputs, preferring Parquet over CSV."""
    root = Path(input_dir)
    summary = _read_table(
        root / "driver_1003_month_to_month_summary.parquet",
        root / "driver_1003_month_to_month_summary.csv",
    )
    nodes = _read_table(
        root / "driver_1003_month_to_month_node_comparisons.parquet",
        root / "driver_1003_month_to_month_node_comparisons.csv",
    )
    edges = _read_table(
        root / "driver_1003_month_to_month_edge_comparisons.parquet",
        root / "driver_1003_month_to_month_edge_comparisons.csv",
    )
    _require_columns(summary, SUMMARY_REQUIRED_COLUMNS, "RCCI summary input")
    return summary, nodes, edges


def normalize_weights(node_weight: float, edge_weight: float) -> tuple[float, float]:
    """Normalize nonnegative node/edge weights so they sum to one."""
    node = float(node_weight)
    edge = float(edge_weight)
    if node < 0 or edge < 0:
        raise ValueError("RCCI weights must be nonnegative")
    total = node + edge
    if total <= 0:
        raise ValueError("At least one RCCI weight must be positive")
    return node / total, edge / total


def compute_trip_count_ratio(trips_a: int | float, trips_b: int | float) -> float:
    """Return max(trips_a, trips_b) / min(trips_a, trips_b), or NaN if zero."""
    a = float(trips_a or 0)
    b = float(trips_b or 0)
    smaller = min(a, b)
    if smaller <= 0:
        return np.nan
    return max(a, b) / smaller


def _bounded_change(overlap: object) -> float:
    value = pd.to_numeric(pd.Series([overlap]), errors="coerce").iloc[0]
    if pd.isna(value):
        return np.nan
    return float(np.clip(1.0 - value, 0.0, 1.0))


def compute_rcci_components(row: pd.Series) -> tuple[float, float]:
    """Return node and edge change components on the 0-1 scale."""
    return (
        _bounded_change(row.get("weighted_node_overlap_min")),
        _bounded_change(row.get("weighted_edge_overlap_min")),
    )


def compute_rcci(
    node_change_component: float,
    edge_change_component: float,
    *,
    node_weight: float,
    edge_weight: float,
) -> float:
    """Compute RCCI on a 0-100 scale."""
    if pd.isna(node_change_component) or pd.isna(edge_change_component):
        return np.nan
    score = 100.0 * (
        node_weight * float(node_change_component)
        + edge_weight * float(edge_change_component)
    )
    return float(np.clip(score, 0.0, 100.0))


def assign_confidence_label(row: pd.Series) -> tuple[str, str]:
    """Assign a confidence label and concise reason string."""
    trips_a = int(row.get("trips_a", 0) or 0)
    trips_b = int(row.get("trips_b", 0) or 0)
    flag = str(row.get("data_quality_flag", "") or "")
    has_node_overlap = "weighted_node_overlap_min" in row.index
    has_edge_overlap = "weighted_edge_overlap_min" in row.index
    node_overlap = row.get("weighted_node_overlap_min")
    edge_overlap = row.get("weighted_edge_overlap_min")

    missing = (
        flag == "missing_files"
        or (
            (
                (has_node_overlap and pd.isna(node_overlap))
                or (has_edge_overlap and pd.isna(edge_overlap))
            )
            and not (trips_a == 0 and trips_b == 0)
        )
    )
    if missing:
        return "LOW", "missing_comparison_data"
    if trips_a == 0 and trips_b == 0:
        return "LOW", "both_months_no_trips"
    if trips_a == 0 or trips_b == 0:
        return "LOW", "zero_trip_month"
    if trips_a < 10 or trips_b < 10:
        return "LOW", "low_trip_count_under_10"

    reasons: list[str] = []
    if trips_a < 25 or trips_b < 25:
        reasons.append("medium_trip_count_10_to_24")
    ratio = compute_trip_count_ratio(trips_a, trips_b)
    if pd.notna(ratio) and ratio > 2.0:
        reasons.append("trip_count_ratio_gt_2")
    if reasons:
        return "MEDIUM", ";".join(reasons)
    return "HIGH", "high_coverage_balanced"


def assign_interpretation_label(row: pd.Series) -> str:
    """Assign Driver 1003 empirical interpretation text."""
    trips_a = int(row.get("trips_a", 0) or 0)
    trips_b = int(row.get("trips_b", 0) or 0)
    confidence = str(row.get("confidence_label", ""))
    score = row.get("rcci_v1")

    if trips_a == 0 and trips_b == 0:
        return "NO COMPARISON"
    if (trips_a == 0) != (trips_b == 0):
        return "ZERO-BASELINE CHANGE"
    if confidence == "LOW":
        return "LOW CONFIDENCE - interpret with trip-count context"
    if pd.isna(score):
        return "NO COMPARISON"
    if score < 60:
        return "LOW RELATIVE CHANGE"
    if score < 70:
        return "MODERATE RELATIVE CHANGE"
    if score < 80:
        return "HIGH RELATIVE CHANGE"
    return "VERY HIGH RELATIVE CHANGE"


def _prepare_summary_input(
    summary: pd.DataFrame,
    *,
    county: str | None = None,
    include_all_counties: bool = False,
) -> pd.DataFrame:
    data = summary.copy()
    data["month_a"] = data["month_a"].astype(str)
    data["month_b"] = data["month_b"].astype(str)
    data["county"] = data["county"].astype(str)
    if not include_all_counties:
        data = data.loc[data["county"] != ALL_COUNTIES].copy()
    if county:
        data = data.loc[data["county"] == county].copy()
    data = data.sort_values(["month_a", "month_b", "county"]).reset_index(drop=True)
    return data


def build_rcci_summary(
    summary: pd.DataFrame,
    *,
    node_weight: float = 0.5,
    edge_weight: float = 0.5,
    county: str | None = None,
    include_all_counties: bool = False,
) -> pd.DataFrame:
    """Build the main RCCI v1 summary table."""
    node_w, edge_w = normalize_weights(node_weight, edge_weight)
    output = _prepare_summary_input(
        summary,
        county=county,
        include_all_counties=include_all_counties,
    )
    if output.empty:
        return pd.DataFrame(columns=RCCI_SUMMARY_COLUMNS)

    numeric_columns = [
        "trips_a",
        "trips_b",
        "nodes_a",
        "nodes_b",
        "edges_a",
        "edges_b",
        "weighted_node_overlap_min",
        "weighted_edge_overlap_min",
        "node_jaccard_similarity",
        "edge_jaccard_similarity",
        "added_nodes",
        "removed_nodes",
        "added_edges",
        "removed_edges",
        "shared_nodes",
        "shared_edges",
    ]
    for column in numeric_columns:
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce")

    output["trip_count_ratio"] = [
        compute_trip_count_ratio(a, b)
        for a, b in zip(output["trips_a"], output["trips_b"], strict=False)
    ]
    components = output.apply(compute_rcci_components, axis=1, result_type="expand")
    output["node_change_component"] = components[0]
    output["edge_change_component"] = components[1]
    both_zero = (output["trips_a"].fillna(0) == 0) & (
        output["trips_b"].fillna(0) == 0
    )
    output.loc[both_zero, ["node_change_component", "edge_change_component"]] = np.nan
    output["node_weight"] = node_w
    output["edge_weight"] = edge_w
    output["rcci_v1"] = [
        compute_rcci(node_change, edge_change, node_weight=node_w, edge_weight=edge_w)
        for node_change, edge_change in zip(
            output["node_change_component"],
            output["edge_change_component"],
            strict=False,
        )
    ]
    labels = output.apply(assign_confidence_label, axis=1)
    output["confidence_label"] = [label for label, _ in labels]
    output["confidence_reason"] = [reason for _, reason in labels]
    output["interpretation_label"] = output.apply(assign_interpretation_label, axis=1)
    return output.reindex(columns=RCCI_SUMMARY_COLUMNS)


def build_sensitivity_table(rcci_summary: pd.DataFrame) -> pd.DataFrame:
    """Build optional sensitivity columns for formula comparison."""
    output = rcci_summary.copy()

    def change(column: str) -> pd.Series:
        return 1.0 - pd.to_numeric(output[column], errors="coerce")

    node_weighted = change("weighted_node_overlap_min")
    edge_weighted = change("weighted_edge_overlap_min")
    node_jaccard = change("node_jaccard_similarity")
    edge_jaccard = change("edge_jaccard_similarity")
    output["rcci_balanced_weighted"] = 100.0 * (
        0.5 * node_weighted + 0.5 * edge_weighted
    )
    output["rcci_edge_heavy_weighted"] = 100.0 * (
        0.3 * node_weighted + 0.7 * edge_weighted
    )
    output["rcci_balanced_jaccard"] = 100.0 * (
        0.5 * node_jaccard + 0.5 * edge_jaccard
    )
    geometric_overlap = np.sqrt(
        pd.to_numeric(output["weighted_node_overlap_min"], errors="coerce")
        * pd.to_numeric(output["weighted_edge_overlap_min"], errors="coerce")
    )
    output["rcci_geometric_weighted"] = 100.0 * (1.0 - geometric_overlap)
    both_zero = (output["trips_a"].fillna(0) == 0) & (
        output["trips_b"].fillna(0) == 0
    )
    sensitivity_cols = [
        "rcci_balanced_weighted",
        "rcci_edge_heavy_weighted",
        "rcci_balanced_jaccard",
        "rcci_geometric_weighted",
    ]
    output.loc[both_zero, sensitivity_cols] = np.nan
    for column in sensitivity_cols:
        output[column] = output[column].clip(lower=0, upper=100)
    return output.reindex(columns=SENSITIVITY_COLUMNS)


def _format_number(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if math.isfinite(numeric) and numeric.is_integer():
        return f"{int(numeric):,}"
    return f"{numeric:.{digits}f}"


def _format_score(value: object) -> str:
    return _format_number(value, 1)


def _html_table(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    limit: int | None = None,
    raw_html_columns: set[str] | None = None,
) -> str:
    data = frame.head(limit).copy() if limit else frame.copy()
    raw_html_columns = raw_html_columns or set()
    if data.empty:
        return "<p class='empty'>No rows.</p>"
    headers = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    rows = []
    for record in data.reindex(columns=columns).to_dict(orient="records"):
        cells = ""
        for column in columns:
            value = record.get(column)
            if column in raw_html_columns and isinstance(value, str):
                cells += f"<td>{value}</td>"
            else:
                cells += f"<td>{_format_number(value)}</td>"
        rows.append(f"<tr>{cells}</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def _county_slug(value: object) -> str:
    return (
        str(value)
        .lower()
        .replace("&", "and")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def _comparison_relative_href(row: pd.Series, *, from_dir: Path) -> str | None:
    pair_dir = f"{row['month_a']}_to_{row['month_b']}"
    county_slug = _county_slug(row["county"])
    filename = f"driver_1003_{county_slug}_comparison.html"
    candidates = [
        # Clean share folder:
        # deliverables/driver_1003/route_choice_change_index/visuals -> graph_comparisons
        from_dir
        / ".."
        / ".."
        / "graph_comparisons"
        / "county_comparisons"
        / pair_dir
        / filename,
        # Large Google Drive bundle:
        # deliverables/google_drive_phase2/driver_1003_route_choice_change_index/visuals
        from_dir
        / ".."
        / ".."
        / "driver_1003_graph_comparisons"
        / "visuals"
        / "county_comparisons"
        / pair_dir
        / filename,
    ]
    return _first_existing_relative(candidates, from_dir=from_dir)


def _first_existing_relative(candidates: Sequence[Path], *, from_dir: Path) -> str | None:
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return os.path.relpath(resolved, from_dir)
    return None


def _share_link(filename: str, *, from_dir: Path) -> str:
    candidates = [
        from_dir / ".." / ".." / filename,
        from_dir / ".." / ".." / ".." / "driver_1003" / filename,
    ]
    href = _first_existing_relative(candidates, from_dir=from_dir)
    if href:
        return href
    return os.path.relpath(candidates[0], from_dir)


def _add_comparison_links(frame: pd.DataFrame, *, from_dir: Path) -> pd.DataFrame:
    output = frame.copy()
    links = []
    for _, row in output.iterrows():
        href = _comparison_relative_href(row, from_dir=from_dir)
        if href:
            links.append(f"<a href='{html.escape(href)}'>Open comparison</a>")
        else:
            links.append("<span class='muted'>Unavailable</span>")
    output["Comparison HTML link"] = links
    output["Node overlap %"] = (
        pd.to_numeric(output["weighted_node_overlap_min"], errors="coerce") * 100
    )
    output["Edge overlap %"] = (
        pd.to_numeric(output["weighted_edge_overlap_min"], errors="coerce") * 100
    )
    return output


def _confidence_color(label: object) -> str:
    return {
        "HIGH": "#178a4c",
        "MEDIUM": "#d99000",
        "LOW": "#cf2e2e",
    }.get(str(label), "#64748b")


def _confidence_fill(label: object) -> str:
    return {
        "HIGH": "#e8f6ee",
        "MEDIUM": "#fff4d6",
        "LOW": "#fdeaea",
    }.get(str(label), "#f1f5f9")


def _timeline_tooltip(row: pd.Series) -> str:
    node_overlap = row.get("weighted_node_overlap_min")
    edge_overlap = row.get("weighted_edge_overlap_min")
    return "\n".join(
        [
            f"{row['month_a']} → {row['month_b']}",
            f"RCCI: {_format_score(row.get('rcci_v1'))}",
            f"Confidence: {row.get('confidence_label')}",
            f"Trips: {int(row.get('trips_a', 0)):,} → {int(row.get('trips_b', 0)):,}",
            "",
            "Node statistics",
            f"Shared nodes: {int(row.get('shared_nodes', 0)):,}",
            f"Added nodes: {int(row.get('added_nodes', 0)):,}",
            f"Removed nodes: {int(row.get('removed_nodes', 0)):,}",
            f"Node overlap: {_format_score(float(node_overlap) * 100 if pd.notna(node_overlap) else np.nan)}%",
            "",
            "Edge statistics",
            f"Shared edges: {int(row.get('shared_edges', 0)):,}",
            f"Added edges: {int(row.get('added_edges', 0)):,}",
            f"Removed edges: {int(row.get('removed_edges', 0)):,}",
            f"Edge overlap: {_format_score(float(edge_overlap) * 100 if pd.notna(edge_overlap) else np.nan)}%",
        ]
    )


def _timeline_svg(
    frame: pd.DataFrame,
    *,
    from_dir: Path,
    width: int = 980,
    height: int = 360,
) -> str:
    data = frame.loc[pd.to_numeric(frame["rcci_v1"], errors="coerce").notna()].copy()
    if data.empty:
        return "<p class='empty'>No RCCI values available for the timeline.</p>"
    data = data.sort_values(["month_a", "month_b"])
    scores = data["rcci_v1"].astype(float).to_list()
    labels = (data["month_a"].astype(str) + "→" + data["month_b"].astype(str)).to_list()
    mean_score = float(np.mean(scores))
    left, right, top, bottom = 64, 34, 38, 84
    inner_w = width - left - right
    inner_h = height - top - bottom
    max_index = max(len(scores) - 1, 1)

    def x_at(index: int) -> float:
        return left + inner_w * index / max_index

    def y_at(score: float) -> float:
        return top + inner_h * (100.0 - score) / 100.0

    points = " ".join(
        f"{x_at(index):.1f},{y_at(score):.1f}" for index, score in enumerate(scores)
    )
    bands = []
    band_width = inner_w / max(len(scores), 1)
    for index, row in enumerate(data.itertuples(index=False)):
        center = x_at(index)
        x = max(left, center - band_width / 2)
        width_rect = band_width
        if index == 0:
            width_rect = band_width / 2
        if index == len(scores) - 1:
            width_rect = min(width_rect, width - right - x)
        bands.append(
            f"<rect class='band' x='{x:.1f}' y='{top}' width='{width_rect:.1f}' "
            f"height='{inner_h}' fill='{_confidence_fill(getattr(row, 'confidence_label'))}'></rect>"
        )
    circles = []
    for index, (_, row) in enumerate(data.iterrows()):
        score = float(row["rcci_v1"])
        tooltip = html.escape(_timeline_tooltip(row))
        circle = (
            "<circle class='point' "
            f"cx='{x_at(index):.1f}' cy='{y_at(score):.1f}' r='6' "
            f"fill='{_confidence_color(row.get('confidence_label'))}'>"
            f"<title>{tooltip}</title></circle>"
        )
        href = _comparison_relative_href(row, from_dir=from_dir)
        if href:
            circle = f"<a href='{html.escape(href)}'>{circle}</a>"
        circles.append(circle)
    y_ticks = []
    for tick in range(0, 101, 20):
        y = y_at(float(tick))
        y_ticks.append(
            f"<line x1='{left}' x2='{width-right}' y1='{y:.1f}' y2='{y:.1f}' />"
            f"<text x='{left-10}' y='{y+4:.1f}' text-anchor='end'>{tick}</text>"
        )
    x_labels = []
    step = max(2, math.ceil(len(labels) / 12))
    for index, label in enumerate(labels):
        if index % step == 0 or index == len(labels) - 1:
            x = x_at(index)
            x_labels.append(
                f"<text x='{x:.1f}' y='{height-28}' transform='rotate(-45 {x:.1f},{height-28})'>{html.escape(label)}</text>"
            )
    mean_y = y_at(mean_score)
    high_idx = int(np.argmax(scores))
    low_idx = int(np.argmin(scores))
    annotations = []
    for text, idx, anchor, y_offset in [
        ("Highest change", high_idx, "start", -16),
        ("Lowest change", low_idx, "end", 18),
    ]:
        x = x_at(idx)
        y = y_at(scores[idx])
        text_x = min(max(x + (10 if anchor == "start" else -10), left + 8), width - right - 8)
        annotations.append(
            f"<g class='callout'><line x1='{x:.1f}' y1='{y:.1f}' x2='{text_x:.1f}' y2='{y + y_offset:.1f}'></line>"
            f"<text x='{text_x:.1f}' y='{y + y_offset:.1f}' text-anchor='{anchor}'>"
            f"{html.escape(text)} · {html.escape(labels[idx])} · RCCI {scores[idx]:.1f}</text></g>"
        )
    return f"""
<svg class="timeline" viewBox="0 0 {width} {height}" role="img" aria-label="Broward RCCI timeline">
  <rect x="0" y="0" width="{width}" height="{height}" rx="14"></rect>
  <g class="bands">{''.join(bands)}</g>
  <g class="grid">{''.join(y_ticks)}</g>
  <line class="mean-line" x1="{left}" x2="{width-right}" y1="{mean_y:.1f}" y2="{mean_y:.1f}"></line>
  <text class="mean-label" x="{width-right-4}" y="{mean_y-6:.1f}" text-anchor="end">Mean RCCI {mean_score:.1f}</text>
  <line class="axis" x1="{left}" x2="{width-right}" y1="{top+inner_h}" y2="{top+inner_h}"></line>
  <line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top+inner_h}"></line>
  <polyline class="series" points="{points}"></polyline>
  <g class="points">{''.join(circles)}</g>
  <g>{''.join(annotations)}</g>
  <g class="xlabels">{''.join(x_labels)}</g>
  <text class="ylabel" x="20" y="{top+inner_h/2}" transform="rotate(-90 20,{top+inner_h/2})">RCCI v1</text>
  <text class="xlabel" x="{left+inner_w/2}" y="{height-6}" text-anchor="middle">Month pair</text>
</svg>
"""


def _confidence_counts(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["confidence_label"]
        .value_counts()
        .reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
    )


def _county_trip_coverage(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "0 comparisons"
    trips_a = pd.to_numeric(frame["trips_a"], errors="coerce").fillna(0)
    trips_b = pd.to_numeric(frame["trips_b"], errors="coerce").fillna(0)
    return (
        f"{int(trips_a.sum()):,} month-A trips, "
        f"{int(trips_b.sum()):,} month-B trips; "
        f"median pair trips {_format_number(pd.concat([trips_a, trips_b]).median())}"
    )


def _county_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for county_name, group in frame.groupby("county", sort=True):
        months = set(group["month_a"].astype(str)) | set(group["month_b"].astype(str))
        scores = pd.to_numeric(group["rcci_v1"], errors="coerce").dropna()
        trip_values = pd.concat(
            [
                pd.to_numeric(group["trips_a"], errors="coerce"),
                pd.to_numeric(group["trips_b"], errors="coerce"),
            ],
            ignore_index=True,
        ).dropna()
        counts = _confidence_counts(group)
        rows.append(
            {
                "County": county_name,
                "Observed months": len(months),
                "Comparisons": len(group),
                "Median trips": trip_values.median() if not trip_values.empty else np.nan,
                "Median RCCI": scores.median() if not scores.empty else np.nan,
                "HIGH confidence": int(counts["HIGH"]),
                "MEDIUM confidence": int(counts["MEDIUM"]),
                "LOW confidence": int(counts["LOW"]),
                "Mean RCCI": scores.mean() if not scores.empty else np.nan,
                "Maximum RCCI": scores.max() if not scores.empty else np.nan,
                "Minimum RCCI": scores.min() if not scores.empty else np.nan,
            }
        )
    order = {"Broward County": 0, "Miami-Dade County": 1, "Palm Beach County": 2}
    return (
        pd.DataFrame(rows)
        .sort_values("County", key=lambda column: column.map(order).fillna(99))
        .reset_index(drop=True)
    )


def _county_stat_cards(frame: pd.DataFrame) -> str:
    counts = _confidence_counts(frame)
    scores = pd.to_numeric(frame["rcci_v1"], errors="coerce").dropna()
    node_retention = (
        pd.to_numeric(frame["weighted_node_overlap_min"], errors="coerce").mean() * 100
    )
    edge_retention = (
        pd.to_numeric(frame["weighted_edge_overlap_min"], errors="coerce").mean() * 100
    )
    cards = [
        ("Comparisons", f"{len(frame):,}"),
        ("Mean RCCI", _format_score(scores.mean() if not scores.empty else np.nan)),
        ("Median RCCI", _format_score(scores.median() if not scores.empty else np.nan)),
        ("Minimum RCCI", _format_score(scores.min() if not scores.empty else np.nan)),
        ("Maximum RCCI", _format_score(scores.max() if not scores.empty else np.nan)),
        ("Avg. node retention", f"{_format_score(node_retention)}%"),
        ("Avg. edge retention", f"{_format_score(edge_retention)}%"),
        ("HIGH confidence", f"{int(counts['HIGH']):,}"),
        ("MEDIUM confidence", f"{int(counts['MEDIUM']):,}"),
        ("LOW confidence", f"{int(counts['LOW']):,}"),
    ]
    return "".join(
        f"<div class='card compact'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
        for label, value in cards
    )


def _county_section(
    frame: pd.DataFrame,
    county_name: str,
    *,
    table_columns: Sequence[str],
) -> str:
    group = frame.loc[frame["county"] == county_name].sort_values(
        ["month_a", "month_b"]
    )
    counts = _confidence_counts(group)
    low_share = (counts["LOW"] / len(group)) if len(group) else 0
    sparse_note = ""
    if low_share >= 0.8:
        sparse_note = (
            "<p class='note'>Most comparisons are low confidence because of "
            "insufficient trip coverage.</p>"
        )
    return f"""
<section>
<h3>{html.escape(county_name)}</h3>
<div class="cards">{_county_stat_cards(group)}</div>
<p><strong>Trip coverage:</strong> {html.escape(_county_trip_coverage(group))}</p>
{sparse_note}
<h4>RCCI table</h4>
{_html_table(group, table_columns, raw_html_columns={"Comparison HTML link"})}
</section>
"""


def generate_rcci_report_html(
    rcci_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    output: str | Path,
    *,
    report_county: str = DEFAULT_REPORT_COUNTY,
    node_weight: float = 0.5,
    edge_weight: float = 0.5,
) -> Path:
    """Write the standalone RCCI report HTML."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = rcci_summary.sort_values(["month_a", "month_b", "county"]).copy()
    ordered_links = _add_comparison_links(ordered, from_dir=output.parent)
    broward = ordered_links.loc[ordered_links["county"] == report_county].copy()
    if broward.empty:
        broward = ordered_links.copy()
        report_county = "All county-specific rows"
    supplemental_counties = [
        county
        for county in ["Miami-Dade County", "Palm Beach County"]
        if county in set(ordered_links["county"])
    ]
    county_counts = ordered_links["county"].value_counts()
    broward_hm = broward.loc[
        broward["confidence_label"].isin(["HIGH", "MEDIUM"])
        & broward["rcci_v1"].notna()
    ].copy()
    broward_highest = broward_hm.sort_values("rcci_v1", ascending=False).head(5)
    broward_lowest = broward_hm.sort_values("rcci_v1", ascending=True).head(5)
    broward_counts = _confidence_counts(broward)
    county_summary = _county_summary(ordered_links)
    executive_scores = pd.to_numeric(broward["rcci_v1"], errors="coerce").dropna()
    executive_node_retention = (
        pd.to_numeric(broward["weighted_node_overlap_min"], errors="coerce").mean()
        * 100
    )
    executive_edge_retention = (
        pd.to_numeric(broward["weighted_edge_overlap_min"], errors="coerce").mean()
        * 100
    )
    cards = [
        ("Primary county", report_county),
        ("Total comparisons", f"{len(ordered_links):,}"),
        ("Broward comparisons", f"{int(county_counts.get('Broward County', 0)):,}"),
        (
            "Miami-Dade comparisons",
            f"{int(county_counts.get('Miami-Dade County', 0)):,}",
        ),
        (
            "Palm Beach comparisons",
            f"{int(county_counts.get('Palm Beach County', 0)):,}",
        ),
        ("Mean RCCI", _format_score(executive_scores.mean())),
        ("Median RCCI", _format_score(executive_scores.median())),
        ("Highest RCCI", _format_score(executive_scores.max())),
        ("Lowest RCCI", _format_score(executive_scores.min())),
        ("Avg. node retention", f"{_format_score(executive_node_retention)}%"),
        ("Avg. edge retention", f"{_format_score(executive_edge_retention)}%"),
        ("HIGH/MEDIUM/LOW", f"{int(broward_counts['HIGH'])}/{int(broward_counts['MEDIUM'])}/{int(broward_counts['LOW'])}"),
    ]
    card_html = "".join(
        f"<div class='card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
        for label, value in cards
    )
    table_columns = [
        "month_a",
        "month_b",
        "county",
        "trips_a",
        "trips_b",
        "trip_count_ratio",
        "rcci_v1",
        "confidence_label",
        "confidence_reason",
        "interpretation_label",
    ]
    change_columns = [
        "month_a",
        "month_b",
        "county",
        "trips_a",
        "trips_b",
        "rcci_v1",
        "weighted_node_overlap_min",
        "weighted_edge_overlap_min",
        "interpretation_label",
        "Comparison HTML link",
    ]
    enhanced_change_columns = [
        "month_a",
        "month_b",
        "county",
        "confidence_label",
        "trips_a",
        "trips_b",
        "rcci_v1",
        "Node overlap %",
        "Edge overlap %",
        "Comparison HTML link",
    ]
    county_summary_columns = [
        "County",
        "Observed months",
        "Comparisons",
        "Median trips",
        "Median RCCI",
        "HIGH confidence",
        "MEDIUM confidence",
        "LOW confidence",
        "Mean RCCI",
        "Maximum RCCI",
        "Minimum RCCI",
    ]
    supplemental_html = "".join(
        _county_section(ordered_links, county, table_columns=table_columns)
        for county in supplemental_counties
    )
    links = {
        "Driver timeline": _share_link(
            "timeline/driver_1003_timeline.html",
            from_dir=output.parent,
        ),
        "Monthly graph overview": _share_link(
            "monthly_graphs/driver_1003_monthly_graph_overview.html",
            from_dir=output.parent,
        ),
        "Graph comparison overview": _share_link(
            "graph_comparisons/driver_1003_graph_comparison_overview.html",
            from_dir=output.parent,
        ),
        "Broward 2023-08 to 2023-09 comparison": _share_link(
            "graph_comparisons/driver_1003_broward_2023-08_to_2023-09_comparison.html",
            from_dir=output.parent,
        ),
    }
    link_html = "".join(
        f"<li><a href='{html.escape(href)}'>{html.escape(label)}</a></li>"
        for label, href in links.items()
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Driver 1003 Route Choice Change Index (RCCI)</title>
<style>
:root{{--bg:#f6f8fb;--card:#ffffff;--text:#182230;--muted:#617085;--blue:#2f6fed;--line:#dbe3ef;--green:#0f8f61;--orange:#b65c00;--red:#b42318}}
html{{scroll-behavior:smooth}} body{{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45}}
	main{{max-width:1380px;margin:0 auto;padding:34px 22px 56px}}
h1{{font-size:34px;margin:0 0 6px}} h2{{margin-top:34px}} p{{color:var(--muted)}}
.nav{{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0 24px}} .nav a{{background:#fff;border:1px solid var(--line);border-radius:999px;padding:8px 12px;text-decoration:none;font-size:13px}}
.subtitle{{font-size:17px;margin-top:0}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:24px 0}}
.card,.box,section{{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:0 8px 24px rgba(24,34,48,.06)}}
.card{{padding:16px}} .card.compact strong{{font-size:21px}} .card span{{display:block;color:var(--muted);font-size:13px}} .card strong{{font-size:25px}}
	section{{padding:22px;margin:22px 0}} .formula{{font-size:20px;color:var(--text);background:#f0f5ff;border-left:5px solid var(--blue);padding:16px;border-radius:12px}}
	.equation-block{{display:grid;gap:12px;margin:16px 0}} .equation{{background:#f0f5ff;border-left:5px solid var(--blue);border-radius:12px;padding:14px 16px;color:var(--text);font-size:18px;line-height:1.55}} .equation small{{display:block;color:var(--muted);font-size:13px;margin-top:4px}} .var-list{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:14px 0}} .var-list div{{background:#f8fbff;border:1px solid var(--line);border-radius:12px;padding:12px}} .tradeoff-table td:nth-child(3){{color:#0f5132}} .tradeoff-table td:nth-child(4){{color:#7c2d12}}
.note{{background:#fff8e6;border-left:5px solid var(--orange);padding:14px;border-radius:12px;color:#5b3b00}}
.disclaimer{{background:#fff1f0;border-left:5px solid var(--red);padding:14px;border-radius:12px;color:#6b1d15}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:12px 0 8px;color:var(--muted);font-size:13px}} .dot{{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px;vertical-align:-1px}} .high{{background:var(--green)}} .medium{{background:var(--orange)}} .low{{background:var(--red)}}
.scale{{display:flex;align-items:center;gap:12px;margin:12px 0 18px;color:var(--muted);font-size:13px}} .scale-bar{{height:10px;flex:1;border-radius:999px;background:linear-gradient(90deg,#d7efe1,#fff0bd,#ffd6d6)}} .muted{{color:var(--muted)}}
.two-col{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}} .mini-box{{background:#f8fbff;border:1px solid var(--line);border-radius:14px;padding:16px}} .mini-box h4{{margin:0 0 8px}} .mini-box ul{{margin:8px 0 0 20px;padding:0}} .example{{background:#f7fbf9;border-left:5px solid var(--green);border-radius:12px;padding:16px;margin-top:14px}} .formula-list{{display:grid;gap:8px;margin:12px 0 16px}}
	.table-wrap{{overflow-x:auto}} table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}} th{{background:#edf3fb;color:#324052;position:sticky;top:0}} tr:nth-child(even){{background:#fafcff}}
	.timeline{{width:100%;height:auto;background:#fff}} .timeline>rect{{fill:#fff}} .band{{opacity:.75}} .grid line{{stroke:#e2e8f0}} .grid text,.xlabels text{{fill:#617085;font-size:11px}} .axis{{stroke:#7b8aa0;stroke-width:1.2}} .series{{fill:none;stroke:#4b6b98;stroke-width:2.5}} .points circle{{stroke:#fff;stroke-width:2.2;cursor:pointer}} .points a:hover circle{{stroke:#111827;stroke-width:3}} .xlabel,.ylabel{{fill:#334155;font-size:13px;font-weight:600}} .mean-line{{stroke:#111827;stroke-dasharray:6 5;stroke-width:1.5;opacity:.65}} .mean-label,.callout text{{fill:#1f2937;font-size:12px;font-weight:700}} .callout line{{stroke:#334155;stroke-width:1;opacity:.7}}
a{{color:var(--blue)}} .empty{{font-style:italic}}
</style>
</head>
<body>
<main>
<h1>Driver 1003 Route Choice Change Index (RCCI)</h1>
<p class="subtitle">A month-to-month route-network change index based on FID road-segment usage and directed transition patterns.</p>
		<nav class="nav">
		<a href="#executive-summary">RCCI Technical Snapshot</a>
	<a href="#how-calculated">How RCCI is calculated</a>
<a href="#timeline">Timeline</a>
<a href="#highest-rcci">Highest RCCI</a>
<a href="#lowest-rcci">Lowest RCCI</a>
<a href="#county-summary">County Summary</a>
<a href="#supplemental-counties">Supplemental Counties</a>
<a href="#full-comparison-table">Full Comparison Table</a>
</nav>

		<!-- DRIVER 1003 REAL WORLD BEHAVIOR INSIGHTS PLACEHOLDER -->

<section id="executive-summary">
<h2>RCCI technical snapshot</h2>
<div class="cards">{card_html}</div>
<p><strong>Primary longitudinal county:</strong> {html.escape(report_county)}.</p>
	<p>{html.escape(report_county)} is emphasized because it has the highest trip counts, strongest month coverage, highest-confidence comparisons, and the most complete longitudinal route history for Driver 1003. This is a data-quality decision, not a geographic preference.</p>
		</section>

	<section id="how-calculated">
	<h2>How RCCI is calculated</h2>
	<p>Each month is represented as a route graph. In the RCCI computation, nodes are road segments/FIDs used during a month, and edges are directed transitions from one FID to the next. RCCI compares two consecutive monthly graphs and measures how much road usage and transition usage changed.</p>
	<div class="var-list">
	  <div><strong>G<sub>t</sub></strong> = route graph for month <em>t</em></div>
	  <div><strong>V<sub>t</sub></strong> = set of FID nodes used in month <em>t</em></div>
	  <div><strong>E<sub>t</sub></strong> = set of directed FID transitions in month <em>t</em></div>
	  <div><strong>w<sub>v</sub><sup>t</sup></strong> = trip-use count for node <em>v</em></div>
	  <div><strong>w<sub>e</sub><sup>t</sup></strong> = transition count for edge <em>e</em></div>
	  <div><strong>N<sub>c</sub>, E<sub>c</sub>, R<sub>c</sub></strong> = node change, edge change, and RCCI</div>
	</div>
	<h3>Node and edge change definitions</h3>
	<div class="equation-block">
	  <div class="equation"><strong>N<sub>c,J</sub>(t,t+1)</strong> = 1 − |V<sub>t</sub> ∩ V<sub>t+1</sub>| / |V<sub>t</sub> ∪ V<sub>t+1</sub>|<small>Unweighted node-set Jaccard distance.</small></div>
	  <div class="equation"><strong>E<sub>c,J</sub>(t,t+1)</strong> = 1 − |E<sub>t</sub> ∩ E<sub>t+1</sub>| / |E<sub>t</sub> ∪ E<sub>t+1</sub>|<small>Unweighted edge-set Jaccard distance.</small></div>
	  <div class="equation"><strong>O<sub>V,w</sub>(t,t+1)</strong> = Σ<sub>v ∈ V<sub>t</sub> ∪ V<sub>t+1</sub></sub> min(w<sub>v</sub><sup>t</sup>, w<sub>v</sub><sup>t+1</sup>) / Σ<sub>v ∈ V<sub>t</sub> ∪ V<sub>t+1</sub></sub> max(w<sub>v</sub><sup>t</sup>, w<sub>v</sub><sup>t+1</sup>)<small>Weighted node overlap.</small></div>
	  <div class="equation"><strong>O<sub>E,w</sub>(t,t+1)</strong> = Σ<sub>e ∈ E<sub>t</sub> ∪ E<sub>t+1</sub></sub> min(w<sub>e</sub><sup>t</sup>, w<sub>e</sub><sup>t+1</sup>) / Σ<sub>e ∈ E<sub>t</sub> ∪ E<sub>t+1</sub></sub> max(w<sub>e</sub><sup>t</sup>, w<sub>e</sub><sup>t+1</sup>)<small>Weighted edge overlap.</small></div>
	  <div class="equation"><strong>N<sub>c,w</sub>(t,t+1)</strong> = 1 − O<sub>V,w</sub>(t,t+1), &nbsp; <strong>E<sub>c,w</sub>(t,t+1)</strong> = 1 − O<sub>E,w</sub>(t,t+1)<small>Weighted node and edge change used for RCCI v1.</small></div>
	</div>
	<p><strong>Weighted overlap</strong> means frequently used roads count more than rarely used roads. If a road segment was used in 40 trips, a change involving that road should matter more than a road segment used once. RCCI therefore uses trip-use counts and transition counts rather than treating every FID equally.</p>
	<h3>RCCI formulas tested</h3>
	<div class="equation-block">
	  <div class="equation"><strong>Node-only:</strong> R<sub>c</sub>(t,t+1) = N<sub>c</sub>(t,t+1)</div>
	  <div class="equation"><strong>Edge-only:</strong> R<sub>c</sub>(t,t+1) = E<sub>c</sub>(t,t+1)</div>
	  <div class="equation"><strong>Equal node-edge:</strong> R<sub>c</sub>(t,t+1) = 0.5N<sub>c</sub>(t,t+1) + 0.5E<sub>c</sub>(t,t+1)</div>
	  <div class="equation"><strong>General weighted:</strong> R<sub>c,α</sub>(t,t+1) = αN<sub>c</sub>(t,t+1) + (1 − α)E<sub>c</sub>(t,t+1)<small>Tested α ∈ {{0.00, 0.25, 0.50, 0.75, 1.00}}. α = 0 is edge-only; α = 1 is node-only.</small></div>
	  <div class="equation"><strong>Reported RCCI v1:</strong> RCCI<sub>v1</sub> = 100 × [0.5N<sub>c,w</sub>(t,t+1) + 0.5E<sub>c,w</sub>(t,t+1)]<small>The 0-100 scale makes month-to-month route change easier to interpret.</small></div>
	</div>
	<h3>Formula tradeoffs</h3>
	<div class="table-wrap"><table class="tradeoff-table">
	<thead><tr><th>Formula</th><th>What it emphasizes</th><th>Pros</th><th>Cons</th></tr></thead>
	<tbody>
	<tr><td>Node-only R<sub>c</sub> = N<sub>c</sub></td><td>Which road segments/FIDs appear or disappear.</td><td>Simple and easy to explain; robust when transition ordering is noisy.</td><td>Ignores how roads are connected and cannot detect route-sequence changes if the same FIDs remain.</td></tr>
	<tr><td>Edge-only R<sub>c</sub> = E<sub>c</sub></td><td>Directed movement from one FID to the next.</td><td>Captures path structure and changes in route connectivity.</td><td>More sensitive to sparse trips, short paths, and map-matching transition noise.</td></tr>
	<tr><td>Equal node-edge R<sub>c</sub> = 0.5N<sub>c</sub> + 0.5E<sub>c</sub></td><td>Balanced road usage and transition change.</td><td>Good default for presentation because it combines structural road changes with sequence changes.</td><td>Assumes node and edge changes are equally important for every driver and month.</td></tr>
	<tr><td>General weighted R<sub>c,α</sub></td><td>Adjustable emphasis on nodes versus edges.</td><td>Supports sensitivity testing; α shows whether conclusions depend on formula choice.</td><td>Requires a defensible α choice if used as the final metric.</td></tr>
	<tr><td>Unweighted Jaccard versions</td><td>Presence/absence of nodes or edges.</td><td>Transparent structural baseline.</td><td>Treats a one-time segment the same as a heavily used segment.</td></tr>
	<tr><td>Weighted overlap versions</td><td>Frequency-weighted route behavior.</td><td>Best aligned with this project because repeated driving patterns matter more than rare observations.</td><td>Can downweight rare but real route changes and depends on month-to-month trip volume quality.</td></tr>
	</tbody>
	</table></div>
<div class="example">
<h3>Worked node example</h3>
<div class="table-wrap"><table>
<thead><tr><th>FID</th><th>Month A usage</th><th>Month B usage</th><th>min</th><th>max</th></tr></thead>
<tbody>
<tr><td>100</td><td>40</td><td>38</td><td>38</td><td>40</td></tr>
<tr><td>200</td><td>10</td><td>11</td><td>10</td><td>11</td></tr>
<tr><td>300</td><td>2</td><td>0</td><td>0</td><td>2</td></tr>
<tr><td>400</td><td>0</td><td>3</td><td>0</td><td>3</td></tr>
</tbody>
</table></div>
<p><strong>weighted overlap</strong> = (38 + 10 + 0 + 0) / (40 + 11 + 2 + 3) = 48 / 56 = 0.857</p>
<p><strong>node change</strong> = 1 − 0.857 = 0.143</p>
<p>Although two roads changed, most high-use driving remained stable, so the node change is relatively small. If FID 100 disappeared, the RCCI contribution would be much larger.</p>
</div>
<div class="two-col">
<div class="mini-box">
<h4>Node component</h4>
<p>Captures changes in which road segments were used.</p>
</div>
<div class="mini-box">
<h4>Edge component</h4>
<p>Captures changes in how the driver moved between road segments. Two months may use many of the same roads but connect them differently. That is why edge changes are included.</p>
</div>
</div>
<div class="two-col">
<div class="mini-box">
<h4>What counts more?</h4>
<ul>
<li>frequently used roads</li>
<li>frequently used transitions</li>
<li>roads/transitions that disappear after heavy use</li>
<li>new roads/transitions that appear repeatedly</li>
</ul>
</div>
<div class="mini-box">
<h4>What counts less?</h4>
<ul>
<li>one-time roads</li>
<li>rare transitions</li>
<li>sparse months, which are flagged by confidence labels</li>
</ul>
</div>
</div>
<p class="note">RCCI measures route change. Confidence measures whether there are enough trips to trust the comparison. These are intentionally reported separately and are not mixed together.</p>
</section>

<section id="timeline">
<h2>Primary Longitudinal Analysis<br>{html.escape(report_county)}</h2>
<p>This primary longitudinal analysis focuses on {html.escape(report_county)} because it is the most suitable county for studying Driver 1003's route-choice behavior over time.</p>
<p class="note">Broward is highlighted because it contains the highest longitudinal trip coverage and the largest number of HIGH-confidence comparisons. This is a data-quality decision, not a geographic preference.</p>
<div class="cards">{_county_stat_cards(broward)}</div>
<div class="legend"><span><i class="dot high"></i>HIGH confidence</span><span><i class="dot medium"></i>MEDIUM confidence</span><span><i class="dot low"></i>LOW confidence</span><span>Subtle vertical shading uses the same confidence colors.</span></div>
<div class="scale"><span>0<br>Very stable</span><span class="scale-bar"></span><span>100<br>Major route change</span></div>
<p class="muted">Higher RCCI indicates greater route-choice change between consecutive months. Click a timeline marker to open the corresponding county-specific comparison page when available.</p>
{_timeline_svg(broward, from_dir=output.parent)}

<h3 id="highest-rcci">Highest Broward RCCI periods</h3>
<p>Top HIGH/MEDIUM confidence Broward rows by RCCI.</p>
{_html_table(broward_highest, enhanced_change_columns, limit=5, raw_html_columns={"Comparison HTML link"})}

<h3 id="lowest-rcci">Lowest Broward RCCI periods</h3>
<p>Lowest HIGH/MEDIUM confidence Broward rows by RCCI.</p>
{_html_table(broward_lowest, enhanced_change_columns, limit=5, raw_html_columns={"Comparison HTML link"})}

<h3>Broward confidence summary</h3>
<p>HIGH: {int(broward_counts['HIGH']):,}; MEDIUM: {int(broward_counts['MEDIUM']):,}; LOW: {int(broward_counts['LOW']):,}.</p>

<h3 id="full-comparison-table">Broward metric table</h3>
{_html_table(broward.sort_values(["month_a", "month_b"]), table_columns, raw_html_columns={"Comparison HTML link"})}
</section>

<section>
<h2>What RCCI means</h2>
<p>RCCI measures how much Driver 1003's route network changed between consecutive months. It combines changes in road-segment usage with changes in directed FID transition patterns. A value near 0 means little route-network change; a value near 100 means very large route-network change.</p>
<p class="disclaimer">This index summarizes route-network change for Driver 1003. It does not explain why the change occurred and should not be interpreted as a clinical measure.</p>
</section>

<section>
<h2>RCCI v1 formula</h2>
<div class="formula">RCCI = 100 × [{node_weight:.2f} × node change + {edge_weight:.2f} × edge change]</div>
<p>Node change = 1 - weighted node overlap. Edge change = 1 - weighted edge overlap. The reported weights are normalized before scoring.</p>
</section>

<section>
<h2>Driver 1003 calibration note</h2>
<p class="note">Interpretation bands are calibrated to Driver 1003's high-coverage Broward County comparisons. They are not universal thresholds for other drivers or datasets.</p>
</section>

<section id="county-summary">
<h2>County summary</h2>
<p>This table preserves all county-specific RCCI results while making the primary Broward coverage clear.</p>
{_html_table(county_summary, county_summary_columns)}
</section>

<section id="supplemental-counties">
<h2>Supplemental County Results</h2>
<p>Miami-Dade and Palm Beach are preserved for completeness. They are supplemental because trip coverage is sparse relative to Broward County.</p>
{supplemental_html}
</section>

<section>
<h2>Related Phase 2 deliverables</h2>
<ul>{link_html}</ul>
</section>

<section>
<h2>Technical details</h2>
<p>Generated at {_generated_at()}. Sensitivity columns were also exported: rcci_balanced_weighted, rcci_edge_heavy_weighted, rcci_balanced_jaccard, and rcci_geometric_weighted.</p>
</section>
</main>
</body>
</html>"""
    rendered = embed_local_html_assets(document, output.parent)
    rendered = "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n"
    output.write_text(rendered, encoding="utf-8")
    return output


def validate_rcci_outputs(
    rcci_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    *,
    original_summary: pd.DataFrame,
    output_paths: Iterable[Path] = (),
) -> dict[str, object]:
    """Validate RCCI row counts, score bounds, and output path existence."""
    errors: list[str] = []
    original_county_rows = int((original_summary["county"] != ALL_COUNTIES).sum())
    original_combined_rows = int((original_summary["county"] == ALL_COUNTIES).sum())
    if len(rcci_summary) != original_county_rows:
        errors.append(
            f"RCCI summary row count {len(rcci_summary)} != county-specific input rows {original_county_rows}"
        )
    if len(sensitivity) != len(rcci_summary):
        errors.append("Sensitivity row count does not match RCCI summary row count")

    scores = pd.to_numeric(rcci_summary["rcci_v1"], errors="coerce").dropna()
    if not scores.between(0, 100).all():
        errors.append("RCCI scores contain values outside [0, 100]")
    required_labels = {"HIGH", "MEDIUM", "LOW"}
    unknown_labels = set(rcci_summary["confidence_label"].dropna().unique()) - required_labels
    if unknown_labels:
        errors.append(f"Unknown confidence labels: {sorted(unknown_labels)}")
    missing_paths = [str(path) for path in output_paths if not Path(path).exists()]
    if missing_paths:
        errors.append(f"Output files missing: {missing_paths}")

    high_medium = rcci_summary.loc[
        rcci_summary["confidence_label"].isin(["HIGH", "MEDIUM"])
        & rcci_summary["rcci_v1"].notna()
    ]
    top = (
        rcci_summary.loc[rcci_summary["rcci_v1"].notna()]
        .sort_values("rcci_v1", ascending=False)
        .head(1)
    )
    lowest_hm = high_medium.sort_values("rcci_v1", ascending=True).head(1)
    return {
        "validation_passed": not errors,
        "errors": errors,
        "rows_processed": int(len(rcci_summary)),
        "county_specific_rows": original_county_rows,
        "all_counties_rows_excluded": original_combined_rows,
        "confidence_counts": rcci_summary["confidence_label"].value_counts().to_dict(),
        "rcci_min": float(scores.min()) if not scores.empty else np.nan,
        "rcci_median": float(scores.median()) if not scores.empty else np.nan,
        "rcci_mean": float(scores.mean()) if not scores.empty else np.nan,
        "rcci_max": float(scores.max()) if not scores.empty else np.nan,
        "top_rcci_month_pair": _row_label(top.iloc[0]) if not top.empty else "None",
        "lowest_high_medium_month_pair": _row_label(lowest_hm.iloc[0])
        if not lowest_hm.empty
        else "None",
    }


def _row_label(row: pd.Series) -> str:
    return (
        f"{row['county']} {row['month_a']}→{row['month_b']} "
        f"(RCCI {_format_score(row['rcci_v1'])})"
    )


def write_validation_report(
    validation: dict[str, object],
    output: str | Path,
) -> Path:
    """Write Markdown validation report."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    confidence = validation.get("confidence_counts", {})
    errors = validation.get("errors", [])
    lines = [
        "# Driver 1003 RCCI Validation Report",
        "",
        f"Generated: {_generated_at()}",
        "",
        f"- Rows processed: {validation.get('rows_processed'):,}",
        f"- County-specific rows: {validation.get('county_specific_rows'):,}",
        f"- ALL_COUNTIES rows excluded from primary scoring: {validation.get('all_counties_rows_excluded'):,}",
        f"- HIGH confidence rows: {int(confidence.get('HIGH', 0)):,}",
        f"- MEDIUM confidence rows: {int(confidence.get('MEDIUM', 0)):,}",
        f"- LOW confidence rows: {int(confidence.get('LOW', 0)):,}",
        f"- Minimum RCCI: {_format_score(validation.get('rcci_min'))}",
        f"- Median RCCI: {_format_score(validation.get('rcci_median'))}",
        f"- Mean RCCI: {_format_score(validation.get('rcci_mean'))}",
        f"- Maximum RCCI: {_format_score(validation.get('rcci_max'))}",
        f"- Top RCCI month pair: {validation.get('top_rcci_month_pair')}",
        f"- Lowest HIGH/MEDIUM RCCI month pair: {validation.get('lowest_high_medium_month_pair')}",
        f"- Scores bounded 0-100: {'yes' if not errors else 'no'}",
        f"- Output files written successfully: {'yes' if not errors else 'no'}",
        f"- Validation passed: {bool(validation.get('validation_passed'))}",
        "",
    ]
    if errors:
        lines.append("## Errors")
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("No validation errors were detected.")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def build_driver_1003_rcci(
    *,
    driver: str = DEFAULT_DRIVER_ID,
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    node_weight: float = 0.5,
    edge_weight: float = 0.5,
    county: str | None = None,
    include_all_counties: bool = False,
    report_county: str = DEFAULT_REPORT_COUNTY,
) -> RCCIResult:
    """End-to-end RCCI builder for Driver 1003."""
    if str(driver) != DEFAULT_DRIVER_ID:
        raise DriverTimelineError("RCCI v1 is currently scoped to Driver 1003")
    output_root = Path(output_dir)
    data_dir = output_root / "data"
    visuals_dir = output_root / "visuals"
    data_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    original_summary, node_details, edge_details = load_comparison_outputs(input_dir)
    del node_details, edge_details  # loaded to verify expected Phase 2C inputs exist
    rcci_summary = build_rcci_summary(
        original_summary,
        node_weight=node_weight,
        edge_weight=edge_weight,
        county=county,
        include_all_counties=include_all_counties,
    )
    sensitivity = build_sensitivity_table(rcci_summary)

    summary_csv = data_dir / "driver_1003_rcci_summary.csv"
    summary_parquet = data_dir / "driver_1003_rcci_summary.parquet"
    sensitivity_csv = data_dir / "driver_1003_rcci_sensitivity.csv"
    sensitivity_parquet = data_dir / "driver_1003_rcci_sensitivity.parquet"
    report_html = visuals_dir / "driver_1003_route_choice_change_index_report.html"
    validation_report = output_root / "driver_1003_rcci_validation.md"

    rcci_summary.to_csv(summary_csv, index=False)
    rcci_summary.to_parquet(summary_parquet, index=False)
    sensitivity.to_csv(sensitivity_csv, index=False)
    sensitivity.to_parquet(sensitivity_parquet, index=False)
    normalized_node_weight, normalized_edge_weight = normalize_weights(
        node_weight,
        edge_weight,
    )
    generate_rcci_report_html(
        rcci_summary,
        sensitivity,
        report_html,
        report_county=report_county,
        node_weight=normalized_node_weight,
        edge_weight=normalized_edge_weight,
    )
    paths = [
        summary_csv,
        summary_parquet,
        sensitivity_csv,
        sensitivity_parquet,
        report_html,
    ]
    validation = validate_rcci_outputs(
        rcci_summary,
        sensitivity,
        original_summary=original_summary,
        output_paths=paths,
    )
    write_validation_report(validation, validation_report)
    return RCCIResult(
        summary_csv=summary_csv,
        summary_parquet=summary_parquet,
        sensitivity_csv=sensitivity_csv,
        sensitivity_parquet=sensitivity_parquet,
        report_html=report_html,
        validation_report=validation_report,
        rows=int(len(rcci_summary)),
        confidence_counts={
            key: int(value) for key, value in validation["confidence_counts"].items()
        },
        validation_passed=bool(validation["validation_passed"]),
    )
