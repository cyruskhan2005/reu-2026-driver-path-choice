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
) -> str:
    data = frame.head(limit).copy() if limit else frame.copy()
    if data.empty:
        return "<p class='empty'>No rows.</p>"
    headers = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    rows = []
    for row in data.reindex(columns=columns).itertuples(index=False, name=None):
        cells = "".join(f"<td>{_format_number(value)}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def _timeline_svg(frame: pd.DataFrame, *, width: int = 980, height: int = 320) -> str:
    data = frame.loc[pd.to_numeric(frame["rcci_v1"], errors="coerce").notna()].copy()
    if data.empty:
        return "<p class='empty'>No RCCI values available for the timeline.</p>"
    data = data.sort_values(["month_a", "month_b"])
    scores = data["rcci_v1"].astype(float).to_list()
    labels = (data["month_a"].astype(str) + "→" + data["month_b"].astype(str)).to_list()
    left, right, top, bottom = 64, 24, 26, 70
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
    circles = []
    for index, (score, label) in enumerate(zip(scores, labels, strict=False)):
        circles.append(
            "<circle "
            f"cx='{x_at(index):.1f}' cy='{y_at(score):.1f}' r='4'>"
            f"<title>{html.escape(label)}: {score:.1f}</title></circle>"
        )
    y_ticks = []
    for tick in range(0, 101, 20):
        y = y_at(float(tick))
        y_ticks.append(
            f"<line x1='{left}' x2='{width-right}' y1='{y:.1f}' y2='{y:.1f}' />"
            f"<text x='{left-10}' y='{y+4:.1f}' text-anchor='end'>{tick}</text>"
        )
    x_labels = []
    step = max(1, math.ceil(len(labels) / 10))
    for index, label in enumerate(labels):
        if index % step == 0 or index == len(labels) - 1:
            x = x_at(index)
            x_labels.append(
                f"<text x='{x:.1f}' y='{height-28}' transform='rotate(-45 {x:.1f},{height-28})'>{html.escape(label)}</text>"
            )
    return f"""
<svg class="timeline" viewBox="0 0 {width} {height}" role="img" aria-label="Broward RCCI timeline">
  <rect x="0" y="0" width="{width}" height="{height}" rx="14"></rect>
  <g class="grid">{''.join(y_ticks)}</g>
  <line class="axis" x1="{left}" x2="{width-right}" y1="{top+inner_h}" y2="{top+inner_h}"></line>
  <line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top+inner_h}"></line>
  <polyline class="series" points="{points}"></polyline>
  <g class="points">{''.join(circles)}</g>
  <g class="xlabels">{''.join(x_labels)}</g>
  <text class="ylabel" x="20" y="{top+inner_h/2}" transform="rotate(-90 20,{top+inner_h/2})">RCCI v1</text>
  <text class="xlabel" x="{left+inner_w/2}" y="{height-6}" text-anchor="middle">Month pair</text>
</svg>
"""


def _relative_link(path: str, *, from_dir: Path) -> str:
    return html.escape(os.path.relpath(Path(path), from_dir))


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
    focus = rcci_summary.loc[rcci_summary["county"] == report_county].copy()
    if focus.empty:
        focus = rcci_summary.copy()
        report_county = "All county-specific rows"
    high_medium = rcci_summary.loc[
        rcci_summary["confidence_label"].isin(["HIGH", "MEDIUM"])
        & rcci_summary["rcci_v1"].notna()
    ].copy()
    highest = high_medium.sort_values("rcci_v1", ascending=False).head(5)
    lowest = high_medium.sort_values("rcci_v1", ascending=True).head(5)
    low_conf = rcci_summary.loc[rcci_summary["confidence_label"] == "LOW"].copy()
    confidence_counts = (
        rcci_summary["confidence_label"]
        .value_counts()
        .reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
    )
    score_values = pd.to_numeric(rcci_summary["rcci_v1"], errors="coerce").dropna()
    cards = [
        ("County-specific rows", f"{len(rcci_summary):,}"),
        ("Broward focus rows", f"{len(focus):,}"),
        ("Median RCCI", _format_score(score_values.median() if not score_values.empty else np.nan)),
        ("HIGH confidence", f"{int(confidence_counts['HIGH']):,}"),
        ("MEDIUM confidence", f"{int(confidence_counts['MEDIUM']):,}"),
        ("LOW confidence", f"{int(confidence_counts['LOW']):,}"),
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
    ]
    links = {
        "Driver timeline": "../../../driver_1003/timeline/driver_1003_timeline.html",
        "Monthly graph overview": "../../../driver_1003/monthly_graphs/driver_1003_monthly_graph_overview.html",
        "Graph comparison overview": "../../../driver_1003/graph_comparisons/driver_1003_graph_comparison_overview.html",
        "Broward 2023-08 to 2023-09 comparison": "../../../driver_1003/graph_comparisons/driver_1003_broward_2023-08_to_2023-09_comparison.html",
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
body{{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45}}
main{{max-width:1180px;margin:0 auto;padding:34px 22px 56px}}
h1{{font-size:34px;margin:0 0 6px}} h2{{margin-top:34px}} p{{color:var(--muted)}}
.subtitle{{font-size:17px;margin-top:0}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:24px 0}}
.card,.box,section{{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:0 8px 24px rgba(24,34,48,.06)}}
.card{{padding:16px}} .card span{{display:block;color:var(--muted);font-size:13px}} .card strong{{font-size:25px}}
section{{padding:22px;margin:22px 0}} .formula{{font-size:20px;color:var(--text);background:#f0f5ff;border-left:5px solid var(--blue);padding:16px;border-radius:12px}}
.note{{background:#fff8e6;border-left:5px solid var(--orange);padding:14px;border-radius:12px;color:#5b3b00}}
.disclaimer{{background:#fff1f0;border-left:5px solid var(--red);padding:14px;border-radius:12px;color:#6b1d15}}
.table-wrap{{overflow-x:auto}} table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}} th{{background:#edf3fb;color:#324052;position:sticky;top:0}} tr:nth-child(even){{background:#fafcff}}
.timeline{{width:100%;height:auto;background:#fff}} .timeline rect{{fill:#fff}} .grid line{{stroke:#edf1f7}} .grid text,.xlabels text{{fill:#617085;font-size:11px}} .axis{{stroke:#7b8aa0;stroke-width:1.2}} .series{{fill:none;stroke:var(--blue);stroke-width:3}} .points circle{{fill:var(--blue);stroke:#fff;stroke-width:2}} .xlabel,.ylabel{{fill:#334155;font-size:13px;font-weight:600}}
a{{color:var(--blue)}} .empty{{font-style:italic}}
</style>
</head>
<body>
<main>
<h1>Driver 1003 Route Choice Change Index (RCCI)</h1>
<p class="subtitle">A month-to-month route-network change index based on FID road-segment usage and directed transition patterns.</p>
<div class="cards">{card_html}</div>

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

<section>
<h2>RCCI timeline: {html.escape(report_county)}</h2>
{_timeline_svg(focus)}
</section>

<section>
<h2>Highest RCCI periods</h2>
<p>Top HIGH/MEDIUM confidence rows by RCCI.</p>
{_html_table(highest, change_columns, limit=5)}
</section>

<section>
<h2>Lowest RCCI periods</h2>
<p>Lowest HIGH/MEDIUM confidence rows by RCCI.</p>
{_html_table(lowest, change_columns, limit=5)}
</section>

<section>
<h2>Low-confidence rows</h2>
<p>Sparse rows are retained for transparency but should be interpreted with trip-count context.</p>
{_html_table(low_conf.sort_values(["month_a", "month_b", "county"]), table_columns)}
</section>

<section>
<h2>Full RCCI metric table</h2>
{_html_table(rcci_summary.sort_values(["month_a", "month_b", "county"]), table_columns)}
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
    output.write_text(
        embed_local_html_assets(document, output.parent),
        encoding="utf-8",
    )
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
