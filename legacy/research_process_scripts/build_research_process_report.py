#!/usr/bin/env python3
"""Build the presentation-ready research process HTML report."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from research_process_utils import OUTPUT_ROOT, dataframe_preview_html, ensure_dir, write_html


def metric_value(path: Path, metric: str) -> str:
    if not path.exists():
        return "missing"
    data = pd.read_csv(path)
    row = data.loc[data["metric"] == metric] if "metric" in data.columns else pd.DataFrame()
    if row.empty:
        return "n/a"
    return f"{int(row.iloc[0]['value']):,}"


def rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix() if path.exists() else path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "research_process_report.html")
    args = parser.parse_args()

    base = ensure_dir(OUTPUT_ROOT)
    raw_dir = base / "raw_gps"
    monthly_dir = base / "monthly_summary"
    segment_dir = base / "segment_usage"
    gif_dir = base / "segment_usage_gif"
    map_matching_dir = base / "map_matching_gif"
    graph_matrix_dir = base / "graph_matrix"
    graph_dir = base / "graph_matrices"
    rcci_dir = base / "rcci_formula_testing"

    total_points = metric_value(raw_dir / "raw_gps_summary.csv", "total_raw_gps_points")
    equations = r"""
<script>
window.MathJax = { tex: { inlineMath: [['\\(', '\\)'], ['$', '$']] } };
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""
    body = f"""
<h1>Driver Path-Choice Research Process</h1>
<p class="note">Raw GPS data → map matching → graph construction → formula exploration → final RCCI interpretation.</p>

<section class="panel">
<h2>1. Raw GPS Data Before Map Matching</h2>
<div class="metric-grid"><div class="metric"><span>Full raw GPS point count</span><strong>{total_points}</strong></div></div>
<p>This figure intentionally plots every raw GPS point before map matching. The overlap is not an error; it shows the scale and density of the trajectory dataset.</p>
<p>This raw trajectory dataset is the input to the Fast Map Matching algorithm, which converts noisy GPS observations into connected road-network paths.</p>
<img src="{rel(raw_dir / 'raw_gps_presentation_scale.png', base)}" alt="Full raw GPS presentation scale plot">
<h3>Density Raster View</h3>
<p class="note">This secondary raster also uses all raw GPS points.</p>
<img src="{rel(raw_dir / 'raw_gps_all_points_density.png', base)}" alt="Full raw GPS point density">
<p><a href="{rel(raw_dir / 'raw_gps_all_points.html', base)}">Open full raw GPS HTML view</a></p>
<h3>Monthly raw GPS counts</h3>
<img src="{rel(monthly_dir / 'monthly_raw_gps_histogram.png', base)}" alt="Monthly raw GPS counts">
</section>

<section class="panel">
<h2>2. Map Matching Process</h2>
<p>FMM does not immediately snap each GPS point to the nearest road. For each GPS point, it first identifies nearby candidate road segments. The candidates are scored using FMM likelihood when available, or normalized distance/error scores when likelihood is unavailable. FMM then selects the most likely connected route through the road network.</p>
<p class="note">For every GPS observation, FMM identifies nearby candidate road segments. The candidates are scored individually, and the highest-scoring connected sequence becomes the final matched route.</p>
<img src="{rel(map_matching_dir / 'map_matching_process.gif', base)}" alt="Map matching process GIF">
<h3>Three-panel summary</h3>
<img src="{rel(map_matching_dir / 'map_matching_process_three_panel_presentation.png', base)}" alt="Map matching process three-panel PNG">
<p><a href="{rel(map_matching_dir / 'fmm_candidates_sample.csv', base)}">Open candidate segment CSV</a></p>
</section>

<section class="panel">
<h2>3. Graph Representation</h2>
<p>After map matching, the matched route is converted into a graph. Nodes represent road-network intersections or segment endpoints, and edges represent traveled road segments.</p>
<img src="{rel(graph_matrix_dir / 'graph_representation_example.png', base)}" alt="Graph representation of matched route">
</section>

<section class="panel">
<h2>4. Graph → Matrix Conversion</h2>
<p>The graph can be written as an adjacency matrix. Rows and columns are nodes; each matrix cell records whether a node-to-node connection exists or how often it was used.</p>
<img src="{rel(graph_matrix_dir / 'graph_to_matrix_example.png', base)}" alt="Graph to adjacency matrix conversion">
<p class="note">The simplified example above illustrates how the graph representation is converted into an adjacency matrix. The full matrices below are generated automatically from every matched route.</p>
</section>

<section class="panel">
<h2>5. Binary Matrix</h2>
<p>A binary matrix captures route structure. A cell value of 1 means that a node-to-node connection appeared in the matched routes; 0 means it did not.</p>
<img src="{rel(graph_matrix_dir / 'adjacency_matrix_binary_heatmap.png', base)}" alt="Actual binary node adjacency matrix">
</section>

<section class="panel">
<h2>6. Weighted Matrix</h2>
<p>A weighted matrix captures route frequency. Larger values mean the matched routes used that node-to-node connection more often.</p>
<img src="{rel(graph_matrix_dir / 'adjacency_matrix_weighted_heatmap.png', base)}" alt="Actual weighted node adjacency matrix">
<p class="note">A 0/1 matrix only shows whether a node-to-node connection exists. A weighted matrix is more useful for frequency-based route-choice analysis because it keeps track of how often roads were used.</p>
<h3>Edge frequency table</h3>
{dataframe_preview_html(graph_dir / 'graph_edges_by_period.csv', rows=8)}
<h3>Node frequency table</h3>
{dataframe_preview_html(graph_dir / 'graph_nodes_by_period.csv', rows=8)}
</section>

<section class="panel">
<h2>7. Node and Edge Change</h2>
<p>Once graphs have been converted into adjacency matrices, route changes can be quantified mathematically.</p>
<p><strong>Node Change (N_c)</strong> compares which graph nodes appear across two time periods.</p>
<p><strong>Edge Change (E_c)</strong> compares which traveled road segments or node-to-node edges appear across two time periods.</p>
<p><strong>Route Choice Change Index (R_c)</strong> combines node and edge changes into one metric.</p>
</section>

<section class="panel">
<h2>8. RCCI Formula Comparison</h2>
<p>\\[N_c(t,t+1)=1-\\frac{{|V_t \\cap V_{{t+1}}|}}{{|V_t \\cup V_{{t+1}}|}}\\]</p>
<p>\\[E_c(t,t+1)=1-\\frac{{|E_t \\cap E_{{t+1}}|}}{{|E_t \\cup E_{{t+1}}|}}\\]</p>
<p>\\[R_c(t,t+1)=N_c(t,t+1)\\]</p>
<p>\\[R_c(t,t+1)=E_c(t,t+1)\\]</p>
<p>\\[R_c(t,t+1)=0.5N_c(t,t+1)+0.5E_c(t,t+1)\\]</p>
<p>\\[R_c(t,t+1)=\\alpha N_c(t,t+1)+(1-\\alpha)E_c(t,t+1)\\]</p>
<p><strong>N_c</strong> means node change. <strong>E_c</strong> means edge change. <strong>R_c</strong> is the Route Choice Change Index. <strong>α</strong> controls how much the formula emphasizes node change versus edge change.</p>
<p>Node and edge changes can be computed from these graph representations. The binary matrix supports structural change, while the weighted matrix supports frequency-based route change.</p>
<p>The purpose of testing multiple RCCI formulas is to compare whether route change is better explained by node changes, edge changes, or a weighted combination of both.</p>
<h3>Average RCCI by month</h3>
<img src="{rel(rcci_dir / 'rcci_average_by_month.png', base)}" alt="Average RCCI by month pair">
<h3>RCCI distribution by formula</h3>
<img src="{rel(rcci_dir / 'rcci_distribution_by_formula.png', base)}" alt="RCCI distribution by formula type">
{dataframe_preview_html(rcci_dir / 'rcci_formula_tests.csv', rows=10)}
</section>

<section class="panel">
<h2>9. Matched Segment Usage Appendix</h2>
<p>Darker/thicker segments represent more trips passing through that segment. These maps support interpretation after the graph and RCCI workflow is understood.</p>
<iframe src="{rel(segment_dir / 'segment_usage_overall.html', base)}"></iframe>
<img src="{rel(monthly_dir / 'monthly_segment_pass_histogram.png', base)}" alt="Monthly segment usage histogram">
<h3>Segment usage over time</h3>
<img src="{rel(gif_dir / 'segment_usage_by_month.gif', base)}" alt="Segment usage by month GIF">
<h3>Candidate segment debug table</h3>
<p class="note">The presentation view above keeps the candidate table out of the main flow. This preview is included for reproducibility and debugging.</p>
{dataframe_preview_html(map_matching_dir / 'fmm_candidates_sample.csv', rows=8)}
</section>

<section class="panel">
<h2>10. Final Interpretation</h2>
<p>The purpose of these steps is to make the research process visible: raw GPS observations become map-matched routes, routes become graph and matrix representations, and those graph objects support RCCI formula testing before final route-choice change results are interpreted.</p>
</section>
"""
    write_html(args.output, "Driver Path-Choice Research Process", body, extra_head=equations)
    print(f"Report: {args.output.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
