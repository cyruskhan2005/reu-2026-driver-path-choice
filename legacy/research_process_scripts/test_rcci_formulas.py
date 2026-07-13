#!/usr/bin/env python3
"""Test multiple RCCI formulas from graph node and edge usage tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from research_process_utils import OUTPUT_ROOT, dataframe_preview_html, ensure_dir, write_html


ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]


def unweighted_change(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 0.0
    return 1.0 - (len(a & b) / len(union))


def weighted_change(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    overlap = sum(min(float(a.get(k, 0)), float(b.get(k, 0))) for k in keys)
    union = sum(max(float(a.get(k, 0)), float(b.get(k, 0))) for k in keys)
    return 0.0 if union == 0 else 1.0 - overlap / union


def compute_tests(edges: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scope_cols = ["county"] + [col for col in ("driver_id", "driver_label") if col in edges.columns and col in nodes.columns]
    for scope, edge_scope in edges.groupby(scope_cols, dropna=False):
        scope_values = scope if isinstance(scope, tuple) else (scope,)
        scope_meta = dict(zip(scope_cols, scope_values))
        node_scope = nodes.copy()
        for col, value in scope_meta.items():
            node_scope = node_scope.loc[node_scope[col].astype(str) == str(value)]
        months = sorted(set(edge_scope["month"].dropna().astype(str)) | set(node_scope["month"].dropna().astype(str)))
        for month_a, month_b in zip(months, months[1:]):
            edge_a = edge_scope.loc[edge_scope["month"].astype(str) == month_a]
            edge_b = edge_scope.loc[edge_scope["month"].astype(str) == month_b]
            node_a = node_scope.loc[node_scope["month"].astype(str) == month_a]
            node_b = node_scope.loc[node_scope["month"].astype(str) == month_b]
            e_unweighted = unweighted_change(set(edge_a["fid"].astype(str)), set(edge_b["fid"].astype(str)))
            n_unweighted = unweighted_change(set(node_a["node_id"].astype(str)), set(node_b["node_id"].astype(str)))
            e_weighted = weighted_change(dict(zip(edge_a["fid"].astype(str), edge_a["edge_weight"])), dict(zip(edge_b["fid"].astype(str), edge_b["edge_weight"])))
            n_weighted = weighted_change(dict(zip(node_a["node_id"].astype(str), node_a["node_weight"])), dict(zip(node_b["node_id"].astype(str), node_b["node_weight"])))
            for alpha in ALPHAS:
                rows.append(
                    {
                        **scope_meta,
                        "month_a": month_a,
                        "month_b": month_b,
                        "alpha": alpha,
                        "N_c": n_unweighted,
                        "E_c": e_unweighted,
                        "R_c": alpha * n_unweighted + (1 - alpha) * e_unweighted,
                        "N_c_weighted": n_weighted,
                        "E_c_weighted": e_weighted,
                        "R_c_weighted": alpha * n_weighted + (1 - alpha) * e_weighted,
                        "variant": "weighted" if alpha not in (0.0, 1.0) else "edge_only" if alpha == 0.0 else "node_only",
                    }
                )
    return pd.DataFrame(rows)


def formula_label(alpha: float) -> str:
    if alpha == 0:
        return "Edge-only"
    if alpha == 0.5:
        return "50/50 node-edge"
    if alpha == 1:
        return "Node-only"
    return f"α={alpha:.2f}"


def save_average_plot(results: pd.DataFrame, output: Path) -> None:
    if results.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=160)
    plot_data = results.copy()
    plot_data["month"] = plot_data["month_b"].astype(str)
    plot_data["_month_sort"] = pd.to_datetime(plot_data["month"], errors="coerce")
    summary = plot_data.groupby(["month", "alpha"], as_index=False).agg(R_c=("R_c", "mean"), _month_sort=("_month_sort", "min"))
    summary = summary.sort_values(["_month_sort", "month", "alpha"])
    month_order = summary.drop_duplicates("month")["month"].tolist()
    for alpha, group in summary.groupby("alpha"):
        ordered = group.set_index("month").reindex(month_order).reset_index()
        ax.plot(ordered["month"], ordered["R_c"], marker="o", linewidth=2.5, label=formula_label(float(alpha)))
    ax.set_title("Average RCCI by Month", fontsize=16, weight="bold")
    ax.set_ylabel("Average Route Choice Change Index")
    ax.set_xlabel("Month")
    tick_step = max(len(month_order) // 12, 1)
    ticks = list(range(0, len(month_order), tick_step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([month_order[i] for i in ticks], rotation=45, ha="right")
    ax.grid(color="#e5e7eb")
    ax.legend(title="Formula")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def save_distribution_plot(results: pd.DataFrame, output: Path) -> None:
    if results.empty:
        return
    plot_data = results.loc[results["alpha"].isin([0.0, 0.5, 1.0])].copy()
    order = [0.0, 0.5, 1.0]
    labels = [formula_label(alpha) for alpha in order]
    values = [plot_data.loc[plot_data["alpha"].eq(alpha), "R_c"].dropna().to_numpy() for alpha in order]
    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=160)
    boxplot_kwargs = {"patch_artist": True, "widths": 0.55}
    try:
        box = ax.boxplot(values, tick_labels=labels, **boxplot_kwargs)
    except TypeError:
        box = ax.boxplot(values, labels=labels, **boxplot_kwargs)
    colors = ["#f97316", "#2563eb", "#10b981"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(2)
    for median in box["medians"]:
        median.set_color("#111827")
        median.set_linewidth(2)
    ax.set_title("RCCI Distribution by Formula Type", fontsize=16, weight="bold")
    ax.set_ylabel("Route Choice Change Index")
    ax.grid(axis="y", color="#e5e7eb")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def save_plot(results: pd.DataFrame, output: Path) -> None:
    save_average_plot(results, output)


def save_report(results: pd.DataFrame, output: Path) -> None:
    equations = r"""
<script>
window.MathJax = { tex: { inlineMath: [['\\(', '\\)'], ['$', '$']] } };
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""
    body = """
<h1>RCCI Formula Testing</h1>
<div class="panel">
<p>These tests compare node-only, edge-only, equal node-edge, and general weighted RCCI formulas.</p>
<p>\\[N_c(t,t+1)=1-\\frac{|V_t \\cap V_{t+1}|}{|V_t \\cup V_{t+1}|}\\]</p>
<p>\\[E_c(t,t+1)=1-\\frac{|E_t \\cap E_{t+1}|}{|E_t \\cup E_{t+1}|}\\]</p>
<p>\\[R_c(t,t+1)=\\alpha N_c(t,t+1)+(1-\\alpha)E_c(t,t+1)\\]</p>
</div>
<div class="panel"><img src="rcci_average_by_month.png" alt="Average RCCI by month pair"></div>
<div class="panel"><img src="rcci_distribution_by_formula.png" alt="RCCI distribution by formula type"></div>
<div class="panel">
<h2>Result Preview</h2>
""" + dataframe_preview_html(output.parent / "rcci_formula_tests.csv", rows=12) + "</div>"
    write_html(output, "RCCI Formula Testing", body, extra_head=equations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OUTPUT_ROOT / "graph_matrices")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "rcci_formula_testing")
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    edge_path = args.input_dir / "graph_edges_by_period.csv"
    node_path = args.input_dir / "graph_nodes_by_period.csv"
    if not edge_path.exists() or not node_path.exists():
        raise FileNotFoundError("Run scripts/build_route_graph_matrices.py before RCCI formula testing.")
    edges = pd.read_csv(edge_path)
    nodes = pd.read_csv(node_path)
    results = compute_tests(edges, nodes)
    results.to_csv(out_dir / "rcci_formula_tests.csv", index=False)
    if "county" in results.columns:
        results.to_csv(out_dir / "rcci_formula_tests_by_county.csv", index=False)
    if "driver_id" in edges.columns:
        results.to_csv(out_dir / "rcci_formula_tests_by_driver.csv", index=False)
    save_average_plot(results, out_dir / "rcci_average_by_month.png")
    save_distribution_plot(results, out_dir / "rcci_distribution_by_formula.png")
    save_plot(results, out_dir / "rcci_formula_comparison.png")
    save_report(results, out_dir / "rcci_formula_comparison.html")
    print(f"RCCI formula rows: {len(results):,}")
    print(f"Wrote {out_dir.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
