#!/usr/bin/env python3
"""Convert matched routes into graph tables and adjacency matrices."""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from research_process_utils import (
    FID_CANDIDATES,
    OUTPUT_ROOT,
    PATH_CANDIDATES,
    county_from_path,
    detect_column,
    ensure_dir,
    explode_segment_observations,
    find_matched_files,
    load_network_edges,
    parse_segment_sequence,
    read_table,
    safe_period_label,
    segment_trip_counts,
    write_html,
)


def build_edge_table(observations: pd.DataFrame) -> pd.DataFrame:
    counts = segment_trip_counts(observations, by_period=True)
    if counts.empty:
        return counts
    counts = counts.rename(columns={"period": "month", "trip_use_count": "edge_weight"})
    extra_cols = [col for col in ("driver_id", "driver_label") if col in observations.columns]
    if extra_cols:
        meta = observations.dropna(subset=["fid"]).drop_duplicates(["county", "period", "fid"])[["county", "period", "fid"] + extra_cols]
        meta = meta.rename(columns={"period": "month"})
        counts = counts.merge(meta, on=["county", "month", "fid"], how="left")
    return counts


def add_node_endpoints(edges: pd.DataFrame) -> pd.DataFrame:
    if {"u", "v"}.issubset(edges.columns):
        return edges
    pieces: list[pd.DataFrame] = []
    for county, group in edges.groupby("county", dropna=False):
        try:
            network = load_network_edges(str(county), group["fid"].unique(), include_geometry=True)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: missing network nodes for {county}: {exc}")
            continue
        keep = ["fid", "u", "v", "geometry"]
        merged = group.merge(network[[col for col in keep if col in network.columns]], on="fid", how="left")
        if "u" not in merged.columns or "v" not in merged.columns or merged[["u", "v"]].isna().all().all():
            # Fallback: derive stable node labels from rounded segment endpoints.
            starts, ends = [], []
            for geom in merged["geometry"]:
                if geom is None or geom.is_empty:
                    starts.append(None)
                    ends.append(None)
                    continue
                line = list((geom.geoms[0] if geom.geom_type == "MultiLineString" else geom).coords)
                starts.append(f"endpoint_{round(line[0][0], 1)}_{round(line[0][1], 1)}")
                ends.append(f"endpoint_{round(line[-1][0], 1)}_{round(line[-1][1], 1)}")
            merged["u"] = starts
            merged["v"] = ends
            merged["node_source"] = "endpoint-derived"
        else:
            merged["node_source"] = "network_node_id"
        pieces.append(merged.drop(columns=["geometry"], errors="ignore"))
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def build_node_table(edge_table: pd.DataFrame) -> pd.DataFrame:
    if edge_table.empty:
        return pd.DataFrame()
    endpoints = edge_table if {"u", "v"}.issubset(edge_table.columns) else add_node_endpoints(edge_table)
    if endpoints.empty:
        return pd.DataFrame()
    rows = []
    meta_cols = [col for col in ("driver_id", "driver_label") if col in endpoints.columns]
    for node_col in ("u", "v"):
        temp = endpoints[["county", "month", "fid", "edge_weight", node_col, "node_source"] + meta_cols].rename(columns={node_col: "node_id"})
        rows.append(temp)
    nodes = pd.concat(rows, ignore_index=True).dropna(subset=["node_id"])
    group_cols = ["county", "month", "node_id", "node_source"] + meta_cols
    return (
        nodes.groupby(group_cols, dropna=False)["edge_weight"]
        .sum()
        .rename("node_weight")
        .reset_index()
    )


def write_adjacency_matrices(edge_table: pd.DataFrame, output_dir: Path, max_nodes: int) -> int:
    endpoints = edge_table if {"u", "v"}.issubset(edge_table.columns) else add_node_endpoints(edge_table)
    if endpoints.empty:
        return 0
    written = 0
    for (county, month), group in endpoints.dropna(subset=["u", "v"]).groupby(["county", "month"], dropna=False):
        node_weights = pd.concat(
            [
                group[["u", "edge_weight"]].rename(columns={"u": "node"}),
                group[["v", "edge_weight"]].rename(columns={"v": "node"}),
            ],
            ignore_index=True,
        )
        top_nodes = set(node_weights.groupby("node")["edge_weight"].sum().sort_values(ascending=False).head(max_nodes).index)
        scoped = group.loc[group["u"].isin(top_nodes) & group["v"].isin(top_nodes)]
        matrix = scoped.pivot_table(index="u", columns="v", values="edge_weight", aggfunc="sum", fill_value=0)
        matrix.index.name = "from_node"
        path = output_dir / f"adjacency_matrix_{safe_period_label(county)}_{safe_period_label(month)}.csv"
        matrix.to_csv(path)
        written += 1
    return written


def load_matched_csv_fid_counts() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in find_matched_files():
        if "sflorida_outputs" not in path.parts:
            continue
        try:
            data = read_table(path)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not read matched CSV {path}: {exc}")
            continue
        route_col = detect_column(data.columns, PATH_CANDIDATES)
        fid_col = detect_column(data.columns, FID_CANDIDATES)
        county = county_from_path(path)
        month_col = detect_column(data.columns, ("month", "trip_month"))
        if route_col:
            for record in data.itertuples(index=False):
                as_dict = record._asdict()
                period = as_dict.get(month_col) if month_col else "overall"
                for fid in parse_segment_sequence(as_dict.get(route_col)):
                    rows.append({"county": county, "period": period or "overall", "fid": fid})
        elif fid_col:
            for record in data.itertuples(index=False):
                as_dict = record._asdict()
                period = as_dict.get(month_col) if month_col else "overall"
                fid = as_dict.get(fid_col)
                if pd.notna(fid):
                    rows.append({"county": county, "period": period or "overall", "fid": int(fid)})
        else:
            print(f"WARNING: no matched route/FID column found in {path}.")
    if not rows:
        return pd.DataFrame(columns=["county", "period", "fid", "edge_pass_count"])
    observed = pd.DataFrame(rows)
    return observed.groupby(["county", "period", "fid"], dropna=False).size().rename("edge_pass_count").reset_index()


def load_fmm_edges(county: str, fids: set[int]) -> gpd.GeoDataFrame:
    path = Path("sflorida_outputs") / county.replace("-", "_").replace(" ", "_") / "fmm" / "edges.shp"
    if path.exists():
        edges = gpd.read_file(path)
        edges = edges.loc[edges["fid"].astype(int).isin(fids)].copy()
        edges["node_source"] = "fmm_edges_node_id" if {"u", "v"}.issubset(edges.columns) else "endpoint-derived"
    else:
        edges = load_network_edges(county, fids, include_geometry=True)
        edges["node_source"] = "enriched_network_node_id" if {"u", "v"}.issubset(edges.columns) else "endpoint-derived"
    if not {"u", "v"}.issubset(edges.columns):
        starts, ends = [], []
        for geom in edges.geometry:
            if geom is None or geom.is_empty:
                starts.append(None)
                ends.append(None)
                continue
            line = geom.geoms[0] if geom.geom_type == "MultiLineString" else geom
            coords = list(line.coords)
            starts.append(f"endpoint_{round(coords[0][0], 6)}_{round(coords[0][1], 6)}")
            ends.append(f"endpoint_{round(coords[-1][0], 6)}_{round(coords[-1][1], 6)}")
        edges["u"] = starts
        edges["v"] = ends
    return edges[["fid", "u", "v", "node_source", "geometry"]]


def build_overall_node_matrix_outputs(output_dir: Path, top_nodes: int) -> None:
    ensure_dir(output_dir)
    counts = load_matched_csv_fid_counts()
    if counts.empty:
        print("WARNING: no matched CSV FID counts available for node matrix visualization.")
        return
    pieces: list[pd.DataFrame] = []
    for county, group in counts.groupby("county", dropna=False):
        fids = set(pd.to_numeric(group["fid"], errors="coerce").dropna().astype(int))
        if not fids:
            continue
        try:
            edges = load_fmm_edges(str(county), fids)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not load FMM edge geometry for {county}: {exc}")
            continue
        merged = group.merge(edges.drop(columns=["geometry"], errors="ignore"), on="fid", how="inner")
        pieces.append(merged)
    if not pieces:
        print("WARNING: no node-edge table could be built.")
        return
    node_edge = pd.concat(pieces, ignore_index=True).rename(columns={"u": "from_node", "v": "to_node"})
    node_edge.to_csv(output_dir / "node_edge_table.csv", index=False)
    weighted = (
        node_edge.groupby(["county", "period", "from_node", "to_node", "node_source"], dropna=False)["edge_pass_count"]
        .sum()
        .rename("weight")
        .reset_index()
    )
    binary = weighted.copy()
    binary["binary"] = 1
    weighted.to_csv(output_dir / "adjacency_matrix_weighted.csv", index=False)
    binary[["county", "period", "from_node", "to_node", "node_source", "binary"]].to_csv(
        output_dir / "adjacency_matrix_binary.csv",
        index=False,
    )
    weighted.to_csv(output_dir / "adjacency_matrix_overall.csv", index=False)
    save_matrix_heatmap(
        binary.rename(columns={"binary": "value"}),
        output_dir / "adjacency_matrix_binary_heatmap.png",
        top_nodes,
        title="Binary Node Adjacency Matrix After Map Matching",
        colorbar_label="Connection exists (0/1)",
        caption="This 0/1 matrix shows whether a connection between two road-network nodes appeared in the matched routes. Top nodes shown for readability.",
        cmap="Greys",
    )
    save_matrix_heatmap(
        weighted.rename(columns={"weight": "value"}),
        output_dir / "adjacency_matrix_weighted_heatmap.png",
        top_nodes,
        title="Weighted Node Adjacency Matrix After Map Matching",
        colorbar_label="log(1 + matched segment passes)",
        caption="This weighted matrix shows how often each node-to-node connection appeared. Larger values mean the route used that connection more often. Top nodes shown for readability.",
        cmap="viridis",
        log_scale=True,
    )
    save_matrix_heatmap(
        weighted.rename(columns={"weight": "value"}),
        output_dir / "adjacency_matrix_heatmap.png",
        top_nodes,
        title="Node Adjacency Matrix After Map Matching",
        colorbar_label="log(1 + matched segment passes)",
        caption="Rows and columns are road-network nodes. Cell intensity shows how often matched routes traveled between node pairs. Top nodes shown for readability.",
        cmap="viridis",
        log_scale=True,
    )
    save_presentation_matrices(weighted, output_dir, max_nodes=12)
    save_matrix_heatmap_html(output_dir / "adjacency_matrix_heatmap.html")


def save_matrix_heatmap(
    sparse: pd.DataFrame,
    output: Path,
    top_nodes: int,
    *,
    title: str,
    colorbar_label: str,
    caption: str,
    cmap: str,
    log_scale: bool = False,
) -> None:
    if sparse.empty:
        return
    node_weights = pd.concat(
        [
            sparse[["from_node", "value"]].rename(columns={"from_node": "node"}),
            sparse[["to_node", "value"]].rename(columns={"to_node": "node"}),
        ],
        ignore_index=True,
    )
    top = node_weights.groupby("node")["value"].sum().sort_values(ascending=False).head(top_nodes).index.astype(str).tolist()
    top_set = set(top)
    scoped = sparse.loc[sparse["from_node"].astype(str).isin(top_set) & sparse["to_node"].astype(str).isin(top_set)].copy()
    scoped["from_node"] = scoped["from_node"].astype(str)
    scoped["to_node"] = scoped["to_node"].astype(str)
    matrix = scoped.pivot_table(index="from_node", columns="to_node", values="value", aggfunc="sum", fill_value=0)
    matrix = matrix.reindex(index=top, columns=top, fill_value=0)
    values = matrix.to_numpy(dtype=float)
    if log_scale:
        values = np.log1p(values)
    fig, ax = plt.subplots(figsize=(10.5, 9), dpi=170)
    im = ax.imshow(values, cmap=cmap, interpolation="nearest", vmin=0)
    ax.set_title(f"{title}\nTop {len(top)} nodes shown for readability", fontsize=15, weight="bold")
    ax.set_xlabel("To node")
    ax.set_ylabel("From node")
    label_step = max(len(top) // 10, 1)
    ticks = np.arange(0, len(top), label_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([top[i] for i in ticks], rotation=60, ha="right", fontsize=6)
    ax.set_yticklabels([top[i] for i in ticks], fontsize=6)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label(colorbar_label)
    fig.text(
        0.5,
        0.02,
        caption,
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output)
    plt.close(fig)


def save_matrix_heatmap_html(output: Path) -> None:
    body = """
<h1>Node Adjacency Matrix After Map Matching</h1>
<div class="panel">
  <h2>Binary 0/1 Matrix</h2>
  <p>This 0/1 matrix shows whether a connection between two road-network nodes appeared in the matched routes. Top nodes are shown for readability.</p>
  <img src="adjacency_matrix_binary_heatmap.png" alt="Binary node adjacency matrix heatmap">
  <h2>Weighted Count Matrix</h2>
  <p>This weighted matrix shows how often each node-to-node connection appeared. Larger values mean the route used that connection more often. Top nodes are shown for readability.</p>
  <img src="adjacency_matrix_weighted_heatmap.png" alt="Weighted node adjacency matrix heatmap">
  <p>The CSV files preserve the full sparse binary and weighted adjacency matrices.</p>
</div>
"""
    write_html(output, "Node Adjacency Matrix After Map Matching", body)


def save_presentation_matrices(weighted: pd.DataFrame, output_dir: Path, max_nodes: int = 12) -> None:
    if weighted.empty:
        return
    scope = weighted.sort_values("weight", ascending=False).copy()
    selected_nodes: list[str] = []
    selected_edges: list[pd.Series] = []
    for _, edge in scope.iterrows():
        edge_nodes = [str(edge["from_node"]), str(edge["to_node"])]
        if not selected_nodes:
            selected_nodes.extend(edge_nodes)
            selected_edges.append(edge)
        elif any(node in selected_nodes for node in edge_nodes):
            selected_edges.append(edge)
            for node in edge_nodes:
                if node not in selected_nodes and len(selected_nodes) < max_nodes:
                    selected_nodes.append(node)
        if len(selected_nodes) >= max_nodes and len(selected_edges) >= max_nodes:
            break
    if len(selected_nodes) < 2:
        return
    selected_set = set(selected_nodes[:max_nodes])
    example = weighted.loc[
        weighted["from_node"].astype(str).isin(selected_set) & weighted["to_node"].astype(str).isin(selected_set)
    ].copy()
    example["from_node"] = example["from_node"].astype(str)
    example["to_node"] = example["to_node"].astype(str)
    node_order = selected_nodes[:max_nodes]
    mapping = pd.DataFrame(
        {
            "presentation_node": [f"N{i + 1}" for i in range(len(node_order))],
            "road_network_node_id": node_order,
        }
    )
    mapping.to_csv(output_dir / "presentation_node_mapping.csv", index=False)
    label_map = dict(zip(mapping["road_network_node_id"], mapping["presentation_node"]))
    example["from_label"] = example["from_node"].map(label_map)
    example["to_label"] = example["to_node"].map(label_map)
    labels = mapping["presentation_node"].tolist()
    weighted_matrix = example.pivot_table(index="from_label", columns="to_label", values="weight", aggfunc="sum", fill_value=0)
    weighted_matrix = weighted_matrix.reindex(index=labels, columns=labels, fill_value=0)
    binary_matrix = (weighted_matrix > 0).astype(int)
    draw_presentation_matrix(
        binary_matrix,
        output_dir / "presentation_adjacency_matrix_binary.png",
        title="Example Node Adjacency Matrix After Map Matching",
        caption="1 means the matched route traveled between two nodes. 0 means no connection in this example.",
        cmap="Blues",
        value_format="{:.0f}",
    )
    draw_presentation_matrix(
        weighted_matrix,
        output_dir / "presentation_adjacency_matrix_weighted.png",
        title="Example Weighted Node Matrix",
        caption="Cell values are matched segment pass counts between presentation nodes.",
        cmap="YlOrRd",
        value_format="{:.0f}",
    )


def draw_presentation_matrix(matrix: pd.DataFrame, output: Path, title: str, caption: str, cmap: str, value_format: str) -> None:
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=180)
    im = ax.imshow(values, cmap=cmap, interpolation="nearest", vmin=0)
    ax.set_title(title, fontsize=15, weight="bold", pad=12)
    ax.set_xlabel("To node")
    ax.set_ylabel("From node")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, fontsize=10)
    ax.set_yticklabels(matrix.index, fontsize=10)
    threshold = values.max() / 2 if values.size and values.max() else 0.5
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            ax.text(
                j,
                i,
                value_format.format(value),
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if value > threshold else "#111827",
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("Matrix value")
    fig.text(0.5, 0.025, caption, ha="center", fontsize=9.5)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "graph_matrices")
    parser.add_argument("--matrix-output-dir", type=Path, default=OUTPUT_ROOT / "graph_matrix")
    parser.add_argument("--max-matrix-nodes", type=int, default=100)
    parser.add_argument("--heatmap-top-nodes", type=int, default=75)
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    observations = explode_segment_observations(prefer_timeline=True)
    if observations.empty:
        print("WARNING: no matched segment observations were found.")
        return
    edge_table = build_edge_table(observations)
    edge_table = add_node_endpoints(edge_table)
    edge_table.to_csv(out_dir / "graph_edges_by_period.csv", index=False)
    node_table = build_node_table(edge_table)
    node_table.to_csv(out_dir / "graph_nodes_by_period.csv", index=False)
    matrix_count = write_adjacency_matrices(edge_table, out_dir, args.max_matrix_nodes)
    build_overall_node_matrix_outputs(ensure_dir(args.matrix_output_dir), args.heatmap_top_nodes)
    print(f"Graph edge rows: {len(edge_table):,}")
    print(f"Graph node rows: {len(node_table):,}")
    print(f"Adjacency matrices written: {matrix_count:,}")
    print(f"Wrote {out_dir.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
