#!/usr/bin/env python3
"""Create a presentation GIF explaining raw GPS -> candidates -> matched route."""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
from shapely.geometry import LineString, Point, box  # noqa: E402
from shapely.ops import nearest_points  # noqa: E402

from research_process_utils import (
    FID_CANDIDATES,
    OUTPUT_ROOT,
    PATH_CANDIDATES,
    county_dir_name,
    detect_column,
    ensure_dir,
    find_matched_files,
    iter_raw_gps,
    parse_segment_sequence,
    read_table,
)


def candidate_value(candidate: object, *names: str):
    for name in names:
        if hasattr(candidate, name):
            return getattr(candidate, name)
    return None


def select_trajectory(
    county: str,
    min_points: int,
    max_points: int,
    edges: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    chunks = []
    for chunk in iter_raw_gps(chunksize=200_000):
        scoped = chunk.loc[chunk["county"].astype(str).eq(county)].copy()
        if not scoped.empty:
            chunks.append(scoped)
        if sum(len(item) for item in chunks) >= 900_000:
            break
    raw = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if raw.empty:
        raise ValueError(f"No raw GPS points found for {county}.")
    if "source_id" not in raw.columns:
        return raw.head(max_points).reset_index(drop=True)
    target_points = min(max(max_points, min_points), 12)
    route_diversity = matched_route_diversity(county)
    group_cols = (["source_file"] if "source_file" in raw.columns else []) + ["source_id"]
    rows = []
    for key, group in raw.groupby(group_cols, dropna=False, sort=False):
        if len(group) < min_points:
            continue
        group = group.sort_values("timestamp") if "timestamp" in group.columns else group
        window_size = min(target_points, len(group), max_points)
        if window_size < min_points:
            continue
        step = max(1, (len(group) - window_size) // 18)
        for start in range(0, len(group) - window_size + 1, step):
            window = group.iloc[start : start + window_size]
            lon_span = float(window["lon"].max() - window["lon"].min())
            lat_span = float(window["lat"].max() - window["lat"].min())
            span = math.hypot(lon_span, lat_span)
            displacement = math.hypot(
                float(window["lon"].iloc[-1] - window["lon"].iloc[0]),
                float(window["lat"].iloc[-1] - window["lat"].iloc[0]),
            )
            lon_steps = window["lon"].diff().abs().fillna(0)
            lat_steps = window["lat"].diff().abs().fillna(0)
            max_step = float((lon_steps.pow(2) + lat_steps.pow(2)).pow(0.5).max())
            if span < 0.0010 or span > 0.014 or displacement < 0.0008 or max_step > 0.0018:
                continue
            source_id = str(window["source_id"].iloc[0])
            diversity = 0
            if not route_diversity.empty:
                match = route_diversity.loc[route_diversity["source_id"].astype(str).eq(source_id)]
                diversity = int(match["unique_fids"].max()) if not match.empty else 0
            score = abs(span - 0.0045) + abs(len(window) - 12) * 0.0005 - min(diversity, 40) * 0.00003
            rows.append({"score": score, "source_id": source_id, "start": start, "window_size": window_size, "key": key})
    def choice_window(choice: dict[str, object]) -> pd.DataFrame:
        mask = pd.Series(True, index=raw.index)
        key_values = choice["key"] if isinstance(choice["key"], tuple) else (choice["key"],)
        for col, value in zip(group_cols, key_values):
            mask &= raw[col].astype(str).eq(str(value))
        traj = raw.loc[mask].copy()
        traj = traj.sort_values("timestamp") if "timestamp" in traj.columns else traj
        return traj.iloc[int(choice["start"]) : int(choice["start"]) + int(choice["window_size"])].reset_index(drop=True)

    if rows:
        ranked_rows = sorted(rows, key=lambda item: item["score"])
        if edges is not None and not edges.empty:
            scored_choices = []
            for choice in ranked_rows[:35]:
                window = choice_window(choice)
                try:
                    near = add_candidate_scores(
                        nearest_candidate_rows(
                            window,
                            edges,
                            "trajectory selection nearest-edge approximation",
                        )
                    )
                    display = display_candidate_rows(near)
                except Exception:  # noqa: BLE001
                    scored_choices.append((float(choice["score"]) + 10.0, choice))
                    continue
                per_point = display.groupby("point_index").size() if not display.empty else pd.Series(dtype=int)
                points_with_alternatives = int((per_point >= 2).sum())
                unique_candidate_edges = int(
                    pd.to_numeric(display.get("candidate_edge_id"), errors="coerce").dropna().nunique()
                )
                span = math.hypot(
                    float(window["lon"].max() - window["lon"].min()),
                    float(window["lat"].max() - window["lat"].min()),
                )
                separability_bonus = min(points_with_alternatives, 8) * 0.00045
                edge_bonus = min(unique_candidate_edges, 25) * 0.00004
                overly_linear_penalty = 0.006 if unique_candidate_edges < max(3, len(window) // 3) else 0.0
                zoom_penalty = max(0.0, span - 0.018) * 0.5
                scored_choices.append(
                    (
                        float(choice["score"])
                        + overly_linear_penalty
                        + zoom_penalty
                        - separability_bonus
                        - edge_bonus,
                        choice,
                    )
                )
            choice = min(scored_choices, key=lambda item: item[0])[1] if scored_choices else ranked_rows[0]
        else:
            choice = ranked_rows[0]
        return choice_window(choice)

    sizes = raw.groupby(group_cols, dropna=False).size()
    source_key = sizes.loc[sizes >= min_points].sub((min_points + max_points) / 2).abs().sort_values().index[0]
    key_values = source_key if isinstance(source_key, tuple) else (source_key,)
    mask = pd.Series(True, index=raw.index)
    for col, value in zip(group_cols, key_values):
        mask &= raw[col].astype(str).eq(str(value))
    traj = raw.loc[mask].copy()
    traj = traj.sort_values("timestamp") if "timestamp" in traj.columns else traj
    return traj.head(max_points).reset_index(drop=True)


def matched_route_diversity(county: str) -> pd.DataFrame:
    rows = []
    for path in find_matched_files():
        if county_dir_name(county) not in str(path):
            continue
        try:
            data = read_table(path)
        except Exception:
            continue
        id_col = detect_column(data.columns, ("id", "trip_id", "trajectory_id"))
        route_col = detect_column(data.columns, PATH_CANDIDATES)
        if not id_col or not route_col:
            continue
        for record in data[[id_col, route_col]].itertuples(index=False):
            source_id, route = record
            rows.append({"source_id": str(source_id), "unique_fids": len(set(parse_segment_sequence(route)))})
    return pd.DataFrame(rows)


def load_county_edges(county: str) -> gpd.GeoDataFrame:
    path = Path("sflorida_outputs") / county_dir_name(county) / "fmm" / "edges.shp"
    if not path.exists():
        raise FileNotFoundError(f"FMM edges shapefile not found: {path}")
    return gpd.read_file(path)


def run_fmm(gps: pd.DataFrame, county: str):
    try:
        from fmm import FastMapMatch, FastMapMatchConfig, Network, NetworkGraph, UBODT  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return None, f"nearest-edge approximation, not FMM candidate output: FMM Python API unavailable ({exc})"
    county_dir = county_dir_name(county)
    network_path = Path("sflorida_outputs") / county_dir / "fmm" / "edges.shp"
    ubodt_path = Path("sflorida_outputs") / county_dir / "fmm" / "ubodt.txt"
    if not network_path.exists() or not ubodt_path.exists():
        return None, "nearest-edge approximation, not FMM candidate output: missing FMM network files"
    try:
        try:
            network = Network(str(network_path))
        except Exception:
            network = Network(str(network_path), "fid", "u", "v")
        graph = NetworkGraph(network)
        ubodt = UBODT.read_ubodt_csv(str(ubodt_path))
        matcher = FastMapMatch(network, graph, ubodt)
        try:
            config = FastMapMatchConfig(8, 300, 30)
        except TypeError:
            config = FastMapMatchConfig()
            for attr, value in (("k", 8), ("radius", 300), ("gps_error", 30)):
                try:
                    setattr(config, attr, value)
                except Exception:
                    pass
        line = LineString(gps[["lon", "lat"]].itertuples(index=False, name=None)).wkt
        return matcher.match_wkt(line, config), "FMM candidates"
    except Exception as exc:  # noqa: BLE001
        return None, f"nearest-edge approximation, not FMM candidate output: FMM matching failed ({exc})"


def nearest_candidate_rows(gps: pd.DataFrame, edges: gpd.GeoDataFrame, source: str) -> pd.DataFrame:
    rows = []
    edge_indexed = edges.reset_index(drop=True)
    for point_index, point in enumerate(gps.itertuples(index=False)):
        geom = Point(point.lon, point.lat)
        radius = 0.002
        local = edge_indexed.cx[point.lon - radius : point.lon + radius, point.lat - radius : point.lat + radius]
        while local.empty and radius < 0.05:
            radius *= 2
            local = edge_indexed.cx[point.lon - radius : point.lon + radius, point.lat - radius : point.lat + radius]
        if local.empty:
            local = edge_indexed
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*")
            nearest = local.assign(_distance=local.geometry.distance(geom)).sort_values("_distance").head(3)
        for rank, (_, edge) in enumerate(nearest.iterrows()):
            rows.append(
                {
                    "point_index": point_index,
                    "lon": point.lon,
                    "lat": point.lat,
                    "candidate_rank": rank,
                    "candidate_edge_id": int(edge["fid"]),
                    "raw_likelihood": None,
                    "distance": float(edge["_distance"]),
                    "error": None,
                    "offset": None,
                    "candidate_source": source,
                }
            )
    return pd.DataFrame(rows)


def extract_candidates(result, gps: pd.DataFrame, edges: gpd.GeoDataFrame, label: str) -> pd.DataFrame:
    rows = []
    if result is not None and hasattr(result, "candidates"):
        raw_candidates = getattr(result, "candidates", []) or []
        for point_index, candidate_list in enumerate(raw_candidates):
            iterable = candidate_list if isinstance(candidate_list, (list, tuple)) else [candidate_list]
            for rank, candidate in enumerate(iterable):
                rows.append(
                    {
                        "point_index": point_index,
                        "lon": gps["lon"].iloc[min(point_index, len(gps) - 1)],
                        "lat": gps["lat"].iloc[min(point_index, len(gps) - 1)],
                        "candidate_rank": rank,
                        "candidate_edge_id": candidate_value(candidate, "edge_id", "id", "eid", "fid"),
                        "raw_likelihood": candidate_value(candidate, "likelihood", "probability", "prob"),
                        "distance": candidate_value(candidate, "distance", "dist"),
                        "error": candidate_value(candidate, "error"),
                        "offset": candidate_value(candidate, "offset"),
                        "candidate_source": label,
                    }
                )
    if rows:
        fmm_rows = pd.DataFrame(rows)
        per_point = fmm_rows.groupby("point_index").size()
        if not per_point.empty and per_point.min() >= 2:
            return fmm_rows
        return nearest_candidate_rows(
            gps,
            edges,
            "nearest-edge approximation, not FMM candidate output: FMM candidate list incomplete",
        )
    return nearest_candidate_rows(gps, edges, "nearest-edge approximation, not FMM candidate output")


def add_candidate_scores(candidates: pd.DataFrame, sigma_m: float = 25.0) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    scored = candidates.copy()
    scored["raw_likelihood"] = pd.to_numeric(scored.get("raw_likelihood"), errors="coerce")
    scored["error"] = pd.to_numeric(scored.get("error"), errors="coerce")
    scored["distance"] = pd.to_numeric(scored.get("distance"), errors="coerce")
    scored["distance_m"] = scored["distance"].where(scored["distance"].abs() > 1.0, scored["distance"] * 111_000.0)
    scores = []
    sources = []
    for _, row in scored.iterrows():
        if pd.notna(row["raw_likelihood"]) and row["raw_likelihood"] > 0:
            scores.append(float(row["raw_likelihood"]))
            sources.append("FMM likelihood")
        elif pd.notna(row["error"]):
            scores.append(math.exp(-float(row["error"])))
            sources.append("exp(-error) normalized")
        elif pd.notna(row["distance_m"]):
            scores.append(math.exp(-float(row["distance_m"]) / sigma_m))
            sources.append(f"exp(-distance_m / {sigma_m:g}m) normalized")
        else:
            scores.append(1.0)
            sources.append("equal fallback: no likelihood/error/distance available")
    scored["candidate_score"] = scores
    scored["score_source"] = sources
    totals = scored.groupby("point_index")["candidate_score"].transform("sum")
    scored["candidate_percent"] = (scored["candidate_score"] / totals.where(totals > 0, 1.0)) * 100.0
    scored["candidate_label"] = scored["candidate_percent"].map(lambda value: f"{value:.0f}%")
    for point_index, group in scored.groupby("point_index"):
        if len(group) < 2:
            continue
        values = group["candidate_score"].astype(float)
        ratio = values.max() / max(values.min(), 1e-12)
        if ratio < 1.05:
            idx = scored["point_index"].eq(point_index)
            scored.loc[idx, "score_source"] = "approx. equal score: candidate scores differ by less than 5%"
    return scored


def display_candidate_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    pieces: list[pd.DataFrame] = []
    sorted_candidates = candidates.sort_values(["point_index", "candidate_percent"], ascending=[True, False])
    for _, group in sorted_candidates.groupby("point_index", sort=True):
        keep = group.head(2).copy()
        if len(group) >= 3:
            second_score = float(group.iloc[1]["candidate_percent"])
            third_score = float(group.iloc[2]["candidate_percent"])
            if second_score - third_score <= 10.0:
                keep = pd.concat([keep, group.iloc[[2]].copy()], ignore_index=True)
        keep["candidate_display_rank"] = np.arange(1, len(keep) + 1)
        pieces.append(keep)
    return pd.concat(pieces, ignore_index=True) if pieces else candidates.iloc[0:0].copy()


def fallback_matched_path_from_candidates(candidates: pd.DataFrame) -> list[int]:
    if candidates.empty:
        return []
    top = (
        candidates.sort_values(["point_index", "candidate_percent"], ascending=[True, False])
        .groupby("point_index", as_index=False)
        .head(1)
        .sort_values("point_index")
    )
    path: list[int] = []
    for value in pd.to_numeric(top["candidate_edge_id"], errors="coerce").dropna().astype(int):
        if not path or path[-1] != int(value):
            path.append(int(value))
    return path


def matched_line_from_candidates(candidates: pd.DataFrame, edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if candidates.empty:
        return edges.iloc[0:0].copy()
    top = (
        candidates.sort_values(["point_index", "candidate_percent"], ascending=[True, False])
        .groupby("point_index", as_index=False)
        .head(1)
        .sort_values("point_index")
    )
    edge_lookup = edges.set_index(edges["fid"].astype(int))["geometry"].to_dict()
    coords: list[tuple[float, float]] = []
    for row in top.itertuples(index=False):
        edge_id = pd.to_numeric(getattr(row, "candidate_edge_id"), errors="coerce")
        if pd.isna(edge_id):
            continue
        geom = edge_lookup.get(int(edge_id))
        if geom is None or geom.is_empty:
            continue
        point = Point(float(row.lon), float(row.lat))
        snapped = nearest_points(point, geom)[1]
        coord = (float(snapped.x), float(snapped.y))
        if not coords or math.hypot(coords[-1][0] - coord[0], coords[-1][1] - coord[1]) > 1e-7:
            coords.append(coord)
    if len(coords) < 2:
        return edges.iloc[0:0].copy()
    return gpd.GeoDataFrame({"fid": ["presentation_matched_route"]}, geometry=[LineString(coords)], crs=edges.crs)


def parse_result_path(result) -> list[int]:
    if result is None:
        return []
    for name in ("cpath", "opath", "path"):
        if hasattr(result, name):
            seq = parse_segment_sequence(getattr(result, name))
            if seq:
                return seq
    return []


def matched_csv_path_for_trajectory(gps: pd.DataFrame, county: str) -> list[int]:
    if "source_id" not in gps.columns:
        return []
    source_id = str(gps["source_id"].iloc[0])
    for path in find_matched_files():
        if county_dir_name(county) not in str(path):
            continue
        try:
            data = read_table(path)
        except Exception:
            continue
        id_col = detect_column(data.columns, ("id", "trip_id", "trajectory_id"))
        route_col = detect_column(data.columns, PATH_CANDIDATES)
        fid_col = detect_column(data.columns, FID_CANDIDATES)
        if id_col and route_col:
            match = data.loc[data[id_col].astype(str).eq(source_id)]
            if not match.empty:
                return parse_segment_sequence(match.iloc[0][route_col])
        if id_col and fid_col:
            match = data.loc[data[id_col].astype(str).eq(source_id)]
            if not match.empty:
                return [int(item) for item in pd.to_numeric(match[fid_col], errors="coerce").dropna()]
    return []


def edge_subset(edges: gpd.GeoDataFrame, fids: set[int]) -> gpd.GeoDataFrame:
    if not fids:
        return edges.iloc[0:0].copy()
    return edges.loc[edges["fid"].astype(int).isin(fids)].copy()


def clip_to_bounds(gdf: gpd.GeoDataFrame, bounds: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    view = box(*bounds)
    clipped = gdf.loc[gdf.intersects(view)].copy()
    if clipped.empty:
        return clipped
    clipped["geometry"] = clipped.geometry.intersection(view)
    return clipped.loc[~clipped.geometry.is_empty].copy()


def draw_lines(
    ax,
    gdf: gpd.GeoDataFrame,
    color: str,
    linewidth: float,
    alpha: float,
    zorder: int,
    linestyle: str = "-",
) -> None:
    for geom in gdf.geometry:
        geoms = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for line in geoms:
            xs, ys = zip(*line.coords)
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=zorder,
            )


def draw_candidate_lines(ax, candidates: gpd.GeoDataFrame, candidate_rows: pd.DataFrame, alpha_scale: float = 1.0) -> None:
    if candidates.empty:
        return
    if candidate_rows.empty or "candidate_edge_id" not in candidate_rows.columns:
        draw_lines(ax, candidates, "#7c3aed", 7.5, 0.94 * alpha_scale, 3)
        return
    rank_by_fid = (
        candidate_rows.assign(candidate_edge_id=pd.to_numeric(candidate_rows["candidate_edge_id"], errors="coerce"))
        .dropna(subset=["candidate_edge_id"])
        .groupby("candidate_edge_id")["candidate_display_rank"]
        .min()
        .to_dict()
    )
    styles = {
        1: {"color": "#7c3aed", "linewidth": 8.0, "alpha": 0.98, "linestyle": "-"},
        2: {"color": "#a855f7", "linewidth": 6.6, "alpha": 0.92, "linestyle": "--"},
        3: {"color": "#d946ef", "linewidth": 5.4, "alpha": 0.86, "linestyle": ":"},
    }
    for row in candidates.itertuples(index=False):
        fid = getattr(row, "fid", None)
        rank = int(rank_by_fid.get(float(fid), 1)) if fid is not None else 1
        style = styles.get(rank, styles[3])
        geoms = row.geometry.geoms if row.geometry.geom_type == "MultiLineString" else [row.geometry]
        for line in geoms:
            xs, ys = zip(*line.coords)
            ax.plot(
                xs,
                ys,
                color=style["color"],
                linewidth=style["linewidth"],
                alpha=style["alpha"] * alpha_scale,
                linestyle=style["linestyle"],
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=6 - min(rank, 3),
            )


def draw_scene(
    ax,
    gps: pd.DataFrame,
    background: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    candidate_rows: pd.DataFrame,
    matched: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    visible_points: int,
    show_candidates: bool,
    show_matched: bool,
    subtitle: str,
    compact: bool = False,
    candidate_alpha: float = 1.0,
    ) -> None:
    ax.set_facecolor("#f8fafc")
    draw_lines(ax, background, "#94a3b8", 1.05, 0.56, 1)
    if show_candidates:
        draw_candidate_lines(ax, candidates, candidate_rows, candidate_alpha)
    if visible_points and not show_matched:
        points = gps.head(visible_points)
        ax.scatter(points["lon"], points["lat"], s=155, color="#dc2626", edgecolors="white", linewidths=1.7, zorder=5, label="Raw GPS points")
    if show_matched:
        ax.scatter(
            gps["lon"],
            gps["lat"],
            s=86,
            color="#dc2626",
            edgecolors="white",
            linewidths=1.1,
            alpha=0.45,
            zorder=5,
        )
        draw_lines(ax, matched, "#2563eb", 12.0, 0.98, 6)
    minx, miny, maxx, maxy = bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    if show_candidates and not candidate_rows.empty and candidate_alpha > 0.65:
        draw_candidate_labels(ax, gps, candidates, candidate_rows, visible_points)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if compact:
        ax.set_title(subtitle, fontsize=12, weight="bold", pad=8)
        legend_lines = [("Road network", "#94a3b8")]
        if visible_points and not show_matched:
            legend_lines.append(("Raw GPS", "#dc2626"))
        if show_candidates:
            legend_lines.append(("Candidate", "#7c3aed"))
        if show_matched:
            legend_lines.append(("Matched route", "#2563eb"))
        for idx, (label, color) in enumerate(legend_lines):
            y = 0.055 + idx * 0.032
            ax.plot([0.035, 0.075], [y, y], transform=ax.transAxes, color=color, linewidth=4.6, solid_capstyle="round")
            ax.text(0.083, y, label, transform=ax.transAxes, va="center", fontsize=7.5, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70})
    else:
        ax.text(
            0.02,
            0.975,
            subtitle,
            transform=ax.transAxes,
            va="top",
            fontsize=12,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.94},
        )
        legend_lines = [("Gray = Road network", "#94a3b8")]
        if visible_points and not show_matched:
            legend_lines.append(("Red = Raw GPS observation", "#dc2626"))
        if show_candidates:
            legend_lines.append(("Purple = Candidate road segment", "#7c3aed"))
        if show_matched:
            legend_lines.append(("Blue = Final matched route", "#2563eb"))
        for idx, (label, color) in enumerate(legend_lines):
            y = 0.08 + idx * 0.035
            ax.plot([0.03, 0.07], [y, y], transform=ax.transAxes, color=color, linewidth=5, solid_capstyle="round")
            ax.text(0.08, y, label, transform=ax.transAxes, va="center", fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72})


def closest_point_on_geometry(point: Point, geom) -> Point:
    if geom is None or geom.is_empty:
        return point
    try:
        return nearest_points(point, geom)[1]
    except Exception:  # noqa: BLE001
        return point


def candidate_label_records(
    gps: pd.DataFrame,
    candidate_edges: gpd.GeoDataFrame,
    candidate_rows: pd.DataFrame,
    visible_points: int,
    max_labels: int = 10,
) -> list[dict[str, object]]:
    visible = candidate_rows.loc[candidate_rows["point_index"] < visible_points].copy()
    if visible.empty or candidate_edges.empty:
        return []
    edge_lookup = candidate_edges.set_index(candidate_edges["fid"].astype(int))["geometry"].to_dict()
    label_records: list[dict[str, object]] = []
    for point_index, group in visible.groupby("point_index"):
        point_idx = int(point_index)
        if point_idx >= len(gps):
            continue
        point = Point(float(gps.iloc[point_idx]["lon"]), float(gps.iloc[point_idx]["lat"]))
        records = []
        for row in group.sort_values("candidate_display_rank").head(3).itertuples(index=False):
            edge_id = pd.to_numeric(getattr(row, "candidate_edge_id"), errors="coerce")
            if pd.isna(edge_id):
                continue
            geom = edge_lookup.get(int(edge_id))
            if geom is None:
                continue
            anchor = closest_point_on_geometry(point, geom)
            distance_to_gps = point.distance(anchor)
            records.append({"point_index": point_idx, "row": row, "anchor": anchor, "distance_to_gps": distance_to_gps})
        if not records:
            continue
        max_sep = 0.0
        if len(records) > 1:
            anchors = [item["anchor"] for item in records]
            max_sep = max(
                math.hypot(a.x - b.x, a.y - b.y)
                for idx, a in enumerate(anchors)
                for b in anchors[idx + 1 :]
            )
        for record in records:
            row = record["row"]
            rank = int(getattr(row, "candidate_display_rank"))
            percent = float(getattr(row, "candidate_percent"))
            # Prefer high-ranked candidates on GPS points where candidate roads visibly separate.
            priority = (4 - rank) * 1000.0 + max_sep * 100000.0 + percent
            label_records.append({**record, "priority": priority})
    label_records.sort(key=lambda item: (-float(item["priority"]), int(item["point_index"]), int(getattr(item["row"], "candidate_display_rank"))))
    return label_records[: max_labels * 4]


def _rects_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _candidate_label_positions(anchor_axes: tuple[float, float]) -> list[tuple[float, float]]:
    ax_x, ax_y = anchor_axes
    column_positions = [
        (0.20, ax_y + 0.02),
        (0.80, ax_y + 0.02),
        (0.20, ax_y + 0.16),
        (0.80, ax_y + 0.16),
        (0.20, ax_y - 0.16),
        (0.80, ax_y - 0.16),
        (0.36, ax_y + 0.13),
        (0.64, ax_y + 0.13),
        (0.36, ax_y - 0.13),
        (0.64, ax_y - 0.13),
    ]
    offsets = [
        (-0.30, 0.16),
        (0.30, 0.16),
        (-0.30, -0.16),
        (0.30, -0.16),
        (-0.36, 0.03),
        (0.36, 0.03),
        (-0.12, 0.24),
        (0.12, 0.24),
        (-0.12, -0.24),
        (0.12, -0.24),
    ]
    positions: list[tuple[float, float]] = []
    for px, py in column_positions:
        x = min(max(px, 0.105), 0.895)
        y = min(max(py, 0.155), 0.890)
        if (x, y) not in positions:
            positions.append((x, y))
    for ox, oy in offsets:
        x = min(max(ax_x + ox, 0.105), 0.895)
        y = min(max(ax_y + oy, 0.155), 0.890)
        if (x, y) not in positions:
            positions.append((x, y))
    return positions


def draw_candidate_labels(
    ax,
    gps: pd.DataFrame,
    candidate_edges: gpd.GeoDataFrame,
    candidate_rows: pd.DataFrame,
    visible_points: int,
) -> None:
    records = candidate_label_records(gps, candidate_edges, candidate_rows, visible_points, max_labels=10)
    placed_rects: list[tuple[float, float, float, float]] = [
        (0.01, 0.00, 0.80, 0.125),  # explanatory caption
        (0.01, 0.89, 0.46, 0.995),  # step label in animation frames
    ]
    label_width = 0.300
    label_height = 0.158
    accepted = 0
    used_point_rank: set[tuple[int, int]] = set()
    for record in records:
        row = record["row"]
        anchor = record["anchor"]
        point_idx = int(record["point_index"])
        rank = int(row.candidate_display_rank)
        point_rank = (point_idx, rank)
        if point_rank in used_point_rank:
            continue
        anchor_axes = ax.transAxes.inverted().transform(
            ax.transData.transform((float(anchor.x), float(anchor.y)))
        )
        chosen_axes = None
        for tx, ty in _candidate_label_positions((float(anchor_axes[0]), float(anchor_axes[1]))):
            rect = (
                tx - label_width / 2,
                ty - label_height / 2,
                tx + label_width / 2,
                ty + label_height / 2,
            )
            if rect[0] < 0.01 or rect[2] > 0.99 or rect[1] < 0.01 or rect[3] > 0.98:
                continue
            if any(_rects_overlap(rect, placed) for placed in placed_rects):
                continue
            chosen_axes = (tx, ty)
            placed_rects.append(rect)
            break
        if chosen_axes is None:
            continue
        used_point_rank.add(point_rank)
        ax.annotate(
            f"GPS {point_idx + 1}\nCandidate {rank}\n{float(row.candidate_percent):.0f}%",
            xy=(float(anchor.x), float(anchor.y)),
            xycoords="data",
            xytext=chosen_axes,
            textcoords=ax.transAxes,
            fontsize=11.2,
            weight="bold",
            color="#581c87",
            ha="center",
            va="center",
            arrowprops={"arrowstyle": "->", "color": "#6d28d9", "lw": 1.5, "shrinkA": 3, "shrinkB": 5},
            bbox={"boxstyle": "round,pad=0.30", "facecolor": "white", "edgecolor": "#7c3aed", "alpha": 0.98},
            zorder=8,
        )
        accepted += 1
        if accepted >= 8:
            break
    ax.text(
        0.02,
        0.03,
        "For every GPS observation, FMM identifies nearby candidate road segments.\n"
        "The candidates are scored individually, and the highest-scoring connected\n"
        "sequence becomes the final matched route.",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#475569",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
        zorder=9,
    )


def label_segment_candidates(ax, candidate_rows: pd.DataFrame) -> None:
    if candidate_rows.empty:
        return
    current = candidate_rows.sort_values(["point_index", "candidate_rank"]).groupby("candidate_edge_id").tail(1).head(10)
    xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
    yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
    for row in current.itertuples(index=False):
        ax.annotate(
            str(row.candidate_label),
            xy=(float(row.lon), float(row.lat)),
            xytext=(float(row.lon) + xspan * 0.018, float(row.lat) + yspan * 0.018),
            fontsize=8,
            color="#581c87",
            arrowprops={"arrowstyle": "->", "color": "#6d28d9", "lw": 1.0},
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "#faf5ff", "edgecolor": "#7c3aed", "alpha": 0.92},
            zorder=8,
        )


def draw_frame(
    output: Path,
    gps: pd.DataFrame,
    background: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    candidate_rows: pd.DataFrame,
    matched: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    visible_points: int,
    show_candidates: bool,
    show_matched: bool,
    subtitle: str,
    candidate_alpha: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    draw_scene(
        ax,
        gps,
        background,
        candidates,
        candidate_rows,
        matched,
        bounds,
        visible_points,
        show_candidates,
        show_matched,
        subtitle,
        compact=False,
        candidate_alpha=candidate_alpha,
    )
    fig.savefig(output)
    plt.close(fig)


def draw_three_panel(
    output: Path,
    gps: pd.DataFrame,
    background: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    candidate_rows: pd.DataFrame,
    matched: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=180)
    panels = [
        ("1. Raw GPS observations", False, False),
        ("2. Candidate road segments", True, False),
        ("3. Final matched route", False, True),
    ]
    for ax, (subtitle, show_candidates, show_matched) in zip(axes, panels):
        draw_scene(
            ax,
            gps,
            background,
            candidates,
            candidate_rows,
            matched,
            bounds,
            len(gps),
            show_candidates,
            show_matched,
            subtitle,
            compact=True,
            candidate_alpha=1.0,
        )
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def sample_graph_nodes(matched: gpd.GeoDataFrame, max_nodes: int = 6) -> tuple[list[tuple[float, float]], pd.DataFrame]:
    if matched.empty:
        return [], pd.DataFrame()
    lines: list[LineString] = []
    for geom in matched.geometry:
        if geom is None or geom.is_empty:
            continue
        lines.extend(list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom])
    if not lines:
        return [], pd.DataFrame()
    line = max(lines, key=lambda item: item.length)
    coords = list(line.coords)
    if len(coords) < 2:
        return [], pd.DataFrame()
    count = min(max_nodes, len(coords))
    indices = np.linspace(0, len(coords) - 1, count).round().astype(int).tolist()
    nodes: list[tuple[float, float]] = []
    for idx in indices:
        coord = (float(coords[idx][0]), float(coords[idx][1]))
        if not nodes or math.hypot(nodes[-1][0] - coord[0], nodes[-1][1] - coord[1]) > 1e-8:
            nodes.append(coord)
    if len(nodes) < 2:
        nodes = [(float(coords[0][0]), float(coords[0][1])), (float(coords[-1][0]), float(coords[-1][1]))]
    mapping = pd.DataFrame(
        {
            "node_label": [f"N{i + 1}" for i in range(len(nodes))],
            "lon": [coord[0] for coord in nodes],
            "lat": [coord[1] for coord in nodes],
        }
    )
    return nodes, mapping


def draw_route_graph(ax, nodes: list[tuple[float, float]], *, show_title: bool = True) -> None:
    if len(nodes) < 2:
        return
    ax.set_facecolor("#f8fafc")
    xs = [coord[0] for coord in nodes]
    ys = [coord[1] for coord in nodes]
    for idx, (start, end) in enumerate(zip(nodes, nodes[1:])):
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 3.2,
                "color": "#2563eb",
                "shrinkA": 12,
                "shrinkB": 12,
                "mutation_scale": 16,
            },
            zorder=2,
        )
    ax.scatter(xs, ys, s=420, color="white", edgecolors="#2563eb", linewidths=2.5, zorder=3)
    for idx, (x, y) in enumerate(nodes):
        ax.text(x, y, f"N{idx + 1}", ha="center", va="center", fontsize=11, weight="bold", color="#1e3a8a", zorder=4)
    span_x = max(max(xs) - min(xs), 0.001)
    span_y = max(max(ys) - min(ys), 0.001)
    ax.set_xlim(min(xs) - span_x * 0.24, max(xs) + span_x * 0.24)
    ax.set_ylim(min(ys) - span_y * 0.24, max(ys) + span_y * 0.24)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if show_title:
        ax.set_title("Graph Representation of the Matched Route", fontsize=16, weight="bold", pad=12)


def save_graph_representation(output: Path, matched: gpd.GeoDataFrame) -> pd.DataFrame:
    nodes, mapping = sample_graph_nodes(matched, max_nodes=6)
    if len(nodes) < 2:
        return mapping
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    draw_route_graph(ax, nodes)
    fig.text(
        0.5,
        0.035,
        "The matched trajectory is converted into a graph.\nIntersections become nodes; traveled road segments become edges.",
        ha="center",
        fontsize=10.5,
        color="#334155",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output)
    plt.close(fig)
    return mapping


def save_graph_to_matrix(output: Path, matched: gpd.GeoDataFrame) -> pd.DataFrame:
    nodes, mapping = sample_graph_nodes(matched, max_nodes=6)
    if len(nodes) < 2:
        return mapping
    labels = mapping["node_label"].tolist()
    matrix = pd.DataFrame(0, index=labels, columns=labels)
    for left, right in zip(labels, labels[1:]):
        matrix.loc[left, right] = 1
        matrix.loc[right, left] = 1
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 6.5),
        dpi=180,
        gridspec_kw={"width_ratios": [1, 0.22, 1.18], "wspace": 0.16},
    )
    graph_ax, arrow_ax, matrix_ax = axes
    draw_route_graph(graph_ax, nodes, show_title=False)
    graph_ax.set_title("Graph", fontsize=15, weight="bold", pad=10)
    values = matrix.to_numpy(dtype=float)
    matrix_ax.imshow(values, cmap="Blues", vmin=0, vmax=1)
    matrix_ax.set_title("Adjacency Matrix", fontsize=15, weight="bold", pad=10)
    matrix_ax.set_xticks(np.arange(len(labels)))
    matrix_ax.set_yticks(np.arange(len(labels)))
    matrix_ax.set_xticklabels(labels, fontsize=11)
    matrix_ax.set_yticklabels(labels, fontsize=11)
    matrix_ax.set_xlabel("To node")
    matrix_ax.set_ylabel("From node")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            matrix_ax.text(j, i, f"{int(values[i, j])}", ha="center", va="center", fontsize=12, weight="bold", color="#0f172a")
    arrow_ax.axis("off")
    arrow_ax.text(0.5, 0.56, "→", ha="center", va="center", fontsize=48, weight="bold", color="#2563eb")
    arrow_ax.text(0.5, 0.43, "Convert\nGraph to\nMatrix", ha="center", va="center", fontsize=12, weight="bold", color="#1e3a8a")
    fig.text(
        0.5,
        0.03,
        "A 1 means two presentation nodes are connected by a traveled matched segment; a 0 means no connection in this example.",
        ha="center",
        fontsize=10.5,
        color="#334155",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output)
    plt.close(fig)
    return mapping


def padded_bounds(gps: pd.DataFrame, gdfs: list[gpd.GeoDataFrame]) -> tuple[float, float, float, float]:
    minx, maxx = gps["lon"].min(), gps["lon"].max()
    miny, maxy = gps["lat"].min(), gps["lat"].max()
    for gdf in gdfs:
        if not gdf.empty:
            bounds = gdf.total_bounds
            minx, miny = min(minx, bounds[0]), min(miny, bounds[1])
            maxx, maxy = max(maxx, bounds[2]), max(maxy, bounds[3])
    width = max(maxx - minx, 0.001)
    height = max(maxy - miny, 0.001)
    pad = max(width, height) * 0.14
    return minx - pad, miny - pad, maxx + pad, maxy + pad


def make_gif(frames: list[Path], output: Path, duration_ms: int) -> None:
    images = [Image.open(frame).convert("P", palette=Image.Palette.ADAPTIVE) for frame in frames]
    images[0].save(output, save_all=True, append_images=images[1:], duration=duration_ms, loop=0, optimize=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--county", default="Broward County")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "map_matching_gif")
    parser.add_argument("--graph-output-dir", type=Path, default=OUTPUT_ROOT / "graph_matrix")
    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=20)
    parser.add_argument("--duration-ms", type=int, default=650)
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    graph_out_dir = ensure_dir(args.graph_output_dir)
    frame_dir = ensure_dir(out_dir / "frames")
    for old_frame in frame_dir.glob("frame_*.png"):
        old_frame.unlink()
    edges = load_county_edges(args.county)
    gps = select_trajectory(args.county, args.min_points, args.max_points, edges)
    result, label = run_fmm(gps, args.county)
    candidates = add_candidate_scores(extract_candidates(result, gps, edges, label))
    candidates.to_csv(out_dir / "fmm_candidates_sample.csv", index=False)
    display_candidates = display_candidate_rows(candidates)
    candidate_fids = set(pd.to_numeric(display_candidates["candidate_edge_id"], errors="coerce").dropna().astype(int))
    result_path = parse_result_path(result)
    chosen_path = result_path if result_path else fallback_matched_path_from_candidates(candidates)
    matched_fids = set(chosen_path)
    candidate_edges = edge_subset(edges, candidate_fids)
    matched_edges = matched_line_from_candidates(candidates, edges)
    if matched_edges.empty:
        matched_edges = edge_subset(edges, matched_fids)
    # Keep the viewport focused on the GPS trajectory and selected route. Long
    # candidate road geometries are clipped to this view instead of controlling it.
    bounds = padded_bounds(gps, [matched_edges])
    candidate_edges = clip_to_bounds(candidate_edges, bounds)
    matched_edges = clip_to_bounds(matched_edges, bounds)
    view = box(*bounds)
    background = clip_to_bounds(edges.loc[edges.intersects(view)].copy(), bounds)
    if len(background) > 450:
        center = Point((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*")
            background = background.assign(_distance=background.geometry.distance(center)).sort_values("_distance").head(650)

    frames: list[Path] = []
    for idx in range(1, len(gps) + 1):
        frame = frame_dir / f"frame_{len(frames):03d}.png"
        draw_frame(frame, gps, background, candidate_edges, candidates.iloc[0:0], matched_edges, bounds, idx, False, False, f"Step 1: Raw GPS observations ({idx} of {len(gps)})")
        frames.append(frame)
    candidate_steps = sorted(set(round(value) for value in pd.Series(range(1, min(len(gps), 7) + 1)).mul(len(gps) / min(len(gps), 7))))
    for idx in candidate_steps:
        visible_fids = set(pd.to_numeric(display_candidates.loc[display_candidates["point_index"] < idx, "candidate_edge_id"], errors="coerce").dropna().astype(int))
        visible_candidates = edge_subset(edges, visible_fids)
        visible_rows = display_candidates.loc[display_candidates["point_index"] < idx]
        frame = frame_dir / f"frame_{len(frames):03d}.png"
        draw_frame(frame, gps, background, visible_candidates, visible_rows, matched_edges, bounds, len(gps), True, False, "Step 2: Candidate road segments")
        frames.append(frame)
    for _ in range(2):
        frame = frame_dir / f"frame_{len(frames):03d}.png"
        draw_frame(
            frame,
            gps,
            background,
            candidate_edges,
            display_candidates,
            matched_edges,
            bounds,
            len(gps),
            True,
            False,
            "Step 2: Candidate road segments",
        )
        frames.append(frame)
    for alpha in (0.45, 0.18):
        frame = frame_dir / f"frame_{len(frames):03d}.png"
        draw_frame(
            frame,
            gps,
            background,
            candidate_edges,
            display_candidates.iloc[0:0],
            matched_edges,
            bounds,
            len(gps),
            True,
            False,
            "Step 2: Candidate road segments",
            candidate_alpha=alpha,
        )
        frames.append(frame)
    for repeat in range(4):
        frame = frame_dir / f"frame_{len(frames):03d}.png"
        draw_frame(frame, gps, background, candidate_edges.iloc[0:0], candidates.iloc[0:0], matched_edges, bounds, len(gps), False, True, "Step 3: Final matched route")
        frames.append(frame)
    draw_three_panel(
        out_dir / "map_matching_process_three_panel_presentation.png",
        gps,
        background,
        candidate_edges,
        display_candidates,
        matched_edges,
        bounds,
    )
    draw_three_panel(
        out_dir / "map_matching_process_three_panel.png",
        gps,
        background,
        candidate_edges,
        display_candidates,
        matched_edges,
        bounds,
    )
    graph_mapping = save_graph_representation(graph_out_dir / "graph_representation_example.png", matched_edges)
    graph_mapping = save_graph_to_matrix(graph_out_dir / "graph_to_matrix_example.png", matched_edges)
    if not graph_mapping.empty:
        graph_mapping.to_csv(graph_out_dir / "graph_representation_example_node_mapping.csv", index=False)
    gif_path = out_dir / "map_matching_process.gif"
    make_gif(frames, gif_path, args.duration_ms)
    print(f"Selected trajectory points: {len(gps)}")
    print(f"Candidate rows: {len(candidates)}")
    print(f"Matched FIDs: {len(matched_fids)}")
    print(f"Frames: {len(frames)}")
    print(f"GIF: {gif_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
