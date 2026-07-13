#!/usr/bin/env python3
"""Create a month-by-month GIF of matched segment usage."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from PIL import Image  # noqa: E402
from shapely.geometry import box  # noqa: E402

from research_process_utils import (
    OUTPUT_ROOT,
    ensure_dir,
    explode_segment_observations,
    load_network_edges,
    segment_trip_counts,
)


def line_segments(series) -> list[list[tuple[float, float]]]:
    segments: list[list[tuple[float, float]]] = []
    for geom in series:
        geoms = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for line in geoms:
            coords = list(line.coords)
            if len(coords) >= 2:
                segments.append(coords)
    return segments


def load_usage_geometry(counts: pd.DataFrame) -> gpd.GeoDataFrame:
    pieces: list[gpd.GeoDataFrame] = []
    for county, group in counts.groupby("county", dropna=False):
        try:
            edges = load_network_edges(str(county), group["fid"].unique(), include_geometry=True)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not load geometry for {county}: {exc}")
            continue
        pieces.append(edges.merge(group, on="fid", how="inner"))
    if not pieces:
        return gpd.GeoDataFrame()
    return gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), geometry="geometry", crs=pieces[0].crs)


def add_padding(bounds, fraction: float = 0.07):
    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, 1.0)
    height = max(maxy - miny, 1.0)
    pad = max(width, height) * fraction
    return minx - pad, miny - pad, maxx + pad, maxy + pad


def draw_frame(gdf: gpd.GeoDataFrame, month: str, months: list[str], bounds, output: Path, width: int, height: int) -> None:
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    minx, miny, maxx, maxy = bounds
    base = gdf.cx[minx:maxx, miny:maxy].drop_duplicates("fid")
    base_segments = line_segments(base.geometry)
    if base_segments:
        ax.add_collection(LineCollection(base_segments, colors="#cbd5e1", linewidths=0.35, alpha=0.45, zorder=1))
    current = gdf.loc[gdf["month"].astype(str) == month].copy()
    max_count = max(float(gdf["trip_use_count"].max()), 1.0)
    for row in current.sort_values("trip_use_count").itertuples(index=False):
        scale = math.log1p(float(row.trip_use_count)) / math.log1p(max_count)
        color = "#08306b" if scale > 0.75 else "#08519c" if scale > 0.5 else "#2171b5"
        lw = 1.2 + 5.5 * scale
        for coords in line_segments([row.geometry]):
            xs, ys = zip(*coords)
            ax.plot(xs, ys, color=color, linewidth=lw, alpha=0.76, solid_capstyle="round", solid_joinstyle="round", zorder=5)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    trips = int(current["trip_use_count"].sum()) if not current.empty else 0
    fids = int(current["fid"].nunique()) if not current.empty else 0
    ax.text(
        0.02,
        0.96,
        f"Matched Segment Usage Over Time\nMonth: {month}\n{trips:,} trip-segment uses\n{fids:,} road segments (FIDs)",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=15,
        linespacing=1.25,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.95},
    )
    timeline = "  ".join([m if m == month else "•" for m in months])
    ax.text(0.5, 0.035, timeline, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="#0f172a")
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "segment_usage_gif")
    parser.add_argument("--duration-ms", type=int, default=600)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    frame_dir = ensure_dir(out_dir / "frames")
    counts_path = OUTPUT_ROOT / "segment_usage" / "segment_usage_counts.csv"
    if counts_path.exists():
        counts = pd.read_csv(counts_path)
    else:
        observations = explode_segment_observations(prefer_timeline=True)
        counts = segment_trip_counts(observations, by_period=True).rename(columns={"period": "month"})
    if counts.empty or "month" not in counts.columns:
        print("WARNING: no monthly segment counts available for GIF creation.")
        return
    gdf = load_usage_geometry(counts.dropna(subset=["month"]))
    if gdf.empty:
        print("WARNING: no segment geometry available for GIF creation.")
        return
    months = sorted(gdf["month"].astype(str).unique())
    bounds = add_padding(gdf.total_bounds)
    frames: list[Path] = []
    for idx, month in enumerate(months, start=1):
        frame = frame_dir / f"{idx:03d}_{month}.png"
        draw_frame(gdf, month, months, bounds, frame, args.width, args.height)
        frames.append(frame)
    images = [Image.open(frame).convert("P", palette=Image.Palette.ADAPTIVE) for frame in frames]
    gif_path = out_dir / "segment_usage_by_month.gif"
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=args.duration_ms, loop=0, optimize=False)
    print(f"Frames: {len(frames)}")
    print(f"GIF: {gif_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
