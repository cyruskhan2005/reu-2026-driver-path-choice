#!/usr/bin/env python3
"""Create HTML maps of matched road segment usage."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd

from research_process_utils import (
    OUTPUT_ROOT,
    ensure_dir,
    explode_segment_observations,
    load_network_edges,
    safe_period_label,
    segment_trip_counts,
    write_html,
)


def color_for_count(count: float, max_count: float) -> str:
    if max_count <= 1:
        scale = 1.0
    else:
        scale = math.log1p(count) / math.log1p(max_count)
    if scale > 0.75:
        return "#08306b"
    if scale > 0.5:
        return "#08519c"
    if scale > 0.25:
        return "#2171b5"
    return "#6baed6"


def add_usage_lines(m: folium.Map, gdf: gpd.GeoDataFrame, count_col: str) -> None:
    if gdf.empty:
        return
    max_count = max(float(gdf[count_col].max()), 1.0)
    for row in gdf.sort_values(count_col).itertuples(index=False):
        geom = row.geometry
        count = float(getattr(row, count_col))
        weight = 1.0 + 6.0 * (math.log1p(count) / math.log1p(max_count))
        color = color_for_count(count, max_count)
        geoms = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for line in geoms:
            coords = [(lat, lon) for lon, lat in line.coords]
            folium.PolyLine(
                coords,
                color=color,
                weight=weight,
                opacity=0.72,
                tooltip=f"FID {row.fid}: {int(count)} trips",
            ).add_to(m)


def build_map(counts: pd.DataFrame, output: Path, title: str, max_segments: int) -> None:
    layers: list[gpd.GeoDataFrame] = []
    for county, group in counts.groupby("county", dropna=False):
        top = group.sort_values("trip_use_count", ascending=False).head(max_segments)
        try:
            edges = load_network_edges(str(county), top["fid"].tolist(), include_geometry=True)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not load network geometry for {county}: {exc}")
            continue
        merged = edges.merge(top, on="fid", how="inner")
        if merged.empty:
            continue
        if merged.crs and str(merged.crs).lower() not in {"epsg:4326", "wgs84"}:
            merged = merged.to_crs(4326)
        layers.append(merged)
    if not layers:
        write_html(output, title, f"<h1>{title}</h1><p>No matched segment geometry could be mapped.</p>")
        return
    gdf = pd.concat(layers, ignore_index=True)
    bounds = gdf.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    m = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron", control_scale=True)
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    add_usage_lines(m, gdf, "trip_use_count")
    legend = """
    <div style="position: fixed; top: 14px; left: 50px; z-index: 9999; background: white;
        border: 1px solid #d1d5db; border-radius: 6px; padding: 10px 12px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 360px;">
        <strong>Matched Road Segment Usage</strong><br>
        Darker/thicker segments represent more trips passing through that segment.
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    ensure_dir(output.parent)
    m.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "segment_usage")
    parser.add_argument("--max-segments", type=int, default=12000)
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    observations = explode_segment_observations(prefer_timeline=True)
    counts_by_month = segment_trip_counts(observations, by_period=True)
    if "period" in counts_by_month.columns:
        counts_by_month = counts_by_month.rename(columns={"period": "month"})
    counts_by_month.to_csv(out_dir / "segment_usage_counts.csv", index=False)

    if counts_by_month.empty:
        print("WARNING: no matched segment observations were available.")
        return
    overall = counts_by_month.groupby(["county", "fid"], dropna=False)[["trip_use_count", "segment_pass_count"]].sum().reset_index()
    build_map(overall, out_dir / "segment_usage_overall.html", "Matched Road Segment Usage", args.max_segments)
    for month, group in counts_by_month.dropna(subset=["month"]).groupby("month"):
        build_map(
            group,
            out_dir / f"segment_usage_by_month_{safe_period_label(month)}.html",
            f"Matched Road Segment Usage: {month}",
            args.max_segments,
        )
    print(f"Segment usage rows: {len(counts_by_month):,}")
    print(f"Wrote {out_dir.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
