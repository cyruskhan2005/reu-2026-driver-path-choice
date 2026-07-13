#!/usr/bin/env python3
"""Generate a presentation-quality month-by-month route evolution animation."""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402
from shapely import wkt  # noqa: E402
from shapely.geometry import LineString, MultiLineString, box  # noqa: E402


DEFAULT_INPUT_TEMPLATE = (
    "deliverables/google_drive_phase2/driver_{driver}_monthly_graphs/data/"
    "driver_{driver}_all_monthly_nodes.csv"
)
DEFAULT_BASEMAP_ROOT = Path("deliverables/google_drive_phase2/enriched_network_parquet")


@dataclass(frozen=True)
class Viewport:
    minx: float
    maxx: float
    miny: float
    maxy: float
    route_fill_x: float
    route_fill_y: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.minx, self.maxx, self.miny, self.maxy


@dataclass(frozen=True)
class AnimationResult:
    gif_path: Path
    mp4_path: Path | None
    frame_count: int
    months: tuple[str, ...]
    viewport: Viewport
    clipped_geometry_count: int
    mode: str


def _county_slug(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _county_dir(value: str) -> str:
    return value.replace("-", "_").replace(" ", "_")


def _default_input_path(driver: str) -> Path:
    return Path(DEFAULT_INPUT_TEMPLATE.format(driver=driver))


def _iter_line_coords(geometry: object):
    if isinstance(geometry, LineString):
        yield list(geometry.coords)
    elif isinstance(geometry, MultiLineString):
        for part in geometry.geoms:
            yield list(part.coords)


def _line_segments(geometries: pd.Series) -> list[list[tuple[float, float]]]:
    segments: list[list[tuple[float, float]]] = []
    for geometry in geometries:
        for coords in _iter_line_coords(geometry):
            if len(coords) >= 2:
                segments.append(coords)
    return segments


def _load_monthly_routes(input_path: Path, driver: str, county: str) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Monthly graph table not found: {input_path}")
    columns = [
        "driver_id",
        "driver_label",
        "month",
        "county",
        "fid",
        "trip_use_count",
        "monthly_trip_count",
        "geometry_wkt",
    ]
    data = pd.read_csv(input_path, usecols=lambda name: name in columns)

    driver_matches = []
    if "driver_id" in data.columns:
        driver_matches.append(data["driver_id"].astype(str) == str(driver))
    if "driver_label" in data.columns:
        driver_matches.append(data["driver_label"].astype(str).str.lower() == f"driver {driver}".lower())
    if driver_matches:
        literal = data.loc[pd.concat(driver_matches, axis=1).any(axis=1)]
        if not literal.empty:
            data = literal

    data = data.loc[data["county"].astype(str) == county].copy()
    data = data.loc[data["geometry_wkt"].notna() & (data["geometry_wkt"].astype(str) != "")]
    if data.empty:
        raise ValueError(f"No route rows found for driver {driver}, county {county}")
    data["month"] = data["month"].astype(str)
    data["trip_use_count"] = pd.to_numeric(data["trip_use_count"], errors="coerce").fillna(1)
    data["monthly_trip_count"] = (
        pd.to_numeric(data["monthly_trip_count"], errors="coerce").fillna(0).astype(int)
    )
    data["geometry"] = data["geometry_wkt"].map(wkt.loads)
    return data


def _compute_viewport(
    geometries: pd.Series,
    *,
    padding_fraction: float,
    bottom_ui_fraction: float = 0.17,
) -> Viewport:
    bounds = [geom.bounds for geom in geometries]
    route_minx = min(item[0] for item in bounds)
    route_miny = min(item[1] for item in bounds)
    route_maxx = max(item[2] for item in bounds)
    route_maxy = max(item[3] for item in bounds)
    route_width = max(route_maxx - route_minx, 1.0)
    route_height = max(route_maxy - route_miny, 1.0)
    pad = max(route_width, route_height) * padding_fraction
    minx = route_minx - pad
    maxx = route_maxx + pad
    miny = route_miny - pad - (route_height * bottom_ui_fraction)
    maxy = route_maxy + pad
    return Viewport(
        minx=minx,
        maxx=maxx,
        miny=miny,
        maxy=maxy,
        route_fill_x=route_width / max(maxx - minx, 1.0),
        route_fill_y=route_height / max(maxy - miny, 1.0),
    )


def _load_basemap(
    *,
    basemap_root: Path,
    county: str,
    viewport: Viewport,
    max_segments: int,
) -> list[list[tuple[float, float]]]:
    path = basemap_root / _county_dir(county) / "enriched_network.parquet"
    if not path.exists():
        return []
    roads = gpd.read_parquet(path, columns=["geometry", "highway"])
    roads = roads.loc[roads.geometry.notna()].copy()
    roads = roads.cx[viewport.minx : viewport.maxx, viewport.miny : viewport.maxy]
    if roads.empty:
        return []

    # Keep the basemap contextual but light. Major roads first, then sample the
    # remaining local roads if the viewport is dense.
    highway_order = {
        "motorway": 0,
        "trunk": 1,
        "primary": 2,
        "secondary": 3,
        "tertiary": 4,
        "residential": 5,
        "unclassified": 6,
        "service": 7,
    }

    def rank_highway(value: object) -> int:
        text = str(value)
        if text.startswith("[") and "," in text:
            text = text.split(",", 1)[0].strip("['\" ")
        return highway_order.get(text, 8)

    roads["highway_rank"] = roads["highway"].map(rank_highway)
    roads = roads.sort_values(["highway_rank"])
    if len(roads) > max_segments:
        major = roads.loc[roads["highway_rank"] <= 4]
        local = roads.loc[roads["highway_rank"] > 4]
        remaining = max(max_segments - len(major), 0)
        if remaining and not local.empty:
            local = local.sample(n=min(remaining, len(local)), random_state=17)
        roads = pd.concat([major, local], ignore_index=True).head(max_segments)

    clip_box = box(viewport.minx, viewport.miny, viewport.maxx, viewport.maxy)
    clipped = roads.geometry.intersection(clip_box)
    return _line_segments(clipped)


def _draw_basemap(ax, basemap_segments: list[list[tuple[float, float]]]) -> None:
    if not basemap_segments:
        return
    collection = LineCollection(
        basemap_segments,
        colors="#c9d0d8",
        linewidths=0.35,
        alpha=0.55,
        antialiaseds=True,
        zorder=1,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_collection(collection)


def _draw_route_layer(
    ax,
    route_data: pd.DataFrame,
    *,
    color: str,
    zorder: int,
    alpha_min: float,
    alpha_max: float,
    linewidth_min: float,
    linewidth_max: float,
    max_use: float | None = None,
) -> None:
    if route_data.empty:
        return
    max_use = max(float(max_use or route_data["trip_use_count"].max()), 1.0)
    for row in route_data.sort_values("trip_use_count").itertuples(index=False):
        weight = float(row.trip_use_count)
        scale = math.log1p(weight) / math.log1p(max_use)
        alpha = alpha_min + (alpha_max - alpha_min) * scale
        linewidth = linewidth_min + (linewidth_max - linewidth_min) * scale
        for coords in _iter_line_coords(row.geometry):
            if len(coords) < 2:
                continue
            xs, ys = zip(*coords)
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                solid_capstyle="round",
                solid_joinstyle="round",
                antialiased=True,
                zorder=zorder,
            )


def _draw_routes(ax, month_data: pd.DataFrame) -> None:
    max_use = max(float(month_data["trip_use_count"].max()), 1.0)
    _draw_route_layer(
        ax,
        month_data,
        color="#0b6f8f",
        zorder=4,
        alpha_min=0.35,
        alpha_max=0.95,
        linewidth_min=1.25,
        linewidth_max=6.5,
        max_use=max_use,
    )
    # A subtle dark centerline keeps dense high-use corridors crisp in GIF form.
    for row in month_data.loc[month_data["trip_use_count"] >= month_data["trip_use_count"].quantile(0.9)].itertuples(index=False):
        for coords in _iter_line_coords(row.geometry):
            if len(coords) < 2:
                continue
            xs, ys = zip(*coords)
            ax.plot(
                xs,
                ys,
                color="#084c61",
                linewidth=0.8,
                alpha=0.75,
                solid_capstyle="round",
                solid_joinstyle="round",
                antialiased=True,
                zorder=5,
            )


def _dedupe_routes(route_data: pd.DataFrame) -> pd.DataFrame:
    if route_data.empty:
        return route_data.copy()
    rows = []
    for fid, group in route_data.groupby("fid", sort=False):
        latest = group.sort_values("month").iloc[-1].copy()
        latest["trip_use_count"] = group["trip_use_count"].max()
        rows.append(latest)
    return pd.DataFrame(rows)


def _draw_progression_routes(
    ax,
    *,
    all_data: pd.DataFrame,
    current_data: pd.DataFrame,
    month: str,
) -> int:
    prior_data = all_data.loc[all_data["month"] < month]
    prior_fids = set(prior_data["fid"].astype(str))
    current_fids = current_data["fid"].astype(str)
    current_existing = current_data.loc[current_fids.isin(prior_fids)]
    current_new = current_data.loc[~current_fids.isin(prior_fids)]

    prior_unique = _dedupe_routes(prior_data)
    if not prior_unique.empty:
        _draw_route_layer(
            ax,
            prior_unique,
            color="#9aa7b5",
            zorder=2,
            alpha_min=0.16,
            alpha_max=0.34,
            linewidth_min=0.85,
            linewidth_max=3.2,
            max_use=all_data["trip_use_count"].max(),
        )
    if not current_existing.empty:
        _draw_route_layer(
            ax,
            current_existing,
            color="#0f7896",
            zorder=4,
            alpha_min=0.34,
            alpha_max=0.72,
            linewidth_min=1.15,
            linewidth_max=4.6,
            max_use=current_data["trip_use_count"].max(),
        )
    if not current_new.empty:
        _draw_route_layer(
            ax,
            current_new,
            color="#0077ff",
            zorder=6,
            alpha_min=0.68,
            alpha_max=0.98,
            linewidth_min=2.2,
            linewidth_max=7.0,
            max_use=current_data["trip_use_count"].max(),
        )
    return int(current_new["fid"].nunique())


def _add_title_block(
    ax,
    *,
    month_data: pd.DataFrame,
    month: str,
    driver: str,
    county: str,
    mode: str,
    new_fid_count: int | None,
) -> None:
    trip_count = int(month_data["monthly_trip_count"].max())
    route_count = int(month_data["fid"].nunique())
    panel = FancyBboxPatch(
        (0.032, 0.768),
        0.31,
        0.19,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=(1, 1, 1, 0.88),
        edgecolor="#d7dee8",
        linewidth=0.8,
        zorder=10,
    )
    ax.add_patch(panel)
    ax.text(
        0.052,
        0.925,
        f"Driver {driver} Route Evolution",
        transform=ax.transAxes,
        fontsize=16,
        weight="bold",
        color="#101828",
        ha="left",
        va="top",
        zorder=11,
    )
    ax.text(
        0.052,
        0.875,
        county,
        transform=ax.transAxes,
        fontsize=10.5,
        color="#344054",
        ha="left",
        va="top",
        zorder=11,
    )
    ax.text(
        0.052,
        0.835,
        "Month:",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#475467",
        ha="left",
        va="top",
        zorder=11,
    )
    ax.text(
        0.118,
        0.835,
        month,
        transform=ax.transAxes,
        fontsize=17,
        weight="bold",
        color="#0b6f8f",
        ha="left",
        va="top",
        zorder=11,
    )
    detail_lines = [
        f"{trip_count:,} trips",
        f"{route_count:,} road segments (FIDs)",
    ]
    if mode == "progression" and new_fid_count is not None:
        detail_lines.append(f"{new_fid_count:,} new this month")
    ax.text(
        0.052,
        0.785,
        "\n".join(detail_lines),
        transform=ax.transAxes,
        fontsize=10.0 if len(detail_lines) > 2 else 10.5,
        color="#1d2939",
        ha="left",
        va="top",
        linespacing=1.18,
        zorder=11,
    )


def _add_timeline(ax, *, months: tuple[str, ...], current_month: str) -> None:
    panel = FancyBboxPatch(
        (0.025, 0.015),
        0.95,
        0.155,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=(1, 1, 1, 0.9),
        edgecolor="#d7dee8",
        linewidth=0.8,
        zorder=10,
    )
    ax.add_patch(panel)
    left, right = 0.065, 0.935
    y = 0.132
    line = LineCollection(
        [[(left, y), (right, y)]],
        transform=ax.transAxes,
        colors="#c8d2dc",
        linewidths=1.2,
        zorder=11,
    )
    ax.add_collection(line)
    count = max(len(months) - 1, 1)
    for index, month in enumerate(months):
        x = left + (right - left) * (index / count)
        is_current = month == current_month
        ax.scatter(
            [x],
            [y],
            transform=ax.transAxes,
            s=46 if is_current else 15,
            color="#0b6f8f" if is_current else "#98a2b3",
            edgecolor="white",
            linewidth=0.8 if is_current else 0.35,
            zorder=12,
        )
        ax.text(
            x,
            0.103,
            month,
            transform=ax.transAxes,
            fontsize=7.5 if is_current else 5.8,
            weight="bold" if is_current else "normal",
            color="#0b6f8f" if is_current else "#667085",
            ha="right",
            va="top",
            rotation=45,
            zorder=12,
        )


def _draw_frame(
    month_data: pd.DataFrame,
    *,
    all_data: pd.DataFrame,
    month: str,
    months: tuple[str, ...],
    driver: str,
    county: str,
    mode: str,
    viewport: Viewport,
    basemap_segments: list[list[tuple[float, float]]],
    output_path: Path,
    figure_size: tuple[float, float],
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
    fig.patch.set_facecolor("#eef2f6")
    ax.set_facecolor("#f6f7f9")
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(viewport.minx, viewport.maxx)
    ax.set_ylim(viewport.miny, viewport.maxy)
    ax.set_aspect("auto")
    ax.axis("off")

    _draw_basemap(ax, basemap_segments)
    new_fid_count: int | None = None
    if mode == "progression":
        new_fid_count = _draw_progression_routes(
            ax,
            all_data=all_data,
            current_data=month_data,
            month=month,
        )
    else:
        _draw_routes(ax, month_data)
    _add_title_block(
        ax,
        month_data=month_data,
        month=month,
        driver=driver,
        county=county,
        mode=mode,
        new_fid_count=new_fid_count,
    )
    _add_timeline(ax, months=months, current_month=month)

    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _write_gif(frame_paths: list[Path], output_path: Path, duration_ms: int) -> None:
    images = [
        Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE)
        for path in frame_paths
    ]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    for image in images:
        image.close()


def _write_mp4(frame_paths: list[Path], output_path: Path, fps: float) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_paths[0].parent / "frame_%04d.png"),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return True


def _count_clipped_geometries(data: pd.DataFrame, viewport: Viewport) -> int:
    return sum(
        1
        for geometry in data["geometry"]
        if not (
            geometry.bounds[0] >= viewport.minx
            and geometry.bounds[2] <= viewport.maxx
            and geometry.bounds[1] >= viewport.miny
            and geometry.bounds[3] <= viewport.maxy
        )
    )


def generate_animation(
    *,
    input_path: Path,
    output_dir: Path,
    driver: str,
    county: str,
    basemap_root: Path,
    include_basemap: bool,
    duration_ms: int,
    width: int,
    height: int,
    dpi: int,
    padding_fraction: float,
    max_basemap_segments: int,
    mode: str,
) -> AnimationResult:
    data = _load_monthly_routes(input_path, driver=driver, county=county)
    months = tuple(sorted(data["month"].astype(str).unique()))
    viewport = _compute_viewport(data["geometry"], padding_fraction=padding_fraction)
    clipped_geometry_count = _count_clipped_geometries(data, viewport)
    basemap_segments = (
        _load_basemap(
            basemap_root=basemap_root,
            county=county,
            viewport=viewport,
            max_segments=max_basemap_segments,
        )
        if include_basemap
        else []
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"driver_{driver}_{_county_slug(county)}_route_evolution"
    gif_path = output_dir / f"{prefix}.gif"
    mp4_path = output_dir / f"{prefix}.mp4"
    figure_size = (width / dpi, height / dpi)

    with tempfile.TemporaryDirectory(prefix="route_evolution_frames_") as tmp:
        tmp_dir = Path(tmp)
        frame_paths: list[Path] = []
        for index, month in enumerate(months):
            frame_path = tmp_dir / f"frame_{index:04d}.png"
            month_data = data.loc[data["month"].astype(str) == month]
            _draw_frame(
                month_data,
                all_data=data,
                month=month,
                months=months,
                driver=driver,
                county=county,
                mode=mode,
                viewport=viewport,
                basemap_segments=basemap_segments,
                output_path=frame_path,
                figure_size=figure_size,
                dpi=dpi,
            )
            frame_paths.append(frame_path)
        _write_gif(frame_paths, gif_path, duration_ms=duration_ms)
        fps = 1000 / duration_ms
        mp4_written = _write_mp4(frame_paths, mp4_path, fps=fps)

    return AnimationResult(
        gif_path=gif_path,
        mp4_path=mp4_path if mp4_written else None,
        frame_count=len(months),
        months=months,
        viewport=viewport,
        clipped_geometry_count=clipped_geometry_count,
        mode=mode,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a fixed-viewport route evolution GIF/MP4 from monthly FID geometry."
    )
    parser.add_argument("--driver", default="1003")
    parser.add_argument("--county", default="Broward County")
    parser.add_argument(
        "--input",
        help=(
            "Monthly nodes CSV with geometry_wkt. Defaults to the standard "
            "driver-specific Google Drive bundle path."
        ),
    )
    parser.add_argument("--basemap-root", default=str(DEFAULT_BASEMAP_ROOT))
    parser.add_argument("--no-basemap", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="deliverables/driver_1003/route_evolution_animation",
    )
    parser.add_argument("--duration-ms", type=int, default=550)
    parser.add_argument(
        "--mode",
        choices=["progression", "monthly"],
        default="progression",
        help=(
            "progression draws prior months in gray and highlights new current-month "
            "FIDs; monthly draws only the selected month."
        ),
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--padding-fraction", type=float, default=0.075)
    parser.add_argument("--max-basemap-segments", type=int, default=22000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input) if args.input else _default_input_path(str(args.driver))
    result = generate_animation(
        input_path=input_path,
        output_dir=Path(args.output_dir),
        driver=str(args.driver),
        county=str(args.county),
        basemap_root=Path(args.basemap_root),
        include_basemap=not args.no_basemap,
        duration_ms=args.duration_ms,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        padding_fraction=args.padding_fraction,
        max_basemap_segments=args.max_basemap_segments,
        mode=args.mode,
    )
    print(f"Generated {result.frame_count} monthly frames")
    print(f"Mode: {result.mode}")
    print(f"Months: {', '.join(result.months)}")
    print(
        "Viewport: "
        f"x=({result.viewport.minx:.1f}, {result.viewport.maxx:.1f}), "
        f"y=({result.viewport.miny:.1f}, {result.viewport.maxy:.1f})"
    )
    print(
        "Route fill: "
        f"{result.viewport.route_fill_x:.1%} width, "
        f"{result.viewport.route_fill_y:.1%} height"
    )
    print(f"Clipped route geometries: {result.clipped_geometry_count}")
    print(f"GIF: {result.gif_path}")
    if result.mp4_path:
        print(f"MP4: {result.mp4_path}")
    else:
        print("MP4: not generated because ffmpeg is not installed")
    return 0 if result.clipped_geometry_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
