#!/usr/bin/env python3
"""Generate a fixed-extent month-by-month route evolution animation."""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from shapely import wkt  # noqa: E402
from shapely.geometry import LineString, MultiLineString  # noqa: E402


DEFAULT_INPUT = Path(
    "deliverables/google_drive_phase2/driver_1003_monthly_graphs/data/"
    "driver_1003_all_monthly_nodes.csv"
)


def _county_slug(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _iter_line_coords(geometry: object):
    if isinstance(geometry, LineString):
        yield list(geometry.coords)
    elif isinstance(geometry, MultiLineString):
        for part in geometry.geoms:
            yield list(part.coords)


def _load_monthly_routes(input_path: Path, driver: str, county: str) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Monthly graph table not found: {input_path}")
    columns = [
        "driver_id",
        "month",
        "county",
        "fid",
        "trip_use_count",
        "monthly_trip_count",
        "geometry_wkt",
    ]
    data = pd.read_csv(input_path, usecols=lambda name: name in columns)
    if "driver_id" in data.columns:
        literal = data.loc[data["driver_id"].astype(str) == str(driver)]
        if not literal.empty:
            data = literal
    data = data.loc[data["county"].astype(str) == county].copy()
    data = data.loc[data["geometry_wkt"].notna() & (data["geometry_wkt"].astype(str) != "")]
    if data.empty:
        raise ValueError(f"No route rows found for driver {driver}, county {county}")
    data["trip_use_count"] = pd.to_numeric(data["trip_use_count"], errors="coerce").fillna(1)
    data["monthly_trip_count"] = (
        pd.to_numeric(data["monthly_trip_count"], errors="coerce").fillna(0).astype(int)
    )
    data["geometry"] = data["geometry_wkt"].map(wkt.loads)
    return data


def _fixed_extent(
    geometries: pd.Series,
    padding_fraction: float = 0.06,
) -> tuple[float, float, float, float]:
    bounds = [geom.bounds for geom in geometries]
    minx = min(item[0] for item in bounds)
    miny = min(item[1] for item in bounds)
    maxx = max(item[2] for item in bounds)
    maxy = max(item[3] for item in bounds)
    width = max(maxx - minx, 1.0)
    height = max(maxy - miny, 1.0)
    pad = max(width, height) * padding_fraction
    return minx - pad, maxx + pad, miny - pad, maxy + pad


def _draw_frame(
    month_data: pd.DataFrame,
    *,
    month: str,
    driver: str,
    county: str,
    extent: tuple[float, float, float, float],
    output_path: Path,
    figure_size: tuple[float, float],
) -> None:
    max_use = max(float(month_data["trip_use_count"].max()), 1.0)
    fig, ax = plt.subplots(figsize=figure_size, dpi=120)
    fig.patch.set_facecolor("#f7f8fb")
    ax.set_facecolor("#ffffff")

    for row in month_data.sort_values("trip_use_count").itertuples(index=False):
        weight = float(row.trip_use_count)
        scale = math.log1p(weight) / math.log1p(max_use)
        alpha = 0.25 + 0.65 * scale
        linewidth = 0.45 + 2.4 * scale
        for coords in _iter_line_coords(row.geometry):
            xs, ys = zip(*coords)
            ax.plot(xs, ys, color="#1f6f8b", linewidth=linewidth, alpha=alpha)

    minx, maxx, miny, maxy = extent
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    trip_count = int(month_data["monthly_trip_count"].max())
    route_count = int(month_data["fid"].nunique())
    ax.text(
        0.03,
        0.94,
        f"Driver {driver} route evolution",
        transform=ax.transAxes,
        fontsize=16,
        weight="bold",
        color="#172033",
        ha="left",
        va="top",
    )
    ax.text(
        0.03,
        0.885,
        f"{county} | {month} | {trip_count:,} trips | {route_count:,} FIDs",
        transform=ax.transAxes,
        fontsize=11,
        color="#344054",
        ha="left",
        va="top",
    )
    ax.text(
        0.03,
        0.045,
        "Line thickness and opacity scale with monthly trip-use count",
        transform=ax.transAxes,
        fontsize=9,
        color="#667085",
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output_path)
    plt.close(fig)


def _write_gif(frame_paths: list[Path], output_path: Path, duration_ms: int) -> None:
    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
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


def generate_animation(
    *,
    input_path: Path,
    output_dir: Path,
    driver: str,
    county: str,
    duration_ms: int,
    width: int,
    height: int,
) -> tuple[Path, Path | None, int]:
    data = _load_monthly_routes(input_path, driver=driver, county=county)
    months = sorted(data["month"].astype(str).unique())
    extent = _fixed_extent(data["geometry"])
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"driver_{driver}_{_county_slug(county)}_route_evolution"
    gif_path = output_dir / f"{prefix}.gif"
    mp4_path = output_dir / f"{prefix}.mp4"
    figure_size = (width / 120, height / 120)

    with tempfile.TemporaryDirectory(prefix="route_evolution_frames_") as tmp:
        tmp_dir = Path(tmp)
        frame_paths: list[Path] = []
        for index, month in enumerate(months):
            frame_path = tmp_dir / f"frame_{index:04d}.png"
            month_data = data.loc[data["month"].astype(str) == month]
            _draw_frame(
                month_data,
                month=month,
                driver=driver,
                county=county,
                extent=extent,
                output_path=frame_path,
                figure_size=figure_size,
            )
            frame_paths.append(frame_path)
        _write_gif(frame_paths, gif_path, duration_ms=duration_ms)
        fps = 1000 / duration_ms
        mp4_written = _write_mp4(frame_paths, mp4_path, fps=fps)

    return gif_path, mp4_path if mp4_written else None, len(months)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a fixed-extent route evolution GIF/MP4 from monthly FID geometry."
    )
    parser.add_argument("--driver", default="1003")
    parser.add_argument("--county", default="Broward County")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Monthly nodes CSV with geometry_wkt")
    parser.add_argument(
        "--output-dir",
        default="deliverables/driver_1003/route_evolution_animation",
    )
    parser.add_argument("--duration-ms", type=int, default=700)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gif_path, mp4_path, frame_count = generate_animation(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        driver=str(args.driver),
        county=str(args.county),
        duration_ms=args.duration_ms,
        width=args.width,
        height=args.height,
    )
    print(f"Generated {frame_count} frames")
    print(f"GIF: {gif_path}")
    if mp4_path:
        print(f"MP4: {mp4_path}")
    else:
        print("MP4: not generated because ffmpeg is not installed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
