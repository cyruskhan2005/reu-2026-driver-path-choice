#!/usr/bin/env python3
"""Visualize raw GPS points before map matching."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402

from research_process_utils import OUTPUT_ROOT, ensure_dir, iter_raw_gps, write_html


def map_pin_marker() -> MplPath:
    """Small filled map-pin silhouette for dense GPS point plots."""
    vertices = [
        (0.0, -1.0),
        (-0.18, -0.76),
        (-0.38, -0.50),
        (-0.54, -0.24),
        (-0.66, 0.04),
        (-0.66, 0.34),
        (-0.52, 0.60),
        (-0.28, 0.78),
        (0.0, 0.86),
        (0.28, 0.78),
        (0.52, 0.60),
        (0.66, 0.34),
        (0.66, 0.04),
        (0.54, -0.24),
        (0.38, -0.50),
        (0.18, -0.76),
        (0.0, -1.0),
    ]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(vertices, codes)


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = [{"metric": "total_raw_gps_points", "group": "all", "value": int(len(data))}]
    for col in ("county", "driver_id", "source_id", "source_file"):
        if col in data.columns:
            counts = data.groupby(col, dropna=False).size().sort_values(ascending=False)
            for key, value in counts.items():
                rows.append({"metric": f"points_by_{col}", "group": key, "value": int(value)})
    return pd.DataFrame(rows)


def load_counts_and_points() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = 0
    summary_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []
    point_parts: list[pd.DataFrame] = []
    for chunk in iter_raw_gps():
        total += len(chunk)
        for col in ("county", "driver_id", "source_id", "source_file"):
            if col in chunk.columns:
                counts = chunk.groupby(col, dropna=False).size().rename("value").reset_index().rename(columns={col: "group"})
                counts["metric"] = f"points_by_{col}"
                summary_parts.append(counts[["metric", "group", "value"]])
        if "month" in chunk.columns:
            monthly_parts.append(chunk.dropna(subset=["month"]).groupby(["county", "month"], dropna=False).size().rename("raw_gps_points").reset_index())
        point_cols = ["lon", "lat"] + [
            col for col in ("county", "source_id", "source_file", "driver_id", "timestamp") if col in chunk.columns
        ]
        point_parts.append(chunk[point_cols].copy())

    summary = pd.DataFrame([{"metric": "total_raw_gps_points", "group": "all", "value": total}])
    if summary_parts:
        grouped = pd.concat(summary_parts, ignore_index=True).groupby(["metric", "group"], dropna=False)["value"].sum().reset_index()
        summary = pd.concat([summary, grouped], ignore_index=True)
    if monthly_parts:
        monthly = pd.concat(monthly_parts, ignore_index=True).groupby(["county", "month"], dropna=False)["raw_gps_points"].sum().reset_index()
    else:
        monthly = pd.DataFrame(columns=["county", "month", "raw_gps_points"])
    points = pd.concat(point_parts, ignore_index=True) if point_parts else pd.DataFrame(columns=["lon", "lat", "county"])
    return summary, monthly, points


def save_density_png(data: pd.DataFrame, output: Path, total_points: int, bins: int) -> None:
    if data.empty:
        return
    lon = data["lon"].to_numpy(dtype=float)
    lat = data["lat"].to_numpy(dtype=float)
    hist, xedges, yedges = np.histogram2d(lon, lat, bins=bins)
    hist = hist.T
    masked = np.ma.masked_where(hist <= 0, hist)
    fig, ax = plt.subplots(figsize=(13.5, 8), dpi=170)
    image = ax.imshow(
        masked,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="magma",
        norm=LogNorm(vmin=1, vmax=max(float(hist.max()), 1.0)),
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title("Raw GPS Points Before Map Matching", fontsize=16, weight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_facecolor("#f8fafc")
    ax.grid(color="#dbe3ee", linewidth=0.4, alpha=0.5)
    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label("Raw GPS point count per raster cell")
    ax.text(
        0.01,
        0.01,
        f"Full raw GPS point count: {total_points:,}\nMain visualization uses all raw GPS points, not a sample.",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.94},
    )
    fig.tight_layout()
    ensure_dir(output.parent)
    fig.savefig(output)
    plt.close(fig)


def save_dense_scatter_png(data: pd.DataFrame, output: Path, total_points: int) -> None:
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.set_facecolor("#f8fafc")
    lon = data["lon"].to_numpy(dtype=float)
    lat = data["lat"].to_numpy(dtype=float)
    hist, xedges, yedges = np.histogram2d(lon, lat, bins=1800)
    # Spread occupied cells slightly so dense corridors look crowded at
    # presentation scale while still being derived from every raw point.
    spread = hist.copy()
    for _ in range(2):
        spread = np.maximum.reduce(
            [
                spread,
                np.roll(spread, 1, axis=0),
                np.roll(spread, -1, axis=0),
                np.roll(spread, 1, axis=1),
                np.roll(spread, -1, axis=1),
            ]
        )
    spread = np.ma.masked_where(spread.T <= 0, spread.T)
    ax.imshow(
        spread,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="Greys",
        norm=LogNorm(vmin=1, vmax=max(float(hist.max()), 1.0)),
        alpha=0.42,
        interpolation="nearest",
        aspect="auto",
    )
    ax.scatter(
        lon,
        lat,
        s=0.45,
        c="#030712",
        alpha=0.16,
        linewidths=0,
        marker="o",
        rasterized=True,
    )
    ax.set_title("Raw GPS Points Before Map Matching", fontsize=18, weight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(color="#dbe3ee", linewidth=0.35, alpha=0.55)
    ax.text(
        0.01,
        0.015,
        (
            f"Full raw GPS point count: {total_points:,}\n"
            "This plot intentionally draws all raw GPS points.\n"
            "Heavy overlap shows the scale of the dataset before map matching."
        ),
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.94},
    )
    fig.tight_layout()
    ensure_dir(output.parent)
    fig.savefig(output)
    plt.close(fig)


def save_big_overlap_png(data: pd.DataFrame, output: Path, total_points: int) -> None:
    if data.empty:
        return
    lon = data["lon"].to_numpy(dtype=float)
    lat = data["lat"].to_numpy(dtype=float)
    x0, x1 = np.nanquantile(lon, [0.03, 0.985])
    y0, y1 = np.nanquantile(lat, [0.03, 0.985])
    xpad = max((x1 - x0) * 0.04, 0.002)
    ypad = max((y1 - y0) * 0.04, 0.002)

    fig, ax = plt.subplots(figsize=(14, 10), dpi=260)
    ax.set_facecolor("#eef5ec")
    # Draw every point twice: a dark low-alpha "marker shadow" underneath and
    # a colored point layer above it. At presentation scale this reads more
    # like a crowded marker map than a clean road-centerline trace.
    ax.scatter(
        lon,
        lat,
        s=6.0,
        c="#020617",
        alpha=0.035,
        linewidths=0,
        marker="o",
        rasterized=True,
    )
    colors = {
        "Broward County": "#4f46e5",
        "Miami-Dade County": "#f59e0b",
        "Palm Beach County": "#22c55e",
    }
    if "county" in data.columns:
        for county, group in data.groupby("county", sort=False):
            ax.scatter(
                group["lon"].to_numpy(dtype=float),
                group["lat"].to_numpy(dtype=float),
                s=2.4,
                c=colors.get(str(county), "#ef4444"),
                alpha=0.12,
                linewidths=0,
                marker="o",
                rasterized=True,
                label=str(county),
            )
        ax.legend(loc="upper left", frameon=True, markerscale=4, fontsize=9)
    else:
        ax.scatter(
            lon,
            lat,
            s=2.4,
            c="#4f46e5",
            alpha=0.12,
            linewidths=0,
            marker="o",
            rasterized=True,
        )
    ax.set_xlim(x0 - xpad, x1 + xpad)
    ax.set_ylim(y0 - ypad, y1 + ypad)
    ax.set_title("All Raw GPS Points Before Map Matching", fontsize=18, weight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(color="#dbe3ee", linewidth=0.35, alpha=0.45)
    ax.text(
        0.015,
        0.02,
        (
            f"Full raw GPS point count: {total_points:,}\n"
            "Every raw GPS point is plotted.\n"
            "Heavy overlap is intentional to show the scale of the dataset."
        ),
        transform=ax.transAxes,
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.94},
    )
    fig.tight_layout()
    ensure_dir(output.parent)
    fig.savefig(output)
    plt.close(fig)


def save_google_style_marker_map(data: pd.DataFrame, output: Path, total_points: int) -> None:
    if data.empty:
        return
    lon = data["lon"].to_numpy(dtype=float)
    lat = data["lat"].to_numpy(dtype=float)
    x0, x1 = np.nanquantile(lon, [0.025, 0.99])
    y0, y1 = np.nanquantile(lat, [0.025, 0.99])
    xpad = max((x1 - x0) * 0.05, 0.003)
    ypad = max((y1 - y0) * 0.05, 0.003)
    rng = np.random.default_rng(1003)
    jitter_lon = rng.normal(0, max((x1 - x0) * 0.0014, 0.00018), size=len(lon))
    jitter_lat = rng.normal(0, max((y1 - y0) * 0.0014, 0.00018), size=len(lat))
    plot_lon = lon + jitter_lon
    plot_lat = lat + jitter_lat

    fig, ax = plt.subplots(figsize=(16, 9), dpi=260)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#e7f1df")

    ax.scatter(
        plot_lon,
        plot_lat,
        s=13.0,
        c="#ef2b12",
        alpha=0.46,
        marker=map_pin_marker(),
        linewidths=0,
        antialiaseds=False,
        rasterized=True,
    )

    ax.set_xlim(x0 - xpad, x1 + xpad)
    ax.set_ylim(y0 - ypad, y1 + ypad)
    ax.grid(color="#b9d7b3", linewidth=0.7, alpha=0.45)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Raw GPS Trajectory Dataset ({total_points / 1_000_000:.2f} Million Observations)",
        fontsize=20,
        weight="bold",
        pad=14,
    )
    ax.text(-80.17, 26.42, "Broward", fontsize=13, weight="bold", color="#334155", alpha=0.8)
    ax.text(-80.27, 26.25, "Miami-Dade", fontsize=13, weight="bold", color="#334155", alpha=0.8)
    ax.text(-80.12, 26.64, "Palm Beach", fontsize=13, weight="bold", color="#334155", alpha=0.8)
    ax.text(
        0.012,
        0.025,
        (
            f"Full raw GPS point count: {total_points:,}\n"
            "Each red map-pin marker represents a raw GPS observation.\n"
            "Dense overlap is intentional: this is the raw input before map matching."
        ),
        transform=ax.transAxes,
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.94},
        zorder=10,
    )
    fig.tight_layout()
    ensure_dir(output.parent)
    fig.savefig(output)
    plt.close(fig)


def trip_count(data: pd.DataFrame) -> int | None:
    if "source_id" not in data.columns:
        return None
    cols = ["source_id"] + (["source_file"] if "source_file" in data.columns else [])
    return int(data.dropna(subset=["source_id"]).drop_duplicates(cols).shape[0])


def driver_count(data: pd.DataFrame) -> int | None:
    if "driver_id" not in data.columns:
        return None
    return int(data["driver_id"].dropna().astype(str).nunique())


def select_example_trajectory(data: pd.DataFrame) -> pd.DataFrame:
    if "source_id" not in data.columns:
        return data.head(40).copy()
    group_cols = (["source_file"] if "source_file" in data.columns else []) + ["source_id"]
    counts = (
        data.groupby(group_cols, dropna=False)
        .agg(
            count=("lon", "size"),
            lon_min=("lon", "min"),
            lon_max=("lon", "max"),
            lat_min=("lat", "min"),
            lat_max=("lat", "max"),
        )
        .reset_index()
    )
    counts["span"] = np.maximum(counts["lon_max"] - counts["lon_min"], counts["lat_max"] - counts["lat_min"])
    preferred = counts.loc[counts["count"].between(20, 50) & counts["span"].between(0.006, 0.08)].copy()
    if preferred.empty:
        preferred = counts.loc[counts["count"].between(50, 250) & counts["span"].between(0.006, 0.12)].copy()
    if preferred.empty:
        preferred = counts.loc[counts["span"].gt(0.006)].sort_values("count", ascending=False).head(1)
    else:
        preferred["target_delta"] = (preferred["count"] - 35).abs()
        preferred = preferred.sort_values(["target_delta", "span", "count"])
    key = preferred.iloc[0]
    mask = pd.Series(True, index=data.index)
    for col in group_cols:
        mask &= data[col].astype(str).eq(str(key[col]))
    traj = data.loc[mask].copy()
    if "timestamp" in traj.columns:
        traj = traj.sort_values("timestamp")
    if len(traj) > 50:
        positions = np.linspace(0, len(traj) - 1, 40).round().astype(int)
        traj = traj.iloc[positions]
    return traj.head(50).copy()


def add_county_labels(ax) -> None:
    labels = [
        ("Palm Beach", -80.13, 26.62),
        ("Broward", -80.17, 26.39),
        ("Miami-Dade", -80.26, 26.20),
    ]
    for label, x, y in labels:
        ax.text(x, y, label, fontsize=10, weight="bold", color="#475569", alpha=0.55)


def save_methodology_scale_png(
    data: pd.DataFrame,
    output: Path,
    total_points: int,
    *,
    show_county_labels: bool = True,
) -> None:
    if data.empty:
        return
    lon = data["lon"].to_numpy(dtype=float)
    lat = data["lat"].to_numpy(dtype=float)
    x0, x1 = np.nanquantile(lon, [0.03, 0.995])
    y0, y1 = np.nanquantile(lat, [0.01, 0.995])
    xpad = max((x1 - x0) * 0.025, 0.002)
    ypad = max((y1 - y0) * 0.025, 0.002)

    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    # Display-only jitter separates heavily overlapping raw observations so the
    # presentation figure reads as millions of GPS points instead of a clean
    # road-centerline drawing. The source coordinates and counts are unchanged.
    rng = np.random.default_rng(1003)
    display_lon = lon + rng.normal(0, 0.0010, size=len(lon))
    display_lat = lat + rng.normal(0, 0.0010, size=len(lat))
    ax.scatter(display_lon, display_lat, s=18.0, c="#111827", alpha=0.022, linewidths=0, marker="o", rasterized=True)
    ax.scatter(display_lon, display_lat, s=4.0, c="#020617", alpha=0.075, linewidths=0, marker="o", rasterized=True)
    ax.set_xlim(x0 - xpad, x1 + xpad)
    ax.set_ylim(y0 - ypad, y1 + ypad)
    ax.grid(color="#e2e8f0", linewidth=0.55, alpha=0.65)
    ax.tick_params(axis="both", colors="#64748b", labelsize=8)
    ax.set_xlabel("Longitude", color="#64748b", fontsize=9)
    ax.set_ylabel("Latitude", color="#64748b", fontsize=9)
    if show_county_labels:
        add_county_labels(ax)

    fig.suptitle(
        f"Raw GPS Trajectory Dataset ({total_points / 1_000_000:.2f} Million Observations)",
        fontsize=24,
        weight="bold",
        y=0.985,
    )
    ax.set_title(
        "Each point represents one recorded GPS observation before map matching.",
        fontsize=15,
        color="#334155",
        pad=14,
    )

    ax.text(
        0.985,
        0.965,
        f"{total_points:,}\nobservations",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=22,
        weight="bold",
        color="#0f172a",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.45", "alpha": 0.94},
    )

    stats = [f"GPS observations: {total_points:,}"]
    trips = trip_count(data)
    drivers = driver_count(data)
    if trips is not None:
        stats.append(f"Individual trips: {trips:,}")
    if drivers:
        stats.append(f"Drivers: {drivers:,}")
    stats.append("")
    stats.append("Input to Fast Map Matching")
    ax.text(
        0.018,
        0.04,
        "Raw trajectory dataset\n\n" + "\n".join(f"• {line}" if line else "" for line in stats),
        transform=ax.transAxes,
        fontsize=11.5,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.5", "alpha": 0.96},
    )

    traj = select_example_trajectory(data)
    inset = inset_axes(ax, width="32%", height="35%", loc="upper left", borderpad=1.4)
    inset.set_facecolor("white")
    if not traj.empty:
        inset.plot(
            traj["lon"],
            traj["lat"],
            color="#94a3b8",
            linewidth=1.2,
            linestyle="--",
            alpha=0.85,
            zorder=1,
        )
        inset.scatter(
            traj["lon"],
            traj["lat"],
            s=30,
            color="#dc2626",
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
        )
        tx0, tx1 = traj["lon"].min(), traj["lon"].max()
        ty0, ty1 = traj["lat"].min(), traj["lat"].max()
        tpad = max(tx1 - tx0, ty1 - ty0, 0.001) * 0.18
        inset.set_xlim(tx0 - tpad, tx1 + tpad)
        inset.set_ylim(ty0 - tpad, ty1 + tpad)
    inset.set_title("Example raw trajectory", fontsize=10, weight="bold")
    inset.text(
        0.03,
        0.04,
        "Noisy GPS observations\nbefore map matching",
        transform=inset.transAxes,
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#e2e8f0", "alpha": 0.9},
    )
    inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in inset.spines.values():
        spine.set_edgecolor("#64748b")
        spine.set_linewidth(1.0)

    fig.text(
        0.5,
        0.018,
        "This raw trajectory dataset is the input to the Fast Map Matching algorithm, which converts noisy GPS observations into connected road-network paths.",
        ha="center",
        fontsize=10.5,
        color="#334155",
    )
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.10, top=0.82)
    ensure_dir(output.parent)
    fig.savefig(output)
    plt.close(fig)


def save_density_html(image_name: str, output: Path, total_points: int) -> None:
    body = f"""
<h1>Raw GPS Points Before Map Matching</h1>
<div class="panel">
  <div class="metric-grid">
    <div class="metric"><span>Full raw GPS point count</span><strong>{total_points:,}</strong></div>
  </div>
  <p>Main visualization uses all raw GPS points, not a sample. The overlap is not an error; it shows the scale and density of the trajectory dataset before map matching.</p>
  <img src="{image_name}" alt="Full raw GPS point density">
  <h2>Raster Density View</h2>
  <p>This secondary density raster also uses all raw GPS points.</p>
  <img src="raw_gps_all_points_density.png" alt="Full raw GPS point density raster">
</div>
"""
    write_html(output, "Raw GPS Points Before Map Matching", body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "raw_gps")
    parser.add_argument("--bins", type=int, default=1600, help="Raster grid cells per axis for the full-point density image.")
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    summary, monthly, data = load_counts_and_points()
    summary.to_csv(out_dir / "raw_gps_summary.csv", index=False)
    monthly.to_csv(out_dir / "raw_gps_monthly_counts.csv", index=False)

    total = int(summary.loc[summary["metric"] == "total_raw_gps_points", "value"].iloc[0]) if not summary.empty else 0
    save_google_style_marker_map(data, out_dir / "raw_gps_presentation_scale.png", total)
    save_google_style_marker_map(data, out_dir / "raw_gps_google_style_marker_map.png", total)
    save_methodology_scale_png(
        data,
        out_dir / "raw_gps_presentation_scale_no_county_color.png",
        total,
        show_county_labels=False,
    )
    save_big_overlap_png(data, out_dir / "raw_gps_all_points_big_overlap.png", total)
    save_dense_scatter_png(data, out_dir / "raw_gps_all_points_dense_scatter.png", total)
    save_density_png(data, out_dir / "raw_gps_all_points_density.png", total, args.bins)
    save_density_html("raw_gps_presentation_scale.png", out_dir / "raw_gps_all_points.html", total)
    save_density_html("raw_gps_presentation_scale.png", out_dir / "raw_gps_points.html", total)
    print(f"Raw GPS points: {total:,}")
    print("Main raw GPS visualization uses all raw GPS points, not a sample.")
    print(f"Wrote {out_dir.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
