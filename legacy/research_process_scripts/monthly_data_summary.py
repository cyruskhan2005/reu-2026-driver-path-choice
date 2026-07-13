#!/usr/bin/env python3
"""Create monthly raw GPS, trip, and matched segment usage summaries."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from research_process_utils import (
    OUTPUT_ROOT,
    dataframe_preview_html,
    ensure_dir,
    explode_segment_observations,
    iter_raw_gps,
    segment_trip_counts,
    write_html,
)


def save_bar(data: pd.DataFrame, x: str, y: str, title: str, output: Path) -> None:
    if data.empty or x not in data.columns or y not in data.columns:
        return
    plot_data = data.groupby(x, dropna=False)[y].sum().reset_index().sort_values(x)
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    ax.bar(plot_data[x].astype(str), plot_data[y], color="#2563eb")
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", color="#e5e7eb")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def wrapper_html(title: str, image_name: str, table_path: Path, output: Path) -> None:
    body = f"<h1>{title}</h1><div class='panel'><img src='{image_name}' alt='{title}'></div><div class='panel'>{dataframe_preview_html(table_path)}</div>"
    write_html(output, title, body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "monthly_summary")
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    raw_parts: list[pd.DataFrame] = []
    trip_parts: list[pd.DataFrame] = []
    trip_cols: set[str] = set()
    for chunk in iter_raw_gps():
        if "month" not in chunk.columns:
            continue
        raw_parts.append(chunk.dropna(subset=["month"]).groupby(["county", "month"], dropna=False).size().rename("raw_gps_points").reset_index())
        if "source_id" in chunk.columns:
            trip_parts.append(chunk.dropna(subset=["month"]).drop_duplicates(["county", "month", "source_id"])[["county", "month", "source_id"]])
            trip_cols.add("source_id")
        if "driver_id" in chunk.columns:
            trip_cols.add("driver_id")
    if raw_parts:
        raw_counts = pd.concat(raw_parts, ignore_index=True).groupby(["county", "month"], dropna=False)["raw_gps_points"].sum().reset_index()
    else:
        print("WARNING: raw GPS timestamp/month column was not available.")
        raw_counts = pd.DataFrame(columns=["county", "month", "raw_gps_points"])
    raw_counts.to_csv(out_dir / "monthly_raw_gps_counts.csv", index=False)

    if trip_parts:
        trip_counts = (
            pd.concat(trip_parts, ignore_index=True)
            .drop_duplicates(["county", "month", "source_id"])
            .groupby(["county", "month"], dropna=False)
            .size()
            .rename("trip_or_session_count")
            .reset_index()
        )
    else:
        trip_counts = pd.DataFrame(columns=["county", "month", "trip_or_session_count"])
    trip_counts.to_csv(out_dir / "monthly_trip_counts.csv", index=False)

    observations = explode_segment_observations(prefer_timeline=True)
    segment_counts = segment_trip_counts(observations, by_period=True)
    if "period" in segment_counts.columns:
        segment_counts = segment_counts.rename(columns={"period": "month"})
    segment_counts.to_csv(out_dir / "monthly_segment_pass_counts.csv", index=False)

    save_bar(raw_counts, "month", "raw_gps_points", "Monthly Raw GPS Point Counts", out_dir / "monthly_raw_gps_histogram.png")
    wrapper_html("Monthly Raw GPS Point Counts", "monthly_raw_gps_histogram.png", out_dir / "monthly_raw_gps_counts.csv", out_dir / "monthly_raw_gps_histogram.html")
    if not segment_counts.empty:
        month_totals = segment_counts.groupby(["county", "month"], dropna=False)["trip_use_count"].sum().reset_index()
        save_bar(month_totals, "month", "trip_use_count", "Monthly Matched Segment Usage", out_dir / "monthly_segment_pass_histogram.png")
        wrapper_html("Monthly Matched Segment Usage", "monthly_segment_pass_histogram.png", out_dir / "monthly_segment_pass_counts.csv", out_dir / "monthly_segment_pass_histogram.html")
    print(f"Raw monthly rows: {len(raw_counts):,}")
    print(f"Matched segment count rows: {len(segment_counts):,}")
    if trip_cols:
        print(f"Detected raw GPS identifier columns: {', '.join(sorted(trip_cols))}")
    print(f"Wrote {out_dir.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
