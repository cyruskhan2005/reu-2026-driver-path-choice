#!/usr/bin/env python3
"""Build one or more Phase 2A Driver/Session timelines from cached outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roadnet.driver_timeline import (  # noqa: E402
    DEFAULT_MATCH_TOLERANCE_SECONDS,
    DEFAULT_TIMEZONE,
    DriverTimelineError,
    build_and_export_driver_timelines,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile cached matched routes into Driver/Session timelines, "
            "monthly summaries, aliases, and HTML views. FMM is not rerun."
        )
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--driver",
        default=None,
        help="Driver/session hash, unique hash prefix, or 'auto' for most usable trips",
    )
    selection.add_argument(
        "--top-n",
        type=int,
        help="Build outputs for the top N ranked Driver/Session groupings",
    )
    parser.add_argument(
        "--gps-root",
        help=(
            "Raw/cached session hierarchy. If omitted, use config.yaml gps_root "
            "or /Volumes/KINGSTON when available."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="sflorida_outputs",
        help="Phase 1 county output root and destination for phase2 outputs",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument(
        "--match-tolerance-seconds",
        type=int,
        default=DEFAULT_MATCH_TOLERANCE_SECONDS,
        help="Maximum difference between source filename time and cached GPS start",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_and_export_driver_timelines(
            driver=args.driver or "auto",
            top_n=args.top_n,
            output_dir=args.output_dir,
            gps_root=args.gps_root,
            config_path=args.config,
            repo_root=ROOT,
            timezone_name=args.timezone,
            tolerance_seconds=args.match_tolerance_seconds,
        )
    except DriverTimelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Phase 2A Driver/Session timeline build complete")
    print(f"  session root: {result.gps_root}")
    print(f"  selected groupings: {len(result.selected_outputs)}")
    for output in result.selected_outputs:
        print(
            f"    - {output.driver_alias}: {output.trip_count:,} trips, "
            f"{output.observed_month_count} observed months, "
            f"{output.first_month} through {output.last_month}"
        )
    print(
        "  source mapping: "
        f"{result.mapped_source_trips:,} usable / "
        f"{result.discovered_source_trips:,} discovered; "
        f"{result.skipped_source_trips:,} skipped"
    )
    print("  matched inputs:")
    for path in result.matched_paths:
        print(f"    - {path}")
    print("  GPS metadata inputs:")
    for path in result.gps_paths:
        print(f"    - {path}")
    print("  population outputs:")
    print(f"    identity audit: {result.identity_audit_path}")
    print(f"    alias map: {result.alias_map_path}")
    print(f"    population index: {result.population_index_path}")
    print(f"    overview: {result.population_overview_path}")
    print("  timeline outputs:")
    for output in result.selected_outputs:
        print(f"    {output.driver_alias} timeline: {output.timeline_path}")
        print(f"    {output.driver_alias} monthly summary: {output.monthly_summary_path}")
        print(f"    {output.driver_alias} visual: {output.visual_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
