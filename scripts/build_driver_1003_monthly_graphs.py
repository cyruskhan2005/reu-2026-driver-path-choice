#!/usr/bin/env python3
"""Build Driver 1003 FID-node and transition-edge monthly deliverables."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roadnet.driver_timeline import DriverTimelineError  # noqa: E402
from roadnet.monthly_attributed_graphs import (  # noqa: E402
    build_driver_1003_monthly_graphs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Driver 1003 monthly attributed graphs with FIDs as nodes "
            "and consecutive matched-FID transitions as directed edges."
        )
    )
    parser.add_argument("--driver", default="1003")
    parser.add_argument("--month", help="Optional YYYY-MM filter")
    parser.add_argument("--county", help="Optional exact county-name filter")
    parser.add_argument("--output-dir", default="sflorida_outputs")
    parser.add_argument(
        "--top-edges",
        type=int,
        default=250,
        help="Maximum transition edges displayed in monthly map overlays",
    )
    parser.add_argument(
        "--no-drive-bundle",
        action="store_true",
        help="Skip rebuilding deliverables/google_drive_phase2",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_driver_1003_monthly_graphs(
            subject=args.driver,
            output_dir=args.output_dir,
            prepare_drive_bundle=not args.no_drive_bundle,
            month=args.month,
            county=args.county,
            top_edges=args.top_edges,
        )
    except DriverTimelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Driver 1003 FID transition graph build complete")
    print(
        f"  months: {result.observed_month_count} observed / "
        f"{result.calendar_month_count} calendar"
    )
    print(
        f"  monthly node datasets: {result.fid_node_dataset_count} "
        f"({result.total_monthly_fid_nodes:,} node rows)"
    )
    print(
        f"  monthly edge datasets: {result.fid_edge_dataset_count} "
        f"({result.total_monthly_fid_edges:,} directed edge rows)"
    )
    print(f"  validation: {result.fid_graph_validation_path}")
    print(f"  overview: {result.visual_overview_path}")
    if result.proof_graph_path:
        print(f"  proof graph: {result.proof_graph_path}")
    if result.upload_bundle_root:
        print(f"  Drive bundle: {result.upload_bundle_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
