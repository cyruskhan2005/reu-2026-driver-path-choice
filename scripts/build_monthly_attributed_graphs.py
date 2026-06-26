#!/usr/bin/env python3
"""Build Driver 1003 monthly attributed road graphs from Phase 2A outputs."""
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
            "Build one directed attributed road graph per observed month/county "
            "for Driver 1003. FMM is not rerun."
        )
    )
    parser.add_argument("--subject", default="1003")
    parser.add_argument("--output-dir", default="sflorida_outputs")
    parser.add_argument("--month", help="Optional YYYY-MM filter")
    parser.add_argument("--county", help="Optional exact county-name filter")
    parser.add_argument(
        "--top-edges",
        type=int,
        default=250,
        help="Maximum transitions shown in each optional map overlay",
    )
    parser.add_argument(
        "--prepare-drive-bundle",
        action="store_true",
        help="Prepare the privacy-checked local Google Drive upload bundle",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_driver_1003_monthly_graphs(
            subject=args.subject,
            output_dir=args.output_dir,
            prepare_drive_bundle=args.prepare_drive_bundle,
            month=args.month,
            county=args.county,
            top_edges=args.top_edges,
        )
    except DriverTimelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Driver 1003 monthly attributed graph build complete")
    print(
        f"  months: {result.observed_month_count} observed / "
        f"{result.calendar_month_count} calendar"
    )
    print(f"  GraphML graphs: {result.graphml_count}")
    print(f"  monthly maps: {result.map_count}")
    print(f"  monthly FID-node datasets: {result.fid_node_dataset_count}")
    print(f"  monthly transition-edge datasets: {result.fid_edge_dataset_count}")
    print(f"  total monthly FID nodes: {result.total_monthly_fid_nodes:,}")
    print(f"  total monthly directed edges: {result.total_monthly_fid_edges:,}")
    print(f"  subject manifest: {result.subject_manifest_path}")
    print(f"  monthly FID usage CSV: {result.monthly_fid_usage_csv_path}")
    print(f"  monthly FID usage GeoParquet: {result.monthly_fid_usage_parquet_path}")
    print(f"  graph manifest: {result.monthly_graph_manifest_path}")
    print(f"  unmatched FIDs: {result.unmatched_fids_path}")
    print(f"  visual overview: {result.visual_overview_path}")
    print(f"  FID graph validation: {result.fid_graph_validation_path}")
    if result.proof_graph_path:
        print(f"  2023-08 proof graph: {result.proof_graph_path}")
    print(f"  graph root: {result.graph_root}")
    if result.upload_bundle_root:
        print(f"  Google Drive bundle: {result.upload_bundle_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
