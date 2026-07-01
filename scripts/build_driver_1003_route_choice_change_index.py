#!/usr/bin/env python3
"""Build Driver 1003 Route Choice Change Index (RCCI) outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roadnet.driver_timeline import DriverTimelineError  # noqa: E402
from roadnet.route_choice_change_index import (  # noqa: E402
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    build_driver_1003_rcci,
    normalize_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Driver 1003 Route Choice Change Index from cached "
            "Phase 2C graph comparison outputs."
        )
    )
    parser.add_argument("--driver", default="1003")
    parser.add_argument("--node-weight", type=float, default=0.5)
    parser.add_argument("--edge-weight", type=float, default=0.5)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--county",
        help="Optional exact county name to process. Default processes all county-specific rows.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Process all county-specific rows. This is the default when --county "
            "is not supplied."
        ),
    )
    parser.add_argument(
        "--include-all-counties",
        action="store_true",
        help="Also include ALL_COUNTIES aggregate rows. Not recommended for primary RCCI.",
    )
    parser.add_argument(
        "--report-county",
        default="Broward County",
        help="County highlighted in the report timeline.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    county = None if args.all else args.county
    try:
        node_weight, edge_weight = normalize_weights(
            args.node_weight,
            args.edge_weight,
        )
        result = build_driver_1003_rcci(
            driver=args.driver,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            node_weight=args.node_weight,
            edge_weight=args.edge_weight,
            county=county,
            include_all_counties=args.include_all_counties,
            report_county=args.report_county,
        )
    except (DriverTimelineError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Driver 1003 RCCI build complete")
    print(f"  normalized node weight: {node_weight:.3f}")
    print(f"  normalized edge weight: {edge_weight:.3f}")
    print(f"  RCCI rows: {result.rows:,}")
    print(f"  confidence counts: {result.confidence_counts}")
    print(f"  validation passed: {result.validation_passed}")
    print(f"  summary CSV: {result.summary_csv}")
    print(f"  summary parquet: {result.summary_parquet}")
    print(f"  sensitivity CSV: {result.sensitivity_csv}")
    print(f"  sensitivity parquet: {result.sensitivity_parquet}")
    print(f"  report HTML: {result.report_html}")
    print(f"  validation report: {result.validation_report}")
    return 0 if result.validation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
