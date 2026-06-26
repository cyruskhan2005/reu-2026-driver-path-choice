#!/usr/bin/env python3
"""Compare consecutive Driver 1003 monthly attributed graphs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roadnet.driver_timeline import DriverTimelineError  # noqa: E402
from roadnet.graph_comparisons import (  # noqa: E402
    DEFAULT_GRAPH_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
    compare_driver_1003_monthly_graphs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare cached Driver 1003 monthly FID-node and directed-transition "
            "graphs without rerunning map matching."
        )
    )
    parser.add_argument("--driver", default="1003")
    parser.add_argument("--county", help="Optional exact county name")
    parser.add_argument("--month-a", help="First month in YYYY-MM format")
    parser.add_argument("--month-b", help="Consecutive second month in YYYY-MM format")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compare every consecutive calendar month (default)",
    )
    parser.add_argument("--graph-root", default=str(DEFAULT_GRAPH_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compare_all = args.all or not (args.month_a or args.month_b)
    try:
        result = compare_driver_1003_monthly_graphs(
            driver=args.driver,
            graph_root=args.graph_root,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            county=args.county,
            month_a=args.month_a,
            month_b=args.month_b,
            compare_all=compare_all,
        )
    except (DriverTimelineError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Driver 1003 graph comparison build complete")
    print(f"  calendar month pairs: {result.month_pair_count}")
    print(f"  county comparisons: {result.county_comparison_count}")
    print(f"  node comparison rows: {result.node_comparison_rows:,}")
    print(f"  edge comparison rows: {result.edge_comparison_rows:,}")
    print(f"  summary rows: {result.summary_rows:,}")
    print(f"  validation passed: {result.validation_passed}")
    print(f"  summary: {result.summary_csv}")
    print(f"  node details: {result.node_comparison_csv}")
    print(f"  edge details: {result.edge_comparison_csv}")
    print(f"  overview: {result.overview_html}")
    if result.detailed_html:
        print(f"  detailed comparison: {result.detailed_html}")
    if result.detailed_map_html:
        print(f"  comparison map: {result.detailed_map_html}")
    print(f"  county comparison pages: {len(result.comparison_map_htmls):,}")
    for path in result.comparison_map_htmls:
        print(f"    - {path}")
    print(f"  validation: {result.validation_report}")
    return 0 if result.validation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
