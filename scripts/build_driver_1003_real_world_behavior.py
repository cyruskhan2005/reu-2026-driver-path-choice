#!/usr/bin/env python3
"""Build place, routine, and OD-route insights for Driver 1003."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roadnet.real_world_behavior import (  # noqa: E402
    CURATED_REPORT_PATH,
    DEFAULT_CACHE_DIR,
    DEFAULT_OUTPUT_DIR,
    BehaviorAnalysisError,
    build_driver_1003_real_world_behavior,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--report", type=Path, default=CURATED_REPORT_PATH)
    parser.add_argument(
        "--pipeline-output-root",
        type=Path,
        default=ROOT / "sflorida_outputs",
        help="Phase 1/2 output root containing county GPS and driver timeline files.",
    )
    parser.add_argument(
        "--phase2-deliverable-root",
        type=Path,
        default=ROOT / "deliverables" / "google_drive_phase2",
        help="Phase 2 deliverable root containing combined monthly road-node data.",
    )
    parser.add_argument(
        "--max-non-home-clusters",
        type=int,
        default=20,
        help="Maximum unique meaningful non-home clusters sent to Google (default: 20).",
    )
    parser.add_argument(
        "--google-request-budget",
        type=int,
        default=90,
        help="Hard HTTP-attempt budget for this invocation; never allowed above 100.",
    )
    parser.add_argument(
        "--prior-google-requests",
        type=int,
        default=0,
        help="Requests already made in the surrounding run (for example access tests).",
    )
    parser.add_argument(
        "--access-test-requests",
        type=int,
        default=0,
        help="Subset of prior requests used by the one-time API access checks.",
    )
    parser.add_argument(
        "--skip-google",
        action="store_true",
        help="Use only cached local OSM/GIS context.",
    )
    parser.add_argument(
        "--google-cache-only",
        action="store_true",
        help="Reuse sanitized Google cache entries but never send a network request.",
    )
    parser.add_argument(
        "--skip-report-update",
        action="store_true",
        help="Generate analysis artifacts without changing the curated RCCI HTML.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_non_home_clusters < 1:
        print("ERROR: --max-non-home-clusters must be positive", file=sys.stderr)
        return 2
    if args.skip_google and args.google_cache_only:
        print("ERROR: --skip-google and --google-cache-only are mutually exclusive", file=sys.stderr)
        return 2
    try:
        result = build_driver_1003_real_world_behavior(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            report_path=args.report,
            update_report=not args.skip_report_update,
            use_google=not args.skip_google,
            google_cache_only=args.google_cache_only,
            max_non_home_clusters=args.max_non_home_clusters,
            google_request_budget=args.google_request_budget,
            prior_google_requests=args.prior_google_requests,
            access_test_requests=args.access_test_requests,
            pipeline_output_root=args.pipeline_output_root,
            phase2_deliverable_root=args.phase2_deliverable_root,
        )
    except (BehaviorAnalysisError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Driver 1003 real-world behavior build complete")
    print(f"  source trips: {result.source_trip_count:,}")
    print(f"  county fragments: {result.county_fragment_count:,}")
    print(f"  selected cluster radius: {result.selected_cluster_radius_m:.0f} m")
    print(f"  Google requests (including prior tests): {result.google_requests}")
    print(f"  cache hits: {result.cache_hits}")
    print(f"  likely home cluster: {result.likely_home_cluster_id} (publicly generalized)")
    print(f"  privacy checks passed: {result.privacy_checks_passed}")
    for label, path in result.paths.items():
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
