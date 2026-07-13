#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RCCI_SCRIPT="${SCRIPT_DIR}/build_driver_1003_route_choice_change_index.py"
BEHAVIOR_SCRIPT="${SCRIPT_DIR}/build_driver_1003_real_world_behavior.py"
VERIFY_SCRIPT="${SCRIPT_DIR}/verify_pipeline_outputs.py"
PACKAGE_SCRIPT="${SCRIPT_DIR}/package_driver_1003_public_release.py"
MANIFEST_SCRIPT="${SCRIPT_DIR}/generate_output_manifest.py"
PUBLIC_VALIDATE_SCRIPT="${SCRIPT_DIR}/validate_public_release.py"
TIMELINE_SCRIPT="${SCRIPT_DIR}/build_driver_timeline.py"
MONTHLY_SCRIPT="${SCRIPT_DIR}/build_driver_1003_monthly_graphs.py"
COMPARISON_SCRIPT="${SCRIPT_DIR}/compare_driver_1003_monthly_graphs.py"

usage() {
  cat <<'EOF'
Regenerate the Driver 1003 RCCI and longitudinal behavior deliverables.

Usage:
  scripts/generate_driver_1003_report.sh --config PATH [options]

Options:
  --config PATH          Pipeline config used by output verification (required)
  --env-name NAME        Required active Conda environment (default: pipeline)
  --google-mode MODE     offline, cache, or network (default: offline)
  --prior-requests N     Google attempts already made in this run (default: 0)
  --google-budget N      Maximum new attempts in network mode (default: 20)
  --max-clusters N       Maximum meaningful non-home clusters (default: 20)
  --output-dir DIR       Behavior output directory (default: outputs)
  --cache-dir DIR        Sanitized Google cache directory (default: cache/google_maps)
  --log-dir DIR          Log directory (default: logs/reports)
  --reuse-phase2         Reuse validated Phase 2A/B/C tables instead of rebuilding them
  -h, --help             Show this help and exit

offline uses only repository OSM/GIS context. cache reuses sanitized Google
responses without network access. network requires GOOGLE_MAPS_API_KEY in the
environment and enforces prior + budget <= 99. The key is never printed.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

CONFIG=""
ENV_NAME="pipeline"
GOOGLE_MODE="offline"
PRIOR_REQUESTS=0
GOOGLE_BUDGET=20
MAX_CLUSTERS=20
OUTPUT_DIR="${REPO_ROOT}/outputs"
CACHE_DIR="${REPO_ROOT}/cache/google_maps"
LOG_DIR="${REPO_ROOT}/logs/reports"
REPORT_PATH="${REPO_ROOT}/deliverables/driver_1003/route_choice_change_index/visuals/driver_1003_route_choice_change_index_report.html"
PHASE2_DELIVERABLE_ROOT="${REPO_ROOT}/deliverables/google_drive_phase2"
CANONICAL_RCCI_ROOT="${REPO_ROOT}/deliverables/driver_1003/route_choice_change_index"
REUSE_PHASE2=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || die "--config requires a path"
      CONFIG="$2"
      shift 2
      ;;
    --env-name)
      [[ $# -ge 2 ]] || die "--env-name requires a name"
      ENV_NAME="$2"
      shift 2
      ;;
    --google-mode)
      [[ $# -ge 2 ]] || die "--google-mode requires offline, cache, or network"
      GOOGLE_MODE="$2"
      shift 2
      ;;
    --prior-requests)
      [[ $# -ge 2 ]] || die "--prior-requests requires an integer"
      PRIOR_REQUESTS="$2"
      shift 2
      ;;
    --google-budget)
      [[ $# -ge 2 ]] || die "--google-budget requires an integer"
      GOOGLE_BUDGET="$2"
      shift 2
      ;;
    --max-clusters)
      [[ $# -ge 2 ]] || die "--max-clusters requires an integer"
      MAX_CLUSTERS="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires a directory"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --cache-dir)
      [[ $# -ge 2 ]] || die "--cache-dir requires a directory"
      CACHE_DIR="$2"
      shift 2
      ;;
    --log-dir)
      [[ $# -ge 2 ]] || die "--log-dir requires a directory"
      LOG_DIR="$2"
      shift 2
      ;;
    --reuse-phase2)
      REUSE_PHASE2=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (use --help)"
      ;;
  esac
done

[[ -n "$CONFIG" ]] || die "--config is required"
case "$GOOGLE_MODE" in
  offline|cache|network) ;;
  *) die "--google-mode must be offline, cache, or network" ;;
esac
for value in "$PRIOR_REQUESTS" "$GOOGLE_BUDGET" "$MAX_CLUSTERS"; do
  case "$value" in
    ''|*[!0-9]*) die "request counts and cluster count must be nonnegative integers" ;;
  esac
done
[[ "$MAX_CLUSTERS" -gt 0 ]] || die "--max-clusters must be positive"
[[ $((PRIOR_REQUESTS + GOOGLE_BUDGET)) -le 99 ]] \
  || die "prior requests plus Google budget must be at most 99"

cd "$REPO_ROOT"
[[ -f "$CONFIG" ]] || die "configuration not found: $CONFIG"
[[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]] \
  || die "activate the '$ENV_NAME' Conda environment first"
if [[ "$GOOGLE_MODE" == "network" && -z "${GOOGLE_MAPS_API_KEY:-}" ]]; then
  die "GOOGLE_MAPS_API_KEY is not set"
fi

PIPELINE_OUTPUT_ROOT="$(python - "$CONFIG" <<'PY'
import pathlib, sys, yaml
raw = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
value = raw.get("output_dir")
if not value:
    raise SystemExit("config output_dir is missing")
print(pathlib.Path(value).expanduser().resolve())
PY
)"

mkdir -p "$LOG_DIR"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="${LOG_DIR}/driver_1003_report_${TIMESTAMP}.log"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/roadnet-matplotlib"
mkdir -p "$MPLCONFIGDIR"

if [[ "$REUSE_PHASE2" -eq 0 ]]; then
  printf 'Stage 1/7: rebuild Phase 2A Driver 1003 timeline from matched outputs\n'
  python "$TIMELINE_SCRIPT" \
    --driver auto \
    --config "$CONFIG" \
    --output-dir "$PIPELINE_OUTPUT_ROOT" 2>&1 | tee "$LOG_PATH"

  printf 'Stage 2/7: rebuild Phase 2B monthly attributed graphs\n'
  python "$MONTHLY_SCRIPT" \
    --driver 1003 \
    --output-dir "$PIPELINE_OUTPUT_ROOT" 2>&1 | tee -a "$LOG_PATH"

  printf 'Stage 3/7: rebuild Phase 2C consecutive-month comparisons\n'
  python "$COMPARISON_SCRIPT" \
    --driver 1003 \
    --all \
    --graph-root "$PHASE2_DELIVERABLE_ROOT/driver_1003_monthly_graphs" \
    --manifest "$PIPELINE_OUTPUT_ROOT/phase2/monthly_graphs/driver_1003/monthly_graph_manifest.csv" \
    --output-dir "$PHASE2_DELIVERABLE_ROOT/driver_1003_graph_comparisons" \
    2>&1 | tee -a "$LOG_PATH"
else
  printf 'Stages 1-3/7: reuse requested; validating Phase 2A/B/C prerequisites\n'
  for required in \
    "$PIPELINE_OUTPUT_ROOT/phase2/driver_timelines/driver_1_timeline.csv" \
    "$PHASE2_DELIVERABLE_ROOT/driver_1003_monthly_graphs/data/driver_1003_all_monthly_nodes.csv" \
    "$PHASE2_DELIVERABLE_ROOT/driver_1003_graph_comparisons/data/driver_1003_month_to_month_summary.csv"; do
    [[ -s "$required" ]] || die "required reusable Phase 2 file is missing: $required"
  done
  : >"$LOG_PATH"
fi

printf 'Stage 4/7: rebuild canonical RCCI technical outputs\n'
python "$RCCI_SCRIPT" \
  --driver 1003 \
  --all \
  --input-dir "$PHASE2_DELIVERABLE_ROOT/driver_1003_graph_comparisons/data" \
  --output-dir "$CANONICAL_RCCI_ROOT" 2>&1 | tee -a "$LOG_PATH"

BEHAVIOR_ARGS=(
  --output-dir "$OUTPUT_DIR"
  --cache-dir "$CACHE_DIR"
  --report "$REPORT_PATH"
  --max-non-home-clusters "$MAX_CLUSTERS"
  --prior-google-requests "$PRIOR_REQUESTS"
  --google-request-budget "$GOOGLE_BUDGET"
  --pipeline-output-root "$PIPELINE_OUTPUT_ROOT"
  --phase2-deliverable-root "$PHASE2_DELIVERABLE_ROOT"
)
case "$GOOGLE_MODE" in
  offline) BEHAVIOR_ARGS+=(--skip-google) ;;
  cache) BEHAVIOR_ARGS+=(--google-cache-only) ;;
  network) ;;
esac

printf 'Stage 5/7: rebuild place, routine, and longitudinal route analysis (%s Google mode)\n' "$GOOGLE_MODE"
python "$BEHAVIOR_SCRIPT" "${BEHAVIOR_ARGS[@]}" 2>&1 | tee -a "$LOG_PATH"

printf 'Stage 6/7: validate Driver 1003 outputs and public secret boundary\n'
python "$VERIFY_SCRIPT" \
  --config "$CONFIG" \
  --stage driver \
  --driver-output-dir "$OUTPUT_DIR" \
  --report "$REPORT_PATH" 2>&1 | tee -a "$LOG_PATH"

printf 'Stage 7/7: package and validate curated public artifacts\n'
python "$PACKAGE_SCRIPT" --public-dir "$REPO_ROOT/outputs/public" 2>&1 | tee -a "$LOG_PATH"
python "$MANIFEST_SCRIPT" \
  --root "$REPO_ROOT/outputs/public" \
  --output "$REPO_ROOT/outputs/public/manifest.json" 2>&1 | tee -a "$LOG_PATH"
python "$PUBLIC_VALIDATE_SCRIPT" \
  --public-dir "$REPO_ROOT/outputs/public" \
  --manifest "$REPO_ROOT/outputs/public/manifest.json" \
  --require-manifest 2>&1 | tee -a "$LOG_PATH"

printf 'Driver 1003 report generation completed.\n'
printf 'Report: %s\n' "$REPORT_PATH"
printf 'Map: %s/driver_1003_poi_route_insights_map.html\n' "$OUTPUT_DIR"
printf 'Log: %s\n' "$LOG_PATH"
