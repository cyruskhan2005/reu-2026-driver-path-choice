#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERIFY_SCRIPT="${SCRIPT_DIR}/verify_pipeline_outputs.py"

usage() {
  cat <<'EOF'
Run the complete road enrichment and FMM workflow on Apple Silicon macOS.

Usage:
  scripts/run_full_pipeline_macos.sh --config PATH [options]

Options:
  --config PATH       Sanitized local YAML configuration (required)
  --env-name NAME     Required active Conda environment (default: pipeline)
  --county NAME       Process one county; repeat for multiple counties
  --resume            Permit existing enrichment outputs; YAML skip flags decide reuse
  --log-dir DIR       Log directory (default: logs/pipeline)
  --dry-run           Print validated command without running the pipeline
  -h, --help          Show this help and exit

By default the script refuses to run over an output tree containing an existing
enriched network. Use --resume only after reviewing the YAML cache/skip flags.
The script never reads or prints Google credentials. Mapillary behavior comes
from mapillary_enabled and mly_token in the ignored local config file.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

CONFIG=""
ENV_NAME="pipeline"
LOG_DIR="${REPO_ROOT}/logs/pipeline"
RESUME=0
DRY_RUN=0
COUNTIES=()

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
    --county)
      [[ $# -ge 2 ]] || die "--county requires a county name"
      COUNTIES+=("$2")
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --log-dir)
      [[ $# -ge 2 ]] || die "--log-dir requires a directory"
      LOG_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
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
cd "$REPO_ROOT"
[[ -f "$CONFIG" ]] || die "configuration not found: $CONFIG"
[[ -f "$VERIFY_SCRIPT" ]] || die "verification script not found: $VERIFY_SCRIPT"
[[ "$(uname -s)" == "Darwin" ]] || die "macOS is required"
[[ "$(uname -m)" == "arm64" ]] || die "Apple Silicon arm64 is required"
[[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]] \
  || die "activate the '$ENV_NAME' Conda environment first"
command -v python >/dev/null 2>&1 || die "python is not on PATH"
command -v roadnet-run >/dev/null 2>&1 || die "roadnet-run is not installed in the active environment"
command -v fmm >/dev/null 2>&1 || die "fmm is not on PATH; run scripts/bootstrap_macos.sh"
command -v ubodt_gen >/dev/null 2>&1 || die "ubodt_gen is not on PATH"

OUTPUT_DIR="$(python - "$CONFIG" <<'PY'
import pathlib, sys, yaml
raw = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
value = raw.get("output_dir")
if not value:
    raise SystemExit("config output_dir is missing")
print(pathlib.Path(value).expanduser().resolve())
PY
)"

GPS_ROOT="$(python - "$CONFIG" "$RESUME" <<'PY'
import pathlib, sys, yaml
raw = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
resume = sys.argv[2] == "1"
if not resume:
    enabled_skips = [
        key for key in ("skip_osm", "skip_mly", "skip_conflation", "skip_fmm")
        if raw.get(key, False)
    ]
    if enabled_skips:
        raise SystemExit(
            "full clean rebuild requires skip flags false; enabled: "
            + ", ".join(enabled_skips)
        )
gps = raw.get("gps_root")
if not gps:
    raise SystemExit("config gps_root is required for the full FMM workflow")
print(pathlib.Path(gps).expanduser().resolve())
PY
)"
[[ -d "$GPS_ROOT" ]] || die "GPS root does not exist: $GPS_ROOT"
[[ -w "$GPS_ROOT" ]] \
  || die "GPS root is not writable; FMM creates master/cache and aggregated files there"

python "$VERIFY_SCRIPT" --config "$CONFIG" --stage environment

if [[ "$RESUME" -eq 0 && -d "$OUTPUT_DIR" ]]; then
  EXISTING="$(find "$OUTPUT_DIR" -name enriched_network.parquet -type f -print -quit 2>/dev/null || true)"
  [[ -z "$EXISTING" ]] \
    || die "existing enrichment output detected; use a new output_dir or pass --resume after reviewing skip flags"
fi

RUN_ARGS=("$CONFIG" --log-level INFO)
VERIFY_ARGS=(--config "$CONFIG" --stage matched)
if [[ "${#COUNTIES[@]}" -gt 0 ]]; then
  RUN_ARGS+=(--counties "${COUNTIES[@]}")
  for county in "${COUNTIES[@]}"; do
    VERIFY_ARGS+=(--county "$county")
  done
fi

printf 'Stage: full road enrichment and FMM map matching\n'
printf 'Repository: %s\n' "$REPO_ROOT"
printf 'Output root: %s\n' "$OUTPUT_DIR"
printf 'Resume mode: %s\n' "$([[ "$RESUME" -eq 1 ]] && printf enabled || printf disabled)"
printf 'Command: roadnet-run %q' "${RUN_ARGS[0]}"
for argument in "${RUN_ARGS[@]:1}"; do printf ' %q' "$argument"; done
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'Dry run complete; no pipeline stage was executed.\n'
  exit 0
fi

mkdir -p "$LOG_DIR"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="${LOG_DIR}/full_pipeline_${TIMESTAMP}.log"
printf 'Log: %s\n' "$LOG_PATH"
roadnet-run "${RUN_ARGS[@]}" 2>&1 | tee "$LOG_PATH"

python "$VERIFY_SCRIPT" "${VERIFY_ARGS[@]}"
printf 'Full pipeline completed and validated.\n'
printf 'Safe to rerun only with --resume and reviewed YAML skip flags.\n'
