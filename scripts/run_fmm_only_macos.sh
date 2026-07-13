#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERIFY_SCRIPT="${SCRIPT_DIR}/verify_pipeline_outputs.py"

usage() {
  cat <<'EOF'
Reuse cached enriched networks and run only FMM preparation/map matching.

Usage:
  scripts/run_fmm_only_macos.sh --config PATH [options]

Options:
  --config PATH         Local YAML configuration (required; skip_fmm must be false)
  --env-name NAME       Required active Conda environment (default: pipeline)
  --county NAME         Process one county; repeat for multiple counties
  --overwrite-matched   Explicitly allow county *_gps.csv/*_matched.csv replacement
  --reuse-matched       Reuse matched CSVs and rerun downstream aggregation only
  --log-dir DIR         Log directory (default: logs/pipeline)
  --dry-run             Validate inputs and print the command only
  -h, --help            Show this help and exit

The script never silently overwrites known-good matched CSVs. Select exactly
one of --overwrite-matched or --reuse-matched when such files already exist.
Cached enriched_network.parquet and OSM files are required.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

CONFIG=""
ENV_NAME="pipeline"
LOG_DIR="${REPO_ROOT}/logs/pipeline"
OVERWRITE=0
REUSE=0
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
    --overwrite-matched)
      OVERWRITE=1
      shift
      ;;
    --reuse-matched)
      REUSE=1
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
[[ "$OVERWRITE" -eq 0 || "$REUSE" -eq 0 ]] \
  || die "--overwrite-matched and --reuse-matched are mutually exclusive"
cd "$REPO_ROOT"
[[ -f "$CONFIG" ]] || die "configuration not found: $CONFIG"
[[ "$(uname -s)" == "Darwin" ]] || die "macOS is required"
[[ "$(uname -m)" == "arm64" ]] || die "Apple Silicon arm64 is required"
[[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]] \
  || die "activate the '$ENV_NAME' Conda environment first"
command -v roadnet-run >/dev/null 2>&1 || die "roadnet-run is not installed"
command -v fmm >/dev/null 2>&1 || die "fmm is not on PATH"
command -v ubodt_gen >/dev/null 2>&1 || die "ubodt_gen is not on PATH"

CONFIG_VALUES=()
while IFS= read -r value; do
  CONFIG_VALUES+=("$value")
done < <(python - "$CONFIG" <<'PY'
import pathlib, sys, yaml
raw = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
if raw.get("skip_fmm", False):
    raise SystemExit("config skip_fmm must be false for an FMM-only run")
output = raw.get("output_dir")
gps = raw.get("gps_root")
if not output or not gps:
    raise SystemExit("config output_dir and gps_root are required")
print(pathlib.Path(output).expanduser().resolve())
print(pathlib.Path(gps).expanduser().resolve())
for county in raw.get("counties", []):
    print(county.get("name", ""))
PY
)
OUTPUT_DIR="${CONFIG_VALUES[0]}"
GPS_ROOT="${CONFIG_VALUES[1]}"
CONFIG_COUNTIES=("${CONFIG_VALUES[@]:2}")
[[ -d "$GPS_ROOT" ]] || die "GPS root does not exist: $GPS_ROOT"
[[ -w "$GPS_ROOT" ]] \
  || die "GPS root is not writable; FMM writes caches and aggregated session outputs"

TARGET_COUNTIES=("${COUNTIES[@]}")
if [[ "${#TARGET_COUNTIES[@]}" -eq 0 ]]; then
  TARGET_COUNTIES=("${CONFIG_COUNTIES[@]}")
fi

EXISTING_MATCHED=()
for county in "${TARGET_COUNTIES[@]}"; do
  slug="${county// /_}"
  slug="${slug//-/_}"
  enriched="${OUTPUT_DIR}/${slug}/enriched_network.parquet"
  [[ -s "$enriched" ]] || die "cached enriched network is missing: $enriched"
  for cache_name in osm_nodes.parquet osm_edges.parquet osm_landuse.parquet; do
    cache_path="${OUTPUT_DIR}/${slug}/${cache_name}"
    [[ -s "$cache_path" ]] || die "cached OSM input is missing: $cache_path"
  done
  for suffix in shp shx dbf prj cpg; do
    edge_path="${OUTPUT_DIR}/${slug}/fmm/edges.${suffix}"
    [[ -s "$edge_path" ]] || die "cached FMM edge input is missing: $edge_path"
  done
  matched="${OUTPUT_DIR}/${slug}/${county}_matched.csv"
  if [[ -s "$matched" ]]; then
    EXISTING_MATCHED+=("$matched")
  fi
done
if [[ "${#EXISTING_MATCHED[@]}" -gt 0 && "$OVERWRITE" -eq 0 && "$REUSE" -eq 0 ]]; then
  die "existing matched outputs found; choose --overwrite-matched or --reuse-matched"
fi

PREFLIGHT_VERIFY_ARGS=(--config "$CONFIG" --stage enrichment)
FINAL_VERIFY_ARGS=(--config "$CONFIG" --stage matched)
if [[ "${#COUNTIES[@]}" -gt 0 ]]; then
  for county in "${COUNTIES[@]}"; do
    PREFLIGHT_VERIFY_ARGS+=(--county "$county")
    FINAL_VERIFY_ARGS+=(--county "$county")
  done
fi
if [[ "$REUSE" -eq 1 ]]; then
  FINAL_VERIFY_ARGS+=(--allow-reused-matched)
fi
python "$VERIFY_SCRIPT" "${PREFLIGHT_VERIFY_ARGS[@]}"

RUN_ARGS=("$CONFIG" --skip-osm --skip-mly --skip-conflation --disable-mapillary --log-level INFO)
if [[ "${#COUNTIES[@]}" -gt 0 ]]; then
  RUN_ARGS+=(--counties "${COUNTIES[@]}")
fi

printf 'Stage: FMM-only map matching using cached enriched networks\n'
printf 'Output root: %s\n' "$OUTPUT_DIR"
printf 'GPS root: %s\n' "$GPS_ROOT"
if [[ "$REUSE" -eq 1 ]]; then
  printf 'Matched CSV behavior: explicit reuse (ROADNET_FMM_REUSE_MATCHED=1)\n'
else
  printf 'Matched CSV behavior: explicit replacement or new creation\n'
fi
printf 'Command: roadnet-run %q' "${RUN_ARGS[0]}"
for argument in "${RUN_ARGS[@]:1}"; do printf ' %q' "$argument"; done
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'Dry run complete; no map matching was executed.\n'
  exit 0
fi

mkdir -p "$LOG_DIR"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="${LOG_DIR}/fmm_only_${TIMESTAMP}.log"
printf 'Log: %s\n' "$LOG_PATH"
if [[ "$REUSE" -eq 1 ]]; then
  ROADNET_FMM_REUSE_MATCHED=1 roadnet-run "${RUN_ARGS[@]}" 2>&1 | tee "$LOG_PATH"
else
  roadnet-run "${RUN_ARGS[@]}" 2>&1 | tee "$LOG_PATH"
fi

python "$VERIFY_SCRIPT" "${FINAL_VERIFY_ARGS[@]}"
printf 'FMM-only run completed and validated.\n'
