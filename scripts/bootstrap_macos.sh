#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/environment-macos.yml"
BUILD_SCRIPT="${SCRIPT_DIR}/build_fmm_macos.sh"

usage() {
  cat <<'EOF'
Bootstrap the roadnet research environment on Apple Silicon macOS.

Usage:
  scripts/bootstrap_macos.sh [options]

Options:
  --env-name NAME    Conda environment name (default: pipeline)
  --work-dir DIR     FMM build workspace (default: build/fmm-macos)
  --jobs N           Parallel FMM build jobs
  --skip-brew        Do not install missing Homebrew build dependencies
  --skip-fmm         Create/update the Python environment without building FMM
  -h, --help         Show this help and exit

This script never reads or writes API keys. Mapillary can be disabled in the
pipeline configuration with `mapillary_enabled: false`.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

ENV_NAME="pipeline"
WORK_DIR="${REPO_ROOT}/build/fmm-macos"
JOBS=""
SKIP_BREW=0
SKIP_FMM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      [[ $# -ge 2 ]] || die "--env-name requires a name"
      ENV_NAME="$2"
      shift 2
      ;;
    --work-dir)
      [[ $# -ge 2 ]] || die "--work-dir requires a directory"
      WORK_DIR="$2"
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || die "--jobs requires a positive integer"
      JOBS="$2"
      shift 2
      ;;
    --skip-brew)
      SKIP_BREW=1
      shift
      ;;
    --skip-fmm)
      SKIP_FMM=1
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

[[ "$(uname -s)" == "Darwin" ]] || die "macOS is required"
[[ "$(uname -m)" == "arm64" ]] || die "Apple Silicon arm64 is required"
[[ -f "$ENV_FILE" ]] || die "missing environment file: $ENV_FILE"
[[ -x "$BUILD_SCRIPT" ]] || die "build script is not executable: $BUILD_SCRIPT"
command -v brew >/dev/null 2>&1 || die "Homebrew is required: https://brew.sh"
command -v conda >/dev/null 2>&1 || die "Conda is required (Miniconda or Miniforge)"

BREW_PREFIX="$(brew --prefix)"
export PATH="${BREW_PREFIX}/bin:${PATH}"
FORMULAE=(cmake swig boost gdal libomp)
MISSING_FORMULAE=()
for formula in "${FORMULAE[@]}"; do
  if ! brew list --versions "$formula" >/dev/null 2>&1; then
    MISSING_FORMULAE+=("$formula")
  fi
done

if [[ "${#MISSING_FORMULAE[@]}" -gt 0 ]]; then
  if [[ "$SKIP_BREW" -eq 1 ]]; then
    die "missing Homebrew dependencies: ${MISSING_FORMULAE[*]}"
  fi
  brew install "${MISSING_FORMULAE[@]}"
fi

conda env update --name "$ENV_NAME" --file "$ENV_FILE" --prune
ENV_PYTHON="$(conda run --no-capture-output -n "$ENV_NAME" \
  python -c 'import sys; print(sys.executable)')"
[[ -x "$ENV_PYTHON" ]] || die "could not locate Python in Conda environment $ENV_NAME"

cd "$REPO_ROOT"
"$ENV_PYTHON" -m pip install --no-deps --editable "$REPO_ROOT"

if [[ "$SKIP_FMM" -eq 0 ]]; then
  BUILD_ARGS=(
    --python "$ENV_PYTHON"
    --work-dir "$WORK_DIR"
    --clean
    --install
  )
  if [[ -n "$JOBS" ]]; then
    BUILD_ARGS+=(--jobs "$JOBS")
  fi
  "$BUILD_SCRIPT" "${BUILD_ARGS[@]}"
fi

"$ENV_PYTHON" -m unittest -v tests.test_pipeline_reproducibility
if [[ "$SKIP_FMM" -eq 0 ]]; then
  "$ENV_PYTHON" -I -c 'import fmm; assert hasattr(fmm, "FastMapMatch")'
fi

ENV_PREFIX="$($ENV_PYTHON -c 'import sys; print(sys.prefix)')"
printf 'Bootstrap complete.\n'
printf 'Activate with: conda activate %s\n' "$ENV_NAME"
if [[ "$SKIP_FMM" -eq 0 ]]; then
  printf 'Set top-level fmm_bin to: %s/bin/fmm\n' "$ENV_PREFIX"
fi
