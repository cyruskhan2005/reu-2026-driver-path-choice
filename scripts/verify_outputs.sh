#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Validate a configured pipeline stage or the curated public-release boundary.

Usage:
  scripts/verify_outputs.sh --config PATH [pipeline verifier options]
  scripts/verify_outputs.sh --public [public validator options]

Examples:
  scripts/verify_outputs.sh --config config.yaml --stage enrichment
  scripts/verify_outputs.sh --public --require-manifest
EOF
}

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

cd "$REPO_ROOT"
if [[ "$1" == "--public" ]]; then
  shift
  exec python scripts/validate_public_release.py --public-dir outputs/public "$@"
fi
exec python scripts/verify_pipeline_outputs.py "$@"
