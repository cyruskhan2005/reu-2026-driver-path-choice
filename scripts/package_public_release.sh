#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Package, manifest, and validate the reviewed Driver 1003 public artifact set.

Usage:
  scripts/package_public_release.sh [--public-dir PATH]

The command reads existing private analysis outputs. It never contacts Google
and never prints credentials. Review artifacts visually before publication.
EOF
}

PUBLIC_DIR="$REPO_ROOT/outputs/public"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --public-dir)
      [[ $# -ge 2 ]] || { printf 'ERROR: --public-dir requires a path\n' >&2; exit 1; }
      PUBLIC_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'ERROR: unknown option: %s (use --help)\n' "$1" >&2
      exit 1
      ;;
  esac
done

cd "$REPO_ROOT"
printf 'Packaging reviewed public artifacts in %s\n' "$PUBLIC_DIR"
python scripts/package_driver_1003_public_release.py --public-dir "$PUBLIC_DIR"
python scripts/generate_output_manifest.py --root "$PUBLIC_DIR" --output "$PUBLIC_DIR/manifest.json"
python scripts/validate_public_release.py --public-dir "$PUBLIC_DIR" --require-manifest
printf 'Public-release package completed and validated.\n'
