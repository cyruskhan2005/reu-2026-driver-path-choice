#!/usr/bin/env bash
set -euo pipefail

FMM_REPOSITORY="https://github.com/cyang-kth/fmm.git"
FMM_SHA="19ef34e1f57ff2f2484231aa0d01dfffea986ec1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_FILE="${REPO_ROOT}/patches/fmm-macos-arm64.patch"

usage() {
  cat <<'EOF'
Build the pinned FMM native CLI and Python 3.11 binding for Apple Silicon.

Usage:
  scripts/build_fmm_macos.sh [options]

Options:
  --python PATH       Python 3.11 executable to build against (default: python3)
  --work-dir DIR      Source/build workspace (default: build/fmm-macos)
  --prefix DIR        Portable installation directory (default: WORK_DIR/install)
  --source-dir DIR    Use an existing clean checkout at the pinned FMM commit
  --jobs N            Parallel build jobs (default: detected CPU count)
  --clean             Remove this script's build and install directories first
  --install           Also install the package and CLI into the Python environment
  -h, --help          Show this help and exit

The portable result contains bin/, lib/, and python/. The Python extension uses
dynamic lookup for CPython symbols and @loader_path for libFMMLIB.dylib; it does
not embed or directly link a libpython path.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

absolute_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PWD" "$1" ;;
  esac
}

PYTHON_BIN="${PYTHON:-python3}"
WORK_DIR="${REPO_ROOT}/build/fmm-macos"
PREFIX=""
SOURCE_DIR=""
JOBS=""
CLEAN=0
INSTALL_ENV=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || die "--python requires a path"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --work-dir)
      [[ $# -ge 2 ]] || die "--work-dir requires a directory"
      WORK_DIR="$2"
      shift 2
      ;;
    --prefix)
      [[ $# -ge 2 ]] || die "--prefix requires a directory"
      PREFIX="$2"
      shift 2
      ;;
    --source-dir)
      [[ $# -ge 2 ]] || die "--source-dir requires a directory"
      SOURCE_DIR="$2"
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || die "--jobs requires a positive integer"
      JOBS="$2"
      shift 2
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --install)
      INSTALL_ENV=1
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

[[ "$(uname -s)" == "Darwin" ]] || die "this build is supported only on macOS"
HOST_ARCH="$(uname -m)"
[[ "$HOST_ARCH" == "arm64" ]] || die "Apple Silicon arm64 is required (found $HOST_ARCH)"
[[ -f "$PATCH_FILE" ]] || die "missing patch: $PATCH_FILE"

require_command brew
require_command git
require_command cmake
require_command swig
require_command otool
require_command install_name_tool
require_command codesign
require_command shasum

if [[ "$PYTHON_BIN" != /* ]]; then
  PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
fi
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || die "Python executable not found"

PYTHON_VERSION="$($PYTHON_BIN -c 'import platform; print(platform.python_version())')"
PYTHON_MINOR="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_ARCH="$($PYTHON_BIN -c 'import platform; print(platform.machine())')"
[[ "$PYTHON_MINOR" == "3.11" ]] || die "Python 3.11 is required (found $PYTHON_VERSION)"
[[ "$PYTHON_ARCH" == "arm64" ]] || die "arm64 Python is required (found $PYTHON_ARCH)"

BREW_PREFIX="$(brew --prefix)"
GDAL_PREFIX="$(brew --prefix gdal 2>/dev/null)" || die "Homebrew gdal is not installed"
BOOST_PREFIX="$(brew --prefix boost 2>/dev/null)" || die "Homebrew boost is not installed"
LIBOMP_PREFIX="$(brew --prefix libomp 2>/dev/null)" || die "Homebrew libomp is not installed"
[[ -d "$BREW_PREFIX" ]] || die "invalid Homebrew prefix: $BREW_PREFIX"
[[ -x "$GDAL_PREFIX/bin/gdal-config" ]] || die "gdal-config not found under $GDAL_PREFIX"
[[ -f "$LIBOMP_PREFIX/include/omp.h" ]] || die "omp.h not found under $LIBOMP_PREFIX"
[[ -f "$LIBOMP_PREFIX/lib/libomp.dylib" ]] || die "libomp.dylib not found under $LIBOMP_PREFIX"

BOOST_CMAKE_DIR=""
for candidate in "$BOOST_PREFIX"/lib/cmake/Boost-*; do
  if [[ -d "$candidate" ]]; then
    BOOST_CMAKE_DIR="$candidate"
  fi
done
[[ -n "$BOOST_CMAKE_DIR" ]] || die "Boost CMake package directory not found"

WORK_DIR="$(absolute_path "$WORK_DIR")"
if [[ -z "$PREFIX" ]]; then
  PREFIX="${WORK_DIR}/install"
else
  PREFIX="$(absolute_path "$PREFIX")"
fi
if [[ -n "$SOURCE_DIR" ]]; then
  SOURCE_DIR="$(absolute_path "$SOURCE_DIR")"
else
  SOURCE_DIR="${WORK_DIR}/source"
fi
BUILD_DIR="${WORK_DIR}/build"

if [[ -z "$JOBS" ]]; then
  JOBS="$(sysctl -n hw.logicalcpu 2>/dev/null || printf '4')"
fi
case "$JOBS" in
  ''|*[!0-9]*) die "--jobs must be a positive integer" ;;
esac
[[ "$JOBS" -gt 0 ]] || die "--jobs must be greater than zero"

mkdir -p "$WORK_DIR"
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  [[ ! -e "$SOURCE_DIR" || -z "$(ls -A "$SOURCE_DIR" 2>/dev/null)" ]] \
    || die "source directory exists but is not a Git checkout: $SOURCE_DIR"
  git clone "$FMM_REPOSITORY" "$SOURCE_DIR"
  git -C "$SOURCE_DIR" checkout --detach "$FMM_SHA"
fi

ACTUAL_SHA="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
[[ "$ACTUAL_SHA" == "$FMM_SHA" ]] \
  || die "FMM checkout is at $ACTUAL_SHA; expected pinned commit $FMM_SHA"

if git -C "$SOURCE_DIR" apply --check "$PATCH_FILE"; then
  [[ -z "$(git -C "$SOURCE_DIR" status --porcelain)" ]] \
    || die "FMM checkout has local changes before patching"
  git -C "$SOURCE_DIR" apply "$PATCH_FILE"
elif git -C "$SOURCE_DIR" apply --reverse --check "$PATCH_FILE"; then
  UNEXPECTED_STATUS="$({ git -C "$SOURCE_DIR" status --porcelain \
    | awk '{print $2}' \
    | grep -Ev '^(CMakeLists.txt|python/CMakeLists.txt|src/network/network.cpp)$'; } || true)"
  [[ -z "$UNEXPECTED_STATUS" ]] \
    || die "FMM checkout has changes beyond the expected macOS patch"
else
  die "macOS patch does not apply cleanly to $SOURCE_DIR"
fi

if [[ "$CLEAN" -eq 1 ]]; then
  case "$BUILD_DIR" in
    "$WORK_DIR"/*) rm -rf "$BUILD_DIR" ;;
    *) die "refusing to clean build directory outside work directory" ;;
  esac
  case "$PREFIX" in
    "$WORK_DIR"/*) rm -rf "$PREFIX" ;;
    *) die "refusing to clean custom prefix outside work directory: $PREFIX" ;;
  esac
fi
mkdir -p "$BUILD_DIR" "$PREFIX/bin" "$PREFIX/lib" "$PREFIX/python"

PYTHON_INCLUDE="$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_path("include"))')"
PYTHON_LIBRARY="$($PYTHON_BIN -c '
import os, sys, sysconfig
libdir = sysconfig.get_config_var("LIBDIR")
candidates = [
    f"libpython{sys.version_info.major}.{sys.version_info.minor}.dylib",
    sysconfig.get_config_var("LDLIBRARY"),
]
for name in candidates:
    if name and os.path.isfile(os.path.join(libdir, name)):
        print(os.path.join(libdir, name))
        break
')"
PYTHON_PREFIX="$($PYTHON_BIN -c 'import sys; print(sys.prefix)')"
[[ -d "$PYTHON_INCLUDE" ]] || die "Python headers not found: $PYTHON_INCLUDE"
[[ -f "$PYTHON_LIBRARY" ]] || die "Python library not found: $PYTHON_LIBRARY"

export PATH="${BREW_PREFIX}/bin:${PATH}"
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_PREFIX_PATH="${BREW_PREFIX};${PYTHON_PREFIX}" \
  -DGDAL_CONFIG="$GDAL_PREFIX/bin/gdal-config" \
  -DGDAL_INCLUDE_DIR="$GDAL_PREFIX/include" \
  -DGDAL_LIBRARY="$GDAL_PREFIX/lib/libgdal.dylib" \
  -DBoost_DIR="$BOOST_CMAKE_DIR" \
  -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" \
  -DOpenMP_C_LIB_NAMES=omp \
  -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY="$LIBOMP_PREFIX/lib/libomp.dylib" \
  -DOpenMP_CXX_LIBRARY="$LIBOMP_PREFIX/lib/libomp.dylib" \
  -DOpenMP_CXX_INCLUDE_DIR="$LIBOMP_PREFIX/include" \
  -DPYTHON_EXECUTABLE="$PYTHON_BIN" \
  -DPYTHON_INCLUDE_DIR="$PYTHON_INCLUDE" \
  -DPYTHON_LIBRARY="$PYTHON_LIBRARY"

cmake --build "$BUILD_DIR" \
  --target pyfmm fmm ubodt_gen stmatch h3mm \
  --parallel "$JOBS"

FMM_EXTENSION="$(find "$BUILD_DIR/python" -maxdepth 1 -type f -name '_fmm*.so' -print -quit)"
[[ -n "$FMM_EXTENSION" ]] || die "built _fmm extension not found"
[[ -f "$BUILD_DIR/python/fmm.py" ]] || die "generated fmm.py not found"
[[ -f "$BUILD_DIR/libFMMLIB.dylib" ]] || die "built libFMMLIB.dylib not found"

install -m 0644 "$BUILD_DIR/python/fmm.py" "$PREFIX/python/fmm.py"
install -m 0755 "$FMM_EXTENSION" "$PREFIX/python/_fmm.so"
install -m 0755 "$BUILD_DIR/libFMMLIB.dylib" "$PREFIX/python/libFMMLIB.dylib"
install -m 0755 "$BUILD_DIR/libFMMLIB.dylib" "$PREFIX/lib/libFMMLIB.dylib"
for executable in fmm ubodt_gen stmatch h3mm; do
  [[ -x "$BUILD_DIR/bin/$executable" ]] || die "built executable not found: $executable"
  install -m 0755 "$BUILD_DIR/bin/$executable" "$PREFIX/bin/$executable"
done

rewrite_fmmlib_dependency() {
  local binary="$1"
  local replacement="$2"
  local current
  current="$(otool -L "$binary" | awk '/libFMMLIB[.]dylib/{print $1; exit}')"
  [[ -n "$current" ]] || die "libFMMLIB dependency missing from $binary"
  install_name_tool -change "$current" "$replacement" "$binary"
}

delete_all_rpaths() {
  local binary="$1"
  local rpath
  while IFS= read -r rpath; do
    [[ -n "$rpath" ]] || continue
    install_name_tool -delete_rpath "$rpath" "$binary"
  done < <(otool -l "$binary" | awk '
    $1 == "cmd" && $2 == "LC_RPATH" { getline; getline; print $2 }
  ')
}

rewrite_fmmlib_dependency "$PREFIX/python/_fmm.so" '@loader_path/libFMMLIB.dylib'
delete_all_rpaths "$PREFIX/python/_fmm.so"
for executable in fmm ubodt_gen stmatch h3mm; do
  rewrite_fmmlib_dependency "$PREFIX/bin/$executable" '@loader_path/../lib/libFMMLIB.dylib'
  delete_all_rpaths "$PREFIX/bin/$executable"
done
install_name_tool -id '@rpath/libFMMLIB.dylib' "$PREFIX/python/libFMMLIB.dylib"
install_name_tool -id '@rpath/libFMMLIB.dylib' "$PREFIX/lib/libFMMLIB.dylib"

for binary in \
  "$PREFIX/python/_fmm.so" \
  "$PREFIX/python/libFMMLIB.dylib" \
  "$PREFIX/lib/libFMMLIB.dylib" \
  "$PREFIX/bin/fmm" \
  "$PREFIX/bin/ubodt_gen" \
  "$PREFIX/bin/stmatch" \
  "$PREFIX/bin/h3mm"; do
  codesign --force --sign - "$binary"
  file "$binary" | grep -q 'arm64' || die "non-arm64 binary produced: $binary"
done

otool -L "$PREFIX/python/_fmm.so" | grep -q '@loader_path/libFMMLIB.dylib' \
  || die "Python extension does not use loader-relative libFMMLIB"
if otool -L "$PREFIX/python/_fmm.so" | grep -q 'libpython'; then
  die "Python extension unexpectedly links libpython"
fi

"$PYTHON_BIN" -I -c '
import pathlib, sys
package = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(package))
import fmm
required = ("Network", "NetworkGraph", "FastMapMatch", "STMATCH", "UBODTGenAlgorithm")
missing = [name for name in required if not hasattr(fmm, name)]
if missing:
    raise SystemExit("missing FMM symbols: " + ", ".join(missing))
' "$PREFIX/python"
"$PREFIX/bin/fmm" >/dev/null
"$PREFIX/bin/ubodt_gen" >/dev/null

{
  printf 'fmm_repository=%s\n' "$FMM_REPOSITORY"
  printf 'fmm_sha=%s\n' "$FMM_SHA"
  printf 'host_arch=%s\n' "$HOST_ARCH"
  printf 'python=%s\n' "$PYTHON_BIN"
  printf 'python_version=%s\n' "$PYTHON_VERSION"
  printf 'homebrew_prefix=%s\n' "$BREW_PREFIX"
  brew list --versions cmake swig boost gdal libomp
  shasum -a 256 \
    "$PREFIX/bin/fmm" \
    "$PREFIX/bin/ubodt_gen" \
    "$PREFIX/python/_fmm.so" \
    "$PREFIX/python/libFMMLIB.dylib"
} >"$PREFIX/build-manifest.txt"

if [[ "$INSTALL_ENV" -eq 1 ]]; then
  PYTHON_SITE="$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
  ENV_BIN="$($PYTHON_BIN -c 'import os, sys; print(os.path.join(sys.prefix, "bin"))')"
  ENV_LIB="$($PYTHON_BIN -c 'import os, sys; print(os.path.join(sys.prefix, "lib"))')"
  [[ -d "$PYTHON_SITE" && -w "$PYTHON_SITE" ]] \
    || die "Python site-packages is not writable: $PYTHON_SITE"
  [[ -d "$ENV_BIN" && -w "$ENV_BIN" ]] || die "environment bin is not writable: $ENV_BIN"
  [[ -d "$ENV_LIB" && -w "$ENV_LIB" ]] || die "environment lib is not writable: $ENV_LIB"
  install -m 0644 "$PREFIX/python/fmm.py" "$PYTHON_SITE/fmm.py"
  install -m 0755 "$PREFIX/python/_fmm.so" "$PYTHON_SITE/_fmm.so"
  install -m 0755 "$PREFIX/python/libFMMLIB.dylib" "$PYTHON_SITE/libFMMLIB.dylib"
  install -m 0755 "$PREFIX/lib/libFMMLIB.dylib" "$ENV_LIB/libFMMLIB.dylib"
  for executable in fmm ubodt_gen stmatch h3mm; do
    install -m 0755 "$PREFIX/bin/$executable" "$ENV_BIN/$executable"
  done
  "$PYTHON_BIN" -I -c 'import fmm; assert hasattr(fmm, "FastMapMatch")'
  "$ENV_BIN/fmm" >/dev/null
fi

printf 'FMM build verified.\n'
printf 'Portable prefix: %s\n' "$PREFIX"
printf 'FMM CLI: %s\n' "$PREFIX/bin/fmm"
printf 'Python package: %s\n' "$PREFIX/python"
