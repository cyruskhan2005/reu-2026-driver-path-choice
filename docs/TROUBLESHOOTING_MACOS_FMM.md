# Troubleshooting the macOS roadnet/FMM pipeline

Use the supported `pipeline` environment and committed scripts first. The alternatives below are diagnostics, not a second setup path. Never paste credentials or private GPS rows into an issue or log.

## `fmm` or `ubodt_gen` is not found

1. **Symptom:** `command not found: fmm`, or the pipeline cannot execute the configured matcher.
2. **Likely cause:** The `pipeline` Conda environment is inactive, bootstrap skipped FMM, or `fmm_bin` points to an obsolete build.
3. **Diagnostic command:**

   ```bash
   echo "${CONDA_DEFAULT_ENV:-none}"
   command -v python
   command -v fmm
   command -v ubodt_gen
   python -c 'from roadnet.cli.run import load_config; print(load_config("config.yaml").fmm_bin)'
   ```

4. **Verified fix:** `conda activate pipeline`; rerun `scripts/bootstrap_macos.sh --env-name pipeline`, and keep `fmm_bin: "fmm"` unless an explicit tested path is required.
5. **Proof:** `fmm --help`, `ubodt_gen --help`, and verifier `--stage environment` exit zero.

## `dyld` cannot find `libFMMLIB.dylib`

1. **Symptom:** Import/CLI fails with `Library not loaded`, `image not found`, or a missing `libFMMLIB.dylib` message.
2. **Likely cause:** A stale FMM build relies on an unresolved RPATH, or the extension/binary was copied without its loader-relative library.
3. **Diagnostic command:**

   ```bash
   python -I -c 'import _fmm; print(_fmm.__file__)'
   otool -L "$(python -I -c 'import _fmm; print(_fmm.__file__)')"
   otool -l "$(python -I -c 'import _fmm; print(_fmm.__file__)')" | grep -A2 LC_RPATH
   ```

4. **Verified fix:** Rebuild/install with `scripts/build_fmm_macos.sh --python "$(command -v python)" --clean --install`. Do not set a broad permanent `DYLD_LIBRARY_PATH`.
5. **Proof:** `otool -L` shows `@loader_path/libFMMLIB.dylib`, and isolated `import fmm` succeeds.

## Python FMM import exits 139 (segmentation fault)

1. **Symptom:** `python -c 'import fmm'` or `import _fmm` exits 139 without a Python exception.
2. **Likely cause:** Confirmed on the obsolete sibling build: `_fmm.so` directly links `@rpath/libpython3.9.dylib` while the active interpreter is Python 3.11. A mixed binding/interpreter runtime is unsafe.
3. **Diagnostic command:**

   ```bash
   python --version
   python -c 'import platform; print(platform.machine())'
   file /path/to/_fmm.so
   otool -L /path/to/_fmm.so | grep -E 'libpython|libFMMLIB'
   ```

4. **Verified fix:** Delete/quarantine only the stale build workspace, then use the pinned build script. Its macOS SWIG target uses undefined dynamic lookup and rejects any direct `libpython` dependency.
5. **Proof:**

   ```bash
   python -I -c 'import fmm; assert hasattr(fmm, "FastMapMatch")'
   ! otool -L "$(python -I -c 'import _fmm; print(_fmm.__file__)')" | grep libpython
   ```

   The clean scripted build and real Miami regression both pass.

## Split matching exits 139 even though import works

1. **Symptom:** Import and CLI help pass, but a Python split-match call crashes while reading `opath`.
2. **Likely cause:** A separate confirmed SWIG lifetime bug: reading `.opath` directly from a temporary `model.match_wkt(...)` result allowed the native `MatchResult` to be freed too soon.
3. **Diagnostic command:** `python -m unittest -v tests.test_fmm_result_lifetime`.
4. **Verified fix:** Current `roadnet/fmm_pipeline.py` retains `match_result` and then copies `list(match_result.opath)`.
5. **Proof:** The isolated 261-point Miami test returns 261 results and exit code zero (11.58 seconds on the audited Mac).

## ARM/Intel architecture mismatch

1. **Symptom:** `bad CPU type`, loader failure, import crash, or `file` reports `x86_64` for one component and `arm64` for another.
2. **Likely cause:** Rosetta/Homebrew/Conda artifacts from different architectures were mixed.
3. **Diagnostic command:**

   ```bash
   uname -m
   python -c 'import platform; print(platform.machine())'
   file "$(command -v fmm)"
   file "$(command -v python)"
   brew --prefix
   ```

4. **Verified fix:** Use an arm64 Terminal, arm64 Conda/Python, Homebrew detected by `brew --prefix`, and a clean scripted build with `CMAKE_OSX_ARCHITECTURES=arm64`.
5. **Proof:** All four architecture checks report `arm64`; verifier environment stage passes.

## CMake cannot find OpenMP, Boost, or GDAL

1. **Symptom:** Configure fails at `find_package(OpenMP)`, Boost, or GDAL; compilation cannot find `omp.h`; GDAL headers cause a const error.
2. **Likely cause:** Homebrew prefixes were not passed to CMake, old FMM assumes C++11/older GDAL, or the formula is absent.
3. **Diagnostic command:**

   ```bash
   brew list --versions cmake swig boost gdal libomp
   brew --prefix boost
   brew --prefix gdal
   brew --prefix libomp
   test -f "$(brew --prefix libomp)/include/omp.h" && echo omp-ok
   ```

4. **Verified fix:** Let bootstrap install formulae; use the committed FMM patch/build script. It sets C++17, the Homebrew prefixes, Apple clang OpenMP flags, and the verified GDAL const correction.
5. **Proof:** The build finishes all five targets and writes `build-manifest.txt` with formula versions/checksums.

## Conda and Homebrew geospatial libraries conflict

1. **Symptom:** Python imports a different GDAL/PROJ/GEOS stack than native FMM, or `otool` shows unexpected architecture/prefix paths.
2. **Likely cause:** A globally modified `PATH`, `DYLD_LIBRARY_PATH`, or mixed Intel/ARM installations.
3. **Diagnostic command:**

   ```bash
   which -a python gdal-config cmake
   python -c 'import geopandas,pyproj,shapely; print(geopandas.__version__, pyproj.__version__, shapely.__version__)'
   env | grep -E '^(PATH|DYLD_LIBRARY_PATH|PROJ|GDAL)='
   otool -L "$(command -v fmm)"
   ```

4. **Verified fix:** Start a fresh Terminal, activate only `pipeline`, unset undocumented DYLD/PROJ/GDAL overrides, and rebuild through the script.
5. **Proof:** Project imports and verifier environment stage pass.

## Python imports the wrong environment

1. **Symptom:** `roadnet`, `fmm`, or a dependency is missing despite bootstrap, or Python path points to a different environment.
2. **Likely cause:** Conda activation did not affect the current shell.
3. **Diagnostic command:**

   ```bash
   echo "${CONDA_DEFAULT_ENV:-none}"
   command -v python
   python -c 'import sys; print(sys.executable); print(sys.prefix)'
   ```

4. **Verified fix:** Initialize Conda for zsh if necessary, reopen Terminal, and run `conda activate pipeline`.
5. **Proof:** `CONDA_DEFAULT_ENV=pipeline`, imports pass, and wrappers no longer refuse the environment.

## Missing enriched network, OSM cache, edges shapefile, or UBODT

1. **Symptom:** FMM-only preflight stops before map matching.
2. **Likely cause:** Only part of the cached pipeline tree was copied. FMM-only needs more than `enriched_network.parquet`.
3. **Diagnostic command:**

   ```bash
   find sflorida_outputs -type f \( \
     -name enriched_network.parquet -o \
     -name osm_nodes.parquet -o \
     -name osm_edges.parquet -o \
     -name osm_landuse.parquet -o \
     -name edges.shp -o \
     -name ubodt.txt \) -print
   ```

4. **Verified fix:** Restore approved caches or run the full clean mode. `ubodt.txt` alone may be generated by FMM-only if the complete edge shapefile exists.
5. **Proof:** Verifier enrichment and FMM-prep stages pass.

## Mapillary token is unavailable

1. **Symptom:** Enabled mode raises `mly_token is empty`.
2. **Likely cause:** Mapillary was enabled without a token. `skip_mly` is not a credential-free mode; it asks to reuse Mapillary cache.
3. **Diagnostic command:** Inspect only booleans, not token values: `python -c 'from roadnet.cli.run import load_config; c=load_config("config.yaml"); print(c.mapillary_enabled, c.skip_mly)'`.
4. **Verified fix:** Set `mapillary_enabled: false` or pass `--disable-mapillary`.
5. **Proof:** The log says `Mapillary disabled` and tests confirm no Mapillary function was called.

## A long conflation or FMM run appears frozen

1. **Symptom:** Terminal output pauses for minutes; CPU/disk behavior is unclear.
2. **Likely cause:** UBODT generation, large county FMM, sensor aggregation, or spatial conflation is legitimately expensive; alternatively a worker may be pathological.
3. **Diagnostic command:**

   ```bash
   pgrep -af 'roadnet-run|fmm|ubodt_gen'
   ps -Ao pid,etime,%cpu,%mem,command | grep -E 'roadnet-run|fmm|ubodt_gen'
   top -o cpu
   tail -f logs/pipeline/*.log
   ls -lhT sflorida_outputs/*_County/fmm/ubodt.txt
   du -sh sflorida_outputs
   ```

4. **Verified fix:** Continue while CPU/output timestamps advance. Current FMM aggregation has progress messages, bounded split attempts, and a watchdog for pathological native postprocessing. If truly stuck, interrupt once, preserve good caches, and resume explicitly.
5. **Proof:** Logs show advancing trip counts/stage completion; post-run verifier passes.

## Safe stopping and resume

1. **Symptom:** A laptop must stop before a long run finishes.
2. **Likely cause:** Normal operational interruption.
3. **Diagnostic command:** Check current process/log/output timestamp with the monitoring commands above.
4. **Verified fix:** Press `Control-C` once, wait for child processes, confirm with `pgrep`, then review which output was mid-write. Move a corrupt partial file to a clearly named quarantine location; do not delete unrelated caches. Set only justified skip flags and pass `--resume`.
5. **Proof:** Dry-run/preflight succeeds, and the resumed stage produces a newer validated output.

## Matched CSV is older than GPS or contains duplicate IDs

1. **Symptom:** Strict verifier rejects freshness or duplicate matched trip IDs. The current Palm Beach artifact has 1,206 rows but 1,160 unique IDs.
2. **Likely cause:** Stale/partial prior output or intentional matched reuse after GPS CSV regeneration.
3. **Diagnostic command:**

   ```bash
   python scripts/verify_pipeline_outputs.py --config config.yaml --stage matched
   ```

4. **Verified fix:** Prefer a clean `--overwrite-matched` run. Use `--reuse-matched` only when reuse is intentional; that relaxes timestamp order, not ID/schema/coverage checks.
5. **Proof:** Verifier reports unique IDs, at least 90% GPS-ID coverage, and populated paths.

## External drive is missing, read-only, or contains spaces

1. **Symptom:** `gps_root` does not exist, caches cannot be written, or an unquoted path splits into words.
2. **Likely cause:** Volume not mounted, permissions, available-space problem, or hand-entered unquoted path.
3. **Diagnostic command:**

   ```bash
   ls -ld "/Volumes/Your Drive"
   test -w "/Volumes/Your Drive" && echo writable
   df -h "/Volumes/Your Drive"
   ```

4. **Verified fix:** Mount/unlock the approved volume, quote every path, and use YAML quoted strings. The wrappers resolve and test `gps_root` writability.
5. **Proof:** FMM-only dry-run passes and `gps_master.parquet`/logs can be created.

## Gatekeeper, execute permission, or local signature failure

1. **Symptom:** macOS refuses a locally built executable, or a script returns permission denied.
2. **Likely cause:** Execute bit missing or native binary changed after signing.
3. **Diagnostic command:** `ls -l scripts/*.sh "$(command -v fmm)"`; `codesign -dv "$(command -v fmm)"`.
4. **Verified fix:** Use the committed executable scripts; rebuild native artifacts so the build script applies ad-hoc signing after link-path rewrites. Do not bypass Gatekeeper globally.
5. **Proof:** `fmm --help` and wrapper `--help` exit zero.

## Report generation cannot find Phase 2 inputs

1. **Symptom:** RCCI reports missing comparison CSV/Parquet or behavior analysis cannot find the Driver 1003 timeline/monthly nodes.
2. **Likely cause:** Only Phase 1 ran; `roadnet-run` does not create Phase 2A/B/C automatically, or config `output_dir` differs from the report source root.
3. **Diagnostic command:**

   ```bash
   test -s "$(python -c 'import yaml; print(yaml.safe_load(open("config.yaml"))["output_dir"])')/phase2/driver_timelines/driver_1_timeline.csv"
   test -s deliverables/google_drive_phase2/driver_1003_graph_comparisons/data/driver_1003_month_to_month_summary.csv
   ```

4. **Verified fix:** Run `scripts/generate_driver_1003_report.sh --config config.yaml --google-mode offline` without `--reuse-phase2`. The wrapper now builds Phase 2A/B/C and passes configured Phase 1/2 roots to behavior analysis.
5. **Proof:** Driver verifier reports 3,284 unique trips, reconciled monthly shares, and generated JSON/report/map.
