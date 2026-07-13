# Phase 1 macOS reproduction source-of-truth audit

Audit date: 2026-07-10. Branch: `feature/driver-1003-longitudinal-insights-and-reproducibility`. Evidence priority was current executable code, rerunnable commands, local build/output metadata, then older notes. No generic FMM command is treated as authoritative over the repository scripts.

## Audit status

Verified on the available Mac:

- pinned FMM checkout, clean temporary scripted build, native linkage, CLI help, and Python import;
- real Miami FMM lifetime regression against 568,030 edges, 42,484,944 UBODT rows, and 261 points (11.58 seconds);
- current Phase 1 output filenames and main schemas;
- Driver 1003 behavior build, request accounting, and privacy checks;
- wrapper syntax/help and focused tests.

Not completed:

- a fresh clone plus fresh `pipeline` Conda environment;
- a full clean three-county OSM/FDOT/conflation/FMM run;
- a replacement Palm Beach matched output;
- an unrestricted public release of legacy mobility-map history.

## Repository clone and initial directory setup

- **Pipeline stage:** Source checkout.
- **Phase 1 evidence found:** Git remotes and current working repository; no required submodule.
- **Historical command or behavior:** Work occurred in a locally named research checkout with a sibling FMM source tree.
- **Current verified command:** `git clone https://github.com/cyruskhan2005/reu-2026-driver-path-choice.git`, `git checkout feature/driver-1003-longitudinal-insights-and-reproducibility`, `git submodule update --init --recursive`.
- **Verification result:** Command syntax and branch existence verified locally; a new remote clone was not completed.
- **Output produced:** Repository checkout.
- **Notes and unresolved differences:** The branch must be pushed before an external clone can select it.

## Conda installation and activation

- **Pipeline stage:** Python runtime.
- **Phase 1 evidence found:** Miniconda/Conda 26.3.2; active tested Python was `/Users/.../miniconda3/envs/roadnet/bin/python`, version 3.11.15 arm64.
- **Historical command or behavior:** Existing `roadnet` environment supplied Python 3.11 for the verified native build.
- **Current verified command:** `scripts/bootstrap_macos.sh --env-name pipeline`, then `conda activate pipeline`.
- **Verification result:** Bootstrap logic, strict mode, help, and tests passed; the environment was not clean-created during this audit.
- **Output produced:** Intended Conda environment `pipeline`.
- **Notes and unresolved differences:** Documentation names the future reproducible environment `pipeline`; native proof currently comes from the older `roadnet` environment.

## Python package installation

- **Pipeline stage:** Install repository modules/CLIs.
- **Phase 1 evidence found:** `pyproject.toml` defines `roadnet-run` and `roadnet-consolidate`.
- **Historical command or behavior:** Editable installation in the working environment.
- **Current verified command:** `python -m pip install --no-deps --editable .` (performed by bootstrap after dependency creation).
- **Verification result:** Current imports and CLI help pass.
- **Output produced:** Importable `roadnet` and console scripts.
- **Notes and unresolved differences:** No global or `sudo pip` installation is supported.

## Road-data configuration

- **Pipeline stage:** Sanitized YAML and private paths.
- **Phase 1 evidence found:** `config.example.yaml`, `roadnet.cli.run.load_config`, and `PipelineConfig`.
- **Historical command or behavior:** Local absolute paths and an external GPS volume.
- **Current verified command:** `cp config.example.yaml config.yaml`, edit ignored paths, then `python -c 'from roadnet.cli.run import load_config; load_config("config.yaml")'`.
- **Verification result:** Top-level `fmm_bin` inheritance and county override tests pass; Mapillary-disabled mode is real.
- **Output produced:** Ignored local `config.yaml`.
- **Notes and unresolved differences:** Credentials are not committed. `config.yaml` is still tracked in historical repository state and must be untracked before release.

## OpenStreetMap retrieval

- **Pipeline stage:** OSM nodes, edges, and land use.
- **Phase 1 evidence found:** `roadnet.osm.download_county` called first by `Pipeline._run_county`.
- **Historical command or behavior:** County OSM caches exist under each county output directory.
- **Current verified command:** `scripts/run_full_pipeline_macos.sh --config config.yaml` with `skip_osm: false`.
- **Verification result:** Existing files confirmed; a new download was not run.
- **Output produced:** `osm_nodes.parquet`, `osm_edges.parquet`, `osm_landuse.parquet`.
- **Notes and unresolved differences:** FMM-only still needs these caches because `_run_county` loads OSM before returning the cached enriched network.

## FDOT and county GIS loading

- **Pipeline stage:** Official/state and optional county road attributes.
- **Phase 1 evidence found:** `roadnet.fdot`, `custom_geojson` mappings in configuration, and existing FDOT caches.
- **Historical command or behavior:** Miami-Dade and Palm Beach local centerline files; no Broward custom file.
- **Current verified command:** Full wrapper with configured `fdot_gdb` and county `custom_geojson` paths.
- **Verification result:** Code/path contracts inspected; proprietary/source files were not redownloaded.
- **Output produced:** Shared FDOT Parquets and county attributes used during conflation.
- **Notes and unresolved differences:** Source licensing and official acquisition remain the researcher's responsibility.

## Mapillary enabled and skipped modes

- **Pipeline stage:** Traffic-control sign enrichment.
- **Phase 1 evidence found:** `Pipeline._enrich_with_mapillary` and CLI `--disable-mapillary`.
- **Historical command or behavior:** Token-bearing enabled runs and cached results existed.
- **Current verified command:** `mapillary_enabled: false` or `roadnet-run config.yaml --disable-mapillary`.
- **Verification result:** Tests prove disabled mode calls neither Mapillary fetch nor attachment; enabled empty-token mode fails before a request.
- **Output produced:** Unchanged OSM edges when disabled; `mly_signs_raw.parquet`/`osm_edges_with_mly.parquet` when enabled.
- **Notes and unresolved differences:** `skip_mly` means reuse cache; it is not the same as disabling Mapillary.

## Conflation and speed enrichment

- **Pipeline stage:** Merge OSM, FDOT, county GIS, land use, and speed evidence.
- **Phase 1 evidence found:** `roadnet.conflation`, `roadnet.speed`, and `Pipeline._run_county`.
- **Historical command or behavior:** Long-running county processing produced known-good enriched networks.
- **Current verified command:** Full wrapper with `skip_conflation: false`.
- **Verification result:** Existing schemas, FID uniqueness, geometry, and CRS pass for three counties.
- **Output produced:** `enriched_network.parquet`.
- **Notes and unresolved differences:** Full recomputation was not attempted; cache reuse requires explicit `skip_conflation`/wrapper mode.

## Enriched-network output

- **Pipeline stage:** Phase 1 network checkpoint.
- **Phase 1 evidence found:** Broward 448,984 rows; Miami-Dade 568,030; Palm Beach 443,988.
- **Historical command or behavior:** Written with `GeoDataFrame.to_parquet` in county directories.
- **Current verified command:** `python scripts/verify_pipeline_outputs.py --config config.yaml --stage enrichment`.
- **Verification result:** Existing nonempty rows, unique FIDs, required fields, geometry, and CRS verified.
- **Output produced:** `<output_dir>/<County_slug>/enriched_network.parquet`.
- **Notes and unresolved differences:** Counts may change when upstream OSM/source data changes.

## FMM source acquisition

- **Pipeline stage:** Pin native matcher source.
- **Phase 1 evidence found:** Sibling `../fmm` and committed build script.
- **Historical command or behavior:** Local sibling checkout at `19ef34e…` with manual fixes.
- **Current verified command:** `scripts/build_fmm_macos.sh` clones `https://github.com/cyang-kth/fmm.git` and checks out the full pinned SHA.
- **Verification result:** Clean temporary checkout and patch application passed.
- **Output produced:** Ignored `build/fmm-macos/source`.
- **Notes and unresolved differences:** The sibling source contains historical local changes and is evidence, not the reproducible source path.

## Native dependencies and CMake compilation

- **Pipeline stage:** Compile FMM CLI/Python binding.
- **Phase 1 evidence found:** Homebrew arm64 formulae and `patches/fmm-macos-arm64.patch`.
- **Historical command or behavior:** C++11/GDAL/OpenMP/Python-link assumptions failed on the modern host.
- **Current verified command:** `scripts/build_fmm_macos.sh --python "$(command -v python)" --clean --install`.
- **Verification result:** arm64 targets built; C++17, GDAL const fix, OpenMP discovery, SWIG/Python 3.11 sysconfig, and dynamic lookup verified.
- **Output produced:** portable `bin/`, `lib/`, `python/`, plus `build-manifest.txt`.
- **Notes and unresolved differences:** Exact formula versions are recorded in the manifest; scripts detect prefixes rather than assuming `/opt/homebrew`.

## FMM executable and native-library discovery

- **Pipeline stage:** Make runtime artifacts discoverable.
- **Phase 1 evidence found:** Historical PATH confusion and a Python extension directly linked to libpython 3.9.
- **Historical command or behavior:** A stale build could be found before the correct executable/binding.
- **Current verified command:** bootstrap installs binaries in the active environment; configuration uses top-level `fmm_bin: "fmm"`.
- **Verification result:** `command -v`, `file`, help, `otool`, and isolated import passed. `_fmm.so` uses `@loader_path/libFMMLIB.dylib` and no `libpython`.
- **Output produced:** working `fmm`, `ubodt_gen`, and Python `fmm` module.
- **Notes and unresolved differences:** Do not use broad `DYLD_LIBRARY_PATH`; the supported fix is loader-relative linkage.

## FMM edge shapefile preparation

- **Pipeline stage:** Export FMM network schema.
- **Phase 1 evidence found:** `Pipeline._write_fmm_shp` exports `fid`, `u`, `v`, geometry in EPSG:4326.
- **Historical command or behavior:** Per-county files already exist.
- **Current verified command:** Full pipeline produces them; verifier checks `edges.shp/.shx/.dbf/.prj/.cpg`, unique IDs, row parity, and native readability.
- **Verification result:** Existing components located and schemas inspected.
- **Output produced:** `<county>/fmm/edges.*`.
- **Notes and unresolved differences:** Cached-network mode does not regenerate this shapefile; it must already exist.

## UBODT generation

- **Pipeline stage:** Precompute bounded network paths.
- **Phase 1 evidence found:** `_generate_ubodt_native`; stale test compares UBODT and shapefile modification times.
- **Historical command or behavior:** Existing tables contain 27,452,304 Broward, 42,484,944 Miami, and 23,527,095 Palm rows.
- **Current verified command:** FMM-only/full wrapper; `run_county` creates UBODT when missing or stale.
- **Verification result:** Files are nonempty; Miami table loaded in the real regression.
- **Output produced:** `<county>/fmm/ubodt.txt`.
- **Notes and unresolved differences:** Generation is expensive; do not delete a current table merely to test setup.

## County GPS input preparation

- **Pipeline stage:** Convert session GPS to FMM CSV.
- **Phase 1 evidence found:** `find_sessions`, `build_master_gps_parquet`, `CountyAssigner`, and point-index schema.
- **Historical command or behavior:** 3,504,264 Broward, 41,816 Miami, and 1,138,172 Palm CSV point rows.
- **Current verified command:** Full/FMM-only wrapper with writable `gps_root`.
- **Verification result:** Existing schemas contain `id`, `lon`, `lat`, `timestamp`; point counts measured.
- **Output produced:** `<county>/<County name>_gps.csv`.
- **Notes and unresolved differences:** The pipeline writes GPS CSV each run; explicit matched reuse is handled separately.

## FMM map matching and matched-output validation

- **Pipeline stage:** Run CLI and aggregate sensor/road observations.
- **Phase 1 evidence found:** `_run_fmm_cli` requests `opath`; `run_county` aligns by `point_idx`.
- **Historical command or behavior:** Broward and Miami matched IDs are unique; existing Palm file has 1,206 rows but only 1,160 unique IDs.
- **Current verified command:** `scripts/run_fmm_only_macos.sh --config config.yaml --overwrite-matched` then verifier `--stage matched`.
- **Verification result:** Real Miami smoke passed; Palm is intentionally a known failing artifact.
- **Output produced:** county `*_matched.csv` and optional per-session `*_fid_aggregated.jsonl`.
- **Notes and unresolved differences:** Strict verification requires unique matched IDs, GPS membership, at least 90% ID coverage, populated paths, and fresh timestamps unless reuse is explicit.

## SWIG MatchResult lifetime regression

- **Pipeline stage:** Native failed-trip/split matching.
- **Phase 1 evidence found:** Older expression read `.opath` from a temporary SWIG result; native memory could be released first.
- **Historical command or behavior:** Miami trip prefix exited 139.
- **Current verified command:** `_match_with_splits` stores `match_result = model.match_wkt(...)` before copying `list(match_result.opath)`; run `python -m unittest -v tests.test_fmm_result_lifetime`.
- **Verification result:** 261 returned assignments, child exit 0, 11.58 seconds.
- **Output produced:** stable split-match results.
- **Notes and unresolved differences:** This is distinct from the stale libpython 3.9 linkage failure.

## Driver 1003 timeline and Phase 2 graph products

- **Pipeline stage:** Phase 2A/B/C.
- **Phase 1 evidence found:** `build_driver_timeline.py`, `build_driver_1003_monthly_graphs.py`, and `compare_driver_1003_monthly_graphs.py`.
- **Historical command or behavior:** Existing cached timeline and graph deliverables fed RCCI.
- **Current verified command:** `scripts/generate_driver_1003_report.sh --config config.yaml --google-mode offline` (rebuilds prerequisites by default).
- **Verification result:** Current outputs exist; wrapper command path and preflights were validated.
- **Output produced:** driver timeline, monthly nodes/edges, comparison summaries/maps.
- **Notes and unresolved differences:** Phase 2 is separate from `roadnet-run`; it is not silently assumed by the report wrapper.

## RCCI report generation

- **Pipeline stage:** Technical route-network change measure.
- **Phase 1 evidence found:** `roadnet.route_choice_change_index` consumes Phase 2C only.
- **Historical command or behavior:** Google-Drive and canonical copies diverged.
- **Current verified command:** report wrapper passes explicit Phase 2C input and canonical `deliverables/driver_1003/route_choice_change_index` output roots.
- **Verification result:** RCCI formulas remain covered; canonical report exists and accepts the behavior section.
- **Output produced:** RCCI CSV/JSON/SVG/HTML and validation report.
- **Notes and unresolved differences:** Legacy linked maps contain private geometry and are not automatically public.

## Longitudinal behavior outputs

- **Pipeline stage:** Stays, roles, POIs, chains, route families, and report.
- **Phase 1 evidence found:** actual Driver 1003 build over 3,284 trips/3,286 county fragments.
- **Historical command or behavior:** Initial insights overclassified a frequent short bank-area stop as work.
- **Current verified command:** report wrapper invokes `build_driver_1003_real_world_behavior.py` with configured Phase 1/2 roots and explicit Google mode.
- **Verification result:** No workplace/school claim; C002 median stay 15.8 minutes; route-family arithmetic and public-home checks pass. Google accounting is 84 cumulative with 51 final cache hits.
- **Output produced:** CSVs, structured JSON, interactive map, and redesigned canonical report.
- **Notes and unresolved differences:** Activity purpose remains inferred. Public packaging requires removal of raw-GPS imagery and legacy private links.

## Confirmed macOS segmentation-fault root cause and fix

The obsolete sibling Python binding directly links `@rpath/libpython3.9.dylib`. Importing that binding while Python 3.11 is active reproducibly exits 139. Binary inspection, not speculation, establishes this incompatibility. The supported build uses macOS undefined dynamic lookup for CPython symbols, never directly links `libpython`, embeds a loader-relative reference to `libFMMLIB.dylib`, removes stale RPATHs, and signs the resulting local binaries. The proof commands are in [TROUBLESHOOTING_MACOS_FMM.md](TROUBLESHOOTING_MACOS_FMM.md).
