# macOS full-pipeline guide

This guide is the copy/paste path for a new REU researcher reproducing the current South Florida road-network, FMM, RCCI, and Driver 1003 longitudinal workflow. It describes the repository as implemented; it is not a generic FMM tutorial.

## Tested scope and important qualifications

The native FMM build and a real Miami lifetime regression were verified on:

- macOS 26.5.1 (build 25F80);
- Apple Silicon `arm64`;
- zsh 5.9 and the system Bash available on macOS;
- Miniconda/Conda 26.3.2;
- Python 3.11.15 (`arm64`);
- Homebrew prefix `/opt/homebrew`;
- CMake 4.3.3, Apple clang 21.0.0, SWIG 4.4.1;
- Boost 1.90.0, GDAL 3.13.1, and libomp 22.1.7;
- FMM commit `19ef34e1f57ff2f2484231aa0d01dfffea986ec1`.

The scripted build was exercised in a temporary workspace using the existing
Python 3.11 `roadnet` environment. The committed environment now uses the name
`pipeline`; creating that environment from scratch and running all three
counties end to end was not completed in this session. A real FMM Python
regression used the 568,030-edge Miami network, its 42,484,944-row UBODT, and
261 GPS observations and passed in 11.58 seconds. Palm Beach's current matched
CSV has duplicate IDs and is intentionally rejected until regenerated.

## 1. Fresh Terminal and system prerequisites

Starting directory: your home directory. Active Conda environment: none.

```bash
xcode-select -p || xcode-select --install
uname -m
sw_vers
zsh --version
```

Expected architecture is `arm64`. This project does not currently claim a verified Intel build.

Verify Homebrew. If it is absent, install it from its official site and reopen Terminal.

```bash
command -v brew
brew --prefix
```

Expected Apple Silicon prefix is `/opt/homebrew`; Intel installations commonly use `/usr/local`. The scripts call `brew --prefix` and do not hardcode either location.

Install Miniconda or Miniforge from its official distribution if `conda` is absent. Do not use `sudo pip`.

```bash
command -v conda
conda --version
```

## 2. Clone the project and select the tested branch

Starting directory: the parent directory where the clone should live. Active environment: none.

```bash
git clone https://github.com/cyruskhan2005/reu-2026-driver-path-choice.git
cd reu-2026-driver-path-choice
git checkout feature/driver-1003-longitudinal-insights-and-reproducibility
git submodule update --init --recursive
git status --short
```

The repository does not currently require a Git submodule for FMM; the initialization command is harmless. `scripts/build_fmm_macos.sh` clones the exact pinned FMM commit into the ignored build workspace.

## 3. Create the Python environment and build FMM

Starting directory: repository root. Active environment: base or none.

```bash
scripts/bootstrap_macos.sh --env-name pipeline
conda activate pipeline
python --version
python -c 'import platform,sys; print(platform.machine()); print(sys.executable)'
```

The bootstrap script:

1. requires macOS `arm64`;
2. verifies Homebrew and Conda;
3. installs missing `cmake`, `swig`, `boost`, `gdal`, and `libomp` formulae;
4. creates/updates `pipeline` from `environment-macos.yml`;
5. installs this repository editable;
6. calls the pinned FMM build with a clean build directory;
7. installs `fmm`, `ubodt_gen`, `stmatch`, `h3mm`, `fmm.py`, `_fmm.so`, and loader-relative `libFMMLIB.dylib` into the environment;
8. runs reproducibility tests and an isolated import.

It is safe to rerun. `--skip-brew` refuses missing formulae instead of installing them. `--skip-fmm` updates only the Python environment.

### Native build details

The approved build command is:

```bash
# Starting directory: repository root; active environment: pipeline
scripts/build_fmm_macos.sh \
  --python "$(command -v python)" \
  --work-dir build/fmm-macos \
  --clean \
  --install
```

The script checks out FMM commit `19ef34e…`, applies `patches/fmm-macos-arm64.patch`, detects Homebrew dependency prefixes, configures CMake for `arm64`, uses Apple clang with Homebrew OpenMP, and builds `pyfmm`, `fmm`, `ubodt_gen`, `stmatch`, and `h3mm`. It rewrites FMM library references to `@loader_path`, deletes stale RPATHs, ad-hoc signs locally built binaries, and fails if `_fmm.so` directly links `libpython`.

Portable output is under `build/fmm-macos/install/{bin,lib,python}` unless `--prefix` changes it. The bootstrap installs only the required files into the active Conda environment.

### Native verification

```bash
# Starting directory: repository root; active environment: pipeline
command -v fmm
command -v ubodt_gen
file "$(command -v fmm)"
file "$(command -v ubodt_gen)"
fmm --help
ubodt_gen --help
python -I -c 'import fmm; assert hasattr(fmm, "FastMapMatch")'
python scripts/verify_pipeline_outputs.py \
  --config config.example.yaml \
  --stage environment
```

Expected verification mentions arm64 CLI help, isolated Python import, no direct `libpython`, and `@loader_path/libFMMLIB.dylib`.

## 4. Prepare private and external data

Starting directory: repository root. Active environment: `pipeline`.

```bash
cp config.example.yaml config.yaml
```

`config.yaml` is ignored. Edit it without copying credentials into tracked files.

| Input | Config/path | Acquisition and use note |
|---|---|---|
| OpenStreetMap roads/land use | `place_query` per county | Downloaded by OSMnx when `skip_osm: false`; retain OSM attribution |
| FDOT File Geodatabase | `fdot_gdb` | Supply the approved local `DOTShapesFGDB.gdb`; not distributed here |
| Miami-Dade roads | county `custom_geojson` | Approved local GeoJSON with `SPEEDLIMIT`, `SNAME`, `MAINTCODE` mappings |
| Palm Beach roads | county `custom_geojson` | Approved local `Road_Centerlines.geojson` with `SPEED_LIM`, `NAME`, `RESP_AUTH` |
| Broward roads | no custom file in current config | Uses OSM + FDOT unless an approved source is later configured |
| Mapillary | `mapillary_enabled`, `mly_token` | Optional; default is disabled. Store a token only in ignored `config.yaml` |
| GPS/sensor sessions | `gps_root` | Restricted session hierarchy; must be readable and writable because caches/aggregates are written beside it |
| County boundary/area | county `place_query` | OSMnx resolves the administrative place boundary used for OSM road and land-use queries; `edges.shp` is an FMM road network, not a county boundary layer |
| Google POIs | environment variable | Optional report enrichment only; never used by Phase 1 or placed in YAML |

The code expects raw session files matching the existing `*_gps.jsonl` hierarchy discovered by `roadnet.fmm_pipeline.find_sessions`. It may create `gps_master.parquet` under `gps_root` and `*_fid_aggregated.jsonl` beside source sessions.

Verify local paths without exposing contents:

```bash
python - <<'PY'
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path("config.yaml").read_text())
for key in ("output_dir", "gps_root", "fdot_gdb"):
    value = cfg.get(key)
    print(key, "configured" if value else "missing", "exists" if value and Path(value).expanduser().exists() else "not found")
PY
```

### Optional Google setup

The report engine reads only the environment and never prints the value:

```bash
# Starting directory: repository root; active environment: pipeline
read -s GOOGLE_MAPS_API_KEY
export GOOGLE_MAPS_API_KEY
printf '\nGoogle key detected in environment: %s\n' "$([[ -n "${GOOGLE_MAPS_API_KEY:-}" ]] && printf yes || printf no)"
```

Do not add that command with a literal value to shell scripts or documentation. Google enrichment has a 100-attempt hard stop and caches successful sanitized responses in an ignored directory.

## 5. Minimal smoke tests before expensive work

```bash
# Starting directory: repository root; active environment: pipeline
python -m unittest -v tests.test_pipeline_reproducibility
python -m unittest -v tests.test_fmm_result_lifetime
scripts/run_full_pipeline_macos.sh --config config.yaml --dry-run
scripts/run_fmm_only_macos.sh --config config.yaml --dry-run
```

The full/FMM-only dry runs validate wrapper preconditions—architecture, environment, config, required paths, and overwrite policy—without executing or proving the scientific pipeline. FMM-only dry-run still requires cached inputs because that is its purpose.

## 6. Mode A: full clean rebuild

Use a new output directory. Set these YAML values:

```yaml
mapillary_enabled: false
skip_osm: false
skip_mly: false
skip_conflation: false
skip_fmm: false
fmm_bin: "fmm"
```

Starting directory: repository root. Active environment: `pipeline`.

```bash
scripts/run_full_pipeline_macos.sh --config config.yaml
```

Without `--resume`, the wrapper rejects enabled skip flags and existing `enriched_network.parquet`. It prints the resolved output root and a timestamped log under `logs/pipeline/`, then runs `roadnet-run` and the strict matched-output verifier.

### Stage boundaries and checks

1. **FDOT extraction** — `roadnet.fdot.extract_fdot_layers` writes shared FDOT Parquets under `<output_dir>/fdot/` when the GDB exists.
2. **OSM retrieval** — `roadnet.osm.download_county` writes `osm_nodes.parquet`, `osm_edges.parquet`, and `osm_landuse.parquet` per county.
3. **Mapillary** — `Pipeline._enrich_with_mapillary`; disabled mode makes no request. Enabled mode requires a token and produces `mly_signs_raw.parquet` and `osm_edges_with_mly.parquet`.
4. **Conflation/speed** — `roadnet.conflation` and `roadnet.speed` combine OSM, FDOT, and optional county roads.
5. **Enriched output** — `roadnet.pipeline.Pipeline._run_county` writes `enriched_network.parquet` and FMM `edges.*`.
6. **UBODT** — `roadnet.fmm_pipeline.run_county` generates `fmm/ubodt.txt` when absent or older than `edges.shp`.
7. **GPS preparation/FMM** — creates county `*_gps.csv`, runs the configured `fmm` CLI, and writes `*_matched.csv`.
8. **Aggregation** — aligns `opath` by point index and may write per-session aggregate JSONL files.

Run boundary checks independently:

```bash
python scripts/verify_pipeline_outputs.py --config config.yaml --stage enrichment
python scripts/verify_pipeline_outputs.py --config config.yaml --stage fmm-prep
python scripts/verify_pipeline_outputs.py --config config.yaml --stage matched
```

Known-good local row counts are 448,984 Broward edges, 568,030 Miami-Dade edges, and 443,988 Palm Beach edges. Treat these as a historical comparison, not a universal required count after OSM/source updates.

## 7. Mode B: cached enriched network and FMM only

Required cached files are listed in the root README. Set `skip_fmm: false`; the wrapper supplies the other skip flags on the command line.

Starting directory: repository root. Active environment: `pipeline`.

```bash
scripts/run_fmm_only_macos.sh \
  --config config.yaml \
  --county "Miami-Dade County" \
  --overwrite-matched
```

If an existing matched CSV should be reused for aggregation rather than replaced:

```bash
scripts/run_fmm_only_macos.sh \
  --config config.yaml \
  --county "Miami-Dade County" \
  --reuse-matched
```

The second form explicitly sets `ROADNET_FMM_REUSE_MATCHED=1`. The verifier allows an older matched-file timestamp only in this explicit reuse mode, while still checking schema, unique IDs, GPS-ID membership, at least 90% trip-ID coverage, and populated `opath` values.

## 8. Generate Phase 2, RCCI, and longitudinal outputs

Starting directory: repository root. Active environment: `pipeline`.

```bash
scripts/generate_driver_1003_report.sh \
  --config config.yaml \
  --google-mode offline
```

This command runs:

1. `build_driver_timeline.py` (Phase 2A);
2. `build_driver_1003_monthly_graphs.py` (Phase 2B and local deliverable bundle);
3. `compare_driver_1003_monthly_graphs.py` (Phase 2C);
4. canonical RCCI generation under `deliverables/driver_1003/route_choice_change_index/`;
5. Driver 1003 stay/place/chain/route-family analysis and report injection;
6. Driver-output verification;
7. self-contained public packaging, deterministic manifest generation, and offline release validation.

Use `--reuse-phase2` only when the wrapper verifies the timeline, combined monthly nodes, and comparison summary. `offline` cannot incur Google charges. `cache` uses existing sanitized cache entries. `network` requires the environment key and bounded request arguments.

## 9. Expected verification results

Short sanitized examples from the current repository:

```text
Verification passed: driver
  - Driver 1003: 3,284 unique trips
  - Driver 1003: 5,192 monthly route-family rows reconcile
  - Driver 1003: JSON/report/map exist; secret scan passed; no workplace claimed
```

Current local Phase 1 artifacts:

| County | Enriched edges | GPS CSV rows | Matched CSV rows | UBODT rows | Status |
|---|---:|---:|---:|---:|---|
| Broward | 448,984 | 3,504,264 | 4,071 | 27,452,304 | structurally consistent |
| Miami-Dade | 568,030 | 41,816 | 28 | 42,484,944 | real FMM smoke regression passed |
| Palm Beach | 443,988 | 1,138,172 | 1,206 | 23,527,095 | stale duplicate matched IDs; regenerate |

The strict `matched` verifier is expected to fail Palm Beach until a clean FMM run replaces that artifact.

## 10. Long-run monitoring, stopping, and resuming

In a second Terminal:

```bash
cd /path/to/reu-2026-driver-path-choice
pgrep -af 'roadnet-run|fmm|ubodt_gen'
ps -Ao pid,etime,%cpu,%mem,command | grep -E 'roadnet-run|fmm|ubodt_gen'
top -o cpu
tail -f logs/pipeline/full_pipeline_*.log
du -sh "$(python -c 'import yaml; print(yaml.safe_load(open("config.yaml"))["output_dir"])')"
ls -lhT sflorida_outputs/*_County/fmm/ubodt.txt
ls -lhT sflorida_outputs/*_County/*_matched.csv
```

Use `Control-C` once in the launching Terminal. Wait for subprocesses to exit, then check `pgrep`. Do not kill the machine or detach the external disk while a Parquet/CSV is being written.

Resume rules:

- OSM, Mapillary, conflation, and FMM reuse are controlled by explicit YAML/CLI flags.
- UBODT is reused only when it is newer than `edges.shp`.
- Full-run `--resume` permits existing output but never changes skip flags for you.
- FMM-only requires explicit `--overwrite-matched` or `--reuse-matched` when a matched CSV exists.
- Quarantine a corrupt partial output manually with a descriptive suffix before retrying; the wrappers never silently delete it.

## 11. Clean-room test record

Completed:

- FMM build script executed from a clean temporary build/install directory;
- pinned checkout and patch application verified;
- arm64 CLI/help, Python import, loader path, and no-`libpython` checks passed;
- real Miami FMM lifetime regression passed in 11.58 seconds;
- wrapper syntax/help and focused tests passed;
- Driver 1003 cache-only output regeneration and privacy scan passed.

Not completed:

- a new `pipeline` Conda environment from an entirely fresh clone;
- a full three-county clean OSM/FDOT/conflation/FMM run;
- a corrected Palm Beach matched run;
- unrestricted publication of legacy route-map history.

Do not describe the entire pipeline as fully clean-room tested until those items are completed. See [PHASE1_MACOS_REPRODUCTION_AUDIT.md](PHASE1_MACOS_REPRODUCTION_AUDIT.md) for stage-by-stage evidence and [TROUBLESHOOTING_MACOS_FMM.md](TROUBLESHOOTING_MACOS_FMM.md) for verified failures and fixes.
