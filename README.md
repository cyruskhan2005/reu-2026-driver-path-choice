# Driver Path Choice and Longitudinal Behavior Research

## 1. What this project does

This repository measures how a driver's destinations and road choices change over time. It enriches South Florida road networks, map-matches recorded GPS trips with Fast Map Matching (FMM), calculates the Route Choice Change Index (RCCI), groups similar trips into route families, and turns those results into privacy-aware behavioral findings.

The primary research subject is Driver 1003. Activity roles are inferred from timing, reconstructed stays, repeated trip chains, and map context; they are not confirmed facts about residence, employment, school, healthcare, or relationships.

## 2. Pipeline overview

```text
restricted GPS/sensor sessions + OSM + FDOT/county road data
  -> roadnet enrichment and speed/context attributes
  -> per-county enriched_network.parquet and FMM edges.shp
  -> UBODT generation and FMM map matching
  -> county *_gps.csv, *_matched.csv, and optional sensor aggregation
  -> Driver 1003 timeline and monthly attributed graphs
  -> month-to-month graph comparisons and RCCI
  -> stays, recurring places, route families, longitudinal report, and map
```

The current stage boundaries are implemented in `roadnet/pipeline.py`, `roadnet/fmm_pipeline.py`, the Phase 2 scripts under `scripts/`, and `roadnet/real_world_behavior.py`.

## 3. Repository layout

| Path | Purpose |
|---|---|
| `roadnet/` | Production enrichment, FMM, RCCI, activity, POI, and report code |
| `scripts/` | Reproducible command-line entry points and validation tools |
| `tests/` | Unit, privacy, route-family, FMM lifetime, and release tests |
| `docs/` | Detailed methods, macOS setup, audit history, and troubleshooting |
| `patches/fmm-macos-arm64.patch` | Verified patch for pinned FMM on Apple Silicon/Python 3.11 |
| `environment-macos.yml` | Python environment definition (`pipeline`) |
| `config.example.yaml` | Credential-free local configuration template |
| `outputs/public/` | Curated public artifacts only; private/generated outputs stay outside Git |
| `sflorida_outputs/` | Large local Phase 1/2 products; generated locally and ignored |
| `deliverables/driver_1003/` | Research report tree; only a reviewed public copy is released |

## 4. Fastest verified path

This path reuses known-good OSM/enriched-network/FMM files. It still requires the restricted GPS session hierarchy named by `gps_root` and a writable output/session volume.

```bash
# Starting directory: repository root; active environment: pipeline
conda activate pipeline
cp config.example.yaml config.yaml       # safe once; config.yaml is ignored
# Edit output_dir, gps_root, fdot_gdb, and county GIS paths in config.yaml.

scripts/run_fmm_only_macos.sh \
  --config config.yaml \
  --overwrite-matched

scripts/generate_driver_1003_report.sh \
  --config config.yaml \
  --google-mode offline
```

Success ends with `FMM-only run completed and validated` and `Driver 1003 report generation completed`. The FMM command explicitly allows replacement; omit it if no matched CSV exists. Report generation rebuilds Phase 2A/B/C by default, packages the reviewed public report/map/JSON, and is safe to rerun over its stable generated filenames. It makes no Google request in `offline` mode.

## 5. Full macOS setup from a fresh Terminal

Tested native-build host: macOS 26.5.1, Apple Silicon `arm64`, zsh 5.9, Miniconda/Conda 26.3.2, Python 3.11.15, Homebrew `/opt/homebrew`, CMake 4.3.3, Apple clang 21, SWIG 4.4.1, Boost 1.90.0, GDAL 3.13.1, and libomp 22.1.7. Intel macOS is not currently a verified target.

```bash
# Starting directory: a fresh Terminal in your home directory
xcode-select -p || xcode-select --install
command -v brew || open https://brew.sh
command -v conda || printf '%s\n' 'Install Miniconda or Miniforge, then reopen Terminal.'

git clone https://github.com/cyruskhan2005/reu-2026-driver-path-choice.git
cd reu-2026-driver-path-choice
git checkout feature/driver-1003-longitudinal-insights-and-reproducibility
git submodule update --init --recursive

# Creates/updates the pipeline environment and builds pinned FMM.
scripts/bootstrap_macos.sh --env-name pipeline
conda activate pipeline

command -v fmm
command -v ubodt_gen
file "$(command -v fmm)"
fmm --help
ubodt_gen --help
python scripts/verify_pipeline_outputs.py --config config.example.yaml --stage environment
```

`bootstrap_macos.sh` installs missing Homebrew build formulae, creates the Conda environment, installs this package editable, builds FMM commit `19ef34e1f57ff2f2484231aa0d01dfffea986ec1`, and verifies loader-relative native linkage. It downloads Homebrew/Conda/FMM dependencies and is safe to rerun; `--skip-brew` and `--skip-fmm` are available for controlled reuse. See [the detailed macOS guide](docs/MACOS_PIPELINE_GUIDE.md) before configuring private data.

## 6. Full clean rebuild

Copy the template to ignored `config.yaml`. For a clean run, use a new `output_dir`; set `skip_osm`, `skip_mly`, `skip_conflation`, and `skip_fmm` to `false`. `mapillary_enabled: false` is the supported credential-free mode. The wrapper refuses existing enriched outputs unless `--resume` is explicit.

```bash
# Starting directory: repository root; active environment: pipeline
conda activate pipeline
scripts/run_full_pipeline_macos.sh --config config.yaml

# Validate individual boundaries after the run.
python scripts/verify_pipeline_outputs.py --config config.yaml --stage enrichment
python scripts/verify_pipeline_outputs.py --config config.yaml --stage fmm-prep
python scripts/verify_pipeline_outputs.py --config config.yaml --stage matched
```

This can download OSM data and take hours; actual duration depends on network, county GIS, GPS volume, and disk. The known local caches occupy roughly 95 MB for three enriched networks and 3.7 GB for three UBODTs, before GPS, maps, logs, and intermediate files. Resume only after reviewing the YAML skip flags.

## 7. FMM-only run

FMM-only mode requires, for every selected county:

- `osm_nodes.parquet`, `osm_edges.parquet`, and `osm_landuse.parquet`;
- nonempty `enriched_network.parquet`;
- `fmm/edges.shp`, `.shx`, `.dbf`, `.prj`, and `.cpg`;
- `ubodt.txt` or permission to create it;
- the writable restricted `gps_root` session hierarchy;
- `skip_fmm: false` and a working `fmm_bin` (normally `fmm`).

```bash
# Starting directory: repository root; active environment: pipeline
conda activate pipeline
scripts/run_fmm_only_macos.sh \
  --config config.yaml \
  --county "Miami-Dade County" \
  --overwrite-matched
```

Use `--reuse-matched` only to rerun aggregation intentionally from an existing matched CSV. The wrapper passes `--skip-osm --skip-mly --skip-conflation --disable-mapillary`, validates cached inputs first, writes a timestamped log, and fails rather than silently replacing a known-good matched file. The current Palm Beach matched artifact is known to contain duplicate IDs and must be regenerated before it can pass the strict verifier.

## 8. Driver 1003 report generation

The report wrapper rebuilds the Driver 1003 timeline, monthly graphs, consecutive-month comparisons, canonical RCCI report, place/stay analysis, route families, JSON, and verification map. Use `--reuse-phase2` only when the wrapper's Phase 2 prerequisites already exist.

```bash
# Starting directory: repository root; active environment: pipeline
conda activate pipeline
scripts/generate_driver_1003_report.sh \
  --config config.yaml \
  --google-mode offline
```

Google modes are explicit:

- `offline`: no Google request; use local OSM/GIS only.
- `cache`: reuse sanitized ignored cache entries; no network request.
- `network`: require `GOOGLE_MAPS_API_KEY` in the environment and enforce `prior + budget <= 99`.

The research run documented here used 84 cumulative Google attempts/requests and the final cache-only rebuild used 51 cache hits. Never carry that count into a separate research run; start a new run's accounting at zero.

## 9. Expected outputs

| Output | Meaning |
|---|---|
| `<output_dir>/<County>/enriched_network.parquet` | Enriched directed road edges |
| `<output_dir>/<County>/fmm/edges.shp` | FMM network input |
| `<output_dir>/<County>/fmm/ubodt.txt` | FMM upper-bounded origin-destination table |
| `<output_dir>/<County>/<County>_gps.csv` | Semicolon-delimited FMM GPS input |
| `<output_dir>/<County>/<County>_matched.csv` | FMM matched path (`opath`) per trip |
| `<output_dir>/phase2/driver_timelines/driver_1_timeline.csv` | Canonical Driver 1003 trip fragments |
| `outputs/driver_1003_trip_summary.csv` | 3,284 reconciled source trips |
| `outputs/driver_1003_poi_enriched_clusters.csv` | Private research cluster/POI audit |
| `outputs/driver_1003_route_family_monthly_shares.csv` | Monthly family counts/shares and sufficiency |
| `outputs/driver_1003_longitudinal_route_transitions.csv` | Supported full-period route changes |
| `outputs/driver_1003_real_world_behavior_insights.json` | Privacy-safe structured findings |
| `outputs/driver_1003_poi_route_insights_map.html` | Interactive verification map |
| `deliverables/driver_1003/route_choice_change_index/visuals/driver_1003_route_choice_change_index_report.html` | Canonical technical + behavioral report |

The canonical public copies and checksums are described in [outputs/README.md](outputs/README.md). Large/private outputs are generated locally and are not normal Git artifacts.

## 10. Validation

```bash
# Starting directory: repository root; active environment: pipeline
conda activate pipeline
python -m unittest -v \
  tests.test_pipeline_reproducibility \
  tests.test_real_world_behavior \
  tests.test_behavior_cluster_stability \
  tests.test_behavior_quality \
  tests.test_behavior_longitudinal \
  tests.test_behavior_routes \
  tests.test_public_release_tools
python scripts/verify_pipeline_outputs.py --config config.yaml --stage driver
python scripts/validate_public_release.py --public-dir outputs/public --require-manifest
python scripts/generate_output_manifest.py \
  --root outputs/public \
  --output outputs/public/manifest.json \
  --check
```

The stage verifier checks imports, native architecture/linkage, Parquet schemas/CRS/FID uniqueness, shapefile/UBODT freshness, matched IDs and paths, route-share reconciliation, and basic secret safety. The public-release validator adds curated-tree link, key-pattern, absolute-path, and privacy checks.

## 11. Configuration and credentials

- `config.yaml` is local and ignored; never commit it.
- Mapillary is off by default. If enabled, keep `mly_token` only in ignored local configuration.
- Google enrichment reads `GOOGLE_MAPS_API_KEY` only from the server-side environment. Never put it in YAML, notebooks, logs, HTML, JavaScript, shell scripts, caches, or documentation.
- Google cache responses are sanitized and ignored under `cache/google_maps/`.
- Raw GPS/sensor sessions, exact home data, large networks, UBODTs, and matched trajectories remain private/local.
- The wrappers print paths and counts, not credential values.

## 12. Troubleshooting

Start with [docs/TROUBLESHOOTING_MACOS_FMM.md](docs/TROUBLESHOOTING_MACOS_FMM.md). It covers command discovery, `dyld`, the reproduced Python 3.9/3.11 segmentation fault, ARM/Intel mismatch, GDAL/Conda conflicts, missing caches, long runs, safe interruption, external volumes, and Gatekeeper.

Monitor a long run from a second Terminal:

```bash
pgrep -af 'roadnet-run|fmm|ubodt_gen'
ps -Ao pid,etime,%cpu,%mem,command | grep -E 'roadnet-run|fmm|ubodt_gen'
tail -f logs/pipeline/full_pipeline_*.log
du -sh sflorida_outputs
ls -lhT sflorida_outputs/*_County/*_matched.csv
```

## 13. Research outputs and limitations

The likely home is reported only at a generalized neighborhood scale. No defensible workplace or school/daycare was identified. POI proximity does not prove a visit; endpoints can fall in shared parking lots, campus access roads, or neighboring parcels. Censored stays are excluded from dwell medians. Route changes do not prove congestion, construction, toll avoidance, or preference as a cause.

Legacy tracked route maps contain private geometry and are not part of the reviewed public package. The public report is separately sanitized; raw-GPS imagery, exact-home geometry, private caches, and unrestricted legacy HTML must not be published.

## 14. Development and testing

```bash
# Starting directory: repository root; active environment: pipeline
python -m unittest -v tests.test_pipeline_reproducibility
python -m unittest -v tests.test_real_world_behavior
python -m unittest -v tests.test_behavior_longitudinal
python -m unittest -v tests.test_fmm_result_lifetime
bash -n scripts/*.sh
```

Offline CI runs synthetic/release checks without Google credentials or private data. See `.github/workflows/offline-ci.yml`. New code should preserve explicit trip boundaries, county-scoped FIDs, secret-free caches, and public-home redaction.

## 15. Citation, license, and data-source notes

No repository-wide software license was present during this audit; do not assume redistribution rights beyond the repository owner's authorization. Cite the project repository, commit, and output manifest in research use.

OpenStreetMap data are obtained through OSMnx and remain subject to OpenStreetMap attribution/licensing. FMM is pinned from `cyang-kth/fmm` and retains its upstream license. FDOT and county GIS files must be acquired from their official custodians or the approved research data store and used under their applicable terms. Mapillary and Google results are optional enrichment sources governed by their respective terms; their credentials and raw caches are not distributed.

Detailed evidence is in [docs/PHASE1_MACOS_REPRODUCTION_AUDIT.md](docs/PHASE1_MACOS_REPRODUCTION_AUDIT.md), and output provenance is in [outputs/README.md](outputs/README.md).
