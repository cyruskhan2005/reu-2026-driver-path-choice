# June 18, 2026 FMM Investigation

## Project Goal

The FAU REU Driver Path Choice pipeline builds enriched road networks for
Miami-Dade, Broward, and Palm Beach counties, map-matches driver GPS traces with
FMM, and aggregates GPS, accelerometer, and OBD measurements by matched road
segment.

## Zero-Skipping Run

The requested validation is a full three-county run with OSM download,
Mapillary collection, conflation, and FMM enabled:

```yaml
skip_osm: false
skip_mly: false
skip_conflation: false
skip_fmm: false
```

The run uses nine aggregation workers, the locally built FMM binary, and GPS
sessions mounted at `/Volumes/KINGSTON`.

## Original Broward Retry Hang

The June 18 run completed Broward FMM and aggregated 3,837 successful trips,
leaving 234 failed trips. It then logged `Retrying 234 failed trips with
split-matching` and remained in native FMM work indefinitely. The parent used
an unbounded `Process.join()`, while each failed trip could make repeated native
`match_wkt()` calls without progress reporting or a time limit.

The stale run consisted of the `roadnet-run` parent, its `tee` logger, a
multiprocessing resource tracker, and a spawned native postprocessing worker.
The isolated process group was inspected and terminated before debugging.

## Investigation Process

The investigation included:

- process-tree and CPU inspection before terminating stale workers;
- review of `roadnet/fmm_pipeline.py`, especially split matching and native
  postprocessing;
- syntax, diff, and focused fake-model checks;
- validation of existing parquets, shapefile component sets, UBODTs, matched
  CSVs, and aggregated JSONL files;
- read/write/create/delete tests on the project output directories and the
  exact KINGSTON directory that previously failed;
- checks of conda, FMM CLI and Python bindings, FDOT, OSM, and Mapillary access;
- cache freshness checks for master GPS and per-session sensor parquets.

## Invalid Mapillary Token

`config.yaml` initially contained the literal placeholder `token`. The fetcher
silently converted HTTP failures into empty result lists, so all grid cells
completed with zero signs. A direct diagnostic request returned OAuth error 190
(`Invalid OAuth access token`). The professor-provided token was then installed
and verified through the project's real `load_config()` path. A small Mapillary
request returned HTTP 200, and a subsequent partial run began finding signs.

The credential itself is intentionally not reproduced in this document.

## KINGSTON Permission Error

The first fixed run reached Miami-Dade aggregation but failed opening a
`*_fid_aggregated.jsonl` file under `/Volumes/KINGSTON`. The drive and session
directories were writable by the macOS user. The failure came from the managed
execution sandbox, which allowed workspace writes but denied writes to the
external volume.

Host-level tests successfully created, read, and deleted a marker in the exact
failing directory. All discovered GPS session directories reported writable,
and the volume had approximately 23 GiB free. Future full runs must be launched
with explicit host filesystem access.

## Retry Watchdog Design

The native postprocessing child now sends `start`, `match_done`, and `done`
events to its parent over a multiprocessing pipe. The parent starts a per-trip
timer when native matching begins. If matching exceeds the default 300-second
limit, the parent terminates the child, escalates to a kill if necessary, skips
the active trip, and restarts postprocessing with only the unfinished trips.

The timer is cleared before aggregation output is written, preventing watchdog
termination from truncating a successfully matched trip's JSONL output.

## Split-Match Cap

`_match_with_splits()` now allows at most 64 native matching attempts per trip
by default. The cap includes the full-trace attempt, binary-search probes, and
candidate segment attempts. A warning records when the cap is reached with
points still unmatched. The limit and watchdog can be overridden with
`ROADNET_FMM_SPLIT_MAX_ATTEMPTS` and `ROADNET_FMM_RETRY_TIMEOUT_S`.

## Progress Logging

Each failed trip now logs:

- its one-based position in the retry set;
- its trip ID;
- whether it was rescued, skipped, or remained unmatched;
- matched-point count for rescued trips;
- elapsed time.

This makes normal progress distinguishable from a native FMM stall.

## Palm Beach Missing-Trip-ID Bug

The existing Palm Beach matched CSV contained 1,206 rows but only 1,160 unique
trip IDs. FMM emitted 47 empty records as trip ID 0 and omitted 46 real IDs.
The old aggregation logic considered only explicit empty `opath` rows failed,
so omitted IDs were never retried and the valid ID 0 could be retried
unnecessarily.

Failed-trip discovery now compares successful matched IDs against every
expected trip ID. Missing IDs are treated as failed, while any ID with a valid
match is removed from the failure set. The existing Palm Beach artifact now
correctly identifies 105 failed trips, including the 46 omitted IDs.

## UBODT Freshness Bug

Full preprocessing rewrites each FMM network shapefile, but UBODT reuse
previously depended only on whether `ubodt.txt` existed. This could combine a
fresh network with stale shortest-path data. Existing network topology was
validated, but the cache rule was unsafe for future OSM changes.

The pipeline now regenerates UBODT when `edges.shp` is newer. Generation writes
to a temporary file in the same directory and atomically replaces the old
UBODT only after the child succeeds. A synthetic two-edge network verified the
atomic generation path.

## Nonzero FMM Exit Handling

The FMM CLI wrapper previously accepted any existing nonempty output CSV even
if the new FMM process exited nonzero. It now requires a zero exit status before
accepting the output, preventing stale matched CSVs from masking a failed run.

## Output and Cache Validation

- Enriched network parquets for all three counties load successfully.
- Each FMM shapefile has `.shp`, `.shx`, `.dbf`, `.prj`, and `.cpg` components.
- Existing UBODTs are readable, though a full fresh run will regenerate them
  after network shapefiles are rewritten.
- Existing Mapillary caches contain zero rows from the invalid-token run and
  are not suitable for the professor-requested result.
- Master GPS and sensor caches are newer than their source JSONL files.
- CSV, JSONL, and parquet create/read/delete tests passed in each county output
  directory.

## Remaining Risks

- UBODT/model initialization and STMatch gap bridging are not covered by the
  per-trip watchdog.
- The main FMM CLI and multiprocessing aggregation still have no global timeout.
- Mapillary and OSM remain subject to service availability and rate limiting.
- Four FDOT layers reject the county filter and are skipped with nonfatal
  warnings; the primary roadway layer loads successfully.
- The removable drive must remain mounted and the Mac must not sleep.
- The repository has no automated unit tests; validation used compilation,
  artifact inspection, and focused executable checks.
- The real Mapillary token is stored in local `config.yaml` and must not be
  copied into logs or documentation.

## Tomorrow Rerun Plan

1. Mount `/Volumes/KINGSTON` and verify at least 5 GiB free.
2. Disable sleep for the duration of the run.
3. Confirm `config.yaml` has the four full-run flags set to `false` and that
   `load_config()` reads a valid, redacted Mapillary token.
4. Confirm no stale `roadnet-run`, FMM, or multiprocessing workers exist.
5. Launch with host filesystem access, the `roadnet` conda environment, and the
   locally built FMM binary on `PATH`.
6. Use a new log file and monitor Mapillary sign counts, UBODT regeneration,
   FMM completion, aggregation progress, and per-trip Broward retry logs.
7. Verify Palm Beach starts after Broward, the pipeline exits zero, and final
   matched CSV, enriched parquet, shapefile, UBODT, and aggregated JSONL outputs
   are nonempty and readable.

Recommended command:

```bash
cd ~/Research/my-last-fmm-contribution
conda activate roadnet
export PATH="/Users/cyruskhan/Research/fmm/build:$PATH"
export MPLCONFIGDIR="/tmp/roadnet-mpl"
ROADNET_FMM_WORKERS=9 roadnet-run config.yaml 2>&1 | tee run_log_full_june19.txt
```
