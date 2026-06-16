# Summer 2026 REU Driver Path Choice FMM Validation

This repository is a clean-history export of the Summer 2026 REU driver path
choice / Fast Map Matching (FMM) validation work. It keeps the source code,
small validation artifacts, and documentation needed to review the workflow
without carrying large local datasets or generated pipeline outputs.

Large datasets and generated outputs are intentionally excluded. The repository
does not include driver GPS JSONL files, parquet outputs, FDOT/OSM generated
data, cache directories, run logs, external drive contents, or the old tracked
`data/` directory from the original working copy.

## Included Files

- `roadnet/`: Python package for road network enrichment, Mapillary sign
  processing, FMM aggregation, and command-line entry points.
- `visualize_134055.py`: validation plotting script for comparing raw GPS
  points with FMM-matched road segments.
- `VALIDATION.md`: notes on validation artifacts and excluded data.
- `sample.csv`: small sample of FMM `id,opath` output.
- `broward_check.html`: small interactive Folium validation map.
- `broward_134055_raw_vs_matched.png`: static plot for Broward trip 134055.
- `tests/test_trip.py` and `tests/test_stmatch.py`: small single-trip FMM
  validation utilities.
- `pyproject.toml`: package metadata and Python dependencies.

## Validation Summary

The Broward County validation run produced:

| Metric | Value |
| --- | ---: |
| Broward total trips | 4,071 |
| Matched successfully | 3,837 |
| Failed / no opath | 234 |
| Success rate | 94.25% |
| GPS points plotted | 517 |
| Matched road segments plotted | 52 |
| Missing FIDs from `edges.shp` | 0 |

## Run The Visualization

Install the package dependencies in a Python 3.10+ environment:

```bash
python -m pip install -e .
```

Run the validation plotter with local copies of the GPS JSONL, aggregated FID
JSONL, and generated FMM edge shapefile:

```bash
python visualize_134055.py \
  --gps-file /path/to/134055_gps.jsonl \
  --agg-file /path/to/Broward_County_134055_fid_aggregated.jsonl \
  --edges-file /path/to/edges.shp \
  --out-png broward_134055_raw_vs_matched.png
```

The script reads raw GPS points, extracts FMM-matched edge IDs from the
aggregated JSONL file, loads the edge shapefile, and writes a comparison plot.

## Code Changes and Validation Work

### 1. FMM Pipeline Stabilization (`roadnet/fmm_pipeline.py`)

#### Problem

The pipeline would occasionally stop during multi-county execution after Fast Map Matching completed. Earlier runs also experienced process crashes and failures during retry/STMatch recovery stages.

#### Implementation Changes

Modified the FMM pipeline execution flow by:

* Refactoring the control flow around FMM execution and retry handling.
* Separating the primary FMM workflow from STMatch fallback processing.
* Adding conditional execution paths using environment variables (e.g., aggregation-only mode).
* Introducing additional logging statements to track county-level progress and identify failure locations.
* Improving subprocess isolation so recovery failures could not terminate the main processing workflow.

#### Concepts Used

* Process isolation
* Conditional execution flags
* Exception handling
* Control flow refactoring
* Logging and debugging instrumentation

#### Result

The pipeline successfully completed county-level matching and aggregation for Miami-Dade, Broward, and Palm Beach counties without terminating during validation runs.

---

### 2. Mapillary Integration Updates (`roadnet/mapillary.py`)

#### Problem

Repeated validation runs required rebuilding enrichment layers, increasing runtime and making debugging slower.

#### Implementation Changes

Modified the Mapillary processing workflow to:

* Detect and reuse cached Mapillary outputs when available.
* Skip redundant downloads and processing steps.
* Preserve compatibility with previously generated enrichment files.

#### Concepts Used

* Cache validation
* File existence checks
* Conditional branching
* Data reuse optimization

#### Result

Subsequent validation runs completed significantly faster because previously generated enrichment data could be reused.

---

### 3. Validation Visualization Tool (`visualize_134055.py`)

#### Problem

There was no simple way to visually verify that FMM outputs aligned with the original GPS trajectory.

#### Implementation

Created a standalone Python validation script using:

* `json` for GPS and aggregated JSONL parsing
* `pandas` for tabular processing
* `GeoPandas` for shapefile loading
* `Shapely` geometry operations
* `Matplotlib` visualization

Workflow:

1. Load raw GPS observations from `134055_gps.jsonl`.
2. Extract matched road-segment FIDs from aggregated FMM output.
3. Load county road geometries from `edges.shp`.
4. Perform a join between matched FIDs and shapefile road segments.
5. Overlay GPS points and matched road segments in a single figure.

#### Concepts Used

* File parsing
* Geospatial joins
* Coordinate geometry
* Data visualization
* Cross-file validation

#### Result

Generated `broward_134055_raw_vs_matched.png` showing:

* 517 GPS observations
* 52 matched road segments

All matched FIDs were successfully located in the road network shapefile.

---

### 4. Interactive Folium Validation Map (`broward_check.html`)

#### Problem

A static image verifies alignment but does not allow interactive inspection of the route.

#### Implementation

Built an interactive Folium map that:

* Reads GPS coordinates from raw JSONL trajectory files.
* Reads matched FIDs from aggregated FMM outputs.
* Loads road geometries from the county edge shapefile.
* Creates separate Folium layers for:

  * Raw GPS points
  * Matched road segments
* Automatically fits the map bounds to the route.

#### Concepts Used

* Geospatial data processing
* Layer based map rendering
* FID to geometry joins
* Interactive web visualization

#### Result

Generated `broward_check.html` containing:

* 517 GPS points
* 52 matched road segments
* 0 missing FID references

This confirmed that every matched road segment could be traced back to a valid road geometry.

---

### 5. Match Quality Evaluation

#### Implementation

Created validation scripts using Pandas to inspect the FMM output table.

Checks performed:

* Null value detection on the `opath` column.
* Successful versus failed trip counts.
* Sampling of matched routes for manual inspection.
* Export of validation samples (`sample.csv`).

#### Concepts Used

* Data quality checks
* Null value analysis
* Exploratory data analysis (EDA)
* CSV export automation

#### Results

Broward County:

* Total trips: 4,071
* Successfully matched: 3,837
* Failed matches: 234
* Success rate: 94.25%

The `opath` column contained valid road-segment identifiers for successfully matched trips. Failed trips corresponded to rows where no valid matched path was produced.

---

### Summary

The primary contribution was validating and stabilizing the end to end FMM workflow rather than modifying the FMM algorithm itself. Work focused on:

* Pipeline debugging
* Process isolation
* Logging improvements
* Cache reuse
* Geospatial validation
* Interactive visualization
* Match quality evaluation

These efforts verified that the generated matched trajectories and aggregated road segment outputs are suitable inputs for the next phase of the project: driver route choice analysis.


## Data Policy

The clean repository is meant for source control and review. Keep large or
machine-local artifacts outside git:

- driver GPS and sensor JSONL files
- generated parquet files
- generated shapefiles and FMM outputs
- run logs
- cache directories
- external drive mount contents
- local `data/` directories

