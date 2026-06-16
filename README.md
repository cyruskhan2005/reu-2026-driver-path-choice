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

