# FMM Validation Notes

This branch contains the macOS/FMM validation changes and small artifacts used to
inspect a Broward County trip match.

## Included validation files

- `visualize_134055.py` plots raw GPS points against matched FMM road segments.
- `sample.csv` is a small sample of FMM `id,opath` output.
- `broward_check.html` is a small interactive Folium check map.
- `broward_134055_raw_vs_matched.png` is a static validation plot.

Large generated outputs are intentionally excluded from git, including
`sflorida_outputs/`, parquet files, JSONL files, run logs, caches, and external
drive mount contents.

## Example

```bash
python visualize_134055.py \
  --gps-file /path/to/134055_gps.jsonl \
  --agg-file /path/to/Broward_County_134055_fid_aggregated.jsonl \
  --edges-file sflorida_outputs/Broward_County/fmm/edges.shp \
  --out-png broward_134055_raw_vs_matched.png
```
