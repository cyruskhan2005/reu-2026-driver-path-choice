# Professor Jang Deliverables

This folder contains the GitHub-safe deliverables for the June 2026 FMM run.

## 1. Shapefiles: edges and nodes

The edge/node shapefile bundles were generated locally from:

- `sflorida_outputs/*_County/enriched_network.parquet`
- `sflorida_outputs/*_County/osm_nodes.parquet`

The actual shapefile bundles are not committed to Git because the `.dbf`
components are hundreds of MB to multiple GB and exceed normal GitHub file
limits. See `shapefiles/MANIFEST.md` for exact local paths, sizes, and the
regeneration command.

## 2. HTML map visualizations

Representative clickable HTML maps are committed in `html_maps/`:

- `html_maps/miami_dade_trip_6_enriched.html`
- `html_maps/broward_trip_3967_enriched.html`
- `html_maps/palm_beach_trip_120_enriched.html`

Each map includes raw GPS, matched road segments, START/END markers, and
clickable enriched road attributes.

## 3. UBODT regeneration/performance check

The UBODT performance summary and pre-run manifest are committed in
`ubodt_performance/`.

The full UBODT files are not committed because they are generated FMM cache
files between roughly 1 GB and 2 GB per county.

## 4. Matched CSV files

Matched CSV files for all three counties are committed in `matched_csv/`.
These define the trip-to-road-segment `opath` mapping used for monthly
attributed graph work.

