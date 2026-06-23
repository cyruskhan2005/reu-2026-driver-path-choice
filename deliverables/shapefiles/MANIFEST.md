# Shapefile Deliverable Manifest

The required edge and node shapefile bundles exist locally under
`sflorida_outputs/`, but are intentionally not committed to GitHub because
the `.dbf` files exceed normal repository limits.

Regenerate the shapefile bundles from the repository root with:

```bash
python parquet_to_shapefiles.py
```

This exports only:

- `enriched_network.parquet` -> edge shapefile bundle
- `osm_nodes.parquet` -> node shapefile bundle

## Local shapefile bundle paths

### Miami-Dade County

Edge bundle:

- `sflorida_outputs/Miami_Dade_County/enriched_network.shp` (63M)
- `sflorida_outputs/Miami_Dade_County/enriched_network.shx` (5.3M)
- `sflorida_outputs/Miami_Dade_County/enriched_network.dbf` (1.9G)
- `sflorida_outputs/Miami_Dade_County/enriched_network.prj` (424B)
- `sflorida_outputs/Miami_Dade_County/enriched_network.cpg` (5B)

Node bundle:

- `sflorida_outputs/Miami_Dade_County/osm_nodes.shp` (6.2M)
- `sflorida_outputs/Miami_Dade_County/osm_nodes.shx` (2.7M)
- `sflorida_outputs/Miami_Dade_County/osm_nodes.dbf` (209M)
- `sflorida_outputs/Miami_Dade_County/osm_nodes.prj` (424B)
- `sflorida_outputs/Miami_Dade_County/osm_nodes.cpg` (5B)

### Broward County

Edge bundle:

- `sflorida_outputs/Broward_County/enriched_network.shp` (48M)
- `sflorida_outputs/Broward_County/enriched_network.shx` (4.4M)
- `sflorida_outputs/Broward_County/enriched_network.dbf` (1.3G)
- `sflorida_outputs/Broward_County/enriched_network.prj` (424B)
- `sflorida_outputs/Broward_County/enriched_network.cpg` (5B)

Node bundle:

- `sflorida_outputs/Broward_County/osm_nodes.shp` (4.7M)
- `sflorida_outputs/Broward_County/osm_nodes.shx` (1.4M)
- `sflorida_outputs/Broward_County/osm_nodes.dbf` (162M)
- `sflorida_outputs/Broward_County/osm_nodes.prj` (424B)
- `sflorida_outputs/Broward_County/osm_nodes.cpg` (5B)

### Palm Beach County

Edge bundle:

- `sflorida_outputs/Palm_Beach_County/enriched_network.shp` (51M)
- `sflorida_outputs/Palm_Beach_County/enriched_network.shx` (4.4M)
- `sflorida_outputs/Palm_Beach_County/enriched_network.dbf` (1.4G)
- `sflorida_outputs/Palm_Beach_County/enriched_network.prj` (424B)
- `sflorida_outputs/Palm_Beach_County/enriched_network.cpg` (5B)

Node bundle:

- `sflorida_outputs/Palm_Beach_County/osm_nodes.shp` (4.8M)
- `sflorida_outputs/Palm_Beach_County/osm_nodes.shx` (1.4M)
- `sflorida_outputs/Palm_Beach_County/osm_nodes.dbf` (165M)
- `sflorida_outputs/Palm_Beach_County/osm_nodes.prj` (424B)
- `sflorida_outputs/Palm_Beach_County/osm_nodes.cpg` (5B)

## Delivery note

If Professor Jang needs the actual shapefile bundles, provide them through
Google Drive, Box, Dropbox, a GitHub Release asset, or another large-file
delivery channel. Do not place the multi-GB shapefile bundles directly in
normal Git history.

