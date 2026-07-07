# Driver 1003 research deliverables

This folder contains presentation-ready HTML deliverables for the REU driver
path-choice project. The current research scope is the longitudinal analysis of
**Driver 1003**.

Raw CSV, Parquet, JSONL, cache, and intermediate outputs are intentionally not
committed here. The HTML pages are committed because they are the review format
for Dr. Jang and Mojtaba.

The committed reports are standalone portable HTML deliverables. Local figures,
screenshots, and other image assets are embedded directly into the HTML as
Base64 data URIs when reports are generated, so a report can be copied,
downloaded, emailed, or uploaded to Google Drive without sibling asset folders.
Interactive Leaflet maps intentionally continue to use external CDN libraries
and map-tile URLs.

## Clean share folder

Use `driver_1003/` as the clean Google Drive folder. It contains standalone HTML
reports, package metadata, JSON summaries where appropriate, and the generated
route evolution animation.

Folder structure:

- `driver_1003/README.md` - share-folder entry point and viewing order.
- `driver_1003/timeline/` - Driver 1003 monthly activity timeline.
- `driver_1003/monthly_graphs/` - monthly attributed graph overview and maps.
- `driver_1003/graph_comparisons/` - month-to-month graph comparison reports.
- `driver_1003/route_choice_change_index/` - Phase 3 RCCI report and JSON
  metric exports.
- `driver_1003/route_evolution_animation/` - GIF/MP4 route evolution animation.
- `driver_1003/assets/` - package CSS and machine-readable report manifest.

## Recommended viewing order

1. `driver_1003/timeline/driver_1003_timeline.html`
   - Shows Driver 1003 monthly route activity over time.
   - Use this first to understand the observation period and month-to-month
     trip coverage.

2. `driver_1003/monthly_graphs/driver_1003_monthly_graph_overview.html`
   - Index of monthly attributed graph maps.
   - Each monthly graph uses matched road-segment FIDs as nodes and directed
     consecutive-FID transitions as edges.

3. `driver_1003/monthly_graphs/maps/*.html`
   - Individual monthly attributed graph pages.
   - Road/FID styling reflects monthly `trip_use_count`.
   - Popups include enriched road attributes and observed Driver 1003 speed
     attributes where available.

4. `driver_1003/graph_comparisons/driver_1003_graph_comparison_overview.html`
   - Index of month-to-month county-specific graph comparison pages.
   - Shows shared, added, and removed FIDs for each county/month pair.

5. `driver_1003/graph_comparisons/county_comparisons/**/*.html`
   - County-specific comparison maps.
   - Gray = shared FIDs, green = added FIDs, red = removed FIDs.
   - County-specific pages avoid FID namespace collisions across counties.

6. `driver_1003/route_choice_change_index/visuals/driver_1003_route_choice_change_index_report.html`
   - Phase 3 Route Choice Change Index report.
   - Begin here for the final RCCI interpretation after reviewing the Phase 2
     foundation.

7. `driver_1003/route_evolution_animation/driver_1003_broward_county_route_evolution.gif`
   - Month-by-month route evolution animation for Broward County.

## Research interpretation

These deliverables demonstrate the Phase 2 foundation:

- Phase 2A: Driver 1003 longitudinal timeline.
- Phase 2B: monthly attributed graphs with FIDs as nodes and transitions as
  directed edges.
- Phase 2C: consecutive monthly graph comparison using shared/added/removed
  nodes and edges, Jaccard similarities, weighted overlaps, and data-quality
  flags.

Phase 3 adds the Route Choice Change Index (RCCI), a weighted node/edge graph
change score for consecutive monthly route behavior.
