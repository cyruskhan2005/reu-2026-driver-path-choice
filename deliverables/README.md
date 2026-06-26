# Driver 1003 HTML research deliverables

This folder contains presentation-ready HTML deliverables for the REU driver
path-choice project. The current research scope is the longitudinal analysis of
**Driver 1003**.

Raw CSV, Parquet, JSONL, cache, and intermediate outputs are intentionally not
committed here. The HTML pages are committed because they are the review format
for Dr. Jang and Mojtaba.

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

## Research interpretation

These deliverables demonstrate the Phase 2 foundation:

- Phase 2A: Driver 1003 longitudinal timeline.
- Phase 2B: monthly attributed graphs with FIDs as nodes and transitions as
  directed edges.
- Phase 2C: consecutive monthly graph comparison using shared/added/removed
  nodes and edges, Jaccard similarities, weighted overlaps, and data-quality
  flags.

The next research phase is to develop a Driver Path Choice Change Metric using
these graph comparison outputs. These pages do not yet claim or implement that
final metric.
