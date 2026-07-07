# Driver 1003 Deliverables

This is the clean share folder for Driver 1003. Start with the timeline, then
move through monthly graphs, graph comparisons, RCCI, and finally the animation.

## Recommended Viewing Order

1. `timeline/driver_1003_timeline.html`
2. `monthly_graphs/driver_1003_monthly_graph_overview.html`
3. `graph_comparisons/driver_1003_graph_comparison_overview.html`
4. `route_choice_change_index/visuals/driver_1003_route_choice_change_index_report.html`
5. `route_evolution_animation/driver_1003_broward_county_route_evolution.gif`

## Folder Contents

- `timeline/` contains the Driver 1003 longitudinal timeline.
- `monthly_graphs/` contains monthly attributed graph reports and individual
  county/month maps.
- `graph_comparisons/` contains month-to-month graph comparison reports.
- `route_choice_change_index/` contains the Phase 3 RCCI report plus
  `data/driver_1003_rcci_summary.json` and
  `data/driver_1003_rcci_sensitivity.json` for sharing.
- `route_evolution_animation/` contains the generated month-by-month route
  evolution animation.
- `assets/` contains package-level CSS and `report_manifest.json`.

All committed HTML reports are standalone and include their report styling
inline. Interactive maps still use external Leaflet CDN and map-tile URLs.
