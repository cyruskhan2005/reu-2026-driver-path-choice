# Legacy and restricted materials

This directory preserves superseded source code that may help a future REU
student understand the project’s development history. Nothing here is part of
the supported production pipeline or the public-release process.

## Research-process scripts

`research_process_scripts/` contains exploratory visualisation and formula
comparison tools developed while documenting the map-matching and RCCI process.
They may require restricted raw GPS, matched trajectories, or local network
files. They are deliberately separated from `scripts/`, whose entry points are
the supported reproducible workflow.

## Restricted artifacts kept outside Git

The following local paths are intentionally ignored rather than stored in the
public Git tree because they can contain detailed route geometry, timestamps,
endpoint coordinates, raw GIS data, or duplicated delivery copies:

- `data/`
- `sflorida_outputs/`
- `cache/`
- `deliverables/google_drive_phase2/`
- `deliverables/driver_1003/graph_comparisons/`
- `deliverables/driver_1003/monthly_graphs/`
- `deliverables/driver_1003/timeline/`
- `deliverables/driver_1003/route_evolution_animation/`
- `deliverables/driver_1003/route_choice_change_index/data/`

These may be retained locally or shared only through an approved restricted
research channel. Their absence from a clone is intentional. See
[`docs/DATA_AND_PRIVACY.md`](../docs/DATA_AND_PRIVACY.md) once that policy is
available, and `outputs/public/` for the only curated public artifacts.
