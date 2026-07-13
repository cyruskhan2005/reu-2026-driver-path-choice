# Project architecture

## Purpose

The repository separates reusable production code from local restricted data
and from reviewed public outputs. The pipeline is reproducible from approved
inputs but a public clone intentionally cannot recreate participant-level maps
without the restricted GPS and GIS sources.

## Data flow

```text
restricted GPS + OSM + FDOT/county GIS
  -> roadnet.pipeline: OSM retrieval, conflation, speed/context enrichment
  -> roadnet.fmm_pipeline: FMM edge preparation, UBODT, map matching
  -> scripts/build_driver_timeline.py: Driver 1003 timeline
  -> monthly graphs and comparisons: FID use and month-to-month change
  -> roadnet.route_choice_change_index: RCCI
  -> roadnet.real_world_behavior: stays, OD pairs, road classes, POI context
  -> roadnet.behavior_report: report/map/JSON rendering
  -> outputs/public: reviewed release package
```

## Main locations

| Location | Responsibility |
|---|---|
| `roadnet/` | Production package: enrichment, FMM, RCCI, behavior analysis, report logic |
| `roadnet/cli/` | `roadnet-run` and consolidation command-line entry points |
| `scripts/` | Supported macOS setup, pipeline, report, manifest, and validation wrappers |
| `tests/` | Offline unit and release-boundary tests |
| `config.example.yaml` | Credential-free template for local paths and stage switches |
| `outputs/` | Ignored private analysis workspace; `outputs/public/` is the reviewed boundary |
| `deliverables/driver_1003/.../visuals/` | Canonical report source |
| `legacy/` | Superseded exploratory source, not the supported workflow |

## Input/output relationships

`enriched_network.parquet` and FMM edge files are county-level intermediates.
Matched paths feed the Driver 1003 timeline. The timeline feeds monthly graph
products and RCCI; matched paths plus enriched road attributes feed OD-specific
road-class and behavioral summaries. The report is rendered only after those
summaries pass reconciliation and privacy checks.

The public package is a copy, not the analysis workspace. Packaging strips
escaping local references and validation rejects credentials, restricted file
names, unsafe local links, and precise home fields.
