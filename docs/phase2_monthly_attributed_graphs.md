# Phase 2B: Driver 1003 monthly attributed FID graphs

## Research definition

This phase focuses on the longitudinal analysis of **Driver 1003**. Phase 1
produced the FMM map matching outputs and enriched road network attributes used
to build these monthly graphs.

Dr. Jang requested a month-by-month graph for the same driver. The graph used
in this phase is:

- **nodes:** matched road-segment FIDs;
- **node weight:** number of distinct trips in the month using the FID;
- **node static attributes:** road geometry, name, type, speed limit, lanes,
  AADT, functional class, land use, connector/roundabout status, and control
  node flags when available;
- **directed edges:** consecutive FID transitions within matched trips;
- **edge weight:** monthly transition frequency.

The study label **Driver 1003** maps to collection `1003 1004` and internal
pseudonymous ID `ca351c04cfabaae40cb77059ef799f4a`.

## FID nodes and monthly counts

For node usage, a FID counts at most once per trip. If one trip traverses the
same FID several times, that trip still contributes one to `trip_use_count`.

Node attributes include:

- `driver_id`, `driver_label`, and `driver_alias`;
- `month`, `county`, and `fid`;
- `trip_use_count`;
- `monthly_trip_count`;
- `trip_use_share`;
- `count_rank`;
- joined enriched road attributes and road geometry.

FID namespaces are county-specific. `(month, county, fid)` is the node key.

## Node Attribute Schema

The canonical node columns are stable across CSV and Parquet outputs:

- `fid`: county-specific matched road-segment identifier.
- `trip_use_count`: number of distinct Driver 1003 trips in the month using
  the FID; this definition is unchanged.
- `trip_use_share`: `trip_use_count / monthly_trip_count`.
- `road_length_m`: enriched-network `length`, expressed in meters.
- `oneway`: enriched OSM one-way value; null when unavailable.
- `observed_avg_speed`: mean monthly Driver 1003 GPS-derived speed for points
  positionally matched to the FID, in mph.
- `observed_median_speed`: median of the same monthly Driver 1003 point-speed
  observations, in mph.
- `road_owner_or_source`: `CUSTOM_OWNER` when available, otherwise `FDOT`
  when an FDOT roadway match exists, otherwise the network `speed_source`
  provenance such as OSM or an estimation source.
- `estimated_speed_limit`: selected network speed limit in mph.
- `lanes`: enriched lane count.
- `FDOT_AADT`: annual average daily traffic where an FDOT match exists.
- `highway`: OSM road/highway type.

Observed speed is not inherited from a global network attribute. It is
recalculated for Driver 1003 by month/county/FID from the cached county GPS CSV
and matched `opath`. Point-to-point haversine distance is divided by timestamp
difference, converted to mph, filtered to 0–120 mph, and then aggregated to
mean and median. Null values indicate that no valid matched speed observations
were available for that monthly FID.

## Transition edges

The ordered matched FID sequence is retained for edge construction. Only
adjacent duplicate FIDs are collapsed:

```text
43208|43208|43208|83107|83107|4257
```

becomes:

```text
43208|83107|4257
```

and creates:

```text
43208 -> 83107
83107 -> 4257
```

Collapsing is necessary because FMM returns one matched FID per GPS observation.
Without collapsing, time spent on one road segment would create artificial
self-transitions.

Nonconsecutive repeated transitions remain valid. Therefore:

- `transition_count` counts every transition occurrence after collapsing;
- `trip_count_using_transition` counts distinct trips containing the edge;
- `transition_share_of_month_trips` divides that distinct-trip count by the
  number of trips in the month/county;
- source and target node usage counts/ranks are joined onto each edge.

## Run

Build the complete deliverable, including maps, validation, and the local Drive
bundle:

```bash
python scripts/build_driver_1003_monthly_graphs.py
```

Optional filters:

```bash
python scripts/build_driver_1003_monthly_graphs.py \
  --driver 1003 \
  --month 2023-08 \
  --county "Broward County" \
  --top-edges 250
```

The compatibility CLI remains available:

```bash
python scripts/build_monthly_attributed_graphs.py \
  --subject 1003 \
  --prepare-drive-bundle
```

No FMM preprocessing is rerun.

## Outputs

The flat graph tables requested for delivery are under:

```text
deliverables/google_drive_phase2/driver_1003_monthly_graphs/data/
```

For every observed month/county:

```text
driver_1003_<county>_<month>_nodes.csv
driver_1003_<county>_<month>_nodes.parquet
driver_1003_<county>_<month>_edges.csv
driver_1003_<county>_<month>_edges.parquet
```

Combined files:

```text
driver_1003_all_monthly_nodes.csv
driver_1003_all_monthly_nodes.parquet
driver_1003_all_monthly_edges.csv
driver_1003_all_monthly_edges.parquet
```

Additional outputs:

```text
deliverables/google_drive_phase2/driver_1003_monthly_graphs/
  driver_1003_graph_validation.md
  driver_1003_monthly_graph_overview.html
  visuals/driver_1003_broward_county_2023-08_graph.html
  visuals/driver_1003_monthly_maps/*.html
```

Canonical generated copies also remain under
`sflorida_outputs/phase2/monthly_graphs/driver_1003/`.

Presentation-ready HTML deliverables are committed under:

```text
deliverables/driver_1003/monthly_graphs/
  driver_1003_monthly_graph_overview.html
  driver_1003_broward_county_2023-08_graph.html
  maps/*.html
```

The committed HTML files are intended for advisor review on GitHub. Raw CSV,
Parquet, JSONL, cache, and intermediate output folders are not committed as
Phase 2 deliverables.

Generated HTML reports are standalone portable deliverables. Any local image
assets referenced by report generators are embedded as Base64 data URIs in the
HTML file itself. Interactive Leaflet maps intentionally retain external CDN
and map-tile URLs, but they do not require local image folders.

## Maps

Monthly maps retain the road/FID usage layer colored and weighted by
`trip_use_count`. A toggleable transition layer displays the top transitions
for that month, ranked by `transition_count`. Transition lines connect the
representative midpoints of the source and target FID road geometries; they are
an abstract graph overlay rather than additional road geometry.

The presentation proof uses Broward County, August 2023:

```text
driver_1003_broward_county_2023-08_graph.html
```

It includes 170 trips, 4,170 FID nodes, summary cards, node popups, edge
popups, and a toggleable top-transition layer.

## Validation

The validation report checks:

- all edge source and target FIDs exist as monthly nodes;
- transition counts are positive;
- eligible observed month/county groups contain edges;
- self-loops are reported;
- zero-trip months create no graph data files;
- node and edge dataset/row totals are recorded.

April and May 2023 remain explicit zero-trip months in the overview, but do not
produce node or edge files.

## Limitations and next step

- Straight transition-overlay lines connect FID representative points and are
  intended for graph interpretation, not as physical road paths.
- County FID namespaces remain separate.
- Monthly frequency differences can reflect both route behavior and differing
  numbers of trips; normalized shares are included for this reason.
- Origin–destination route comparison remains deferred.

The next research stage is Phase 2C graph-to-graph comparison across
consecutive months using these stable FID-node and directed-transition-edge
tables. The subsequent research phase is development of a Driver Path Choice
Change Metric using the comparison outputs.
