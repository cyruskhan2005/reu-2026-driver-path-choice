# Phase 3: Driver 1003 Route Choice Change Index

Phase 3 implements the Route Choice Change Index (RCCI) for Driver 1003. RCCI converts the completed Phase 2C month-to-month graph comparison outputs into an interpretable route-network change index.

RCCI is a transportation research metric. It is not a clinical score, not a dementia detector, and not a diagnostic interpretation.

## Why RCCI is needed

Phase 2C already compares consecutive monthly attributed graphs and reports:

- shared / added / removed FIDs;
- shared / added / removed directed FID transitions;
- node Jaccard similarity;
- edge Jaccard similarity;
- weighted node overlap;
- weighted edge overlap;
- data-quality flags.

RCCI summarizes those comparison outputs into a single month-to-month value while preserving confidence labels and supporting diagnostics.

## Inputs

RCCI uses existing Phase 2C outputs only:

```text
deliverables/google_drive_phase2/driver_1003_graph_comparisons/data/
  driver_1003_month_to_month_summary.parquet
  driver_1003_month_to_month_node_comparisons.parquet
  driver_1003_month_to_month_edge_comparisons.parquet
```

If Parquet files are unavailable, the script falls back to the CSV equivalents.

The primary RCCI table uses county-specific rows. `ALL_COUNTIES` rows are excluded from primary scoring because FID namespaces are county-specific and sparse county observations can be hidden by Broward County activity.

## RCCI v1 formula

RCCI v1 uses the balanced weighted formula selected in `docs/path_choice_metric_design.md`:

```text
RCCI = 100 * (
  node_weight * (1 - weighted_node_overlap_min)
  +
  edge_weight * (1 - weighted_edge_overlap_min)
)
```

Default weights:

```text
node_weight = 0.5
edge_weight = 0.5
```

The command-line script accepts custom weights. If the supplied weights do not sum to 1, they are normalized automatically and the normalized weights are written to the output table.

## How weighted overlap works

Each month is represented as a graph:

- nodes are road segments/FIDs;
- edges are directed transitions from one FID to the next.

RCCI compares two consecutive monthly graphs and measures how much road-segment usage and transition usage changed.

Weighted overlap means frequently used roads and transitions count more than rarely used roads and transitions. If a road segment was used in 40 trips, a change involving that road should matter more than a road segment used once. RCCI therefore uses trip-use counts and transition counts rather than treating every FID equally.

The exact formula is a weighted Jaccard / min-max overlap:

```text
weighted overlap =
  Σ min(weight in Month A, weight in Month B)
  /
  Σ max(weight in Month A, weight in Month B)
```

For nodes:

```text
node weight = road-segment trip-use count
weighted_node_overlap_min =
  Σ over FIDs min(trip_use_count_month_a, trip_use_count_month_b)
  /
  Σ over FIDs max(trip_use_count_month_a, trip_use_count_month_b)
```

For edges:

```text
edge weight = directed transition count
weighted_edge_overlap_min =
  Σ over directed transitions min(transition_count_month_a, transition_count_month_b)
  /
  Σ over directed transitions max(transition_count_month_a, transition_count_month_b)
```

RCCI uses these weighted overlaps, not raw FID counts, as the primary formula inputs. Raw shared/added/removed counts and Jaccard similarities are still reported as supporting diagnostics.

Node example:

| FID | Month A usage | Month B usage | min | max |
|---:|---:|---:|---:|---:|
| 100 | 40 | 38 | 38 | 40 |
| 200 | 10 | 11 | 10 | 11 |
| 300 | 2 | 0 | 0 | 2 |
| 400 | 0 | 3 | 0 | 3 |

```text
weighted overlap = (38 + 10 + 0 + 0) / (40 + 11 + 2 + 3)
                 = 48 / 56
                 = 0.857

node change = 1 - 0.857
            = 0.143
```

Although two roads changed, most high-use driving remained stable, so the node change is relatively small. If FID 100 disappeared instead, the RCCI contribution would be much larger.

Plain-language example:

```text
Month A:
  FID 100 used 40 times
  FID 200 used 1 time

Month B:
  FID 100 still used 38 times
  FID 200 disappears
```

In this case, the route network changed slightly, but not dramatically, because the heavily used road remained mostly stable. If FID 100 disappeared instead, the RCCI contribution would be much larger.

Higher-impact changes include:

- frequently used roads;
- frequently used transitions;
- roads or transitions that disappear after heavy use;
- new roads or transitions that appear repeatedly.

Lower-impact changes include:

- one-time roads;
- rare transitions;
- sparse months, which are flagged by confidence labels.

## Node component

The node component measures change in monthly FID road-segment usage:

```text
node_change_component = 1 - weighted_node_overlap_min
```

The weighted overlap is based on monthly FID trip-use counts. A low overlap means the road segments used, or their usage intensity, changed substantially.

## Edge component

The edge component measures change in directed FID transition usage:

```text
edge_change_component = 1 - weighted_edge_overlap_min
```

Edges represent directed consecutive-FID transitions within matched trips. A low edge overlap means the movement pattern through road segments changed substantially.

Two months may use many of the same roads but connect them differently. That is why edge changes are included in addition to node changes.

## Confidence labels

Confidence is kept separate from the RCCI value. RCCI is not penalized or altered by confidence.

RCCI measures route change. Confidence measures whether there are enough trips to trust the comparison. These concepts are intentionally reported separately and are not mixed into a single adjusted score.

| Confidence | Rule |
|---|---|
| LOW | either month has zero trips |
| LOW | either month has fewer than 10 trips |
| LOW | missing node or edge comparison data |
| LOW | both months have no graph |
| MEDIUM | either month has 10-24 trips |
| MEDIUM | trip count ratio is greater than 2.0 |
| HIGH | both months have at least 25 trips, trip count ratio is at most 2.0, and graph data are present |

The output also includes `confidence_reason`, such as:

- `zero_trip_month`
- `low_trip_count_under_10`
- `medium_trip_count_10_to_24`
- `trip_count_ratio_gt_2`
- `missing_comparison_data`
- `high_coverage_balanced`

## Driver 1003 interpretation bands

The interpretation bands are empirical Driver 1003 v1 thresholds. They are calibrated to the high-coverage Broward County comparison distribution and should not be treated as universal cutoffs.

For HIGH or MEDIUM confidence rows:

| RCCI v1 | Interpretation |
|---:|---|
| `< 60` | LOW RELATIVE CHANGE |
| `60-70` | MODERATE RELATIVE CHANGE |
| `70-80` | HIGH RELATIVE CHANGE |
| `>= 80` | VERY HIGH RELATIVE CHANGE |

Special cases:

- both months no trips: `NO COMPARISON`, with RCCI blank/null;
- one zero-trip month and one observed month: `ZERO-BASELINE CHANGE`, confidence LOW;
- LOW confidence rows otherwise: `LOW CONFIDENCE - interpret with trip-count context`.

## Why Broward is highlighted

Broward County is the dense longitudinal dataset for Driver 1003. The exploratory design analysis found that high-coverage comparisons with at least 10 trips in both months occur only in Broward County.

Miami-Dade and Palm Beach rows are still reported, but many of them are one-trip or zero-baseline comparisons. Those rows can have high RCCI values because the graph appears or disappears, but they should be interpreted with LOW confidence.

## Outputs

Running the script creates:

```text
deliverables/google_drive_phase2/driver_1003_route_choice_change_index/
  data/
    driver_1003_rcci_summary.csv
    driver_1003_rcci_summary.parquet
    driver_1003_rcci_sensitivity.csv
    driver_1003_rcci_sensitivity.parquet
  visuals/
    driver_1003_route_choice_change_index_report.html
  driver_1003_rcci_validation.md
```

The clean advisor-facing share folder is:

```text
deliverables/driver_1003/
```

The RCCI share-folder copy is:

```text
deliverables/driver_1003/route_choice_change_index/
```

The main summary table includes:

- driver ID;
- month pair;
- county;
- trip counts and trip count ratio;
- node and edge counts;
- weighted overlaps;
- Jaccard similarities;
- node and edge change components;
- normalized node and edge weights;
- `rcci_v1`;
- confidence label and reason;
- interpretation label;
- shared / added / removed node and edge counts;
- Phase 2C data-quality flag.

The sensitivity table includes:

- `rcci_balanced_weighted`;
- `rcci_edge_heavy_weighted`;
- `rcci_balanced_jaccard`;
- `rcci_geometric_weighted`.

## HTML report

The report is written to:

```text
deliverables/google_drive_phase2/driver_1003_route_choice_change_index/visuals/
  driver_1003_route_choice_change_index_report.html
```

It is a standalone portable HTML report using embedded CSS and SVG. It highlights the Broward County RCCI timeline, lists highest and lowest HIGH/MEDIUM confidence periods, separately reports LOW confidence rows, and links back to the Phase 2 graph deliverables.

## How to run

From the repository root:

```bash
python scripts/build_driver_1003_route_choice_change_index.py --driver 1003
```

Optional examples:

```bash
python scripts/build_driver_1003_route_choice_change_index.py \
  --driver 1003 \
  --node-weight 0.6 \
  --edge-weight 0.4
```

```bash
python scripts/build_driver_1003_route_choice_change_index.py \
  --driver 1003 \
  --county "Broward County"
```

## Deliverable folder structure

The clean Google Drive share folder is organized as:

```text
deliverables/driver_1003/
  README.md
  timeline/
  monthly_graphs/
  graph_comparisons/
  route_choice_change_index/
    data/
    visuals/
  route_evolution_animation/
  assets/
```

Recommended viewing order:

1. `timeline/driver_1003_timeline.html`
2. `monthly_graphs/driver_1003_monthly_graph_overview.html`
3. `graph_comparisons/driver_1003_graph_comparison_overview.html`
4. `route_choice_change_index/visuals/driver_1003_route_choice_change_index_report.html`
5. `route_evolution_animation/driver_1003_broward_county_route_evolution.gif`

## Regenerating reports

Run the report stages in order:

```bash
python scripts/build_driver_timeline.py --driver auto
python scripts/build_driver_1003_monthly_graphs.py --driver 1003
python scripts/compare_driver_1003_monthly_graphs.py --driver 1003 --all
python scripts/build_driver_1003_route_choice_change_index.py --driver 1003 --all
```

The monthly graph and comparison stages use cached Phase 2 outputs and write the
large Google Drive bundle under `deliverables/google_drive_phase2/`. The curated
GitHub/share folder is `deliverables/driver_1003/`.

## HTML validation

Validate every generated HTML report in the share folder with:

```bash
python scripts/validate_html_deliverables.py deliverables/driver_1003
```

The validator checks local HTML, CSS, JavaScript, image, and iframe references,
internal anchors, and map-like pages without a detectable map container. External
Leaflet CDN and tile URLs are intentionally treated as external dependencies.

## Route evolution animation

Generate the reusable route evolution animation with:

```bash
python scripts/generate_route_evolution_animation.py \
  --driver 1003 \
  --county "Broward County" \
  --output-dir deliverables/driver_1003/route_evolution_animation
```

The generator reads monthly FID geometry from
`deliverables/google_drive_phase2/driver_1003_monthly_graphs/data/driver_1003_all_monthly_nodes.csv`
by default. Each frame uses the same geographic extent, same map style, same
zoom level, a month label, and only that month's routes. It always writes a GIF.
It also writes an MP4 when `ffmpeg` is installed.

Dependencies for the animation generator:

```bash
python -m pip install -e ".[deliverables]"
```

Optional MP4 dependency:

```bash
brew install ffmpeg
```

## Limitations

- RCCI v1 is calibrated to Driver 1003 and should not be generalized without further validation.
- Sparse county rows can produce extreme values from very small trip counts.
- RCCI measures route-network change, not cause.
- RCCI does not distinguish planned travel changes, external events, roadway disruptions, seasonal behavior, or health-related explanations.
- This is not a clinical or diagnostic score.

## Next step

The next research step is to validate RCCI against expert review, known travel context, and manually inspected high-change periods. After validation, the metric can be refined or compared across additional longitudinal subjects if appropriate data access becomes available.
