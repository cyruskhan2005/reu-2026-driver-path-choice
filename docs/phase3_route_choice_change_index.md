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

## Confidence labels

Confidence is kept separate from the RCCI value. RCCI is not penalized or altered by confidence.

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

## Limitations

- RCCI v1 is calibrated to Driver 1003 and should not be generalized without further validation.
- Sparse county rows can produce extreme values from very small trip counts.
- RCCI measures route-network change, not cause.
- RCCI does not distinguish planned travel changes, external events, roadway disruptions, seasonal behavior, or health-related explanations.
- This is not a clinical or diagnostic score.

## Next step

The next research step is to validate RCCI against expert review, known travel context, and manually inspected high-change periods. After validation, the metric can be refined or compared across additional longitudinal subjects if appropriate data access becomes available.
