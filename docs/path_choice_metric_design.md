# Route Choice Change Index (RCCI) Design Analysis

This document is an exploratory analysis of the existing Driver 1003 month-to-month graph comparison outputs. It supported the data-driven design of the Phase 3 Route Choice Change Index (RCCI).

The implemented Phase 3 metric name is:

```text
Route Choice Change Index (RCCI) v1
```

The analysis uses the completed Phase 2C outputs:

```text
deliverables/google_drive_phase2/driver_1003_graph_comparisons/data/
  driver_1003_month_to_month_summary.parquet
  driver_1003_month_to_month_node_comparisons.parquet
  driver_1003_month_to_month_edge_comparisons.parquet
```

## Input inventory

The comparison outputs contain:

| Item | Count |
|---|---:|
| Summary rows | 132 |
| County-specific summary rows | 99 |
| Combined `ALL_COUNTIES` rows | 33 |
| Calendar month pairs | 33 |
| Counties represented | Broward County, Miami-Dade County, Palm Beach County |
| Node comparison rows | 124,240 |
| Edge comparison rows | 138,210 |

The metric should primarily use county-specific rows. The `ALL_COUNTIES` rows are useful as a dashboard summary, but county-specific scoring is safer because FID namespaces are county-specific and because sparse county observations can otherwise be hidden by Broward activity.

## County coverage

| County | Comparison rows | Non-empty rows | Total trips in month A | Total trips in month B | Shared nodes | Added nodes | Removed nodes | Shared edges | Added edges | Removed edges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Broward County | 33 | 32 | 3,197 | 3,225 | 38,356 | 41,428 | 40,567 | 38,322 | 48,469 | 47,564 |
| Miami-Dade County | 33 | 2 | 1 | 1 | 0 | 120 | 120 | 0 | 119 | 119 |
| Palm Beach County | 33 | 24 | 26 | 26 | 233 | 1,708 | 1,708 | 217 | 1,700 | 1,700 |

Interpretation:

- Broward County is the only county with dense longitudinal coverage.
- Miami-Dade and Palm Beach rows are important but usually sparse. They should be scored, but confidence should usually be low.
- Extreme scores are often caused by zero-baseline or one-trip county observations, not necessarily sustained route-choice changes.

## Distribution analysis

Statistics below use county-specific rows and exclude rows where both months have no trips.

### Node metrics

| Metric | Count | Min | Q1 | Median | Mean | Q3 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Node Jaccard similarity | 58 | 0.000 | 0.000 | 0.193 | 0.190 | 0.320 | 0.937 |
| Weighted node overlap | 58 | 0.000 | 0.000 | 0.194 | 0.190 | 0.340 | 0.937 |
| Added node percentage | 58 | 0.000 | 0.152 | 0.334 | 0.406 | 0.659 | 1.000 |
| Removed node percentage | 58 | 0.000 | 0.143 | 0.293 | 0.405 | 0.565 | 1.000 |

### Edge metrics

| Metric | Count | Min | Q1 | Median | Mean | Q3 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Edge Jaccard similarity | 58 | 0.000 | 0.000 | 0.176 | 0.169 | 0.288 | 0.839 |
| Weighted edge overlap | 58 | 0.000 | 0.000 | 0.177 | 0.173 | 0.310 | 0.839 |
| Added edge percentage | 58 | 0.000 | 0.172 | 0.346 | 0.416 | 0.665 | 1.000 |
| Removed edge percentage | 58 | 0.000 | 0.155 | 0.313 | 0.415 | 0.573 | 1.000 |

### Trip metrics

| Metric | Count | Min | Q1 | Median | Mean | Q3 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Trips in month A | 58 | 0.000 | 1.000 | 44.000 | 55.586 | 110.000 | 170.000 |
| Trips in month B | 58 | 0.000 | 1.000 | 55.000 | 56.069 | 110.000 | 170.000 |
| Minimum trips in pair | 58 | 0.000 | 0.000 | 5.000 | 43.000 | 84.000 | 167.000 |
| Average trips in pair | 58 | 0.500 | 1.000 | 56.500 | 55.828 | 105.500 | 168.500 |
| Trip count ratio | 38 | 1.000 | 1.073 | 1.438 | 2.535 | 1.982 | 28.400 |

Sparse comparison counts:

- County-specific rows: 99
- Non-empty rows: 58
- Both-months-no-trips rows: 41
- One-month-zero-trip rows: 20
- Low-trip rows already flagged by Phase 2C `< 10 trips`: 10
- Nonzero rows with minimum trip count `< 10`: 10
- Nonzero rows with minimum trip count `< 25`: 10
- Rows with trip count ratio `> 2`: 7

## Meaningful high-coverage subset

The data behave differently after filtering to rows with at least 10 trips in both months. This subset contains 28 rows, all in Broward County.

| Metric | Count | Min | Q1 | Median | Mean | Q3 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Weighted node overlap | 28 | 0.171 | 0.256 | 0.340 | 0.345 | 0.431 | 0.514 |
| Weighted edge overlap | 28 | 0.159 | 0.238 | 0.309 | 0.315 | 0.396 | 0.466 |
| Node Jaccard similarity | 28 | 0.184 | 0.276 | 0.318 | 0.340 | 0.399 | 0.480 |
| Edge Jaccard similarity | 28 | 0.172 | 0.246 | 0.287 | 0.303 | 0.359 | 0.423 |
| Balanced weighted score | 28 | 51.006 | 58.657 | 67.559 | 67.022 | 75.335 | 83.471 |
| Trip count ratio | 28 | 1.009 | 1.146 | 1.438 | 1.577 | 1.811 | 3.676 |

Important observation: for meaningful Broward comparisons, even the most stable month pair still has a balanced weighted change score around 51. This means the raw score should not use generic bands such as 0-25 = low, 25-50 = moderate, 50-75 = high without calibration. The observed dataset has a high baseline month-to-month graph turnover.

## Interesting month pairs

| Category | Month pair | County | Trips | Nodes | Edges | Quality | Weighted node overlap | Weighted edge overlap |
|---|---|---|---:|---:|---:|---|---:|---:|
| Most similar overall by balanced weighted score | 2024-01 → 2024-02 | Palm Beach County | 1 → 1 | 184 → 184 | 183 → 183 | low_trip_count_month | 0.937 | 0.839 |
| Most different overall by balanced weighted score | 2023-08 → 2023-09 | Miami-Dade County | 0 → 1 | 0 → 120 | 0 → 119 | month_a_no_trips | 0.000 | 0.000 |
| Largest increase in road usage by total node weight | 2023-05 → 2023-06 | Broward County | 0 → 110 | 0 → 3,092 | 0 → 3,319 | month_a_no_trips | 0.000 | 0.000 |
| Largest decrease in road usage by total node weight | 2022-07 → 2022-08 | Broward County | 142 → 5 | 3,176 → 407 | 3,520 → 410 | low_trip_count_month | 0.051 | 0.047 |
| Largest edge change by weighted edge change | 2023-08 → 2023-09 | Miami-Dade County | 0 → 1 | 0 → 120 | 0 → 119 | month_a_no_trips | 0.000 | 0.000 |
| Largest node change by weighted node change | 2023-08 → 2023-09 | Miami-Dade County | 0 → 1 | 0 → 120 | 0 → 119 | month_a_no_trips | 0.000 | 0.000 |
| Largest trip count imbalance | 2022-07 → 2022-08 | Broward County | 142 → 5 | 3,176 → 407 | 3,520 → 410 | low_trip_count_month | 0.051 | 0.047 |

For high-coverage Broward rows only, the largest balanced weighted changes are:

| Month pair | Trips | Nodes | Edges | Weighted node overlap | Weighted edge overlap | Balanced weighted score | Quality |
|---|---:|---:|---:|---:|---:|---:|---|
| 2021-09 → 2021-10 | 34 → 125 | 808 → 2,955 | 871 → 3,237 | 0.171 | 0.159 | 83.471 | ok |
| 2021-11 → 2021-12 | 70 → 135 | 1,863 → 4,094 | 1,974 → 4,459 | 0.216 | 0.195 | 79.441 | ok |
| 2024-01 → 2024-02 | 109 → 61 | 2,990 → 1,667 | 3,181 → 1,815 | 0.226 | 0.208 | 78.285 | ok |
| 2024-03 → 2024-04 | 56 → 124 | 1,744 → 2,474 | 1,836 → 2,765 | 0.228 | 0.209 | 78.152 | ok |
| 2021-10 → 2021-11 | 125 → 70 | 2,955 → 1,863 | 3,237 → 1,974 | 0.237 | 0.214 | 77.458 | ok |

The most stable high-coverage Broward rows are:

| Month pair | Trips | Nodes | Edges | Weighted node overlap | Weighted edge overlap | Balanced weighted score | Quality |
|---|---:|---:|---:|---:|---:|---:|---|
| 2023-07 → 2023-08 | 167 → 170 | 3,353 → 4,170 | 3,732 → 4,548 | 0.514 | 0.466 | 51.006 | ok |
| 2022-05 → 2022-06 | 122 → 120 | 3,729 → 3,152 | 3,998 → 3,478 | 0.478 | 0.432 | 54.519 | ok |
| 2022-06 → 2022-07 | 120 → 142 | 3,152 → 3,176 | 3,478 → 3,520 | 0.463 | 0.419 | 55.923 | ok |
| 2022-10 → 2022-11 | 161 → 104 | 3,166 → 1,923 | 3,523 → 2,117 | 0.452 | 0.422 | 56.305 | ok |
| 2024-05 → 2024-06 | 54 → 62 | 1,591 → 1,669 | 1,675 → 1,776 | 0.453 | 0.406 | 57.048 | ok |

## Correlation analysis

Pearson correlations on non-empty county-specific rows:

| Relationship | Correlation |
|---|---:|
| Minimum trip count vs node Jaccard similarity | 0.727 |
| Minimum trip count vs edge Jaccard similarity | 0.727 |
| Minimum trip count vs weighted node overlap | 0.771 |
| Minimum trip count vs weighted edge overlap | 0.775 |
| Trip count ratio vs weighted node overlap | -0.297 |
| Trip count ratio vs weighted edge overlap | -0.296 |
| Node Jaccard similarity vs edge Jaccard similarity | 1.000 |
| Weighted node overlap vs weighted edge overlap | 1.000 |
| Weighted node overlap vs node Jaccard similarity | 0.982 |
| Weighted edge overlap vs edge Jaccard similarity | 0.984 |

Pearson correlations on the high-coverage subset with at least 10 trips in both months:

| Relationship | Correlation |
|---|---:|
| Minimum trip count vs weighted node overlap | 0.800 |
| Minimum trip count vs weighted edge overlap | 0.802 |
| Trip count ratio vs weighted node overlap | -0.710 |
| Trip count ratio vs weighted edge overlap | -0.696 |
| Node Jaccard similarity vs edge Jaccard similarity | 0.998 |
| Weighted node overlap vs weighted edge overlap | 0.999 |
| Weighted node overlap vs node Jaccard similarity | 0.835 |
| Weighted edge overlap vs edge Jaccard similarity | 0.845 |

Interpretation:

- Similarity is strongly affected by trip count. Sparse or imbalanced month pairs tend to look more different.
- Node and edge similarities are nearly collinear in this dataset. Edge metrics still matter conceptually, but the current Driver 1003 data do not show large disagreement between node and edge change.
- Weighted overlap and Jaccard similarity are highly correlated. Weighted overlap is still preferable because it accounts for repeated use rather than only membership.

## Candidate metric evaluation

The following candidates were computed offline:

| Candidate | Formula |
|---|---|
| A: balanced weighted | `100 * (0.5 * (1 - weighted_node_overlap) + 0.5 * (1 - weighted_edge_overlap))` |
| B: edge-heavy weighted | `100 * (0.3 * (1 - weighted_node_overlap) + 0.7 * (1 - weighted_edge_overlap))` |
| C: balanced Jaccard | `100 * (0.5 * (1 - node_jaccard) + 0.5 * (1 - edge_jaccard))` |
| D: geometric weighted | `100 * (1 - sqrt(weighted_node_overlap * weighted_edge_overlap))` |

Candidate score distributions on non-empty county-specific rows:

| Candidate | Count | Min | Q1 | Median | Mean | Q3 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: balanced weighted | 58 | 11.198 | 67.531 | 81.456 | 81.866 | 100.000 | 100.000 |
| B: edge-heavy weighted | 58 | 13.151 | 68.128 | 81.788 | 82.201 | 100.000 | 100.000 |
| C: balanced Jaccard | 58 | 11.198 | 69.601 | 81.518 | 82.052 | 100.000 | 100.000 |
| D: geometric weighted | 58 | 11.332 | 67.566 | 81.475 | 81.886 | 100.000 | 100.000 |

Candidate score distributions on high-coverage rows with at least 10 trips in both months:

| Candidate | Count | Min | Q1 | Median | Mean | Q3 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: balanced weighted | 28 | 51.006 | 58.657 | 67.559 | 67.022 | 75.335 | 83.471 |
| B: edge-heavy weighted | 28 | 51.975 | 59.357 | 68.168 | 67.623 | 75.697 | 83.711 |
| C: balanced Jaccard | 28 | 54.953 | 62.100 | 69.741 | 67.825 | 73.920 | 82.244 |
| D: geometric weighted | 28 | 51.066 | 58.695 | 67.595 | 67.057 | 75.352 | 83.482 |

Candidate observations:

- Candidate A and Candidate D are almost identical because node and edge weighted overlaps are highly correlated.
- Candidate B changes rankings only slightly because edge and node overlap move together in this dataset.
- Candidate C is more set-based and ignores usage intensity. It is useful as a supporting metric, but it should not be the primary index.
- All candidates produce extreme scores for zero-baseline and one-trip county comparisons. That is not a formula problem; it is a confidence/reporting problem.
- The most intuitive rankings come from Candidate A with explicit confidence labels and calibrated interpretation bands.

## Exploratory plots

Exploratory plots were generated for internal review only under:

```text
/tmp/path_choice_metric_design_figures/
```

Generated files:

- `weighted_node_overlap_hist.png`
- `weighted_edge_overlap_hist.png`
- `broward_trip_count_timeline.png`
- `broward_weighted_overlap_timeline.png`
- `node_vs_edge_weighted_overlap_scatter.png`
- `trip_count_vs_change_scatter.png`

These plots are not committed because they are exploratory artifacts rather than final research deliverables.

## Recommendations for Phase 3 RCCI metric design

### Recommended base formula

Use Candidate A as the first implementation:

```text
RCCI v1 =
100 * (
  0.5 * (1 - weighted_node_overlap_min)
  +
  0.5 * (1 - weighted_edge_overlap_min)
)
```

Rationale:

- Weighted overlap captures usage intensity, not only membership.
- A 50/50 node-edge split is defensible because the graph definition includes both road segments and transitions.
- Edge-heavy weighting does not materially change rankings for Driver 1003 because node and edge metrics are nearly collinear.
- Keeping the formula simple makes the first version easier to explain to Dr. Jang and Mojtaba.

### Recommended supporting metrics

Report these alongside the index:

- node Jaccard similarity
- edge Jaccard similarity
- weighted node overlap
- weighted edge overlap
- added nodes
- removed nodes
- added edges
- removed edges
- trips in month A
- trips in month B
- trip count ratio
- confidence label

Do not hide low-quality rows. Instead, show the score and label its reliability.

### Recommended confidence thresholds

Use these confidence labels:

| Confidence | Rule |
|---|---|
| LOW | either month has zero trips |
| LOW | either month has fewer than 10 trips |
| LOW | missing node or edge comparison data |
| LOW | both months have no graph |
| MEDIUM | either month has 10-24 trips |
| MEDIUM | trip count ratio is greater than 2.0 |
| HIGH | both months have at least 25 trips and trip count ratio is at most 2.0 |

Rationale:

- The existing Phase 2C `low_trip_count_month` threshold of 10 is supported by the data. Rows below 10 trips include many one-trip or zero-baseline comparisons with unstable similarity.
- A second threshold at 25 trips is useful because the high-coverage Broward rows all have at least 34 trips in both months. Comparisons with 10-24 trips may be usable but should not receive the same confidence as dense Broward months.
- Trip imbalance is a major confounder. In high-coverage rows, trip count ratio correlates negatively with weighted overlap around -0.70. A ratio above 2.0 should reduce confidence even if both months have many trips.

### Recommended sparse-month handling

Do not assign the same interpretation to sparse and dense comparisons.

Recommended handling:

- If both months have no trips: score should be blank or `NA`, confidence `LOW`, interpretation `NO COMPARISON`.
- If one month has zero trips and the other has trips: score may be 100, confidence `LOW`, interpretation `ZERO-BASELINE CHANGE`.
- If either month has fewer than 10 trips: show score, confidence `LOW`.
- If either month has 10-24 trips: show score, confidence `MEDIUM`.

### Recommended interpretation bands

Generic 0-25 / 25-50 / 50-75 / 75-100 bands are not appropriate for this dataset. The meaningful Broward rows have a baseline score range of about 51-83.

For Phase 3 v1, use data-calibrated interpretation bands for rows with `HIGH` or `MEDIUM` confidence:

| Label | Score range | Basis |
|---|---:|---|
| LOW RELATIVE CHANGE | `< 60` | approximately below the high-coverage Broward lower quartile |
| MODERATE RELATIVE CHANGE | `60-70` | around the high-coverage median |
| HIGH RELATIVE CHANGE | `70-80` | upper quartile region |
| VERY HIGH RELATIVE CHANGE | `>= 80` | near the maximum high-coverage Broward comparisons |

For `LOW` confidence rows, use the same numeric score but display the interpretation as:

```text
LOW CONFIDENCE - interpret with trip-count context
```

Rationale:

- The most stable high-coverage comparison still scores around 51.
- A score of 60 is relatively low for dense Driver 1003 Broward comparisons.
- A score above 80 is unusual among meaningful rows and corresponds to major month-to-month graph turnover.
- Confidence should be visually prominent to prevent sparse one-trip comparisons from being misread as stronger evidence than dense comparisons.

## Proposed Phase 3 implementation scope

The Phase 3 implementation should:

1. Load the existing Phase 2C summary, node comparison, and edge comparison tables.
2. Compute Candidate A as the default Path Choice Change Index.
3. Preserve Candidate B, C, and D as optional sensitivity-analysis columns if useful.
4. Add confidence labels using trip counts, zero-trip flags, missing-data flags, and trip-count ratio.
5. Use calibrated interpretation bands rather than generic bands.
6. Generate an HTML report focused on Driver 1003, with Broward County highlighted as the strongest longitudinal dataset.
7. Keep county-specific rows primary and avoid combining FIDs across counties for the main metric.

## Limitations

- This analysis uses only one longitudinal subject: Driver 1003.
- County-specific sparse rows can produce extreme scores from a single trip.
- Node and edge metrics are nearly collinear in the current data, so edge weighting cannot yet be empirically tuned from divergence between node and edge behavior.
- The metric should be described as a route-choice or route-network change index, not a clinical or diagnostic score.
- The next step after implementation should be validation against expert review, known travel context, or manually inspected high-change periods.
