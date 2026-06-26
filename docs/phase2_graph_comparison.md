# Phase 2C: Driver 1003 month-to-month graph comparison

## Purpose

This phase compares consecutive monthly attributed graphs for Driver 1003.
It quantifies changes in monthly route-network structure without defining the
project's final driver-choice change metric.

The comparison does not rerun FMM, enrichment, or monthly graph construction.
It consumes the cached Phase 2B node and edge tables.

## Inputs and graph identity

Primary inputs:

```text
deliverables/google_drive_phase2/driver_1003_monthly_graphs/data/
  driver_1003_all_monthly_nodes.parquet
  driver_1003_all_monthly_edges.parquet

sflorida_outputs/phase2/monthly_graphs/driver_1003/
  monthly_graph_manifest.csv
```

The manifest supplies every calendar month and county, including months with
no observed trips.

FID namespaces are county-specific:

- node key: `(county, fid)`
- directed edge key: `(county, source_fid, target_fid)`

County-specific comparisons are the primary output. The summary also includes
`ALL_COUNTIES` rows whose keys remain county-qualified.

Interactive comparison pages are generated separately for each county/month
pair. A month pair such as `2021-09 -> 2021-10` can therefore have separate
pages for Broward County, Palm Beach County, and Miami-Dade County. This avoids
mixing county FID namespaces and prevents a zero-baseline county from obscuring
a more informative county in the same month pair.

## Month pairs

Every consecutive calendar pair from September 2021 through June 2024 is
considered. Zero-trip months are retained and flagged rather than skipped.

Examples:

```text
2021-09 -> 2021-10
2021-10 -> 2021-11
...
2024-05 -> 2024-06
```

## Node comparison

Each FID in the union of two monthly node sets is classified as:

- `shared`: present in both months
- `added`: present only in month B
- `removed`: present only in month A

Set metrics include node Jaccard similarity, retention rate, new rate, and
removed rate.

Weighted metrics use `trip_use_count`:

```text
weighted overlap =
  sum(min(weight_a, weight_b)) / sum(max(weight_a, weight_b))

normalized L1 change =
  sum(abs(weight_b - weight_a)) / (total_weight_a + total_weight_b)
```

Detailed rows preserve monthly counts, shares, deltas, road name/type, speed
limit, lanes, AADT, road length, one-way status, owner/source, and monthly
observed average/median speeds where available.

## Edge comparison

Each county-qualified directed FID transition is similarly classified as
shared, added, or removed.

Set metrics include edge Jaccard similarity, retention rate, new rate, and
removed rate. Weighted overlap and normalized L1 change use
`transition_count`.

Detailed rows preserve transition counts, distinct-trip transition counts,
transition shares, and their month-to-month deltas.

## Empty sets and data quality

- If both graph sets are empty, similarity and normalized weighted metrics are
  null because no route activity was observed.
- If exactly one set is empty, Jaccard and weighted overlap are zero.
- Rates with zero denominators remain null.

Primary quality flags use this precedence:

1. `missing_files`
2. `both_months_no_trips`
3. `month_a_no_trips`
4. `month_b_no_trips`
5. `nodes_but_no_edges`
6. `low_trip_count_month`
7. `ok`

An observed month with fewer than 10 trips is low-trip.

County comparison pages explicitly call out zero-baseline cases:

- If month A has zero county nodes and month B has nodes, every displayed FID
  is newly observed in month B.
- If month A has nodes and month B has zero county nodes, every displayed FID
  was removed after month A.
- If both months have zero county nodes, no comparison page is generated for
  that county/month pair.

## Run

Compare all consecutive months and counties:

```bash
python scripts/compare_driver_1003_monthly_graphs.py --driver 1003 --all
```

Run one consecutive county comparison:

```bash
python scripts/compare_driver_1003_monthly_graphs.py \
  --driver 1003 \
  --county "Broward County" \
  --month-a 2023-08 \
  --month-b 2023-09
```

## Outputs

```text
deliverables/google_drive_phase2/driver_1003_graph_comparisons/
  data/
    driver_1003_month_to_month_node_comparisons.csv
    driver_1003_month_to_month_node_comparisons.parquet
    driver_1003_month_to_month_edge_comparisons.csv
    driver_1003_month_to_month_edge_comparisons.parquet
    driver_1003_month_to_month_summary.csv
    driver_1003_month_to_month_summary.parquet
  visuals/
    driver_1003_graph_comparison_overview.html
    driver_1003_broward_2023-08_to_2023-09_comparison.html
    county_comparisons/
      2021-09_to_2021-10/
        driver_1003_broward_county_comparison.html
        driver_1003_palm_beach_county_comparison.html
      2023-08_to_2023-09/
        driver_1003_broward_county_comparison.html
        ...
  driver_1003_graph_comparison_validation.md
```

The overview groups links by month pair, then lists each county-specific
comparison underneath. Within each month pair, counties with meaningful shared
overlap are listed first, higher-activity counties next, and zero-baseline
counties last.

County comparison pages use neutral lines for shared FIDs, green for added
FIDs, and red for removed FIDs. Transition changes remain in the detailed
tables and summary outputs to avoid an unreadable overlay.

## Interpretation and limitations

These outputs describe graph change, route-network change, FID usage change,
transition change, and monthly route-activity structure. Differences may also
reflect unequal trip counts or missing observations.

This phase does not establish cognitive decline, dementia, impairment, or
causation. Edge Jaccard is used only to order descriptive overview cards; it is
not the final driver-choice change metric.

The next research step is to use these validated component metrics and quality
flags to design and evaluate a defensible driver path-choice change metric.
