# Phase 2A: Driver 1003 route timeline

## Purpose

Phase 2A converts cached Phase 1 trip-level map-matching outputs into a
longitudinal route-activity timeline for **Driver 1003**. Phase 1 produced the
FMM map matching, enriched road network, matched trip outputs, and individual
trip visualizations used as inputs here.

This phase does not calculate the final Driver Path Choice Change Metric or
compare monthly graphs. It prepares the Driver 1003 observation timeline used by
the monthly attributed graph and graph-comparison stages.

The immediate products are:

- one Driver 1003 trip timeline;
- complete calendar-month summaries, including zero-observation months;
- a presentation-ready HTML timeline.

Each timeline row is one county-specific matched portion of a source GPS trip.
A source trip crossing county boundaries can produce more than one row because
each county has a separate FID namespace and enriched network.

## Inputs

The workflow reuses cached files and does not rerun FMM:

- `<gps_root>/<collection>/<32-character ID>/<YYMMDD>/*_fid_aggregated.jsonl`
  filenames provide grouping, session date, source trip token, and county. The
  aggregate file bodies are not required to construct timelines.
- `sflorida_outputs/*_County/*_gps.csv` provides county-local trip IDs,
  timestamps, durations, and GPS point counts.
- `sflorida_outputs/*_County/*_matched.csv` provides matched `opath` sequences.
- `sflorida_outputs/*_County/enriched_network.parquet` is retained as the
  FID-level road-attribute source for monthly graph construction.

Aggregate filenames are joined to county GPS trips by county and trip start
time. The default maximum difference is 300 seconds. Stale aggregate files,
ambiguous assignments, and empty FMM matches are excluded rather than guessed.

## Why the current identity is inferred

The source hierarchy has the form:

```text
<collection>/<32-character hexadecimal ID>/<session date>/<trip files>
```

The 32-character ID is stable across many dated sessions and trips, but sampled
GPS, accelerometer, OBD, aggregate, capability-log, and cache schemas do not
contain an explicit `driver_id`, `vehicle_id`, `device_id`, participant ID, or
user ID.

OBD capability files occur under the same ID, so it may identify a logger,
vehicle, account, collection subject, or human participant. The available
metadata does not distinguish these possibilities.

Unless confirmed by project metadata, the inferred ID should be treated as a
pseudonymous driver/session identifier rather than guaranteed human driver identity.

For the current project scope, Dr. Jang directed us to work with **Driver 1003**
as the longitudinal case study. Technical outputs still preserve
`internal_driver_id` with
`driver_id_source=session_dir_parent`. See:

```text
sflorida_outputs/phase2/driver_identity_audit.md
```

## Internal ID and presentation label

Long pseudonymous IDs are preserved for traceability in timeline CSVs and
technical HTML details. Older intermediate outputs may still include alias-map
files such as:

```text
sflorida_outputs/phase2/driver_alias_map.csv
```

Presentation-facing Phase 2 deliverables now label the subject as
**Driver 1003**.

## Complete calendar months

Monthly summaries contain every calendar month from first observation through
last observation. Months without trips are written with zero activity and:

```text
observed_month = false
```

`observed_month_count` counts months with real observations.
`calendar_month_span` counts the full inclusive date span. Keeping these values
separate avoids misleading visual jumps over missing months.

## Build the Driver 1003 timeline

From the repository root:

```bash
python scripts/build_driver_timeline.py --driver auto
```

This selects the Driver 1003 working subject used by the rest of Phase 2 in the
current cached dataset.

Select a specific internal ID or unique prefix:

```bash
python scripts/build_driver_timeline.py --driver ca351c04
```

The expensive Phase 1 work is not repeated. County GPS/matched tables are
loaded from existing cached outputs.

## Outputs

Local audit and traceability outputs:

```text
sflorida_outputs/phase2/driver_identity_audit.md
sflorida_outputs/phase2/driver_alias_map.csv
```

Driver 1003 timeline outputs:

```text
sflorida_outputs/phase2/driver_timelines/driver_1_timeline.csv
sflorida_outputs/phase2/driver_timelines/driver_1_monthly_summary.csv
sflorida_outputs/phase2/visuals/driver_1_activity_over_time.html
```

Individual HTML pages include summary cards, complete monthly axes, spacious
SVG charts with axis titles, an interpretation note, and collapsed technical
details containing the internal ID and cache sources. The raw hash is not used
in the page title.

The committed advisor-review HTML copy is:

```text
deliverables/driver_1003/timeline/driver_1003_timeline.html
```

## Connection to monthly attributed graphs

For Driver 1003 and each month, the outputs provide:

- ordered road-segment traversal sequences;
- route signatures;
- start/end FIDs;
- monthly FID unions;
- explicit missing-month rows;
- county and enriched-network references;
- stable internal IDs plus deterministic presentation aliases.

The next phase groups by `internal_driver_id`, `trip_month`, and `county`,
loads matching FIDs from `enriched_network.parquet`, and constructs directed
monthly attributed graphs without revisiting map matching.

## Current limitations and mentor questions

- The internal ID is not proven to represent a human driver.
- A human may potentially use multiple IDs, or multiple humans may share one
  vehicle/logger ID.
- Collection labels such as `1003 1004` are not documented in this repository.
- County-specific FID namespaces are not reconciled across county boundaries.
- The timeline visual summarizes monthly activity; route-network structure is
  represented in Phase 2B monthly attributed graph maps.
- The final path-choice change metric remains intentionally out of scope.

Questions for project mentors:

1. What process generated the 32-character directory IDs?
2. Are they tied to participant, vehicle, device/logger, account, or deployment?
3. Is there a privacy-safe crosswalk for participant, vehicle, and logger IDs?
4. Can one person have multiple IDs or one ID represent multiple drivers?
