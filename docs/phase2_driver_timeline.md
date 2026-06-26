# Phase 2A: Driver/Session route timelines

## Purpose

Phase 2A converts cached Phase 1 trip-level map-matching outputs into reusable
longitudinal tables for one or many inferred Driver/Session groupings. It does
not calculate the final path-choice change metric or compare monthly graphs.

The immediate products are:

- one trip timeline per selected grouping;
- complete calendar-month summaries, including zero-observation months;
- deterministic presentation aliases such as `Driver 1`;
- a population ranking/index suitable for selecting the top 50–100 groupings;
- readable individual HTML timelines and a population overview.

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

Presentation text therefore uses **Driver/Session**. Technical outputs use
`internal_driver_id` with `driver_id_source=session_dir_parent`. See:

```text
sflorida_outputs/phase2/driver_identity_audit.md
```

## Internal IDs and presentation aliases

Long pseudonymous IDs are preserved for traceability in timeline CSVs,
technical HTML details, and:

```text
sflorida_outputs/phase2/driver_alias_map.csv
```

Presentation-facing titles, tables, links, and filenames use aliases:

```text
Driver 1
Driver 2
Driver 3
```

Aliases are deterministic. The population is ranked by:

1. usable trip count;
2. observed month count;
3. inclusive calendar coverage span;
4. internal ID as a stable final tiebreaker.

The same input population therefore produces the same aliases.

## Complete calendar months

Monthly summaries contain every calendar month from first observation through
last observation. Months without trips are written with zero activity and:

```text
observed_month = false
```

`observed_month_count` counts months with real observations.
`calendar_month_span` counts the full inclusive date span. Keeping these values
separate avoids misleading visual jumps over missing months.

## Build one Driver/Session timeline

From the repository root:

```bash
python scripts/build_driver_timeline.py --driver auto
```

This selects the highest-ranked grouping and normally presents it as
`Driver 1`.

Select a specific internal ID or unique prefix:

```bash
python scripts/build_driver_timeline.py --driver ca351c04
```

## Build the top N groupings

```bash
python scripts/build_driver_timeline.py --top-n 5
python scripts/build_driver_timeline.py --top-n 50
python scripts/build_driver_timeline.py --top-n 100
```

The mounted dataset currently determines how many groupings are available. If
N exceeds the population size, all available groupings are built and the
overview records the requested and available counts.

The expensive Phase 1 work is not repeated. County GPS/matched tables are
loaded once, the population is ranked once, and only selected per-grouping CSV
and HTML outputs are then written. This structure is intended to scale to
50–100 groupings when a larger source population is mounted.

If `gps_root` is not configured and `/Volumes/KINGSTON` is unavailable:

```bash
python scripts/build_driver_timeline.py \
  --top-n 50 \
  --gps-root /path/to/session/root
```

## Outputs

Population outputs:

```text
sflorida_outputs/phase2/driver_identity_audit.md
sflorida_outputs/phase2/driver_alias_map.csv
sflorida_outputs/phase2/driver_population_index.csv
sflorida_outputs/phase2/visuals/driver_population_overview.html
```

Per-alias outputs:

```text
sflorida_outputs/phase2/driver_timelines/driver_1_timeline.csv
sflorida_outputs/phase2/driver_timelines/driver_1_monthly_summary.csv
sflorida_outputs/phase2/visuals/driver_1_activity_over_time.html
```

Individual HTML pages include summary cards, complete monthly axes, spacious
SVG charts with axis titles, an interpretation note, and collapsed technical
details containing the internal ID and cache sources. The raw hash is not used
in the page title.

## Connection to monthly attributed graphs

For every selected Driver/Session and month, the outputs provide:

- ordered road-segment traversal sequences;
- route signatures;
- start/end FIDs;
- monthly FID unions;
- explicit missing-month rows;
- county and enriched-network references;
- stable internal IDs plus deterministic presentation aliases.

The next phase can group by `internal_driver_id`, `trip_month`, and `county`,
load matching FIDs from `enriched_network.parquet`, and construct directed
monthly attributed graphs without revisiting map matching. The population index
provides the selection layer for running that process over 50–100 groupings.

## Current limitations and mentor questions

- The internal ID is not proven to represent a human driver.
- A human may potentially use multiple IDs, or multiple humans may share one
  vehicle/logger ID.
- Collection labels such as `1003 1004` are not documented in this repository.
- County-specific FID namespaces are not reconciled across county boundaries.
- The visual summarizes activity but does not overlay routes or compare graphs.
- The final path-choice change metric remains intentionally out of scope.

Questions for project mentors:

1. What process generated the 32-character directory IDs?
2. Are they tied to participant, vehicle, device/logger, account, or deployment?
3. Is there a privacy-safe crosswalk for participant, vehicle, and logger IDs?
4. Can one person have multiple IDs or one ID represent multiple drivers?
