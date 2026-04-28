# roadnet

A modular Python library for road-network enrichment, designed for South Florida
but usable with any OSMnx-accessible region.

## What it does

| Stage | Module | Output |
|---|---|---|
| Download OSM road network & land-use | `roadnet.osm` | `osm_nodes.parquet`, `osm_edges.parquet`, `osm_landuse.parquet` |
| Fetch Mapillary traffic signs | `roadnet.mapillary` | `mly_signs_raw.parquet` |
| Snap signs → edges | `roadnet.mapillary` | `osm_edges_with_mly.parquet` |
| Extract + merge FDOT GDB | `roadnet.fdot` | `fdot_merged.parquet` |
| Bearing-aware spatial conflation | `roadnet.conflation` | Joins FDOT + custom county GeoJSON onto OSM edges using 20m sub-segment local bearings |
| Connector speed graph-walk | `roadnet.speed` | Fills ramp/link speed gaps via directed graph walk |
| Land-use context | `roadnet.speed` | Tags each segment with modal land-use polygon |
| Multi-source speed arbitration | `roadnet.speed` | `estimated_speed_limit`, `speed_source`, `speed_limit_confidence_score` |
| Control node boolean features | `roadnet.pipeline` | `has_stop_sign_u/v`, `has_yield_u/v`, `has_traffic_signal_u/v` |
| FMM map-matching + aggregation | `roadnet.fmm_pipeline` | Per-segment sensor aggregates (`sog`, accelerometer, OBD) |

## Quick start

```python
from pathlib import Path
from roadnet import CountyConfig, PipelineConfig, Pipeline

cfg = PipelineConfig(
    output_dir = Path("outputs"),
    mly_token  = "MLY|...",
    fdot_gdb   = Path("DOTShapesFGDB.gdb"),
    counties   = [
        CountyConfig(
            name             = "Miami-Dade County",
            place_query      = "Miami-Dade County, Florida, USA",
            fdot_county_name = "Miami-Dade",
            fdot_county_code = "87",
            custom_geojson   = Path("miami.geojson"),
            custom_speed_col = "SPEEDLIMIT",
            custom_name_col  = "SNAME",
        ),
    ],
    gps_root = Path("gps_sessions"),
)

results = Pipeline(cfg).run()
network = results["Miami-Dade County"]   # GeoDataFrame
```

## YAML config (recommended)

```yaml
output_dir: "sflorida_outputs"
mly_token: "MLY|YOUR_TOKEN_HERE"
fdot_gdb: "DOTShapesFGDB.gdb"
gps_root: "kingston_miami"

skip_osm: true
skip_mly: true
skip_conflation: false
skip_fmm: false

counties:
  - name: "Miami-Dade County"
    place_query: "Miami-Dade County, Florida, USA"
    fdot_county_name: "Miami-Dade"
    fdot_county_code: "87"
    custom_geojson: "miami.geojson"
    custom_speed_col: "SPEEDLIMIT"
    custom_name_col: "SNAME"
    custom_min_vote: 0.6

  - name: "Broward County"
    place_query: "Broward County, Florida, USA"
    fdot_county_name: "Broward"
    fdot_county_code: "86"

  - name: "Palm Beach County"
    place_query: "Palm Beach County, Florida, USA"
    fdot_county_name: "Palm Beach"
    fdot_county_code: "93"
    custom_geojson: "Road_Centerlines.geojson"
    custom_speed_col: "SPEED_LIM"
    custom_name_col: "NAME"
    custom_lane_col: "LANES"
    custom_owner_col: "RESP_AUTH"
    custom_func_class_col: "FUNC_CLASS"
    custom_min_vote: 0.6
```

Run with:

```bash
roadnet-run config.yaml
roadnet-consolidate --config config.yaml
```

## Per-county custom GeoJSON

Any county can supply its own road-centreline file. Map the column names once
in `CountyConfig` and the library handles the conflation automatically:

```python
CountyConfig(
    name                  = "Palm Beach County",
    place_query           = "Palm Beach County, Florida, USA",
    fdot_county_name      = "Palm Beach",
    fdot_county_code      = "93",
    custom_geojson        = Path("Road_Centerlines.geojson"),
    custom_speed_col      = "SPEED_LIM",
    custom_name_col       = "NAME",
    custom_owner_col      = "RESP_AUTH",
    custom_min_vote       = 0.6,
)
```

## Enriched network columns

Key columns in the output `enriched_network.parquet`:

| Column | Description |
|---|---|
| `estimated_speed_limit` | Best available speed limit estimate (mph) |
| `speed_source` | Which data source the speed came from (`osm`, `custom_primary`, `fdot_primary`, `functional_class`, …) |
| `speed_limit_confidence_score` | 0–1 confidence score incorporating source authority, name match, multi-source agreement, and land-use plausibility |
| `FDOT_SPEED` | Raw FDOT speed limit where matched |
| `FDOT_DESCR` | FDOT road description |
| `FDOT_FUNCTIONAL_CLASS` | FDOT functional classification |
| `FDOT_AADT` | Annual average daily traffic |
| `FDOT_BRIDGES` | FDOT bridge structure ID (non-null if segment is a bridge) |
| `CUSTOM_SPEED` | County GeoJSON speed limit |
| `CUSTOM_NAME` | County GeoJSON road name |
| `CUSTOM_OWNER` | Road authority / owner (e.g. `FDOT`, `COUNTY`, `MUN`, `PRIT`) |
| `CUSTOM_FUNC_CLASS` | County functional class (e.g. `U-PA`, `U-MA`, `U-COLL`) |
| `has_stop_sign_u` | Stop sign present at source node (Mapillary or OSM) |
| `has_stop_sign_v` | Stop sign present at destination node |
| `has_yield_u` | Yield sign at source node |
| `has_yield_v` | Yield sign at destination node |
| `has_traffic_signal_u` | Traffic signal at source node |
| `has_traffic_signal_v` | Traffic signal at destination node |
| `MAP_has_stop_u/v` | Raw Mapillary stop sign count at u/v |
| `MAP_has_yield_u/v` | Raw Mapillary yield sign count at u/v |
| `OSM_has_stop_u/v` | OSM stop node at u/v |
| `OSM_has_signal_u/v` | OSM traffic signal node at u/v |
| `landuse` | Modal land-use type for this segment |
| `is_roundabout` | Boolean — segment is part of a roundabout |
| `is_connector` | Boolean — segment is a ramp/link connector |

## Conflation algorithm

The FDOT and county conflation uses a **bearing-aware spatial voter system**:

1. Each FDOT/county target geometry is split into **20m sub-segments**, each with its own local bearing (fixes false positives from long curved geometries whose start-to-end bearing is misleading).
2. Each OSM source edge is also split into **20m sub-segments** for voter points, each with its own local bearing.
3. A nearest-neighbour spatial join snaps each voter point to the nearest target sub-segment within `max_dist` metres.
4. The **bearing filter** rejects matches where the local angle difference exceeds `angle_tol` degrees.
5. A **vote tally** requires at least `min_vote_ratio` of voter points to agree on the same target feature.
6. A **vertical separation filter** rejects matches where an OSM bridge/elevated edge (`bridge=yes` or `layer≥1`) doesn't correspond to a FDOT bridge structure (`FDOT_BRIDGES` notna).
7. A **name-match tiebreaker** strips directional suffixes (NB/SB/EB/WB) and road-type words (St/Ave/Blvd/Rd etc.) from the OSM `name` and checks whether it appears in `FDOT_DESCR`. Among all surviving vote candidates, name-matched roads are preferred; closest by snap distance is used as fallback when no name match exists.

| Parameter | FDOT | Custom county |
|---|---|---|
| `max_dist` | 30m | 30m |
| `angle_tol` | 30° | 30° |
| `source_offset` | 10m | 5m |
| `target_offset` | 20m | 5m |
| `min_vote_ratio` | 0.4 | 0.5–0.6 |

## Mapillary sign snapping

Signs are snapped to the correct approach edge using a two-strategy bearing-aware
algorithm. Camera images that observed each sign are fetched from the Mapillary
API and their GPS positions used as a trajectory to vote for the most likely road.

### Camera trajectory voting

For each sign, the GPS positions of all contributing cameras are projected to UTM
and treated as a mini trajectory sampled every 5m. Each sample point votes for
the nearest OSM edge within `snap_radius` (5m) whose local bearing agrees with
the camera's travel direction within 40°. The tight snap radius prevents cameras
on one road from casting votes for a perpendicular road at an intersection.

Two tallies are maintained per edge:

- **Sample votes** — total number of 5m interpolated sample points that voted
- **Raw camera GPS points** — the set of distinct original camera GPS positions
  that contributed at least one sample vote (used as the primary selection
  criterion in Strategy A)

Votes are grouped by road name for bearing lookup and Strategy B. Pipe-separated
OSM names are normalised and sorted so `"A|B"` and `"B|A"` group together;
unnamed edges group by sorted u/v node pair.

### Strategy A — Most camera GPS points wins (primary)

The road segment physically driven by the most Mapillary cameras near the sign
wins, regardless of total sample vote count. This prevents a busy through-road
with many cameras far away from stealing the snap from the correct road that
fewer cameras actually drove on.

1. Find all edges with **≥ `MIN_RAW_CAM_POINTS` (2) distinct original camera GPS
   points** and a node within `NODE_SNAP_M` of the sign.
2. Find the **maximum raw camera count** among eligible edges.
3. Only consider edges that match the maximum raw count — the road with the most
   cameras physically on it wins.
4. Among tied edges, score by `sample_votes / min_camera_dist_to_sign` — more
   votes and cameras closer to the sign score higher.
5. The circular mean of compass angles from cameras on the winning edge must agree
   with `aligned_direction` within 65° (mod 180°). If not — reject.
6. **Service road check** — if any `highway=service` road is within 20m of the
   sign AND the snapped edge has `osm_maxspeed > 30 mph`, reject. Signs on fast
   roads near service roads likely control parking lot access, not the main road.
   This check is skipped if the snapped edge is itself a service road or has
   `osm_maxspeed ≤ 30 mph`.
7. **Landuse zone check** — if the sign falls inside a non-residential landuse
   polygon (commercial, retail, industrial, etc.), the snapped edge must also
   intersect that same polygon. If the edge lies entirely outside the zone the
   snap is rejected — the sign controls a road inside the zone, not the public
   street running past it.

### Strategy B — Weak consensus + bearing (no clear Strategy A winner)

Used when no edge passes the raw camera count + node distance filter.

1. All road names with **≥ 4 total sample votes** qualify.
2. Exactly **one** OSM node must be within `NODE_SNAP_M` of the sign (ambiguous
   intersections are skipped).
3. That node must touch at least one qualified edge.
4. Among qualifying edges, candidates are scored by `bearing_diff − votes × 10`.
   The per-road circular mean camera bearing is used where available; geometric
   edge bearing is the fallback. Candidates with bearing difference > 40° are
   rejected.
5. The same service road proximity check and landuse zone check as Strategy A are
   applied to the winning candidate.

### Reverse edge propagation

After snapping, sign flags are propagated to the exact reverse edge on two-way
streets. For edge A (u→v), the reverse edge B where B.u==A.v AND B.v==A.u
receives the flipped flag (e.g. `MAP_has_stop_u` on A → `MAP_has_stop_v` on B).
One-way edges are excluded. This ensures drivers approaching the same intersection
from the opposite direction are correctly flagged even if no Mapillary camera drove
that direction.

### Key parameters

| Parameter | Value |
|---|---|
| `NODE_SNAP_M` | 20m |
| `MAX_EDGE_DIST_M` | 20m |
| `MIN_RAW_CAM_POINTS` | 2 |
| Camera trajectory step | 5m |
| Camera snap radius | 5m |
| Bearing tolerance (voting) | 40° |
| Bearing gate — Strategy A | 65° mod 180° |
| Bearing gate — Strategy B | 40° mod 180° |
| Strategy B min votes | 4 |
| Service road rejection radius | 20m |
| Service road speed threshold | > 30 mph on snapped edge |
| Speed filter | edges with `osm_maxspeed ≥ 50 mph` excluded |
| Residential landuse excluded from zone check | `residential`, `house`, `houses`, `apartments`, `garages` |

## FMM map-matching

GPS traces are map-matched using the Fast Map-Matching (FMM) library:

- **opath alignment** — FMM returns `opath` (one matched FID per GPS point), which is merged to sensor data by positional index (`point_idx`), not by timestamp. This correctly handles timestamp collisions where multiple GPS points share the same floored second.
- **Sensor aggregation** — `sog`, `cog`, `nsat` are averaged directly from GPS points per FID. `acc`/`obd` sensors (high-frequency) are resampled to 1s bins and joined to each FID by its traversal time window (`ts_min` to `ts_max`).
- **`seconds_total`** — actual elapsed seconds on each segment (`ts_max - ts_min + 1`), not a GPS point count.
- **STMatch gap bridging** — time gaps in GPS traces are bridged using STMatch. Gaps are skipped if: (a) either endpoint is outside the current county's shapefile, or (b) the straight-line distance between endpoints exceeds 1km.
- **Multiprocessing** — trip aggregation runs in parallel across all CPU cores using `multiprocessing.Pool`.
- **Caching** — sensor data (acc/obd) is cached per session as parquet files. Master GPS parquet cache avoids re-reading all JSONL files on subsequent runs.

## Confidence scoring

`speed_limit_confidence_score` is built from:

- **Base score** by speed source (OSM: 0.40, FDOT: 0.38, custom county: 0.35, functional class default: 0.04)
- **Authority bonus** — Miami-Dade `MAINTCODE=SR` +0.20, PBC `RESP_AUTH=FDOT` +0.20
- **Functional class bonus** — FDOT interstate +0.12, arterial +0.08; PBC `U-PA` +0.10
- **Name match bonus** — vectorized word-intersection between OSM name, county name, and FDOT description
- **Multi-source agreement** — OSM + FDOT within 5 mph +0.12, all three sources agree +0.18 additional
- **Land-use plausibility** — speed consistent with land-use type +0.08, residential >45 mph −0.10

## Caching flags

| Flag | Skips |
|---|---|
| `skip_osm` | OSM download (also skips FDOT re-extraction) |
| `skip_mly` | Mapillary fetch |
| `skip_conflation` | Full conflation + speed stages (loads cached enriched network) |
| `skip_fmm` | Map-matching (FMM) |

## Sanity check tool

```bash
python sanity_check.py \
  --name "Glades Road" \
  --county "Palm Beach County" \
  --fdot_parquet "sflorida_outputs/fdot/fdot_merged.parquet" \
  --county_geojson "Road_Centerlines.geojson"
```

Outputs a `.log` file and an interactive `.html` map showing OSM edges (red/blue
alternating with FID labels), FDOT geometry (green dashed), and county geometry
(orange dashed).

## Installation

```bash
pip install -e .
# For map-matching support, also install fmm:
# https://fmm-wiki.github.io/docs/installation/
```

## Module overview

```
roadnet/
├── __init__.py         Public API (CountyConfig, PipelineConfig, Pipeline)
├── config.py           Dataclasses + all tunable constants
├── osm.py              OSM download, tag cleaning, control-node flags
├── mapillary.py        Mapillary fetch + camera-trajectory sign snapping
├── fdot.py             Generic FDOT GDB extraction + attribute merge
├── conflation.py       Bearing-aware spatial conflation engine (20m sub-segments, name-match tiebreaker)
├── speed.py            Connector graph-walk + speed arbitration + confidence scoring
├── fmm_pipeline.py     FMM map-matching, sensor aggregation, split-matching, STMatch gap bridging
└── pipeline.py         Top-level orchestrator
```