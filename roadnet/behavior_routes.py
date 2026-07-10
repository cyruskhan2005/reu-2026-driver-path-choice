"""OD-specific route summaries and route-change stories.

This module is deliberately independent of geocoding and POI enrichment.  It
turns county-local FMM FIDs into county-scoped keys, joins those keys to the
enriched monthly-node export, and compares route use only after an
origin/destination clustering step has assigned stable cluster IDs.

Assumptions
-----------
* FIDs are unique only within a county.  A bare FID therefore requires an
  explicit ``county`` argument; silently joining a bare FID across counties is
  not allowed.
* Adjacent duplicate FIDs are GPS-sampling repeats and are collapsed.  The
  reported ``route_distance_m`` counts each county/FID once per trip, which is
  a conservative deduplicated matched-route distance.  A diagnostic
  ``traversal_distance_m`` also counts nonconsecutive revisits.
* OSM ``motorway``/``trunk`` classes and their links are treated as the
  controlled-access/highway proxy.  Other known classes are surface streets;
  primary/secondary/tertiary classes are the arterial proxy.
* OD-specific weighted overlaps compare normalized within-month route-use
  distributions.  Thus, trip-volume changes alone do not create route change.
  RCCI is the balanced mean of FID and directed-transition change, on 0--100.
* Trip duration is an elapsed-recording duration.  Speeds and travel-time
  changes are descriptive and do not establish congestion or a causal reason.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Hashable, Iterable, Mapping, Sequence, TypeAlias

import pandas as pd


CountyFID: TypeAlias = tuple[str, int]
Transition: TypeAlias = tuple[CountyFID, CountyFID]

HIGHWAY_CLASSES = frozenset(
    {"motorway", "motorway_link", "trunk", "trunk_link"}
)
ARTERIAL_CLASSES = frozenset(
    {
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
    }
)
LOCAL_CLASSES = frozenset(
    {"residential", "service", "living_street", "unclassified"}
)

OD_ROUTE_CHANGE_COLUMNS = [
    "origin_cluster_id",
    "origin_label",
    "destination_cluster_id",
    "destination_label",
    "month_a",
    "month_b",
    "trip_count_a",
    "trip_count_b",
    "dominant_route_a",
    "dominant_route_b",
    "major_roads_removed",
    "major_roads_added",
    "highway_share_a",
    "highway_share_b",
    "surface_street_share_a",
    "surface_street_share_b",
    "distance_change",
    "travel_time_change",
    "RCCI",
    "plain_english_story",
    "confidence",
    "limitations",
]


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "null", "<na>"}
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    try:
        return bool(result)
    except (TypeError, ValueError):
        # Array-like route sequences produce an array of missingness flags and
        # are not themselves a scalar missing value.
        return False


def _text(value: object) -> str | None:
    if _is_missing(value):
        return None
    result = str(value).strip()
    return result or None


def _number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _boolean(value: object) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t", "toll", "tolled"}:
        return True
    if text in {"0", "false", "no", "n", "f", "not_toll", "untolled"}:
        return False
    return None


def _clean_county(value: object) -> str | None:
    text = _text(value)
    return re.sub(r"\s+", " ", text) if text else None


def _is_scoped_pair(value: object) -> bool:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return False
    county, fid = value
    county_text = _clean_county(county)
    if not county_text:
        return False
    try:
        int(str(fid).strip())
    except (TypeError, ValueError):
        return False
    return not county_text.lstrip("+-").isdigit()


def _sequence_items(value: object) -> list[object]:
    if _is_missing(value):
        return []
    if isinstance(value, Mapping) or _is_scoped_pair(value):
        return [value]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                decoded = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                decoded = None
            if isinstance(decoded, list):
                return decoded
        if re.fullmatch(r"[+-]?\d+(?:\s+[+-]?\d+)+", text):
            return re.split(r"\s+", text)
        return [token.strip() for token in re.split(r"[|,;]+", text)]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return list(value)
    return [value]


def _parse_scoped_item(
    item: object,
    default_county: str | None,
) -> CountyFID | None:
    county: str | None = default_county
    fid_value: object = item

    if isinstance(item, Mapping):
        county = _clean_county(item.get("county")) or default_county
        fid_value = item.get("fid", item.get("FID"))
    elif _is_scoped_pair(item):
        pair = list(item)  # type: ignore[arg-type]
        county = _clean_county(pair[0])
        fid_value = pair[1]
    elif isinstance(item, str):
        token = item.strip()
        scoped = re.fullmatch(r"(.+?)\s*(?:::|:)\s*([+-]?\d+)", token)
        if scoped:
            county = _clean_county(scoped.group(1))
            fid_value = scoped.group(2)
        else:
            fid_value = token

    try:
        fid = int(str(fid_value).strip())
    except (TypeError, ValueError):
        return None
    if fid < 0:
        return None
    if not county:
        raise ValueError(
            "Bare FID sequence has no county scope; pass county=... or use "
            "(county, fid) entries"
        )
    return county, fid


def parse_county_fid_sequence(
    value: object,
    county: str | None = None,
    *,
    collapse_consecutive: bool = True,
    strict: bool = False,
) -> list[CountyFID]:
    """Parse a route into ``(county, fid)`` keys.

    Accepted values include the repository's ``"1|2|3"`` representation
    (with ``county=``), iterables of integers (with ``county=``), iterables of
    ``(county, fid)`` pairs, mappings with ``county``/``fid`` keys, and scoped
    string tokens such as ``"Broward County:123|Broward County:456"``.
    Negative FMM non-matches are omitted.  Malformed tokens are skipped unless
    ``strict=True``.
    """
    default_county = _clean_county(county)
    result: list[CountyFID] = []
    for item in _sequence_items(value):
        parsed = _parse_scoped_item(item, default_county)
        if parsed is None:
            if strict and not _is_missing(item):
                raise ValueError(f"Invalid county/FID sequence item: {item!r}")
            continue
        if collapse_consecutive and result and result[-1] == parsed:
            continue
        result.append(parsed)
    return result


def serialize_county_fid_sequence(sequence: Iterable[CountyFID]) -> str:
    """Serialize scoped FIDs without losing the county component."""
    return "|".join(f"{county}::{fid}" for county, fid in sequence)


_ROAD_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "county": ("county",),
    "fid": ("fid", "FID"),
    "road_name": ("road_name", "name", "FDOT_ROADWAY"),
    "highway": ("highway", "road_class", "road_type"),
    "length_m": ("road_length_m", "length_m", "length"),
    "speed_limit": (
        "estimated_speed_limit",
        "speed_limit",
        "speed_limit_mph",
    ),
    "geometry_wkt": ("geometry_wkt", "geometry"),
    "toll": ("toll", "is_toll", "toll_road", "uses_toll_road"),
}


def _resolve_alias(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    available = {str(column): str(column) for column in columns}
    folded = {column.casefold(): column for column in available}
    for alias in aliases:
        if alias in available:
            return available[alias]
        if alias.casefold() in folded:
            return folded[alias.casefold()]
    return None


def _representative_value(values: pd.Series) -> object:
    valid = [value for value in values if not _is_missing(value)]
    if not valid:
        return pd.NA
    # Monthly rows normally agree.  Mode makes the result deterministic if an
    # attribute was updated in one export, while first occurrence breaks ties.
    counts: Counter[str] = Counter(str(value) for value in valid)
    # Counter preserves first-seen order, so ties resolve to the earliest row.
    winning_text = counts.most_common(1)[0][0]
    return next(value for value in valid if str(value) == winning_text)


def load_unique_road_context(
    path: str | Path,
    *,
    parse_geometry: bool = True,
) -> pd.DataFrame:
    """Load one enriched road-context row per county/FID.

    The input is the combined ``driver_1003_all_monthly_nodes.csv``.  Monthly
    duplicates are collapsed by county and FID, never by FID alone.  Returned
    canonical columns are ``county``, ``fid``, ``road_name``, ``highway``,
    ``length_m``, ``speed_limit``, ``geometry_wkt``, ``geometry``, and ``toll``.
    ``geometry`` contains parsed Shapely objects when requested; malformed WKT
    remains available in ``geometry_wkt`` and yields ``None`` in ``geometry``.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Road-context CSV does not exist: {source}")

    header = pd.read_csv(source, nrows=0).columns.tolist()
    resolved = {
        canonical: _resolve_alias(header, aliases)
        for canonical, aliases in _ROAD_COLUMN_ALIASES.items()
    }
    if not resolved["county"] or not resolved["fid"]:
        raise ValueError("Road-context CSV must contain county and fid columns")
    usecols = sorted({column for column in resolved.values() if column})
    raw = pd.read_csv(source, usecols=usecols, low_memory=False)

    canonical = pd.DataFrame(index=raw.index)
    for output_name, input_name in resolved.items():
        canonical[output_name] = raw[input_name] if input_name else pd.NA
    canonical["county"] = canonical["county"].map(_clean_county)
    canonical["fid"] = pd.to_numeric(canonical["fid"], errors="coerce")
    canonical = canonical.dropna(subset=["county", "fid"]).copy()
    canonical = canonical.loc[canonical["fid"] >= 0]
    canonical["fid"] = canonical["fid"].astype("int64")

    rows: list[dict[str, object]] = []
    value_columns = [
        "road_name",
        "highway",
        "length_m",
        "speed_limit",
        "geometry_wkt",
        "toll",
    ]
    for (county_value, fid), group in canonical.groupby(
        ["county", "fid"], sort=True, dropna=False
    ):
        row: dict[str, object] = {"county": str(county_value), "fid": int(fid)}
        for column in value_columns:
            row[column] = _representative_value(group[column])
        rows.append(row)
    result = pd.DataFrame(
        rows,
        columns=["county", "fid", *value_columns],
    )
    for column in ("length_m", "speed_limit"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["road_name"] = result["road_name"].map(_text)
    result["highway"] = result["highway"].map(
        lambda value: (_text(value) or "unknown").lower()
    )
    result["toll"] = result["toll"].map(_boolean)

    if parse_geometry:
        try:
            from shapely import wkt
        except ImportError:  # pragma: no cover - Shapely is a project dependency.
            result["geometry"] = None
        else:
            def parse_wkt(value: object) -> object | None:
                text = _text(value)
                if not text:
                    return None
                try:
                    return wkt.loads(text)
                except Exception:
                    # GEOS exception types differ between supported Shapely
                    # versions; a malformed optional geometry must not prevent
                    # use of the remaining road attributes.
                    return None

            result["geometry"] = result["geometry_wkt"].map(parse_wkt)
    else:
        result["geometry"] = None
    return result.reset_index(drop=True)


def build_road_context_lookup(
    road_context: pd.DataFrame | Mapping[CountyFID, Mapping[str, object]],
) -> dict[CountyFID, dict[str, object]]:
    """Return a normalized county/FID lookup for repeated trip summaries."""
    if isinstance(road_context, Mapping):
        result: dict[CountyFID, dict[str, object]] = {}
        for raw_key, raw_value in road_context.items():
            key = parse_county_fid_sequence([raw_key], strict=True)[0]
            value = dict(raw_value)
            result[key] = {
                "county": key[0],
                "fid": key[1],
                "road_name": _text(value.get("road_name", value.get("name"))),
                "highway": (
                    _text(value.get("highway", value.get("road_class"))) or "unknown"
                ).lower(),
                "length_m": _number(
                    value.get("length_m", value.get("road_length_m", value.get("length")))
                ),
                "speed_limit": _number(
                    value.get("speed_limit", value.get("estimated_speed_limit"))
                ),
                "geometry": value.get("geometry"),
                "geometry_wkt": value.get("geometry_wkt"),
                "toll": _boolean(
                    value.get("toll", value.get("is_toll", value.get("toll_road")))
                ),
            }
        return result

    required = {"county", "fid"}
    if missing := required - set(road_context.columns):
        raise ValueError(f"Road context missing required columns: {sorted(missing)}")
    result = {}
    for row in road_context.to_dict(orient="records"):
        county = _clean_county(row.get("county"))
        fid = _number(row.get("fid"))
        if not county or math.isnan(fid) or fid < 0:
            continue
        key = (county, int(fid))
        result[key] = {
            **row,
            "county": county,
            "fid": int(fid),
            "road_name": _text(row.get("road_name", row.get("name"))),
            "highway": (
                _text(row.get("highway", row.get("road_class"))) or "unknown"
            ).lower(),
            "length_m": _number(
                row.get("length_m", row.get("road_length_m", row.get("length")))
            ),
            "speed_limit": _number(
                row.get("speed_limit", row.get("estimated_speed_limit"))
            ),
            "toll": _boolean(
                row.get("toll", row.get("is_toll", row.get("toll_road")))
            ),
        }
    return result


def load_road_context(
    path: str | Path,
    *,
    parse_geometry: bool = True,
) -> dict[CountyFID, dict[str, object]]:
    """Load the monthly-node CSV directly as a county/FID lookup."""
    return build_road_context_lookup(
        load_unique_road_context(path, parse_geometry=parse_geometry)
    )


def _ordered_unique(values: Iterable[Hashable]) -> list[Any]:
    result: list[Any] = []
    seen: set[Hashable] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _collapse_adjacent_text(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if not result or result[-1].casefold() != value.casefold():
            result.append(value)
    return result


def _route_family_tokens(
    sequence: Sequence[CountyFID],
    context: Mapping[CountyFID, Mapping[str, object]],
) -> list[str]:
    result: list[str] = []
    for key in sequence:
        info = context.get(key, {})
        name = _text(info.get("road_name", info.get("name")))
        highway = (_text(info.get("highway")) or "unknown").lower()
        # Retain a scoped FID for unnamed segments so distinct unnamed paths do
        # not collapse into one apparent route merely because their classes match.
        token = f"name:{name.casefold()}" if name else f"fid:{key[0]}:{key[1]}:{highway}"
        if not result or result[-1] != token:
            result.append(token)
    return result


def summarize_trip_route(
    fid_sequence: object,
    road_context: pd.DataFrame | Mapping[CountyFID, Mapping[str, object]],
    *,
    county: str | None = None,
    duration_seconds: object = None,
    toll_used: object = None,
) -> dict[str, object]:
    """Summarize one matched trip route using enriched road attributes.

    Distance and class shares use unique county/FIDs.  ``traversal_distance_m``
    separately sums the adjacent-deduplicated traversal and can therefore be
    larger if a trip genuinely revisits a segment.  The derived average speed
    uses ``route_distance_m / duration_seconds`` and is not an observed-speed
    average.
    """
    context = (
        build_road_context_lookup(road_context)
        if isinstance(road_context, pd.DataFrame)
        else road_context
    )
    sequence = parse_county_fid_sequence(
        fid_sequence, county=county, collapse_consecutive=True
    )
    unique_sequence: list[CountyFID] = _ordered_unique(sequence)

    road_names: list[str] = []
    for key in sequence:
        info = context.get(key, {})
        name = _text(info.get("road_name", info.get("name")))
        if name:
            road_names.append(name)
    road_names = _collapse_adjacent_text(road_names)

    class_distances: Counter[str] = Counter()
    known_length = 0.0
    missing_context: list[CountyFID] = []
    missing_length: list[CountyFID] = []
    speed_limit_numerator = 0.0
    speed_limit_denominator = 0.0
    inferred_toll_values: list[bool] = []
    for key in unique_sequence:
        info = context.get(key)
        if info is None:
            missing_context.append(key)
            continue
        length = _number(
            info.get("length_m", info.get("road_length_m", info.get("length")))
        )
        toll_value = _boolean(
            info.get("toll", info.get("is_toll", info.get("toll_road")))
        )
        if toll_value is not None:
            inferred_toll_values.append(toll_value)
        if math.isnan(length) or length < 0:
            missing_length.append(key)
            continue
        highway = (_text(info.get("highway")) or "unknown").lower()
        class_distances[highway] += length
        known_length += length
        speed_limit = _number(
            info.get("speed_limit", info.get("estimated_speed_limit"))
        )
        if not math.isnan(speed_limit) and speed_limit >= 0:
            speed_limit_numerator += speed_limit * length
            speed_limit_denominator += length
    traversal_distance = 0.0
    for key in sequence:
        info = context.get(key, {})
        length = _number(
            info.get("length_m", info.get("road_length_m", info.get("length")))
        )
        if not math.isnan(length) and length >= 0:
            traversal_distance += length

    highway_distance = sum(class_distances[name] for name in HIGHWAY_CLASSES)
    arterial_distance = sum(class_distances[name] for name in ARTERIAL_CLASSES)
    local_distance = sum(class_distances[name] for name in LOCAL_CLASSES)
    surface_distance = known_length - highway_distance
    classified_distance = highway_distance + arterial_distance + local_distance
    other_surface_distance = max(known_length - classified_distance, 0.0)

    def share(distance: float) -> float:
        return distance / known_length if known_length > 0 else float("nan")

    duration = _number(duration_seconds)
    speed_mps = (
        known_length / duration
        if known_length > 0 and not math.isnan(duration) and duration > 0
        else float("nan")
    )
    supplied_toll = _boolean(toll_used)
    toll = (
        supplied_toll
        if supplied_toll is not None
        else (any(inferred_toll_values) if inferred_toll_values else None)
    )

    scoped_serialized = serialize_county_fid_sequence(sequence)
    route_signature = hashlib.sha256(scoped_serialized.encode("utf-8")).hexdigest()[:16]
    family_tokens = _route_family_tokens(sequence, context)
    family_payload = "|".join(family_tokens)
    route_family_signature = hashlib.sha256(
        family_payload.encode("utf-8")
    ).hexdigest()[:16]
    road_label = " → ".join(road_names) if road_names else "Unnamed matched route"
    limitations: list[str] = []
    if missing_context:
        limitations.append(f"{len(missing_context)} matched FID(s) lacked road context")
    if missing_length:
        limitations.append(f"{len(missing_length)} matched FID(s) lacked usable length")
    if not sequence:
        limitations.append("no usable matched FIDs")

    return {
        "county_fid_sequence": sequence,
        "fid_sequence": [fid for _, fid in sequence],
        "unique_county_fids": unique_sequence,
        "directed_transitions": list(zip(sequence, sequence[1:])),
        "route_signature": route_signature,
        "route_family_signature": route_family_signature,
        "road_name_sequence": road_names,
        "ordered_road_name_sequence": road_names,
        "matched_road_name_sequence": road_label,
        "road_name_sequence_text": road_label,
        "route_distance": known_length,
        "route_distance_m": known_length,
        "unique_segment_distance_m": known_length,
        "route_distance_km": known_length / 1_000.0,
        "route_distance_miles": known_length / 1_609.344,
        "traversal_distance_m": traversal_distance,
        "highway_distance_m": highway_distance,
        "highway_share": share(highway_distance),
        "arterial_distance_m": arterial_distance,
        "arterial_share": share(arterial_distance),
        "local_road_distance_m": local_distance,
        "local_road_share": share(local_distance),
        "surface_street_distance_m": surface_distance,
        "surface_street_share": share(surface_distance),
        "other_surface_distance_m": other_surface_distance,
        "other_surface_share": share(other_surface_distance),
        "distance_weighted_speed_limit_mph": (
            speed_limit_numerator / speed_limit_denominator
            if speed_limit_denominator > 0
            else float("nan")
        ),
        "duration_seconds": duration,
        "average_matched_route_speed": speed_mps * 2.2369362920544,
        "average_matched_route_speed_mph": speed_mps * 2.2369362920544,
        "average_speed_mph": speed_mps * 2.2369362920544,
        "average_matched_route_speed_kph": speed_mps * 3.6,
        "toll_road_usage": toll,
        "toll_indicator": toll,
        "matched_fid_count": len(sequence),
        "unique_fid_count": len(unique_sequence),
        "missing_context_fid_count": len(missing_context),
        "missing_length_fid_count": len(missing_length),
        "route_summary_limitations": "; ".join(limitations),
    }


_TRIP_COLUMN_CANDIDATES = {
    "origin": ("origin_cluster_id", "origin_area_id", "origin_cluster"),
    "destination": (
        "destination_cluster_id",
        "destination_area_id",
        "destination_cluster",
    ),
    "origin_label": ("origin_label", "origin_area_label"),
    "destination_label": ("destination_label", "destination_area_label"),
    "month": ("month", "trip_month"),
    "sequence": ("matched_fid_sequence", "fid_sequence", "fid_list"),
    "county": ("county", "origin_county"),
    "duration": ("trip_duration", "duration_seconds", "travel_time_seconds"),
    "toll": ("toll_road_usage", "toll_used", "uses_toll_road"),
}


def _resolve_frame_column(
    frame: pd.DataFrame,
    requested: str | None,
    kind: str,
    *,
    required: bool,
) -> str | None:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Requested {kind} column is absent: {requested}")
        return requested
    candidate = _resolve_alias(frame.columns, _TRIP_COLUMN_CANDIDATES[kind])
    if required and not candidate:
        raise ValueError(
            f"Trip table needs a {kind} column; tried "
            f"{list(_TRIP_COLUMN_CANDIDATES[kind])}"
        )
    return candidate


def _valid_numbers(values: Iterable[object]) -> list[float]:
    result = [_number(value) for value in values]
    return [value for value in result if not math.isnan(value)]


def _median(values: Iterable[object]) -> float:
    valid = _valid_numbers(values)
    return float(pd.Series(valid).median()) if valid else float("nan")


def _distance_weighted_share(group: pd.DataFrame, distance_column: str) -> float:
    numerator = sum(_valid_numbers(group[distance_column]))
    denominator = sum(_valid_numbers(group["route_distance_m"]))
    return numerator / denominator if denominator > 0 else float("nan")


def _month_text(value: object) -> str | None:
    text = _text(value)
    if not text:
        return None
    try:
        return str(pd.Period(text, freq="M"))
    except (TypeError, ValueError):
        return None


def compute_dominant_routes(
    trips: pd.DataFrame,
    road_context: pd.DataFrame | Mapping[CountyFID, Mapping[str, object]],
    *,
    origin_col: str | None = None,
    destination_col: str | None = None,
    origin_label_col: str | None = None,
    destination_label_col: str | None = None,
    month_col: str | None = None,
    sequence_col: str | None = None,
    county_col: str | None = None,
    duration_col: str | None = None,
    toll_col: str | None = None,
) -> pd.DataFrame:
    """Compute a dominant named-route family for every OD pair and month.

    Route families collapse adjacent segments carrying the same road name;
    unnamed segments retain their scoped FID.  This avoids treating harmless
    segment splits on one named corridor as different route choices, without
    merging genuinely different unnamed paths.
    """
    origin = _resolve_frame_column(trips, origin_col, "origin", required=True)
    destination = _resolve_frame_column(
        trips, destination_col, "destination", required=True
    )
    origin_label = _resolve_frame_column(
        trips, origin_label_col, "origin_label", required=False
    )
    destination_label = _resolve_frame_column(
        trips, destination_label_col, "destination_label", required=False
    )
    month = _resolve_frame_column(trips, month_col, "month", required=True)
    sequence = _resolve_frame_column(trips, sequence_col, "sequence", required=True)
    county = _resolve_frame_column(trips, county_col, "county", required=False)
    duration = _resolve_frame_column(trips, duration_col, "duration", required=False)
    toll = _resolve_frame_column(trips, toll_col, "toll", required=False)

    context = build_road_context_lookup(road_context)
    summarized_rows: list[dict[str, object]] = []
    for row in trips.to_dict(orient="records"):
        month_value = _month_text(row.get(month))
        if month_value is None or _is_missing(row.get(origin)) or _is_missing(row.get(destination)):
            continue
        route = summarize_trip_route(
            row.get(sequence),
            context,
            county=row.get(county) if county else None,
            duration_seconds=row.get(duration) if duration else None,
            toll_used=row.get(toll) if toll else None,
        )
        summarized_rows.append(
            {
                "origin_cluster_id": str(row.get(origin)),
                "destination_cluster_id": str(row.get(destination)),
                "origin_label": (
                    _text(row.get(origin_label)) if origin_label else None
                ) or str(row.get(origin)),
                "destination_label": (
                    _text(row.get(destination_label)) if destination_label else None
                ) or str(row.get(destination)),
                "month": month_value,
                **route,
            }
        )
    summarized = pd.DataFrame(summarized_rows)
    if summarized.empty:
        return pd.DataFrame()

    output: list[dict[str, object]] = []
    keys = ["origin_cluster_id", "destination_cluster_id", "month"]
    for (origin_id, destination_id, month_value), group in summarized.groupby(
        keys, sort=True, dropna=False
    ):
        valid = group.loc[group["matched_fid_count"] > 0]
        family_counts = Counter(valid["route_family_signature"])
        if family_counts:
            dominant_family = sorted(
                family_counts, key=lambda value: (-family_counts[value], str(value))
            )[0]
            dominant = valid.loc[
                valid["route_family_signature"] == dominant_family
            ]
            representative = dominant.sort_values("route_signature").iloc[0]
            dominant_count = int(len(dominant))
            dominant_route = str(representative["matched_road_name_sequence"])
            dominant_names = list(representative["road_name_sequence"])
            dominant_fids = list(representative["county_fid_sequence"])
        else:
            dominant_family = None
            dominant_count = 0
            dominant_route = "Route unavailable"
            dominant_names = []
            dominant_fids = []

        fid_counts: Counter[CountyFID] = Counter()
        transition_counts: Counter[Transition] = Counter()
        for route_sequence in valid["county_fid_sequence"]:
            # Monthly node use follows the repository RCCI convention: one
            # contribution per trip/FID, regardless of an in-trip revisit.
            fid_counts.update(set(route_sequence))
            transition_counts.update(zip(route_sequence, route_sequence[1:]))

        metric_group = valid if not valid.empty else group
        toll_values = [
            parsed
            for value in group["toll_road_usage"]
            if (parsed := _boolean(value)) is not None
        ]
        output.append(
            {
                "origin_cluster_id": str(origin_id),
                "origin_label": str(group["origin_label"].iloc[0]),
                "destination_cluster_id": str(destination_id),
                "destination_label": str(group["destination_label"].iloc[0]),
                "month": str(month_value),
                "trip_count": int(len(group)),
                "route_valid_trip_count": int(len(valid)),
                "route_coverage_share": len(valid) / len(group),
                "dominant_route": dominant_route,
                "dominant_route_frequency": dominant_count,
                "dominant_route_share": (
                    dominant_count / len(valid) if len(valid) else float("nan")
                ),
                "dominant_route_family_signature": dominant_family,
                "dominant_road_name_sequence": dominant_names,
                "dominant_county_fid_sequence": dominant_fids,
                "median_route_distance_m": _median(metric_group["route_distance_m"]),
                "median_route_distance_miles": _median(
                    metric_group["route_distance_miles"]
                ),
                "median_travel_time_seconds": _median(
                    metric_group["duration_seconds"]
                ),
                "median_average_speed_mph": _median(
                    metric_group["average_matched_route_speed_mph"]
                ),
                "highway_share": _distance_weighted_share(
                    metric_group, "highway_distance_m"
                ),
                "arterial_share": _distance_weighted_share(
                    metric_group, "arterial_distance_m"
                ),
                "local_road_share": _distance_weighted_share(
                    metric_group, "local_road_distance_m"
                ),
                "surface_street_share": _distance_weighted_share(
                    metric_group, "surface_street_distance_m"
                ),
                "toll_trip_share": (
                    sum(toll_values) / len(toll_values)
                    if toll_values
                    else float("nan")
                ),
                "_fid_weights": fid_counts,
                "_transition_weights": transition_counts,
            }
        )
    return pd.DataFrame(output).sort_values(keys).reset_index(drop=True)


def compute_dominant_route_by_od_month(*args: object, **kwargs: object) -> pd.DataFrame:
    """Compatibility alias for :func:`compute_dominant_routes`."""
    return compute_dominant_routes(*args, **kwargs)  # type: ignore[arg-type]


def weighted_route_overlap(
    left: Mapping[Hashable, int | float],
    right: Mapping[Hashable, int | float],
    *,
    normalize: bool = True,
) -> float:
    """Generalized weighted-Jaccard overlap for two sparse route vectors.

    Normalization is on by default so identical route distributions with
    different trip counts have overlap 1.0.
    """
    keys = set(left) | set(right)
    if not keys:
        return float("nan")
    def nonnegative_weight(value: object) -> float:
        number = _number(value)
        return 0.0 if math.isnan(number) else max(number, 0.0)

    left_values = {key: nonnegative_weight(left.get(key, 0)) for key in keys}
    right_values = {key: nonnegative_weight(right.get(key, 0)) for key in keys}
    left_total = sum(left_values.values())
    right_total = sum(right_values.values())
    if left_total <= 0 or right_total <= 0:
        return float("nan")
    if normalize:
        left_values = {key: value / left_total for key, value in left_values.items()}
        right_values = {
            key: value / right_total for key, value in right_values.items()
        }
    denominator = sum(max(left_values[key], right_values[key]) for key in keys)
    if denominator <= 0:
        return float("nan")
    numerator = sum(min(left_values[key], right_values[key]) for key in keys)
    return float(min(max(numerator / denominator, 0.0), 1.0))


def compute_balanced_od_rcci(
    fid_overlap: object,
    transition_overlap: object,
) -> float:
    """Return balanced OD-specific RCCI on 0--100.

    Both components receive 0.5 weight when available.  For a one-segment
    route with no transitions, the FID component is used alone and callers
    should retain the resulting limitation note.
    """
    overlaps = [_number(fid_overlap), _number(transition_overlap)]
    changes = [
        1.0 - min(max(value, 0.0), 1.0)
        for value in overlaps
        if not math.isnan(value)
    ]
    if not changes:
        return float("nan")
    return float(min(max(100.0 * sum(changes) / len(changes), 0.0), 100.0))


def _ordered_name_difference(left: Sequence[str], right: Sequence[str]) -> list[str]:
    left_keys = {value.casefold() for value in left}
    return _ordered_unique(value for value in right if value.casefold() not in left_keys)


def _difference(value_b: object, value_a: object) -> float:
    a = _number(value_a)
    b = _number(value_b)
    return b - a if not math.isnan(a) and not math.isnan(b) else float("nan")


def _comparison_confidence(a: Mapping[str, object], b: Mapping[str, object]) -> str:
    minimum_trips = min(int(a["trip_count"]), int(b["trip_count"]))
    coverage = min(_number(a["route_coverage_share"]), _number(b["route_coverage_share"]))
    dominance = min(_number(a["dominant_route_share"]), _number(b["dominant_route_share"]))
    if minimum_trips >= 5 and coverage >= 0.8 and dominance >= 0.4:
        return "high"
    if minimum_trips >= 3 and coverage >= 0.6 and dominance >= 0.25:
        return "medium"
    return "low"


def _format_roads(values: Sequence[str], maximum: int = 4) -> str:
    if not values:
        return ""
    shown = list(values[:maximum])
    suffix = f" and {len(values) - maximum} more" if len(values) > maximum else ""
    return ", ".join(shown) + suffix


def _plain_english_story(
    a: Mapping[str, object],
    b: Mapping[str, object],
    *,
    removed: Sequence[str],
    added: Sequence[str],
    rcci: float,
) -> str:
    origin = str(a["origin_label"])
    destination = str(a["destination_label"])
    month_a = str(a["month"])
    month_b = str(b["month"])
    route_a = str(a["dominant_route"])
    route_b = str(b["dominant_route"])
    parts = [
        f"Trips continued between the same clustered origin and destination ({origin} to {destination}) in {month_a} and {month_b}."
    ]
    if route_a != route_b:
        parts.append(
            f"The dominant matched route changed from {route_a} to {route_b}."
        )
    elif not math.isnan(rcci) and rcci <= 10:
        parts.append(
            f"The dominant named-road sequence remained {route_a}, and the overall matched-route mix was similar."
        )
    else:
        parts.append(
            f"The dominant named-road sequence remained {route_a}, although the overall matched-route mix changed."
        )
    if removed:
        parts.append(f"Roads no longer in the dominant sequence included {_format_roads(removed)}.")
    if added:
        parts.append(f"New roads in the dominant sequence included {_format_roads(added)}.")

    highway_delta = _difference(b["highway_share"], a["highway_share"])
    if not math.isnan(highway_delta) and abs(highway_delta) >= 0.10:
        direction = "more" if highway_delta > 0 else "less"
        parts.append(
            f"The monthly route mix used {direction} highway mileage ({_number(a['highway_share']):.0%} to {_number(b['highway_share']):.0%})."
        )
    distance_delta = _difference(
        b["median_route_distance_m"], a["median_route_distance_m"]
    )
    if not math.isnan(distance_delta) and abs(distance_delta) >= 100:
        direction = "increased" if distance_delta > 0 else "decreased"
        parts.append(
            f"Median deduplicated matched-route distance {direction} by {abs(distance_delta) / 1609.344:.1f} miles."
        )
    time_delta = _difference(
        b["median_travel_time_seconds"], a["median_travel_time_seconds"]
    )
    if not math.isnan(time_delta) and abs(time_delta) >= 60:
        direction = "increased" if time_delta > 0 else "decreased"
        parts.append(
            f"Median recorded travel time {direction} by {abs(time_delta) / 60:.1f} minutes."
        )
    toll_a = _number(a["toll_trip_share"])
    toll_b = _number(b["toll_trip_share"])
    if (
        not math.isnan(toll_a)
        and not math.isnan(toll_b)
        and abs(toll_b - toll_a) >= 0.25
    ):
        parts.append(
            f"The supplied toll-road indicator changed from {toll_a:.0%} to {toll_b:.0%} of trips."
        )
    if not math.isnan(rcci):
        parts.append(
            f"The OD-specific balanced RCCI was {rcci:.1f} out of 100; it measures route-pattern change, not its cause."
        )
    return " ".join(parts)


def compare_consecutive_od_months(
    trips_or_profiles: pd.DataFrame,
    road_context: pd.DataFrame | Mapping[CountyFID, Mapping[str, object]] | None = None,
    *,
    min_trips_per_month: int = 1,
    **dominant_route_kwargs: object,
) -> pd.DataFrame:
    """Compare consecutive months for OD pairs whose cluster IDs are stable.

    Pass raw trip rows plus ``road_context``, or pass the result of
    :func:`compute_dominant_routes`.  Only identical origin/destination cluster
    IDs in consecutive calendar months are compared.  This separates
    ``same OD, different route`` from a destination-cluster change.
    """
    if min_trips_per_month < 1:
        raise ValueError("min_trips_per_month must be at least 1")
    if trips_or_profiles.empty:
        return pd.DataFrame(columns=OD_ROUTE_CHANGE_COLUMNS)
    internal = {"_fid_weights", "_transition_weights", "dominant_route", "month"}
    if internal.issubset(trips_or_profiles.columns):
        profiles = trips_or_profiles.copy()
    else:
        if road_context is None:
            raise ValueError("road_context is required when passing raw trip rows")
        profiles = compute_dominant_routes(
            trips_or_profiles, road_context, **dominant_route_kwargs
        )
    if profiles.empty:
        return pd.DataFrame(columns=OD_ROUTE_CHANGE_COLUMNS)
    required_profiles = {
        "origin_cluster_id",
        "origin_label",
        "destination_cluster_id",
        "destination_label",
        "month",
        "trip_count",
        "route_coverage_share",
        "dominant_route",
        "dominant_route_frequency",
        "dominant_route_share",
        "dominant_road_name_sequence",
        "median_route_distance_m",
        "median_travel_time_seconds",
        "highway_share",
        "arterial_share",
        "local_road_share",
        "surface_street_share",
        "toll_trip_share",
        "_fid_weights",
        "_transition_weights",
    }
    if missing := required_profiles - set(profiles.columns):
        raise ValueError(
            f"OD/month profiles are missing required columns: {sorted(missing)}"
        )

    rows: list[dict[str, object]] = []
    pair_keys = ["origin_cluster_id", "destination_cluster_id"]
    for _pair, od in profiles.groupby(pair_keys, sort=True, dropna=False):
        if od.duplicated("month").any():
            raise ValueError(
                "OD/month profile table contains duplicate rows for an OD month"
            )
        ordered = od.assign(
            _period=od["month"].map(lambda value: pd.Period(str(value), freq="M"))
        ).sort_values("_period")
        records = ordered.to_dict(orient="records")
        for a, b in zip(records, records[1:]):
            if b["_period"] != a["_period"] + 1:
                continue
            if min(int(a["trip_count"]), int(b["trip_count"])) < min_trips_per_month:
                continue
            fid_overlap = weighted_route_overlap(
                a["_fid_weights"], b["_fid_weights"]
            )
            transition_overlap = weighted_route_overlap(
                a["_transition_weights"], b["_transition_weights"]
            )
            rcci = compute_balanced_od_rcci(fid_overlap, transition_overlap)
            if math.isnan(rcci):
                # A stable OD assignment without comparable matched-route
                # evidence is not a route-change insight.
                continue
            names_a = list(a["dominant_road_name_sequence"])
            names_b = list(b["dominant_road_name_sequence"])
            removed = _ordered_name_difference(names_b, names_a)
            added = _ordered_name_difference(names_a, names_b)
            distance_change = _difference(
                b["median_route_distance_m"], a["median_route_distance_m"]
            )
            travel_time_change = _difference(
                b["median_travel_time_seconds"], a["median_travel_time_seconds"]
            )
            limitations = (
                "Origin/destination equality means the same spatial clusters, not a verified "
                "activity or exact place. Road classes are OSM-based proxies; FMM and endpoint "
                "error can alter FIDs. Distance is deduplicated matched-segment length, and "
                "recorded duration may include stops. The data cannot establish whether any "
                "change was caused by traffic, construction, toll avoidance, or preference."
            )
            if math.isnan(transition_overlap):
                limitations += " No directed transitions were available, so RCCI uses FIDs only."
            row = {
                "origin_cluster_id": a["origin_cluster_id"],
                "origin_label": a["origin_label"],
                "destination_cluster_id": a["destination_cluster_id"],
                "destination_label": a["destination_label"],
                "month_a": a["month"],
                "month_b": b["month"],
                "trip_count_a": int(a["trip_count"]),
                "trip_count_b": int(b["trip_count"]),
                "dominant_route_a": a["dominant_route"],
                "dominant_route_b": b["dominant_route"],
                "dominant_route_frequency_a": int(a["dominant_route_frequency"]),
                "dominant_route_frequency_b": int(b["dominant_route_frequency"]),
                "route_frequency_a": int(a["dominant_route_frequency"]),
                "route_frequency_b": int(b["dominant_route_frequency"]),
                "dominant_route_share_a": a["dominant_route_share"],
                "dominant_route_share_b": b["dominant_route_share"],
                "major_roads_removed": "; ".join(removed),
                "major_roads_added": "; ".join(added),
                "major_roads_removed_list": removed,
                "major_roads_added_list": added,
                "weighted_fid_overlap": fid_overlap,
                "weighted_transition_overlap": transition_overlap,
                "highway_share_a": a["highway_share"],
                "highway_share_b": b["highway_share"],
                "arterial_share_a": a["arterial_share"],
                "arterial_share_b": b["arterial_share"],
                "local_road_share_a": a["local_road_share"],
                "local_road_share_b": b["local_road_share"],
                "surface_street_share_a": a["surface_street_share"],
                "surface_street_share_b": b["surface_street_share"],
                "toll_trip_share_a": a["toll_trip_share"],
                "toll_trip_share_b": b["toll_trip_share"],
                # Canonical unsuffixed changes use metres and seconds; explicit
                # convenience columns make CSV interpretation unambiguous.
                "distance_change": distance_change,
                "distance_change_m": distance_change,
                "distance_change_units": "metres",
                "distance_change_miles": (
                    distance_change / 1_609.344
                    if not math.isnan(distance_change)
                    else float("nan")
                ),
                "travel_time_change": travel_time_change,
                "travel_time_change_seconds": travel_time_change,
                "travel_time_change_units": "seconds",
                "travel_time_change_minutes": (
                    travel_time_change / 60
                    if not math.isnan(travel_time_change)
                    else float("nan")
                ),
                "RCCI": rcci,
                "rcci": rcci,
                "change_type": (
                    "same_od_stable_route"
                    if rcci <= 10
                    else "same_od_route_change"
                ),
                "plain_english_story": _plain_english_story(
                    a, b, removed=removed, added=added, rcci=rcci
                ),
                "confidence": _comparison_confidence(a, b),
                "limitations": limitations,
            }
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=OD_ROUTE_CHANGE_COLUMNS)
    return pd.DataFrame(rows).sort_values(
        ["month_a", "origin_cluster_id", "destination_cluster_id"]
    ).reset_index(drop=True)


def build_od_route_change_insights(*args: object, **kwargs: object) -> pd.DataFrame:
    """Compatibility alias for :func:`compare_consecutive_od_months`."""
    return compare_consecutive_od_months(*args, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "ARTERIAL_CLASSES",
    "CountyFID",
    "HIGHWAY_CLASSES",
    "LOCAL_CLASSES",
    "OD_ROUTE_CHANGE_COLUMNS",
    "Transition",
    "build_od_route_change_insights",
    "build_road_context_lookup",
    "compare_consecutive_od_months",
    "compute_balanced_od_rcci",
    "compute_dominant_route_by_od_month",
    "compute_dominant_routes",
    "load_road_context",
    "load_unique_road_context",
    "parse_county_fid_sequence",
    "serialize_county_fid_sequence",
    "summarize_trip_route",
    "weighted_route_overlap",
]
