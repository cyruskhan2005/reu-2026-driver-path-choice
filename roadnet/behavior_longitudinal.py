"""Longitudinal route-family and road-class analysis.

The existing Driver 1003 RCCI compares monthly road graphs.  This module adds
the complementary question needed for behavioral interpretation: for a stable
directed origin/destination pair, which *route family* did each direct trip use,
how did the family shares change, and were apparent changes sustained or only
temporary?

The implementation deliberately keeps this layer independent of POI lookup and
report rendering.  It accepts a trip table and county/FID road context, performs
no I/O, and returns seven analysis DataFrames from
:func:`analyze_longitudinal_routes`.

Key safeguards
--------------
* Direct-route analysis requires at least 500 m endpoint separation, circuity
  no greater than 3, and an OD-specific median/MAD distance screen.
* First/last local-access segments within 250 m are removed before route-family
  comparison.  Arterial/highway road names form the primary backbone.
* Routes with the same canonical backbone seed one family.  Near seeds merge
  only when ordered-road LCS coverage is at least 0.80 and the combined
  LCS/FID/transition similarity is at least 0.70 by default.
* Every trip has equal weight in route-family shares.  Long routes cannot
  dominate merely because they contain more FIDs.
* Monthly claims require at least five eligible OD trips.  Longitudinal claims
  use pooled windows with at least ten trips per window and persistence across
  at least three observed months.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import calendar
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Hashable, Iterable, Mapping, Sequence, TypeAlias

import numpy as np
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
LOCAL_ACCESS_CLASSES = frozenset(
    {
        "residential",
        "service",
        "living_street",
        "unclassified",
        "road",
        "unknown",
    }
)


class LongitudinalAnalysisError(RuntimeError):
    """Raised when inputs cannot support a reconciled longitudinal analysis."""


@dataclass(frozen=True)
class _FamilyCluster:
    members: tuple[int, ...]
    medoid_index: int
    supported: bool


_TRIP_ALIASES: dict[str, tuple[str, ...]] = {
    "trip_id": ("trip_id",),
    "origin": ("origin_cluster_id", "origin_area_id", "origin_cluster"),
    "destination": (
        "destination_cluster_id",
        "destination_area_id",
        "destination_cluster",
    ),
    "origin_label": ("origin_label", "origin_area_label"),
    "destination_label": ("destination_label", "destination_area_label"),
    "month": ("month", "trip_month"),
    "timestamp": ("start_timestamp", "trip_start_time", "start_time"),
    "sequence": ("matched_fid_sequence", "fid_sequence", "fid_list"),
    "county": ("origin_county", "county"),
    "start_lat": ("start_latitude", "origin_latitude", "start_lat"),
    "start_lon": ("start_longitude", "origin_longitude", "start_lon"),
    "end_lat": ("end_latitude", "destination_latitude", "end_lat"),
    "end_lon": ("end_longitude", "destination_longitude", "end_lon"),
    "distance": ("route_distance_m", "distance_m"),
    "duration": ("trip_duration_seconds", "duration_seconds", "travel_time_seconds"),
    "highway_distance": ("highway_distance_m", "controlled_distance_m"),
    "arterial_distance": ("arterial_distance_m",),
    "local_distance": ("local_road_distance_m", "local_distance_m"),
    "surface_distance": ("surface_street_distance_m", "surface_distance_m"),
    "toll_distance": ("toll_distance_m",),
}

_ROAD_ALIASES: dict[str, tuple[str, ...]] = {
    "county": ("county",),
    "fid": ("fid", "FID"),
    "name": ("road_name", "name"),
    "fdot_name": ("FDOT_ROADWAY", "fdot_roadway"),
    "highway": ("highway", "road_class", "road_type"),
    "length": ("length_m", "road_length_m", "length"),
    "geometry": ("geometry",),
    "geometry_wkt": ("geometry_wkt",),
    "toll": ("toll", "is_toll"),
}


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "null", "<na>"}
    try:
        missing = pd.isna(value)
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _text(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _resolve_column(
    columns: Iterable[str],
    aliases: Sequence[str],
    *,
    required: bool,
    label: str,
) -> str | None:
    available = {str(column).casefold(): str(column) for column in columns}
    for alias in aliases:
        if alias.casefold() in available:
            return available[alias.casefold()]
    if required:
        raise LongitudinalAnalysisError(
            f"Missing {label} column; tried {list(aliases)}"
        )
    return None


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _safe_slug(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return text or "UNKNOWN"


def _month(value: object) -> str | None:
    text = _text(value)
    if not text:
        return None
    try:
        return str(pd.Period(text, freq="M"))
    except (TypeError, ValueError):
        return None


def _timestamp(value: object) -> pd.Timestamp:
    if _is_missing(value):
        return pd.NaT
    return pd.to_datetime(value, errors="coerce", utc=True)


def _haversine_m(lat1: object, lon1: object, lat2: object, lon2: object) -> float:
    values = [_number(value) for value in (lat1, lon1, lat2, lon2)]
    if any(math.isnan(value) for value in values):
        return float("nan")
    lat1_f, lon1_f, lat2_f, lon2_f = values
    radius = 6_371_008.8
    phi1, phi2 = math.radians(lat1_f), math.radians(lat2_f)
    dphi = math.radians(lat2_f - lat1_f)
    dlambda = math.radians(lon2_f - lon1_f)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return radius * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))


_ROAD_WORDS = {
    "rd": "Road",
    "road": "Road",
    "st": "Street",
    "street": "Street",
    "ave": "Avenue",
    "avenue": "Avenue",
    "av": "Avenue",
    "blvd": "Boulevard",
    "boulevard": "Boulevard",
    "dr": "Drive",
    "drive": "Drive",
    "ln": "Lane",
    "lane": "Lane",
    "hwy": "Highway",
    "highway": "Highway",
    "pkwy": "Parkway",
    "parkway": "Parkway",
    "cir": "Circle",
    "circle": "Circle",
    "ter": "Terrace",
    "terrace": "Terrace",
    "ct": "Court",
    "court": "Court",
}
_DIRECTION_PREFIXES = {
    "n",
    "north",
    "s",
    "south",
    "e",
    "east",
    "w",
    "west",
    "ne",
    "northeast",
    "nw",
    "northwest",
    "se",
    "southeast",
    "sw",
    "southwest",
}


def _canonical_name_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9' -]+", " ", value).strip()
    words = cleaned.split()
    if words and words[0].casefold().rstrip(".") in _DIRECTION_PREFIXES:
        words = words[1:]
    normalized: list[str] = []
    for word in words:
        key = word.casefold().rstrip(".")
        normalized.append(_ROAD_WORDS.get(key, word.title()))
    return " ".join(normalized).strip()


def canonicalize_road_name(value: object) -> str | None:
    """Return a stable corridor name across case, abbreviations, and direction.

    Pipe-delimited OSM alternatives are canonicalized separately, deduplicated,
    and sorted so their storage order cannot change a route-family seed.
    """
    text = _text(value)
    if not text:
        return None
    parts = [_canonical_name_part(part) for part in re.split(r"[|;/]+", text)]
    unique = sorted({part for part in parts if part}, key=str.casefold)
    return " / ".join(unique) if unique else None


def _parse_county_fids(value: object, county: object = None) -> list[CountyFID]:
    default_county = _text(county)
    if _is_missing(value):
        return []
    decoded: object = value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                decoded = json.loads(stripped)
            except (TypeError, json.JSONDecodeError):
                decoded = value
        else:
            decoded = [token for token in re.split(r"[|,;]+", stripped) if token.strip()]
    if isinstance(decoded, Mapping):
        items: list[object] = [decoded]
    elif isinstance(decoded, Iterable) and not isinstance(decoded, (str, bytes, bytearray)):
        items = list(decoded)
    else:
        items = [decoded]

    result: list[CountyFID] = []
    for item in items:
        item_county = default_county
        fid_value: object = item
        if isinstance(item, Mapping):
            item_county = _text(item.get("county")) or default_county
            fid_value = item.get("fid", item.get("FID"))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            item_county = _text(item[0]) or default_county
            fid_value = item[1]
        elif isinstance(item, str):
            scoped = re.fullmatch(r"(.+?)(?:::|:)\s*(-?\d+)", item.strip())
            if scoped:
                item_county = scoped.group(1).strip()
                fid_value = scoped.group(2)
        try:
            fid = int(str(fid_value).strip())
        except (TypeError, ValueError):
            continue
        if fid < 0:
            continue
        if not item_county:
            raise LongitudinalAnalysisError(
                "Bare FIDs require a county column or county-scoped sequence"
            )
        key = (item_county, fid)
        if not result or result[-1] != key:
            result.append(key)
    return result


def _normalize_road_context(
    road_context: pd.DataFrame | Mapping[CountyFID, Mapping[str, object]],
) -> dict[CountyFID, dict[str, object]]:
    if isinstance(road_context, Mapping):
        rows: list[dict[str, object]] = []
        for key, value in road_context.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise LongitudinalAnalysisError(
                    "Road-context mapping keys must be (county, fid) tuples"
                )
            rows.append({"county": key[0], "fid": key[1], **dict(value)})
        frame = pd.DataFrame(rows)
    else:
        frame = road_context.copy()
    county_col = _resolve_column(
        frame.columns, _ROAD_ALIASES["county"], required=True, label="road county"
    )
    fid_col = _resolve_column(
        frame.columns, _ROAD_ALIASES["fid"], required=True, label="road FID"
    )
    resolved = {
        name: _resolve_column(
            frame.columns,
            aliases,
            required=False,
            label=f"road {name}",
        )
        for name, aliases in _ROAD_ALIASES.items()
        if name not in {"county", "fid"}
    }
    output: dict[CountyFID, dict[str, object]] = {}
    for row in frame.to_dict(orient="records"):
        county = _text(row.get(county_col))
        fid_number = _number(row.get(fid_col))
        if not county or math.isnan(fid_number) or fid_number < 0:
            continue
        key = (county, int(fid_number))
        if key in output:
            continue
        name = row.get(resolved["name"]) if resolved["name"] else None
        fdot_name = row.get(resolved["fdot_name"]) if resolved["fdot_name"] else None
        canonical_name = canonicalize_road_name(name) or canonicalize_road_name(fdot_name)
        highway = (
            _text(row.get(resolved["highway"])) if resolved["highway"] else None
        ) or "unknown"
        length = _number(row.get(resolved["length"])) if resolved["length"] else float("nan")
        output[key] = {
            "road_name": _text(name) or _text(fdot_name),
            "canonical_name": canonical_name,
            "highway": highway.split("|")[0].casefold(),
            "length_m": length,
            "geometry": row.get(resolved["geometry"]) if resolved["geometry"] else None,
            "geometry_wkt": (
                row.get(resolved["geometry_wkt"]) if resolved["geometry_wkt"] else None
            ),
            "toll": bool(row.get(resolved["toll"])) if resolved["toll"] else False,
        }
    return output


def _collapse_adjacent(values: Iterable[Hashable]) -> list[Any]:
    output: list[Any] = []
    for value in values:
        if not output or output[-1] != value:
            output.append(value)
    return output


def _trim_local_access(
    sequence: Sequence[CountyFID],
    roads: Mapping[CountyFID, Mapping[str, object]],
    trim_m: float,
) -> list[CountyFID]:
    if not sequence or trim_m <= 0:
        return list(sequence)
    remove: set[int] = set()
    cumulative = 0.0
    for index, key in enumerate(sequence):
        road = roads.get(key, {})
        if str(road.get("highway", "unknown")) not in LOCAL_ACCESS_CLASSES:
            break
        if cumulative >= trim_m:
            break
        remove.add(index)
        cumulative += max(_number(road.get("length_m")), 0.0)
    cumulative = 0.0
    for index in range(len(sequence) - 1, -1, -1):
        key = sequence[index]
        road = roads.get(key, {})
        if str(road.get("highway", "unknown")) not in LOCAL_ACCESS_CLASSES:
            break
        if cumulative >= trim_m:
            break
        remove.add(index)
        cumulative += max(_number(road.get("length_m")), 0.0)
    trimmed = [key for index, key in enumerate(sequence) if index not in remove]
    return trimmed or list(sequence)


def _backbone(
    sequence: Sequence[CountyFID],
    roads: Mapping[CountyFID, Mapping[str, object]],
) -> tuple[tuple[str, ...], str]:
    major: list[str] = []
    named: list[str] = []
    for key in sequence:
        road = roads.get(key, {})
        name = _text(road.get("canonical_name"))
        highway = str(road.get("highway", "unknown"))
        if name:
            named.append(name)
            if highway in HIGHWAY_CLASSES or highway in ARTERIAL_CLASSES:
                major.append(name)
    if major:
        return tuple(_collapse_adjacent(major)), "arterial_highway"
    if named:
        return tuple(_collapse_adjacent(named)), "named_local"
    return tuple(), "unnamed"


def ordered_lcs_similarity(left: Sequence[Hashable], right: Sequence[Hashable]) -> float:
    """Return ordered LCS coverage relative to the longer sequence."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1] / max(len(left), len(right))


def weighted_overlap(
    left: Mapping[Hashable, int | float],
    right: Mapping[Hashable, int | float],
) -> float:
    """Return normalized generalized weighted-Jaccard overlap."""
    keys = set(left) | set(right)
    if not keys:
        return float("nan")

    def weights(values: Mapping[Hashable, int | float]) -> dict[Hashable, float]:
        result: dict[Hashable, float] = {}
        for key in keys:
            value = _number(values.get(key, 0))
            result[key] = 0.0 if math.isnan(value) else max(value, 0.0)
        total = sum(result.values())
        return {key: value / total for key, value in result.items()} if total > 0 else result

    left_weights = weights(left)
    right_weights = weights(right)
    if sum(left_weights.values()) <= 0 or sum(right_weights.values()) <= 0:
        return float("nan")
    denominator = sum(max(left_weights[key], right_weights[key]) for key in keys)
    numerator = sum(min(left_weights[key], right_weights[key]) for key in keys)
    return numerator / denominator if denominator > 0 else float("nan")


def route_similarity(
    left_backbone: Sequence[str],
    right_backbone: Sequence[str],
    left_fids: Mapping[CountyFID, int | float],
    right_fids: Mapping[CountyFID, int | float],
    left_transitions: Mapping[Transition, int | float],
    right_transitions: Mapping[Transition, int | float],
) -> dict[str, float]:
    """Combine ordered-road and balanced FID/transition evidence."""
    lcs = ordered_lcs_similarity(left_backbone, right_backbone)
    fid_overlap = weighted_overlap(left_fids, right_fids)
    transition_overlap = weighted_overlap(left_transitions, right_transitions)
    route_components = [
        value for value in (fid_overlap, transition_overlap) if not math.isnan(value)
    ]
    balanced = sum(route_components) / len(route_components) if route_components else 0.0
    combined = 0.5 * lcs + 0.5 * balanced
    return {
        "ordered_lcs_similarity": lcs,
        "weighted_fid_overlap": fid_overlap,
        "weighted_transition_overlap": transition_overlap,
        "balanced_fid_transition_overlap": balanced,
        "combined_similarity": combined,
    }


def _route_similarity_rows(left: Mapping[str, object], right: Mapping[str, object]) -> dict[str, float]:
    return route_similarity(
        left["_backbone"],
        right["_backbone"],
        left["_fid_weights"],
        right["_fid_weights"],
        left["_transition_weights"],
        right["_transition_weights"],
    )


def _route_features(
    trips: pd.DataFrame,
    roads: Mapping[CountyFID, Mapping[str, object]],
    *,
    trim_access_m: float,
) -> pd.DataFrame:
    columns = {
        key: _resolve_column(
            trips.columns,
            aliases,
            required=key
            in {
                "trip_id",
                "origin",
                "destination",
                "month",
                "sequence",
                "start_lat",
                "start_lon",
                "end_lat",
                "end_lon",
            },
            label=f"trip {key}",
        )
        for key, aliases in _TRIP_ALIASES.items()
    }
    records: list[dict[str, object]] = []
    for raw in trips.to_dict(orient="records"):
        trip_id = _text(raw.get(columns["trip_id"]))
        month = _month(raw.get(columns["month"]))
        if not trip_id or not month:
            continue
        county = raw.get(columns["county"]) if columns["county"] else None
        sequence = _parse_county_fids(raw.get(columns["sequence"]), county)
        segment_lengths: list[float] = []
        class_distances: Counter[str] = Counter()
        road_names: list[str] = []
        for key in sequence:
            road = roads.get(key, {})
            length = _number(road.get("length_m"))
            length = length if not math.isnan(length) and length >= 0 else 0.0
            segment_lengths.append(length)
            highway = str(road.get("highway", "unknown"))
            class_distances[highway] += length
            name = _text(road.get("canonical_name"))
            if name:
                road_names.append(name)
        calculated_distance = sum(segment_lengths)
        supplied_distance = (
            _number(raw.get(columns["distance"])) if columns["distance"] else float("nan")
        )
        route_distance = supplied_distance if supplied_distance > 0 else calculated_distance
        highway_distance_calculated = sum(
            class_distances[road_class] for road_class in HIGHWAY_CLASSES
        )
        arterial_distance_calculated = sum(
            class_distances[road_class] for road_class in ARTERIAL_CLASSES
        )
        local_distance_calculated = sum(
            class_distances[road_class] for road_class in LOCAL_ACCESS_CLASSES
        )
        toll_distance_calculated = sum(
            length
            for key, length in zip(sequence, segment_lengths, strict=True)
            if bool(roads.get(key, {}).get("toll", False))
        )

        def supplied_or_calculated(column_key: str, calculated: float) -> float:
            supplied = (
                _number(raw.get(columns[column_key])) if columns[column_key] else float("nan")
            )
            return supplied if not math.isnan(supplied) and supplied >= 0 else calculated

        highway_distance = supplied_or_calculated(
            "highway_distance", highway_distance_calculated
        )
        arterial_distance = supplied_or_calculated(
            "arterial_distance", arterial_distance_calculated
        )
        local_distance = supplied_or_calculated("local_distance", local_distance_calculated)
        surface_distance = supplied_or_calculated(
            "surface_distance", max(route_distance - highway_distance, 0.0)
        )
        toll_distance = supplied_or_calculated(
            "toll_distance", toll_distance_calculated
        )
        trimmed_sequence = _trim_local_access(sequence, roads, trim_access_m)
        backbone, backbone_kind = _backbone(trimmed_sequence, roads)
        if not backbone:
            # Keep unnamed routes distinct until similarity evidence can join them.
            digest = hashlib.sha256(
                _json([[county_name, fid] for county_name, fid in trimmed_sequence]).encode()
            ).hexdigest()[:12]
            backbone = (f"UNNAMED-{digest}",)
        origin = _text(raw.get(columns["origin"])) or ""
        destination = _text(raw.get(columns["destination"])) or ""
        timestamp = _timestamp(raw.get(columns["timestamp"])) if columns["timestamp"] else pd.NaT
        event_date = (
            timestamp.tz_convert("America/New_York").date().isoformat()
            if pd.notna(timestamp)
            else ""
        )
        separation = _haversine_m(
            raw.get(columns["start_lat"]),
            raw.get(columns["start_lon"]),
            raw.get(columns["end_lat"]),
            raw.get(columns["end_lon"]),
        )
        circuity = (
            route_distance / separation
            if route_distance > 0 and not math.isnan(separation) and separation > 0
            else float("nan")
        )
        records.append(
            {
                "trip_id": trip_id,
                "origin_cluster_id": origin,
                "origin_label": (
                    _text(raw.get(columns["origin_label"])) if columns["origin_label"] else None
                )
                or origin,
                "destination_cluster_id": destination,
                "destination_label": (
                    _text(raw.get(columns["destination_label"]))
                    if columns["destination_label"]
                    else None
                )
                or destination,
                "month": month,
                "start_timestamp": timestamp,
                "event_date": event_date,
                "start_latitude": _number(raw.get(columns["start_lat"])),
                "start_longitude": _number(raw.get(columns["start_lon"])),
                "end_latitude": _number(raw.get(columns["end_lat"])),
                "end_longitude": _number(raw.get(columns["end_lon"])),
                "route_distance_m": route_distance,
                "duration_seconds": (
                    _number(raw.get(columns["duration"])) if columns["duration"] else float("nan")
                ),
                "highway_distance_m": highway_distance,
                "arterial_distance_m": arterial_distance,
                "local_road_distance_m": local_distance,
                "surface_street_distance_m": surface_distance,
                "toll_distance_m": toll_distance,
                "od_separation_m": separation,
                "circuity": circuity,
                "county_fid_sequence": sequence,
                "analysis_fid_sequence": trimmed_sequence,
                "road_name_sequence": _collapse_adjacent(road_names),
                "_backbone": backbone,
                "backbone_kind": backbone_kind,
                "_fid_weights": Counter(set(trimmed_sequence)),
                "_transition_weights": Counter(
                    zip(trimmed_sequence, trimmed_sequence[1:])
                ),
            }
        )
    features = pd.DataFrame(records)
    if features.empty:
        raise LongitudinalAnalysisError("No usable trip records were found")
    if features["trip_id"].duplicated().any():
        raise LongitudinalAnalysisError("Trip IDs must be unique")
    return features


def _robust_distance_threshold(values: pd.Series, z: float) -> tuple[float, bool]:
    data = pd.to_numeric(values, errors="coerce").dropna()
    data = data.loc[data > 0]
    if len(data) < 5:
        return float("inf"), False
    median = float(data.median())
    mad = float((data - median).abs().median())
    if mad > 0:
        allowance = z * 1.4826 * mad
    else:
        q1, q3 = float(data.quantile(0.25)), float(data.quantile(0.75))
        iqr = q3 - q1
        allowance = 1.5 * iqr if iqr > 0 else 0.0
    allowance = max(allowance, 250.0, 0.25 * median)
    return median + allowance, True


def _mark_direct_eligibility(
    features: pd.DataFrame,
    *,
    min_od_separation_m: float,
    max_circuity: float,
    robust_outlier_z: float,
) -> pd.DataFrame:
    output = features.copy()
    valid_od = (
        output["origin_cluster_id"].ne("")
        & output["destination_cluster_id"].ne("")
        & output["origin_cluster_id"].ne("UNCLUSTERED")
        & output["destination_cluster_id"].ne("UNCLUSTERED")
        & output["origin_cluster_id"].ne(output["destination_cluster_id"])
    )
    route_available = (
        output["route_distance_m"].gt(0)
        & output["analysis_fid_sequence"].map(bool)
        & output["circuity"].notna()
    )
    base = (
        valid_od
        & route_available
        & output["od_separation_m"].ge(min_od_separation_m)
        & output["circuity"].le(max_circuity)
    )
    output["robust_distance_threshold_m"] = float("nan")
    output["robust_distance_screen_sufficient"] = False
    distance_pass = pd.Series(False, index=output.index)
    for _, group in output.loc[base].groupby(
        ["origin_cluster_id", "destination_cluster_id"], sort=True
    ):
        threshold, sufficient = _robust_distance_threshold(
            group["route_distance_m"], robust_outlier_z
        )
        output.loc[group.index, "robust_distance_threshold_m"] = threshold
        output.loc[group.index, "robust_distance_screen_sufficient"] = sufficient
        distance_pass.loc[group.index] = group["route_distance_m"].le(threshold)
    output["direct_route_eligible"] = base & distance_pass

    reasons: list[str] = []
    for index, row in output.iterrows():
        if not valid_od.loc[index]:
            if row["origin_cluster_id"] == row["destination_cluster_id"]:
                reasons.append("same_cluster")
            else:
                reasons.append("unclustered_or_missing_od")
        elif not route_available.loc[index]:
            reasons.append("missing_route_evidence")
        elif row["od_separation_m"] < min_od_separation_m:
            reasons.append("od_separation_under_minimum")
        elif row["circuity"] > max_circuity:
            reasons.append("circuity_above_maximum")
        elif not distance_pass.loc[index]:
            reasons.append("robust_od_distance_outlier")
        else:
            reasons.append("eligible")
    output["direct_route_exclusion_reason"] = reasons
    return output


def _medoid_index(group: pd.DataFrame, members: Sequence[int]) -> int:
    if len(members) == 1:
        return int(members[0])
    best: tuple[float, str, int] | None = None
    records = group.to_dict(orient="index")
    for index in members:
        similarities = [
            _route_similarity_rows(records[index], records[other])["combined_similarity"]
            for other in members
        ]
        candidate = (
            -float(np.mean(similarities)),
            str(records[index]["trip_id"]),
            int(index),
        )
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[2]


def _cluster_routes(
    group: pd.DataFrame,
    *,
    similarity_threshold: float,
    lcs_merge_threshold: float,
    min_family_trips: int,
    min_family_days: int,
) -> list[_FamilyCluster]:
    records = group.to_dict(orient="index")
    seed_members: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, row in records.items():
        seed_members[tuple(row["_backbone"])].append(int(index))
    seeds = [tuple(sorted(members)) for _, members in sorted(seed_members.items())]
    parent = list(range(len(seeds)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    seed_medoids = [_medoid_index(group, members) for members in seeds]
    candidates: list[tuple[float, str, str, int, int]] = []
    for left in range(len(seeds)):
        for right in range(left + 1, len(seeds)):
            left_row = records[seed_medoids[left]]
            right_row = records[seed_medoids[right]]
            similarity = _route_similarity_rows(left_row, right_row)
            if (
                similarity["ordered_lcs_similarity"] >= lcs_merge_threshold
                and similarity["combined_similarity"] >= similarity_threshold
            ):
                candidates.append(
                    (
                        -similarity["combined_similarity"],
                        str(left_row["trip_id"]),
                        str(right_row["trip_id"]),
                        left,
                        right,
                    )
                )
    for _, _, _, left, right in sorted(candidates):
        union(left, right)
    merged: dict[int, list[int]] = defaultdict(list)
    for seed_index, members in enumerate(seeds):
        merged[find(seed_index)].extend(members)
    clusters: list[_FamilyCluster] = []
    for members in merged.values():
        members_tuple = tuple(sorted(members))
        medoid = _medoid_index(group, members_tuple)
        dates = {str(records[index]["event_date"]) for index in members_tuple if records[index]["event_date"]}
        supported = len(members_tuple) >= min_family_trips and len(dates) >= min_family_days
        clusters.append(_FamilyCluster(members_tuple, medoid, supported))
    return sorted(
        clusters,
        key=lambda cluster: (
            -len(cluster.members),
            str(records[cluster.medoid_index]["trip_id"]),
        ),
    )


def _coassignment_agreement(
    members: Sequence[int],
    left: Sequence[_FamilyCluster],
    right: Sequence[_FamilyCluster],
) -> float:
    if len(members) < 2:
        return 1.0
    left_assignment = {
        member: cluster_index
        for cluster_index, cluster in enumerate(left)
        for member in cluster.members
    }
    right_assignment = {
        member: cluster_index
        for cluster_index, cluster in enumerate(right)
        for member in cluster.members
    }
    agreements = 0
    comparisons = 0
    for position, left_member in enumerate(members):
        for right_member in members[position + 1 :]:
            agreements += int(
                (left_assignment[left_member] == left_assignment[right_member])
                == (right_assignment[left_member] == right_assignment[right_member])
            )
            comparisons += 1
    return agreements / comparisons if comparisons else 1.0


def _family_analysis(
    features: pd.DataFrame,
    *,
    similarity_threshold: float,
    sensitivity_thresholds: Sequence[float],
    lcs_merge_threshold: float,
    min_family_trips: int,
    min_family_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = features.loc[features["direct_route_eligible"]].copy()
    assignment_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    for (origin, destination), group in eligible.groupby(
        ["origin_cluster_id", "destination_cluster_id"], sort=True
    ):
        default_clusters = _cluster_routes(
            group,
            similarity_threshold=similarity_threshold,
            lcs_merge_threshold=lcs_merge_threshold,
            min_family_trips=min_family_trips,
            min_family_days=min_family_days,
        )
        sensitivity: dict[float, list[_FamilyCluster]] = {}
        for threshold in sensitivity_thresholds:
            sensitivity[float(threshold)] = _cluster_routes(
                group,
                similarity_threshold=float(threshold),
                lcs_merge_threshold=lcs_merge_threshold,
                min_family_trips=min_family_trips,
                min_family_days=min_family_days,
            )
        od_prefix = f"{_safe_slug(origin)}__{_safe_slug(destination)}"
        supported = [cluster for cluster in default_clusters if cluster.supported]
        unsupported = [cluster for cluster in default_clusters if not cluster.supported]
        records = group.to_dict(orient="index")
        member_to_family: dict[int, str] = {}
        for rank, cluster in enumerate(supported, start=1):
            family_id = f"{od_prefix}__F{rank:02d}"
            for member in cluster.members:
                member_to_family[member] = family_id
            medoid = records[cluster.medoid_index]
            family_rows.append(
                {
                    "route_family_id": family_id,
                    "origin_cluster_id": origin,
                    "origin_label": medoid["origin_label"],
                    "destination_cluster_id": destination,
                    "destination_label": medoid["destination_label"],
                    "family_rank": rank,
                    "is_other": False,
                    "family_name": " → ".join(medoid["_backbone"]),
                    "backbone_roads_json": _json(list(medoid["_backbone"])),
                    "backbone_kind": medoid["backbone_kind"],
                    "trip_count": len(cluster.members),
                    "unique_days": len(
                        {records[index]["event_date"] for index in cluster.members}
                    ),
                    "months_seen": len(
                        {records[index]["month"] for index in cluster.members}
                    ),
                    "first_month": min(records[index]["month"] for index in cluster.members),
                    "last_month": max(records[index]["month"] for index in cluster.members),
                    "overall_route_share": len(cluster.members) / len(group),
                    "medoid_trip_id": medoid["trip_id"],
                    "median_route_distance_m": float(
                        pd.Series(
                            [records[index]["route_distance_m"] for index in cluster.members]
                        ).median()
                    ),
                    "median_duration_seconds": float(
                        pd.Series(
                            [records[index]["duration_seconds"] for index in cluster.members]
                        ).median()
                    ),
                    "similarity_threshold": similarity_threshold,
                    "lcs_merge_threshold": lcs_merge_threshold,
                }
            )
        if unsupported:
            other_id = f"{od_prefix}__OTHER"
            other_members = tuple(
                member for cluster in unsupported for member in cluster.members
            )
            for member in other_members:
                member_to_family[member] = other_id
            representative = min(
                (records[index] for index in other_members), key=lambda row: str(row["trip_id"])
            )
            family_rows.append(
                {
                    "route_family_id": other_id,
                    "origin_cluster_id": origin,
                    "origin_label": representative["origin_label"],
                    "destination_cluster_id": destination,
                    "destination_label": representative["destination_label"],
                    "family_rank": 999,
                    "is_other": True,
                    "family_name": "Other infrequent routes",
                    "backbone_roads_json": "[]",
                    "backbone_kind": "mixed",
                    "trip_count": len(other_members),
                    "unique_days": len(
                        {records[index]["event_date"] for index in other_members}
                    ),
                    "months_seen": len({records[index]["month"] for index in other_members}),
                    "first_month": min(records[index]["month"] for index in other_members),
                    "last_month": max(records[index]["month"] for index in other_members),
                    "overall_route_share": len(other_members) / len(group),
                    "medoid_trip_id": representative["trip_id"],
                    "median_route_distance_m": float(
                        pd.Series(
                            [records[index]["route_distance_m"] for index in other_members]
                        ).median()
                    ),
                    "median_duration_seconds": float(
                        pd.Series(
                            [records[index]["duration_seconds"] for index in other_members]
                        ).median()
                    ),
                    "similarity_threshold": similarity_threshold,
                    "lcs_merge_threshold": lcs_merge_threshold,
                }
            )
        for index in group.index:
            assignment_rows.append(
                {
                    "trip_id": records[index]["trip_id"],
                    "origin_cluster_id": origin,
                    "destination_cluster_id": destination,
                    "route_family_id": member_to_family[index],
                    "month": records[index]["month"],
                    "event_date": records[index]["event_date"],
                    "_feature_index": index,
                }
            )
        sensitivity_record: dict[str, object] = {
            "origin_cluster_id": origin,
            "destination_cluster_id": destination,
            "default_family_count": len(supported),
        }
        for threshold, clusters in sensitivity.items():
            suffix = f"{threshold:.2f}".replace(".", "_")
            sensitivity_record[f"family_count_similarity_{suffix}"] = sum(
                cluster.supported for cluster in clusters
            )
            sensitivity_record[f"assignment_agreement_similarity_{suffix}"] = (
                _coassignment_agreement(list(group.index), default_clusters, clusters)
            )
        sensitivity_rows.append(sensitivity_record)
    assignments = pd.DataFrame(assignment_rows)
    families = pd.DataFrame(family_rows)
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    return assignments, families, sensitivity_frame


def _observed_months(features: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    values = sorted({_month(value) for value in features["month"] if _month(value)})
    months = [value for value in values if value is not None]
    return months, {month: index for index, month in enumerate(months)}


def _od_summary(
    features: pd.DataFrame,
    assignments: pd.DataFrame,
    families: pd.DataFrame,
    sensitivity: pd.DataFrame,
    observed_months: Sequence[str],
) -> pd.DataFrame:
    valid = features.loc[
        features["origin_cluster_id"].ne("")
        & features["destination_cluster_id"].ne("")
        & features["origin_cluster_id"].ne("UNCLUSTERED")
        & features["destination_cluster_id"].ne("UNCLUSTERED")
        & features["origin_cluster_id"].ne(features["destination_cluster_id"])
    ]
    half = len(observed_months) // 2
    early = set(observed_months[:half])
    late = set(observed_months[half:])
    rows: list[dict[str, object]] = []
    for (origin, destination), group in valid.groupby(
        ["origin_cluster_id", "destination_cluster_id"], sort=True
    ):
        direct = group.loc[group["direct_route_eligible"]]
        od_assignments = assignments.loc[
            assignments["origin_cluster_id"].eq(origin)
            & assignments["destination_cluster_id"].eq(destination)
        ]
        family_counts = od_assignments["route_family_id"].value_counts()
        dominant_id = str(family_counts.index[0]) if len(family_counts) else ""
        dominant_share = (
            float(family_counts.iloc[0] / len(od_assignments))
            if len(od_assignments)
            else float("nan")
        )
        early_assignments = od_assignments.loc[od_assignments["month"].isin(early)]
        late_assignments = od_assignments.loc[od_assignments["month"].isin(late)]
        keys = set(early_assignments["route_family_id"]) | set(
            late_assignments["route_family_id"]
        )
        total_variation = float("nan")
        if len(early_assignments) and len(late_assignments):
            total_variation = 0.5 * sum(
                abs(
                    (early_assignments["route_family_id"].eq(key).sum() / len(early_assignments))
                    - (late_assignments["route_family_id"].eq(key).sum() / len(late_assignments))
                )
                for key in keys
            )
        reasons = group["direct_route_exclusion_reason"].value_counts()
        threshold_values = direct["robust_distance_threshold_m"].replace(
            [np.inf, -np.inf], np.nan
        )
        rows.append(
            {
                "origin_cluster_id": origin,
                "origin_label": str(group["origin_label"].iloc[0]),
                "destination_cluster_id": destination,
                "destination_label": str(group["destination_label"].iloc[0]),
                "total_trip_count": len(group),
                "eligible_direct_trip_count": len(direct),
                "excluded_trip_count": len(group) - len(direct),
                "direct_trip_share": len(direct) / len(group),
                "eligible_unique_days": direct["event_date"].replace("", np.nan).nunique(),
                "eligible_months": direct["month"].nunique(),
                "median_od_separation_m": float(group["od_separation_m"].median()),
                "median_direct_route_distance_m": float(direct["route_distance_m"].median())
                if len(direct)
                else float("nan"),
                "median_direct_circuity": float(direct["circuity"].median())
                if len(direct)
                else float("nan"),
                "robust_distance_threshold_m": float(threshold_values.median())
                if threshold_values.notna().any()
                else float("nan"),
                "excluded_low_separation": int(
                    reasons.get("od_separation_under_minimum", 0)
                ),
                "excluded_high_circuity": int(reasons.get("circuity_above_maximum", 0)),
                "excluded_distance_outlier": int(
                    reasons.get("robust_od_distance_outlier", 0)
                ),
                "route_family_count": int(
                    families.loc[
                        families["origin_cluster_id"].eq(origin)
                        & families["destination_cluster_id"].eq(destination)
                        & ~families["is_other"],
                    ].shape[0]
                ),
                "dominant_route_family_id": dominant_id,
                "dominant_route_family_share": dominant_share,
                "early_window_trip_count": len(early_assignments),
                "late_window_trip_count": len(late_assignments),
                "early_late_total_variation": total_variation,
                "early_window_start": observed_months[0] if early else "",
                "early_window_end": observed_months[half - 1] if early else "",
                "late_window_start": observed_months[half] if late else "",
                "late_window_end": observed_months[-1] if late else "",
            }
        )
    result = pd.DataFrame(rows)
    if not sensitivity.empty:
        result = result.merge(
            sensitivity,
            on=["origin_cluster_id", "destination_cluster_id"],
            how="left",
            validate="one_to_one",
        )
    return result.sort_values(
        ["eligible_direct_trip_count", "total_trip_count"], ascending=False
    ).reset_index(drop=True)


def _monthly_family_shares(
    features: pd.DataFrame,
    assignments: pd.DataFrame,
    families: pd.DataFrame,
    observed_months: Sequence[str],
    observed_index: Mapping[str, int],
    *,
    min_month_trips: int,
    rolling_months: int,
    min_window_trips: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    feature_month_counts = (
        features.groupby(
            ["origin_cluster_id", "destination_cluster_id", "month"]
        )["trip_id"]
        .nunique()
        .to_dict()
    )
    family_lookup = families.set_index("route_family_id").to_dict(orient="index")
    for (origin, destination), od_assignments in assignments.groupby(
        ["origin_cluster_id", "destination_cluster_id"], sort=True
    ):
        family_ids = sorted(od_assignments["route_family_id"].unique())
        start_index = min(observed_index[month] for month in od_assignments["month"])
        end_index = max(observed_index[month] for month in od_assignments["month"])
        for month_index in range(start_index, end_index + 1):
            month = observed_months[month_index]
            month_assignments = od_assignments.loc[od_assignments["month"].eq(month)]
            denominator = len(month_assignments)
            for family_id in family_ids:
                family_month = month_assignments.loc[
                    month_assignments["route_family_id"].eq(family_id)
                ]
                family = family_lookup[family_id]
                member_features = features.loc[
                    family_month["_feature_index"].astype(int).tolist()
                ]
                distance_total = float(
                    member_features["route_distance_m"].clip(lower=0).sum()
                )

                def distance_share(column: str) -> float:
                    if distance_total <= 0 or member_features.empty:
                        return float("nan")
                    return float(
                        member_features[column].clip(lower=0).sum() / distance_total
                    )

                rows.append(
                    {
                        "origin_cluster_id": origin,
                        "origin_label": family["origin_label"],
                        "destination_cluster_id": destination,
                        "destination_label": family["destination_label"],
                        "route_family_id": family_id,
                        "family_name": family["family_name"],
                        "major_roads_json": family["backbone_roads_json"],
                        "is_other": bool(family["is_other"]),
                        "month": month,
                        "observed_month_index": month_index,
                        "observed_month_number": month_index + 1,
                        "calendar_month_index": (
                            pd.Period(month, freq="M")
                            - pd.Period(observed_months[0], freq="M")
                        ).n,
                        "family_trip_count": len(family_month),
                        "eligible_od_trip_count": denominator,
                        "all_od_trip_count": int(
                            feature_month_counts.get((origin, destination, month), 0)
                        ),
                        "family_unique_days": family_month["event_date"].replace(
                            "", np.nan
                        ).nunique(),
                        "route_share": len(family_month) / denominator
                        if denominator
                        else float("nan"),
                        "distance_weighted_highway_share": distance_share(
                            "highway_distance_m"
                        ),
                        "distance_weighted_arterial_share": distance_share(
                            "arterial_distance_m"
                        ),
                        "distance_weighted_local_share": distance_share(
                            "local_road_distance_m"
                        ),
                        "distance_weighted_surface_street_share": distance_share(
                            "surface_street_distance_m"
                        ),
                        "distance_weighted_toll_share": distance_share(
                            "toll_distance_m"
                        ),
                        "median_route_distance_m": float(
                            member_features["route_distance_m"].median()
                        )
                        if not member_features.empty
                        else float("nan"),
                        "median_duration_seconds": float(
                            member_features["duration_seconds"].median()
                        )
                        if not member_features.empty
                        else float("nan"),
                        "month_trip_sufficient": denominator >= min_month_trips,
                        "data_sufficiency": (
                            "sufficient" if denominator >= min_month_trips else "sparse"
                        ),
                        "confidence": (
                            "high"
                            if denominator >= max(10, min_month_trips)
                            and family_month["event_date"].replace("", np.nan).nunique()
                            >= 3
                            else "medium"
                            if denominator >= min_month_trips
                            else "low"
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    rolling_rows: list[pd.DataFrame] = []
    for (_, _, _), group in result.groupby(
        ["origin_cluster_id", "destination_cluster_id", "route_family_id"],
        sort=True,
    ):
        group = group.sort_values("observed_month_index").copy()
        group["rolling_3_observed_month_family_trips"] = group[
            "family_trip_count"
        ].rolling(rolling_months, min_periods=1).sum()
        group["rolling_3_observed_month_od_trips"] = group[
            "eligible_od_trip_count"
        ].rolling(rolling_months, min_periods=1).sum()
        group["rolling_3_observed_month_share"] = (
            group["rolling_3_observed_month_family_trips"]
            / group["rolling_3_observed_month_od_trips"].replace(0, np.nan)
        )
        contiguous_flags: list[bool] = []
        enough_rows: list[bool] = []
        month_values = list(group["month"])
        for position in range(len(group)):
            window = month_values[max(0, position - rolling_months + 1) : position + 1]
            enough = len(window) == rolling_months
            enough_rows.append(enough)
            periods = [pd.Period(value, freq="M") for value in window]
            contiguous_flags.append(
                enough
                and all(right == left + 1 for left, right in zip(periods, periods[1:]))
            )
        group["rolling_window_has_3_observed_months"] = enough_rows
        group["rolling_window_calendar_contiguous"] = contiguous_flags
        group["rolling_window_sufficient"] = (
            group["rolling_window_has_3_observed_months"]
            & group["rolling_window_calendar_contiguous"]
            & group["rolling_3_observed_month_od_trips"].ge(min_window_trips)
        )
        rolling_rows.append(group)
    result = pd.concat(rolling_rows, ignore_index=True)

    dominant_rows: list[dict[str, object]] = []
    for (origin, destination, month), group in result.groupby(
        ["origin_cluster_id", "destination_cluster_id", "month"], sort=True
    ):
        present = group.loc[group["family_trip_count"] > 0].sort_values(
            ["family_trip_count", "route_family_id"], ascending=[False, True]
        )
        dominant = present.iloc[0] if not present.empty else None
        dominant_rows.append(
            {
                "origin_cluster_id": origin,
                "destination_cluster_id": destination,
                "month": month,
                "monthly_route_family_count": int(len(present)),
                "dominant_route_family_id": (
                    str(dominant["route_family_id"]) if dominant is not None else ""
                ),
                "dominant_route_family_name": (
                    str(dominant["family_name"]) if dominant is not None else ""
                ),
                "dominant_route_family_share": (
                    float(dominant["route_share"])
                    if dominant is not None
                    else float("nan")
                ),
            }
        )
    result = result.merge(
        pd.DataFrame(dominant_rows),
        on=["origin_cluster_id", "destination_cluster_id", "month"],
        how="left",
        validate="many_to_one",
    )

    observed = result.loc[result["eligible_od_trip_count"] > 0]
    sums = observed.groupby(
        ["origin_cluster_id", "destination_cluster_id", "month"]
    ).agg(
        count_sum=("family_trip_count", "sum"),
        denominator=("eligible_od_trip_count", "first"),
        share_sum=("route_share", "sum"),
    )
    if not (sums["count_sum"] == sums["denominator"]).all():
        raise LongitudinalAnalysisError("Monthly family trip counts do not reconcile")
    if not np.allclose(sums["share_sum"], 1.0, atol=1e-9):
        raise LongitudinalAnalysisError("Monthly family shares do not sum to one")
    return result.sort_values(
        ["origin_cluster_id", "destination_cluster_id", "observed_month_index", "route_family_id"]
    ).reset_index(drop=True)


def _longitudinal_transitions(
    assignments: pd.DataFrame,
    families: pd.DataFrame,
    features: pd.DataFrame,
    monthly: pd.DataFrame,
    observed_months: Sequence[str],
    *,
    min_window_trips: int,
    share_change_threshold: float,
    persistence_months: int,
) -> pd.DataFrame:
    if assignments.empty:
        return pd.DataFrame()
    half = len(observed_months) // 2
    early_months = set(observed_months[:half])
    late_months = set(observed_months[half:])
    family_lookup = families.set_index("route_family_id").to_dict(orient="index")
    rows: list[dict[str, object]] = []

    def dominant_family(frame: pd.DataFrame) -> str:
        counts = frame["route_family_id"].value_counts().to_dict()
        return min(counts, key=lambda key: (-int(counts[key]), str(key)))

    def family_share(frame: pd.DataFrame, family_id: str) -> float:
        return float(frame["route_family_id"].eq(family_id).mean()) if len(frame) else 0.0

    def route_metrics(frame: pd.DataFrame) -> dict[str, float]:
        member_features = features.loc[frame["_feature_index"].astype(int).tolist()]
        distance = float(member_features["route_distance_m"].clip(lower=0).sum())

        def share(column: str) -> float:
            return (
                float(member_features[column].clip(lower=0).sum() / distance)
                if distance > 0
                else float("nan")
            )

        return {
            "highway_share": share("highway_distance_m"),
            "surface_share": share("surface_street_distance_m"),
            "local_share": share("local_road_distance_m"),
            "toll_share": share("toll_distance_m"),
            "median_distance_m": float(member_features["route_distance_m"].median()),
            "median_duration_seconds": float(member_features["duration_seconds"].median()),
        }

    for (origin, destination), od in assignments.groupby(
        ["origin_cluster_id", "destination_cluster_id"], sort=True
    ):
        early = od.loc[od["month"].isin(early_months)]
        late = od.loc[od["month"].isin(late_months)]
        if len(early) < min_window_trips or len(late) < min_window_trips:
            continue
        baseline_id = dominant_family(early)
        late_dominant_id = dominant_family(late)
        candidate_id = late_dominant_id
        if candidate_id == baseline_id:
            alternatives = []
            for family_id in sorted(set(od["route_family_id"])):
                if family_id == baseline_id or bool(family_lookup[family_id]["is_other"]):
                    continue
                delta = family_share(late, family_id) - family_share(early, family_id)
                alternatives.append((delta, family_share(late, family_id), family_id))
            if not alternatives:
                continue
            delta, _, candidate_id = max(alternatives)
            if delta < share_change_threshold:
                continue
        baseline = family_lookup[baseline_id]
        candidate = family_lookup[candidate_id]
        if bool(candidate["is_other"]):
            continue
        early_candidate_share = family_share(early, candidate_id)
        late_candidate_share = family_share(late, candidate_id)
        delta = late_candidate_share - early_candidate_share
        if delta < share_change_threshold:
            continue

        od_monthly = monthly.loc[
            monthly["origin_cluster_id"].eq(origin)
            & monthly["destination_cluster_id"].eq(destination)
        ].copy()
        candidate_monthly = od_monthly.loc[
            od_monthly["route_family_id"].eq(candidate_id)
        ].sort_values("observed_month_index")
        baseline_monthly = od_monthly.loc[
            od_monthly["route_family_id"].eq(baseline_id),
            ["month", "route_share"],
        ].rename(columns={"route_share": "baseline_month_share"})
        candidate_monthly = candidate_monthly.merge(
            baseline_monthly, on="month", how="left", validate="one_to_one"
        )
        candidate_present = candidate_monthly.loc[
            candidate_monthly["family_trip_count"] > 0
        ]
        adequate_appearance = candidate_present.loc[
            (candidate_present["family_trip_count"] >= 2)
            | candidate_present["month_trip_sufficient"]
        ]
        first_appearance = (
            adequate_appearance.iloc[0]["month"]
            if not adequate_appearance.empty
            else candidate_present.iloc[0]["month"]
            if not candidate_present.empty
            else ""
        )
        late_candidate_months = candidate_monthly.loc[
            candidate_monthly["month"].isin(late_months)
            & candidate_monthly["family_trip_count"].gt(0)
        ]
        persistence = int(late_candidate_months["month"].nunique())
        if persistence < persistence_months:
            continue
        crossover_rows = candidate_monthly.loc[
            candidate_monthly["route_share"].ge(
                candidate_monthly["baseline_month_share"].fillna(float("inf"))
            )
            & candidate_monthly["eligible_od_trip_count"].gt(0)
        ]
        crossover = crossover_rows.iloc[0]["month"] if not crossover_rows.empty else ""
        dominance_rows = candidate_monthly.loc[
            candidate_monthly["dominant_route_family_id"].eq(candidate_id)
            & candidate_monthly["eligible_od_trip_count"].gt(0)
        ]
        dominance = dominance_rows.iloc[0]["month"] if not dominance_rows.empty else ""
        late_adoption_rows = late_candidate_months.loc[
            late_candidate_months["route_share"].ge(
                early_candidate_share + min(share_change_threshold / 2.0, 0.10)
            )
        ]
        adoption = (
            late_adoption_rows.iloc[0]["month"]
            if not late_adoption_rows.empty
            else late_candidate_months.iloc[0]["month"]
        )
        adoption_index = int(
            candidate_monthly.loc[
                candidate_monthly["month"].eq(adoption), "observed_month_index"
            ].iloc[0]
        )
        crossover_rows = crossover_rows.loc[
            crossover_rows["observed_month_index"].ge(adoption_index)
        ]
        crossover = crossover_rows.iloc[0]["month"] if not crossover_rows.empty else ""
        dominance_rows = dominance_rows.loc[
            dominance_rows["observed_month_index"].ge(adoption_index)
        ]
        dominance = dominance_rows.iloc[0]["month"] if not dominance_rows.empty else ""
        reversion = ""
        reversion_months: list[str] = []
        if dominance:
            dominance_index = int(
                candidate_monthly.loc[
                    candidate_monthly["month"].eq(dominance), "observed_month_index"
                ].iloc[0]
            )
            reverted = candidate_monthly.loc[
                candidate_monthly["observed_month_index"].gt(dominance_index)
                & candidate_monthly["dominant_route_family_id"].eq(baseline_id)
                & candidate_monthly["eligible_od_trip_count"].gt(0)
            ]
            if not reverted.empty:
                reversion = str(reverted.iloc[0]["month"])
                reversion_months = [str(value) for value in reverted["month"]]

        early_candidate = early.loc[early["route_family_id"].eq(candidate_id)]
        late_candidate = late.loc[late["route_family_id"].eq(candidate_id)]
        early_baseline = early.loc[early["route_family_id"].eq(baseline_id)]
        baseline_metrics = route_metrics(early_baseline)
        later_metrics = route_metrics(late_candidate)
        confidence = (
            "high"
            if len(early) >= 25 and len(late) >= 25 and persistence >= 5
            else "medium"
        )
        transition_type = (
            "sustained_distribution_shift_with_intermediate_reversions"
            if reversion_months
            else "sustained_route_family_transition"
            if candidate_id != baseline_id and late_dominant_id == candidate_id
            else "sustained_alternate_route_increase"
        )
        rows.append(
            {
                "origin_cluster_id": origin,
                "origin_label": candidate["origin_label"],
                "destination_cluster_id": destination,
                "destination_label": candidate["destination_label"],
                "baseline_start": observed_months[0],
                "baseline_end": observed_months[half - 1],
                "baseline_route_family_id": baseline_id,
                "baseline_route_family": baseline["family_name"],
                "baseline_major_roads_json": baseline["backbone_roads_json"],
                "baseline_share": family_share(early, baseline_id),
                "first_alternate_appearance": first_appearance,
                "adoption_start": adoption,
                "crossover_month": crossover,
                "dominance_month": dominance,
                "later_start": observed_months[half],
                "later_end": observed_months[-1],
                "later_route_family_id": candidate_id,
                "later_route_family": candidate["family_name"],
                "later_major_roads_json": candidate["backbone_roads_json"],
                "later_share": late_candidate_share,
                "persistence_months": persistence,
                "reversion_month": reversion,
                "reversion_months_json": _json(reversion_months),
                "trips_before": len(early),
                "trips_after": len(late),
                "baseline_family_trips_before": len(early_baseline),
                "later_family_trips_before": len(early_candidate),
                "later_family_trips_after": len(late_candidate),
                "baseline_highway_share": baseline_metrics["highway_share"],
                "later_highway_share": later_metrics["highway_share"],
                "highway_share_change": later_metrics["highway_share"]
                - baseline_metrics["highway_share"],
                "baseline_surface_street_share": baseline_metrics["surface_share"],
                "later_surface_street_share": later_metrics["surface_share"],
                "surface_street_share_change": later_metrics["surface_share"]
                - baseline_metrics["surface_share"],
                "baseline_local_share": baseline_metrics["local_share"],
                "later_local_share": later_metrics["local_share"],
                "local_share_change": later_metrics["local_share"]
                - baseline_metrics["local_share"],
                "baseline_toll_share": baseline_metrics["toll_share"],
                "later_toll_share": later_metrics["toll_share"],
                "toll_share_change": later_metrics["toll_share"]
                - baseline_metrics["toll_share"],
                "baseline_median_distance_m": baseline_metrics["median_distance_m"],
                "later_median_distance_m": later_metrics["median_distance_m"],
                "distance_change_m": later_metrics["median_distance_m"]
                - baseline_metrics["median_distance_m"],
                "baseline_median_duration_seconds": baseline_metrics[
                    "median_duration_seconds"
                ],
                "later_median_duration_seconds": later_metrics[
                    "median_duration_seconds"
                ],
                "duration_change_seconds": later_metrics["median_duration_seconds"]
                - baseline_metrics["median_duration_seconds"],
                "route_family_id": candidate_id,
                "family_name": candidate["family_name"],
                "transition_type": transition_type,
                "early_window_start": observed_months[0],
                "early_window_end": observed_months[half - 1],
                "late_window_start": observed_months[half],
                "late_window_end": observed_months[-1],
                "early_od_trip_count": len(early),
                "late_od_trip_count": len(late),
                "early_family_trip_count": len(early_candidate),
                "late_family_trip_count": len(late_candidate),
                "early_route_share": early_candidate_share,
                "late_route_share": late_candidate_share,
                "route_share_change": delta,
                "route_share_change_percentage_points": 100.0 * delta,
                "persistence_observed_months": persistence,
                "confidence": confidence,
                "plain_english_story": (
                    f"For {candidate['origin_label']} to {candidate['destination_label']}, "
                    f"the {candidate['family_name']} family rose from "
                    f"{early_candidate_share:.0%} of {len(early)} eligible direct trips "
                    f"in {observed_months[0]}–{observed_months[half - 1]} to "
                    f"{late_candidate_share:.0%} of {len(late)} trips in "
                    f"{observed_months[half]}–{observed_months[-1]}. It appeared in "
                    f"{persistence} late-period observed months; this supports a sustained "
                    "route-distribution change rather than a permanent replacement."
                    + (
                        f" The earlier family reappeared in {', '.join(reversion_months)}, "
                        "so the shift included intermittent reversions."
                        if reversion_months
                        else ""
                    )
                    + " The GPS data do not identify the cause."
                ),
                "limitations": (
                    "Activity purpose and cause are unknown; observed-month coverage varies, "
                    "and route-family assignment depends on map matching and the declared "
                    "directness/similarity thresholds. Persistence counts observed months "
                    "with the later family, including sparse months."
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["route_share_change_percentage_points"], key=lambda values: values.abs(), ascending=False
    ).reset_index(drop=True)


def _temporary_deviations(
    monthly: pd.DataFrame,
    *,
    share_threshold: float,
    max_episode_months: int,
    reversion_months: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    usable = monthly.loc[
        ~monthly["is_other"] & monthly["month_trip_sufficient"]
    ].copy()
    for (origin, destination, family_id), group in usable.groupby(
        ["origin_cluster_id", "destination_cluster_id", "route_family_id"],
        sort=True,
    ):
        group = group.sort_values("observed_month_index").reset_index(drop=True)
        high_positions = set(group.index[group["route_share"].ge(share_threshold)])
        visited: set[int] = set()
        for position in sorted(high_positions):
            if position in visited:
                continue
            episode = [position]
            visited.add(position)
            next_position = position + 1
            while (
                next_position in high_positions
                and group.loc[next_position, "observed_month_index"]
                == group.loc[episode[-1], "observed_month_index"] + 1
                and pd.Period(group.loc[next_position, "month"], freq="M")
                == pd.Period(group.loc[episode[-1], "month"], freq="M") + 1
            ):
                episode.append(next_position)
                visited.add(next_position)
                next_position += 1
            if len(episode) > max_episode_months:
                continue
            before = group.loc[: episode[0] - 1]
            before = before.loc[
                group.loc[episode[0], "observed_month_index"]
                - before["observed_month_index"]
                <= reversion_months
            ]
            before = before.loc[before["route_share"].lt(share_threshold)]
            after = group.loc[episode[-1] + 1 :]
            after = after.loc[
                after["observed_month_index"]
                - group.loc[episode[-1], "observed_month_index"]
                <= reversion_months
            ]
            after = after.loc[after["route_share"].lt(share_threshold)]
            if before.empty or after.empty:
                continue
            start = group.loc[episode[0]]
            end = group.loc[episode[-1]]
            revert = after.iloc[0]
            rows.append(
                {
                    "origin_cluster_id": origin,
                    "origin_label": start["origin_label"],
                    "destination_cluster_id": destination,
                    "destination_label": start["destination_label"],
                    "route_family_id": family_id,
                    "family_name": start["family_name"],
                    "episode_start_month": start["month"],
                    "episode_end_month": end["month"],
                    "episode_observed_months": len(episode),
                    "peak_route_share": float(group.loc[episode, "route_share"].max()),
                    "episode_family_trips": int(
                        group.loc[episode, "family_trip_count"].sum()
                    ),
                    "episode_od_trips": int(
                        group.loc[episode, "eligible_od_trip_count"].sum()
                    ),
                    "reversion_month": revert["month"],
                    "reversion_route_share": float(revert["route_share"]),
                    "confidence": "medium",
                    "plain_english_story": (
                        f"The {start['family_name']} family temporarily reached "
                        f"{group.loc[episode, 'route_share'].max():.0%} of eligible "
                        f"{start['origin_label']} to {start['destination_label']} trips, "
                        f"then fell below {share_threshold:.0%} by {revert['month']}."
                    ),
                    "limitations": (
                        "This identifies a temporary route-share deviation, not its cause; "
                        "only months meeting the minimum OD trip threshold are evaluated."
                    ),
                }
            )
    return pd.DataFrame(rows)


def _monthly_trends(
    features: pd.DataFrame,
    od_summary: pd.DataFrame,
    observed_months: Sequence[str],
    observed_index: Mapping[str, int],
    *,
    rolling_months: int,
) -> pd.DataFrame:
    recurring_pairs = {
        (str(row.origin_cluster_id), str(row.destination_cluster_id))
        for row in od_summary.itertuples(index=False)
        if int(row.eligible_direct_trip_count) >= 15 and int(row.eligible_months) >= 3
    }
    scopes = {
        "all_trips": features,
        "eligible_direct_routes": features.loc[features["direct_route_eligible"]],
        "recurring_direct_od": features.loc[
            features.apply(
                lambda row: (
                    row["origin_cluster_id"], row["destination_cluster_id"]
                )
                in recurring_pairs
                and bool(row["direct_route_eligible"]),
                axis=1,
            )
        ],
    }
    rows: list[dict[str, object]] = []
    first_period = pd.Period(observed_months[0], freq="M")
    for scope, data in scopes.items():
        for month in observed_months:
            group = data.loc[data["month"].eq(month)]
            total_distance = float(group["route_distance_m"].clip(lower=0).sum())
            highway_distance = float(group["highway_distance_m"].clip(lower=0).sum())
            surface_distance = float(group["surface_street_distance_m"].clip(lower=0).sum())
            arterial_distance = float(group["arterial_distance_m"].clip(lower=0).sum())
            local_distance = float(group["local_road_distance_m"].clip(lower=0).sum())
            per_trip_highway = (
                group["highway_distance_m"] / group["route_distance_m"].replace(0, np.nan)
            )
            period = pd.Period(month, freq="M")
            observation_days = group["event_date"].replace("", np.nan).nunique()
            rows.append(
                {
                    "scope": scope,
                    "month": month,
                    "observed_month_index": observed_index[month],
                    "observed_month_number": observed_index[month] + 1,
                    "calendar_month_index": (period - first_period).n,
                    "trip_count": len(group),
                    "observation_days": observation_days,
                    "calendar_days": calendar.monthrange(period.year, period.month)[1],
                    "observation_day_share": observation_days
                    / calendar.monthrange(period.year, period.month)[1],
                    "total_route_distance_m": total_distance,
                    "highway_distance_m": highway_distance,
                    "surface_street_distance_m": surface_distance,
                    "arterial_distance_m": arterial_distance,
                    "local_road_distance_m": local_distance,
                    "distance_weighted_highway_share": highway_distance / total_distance
                    if total_distance > 0
                    else float("nan"),
                    "distance_weighted_surface_share": surface_distance / total_distance
                    if total_distance > 0
                    else float("nan"),
                    "distance_weighted_arterial_share": arterial_distance / total_distance
                    if total_distance > 0
                    else float("nan"),
                    "distance_weighted_local_share": local_distance / total_distance
                    if total_distance > 0
                    else float("nan"),
                    "highway_trip_count": int(group["highway_distance_m"].gt(0).sum()),
                    "highway_trip_share": float(group["highway_distance_m"].gt(0).mean())
                    if len(group)
                    else float("nan"),
                    "mean_trip_highway_share": float(per_trip_highway.mean())
                    if len(group)
                    else float("nan"),
                    "median_trip_highway_share": float(per_trip_highway.median())
                    if len(group)
                    else float("nan"),
                    "trend_month_sufficient": len(group) >= 30 and observation_days >= 10,
                }
            )
    result = pd.DataFrame(rows)
    rolling_parts: list[pd.DataFrame] = []
    for _, group in result.groupby("scope", sort=True):
        group = group.sort_values("observed_month_index").copy()
        for source, destination in (
            ("trip_count", "rolling_3_observed_month_trips"),
            ("total_route_distance_m", "rolling_3_observed_month_distance_m"),
            ("highway_distance_m", "rolling_3_observed_month_highway_m"),
            ("surface_street_distance_m", "rolling_3_observed_month_surface_m"),
        ):
            group[destination] = group[source].rolling(
                rolling_months, min_periods=1
            ).sum()
        group["rolling_3_observed_month_highway_share"] = (
            group["rolling_3_observed_month_highway_m"]
            / group["rolling_3_observed_month_distance_m"].replace(0, np.nan)
        )
        group["rolling_3_observed_month_surface_share"] = (
            group["rolling_3_observed_month_surface_m"]
            / group["rolling_3_observed_month_distance_m"].replace(0, np.nan)
        )
        contiguous: list[bool] = []
        months = list(group["month"])
        for position in range(len(group)):
            window = months[max(0, position - rolling_months + 1) : position + 1]
            periods = [pd.Period(value, freq="M") for value in window]
            contiguous.append(
                len(window) == rolling_months
                and all(right == left + 1 for left, right in zip(periods, periods[1:]))
            )
        group["rolling_window_calendar_contiguous"] = contiguous
        rolling_parts.append(group)
    return pd.concat(rolling_parts, ignore_index=True).sort_values(
        ["scope", "observed_month_index"]
    ).reset_index(drop=True)


def _geometry_wkt(
    sequence: Sequence[CountyFID],
    roads: Mapping[CountyFID, Mapping[str, object]],
) -> str:
    geometries: list[object] = []
    try:
        from shapely import wkt
        from shapely.ops import unary_union
    except ImportError:
        return ""
    for key in sequence:
        road = roads.get(key, {})
        geometry = road.get("geometry")
        if geometry is None:
            text = _text(road.get("geometry_wkt"))
            if text:
                try:
                    geometry = wkt.loads(text)
                except Exception:
                    geometry = None
        if geometry is not None:
            geometries.append(geometry)
    if not geometries:
        return ""
    try:
        return unary_union(geometries).wkt
    except Exception:
        return ""


def _map_representatives(
    families: pd.DataFrame,
    assignments: pd.DataFrame,
    features: pd.DataFrame,
    roads: Mapping[CountyFID, Mapping[str, object]],
) -> pd.DataFrame:
    if families.empty:
        return pd.DataFrame()
    feature_lookup = features.set_index("trip_id").to_dict(orient="index")
    rows: list[dict[str, object]] = []
    for family in families.loc[~families["is_other"]].itertuples(index=False):
        medoid = feature_lookup[str(family.medoid_trip_id)]
        sequence = list(medoid["county_fid_sequence"])
        rows.append(
            {
                "route_family_id": family.route_family_id,
                "origin_cluster_id": family.origin_cluster_id,
                "origin_label": family.origin_label,
                "destination_cluster_id": family.destination_cluster_id,
                "destination_label": family.destination_label,
                "family_name": family.family_name,
                "medoid_trip_id": family.medoid_trip_id,
                "trip_count": family.trip_count,
                "overall_route_share": family.overall_route_share,
                "backbone_roads_json": family.backbone_roads_json,
                "county_fid_sequence_json": _json(
                    [{"county": county, "fid": fid} for county, fid in sequence]
                ),
                "road_name_sequence_json": _json(medoid["road_name_sequence"]),
                "start_latitude": medoid["start_latitude"],
                "start_longitude": medoid["start_longitude"],
                "end_latitude": medoid["end_latitude"],
                "end_longitude": medoid["end_longitude"],
                "route_distance_m": medoid["route_distance_m"],
                "highway_share": medoid["highway_distance_m"]
                / medoid["route_distance_m"]
                if medoid["route_distance_m"] > 0
                else float("nan"),
                "surface_street_share": medoid["surface_street_distance_m"]
                / medoid["route_distance_m"]
                if medoid["route_distance_m"] > 0
                else float("nan"),
                "geometry_wkt_projected": _geometry_wkt(sequence, roads),
                "geometry_source_crs": "EPSG:26917",
            }
        )
    return pd.DataFrame(rows)


def analyze_longitudinal_routes(
    trips: pd.DataFrame,
    road_context: pd.DataFrame | Mapping[CountyFID, Mapping[str, object]],
    *,
    min_od_separation_m: float = 500.0,
    max_circuity: float = 3.0,
    robust_outlier_z: float = 3.0,
    trim_access_m: float = 250.0,
    similarity_threshold: float = 0.70,
    sensitivity_thresholds: Sequence[float] = (0.65, 0.75),
    lcs_merge_threshold: float = 0.80,
    min_family_trips: int = 3,
    min_family_days: int = 2,
    min_month_trips: int = 5,
    rolling_months: int = 3,
    min_window_trips: int = 10,
    sustained_share_change: float = 0.20,
    sustained_persistence_months: int = 3,
    temporary_share_threshold: float = 0.30,
    temporary_max_months: int = 2,
    temporary_reversion_months: int = 2,
) -> dict[str, pd.DataFrame]:
    """Run the complete longitudinal route analysis without reading or writing files.

    Returned keys are ``od_summary``, ``route_families``,
    ``route_family_monthly_shares``, ``longitudinal_route_transitions``,
    ``temporary_route_deviations``, ``monthly_highway_surface_trends``, and
    ``route_family_map_representatives``.
    """
    if min_od_separation_m <= 0 or max_circuity <= 0 or trim_access_m < 0:
        raise ValueError("Distance, circuity, and trim thresholds must be valid")
    if not 0 <= similarity_threshold <= 1 or not 0 <= lcs_merge_threshold <= 1:
        raise ValueError("Similarity thresholds must be on the 0-1 scale")
    if any(not 0 <= float(value) <= 1 for value in sensitivity_thresholds):
        raise ValueError("Sensitivity thresholds must be on the 0-1 scale")
    if min_family_trips < 1 or min_family_days < 1 or min_month_trips < 1:
        raise ValueError("Count thresholds must be positive")
    roads = _normalize_road_context(road_context)
    features = _route_features(trips, roads, trim_access_m=trim_access_m)
    features = _mark_direct_eligibility(
        features,
        min_od_separation_m=min_od_separation_m,
        max_circuity=max_circuity,
        robust_outlier_z=robust_outlier_z,
    )
    observed_months, observed_index = _observed_months(features)
    if len(observed_months) < 2:
        raise LongitudinalAnalysisError(
            "At least two observed months are required for longitudinal analysis"
        )
    assignments, families, sensitivity = _family_analysis(
        features,
        similarity_threshold=similarity_threshold,
        sensitivity_thresholds=sensitivity_thresholds,
        lcs_merge_threshold=lcs_merge_threshold,
        min_family_trips=min_family_trips,
        min_family_days=min_family_days,
    )
    od_summary = _od_summary(
        features, assignments, families, sensitivity, observed_months
    )
    monthly = _monthly_family_shares(
        features,
        assignments,
        families,
        observed_months,
        observed_index,
        min_month_trips=min_month_trips,
        rolling_months=rolling_months,
        min_window_trips=min_window_trips,
    )
    transitions = _longitudinal_transitions(
        assignments,
        families,
        features,
        monthly,
        observed_months,
        min_window_trips=min_window_trips,
        share_change_threshold=sustained_share_change,
        persistence_months=sustained_persistence_months,
    )
    temporary = _temporary_deviations(
        monthly,
        share_threshold=temporary_share_threshold,
        max_episode_months=temporary_max_months,
        reversion_months=temporary_reversion_months,
    )
    trends = _monthly_trends(
        features,
        od_summary,
        observed_months,
        observed_index,
        rolling_months=rolling_months,
    )
    representatives = _map_representatives(families, assignments, features, roads)

    eligible_trip_count = int(features["direct_route_eligible"].sum())
    if int(families["trip_count"].sum()) != eligible_trip_count if not families.empty else eligible_trip_count != 0:
        raise LongitudinalAnalysisError("Route-family totals do not reconcile to eligible trips")
    if len(assignments) != eligible_trip_count:
        raise LongitudinalAnalysisError("Route-family assignments do not reconcile")
    return {
        "od_summary": od_summary,
        "route_families": families.sort_values(
            ["origin_cluster_id", "destination_cluster_id", "family_rank"]
        ).reset_index(drop=True)
        if not families.empty
        else families,
        "route_family_monthly_shares": monthly,
        "longitudinal_route_transitions": transitions,
        "temporary_route_deviations": temporary,
        "monthly_highway_surface_trends": trends,
        "route_family_map_representatives": representatives,
    }


__all__ = [
    "ARTERIAL_CLASSES",
    "HIGHWAY_CLASSES",
    "LOCAL_ACCESS_CLASSES",
    "LongitudinalAnalysisError",
    "analyze_longitudinal_routes",
    "canonicalize_road_name",
    "ordered_lcs_similarity",
    "route_similarity",
    "weighted_overlap",
]
