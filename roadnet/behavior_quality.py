"""Deterministic, non-destructive quality checks for trip summaries.

The helpers in this module are intentionally independent of the Driver 1003
pipeline.  :func:`validate_trip_quality` accepts the dataframe produced by
``real_world_behavior.build_trip_summary``, returns a copy with merged JSON
quality flags, and never removes or reorders a row.  Existing flags may be a
JSON string or a Python collection and are retained.

These checks identify records that require review; they do not assert that a
flagged trip is unusable.  In particular, long round trips can be legitimate,
so circuity is evaluated only when the straight-line endpoint separation is
large enough to make the ratio meaningful.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


EARTH_RADIUS_M = 6_371_008.8
MAX_AVERAGE_SPEED_MPH = 100.0
MAX_ROUTE_DISTANCE_M = 250_000.0
MIN_CIRCUITY_ENDPOINT_DISTANCE_M = 500.0
MIN_PLAUSIBLE_CIRCUITY = 0.75
MAX_PLAUSIBLE_CIRCUITY = 5.0
MIN_OD_PAIR_SAMPLES = 5
OD_DISTANCE_MAD_MULTIPLIER = 6.0
OD_DISTANCE_MIN_RELATIVE_DEVIATION = 0.75
OD_DISTANCE_MIN_ABSOLUTE_DEVIATION_M = 1_000.0
UNCLUSTERED_LABEL = "UNCLUSTERED"


# Known flags have a stable semantic order.  Caller-defined flags are retained
# after these values in lexical order, making repeated runs byte-for-byte
# deterministic without discarding prior pipeline annotations.
QUALITY_FLAG_ORDER = (
    "nonpositive_duration",
    "missing_or_invalid_endpoint",
    "repeated_start_timestamp",
    "repeated_end_timestamp",
    "no_matched_route",
    "empty_fid_sequence",
    "empty_road_sequence",
    "average_speed_unavailable",
    "implausible_average_speed",
    "implausible_route_distance",
    "implausible_circuity",
    "od_pair_distance_outlier",
    "cross_county_trip",
)
_QUALITY_FLAG_RANK = {flag: index for index, flag in enumerate(QUALITY_FLAG_ORDER)}


def _normalize_flag(value: Any) -> str:
    text = str(value).strip().casefold().replace("-", " ")
    return "_".join(text.split())


def parse_quality_flags(value: Any) -> tuple[str, ...]:
    """Return normalized flags from JSON, a Python collection, or one string.

    Invalid JSON is treated as a single existing flag rather than discarded.
    Mapping inputs may use ``flag: bool`` entries; only truthy entries remain.
    """

    if value is None:
        return ()
    try:
        if bool(pd.isna(value)):
            return ()
    except (TypeError, ValueError):
        pass

    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text or text.casefold() in {"nan", "none", "null"}:
            return ()
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = [text]

    if isinstance(parsed, Mapping):
        values: Iterable[Any] = [key for key, active in parsed.items() if active]
    elif isinstance(parsed, (list, tuple, set, frozenset, pd.Series)):
        values = parsed
    else:
        values = [parsed]

    normalized = {_normalize_flag(item) for item in values if str(item).strip()}
    normalized.discard("")
    return tuple(
        sorted(
            normalized,
            key=lambda flag: (_QUALITY_FLAG_RANK.get(flag, len(QUALITY_FLAG_ORDER)), flag),
        )
    )


def merge_quality_flags(existing: Any, new_flags: Iterable[str]) -> tuple[str, ...]:
    """Merge prior and new flags using :data:`QUALITY_FLAG_ORDER`."""

    combined = set(parse_quality_flags(existing))
    combined.update(
        normalized
        for flag in new_flags
        if (normalized := _normalize_flag(flag))
    )
    return tuple(
        sorted(
            combined,
            key=lambda flag: (_QUALITY_FLAG_RANK.get(flag, len(QUALITY_FLAG_ORDER)), flag),
        )
    )


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _valid_latitude(value: Any) -> bool:
    number = _finite_number(value)
    return number is not None and -90.0 <= number <= 90.0


def _valid_longitude(value: Any) -> bool:
    number = _finite_number(value)
    return number is not None and -180.0 <= number <= 180.0


def _valid_endpoints(row: Mapping[str, Any]) -> bool:
    return (
        _valid_latitude(row.get("start_latitude"))
        and _valid_longitude(row.get("start_longitude"))
        and _valid_latitude(row.get("end_latitude"))
        and _valid_longitude(row.get("end_longitude"))
    )


def _sequence_is_empty(value: Any) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass

    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text or text.casefold() in {"nan", "none", "null"}:
            return True
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False

    if isinstance(parsed, Mapping):
        return not parsed
    if isinstance(parsed, (list, tuple, set, frozenset, pd.Series)):
        return not any(not _sequence_is_empty(item) for item in parsed)
    return False


def _haversine_m(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    phi_a = math.radians(latitude_a)
    phi_b = math.radians(latitude_b)
    delta_phi = math.radians(latitude_b - latitude_a)
    delta_lambda = math.radians(longitude_b - longitude_a)
    value = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_a)
        * math.cos(phi_b)
        * math.sin(delta_lambda / 2.0) ** 2
    )
    return EARTH_RADIUS_M * 2.0 * math.atan2(
        math.sqrt(value), math.sqrt(max(1.0 - value, 0.0))
    )


def _trip_identities(frame: pd.DataFrame) -> list[str]:
    if "trip_id" not in frame:
        return [f"__row_{position}" for position in range(len(frame))]
    result: list[str] = []
    for position, value in enumerate(frame["trip_id"].tolist()):
        text = "" if value is None else str(value).strip()
        result.append(text if text and text.casefold() != "nan" else f"__row_{position}")
    return result


def _repeated_timestamp_positions(
    frame: pd.DataFrame,
    column: str,
    identities: Sequence[str],
) -> set[int]:
    if column not in frame:
        return set()
    timestamps = pd.to_datetime(frame[column], errors="coerce", utc=True)
    groups: dict[pd.Timestamp, list[tuple[int, str]]] = defaultdict(list)
    for position, (timestamp, identity) in enumerate(
        zip(timestamps.tolist(), identities, strict=True)
    ):
        if pd.notna(timestamp):
            groups[timestamp].append((position, identity))
    repeated: set[int] = set()
    for members in groups.values():
        if len({identity for _, identity in members}) > 1:
            repeated.update(position for position, _ in members)
    return repeated


def _add_od_pair_distance_outliers(
    frame: pd.DataFrame,
    flags: list[set[str]],
    *,
    min_samples: int,
    mad_multiplier: float,
    min_relative_deviation: float,
    min_absolute_deviation_m: float,
) -> None:
    required = {"origin_cluster_id", "destination_cluster_id", "route_distance_m"}
    if not required.issubset(frame.columns):
        return

    groups: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for position, row in enumerate(frame.to_dict(orient="records")):
        origin = str(row.get("origin_cluster_id", "")).strip()
        destination = str(row.get("destination_cluster_id", "")).strip()
        distance = _finite_number(row.get("route_distance_m"))
        if (
            not origin
            or not destination
            or origin == UNCLUSTERED_LABEL
            or destination == UNCLUSTERED_LABEL
            or distance is None
            or distance <= 0
        ):
            continue
        groups[(origin, destination)].append((position, distance))

    for members in groups.values():
        if len(members) < min_samples:
            continue
        distances = [distance for _, distance in members]
        center = float(median(distances))
        mad = float(median(abs(distance - center) for distance in distances))
        scaled_mad = 1.4826 * mad
        allowed_deviation = max(
            mad_multiplier * scaled_mad,
            min_relative_deviation * center,
            min_absolute_deviation_m,
        )
        for position, distance in members:
            if abs(distance - center) > allowed_deviation:
                flags[position].add("od_pair_distance_outlier")


def validate_trip_quality(
    trips: pd.DataFrame,
    *,
    flag_column: str = "data_quality_flags",
    max_average_speed_mph: float = MAX_AVERAGE_SPEED_MPH,
    max_route_distance_m: float = MAX_ROUTE_DISTANCE_M,
    min_circuity_endpoint_distance_m: float = MIN_CIRCUITY_ENDPOINT_DISTANCE_M,
    min_plausible_circuity: float = MIN_PLAUSIBLE_CIRCUITY,
    max_plausible_circuity: float = MAX_PLAUSIBLE_CIRCUITY,
    min_od_pair_samples: int = MIN_OD_PAIR_SAMPLES,
    od_distance_mad_multiplier: float = OD_DISTANCE_MAD_MULTIPLIER,
    od_distance_min_relative_deviation: float = OD_DISTANCE_MIN_RELATIVE_DEVIATION,
    od_distance_min_absolute_deviation_m: float = OD_DISTANCE_MIN_ABSOLUTE_DEVIATION_M,
) -> pd.DataFrame:
    """Return a row-preserving copy annotated with deterministic JSON flags.

    OD-pair distance outliers use a median/MAD rule with relative and absolute
    floors.  This makes the result robust when most repeated trips have exactly
    the same distance.  No row is removed, and the input dataframe is untouched.
    """

    if min_od_pair_samples < 1:
        raise ValueError("min_od_pair_samples must be positive")
    if max_average_speed_mph <= 0 or max_route_distance_m <= 0:
        raise ValueError("speed and route-distance limits must be positive")
    if not 0 < min_plausible_circuity < max_plausible_circuity:
        raise ValueError("circuity bounds must be positive and increasing")

    output = trips.copy(deep=True)
    existing = (
        output[flag_column].tolist()
        if flag_column in output
        else [()] * len(output)
    )
    additions: list[set[str]] = [set() for _ in range(len(output))]
    identities = _trip_identities(output)
    repeated_starts = _repeated_timestamp_positions(
        output, "start_timestamp", identities
    )
    repeated_ends = _repeated_timestamp_positions(output, "end_timestamp", identities)

    for position, row in enumerate(output.to_dict(orient="records")):
        duration = _finite_number(row.get("trip_duration_seconds"))
        if duration is not None and duration <= 0:
            additions[position].add("nonpositive_duration")

        endpoints_valid = _valid_endpoints(row)
        if not endpoints_valid:
            additions[position].add("missing_or_invalid_endpoint")
        if position in repeated_starts:
            additions[position].add("repeated_start_timestamp")
        if position in repeated_ends:
            additions[position].add("repeated_end_timestamp")

        if _sequence_is_empty(row.get("matched_fid_sequence")):
            additions[position].add("empty_fid_sequence")
        if _sequence_is_empty(row.get("matched_road_name_sequence")):
            additions[position].add("empty_road_sequence")

        average_speed = _finite_number(row.get("average_speed_mph"))
        if average_speed is not None and (
            average_speed < 0 or average_speed > max_average_speed_mph
        ):
            additions[position].add("implausible_average_speed")

        route_distance = _finite_number(row.get("route_distance_m"))
        if (
            route_distance is None
            or route_distance <= 0
            or route_distance > max_route_distance_m
        ):
            additions[position].add("implausible_route_distance")
        elif endpoints_valid:
            endpoint_distance = _haversine_m(
                float(row["start_latitude"]),
                float(row["start_longitude"]),
                float(row["end_latitude"]),
                float(row["end_longitude"]),
            )
            if endpoint_distance >= min_circuity_endpoint_distance_m:
                circuity = route_distance / endpoint_distance
                if not min_plausible_circuity <= circuity <= max_plausible_circuity:
                    additions[position].add("implausible_circuity")

    _add_od_pair_distance_outliers(
        output,
        additions,
        min_samples=min_od_pair_samples,
        mad_multiplier=od_distance_mad_multiplier,
        min_relative_deviation=od_distance_min_relative_deviation,
        min_absolute_deviation_m=od_distance_min_absolute_deviation_m,
    )

    output[flag_column] = [
        json.dumps(
            merge_quality_flags(previous, new),
            separators=(",", ":"),
            ensure_ascii=True,
        )
        for previous, new in zip(existing, additions, strict=True)
    ]
    return output


def quality_flag_counts(
    trips_or_flags: pd.DataFrame | pd.Series | Iterable[Any],
    *,
    flag_column: str = "data_quality_flags",
) -> dict[str, int]:
    """Return deterministic counts of rows containing each quality flag."""

    if isinstance(trips_or_flags, pd.DataFrame):
        if flag_column not in trips_or_flags:
            raise ValueError(f"Missing quality flag column: {flag_column}")
        values: Iterable[Any] = trips_or_flags[flag_column]
    elif isinstance(trips_or_flags, pd.Series):
        values = trips_or_flags
    else:
        values = trips_or_flags

    counts: Counter[str] = Counter()
    for value in values:
        counts.update(set(parse_quality_flags(value)))
    ordered = sorted(
        counts,
        key=lambda flag: (_QUALITY_FLAG_RANK.get(flag, len(QUALITY_FLAG_ORDER)), flag),
    )
    return {flag: int(counts[flag]) for flag in ordered}


__all__ = [
    "MAX_AVERAGE_SPEED_MPH",
    "MAX_ROUTE_DISTANCE_M",
    "MIN_CIRCUITY_ENDPOINT_DISTANCE_M",
    "MIN_PLAUSIBLE_CIRCUITY",
    "MAX_PLAUSIBLE_CIRCUITY",
    "MIN_OD_PAIR_SAMPLES",
    "QUALITY_FLAG_ORDER",
    "parse_quality_flags",
    "merge_quality_flags",
    "validate_trip_quality",
    "quality_flag_counts",
]
