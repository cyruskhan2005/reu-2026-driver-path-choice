"""Stay reconstruction and trip-chain evidence for mobility interpretation.

The public behavior report must not confuse an interval between two recorded
trips with a fully observed activity.  This module therefore distinguishes
measured within-session stays from cross-session continuity, micro boundaries,
and spatially discontinuous recording gaps.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


LOCAL_TIMEZONE = "America/New_York"
MIN_ACTIVITY_STAY_MINUTES = 5.0
MAX_CONTINUITY_GAP_MINUTES = 48.0 * 60.0
MAX_STAY_SPATIAL_GAP_M = 150.0


def haversine_m(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    """Return WGS84 great-circle distance in metres."""
    radius = 6_371_008.8
    phi_a = math.radians(float(latitude_a))
    phi_b = math.radians(float(latitude_b))
    delta_phi = math.radians(float(latitude_b) - float(latitude_a))
    delta_lambda = math.radians(float(longitude_b) - float(longitude_a))
    value = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_a)
        * math.cos(phi_b)
        * math.sin(delta_lambda / 2.0) ** 2
    )
    return radius * 2.0 * math.atan2(
        math.sqrt(value), math.sqrt(max(1.0 - value, 0.0))
    )


def _local_time(values: pd.Series, timezone_name: str) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(
        timezone_name
    )


def _crosses_local_hour(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    hour: int,
) -> bool:
    """Return whether ``(start, end]`` contains the requested local hour."""
    if pd.isna(start) or pd.isna(end) or end <= start:
        return False
    anchor = start.normalize() + pd.Timedelta(hours=hour)
    if anchor <= start:
        anchor += pd.Timedelta(days=1)
    return bool(anchor <= end)


def _time_of_day(timestamp: pd.Timestamp) -> str:
    if pd.isna(timestamp):
        return "unknown"
    hour = timestamp.hour + timestamp.minute / 60.0
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "nighttime"


def reconstruct_stays(
    trips: pd.DataFrame,
    *,
    timezone_name: str = LOCAL_TIMEZONE,
    minimum_activity_minutes: float = MIN_ACTIVITY_STAY_MINUTES,
    maximum_continuity_minutes: float = MAX_CONTINUITY_GAP_MINUTES,
    maximum_spatial_gap_m: float = MAX_STAY_SPATIAL_GAP_M,
) -> pd.DataFrame:
    """Classify every consecutive inter-trip boundary.

    A ``MEASURED_STAY`` requires the next recorded trip to begin in the same
    cluster and session, within ``maximum_spatial_gap_m``, after at least
    ``minimum_activity_minutes``.  Cross-session matches are useful continuity
    and overnight evidence, but their duration is censored because the recorder
    was not continuously observed.
    """
    required = {
        "trip_id",
        "session_id",
        "start_timestamp",
        "end_timestamp",
        "end_latitude",
        "end_longitude",
        "start_latitude",
        "start_longitude",
        "origin_cluster_id",
        "destination_cluster_id",
    }
    missing = required - set(trips.columns)
    if missing:
        raise ValueError(f"Trip table is missing stay fields: {sorted(missing)}")

    ordered = trips.copy()
    ordered["_start"] = _local_time(ordered["start_timestamp"], timezone_name)
    ordered["_end"] = _local_time(ordered["end_timestamp"], timezone_name)
    ordered = ordered.sort_values(["_start", "trip_id"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for index, arrival in ordered.iterrows():
        departure = ordered.iloc[index + 1] if index + 1 < len(ordered) else None
        arrival_time = arrival["_end"]
        cluster_id = str(arrival["destination_cluster_id"])
        status = "RIGHT_CENSORED_RECORD_END"
        reason = "No later recorded trip is available."
        observed_gap = float("nan")
        dwell_minutes = float("nan")
        spatial_gap = float("nan")
        same_session = False
        same_cluster = False
        censored = True
        quality = 0.15
        departure_time = pd.NaT
        departure_trip_id = ""

        if departure is not None:
            departure_time = departure["_start"]
            departure_trip_id = str(departure["trip_id"])
            observed_gap = (
                (departure_time - arrival_time).total_seconds() / 60.0
                if pd.notna(arrival_time) and pd.notna(departure_time)
                else float("nan")
            )
            same_session = str(arrival["session_id"]) == str(departure["session_id"])
            same_cluster = (
                cluster_id != "UNCLUSTERED"
                and cluster_id == str(departure["origin_cluster_id"])
            )
            coordinates = (
                arrival.get("end_latitude"),
                arrival.get("end_longitude"),
                departure.get("start_latitude"),
                departure.get("start_longitude"),
            )
            try:
                values = [float(value) for value in coordinates]
                if all(math.isfinite(value) for value in values):
                    spatial_gap = haversine_m(*values)
            except (TypeError, ValueError):
                spatial_gap = float("nan")

            if not math.isfinite(observed_gap) or observed_gap <= 0:
                status = "INVALID_TIME_ORDER"
                reason = "The next trip does not begin after this trip ended."
                quality = 0.0
            elif cluster_id == "UNCLUSTERED":
                status = "CENSORED_UNCLUSTERED_DESTINATION"
                reason = "The arrival endpoint was not assigned to a stable place."
                quality = 0.2
            elif not same_cluster:
                status = "CENSORED_CHAIN_BREAK"
                reason = (
                    "The next recorded trip begins in a different cluster; "
                    "an intervening movement may be unobserved."
                )
                quality = 0.25
            elif not math.isfinite(spatial_gap) or spatial_gap > maximum_spatial_gap_m:
                status = "CENSORED_SPATIAL_MISMATCH"
                reason = (
                    "The cluster IDs match, but the physical endpoints are too far "
                    "apart for a measured stay."
                )
                quality = 0.3
            elif not same_session:
                status = (
                    "CENSORED_CONTINUITY"
                    if observed_gap <= maximum_continuity_minutes
                    else "RIGHT_CENSORED_LONG_GAP"
                )
                reason = (
                    "The next recording begins in the same area, but a session "
                    "boundary prevents treating the interval as measured dwell."
                )
                quality = 0.65 if status == "CENSORED_CONTINUITY" else 0.4
            elif observed_gap < minimum_activity_minutes:
                status = "MICRO_STOP_BOUNDARY"
                reason = (
                    "The spatial boundary is continuous, but the gap is too short "
                    "to support an activity-duration inference."
                )
                quality = 0.65
            elif observed_gap > maximum_continuity_minutes:
                status = "RIGHT_CENSORED_LONG_GAP"
                reason = "The inter-trip interval exceeds the supported continuity window."
                quality = 0.4
            else:
                status = "MEASURED_STAY"
                reason = (
                    "Consecutive trips share the same session and spatially aligned "
                    "destination/origin area."
                )
                dwell_minutes = observed_gap
                censored = False
                quality = 1.0

        overnight = (
            _crosses_local_hour(arrival_time, departure_time, hour=4)
            if pd.notna(departure_time)
            else False
        )
        stay_id = hashlib.sha256(
            f"{arrival['trip_id']}|{departure_trip_id}|{cluster_id}".encode("utf-8")
        ).hexdigest()[:16]
        rows.append(
            {
                "stay_id": f"S{stay_id}",
                "cluster_id": cluster_id,
                "arrival_trip_id": str(arrival["trip_id"]),
                "departure_trip_id": departure_trip_id,
                "arrival_timestamp": (
                    arrival_time.isoformat() if pd.notna(arrival_time) else ""
                ),
                "departure_timestamp": (
                    departure_time.isoformat() if pd.notna(departure_time) else ""
                ),
                "observed_gap_minutes": observed_gap,
                "dwell_minutes": dwell_minutes,
                "weekday_weekend": (
                    "weekday" if pd.notna(arrival_time) and arrival_time.dayofweek < 5 else "weekend"
                ),
                "time_of_day": _time_of_day(arrival_time),
                "overnight_flag": bool(overnight),
                "same_session": bool(same_session),
                "spatial_gap_m": spatial_gap,
                "stay_status": status,
                "censored_flag": bool(censored),
                "censoring_reason": reason if censored else "",
                "data_quality_score": quality,
            }
        )
    return pd.DataFrame(rows)


def summarize_cluster_stays(stays: pd.DataFrame) -> pd.DataFrame:
    """Summarize measured dwell and censored continuity by place cluster."""
    required = {"cluster_id", "stay_status", "dwell_minutes", "overnight_flag"}
    missing = required - set(stays.columns)
    if missing:
        raise ValueError(f"Stay table is missing fields: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for cluster_id, group in stays.groupby("cluster_id", sort=True):
        measured = group.loc[group["stay_status"].eq("MEASURED_STAY")].copy()
        dwell = pd.to_numeric(measured["dwell_minutes"], errors="coerce").dropna()
        cross_session = group.loc[
            group["stay_status"].isin(
                ["CENSORED_CONTINUITY", "RIGHT_CENSORED_LONG_GAP"]
            )
        ]
        micro_count = int(group["stay_status"].eq("MICRO_STOP_BOUNDARY").sum())
        denominator = max(len(dwell) + micro_count, 1)
        weekday = measured.loc[measured["weekday_weekend"].eq("weekday"), "dwell_minutes"]
        weekend = measured.loc[measured["weekday_weekend"].eq("weekend"), "dwell_minutes"]
        rows.append(
            {
                "cluster_id": str(cluster_id),
                "valid_stay_count": int(len(dwell)),
                "median_dwell_minutes": float(dwell.median()) if len(dwell) else float("nan"),
                "dwell_q25_minutes": float(dwell.quantile(0.25)) if len(dwell) else float("nan"),
                "dwell_q75_minutes": float(dwell.quantile(0.75)) if len(dwell) else float("nan"),
                "mean_dwell_minutes": float(dwell.mean()) if len(dwell) else float("nan"),
                "share_under_5_minutes": float(micro_count / denominator),
                "share_5_to_20_minutes": float(dwell.between(5, 20, inclusive="left").sum() / denominator),
                "share_20_to_60_minutes": float(dwell.between(20, 60, inclusive="left").sum() / denominator),
                "share_1_to_3_hours": float(dwell.between(60, 180, inclusive="left").sum() / denominator),
                "share_over_3_hours": float(dwell.ge(180).sum() / denominator),
                "measured_overnight_stay_count": int(measured["overnight_flag"].sum()),
                "censored_continuity_count": int(len(cross_session)),
                "censored_overnight_association_count": int(cross_session["overnight_flag"].sum()),
                "micro_stop_boundary_count": micro_count,
                "weekday_median_dwell_minutes": float(pd.to_numeric(weekday, errors="coerce").median()) if len(weekday) else float("nan"),
                "weekend_median_dwell_minutes": float(pd.to_numeric(weekend, errors="coerce").median()) if len(weekend) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _collapse(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for raw in values:
        value = str(raw)
        if not value or value in {"nan", "UNCLUSTERED"}:
            continue
        if not result or result[-1] != value:
            result.append(value)
    return result


def build_repeated_trip_chains(
    trips: pd.DataFrame,
    *,
    minimum_occurrences: int = 2,
    timezone_name: str = LOCAL_TIMEZONE,
    service_day_start_hour: int = 4,
    maximum_continuity_m: float = MAX_STAY_SPATIAL_GAP_M,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct continuous service-day chains and summarize repeated sequences.

    A service day begins at 04:00 local time.  A new chain starts when the next
    recorded trip does not begin near the preceding endpoint, preventing an
    unobserved movement from being silently inserted into an activity chain.
    """
    required = {
        "trip_id",
        "session_id",
        "start_timestamp",
        "end_timestamp",
        "origin_cluster_id",
        "destination_cluster_id",
        "start_latitude",
        "start_longitude",
        "end_latitude",
        "end_longitude",
    }
    missing = required - set(trips.columns)
    if missing:
        raise ValueError(f"Trip table is missing chain fields: {sorted(missing)}")

    data = trips.copy()
    data["_start"] = _local_time(data["start_timestamp"], timezone_name)
    data["_end"] = _local_time(data["end_timestamp"], timezone_name)
    data["_service_day"] = (
        data["_start"] - pd.Timedelta(hours=service_day_start_hour)
    ).dt.date.astype(str)
    occurrences: list[dict[str, object]] = []
    for service_day, day_group in data.groupby("_service_day", sort=True):
        day_group = day_group.sort_values(["_start", "trip_id"])
        partitions: list[list[int]] = []
        active: list[int] = []
        previous: Mapping[str, object] | None = None
        for index, row in day_group.iterrows():
            continuous = False
            if previous is not None:
                same_cluster = (
                    str(previous.get("destination_cluster_id")) != "UNCLUSTERED"
                    and str(previous.get("destination_cluster_id"))
                    == str(row.get("origin_cluster_id"))
                )
                try:
                    separation = haversine_m(
                        float(previous.get("end_latitude")),
                        float(previous.get("end_longitude")),
                        float(row.get("start_latitude")),
                        float(row.get("start_longitude")),
                    )
                except (TypeError, ValueError):
                    separation = float("inf")
                continuous = same_cluster and separation <= maximum_continuity_m
            if active and not continuous:
                partitions.append(active)
                active = []
            active.append(index)
            previous = row.to_dict()
        if active:
            partitions.append(active)

        for chain_number, indices in enumerate(partitions, start=1):
            group = data.loc[indices].sort_values(["_start", "trip_id"])
            session_id = ",".join(dict.fromkeys(group["session_id"].astype(str)))
            sequence = _collapse(
                [str(group.iloc[0]["origin_cluster_id"])]
                + group["destination_cluster_id"].astype(str).tolist()
            )
            labels_by_cluster: dict[str, str] = {}
            for row in group.to_dict(orient="records"):
                labels_by_cluster.setdefault(
                    str(row["origin_cluster_id"]),
                    str(row.get("origin_label") or row["origin_cluster_id"]),
                )
                labels_by_cluster.setdefault(
                    str(row["destination_cluster_id"]),
                    str(row.get("destination_label") or row["destination_cluster_id"]),
                )
            labels = [labels_by_cluster.get(cluster, cluster) for cluster in sequence]
            start = group["_start"].min()
            end = group["_end"].max()
            key = "|".join(sequence)
            stop_records: list[dict[str, object]] = []
            ordered = group.reset_index(drop=True)
            for position in range(len(ordered) - 1):
                arrival = ordered.iloc[position]
                departure = ordered.iloc[position + 1]
                dwell_minutes = float(
                    (departure["_start"] - arrival["_end"]).total_seconds() / 60.0
                )
                if dwell_minutes < 0:
                    continue
                stop_records.append(
                    {
                        "position": position + 1,
                        "cluster_id": str(arrival["destination_cluster_id"]),
                        "dwell_minutes": dwell_minutes,
                    }
                )
            stop_durations = [
                float(record["dwell_minutes"]) for record in stop_records
            ]
            occurrences.append(
                {
                    "session_id": str(session_id),
                    "service_day": str(service_day),
                    "chain_number_within_service_day": chain_number,
                    "chain_key": key,
                    "cluster_sequence_json": json.dumps(sequence, separators=(",", ":")),
                    "place_count": len(sequence),
                    "public_chain": " → ".join(labels),
                    "trip_count": int(len(group)),
                    "chain_start_timestamp": start.isoformat(),
                    "chain_end_timestamp": end.isoformat(),
                    "chain_duration_minutes": float((end - start).total_seconds() / 60.0),
                    "intermediate_stop_count": len(stop_records),
                    "median_intermediate_stop_minutes": (
                        float(pd.Series(stop_durations, dtype=float).median())
                        if stop_durations
                        else float("nan")
                    ),
                    "intermediate_stops_json": json.dumps(
                        stop_records, separators=(",", ":")
                    ),
                    "event_date": start.date().isoformat(),
                    "event_week": start.strftime("%G-W%V"),
                    "month": start.strftime("%Y-%m"),
                    "start_hour": start.hour + start.minute / 60.0,
                    "weekday_weekend": "weekday" if start.dayofweek < 5 else "weekend",
                }
            )
    occurrence_frame = pd.DataFrame(occurrences)
    if occurrence_frame.empty:
        return pd.DataFrame(), occurrence_frame

    summary_rows: list[dict[str, object]] = []
    for chain_key, group in occurrence_frame.groupby("chain_key", sort=False):
        if len(group) < minimum_occurrences or int(group["place_count"].max()) < 2:
            continue
        months = sorted(group["month"].unique())
        month_counts = Counter(group["month"])
        first = pd.to_datetime(group["chain_start_timestamp"], utc=True).min()
        last = pd.to_datetime(group["chain_start_timestamp"], utc=True).max()
        span_months = max((last.year - first.year) * 12 + last.month - first.month + 1, 1)
        active_share = len(months) / span_months
        recurrence = (
            "stable across the observation span"
            if len(months) >= 6 and active_share >= 0.6
            else "intermittent"
            if len(months) >= 3
            else "limited-period"
        )
        stop_durations: list[float] = []
        stop_durations_by_cluster: dict[str, list[float]] = {}
        for value in group["intermediate_stops_json"]:
            try:
                decoded = json.loads(str(value))
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded = []
            if not isinstance(decoded, list):
                continue
            for record in decoded:
                if not isinstance(record, Mapping):
                    continue
                try:
                    dwell = float(record.get("dwell_minutes"))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(dwell) or dwell < 0:
                    continue
                cluster_id = str(record.get("cluster_id") or "UNCLUSTERED")
                stop_durations.append(dwell)
                stop_durations_by_cluster.setdefault(cluster_id, []).append(dwell)
        typical_stop_durations = {
            cluster_id: float(pd.Series(values, dtype=float).median())
            for cluster_id, values in sorted(stop_durations_by_cluster.items())
            if cluster_id != "UNCLUSTERED" and values
        }
        chain_id = hashlib.sha256(chain_key.encode("utf-8")).hexdigest()[:12]
        summary_rows.append(
            {
                "chain_id": f"CH{chain_id}",
                "cluster_sequence_json": group["cluster_sequence_json"].iloc[0],
                "public_chain": group["public_chain"].mode().iloc[0],
                "occurrence_count": int(len(group)),
                "unique_days": int(group["event_date"].nunique()),
                "unique_weeks": int(group["event_week"].nunique()),
                "months_visited": int(len(months)),
                "first_observed_date": str(group["event_date"].min()),
                "last_observed_date": str(group["event_date"].max()),
                "typical_start_time": (
                    pd.Timestamp("2000-01-01")
                    + pd.Timedelta(minutes=int(round(group["start_hour"].median() * 60)))
                ).strftime("%-I:%M %p"),
                "median_chain_duration_minutes": float(group["chain_duration_minutes"].median()),
                "median_intermediate_stop_minutes": (
                    float(pd.Series(stop_durations, dtype=float).median())
                    if stop_durations
                    else float("nan")
                ),
                "typical_stop_durations_json": json.dumps(
                    typical_stop_durations, separators=(",", ":")
                ),
                "weekday_share": float(group["weekday_weekend"].eq("weekday").mean()),
                "month_counts_json": json.dumps(dict(sorted(month_counts.items())), separators=(",", ":")),
                "seasonal_pattern": recurrence,
                "stability": recurrence,
                "limitations": "A repeated recorded chain does not confirm the purpose of any stop.",
            }
        )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["occurrence_count", "months_visited", "public_chain"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    return summary, occurrence_frame


def workplace_plausibility(evidence: Mapping[str, object]) -> dict[str, object]:
    """Apply the report's minimum common-sense workplace evidence gate."""
    def number(key: str) -> float:
        try:
            value = float(evidence.get(key, float("nan")))
        except (TypeError, ValueError):
            return float("nan")
        return value

    valid_stays = int(number("valid_stay_count") if math.isfinite(number("valid_stay_count")) else 0)
    median = number("median_dwell_minutes")
    long_share = number("share_over_3_hours")
    weekday_share = number("weekday_share")
    months = int(number("months_visited") if math.isfinite(number("months_visited")) else 0)
    home_connections = int(number("home_connection_count") if math.isfinite(number("home_connection_count")) else 0)
    gates = {
        "enough_valid_stays": valid_stays >= 10,
        "multi_hour_median": math.isfinite(median) and median >= 180.0,
        "substantial_long_stay_share": math.isfinite(long_share) and long_share >= 0.50,
        "weekday_pattern": math.isfinite(weekday_share) and weekday_share >= 0.60,
        "recurs_across_months": months >= 6,
        "connected_to_home": home_connections >= 5,
    }
    supported = all(gates.values())
    return {
        "supported": supported,
        "gates": gates,
        "passed_gate_count": int(sum(gates.values())),
        "required_gate_count": len(gates),
        "reason": (
            "Repeated multi-hour weekday stays satisfy the minimum workplace evidence gate."
            if supported
            else "The location lacks sufficient repeated multi-hour weekday stay evidence for a workplace label."
        ),
    }
