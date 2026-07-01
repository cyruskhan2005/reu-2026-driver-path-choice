"""
Build scalable Driver/Session route timelines from cached Phase 1 outputs.

The Phase 1 county-local FMM trip IDs do not retain source identity and can
change when the input set changes. The mounted source hierarchy is more stable:

    <gps_root>/<collection>/<32-char-id>/<YYMMDD>/<HHMMSS>_gps.jsonl

No project metadata currently proves that the 32-character directory identifies
a human driver. This module therefore treats it as an ``internal_driver_id``
whose source is ``session_dir_parent`` and presents deterministic aliases such
as ``Driver 1`` in user-facing outputs.

All processing uses cached GPS/matched outputs and aggregate filenames. FMM is
not rerun and raw GPS record values are not exported.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .html_assets import embed_local_html_assets
import yaml


AGGREGATED_SUFFIX = "_fid_aggregated.jsonl"
DEFAULT_TIMEZONE = "America/New_York"
DEFAULT_MATCH_TOLERANCE_SECONDS = 300
IDENTITY_SOURCE = "session_dir_parent"
PUBLIC_IDENTITY_TERM = "Driver/Session"


class DriverTimelineError(RuntimeError):
    """Raised when cached Phase 1 data cannot support a timeline build."""


@dataclass(frozen=True)
class CountyInput:
    county: str
    directory: Path
    gps_path: Path
    matched_path: Path
    enriched_network_path: Path | None


@dataclass(frozen=True)
class DriverOutput:
    driver_alias: str
    internal_driver_id: str
    trip_count: int
    observed_month_count: int
    calendar_month_span: int
    first_month: str
    last_month: str
    timeline_path: Path
    monthly_summary_path: Path
    visual_path: Path


@dataclass(frozen=True)
class TimelineBuildResult:
    selected_outputs: tuple[DriverOutput, ...]
    population_index_path: Path
    alias_map_path: Path
    population_overview_path: Path
    identity_audit_path: Path
    gps_root: Path
    matched_paths: tuple[Path, ...]
    gps_paths: tuple[Path, ...]
    enriched_network_paths: tuple[Path, ...]
    discovered_source_trips: int
    mapped_source_trips: int
    skipped_source_trips: int
    requested_top_n: int | None


def _slug(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("_") or "unknown"


def _generated_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_trip_id(county: str, session_id: str, source_trip_id: str) -> str:
    """Return a stable trip identifier independent of county-local FMM IDs."""
    session_digits = re.sub(r"\D", "", session_id)
    if len(session_digits) == 6:
        session_digits = f"20{session_digits}"
    return "_".join(
        (_slug(county).lower(), session_digits or "unknown_date", _slug(source_trip_id))
    )


def parse_fid_sequence(
    value: object,
    *,
    collapse_consecutive: bool = True,
) -> list[int]:
    """
    Parse an FMM ``opath`` value.

    FMM stores one FID per GPS point. Consecutive duplicates are collapsed by
    default so the result represents segment traversal order rather than GPS
    sampling density. Non-matches (FID ``-1``) are omitted.
    """
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    tokens: Iterable[object]
    if isinstance(value, str):
        tokens = re.split(r"[,|]", value)
    elif isinstance(value, Sequence):
        tokens = value
    else:
        tokens = [value]

    fids: list[int] = []
    for token in tokens:
        try:
            fid = int(str(token).strip())
        except (TypeError, ValueError):
            continue
        if fid < 0:
            continue
        if collapse_consecutive and fids and fids[-1] == fid:
            continue
        fids.append(fid)
    return fids


def route_signature(fid_sequence: Sequence[int]) -> str:
    """Return a compact deterministic hash for an ordered FID sequence."""
    payload = "|".join(str(fid) for fid in fid_sequence).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def calendar_month_span(first_month: str, last_month: str) -> int:
    """Return the inclusive number of calendar months between two YYYY-MM values."""
    first = pd.Period(first_month, freq="M")
    last = pd.Period(last_month, freq="M")
    return (last.year - first.year) * 12 + last.month - first.month + 1


def discover_gps_root(
    *,
    explicit_root: str | Path | None = None,
    config_path: str | Path | None = "config.yaml",
    repo_root: str | Path = ".",
) -> Path:
    """Locate the raw/cached session hierarchy without modifying it."""
    candidates: list[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root).expanduser())

    if config_path:
        config = Path(config_path)
        if config.exists():
            try:
                data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
                configured = data.get("gps_root")
                if configured:
                    candidates.append(Path(configured).expanduser())
            except (OSError, yaml.YAMLError):
                pass

    repo = Path(repo_root)
    candidates.extend((repo / "KINGSTON", Path("/Volumes/KINGSTON")))
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    checked = ", ".join(str(path) for path in candidates) or "(none)"
    raise DriverTimelineError(
        "Could not locate the GPS/session root. Pass --gps-root explicitly or "
        f"set gps_root in config.yaml. Checked: {checked}"
    )


def discover_county_inputs(output_dir: str | Path) -> list[CountyInput]:
    """Find county GPS, matched, and enriched-network caches."""
    output_root = Path(output_dir)
    if not output_root.exists():
        raise DriverTimelineError(f"Output directory does not exist: {output_root}")

    inputs: list[CountyInput] = []
    for directory in sorted(path for path in output_root.iterdir() if path.is_dir()):
        gps_files = sorted(directory.glob("*_gps.csv"))
        matched_files = sorted(directory.glob("*_matched.csv"))
        if not gps_files or not matched_files:
            continue

        gps_path = gps_files[0]
        county = gps_path.name[: -len("_gps.csv")]
        matched_by_county = [
            path
            for path in matched_files
            if path.name[: -len("_matched.csv")] == county
        ]
        matched_path = matched_by_county[0] if matched_by_county else matched_files[0]
        network_path = directory / "enriched_network.parquet"
        inputs.append(
            CountyInput(
                county=county,
                directory=directory,
                gps_path=gps_path,
                matched_path=matched_path,
                enriched_network_path=network_path if network_path.exists() else None,
            )
        )

    if not inputs:
        raise DriverTimelineError(
            f"No county *_gps.csv and *_matched.csv pairs found under {output_root}"
        )
    return inputs


def load_matched_trip_data(
    county_input: CountyInput,
    *,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Load one county's cached trip metadata and matched ``opath`` values."""
    try:
        gps = pd.read_csv(
            county_input.gps_path,
            sep=";",
            usecols=["id", "timestamp", "point_idx"],
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DriverTimelineError(
            f"Could not read county GPS cache {county_input.gps_path}: {exc}"
        ) from exc

    required = {"id", "timestamp", "point_idx"}
    if not required.issubset(gps.columns):
        missing = sorted(required - set(gps.columns))
        raise DriverTimelineError(
            f"GPS cache {county_input.gps_path} is missing columns: {missing}"
        )

    gps["id"] = pd.to_numeric(gps["id"], errors="coerce")
    gps["timestamp"] = pd.to_numeric(gps["timestamp"], errors="coerce")
    gps = gps.dropna(subset=["id", "timestamp"])
    gps["id"] = gps["id"].astype("int64")

    stats = (
        gps.groupby("id", sort=False)
        .agg(
            gps_point_count=("point_idx", "size"),
            trip_start_epoch=("timestamp", "min"),
            trip_end_epoch=("timestamp", "max"),
        )
        .reset_index()
        .rename(columns={"id": "matched_trip_id"})
    )
    stats["duration_seconds"] = (
        stats["trip_end_epoch"] - stats["trip_start_epoch"]
    ).clip(lower=0)

    try:
        matched = pd.read_csv(
            county_input.matched_path,
            sep=";",
            usecols=["id", "opath"],
            dtype={"opath": "string"},
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DriverTimelineError(
            f"Could not read matched cache {county_input.matched_path}: {exc}"
        ) from exc

    if not {"id", "opath"}.issubset(matched.columns):
        raise DriverTimelineError(
            f"Matched cache {county_input.matched_path} must contain id and opath"
        )
    matched["id"] = pd.to_numeric(matched["id"], errors="coerce")
    matched = matched.dropna(subset=["id"]).copy()
    matched["id"] = matched["id"].astype("int64")

    # FMM can emit repeated empty fallback rows for one ID. Prefer a usable
    # route, then the longest route, to keep the join deterministic.
    matched["_opath_present"] = matched["opath"].notna() & matched["opath"].str.len().gt(0)
    matched["_opath_length"] = matched["opath"].str.len().fillna(-1)
    matched = (
        matched.sort_values(
            ["id", "_opath_present", "_opath_length"],
            ascending=[True, False, False],
        )
        .drop_duplicates("id", keep="first")
        .drop(columns=["_opath_present", "_opath_length"])
        .rename(columns={"id": "matched_trip_id"})
    )

    trips = stats.merge(matched, on="matched_trip_id", how="inner", validate="one_to_one")
    trips["county"] = county_input.county
    trips["gps_path"] = str(county_input.gps_path)
    trips["matched_path"] = str(county_input.matched_path)
    trips["enriched_network_path"] = (
        str(county_input.enriched_network_path)
        if county_input.enriched_network_path
        else pd.NA
    )

    utc_start = pd.to_datetime(trips["trip_start_epoch"], unit="s", utc=True)
    local_start = utc_start.dt.tz_convert(timezone_name)
    trips["trip_start_time"] = local_start.map(lambda value: value.isoformat())
    trips["trip_month"] = local_start.dt.strftime("%Y-%m")
    trips["_local_start_naive"] = local_start.dt.tz_localize(None)
    return trips


def _parse_session_date(session_id: str) -> datetime | None:
    digits = re.sub(r"\D", "", session_id)
    for fmt, length in (("%y%m%d", 6), ("%Y%m%d", 8)):
        if len(digits) == length:
            try:
                return datetime.strptime(digits, fmt)
            except ValueError:
                return None
    return None


def _identity_directories(gps_root: Path) -> list[tuple[Path, Path, list[Path]]]:
    """Return collection, identity, and valid date-session directories."""
    results: list[tuple[Path, Path, list[Path]]] = []
    try:
        collections = sorted(path for path in gps_root.iterdir() if path.is_dir())
    except OSError as exc:
        raise DriverTimelineError(f"Could not inspect GPS root {gps_root}: {exc}") from exc

    for collection in collections:
        if collection.name.startswith(".") or collection.name == "System Volume Information":
            continue
        try:
            identities = sorted(path for path in collection.iterdir() if path.is_dir())
        except OSError:
            continue
        for identity in identities:
            try:
                sessions = sorted(
                    path
                    for path in identity.iterdir()
                    if path.is_dir() and _parse_session_date(path.name) is not None
                )
            except OSError:
                continue
            if sessions:
                results.append((collection, identity, sessions))
    return results


def discover_source_trips(
    gps_root: str | Path,
    counties: Sequence[str],
) -> pd.DataFrame:
    """
    Discover cached trip aggregate files and infer stable internal IDs.

    Aggregate contents are not read. Filenames provide county/source trip;
    parent directories provide date session and inferred grouping identity.
    """
    root = Path(gps_root)
    county_names = sorted(counties, key=len, reverse=True)
    rows: list[dict[str, object]] = []

    for collection, identity, sessions in _identity_directories(root):
        for session_dir in sessions:
            session_date = _parse_session_date(session_dir.name)
            if session_date is None:
                continue
            try:
                aggregate_paths = sorted(session_dir.glob(f"*{AGGREGATED_SUFFIX}"))
            except OSError:
                continue
            for aggregate_path in aggregate_paths:
                if aggregate_path.name.startswith("._"):
                    continue
                county = next(
                    (
                        name
                        for name in county_names
                        if aggregate_path.name.startswith(f"{name}_")
                    ),
                    None,
                )
                if county is None:
                    continue

                prefix_length = len(county) + 1
                source_trip_id = aggregate_path.name[
                    prefix_length : -len(AGGREGATED_SUFFIX)
                ]
                time_match = re.search(r"(\d{6})", source_trip_id)
                expected_start = None
                if time_match:
                    try:
                        expected_time = datetime.strptime(
                            time_match.group(1), "%H%M%S"
                        ).time()
                        expected_start = datetime.combine(
                            session_date.date(), expected_time
                        )
                    except ValueError:
                        pass

                rows.append(
                    {
                        "internal_driver_id": identity.name,
                        # Backward-compatible research column; this remains the
                        # internal pseudonymous identifier, never the alias.
                        "driver_id": identity.name,
                        "driver_id_source": IDENTITY_SOURCE,
                        "collection_id": collection.name,
                        "session_id": session_dir.name,
                        "session_dir": str(session_dir),
                        "source_trip_id": source_trip_id,
                        "source_gps_path": str(
                            session_dir / f"{source_trip_id}_gps.jsonl"
                        ),
                        "aggregate_path": str(aggregate_path),
                        "county": county,
                        "_expected_start_naive": expected_start,
                    }
                )

    if not rows:
        raise DriverTimelineError(
            f"No *{AGGREGATED_SUFFIX} files found under valid session folders in {root}"
        )
    return pd.DataFrame(rows)


def map_source_trips_to_matches(
    source_trips: pd.DataFrame,
    matched_trips: pd.DataFrame,
    *,
    tolerance_seconds: int = DEFAULT_MATCH_TOLERANCE_SECONDS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Join source identity to county-local FMM IDs by nearest trip start time."""
    mapped_rows: list[dict[str, object]] = []
    source = source_trips.dropna(subset=["_expected_start_naive"]).copy()

    for county, county_source in source.groupby("county", sort=False):
        county_matches = matched_trips.loc[matched_trips["county"] == county].copy()
        if county_matches.empty:
            continue

        county_matches = county_matches.sort_values("_local_start_naive")
        starts_ns = (
            county_matches["_local_start_naive"]
            .to_numpy(dtype="datetime64[ns]")
            .astype("int64")
        )
        match_ids = county_matches["matched_trip_id"].to_numpy()

        for record in county_source.to_dict(orient="records"):
            expected_ns = pd.Timestamp(record["_expected_start_naive"]).value
            position = int(np.searchsorted(starts_ns, expected_ns))
            candidates = {
                candidate
                for candidate in (position - 1, position)
                if 0 <= candidate < len(starts_ns)
            }
            if not candidates:
                continue
            best = min(
                candidates,
                key=lambda candidate: abs(int(starts_ns[candidate]) - expected_ns),
            )
            delta_seconds = abs(int(starts_ns[best]) - expected_ns) / 1_000_000_000
            if delta_seconds > tolerance_seconds:
                continue

            record["matched_trip_id"] = int(match_ids[best])
            record["start_time_match_delta_seconds"] = float(delta_seconds)
            mapped_rows.append(record)

    if not mapped_rows:
        raise DriverTimelineError(
            "No source trips could be joined to cached county GPS trips. "
            "Check --gps-root, timezone, and match tolerance."
        )

    mapped = pd.DataFrame(mapped_rows).sort_values("start_time_match_delta_seconds")
    before_dedup = len(mapped)
    mapped = mapped.drop_duplicates(
        subset=["county", "matched_trip_id"], keep="first"
    )
    duplicate_assignments = before_dedup - len(mapped)

    joined = mapped.merge(
        matched_trips,
        on=["county", "matched_trip_id"],
        how="inner",
        validate="one_to_one",
    )
    usable = joined["opath"].notna() & joined["opath"].astype("string").str.len().gt(0)
    failed_or_empty = int((~usable).sum())
    joined = joined.loc[usable].copy()

    stats = {
        "discovered": int(len(source_trips)),
        "mapped_before_filter": int(before_dedup),
        "duplicate_assignments": int(duplicate_assignments),
        "failed_or_empty_opath": failed_or_empty,
        "usable": int(len(joined)),
        "skipped": int(len(source_trips) - len(joined)),
    }
    return joined, stats


def build_population_index(mapped_trips: pd.DataFrame) -> pd.DataFrame:
    """
    Rank all inferred Driver/Session IDs and assign deterministic aliases.

    Ranking is descending by usable trip count, observed month count, and
    inclusive date coverage span, with internal ID as a stable final tiebreaker.
    FID uniqueness uses ``(county, fid)`` pairs because county networks have
    separate FID namespaces.
    """
    rows: list[dict[str, object]] = []
    for internal_id, group in mapped_trips.groupby("internal_driver_id", sort=False):
        months = sorted(group["trip_month"].astype(str).unique())
        if not months:
            continue
        unique_fids: set[tuple[str, int]] = set()
        usable_trip_count = 0
        for row in group.itertuples(index=False):
            fids = parse_fid_sequence(row.opath)
            if not fids:
                continue
            usable_trip_count += 1
            unique_fids.update((str(row.county), fid) for fid in fids)
        if usable_trip_count == 0:
            continue
        rows.append(
            {
                "internal_driver_id": str(internal_id),
                "driver_id_source": str(group["driver_id_source"].iloc[0]),
                "usable_trip_count": usable_trip_count,
                "observed_month_count": len(months),
                "calendar_month_span": calendar_month_span(months[0], months[-1]),
                "first_month": months[0],
                "last_month": months[-1],
                "total_unique_fids": len(unique_fids),
            }
        )

    population = pd.DataFrame(rows)
    if population.empty:
        raise DriverTimelineError("No inferred Driver/Session IDs have usable routes")
    population = population.sort_values(
        [
            "usable_trip_count",
            "observed_month_count",
            "calendar_month_span",
            "internal_driver_id",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    population.insert(
        0,
        "driver_alias",
        [f"Driver {index}" for index in range(1, len(population) + 1)],
    )
    population["selected_for_top_n"] = False
    return population


def build_alias_map(population_index: pd.DataFrame) -> pd.DataFrame:
    """Return the public-alias to internal-ID traceability table."""
    return population_index.rename(
        columns={"usable_trip_count": "trip_count"}
    )[
        [
            "driver_alias",
            "internal_driver_id",
            "driver_id_source",
            "trip_count",
            "observed_month_count",
            "first_month",
            "last_month",
        ]
    ].copy()


def select_population(
    population_index: pd.DataFrame,
    *,
    driver: str = "auto",
    top_n: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one ID or the top N while preserving deterministic global aliases."""
    if top_n is not None:
        if top_n < 1:
            raise DriverTimelineError("--top-n must be at least 1")
        selected = population_index.head(top_n).copy()
    else:
        if driver == "auto":
            selected = population_index.head(1).copy()
        else:
            exact = population_index.loc[
                population_index["internal_driver_id"] == driver
            ]
            if exact.empty:
                prefix = population_index.loc[
                    population_index["internal_driver_id"].str.startswith(driver)
                ]
                if len(prefix) != 1:
                    available = ", ".join(
                        f"{row.driver_alias}={row.internal_driver_id}"
                        for row in population_index.itertuples(index=False)
                    )
                    raise DriverTimelineError(
                        f"Internal ID {driver!r} was not found or was ambiguous. "
                        f"Available: {available}"
                    )
                exact = prefix
            selected = exact.copy()

    updated = population_index.copy()
    updated["selected_for_top_n"] = updated["internal_driver_id"].isin(
        selected["internal_driver_id"]
    )
    return selected.reset_index(drop=True), updated


def build_driver_trip_table(
    mapped_trips: pd.DataFrame,
    internal_driver_id: str,
    driver_alias: str,
) -> pd.DataFrame:
    """Build the normalized one-row-per-county-trip table for one grouping ID."""
    selected = mapped_trips.loc[
        mapped_trips["internal_driver_id"] == internal_driver_id
    ].copy()
    if selected.empty:
        raise DriverTimelineError(
            f"No usable trips found for internal ID {internal_driver_id}"
        )

    rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        point_fids = parse_fid_sequence(row.opath, collapse_consecutive=False)
        traversal_fids = parse_fid_sequence(row.opath, collapse_consecutive=True)
        if not traversal_fids:
            continue
        rows.append(
            {
                "driver_alias": driver_alias,
                "internal_driver_id": internal_driver_id,
                # Backward compatibility: driver_id remains internal, not public.
                "driver_id": internal_driver_id,
                "driver_id_source": str(row.driver_id_source),
                "trip_id": normalize_trip_id(
                    str(row.county), str(row.session_id), str(row.source_trip_id)
                ),
                "county": str(row.county),
                "trip_start_time": str(row.trip_start_time),
                "trip_month": str(row.trip_month),
                "gps_point_count": int(row.gps_point_count),
                "matched_gps_point_count": len(point_fids),
                "matched_fid_count": len(traversal_fids),
                "unique_fid_count": len(set(traversal_fids)),
                "duration_seconds": int(row.duration_seconds),
                "fid_sequence": "|".join(str(fid) for fid in traversal_fids),
                "route_signature": route_signature(traversal_fids),
                "start_fid": traversal_fids[0],
                "end_fid": traversal_fids[-1],
                "matched_trip_id": int(row.matched_trip_id),
                "source_trip_id": str(row.source_trip_id),
                "collection_id": str(row.collection_id),
                "session_id": str(row.session_id),
                "session_dir": str(row.session_dir),
                "source_gps_path": str(row.source_gps_path),
                "aggregate_path": str(row.aggregate_path),
                "matched_path": str(row.matched_path),
                "gps_path": str(row.gps_path),
                "enriched_network_path": row.enriched_network_path,
                "start_time_match_delta_seconds": float(
                    row.start_time_match_delta_seconds
                ),
            }
        )

    timeline = pd.DataFrame(rows)
    if timeline.empty:
        raise DriverTimelineError(
            f"Internal ID {internal_driver_id} has no matched FID sequences"
        )
    return timeline.sort_values(
        ["trip_start_time", "county", "trip_id"]
    ).reset_index(drop=True)


def _most_common(values: Iterable[object]) -> object:
    counter = Counter(value for value in values if pd.notna(value))
    if not counter:
        return pd.NA
    return sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def build_monthly_summary(timeline: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate one timeline and insert zero rows for missing calendar months.

    ``total_unique_fids`` is the monthly FID union. ``observed_month`` separates
    months with real trips from zero-filled calendar months used by plots.
    """
    if timeline.empty:
        raise DriverTimelineError("Cannot summarize an empty timeline")

    grouped = {str(month): group for month, group in timeline.groupby("trip_month")}
    first_month = min(grouped)
    last_month = max(grouped)
    months = pd.period_range(first_month, last_month, freq="M").astype(str)

    rows: list[dict[str, object]] = []
    for month in months:
        group = grouped.get(month)
        observed = group is not None and not group.empty
        monthly_fids: set[int] = set()
        if observed:
            for sequence in group["fid_sequence"]:
                monthly_fids.update(parse_fid_sequence(sequence))
        rows.append(
            {
                "driver_alias": str(timeline["driver_alias"].iloc[0])
                if "driver_alias" in timeline
                else pd.NA,
                "internal_driver_id": str(timeline["internal_driver_id"].iloc[0])
                if "internal_driver_id" in timeline
                else str(timeline["driver_id"].iloc[0]),
                "driver_id": str(timeline["driver_id"].iloc[0]),
                "driver_id_source": str(timeline["driver_id_source"].iloc[0])
                if "driver_id_source" in timeline
                else IDENTITY_SOURCE,
                "trip_month": month,
                "observed_month": bool(observed),
                "trip_count": int(len(group)) if observed else 0,
                "total_gps_points": int(group["gps_point_count"].sum())
                if observed
                else 0,
                "total_matched_fids": int(group["matched_fid_count"].sum())
                if observed
                else 0,
                "total_unique_fids": len(monthly_fids),
                "unique_route_signatures": int(
                    group["route_signature"].nunique(dropna=True)
                )
                if observed
                else 0,
                "most_common_start_fid": _most_common(group["start_fid"])
                if observed
                else pd.NA,
                "most_common_end_fid": _most_common(group["end_fid"])
                if observed
                else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _svg_line_chart(
    labels: Sequence[str],
    values: Sequence[int],
    *,
    title: str,
    y_axis_title: str,
    color: str,
) -> str:
    """Render a spacious presentation-oriented SVG monthly line chart."""
    width, height = 1200, 470
    left, right, top, bottom = 105, 42, 62, 122
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(max(values, default=0), 1)
    count = max(len(values), 1)

    def x_position(index: int) -> float:
        return left + (
            plot_width / 2 if count == 1 else index * plot_width / (count - 1)
        )

    def y_position(value: float) -> float:
        return top + plot_height - (value / maximum) * plot_height

    points = " ".join(
        f"{x_position(index):.1f},{y_position(int(value)):.1f}"
        for index, value in enumerate(values)
    )
    grid: list[str] = []
    for tick in range(6):
        tick_value = maximum * tick / 5
        y = y_position(tick_value)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
            f'<text x="{left-14}" y="{y+5:.1f}" text-anchor="end" '
            f'font-size="13" fill="#64748b">{tick_value:.0f}</text>'
        )

    label_step = max(1, int(np.ceil(len(labels) / 16)))
    label_indices = list(range(0, len(labels), label_step))
    if labels and label_indices[-1] != len(labels) - 1:
        if len(labels) - 1 - label_indices[-1] < max(2, label_step):
            label_indices[-1] = len(labels) - 1
        else:
            label_indices.append(len(labels) - 1)

    x_labels = []
    for index in sorted(set(label_indices)):
        x = x_position(index)
        x_labels.append(
            f'<text x="{x:.1f}" y="{height-bottom+28}" text-anchor="end" '
            f'transform="rotate(-42 {x:.1f} {height-bottom+28})" '
            f'font-size="12" fill="#475569">{html.escape(str(labels[index]))}</text>'
        )

    circles = []
    for index, (label, value) in enumerate(zip(labels, values)):
        circles.append(
            f'<circle cx="{x_position(index):.1f}" cy="{y_position(int(value)):.1f}" '
            f'r="4" fill="{color}" stroke="white" stroke-width="1.5">'
            f'<title>{html.escape(str(label))}: {int(value):,}</title></circle>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">'
        f'<text x="{left}" y="31" font-size="22" font-weight="700" '
        f'fill="#0f172a">{html.escape(title)}</text>'
        + "".join(grid)
        + f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" '
        'stroke="#94a3b8" stroke-width="1.5"/>'
        + f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
        f'y2="{height-bottom}" stroke="#94a3b8" stroke-width="1.5"/>'
        + (
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="3.5" stroke-linejoin="round" stroke-linecap="round"/>'
            if points
            else ""
        )
        + "".join(circles)
        + "".join(x_labels)
        + f'<text x="{left + plot_width/2:.1f}" y="{height-10}" '
        'text-anchor="middle" font-size="14" font-weight="600" fill="#334155">'
        'Month</text>'
        + f'<text x="24" y="{top + plot_height/2:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 24 {top + plot_height/2:.1f})" '
        f'font-size="14" font-weight="600" fill="#334155">'
        f'{html.escape(y_axis_title)}</text>'
        + "</svg>"
    )


def write_timeline_visual(
    monthly_summary: pd.DataFrame,
    output_path: str | Path,
    *,
    driver_alias: str,
    internal_driver_id: str,
    driver_id_source: str,
    source_files: Sequence[str],
) -> Path:
    """Write a presentation-ready HTML Driver/Session timeline."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = monthly_summary["trip_month"].astype(str).tolist()
    trip_counts = monthly_summary["trip_count"].astype(int).tolist()
    unique_fids = monthly_summary["total_unique_fids"].astype(int).tolist()

    trip_chart = _svg_line_chart(
        labels,
        trip_counts,
        title="Trips by month",
        y_axis_title="Number of trips",
        color="#2563eb",
    )
    fid_chart = _svg_line_chart(
        labels,
        unique_fids,
        title="Distinct matched road segments by month",
        y_axis_title="Distinct matched FIDs",
        color="#ea580c",
    )
    first_month = labels[0]
    last_month = labels[-1]
    total_trips = int(sum(trip_counts))
    observed_months = int(monthly_summary["observed_month"].sum())
    month_span = len(monthly_summary)
    source_items = "".join(
        f"<li><code>{html.escape(str(path))}</code></li>"
        for path in sorted(set(source_files))
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(driver_alias)} Route Activity Timeline</title>
  <style>
    :root {{ color-scheme: light; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #f1f5f9; color: #0f172a;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 46px 34px 64px; }}
    h1 {{ margin: 0; font-size: 36px; letter-spacing: -0.025em; }}
    .subtitle {{ margin-top: 10px; color: #475569; font-size: 17px; line-height: 1.6; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr));
                     gap: 16px; margin: 30px 0; }}
    .summary-card, .chart-card, .note, .technical {{
      background: white; border: 1px solid #dbe4ee; border-radius: 15px;
      box-shadow: 0 6px 22px rgba(15, 23, 42, .055);
    }}
    .summary-card {{ padding: 20px; min-height: 112px; }}
    .summary-label {{ color: #64748b; font-size: 12px; font-weight: 700;
                      letter-spacing: .08em; text-transform: uppercase; }}
    .summary-value {{ margin-top: 11px; font-size: 22px; font-weight: 750;
                      line-height: 1.25; }}
    .chart-card {{ padding: 24px 26px 30px; margin-top: 28px; min-height: 500px; }}
    .note {{ margin-top: 28px; padding: 20px 24px; color: #334155;
             font-size: 15px; line-height: 1.65; border-left: 5px solid #2563eb; }}
    .technical {{ margin-top: 34px; padding: 22px 25px; color: #475569; }}
    .technical summary {{ cursor: pointer; color: #1e293b; font-weight: 700; }}
    .technical-content {{ margin-top: 18px; font-size: 13px; line-height: 1.7; }}
    .technical ul {{ padding-left: 22px; }}
    code {{ overflow-wrap: anywhere; }}
    svg {{ display: block; width: 100%; height: auto; min-height: 430px; }}
    @media (max-width: 900px) {{
      .summary-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      main {{ padding: 30px 18px 45px; }}
      h1 {{ font-size: 29px; }}
      .chart-card {{ padding: 14px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>{html.escape(driver_alias)} Route Activity Timeline</h1>
    <div class="subtitle">
      Monthly route activity for one inferred Driver/Session grouping.
      Calendar months without observations are shown as zero.
    </div>
  </header>

  <section class="summary-grid" aria-label="Timeline summary">
    <div class="summary-card"><div class="summary-label">Driver alias</div>
      <div class="summary-value">{html.escape(driver_alias)}</div></div>
    <div class="summary-card"><div class="summary-label">Internal ID source</div>
      <div class="summary-value">{html.escape(driver_id_source)}</div></div>
    <div class="summary-card"><div class="summary-label">Trips</div>
      <div class="summary-value">{total_trips:,}</div></div>
    <div class="summary-card"><div class="summary-label">Observed months</div>
      <div class="summary-value">{observed_months} of {month_span}</div></div>
    <div class="summary-card"><div class="summary-label">Coverage</div>
      <div class="summary-value">{html.escape(first_month)} – {html.escape(last_month)}</div></div>
  </section>

  <section class="note">
    This timeline summarizes monthly route activity for one inferred driver/session.
    It is intended as the foundation for monthly attributed graph construction.
  </section>

  <section class="chart-card">{trip_chart}</section>
  <section class="chart-card">{fid_chart}</section>

  <details class="technical">
    <summary>Technical details</summary>
    <div class="technical-content">
      <div><strong>Internal pseudonymous ID:</strong>
        <code>{html.escape(internal_driver_id)}</code></div>
      <div><strong>ID source:</strong> {html.escape(driver_id_source)}</div>
      <div><strong>Generated:</strong> {html.escape(_generated_at())}</div>
      <div><strong>Source files used:</strong></div>
      <ul>{source_items}</ul>
    </div>
  </details>
</main>
</body>
</html>
"""
    output.write_text(
        embed_local_html_assets(document, output.parent),
        encoding="utf-8",
    )
    return output


def write_population_overview(
    selected: pd.DataFrame,
    driver_outputs: Sequence[DriverOutput],
    output_path: str | Path,
    *,
    requested_top_n: int | None,
) -> Path:
    """Write a clean alias-only index for the selected Driver/Session timelines."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_by_id = {item.internal_driver_id: item for item in driver_outputs}
    rows = []
    for record in selected.itertuples(index=False):
        driver_output = output_by_id[str(record.internal_driver_id)]
        relative_link = driver_output.visual_path.name
        rows.append(
            "<tr>"
            f'<td><a href="{html.escape(relative_link)}">'
            f"{html.escape(str(record.driver_alias))}</a></td>"
            f"<td>{int(record.usable_trip_count):,}</td>"
            f"<td>{int(record.observed_month_count)}</td>"
            f"<td>{int(record.calendar_month_span)}</td>"
            f"<td>{html.escape(str(record.first_month))} – "
            f"{html.escape(str(record.last_month))}</td>"
            f"<td>{int(record.total_unique_fids):,}</td>"
            "</tr>"
        )

    actual_n = len(selected)
    requested_note = (
        f" Requested top N: {requested_top_n}; {actual_n} available."
        if requested_top_n is not None and requested_top_n != actual_n
        else ""
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Top {actual_n} Driver/Session Timeline Dataset</title>
  <style>
    body {{ margin: 0; background: #f1f5f9; color: #0f172a;
            font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 46px 30px 64px; }}
    h1 {{ margin: 0 0 12px; font-size: 34px; }}
    p {{ color: #475569; line-height: 1.65; max-width: 920px; }}
    .table-card {{ margin-top: 28px; overflow-x: auto; background: white;
                   border: 1px solid #dbe4ee; border-radius: 15px;
                   box-shadow: 0 6px 22px rgba(15,23,42,.055); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{ background: #eaf1f8; color: #334155; text-align: left;
          font-size: 12px; letter-spacing: .06em; text-transform: uppercase; }}
    th, td {{ padding: 15px 17px; border-bottom: 1px solid #e2e8f0; }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    tbody tr:hover {{ background: #f8fafc; }}
    a {{ color: #1d4ed8; font-weight: 700; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
<main>
  <h1>Top {actual_n} Driver/Session Timeline Dataset</h1>
  <p>
    Ranked Driver/Session groupings selected from cached Phase 1 matched trips.
    Public aliases are used here; internal pseudonymous IDs remain in the alias
    map and technical details.{html.escape(requested_note)}
  </p>
  <p>
    These timelines are the input foundation for later monthly attributed graph
    networks. No graph comparison or driver-choice change metric is calculated here.
  </p>
  <div class="table-card">
    <table>
      <thead><tr>
        <th>Alias</th><th>Trips</th><th>Observed months</th>
        <th>Calendar span</th><th>Coverage</th><th>Unique county/FIDs</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</main>
</body>
</html>
"""
    output.write_text(
        embed_local_html_assets(document, output.parent),
        encoding="utf-8",
    )
    return output


def _sample_json_field_names(paths: Sequence[Path], line_limit: int = 4) -> set[str]:
    """Read only JSON object keys from a small raw-file sample."""
    keys: set[str] = set()
    for path in paths:
        try:
            with path.open(encoding="utf-8", errors="ignore") as handle:
                for _ in range(line_limit):
                    line = handle.readline()
                    if not line:
                        break
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    stack = [value]
                    while stack:
                        item = stack.pop()
                        if isinstance(item, dict):
                            for key, nested in item.items():
                                keys.add(str(key))
                                if isinstance(nested, (dict, list)):
                                    stack.append(nested)
                        elif isinstance(item, list):
                            stack.extend(item)
        except OSError:
            continue
    return keys


def write_identity_audit(gps_root: str | Path, output_path: str | Path) -> Path:
    """Write a structural identity audit without exporting raw record values."""
    root = Path(gps_root)
    identities = _identity_directories(root)
    sample_by_type: dict[str, list[Path]] = {
        "gps": [],
        "aggregated": [],
        "acc": [],
        "obd": [],
        "supported_pids": [],
    }
    identity_rows = []
    for collection, identity, sessions in identities:
        aggregate_count = 0
        gps_count = 0
        for session in sessions:
            gps_files = [
                path for path in session.glob("*_gps.jsonl")
                if not path.name.startswith("._")
            ]
            aggregate_files = [
                path for path in session.glob(f"*{AGGREGATED_SUFFIX}")
                if not path.name.startswith("._")
            ]
            gps_count += len(gps_files)
            aggregate_count += len(aggregate_files)
            if gps_files and len(sample_by_type["gps"]) < 6:
                sample_by_type["gps"].append(gps_files[0])
            if aggregate_files and len(sample_by_type["aggregated"]) < 6:
                sample_by_type["aggregated"].append(aggregate_files[0])
            for category, pattern in (("acc", "*_acc.jsonl"), ("obd", "*_obd.jsonl")):
                candidates = [
                    path for path in session.glob(pattern)
                    if not path.name.startswith("._")
                ]
                if candidates and len(sample_by_type[category]) < 6:
                    sample_by_type[category].append(candidates[0])
        supported = sorted(identity.glob("supported_pids*.json"))
        sample_by_type["supported_pids"].extend(
            supported[: max(0, 6 - len(sample_by_type["supported_pids"]))]
        )
        months = sorted(
            {
                f"20{session.name[:2]}-{session.name[2:4]}"
                for session in sessions
                if len(session.name) == 6 and session.name.isdigit()
            }
        )
        identity_rows.append(
            {
                "collection": collection.name,
                "internal_id": identity.name,
                "session_count": len(sessions),
                "gps_count": gps_count,
                "aggregate_count": aggregate_count,
                "first_month": months[0] if months else "unknown",
                "last_month": months[-1] if months else "unknown",
            }
        )

    fields = {
        category: sorted(_sample_json_field_names(paths))
        for category, paths in sample_by_type.items()
    }
    identity_terms = (
        "driver", "vehicle", "device", "user", "participant", "subject",
        "session", "uuid", "hash", "account", "trip",
    )
    explicit_identity_fields = sorted(
        {
            field
            for names in fields.values()
            for field in names
            if any(term in field.lower() for term in identity_terms)
        }
    )
    identity_table = "\n".join(
        "| {collection} | `{internal_id}` | {session_count} | {gps_count} | "
        "{aggregate_count} | {first_month}–{last_month} |".format(**row)
        for row in identity_rows
    )
    field_sections = "\n".join(
        f"- `{category}` sample field names: "
        + (", ".join(f"`{field}`" for field in names) if names else "(none parsed)")
        for category, names in fields.items()
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""# Driver identity audit

Generated: {_generated_at()}

## Inference source

The Phase 2 grouping ID is the parent directory of each dated session folder:

```text
<gps_root>/<collection folder>/<32-character hexadecimal ID>/<YYMMDD>/<trip>_gps.jsonl
```

Example structural interpretation:

- collection folder: cohort/batch-like label such as `1003 1004`
- internal ID: 32-character hexadecimal directory name
- session folder: six-digit calendar date
- trip file: six-digit time token plus sensor/file suffix

The code exports the 32-character directory name as `internal_driver_id` with
`driver_id_source={IDENTITY_SOURCE}`.

## Structural stability

| Collection | Internal pseudonymous ID | Dated sessions | GPS files | Aggregates | Month range |
|---|---|---:|---:|---:|---|
{identity_table}

The IDs are stable across many trip files and date folders, so they are useful
longitudinal grouping keys.

## Schema-only inspection

Only field names from a small file sample were inspected; raw coordinates,
sensor values, and individual records were not copied into this audit.

{field_sections}

Explicit identity-related fields found: {
    ", ".join(f"`{field}`" for field in explicit_identity_fields)
    if explicit_identity_fields else "**none**"
}.

The master GPS cache contains `gps_path`, `session_dir`, coordinates, timestamps,
and county, but no explicit driver, person, vehicle, device, or participant ID.
The county matched files contain only county-local trip ID and `opath`.

## Interpretation

- **Explicit human driver ID available:** No.
- **Stable across trips:** Yes; each hash spans many dated sessions and trips.
- **Proven entity type:** No. The directory could represent a logger/device,
  vehicle, account, collection subject, or human participant.
- **Additional clue:** OBD capability logs (`supported_pids*.json`) live under
  the same hash, which suggests a persistent data-collection namespace but does
  not distinguish vehicle, device, or person.
- **Confidence as a stable longitudinal grouping key:** High.
- **Confidence that it identifies a real human driver:** Low.
- **Overall terminology confidence:** Medium for `inferred_subject_id` or
  pseudonymous Driver/Session grouping.

## Recommendation

Use **Driver/Session** in presentation-facing text and `internal_driver_id` in
technical outputs. Treat it as an inferred pseudonymous subject identifier,
not a guaranteed human driver identity.

Unless confirmed by project metadata, the inferred ID should be treated as a
pseudonymous driver/session identifier rather than guaranteed human driver identity.

## Mentor questions

1. What generated the 32-character directory name: participant enrollment,
   vehicle/logger setup, mobile device/account, or another process?
2. Do collection labels such as `1003 1004` encode participant, vehicle, or
   deployment dates?
3. Can a single human driver use multiple hashes, or can multiple drivers share
   one hash/vehicle?
4. Is there an external participant-to-device/vehicle crosswalk that can be
   joined without exposing personally identifiable information?
""",
        encoding="utf-8",
    )
    return output


def load_road_attributes(
    network_path: str | Path,
    *,
    fids: Iterable[int] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load cached FID attributes for later monthly graph construction."""
    import geopandas as gpd

    network = gpd.read_parquet(network_path)
    frame = pd.DataFrame(network.reset_index())
    if "fid" not in frame.columns:
        raise DriverTimelineError(
            f"Enriched network has no fid column or fid index: {network_path}"
        )
    if fids is not None:
        wanted = {int(fid) for fid in fids}
        frame = frame.loc[frame["fid"].astype(int).isin(wanted)]
    if columns is not None:
        selected = ["fid"] + [
            column for column in columns if column != "fid" and column in frame.columns
        ]
        frame = frame[selected]
    elif "geometry" in frame.columns:
        frame = frame.drop(columns="geometry")
    return frame


def export_driver_timeline(
    timeline: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    *,
    output_root: str | Path,
    driver_alias: str,
    internal_driver_id: str,
) -> DriverOutput:
    """Write alias-named CSV and HTML outputs for one Driver/Session grouping."""
    root = Path(output_root)
    timeline_dir = root / "phase2" / "driver_timelines"
    visual_dir = root / "phase2" / "visuals"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    alias_slug = _slug(driver_alias).lower()
    timeline_path = timeline_dir / f"{alias_slug}_timeline.csv"
    summary_path = timeline_dir / f"{alias_slug}_monthly_summary.csv"
    visual_path = visual_dir / f"{alias_slug}_activity_over_time.html"

    timeline.to_csv(timeline_path, index=False)
    monthly_summary.to_csv(summary_path, index=False)
    source_files = (
        timeline["gps_path"].dropna().astype(str).unique().tolist()
        + timeline["matched_path"].dropna().astype(str).unique().tolist()
        + timeline["enriched_network_path"].dropna().astype(str).unique().tolist()
    )
    write_timeline_visual(
        monthly_summary,
        visual_path,
        driver_alias=driver_alias,
        internal_driver_id=internal_driver_id,
        driver_id_source=str(timeline["driver_id_source"].iloc[0]),
        source_files=source_files,
    )

    observed = monthly_summary.loc[monthly_summary["observed_month"]]
    return DriverOutput(
        driver_alias=driver_alias,
        internal_driver_id=internal_driver_id,
        trip_count=len(timeline),
        observed_month_count=len(observed),
        calendar_month_span=len(monthly_summary),
        first_month=str(monthly_summary["trip_month"].iloc[0]),
        last_month=str(monthly_summary["trip_month"].iloc[-1]),
        timeline_path=timeline_path.resolve(),
        monthly_summary_path=summary_path.resolve(),
        visual_path=visual_path.resolve(),
    )


def build_and_export_driver_timelines(
    *,
    driver: str = "auto",
    top_n: int | None = None,
    output_dir: str | Path = "sflorida_outputs",
    gps_root: str | Path | None = None,
    config_path: str | Path | None = "config.yaml",
    repo_root: str | Path = ".",
    timezone_name: str = DEFAULT_TIMEZONE,
    tolerance_seconds: int = DEFAULT_MATCH_TOLERANCE_SECONDS,
) -> TimelineBuildResult:
    """Run the complete cached-data Phase 2A population/timeline workflow."""
    resolved_gps_root = discover_gps_root(
        explicit_root=gps_root,
        config_path=config_path,
        repo_root=repo_root,
    )
    county_inputs = discover_county_inputs(output_dir)
    matched_trips = pd.concat(
        [
            load_matched_trip_data(item, timezone_name=timezone_name)
            for item in county_inputs
        ],
        ignore_index=True,
    )
    source_trips = discover_source_trips(
        resolved_gps_root, [item.county for item in county_inputs]
    )
    mapped, mapping_stats = map_source_trips_to_matches(
        source_trips,
        matched_trips,
        tolerance_seconds=tolerance_seconds,
    )

    population = build_population_index(mapped)
    alias_map = build_alias_map(population)
    selected, population = select_population(
        population,
        driver=driver,
        top_n=top_n,
    )

    phase2_root = Path(output_dir) / "phase2"
    phase2_root.mkdir(parents=True, exist_ok=True)
    population_index_path = phase2_root / "driver_population_index.csv"
    alias_map_path = phase2_root / "driver_alias_map.csv"
    identity_audit_path = phase2_root / "driver_identity_audit.md"
    population.to_csv(population_index_path, index=False)
    alias_map.to_csv(alias_map_path, index=False)
    write_identity_audit(resolved_gps_root, identity_audit_path)

    driver_outputs: list[DriverOutput] = []
    for record in selected.itertuples(index=False):
        timeline = build_driver_trip_table(
            mapped,
            internal_driver_id=str(record.internal_driver_id),
            driver_alias=str(record.driver_alias),
        )
        monthly_summary = build_monthly_summary(timeline)
        driver_outputs.append(
            export_driver_timeline(
                timeline,
                monthly_summary,
                output_root=output_dir,
                driver_alias=str(record.driver_alias),
                internal_driver_id=str(record.internal_driver_id),
            )
        )

    overview_path = phase2_root / "visuals" / "driver_population_overview.html"
    write_population_overview(
        selected,
        driver_outputs,
        overview_path,
        requested_top_n=top_n,
    )

    return TimelineBuildResult(
        selected_outputs=tuple(driver_outputs),
        population_index_path=population_index_path.resolve(),
        alias_map_path=alias_map_path.resolve(),
        population_overview_path=overview_path.resolve(),
        identity_audit_path=identity_audit_path.resolve(),
        gps_root=resolved_gps_root,
        matched_paths=tuple(item.matched_path.resolve() for item in county_inputs),
        gps_paths=tuple(item.gps_path.resolve() for item in county_inputs),
        enriched_network_paths=tuple(
            item.enriched_network_path.resolve()
            for item in county_inputs
            if item.enriched_network_path
        ),
        discovered_source_trips=mapping_stats["discovered"],
        mapped_source_trips=mapping_stats["usable"],
        skipped_source_trips=mapping_stats["skipped"],
        requested_top_n=top_n,
    )


def build_and_export_driver_timeline(**kwargs: object) -> TimelineBuildResult:
    """Backward-compatible singular wrapper around the population workflow."""
    return build_and_export_driver_timelines(**kwargs)
