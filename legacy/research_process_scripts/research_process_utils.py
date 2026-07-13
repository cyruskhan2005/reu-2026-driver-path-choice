#!/usr/bin/env python3
"""Shared helpers for presentation-oriented research process visuals.

These functions intentionally favor robust local data discovery over a single
hard-coded file. The workflow is meant to explain the research process from raw
GPS through map matching, graph construction, and RCCI formula testing.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "visuals" / "research_process"

LON_CANDIDATES = ("lon", "lng", "longitude", "x")
LAT_CANDIDATES = ("lat", "latitude", "y")
TIME_CANDIDATES = (
    "timestamp",
    "time",
    "datetime",
    "date_time",
    "gps_time",
    "trip_start_time",
    "start_time",
)
DRIVER_CANDIDATES = ("driver_id", "driver", "driver_alias", "subject_label", "internal_driver_id")
SESSION_CANDIDATES = ("session_id", "session", "trip_id", "trajectory_id", "collection_id", "id")
PATH_CANDIDATES = ("fid_sequence", "opath", "cpath", "path", "fids", "matched_fids", "matched_fid")
FID_CANDIDATES = ("fid", "edge_id", "matched_fid", "id")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def county_from_path(path: Path) -> str:
    parts = [part for part in path.parent.parts if part]
    for part in reversed(parts):
        if part.endswith("_County") or part.endswith(" County") or part.endswith("-County"):
            return part.replace("_", " ").replace("Miami Dade", "Miami-Dade")
    return "Unknown County"


def county_slug(county: str) -> str:
    return str(county).lower().replace("-", "_").replace(" ", "_")


def county_dir_name(county: str) -> str:
    return str(county).replace("-", "_").replace(" ", "_")


def detect_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {str(col).lower().strip(): str(col) for col in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    for col in columns:
        normalized = str(col).lower().strip()
        for candidate in candidates:
            if normalized.endswith(candidate.lower()) or candidate.lower() in normalized:
                return str(col)
    return None


def sniff_delimiter(path: Path) -> str:
    try:
        first = path.open("r", encoding="utf-8", errors="replace").readline()
    except OSError:
        return ","
    counts = {";": first.count(";"), "\t": first.count("\t"), ",": first.count(",")}
    return max(counts, key=counts.get) if max(counts.values()) > 0 else ","


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True, **kwargs)
    if suffix == ".json":
        return pd.read_json(path, **kwargs)
    if suffix == ".csv":
        return pd.read_csv(path, sep=sniff_delimiter(path), **kwargs)
    raise ValueError(f"Unsupported table file: {path}")


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        median = numeric.dropna().median()
        if median > 1e12:
            return pd.to_datetime(numeric, unit="ms", errors="coerce", utc=True)
        if median > 1e9:
            return pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)
    return pd.to_datetime(series, errors="coerce", utc=True)


def parse_month_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        median = numeric.dropna().median()
        if median > 1e12:
            dt = pd.to_datetime(numeric, unit="ms", errors="coerce")
        elif median > 1e9:
            dt = pd.to_datetime(numeric, unit="s", errors="coerce")
        else:
            dt = pd.to_datetime(series, errors="coerce")
    else:
        dt = pd.to_datetime(series, errors="coerce")
    months = pd.Series(dt.values.astype("datetime64[M]").astype(str), index=series.index)
    return months.mask(months.eq("NaT"))


def find_raw_gps_files() -> list[Path]:
    patterns = [
        "sflorida_outputs/*_County/*_gps.csv",
        "sflorida_outputs/*_County/*_gps.parquet",
        "sflorida_outputs/*_County/*_gps.jsonl",
        "data/drivers/gps_master.parquet",
        "data/**/*_gps.csv",
        "data/**/*_gps.jsonl",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(ROOT.glob(pattern))
    return sorted(dict.fromkeys(path for path in files if path.is_file()))


def standardize_raw_gps_frame(data: pd.DataFrame, path: Path, include_datetime: bool = False) -> pd.DataFrame:
        lon_col = detect_column(data.columns, LON_CANDIDATES)
        lat_col = detect_column(data.columns, LAT_CANDIDATES)
        if not lon_col or not lat_col:
            return pd.DataFrame(columns=["source_file", "county", "lon", "lat"])
        out = pd.DataFrame(
            {
                "source_file": str(path.relative_to(ROOT)),
                "county": data["county"] if "county" in data.columns else county_from_path(path),
                "lon": pd.to_numeric(data[lon_col], errors="coerce"),
                "lat": pd.to_numeric(data[lat_col], errors="coerce"),
            }
        )
        time_col = detect_column(data.columns, TIME_CANDIDATES)
        if time_col:
            out["timestamp"] = data[time_col]
            if include_datetime:
                out["datetime"] = parse_timestamp_series(data[time_col])
                out["month"] = out["datetime"].dt.strftime("%Y-%m")
            else:
                out["month"] = parse_month_series(data[time_col])
        id_col = detect_column(data.columns, SESSION_CANDIDATES)
        if id_col:
            out["source_id"] = data[id_col].astype(str)
        driver_col = detect_column(data.columns, DRIVER_CANDIDATES)
        if driver_col:
            out["driver_id"] = data[driver_col].astype(str)
        return out.dropna(subset=["lon", "lat"])


def iter_raw_gps(chunksize: int = 250_000, files: list[Path] | None = None, include_datetime: bool = False):
    for path in files or find_raw_gps_files():
        try:
            if path.suffix.lower() == ".csv":
                for chunk in pd.read_csv(path, sep=sniff_delimiter(path), chunksize=chunksize):
                    standardized = standardize_raw_gps_frame(chunk, path, include_datetime=include_datetime)
                    if not standardized.empty:
                        yield standardized
            else:
                data = read_table(path)
                standardized = standardize_raw_gps_frame(data, path, include_datetime=include_datetime)
                if not standardized.empty:
                    yield standardized
        except Exception as exc:  # noqa: BLE001 - keep workflow partial.
            print(f"WARNING: could not read raw GPS file {path}: {exc}")


def load_raw_gps(files: list[Path] | None = None, max_rows: int | None = None, include_datetime: bool = False) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    remaining = max_rows
    for chunk in iter_raw_gps(files=files, include_datetime=include_datetime):
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
            remaining -= len(chunk)
        rows.append(chunk)
    if not rows:
        return pd.DataFrame(columns=["source_file", "county", "lon", "lat"])
    return pd.concat(rows, ignore_index=True)


def find_matched_files() -> list[Path]:
    patterns = [
        "sflorida_outputs/*_County/*_matched.csv",
        "deliverables/matched_csv/*.csv",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(ROOT.glob(pattern))
    return sorted(dict.fromkeys(path for path in files if path.is_file()))


def find_timeline_files() -> list[Path]:
    return sorted((ROOT / "sflorida_outputs" / "phase2" / "driver_timelines").glob("*timeline.csv"))


def parse_segment_sequence(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    text = text.strip("[]()")
    parts = re.split(r"[|,;\s]+", text)
    out: list[int] = []
    for part in parts:
        token = part.strip().strip("'\"")
        if not token:
            continue
        try:
            out.append(int(float(token)))
        except ValueError:
            continue
    return out


def load_timeline_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in find_timeline_files():
        try:
            data = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not read timeline file {path}: {exc}")
            continue
        if "fid_sequence" not in data.columns:
            continue
        if "county" not in data.columns:
            data["county"] = county_from_path(path)
        frames.append(data)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def explode_segment_observations(prefer_timeline: bool = True) -> pd.DataFrame:
    """Return one row per matched segment observation.

    Timeline files are preferred because they already align matched FIDs with
    driver, trip, and month metadata. If unavailable, the county matched CSVs
    are parsed as a fallback, without month information.
    """
    if prefer_timeline:
        timeline = load_timeline_rows()
        if not timeline.empty:
            rows: list[dict[str, object]] = []
            for item in timeline.itertuples(index=False):
                as_dict = item._asdict()
                seq = parse_segment_sequence(as_dict.get("fid_sequence"))
                trip_id = as_dict.get("trip_id") or as_dict.get("matched_trip_id") or as_dict.get("source_trip_id")
                for position, fid in enumerate(seq):
                    rows.append(
                        {
                            "source": "phase2_timeline",
                            "county": as_dict.get("county", "Unknown County"),
                            "period": as_dict.get("trip_month"),
                            "trip_id": trip_id,
                            "driver_id": as_dict.get("driver_id") or as_dict.get("internal_driver_id"),
                            "driver_label": as_dict.get("driver_alias") or as_dict.get("subject_label"),
                            "fid": fid,
                            "position": position,
                        }
                    )
            return pd.DataFrame(rows)

    rows = []
    for path in find_matched_files():
        try:
            data = read_table(path)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not read matched file {path}: {exc}")
            continue
        path_col = detect_column(data.columns, PATH_CANDIDATES)
        fid_col = detect_column(data.columns, FID_CANDIDATES)
        id_col = detect_column(data.columns, SESSION_CANDIDATES)
        if path_col:
            for record in data.itertuples(index=False):
                as_dict = record._asdict()
                trip_id = as_dict.get(id_col) if id_col else None
                for position, fid in enumerate(parse_segment_sequence(as_dict.get(path_col))):
                    rows.append(
                        {
                            "source": str(path.relative_to(ROOT)),
                            "county": county_from_path(path),
                            "period": None,
                            "trip_id": trip_id,
                            "fid": fid,
                            "position": position,
                        }
                    )
        elif fid_col:
            for record in data.itertuples(index=False):
                as_dict = record._asdict()
                rows.append(
                    {
                        "source": str(path.relative_to(ROOT)),
                        "county": county_from_path(path),
                        "period": None,
                        "trip_id": as_dict.get(id_col) if id_col else None,
                        "fid": as_dict.get(fid_col),
                        "position": None,
                    }
                )
    return pd.DataFrame(rows)


def segment_trip_counts(observations: pd.DataFrame, by_period: bool = True) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame(columns=["county", "period", "fid", "trip_use_count", "segment_pass_count"])
    data = observations.copy()
    data["fid"] = pd.to_numeric(data["fid"], errors="coerce").astype("Int64")
    data = data.dropna(subset=["fid"])
    data["fid"] = data["fid"].astype(int)
    group_cols = ["county", "fid"]
    if by_period and "period" in data.columns and data["period"].notna().any():
        group_cols.insert(1, "period")
    pass_counts = data.groupby(group_cols, dropna=False).size().rename("segment_pass_count").reset_index()
    if "trip_id" in data.columns and data["trip_id"].notna().any():
        trip_counts = (
            data.drop_duplicates(group_cols + ["trip_id"])
            .groupby(group_cols, dropna=False)
            .size()
            .rename("trip_use_count")
            .reset_index()
        )
        return pass_counts.merge(trip_counts, on=group_cols, how="left")
    pass_counts["trip_use_count"] = pass_counts["segment_pass_count"]
    return pass_counts


def find_enriched_network(county: str) -> Path | None:
    candidates = [
        ROOT / "sflorida_outputs" / county_dir_name(county) / "enriched_network.parquet",
        ROOT / "deliverables" / "google_drive_phase2" / "enriched_network_parquet" / county_dir_name(county) / "enriched_network.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    slug = county_slug(county)
    for path in ROOT.glob("sflorida_outputs/*_County/enriched_network.parquet"):
        if county_slug(county_from_path(path)) == slug:
            return path
    return None


def load_network_edges(county: str, fids: Iterable[int] | None = None, include_geometry: bool = True) -> gpd.GeoDataFrame:
    path = find_enriched_network(county)
    if not path:
        raise FileNotFoundError(f"No enriched network parquet found for {county}.")
    columns = ["u", "v", "highway", "name", "length"]
    if include_geometry:
        columns.append("geometry")
    data = gpd.read_parquet(path, columns=[col for col in columns if col != "fid"])
    data = data.reset_index()
    if "fid" not in data.columns:
        data = data.rename(columns={data.columns[0]: "fid"})
    data["fid"] = pd.to_numeric(data["fid"], errors="coerce").astype("Int64")
    data = data.dropna(subset=["fid"])
    data["fid"] = data["fid"].astype(int)
    if fids is not None:
        fid_set = {int(fid) for fid in fids if pd.notna(fid)}
        data = data.loc[data["fid"].isin(fid_set)]
    return data


def safe_period_label(value: object) -> str:
    text = str(value) if value is not None and not pd.isna(value) else "unknown_period"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def write_html(path: Path, title: str, body: str, extra_head: str = "") -> None:
    ensure_dir(path.parent)
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{extra_head}
<style>
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; background: #f7f8fa; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
h1, h2, h3 {{ color: #18212f; }}
.panel {{ background: white; border: 1px solid #d9dee7; border-radius: 8px; padding: 20px; margin: 18px 0; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
.metric {{ background: #f1f5f9; border-left: 4px solid #2563eb; padding: 12px; border-radius: 6px; }}
.metric strong {{ display: block; font-size: 1.5rem; color: #0f172a; }}
iframe {{ width: 100%; height: 560px; border: 1px solid #d9dee7; border-radius: 6px; background: white; }}
img {{ max-width: 100%; height: auto; border: 1px solid #d9dee7; border-radius: 6px; background: white; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; }}
th {{ background: #eef2f7; }}
code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
.note {{ color: #526173; }}
</style>
</head>
<body><main>
{body}
</main></body>
</html>
""",
        encoding="utf-8",
    )


def dataframe_preview_html(path: Path, rows: int = 8) -> str:
    if not path.exists():
        return f"<p class='note'>Missing expected file: <code>{path.name}</code></p>"
    try:
        data = pd.read_csv(path, nrows=rows)
    except Exception as exc:  # noqa: BLE001
        return f"<p class='note'>Could not preview <code>{path.name}</code>: {exc}</p>"
    if data.empty:
        return f"<p class='note'><code>{path.name}</code> is empty.</p>"
    return data.to_html(index=False, classes="preview")
