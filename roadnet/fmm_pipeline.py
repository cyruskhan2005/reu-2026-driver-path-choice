"""
roadnet.fmm_pipeline
====================
Map-match GPS traces onto a road network using the Fast Map-Matching (fmm)
library, then aggregate per-second sensor data (speed, accelerometer, OBD)
onto each matched road segment.

Key design:
  - FMM opath gives one FID per GPS point (1:1 positional alignment)
  - GPS points are merged to FIDs by point_idx (position), NOT by timestamp
  - acc/obd are still resampled to 1s and joined by timestamp range per FID
  - This preserves all GPS points and their sensor readings regardless of
    timestamp collisions (multiple GPS points sharing the same floored second)

Optimizations:
  1. Master GPS parquet cache
  2. GPS JSONL in-memory cache
  3. Sensor parquet cache (acc/obd)
  4. Multiprocessing aggregation

The public entry-point is ``run_county``.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count, get_context
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from pyproj import Transformer

from .config import (
    FMM_K,
    FMM_RADIUS_M,
    FMM_ERROR_M,
    FMM_RETRY_RADIUS_M,
    FMM_RETRY_ERROR_M,
    FMM_MIN_SEGMENT,
    FMM_SKIP_ON_GAP,
    STM_K,
    STM_RADIUS_M,
    STM_ERROR_M,
    STM_VMAX_MS,
    STM_FACTOR,
    GPS_GAP_THRESHOLD_S,
    WGS84,
    WEB_MERC,
)

log = logging.getLogger(__name__)
_DEG = 1.1e5


# ─────────────────────────────────────────────────────────────────────────────
# Session discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_sessions(gps_root: Path) -> list[tuple[Path, list[Path]]]:
    sessions = []
    for session_dir in Path(gps_root).glob("*/*/*"):
        gps_files = list(session_dir.glob("*_gps.jsonl"))
        if gps_files:
            sessions.append((session_dir, gps_files))
    return sessions


# ─────────────────────────────────────────────────────────────────────────────
# County assignment
# ─────────────────────────────────────────────────────────────────────────────

class CountyAssigner:
    _BORDER_BUFFER_DEG = 0.02

    def __init__(self, county_shapefiles: dict[str, gpd.GeoDataFrame]) -> None:
        self._shps:         dict[str, gpd.GeoDataFrame] = {}
        self._shps_proj:    dict[str, gpd.GeoDataFrame] = {}
        self._sindices:     dict[str, object]            = {}
        self._transformers: dict[str, Transformer]       = {}
        self._lat_bounds:   dict[str, tuple[float, float]] = {}
        self._border_zones: list[tuple[float, float]]    = []

        for name, shp in county_shapefiles.items():
            bounds = shp.to_crs(WGS84).total_bounds
            self._lat_bounds[name]   = (bounds[1], bounds[3])
            shp_proj = shp.to_crs(WEB_MERC)
            self._shps[name]         = shp.to_crs(WGS84)
            self._shps_proj[name]    = shp_proj
            self._sindices[name]     = self._shps[name].sindex
            self._transformers[name] = Transformer.from_crs(WGS84, WEB_MERC, always_xy=True)

        sorted_counties = sorted(self._lat_bounds.items(), key=lambda x: x[1][0])
        for i in range(len(sorted_counties) - 1):
            _, (_, top) = sorted_counties[i]
            _, (bot, _) = sorted_counties[i + 1]
            mid = (top + bot) / 2
            self._border_zones.append((mid - self._BORDER_BUFFER_DEG,
                                       mid + self._BORDER_BUFFER_DEG))

    def assign(self, lon: float, lat: float) -> Optional[str]:
        in_border = any(s <= lat <= n for s, n in self._border_zones)
        if not in_border:
            for county, (south, north) in self._lat_bounds.items():
                if south <= lat < north:
                    return county
            return None
        p   = Point(lon, lat)
        buf = p.buffer(0.003)
        best_county, best_dist = None, 9999.0
        for county, sindex in self._sindices.items():
            candidates = list(sindex.intersection(buf.bounds))
            if not candidates:
                continue
            x, y   = self._transformers[county].transform(lon, lat)
            p_proj = Point(x, y)
            dist   = self._shps_proj[county].iloc[candidates].geometry.distance(p_proj).min()
            if dist < best_dist:
                best_dist, best_county = dist, county
        return best_county if best_dist <= 300 else None


# ─────────────────────────────────────────────────────────────────────────────
# Master GPS parquet cache
# ─────────────────────────────────────────────────────────────────────────────

def build_master_gps_parquet(
    sessions:   list[tuple[Path, list[Path]]],
    assigner:   CountyAssigner,
    cache_path: Path,
) -> pd.DataFrame:
    meta_path        = cache_path.with_suffix(".meta.json")
    current_counties = sorted(assigner._lat_bounds.keys())

    if cache_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            cached_counties = meta.get("counties")
            if (
                cached_counties == current_counties or
                set(cached_counties or []).issuperset(current_counties)
            ):
                log.info("Loading master GPS parquet from %s", cache_path)
                return pd.read_parquet(cache_path)
            else:
                log.info("County config changed — rebuilding GPS parquet")
                cache_path.unlink()
                meta_path.unlink()
        except Exception:
            cache_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)

    log.info("Building master GPS parquet — reading %d sessions …", len(sessions))
    t0   = time.time()
    rows: list[dict] = []

    for session_dir, gps_files in sessions:
        for gps_path in gps_files:
            with open(gps_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        d   = json.loads(line)
                        loc = d.get("loc", {})
                        if not isinstance(loc, dict):
                            continue
                        lon, lat = loc["lon"], loc["lat"]
                        ts_iso   = d.get("@ts", "")
                        county   = assigner.assign(lon, lat)
                        if county is None:
                            continue
                        try:
                            ts_epoch = int(datetime.fromisoformat(
                                ts_iso.replace("Z", "+00:00")
                            ).timestamp())
                        except Exception:
                            ts_epoch = 0
                        rows.append({
                            "gps_path":    str(gps_path),
                            "session_dir": str(session_dir),
                            "lon":         lon,
                            "lat":         lat,
                            "ts_iso":      ts_iso,
                            "ts_epoch":    ts_epoch,
                            "county":      county,
                        })
                    except Exception:
                        pass

    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    meta_path.write_text(json.dumps({"counties": current_counties}))
    log.info("Master GPS parquet written — %d points in %.1f s",
             len(df), time.time() - t0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Sensor parsing
# ─────────────────────────────────────────────────────────────────────────────

def _to_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed").dt.floor("s")


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return pd.DataFrame(rows)


def _parse_acc(path: Path) -> pd.DataFrame:
    df = _read_jsonl(path)
    if df.empty or "@ts" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = _to_ts(df["@ts"])
    if "@t" in df.columns and "x" in df.columns:
        acc = df[df["@t"] == "acc.xyz"].copy()
        if acc.empty:
            return pd.DataFrame()
        return acc[["timestamp"]].assign(
            acc_x=acc["x"].values,
            acc_y=acc["y"].values,
            acc_z=acc["z"].values,
        )
    if "acc" not in df.columns:
        return pd.DataFrame()
    df["acc_x"] = df["acc"].apply(lambda d: d.get("x") if isinstance(d, dict) else None)
    df["acc_y"] = df["acc"].apply(lambda d: d.get("y") if isinstance(d, dict) else None)
    df["acc_z"] = df["acc"].apply(lambda d: d.get("z") if isinstance(d, dict) else None)
    return df[["timestamp", "acc_x", "acc_y", "acc_z"]]


def _parse_obd(path: Path) -> pd.DataFrame:
    df = _read_jsonl(path)
    if df.empty or "@ts" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = _to_ts(df["@ts"])
    wide = df.pivot_table(index="timestamp", columns="@t", values="value", aggfunc="mean")
    wide.columns = [c.replace("obd.", "obd_").replace(".", "_") for c in wide.columns]
    return wide.reset_index()


def _load_session_sensors(
    session_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    acc_cache_path = session_dir / "acc_cache.parquet"
    obd_cache_path = session_dir / "obd_cache.parquet"

    if acc_cache_path.exists() and obd_cache_path.exists():
        try:
            return pd.read_parquet(acc_cache_path), pd.read_parquet(obd_cache_path)
        except Exception:
            pass

    all_files = list(session_dir.glob("*.jsonl"))
    acc_files = [f for f in all_files if "_acc" in f.name]
    obd_files = [f for f in all_files if "_obd" in f.name]

    acc_agg = pd.DataFrame()
    if acc_files:
        frames = [_parse_acc(f) for f in acc_files]
        valid  = [df for df in frames if not df.empty]
        if valid:
            raw     = pd.concat(valid, ignore_index=True).set_index("timestamp").sort_index()
            acc_agg = pd.concat(
                [raw.resample("1s").mean().add_suffix("_mean"),
                 raw.resample("1s").std().add_suffix("_std")],
                axis=1,
            ).reset_index()

    obd_agg = pd.DataFrame()
    if obd_files:
        frames = [_parse_obd(f) for f in obd_files]
        valid  = [df for df in frames if not df.empty]
        if valid:
            raw     = pd.concat(valid, ignore_index=True).set_index("timestamp").sort_index()
            obd_agg = raw.resample("1s").mean().reset_index()

    try:
        if not acc_agg.empty:
            acc_agg.to_parquet(acc_cache_path, index=False)
        else:
            pd.DataFrame().to_parquet(acc_cache_path, index=False)
        if not obd_agg.empty:
            obd_agg.to_parquet(obd_cache_path, index=False)
        else:
            pd.DataFrame().to_parquet(obd_cache_path, index=False)
    except Exception:
        pass

    return acc_agg, obd_agg


def _get_session_sensors(
    session_dir: Path,
    cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = str(session_dir)
    if key not in cache:
        cache[key] = _load_session_sensors(session_dir)
    return cache[key]


# ─────────────────────────────────────────────────────────────────────────────
# Map-matching helpers
# ─────────────────────────────────────────────────────────────────────────────

def _match_with_splits(
    coords: list[tuple[float, float]],
    model:  object,
    config: object,
) -> list[tuple[tuple, int]]:
    try:
        from fmm import FastMapMatchConfig  # noqa: F401
    except ImportError:
        raise RuntimeError("fmm Python bindings not installed")

    results:   list[tuple[tuple, int]] = []
    remaining: list                    = list(coords)

    while len(remaining) >= FMM_MIN_SEGMENT:
        try:
            r     = model.match_wkt(LineString(remaining).wkt, config)
            opath = list(r.opath)
        except Exception:
            opath = []

        if opath:
            results.extend(zip(remaining[: len(opath)], opath))
            break

        lo, hi = FMM_MIN_SEGMENT, len(remaining)
        while lo < hi - 1:
            mid = (lo + hi) // 2
            try:
                r_test     = model.match_wkt(LineString(remaining[:mid]).wkt, config)
                opath_test = list(r_test.opath)
            except Exception:
                opath_test = []
            if opath_test:
                lo = mid
            else:
                hi = mid

        try:
            r_good     = model.match_wkt(LineString(remaining[:lo]).wkt, config)
            opath_good = list(r_good.opath)
        except Exception:
            opath_good = []

        if not opath_good:
            break
        results.extend(zip(remaining[: len(opath_good)], opath_good))
        gap_end = lo + FMM_SKIP_ON_GAP
        results.extend((c, -1) for c in remaining[len(opath_good): gap_end])
        remaining = remaining[gap_end:]

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Gap detection + STMatch bridging
# ─────────────────────────────────────────────────────────────────────────────

def _find_gaps(
    ts_list:     list[str],
    threshold_s: float = GPS_GAP_THRESHOLD_S,
) -> list[int]:
    gaps   = []
    parsed = []
    for t in ts_list:
        try:
            parsed.append(datetime.fromisoformat(t.replace("Z", "+00:00")))
        except Exception:
            parsed.append(None)
    for i in range(len(parsed) - 1):
        if parsed[i] is None or parsed[i + 1] is None:
            continue
        if (parsed[i + 1] - parsed[i]).total_seconds() > threshold_s:
            gaps.append(i)
    return gaps


def _stmatch_gap(
    coord_before: tuple[float, float],
    coord_after:  tuple[float, float],
    stm_model:    object,
    stm_config:   object,
) -> list[int]:
    try:
        wkt    = LineString([coord_before, coord_after]).wkt
        result = stm_model.match_wkt(wkt, stm_config)
        return list(result.cpath)
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Core aggregation — positional GPS-to-FID alignment
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_one_trip(
    trip_rows:    pd.DataFrame,
    gps_df_full:  pd.DataFrame,
    acc_agg:      pd.DataFrame,
    obd_agg:      pd.DataFrame,
    out_path:     Path,
    stmatch_fids: set | None = None,
) -> bool:
    """
    Aggregate sensor data for a single trip and write JSONL output.

    trip_rows columns: point_idx, timestamp, fid
      - point_idx: positional index into the GPS file (0-based)
      - timestamp:  floored-to-second datetime
      - fid:        matched road segment FID (-1 = unmatched)

    Strategy:
      1. Join trip_rows to raw GPS data by point_idx — preserves every GPS
         point with its FID regardless of timestamp collisions.
      2. For each FID, use the min/max timestamp to slice acc/obd, then
         compute mean/std of sensor values over that time window.
      3. Write one JSONL record per FID in traversal order.
    """
    # ── Parse raw GPS ─────────────────────────────────────────────────────────
    gps_df = gps_df_full.copy()
    if "@ts" in gps_df.columns:
        gps_df["timestamp"] = _to_ts(gps_df["@ts"])
    elif "timestamp" not in gps_df.columns:
        return False

    # Ensure point_idx exists in raw GPS
    if "point_idx" not in gps_df.columns:
        gps_df = gps_df.reset_index(drop=True)
        gps_df["point_idx"] = gps_df.index

    # ── Positional merge: each GPS point gets its FID ─────────────────────────
    matched_rows = trip_rows[trip_rows["fid"] != -1].copy()
    if matched_rows.empty:
        return False

    # Join on point_idx — 1:1, no timestamp collision issues
    gps_with_fid = gps_df.merge(
        matched_rows[["point_idx", "fid"]],
        on="point_idx",
        how="inner",
    )
    if gps_with_fid.empty:
        return False

    # Extract GPS-level sensor columns available in the raw JSONL
    gps_cols = [c for c in ("sog", "cog", "nsat") if c in gps_with_fid.columns]

    # ── Per-FID aggregation ───────────────────────────────────────────────────
    # GPS sensors: mean/std per FID directly from GPS points
    gps_agg_dict = {c: (c, "mean") for c in gps_cols}
    std_gps      = {f"{c}_variability": (c, "std") for c in ("sog",) if c in gps_cols}
    gps_agg_dict.update(std_gps)
    gps_agg_dict["seconds_total"] = ("timestamp", lambda x: int((x.max() - x.min()).total_seconds()) + 1)
    gps_agg_dict["ts_min"]        = ("timestamp", "min")
    gps_agg_dict["ts_max"]        = ("timestamp", "max")

    fid_gps = gps_with_fid.groupby("fid").agg(**gps_agg_dict).reset_index()

    # Traversal order: sort FIDs by their first GPS timestamp
    fid_order = (
        gps_with_fid.groupby("fid")["timestamp"].min()
        .reset_index()
        .sort_values("timestamp")
    )
    fid_gps = fid_order[["fid"]].merge(fid_gps, on="fid", how="left")

    # ── acc/obd: join by time range per FID ──────────────────────────────────
    # For each FID, pull sensor readings between ts_min and ts_max.
    # This is still time-based but within each FID's window — correct because
    # we know exactly what time range each FID was traversed.
    acc_cols = [c for c in acc_agg.columns if c != "timestamp"] if not acc_agg.empty else []
    obd_cols = [c for c in obd_agg.columns if c != "timestamp"] if not obd_agg.empty else []

    acc_per_fid: dict[int, dict] = {}
    obd_per_fid: dict[int, dict] = {}

    for _, row in fid_gps.iterrows():
        fid    = int(row["fid"])
        t_min  = row["ts_min"]
        t_max  = row["ts_max"]

        if not acc_agg.empty and acc_cols:
            window = acc_agg[
                (acc_agg["timestamp"] >= t_min) &
                (acc_agg["timestamp"] <= t_max)
            ]
            if not window.empty:
                acc_per_fid[fid] = {
                    col: window[col].mean()
                    for col in acc_cols
                    if col in window.columns
                }

        if not obd_agg.empty and obd_cols:
            window = obd_agg[
                (obd_agg["timestamp"] >= t_min) &
                (obd_agg["timestamp"] <= t_max)
            ]
            if not window.empty:
                obd_per_fid[fid] = {
                    col: window[col].mean()
                    for col in obd_cols
                    if col in window.columns
                }
                # variability for engine load
                if "obd_engine_load" in window.columns and len(window) > 1:
                    obd_per_fid[fid]["obd_engine_load_variability"] = \
                        window["obd_engine_load"].std()

    # ── Write output ──────────────────────────────────────────────────────────
    drop_cols = {"ts_min", "ts_max"}

    with open(out_path, "w") as f:
        for seq_i, (_, row) in enumerate(fid_gps.iterrows()):
            fid    = int(row["fid"])
            record: dict = {}

            # GPS-derived fields
            for col in fid_gps.columns:
                if col in drop_cols or col == "fid":
                    continue
                val = row[col]
                record[col] = None if pd.isna(val) else val

            record["fid"] = fid

            # acc fields
            for col, val in acc_per_fid.get(fid, {}).items():
                record[col] = None if pd.isna(val) else val

            # obd fields
            for col, val in obd_per_fid.get(fid, {}).items():
                record[col] = None if pd.isna(val) else val

            # match method
            if stmatch_fids:
                record["match_method"] = "stmatch" if fid in stmatch_fids else "fmm"
            else:
                record["match_method"] = "fmm"

            record["seq_idx"] = seq_i
            f.write(json.dumps(record) + "\n")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Multiprocessing worker
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args: dict) -> dict:
    tid         = args["tid"]
    trip_rows   = args["trip_rows"]
    gps_df_full = args.get("gps_df_full")
    gps_path    = args.get("gps_path")
    session_dir = Path(args["session_dir"])
    out_path    = Path(args["out_path"])
    meta_coords = args["meta_coords"]
    meta_ts     = args["meta_ts"]
    gap_indices = args["gap_indices"]

    if gps_df_full is None:
        try:
            gps_df_full = _read_jsonl(Path(gps_path)) if gps_path else pd.DataFrame()
        except Exception:
            gps_df_full = pd.DataFrame()

    acc_agg, obd_agg = _load_session_sensors(session_dir)

    success = _aggregate_one_trip(
        trip_rows, gps_df_full, acc_agg, obd_agg, out_path,
    )

    gap_entries: list[tuple] = []
    if gap_indices:
        gap_pairs    = []
        total_coords = len(meta_coords)
        for gap_idx in gap_indices:
            if gap_idx < total_coords - 1:
                gap_pairs.append((
                    meta_coords[gap_idx],
                    meta_coords[gap_idx + 1],
                    meta_ts[gap_idx],
                    meta_ts[gap_idx + 1],
                    gap_idx,
                    total_coords,
                ))
        if gap_pairs:
            gap_entries.append((str(out_path), str(session_dir), gap_pairs))

    return {
        "tid":         tid,
        "success":     success,
        "gap_entries": gap_entries,
    }


# Backward-compatible alias
def _write_aggregation(
    trip_rows:    pd.DataFrame,
    gps_df_full:  pd.DataFrame,
    session_dir:  Path,
    acc_agg:      pd.DataFrame,
    obd_agg:      pd.DataFrame,
    session_cache: dict,
    county_name:  str,
    prefix:       str,
    out_path:     Path,
    stmatch_fids: set | None = None,
) -> bool:
    return _aggregate_one_trip(
        trip_rows, gps_df_full, acc_agg, obd_agg, out_path, stmatch_fids,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FMM CLI runner — requests opath (1:1 with GPS points)
# ─────────────────────────────────────────────────────────────────────────────

def _run_fmm_cli(
    shp_path:   str,
    ubodt_path: str,
    gps_csv:    str,
    out_csv:    str,
    fmm_bin:    str = "fmm",
    radius_m:   int = FMM_RADIUS_M,
    error_m:    int = FMM_ERROR_M,
    k:          int = FMM_K,
) -> bool:
    cmd = [
        fmm_bin,
        "--network",       shp_path,
        "--network_id",    "fid",
        "--source",        "u",
        "--target",        "v",
        "--ubodt",         ubodt_path,
        "--gps",           gps_csv,
        "--gps_point",
        "--gps_id",        "id",
        "--gps_x",         "lon",
        "--gps_y",         "lat",
        "--gps_timestamp", "timestamp",
        "--output",        out_csv,
        "--output_fields", "opath",   # 1:1 with GPS points
        "-k",  str(k),
        "-r",  str(radius_m / _DEG),
        "-e",  str(error_m  / _DEG),
        "--reverse_tolerance", "1",
        "--use_omp",
        "-l",  "2",
    ]
    log.info("Running fmm CLI …")
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
    for line in proc.stderr:
        log.info("[fmm] %s", line.rstrip())
    proc.wait()
    ok = os.path.exists(out_csv) and os.path.getsize(out_csv) > 0
    if not ok:
        log.warning("fmm produced no output (exit %d)", proc.returncode)
    return ok


def _ubodt_generation_child(payload: dict) -> None:
    try:
        logging.basicConfig(
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
            level=logging.INFO,
            force=True,
        )

        county_name = payload["county_name"]
        shp_path    = payload["shp_path"]
        ubodt_path  = payload["ubodt_path"]

        from fmm import Network, NetworkGraph, UBODTGenAlgorithm

        log.info("[%s] Generating UBODT …", county_name)
        net   = Network(shp_path, "fid", "u", "v")
        graph = NetworkGraph(net)
        UBODTGenAlgorithm(net, graph).generate_ubodt(
            ubodt_path, 0.007, binary=False, use_omp=True)
        log.info("[%s] UBODT written to %s", county_name, ubodt_path)
        logging.shutdown()
        # Bypass _fmm's SWIG/C++ destructors; normal teardown can segfault.
        os._exit(0)
    except Exception:
        log.exception("UBODT generation failed")
        logging.shutdown()
        os._exit(1)


def _generate_ubodt_native(county_name: str, shp_path: str, ubodt_path: str) -> None:
    ctx = get_context("spawn")
    proc = ctx.Process(
        target=_ubodt_generation_child,
        args=({
            "county_name": county_name,
            "shp_path":    shp_path,
            "ubodt_path":  ubodt_path,
        },),
    )
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(
            f"UBODT generation failed with exit code {proc.exitcode}"
        )


def _native_postprocess_child(payload: dict) -> None:
    try:
        logging.basicConfig(
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
            level=logging.INFO,
            force=True,
        )

        county_name = payload["county_name"]
        shp_path    = payload["shp_path"]
        ubodt_path  = payload["ubodt_path"]
        trip_meta   = payload["trip_meta"]
        failed_ids  = payload["failed_ids"]
        gap_work    = payload["gap_work"]

        net = graph = None

        if not failed_ids:
            log.info("[%s] No failed trips to retry", county_name)
        else:
            log.info("[%s] Retrying %d failed trips with split-matching …",
                     county_name, len(failed_ids))
            from fmm import Network, NetworkGraph, FastMapMatch, UBODT, FastMapMatchConfig

            net    = Network(shp_path, "fid", "u", "v")
            graph  = NetworkGraph(net)
            ubodt  = UBODT.read_ubodt_csv(ubodt_path)
            model  = FastMapMatch(net, graph, ubodt)
            config = FastMapMatchConfig(
                FMM_K, FMM_RETRY_RADIUS_M / _DEG,
                FMM_RETRY_ERROR_M / _DEG, reverse_tolerance=1)
            session_cache: dict = {}
            rescued = 0

            for tid in failed_ids:
                meta = trip_meta.get(tid)
                if meta is None or len(meta["coords"]) < 2:
                    continue

                results       = _match_with_splits(meta["coords"], model, config)
                matched_count = sum(1 for _, fid in results if fid != -1)
                if matched_count == 0:
                    continue

                coord_to_ts  = {c: t for c, t in zip(meta["coords"], meta["ts"])}
                result_fids  = [fid for _, fid in results]
                result_ts    = [coord_to_ts.get(c, meta["ts"][0]) for c, _ in results]
                n            = len(result_fids)

                trip_rows_df = pd.DataFrame({
                    "point_idx": list(range(n)),
                    "timestamp": _to_ts(pd.Series(result_ts[:n])),
                    "fid":       result_fids[:n],
                })

                acc_agg, obd_agg = _get_session_sensors(meta["session_dir"], session_cache)
                try:
                    gps_df_full = _read_jsonl(meta["gps_path"])
                except Exception:
                    gps_df_full = pd.DataFrame()
                out_p = meta["gps_path"].parent / \
                    f"{county_name}_{meta['prefix']}_fid_aggregated.jsonl"

                if _write_aggregation(
                    trip_rows_df, gps_df_full, meta["session_dir"],
                    acc_agg, obd_agg, session_cache,
                    county_name, meta["prefix"], out_p,
                ):
                    rescued += 1

            log.info("[%s] Split-match rescued %d / %d failed trips",
                     county_name, rescued, len(failed_ids))

        if gap_work:
            if net is None or graph is None:
                from fmm import Network, NetworkGraph

                net   = Network(shp_path, "fid", "u", "v")
                graph = NetworkGraph(net)

            from fmm import STMATCH, STMATCHConfig
            log.info("[%s] STMatch gap bridging for %d trips …",
                     county_name, len(gap_work))
            stm_model  = STMATCH(net, graph)
            stm_config = STMATCHConfig(
                STM_K, STM_RADIUS_M / _DEG, STM_ERROR_M / _DEG,
                STM_VMAX_MS, STM_FACTOR,
            )
            gap_bridged = 0

            for out_p, _gps_path, _session_dir, _prefix, gap_pairs in gap_work:
                if not out_p.exists():
                    continue

                stmatch_fids: set[int] = set()
                insertions:   list[tuple[int, list[int]]] = []

                for gap_pair in gap_pairs:
                    coord_before  = gap_pair[0]
                    coord_after   = gap_pair[1]
                    gap_coord_idx = gap_pair[4]
                    total_coords  = gap_pair[5]

                    gap_cpath = _stmatch_gap(
                        coord_before, coord_after, stm_model, stm_config)
                    if not gap_cpath:
                        continue

                    gap_bridged += 1
                    stmatch_fids.update(gap_cpath)
                    insertions.append((gap_coord_idx, total_coords, gap_cpath))

                if not insertions:
                    continue

                existing: list[dict] = []
                fmm_fids: set[int]   = set()
                with open(out_p) as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                            existing.append(r)
                            if r.get("match_method", "fmm") == "fmm":
                                fmm_fids.add(int(r["fid"]))
                        except Exception:
                            pass

                if not existing:
                    continue

                n_records = len(existing)
                resolved: list[tuple[int, list[int]]] = []
                for gap_coord_idx, total_coords, gap_cpath in insertions:
                    if total_coords > 1:
                        frac         = gap_coord_idx / (total_coords - 1)
                        insert_after = min(int(frac * n_records), n_records - 1)
                    else:
                        insert_after = n_records - 1
                    resolved.append((insert_after, gap_cpath))

                resolved.sort(key=lambda x: x[0])

                result: list[dict] = []
                prev  = -1

                for insert_after, gap_cpath in resolved:
                    result.extend(existing[prev + 1: insert_after + 1])
                    prev = insert_after
                    for fid in gap_cpath:
                        if int(fid) in fmm_fids:
                            continue
                        result.append({"fid": fid, "match_method": "stmatch"})

                result.extend(existing[prev + 1:])

                for i, r in enumerate(result):
                    r["seq_idx"] = i

                with open(out_p, "w") as f:
                    for record in result:
                        f.write(json.dumps(
                            {k: (None if pd.isna(v) else v)
                             for k, v in record.items()}
                        ) + "\n")

            log.info("[%s] STMatch bridged %d gap segments", county_name, gap_bridged)

        logging.shutdown()
        # Bypass _fmm's SWIG/C++ destructors; normal teardown can segfault.
        os._exit(0)
    except Exception:
        log.exception("Native FMM postprocess failed")
        logging.shutdown()
        os._exit(1)


def _run_native_postprocess(payload: dict) -> None:
    ctx = get_context("spawn")
    proc = ctx.Process(target=_native_postprocess_child, args=(payload,))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(
            f"Native FMM postprocess failed with exit code {proc.exitcode}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry-point
# ─────────────────────────────────────────────────────────────────────────────

def run_county(
    county_name:   str,
    shp_dir:       Path,
    sessions:      list[tuple[Path, list[Path]]],
    assigner:      CountyAssigner,
    out_dir:       Path,
    fmm_bin:       str = "fmm",
    master_gps_df: Optional[pd.DataFrame] = None,
    n_workers:     int = max(1, cpu_count() - 1),
) -> None:
    shp_path   = str(shp_dir / "edges.shp")
    ubodt_path = str(shp_dir / "ubodt.txt")
    gps_csv    = str(out_dir / f"{county_name}_gps.csv")
    fmm_out    = str(out_dir / f"{county_name}_matched.csv")

    worker_override = os.getenv("ROADNET_FMM_WORKERS")
    if worker_override:
        try:
            n_workers = max(1, int(worker_override))
        except ValueError:
            log.warning("Ignoring invalid ROADNET_FMM_WORKERS=%r", worker_override)

    # ── Generate UBODT if absent ──────────────────────────────────────────────
    if not os.path.exists(ubodt_path):
        _generate_ubodt_native(county_name, shp_path, ubodt_path)
    else:
        log.info("[%s] UBODT already exists — skipping generation", county_name)

    # ── Build GPS CSV ─────────────────────────────────────────────────────────
    gps_file_cache: dict[str, pd.DataFrame] = {}
    rows:      list[dict] = []
    trip_meta: dict       = {}
    trip_id               = 0

    if master_gps_df is not None:
        log.info("[%s] Filtering master GPS parquet for county …", county_name)
        county_df = master_gps_df[master_gps_df["county"] == county_name].copy()

        for gps_path_str, grp in county_df.groupby("gps_path", sort=False):
            grp         = grp.sort_values("ts_iso")
            gps_path    = Path(gps_path_str)
            session_dir = Path(grp["session_dir"].iloc[0])
            coords_in   = list(zip(grp["lon"], grp["lat"]))
            ts_in       = grp["ts_iso"].tolist()

            if len(coords_in) < 2:
                continue

            for i, (c, t, epoch) in enumerate(
                zip(coords_in, ts_in, grp["ts_epoch"])
            ):
                rows.append({
                    "id":        trip_id,
                    "lon":       c[0],
                    "lat":       c[1],
                    "timestamp": int(epoch),
                    "ts_iso":    t,
                    "point_idx": i,
                })

            gap_indices = _find_gaps(ts_in)
            trip_meta[trip_id] = {
                "session_dir": session_dir,
                "gps_path":    gps_path,
                "prefix":      gps_path.name.replace("_gps.jsonl", ""),
                "coords":      coords_in,
                "ts":          ts_in,
                "gap_indices": gap_indices,
            }
            trip_id += 1

    else:
        log.info("[%s] Building GPS CSV from %d sessions …",
                 county_name, len(sessions))
        for session_dir, gps_files in sessions:
            for gps_path in gps_files:
                coords_in: list[tuple[float, float]] = []
                ts_in:     list[str]                 = []

                with open(gps_path, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        try:
                            d   = json.loads(line)
                            loc = d.get("loc", {})
                            if not isinstance(loc, dict):
                                continue
                            lon, lat = loc["lon"], loc["lat"]
                            if assigner.assign(lon, lat) == county_name:
                                coords_in.append((lon, lat))
                                ts_in.append(d.get("@ts", ""))
                        except Exception:
                            pass

                if len(coords_in) < 2:
                    continue

                for i, (c, t) in enumerate(zip(coords_in, ts_in)):
                    try:
                        ts_epoch = int(datetime.fromisoformat(
                            t.replace("Z", "+00:00")
                        ).timestamp())
                    except Exception:
                        ts_epoch = i
                    rows.append({
                        "id":        trip_id,
                        "lon":       c[0],
                        "lat":       c[1],
                        "timestamp": ts_epoch,
                        "ts_iso":    t,
                        "point_idx": i,
                    })

                gap_indices = _find_gaps(ts_in)
                trip_meta[trip_id] = {
                    "session_dir": session_dir,
                    "gps_path":    gps_path,
                    "prefix":      gps_path.name.replace("_gps.jsonl", ""),
                    "coords":      coords_in,
                    "ts":          ts_in,
                    "gap_indices": gap_indices,
                }
                trip_id += 1

    if not rows:
        log.warning("[%s] No GPS points found — skipping", county_name)
        return

    gps_df = pd.DataFrame(rows)[["id", "lon", "lat", "timestamp", "ts_iso", "point_idx"]]
    gps_df.drop(columns=["ts_iso"]).to_csv(
        gps_csv, index=False, sep=";", encoding="utf-8")
    log.info("[%s] GPS CSV ready — %d points / %d trips",
             county_name, len(gps_df), trip_id)

    # ── Pre-load GPS JSONL files for the serial path only ─────────────────────
    # Sending these full dataframes through multiprocessing pickles a large
    # object per trip and can stall the pool after FMM on large counties.
    if n_workers <= 1:
        log.info("[%s] Pre-loading GPS JSONL files into memory …", county_name)
        unique_gps_paths = {str(meta["gps_path"]) for meta in trip_meta.values()}
        for gps_path_str in unique_gps_paths:
            try:
                gps_file_cache[gps_path_str] = _read_jsonl(Path(gps_path_str))
            except Exception:
                gps_file_cache[gps_path_str] = pd.DataFrame()
        log.info("[%s] Loaded %d GPS files into memory",
                 county_name, len(gps_file_cache))

    # ── Run fmm CLI ───────────────────────────────────────────────────────────
    reuse_matched = os.getenv("ROADNET_FMM_REUSE_MATCHED") == "1"
    if reuse_matched and os.path.exists(fmm_out):
        log.info("[%s] Reusing existing FMM output %s", county_name, fmm_out)
    else:
        t_fmm = time.time()
        ok = _run_fmm_cli(shp_path, ubodt_path, gps_csv, fmm_out, fmm_bin=fmm_bin)
        log.info("[%s] FMM CLI finished in %.1f s", county_name, time.time() - t_fmm)
        if not ok:
            return

    # ── Read matched CSV, build worker args ───────────────────────────────────
    mode = "sequential" if n_workers <= 1 else "parallel"
    log.info("[%s] Aggregating sensor data (%s, %d worker%s) …",
             county_name, mode, n_workers, "" if n_workers == 1 else "s")
    t_agg   = time.time()
    matched = pd.read_csv(fmm_out, sep=";")

    matched_map: dict[int, str] = {
        int(row["id"]): str(row["opath"])
        for _, row in matched.iterrows()
        if not pd.isna(row.get("opath", float("nan")))
        and str(row.get("opath", "")) != ""
    }
    failed_ids = set(
        matched.loc[
            matched["opath"].isna() | (matched["opath"] == ""), "id"
        ].astype(int).tolist()
    )

    worker_args: list[dict] = []

    for tid, meta in trip_meta.items():
        opath_str = matched_map.get(tid)
        if opath_str is None:
            continue

        opath_fids = [int(x) for x in opath_str.split(",")]
        trip_gps   = gps_df[gps_df["id"] == tid].sort_values("point_idx")
        if trip_gps.empty:
            continue

        # opath is 1:1 with GPS points — align by position
        n = min(len(opath_fids), len(trip_gps))

        trip_rows_df = pd.DataFrame({
            "point_idx": trip_gps["point_idx"].values[:n],
            "timestamp": _to_ts(pd.Series(trip_gps["ts_iso"].values[:n]
                                          if "ts_iso" in trip_gps.columns
                                          else trip_gps["timestamp"].values[:n])),
            "fid":       opath_fids[:n],
        })

        out_p = meta["gps_path"].parent / \
            f"{county_name}_{meta['prefix']}_fid_aggregated.jsonl"

        worker_arg = {
            "tid":         tid,
            "trip_rows":   trip_rows_df,
            "gps_path":    str(meta["gps_path"]),
            "session_dir": str(meta["session_dir"]),
            "out_path":    str(out_p),
            "meta_coords": meta["coords"],
            "meta_ts":     meta["ts"],
            "gap_indices": meta.get("gap_indices", []),
        }
        if n_workers <= 1:
            worker_arg["gps_df_full"] = gps_file_cache.get(
                str(meta["gps_path"]), pd.DataFrame()
            )
        worker_args.append(worker_arg)

    # ── Run workers in parallel ───────────────────────────────────────────────
    done      = 0
    gap_work: list[tuple] = []

    total_work = len(worker_args)
    if worker_args:
        log.info("[%s] Aggregating %d trips with %d worker(s)",
                 county_name, total_work, n_workers)

        pool = None
        if n_workers <= 1:
            result_iter = map(_worker, worker_args)
        else:
            pool = Pool(processes=n_workers)
            result_iter = pool.imap_unordered(_worker, worker_args, chunksize=1)

        completed = 0
        pool_failed = False
        try:
            for res in result_iter:
                completed += 1
                if res["success"]:
                    done += 1
                if completed == 1 or completed % 25 == 0 or completed == total_work:
                    log.info(
                        "[%s] Aggregation progress: %d / %d trips (%d successful)",
                        county_name, completed, total_work, done,
                    )
                for gap_entry in res["gap_entries"]:
                    out_p_str, session_dir_str, gap_pairs = gap_entry
                    tid  = res["tid"]
                    meta = trip_meta.get(tid)
                    if meta:
                        gap_work.append((
                            Path(out_p_str),
                            meta["gps_path"],
                            Path(session_dir_str),
                            meta["prefix"],
                            gap_pairs,
                        ))
        except Exception:
            pool_failed = True
            raise
        finally:
            if pool is not None:
                if pool_failed:
                    pool.terminate()
                else:
                    pool.close()
                pool.join()
                result_iter = None
                pool = None

    log.info("[%s] Aggregated %d trips in %.1f s",
             county_name, done, time.time() - t_agg)
    log.info("[%s] Failed trips: %d", county_name, len(failed_ids))

    agg_only = os.getenv("ROADNET_FMM_AGG_ONLY") == "1"
    if agg_only:
        log.info("[%s] ROADNET_FMM_AGG_ONLY=1 — skipping failed-trip retry and STMatch",
                 county_name)
        del matched, gps_df
        gc.collect()
        log.info("[%s] County aggregation complete", county_name)
        return

    # Collect gap_work for failed trips
    for tid in failed_ids:
        meta = trip_meta.get(tid)
        if not meta or not meta.get("gap_indices"):
            continue
        out_p        = meta["gps_path"].parent / \
            f"{county_name}_{meta['prefix']}_fid_aggregated.jsonl"
        gap_pairs    = []
        total_coords = len(meta["coords"])
        for gap_idx in meta["gap_indices"]:
            if gap_idx < total_coords - 1:
                gap_pairs.append((
                    meta["coords"][gap_idx],
                    meta["coords"][gap_idx + 1],
                    meta["ts"][gap_idx],
                    meta["ts"][gap_idx + 1],
                    gap_idx,
                    total_coords,
                ))
        if gap_pairs:
            gap_work.append((out_p, meta["gps_path"], meta["session_dir"],
                             meta["prefix"], gap_pairs))

    del matched, gps_df
    gc.collect()

    # ── Native FMM retry + STMatch gap bridging ───────────────────────────────
    filtered_gap_work: list[tuple] = []
    for out_p, gps_path, session_dir, prefix, gap_pairs in gap_work:
        filtered_pairs = []
        for gap_pair in gap_pairs:
            coord_before = gap_pair[0]
            coord_after  = gap_pair[1]

            # Skip if gap crosses a county boundary — STMatch only has the
            # current county's shapefile so it can't bridge across.
            county_before = assigner.assign(coord_before[0], coord_before[1])
            county_after  = assigner.assign(coord_after[0],  coord_after[1])
            if county_before != county_name or county_after != county_name:
                log.debug("Skipping cross-county gap (%s -> %s)",
                          county_before, county_after)
                continue

            # Skip if gap endpoints are more than 1km apart — STMatch can't
            # reliably bridge large gaps and may route through wrong roads.
            gap_dist_deg = ((coord_after[0] - coord_before[0])**2 +
                            (coord_after[1] - coord_before[1])**2) ** 0.5
            gap_dist_m   = gap_dist_deg * 111000
            if gap_dist_m > 1000:
                log.debug("Skipping large gap (%.0fm)", gap_dist_m)
                continue

            filtered_pairs.append(gap_pair)

        if filtered_pairs:
            filtered_gap_work.append((out_p, gps_path, session_dir, prefix,
                                      filtered_pairs))

    if failed_ids or filtered_gap_work:
        _run_native_postprocess({
            "county_name": county_name,
            "shp_path":    shp_path,
            "ubodt_path":  ubodt_path,
            "trip_meta":   trip_meta,
            "failed_ids":  failed_ids,
            "gap_work":    filtered_gap_work,
        })
    else:
        log.info("[%s] No failed trips to retry", county_name)

    gc.collect()
    log.info("[%s] County complete", county_name)
