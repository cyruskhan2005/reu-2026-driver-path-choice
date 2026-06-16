"""
test_trip.py
============
Simulates the full FMM + retry split-match + STMatch gap bridging pipeline
on a single *_gps.jsonl file, without running the whole pipeline.

Usage:
    python test_trip.py \
        --gps  "../data/drivers/.../...._gps.jsonl" \
        --shp  ../sflorida_outputs/Broward_County/fmm/edges.shp \
        --ubodt ../sflorida_outputs/Broward_County/fmm/ubodt.txt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import gc
from datetime import datetime
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

# Match pipeline constants
_DEG               = 1.1e5
FMM_K              = 16
FMM_RADIUS_M       = 300
FMM_ERROR_M        = 50
FMM_RETRY_RADIUS_M = 700
FMM_RETRY_ERROR_M  = 100
FMM_MIN_SEGMENT    = 10
FMM_SKIP_ON_GAP    = 5
STM_K              = 8
STM_RADIUS_M       = 300
STM_ERROR_M        = 50
STM_VMAX_MS        = 40.0
STM_FACTOR         = 1.5
GPS_GAP_THRESHOLD  = 60.0

_GPS_COLOUR     = "#adb5bd"
_FMM_COLOUR     = "#1d3557"
_STMATCH_COLOUR = "#e63946"


# ─────────────────────────────────────────────────────────────────────────────
# GPS loading
# ─────────────────────────────────────────────────────────────────────────────

def load_gps(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                d   = json.loads(line)
                loc = d.get("loc", {})
                if not isinstance(loc, dict):
                    continue
                ts_iso = d.get("@ts", "")
                ts_dt  = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
                ts_epoch = int(ts_dt.timestamp())
                rows.append({
                    "ts_iso":  ts_iso,
                    "ts_dt":   ts_dt,
                    "ts_epoch": ts_epoch,
                    "lon":     loc["lon"],
                    "lat":     loc["lat"],
                })
            except Exception:
                pass
    return sorted(rows, key=lambda r: r["ts_dt"])


def find_gaps(rows: list[dict]) -> list[int]:
    return [
        i for i in range(len(rows) - 1)
        if (rows[i+1]["ts_dt"] - rows[i]["ts_dt"]).total_seconds() > GPS_GAP_THRESHOLD
    ]


# ─────────────────────────────────────────────────────────────────────────────
# FMM CLI
# ─────────────────────────────────────────────────────────────────────────────

def run_fmm_cli(shp: str, ubodt: str, gps_csv: str, out_csv: str,
                radius_m=FMM_RADIUS_M, error_m=FMM_ERROR_M, k=FMM_K) -> bool:
    cmd = [
        "fmm",
        "--network", shp, "--network_id", "fid", "--source", "u", "--target", "v",
        "--ubodt", ubodt,
        "--gps", gps_csv, "--gps_point",
        "--gps_id", "id", "--gps_x", "lon", "--gps_y", "lat",
        "--gps_timestamp", "timestamp",
        "--output", out_csv, "--output_fields", "opath",
        "-k", str(k),
        "-r", str(radius_m / _DEG),
        "-e", str(error_m  / _DEG),
        "--reverse_tolerance", "1", "--use_omp", "-l", "2",
    ]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
    for line in proc.stderr:
        pass   # suppress FMM noise
    proc.wait()
    return os.path.exists(out_csv) and os.path.getsize(out_csv) > 0


# ─────────────────────────────────────────────────────────────────────────────
# FMM retry split-match (mirrors _match_with_splits in fmm_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def match_with_splits(coords, model, config) -> list[tuple]:
    from fmm import FastMapMatchConfig  # noqa
    results = []
    remaining = list(coords)
    while len(remaining) >= FMM_MIN_SEGMENT:
        try:
            r = model.match_wkt(LineString(remaining).wkt, config)
            opath = list(r.opath)
        except Exception:
            opath = []
        if opath:
            results.extend(zip(remaining[:len(opath)], opath))
            break
        lo, hi = FMM_MIN_SEGMENT, len(remaining)
        while lo < hi - 1:
            mid = (lo + hi) // 2
            try:
                rt = model.match_wkt(LineString(remaining[:mid]).wkt, config)
                opath_t = list(rt.opath)
            except Exception:
                opath_t = []
            if opath_t:
                lo = mid
            else:
                hi = mid
        try:
            rg = model.match_wkt(LineString(remaining[:lo]).wkt, config)
            opath_g = list(rg.opath)
        except Exception:
            opath_g = []
        if not opath_g:
            break
        results.extend(zip(remaining[:len(opath_g)], opath_g))
        gap_end = lo + FMM_SKIP_ON_GAP
        results.extend((c, -1) for c in remaining[len(opath_g):gap_end])
        remaining = remaining[gap_end:]
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STMatch gap bridging
# ─────────────────────────────────────────────────────────────────────────────

def stmatch_gap(coord_before, coord_after, model, config) -> list[int]:
    try:
        result = model.match_wkt(LineString([coord_before, coord_after]).wkt, config)
        return list(result.cpath)
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Map
# ─────────────────────────────────────────────────────────────────────────────

def build_map(gps_rows, gap_indices, fmm_fids, stmatch_fids, shp_wgs, out_html):
    lats   = [r["lat"] for r in gps_rows]
    lons   = [r["lon"] for r in gps_rows]
    m = folium.Map(location=[sum(lats)/len(lats), sum(lons)/len(lons)],
                   zoom_start=13, tiles="CartoDB positron")

    # Raw GPS
    folium.PolyLine([[r["lat"], r["lon"]] for r in gps_rows],
                    color=_GPS_COLOUR, weight=2, opacity=0.7, tooltip="Raw GPS").add_to(m)
    folium.CircleMarker([gps_rows[0]["lat"],  gps_rows[0]["lon"]],
                        radius=6, color="#2d6a4f", fill=True, fill_color="#2d6a4f",
                        tooltip=f"Start {gps_rows[0]['ts_dt'].strftime('%H:%M:%S')}").add_to(m)
    folium.CircleMarker([gps_rows[-1]["lat"], gps_rows[-1]["lon"]],
                        radius=6, color="#6d023f", fill=True, fill_color="#6d023f",
                        tooltip=f"End {gps_rows[-1]['ts_dt'].strftime('%H:%M:%S')}").add_to(m)

    # Gap markers
    for idx in gap_indices:
        gap_s = (gps_rows[idx+1]["ts_dt"] - gps_rows[idx]["ts_dt"]).total_seconds()
        folium.Marker([gps_rows[idx]["lat"],   gps_rows[idx]["lon"]],
                      icon=folium.Icon(color="orange", icon="pause", prefix="fa"),
                      tooltip=f"Gap start — {gap_s:.0f}s").add_to(m)
        folium.Marker([gps_rows[idx+1]["lat"], gps_rows[idx+1]["lon"]],
                      icon=folium.Icon(color="orange", icon="play", prefix="fa"),
                      tooltip=f"Gap end — {gap_s:.0f}s").add_to(m)

    # FMM edges
    all_fids = set(fmm_fids) | set(stmatch_fids)
    for _, edge in shp_wgs[shp_wgs["fid"].isin(all_fids)].iterrows():
        fid    = int(edge["fid"])
        colour = _STMATCH_COLOUR if fid in stmatch_fids else _FMM_COLOUR
        method = "stmatch" if fid in stmatch_fids else "fmm"
        geom   = edge.geometry
        parts  = list(geom.geoms) if geom.geom_type != "LineString" else [geom]
        for part in parts:
            folium.PolyLine([[y, x] for x, y in part.coords],
                            color=colour, weight=6, opacity=0.85,
                            tooltip=f"FID {fid} [{method}]").add_to(m)

    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:white;padding:12px;border-radius:8px;
                border:1px solid #ccc;font-size:13px;line-height:1.9">
        <b>Pipeline simulation</b><br>
        <span style="color:{_GPS_COLOUR}">━━</span> Raw GPS ({len(gps_rows)} pts)<br>
        <span style="color:{_FMM_COLOUR}">━━</span> FMM ({len(fmm_fids)} FIDs)<br>
        <span style="color:{_STMATCH_COLOUR}">━━</span> STMatch gap bridge ({len(stmatch_fids)} FIDs)<br>
        🟠 Time gap (&gt;{GPS_GAP_THRESHOLD:.0f}s)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(out_html)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Simulate FMM+STMatch pipeline on a single trip")
    parser.add_argument("--gps",   required=True)
    parser.add_argument("--shp",   required=True)
    parser.add_argument("--ubodt", required=True)
    parser.add_argument("--tmp",   default="/tmp/test_trip")
    args = parser.parse_args()

    from fmm import Network, NetworkGraph, FastMapMatch, UBODT, FastMapMatchConfig, STMATCH, STMATCHConfig

    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    gps_path = Path(args.gps)

    # ── Load GPS ──────────────────────────────────────────────────────────────
    print(f"Loading GPS: {gps_path.name}")
    gps_rows   = load_gps(gps_path)
    gap_indices = find_gaps(gps_rows)
    coords     = [(r["lon"], r["lat"]) for r in gps_rows]
    print(f"  {len(gps_rows)} points  "
          f"{gps_rows[0]['ts_dt'].strftime('%H:%M:%S')} → "
          f"{gps_rows[-1]['ts_dt'].strftime('%H:%M:%S')}")
    for idx in gap_indices:
        gap_s = (gps_rows[idx+1]["ts_dt"] - gps_rows[idx]["ts_dt"]).total_seconds()
        print(f"  Gap at point {idx}: {gap_s:.0f}s")

    # ── Write GPS CSV (epoch timestamps for FMM) ──────────────────────────────
    gps_csv = str(tmp / "gps.csv")
    fmm_out = str(tmp / "matched.csv")
    pd.DataFrame([{
        "id": 0, "lon": r["lon"], "lat": r["lat"],
        "timestamp": r["ts_epoch"], "point_idx": i,
    } for i, r in enumerate(gps_rows)])[["id","lon","lat","timestamp","point_idx"]]\
      .to_csv(gps_csv, index=False, sep=";")

    # ── Step 1: FMM CLI ───────────────────────────────────────────────────────
    print("\n── Step 1: FMM CLI ──────────────────────────────────────────────")
    ok = run_fmm_cli(args.shp, args.ubodt, gps_csv, fmm_out)
    fmm_fids: list[int] = []
    fmm_failed = True

    if ok:
        matched = pd.read_csv(fmm_out, sep=";")
        opath_str = str(matched.iloc[0].get("opath", ""))
        if opath_str and opath_str != "nan":
            fmm_fids = [int(x) for x in opath_str.split(",") if x.strip().lstrip("-").isdigit()]
            fmm_failed = len(fmm_fids) == 0

    print(f"  FMM result: {len(fmm_fids)} FIDs matched  {'✓' if not fmm_failed else '✗ failed'}")

    # ── Step 2: Retry split-match (if FMM failed) ─────────────────────────────
    print("\n── Step 2: Retry split-match ────────────────────────────────────")
    stmatch_fids: set[int] = set()

    if fmm_failed:
        net    = Network(args.shp, "fid", "u", "v")
        graph  = NetworkGraph(net)
        ubodt  = UBODT.read_ubodt_csv(args.ubodt)
        model  = FastMapMatch(net, graph, ubodt)
        config = FastMapMatchConfig(FMM_K, FMM_RETRY_RADIUS_M/_DEG,
                                    FMM_RETRY_ERROR_M/_DEG, reverse_tolerance=1)
        results = match_with_splits(coords, model, config)
        fmm_fids = [fid for _, fid in results if fid != -1]
        del model, ubodt
        gc.collect()
        print(f"  Split-match result: {len(fmm_fids)} FIDs rescued")
    else:
        # Still need net+graph for STMatch
        net   = Network(args.shp, "fid", "u", "v")
        graph = NetworkGraph(net)
        print("  FMM succeeded — skipping retry")

    # ── Step 3: STMatch gap bridging ──────────────────────────────────────────
    print("\n── Step 3: STMatch gap bridging ─────────────────────────────────")
    if gap_indices:
        stm_model  = STMATCH(net, graph)
        stm_config = STMATCHConfig(STM_K, STM_RADIUS_M/_DEG, STM_ERROR_M/_DEG,
                                   STM_VMAX_MS, STM_FACTOR)
        for idx in gap_indices:
            gap_s = (gps_rows[idx+1]["ts_dt"] - gps_rows[idx]["ts_dt"]).total_seconds()
            cpath = stmatch_gap(coords[idx], coords[idx+1], stm_model, stm_config)
            print(f"  Gap {idx} ({gap_s:.0f}s): STMatch found {len(cpath)} bridging FIDs")
            stmatch_fids.update(cpath)
        del stm_model
    else:
        print("  No gaps found — STMatch not needed")

    del net, graph
    gc.collect()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Summary ──────────────────────────────────────────────────────")
    print(f"  FMM FIDs:     {len(fmm_fids)}")
    print(f"  STMatch FIDs: {len(stmatch_fids)}")
    print(f"  Total unique: {len(set(fmm_fids) | stmatch_fids)}")

    # ── Map ───────────────────────────────────────────────────────────────────
    print(f"\n── Building map ─────────────────────────────────────────────────")
    shp = gpd.read_file(args.shp)
    if "fid" not in shp.columns:
        shp["fid"] = shp.index
    shp["fid"] = shp["fid"].astype(int)
    shp_wgs = shp.to_crs("EPSG:4326") if shp.crs and shp.crs.to_epsg() != 4326 else shp

    Path("visuals").mkdir(exist_ok=True)
    out_html = str(Path("visuals") / f"{gps_path.stem}_pipeline_sim.html")
    build_map(gps_rows, gap_indices, set(fmm_fids), stmatch_fids, shp_wgs, out_html)
    print(f"Map saved → {out_html}")
    print(f"Open with: xdg-open {out_html}")


if __name__ == "__main__":
    main()