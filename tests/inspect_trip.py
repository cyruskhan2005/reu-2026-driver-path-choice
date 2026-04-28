"""
inspect_trip.py
===============
Discovers all *_fid_aggregated.jsonl files under a driver data root,
pairs each with its GPS file and the correct county shapefile, then
generates paginated HTML viewers — one file per 100 trips.

Usage (from pipeline root):
    python tests/inspect_trip.py   --drivers data/drivers   --outputs sflorida_outputs   --out visuals/trip_viewer.html
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

_GPS_COLOUR     = "#fd7e14"
_FMM_COLOUR     = "#1d3557"
_STMATCH_COLOUR = "#e63946"
_PAGE_SIZE      = 100

_COUNTY_DIR_MAP = {
    "Broward County":    "Broward_County",
    "Miami-Dade County": "Miami_Dade_County",
    "Palm Beach County": "Palm_Beach_County",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
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
                rows.append({
                    "ts":  d["@ts"],
                    "lon": loc["lon"],
                    "lat": loc["lat"],
                })
            except Exception:
                pass
    rows.sort(key=lambda r: r["ts"])
    return rows


def find_gaps(rows: list[dict], threshold_s: float) -> list[int]:
    gaps = []
    for i in range(len(rows) - 1):
        try:
            t0 = datetime.fromisoformat(rows[i]["ts"].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(rows[i+1]["ts"].replace("Z", "+00:00"))
            if (t1 - t0).total_seconds() > threshold_s:
                gaps.append(i)
        except Exception:
            pass
    return gaps


def load_aggregated(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    df = pd.DataFrame(rows)
    if not df.empty and "fid" in df.columns:
        df["fid"] = df["fid"].astype(int)
    return df


def county_from_agg_name(agg_path: Path) -> str | None:
    for county in _COUNTY_DIR_MAP:
        if agg_path.name.startswith(county):
            return county
    return None


def shp_path_for_county(county: str, outputs_root: Path) -> Path | None:
    subdir = _COUNTY_DIR_MAP.get(county)
    if subdir is None:
        return None
    p = outputs_root / subdir / "fmm" / "edges.shp"
    return p if p.exists() else None


# ─────────────────────────────────────────────────────────────────────────────
# Trip discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_trips(drivers_root: Path, outputs_root: Path) -> list[dict]:
    trips = []
    for agg_path in sorted(drivers_root.rglob("*_fid_aggregated.jsonl")):
        county = county_from_agg_name(agg_path)
        if county is None:
            continue
        shp = shp_path_for_county(county, outputs_root)
        if shp is None:
            continue
        county_prefix = county + "_"
        stem      = agg_path.stem
        remainder = stem[len(county_prefix):] if stem.startswith(county_prefix) else stem
        gps_stem  = remainder.replace("_fid_aggregated", "")
        gps_path  = agg_path.parent / f"{gps_stem}_gps.jsonl"
        if not gps_path.exists():
            continue
        trips.append({
            "agg":    agg_path,
            "gps":    gps_path,
            "shp":    shp,
            "county": county,
            "label":  f"{county} / {gps_stem}",
        })
    return trips


# ─────────────────────────────────────────────────────────────────────────────
# Shapefile cache
# ─────────────────────────────────────────────────────────────────────────────

_shp_cache: dict[str, dict[int, list]] = {}


def load_edge_map(shp_path: Path) -> dict[int, list]:
    key = str(shp_path)
    if key not in _shp_cache:
        print(f"    Loading shapefile {shp_path.parent.parent.name} …")
        shp = gpd.read_file(shp_path)
        if "fid" not in shp.columns:
            shp["fid"] = shp.index
        shp["fid"] = shp["fid"].astype(int)
        shp_wgs = shp.to_crs("EPSG:4326") if shp.crs and shp.crs.to_epsg() != 4326 else shp
        em: dict[int, list] = {}
        for _, row in shp_wgs.iterrows():
            fid   = int(row["fid"])
            geom  = row.geometry
            parts = list(geom.geoms) if geom.geom_type != "LineString" else [geom]
            em[fid] = [[[round(y, 6), round(x, 6)] for x, y in part.coords]
                       for part in parts]
        _shp_cache[key] = em
    return _shp_cache[key]


# ─────────────────────────────────────────────────────────────────────────────
# Build one trip payload
# ─────────────────────────────────────────────────────────────────────────────

def build_trip_payload(trip: dict, gap_threshold: float) -> dict:
    gps_rows    = load_gps(trip["gps"])
    gap_indices = find_gaps(gps_rows, gap_threshold)
    agg_df      = load_aggregated(trip["agg"])
    edge_map    = load_edge_map(trip["shp"])
    has_method  = "match_method" in agg_df.columns
    skip_attr   = {"seq_idx", "match_method", "fid"}

    gps_points = [[round(r["lat"], 6), round(r["lon"], 6)] for r in gps_rows]

    gaps = []
    for idx in gap_indices:
        try:
            t0    = datetime.fromisoformat(gps_rows[idx]["ts"].replace("Z", "+00:00"))
            t1    = datetime.fromisoformat(gps_rows[idx+1]["ts"].replace("Z", "+00:00"))
            gap_s = (t1 - t0).total_seconds()
        except Exception:
            gap_s = 0
        gaps.append({
            "b": [round(gps_rows[idx]["lat"], 6),   round(gps_rows[idx]["lon"], 6)],
            "a": [round(gps_rows[idx+1]["lat"], 6), round(gps_rows[idx+1]["lon"], 6)],
            "s": round(gap_s),
        })

    # Dedup edge coords within this trip
    local_coords: dict[int, list] = {}
    sort_col = "seq_idx" if "seq_idx" in agg_df.columns else agg_df.columns[0]
    edges_compact = []
    for _, row in agg_df.sort_values(sort_col).iterrows():
        fid    = int(row["fid"])
        method = row.get("match_method", "fmm") if has_method else "fmm"
        coords = edge_map.get(fid)
        if not coords:
            continue
        if fid not in local_coords:
            local_coords[fid] = coords
        attrs: dict = {}
        for col in agg_df.columns:
            if col in skip_attr:
                continue
            val = row.get(col)
            if pd.isna(val):
                continue
            if col == "sog":
                attrs["sog_mph"] = round(val * 0.621371, 2)
            elif col == "sog_variability":
                attrs["sog_variability_mph"] = round(val * 0.621371, 2)
            else:
                attrs[col] = round(val, 3) if isinstance(val, float) else val
        # u = start of edge (first point), v = end (last point)
        u_coord = coords[0][0]   if coords and coords[0] else None
        v_coord = coords[0][-1]  if coords and coords[0] else None
        edges_compact.append([fid, method, attrs, u_coord, v_coord])

    report   = build_report(agg_df, gps_rows, gap_indices, trip["agg"])
    start_ts = gps_rows[0]["ts"][:19].replace("T", " ") if gps_rows else ""
    end_ts   = gps_rows[-1]["ts"][11:19]                 if gps_rows else ""

    return {
        "label":    trip["label"],
        "gps":      gps_points,
        "start":    gps_points[0]  if gps_points else [],
        "end":      gps_points[-1] if gps_points else [],
        "start_ts": start_ts,
        "end_ts":   end_ts,
        "gaps":     gaps,
        "edges":    edges_compact,
        "coords":   {str(k): v for k, v in local_coords.items()},
        "report":   report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Text report
# ─────────────────────────────────────────────────────────────────────────────

def build_report(
    agg_df:      pd.DataFrame,
    gps_rows:    list[dict],
    gap_indices: list[int],
    agg_path:    Path,
) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"TRIP REPORT: {agg_path.stem}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("TRIP SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  GPS points   : {len(gps_rows)}")
    if gps_rows:
        lines.append(f"  Start        : {gps_rows[0]['ts'][:19].replace('T',' ')}")
        lines.append(f"  End          : {gps_rows[-1]['ts'][:19].replace('T',' ')}")
        try:
            t0 = datetime.fromisoformat(gps_rows[0]["ts"].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(gps_rows[-1]["ts"].replace("Z", "+00:00"))
            lines.append(f"  Duration     : {(t1-t0).total_seconds()/60:.1f} min")
        except Exception:
            pass
    lines.append(f"  Time gaps    : {len(gap_indices)}")
    for idx in gap_indices:
        try:
            t0    = datetime.fromisoformat(gps_rows[idx]["ts"].replace("Z", "+00:00"))
            t1    = datetime.fromisoformat(gps_rows[idx+1]["ts"].replace("Z", "+00:00"))
            gap_s = (t1 - t0).total_seconds()
            lines.append(f"    Gap at point {idx}: {gap_s:.0f}s "
                         f"({t0.strftime('%H:%M:%S')} -> {t1.strftime('%H:%M:%S')})")
        except Exception:
            pass
    lines.append(f"  Matched FIDs : {len(agg_df)}")
    has_method = "match_method" in agg_df.columns
    n_stmatch  = len(agg_df[agg_df["match_method"] == "stmatch"]) if has_method else 0
    lines.append(f"    FMM        : {len(agg_df) - n_stmatch}")
    lines.append(f"    STMatch    : {n_stmatch}")

    skip_cols = {"fid", "seq_idx", "match_method"}
    attr_cols = [c for c in agg_df.columns if c not in skip_cols]
    sort_col  = "seq_idx" if "seq_idx" in agg_df.columns else agg_df.columns[0]

    lines.append("")
    lines.append("ROAD SEGMENTS (in traversal order)")
    lines.append("-" * 40)
    for _, row in agg_df.sort_values(sort_col).iterrows():
        fid    = int(row["fid"])
        method = row.get("match_method", "fmm") if has_method else "fmm"
        lines.append(f"\n  FID {fid}  [{method}]")
        for col in attr_cols:
            val = row.get(col)
            if pd.isna(val):
                continue
            if col == "sog":
                lines.append(f"    {'sog_mph':<30} {val * 0.621371:.4f}")
            elif col == "sog_variability":
                lines.append(f"    {'sog_variability_mph':<30} {val * 0.621371:.4f}")
            elif isinstance(val, float):
                lines.append(f"    {col:<30} {val:.4f}")
            else:
                lines.append(f"    {col:<30} {val}")

    lines.append("")
    lines.append("TRIP AGGREGATES")
    lines.append("-" * 40)
    for col in agg_df[attr_cols].select_dtypes(include="number").columns:
        vals = agg_df[col].dropna()
        if vals.empty:
            continue
        lines.append(f"  {col}")
        lines.append(f"    mean={vals.mean():.4f}  min={vals.min():.4f}  "
                     f"max={vals.max():.4f}  std={vals.std():.4f}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Generate one page HTML
# ─────────────────────────────────────────────────────────────────────────────

def make_page_html(
    payloads:    list[dict],
    page_num:    int,
    total_pages: int,
    page_size:   int,
    total_trips: int,
    base_name:   str,
) -> str:
    NAV_H      = 54
    total      = len(payloads)
    page_start = (page_num - 1) * page_size + 1
    page_end   = min(page_num * page_size, total_trips)
    prev_page  = f"{base_name}_{page_num-1:03d}.html" if page_num > 1        else ""
    next_page  = f"{base_name}_{page_num+1:03d}.html" if page_num < total_pages else ""

    prev_btn = (f'<a class="page-btn" href="{prev_page}">\u2190 Pg {page_num-1}</a>'
                if prev_page else
                f'<span class="page-btn" style="opacity:.2">\u2190 Pg {page_num-1}</span>')
    next_btn = (f'<a class="page-btn" href="{next_page}">Pg {page_num+1} \u2192</a>'
                if next_page else
                f'<span class="page-btn" style="opacity:.2">Pg {page_num+1} \u2192</span>')

    payloads_json = json.dumps(payloads)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Trips — Page {page_num}/{total_pages}</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@300;400;600&display=swap');
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  :root{{--bg:#0f1117;--surface:#1a1d27;--border:#2a2d3a;--accent:#fd7e14;--text:#e8eaf0;--muted:#6b7280;--nav:{NAV_H}px;--panel:280px}}
  html,body{{height:100%;overflow:hidden;font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text)}}
  body{{display:flex;flex-direction:column}}
  #navbar{{height:var(--nav);min-height:var(--nav);background:var(--surface);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 14px;gap:8px;flex-shrink:0;z-index:1000}}
  #navbar h1{{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;color:var(--accent);letter-spacing:.1em;text-transform:uppercase;white-space:nowrap}}
  #trip-label{{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
  #counter{{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted);white-space:nowrap}}
  .nav-btn{{background:var(--border);border:1px solid var(--border);color:var(--text);font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;padding:7px 14px;border-radius:5px;cursor:pointer;transition:background .15s,border-color .15s,color .15s;white-space:nowrap;text-decoration:none;display:inline-flex;align-items:center}}
  .nav-btn:hover{{background:var(--accent);border-color:var(--accent);color:#000}}
  .nav-btn:disabled{{opacity:.25;cursor:default}}
  .nav-btn:disabled:hover{{background:var(--border);border-color:var(--border);color:var(--text)}}
  .page-btn{{background:transparent;border:1px solid var(--border);color:var(--muted);font-family:'JetBrains Mono',monospace;font-size:11px;padding:6px 10px;border-radius:5px;cursor:pointer;text-decoration:none;display:inline-flex;align-items:center;white-space:nowrap;transition:all .15s}}
  .page-btn:hover{{border-color:var(--accent);color:var(--accent)}}
  .sep{{color:var(--border);font-size:18px;line-height:1}}
  .pg-info{{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted);white-space:nowrap}}
  #report-toggle{{background:transparent;border:1px solid var(--border);color:var(--muted);font-family:'JetBrains Mono',monospace;font-size:11px;padding:7px 12px;border-radius:5px;cursor:pointer;transition:all .15s;white-space:nowrap}}
  #report-toggle:hover{{border-color:var(--accent);color:var(--accent)}}
  #jump-wrap{{display:flex;align-items:center;gap:5px;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted);white-space:nowrap}}
  #jump-input{{background:var(--border);border:1px solid var(--border);color:var(--text);font-family:'JetBrains Mono',monospace;font-size:11px;padding:6px 6px;border-radius:5px;width:46px;text-align:center;outline:none}}
  #jump-input:focus{{border-color:var(--accent)}}
  #content{{display:flex;width:100%;height:calc(100vh - {NAV_H}px);overflow:hidden}}
  #map{{flex:1;min-width:0;height:100%}}
  #report-panel{{width:var(--panel);min-width:var(--panel);background:var(--surface);border-left:1px solid var(--border);display:none;flex-direction:column;overflow:hidden}}
  #report-panel.open{{display:flex}}
  #report-header{{padding:12px 14px 10px;border-bottom:1px solid var(--border);font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;color:var(--accent);letter-spacing:.1em;text-transform:uppercase;flex-shrink:0}}
  #report-body{{flex:1;overflow-y:auto;padding:12px 14px;font-family:'JetBrains Mono',monospace;font-size:10px;line-height:1.75;color:var(--muted);white-space:pre-wrap;word-break:break-word}}
  #report-body::-webkit-scrollbar{{width:4px}}
  #report-body::-webkit-scrollbar-track{{background:var(--surface)}}
  #report-body::-webkit-scrollbar-thumb{{background:var(--border);border-radius:2px}}
  .leaflet-container{{background:#1a1d27}}
  .leaflet-tooltip{{font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.6;background:#1a1d27;color:#e8eaf0;border:1px solid #2a2d3a;border-radius:6px;padding:8px 10px;box-shadow:0 4px 12px rgba(0,0,0,.4);max-width:340px}}
  .leaflet-tooltip-top::before{{border-top-color:#2a2d3a}}
</style>
</head>
<body>
<div id="navbar">
  <h1>Trips</h1>
  <span id="trip-label">—</span>
  <span id="counter">1/{total}</span>
  <div id="jump-wrap">go to<input id="jump-input" type="number" min="1" max="{total}" value="1"></div>
  <button class="nav-btn" id="btn-prev" onclick="navigate(-1)">&#8592;</button>
  <button class="nav-btn" id="btn-next" onclick="navigate(1)">&#8594;</button>
  <span class="sep">|</span>
  {prev_btn}
  <span class="pg-info">pg {page_num}/{total_pages} &nbsp;({page_start}–{page_end} of {total_trips})</span>
  {next_btn}
  <span class="sep">|</span>
  <button id="report-toggle" onclick="toggleReport()">Report</button>
</div>
<div id="content">
  <div id="map"></div>
  <div id="report-panel">
    <div id="report-header">Segment Report</div>
    <div id="report-body"></div>
  </div>
</div>
<script>
const PAYLOADS={payloads_json};
const TOTAL={total};
const GPS="{_GPS_COLOUR}",FMM="{_FMM_COLOUR}",STM="{_STMATCH_COLOUR}";
let cur=0,map,lg;
const ji=document.getElementById('jump-input');

function tip(fid,method,attrs){{
  let s='<b>FID '+fid+'</b> ['+method+']';
  for(const[k,v]of Object.entries(attrs))s+='<br>'+k+': '+v;
  return s;
}}

window.addEventListener('load',function(){{
  map=L.map('map',{{zoomControl:true,preferCanvas:true}});
  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
    {{attribution:'&copy; CARTO',subdomains:'abcd',maxZoom:19}}).addTo(map);
  lg=L.layerGroup().addTo(map);
  show(0);
}});

function render(idx){{
  lg.clearLayers();
  const p=PAYLOADS[idx];
  if(!p||!p.gps||!p.gps.length){{document.getElementById('report-body').textContent=p?p.report:'No data';return;}}
  L.polyline(p.gps,{{color:GPS,weight:2,opacity:.8}}).bindTooltip('Raw GPS',{{sticky:true}}).addTo(lg);
  if(p.start&&p.start.length===2)
    L.circleMarker(p.start,{{radius:7,color:'#2d6a4f',fillColor:'#2d6a4f',fillOpacity:1,weight:2}}).bindTooltip('Start '+p.start_ts).addTo(lg);
  if(p.end&&p.end.length===2)
    L.circleMarker(p.end,{{radius:7,color:'#6d023f',fillColor:'#6d023f',fillOpacity:1,weight:2}}).bindTooltip('End '+p.end_ts).addTo(lg);
  (p.gaps||[]).forEach(function(g){{
    L.circleMarker(g.b,{{radius:6,color:GPS,fillColor:GPS,fillOpacity:.9,weight:2}}).bindTooltip('Gap \u2014 '+g.s+'s').addTo(lg);
    L.circleMarker(g.a,{{radius:6,color:GPS,fillColor:'#fff',fillOpacity:.9,weight:2}}).bindTooltip('Resumed after '+g.s+'s').addTo(lg);
  }});
  const co=p.coords||{{}};
  (p.edges||[]).forEach(function(e){{
    const fid=e[0],method=e[1],attrs=e[2],u=e[3],v=e[4];
    const colour=method==='stmatch'?STM:FMM;
    const parts=co[String(fid)];
    if(!parts)return;
    const t=tip(fid,method,attrs);
    parts.forEach(function(part){{L.polyline(part,{{color:colour,weight:6,opacity:.85}}).bindTooltip(t,{{sticky:true}}).addTo(lg);}});
    if(u&&attrs.has_stop_sign_u)L.circleMarker(u,{{radius:8,color:'#fff',fillColor:'#e63946',fillOpacity:1,weight:2}}).bindTooltip('Stop sign (u)').addTo(lg);
    if(v&&attrs.has_stop_sign_v)L.circleMarker(v,{{radius:8,color:'#fff',fillColor:'#e63946',fillOpacity:1,weight:2}}).bindTooltip('Stop sign (v)').addTo(lg);
    if(u&&attrs.has_yield_u)L.circleMarker(u,{{radius:8,color:'#fff',fillColor:'#f4a261',fillOpacity:1,weight:2}}).bindTooltip('Yield (u)').addTo(lg);
    if(v&&attrs.has_yield_v)L.circleMarker(v,{{radius:8,color:'#fff',fillColor:'#f4a261',fillOpacity:1,weight:2}}).bindTooltip('Yield (v)').addTo(lg);
    if(u&&attrs.has_traffic_signal_u)L.circleMarker(u,{{radius:8,color:'#fff',fillColor:'#2a9d8f',fillOpacity:1,weight:2}}).bindTooltip('Traffic signal (u)').addTo(lg);
    if(v&&attrs.has_traffic_signal_v)L.circleMarker(v,{{radius:8,color:'#fff',fillColor:'#2a9d8f',fillOpacity:1,weight:2}}).bindTooltip('Traffic signal (v)').addTo(lg);
  }});
  try{{const b=L.latLngBounds(p.gps);if(b.isValid())map.fitBounds(b.pad(.05));}}catch(e){{}}
  document.getElementById('report-body').textContent=p.report||'';
}}

function show(idx){{
  cur=Math.max(0,Math.min(idx,TOTAL-1));
  const p=PAYLOADS[cur];
  document.getElementById('trip-label').textContent=p?p.label:'—';
  document.getElementById('counter').textContent=(cur+1)+'/'+TOTAL;
  document.getElementById('btn-prev').disabled=cur===0;
  document.getElementById('btn-next').disabled=cur===TOTAL-1;
  ji.value=cur+1;
  render(cur);
}}

function navigate(d){{show(cur+d);}}
function toggleReport(){{
  document.getElementById('report-panel').classList.toggle('open');
  setTimeout(function(){{if(map)map.invalidateSize();}},310);
}}
ji.addEventListener('change',function(){{
  const v=parseInt(this.value,10);
  if(!isNaN(v)&&v>=1&&v<=TOTAL)show(v-1);
}});
document.addEventListener('keydown',function(e){{
  if(e.target===ji)return;
  if(e.key==='ArrowRight'||e.key==='ArrowDown')navigate(1);
  if(e.key==='ArrowLeft'||e.key==='ArrowUp')navigate(-1);
}});
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Build viewer
# ─────────────────────────────────────────────────────────────────────────────

def build_viewer(
    trips:         list[dict],
    out_path:      Path,
    gap_threshold: float,
    page_size:     int = _PAGE_SIZE,
) -> None:
    total       = len(trips)
    total_pages = (total + page_size - 1) // page_size
    out_dir     = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name   = out_path.stem

    print(f"Building {total} trips → {total_pages} pages of {page_size}")

    for page_num in range(1, total_pages + 1):
        start      = (page_num - 1) * page_size
        end        = min(start + page_size, total)
        page_trips = trips[start:end]
        print(f"  Page {page_num}/{total_pages} (trips {start+1}–{end}) …")

        payloads = []
        for trip in page_trips:
            try:
                payloads.append(build_trip_payload(trip, gap_threshold))
            except Exception as e:
                print(f"    ERROR {trip['label']}: {e}")
                payloads.append({
                    "label": trip["label"], "gps": [], "start": [], "end": [],
                    "start_ts": "", "end_ts": "", "gaps": [], "edges": [],
                    "coords": {}, "report": f"Error: {e}",
                })

        page_path = out_dir / f"{base_name}_{page_num:03d}.html"
        page_path.write_text(
            make_page_html(payloads, page_num, total_pages, page_size, total, base_name),
            encoding="utf-8",
        )
        size_mb = page_path.stat().st_size / 1024 / 1024
        print(f"    → {page_path.name} ({size_mb:.1f} MB)")

    first = out_dir / f"{base_name}_001.html"
    print(f"\nDone. Open: xdg-open {first}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GPS trip viewer — paginated HTML")
    parser.add_argument("--drivers",       default=None)
    parser.add_argument("--outputs",       default="sflorida_outputs")
    parser.add_argument("--gps",           default=None)
    parser.add_argument("--agg",           default=None)
    parser.add_argument("--shp",           default=None)
    parser.add_argument("--gap_threshold", type=float, default=60.0)
    parser.add_argument("--page_size",     type=int,   default=_PAGE_SIZE)
    parser.add_argument("--out",           default=None)
    args = parser.parse_args()

    if args.drivers:
        trips    = discover_trips(Path(args.drivers), Path(args.outputs))
        if not trips:
            print("No trips found")
            return
        build_viewer(trips, Path(args.out or "visuals/trip_viewer.html"),
                     args.gap_threshold, args.page_size)

    elif args.gps and args.agg and args.shp:
        trips = [{
            "agg":    Path(args.agg),
            "gps":    Path(args.gps),
            "shp":    Path(args.shp),
            "county": county_from_agg_name(Path(args.agg)) or "Unknown",
            "label":  Path(args.gps).stem,
        }]
        build_viewer(trips,
                     Path(args.out or f"visuals/{Path(args.gps).stem}_inspect.html"),
                     args.gap_threshold, args.page_size)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()