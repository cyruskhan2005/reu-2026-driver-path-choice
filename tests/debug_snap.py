"""
Run from pipeline root: python tests/debug_snap2.py
Shows stop sign snapping final result with candidate edge vote counts.
- Blue markers  = successfully snapped signs
- Red markers   = unmatched signs
- Orange dots   = camera positions that photographed the sign
- Yellow line   = snapped edge (algorithm final result)
- Teal lines    = candidate edges with vote counts (grouped by name)
"""
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from pyproj import Transformer
from collections import defaultdict
import json

ROOT = Path(__file__).parent.parent

from roadnet.config import NODE_SNAP_M, MAX_EDGE_DIST_M
from roadnet.mapillary import (
    _snap_signs_to_network,
    _to_list,
    _camera_votes_bearing_aware,
    _group_votes_by_name,
    _closer_service_exists,
    _circular_mean,
    _bearing as _brg,
    _angular_diff as _adiff,
    MIN_RAW_CAM_POINTS,
)

print('Loading data...')
edges     = gpd.read_parquet(ROOT / 'sflorida_outputs/Broward_County/enriched_network.parquet')
nodes     = gpd.read_parquet(ROOT / 'sflorida_outputs/Broward_County/osm_nodes.parquet')
signs_raw = gpd.read_parquet(ROOT / 'sflorida_outputs/Broward_County/mly_signs_raw.parquet')

projected_crs = 'EPSG:26917'

print('Running snap...')
snapped     = _snap_signs_to_network(signs_raw, nodes, edges, projected_crs)
snapped_wgs = snapped.to_crs('EPSG:4326')
edges_proj  = edges.to_crs(projected_crs)
edges_wgs   = edges.to_crs('EPSG:4326')
nodes_proj  = nodes.to_crs(projected_crs)

edge_geoms     = edges_proj.geometry.values
edge_tree      = STRtree(edge_geoms)
node_geoms_arr = nodes_proj.geometry.values
node_tree_dbg  = STRtree(node_geoms_arr)

print('Building lookups...')
if 'osmid' in nodes_proj.columns:
    osmid_to_geom  = dict(zip(nodes_proj['osmid'].astype(int), nodes_proj.geometry))
    node_idx_osmid = nodes_proj['osmid'].astype(int).tolist()
else:
    osmid_to_geom  = dict(enumerate(nodes_proj.geometry))
    node_idx_osmid = list(range(len(nodes_proj)))

u_vals = edges_proj['u'].astype(int).tolist()
v_vals = edges_proj['v'].astype(int).tolist()
n2u: defaultdict[int, list] = defaultdict(list)
n2v: defaultdict[int, list] = defaultdict(list)
for idx2, (u, v) in enumerate(zip(u_vals, v_vals)):
    n2u[u].append(idx2)
    n2v[v].append(idx2)

t_fwd = Transformer.from_crs('EPSG:4326', projected_crs, always_xy=True)

def _is_unnamed(name: str) -> bool:
    return name.startswith('__unnamed_') or name.startswith('__uv_')

# Only stop signs
stops       = snapped_wgs[snapped_wgs['value'].str.contains('stop', case=False, na=False)].copy()
total_stops = len(stops)
print(f'Processing {total_stops} stop signs...')

sign_payloads  = []
edge_payloads  = {}
snapped_so_far = 0

for i, (_, row) in enumerate(stops.iterrows()):
    if i % 500 == 0:
        pct = round(100 * i / total_stops) if total_stops > 0 else 0
        print(f'  {i}/{total_stops} ({pct}%) | snapped so far: {snapped_so_far}', end='\r', flush=True)

    snapped_ei = row.get('snap_edge_idx')
    snapped_ei = int(snapped_ei) if snapped_ei is not None and snapped_ei == snapped_ei else None
    is_snapped = snapped_ei is not None

    # Camera positions
    camera_lngs   = _to_list(row.get('camera_lngs'))
    camera_lats   = _to_list(row.get('camera_lats'))
    camera_angles = _to_list(row.get('camera_angles'))
    camera_pts    = []
    for clng, clat in zip(camera_lngs, camera_lats):
        camera_pts.append([round(clat, 6), round(clng, 6)])

    # Camera bearing (stored per-road value from algorithm)
    raw_cb = row.get('camera_bearing')
    try:
        camera_bearing = round(float(raw_cb), 1) if raw_cb is not None and raw_cb == raw_cb else None
    except Exception:
        camera_bearing = None

    # Sign position in projected CRS
    sx, sy  = t_fwd.transform(row['lng'], row['lat'])
    pt_proj = Point(sx, sy)

    # Recompute bearing-aware camera votes — returns 4 values
    camera_votes, edge_camera_angles, edge_min_sign_dist, edge_raw_cam_points = \
        _camera_votes_bearing_aware(
            camera_lngs, camera_lats, camera_angles,
            edge_geoms, edge_tree, t_fwd,
            sign_pt=pt_proj,
        )

    # Group by road name
    name_votes, name_to_best_ei, name_camera_angles = _group_votes_by_name(
        camera_votes, edge_camera_angles, edges
    )
    total_name_votes = sum(name_votes.values())
    ei_to_name = {ei: name for name, ei in name_to_best_ei.items()}

    # Build candidate list grouped by name, sorted by grouped votes
    candidate_edges = []
    for name, grouped_votes in sorted(name_votes.items(), key=lambda x: -x[1])[:8]:
        ei           = name_to_best_ei[name]
        pct          = round(100 * grouped_votes / total_name_votes) if total_name_votes > 0 else 0
        display_name = name if not _is_unnamed(name) else '(unnamed)'
        coords_wgs   = [[round(y,6), round(x,6)] for x,y in edges_wgs.iloc[ei].geometry.coords]
        edge_payloads[int(ei)] = {'coords': coords_wgs, 'name': display_name}
        road_angles  = name_camera_angles.get(name, [])
        road_bearing = round(_circular_mean(road_angles), 1) if road_angles else None
        raw_count    = len(edge_raw_cam_points.get(ei, set()))
        candidate_edges.append({
            'idx':          int(ei),
            'votes':        grouped_votes,
            'edge_votes':   camera_votes.get(ei, 0),
            'pct':          pct,
            'name':         display_name,
            'road_bearing': road_bearing,
            'raw_cams':     raw_count,
        })

    # Snapped edge geometry
    if snapped_ei is not None and snapped_ei not in edge_payloads:
        raw_name   = str(edges.iloc[snapped_ei].get('name', '') or '')
        coords_wgs = [[round(y,6), round(x,6)] for x,y in edges_wgs.iloc[snapped_ei].geometry.coords]
        edge_payloads[snapped_ei] = {'coords': coords_wgs, 'name': raw_name or '(unnamed)'}

    # ── Trace rejection reason — mirrors new Strategy A logic ─────────────────
    reject_reason = None
    if not is_snapped:
        if total_name_votes == 0:
            reject_reason = "no camera votes at all"
        else:
            # Replicate Strategy A: find eligible edges
            eligible = []
            for ei, votes in camera_votes.items():
                raw_count = len(edge_raw_cam_points.get(ei, set()))
                if raw_count < MIN_RAW_CAM_POINTS:
                    continue
                u_osmid = int(edges_proj.iloc[ei]['u'])
                v_osmid = int(edges_proj.iloc[ei]['v'])
                u_geom  = osmid_to_geom.get(u_osmid)
                v_geom  = osmid_to_geom.get(v_osmid)
                d_u = pt_proj.distance(u_geom) if u_geom is not None else np.inf
                d_v = pt_proj.distance(v_geom) if v_geom is not None else np.inf
                if min(d_u, d_v) <= NODE_SNAP_M:
                    role   = "u_node" if d_u < d_v else "v_node"
                    osmid  = u_osmid  if d_u < d_v else v_osmid
                    n_dist = d_u      if d_u < d_v else d_v
                    eligible.append((ei, raw_count, role, osmid, votes, n_dist))

            if not eligible:
                # No edge passed the raw camera count + node distance filter
                max_raw_any = max(
                    (len(edge_raw_cam_points.get(ei, set())) for ei in camera_votes),
                    default=0
                )
                node_cands = node_tree_dbg.query(pt_proj.buffer(NODE_SNAP_M))
                nearby = [i for i in node_cands if pt_proj.distance(node_geoms_arr[i]) <= NODE_SNAP_M]

                if max_raw_any < MIN_RAW_CAM_POINTS:
                    reject_reason = (
                        f"Strategy A: all roads have <{MIN_RAW_CAM_POINTS} raw camera GPS points "
                        f"(max={max_raw_any}) — not enough camera coverage near sign"
                    )
                else:
                    reject_reason = (
                        f"Strategy A: roads with >={MIN_RAW_CAM_POINTS} cameras exist "
                        f"but none have a node within {NODE_SNAP_M}m of sign"
                    )

                # Check Strategy B fallback
                qualified = {
                    name: name_to_best_ei[name]
                    for name, total in name_votes.items()
                    if total >= 4
                }
                if not qualified:
                    reject_reason += f" | Strategy B: no road with >=4 votes"
                elif len(nearby) == 0:
                    reject_reason += f" | Strategy B: no node within {NODE_SNAP_M}m"
                elif len(nearby) > 1:
                    reject_reason += f" | Strategy B: {len(nearby)} nodes ambiguous"
                else:
                    nd_osmid = node_idx_osmid[nearby[0]]
                    qual_eis = set(qualified.values())
                    touching = [
                        ei for ei in n2u.get(nd_osmid, []) + n2v.get(nd_osmid, [])
                        if ei in qual_eis
                    ]
                    if not touching:
                        reject_reason += " | Strategy B: node doesn't touch qualified edge"
                    else:
                        nd_geom = osmid_to_geom.get(nd_osmid)
                        bearing_rejected = False
                        service_rejected = False
                        for ei in touching:
                            geom = edge_geoms[ei]
                            road_name   = ei_to_name.get(ei)
                            road_angles = name_camera_angles.get(road_name, []) if road_name else []
                            avg_road_cam = _circular_mean(road_angles) if road_angles else None
                            ad = float(row['aligned_direction'])
                            if avg_road_cam is not None:
                                raw_diff = _adiff(ad, avg_road_cam)
                                diff = raw_diff if raw_diff <= 90 else 180 - raw_diff
                            else:
                                d_node = geom.project(nd_geom)
                                d_out  = min(geom.length, d_node + 15)
                                p_out  = geom.interpolate(d_out)
                                road_brg = _brg(Point(nd_geom.x, nd_geom.y), Point(p_out.x, p_out.y))
                                diff = _adiff(ad, (road_brg + 180) % 360)
                            if diff > 40:
                                bearing_rejected = True
                                continue
                            snap_dist_b = pt_proj.distance(geom)
                            if _closer_service_exists(pt_proj, ei, snap_dist_b, edge_geoms, edge_tree, edges):
                                service_rejected = True
                        if service_rejected and not bearing_rejected:
                            reject_reason += " | Strategy B: closer service edge (parking lot)"
                        elif bearing_rejected:
                            reject_reason += " | Strategy B: bearing gate rejected all"
                        else:
                            reject_reason += " | Strategy B: unknown"
            else:
                # Had eligible edges — find max raw and check why winner failed
                max_raw = max(e[1] for e in eligible)
                top_eligible = [e for e in eligible if e[1] == max_raw]
                best_ei = max(top_eligible, key=lambda e: e[4] / max(edge_min_sign_dist.get(e[0], 160), 1))[0]

                road_name_a   = ei_to_name.get(best_ei)
                road_angles_a = name_camera_angles.get(road_name_a, []) if road_name_a else []
                avg_cam_angle = _circular_mean(road_angles_a) if road_angles_a else None
                ad = float(row['aligned_direction'])

                if avg_cam_angle is not None:
                    raw_diff     = _adiff(float(avg_cam_angle), ad)
                    bearing_diff = raw_diff if raw_diff <= 90 else 180 - raw_diff
                    if bearing_diff > 65:
                        reject_reason = (
                            f"Strategy A: best edge {best_ei} ({road_name_a}) "
                            f"failed bearing gate: mod180={bearing_diff:.1f}° > 65°"
                        )
                    else:
                        consensus_dist = pt_proj.distance(edge_geoms[best_ei])
                        if _closer_service_exists(pt_proj, best_ei, consensus_dist, edge_geoms, edge_tree, edges):
                            reject_reason = (
                                f"Strategy A: best edge {best_ei} rejected — closer service edge"
                            )
                        else:
                            reject_reason = f"Strategy A: best edge {best_ei} — unknown rejection"
                else:
                    reject_reason = (
                        f"Strategy A: best edge {best_ei} ({road_name_a}) "
                        f"— no camera angles for bearing gate"
                    )

    sign_payloads.append({
        'lat':            round(row.geometry.y, 6),
        'lon':            round(row.geometry.x, 6),
        'direction':      round(float(row['aligned_direction']), 1),
        'camera_bearing': camera_bearing,
        'snapped':        snapped_ei,
        'is_snapped':     is_snapped,
        'cameras':        camera_pts,
        'bearing_source': row.get('bearing_source', ''),
        'candidates':     candidate_edges,
        'total_votes':    total_name_votes,
        'reject_reason':  reject_reason,
    })

    if is_snapped:
        snapped_so_far += 1

snapped_count   = snapped_so_far
unsnapped_count = total_stops - snapped_count
print(f'\n{total_stops}/{total_stops} (100%) | snapped: {snapped_count} | unmatched: {unsnapped_count}')
print('Building HTML...')

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Stop Sign Debug</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  body,html{{margin:0;padding:0;height:100%;font-family:monospace}}
  #map{{width:100%;height:100vh}}
  #panel{{position:fixed;top:10px;right:10px;background:#1a1d27;color:#e8eaf0;
          padding:12px;border-radius:8px;z-index:1000;font-size:11px;
          max-width:400px;max-height:85vh;overflow-y:auto}}
  table{{border-collapse:collapse;width:100%;margin-top:6px}}
  td,th{{padding:2px 5px;border:1px solid #333;white-space:nowrap}}
  th{{background:#2a2d3a}}
  .snapped-row{{color:#ffdd00;font-weight:bold}}
  #legend{{position:fixed;bottom:20px;left:10px;background:#1a1d27;color:#e8eaf0;
           padding:10px 14px;border-radius:8px;z-index:1000;font-size:11px;line-height:1.8}}
  .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}
  #stats{{font-size:10px;color:#6b7280;margin-top:4px}}
  .reject{{color:#f4a261;margin-top:4px;word-break:break-word}}
</style>
</head>
<body>
<div id="panel">
  <b>Stop Sign Debug</b> (NODE_SNAP_M={NODE_SNAP_M}m, MIN_RAW_CAMS={MIN_RAW_CAM_POINTS})<br>
  <div id="stats">Snapped: {snapped_count} | Unmatched: {unsnapped_count}</div>
  <div id="info" style="color:#aaa;margin-top:6px">Hover a sign</div>
</div>
<div id="legend">
  <div><span class="dot" style="background:#4d9de0"></span>Snapped sign</div>
  <div><span class="dot" style="background:#e63946"></span>Unmatched sign</div>
  <div><span class="dot" style="background:#fd7e14"></span>Camera position</div>
  <div style="margin-top:4px">
    <span style="display:inline-block;width:20px;height:3px;background:#ff0;margin-right:6px;vertical-align:middle"></span>Snapped edge<br>
    <span style="display:inline-block;width:20px;height:3px;background:#2a9d8f;margin-right:6px;vertical-align:middle"></span>Candidates (grouped by name)
  </div>
</div>
<div id="map"></div>
<script>
const SIGNS = {json.dumps(sign_payloads)};
const EDGES = {json.dumps(edge_payloads)};

const map = L.map('map',{{preferCanvas:true}}).setView([26.148,-80.206],14);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
  {{attribution:'&copy; CARTO',subdomains:'abcd',maxZoom:19}}).addTo(map);

const signLayer = L.layerGroup().addTo(map);
const edgeLayer = L.layerGroup().addTo(map);
const camLayer  = L.layerGroup().addTo(map);

SIGNS.forEach(s => {{
  const color = s.is_snapped ? '#4d9de0' : '#e63946';
  const m = L.circleMarker([s.lat,s.lon],{{
    radius:7, color:'#fff', fillColor:color, fillOpacity:1, weight:2
  }});

  m.on('mouseover', () => {{
    edgeLayer.clearLayers();
    camLayer.clearLayers();

    (s.cameras || []).forEach(c => {{
      L.circleMarker(c, {{
        radius:5, color:'#fff', fillColor:'#fd7e14', fillOpacity:0.9, weight:1.5
      }}).bindTooltip('Camera position', {{sticky:true}}).addTo(camLayer);
    }});

    (s.candidates || []).forEach((c, i) => {{
      if (!EDGES[c.idx]) return;
      const isSnapped = c.idx === s.snapped;
      const color   = isSnapped ? '#ff0' : '#2a9d8f';
      const weight  = isSnapped ? 8 : Math.max(2, Math.round(c.pct / 12));
      const opacity = isSnapped ? 0.95 : 0.35 + (c.pct / 100) * 0.55;
      L.polyline(EDGES[c.idx].coords, {{color, weight, opacity}})
       .bindTooltip(
         `${{c.name}}<br>grouped votes=${{c.votes}} (${{c.pct}}%)<br>raw_cams=${{c.raw_cams}}<br>cam_brg=${{c.road_bearing !== null ? c.road_bearing + '°' : '—'}}`,
         {{sticky:true}}
       ).addTo(edgeLayer);
    }});

    if (s.snapped !== null && EDGES[s.snapped]) {{
      const inCandidates = (s.candidates || []).some(c => c.idx === s.snapped);
      if (!inCandidates) {{
        L.polyline(EDGES[s.snapped].coords, {{color:'#ff0', weight:8, opacity:.9}})
         .bindTooltip(`${{EDGES[s.snapped].name}} (snapped)`, {{sticky:true}})
         .addTo(edgeLayer);
      }}
    }}

    const cbStr   = s.camera_bearing !== null ? s.camera_bearing + '°' : 'none';
    const diff    = s.camera_bearing !== null
      ? Math.min(Math.abs(s.camera_bearing - s.direction) % 360,
                 360 - Math.abs(s.camera_bearing - s.direction) % 360)
      : null;
    const diffMod = diff !== null ? Math.min(diff, 180 - diff) : null;
    const diffStr = diffMod !== null ? ` (mod180=${{diffMod.toFixed(1)}}°)` : '';

    let html = `<b>aligned_direction=${{s.direction}}°</b><br>`;
    html += `camera_bearing=${{cbStr}}${{diffStr}}<br>`;
    html += `source=${{s.bearing_source}} | total grouped votes=${{s.total_votes}}<br>`;

    if (s.is_snapped) {{
      html += `<span style="color:#4d9de0">✓ Snapped → idx=${{s.snapped}}</span>`;
    }} else {{
      html += `<span style="color:#e63946">✗ Unmatched</span>`;
      if (s.reject_reason) {{
        html += `<div class="reject">⚠ ${{s.reject_reason}}</div>`;
      }}
    }}

    if (s.candidates && s.candidates.length > 0) {{
      html += `<table style="margin-top:6px">`;
      html += `<tr><th>idx</th><th>name</th><th>votes</th><th>%</th><th>raw</th><th>cam_brg</th></tr>`;
      s.candidates.forEach(c => {{
        const cls     = c.idx === s.snapped ? 'snapped-row' : '';
        const brg     = c.road_bearing !== null ? c.road_bearing + '°' : '—';
        const mod180  = c.road_bearing !== null
          ? Math.min(
              Math.abs(c.road_bearing - s.direction) % 360,
              360 - Math.abs(c.road_bearing - s.direction) % 360
            )
          : null;
        const mod180r = mod180 !== null ? Math.min(mod180, 180 - mod180) : null;
        const brgStr  = mod180r !== null
          ? `${{brg}}<br><span style="color:#aaa">(Δ${{mod180r.toFixed(0)}}°)</span>`
          : brg;
        const rawStyle = c.raw_cams < {MIN_RAW_CAM_POINTS} ? 'color:#e63946' : 'color:#2a9d8f';
        html += `<tr class="${{cls}}">
          <td>${{c.idx}}</td>
          <td>${{c.name}}</td>
          <td>${{c.votes}}</td>
          <td>${{c.pct}}%</td>
          <td style="${{rawStyle}}">${{c.raw_cams}}</td>
          <td>${{brgStr}}</td>
        </tr>`;
      }});
      html += `</table>`;
    }} else {{
      html += `<br><span style="color:#888">No camera votes</span>`;
    }}

    document.getElementById('info').innerHTML = html;
  }});

  m.addTo(signLayer);
}});
</script>
</body>
</html>"""

out = ROOT / 'visuals/debug_snap2.html'
out.parent.mkdir(exist_ok=True)
out.write_text(html, encoding='utf-8')
print(f'Written to {out}')
print('Open: xdg-open visuals/debug_snap2.html')