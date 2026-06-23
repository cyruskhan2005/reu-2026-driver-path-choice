"""
Run this from the pipeline root:
    python tests/visualize_snap.py
Generates visuals/snap_debug.html
"""
from pathlib import Path
import json
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from pyproj import Transformer

ROOT = Path(__file__).parent.parent

edges = gpd.read_parquet(ROOT / 'sflorida_outputs/Broward_County/osm_edges.parquet')
nodes = gpd.read_parquet(ROOT / 'sflorida_outputs/Broward_County/osm_nodes.parquet')
signs_raw = gpd.read_parquet(ROOT / 'sflorida_outputs/Broward_County/mly_signs_raw.parquet')

from roadnet.mapillary import _snap_signs_to_network
projected_crs = 'EPSG:26917'

print('Running snap...')
snapped = _snap_signs_to_network(signs_raw, nodes, edges, projected_crs)
snapped_wgs = snapped.to_crs('EPSG:4326')
edges_wgs   = edges.to_crs('EPSG:4326')

# Build edge lookup: idx -> {fid, u, v, coords, name}
print('Building edge data...')
edge_data = {}
for i, row in edges_wgs.iterrows():
    coords = [[round(y,6), round(x,6)] for x,y in row.geometry.coords]
    edge_data[i] = {
        'u': int(row.u),
        'v': int(row.v),
        'name': str(row.get('name', '') or ''),
        'coords': coords,
    }

# Build sign payloads
print('Building sign data...')
sign_payloads = []
for i, row in snapped_wgs.iterrows():
    ei = row['snap_edge_idx']
    sign_payloads.append({
        'lat':       round(row.geometry.y, 6),
        'lon':       round(row.geometry.x, 6),
        'value':     row['value'],
        'direction': round(float(row['aligned_direction']), 1),
        'snap_type': row['snap_type'],
        'edge_idx':  int(ei) if ei is not None and not (isinstance(ei, float) and np.isnan(ei)) else None,
        'role':      row['mly:snap_edge_role'],
        'label':     row['mly:sign_label'],
    })

# Only include edges that have at least one snapped sign
used_edge_idxs = {s['edge_idx'] for s in sign_payloads if s['edge_idx'] is not None}
edge_subset = {k: v for k, v in edge_data.items() if k in used_edge_idxs}

print(f'{len(sign_payloads)} signs, {len(edge_subset)} edges with snapped signs')

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Sign Snap Debug</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  body,html{{margin:0;padding:0;height:100%}}
  #map{{width:100%;height:100vh}}
  #controls{{position:fixed;top:10px;right:10px;background:#1a1d27;color:#e8eaf0;
             padding:12px;border-radius:8px;z-index:1000;font-family:monospace;font-size:12px;
             max-width:280px}}
  #controls label{{display:block;margin:4px 0;cursor:pointer}}
  .legend{{margin-top:8px;border-top:1px solid #333;padding-top:8px}}
  .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}
</style>
</head>
<body>
<div id="controls">
  <b>Sign Snap Debug</b><br>
  <label><input type="checkbox" id="show-signs" checked> Show signs</label>
  <label><input type="checkbox" id="show-edges" checked> Show snapped edges</label>
  <label><input type="checkbox" id="stops-only"> Stop signs only</label>
  <div class="legend">
    <div><span class="dot" style="background:#e63946"></span>Stop sign</div>
    <div><span class="dot" style="background:#f4a261"></span>Yield sign</div>
    <div><span class="dot" style="background:#2a9d8f"></span>Other sign</div>
    <div><span class="dot" style="background:#1d3557;border:1px solid #fff"></span>Snapped edge</div>
  </div>
  <div id="info" style="margin-top:8px;color:#aaa">Hover a sign</div>
</div>
<div id="map"></div>
<script>
const SIGNS = {json.dumps(sign_payloads)};
const EDGES = {json.dumps(edge_subset)};

const map = L.map('map', {{preferCanvas:true}}).setView([26.3516, -80.0833], 17);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
  {{attribution:'&copy; CARTO',subdomains:'abcd',maxZoom:19}}).addTo(map);

let signLayer = L.layerGroup().addTo(map);
let edgeLayer = L.layerGroup().addTo(map);

function render() {{
  signLayer.clearLayers();
  edgeLayer.clearLayers();
  const stopsOnly = document.getElementById('stops-only').checked;
  const showSigns = document.getElementById('show-signs').checked;
  const showEdges = document.getElementById('show-edges').checked;

  // draw edges first
  if (showEdges) {{
    const drawn = new Set();
    SIGNS.forEach(s => {{
      if (stopsOnly && !s.value.includes('stop')) return;
      if (s.edge_idx === null) return;
      if (drawn.has(s.edge_idx)) return;
      drawn.add(s.edge_idx);
      const e = EDGES[s.edge_idx];
      if (!e) return;
      L.polyline(e.coords, {{color:'#1d3557',weight:5,opacity:.8}})
       .bindTooltip(`edge_idx=${{s.edge_idx}}<br>u=${{e.u}}<br>v=${{e.v}}<br>name=${{e.name}}`, {{sticky:true}})
       .addTo(edgeLayer);
    }});
  }}

  // draw signs
  if (showSigns) {{
    SIGNS.forEach(s => {{
      if (stopsOnly && !s.value.includes('stop')) return;
      const col = s.value.includes('stop') ? '#e63946' :
                  s.value.includes('yield') ? '#f4a261' : '#2a9d8f';
      const marker = L.circleMarker([s.lat, s.lon], {{
        radius: 7, color: '#fff', fillColor: col, fillOpacity: 1, weight: 2
      }});
      // draw arrow showing aligned_direction
      const tip = `${{s.value}}<br>direction=${{s.direction}}°<br>snap=${{s.snap_type}}<br>role=${{s.role}}<br>edge_idx=${{s.edge_idx}}`;
      marker.bindTooltip(tip, {{sticky:true}});
      marker.on('mouseover', () => {{
        document.getElementById('info').innerHTML = tip.replace(/<br>/g,'\\n');
        // highlight snapped edge
        if (s.edge_idx !== null) {{
          const e = EDGES[s.edge_idx];
          if (e) L.polyline(e.coords, {{color:'#ff0',weight:8,opacity:.9}}).addTo(edgeLayer);
        }}
      }});
      marker.addTo(signLayer);
    }});
  }}
}}

document.getElementById('show-signs').addEventListener('change', render);
document.getElementById('show-edges').addEventListener('change', render);
document.getElementById('stops-only').addEventListener('change', render);

render();
</script>
</body>
</html>"""

out = ROOT / 'visuals/snap_debug.html'
out.parent.mkdir(exist_ok=True)
out.write_text(html, encoding='utf-8')
print(f'Written to {out}')
print('Open with: xdg-open visuals/snap_debug.html')