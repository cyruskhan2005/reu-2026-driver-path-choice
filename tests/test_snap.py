from pathlib import Path
import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely.geometry import Point
from roadnet.mapillary import _snap_signs_to_network

ROOT = Path(__file__).parent.parent

signs = gpd.read_parquet(ROOT / 'sflorida_outputs/Broward_County/mly_signs_raw.parquet')
nodes = gpd.read_parquet(ROOT / 'sflorida_outputs/Broward_County/osm_nodes.parquet')
edges = gpd.read_parquet(ROOT / 'sflorida_outputs/Broward_County/osm_edges.parquet')

projected_crs = 'EPSG:26917'

print('Running snap...')
snapped = _snap_signs_to_network(signs, nodes, edges, projected_crs)

# filter to stop signs near Banks Rd / NW 26th Court intersection
# node 3573210997 is at that intersection
# find snapped signs that ended up on edges 338533/338540 (NW 26th Court)
# or 338531/338535 (Banks Road)
target_fids = {338533, 338540, 338531, 338535, 338532, 338537}

# get edges index positions for those FIDs
edges_reset = edges.reset_index(drop=True)
fid_to_idx = {row.fid: i for i, row in edges_reset.iterrows()} if 'fid' in edges.columns else {}

# just show all snapped signs near the intersection
t = Transformer.from_crs('EPSG:4326', projected_crs, always_xy=True)
tx, ty = t.transform(-80.2058, 26.1482)
pt = Point(tx, ty)

snapped_proj = snapped.to_crs(projected_crs)
snapped_proj['dist'] = snapped_proj.geometry.distance(pt)
nearby = snapped_proj.nsmallest(10, 'dist')[
    ['mly_id', 'value', 'aligned_direction', 'snap_type',
     'snap_edge_idx', 'mly:snap_edge_role', 'mly:snap_dist_m', 'dist']
]
print(nearby.to_string())