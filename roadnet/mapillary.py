"""
roadnet.mapillary
=================
Fetch Mapillary map-features (traffic signs), then snap them onto OSM
edges to produce per-edge stop / yield indicator columns.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from shapely.strtree import STRtree
from pyproj import Transformer

from .config import (
    CountyConfig,
    MLY_ENDPOINT,
    MLY_TARGET_VALUES,
    MLY_SIGN_LABELS,
    NODE_SNAP_M,
    BEARING_TOL_DEG,
    MAX_EDGE_DIST_M,
    WGS84,
)

log = logging.getLogger(__name__)

MLY_GRAPH_ENDPOINT = "https://graph.mapillary.com"

# Minimum number of original camera GPS points an edge must have to be
# considered in Strategy A. Edges with fewer are ignored regardless of
# how many interpolated sample votes they accumulated.
MIN_RAW_CAM_POINTS = 2

# Landuse tags treated as residential — signs in these zones are NOT subject
# to the landuse containment check.
_RESIDENTIAL_LANDUSE = frozenset({
    "residential", "house", "houses", "apartments", "garages",
})


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _bearing(p1: Point, p2: Point) -> float:
    dx, dy = p2.x - p1.x, p2.y - p1.y
    return np.degrees(np.arctan2(dx, dy)) % 360


def _angular_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def _circular_mean(angles: list[float]) -> float:
    """Circular mean of angles in degrees."""
    rads     = [np.radians(a) for a in angles]
    sin_mean = np.mean([np.sin(r) for r in rads])
    cos_mean = np.mean([np.cos(r) for r in rads])
    return float(np.degrees(np.arctan2(sin_mean, cos_mean)) % 360)


def _to_list(val) -> list:
    """Safely convert a value that may be a numpy array or None to a list."""
    if val is None:
        return []
    try:
        return list(val)
    except Exception:
        return []


def _normalize_edge_name(raw) -> str | None:
    if not raw or str(raw).strip() == "" or str(raw) == "nan":
        return None
    parts = [p.strip() for p in str(raw).strip().split("|")]
    parts = [p for p in parts if p]
    if not parts:
        return None
    return "|".join(sorted(parts))


def _edge_name_key(ei: int, edges: gpd.GeoDataFrame) -> str:
    has_name = "name" in edges.columns
    has_uv   = "u" in edges.columns and "v" in edges.columns
    if has_name:
        normalized = _normalize_edge_name(edges["name"].iloc[ei])
        if normalized:
            return normalized
        if has_uv:
            u = int(edges["u"].iloc[ei])
            v = int(edges["v"].iloc[ei])
            return f"__uv_{min(u,v)}_{max(u,v)}"
    return f"__unnamed_{ei}"


def _closer_service_exists(
    pt: Point,
    ref_edge_idx: int,
    ref_dist: float,
    edge_geoms,
    edge_tree: STRtree,
    edges: gpd.GeoDataFrame,
    service_radius: float = 20.0,
) -> bool:
    """
    Return True if the snap should be rejected because a service road is
    nearby and the snapped edge is a fast public road.

    Rules:
      - If the snapped edge is itself a service road → never reject.
      - If a service road is within service_radius (20m) of the sign AND
        the snapped edge has maxspeed > 30 mph → reject. The sign likely
        controls traffic inside a parking lot / private drive, not the
        public road.
      - If the snapped edge has maxspeed <= 30 mph it is already a slow
        local road, so parking-lot rejection does not apply.
    """
    if "highway" not in edges.columns:
        return False

    def _speed_mph(v) -> float:
        if not v or str(v) == "nan":
            return 0.0
        v = str(v).strip().lower()
        try:
            if "mph" in v:
                return float(v.replace("mph", "").strip())
            return float(v) * 0.621371
        except Exception:
            return 0.0

    # If the snapped edge is itself a service road, the sign belongs there
    ref_hw = str(edges["highway"].iloc[ref_edge_idx]).strip().lower()
    if ref_hw == "service":
        return False

    # If the snapped edge is slow (<= 30 mph), parking-lot rejection
    # does not apply — it is already a local road
    if "osm_maxspeed" in edges.columns:
        ref_speed = _speed_mph(edges["osm_maxspeed"].iloc[ref_edge_idx])
        if 0 < ref_speed <= 30:
            return False

    # Reject if any service road is within service_radius of the sign
    cands = edge_tree.query(pt.buffer(service_radius))
    for ei in cands:
        if int(ei) == ref_edge_idx:
            continue
        d = pt.distance(edge_geoms[ei])
        if d <= service_radius:
            hw = str(edges["highway"].iloc[int(ei)]).strip().lower()
            if hw == "service":
                return True
    return False


# ── Landuse helpers ───────────────────────────────────────────────────────────

def _build_nonresidential_landuse_tree(
    landuse: gpd.GeoDataFrame | None,
    projected_crs: str,
) -> tuple[STRtree | None, list]:
    """
    Build an STRtree of non-residential landuse polygons.

    Residential landuse is excluded — we only care about commercial, retail,
    industrial, etc. where a sign might be inside a landuse zone but the
    snapped road runs outside it (e.g. a parking lot sign snapping to a
    public street).

    Returns (tree, geoms_list) or (None, []) if no landuse data.
    """
    if landuse is None or len(landuse) == 0:
        return None, []

    lu = landuse.copy()
    epsg = int(projected_crs.split(":")[-1])
    if lu.crs is None or lu.crs.to_epsg() != epsg:
        lu = lu.to_crs(projected_crs)

    # Keep only polygon geometries
    lu = lu[lu.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    # Exclude residential landuse
    if "landuse" in lu.columns:
        lu = lu[~lu["landuse"].astype(str).str.lower().isin(_RESIDENTIAL_LANDUSE)]

    if len(lu) == 0:
        return None, []

    geoms = lu.geometry.values.tolist()
    tree  = STRtree(geoms)
    log.info("Built non-residential landuse STRtree with %d polygons", len(geoms))
    return tree, geoms


def _get_zone_for_point(
    pt: Point,
    lu_tree: STRtree | None,
    lu_geoms: list,
) -> int | None:
    """
    Return the index into lu_geoms of the non-residential zone containing pt,
    or None if pt is not inside any zone.
    """
    if lu_tree is None:
        return None
    for idx in lu_tree.query(pt):
        if lu_geoms[idx].contains(pt):
            return int(idx)
    return None


def _edge_in_zone(edge_geom, zone_geom) -> bool:
    """
    Return True if the edge geometry intersects the landuse zone polygon.
    We use intersects (not contains) so roads along a zone boundary still count.
    """
    return bool(edge_geom.intersects(zone_geom))


# ── Camera voting ─────────────────────────────────────────────────────────────

def _camera_votes_bearing_aware(
    camera_lngs: list[float],
    camera_lats: list[float],
    camera_angles: list[float | None],
    edge_geoms,
    edge_tree: STRtree,
    t: Transformer,
    snap_radius: float = 5.0,
    step_m: float = 5.0,
    bearing_tol: float = 40.0,
    sign_pt: Point | None = None,
    sign_filter_m: float = 0.0,
) -> tuple[
    defaultdict[int, int],
    defaultdict[int, list[float]],
    defaultdict[int, float],
    defaultdict[int, set],
]:
    """
    Snap camera GPS trajectory to OSM edges using bearing-aware voting.

    Returns:
        camera_votes        - sample-point votes per edge index
        edge_camera_angles  - compass angles of cameras that voted for each edge
        edge_min_sign_dist  - minimum distance from any voting sample to the sign
        edge_raw_cam_points - set of original camera GPS point indices that
                              contributed votes to each edge
    """
    camera_votes:        defaultdict[int, int]         = defaultdict(int)
    edge_camera_angles:  defaultdict[int, list[float]] = defaultdict(list)
    edge_min_sign_dist:  defaultdict[int, float]       = defaultdict(lambda: np.inf)
    edge_raw_cam_points: defaultdict[int, set]         = defaultdict(set)

    if not camera_lngs:
        return camera_votes, edge_camera_angles, edge_min_sign_dist, edge_raw_cam_points

    cam_pts_proj: list[tuple[int, float, float]] = []
    for orig_idx, (clng, clat) in enumerate(zip(camera_lngs, camera_lats)):
        try:
            cx, cy = t.transform(clng, clat)
            cam_pts_proj.append((orig_idx, cx, cy))
        except Exception:
            pass

    if not cam_pts_proj:
        return camera_votes, edge_camera_angles, edge_min_sign_dist, edge_raw_cam_points

    if len(cam_pts_proj) == 1:
        oi, cx, cy = cam_pts_proj[0]
        cam_pt     = Point(cx, cy)
        cands      = edge_tree.query(cam_pt.buffer(snap_radius))
        best_d, best_ei = np.inf, None
        for ei in cands:
            d = cam_pt.distance(edge_geoms[ei])
            if d < best_d:
                best_d, best_ei = d, ei
        if best_ei is not None and best_d <= snap_radius:
            camera_votes[int(best_ei)] += 1
            if oi < len(camera_angles) and camera_angles[oi] is not None:
                edge_camera_angles[int(best_ei)].append(camera_angles[oi])
            if sign_pt is not None:
                dist_to_sign = sign_pt.distance(cam_pt)
                edge_min_sign_dist[int(best_ei)] = min(
                    edge_min_sign_dist[int(best_ei)], dist_to_sign
                )
            edge_raw_cam_points[int(best_ei)].add(oi)
        return camera_votes, edge_camera_angles, edge_min_sign_dist, edge_raw_cam_points

    coords   = [(cx, cy) for _, cx, cy in cam_pts_proj]
    cam_line = LineString(coords)
    if cam_line.length < 1:
        return camera_votes, edge_camera_angles, edge_min_sign_dist, edge_raw_cam_points

    cum_dists = [0.0]
    for k in range(1, len(coords)):
        dx = coords[k][0] - coords[k-1][0]
        dy = coords[k][1] - coords[k-1][1]
        cum_dists.append(cum_dists[-1] + (dx**2 + dy**2) ** 0.5)
    total_len = cum_dists[-1]

    n_steps   = max(1, int(np.ceil(cam_line.length / step_m)))
    distances = np.linspace(0, cam_line.length, n_steps + 1)

    for i in range(len(distances) - 1):
        p0 = cam_line.interpolate(distances[i])
        p1 = cam_line.interpolate(distances[i + 1])
        dx = p1.x - p0.x
        dy = p1.y - p0.y
        local_brg = np.degrees(np.arctan2(dx, dy)) % 180
        mx     = (p0.x + p1.x) / 2
        my     = (p0.y + p1.y) / 2
        mid_pt = Point(mx, my)

        mid_dist  = (distances[i] + distances[i + 1]) / 2
        if total_len > 0:
            scaled    = mid_dist / cam_line.length * total_len
            nearest_k = int(np.argmin([abs(cd - scaled) for cd in cum_dists]))
        else:
            nearest_k = 0
        orig_idx = cam_pts_proj[min(nearest_k, len(cam_pts_proj) - 1)][0]

        cands = edge_tree.query(mid_pt.buffer(snap_radius))
        best_score, best_ei = np.inf, None

        for ei in cands:
            geom = edge_geoms[ei]
            d    = mid_pt.distance(geom)
            if d > snap_radius:
                continue

            d_proj  = geom.project(mid_pt)
            d_ahead = min(geom.length, d_proj + 5)
            if d_ahead > d_proj:
                ep0      = geom.interpolate(d_proj)
                ep1      = geom.interpolate(d_ahead)
                edx      = ep1.x - ep0.x
                edy      = ep1.y - ep0.y
                edge_brg = np.degrees(np.arctan2(edx, edy)) % 180
            else:
                edge_brg = local_brg

            brg_diff = abs(local_brg - edge_brg) % 180
            brg_diff = min(brg_diff, 180 - brg_diff)

            if brg_diff > bearing_tol:
                continue

            score = d + brg_diff * 0.5
            if score < best_score:
                best_score, best_ei = score, ei

        if best_ei is not None:
            camera_votes[int(best_ei)] += 1
            if orig_idx < len(camera_angles) and camera_angles[orig_idx] is not None:
                edge_camera_angles[int(best_ei)].append(camera_angles[orig_idx])
            if sign_pt is not None:
                dist_to_sign = sign_pt.distance(mid_pt)
                edge_min_sign_dist[int(best_ei)] = min(
                    edge_min_sign_dist[int(best_ei)], dist_to_sign
                )
            edge_raw_cam_points[int(best_ei)].add(orig_idx)

    return camera_votes, edge_camera_angles, edge_min_sign_dist, edge_raw_cam_points


def _group_votes_by_name(
    camera_votes: defaultdict[int, int],
    edge_camera_angles: defaultdict[int, list[float]],
    edges: gpd.GeoDataFrame,
) -> tuple[dict[str, int], dict[str, int], dict[str, list[float]]]:
    name_votes:         defaultdict[str, int]         = defaultdict(int)
    name_to_best_ei:    dict[str, int]                = {}
    name_to_best_v:     dict[str, int]                = {}
    name_camera_angles: defaultdict[str, list[float]] = defaultdict(list)

    has_name = "name" in edges.columns
    has_uv   = "u" in edges.columns and "v" in edges.columns

    for ei, votes in camera_votes.items():
        edge_name: str
        if has_name:
            normalized = _normalize_edge_name(edges["name"].iloc[ei])
            if normalized:
                edge_name = normalized
            elif has_uv:
                u = int(edges["u"].iloc[ei])
                v = int(edges["v"].iloc[ei])
                edge_name = f"__uv_{min(u,v)}_{max(u,v)}"
            else:
                edge_name = f"__unnamed_{ei}"
        else:
            edge_name = f"__unnamed_{ei}"

        name_votes[edge_name] += votes
        name_camera_angles[edge_name].extend(edge_camera_angles.get(ei, []))

        if votes > name_to_best_v.get(edge_name, -1):
            name_to_best_v[edge_name]  = votes
            name_to_best_ei[edge_name] = ei

    return dict(name_votes), name_to_best_ei, dict(name_camera_angles)


# ── Fetch ─────────────────────────────────────────────────────────────────────

def _fetch_cell(
    bbox: tuple[float, float, float, float],
    token: str,
    overlap: float,
    target_values: set[str],
    retries: int = 3,
) -> list[dict]:
    w, s, e, n = bbox
    params = {
        "access_token":  token,
        "fields":        "id,object_value,geometry,aligned_direction,first_seen_at,last_seen_at",
        "bbox":          f"{w - overlap},{s - overlap},{e + overlap},{n + overlap}",
        "object_values": ",".join(target_values),
        "limit":         5000,
    }
    for attempt in range(retries):
        try:
            r = requests.get(MLY_ENDPOINT, params=params, timeout=30)
            if r.status_code == 200:
                return r.json().get("data", [])
            if r.status_code == 429:
                time.sleep(5)
        except Exception:
            time.sleep(2)
    return []


def _fetch_sign_images(
    mly_id: str,
    token: str,
    retries: int = 3,
) -> list[dict]:
    for attempt in range(retries):
        try:
            r = requests.get(
                f"{MLY_GRAPH_ENDPOINT}/{mly_id}",
                params={
                    "access_token": token,
                    "fields":       "images.id,images.computed_geometry,images.computed_compass_angle",
                },
                timeout=30,
            )
            if r.status_code == 200:
                data    = r.json()
                images  = data.get("images", {}).get("data", [])
                results = []
                for img in images:
                    iid    = img.get("id")
                    coords = img.get("computed_geometry", {}).get("coordinates")
                    angle  = img.get("computed_compass_angle")
                    if iid and coords:
                        results.append({
                            "id":            iid,
                            "lng":           coords[0],
                            "lat":           coords[1],
                            "compass_angle": float(angle) if angle is not None else None,
                        })
                return results
            if r.status_code == 429:
                time.sleep(5)
        except Exception:
            time.sleep(2)
    return []


def _enrich_sign_images(
    gdf: gpd.GeoDataFrame,
    token: str,
    workers: int = 32,
) -> gpd.GeoDataFrame:
    log.info("Fetching contributing image data for %d signs ...", len(gdf))
    t0 = time.time()

    mly_ids = gdf["mly_id"].tolist()
    sign_images: dict[str, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_sign_images, mly_id, token): mly_id
            for mly_id in mly_ids
        }
        done = 0
        for future in as_completed(futures):
            mly_id = futures[future]
            try:
                sign_images[str(mly_id)] = future.result()
            except Exception:
                sign_images[str(mly_id)] = []
            done += 1
            if done % 1000 == 0:
                elapsed = time.time() - t0
                eta     = (elapsed / done) * (len(mly_ids) - done)
                log.info("  Image data: %d/%d | ETA %.1f min",
                         done, len(mly_ids), eta / 60)

    camera_lngs:    list[list[float]]        = []
    camera_lats:    list[list[float]]        = []
    camera_angles:  list[list[float | None]] = []
    camera_bearing: list[float | None]       = []

    for mly_id in mly_ids:
        images = sign_images.get(str(mly_id), [])
        lngs   = [img["lng"] for img in images]
        lats   = [img["lat"] for img in images]

        angles_full  = [img["compass_angle"] for img in images]
        angles_valid = [a for a in angles_full if a is not None]

        camera_lngs.append(lngs)
        camera_lats.append(lats)
        camera_angles.append(angles_full)
        camera_bearing.append(_circular_mean(angles_valid) if angles_valid else None)

    gdf = gdf.copy()
    gdf["camera_lngs"]    = camera_lngs
    gdf["camera_lats"]    = camera_lats
    gdf["camera_angles"]  = camera_angles
    gdf["camera_bearing"] = camera_bearing

    log.info(
        "Image enrichment done in %.1f min — %d/%d signs have camera data",
        (time.time() - t0) / 60,
        sum(1 for l in camera_lngs if l),
        len(mly_ids),
    )
    return gdf


def fetch_signs(
    token: str,
    bounds: tuple[float, float, float, float],
    out_path: Path,
    step: float = 0.009,
    overlap: float = 0.001,
    workers: int = 32,
    skip_if_exists: bool = False,
) -> gpd.GeoDataFrame:
    if skip_if_exists and out_path.exists():
        log.info("Loading cached Mapillary signs from %s", out_path)
        return gpd.read_parquet(out_path)

    west, south, east, north = bounds
    lons  = np.arange(west,  east,  step)
    lats  = np.arange(south, north, step)
    cells = [(lon, lat, lon + step, lat + step) for lon in lons for lat in lats]

    log.info("Fetching %d Mapillary cells with %d workers ...", len(cells), workers)
    t0       = time.time()
    all_signs: list[dict] = []
    seen_ids: set[str]    = set()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_cell, c, token, overlap, MLY_TARGET_VALUES): c
            for c in cells
        }
        done = 0
        for future in as_completed(futures):
            for feat in future.result():
                fid = feat.get("id")
                val = feat.get("object_value", "")
                if fid in seen_ids or val not in MLY_TARGET_VALUES:
                    continue
                direction = feat.get("aligned_direction")
                coords    = feat.get("geometry", {}).get("coordinates", [None, None])
                if direction is None or coords[0] is None:
                    continue
                seen_ids.add(fid)
                all_signs.append({
                    "mly_id":            fid,
                    "value":             val,
                    "aligned_direction": float(direction),
                    "first_seen_at":     feat.get("first_seen_at"),
                    "last_seen_at":      feat.get("last_seen_at"),
                    "lng":               coords[0],
                    "lat":               coords[1],
                })
            done += 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                eta     = (elapsed / done) * (len(cells) - done)
                log.info("  %d/%d cells | %d signs | ETA %.1f min",
                         done, len(cells), len(all_signs), eta / 60)

    if not all_signs:
        log.warning("No signs fetched — check API token or bounds")
        gdf = gpd.GeoDataFrame(
            columns=["mly_id", "value", "aligned_direction", "first_seen_at",
                     "last_seen_at", "lng", "lat", "camera_lngs", "camera_lats",
                     "camera_angles", "camera_bearing"],
            geometry=gpd.points_from_xy([], []),
            crs=WGS84,
        )
        gdf.to_parquet(out_path, index=False)
        return gdf

    df  = pd.DataFrame(all_signs).dropna(subset=["lng", "lat"])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lng"], df["lat"]),
        crs=WGS84,
    )
    log.info("Fetched %d signs in %.1f min", len(gdf), (time.time() - t0) / 60)

    gdf = _enrich_sign_images(gdf, token, workers=workers)
    gdf.to_parquet(out_path, index=False)
    return gdf


# ── Snap ──────────────────────────────────────────────────────────────────────

_SNAPPED_SIGN_COLUMNS = [
    "mly_id",
    "value",
    "aligned_direction",
    "camera_bearing",
    "camera_lngs",
    "camera_lats",
    "camera_angles",
    "bearing_source",
    "lng",
    "lat",
    "snap_type",
    "snap_edge_idx",
    "mly:snap_edge_role",
    "mly:snap_dist_m",
    "mly:sign_label",
]


def _empty_snapped_signs(projected_crs: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {col: pd.Series(dtype="object") for col in _SNAPPED_SIGN_COLUMNS},
        geometry=gpd.GeoSeries([], crs=projected_crs),
        crs=projected_crs,
    )


def _snap_signs_to_network(
    signs_gdf: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    projected_crs: str,
    landuse: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """
    Snap each sign to the correct approach edge using two strategies.

    Strategy A — Most camera GPS points wins (primary):
      Among all edges that have a node within NODE_SNAP_M of the sign,
      only consider those with >= MIN_RAW_CAM_POINTS (2) original camera
      GPS points. Then find the maximum raw camera count among eligible
      edges. Only edges that match that maximum count are considered.
      Among tied edges, score by votes / min_camera_dist_to_sign.

    Strategy B — Weak consensus + bearing (no clear winner from Strategy A):
      1. Find road names with >=4 total bearing-aware votes.
      2. Exactly 1 node must be within NODE_SNAP_M of the sign.
      3. That node must touch one of the qualified edges.
      4. Per-road camera bearing vs aligned_direction must agree within 40°.
      5. No service edge closer to sign than snapped edge. If so — reject.

    Landuse check (applied after both strategies):
      If the sign falls inside a non-residential landuse polygon (commercial,
      retail, industrial, etc.) the snapped edge must also intersect that
      same polygon. If the edge lies entirely outside the zone the snap is
      rejected — the sign likely controls a road inside the zone (e.g. a
      parking lot aisle), not the public street running past it.
    """
    CLOSE_CAMERA_M = 80.0

    # Build non-residential landuse spatial index
    lu_tree, lu_geoms = _build_nonresidential_landuse_tree(landuse, projected_crs)

    if signs_gdf.empty or "value" not in signs_gdf.columns:
        return _empty_snapped_signs(projected_crs)

    signs_gdf = signs_gdf[signs_gdf["value"].isin(MLY_SIGN_LABELS)].reset_index(drop=True)
    if signs_gdf.empty:
        return _empty_snapped_signs(projected_crs)

    signs_gdf = (
        gpd.GeoDataFrame(
            signs_gdf,
            geometry=gpd.points_from_xy(signs_gdf["lng"], signs_gdf["lat"]),
            crs=WGS84,
        ).to_crs(projected_crs)
    )

    node_geoms  = nodes.geometry.values
    node_osmids = nodes["osmid"].values if "osmid" in nodes.columns else nodes.index.values
    node_tree   = STRtree(node_geoms)
    edge_geoms  = edges.geometry.values
    edge_tree   = STRtree(edge_geoms)
    edge_u      = edges["u"].values
    edge_v      = edges["v"].values

    node_to_edges_u: defaultdict[int, list[int]] = defaultdict(list)
    node_to_edges_v: defaultdict[int, list[int]] = defaultdict(list)
    for idx, (u, v) in enumerate(zip(edge_u, edge_v)):
        node_to_edges_u[int(u)].append(idx)
        node_to_edges_v[int(v)].append(idx)

    osmid_to_geom: dict[int, Point] = {
        int(node_osmids[i]): node_geoms[i] for i in range(len(node_osmids))
    }

    t = Transformer.from_crs(WGS84, projected_crs, always_xy=True)

    landuse_rejected = 0
    results = []

    for row in signs_gdf.itertuples():
        pt              = row.geometry
        aligned_bearing = row.aligned_direction
        camera_lngs     = _to_list(getattr(row, "camera_lngs", None))
        camera_lats     = _to_list(getattr(row, "camera_lats", None))
        camera_angles   = _to_list(getattr(row, "camera_angles", None))

        # Check if sign is inside a non-residential landuse zone
        sign_zone_idx = _get_zone_for_point(pt, lu_tree, lu_geoms)

        # ── Bearing-aware camera trajectory voting ────────────────────────────
        camera_votes, edge_camera_angles, edge_min_sign_dist, edge_raw_cam_points = \
            _camera_votes_bearing_aware(
                camera_lngs, camera_lats, camera_angles, edge_geoms, edge_tree, t,
                sign_pt=pt,
            )

        # Group votes by road name (used for bearing lookup + Strategy B)
        name_votes, name_to_best_ei, name_camera_angles = _group_votes_by_name(
            camera_votes, edge_camera_angles, edges
        )
        total_votes = sum(name_votes.values())

        ei_to_name = {ei: name for name, ei in name_to_best_ei.items()}

        snap_type      = "unmatched"
        snap_osmid     = None
        snap_edge_idx  = None
        snap_edge_role = None
        snap_dist      = None
        bearing_source = None
        top_name       = None

        # ── Strategy A: most raw camera GPS points wins ───────────────────────
        if total_votes > 0:
            eligible: list[tuple[int, int, str, int, int, float]] = []

            for ei, votes in camera_votes.items():
                raw_count = len(edge_raw_cam_points.get(ei, set()))
                if raw_count < MIN_RAW_CAM_POINTS:
                    continue

                u_osmid = int(edge_u[ei])
                v_osmid = int(edge_v[ei])
                u_geom  = osmid_to_geom.get(u_osmid)
                v_geom  = osmid_to_geom.get(v_osmid)

                d_u = pt.distance(u_geom) if u_geom is not None else np.inf
                d_v = pt.distance(v_geom) if v_geom is not None else np.inf
                min_node_dist = min(d_u, d_v)

                if min_node_dist > NODE_SNAP_M:
                    continue

                role   = "u_node" if d_u < d_v else "v_node"
                osmid  = u_osmid  if d_u < d_v else v_osmid
                n_dist = d_u      if d_u < d_v else d_v
                eligible.append((ei, raw_count, role, osmid, votes, n_dist))

            if eligible:
                max_raw      = max(e[1] for e in eligible)
                top_eligible = [e for e in eligible if e[1] == max_raw]

                best_score_a = -np.inf
                best_ei_a    = None
                best_role_a  = None
                best_osmid_a = None
                best_dist_a  = None

                for ei, raw_count, role, osmid, votes, n_dist in top_eligible:
                    cam_dist = edge_min_sign_dist.get(ei, np.inf)
                    if cam_dist == np.inf:
                        cam_dist = CLOSE_CAMERA_M * 2

                    proximity_score = votes / max(cam_dist, 1.0)

                    if proximity_score > best_score_a:
                        best_score_a = proximity_score
                        best_ei_a    = ei
                        best_role_a  = role
                        best_osmid_a = osmid
                        best_dist_a  = n_dist

                if best_ei_a is not None:
                    road_name_a   = ei_to_name.get(best_ei_a)
                    top_name      = road_name_a
                    road_angles_a = name_camera_angles.get(road_name_a, []) if road_name_a else []
                    avg_cam_angle = _circular_mean(road_angles_a) if road_angles_a else None

                    bearing_ok = True
                    if (avg_cam_angle is not None
                            and aligned_bearing is not None
                            and not np.isnan(float(aligned_bearing))):
                        raw_diff     = _angular_diff(float(avg_cam_angle), float(aligned_bearing))
                        bearing_diff = raw_diff if raw_diff <= 90 else 180 - raw_diff
                        if bearing_diff > 65:
                            bearing_ok = False

                    if bearing_ok:
                        consensus_dist = pt.distance(edge_geoms[best_ei_a])
                        if not _closer_service_exists(
                            pt, best_ei_a, consensus_dist, edge_geoms, edge_tree, edges
                        ):
                            # ── Landuse check ─────────────────────────────
                            landuse_ok = True
                            if sign_zone_idx is not None:
                                if not _edge_in_zone(edge_geoms[best_ei_a], lu_geoms[sign_zone_idx]):
                                    landuse_ok = False
                                    landuse_rejected += 1

                            if landuse_ok:
                                snap_edge_idx  = best_ei_a
                                snap_edge_role = best_role_a
                                snap_osmid     = best_osmid_a
                                snap_dist      = best_dist_a
                                snap_type      = "node"
                                bearing_source = "camera_consensus"

        # ── Strategy B: weak consensus + bearing ─────────────────────────────
        if snap_edge_idx is None:
            qualified_names = {
                name: name_to_best_ei[name]
                for name, total in name_votes.items()
                if total >= 4
            }
            qualified_edges = {
                ei: camera_votes[ei]
                for ei in qualified_names.values()
            }

            if qualified_edges:
                node_cands = node_tree.query(pt.buffer(NODE_SNAP_M))
                nearby_nodes = [
                    node_osmids[ni]
                    for ni in node_cands
                    if pt.distance(node_geoms[ni]) <= NODE_SNAP_M
                ]

                if len(nearby_nodes) == 1:
                    nd_id   = int(nearby_nodes[0])
                    nd_geom = osmid_to_geom.get(nd_id)

                    node_edge_candidates = []
                    for ei in node_to_edges_u.get(nd_id, []):
                        if ei in qualified_edges:
                            node_edge_candidates.append((ei, "u_node", nd_id))
                    for ei in node_to_edges_v.get(nd_id, []):
                        if ei in qualified_edges:
                            node_edge_candidates.append((ei, "v_node", nd_id))

                    best_score = np.inf
                    for ei, role, nd_id in node_edge_candidates:
                        geom = edge_geoms[ei]
                        if geom.length < 1 or nd_geom is None:
                            continue

                        road_name    = ei_to_name.get(ei)
                        road_angles  = name_camera_angles.get(road_name, []) if road_name else []
                        avg_road_cam = _circular_mean(road_angles) if road_angles else None

                        if (avg_road_cam is not None
                                and aligned_bearing is not None
                                and not np.isnan(float(aligned_bearing))):
                            raw_diff = _angular_diff(float(aligned_bearing), float(avg_road_cam))
                            diff = raw_diff if raw_diff <= 90 else 180 - raw_diff
                        else:
                            d_node = geom.project(nd_geom)
                            d_out  = (
                                min(geom.length, d_node + 15) if role == "u_node"
                                else max(0, d_node - 15)
                            )
                            p_out    = geom.interpolate(d_out)
                            road_brg = _bearing(
                                Point(nd_geom.x, nd_geom.y),
                                Point(p_out.x,   p_out.y)
                            )
                            diff = (
                                _angular_diff(
                                    float(aligned_bearing), (road_brg + 180) % 360
                                )
                                if (aligned_bearing is not None
                                    and not np.isnan(float(aligned_bearing)))
                                else 0.0
                            )

                        if diff > 40:
                            continue

                        edge_dist = pt.distance(geom)
                        if edge_dist > MAX_EDGE_DIST_M:
                            continue

                        votes = qualified_edges[ei]
                        score = diff - votes * 10
                        if score < best_score:
                            best_score     = score
                            snap_edge_idx  = ei
                            snap_edge_role = role
                            snap_osmid     = nd_id
                            snap_dist      = pt.distance(nd_geom)
                            bearing_source = "weak_consensus"
                            top_name       = ei_to_name.get(ei)

                    if snap_edge_idx is not None:
                        snapped_dist = pt.distance(edge_geoms[snap_edge_idx])
                        if _closer_service_exists(
                            pt, snap_edge_idx, snapped_dist, edge_geoms, edge_tree, edges
                        ):
                            snap_edge_idx  = None
                            snap_edge_role = None
                            snap_osmid     = None
                            snap_dist      = None
                            bearing_source = None
                            top_name       = None

                        # ── Landuse check for Strategy B ──────────────────
                        elif sign_zone_idx is not None and snap_edge_idx is not None:
                            if not _edge_in_zone(edge_geoms[snap_edge_idx], lu_geoms[sign_zone_idx]):
                                snap_edge_idx  = None
                                snap_edge_role = None
                                snap_osmid     = None
                                snap_dist      = None
                                bearing_source = None
                                top_name       = None
                                landuse_rejected += 1

                if snap_edge_idx is not None:
                    snap_type = "node"

        # Per-road bearing for debug output
        snapped_road_angles = []
        if top_name is not None:
            snapped_road_angles = name_camera_angles.get(top_name, [])
        avg_cam_angle = _circular_mean(snapped_road_angles) if snapped_road_angles else None

        results.append({
            "mly_id":             row.mly_id,
            "value":              row.value,
            "aligned_direction":  aligned_bearing,
            "camera_bearing":     avg_cam_angle,
            "camera_lngs":        camera_lngs,
            "camera_lats":        camera_lats,
            "camera_angles":      camera_angles,
            "bearing_source":     bearing_source,
            "lng":                row.lng,
            "lat":                row.lat,
            "snap_type":          snap_type,
            "snap_edge_idx":      snap_edge_idx,
            "mly:snap_edge_role": snap_edge_role,
            "mly:snap_dist_m":    snap_dist,
        })

    log.info("Landuse zone check rejected %d snaps", landuse_rejected)

    snapped = gpd.GeoDataFrame(
        results,
        columns=_SNAPPED_SIGN_COLUMNS[:-1],
        geometry=gpd.points_from_xy(
            [r["lng"] for r in results],
            [r["lat"] for r in results],
        ),
        crs=WGS84,
    ).to_crs(projected_crs)
    if snapped.empty or "value" not in snapped.columns:
        return _empty_snapped_signs(projected_crs)

    snapped["mly:sign_label"] = (
        snapped["value"].map(MLY_SIGN_LABELS).fillna(snapped["value"])
    )
    return snapped


def _propagate_signs_to_reverse_edges(
    enriched: gpd.GeoDataFrame,
    map_cols: list[str],
) -> gpd.GeoDataFrame:
    """
    Propagate MAP sign flags to the exact reverse edge on two-way streets.

    For edge A (u→v), the reverse edge B is B.u==A.v AND B.v==A.u.
    Flips _u↔_v suffix on the column name for the reverse edge.
    Only applied when the source edge is NOT oneway.
    """
    if not map_cols:
        return enriched

    enriched = enriched.copy()

    u_vals = pd.to_numeric(enriched["u"], errors="coerce").astype("Int64").values
    v_vals = pd.to_numeric(enriched["v"], errors="coerce").astype("Int64").values

    uv_to_pos: dict[tuple[int, int], int] = {}
    for pos in range(len(enriched)):
        u = int(u_vals[pos]) if not pd.isna(u_vals[pos]) else None
        v = int(v_vals[pos]) if not pd.isna(v_vals[pos]) else None
        if u is not None and v is not None:
            uv_to_pos[(u, v)] = pos

    if "oneway" in enriched.columns:
        oneway_mask = enriched["oneway"].astype(str).str.strip().str.lower().isin(
            ["true", "yes", "1"]
        ).values
    else:
        oneway_mask = np.zeros(len(enriched), dtype=bool)

    col_arrays: dict[str, np.ndarray] = {
        col: pd.to_numeric(enriched[col], errors="coerce").fillna(0).values.astype(int)
        for col in map_cols
        if col in enriched.columns
    }

    propagated = 0

    for pos in range(len(enriched)):
        if oneway_mask[pos]:
            continue

        u = int(u_vals[pos]) if not pd.isna(u_vals[pos]) else None
        v = int(v_vals[pos]) if not pd.isna(v_vals[pos]) else None
        if u is None or v is None:
            continue

        rev_pos = uv_to_pos.get((v, u))
        if rev_pos is None or rev_pos == pos:
            continue

        if oneway_mask[rev_pos]:
            continue

        for col in map_cols:
            if col not in col_arrays:
                continue
            if col_arrays[col][pos] <= 0:
                continue

            if col.endswith("_u"):
                rev_col = col[:-2] + "_v"
            elif col.endswith("_v"):
                rev_col = col[:-2] + "_u"
            else:
                continue

            if rev_col not in col_arrays:
                col_arrays[rev_col] = np.zeros(len(enriched), dtype=int)

            if col_arrays[rev_col][rev_pos] == 0:
                col_arrays[rev_col][rev_pos] = col_arrays[col][pos]
                propagated += 1

    for col, arr in col_arrays.items():
        enriched[col] = arr

    log.info("Propagated MAP signs to %d reverse edges", propagated)
    return enriched


def attach_signs_to_edges(
    signs_raw: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    projected_crs: str,
    out_path: Path,
    landuse: gpd.GeoDataFrame | None = None,
    skip_if_exists: bool = False,
) -> gpd.GeoDataFrame:
    if skip_if_exists and out_path.exists():
        log.info("Loading cached MLY-enriched edges from %s", out_path)
        return gpd.read_parquet(out_path)

    snap_edges = edges.copy()
    if "osm_maxspeed" in snap_edges.columns:
        def _speed_mph(v):
            if pd.isna(v):
                return 0
            v = str(v).strip().lower()
            try:
                if "mph" in v:
                    return float(v.replace("mph", "").strip())
                return float(v) * 0.621371
            except Exception:
                return 0
        speeds     = snap_edges["osm_maxspeed"].apply(_speed_mph)
        snap_edges = snap_edges[speeds < 50]

    log.info("Snapping %d Mapillary signs to network (%d eligible edges) ...",
             len(signs_raw), len(snap_edges))
    snapped = _snap_signs_to_network(
        signs_raw, nodes, snap_edges, projected_crs, landuse=landuse
    )

    required_cols = {"snap_type", "mly:snap_edge_role", "mly:sign_label"}
    if snapped.empty or not required_cols.issubset(snapped.columns):
        log.info("No Mapillary signs snapped; returning original edges unchanged")
        enriched = edges.copy()
        enriched.to_parquet(out_path)
        return enriched

    matched = snapped[
        (snapped["snap_type"] != "unmatched")
        & (snapped["mly:snap_edge_role"].isin(["u_node", "v_node"]))
        & (snapped["mly:sign_label"].isin(MLY_SIGN_LABELS.values()))
    ].copy()

    if matched.empty:
        log.info("No Mapillary signs matched eligible network nodes; returning original edges unchanged")
        enriched = edges.copy()
        enriched.to_parquet(out_path)
        return enriched

    role_prefix = {"u_node": "u", "v_node": "v"}
    matched["role_prefix"] = matched["mly:snap_edge_role"].map(role_prefix)
    matched["sign_col"] = (
        "MAP_has_"
        + matched["mly:sign_label"]
            .str.replace("mly:", "")
            .str.replace(" ", "_")
            .str.lower()
        + "_"
        + matched["role_prefix"]
    )

    sign_counts = (
        matched.groupby(["snap_edge_idx", "sign_col"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    enriched = edges.reset_index(drop=True).merge(
        sign_counts, left_index=True, right_on="snap_edge_idx", how="left"
    ).drop(columns=["snap_edge_idx"])

    map_cols = [c for c in enriched.columns if c.startswith("MAP_")]
    enriched[map_cols] = enriched[map_cols].fillna(0).astype(int)

    enriched = _propagate_signs_to_reverse_edges(enriched, map_cols)

    log.info(
        "Edges with >=1 Mapillary sign: %d / %d",
        (enriched[map_cols].sum(axis=1) > 0).sum(), len(enriched),
    )
    enriched.to_parquet(out_path)
    return enriched
