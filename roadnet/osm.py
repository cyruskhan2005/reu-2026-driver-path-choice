"""
roadnet.osm
===========
Download, clean and save OSM road-network and land-use data for a county.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import osmnx as ox
import pandas as pd
import geopandas as gpd

from .config import (
    CountyConfig,
    OSM_NODE_TAGS,
    OSM_WAY_TAGS,
    OSM_HIGHWAY_FILTER,
    OSM_LANDUSE_TAGS,
)

log = logging.getLogger(__name__)

# ── OSMnx global tag settings ─────────────────────────────────────────────────
ox.settings.useful_tags_node = OSM_NODE_TAGS
ox.settings.useful_tags_way  = OSM_WAY_TAGS

_SKIP_COLS = {"geometry", "u", "v", "key"}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _stringify(val: Any) -> Any:
    """Flatten list/array values and coerce to str or numeric scalar."""
    if isinstance(val, (list, np.ndarray)):
        return "|".join(map(str, val))
    if isinstance(val, (int, float, np.integer, np.floating)):
        return val
    if pd.isna(val):
        return np.nan
    return str(val)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ``_stringify`` to every non-geometry column."""
    for col in df.columns:
        if col in _SKIP_COLS:
            continue
        df[col] = df[col].apply(_stringify).astype(str).replace("nan", np.nan)
    return df


def _annotate_control_nodes(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Add binary stop / signal / crossing flags to *edges* based on their
    endpoint nodes (u and v).
    """
    nodes["has_stop"]     = (nodes["highway"] == "stop").astype(int)
    nodes["has_signal"]   = (nodes["highway"] == "traffic_signals").astype(int)
    nodes["has_crossing"] = nodes["crossing"].notna().astype(int)

    lookup = nodes.set_index("osmid")[["has_stop", "has_signal", "has_crossing"]]

    for endpoint, prefix in [("u", "u"), ("v", "v")]:
        for flag in ["stop", "signal", "crossing"]:
            col = f"OSM_has_{flag}_{prefix}"
            edges[col] = (
                edges[endpoint]
                .map(lookup[f"has_{flag}"])
                .fillna(0)
                .astype(int)
            )

    return edges


# ── Public API ────────────────────────────────────────────────────────────────

def download_county(
    county: CountyConfig,
    out_dir: Path,
    skip_if_exists: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Download OSM road network and land-use polygons for *county*.

    Returns
    -------
    nodes, edges, landuse
        All projected to ``county.projected_crs``.
    """
    nodes_path   = out_dir / "osm_nodes.parquet"
    edges_path   = out_dir / "osm_edges.parquet"
    landuse_path = out_dir / "osm_landuse.parquet"

    if skip_if_exists and all(p.exists() for p in [nodes_path, edges_path, landuse_path]):
        log.info("[%s] Loading cached OSM data", county.name)
        nodes   = gpd.read_parquet(nodes_path)
        edges   = gpd.read_parquet(edges_path)
        landuse = gpd.read_parquet(landuse_path)
        return nodes, edges, landuse

    log.info("[%s] Downloading road network from OSM …", county.name)
    G = ox.graph_from_place(
        county.place_query,
        custom_filter=OSM_HIGHWAY_FILTER,
        truncate_by_edge=True,
        simplify=True,
    )
    G = ox.project_graph(G, to_crs=county.projected_crs)
    nodes, edges = ox.graph_to_gdfs(G)
    nodes = nodes.reset_index()
    edges = edges.reset_index()

    nodes["county"] = county.name
    edges["county"] = county.name

    log.info("[%s] Downloading land-use polygons …", county.name)
    landuse = (
        ox.features_from_place(county.place_query, tags=OSM_LANDUSE_TAGS)
        .to_crs(county.projected_crs)
        .reset_index()
    )
    landuse["county"] = county.name

    # ── Control-node flags ────────────────────────────────────────────────────
    edges = _annotate_control_nodes(nodes, edges)

    # ── Serialisation cleanup ─────────────────────────────────────────────────
    nodes   = _clean_columns(nodes.copy())
    edges   = _clean_columns(edges.copy())
    landuse = _clean_columns(landuse)

    log.info(
        "[%s] %d nodes | %d edges | %d landuse polygons",
        county.name, len(nodes), len(edges), len(landuse),
    )

    nodes.to_parquet(nodes_path)
    edges.to_parquet(edges_path)
    landuse.to_parquet(landuse_path)

    return nodes, edges, landuse


def load_county(out_dir: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load previously saved OSM parquets from *out_dir*."""
    return (
        gpd.read_parquet(out_dir / "osm_nodes.parquet"),
        gpd.read_parquet(out_dir / "osm_edges.parquet"),
        gpd.read_parquet(out_dir / "osm_landuse.parquet"),
    )
