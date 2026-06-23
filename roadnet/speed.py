"""
roadnet.speed
=============
Two-stage speed pipeline:

1. **Connector resolution** — walk the directed graph to infer speeds on
   ramp/link segments that lack an explicit speed limit.
2. **Speed arbitration** — hierarchical priority merge across OSM,
   FDOT, and any custom county datasets, with authority-aware confidence scoring.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx

from .config import (
    HIGHWAY_SPEED_DEFAULTS,
    LINK_TYPES,
    LINK_TO_PARENT,
)

log = logging.getLogger(__name__)


ESTIMATED_SPEED_SOURCES = {
    "road_name_mode",
    "functional_class",
    "roundabout_default",
}


def _round_speed_to_nearest_5(speed: float) -> int:
    """Round mph to the nearest 5, with .5 increments rounded upward."""
    return int(np.floor((float(speed) / 5.0) + 0.5) * 5)


def _round_speed_series_to_nearest_5(speed: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(speed, errors="coerce")
    return np.floor((numeric / 5.0) + 0.5) * 5


# ─────────────────────────────────────────────────────────────────────────────
# Connector speed resolution
# ─────────────────────────────────────────────────────────────────────────────

def _find_mainline(
    G: nx.DiGraph,
    start_node: int,
    direction: str = "out",
    max_depth: int = 50,
) -> tuple[Optional[str], Optional[float]]:
    """BFS walk to find the first non-link mainline road from *start_node*."""
    visited = {start_node}
    queue   = [(start_node, 0)]

    while queue:
        node, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        edges = G.out_edges(node, data=True) if direction == "out" else G.in_edges(node, data=True)
        for u, v, data in edges:
            neighbour = v if direction == "out" else u
            if neighbour in visited:
                continue
            visited.add(neighbour)

            hw    = data.get("highway", "")
            speed = data.get("speed")

            if hw in HIGHWAY_SPEED_DEFAULTS:
                if hw in ("service", "residential", "living_street"):
                    queue.append((neighbour, depth + 1))
                else:
                    return hw, speed
                continue
            queue.append((neighbour, depth + 1))

    return None, None


def _infer_connector_speed(
    row: pd.Series,
    up_spd: Optional[float],
    down_spd: Optional[float],
    parent_spd: int,
) -> tuple[str, Optional[int]]:
    """Return (transition_label, inferred_speed_mph)."""
    is_oneway = str(row.get("oneway", "")).strip() in ("True", "yes", "1")
    up_valid   = pd.notna(up_spd)
    down_valid = pd.notna(down_spd)

    def _connector_speed(spd: float) -> int:
        capped = max(15, min(float(spd), float(parent_spd)))
        return _round_speed_to_nearest_5(capped)

    if up_valid and down_valid:
        delta = down_spd - up_spd
        if not is_oneway:
            return "lateral", _connector_speed((up_spd + down_spd) / 2)
        if delta > 5:
            spd = up_spd + delta * 0.65
            return "acceleration", _connector_speed(spd)
        if delta < -5:
            spd = up_spd + delta * 0.65
            return "deceleration", _connector_speed(spd)
        return "lateral", _connector_speed((up_spd + down_spd) / 2)

    if down_valid:
        t = "entry_only" if is_oneway else "lateral"
        return t, _connector_speed(down_spd)
    if up_valid:
        t = "exit_only" if is_oneway else "lateral"
        return t, _connector_speed(up_spd)

    return "unknown", None


def resolve_connector_speeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a directed graph from *df* and walk it to fill missing speeds on
    ramp/link segments.

    Modifies *df* in-place (adds / updates columns):
    ``osm_maxspeed``, ``connector_transition``,
    ``connector_upstream``, ``connector_downstream``.

    Returns the modified DataFrame.
    """
    log.info("Resolving connector speeds for %d link segments …", df["highway"].isin(LINK_TYPES).sum())

    if "osm_maxspeed" not in df.columns:
        if "maxspeed" in df.columns:
            df["osm_maxspeed"] = (
                df["maxspeed"].astype(str)
                .str.extract(r"(\d+)", expand=False)
                .astype(float)
            )
        else:
            df["osm_maxspeed"] = np.nan

    for col in ("connector_transition", "connector_upstream", "connector_downstream"):
        if col not in df.columns:
            df[col] = pd.Series(pd.NA, index=df.index, dtype=object)

    # ── Build graph ───────────────────────────────────────────────────────────
    edge_data = df[["u", "v", "highway", "osm_maxspeed"]].copy()
    G = nx.DiGraph()
    G.add_nodes_from(pd.concat([df["u"], df["v"]]).unique())
    for row in edge_data.itertuples(index=True):
        G.add_edge(row.u, row.v, idx=row.Index, highway=row.highway, speed=row.osm_maxspeed)

    # ── Walk links ────────────────────────────────────────────────────────────
    link_rows = df[df["highway"].isin(LINK_TYPES)]
    filled    = 0

    for idx, row in link_rows.iterrows():
        if pd.notna(df.at[idx, "osm_maxspeed"]):
            continue

        parent_hw  = LINK_TO_PARENT.get(row["highway"], "primary")
        parent_spd = HIGHWAY_SPEED_DEFAULTS.get(parent_hw, 45)

        up_hw,   up_spd_raw   = _find_mainline(G, row["u"], direction="in")
        down_hw, down_spd_raw = _find_mainline(G, row["v"], direction="out")

        up_spd   = up_spd_raw   if pd.notna(up_spd_raw)   else HIGHWAY_SPEED_DEFAULTS.get(up_hw)
        down_spd = down_spd_raw if pd.notna(down_spd_raw) else HIGHWAY_SPEED_DEFAULTS.get(down_hw)

        transition, inferred = _infer_connector_speed(row, up_spd, down_spd, parent_spd)
        if transition == "unknown" or inferred is None:
            continue

        df.at[idx, "osm_maxspeed"]         = float(inferred)
        df.at[idx, "connector_transition"] = transition
        df.at[idx, "connector_upstream"]   = up_hw
        df.at[idx, "connector_downstream"] = down_hw
        filled += 1

    log.info("Connector speed: filled %d links", filled)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Land-use context
# ─────────────────────────────────────────────────────────────────────────────

LANDUSE_SPEED_RANGES: dict[str, tuple[int, int]] = {
    "residential":  (15, 35),
    "retail":       (20, 35),
    "commercial":   (25, 45),
    "industrial":   (25, 45),
    "construction": (15, 25),
    "education":    (15, 25),
    "institutional":(20, 35),
    "forest":       (25, 55),
}


def add_landuse(
    network: gpd.GeoDataFrame,
    landuse: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Spatial-join land-use polygons onto road segments (modal landuse per edge)."""
    log.info("Joining landuse context …")
    lu_col = next((c for c in landuse.columns if "landuse" in c.lower()), None)
    if lu_col is None:
        log.warning("No landuse column found — skipping")
        return network

    if lu_col != "landuse":
        landuse = landuse.rename(columns={lu_col: "landuse"})

    if landuse.crs != network.crs:
        landuse = landuse.to_crs(network.crs)

    roads  = network.reset_index()
    id_col = "fid" if "fid" in roads.columns else roads.columns[0]

    joined = gpd.sjoin(
        roads[[id_col, "geometry"]],
        landuse[["geometry", "landuse"]],
        how="left",
        predicate="intersects",
    )
    mode_lu = (
        joined.groupby(id_col)["landuse"]
        .apply(lambda x: x.mode().iloc[0] if len(x.mode()) else None)
    )
    network["landuse"] = mode_lu
    log.info("Landuse tagged on %d segments", network["landuse"].notna().sum())
    return network


# ─────────────────────────────────────────────────────────────────────────────
# Speed arbitration
# ─────────────────────────────────────────────────────────────────────────────

# Road-type tokens to strip before name comparison
_NAME_STRIP = {
    "st", "ave", "blvd", "rd", "dr", "ln", "ct", "pl", "ter", "terrace",
    "hwy", "pkwy", "expy", "fwy", "cir", "way", "loop", "path", "trail",
    "sw", "nw", "se", "ne", "n", "s", "e", "w",
    "street", "avenue", "boulevard", "road", "drive", "lane", "court",
    "place", "highway", "parkway", "expressway", "freeway", "circle",
    "sr", "us", "i",
}


def _base_words(name: str) -> set:
    """Lowercase, strip punctuation, remove type/directional tokens."""
    tokens = re.sub(r"[^a-z0-9\s]", " ", name.lower()).split()
    return {t for t in tokens if t not in _NAME_STRIP and len(t) > 1}


def arbitrate_speed(
    df: pd.DataFrame,
    has_fdot: bool                  = True,
    has_miami: bool                 = False,
    has_pbc: bool                   = False,
    custom_speed_col: Optional[str] = None,
    custom_owner_col: Optional[str] = None,
    custom_name_col:  Optional[str] = None,
) -> pd.DataFrame:
    """
    Authority-aware hierarchical speed assignment.

    Priority for all roads:
      1. OSM (if available) — always the first choice
      2. Authority-specific speed based on road ownership:
           Miami-Dade SR  → FDOT speed then Miami custom
           Miami-Dade CM/CI/CO → Miami custom then FDOT
           Palm Beach FDOT-owned → FDOT speed then PBC custom
           Palm Beach COUNTY/MUN → PBC custom then FDOT
           Everything else → custom then FDOT
      3. Mode by road name
      4. Functional-class default
      5. Roundabout default

    Writes ``estimated_speed_limit``, ``speed_source``,
    ``speed_limit_confidence_score``.
    """
    log.info("Running speed arbitration …")

    def _col(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    def _scol(name: str) -> pd.Series:
        return df[name].fillna("").astype(str) if name in df.columns else pd.Series("", index=df.index)

    osm_speed    = _col("osm_maxspeed")
    fdot_speed   = _col("FDOT_SPEED")
    custom_speed = _col(custom_speed_col) if custom_speed_col else pd.Series(np.nan, index=df.index)

    # ── Authority flags ───────────────────────────────────────────────────────
    # All counties use CUSTOM_OWNER (renamed from whatever column the county
    # uses: MAINTCODE for Miami-Dade, RESP_AUTH for Palm Beach, etc.)
    # and CUSTOM_FUNC_CLASS for functional classification.
    #
    # State-road authority values (across all counties):
    #   Miami-Dade MAINTCODE=SR, Palm Beach RESP_AUTH=FDOT
    # County/city authority values:
    #   Miami-Dade MAINTCODE=CM, Palm Beach RESP_AUTH=COUNTY/MUN/CITY
    #   Miami-Dade MAINTCODE=CI/CO (city/other)
    # Private:
    #   Palm Beach RESP_AUTH=PRIT

    road_authority     = _scol("CUSTOM_OWNER")
    is_state_road      = road_authority.isin(["SR", "FDOT"])
    is_county_road     = road_authority.isin(["CM", "COUNTY", "MUN", "CITY"])
    is_city_road       = road_authority.isin(["CI", "CO"])
    is_private_road    = road_authority.isin(["PRIT"])
    has_authority      = road_authority.ne("").any()



    # ── Speed assignment ──────────────────────────────────────────────────────
    # OSM is always the first choice when available.
    # After OSM, authority-based priority applies generically:
    #   State road  (SR/FDOT)        → FDOT speed → custom
    #   County/city (CM/COUNTY/MUN)  → custom → FDOT
    #   City/other  (CI/CO)          → custom → FDOT
    #   Private     (PRIT)           → custom → FDOT
    #   No authority (Broward etc.)  → custom → FDOT
    estimated = osm_speed.copy()

    # State roads: FDOT speed is authoritative
    mask = estimated.isna() & is_state_road & fdot_speed.notna()
    estimated[mask] = fdot_speed[mask]

    mask = estimated.isna() & is_state_road & custom_speed.notna()
    estimated[mask] = custom_speed[mask]

    # County/city/other roads: custom speed first, FDOT as fallback
    is_non_state = is_county_road | is_city_road | is_private_road
    mask = estimated.isna() & is_non_state & custom_speed.notna() & (custom_speed != 35)
    estimated[mask] = custom_speed[mask]

    mask = estimated.isna() & is_non_state & fdot_speed.notna()
    estimated[mask] = fdot_speed[mask]

    # No authority data (Broward, unclassified): custom → FDOT
    mask = estimated.isna() & custom_speed.notna() & (custom_speed != 35)
    estimated[mask] = custom_speed[mask]

    mask = estimated.isna() & fdot_speed.notna()
    estimated[mask] = fdot_speed[mask]

    # Accept suspect 35 mph as last resort before defaults
    mask = estimated.isna() & custom_speed.notna() & (custom_speed == 35)
    estimated[mask] = custom_speed[mask]

    # Mode by road name
    road_mode_source = pd.Series(False, index=df.index)
    if "name" in df.columns:
        road_mode = (
            pd.DataFrame({"name": df["name"], "spd": estimated})
            .groupby("name")["spd"]
            .transform(lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan)
        )
        road_mode_source = estimated.isna() & road_mode.notna()
        estimated[road_mode_source] = road_mode[road_mode_source]

    # Functional-class defaults
    for hw_type, default_spd in HIGHWAY_SPEED_DEFAULTS.items():
        mask = estimated.isna() & (df["highway"] == hw_type)
        estimated[mask] = default_spd

    # Roundabout handling
    if "junction" in df.columns:
        ra_mask = df["junction"] == "roundabout"
        estimated[ra_mask & estimated.isna()] = 20
        over30 = ra_mask & (estimated > 30)
        if over30.sum():
            estimated[over30] = 25

    # ── Speed source tracking ─────────────────────────────────────────────────
    source = pd.Series("none", index=df.index)
    source[osm_speed.notna()] = "osm"

    # State roads — FDOT speed wins after OSM
    source[
        (estimated == fdot_speed) & fdot_speed.notna() & is_state_road & (source == "none")
    ] = "fdot_state_road"
    source[
        (estimated == custom_speed) & custom_speed.notna() & is_state_road & (source == "none")
    ] = "custom_sr_fallback"

    # County/city/other roads — custom wins after OSM
    source[
        (estimated == custom_speed) & custom_speed.notna() & is_non_state & (source == "none")
    ] = "custom_primary"
    source[
        (estimated == fdot_speed) & fdot_speed.notna() & is_non_state & (source == "none")
    ] = "fdot_fallback"

    # No authority data
    source[
        (estimated == custom_speed) & custom_speed.notna() & (source == "none")
    ] = "custom_primary"
    source[
        (estimated == fdot_speed) & fdot_speed.notna() & (source == "none")
    ] = "fdot_primary"

    # Connector graph transitions
    if "connector_transition" in df.columns:
        for t in ("acceleration", "deceleration", "lateral", "entry_only", "exit_only"):
            mask = (df["connector_transition"] == t) & (source == "osm")
            source[mask] = f"graph_{t}"

    source[road_mode_source & (source == "none")] = "road_name_mode"

    # Functional class and roundabout defaults
    for hw_type in HIGHWAY_SPEED_DEFAULTS:
        mask = (df.get("highway") == hw_type) & (source == "none")
        source[mask] = "functional_class"

    if "junction" in df.columns:
        source[(df["junction"] == "roundabout") & (source == "none")] = "roundabout_default"

    df["speed_source"] = source

    is_estimated_source = source.str.startswith("graph_", na=False) | source.isin(ESTIMATED_SPEED_SOURCES)
    estimated[is_estimated_source] = _round_speed_series_to_nearest_5(estimated[is_estimated_source])

    df["estimated_speed_limit"] = estimated
    df["speed_limit_is_estimated"] = is_estimated_source
    df["speed_limit_label"] = np.where(is_estimated_source, "Estimated speed limit", "Speed limit")

    # ── Confidence scoring (fully vectorized) ─────────────────────────────────
    source_base = {
        "osm":               0.40,
        "fdot_state_road":   0.42,
        "custom_primary":    0.35,
        "custom_sr_fallback":0.28,
        "fdot_fallback":     0.25,
        "fdot_primary":      0.35,
        "roundabout_default":0.25,
        "graph_lateral":     0.22,
        "graph_acceleration":0.18,
        "graph_deceleration":0.18,
        "graph_entry_only":  0.12,
        "graph_exit_only":   0.12,
        "road_name_mode":    0.18,
        "functional_class":  0.04,
    }
    conf = np.zeros(len(df))
    src_arr = source.values
    for src, score in source_base.items():
        conf[src_arr == src] += score

    # ── Authority bonus ───────────────────────────────────────────────────────
    if has_authority:
        conf[is_state_road.values]   += 0.20
        conf[is_county_road.values]  += 0.10
        conf[is_city_road.values]    += 0.05
        conf[is_private_road.values] -= 0.05



    # ── Name match bonus (vectorized) ─────────────────────────────────────────
    osm_name_col    = _scol("name")
    custom_name_col_data = _scol(custom_name_col) if custom_name_col else pd.Series("", index=df.index)
    fdot_name_col   = _scol("FDOT_DESCR")

    osm_words    = osm_name_col.map(_base_words)
    custom_words = custom_name_col_data.map(_base_words)
    fdot_words   = fdot_name_col.map(_base_words)

    custom_osm_match = pd.Series(
        [bool(a & b) for a, b in zip(osm_words, custom_words)], index=df.index
    )
    fdot_osm_match = pd.Series(
        [bool(a & b) for a, b in zip(osm_words, fdot_words)], index=df.index
    )
    custom_fdot_match = pd.Series(
        [bool(a & b) for a, b in zip(custom_words, fdot_words)], index=df.index
    )

    is_custom_src = source.str.contains("custom", na=False)
    is_fdot_src   = source.str.contains("fdot", na=False)

    conf[(is_custom_src & custom_osm_match).values]  += 0.12
    conf[(is_fdot_src   & fdot_osm_match).values]    += 0.10
    conf[(is_custom_src & custom_fdot_match).values] += 0.05

    # ── Land-use plausibility ─────────────────────────────────────────────────
    if "landuse" in df.columns:
        for lu, (lo, hi) in LANDUSE_SPEED_RANGES.items():
            mask = (df["landuse"] == lu) & estimated.notna() & (estimated >= lo) & (estimated <= hi)
            conf[mask.values] += 0.08
        severe = (df["landuse"] == "residential") & (estimated > 45)
        conf[severe.values] -= 0.10

    # ── Multi-source agreement ────────────────────────────────────────────────
    fdot_v   = fdot_speed.fillna(-999)
    custom_v = custom_speed.fillna(-999)
    osm_v    = osm_speed.fillna(-999)

    osm_fdot_agree   = (np.abs(osm_v - fdot_v)   <= 5) & fdot_speed.notna()   & osm_speed.notna()
    osm_custom_agree = (np.abs(osm_v - custom_v)  <= 5) & custom_speed.notna() & osm_speed.notna() & (custom_speed != 35)
    fdot_custom_agree= (np.abs(fdot_v - custom_v) <= 5) & fdot_speed.notna()   & custom_speed.notna() & (custom_speed != 35)

    two_agree         = osm_fdot_agree | osm_custom_agree
    triple_agree      = osm_fdot_agree & osm_custom_agree
    fdot_county_agree = fdot_custom_agree & ~osm_speed.notna()

    conf[two_agree.values]         += 0.12
    conf[triple_agree.values]      += 0.18
    conf[fdot_county_agree.values] += 0.08

    df["speed_limit_confidence_score"] = np.clip(conf, 0, 1)

    # ── Summary ───────────────────────────────────────────────────────────────
    cov  = estimated.notna().mean() * 100
    cscr = df["speed_limit_confidence_score"]
    log.info("Speed coverage: %.1f%%", cov)
    log.info("High confidence (≥0.6): %d", (cscr >= 0.6).sum())
    log.info("Medium (0.3–0.6):       %d", ((cscr >= 0.3) & (cscr < 0.6)).sum())
    log.info("Low (<0.3):             %d", (cscr < 0.3).sum())

    return df
