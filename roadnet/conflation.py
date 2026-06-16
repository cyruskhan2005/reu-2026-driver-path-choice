"""
roadnet.conflation
==================
Generic bearing-aware spatial conflation of two road-network GeoDataFrames.

The algorithm works by:
  1. Sampling voter points along each source edge (clipping endpoints to
     avoid intersection artefacts).
  2. Performing a nearest-neighbour spatial join to the target layer
     within ``max_dist`` metres.
  3. Filtering matches by angular parallelism (``angle_tol`` degrees).
  4. Tallying how many source-edge points voted for each target edge, then
     keeping only matches where the vote ratio exceeds ``min_vote_ratio``.
  5. Optionally rejecting overpass / underpass pairs that differ in bridge /
     tunnel / layer attributes.
  6. (FDOT only) Among tied candidates, preferring the FDOT road whose
     FDOT_DESCR contains the OSM base name of the source edge.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import substring

from .config import (
    CONFLATION_MAX_DIST_M,
    CONFLATION_ANGLE_TOL_DEG,
    CONFLATION_MIN_VOTE,
    CONFLATION_CLIP_OFFSET,
)

log = logging.getLogger(__name__)


# ── Name normalisation ────────────────────────────────────────────────────────

# Directional suffixes that appear at the end of OSM names or inside FDOT DESCR
_DIRECTIONAL = re.compile(
    r"\b(northbound|southbound|eastbound|westbound|nb|sb|eb|wb)\b",
    re.IGNORECASE,
)

# Road-type suffixes to strip so "NW 7th Ave" and "NW 7TH AVENUE" both reduce
# to "NW 7TH"
_ROAD_TYPES = re.compile(
    r"\b(street|st|avenue|ave|boulevard|blvd|road|rd|drive|dr|lane|ln"
    r"|court|ct|circle|cir|place|pl|way|trail|trl|highway|hwy|parkway|pkwy"
    r"|expressway|expy|freeway|fwy|turnpike|tpke?|causeway|cswy"
    r"|terrace|ter|trace|trce|run|loop|path)\b\.?",
    re.IGNORECASE,
)

# Collapse leftover whitespace / punctuation
_WHITESPACE = re.compile(r"[\s\-/]+")


def _base_name(name: str | None) -> str:
    """
    Return a normalised base name suitable for fuzzy matching.

    Steps:
      1. Upper-case
      2. Strip directional suffixes (NB / SB / EB / WB / *BOUND)
      3. Strip road-type words (ST / AVE / BLVD / RD …)
      4. Collapse whitespace
      5. Strip leading/trailing whitespace
    """
    if not name or not isinstance(name, str):
        return ""
    s = name.upper()
    s = _DIRECTIONAL.sub(" ", s)
    s = _ROAD_TYPES.sub(" ", s)
    s = _WHITESPACE.sub(" ", s)
    return s.strip()


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _get_bearing_vectorized(gdf: gpd.GeoDataFrame) -> pd.Series:
    if gdf.empty:
        return pd.Series(dtype=float)
    exploded = gdf.explode(index_parts=False)
    exploded = exploded[exploded.geometry.type.isin(["LineString", "LinearRing"])]

    def _ends(g: object) -> tuple:
        if g and not g.is_empty:
            return g.coords[0], g.coords[-1]
        return (None, None), (None, None)

    coords = exploded.geometry.apply(_ends)
    arr    = np.array(coords.tolist())
    dx     = arr[:, 1, 0] - arr[:, 0, 0]
    dy     = arr[:, 1, 1] - arr[:, 0, 1]
    bearings = np.degrees(np.arctan2(dx, dy)) % 180
    return pd.Series(bearings, index=exploded.index).groupby(level=0).first()


def _clip_endpoints(
    geom: object,
    offset: float = CONFLATION_CLIP_OFFSET,
) -> object:
    """Remove ``offset`` metres from both ends of a LineString/MultiLineString."""
    if geom is None or geom.is_empty:
        return geom
    if isinstance(geom, LineString):
        return geom if geom.length <= offset * 2 else substring(geom, offset, geom.length - offset)
    if isinstance(geom, MultiLineString):
        parts = [
            substring(p, offset, p.length - offset) if p.length > offset * 2 else p
            for p in geom.geoms
        ]
        return MultiLineString(parts)
    return geom


def _extract_coords(geom: object) -> list[tuple]:
    if geom is None or geom.is_empty:
        return []
    if hasattr(geom, "coords"):
        return list(geom.coords)
    return []


def _explode_to_subsegments(
    gdf: gpd.GeoDataFrame,
    step_m: float = 20.0,
) -> gpd.GeoDataFrame:
    """
    Split every geometry in *gdf* into sub-segments of at most *step_m* metres.
    Each sub-segment row inherits all columns from the parent row plus a
    ``local_bearing`` computed from its own start and end coordinates.

    This replaces the global start-to-end bearing with a bearing that reflects
    the **local direction** of the geometry at each point — critical for long
    curved FDOT segments whose overall bearing is misleading.
    """
    rows = []
    orig_idx = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Collect all individual LineString parts
        if geom.geom_type == "LineString":
            parts = [geom]
        elif geom.geom_type == "MultiLineString":
            parts = list(geom.geoms)
        else:
            parts = [geom]

        for part in parts:
            if part.is_empty or part.length == 0:
                continue
            n_steps = max(1, int(np.ceil(part.length / step_m)))
            distances = np.linspace(0, part.length, n_steps + 1)
            for i in range(len(distances) - 1):
                d0, d1 = distances[i], distances[i + 1]
                p0 = part.interpolate(d0)
                p1 = part.interpolate(d1)
                dx = p1.x - p0.x
                dy = p1.y - p0.y
                bearing = np.degrees(np.arctan2(dx, dy)) % 180
                seg = LineString([(p0.x, p0.y), (p1.x, p1.y)])
                r = row.to_dict()
                r["geometry"]      = seg
                r["local_bearing"] = bearing
                rows.append(r)
                orig_idx.append(idx)

    if not rows:
        return gpd.GeoDataFrame(columns=list(gdf.columns) + ["local_bearing"], crs=gdf.crs)

    result = gpd.GeoDataFrame(rows, crs=gdf.crs)
    result["_orig_idx"] = orig_idx
    return result


# ── Main conflation function ──────────────────────────────────────────────────

def conflate(
    source_gdf: gpd.GeoDataFrame,
    target_gdf: gpd.GeoDataFrame,
    target_cols: list[str],
    max_dist: float             = CONFLATION_MAX_DIST_M,
    angle_tol: float            = CONFLATION_ANGLE_TOL_DEG,
    clip_target: bool           = True,
    target_offset: float        = CONFLATION_CLIP_OFFSET,
    clip_source: bool           = True,
    source_offset: float        = 5.0,
    min_vote_ratio: float       = CONFLATION_MIN_VOTE,
    check_vertical_separation: bool = True,
    label: str                  = "target",
    # Name-match tiebreaker
    source_name_col: str | None = None,
    target_name_col: str | None = None,
) -> pd.DataFrame:
    """
    Bearing-aware spatial conflation.

    Parameters
    ----------
    source_gdf:
        The network whose edges will receive attributes (indexed by ``fid``).
    target_gdf:
        The reference dataset that supplies the attributes.
    target_cols:
        Column names in *target_gdf* to transfer.
    max_dist:
        Maximum perpendicular snap distance in metres.
    angle_tol:
        Maximum allowed bearing difference (degrees) for a match.
    clip_target / target_offset:
        Whether to clip ``target_offset`` metres from target-edge endpoints.
    clip_source / source_offset:
        Whether to clip ``source_offset`` metres from source-edge endpoints
        before sampling voter points.
    min_vote_ratio:
        Fraction of source-edge voter points that must match a target edge.
    check_vertical_separation:
        Reject matches where source and target differ in bridge/tunnel/layer.
    label:
        Human-readable label for logging.
    source_name_col:
        Column in *source_gdf* containing the road name (e.g. ``"name"``).
        When supplied together with *target_name_col*, a name-match bonus is
        applied as a tiebreaker after vote filtering.
    target_name_col:
        Column in *target_gdf* containing the road name to match against
        (e.g. ``"FDOT_DESCR"``).

    Returns
    -------
    DataFrame indexed by source ``fid`` with *target_cols* + diagnostics.
    """
    if source_gdf.index.name != "fid":
        source_gdf = source_gdf.copy()
        source_gdf.index.name = "fid"

    # ── Prepare target ────────────────────────────────────────────────────────
    target = target_gdf.copy()
    if clip_target:
        target["geometry"] = target.geometry.apply(
            lambda g: _clip_endpoints(g, offset=target_offset)
        )

    # Explode into 20m sub-segments so bearing reflects local direction
    target_segs = _explode_to_subsegments(target, step_m=20.0)
    target_segs["target_bearing"] = target_segs["local_bearing"]
    log.debug("[%s] %d target features → %d sub-segments", label, len(target), len(target_segs))

    # ── Sample voter points from source edges ─────────────────────────────────
    source_point_counts: dict[int, int] = {}
    voter_data: list[dict] = []

    for idx, geom in source_gdf.geometry.items():
        if geom is None or geom.is_empty:
            source_point_counts[idx] = 1
            continue

        if clip_source:
            geom = _clip_endpoints(geom, offset=source_offset) or geom

        if geom.geom_type == "LineString":
            parts = [geom]
        elif geom.geom_type == "MultiLineString":
            parts = list(geom.geoms)
        else:
            parts = [geom]

        voters_for_edge = []
        for part in parts:
            if part.is_empty or part.length == 0:
                continue
            n_steps   = max(1, int(np.ceil(part.length / 20.0)))
            distances = np.linspace(0, part.length, n_steps + 1)
            for i in range(len(distances) - 1):
                p0 = part.interpolate(distances[i])
                p1 = part.interpolate(distances[i + 1])
                dx = p1.x - p0.x
                dy = p1.y - p0.y
                local_bearing = np.degrees(np.arctan2(dx, dy)) % 180
                mx = (p0.x + p1.x) / 2
                my = (p0.y + p1.y) / 2
                voters_for_edge.append((mx, my, local_bearing))

        if not voters_for_edge:
            mid = geom.interpolate(0.5, normalized=True)
            voters_for_edge = [(mid.x, mid.y, 0.0)]

        source_point_counts[idx] = len(voters_for_edge)
        for x, y, bearing in voters_for_edge:
            voter_data.append({"fid": idx, "x": x, "y": y, "source_bearing": bearing})

    if not voter_data:
        return pd.DataFrame(index=source_gdf.index)

    voters_df = pd.DataFrame(voter_data)
    voters = gpd.GeoDataFrame(
        voters_df[["fid", "source_bearing"]],
        geometry=gpd.points_from_xy(voters_df["x"], voters_df["y"]),
        crs=source_gdf.crs,
    )

    # ── Spatial join against sub-segments ─────────────────────────────────────
    seg_cols = ["target_bearing", "_orig_idx", "geometry"] + [
        c for c in target_cols if c in target_segs.columns
    ]
    joined = gpd.sjoin_nearest(
        voters,
        target_segs[seg_cols],
        max_distance=max_dist,
        distance_col="snap_dist",
    )
    if joined.empty:
        return pd.DataFrame(index=source_gdf.index)

    joined["index_right"] = joined["index_right"].map(target_segs["_orig_idx"])

    # ── Bearing filter ────────────────────────────────────────────────────────
    angle_diff    = np.abs(joined["source_bearing"] - joined["target_bearing"]) % 180
    parallel_mask = (angle_diff < angle_tol) | (angle_diff > (180 - angle_tol))
    valid = joined[parallel_mask].copy()
    if valid.empty:
        return pd.DataFrame(index=source_gdf.index)

    # ── Vertical-separation filter ────────────────────────────────────────────
    if check_vertical_separation:
        def _layer_or_zero(df: pd.DataFrame) -> pd.Series:
            if "layer" not in df.columns:
                return pd.Series(0, index=df.index)
            return pd.to_numeric(df["layer"], errors="coerce").fillna(0)

        src_layer  = _layer_or_zero(source_gdf)
        src_bridge = source_gdf["bridge"].notna() if "bridge" in source_gdf.columns else pd.Series(False, index=source_gdf.index)
        src_tunnel = source_gdf["tunnel"].notna() if "tunnel" in source_gdf.columns else pd.Series(False, index=source_gdf.index)
        src_elevated = src_bridge | (src_layer >= 1)

        tgt_bridge     = target["bridge"].notna()       if "bridge"       in target.columns else pd.Series(False, index=target.index)
        tgt_fdot_br    = target["FDOT_BRIDGES"].notna() if "FDOT_BRIDGES" in target.columns else pd.Series(False, index=target.index)
        tgt_layer      = _layer_or_zero(target)
        tgt_has_bridge = tgt_bridge | tgt_fdot_br | (tgt_layer >= 1)
        tgt_tunnel     = target["tunnel"].notna() if "tunnel" in target.columns else pd.Series(False, index=target.index)

        valid["_src_elevated"] = valid["fid"].map(src_elevated)
        valid["_tgt_bridge"]   = valid["index_right"].map(tgt_has_bridge)
        valid["_src_tunnel"]   = valid["fid"].map(src_tunnel)
        valid["_tgt_tunnel"]   = valid["index_right"].map(tgt_tunnel)

        vert_sep = (
            (valid["_src_elevated"] & ~valid["_tgt_bridge"])
            | (valid["_src_tunnel"]  & ~valid["_tgt_tunnel"])
        )
        rejected = vert_sep.sum()
        if rejected:
            log.debug("[%s] Rejected %d overpass/tunnel matches", label, rejected)
        valid = valid[~vert_sep].drop(
            columns=[c for c in valid.columns if c.startswith("_src_") or c.startswith("_tgt_")]
        )
    if valid.empty:
        return pd.DataFrame(index=source_gdf.index)

    # ── Vote tally ────────────────────────────────────────────────────────────
    vote_summary = (
        valid.groupby(["fid", "index_right"])
        .agg(
            vote_count   =("snap_dist", "count"),
            avg_snap_dist=("snap_dist", "mean"),
        )
        .reset_index()
    )
    vote_summary["total_points"] = vote_summary["fid"].map(source_point_counts)
    vote_summary["vote_ratio"]   = vote_summary["vote_count"] / vote_summary["total_points"]
    vote_summary = vote_summary[vote_summary["vote_ratio"] >= min_vote_ratio]

    if vote_summary.empty:
        return pd.DataFrame(index=source_gdf.index)

    # ── Name-match tiebreaker ─────────────────────────────────────────────────
    # For each source edge, among all surviving candidates:
    #   - compute a name_match flag (1 if OSM base name appears in FDOT_DESCR, else 0)
    #   - sort by (name_match DESC, vote_count DESC, avg_snap_dist ASC)
    # This means a name-matched road wins even if it scored slightly fewer votes
    # than a closer-but-wrongly-named road, while falling back gracefully to
    # nearest when no name match exists.
    if (
        source_name_col is not None
        and target_name_col is not None
        and source_name_col in source_gdf.columns
        and target_name_col in target.columns
    ):
        # Build lookup: fid → normalised OSM base name
        src_names: pd.Series = source_gdf[source_name_col].apply(_base_name)

        # Build lookup: target original index → normalised FDOT description
        tgt_names: pd.Series = target[target_name_col].apply(_base_name)

        vote_summary["_src_base"] = vote_summary["fid"].map(src_names)
        vote_summary["_tgt_base"] = vote_summary["index_right"].map(tgt_names)

        def _name_match(row: pd.Series) -> int:
            src = row["_src_base"]
            tgt = row["_tgt_base"]
            if not src or not tgt:
                return 0
            # Match if the OSM base name appears anywhere inside the FDOT description
            # (handles cases like "NW 7TH" matching "NW 7TH AVE NB")
            return int(src in tgt or tgt in src)

        vote_summary["name_match"] = vote_summary.apply(_name_match, axis=1)

        n_name_helped = (
            vote_summary[vote_summary["name_match"] == 1]
            .drop_duplicates("fid")
            .shape[0]
        )
        log.debug("[%s] Name tiebreaker active; %d edges have a name match",
                  label, n_name_helped)

        winners = (
            vote_summary
            .sort_values(
                ["fid", "name_match", "vote_count", "avg_snap_dist"],
                ascending=[True, False, False, True],
            )
            .drop_duplicates("fid")
            .set_index("fid")
            .drop(columns=["_src_base", "_tgt_base"])
        )
    else:
        # Original behaviour: best vote count, then closest
        winners = (
            vote_summary
            .sort_values(["fid", "vote_count", "avg_snap_dist"], ascending=[True, False, True])
            .drop_duplicates("fid")
            .set_index("fid")
        )

    log.info("[%s] %d / %d source edges matched", label, len(winners), len(source_gdf))

    target_attrs = target[target_cols].copy()
    result = winners.join(target_attrs, on="index_right")
    return result


# ── FDOT helper ───────────────────────────────────────────────────────────────

FDOT_RENAME: dict[str, str] = {
    "ROADWAY":            "FDOT_ROADWAY",
    "DESCR":              "FDOT_DESCR",
    "FUNCLASS":           "FDOT_FUNCTIONAL_CLASS",
    "SPEED":              "FDOT_SPEED",
    "LANE_CNT":           "FDOT_LANE_COUNT",
    "AADT":               "FDOT_AADT",
    "TruckAADT":          "FDOT_TruckAADT",
    "PFC":                "FDOT_RAMP_TYPE",
    "MEDIAN_TYP":         "FDOT_MEDIAN_TYPE",
    "ACCESS_CLA":         "FDOT_ACCESS_CLASS",
    "ROAD_STATU":         "FDOT_ROAD_STATUS",
    "ON_OFF_SYS":         "FDOT_ON_OFF_SYSTEM",
    "STRUCTURE__bridges": "FDOT_BRIDGES",
}

FDOT_COLS = list(FDOT_RENAME.values())


def conflate_fdot(
    network: gpd.GeoDataFrame,
    fdot_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Run FDOT conflation and join results back onto *network*."""
    fdot_renamed = fdot_gdf.rename(columns=FDOT_RENAME)
    available    = [c for c in FDOT_COLS if c in fdot_renamed.columns]

    voter_mask = network["highway"] != "service"
    voter_net  = network[voter_mask].copy()

    matches = conflate(
        voter_net, fdot_renamed, available,
        max_dist=30, angle_tol=30,
        clip_target=True, target_offset=20,
        clip_source=True, source_offset=10,
        min_vote_ratio=0.4,
        label="FDOT",
        # Name-match tiebreaker: prefer FDOT roads whose description
        # contains the OSM road name
        source_name_col="name",
        target_name_col="FDOT_DESCR",
    )

    network = network.drop(columns=available + ["fdot_snap_dist"], errors="ignore")
    result  = matches.drop(
        columns=["vote_count", "index_right", "total_points", "vote_ratio",
                 "name_match"],
        errors="ignore",
    ).rename(columns={"avg_snap_dist": "fdot_snap_dist"})
    return network.join(result)


def conflate_custom(
    network: gpd.GeoDataFrame,
    county_gdf: gpd.GeoDataFrame,
    col_map: dict[str, str],
    prefix: str,
    min_vote: float = 0.5,
) -> gpd.GeoDataFrame:
    """
    Conflate a user-supplied county GeoJSON onto *network*.

    Parameters
    ----------
    col_map:
        Mapping from source column names to target column names,
        e.g. ``{"SPEED_LIM": "CUSTOM_SPEED", "NAME": "CUSTOM_NAME"}``.
    prefix:
        Short identifier used in the ``snap_dist`` column name.
    """
    renamed     = county_gdf.rename(columns=col_map)
    target_cols = [v for v in col_map.values() if v in renamed.columns]

    voter_mask = network["highway"] != "service"
    voter_net  = network[voter_mask].copy()

    matches = conflate(
        voter_net, renamed, target_cols,
        max_dist=15, angle_tol=30,
        clip_target=True, target_offset=5,
        clip_source=True, source_offset=5,
        min_vote_ratio=min_vote,
        label=prefix,
    )

    snap_col = f"{prefix}_snap_dist"
    network  = network.drop(columns=target_cols + [snap_col], errors="ignore")
    result   = matches.drop(
        columns=["vote_count", "index_right", "total_points", "vote_ratio"],
        errors="ignore",
    ).rename(columns={"avg_snap_dist": snap_col})
    return network.join(result)