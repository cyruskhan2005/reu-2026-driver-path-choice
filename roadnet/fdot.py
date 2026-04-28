"""
roadnet.fdot
============
Extract and merge FDOT layers from a File Geodatabase into a single
enriched GeoDataFrame for conflation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import geopandas as gpd
from shapely import wkb

log = logging.getLogger(__name__)

_EXCLUDE_LAYERS = {
    "traffic_signal_locations",
    "intersection",
    "railcross",
    "interchange",
    "roadway",
    "weigh_in_motion",
}

_BLOAT_EXACT    = {"OBJECTID", "Shape_Length", "Shape_Leng", "DISTRICT", "COUNTYDOT"}
_BLOAT_PREFIXES = ["OBJECTID_", "DISTRICT_", "COUNTYDOT_"]


def _build_county_filter(
    county_names: list[str],
    county_codes: list[str],
) -> Optional[str]:
    """
    Build a SQL WHERE clause for FDOT county filtering.

    Parameters
    ----------
    county_names:
        Values to match against the FDOT ``COUNTY`` column
        (e.g. ``["Miami-Dade", "Broward", "Palm Beach"]``).
    county_codes:
        Values to match against the FDOT ``COUNTYDOT`` column
        (e.g. ``["87", "86", "93"]``).

    Returns None if both lists are empty (no filter applied).
    """
    parts = []
    if county_names:
        quoted = ", ".join(f"'{n}'" for n in county_names)
        parts.append(f"COUNTY IN ({quoted})")
    if county_codes:
        quoted = ", ".join(f"'{c}'" for c in county_codes)
        parts.append(f"COUNTYDOT IN ({quoted})")
    return " OR ".join(parts) if parts else None


def extract_fdot_layers(
    gdb_path: Path,
    out_dir: Path,
    county_names: Optional[list[str]] = None,
    county_codes: Optional[list[str]] = None,
    projected_crs: str = "EPSG:26917",
    skip_if_exists: bool = False,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Extract every road-centreline layer from *gdb_path*, optionally
    filtering to specific counties, project to *projected_crs*, and
    save each layer as a parquet file inside *out_dir*.

    Parameters
    ----------
    gdb_path:
        Path to the FDOT File Geodatabase (.gdb).
    out_dir:
        Output directory for layer parquets and manifest.
    county_names:
        List of county names to filter (FDOT COUNTY column).
        Pass None or empty list to load all counties.
    county_codes:
        List of COUNTYDOT codes to filter.
        Pass None or empty list to load all counties.
    projected_crs:
        Metre-unit CRS for spatial operations.
    skip_if_exists:
        If True and outputs already exist, load and return cached data.

    Returns
    -------
    manifest : DataFrame
        Columns ``layer``, ``count``, ``path``.
    roadway_gdf : GeoDataFrame
        The merged FDOT feature collection used for conflation.
    """
    manifest_path = out_dir / "fdot_manifest.parquet"
    roadway_path  = out_dir / "fdot_merged.parquet"

    if skip_if_exists and manifest_path.exists() and roadway_path.exists():
        log.info("Loading cached FDOT data from %s", out_dir)
        manifest = pd.read_parquet(manifest_path)
        merged   = _load_merged(roadway_path)
        return manifest, merged

    out_dir.mkdir(parents=True, exist_ok=True)

    county_filter = _build_county_filter(
        county_names or [],
        county_codes or [],
    )
    if county_filter:
        log.info("FDOT county filter: %s", county_filter)
    else:
        log.info("FDOT county filter: none (loading all counties)")

    layers = gpd.list_layers(gdb_path)
    metadata: list[dict] = []

    for layer_name in layers["name"]:
        try:
            kwargs = dict(
                layer  = layer_name,
                engine = "pyogrio",
            )
            if county_filter:
                kwargs["where"] = county_filter

            df = gpd.read_file(gdb_path, **kwargs)
            if df.empty:
                log.debug("FDOT layer %s is empty — skipping", layer_name)
                continue
            df = df.to_crs(projected_crs)
            layer_path = out_dir / f"{layer_name}.parquet"
            df.to_parquet(layer_path)
            metadata.append({
                "layer": layer_name,
                "count": len(df),
                "path":  str(layer_path),
            })
            log.info("Saved FDOT layer %-40s  (%d rows)", layer_name, len(df))
        except Exception as exc:
            log.warning("Error extracting FDOT layer %s: %s", layer_name, exc)

    manifest = pd.DataFrame(metadata)
    manifest.to_parquet(manifest_path, index=False)

    merged = _merge_fdot_layers(manifest, out_dir, projected_crs)
    merged.to_parquet(roadway_path, index=False)
    return manifest, merged


def _load_merged(path: Path) -> gpd.GeoDataFrame:
    raw = pd.read_parquet(path)
    if isinstance(raw["geometry"].iloc[0], (bytes, bytearray)):
        raw["geometry"] = raw["geometry"].apply(wkb.loads)
    return gpd.GeoDataFrame(raw, geometry="geometry", crs="EPSG:26917")


def _merge_fdot_layers(
    manifest: pd.DataFrame,
    out_dir: Path,
    projected_crs: str,
) -> gpd.GeoDataFrame:
    """Join attribute layers onto the base roadway centrelines by ROADWAY key."""
    roadway_row = manifest[manifest["layer"] == "roadway"]
    if roadway_row.empty:
        raise FileNotFoundError("No 'roadway' layer found in FDOT GDB")

    base = _load_merged(Path(roadway_row["path"].iloc[0]))

    for _, row in manifest.iterrows():
        if row["layer"] in _EXCLUDE_LAYERS:
            continue
        try:
            attr_df     = pd.read_parquet(row["path"]).drop(columns=["geometry"], errors="ignore")
            attr_unique = attr_df.groupby("ROADWAY").first().reset_index()
            base = base.merge(
                attr_unique,
                on       = "ROADWAY",
                how      = "left",
                suffixes = ("", f"_{row['layer']}"),
            )
        except Exception as exc:
            log.warning("Error merging FDOT layer %s: %s", row["layer"], exc)

    drop_cols = [
        c for c in base.columns
        if c in _BLOAT_EXACT or any(c.startswith(p) for p in _BLOAT_PREFIXES)
    ]
    base = base.drop(columns=drop_cols, errors="ignore")
    log.info("FDOT merged: %d columns, %d rows", len(base.columns), len(base))
    return base


def load_fdot(out_dir: Path) -> gpd.GeoDataFrame:
    """Load previously built merged FDOT parquet."""
    return _load_merged(out_dir / "fdot_merged.parquet")