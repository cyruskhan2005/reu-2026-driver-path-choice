"""
roadnet.pipeline
================
Top-level orchestrator that wires together all pipeline stages for a
single county or a list of counties.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd

from .config import (
    CountyConfig,
    PipelineConfig,
    WGS84,
)
from .osm        import download_county, load_county
from .mapillary  import fetch_signs, attach_signs_to_edges
from .fdot       import extract_fdot_layers, load_fdot
from .conflation import conflate_fdot, conflate_custom
from .speed      import resolve_connector_speeds, add_landuse, arbitrate_speed
from .fmm_pipeline import (
    find_sessions,
    CountyAssigner,
    build_master_gps_parquet,
    run_county as fmm_run_county,
)

log = logging.getLogger(__name__)


def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


class Pipeline:
    def __init__(self, cfg: PipelineConfig, log_level: int = logging.INFO) -> None:
        self.cfg = cfg
        _setup_logging(log_level)

    def run(self, counties: Optional[list[str]] = None) -> dict[str, gpd.GeoDataFrame]:
        cfg = self.cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        # ── FDOT (shared across counties) ─────────────────────────────────────
        fdot_gdf: Optional[gpd.GeoDataFrame] = None
        if cfg.fdot_gdb and cfg.fdot_gdb.exists():
            fdot_dir = cfg.output_dir / "fdot"
            county_names = [c.fdot_county_name for c in cfg.counties if c.fdot_county_name]
            county_codes = [c.fdot_county_code for c in cfg.counties if c.fdot_county_code]
            projected_crs = cfg.counties[0].projected_crs if cfg.counties else "EPSG:26917"
            _, fdot_gdf = extract_fdot_layers(
                gdb_path      = cfg.fdot_gdb,
                out_dir       = fdot_dir,
                county_names  = county_names or None,
                county_codes  = county_codes or None,
                projected_crs = projected_crs,
                skip_if_exists= cfg.skip_osm,
            )
        else:
            log.warning("No FDOT GDB path set — FDOT conflation will be skipped")

        # ── Per-county processing ─────────────────────────────────────────────
        target_counties = [
            c for c in cfg.counties
            if counties is None or c.name in counties
        ]

        results: dict[str, gpd.GeoDataFrame] = {}
        for county in target_counties:
            log.info("\n%s\nProcessing: %s\n%s", "=" * 60, county.name, "=" * 60)
            enriched = self._run_county(county, fdot_gdf)
            results[county.name] = enriched

        # ── FMM stage ─────────────────────────────────────────────────────────
        if not cfg.skip_fmm and cfg.gps_root and cfg.gps_root.exists():
            self._run_fmm(target_counties, results)

        log.info("Pipeline complete in %.1f min", (time.time() - t0) / 60)
        return results

    def _run_county(
        self,
        county: CountyConfig,
        fdot_gdf: Optional[gpd.GeoDataFrame],
    ) -> gpd.GeoDataFrame:
        cfg     = self.cfg
        out_dir = cfg.county_output(county)

        nodes, edges, landuse = download_county(
            county, out_dir, skip_if_exists=cfg.skip_osm
        )

        bounds_wgs = (
            nodes.to_crs(WGS84).geometry if nodes.crs and nodes.crs.to_epsg() != 4326
            else nodes.geometry
        )
        west  = bounds_wgs.x.min()
        east  = bounds_wgs.x.max()
        south = bounds_wgs.y.min()
        north = bounds_wgs.y.max()

        signs_raw = fetch_signs(
            token     = cfg.mly_token,
            bounds    = (west, south, east, north),
            out_path  = out_dir / "mly_signs_raw.parquet",
            step      = cfg.mly_grid_step,
            overlap   = cfg.mly_grid_overlap,
            workers   = cfg.mly_workers,
            skip_if_exists = cfg.skip_mly,
        )

        edges_with_mly = attach_signs_to_edges(
            signs_raw     = signs_raw,
            nodes         = nodes,
            edges         = edges,
            projected_crs = county.projected_crs,
            out_path      = out_dir / "osm_edges_with_mly.parquet",
            landuse       = landuse,
            skip_if_exists= cfg.skip_conflation,
        )

        if cfg.skip_conflation and (out_dir / "enriched_network.parquet").exists():
            log.info("[%s] Loading cached enriched network", county.name)
            return gpd.read_parquet(out_dir / "enriched_network.parquet")

        network = edges_with_mly.copy()

        if "fid" not in network.columns:
            network = network.reset_index(drop=True)
            network["fid"] = network.index
        network = network.set_index("fid")
        network.index.name = "fid"

        if fdot_gdf is not None:
            log.info("[%s] Conflating FDOT …", county.name)
            network = conflate_fdot(network, fdot_gdf)

        if county.custom_geojson and county.custom_geojson.exists():
            log.info("[%s] Conflating custom GeoJSON %s …", county.name, county.custom_geojson.name)
            custom_gdf = gpd.read_file(county.custom_geojson).to_crs(county.projected_crs)
            col_map    = self._build_custom_col_map(county)
            network    = conflate_custom(
                network, custom_gdf, col_map,
                prefix   = county.slug[:8],
                min_vote = county.custom_min_vote,
            )
        elif county.custom_geojson:
            log.warning("[%s] Custom GeoJSON not found: %s", county.name, county.custom_geojson)

        if "maxspeed" in network.columns:
            network["osm_maxspeed"] = (
                network["maxspeed"].astype(str)
                .str.extract(r"(\d+)", expand=False)
                .astype(float)
            )

        network["length"]        = network["length"].astype(float)
        network["is_roundabout"] = network.get("junction", pd.Series()) == "roundabout"
        network["is_connector"]  = network["highway"].str.contains("link", na=False)

        network_df = pd.DataFrame(network)
        network_df = resolve_connector_speeds(network_df)
        for col in ("osm_maxspeed", "connector_transition", "connector_upstream", "connector_downstream"):
            if col in network_df.columns:
                network[col] = network_df[col].values

        network = add_landuse(network, landuse)

        has_miami  = "COUNTY_SPEED" in network.columns
        has_pbc    = "PBC_SPEED"    in network.columns
        has_custom = county.custom_speed_col is not None

        custom_speed_col = None
        if has_custom and county.custom_speed_col:
            custom_speed_col = f"CUSTOM_SPEED_{county.slug}"
            src_col = self._resolved_custom_col(county, "speed")
            if src_col and src_col in network.columns:
                network[custom_speed_col] = network[src_col]

        custom_name_col  = self._resolved_custom_col(county, "name")
        custom_owner_col = self._resolved_custom_col(county, "owner")

        network_df = arbitrate_speed(
            pd.DataFrame(network),
            has_fdot         = fdot_gdf is not None,
            has_miami        = has_miami,
            has_pbc          = has_pbc,
            custom_speed_col = custom_speed_col,
            custom_owner_col = custom_owner_col,
            custom_name_col  = custom_name_col,
        )
        for col in ("estimated_speed_limit", "speed_source", "speed_limit_confidence_score"):
            if col in network_df.columns:
                network[col] = network_df[col].values

        def _any_notna(*cols) -> pd.Series:
            result = pd.Series(False, index=network.index)
            for col in cols:
                if col in network.columns:
                    numeric = pd.to_numeric(network[col], errors="coerce").fillna(0)
                    result = result | (numeric > 0)
            return result

        network["has_stop_sign_u"]      = _any_notna("MAP_has_stop_u", "OSM_has_stop_u")
        network["has_stop_sign_v"]      = _any_notna("MAP_has_stop_v", "OSM_has_stop_v")
        network["has_yield_u"]          = _any_notna("MAP_has_yield_u", "OSM_has_yield_u")
        network["has_yield_v"]          = _any_notna("MAP_has_yield_v", "OSM_has_yield_v")
        network["has_traffic_signal_u"] = _any_notna("OSM_has_signal_u", "MAP_has_signal_u")
        network["has_traffic_signal_v"] = _any_notna("OSM_has_signal_v", "MAP_has_signal_v")

        log.info(
            "[%s] Control nodes — stop_u: %d  stop_v: %d  yield_u: %d  yield_v: %d  signal_u: %d  signal_v: %d",
            county.name,
            network["has_stop_sign_u"].sum(), network["has_stop_sign_v"].sum(),
            network["has_yield_u"].sum(),     network["has_yield_v"].sum(),
            network["has_traffic_signal_u"].sum(), network["has_traffic_signal_v"].sum(),
        )

        # Ensure MAP/OSM control columns are stored as integers not strings.
        for col in network.columns:
            if col.startswith("MAP_") or col.startswith("OSM_has_"):
                network[col] = pd.to_numeric(network[col], errors="coerce").fillna(0).astype(int)

        network = network.drop(columns=["maxspeed", "key"], errors="ignore")
        out_path = out_dir / "enriched_network.parquet"
        network.to_parquet(out_path)
        log.info("[%s] Saved enriched network → %s", county.name, out_path)

        self._write_fmm_shp(county, network, out_dir)

        return network

    def _run_fmm(
        self,
        counties: list[CountyConfig],
        networks: dict[str, gpd.GeoDataFrame],
    ) -> None:
        cfg      = self.cfg
        sessions = find_sessions(cfg.gps_root)
        log.info("Found %d GPS sessions", len(sessions))

        county_shapefiles: dict[str, gpd.GeoDataFrame] = {}
        for county in counties:
            shp_path = cfg.county_output(county) / "fmm" / "edges.shp"
            if shp_path.exists():
                county_shapefiles[county.name] = gpd.read_file(str(shp_path))

        if not county_shapefiles:
            log.warning("No FMM shapefiles found — skipping map-matching")
            return

        assigner = CountyAssigner(county_shapefiles)

        # ── Build master GPS parquet once for all counties ────────────────────
        master_cache = cfg.gps_root / "gps_master.parquet"
        master_gps_df = build_master_gps_parquet(sessions, assigner, master_cache)

        for county in counties:
            shp_dir = cfg.county_output(county) / "fmm"
            if not shp_dir.exists():
                log.warning("[%s] FMM directory not found — skipping", county.name)
                continue
            fmm_run_county(
                county_name   = county.name,
                shp_dir       = shp_dir,
                sessions      = sessions,
                assigner      = assigner,
                out_dir       = cfg.county_output(county),
                fmm_bin       = county.fmm_bin,
                master_gps_df = master_gps_df,
            )

    @staticmethod
    def _build_custom_col_map(county: CountyConfig) -> dict[str, str]:
        mapping: dict[str, str] = {}
        if county.custom_speed_col:
            mapping[county.custom_speed_col] = "CUSTOM_SPEED"
        if county.custom_name_col:
            mapping[county.custom_name_col]  = "CUSTOM_NAME"
        if county.custom_lane_col:
            mapping[county.custom_lane_col]  = "CUSTOM_LANES"
        if county.custom_owner_col:
            mapping[county.custom_owner_col] = "CUSTOM_OWNER"
        if county.custom_func_class_col:
            mapping[county.custom_func_class_col] = "CUSTOM_FUNC_CLASS"
        return mapping

    @staticmethod
    def _resolved_custom_col(county: CountyConfig, kind: str) -> Optional[str]:
        return {
            "speed": "CUSTOM_SPEED" if county.custom_speed_col else None,
            "name":  "CUSTOM_NAME"  if county.custom_name_col  else None,
            "owner": "CUSTOM_OWNER" if county.custom_owner_col else None,
        }.get(kind)

    @staticmethod
    def _write_fmm_shp(
        county: CountyConfig,
        network: gpd.GeoDataFrame,
        out_dir: Path,
    ) -> None:
        fmm_dir = out_dir / "fmm"
        fmm_dir.mkdir(parents=True, exist_ok=True)
        net = network.reset_index()
        fmm_export = net[["fid", "u", "v", "geometry"]].copy()
        fmm_export["fid"] = pd.to_numeric(fmm_export["fid"]).astype(np.int64)
        fmm_export["u"]   = pd.to_numeric(fmm_export["u"]).astype(np.int64)
        fmm_export["v"]   = pd.to_numeric(fmm_export["v"]).astype(np.int64)
        fmm_export = fmm_export.to_crs(WGS84)
        fmm_export.to_file(str(fmm_dir / "edges.shp"), index=False)
        log.info("[%s] FMM shapefile → %s", county.name, fmm_dir / "edges.shp")