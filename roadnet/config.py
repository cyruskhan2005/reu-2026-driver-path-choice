"""
roadnet.config
==============
Central configuration dataclasses and project-wide constants.
All tuneable parameters live here; nothing is hard-coded elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── CRS ───────────────────────────────────────────────────────────────────────
WGS84     = "EPSG:4326"
UTM17N    = "EPSG:26917"   # South-Florida projected CRS (metres)
WEB_MERC  = "EPSG:3857"


# ── OSM tag settings ──────────────────────────────────────────────────────────
OSM_NODE_TAGS = ["highway", "traffic_signals", "stop", "crossing"]

OSM_WAY_TAGS = [
    "highway", "name", "ref", "lanes", "maxspeed", "oneway",
    "bridge", "tunnel", "layer", "junction",
    "turn:lanes", "surface", "lit", "width", "access",
]

OSM_HIGHWAY_FILTER = (
    '["highway"]["highway"~"motorway|trunk|primary|secondary|tertiary|'
    'residential|unclassified|service|living_street|'
    'motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"]'
    '["access"!~"no"]'
)

OSM_LANDUSE_TAGS = {
    "landuse": [
        "residential", "retail", "commercial", "industrial",
        "construction", "education", "institutional", "forest",
    ]
}


# ── Mapillary ─────────────────────────────────────────────────────────────────
MLY_ENDPOINT = "https://graph.mapillary.com/map_features"

MLY_TARGET_VALUES: set[str] = {
    "regulatory--stop--g1",
    "regulatory--yield--g1",
    "regulatory--end-of-school-zone--g1",
}

MLY_SIGN_LABELS: dict[str, str] = {
    "regulatory--stop--g1":  "mly:Stop",
    "regulatory--yield--g1": "mly:Yield",
}


# ── Snap / conflation thresholds ──────────────────────────────────────────────
NODE_SNAP_M       = 20
BEARING_TOL_DEG   = 50
MAX_EDGE_DIST_M   = 20

CONFLATION_MAX_DIST_M    = 15.0
CONFLATION_ANGLE_TOL_DEG = 40.0
CONFLATION_MIN_VOTE      = 0.4
CONFLATION_CLIP_OFFSET   = 20


# ── Speed / functional-class defaults ────────────────────────────────────────
HIGHWAY_SPEED_DEFAULTS: dict[str, int] = {
    "motorway":      70,
    "motorway_link": 45,
    "trunk":         55,
    "trunk_link":    45,
    "primary":       45,
    "primary_link":  35,
    "secondary":     35,
    "secondary_link":30,
    "tertiary":      30,
    "tertiary_link": 25,
    "residential":   25,
    "unclassified":  25,
    "living_street": 15,
    "service":       15,
}

LINK_TO_PARENT: dict[str, str] = {
    "motorway_link":  "motorway",
    "trunk_link":     "trunk",
    "primary_link":   "primary",
    "secondary_link": "secondary",
    "tertiary_link":  "tertiary",
}

LINK_TYPES = list(LINK_TO_PARENT.keys())


# ── FMM ───────────────────────────────────────────────────────────────────────
FMM_K        = 16
FMM_RADIUS_M = 300
FMM_ERROR_M  = 50
FMM_RETRY_RADIUS_M = 700
FMM_RETRY_ERROR_M  = 100
FMM_MIN_SEGMENT    = 10
FMM_SKIP_ON_GAP    = 5

# ── STMatch (gap bridging) ────────────────────────────────────────────────────
STM_K        = 8
STM_RADIUS_M = 300
STM_ERROR_M  = 50
STM_VMAX_MS  = 40.0   # max vehicle speed m/s (~144 km/h)
STM_FACTOR   = 1.5
GPS_GAP_THRESHOLD_S = 8   # seconds — gaps larger than this trigger STMatch


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class CountyConfig:
    """
    All information needed to process a single county.

    Parameters
    ----------
    name:
        Human-readable name used in column values and file names
        (e.g. ``"Miami-Dade County"``).
    place_query:
        OSMnx place string (e.g. ``"Miami-Dade County, Florida, USA"``).
    projected_crs:
        Metre-unit CRS for spatial operations (default ``UTM17N``).
    custom_geojson:
        Optional path to a county-supplied road-centreline GeoJSON.
        When present the file is conflated after OSM/FDOT/Mapillary.
    custom_speed_col:
        Column in the custom GeoJSON that holds the speed limit.
    custom_name_col:
        Column in the custom GeoJSON that holds the road name.
    custom_lane_col:
        Column in the custom GeoJSON that holds lane count.
    custom_owner_col:
        Column in the custom GeoJSON that holds road-authority / owner.
    custom_func_class_col:
        Column in the custom GeoJSON that holds functional class.
    custom_min_vote:
        Override ``min_vote_ratio`` for this county's custom conflation.
    fmm_bin:
        Path to the ``fmm`` CLI executable.
    """
    name: str
    place_query: str
    projected_crs: str = UTM17N

    # Optional county GeoJSON
    custom_geojson: Optional[Path] = None
    custom_speed_col: Optional[str] = None
    custom_name_col: Optional[str] = None
    custom_lane_col: Optional[str] = None
    custom_owner_col: Optional[str] = None
    custom_func_class_col: Optional[str] = None
    custom_min_vote: float = 0.5

    fmm_bin: str = "fmm"

    # FDOT county filter — name as it appears in FDOT COUNTY column
    # and numeric COUNTYDOT code. If None, no FDOT county filter is applied.
    fdot_county_name: Optional[str] = None
    fdot_county_code: Optional[str] = None

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def slug(self) -> str:
        """File-system-safe name, e.g. ``Miami_Dade_County``."""
        return self.name.replace(" ", "_").replace("-", "_")

    @property
    def lat_bounds(self) -> tuple[float, float]:
        """
        Approximate latitude band for quick GPS point assignment.
        Populated by the pipeline after downloading OSM.
        """
        return getattr(self, "_lat_bounds", (0.0, 90.0))

    @lat_bounds.setter
    def lat_bounds(self, value: tuple[float, float]) -> None:
        self._lat_bounds = value


@dataclass
class PipelineConfig:
    """
    Top-level pipeline settings, shared across all counties.

    Parameters
    ----------
    output_dir:
        Root directory for all outputs.
    mly_token:
        Mapillary API token.
    fdot_gdb:
        Path to the FDOT file-geodatabase (``*.gdb``).
    counties:
        List of county configurations to process.
    mly_grid_step:
        Degree step for Mapillary bounding-box grid (default 0.009°).
    mly_grid_overlap:
        Degree overlap between adjacent grid cells (default 0.001°).
    mly_workers:
        Thread-pool workers for concurrent Mapillary fetches.
    skip_osm:
        Re-use cached OSM parquets instead of re-downloading.
    skip_mly:
        Re-use cached Mapillary parquets.
    skip_conflation:
        Re-use cached conflated network.
    skip_fmm:
        Re-use cached FMM outputs (map-matching only).
    gps_root:
        Root directory that contains raw GPS session folders.
    fmm_bin:
        Path to the ``fmm`` CLI executable.
    """
    output_dir: Path
    mly_token: str
    fdot_gdb: Optional[Path]
    counties: list[CountyConfig]

    mly_grid_step: float    = 0.009
    mly_grid_overlap: float = 0.001
    mly_workers: int        = 32

    skip_osm:         bool = False
    skip_mly:         bool = False
    skip_conflation:  bool = False
    skip_fmm:         bool = False

    gps_root: Optional[Path] = None
    fmm_bin:  str            = "fmm"

    def county_output(self, county: CountyConfig) -> Path:
        """Return (and create) the per-county output subdirectory."""
        p = self.output_dir / county.slug
        p.mkdir(parents=True, exist_ok=True)
        return p