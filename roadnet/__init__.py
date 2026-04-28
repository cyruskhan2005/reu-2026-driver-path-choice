"""
roadnet
=======
A modular road-network enrichment library for South Florida (and beyond).

Quick start::

    from pathlib import Path
    from roadnet import Pipeline, CountyConfig, PipelineConfig

    cfg = PipelineConfig(
        output_dir = Path("outputs"),
        mly_token  = "MLY|...",
        fdot_gdb   = Path("DOTShapesFGDB.gdb"),
        counties   = [
            CountyConfig(
                name        = "Miami-Dade County",
                place_query = "Miami-Dade County, Florida, USA",
                # optional custom county road file:
                custom_geojson    = Path("miami.geojson"),
                custom_speed_col  = "SPEEDLIMIT",
                custom_name_col   = "SNAME",
            ),
        ],
        gps_root = Path("kingston_miami"),
    )

    pipe = Pipeline(cfg)
    results = pipe.run()
    # results["Miami-Dade County"]  →  enriched GeoDataFrame
"""

from .config import CountyConfig, PipelineConfig
from .pipeline import Pipeline

__all__ = ["CountyConfig", "PipelineConfig", "Pipeline"]
__version__ = "1.0.0"
