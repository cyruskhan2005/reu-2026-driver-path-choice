"""
run.py
======
Entry point for the roadnet pipeline.
Reads a YAML config file and runs the pipeline.
"""
import argparse
import logging
from pathlib import Path

import yaml

from roadnet import CountyConfig, PipelineConfig, Pipeline


def load_config(yaml_path: str) -> PipelineConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    counties = []
    for c in raw.get("counties", []):
        counties.append(CountyConfig(
            name                 = c["name"],
            place_query          = c["place_query"],
            custom_geojson       = Path(c["custom_geojson"])       if c.get("custom_geojson")       else None,
            custom_speed_col     = c.get("custom_speed_col"),
            custom_name_col      = c.get("custom_name_col"),
            custom_lane_col      = c.get("custom_lane_col"),
            custom_owner_col     = c.get("custom_owner_col"),
            custom_func_class_col= c.get("custom_func_class_col"),
            custom_min_vote      = c.get("custom_min_vote", 0.5),
            fdot_county_name     = c.get("fdot_county_name"),
            fdot_county_code     = c.get("fdot_county_code"),
        ))

    return PipelineConfig(
        output_dir       = Path(raw["output_dir"]),
        mly_token        = raw["mly_token"],
        fdot_gdb         = Path(raw["fdot_gdb"])   if raw.get("fdot_gdb")   else None,
        gps_root         = Path(raw["gps_root"])   if raw.get("gps_root")   else None,
        counties         = counties,
        mly_grid_step    = raw.get("mly_grid_step",    0.009),
        mly_grid_overlap = raw.get("mly_grid_overlap", 0.001),
        mly_workers      = raw.get("mly_workers",      32),
        skip_osm         = raw.get("skip_osm",         False),
        skip_mly         = raw.get("skip_mly",         False),
        skip_conflation  = raw.get("skip_conflation",  False),
        skip_fmm         = raw.get("skip_fmm",         False),
    )


def list_fdot_counties(gdb_path: str) -> None:
    """Write all unique COUNTY / COUNTYDOT pairs to fdot_counties.txt."""
    import geopandas as gpd
    gdb = Path(gdb_path)
    if not gdb.exists():
        print(f"ERROR: GDB not found at {gdb_path}")
        return
    print(f"Reading FDOT roadway layer from {gdb_path} …")
    try:
        df = gpd.read_file(gdb, layer="roadway", engine="pyogrio",
                           columns=["COUNTY", "COUNTYDOT"])
    except Exception as e:
        print(f"ERROR: Could not read roadway layer: {e}")
        return

    pairs = (
        df[["COUNTY", "COUNTYDOT"]]
        .dropna(subset=["COUNTY"])
        .drop_duplicates()
        .sort_values("COUNTY")
    )
    out_path = Path("fdot_counties.txt")
    with open(out_path, "w") as f:
        f.write("{:<30} {}\n".format("COUNTY", "COUNTYDOT"))
        f.write("-" * 45 + "\n")
        for _, row in pairs.iterrows():
            f.write("{:<30} {}\n".format(str(row["COUNTY"]), row["COUNTYDOT"]))
        f.write(f"\nTotal: {len(pairs)} counties\n")
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the roadnet enrichment pipeline")
    parser.add_argument("config", nargs="?", help="Path to YAML config file")
    parser.add_argument("--list-fdot-counties", metavar="GDB_PATH",
                        help="List all FDOT county names and codes from the given GDB and exit")
    parser.add_argument("--counties", nargs="+", help="Only process these counties (by name)")
    parser.add_argument("--skip-osm",        action="store_true")
    parser.add_argument("--skip-mly",        action="store_true")
    parser.add_argument("--skip-conflation", action="store_true")
    parser.add_argument("--skip-fmm",        action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    if args.list_fdot_counties:
        list_fdot_counties(args.list_fdot_counties)
        return

    if not args.config:
        parser.print_help()
        return

    cfg = load_config(args.config)

    # CLI flags override YAML
    if args.skip_osm:        cfg.skip_osm        = True
    if args.skip_mly:        cfg.skip_mly        = True
    if args.skip_conflation: cfg.skip_conflation = True
    if args.skip_fmm:        cfg.skip_fmm        = True

    log_level = getattr(logging, args.log_level)
    Pipeline(cfg, log_level=log_level).run(counties=args.counties)


if __name__ == "__main__":
    main()