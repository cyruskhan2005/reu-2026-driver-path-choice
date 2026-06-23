#!/usr/bin/env python3
"""Convert deliverable county GeoParquet layers to shapefile bundles in place.

By default, only the professor-requested GIS deliverable layers are converted
inside each ``*_County`` directory under ``sflorida_outputs``:
``enriched_network.parquet`` and ``osm_nodes.parquet``. A shapefile is a
bundle; GeoPandas writes the required ``.shp``, ``.shx``, ``.dbf``, ``.prj``,
and ``.cpg`` files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd


SHAPEFILE_SUFFIXES = (".shp", ".shx", ".dbf", ".prj", ".cpg")
DEFAULT_LAYERS = ("enriched_network.parquet", "osm_nodes.parquet")


def _dbf_value(value: object) -> object:
    """Return a scalar value supported by the shapefile DBF format."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, set):
        value = sorted(value, key=str)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=str, ensure_ascii=True)
    return value


def _prepare_attributes(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Serialize nested object values while leaving geometry untouched."""
    prepared = frame.copy()
    geometry_name = prepared.geometry.name

    for column in prepared.columns:
        if column == geometry_name:
            continue
        if prepared[column].dtype == "object":
            prepared[column] = prepared[column].map(_dbf_value)
        elif isinstance(prepared[column].dtype, pd.BooleanDtype):
            prepared[column] = prepared[column].astype("Int8")

    return prepared


def _remove_shapefile(path: Path) -> None:
    for suffix in SHAPEFILE_SUFFIXES:
        path.with_suffix(suffix).unlink(missing_ok=True)


def convert_parquet(path: Path, overwrite: bool = False) -> bool:
    """Convert one GeoParquet file and return whether output was written."""
    output = path.with_suffix(".shp")
    if output.exists() and not overwrite:
        print(f"SKIP  {output} already exists (use --overwrite)")
        return False

    try:
        frame = gpd.read_parquet(path)
    except ValueError as exc:
        print(f"SKIP  {path} is not a GeoParquet file: {exc}")
        return False

    if frame.empty:
        print(f"SKIP  {path} contains no features")
        return False
    if frame.crs is None:
        raise ValueError(f"{path} has no CRS; refusing to write an invalid .prj")

    if overwrite:
        _remove_shapefile(output)

    prepared = _prepare_attributes(frame)
    prepared.to_file(output, driver="ESRI Shapefile", index=False, encoding="UTF-8")
    print(f"WROTE {output} ({len(prepared):,} features, CRS {prepared.crs})")
    return True


def county_parquets(
    root: Path,
    counties: Iterable[str],
    layers: Iterable[str],
) -> list[Path]:
    requested = list(counties)
    requested_layers = list(layers)
    directories = (
        [root / county for county in requested]
        if requested
        else sorted(path for path in root.glob("*_County") if path.is_dir())
    )

    missing = [path for path in directories if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "County directories not found: " + ", ".join(map(str, missing))
        )

    inputs: list[Path] = []
    for directory in directories:
        for layer in requested_layers:
            path = directory / layer
            if path.exists():
                inputs.append(path)
            else:
                print(f"SKIP  {path} not found")
    return inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("sflorida_outputs"),
        help="directory containing *_County folders (default: sflorida_outputs)",
    )
    parser.add_argument(
        "--county",
        action="append",
        default=[],
        help="county folder name to convert; repeat for multiple counties",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing shapefile bundles",
    )
    parser.add_argument(
        "--layer",
        action="append",
        default=[],
        help=(
            "GeoParquet filename to convert within each county folder; repeat for "
            "multiple layers (default: enriched_network.parquet and osm_nodes.parquet)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = args.layer or list(DEFAULT_LAYERS)
    inputs = county_parquets(args.root, args.county, layers)
    if not inputs:
        raise SystemExit(f"No county parquet files found under {args.root}")

    written = sum(convert_parquet(path, overwrite=args.overwrite) for path in inputs)
    print(f"Finished: wrote {written} of {len(inputs)} shapefiles")


if __name__ == "__main__":
    main()
