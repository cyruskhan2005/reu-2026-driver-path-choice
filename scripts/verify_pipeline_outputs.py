#!/usr/bin/env python3
"""Validate macOS pipeline stages and Driver 1003 deliverables."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Iterable

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("environment", "enrichment", "fmm-prep", "matched", "driver", "all")


class VerificationError(RuntimeError):
    """Raised when one or more required outputs fail validation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _nonempty(path: Path, label: str) -> Path:
    _require(path.is_file(), f"{label} is missing: {path}")
    _require(path.stat().st_size > 0, f"{label} is empty: {path}")
    return path


def _slug(county: str) -> str:
    return county.replace(" ", "_").replace("-", "_")


def _count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            count += block.count(b"\n")
    return count


def _read_config(path: Path) -> dict[str, object]:
    _nonempty(path, "configuration")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _require(isinstance(raw, dict), "configuration root must be a mapping")
    return raw


def _target_counties(raw: dict[str, object], requested: Iterable[str]) -> list[str]:
    configured = [
        str(value.get("name"))
        for value in raw.get("counties", [])
        if isinstance(value, dict) and value.get("name")
    ]
    requested_list = list(requested)
    if requested_list:
        missing = sorted(set(requested_list) - set(configured))
        _require(not missing, f"counties are not configured: {', '.join(missing)}")
        return requested_list
    _require(bool(configured), "configuration contains no counties")
    return configured


def verify_environment(raw: dict[str, object]) -> list[str]:
    import networkx  # noqa: F401
    import numpy  # noqa: F401
    import pyproj  # noqa: F401
    import shapely  # noqa: F401
    import roadnet  # noqa: F401

    messages = [
        f"Python {platform.python_version()} ({platform.machine()}) imports passed",
        "pandas/GeoPandas/Shapely/PyProj/roadnet imports passed",
    ]
    fmm_value = str(raw.get("fmm_bin") or "fmm")
    fmm_path = Path(fmm_value).expanduser() if os.sep in fmm_value else None
    resolved = str(fmm_path.resolve()) if fmm_path and fmm_path.exists() else shutil.which(fmm_value)
    _require(bool(resolved), f"configured FMM executable is not discoverable: {fmm_value}")
    _require(platform.system() == "Darwin", "macOS is required for native FMM verification")
    _require(platform.machine() == "arm64", "arm64 Python is required")
    ubodt = shutil.which("ubodt_gen")
    _require(bool(ubodt), "ubodt_gen is not discoverable")
    for executable in (str(resolved), str(ubodt)):
        architecture = subprocess.run(
            ["file", executable], capture_output=True, text=True, check=False
        )
        _require(architecture.returncode == 0, f"file failed for {executable}")
        _require("arm64" in architecture.stdout, f"native executable is not arm64: {executable}")
        help_result = subprocess.run(
            [executable, "--help"], capture_output=True, text=True, timeout=15, check=False
        )
        _require(help_result.returncode == 0, f"native help command failed: {executable}")
    import_check = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import fmm, _fmm; assert hasattr(fmm, 'FastMapMatch'); print(_fmm.__file__)",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    _require(import_check.returncode == 0, "isolated FMM Python import failed")
    extension = import_check.stdout.strip().splitlines()[-1]
    linkage = subprocess.run(
        ["otool", "-L", extension], capture_output=True, text=True, check=False
    )
    _require(linkage.returncode == 0, "otool failed for the FMM Python extension")
    _require("libpython" not in linkage.stdout, "FMM Python extension directly links libpython")
    _require(
        "@loader_path/libFMMLIB.dylib" in linkage.stdout,
        "FMM Python extension does not use loader-relative libFMMLIB",
    )
    messages.append(f"FMM/ubodt_gen arm64 help passed: {resolved}")
    messages.append("isolated FMM Python import and loader-relative linkage passed")
    return messages


def verify_enrichment(
    output_root: Path, counties: Iterable[str]
) -> list[str]:
    messages: list[str] = []
    required = {"geometry", "estimated_speed_limit", "highway", "length"}
    for county in counties:
        path = _nonempty(
            output_root / _slug(county) / "enriched_network.parquet",
            f"{county} enriched network",
        )
        parquet = pq.ParquetFile(path)
        rows = int(parquet.metadata.num_rows)
        columns = set(parquet.schema.names)
        _require(rows > 0, f"{county} enriched network has no rows")
        missing = sorted(required - columns)
        _require(not missing, f"{county} enriched network lacks columns: {missing}")
        fid_columns = ["estimated_speed_limit", "highway", "length"]
        if "fid" in columns:
            fid_columns.append("fid")
        fid_frame = pd.read_parquet(path, columns=fid_columns)
        if "fid" in fid_frame.columns:
            fid = fid_frame["fid"]
        else:
            _require(
                fid_frame.index.name == "fid",
                f"{county} enriched network lacks a fid column or named index",
            )
            fid = pd.Series(fid_frame.index, index=fid_frame.index)
        _require(fid.notna().all(), f"{county} enriched network contains null FIDs")
        _require(fid.is_unique, f"{county} enriched-network FIDs are not unique")
        geometry = gpd.read_parquet(path, columns=["geometry"])
        _require(geometry.crs is not None, f"{county} enriched network has no CRS")
        _require(not geometry.geometry.is_empty.all(), f"{county} geometry is empty")
        messages.append(
            f"{county}: enriched_network.parquet {rows:,} rows, unique FIDs, CRS {geometry.crs}"
        )
    return messages


def verify_fmm_preparation(
    output_root: Path, counties: Iterable[str]
) -> list[str]:
    messages: list[str] = []
    for county in counties:
        directory = output_root / _slug(county) / "fmm"
        shp = _nonempty(directory / "edges.shp", f"{county} FMM edges shapefile")
        for suffix in (".shx", ".dbf", ".prj", ".cpg"):
            _nonempty(shp.with_suffix(suffix), f"{county} shapefile companion {suffix}")
        edges = gpd.read_file(shp)
        _require(not edges.empty, f"{county} FMM shapefile has no rows")
        missing = sorted({"fid", "u", "v", "geometry"} - set(edges.columns))
        _require(not missing, f"{county} FMM shapefile lacks columns: {missing}")
        _require(edges["fid"].is_unique, f"{county} FMM edge FIDs are not unique")
        enriched_rows = int(
            pq.ParquetFile(output_root / _slug(county) / "enriched_network.parquet")
            .metadata.num_rows
        )
        _require(
            len(edges) == enriched_rows,
            f"{county} FMM edge count differs from enriched network",
        )
        read_test = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "import fmm,sys; net=fmm.Network(sys.argv[1], 'fid', 'u', 'v'); assert net is not None",
                str(shp),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        _require(read_test.returncode == 0, f"{county} FMM Network could not read edges.shp")
        ubodt = _nonempty(directory / "ubodt.txt", f"{county} UBODT")
        _require(ubodt.stat().st_mtime >= shp.stat().st_mtime, f"{county} UBODT is older than edges.shp")
        messages.append(
            f"{county}: edges.shp {len(edges):,} unique edges; ubodt.txt {ubodt.stat().st_size:,} bytes"
        )
    return messages


def _csv_header(path: Path) -> tuple[list[str], str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        sample = handle.read(8192)
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=";,\t").delimiter
    except csv.Error:
        delimiter = ";"
    header = next(csv.reader(sample.splitlines(), delimiter=delimiter), [])
    return header, delimiter


def verify_matched(
    output_root: Path,
    counties: Iterable[str],
    *,
    allow_reused_matched: bool = False,
) -> list[str]:
    messages: list[str] = []
    for county in counties:
        directory = output_root / _slug(county)
        gps = _nonempty(directory / f"{county}_gps.csv", f"{county} FMM GPS CSV")
        matched = _nonempty(
            directory / f"{county}_matched.csv", f"{county} matched CSV"
        )
        if not allow_reused_matched:
            _require(
                matched.stat().st_mtime >= gps.stat().st_mtime,
                f"{county} matched CSV is older than its GPS input",
            )
        gps_header, gps_delimiter = _csv_header(gps)
        matched_header, matched_delimiter = _csv_header(matched)
        _require(
            {"id", "lon", "lat", "timestamp"}.issubset(gps_header),
            f"{county} GPS CSV has an unexpected schema",
        )
        _require(
            {"id", "opath"}.issubset(matched_header),
            f"{county} matched CSV lacks id/opath",
        )
        matched_frame = pd.read_csv(
            matched, sep=matched_delimiter, usecols=["id", "opath"]
        )
        _require(not matched_frame.empty, f"{county} matched CSV has no records")
        _require(
            not matched_frame["id"].duplicated().any(),
            f"{county} matched CSV contains duplicate trip IDs",
        )
        populated = matched_frame["opath"].fillna("").astype(str).str.len().gt(0).mean()
        _require(populated > 0, f"{county} sample has no populated matched paths")
        gps_ids: set[int] = set()
        for chunk in pd.read_csv(
            gps, sep=gps_delimiter, usecols=["id"], chunksize=500_000
        ):
            gps_ids.update(pd.to_numeric(chunk["id"], errors="coerce").dropna().astype(int))
        matched_ids = set(
            pd.to_numeric(matched_frame["id"], errors="coerce").dropna().astype(int)
        )
        _require(matched_ids.issubset(gps_ids), f"{county} matched IDs are absent from GPS input")
        coverage = len(matched_ids) / len(gps_ids) if gps_ids else 0.0
        _require(coverage >= 0.90, f"{county} matched-trip coverage is below 90%")
        gps_rows = max(_count_lines(gps) - 1, 0)
        matched_rows = max(_count_lines(matched) - 1, 0)
        _require(gps_rows > 0 and matched_rows > 0, f"{county} CSV row counts are zero")
        messages.append(
            f"{county}: {gps_rows:,} GPS points, {matched_rows:,} matched trips, {coverage:.1%} ID coverage, opath {populated:.1%} populated"
        )
    return messages


def verify_driver_outputs(output_dir: Path, report: Path) -> list[str]:
    expected = {
        "trip summary": "driver_1003_trip_summary.csv",
        "location clusters": "driver_1003_location_clusters.csv",
        "POI enrichment": "driver_1003_poi_enriched_clusters.csv",
        "activity validation": "driver_1003_activity_role_validation.csv",
        "route families": "driver_1003_route_families.csv",
        "monthly route shares": "driver_1003_route_family_monthly_shares.csv",
        "longitudinal transitions": "driver_1003_longitudinal_route_transitions.csv",
        "behavior JSON": "driver_1003_real_world_behavior_insights.json",
        "verification map": "driver_1003_poi_route_insights_map.html",
    }
    paths = {label: _nonempty(output_dir / name, label) for label, name in expected.items()}
    _nonempty(report, "Driver 1003 RCCI/behavior report")
    trips = pd.read_csv(paths["trip summary"])
    _require(len(trips) > 0, "Driver 1003 trip summary has no rows")
    _require(trips["trip_id"].is_unique, "Driver 1003 trip IDs are duplicated")
    monthly = pd.read_csv(paths["monthly route shares"])
    observed = monthly.loc[monthly["eligible_od_trip_count"] > 0]
    share_sums = observed.groupby(
        ["origin_cluster_id", "destination_cluster_id", "month"]
    )["route_share"].sum()
    _require(((share_sums - 1.0).abs() < 1e-8).all(), "monthly route shares do not sum to one")
    document = paths["behavior JSON"].read_text(encoding="utf-8")
    parsed = json.loads(document)
    _require(parsed.get("driver_id") == 1003, "behavior JSON driver ID is incorrect")
    _require(not parsed.get("likely_workplaces"), "behavior JSON unexpectedly claims a workplace")
    public_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (paths["behavior JSON"], paths["verification map"], report)
    )
    _require("GOOGLE_MAPS_API_KEY" not in public_text, "public output contains a key variable name")
    _require(not re.search(r"AIza[0-9A-Za-z_-]{20,}", public_text), "public output contains a Google key pattern")
    return [
        f"Driver 1003: {len(trips):,} unique trips",
        f"Driver 1003: {len(monthly):,} monthly route-family rows reconcile",
        "Driver 1003: JSON/report/map exist; secret scan passed; no workplace claimed",
    ]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--county", action="append", default=[])
    parser.add_argument(
        "--allow-reused-matched",
        action="store_true",
        help="Allow an explicitly reused matched CSV to predate a regenerated GPS CSV",
    )
    parser.add_argument("--driver-output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "deliverables/driver_1003/route_choice_change_index/visuals/driver_1003_route_choice_change_index_report.html",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        raw = _read_config(args.config)
        output_value = raw.get("output_dir")
        _require(bool(output_value), "configuration output_dir is required")
        output_root = Path(str(output_value)).expanduser().resolve()
        counties = _target_counties(raw, args.county)
        messages: list[str] = []
        if args.stage in {"environment", "all"}:
            messages.extend(verify_environment(raw))
        if args.stage in {"enrichment", "fmm-prep", "matched", "all"}:
            messages.extend(verify_enrichment(output_root, counties))
        if args.stage in {"fmm-prep", "matched", "all"}:
            messages.extend(verify_fmm_preparation(output_root, counties))
        if args.stage in {"matched", "all"}:
            messages.extend(
                verify_matched(
                    output_root,
                    counties,
                    allow_reused_matched=args.allow_reused_matched,
                )
            )
        if args.stage in {"driver", "all"}:
            messages.extend(
                verify_driver_outputs(args.driver_output_dir.resolve(), args.report.resolve())
            )
        print(f"Verification passed: {args.stage}")
        for message in messages:
            print(f"  - {message}")
        return 0
    except (VerificationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
