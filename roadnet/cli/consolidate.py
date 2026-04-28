"""
consolidate.py
==============
Enriches each *_fid_aggregated.jsonl file in-place with road attributes
from the enriched_network.parquet for the corresponding county.

For each county it:
  1. Loads the enriched_network.parquet (indexed by fid)
  2. Finds all *_fid_aggregated.jsonl files under gps_root for that county
  3. For each file, joins road attributes onto each FID record and rewrites it

Usage:
    python consolidate.py --config config.yaml
    python consolidate.py --config config.yaml --counties "Broward County"
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

_KEEP_COLS = [
    "estimated_speed_limit", "speed_limit_confidence_score", "FDOT_AADT", "FDOT_TruckAADT",
    "has_stop_sign_u", "has_stop_sign_v", "has_yield_u", "has_yield_v",
    "has_traffic_signal_u", "has_traffic_signal_v", "connector_transition", "name",
]

# Road attribute columns are always overwritten from the enriched network.
# Sensor columns (acc, obd, sog, etc.) that don't appear in _KEEP_COLS are
# never touched.
_ROAD_ATTR_COLS = set(_KEEP_COLS)


def _load_network_attrs(network_path: Path) -> pd.DataFrame:
    """Load enriched network and return a flat DataFrame indexed by fid."""
    log.info("Loading enriched network: %s", network_path)
    network = gpd.read_parquet(network_path)

    net_df = pd.DataFrame(network.reset_index()[
        ["fid"] + [c for c in _KEEP_COLS if c in network.columns]
    ])
    net_df["fid"] = pd.to_numeric(net_df["fid"], errors="coerce").astype("Int64")
    net_df = net_df.dropna(subset=["fid"]).set_index("fid")
    log.info("  %d edges, %d attribute columns", len(net_df), len(net_df.columns))
    return net_df


def _enrich_file(path: Path, attrs: pd.DataFrame) -> int:
    """Rewrite a single *_fid_aggregated.jsonl with road attributes added. Returns rows written."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    if not records:
        return 0

    enriched = []
    for r in records:
        try:
            fid = int(r["fid"])
        except (KeyError, TypeError, ValueError):
            enriched.append(r)
            continue

        if fid in attrs.index:
            row = attrs.loc[fid]
            for col, val in row.items():
                # Always overwrite road attribute columns so re-runs of the
                # pipeline (e.g. after sign propagation fixes) are reflected
                # in existing JSONL files. Sensor columns not in _KEEP_COLS
                # are never touched.
                if col in _ROAD_ATTR_COLS or col not in r:
                    if pd.isna(val):
                        r[col] = None
                    elif hasattr(val, "item"):  # numpy scalar
                        r[col] = val.item()
                    else:
                        r[col] = val
        enriched.append(r)

    with open(path, "w", encoding="utf-8") as f:
        for r in enriched:
            f.write(json.dumps(r) + "\n")

    return len(enriched)


def _consolidate_county(
    county_name: str,
    county_slug: str,
    gps_root: Path,
    output_dir: Path,
) -> None:
    network_path = output_dir / county_slug / "enriched_network.parquet"
    if not network_path.exists():
        log.error("Enriched network not found: %s", network_path)
        return

    attrs = _load_network_attrs(network_path)

    pattern = f"{county_name}_*_fid_aggregated.jsonl"
    files   = list(gps_root.rglob(pattern))
    if not files:
        log.warning("No aggregated files found for %s", county_name)
        return

    log.info("Enriching %d files for %s …", len(files), county_name)
    total_rows = 0
    for path in files:
        n = _enrich_file(path, attrs)
        total_rows += n

    log.info("[%s] Done — %d files, %d total records enriched", county_name, len(files), total_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich fid_aggregated JSONL files with road attributes")
    parser.add_argument("--config",   required=True)
    parser.add_argument("--counties", nargs="+")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output_dir"])
    gps_root   = Path(cfg.get("gps_root", ""))

    if not gps_root.exists():
        log.error("gps_root not set or does not exist")
        return

    counties = cfg.get("counties", [])
    if args.counties:
        counties = [c for c in counties if c["name"] in args.counties]

    for county in counties:
        name = county["name"]
        slug = name.replace(" ", "_").replace("-", "_")
        log.info("\n%s\nConsolidating: %s\n%s", "=" * 60, name, "=" * 60)
        _consolidate_county(name, slug, gps_root, output_dir)


if __name__ == "__main__":
    main()