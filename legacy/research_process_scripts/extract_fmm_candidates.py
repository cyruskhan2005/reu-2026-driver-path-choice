#!/usr/bin/env python3
"""Extract FMM candidate road segments for a small GPS trajectory sample."""
from __future__ import annotations

import argparse
from pathlib import Path

import folium
import pandas as pd
from shapely.geometry import LineString

from research_process_utils import OUTPUT_ROOT, county_dir_name, ensure_dir, load_network_edges, load_raw_gps, write_html


def candidate_value(candidate: object, *names: str):
    for name in names:
        if hasattr(candidate, name):
            return getattr(candidate, name)
    return None


def write_empty(output_dir: Path, reason: str, gps_sample: pd.DataFrame | None = None) -> None:
    columns = ["trajectory_id", "point_index", "lon", "lat", "candidate_rank", "candidate_edge_id", "distance", "error", "offset"]
    pd.DataFrame(columns=columns).to_csv(output_dir / "fmm_candidates_sample.csv", index=False)
    (output_dir / "fmm_candidates_status.txt").write_text(reason + "\n", encoding="utf-8")
    if gps_sample is not None and not gps_sample.empty:
        save_candidate_map(gps_sample, pd.DataFrame(columns=columns), output_dir / "fmm_candidates_sample.html", reason)
    else:
        write_html(output_dir / "fmm_candidates_sample.html", "FMM Candidate Segment Sample", f"<h1>FMM Candidate Segment Sample</h1><p>{reason}</p>")
    print(f"WARNING: {reason}")


def select_trajectory(raw: pd.DataFrame, county: str, max_points: int) -> pd.DataFrame:
    scoped = raw.loc[raw["county"].astype(str).str.lower() == county.lower()].copy()
    if scoped.empty:
        scoped = raw.copy()
    if "source_id" in scoped.columns:
        candidates = scoped.groupby("source_id").size().sort_values(ascending=False)
        if not candidates.empty:
            scoped = scoped.loc[scoped["source_id"] == candidates.index[0]]
    if "datetime" in scoped.columns:
        scoped = scoped.sort_values("datetime")
    elif "timestamp" in scoped.columns:
        scoped = scoped.sort_values("timestamp")
    return scoped.head(max_points).reset_index(drop=True)


def save_candidate_map(gps: pd.DataFrame, candidates: pd.DataFrame, output: Path, note: str = "") -> None:
    center = [float(gps["lat"].mean()), float(gps["lon"].mean())]
    m = folium.Map(location=center, zoom_start=15, tiles="CartoDB positron", control_scale=True)
    folium.PolyLine(gps[["lat", "lon"]].values.tolist(), color="#111827", weight=3, opacity=0.8, tooltip="Sample raw GPS trajectory").add_to(m)
    for row in gps.itertuples():
        folium.CircleMarker([row.lat, row.lon], radius=4, color="#dc2626", fill=True, fill_opacity=0.85, tooltip=f"GPS point {row.Index}").add_to(m)
    if not candidates.empty:
        candidate_fids = pd.to_numeric(candidates["candidate_edge_id"], errors="coerce").dropna().astype(int).unique()
        county = str(gps["county"].iloc[0]) if "county" in gps.columns else "Broward County"
        try:
            edges = load_network_edges(county, candidate_fids, include_geometry=True)
            if edges.crs:
                edges = edges.to_crs(4326)
            for edge in edges.itertuples(index=False):
                geoms = edge.geometry.geoms if edge.geometry.geom_type == "MultiLineString" else [edge.geometry]
                for line in geoms:
                    folium.PolyLine([(lat, lon) for lon, lat in line.coords], color="#2563eb", weight=4, opacity=0.65, tooltip=f"Candidate FID {edge.fid}").add_to(m)
        except Exception as exc:  # noqa: BLE001
            note = f"{note} Candidate table was created, but candidate geometry could not be mapped: {exc}"
    banner = f"""
    <div style="position: fixed; top: 14px; left: 50px; z-index: 9999; background: white;
        border: 1px solid #d1d5db; border-radius: 6px; padding: 10px 12px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 420px;">
        <strong>FMM Candidate Segment Sample</strong><br>{note}
    </div>
    """
    m.get_root().html.add_child(folium.Element(banner))
    m.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--county", default="Broward County")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "fmm_candidates")
    parser.add_argument("--max-points", type=int, default=60)
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    raw = load_raw_gps(max_rows=250_000)
    if raw.empty:
        write_empty(out_dir, "No raw GPS points were available for candidate extraction.")
        return
    sample = select_trajectory(raw, args.county, args.max_points)
    if len(sample) < 2:
        write_empty(out_dir, "A trajectory needs at least two GPS points for FMM matching.", sample)
        return

    try:
        from fmm import FastMapMatch, FastMapMatchConfig, Network, NetworkGraph, UBODT  # type: ignore
    except Exception as exc:  # noqa: BLE001
        write_empty(out_dir, f"FMM Python API is not available in this environment: {exc}", sample)
        return

    county_dir = county_dir_name(args.county)
    network_path = Path("sflorida_outputs") / county_dir / "fmm" / "edges.shp"
    ubodt_path = Path("sflorida_outputs") / county_dir / "fmm" / "ubodt.txt"
    if not network_path.exists() or not ubodt_path.exists():
        write_empty(out_dir, f"Missing FMM network files: {network_path} or {ubodt_path}", sample)
        return

    line = LineString(sample[["lon", "lat"]].itertuples(index=False, name=None)).wkt
    try:
        network = Network(str(network_path), "fid", "u", "v")
        graph = NetworkGraph(network)
        ubodt = UBODT.read_ubodt_csv(str(ubodt_path))
        matcher = FastMapMatch(network, graph, ubodt)
        config = FastMapMatchConfig()
        result = matcher.match_wkt(line, config)
    except Exception as exc:  # noqa: BLE001
        write_empty(out_dir, f"FMM match_wkt failed for the sample trajectory: {exc}", sample)
        return

    if not hasattr(result, "candidates"):
        write_empty(out_dir, "This FMM build did not expose result.candidates; final matching can run, but candidate internals are unavailable.", sample)
        return

    rows = []
    raw_candidates = getattr(result, "candidates", []) or []
    for point_index, candidate_list in enumerate(raw_candidates):
        if isinstance(candidate_list, (list, tuple)):
            iterable = candidate_list
        else:
            iterable = [candidate_list]
        for rank, candidate in enumerate(iterable):
            rows.append(
                {
                    "trajectory_id": sample["source_id"].iloc[0] if "source_id" in sample.columns else "sample",
                    "point_index": point_index,
                    "lon": sample["lon"].iloc[point_index] if point_index < len(sample) else None,
                    "lat": sample["lat"].iloc[point_index] if point_index < len(sample) else None,
                    "candidate_rank": rank,
                    "candidate_edge_id": candidate_value(candidate, "edge_id", "id", "eid", "fid"),
                    "distance": candidate_value(candidate, "distance", "dist"),
                    "error": candidate_value(candidate, "error"),
                    "offset": candidate_value(candidate, "offset"),
                }
            )
    candidates = pd.DataFrame(rows)
    candidates.to_csv(out_dir / "fmm_candidates_sample.csv", index=False)
    save_candidate_map(sample, candidates, out_dir / "fmm_candidates_sample.html", "Blue lines are candidate road segments considered before final matching.")
    print(f"Candidate rows: {len(candidates):,}")
    print(f"Wrote {out_dir.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
