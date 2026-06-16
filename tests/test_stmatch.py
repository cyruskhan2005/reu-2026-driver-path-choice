"""
test_stmatch.py
===============
Run STMatch via Python bindings on a single *_gps.jsonl file.
No UBODT required — STMatch uses speed constraints instead.

Usage:
    python test_stmatch.py \
        --gps "kingston_miami/.../160811_gps.jsonl" \
        --shp ../sflorida_outputs/Broward_County/fmm/edges.shp

    # Tune parameters:
    --radius  0.003   (degrees, ~300m)
    --error   0.0005  (degrees, ~50m)
    --vmax    30      (m/s, ~108 km/h)
    --factor  1.5
    -k        8
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import folium
import geopandas as gpd
from shapely.geometry import LineString

_GPS_COLOUR  = "#adb5bd"
_EDGE_COLOUR = "#e63946"
_DEG         = 1.1e5


def load_gps(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                d   = json.loads(line)
                loc = d.get("loc", {})
                if not isinstance(loc, dict):
                    continue
                rows.append({
                    "ts":  datetime.fromisoformat(d["@ts"].replace("Z", "+00:00")),
                    "lon": loc["lon"],
                    "lat": loc["lat"],
                    "sog": d.get("sog"),
                })
            except Exception:
                pass
    return sorted(rows, key=lambda r: r["ts"])


def find_gaps(rows: list[dict], threshold_s: float = 60.0) -> list[int]:
    return [
        i for i in range(len(rows) - 1)
        if (rows[i+1]["ts"] - rows[i]["ts"]).total_seconds() > threshold_s
    ]


def build_map(
    gps_rows:  list[dict],
    gap_indices: list[int],
    fids:      list[int],
    shp:       gpd.GeoDataFrame,
    out_html:  str,
    params:    dict,
    mgeom_wkt: str | None = None,
    cpath:     list[int] | None = None,
) -> None:
    lats   = [r["lat"] for r in gps_rows]
    lons   = [r["lon"] for r in gps_rows]
    centre = [sum(lats)/len(lats), sum(lons)/len(lons)]

    m = folium.Map(location=centre, zoom_start=14, tiles="CartoDB positron")

    # Raw GPS
    folium.PolyLine(
        [[r["lat"], r["lon"]] for r in gps_rows],
        color=_GPS_COLOUR, weight=2, opacity=0.7, tooltip="Raw GPS",
    ).add_to(m)
    folium.CircleMarker(
        [gps_rows[0]["lat"],  gps_rows[0]["lon"]],
        radius=6, color="#2d6a4f", fill=True, fill_color="#2d6a4f",
        tooltip=f"Start {gps_rows[0]['ts'].strftime('%H:%M:%S')}",
    ).add_to(m)
    folium.CircleMarker(
        [gps_rows[-1]["lat"], gps_rows[-1]["lon"]],
        radius=6, color="#6d023f", fill=True, fill_color="#6d023f",
        tooltip=f"End {gps_rows[-1]['ts'].strftime('%H:%M:%S')}",
    ).add_to(m)

    # Gap markers
    for idx in gap_indices:
        gap_s = (gps_rows[idx+1]["ts"] - gps_rows[idx]["ts"]).total_seconds()
        folium.Marker(
            [gps_rows[idx]["lat"],   gps_rows[idx]["lon"]],
            icon=folium.Icon(color="orange", icon="pause", prefix="fa"),
            tooltip=f"Gap start — {gap_s:.0f}s",
        ).add_to(m)
        folium.Marker(
            [gps_rows[idx+1]["lat"], gps_rows[idx+1]["lon"]],
            icon=folium.Icon(color="orange", icon="play", prefix="fa"),
            tooltip=f"Gap end — resumed after {gap_s:.0f}s",
        ).add_to(m)

    # Matched edges
    if fids:
        shp_wgs = shp.to_crs("EPSG:4326") if shp.crs and shp.crs.to_epsg() != 4326 else shp
        matched = shp_wgs[shp_wgs["fid"].isin(set(fids))]
        for _, edge in matched.iterrows():
            geom  = edge.geometry
            parts = list(geom.geoms) if geom.geom_type != "LineString" else [geom]
            for part in parts:
                folium.PolyLine(
                    [[y, x] for x, y in part.coords],
                    color=_EDGE_COLOUR, weight=6, opacity=0.85,
                    tooltip=f"FID {int(edge['fid'])}",
                ).add_to(m)

    # cpath edges — all road edges along the matched path
    if cpath:
        shp_wgs = shp.to_crs("EPSG:4326") if shp.crs and shp.crs.to_epsg() != 4326 else shp
        cpath_edges = shp_wgs[shp_wgs["fid"].isin(set(cpath))]
        for _, edge in cpath_edges.iterrows():
            geom  = edge.geometry
            parts = list(geom.geoms) if geom.geom_type != "LineString" else [geom]
            for part in parts:
                folium.PolyLine(
                    [[y, x] for x, y in part.coords],
                    color="#f4a261", weight=5, opacity=0.8,
                    tooltip=f"cpath FID {int(edge['fid'])}",
                ).add_to(m)

    # Matched geometry — the connected snapped path along the road network
    if mgeom_wkt:
        from shapely import wkt as shapely_wkt
        try:
            mgeom = shapely_wkt.loads(mgeom_wkt)
            if mgeom.geom_type == "LineString":
                mgeom_parts = [mgeom]
            else:
                mgeom_parts = list(mgeom.geoms)
            for part in mgeom_parts:
                folium.PolyLine(
                    [[y, x] for x, y in part.coords],
                    color="#2a9d8f", weight=4, opacity=0.9,
                    dash_array="6",
                    tooltip="STMatch matched geometry (mgeom)",
                ).add_to(m)
        except Exception as e:
            print(f"  Warning: could not render mgeom: {e}")

    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:white;padding:12px;border-radius:8px;
                border:1px solid #ccc;font-size:13px;line-height:1.9">
        <b>STMatch result</b><br>
        <span style="color:{_GPS_COLOUR}">━━</span> Raw GPS ({len(gps_rows)} pts)<br>
        <span style="color:{_EDGE_COLOUR}">━━</span> Matched FIDs ({len(set(fids))} unique)<br>
        <span style="color:#f4a261">━━</span> cpath FIDs ({len(cpath) if cpath else 0} edges)<br>
        <span style="color:#2a9d8f">╌╌</span> mgeom snapped path<br>
        radius={params['radius']}  error={params['error']}<br>
        vmax={params['vmax']}m/s  factor={params['factor']}  k={params['k']}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(out_html)
    print(f"Map saved -> {out_html}")
    print(f"Open with: xdg-open {out_html}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run STMatch (Python API) on a GPS file")
    parser.add_argument("--gps",    required=True)
    parser.add_argument("--shp",    required=True)
    parser.add_argument("--radius", type=float, default=300 / _DEG,
                        help="Search radius in degrees (default: 300m)")
    parser.add_argument("--error",  type=float, default=50 / _DEG,
                        help="GPS error in degrees (default: 50m)")
    parser.add_argument("--vmax",   type=float, default=30.0,
                        help="Max speed m/s (default: 30 ~108 km/h)")
    parser.add_argument("--factor", type=float, default=1.5)
    parser.add_argument("-k",       type=int,   default=8)
    parser.add_argument("--gap_threshold", type=float, default=60.0)
    parser.add_argument("--max_point",     type=int,   default=None,
                        help="Only use GPS points up to this index (for testing sub-segments)")
    parser.add_argument("--min_point",     type=int,   default=None,
                        help="Only use GPS points from this index onwards")
    args = parser.parse_args()

    # ── Import fmm ────────────────────────────────────────────────────────────
    try:
        from fmm import Network, NetworkGraph, STMATCH, STMATCHConfig
    except ImportError:
        print("ERROR: fmm Python bindings not installed")
        print("  Install from: https://fmm-wiki.github.io/docs/installation/")
        return

    gps_path = Path(args.gps)

    # ── Load GPS ──────────────────────────────────────────────────────────────
    print(f"Loading GPS: {gps_path}")
    gps_rows = load_gps(gps_path)
    print(f"  {len(gps_rows)} points  "
          f"{gps_rows[0]['ts'].strftime('%Y-%m-%d %H:%M:%S')} -> "
          f"{gps_rows[-1]['ts'].strftime('%H:%M:%S')}")

    if args.min_point is not None or args.max_point is not None:
        lo = args.min_point or 0
        hi = args.max_point or len(gps_rows)
        gps_rows = gps_rows[lo:hi]
        print(f"  Sliced to points [{lo}:{hi}] → {len(gps_rows)} points")

    gap_indices = find_gaps(gps_rows, args.gap_threshold)
    for idx in gap_indices:
        gap_s = (gps_rows[idx+1]["ts"] - gps_rows[idx]["ts"]).total_seconds()
        print(f"  Gap at point {idx}: {gap_s:.0f}s "
              f"({gps_rows[idx]['ts'].strftime('%H:%M:%S')} -> "
              f"{gps_rows[idx+1]['ts'].strftime('%H:%M:%S')})")

    # ── Load network ──────────────────────────────────────────────────────────
    print(f"\nLoading network: {args.shp}")
    net   = Network(args.shp, "fid", "u", "v")
    graph = NetworkGraph(net)
    print(f"  Nodes {net.get_node_count()}  Edges {net.get_edge_count()}")

    # ── Load shapefile for map display ────────────────────────────────────────
    shp = gpd.read_file(args.shp)
    if "fid" not in shp.columns:
        shp["fid"] = shp.index
    shp["fid"] = shp["fid"].astype(int)

    # ── Build STMatch model ───────────────────────────────────────────────────
    model  = STMATCH(net, graph)
    config = STMATCHConfig(args.k, args.radius, args.error, args.vmax, args.factor)
    print(f"\nSTMATCHConfig: k={args.k} radius={args.radius:.6f} "
          f"error={args.error:.6f} vmax={args.vmax} factor={args.factor}")

    # ── Build WKT trajectory and match ───────────────────────────────────────
    coords = [(r["lon"], r["lat"]) for r in gps_rows]
    wkt    = LineString(coords).wkt
    print(f"\nMatching {len(coords)} points as trajectory …")

    result = model.match_wkt(wkt, config)
    opath  = list(result.opath)
    cpath  = list(result.cpath)
    mgeom  = result.mgeom.export_wkt() if opath else None

    print(f"  opath length: {len(opath)}")
    print(f"  cpath length: {len(cpath)}")
    print(f"  First 10 opath: {opath[:10]}")
    print(f"  Unique opath FIDs: {len(set(opath))}")
    print(f"  cpath FIDs: {cpath[:20]}{'...' if len(cpath)>20 else ''}")
    print(f"  Unique cpath FIDs: {len(set(cpath))}")
    if mgeom:
        print(f"  Matched geom:   {mgeom[:80]}...")

    if not opath:
        print("\nNo matches found. Try:")
        print("  --radius larger  (current: {:.4f} deg = {:.0f}m)".format(
            args.radius, args.radius * _DEG))
        print("  --error  larger  (current: {:.4f} deg = {:.0f}m)".format(
            args.error, args.error * _DEG))
        print("  --vmax   larger  (current: {:.0f} m/s)".format(args.vmax))
        print("  -k       larger  (current: {})".format(args.k))

    # ── Build map ─────────────────────────────────────────────────────────────
    params = dict(radius=f"{args.radius:.4f}", error=f"{args.error:.4f}",
                  vmax=args.vmax, factor=args.factor, k=args.k, mgeom=mgeom)
    Path("../visuals").mkdir(exist_ok=True)
    # Include segment info in filename if sliced
    if args.min_point is not None or args.max_point is not None:
        lo = args.min_point or 0
        hi = args.max_point or len(gps_rows) + lo
        seg_tag = f"_pts{lo}-{hi}"
    else:
        seg_tag = ""
    shp_tag = Path(args.shp).parent.parent.name.replace("_County", "")
    out_html = str(Path("../visuals") / f"{gps_path.stem}_stmatch{seg_tag}_{shp_tag}.html")
    # Use cpath for edge display — it contains all traversed FIDs along the route
    # opath only has one FID per GPS point, cpath has the full connected path
    display_fids = cpath if cpath else opath
    build_map(gps_rows, gap_indices, display_fids, shp, out_html, params, mgeom_wkt=mgeom)


if __name__ == "__main__":
    main()