"""
sanity_check.py
===============
Validate conflation results for a given road name.
Outputs results to terminal, a log file, and an interactive HTML map.

Usage:
python sanity_check.py   --name "northwest 54th street"   --county "Miami-Dade County"   --fdot_parquet "../sflorida_outputs/fdot/fdot_merged.parquet"
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import geopandas as gpd

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False


class Tee:
    """Write output to both terminal and a log file simultaneously."""
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def make_map(matches: gpd.GeoDataFrame, map_path: Path, fdot_gdf=None, county_gdf=None) -> None:
    """Render matched edges with alternating colours and FID labels."""
    if not HAS_FOLIUM:
        print("  (folium not installed — skipping map. Run: pip install folium)")
        return

    edges_wgs = matches.to_crs("EPSG:4326")

    # Deduplicate bidirectional edges — OSM stores u→v and v→u as separate
    # edges that overlap exactly. Keep one per unique geometry to make
    # alternating colours visible.
    if "u" in edges_wgs.columns and "v" in edges_wgs.columns:
        edges_wgs["_edge_key"] = edges_wgs.apply(
            lambda r: tuple(sorted([str(r["u"]), str(r["v"])])), axis=1
        )
        edges_wgs = edges_wgs.drop_duplicates(subset="_edge_key").drop(columns=["_edge_key"])
    bounds    = edges_wgs.total_bounds
    centre    = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    m         = folium.Map(location=centre, zoom_start=16, tiles="CartoDB positron")

    colours = ["#e74c3c", "#2980b9"]  # red / blue — alternates per FID

    for i, (idx, row) in enumerate(edges_wgs.iterrows()):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        colour     = colours[i % 2]
        speed      = row.get("estimated_speed_limit", "?")
        source     = row.get("speed_source", "?")
        conf       = row.get("speed_limit_confidence_score", None)
        conf_str   = f"{conf:.2f}" if conf is not None and conf == conf else "?"
        custom_spd = row.get("CUSTOM_SPEED", None)
        fdot_descr = row.get("FDOT_DESCR", None)

        tooltip = (
            f"<b>FID: {idx}</b><br>"
            f"<b>{row.get('name', '?')}</b><br>"
            f"highway: {row.get('highway', '?')}<br>"
            f"estimated_speed_limit: {speed} mph<br>"
            f"speed_source: {source}<br>"
            f"confidence: {conf_str}<br>"
            f"osm_maxspeed: {row.get('osm_maxspeed', '—')}<br>"
            f"custom_speed: {custom_spd if custom_spd == custom_spd else '—'}<br>"
            f"FDOT_DESCR: {fdot_descr if fdot_descr and fdot_descr == fdot_descr else '—'}<br>"
            f"landuse: {row.get('landuse', '—')}"
        )

        def add_line(coords_list, fid=idx, col=colour, tip=tooltip):
            if len(coords_list) < 2:
                return
            folium.PolyLine(
                coords_list,
                color=col,
                weight=6,
                opacity=0.9,
                tooltip=folium.Tooltip(tip),
            ).add_to(m)
            mid = coords_list[len(coords_list) // 2]
            folium.Marker(
                location=mid,
                icon=folium.DivIcon(
                    html=(
                        f'<div style="font-size:9px;font-weight:bold;color:{col};'
                        f'background:white;border:1px solid {col};padding:1px 3px;'
                        f'border-radius:3px;white-space:nowrap;">{fid}</div>'
                    ),
                    icon_size=(60, 18),
                    icon_anchor=(30, 9),
                ),
                tooltip=folium.Tooltip(tip),
            ).add_to(m)

        if geom.geom_type == "LineString":
            add_line([(lat, lon) for lon, lat in geom.coords])
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                add_line([(lat, lon) for lon, lat in part.coords])

    # ── FDOT geometry overlay ────────────────────────────────────────────────
    if fdot_gdf is not None and not fdot_gdf.empty:
        fdot_wgs = fdot_gdf.to_crs("EPSG:4326")
        for _, row in fdot_wgs.iterrows():
            geom   = row.geometry
            descr  = str(row.get("DESCR", row.get("FDOT_DESCR", "?")))
            roadway = str(row.get("ROADWAY", "?"))
            tip    = f"<b>FDOT: {descr}</b><br>ROADWAY: {roadway}"
            if geom is None or geom.is_empty:
                continue
            def add_fdot(coords_list, t=tip):
                if len(coords_list) < 2:
                    return
                folium.PolyLine(
                    coords_list,
                    color="#27ae60",
                    weight=4,
                    opacity=0.7,
                    dash_array="8 4",
                    tooltip=folium.Tooltip(t),
                ).add_to(m)
            if geom.geom_type == "LineString":
                add_fdot([(lat, lon) for lon, lat in geom.coords])
            elif geom.geom_type == "MultiLineString":
                for part in geom.geoms:
                    add_fdot([(lat, lon) for lon, lat in part.coords])

    # ── County geometry overlay ─────────────────────────────────────────────
    if county_gdf is not None and not county_gdf.empty:
        county_wgs = county_gdf.to_crs("EPSG:4326")
        for _, row in county_wgs.iterrows():
            geom  = row.geometry
            name_col  = next((c for c in ["SNAME", "NAME", "CUSTOM_NAME"] if c in row.index), None)
            speed_col = next((c for c in ["SPEEDLIMIT", "SPEED_LIM", "SPEED"] if c in row.index), None)
            label     = str(row[name_col])  if name_col  else "?"
            speed     = str(row[speed_col]) if speed_col else "?"
            tip       = (
                f"<b>County: {label}</b><br>"
                f"speed_limit: {speed}"
            )
            if geom is None or geom.is_empty:
                continue
            def add_county(coords_list, t=tip):
                if len(coords_list) < 2:
                    return
                folium.PolyLine(
                    coords_list,
                    color="#e67e22",
                    weight=4,
                    opacity=0.7,
                    dash_array="8 4",
                    tooltip=folium.Tooltip(t),
                ).add_to(m)
            if geom.geom_type == "LineString":
                add_county([(lat, lon) for lon, lat in geom.coords])
            elif geom.geom_type == "MultiLineString":
                for part in geom.geoms:
                    add_county([(lat, lon) for lon, lat in part.coords])

        # Legend
        legend = (
            '<div style="position:fixed;bottom:30px;left:30px;z-index:1000;'
            'background:white;padding:8px 12px;border-radius:6px;'
            'border:1px solid #ccc;font-size:12px;line-height:1.8;">' 
            '<b>Legend</b><br>'
            '<span style="color:#e74c3c;">&#9644;</span> OSM edge (red)<br>'
            '<span style="color:#2980b9;">&#9644;</span> OSM edge (blue)<br>'
            '<span style="color:#27ae60;">&#9644;</span> FDOT geometry<br>'
            '<span style="color:#e67e22;">&#9644;</span> County geometry'
            '</div>'
        )
        m.get_root().html.add_child(folium.Element(legend))

    m.save(str(map_path))
    print(f"  Map saved → {map_path}")


def check_road(
    name: str,
    county: str,
    output_dir: Path = Path("../sflorida_outputs"),
    map_path: Path = None,
    fdot_parquet: Path = None,
    county_geojson: Path = None,
) -> None:
    # ── Load enriched network ─────────────────────────────────────────────────
    slug = county.replace(" ", "_").replace("-", "_")
    parquet_path = output_dir / slug / "enriched_network.parquet"

    if not parquet_path.exists():
        print(f"ERROR: No enriched network found at {parquet_path}")
        print(f"  Run the pipeline first for {county}")
        return

    print(f"Loading {parquet_path} ...")
    net = gpd.read_parquet(parquet_path)

    # ── Search by name (case-insensitive partial match) ───────────────────────
    if "name" not in net.columns:
        print("ERROR: No 'name' column found in network")
        return

    mask = net["name"].str.contains(name, case=False, na=False)
    matches = net[mask].copy()
    matches = matches[~matches["name"].str.contains("|", regex=False, na=False)]

    if matches.empty:
        print(f"\nNo edges found with name containing '{name}'")
        print("\nSample road names in this network:")
        sample = net["name"].dropna().unique()[:20]
        for n in sorted(sample):
            print(f"  {n}")
        return

    print(f"\n{'='*60}")
    print(f"Road: '{name}'  |  County: {county}")
    print(f"{'='*60}")
    print(f"Matched {len(matches):,} edges\n")

    # ── OSM info ──────────────────────────────────────────────────────────────
    print("── OSM ──────────────────────────────────────────────────")
    osm_cols = ["highway", "oneway", "osm_maxspeed", "is_roundabout", "is_connector"]
    for col in osm_cols:
        if col in matches.columns:
            vals = matches[col].dropna().unique()
            print(f"  {col:<20}: {', '.join(str(v) for v in vals[:5])}")

    # ── Speed ─────────────────────────────────────────────────────────────────
    print("\n── Speed ────────────────────────────────────────────────")
    speed_cols = ["estimated_speed_limit", "speed_source", "speed_limit_confidence_score", "osm_maxspeed"]
    for col in speed_cols:
        if col in matches.columns:
            if col == "speed_limit_confidence_score":
                vals = matches[col].dropna()
                if not vals.empty:
                    print(f"  {col:<20}: mean={vals.mean():.2f}  min={vals.min():.2f}  max={vals.max():.2f}")
            elif col == "speed_source":
                counts = matches[col].value_counts()
                print(f"  {col:<20}:")
                for src, cnt in counts.items():
                    print(f"    {src:<25} {cnt:>5} edges")
            else:
                vals = matches[col].dropna().unique()
                print(f"  {col:<20}: {', '.join(str(v) for v in sorted(vals)[:10])}")

    # ── FDOT ─────────────────────────────────────────────────────────────────
    print("\n── FDOT ─────────────────────────────────────────────────")
    fdot_cols = [c for c in matches.columns if c.startswith("FDOT_")]
    if fdot_cols:
        fdot_matched = matches[matches["FDOT_ROADWAY"].notna()] if "FDOT_ROADWAY" in matches.columns else pd.DataFrame()
        print(f"  FDOT matched edges : {len(fdot_matched):,} / {len(matches):,}")
        for col in ["FDOT_SPEED", "FDOT_FUNCTIONAL_CLASS", "FDOT_LANE_COUNT", "FDOT_AADT", "FDOT_DESCR"]:
            if col in matches.columns:
                vals = matches[col].dropna().unique()
                print(f"  {col:<25}: {', '.join(str(v) for v in vals[:5])}")
    else:
        print("  No FDOT columns found")

    # ── Custom county data ────────────────────────────────────────────────────
    print("\n── Custom County Data ───────────────────────────────────")
    custom_cols = [c for c in matches.columns if c.startswith("CUSTOM_")]
    county_cols = [c for c in matches.columns if c.startswith("COUNTY_") or c.startswith("PBC_")]
    all_custom = custom_cols + county_cols

    if all_custom:
        for col in all_custom:
            vals = matches[col].dropna().unique()
            if len(vals):
                print(f"  {col:<25}: {', '.join(str(v) for v in vals[:5])}")
    else:
        print("  No custom county columns found")

    # ── Mapillary signs ───────────────────────────────────────────────────────
    print("\n── Mapillary Signs ──────────────────────────────────────")
    map_cols = [c for c in matches.columns if c.startswith("MAP_")]
    if map_cols:
        for col in map_cols:
            numeric = pd.to_numeric(matches[col], errors="coerce").fillna(0)
            total = numeric.sum()
            if total > 0:
                print(f"  {col:<25}: {int(total)} signs across {int((numeric > 0).sum())} edges")
    else:
        print("  No Mapillary sign columns found")

    # ── OSM control nodes ─────────────────────────────────────────────────────
    print("\n── OSM Control Nodes ────────────────────────────────────")
    osm_ctrl = [c for c in matches.columns if c.startswith("OSM_has_")]
    for col in osm_ctrl:
        numeric = pd.to_numeric(matches[col], errors="coerce").fillna(0)
        total = numeric.sum()
        if total > 0:
            print(f"  {col:<30}: {int(total)}")

    # ── Land use ─────────────────────────────────────────────────────────────
    if "landuse" in matches.columns:
        print("\n── Land Use ─────────────────────────────────────────────")
        counts = matches["landuse"].value_counts()
        for lu, cnt in counts.items():
            print(f"  {lu:<20}: {cnt} edges")

    # ── All rows ──────────────────────────────────────────────────────────────
    print(f"\n── All Edges ({len(matches):,}) ──────────────────────────────────")
    show_cols = [c for c in [
        "name", "highway", "estimated_speed_limit", "speed_source", "speed_limit_confidence_score",
        "FDOT_SPEED", "FDOT_DESCR", "CUSTOM_SPEED", "CUSTOM_NAME",
        "osm_maxspeed", "landuse",
    ] if c in matches.columns]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 30)
    print(matches[show_cols].to_string(index=True))
    print()

    # ── Map ───────────────────────────────────────────────────────────────────
    if map_path:
        print("\n── Map ──────────────────────────────────────────────────")
        # Load FDOT geometry for matched descriptions
        fdot_overlay = None
        if fdot_parquet and fdot_parquet.exists():
            pass  # gpd already imported
            fdot_all = gpd.read_parquet(fdot_parquet)
            # Find FDOT segments whose DESCR appears in our matched edges
            matched_descrs = matches["FDOT_DESCR"].dropna().unique()
            if len(matched_descrs):
                fdot_overlay = fdot_all[fdot_all["DESCR"].isin(matched_descrs)]
            else:
                # Show all FDOT within the bounding box of matched edges
                from shapely.geometry import box
                bounds = matches.total_bounds
                buf    = 500  # metres buffer
                bbox   = box(bounds[0]-buf, bounds[1]-buf, bounds[2]+buf, bounds[3]+buf)
                fdot_overlay = fdot_all[fdot_all.geometry.intersects(bbox)]

        # Load county geometry overlay — filter by matched CUSTOM_NAME values
        county_overlay = None
        if county_geojson and county_geojson.exists():
            try:
                c_gdf = gpd.read_file(county_geojson).to_crs(matches.crs)
                # Try to match by name column
                matched_names = matches["CUSTOM_NAME"].dropna().unique() if "CUSTOM_NAME" in matches.columns else []
                name_col = next((c for c in ["SNAME", "NAME"] if c in c_gdf.columns), None)
                if len(matched_names) and name_col:
                    county_overlay = c_gdf[c_gdf[name_col].isin(matched_names)]
                else:
                    # Fallback: spatial filter within 100m of matched edges
                    from shapely.geometry import box as sbox
                    bounds = matches.total_bounds
                    bbox   = sbox(bounds[0]-100, bounds[1]-100, bounds[2]+100, bounds[3]+100)
                    county_overlay = c_gdf[c_gdf.geometry.intersects(bbox)]
            except Exception as e:
                print(f"  Warning: could not load county GeoJSON: {e}")

        make_map(matches, map_path, fdot_gdf=fdot_overlay, county_gdf=county_overlay)


def main():
    parser = argparse.ArgumentParser(description="Sanity check conflation results for a road name")
    parser.add_argument("--name",       required=True, help="Road name to search (partial, case-insensitive)")
    parser.add_argument("--county",     required=True, help="County name e.g. 'Miami-Dade County'")
    parser.add_argument("--output_dir", default="../sflorida_outputs", help="Pipeline output directory")
    parser.add_argument("--fdot_parquet", default=None, help="Path to FDOT merged parquet for geometry overlay")
    parser.add_argument("--county_geojson", default=None, help="Path to county GeoJSON for geometry overlay")
    args = parser.parse_args()

    # ── Set up log file and map path ──────────────────────────────────────────
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_slug   = args.name.lower().replace(" ", "_")
    county_slug = args.county.lower().replace(" ", "_").replace("-", "_")
    Path("../visuals").mkdir(exist_ok=True)
    log_path = Path("../visuals") / f"sanity_{county_slug}_{name_slug}_{timestamp}.log"
    map_path = Path("../visuals") / f"sanity_{county_slug}_{name_slug}_{timestamp}.html"

    tee = Tee(log_path)
    sys.stdout = tee

    try:
        print(f"Sanity Check Log")
        print(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Road      : {args.name}")
        print(f"County    : {args.county}")
        print(f"Output dir: {args.output_dir}")
        print()

        check_road(
            name           = args.name,
            county         = args.county,
            output_dir     = Path(args.output_dir),
            map_path       = map_path,
            fdot_parquet   = Path(args.fdot_parquet) if args.fdot_parquet else None,
            county_geojson = Path(args.county_geojson) if args.county_geojson else None,
        )
    finally:
        sys.stdout = tee.terminal
        tee.close()
        print(f"\nLog saved → {log_path}")


if __name__ == "__main__":
    main()