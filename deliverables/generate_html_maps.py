from __future__ import annotations

import html
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from folium import plugins

from roadnet.speed import arbitrate_speed


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "sflorida_outputs"
HTML_DIR = ROOT / "deliverables" / "html_maps"

WGS84 = "EPSG:4326"

TRIPS = [
    {
        "county": "Miami-Dade County",
        "slug": "Miami_Dade_County",
        "trip_id": 6,
        "month": "2021-10",
        "gps_file": "Miami-Dade County_gps.csv",
        "matched_file": "Miami-Dade County_matched.csv",
        "html_file": "miami_dade_trip_6_enriched.html",
        "custom_speed_col": "CUSTOM_SPEED_Miami_Dade_County",
    },
    {
        "county": "Broward County",
        "slug": "Broward_County",
        "trip_id": 3967,
        "month": "2021-09",
        "gps_file": "Broward County_gps.csv",
        "matched_file": "Broward County_matched.csv",
        "html_file": "broward_trip_3967_enriched.html",
        "custom_speed_col": None,
    },
    {
        "county": "Palm Beach County",
        "slug": "Palm_Beach_County",
        "trip_id": 120,
        "month": "2022-02",
        "gps_file": "Palm Beach County_gps.csv",
        "matched_file": "Palm Beach County_matched.csv",
        "html_file": "palm_beach_trip_120_enriched.html",
        "custom_speed_col": "CUSTOM_SPEED_Palm_Beach_County",
    },
]

POPUP_BASE_FIELDS = [
    ("County", "county"),
    ("Trip ID", "trip_id"),
    ("FID", "fid"),
    ("Road name", "name"),
    ("Highway / road type", "highway"),
    ("speed", "estimated_speed_limit"),
    ("Lanes", "lanes"),
    ("Oneway", "oneway"),
    ("Length", "length"),
    ("GPS observations matched to FID", "gps_count"),
    ("Observed average speed", "observed_avg_speed"),
    ("Observed median speed", "observed_median_speed"),
    ("Observed - limit", "observed_delta"),
    ("Status", "status"),
]

EXTRA_FIELD_PREFIXES = (
    "fdot_",
    "FDOT_",
    "CUSTOM_",
    "MAP_",
    "OSM_has_",
    "speed_source",
    "speed_limit_confidence_score",
    "speed_limit_is_estimated",
    "connector_",
)


def _safe(value: object, default: str = "unknown") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.1f}"
    return html.escape(str(value))


def _format_speed(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    return f"{float(value):.1f} mph"


def _format_length(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    return f"{float(value):,.1f} m"


def _trip_opath(match_path: Path, trip_id: int) -> list[int]:
    matched = pd.read_csv(match_path, sep=";")
    row = matched.loc[matched["id"] == trip_id]
    if row.empty:
        raise ValueError(f"Trip {trip_id} not found in {match_path}")
    opath = str(row.iloc[0]["opath"])
    return [int(fid) for fid in opath.split(",") if fid and fid != "-1"]


def _load_trip_gps(gps_path: Path, trip_id: int) -> pd.DataFrame:
    gps = pd.read_csv(gps_path, sep=";")
    gps = gps.loc[gps["id"] == trip_id].copy()
    gps = gps.sort_values("point_idx")
    return gps.reset_index(drop=True)


def _observed_speeds(gps: pd.DataFrame, opath: list[int]) -> pd.Series:
    if gps.empty:
        return pd.Series(dtype=float)

    lat1 = gps["lat"].shift()
    lon1 = gps["lon"].shift()
    lat2 = gps["lat"]
    lon2 = gps["lon"]
    dt = gps["timestamp"].diff()

    rad = np.pi / 180.0
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2) ** 2
    )
    meters = 6_371_000.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    mph = (meters / dt.replace(0, np.nan)) * 2.236936
    mph = mph.where((mph >= 0) & (mph <= 120))

    fid_series = pd.Series(opath[: len(gps)], index=gps.index, dtype="Int64")
    return mph.groupby(fid_series).agg(["count", "mean", "median"])


def _refresh_speed_columns(network: gpd.GeoDataFrame, custom_speed_col: str | None) -> gpd.GeoDataFrame:
    updated = arbitrate_speed(
        pd.DataFrame(network),
        has_fdot="FDOT_ROADWAY" in network.columns,
        custom_speed_col=custom_speed_col,
        custom_name_col="CUSTOM_NAME" if "CUSTOM_NAME" in network.columns else None,
    )
    for col in (
        "estimated_speed_limit",
        "speed_source",
        "speed_limit_confidence_score",
        "speed_limit_is_estimated",
        "speed_limit_label",
    ):
        if col in updated.columns:
            network[col] = updated[col].values
    return network


def _status(observed: object, limit: object) -> tuple[str, str]:
    if observed is None or limit is None or pd.isna(observed) or pd.isna(limit):
        return "unknown", "unknown"
    delta = float(observed) - float(limit)
    if delta > 5:
        status = "above speed limit"
    elif delta < -5:
        status = "below speed limit"
    else:
        status = "near speed limit"
    return f"{delta:+.1f} mph", status


def _row_popup(row: pd.Series) -> str:
    speed_label = row.get("speed_limit_label", "Speed limit")
    rows = []
    for label, key in POPUP_BASE_FIELDS:
        display_label = speed_label if label == "speed" else label
        value = row.get(key)
        if key == "estimated_speed_limit":
            rendered = _format_speed(value)
        elif key == "length":
            rendered = _format_length(value)
        elif key in {"observed_avg_speed", "observed_median_speed"}:
            rendered = _format_speed(value)
        else:
            rendered = _safe(value)
        rows.append(
            "<tr>"
            f'<th style="text-align:left;padding:3px 8px 3px 0;vertical-align:top;color:#333;">{display_label}</th>'
            f'<td style="padding:3px 0;vertical-align:top;max-width:280px;word-break:break-word;">{rendered}</td>'
            "</tr>"
        )

    extras = []
    for col, value in row.items():
        if not col.startswith(EXTRA_FIELD_PREFIXES):
            continue
        try:
            if pd.isna(value) or value == "":
                continue
        except (TypeError, ValueError):
            pass
        if (col.startswith("OSM_has_") or col.startswith("MAP_")) and float(value) == 0:
            continue
        extras.append(
            "<tr>"
            f'<th style="text-align:left;padding:3px 8px 3px 0;vertical-align:top;color:#333;">{html.escape(col)}</th>'
            f'<td style="padding:3px 0;vertical-align:top;max-width:280px;word-break:break-word;">{_safe(value)}</td>'
            "</tr>"
        )

    if extras:
        extra_html = (
            '<div style="font-weight:700;margin:8px 0 4px 0;">Enrichment fields</div>'
            f'<table style="border-collapse:collapse;font-size:12px;line-height:1.35;">{"".join(extras)}</table>'
        )
    else:
        extra_html = (
            '<div style="font-weight:700;margin:8px 0 4px 0;">Enrichment fields</div>'
            '<div style="font-size:12px;color:#666;">No extra non-empty FDOT/Mapillary/source fields on this segment.</div>'
        )

    return (
        '<div style="font-family:Arial,sans-serif;min-width:320px;max-width:460px;">'
        f'<h4 style="margin:0 0 8px 0;">{_safe(row["county"])} Trip {row["trip_id"]} - FID {row["fid"]}</h4>'
        f'<table style="border-collapse:collapse;font-size:12px;line-height:1.35;">{"".join(rows)}</table>'
        f"{extra_html}</div>"
    )


def _legend(spec: dict, gps_count: int, unique_fids: int, duration: int) -> str:
    return f"""
    <div style="position: fixed; top: 18px; left: 56px; z-index: 9999; background: white; border: 1px solid #bbb; border-radius: 6px; padding: 12px 14px; font-family: Arial, sans-serif; max-width: 450px; box-shadow: 0 2px 10px rgba(0,0,0,.18);">
      <div style="font-size: 17px; font-weight: 700; margin-bottom: 6px;">{spec["county"]} Trip {spec["trip_id"]}: Enriched Matched Road Segments</div>
      <div style="font-size: 12px; line-height: 1.55; color: #333;">
        <b>County:</b> {spec["county"]}<br>
        <b>Trip ID:</b> {spec["trip_id"]}<br>
        <b>Trip month:</b> {spec["month"]}<br>
        <b>GPS points:</b> {gps_count:,}; aligned to opath: {gps_count:,}<br>
        <b>Unique matched FIDs:</b> {unique_fids:,}<br>
        <b>Clickable segment features:</b> {unique_fids:,}<br>
        <b>Speed field:</b> estimated_speed_limit<br>
        <b>Duration:</b> {duration:,} seconds<br>
        <span style="color:#1f77b4; font-weight:700;">━━</span> Raw GPS path/points<br>
        <span style="color:#d95f02; font-weight:700;">━━</span> Clickable enriched matched road segments<br>
        <span style="color:#138a36; font-weight:700;">●</span> START &nbsp; <span style="color:#c62828; font-weight:700;">●</span> END
      </div>
    </div>
    """


def generate_map(spec: dict) -> None:
    county_dir = OUTPUT_DIR / spec["slug"]
    network = gpd.read_parquet(county_dir / "enriched_network.parquet")
    network = _refresh_speed_columns(network, spec["custom_speed_col"])

    gps = _load_trip_gps(county_dir / spec["gps_file"], spec["trip_id"])
    opath = _trip_opath(county_dir / spec["matched_file"], spec["trip_id"])
    observed = _observed_speeds(gps, opath)

    fid_order = pd.Series(opath).drop_duplicates().tolist()
    edges = network.loc[network.index.intersection(fid_order)].copy()
    edges = edges.loc[fid_order]
    edges = edges[~edges.index.duplicated(keep="first")]
    edges_wgs = edges.to_crs(WGS84)

    gps_count_by_fid = pd.Series(opath).value_counts()
    edges_wgs["fid"] = edges_wgs.index.astype(int)
    edges_wgs["county"] = spec["county"]
    edges_wgs["trip_id"] = spec["trip_id"]
    edges_wgs["gps_count"] = edges_wgs["fid"].map(gps_count_by_fid).fillna(0).astype(int)
    edges_wgs["observed_avg_speed"] = edges_wgs["fid"].map(observed["mean"]) if not observed.empty else np.nan
    edges_wgs["observed_median_speed"] = edges_wgs["fid"].map(observed["median"]) if not observed.empty else np.nan
    status_values = edges_wgs.apply(
        lambda row: _status(row["observed_avg_speed"], row.get("estimated_speed_limit")),
        axis=1,
        result_type="expand",
    )
    edges_wgs["observed_delta"] = status_values[0]
    edges_wgs["status"] = status_values[1]

    center = [gps["lat"].mean(), gps["lon"].mean()]
    fmap = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron", control_scale=True)
    plugins.MousePosition(position="bottomright", separator=", ", prefix="Lat/Lon").add_to(fmap)

    gps_coords = gps[["lat", "lon"]].values.tolist()
    folium.PolyLine(
        gps_coords,
        color="#1f77b4",
        weight=3,
        opacity=0.75,
        tooltip=f"Raw GPS path ({len(gps_coords):,} points)",
        name=f"Raw GPS path ({len(gps_coords):,} points)",
    ).add_to(fmap)

    if gps_coords:
        folium.CircleMarker(gps_coords[0], radius=5, color="#138a36", fill=True, fill_opacity=1, tooltip="START").add_to(fmap)
        folium.CircleMarker(gps_coords[-1], radius=5, color="#c62828", fill=True, fill_opacity=1, tooltip="END").add_to(fmap)

    feature_group = folium.FeatureGroup(
        name=f"Clickable matched road segments ({len(edges_wgs):,} FIDs)",
        show=True,
    )
    for _, row in edges_wgs.iterrows():
        popup = folium.Popup(_row_popup(row), max_width=520)
        tooltip = f"FID {row['fid']} | {row['status']}"
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        geoms = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for part in geoms:
            coords = [[lat, lon] for lon, lat in part.coords]
            folium.PolyLine(
                coords,
                color="#d95f02",
                weight=6,
                opacity=0.8,
                popup=popup,
                tooltip=tooltip,
            ).add_to(feature_group)
    feature_group.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    duration = int(gps["timestamp"].max() - gps["timestamp"].min()) if not gps.empty else 0
    fmap.get_root().html.add_child(
        folium.Element(_legend(spec, len(gps), len(edges_wgs), duration))
    )

    HTML_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HTML_DIR / spec["html_file"]
    fmap.save(out_path)
    print(f"Wrote {out_path}")


def main() -> None:
    for spec in TRIPS:
        generate_map(spec)


if __name__ == "__main__":
    main()
