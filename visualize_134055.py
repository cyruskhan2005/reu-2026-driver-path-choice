import json
import argparse
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot raw GPS points against FMM-matched road segments."
    )
    parser.add_argument("--gps-file", required=True, help="Input GPS JSONL file")
    parser.add_argument(
        "--agg-file",
        required=True,
        help="Aggregated FID JSONL output for the same trip",
    )
    parser.add_argument(
        "--edges-file",
        default="sflorida_outputs/Broward_County/fmm/edges.shp",
        help="FMM edge shapefile",
    )
    parser.add_argument(
        "--out-png",
        default="broward_134055_raw_vs_matched.png",
        help="Output PNG path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    gps_rows = []
    with open(args.gps_file) as f:
        for line in f:
            r = json.loads(line)
            gps_rows.append({
                "lat": r["loc"]["lat"],
                "lon": r["loc"]["lon"],
                "time": r.get("@ts")
            })

    gps = pd.DataFrame(gps_rows)
    gps_gdf = gpd.GeoDataFrame(
        gps,
        geometry=[Point(xy) for xy in zip(gps["lon"], gps["lat"])],
        crs="EPSG:4326"
    )

    fids = []
    with open(args.agg_file) as f:
        for line in f:
            r = json.loads(line)
            if r.get("match_method") == "fmm":
                fids.append(r["fid"])

    edges = gpd.read_file(args.edges_file)
    matched_edges = edges[edges["fid"].isin(fids)]

    ax = matched_edges.plot(linewidth=2, figsize=(10, 8))
    gps_gdf.plot(ax=ax, markersize=8)

    plt.title("Broward Trip 134055: Raw GPS Points vs FMM Matched Road Segments")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"Saved visualization to {args.out_png}")
    print(f"GPS points: {len(gps_gdf)}")
    print(f"Matched road segments: {len(matched_edges)}")


if __name__ == "__main__":
    main()
