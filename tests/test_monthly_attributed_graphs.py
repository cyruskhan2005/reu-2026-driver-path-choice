from __future__ import annotations

from pathlib import Path
import tempfile

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point

from roadnet.monthly_attributed_graphs import (
    build_calendar_manifest,
    build_monthly_fid_usage,
    build_monthly_graph,
    build_trip_fid_membership,
)


def _timeline() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "internal_driver_id": "subject",
                "collection_id": "1003 1004",
                "trip_id": "trip-a",
                "county": "Broward County",
                "trip_month": "2022-01",
                "fid_sequence": "1|2|1|3",
            },
            {
                "internal_driver_id": "subject",
                "collection_id": "1003 1004",
                "trip_id": "trip-b",
                "county": "Broward County",
                "trip_month": "2022-01",
                "fid_sequence": "2|3",
            },
            {
                "internal_driver_id": "subject",
                "collection_id": "1003 1004",
                "trip_id": "trip-c",
                "county": "Palm Beach County",
                "trip_month": "2022-03",
                "fid_sequence": "1|4",
            },
        ]
    )


def test_repeated_fid_counts_once_per_trip() -> None:
    timeline = _timeline()
    membership = build_trip_fid_membership(timeline)
    trip_a = membership.loc[membership["trip_id"] == "trip-a"]
    assert trip_a["fid"].tolist() == [1, 2, 3]

    usage = build_monthly_fid_usage(timeline, membership)
    broward = usage.loc[
        (usage["trip_month"] == "2022-01")
        & (usage["county"] == "Broward County")
    ].set_index("fid")
    assert broward.loc[1, "trip_use_count"] == 1
    assert broward.loc[2, "trip_use_count"] == 2
    assert broward.loc[3, "trip_use_count"] == 2
    assert broward.loc[2, "trip_use_share"] == 1.0


def test_same_fid_in_different_counties_remains_separate() -> None:
    timeline = _timeline()
    usage = build_monthly_fid_usage(
        timeline, build_trip_fid_membership(timeline)
    )
    fid_one = usage.loc[usage["fid"] == 1]
    assert set(fid_one["county"]) == {"Broward County", "Palm Beach County"}


def test_calendar_manifest_includes_missing_months() -> None:
    timeline = _timeline()
    usage = build_monthly_fid_usage(
        timeline, build_trip_fid_membership(timeline)
    )
    attributed = gpd.GeoDataFrame(
        usage.assign(
            u=usage["fid"] * 10,
            v=usage["fid"] * 10 + 1,
            geometry=[
                LineString([(index, index), (index + 1, index + 1)])
                for index in range(len(usage))
            ],
        ),
        geometry="geometry",
        crs="EPSG:4326",
    )
    manifest = build_calendar_manifest(timeline, attributed)
    assert manifest["trip_month"].nunique() == 3
    february = manifest.loc[manifest["trip_month"] == "2022-02"]
    assert not february["observed_month"].any()
    assert (february["trip_count"] == 0).all()


def test_directed_parallel_edges_survive_graphml() -> None:
    edges = gpd.GeoDataFrame(
        [
            {
                "fid": 10,
                "u": 1,
                "v": 2,
                "trip_use_count": 2,
                "trip_use_share": 1.0,
                "geometry": LineString([(0, 0), (1, 1)]),
            },
            {
                "fid": 11,
                "u": 1,
                "v": 2,
                "trip_use_count": 1,
                "trip_use_share": 0.5,
                "geometry": LineString([(0, 0), (1, 0.8)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    nodes = gpd.GeoDataFrame(
        [
            {"osmid": 1, "coordinate_source": "test", "geometry": Point(0, 0)},
            {"osmid": 2, "coordinate_source": "test", "geometry": Point(1, 1)},
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    graph = build_monthly_graph(
        edges, nodes, month="2022-01", county="Broward County"
    )
    assert graph.number_of_edges() == 2
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "graph.graphml"
        nx.write_graphml(graph, path)
        loaded = nx.read_graphml(path, force_multigraph=True)
        assert loaded.number_of_edges() == 2
        assert {str(key) for _, _, key in loaded.edges(keys=True)} == {"10", "11"}
