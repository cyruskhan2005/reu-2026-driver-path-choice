from __future__ import annotations

from pathlib import Path
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from roadnet.monthly_attributed_graphs import (
    CANONICAL_NODE_ATTRIBUTE_COLUMNS,
    add_canonical_node_attributes,
    build_monthly_transition_edges,
    build_trip_fid_transitions,
    collapse_consecutive_fids,
    validate_fid_transition_graphs,
    write_month_map,
)


def _timeline() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "internal_driver_id": "driver-hash",
                "trip_id": "trip-1",
                "trip_month": "2023-08",
                "county": "Broward County",
                "fid_sequence": "10|10|20|20|30|20|30",
            },
            {
                "internal_driver_id": "driver-hash",
                "trip_id": "trip-2",
                "trip_month": "2023-08",
                "county": "Broward County",
                "fid_sequence": "10|20|30",
            },
        ]
    )


def _nodes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "month": "2023-08",
                "county": "Broward County",
                "fid": 10,
                "trip_use_count": 2,
                "count_rank": 1,
            },
            {
                "month": "2023-08",
                "county": "Broward County",
                "fid": 20,
                "trip_use_count": 2,
                "count_rank": 1,
            },
            {
                "month": "2023-08",
                "county": "Broward County",
                "fid": 30,
                "trip_use_count": 2,
                "count_rank": 1,
            },
        ]
    )


def test_duplicate_fid_collapsing() -> None:
    assert collapse_consecutive_fids("10|10|20|20|30|20") == [
        10,
        20,
        30,
        20,
    ]


def test_edge_creation_from_fid_sequence() -> None:
    transitions = build_trip_fid_transitions(_timeline().iloc[[0]])
    assert list(
        transitions[["source_fid", "target_fid"]].itertuples(
            index=False, name=None
        )
    ) == [(10, 20), (20, 30), (30, 20), (20, 30)]


def test_monthly_edge_aggregation_distinguishes_occurrences_and_trips() -> None:
    edges = build_monthly_transition_edges(_timeline(), _nodes())
    edge = edges.loc[
        (edges["source_fid"] == 20) & (edges["target_fid"] == 30)
    ].iloc[0]
    assert edge["transition_count"] == 3
    assert edge["trip_count_using_transition"] == 2
    assert edge["transition_share_of_month_trips"] == 1.0
    assert edge["source_trip_use_count"] == 2
    assert edge["target_trip_use_count"] == 2


def test_transition_output_columns() -> None:
    edges = build_monthly_transition_edges(_timeline(), _nodes())
    assert edges.columns.tolist() == [
        "driver_id",
        "driver_label",
        "driver_alias",
        "month",
        "county",
        "source_fid",
        "target_fid",
        "transition_count",
        "trip_count_using_transition",
        "source_trip_use_count",
        "target_trip_use_count",
        "transition_share_of_month_trips",
        "source_count_rank",
        "target_count_rank",
    ]


def test_source_target_node_validation() -> None:
    edges = build_monthly_transition_edges(_timeline(), _nodes())
    manifest = pd.DataFrame(
        [
            {
                "trip_month": "2023-08",
                "county": "Broward County",
                "observed_month": True,
            },
            {
                "trip_month": "2023-09",
                "county": "Broward County",
                "observed_month": False,
            },
        ]
    )
    validation = validate_fid_transition_graphs(
        _timeline(), _nodes(), edges, manifest
    )
    assert validation["passed"]
    assert validation["missing_source_references"] == []
    assert validation["missing_target_references"] == []
    assert validation["self_loop_count"] == 0
    assert validation["zero_trip_months"] == ["2023-09"]


def test_missing_node_reference_fails_validation() -> None:
    edges = build_monthly_transition_edges(_timeline(), _nodes())
    manifest = pd.DataFrame(
        [
            {
                "trip_month": "2023-08",
                "county": "Broward County",
                "observed_month": True,
            }
        ]
    )
    validation = validate_fid_transition_graphs(
        _timeline(),
        _nodes().loc[_nodes()["fid"] != 30],
        edges,
        manifest,
    )
    assert not validation["passed"]
    assert validation["missing_target_references"]


def _geographic_nodes() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [
            {
                "driver_id": "driver-hash",
                "driver_label": "Driver 1003",
                "driver_alias": "Driver 1",
                "month": "2023-08",
                "county": "Broward County",
                "fid": 10,
                "trip_use_count": 2,
                "monthly_trip_count": 2,
                "trip_use_share": 1.0,
                "count_rank": 1,
                "length": 123.4,
                "oneway": "yes",
                "CUSTOM_OWNER": "COUNTY",
                "FDOT_ROADWAY": pd.NA,
                "speed_source": "osm",
                "name": "Example Road",
                "highway": "primary",
                "estimated_speed_limit": 40.0,
                "lanes": 2,
                "FDOT_AADT": 12000,
                "geometry": LineString([(-80.2, 26.1), (-80.19, 26.11)]),
            },
            {
                "driver_id": "driver-hash",
                "driver_label": "Driver 1003",
                "driver_alias": "Driver 1",
                "month": "2023-08",
                "county": "Broward County",
                "fid": 20,
                "trip_use_count": 1,
                "monthly_trip_count": 2,
                "trip_use_share": 0.5,
                "count_rank": 2,
                "length": np.nan,
                "oneway": pd.NA,
                "CUSTOM_OWNER": pd.NA,
                "FDOT_ROADWAY": pd.NA,
                "speed_source": pd.NA,
                "name": pd.NA,
                "highway": "residential",
                "estimated_speed_limit": 25.0,
                "lanes": pd.NA,
                "FDOT_AADT": pd.NA,
                "geometry": LineString([(-80.19, 26.11), (-80.18, 26.12)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )


def test_canonical_node_columns_and_observed_speeds() -> None:
    speeds = pd.DataFrame(
        [
            {
                "month": "2023-08",
                "county": "Broward County",
                "fid": 10,
                "observed_avg_speed": 31.5,
                "observed_median_speed": 30.0,
                "observed_speed_observation_count": 12,
            }
        ]
    )
    nodes = add_canonical_node_attributes(_geographic_nodes(), speeds)
    assert set(CANONICAL_NODE_ATTRIBUTE_COLUMNS).issubset(nodes.columns)
    first = nodes.loc[nodes["fid"] == 10].iloc[0]
    assert first["road_length_m"] == 123.4
    assert first["oneway"] == "yes"
    assert first["observed_avg_speed"] == 31.5
    assert first["observed_median_speed"] == 30.0
    assert first["road_owner_or_source"] == "COUNTY"


def test_missing_canonical_attributes_are_safe() -> None:
    nodes = add_canonical_node_attributes(_geographic_nodes(), pd.DataFrame())
    missing = nodes.loc[nodes["fid"] == 20].iloc[0]
    assert pd.isna(missing["road_length_m"])
    assert pd.isna(missing["oneway"])
    assert pd.isna(missing["observed_avg_speed"])
    assert pd.isna(missing["observed_median_speed"])
    assert pd.isna(missing["road_owner_or_source"])
    assert {"fid", "trip_use_count", "trip_use_share"}.issubset(nodes.columns)


def test_map_popup_contains_readable_new_labels() -> None:
    nodes = add_canonical_node_attributes(
        _geographic_nodes(),
        pd.DataFrame(
            [
                {
                    "month": "2023-08",
                    "county": "Broward County",
                    "fid": 10,
                    "observed_avg_speed": 31.5,
                    "observed_median_speed": 30.0,
                    "observed_speed_observation_count": 12,
                }
            ]
        ),
    ).rename(columns={"month": "trip_month"})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "map.html"
        write_month_map(
            nodes,
            path,
            month="2023-08",
            county="Broward County",
        )
        document = path.read_text(encoding="utf-8")
    for label in (
        "Basic",
        "Usage",
        "Road attributes",
        "Observed avg speed",
        "Observed median speed",
        "Road length",
        "One-way",
        "Owner/source",
    ):
        assert label in document
