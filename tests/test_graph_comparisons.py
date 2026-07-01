from __future__ import annotations

import math
from pathlib import Path
import tempfile

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from roadnet.graph_comparisons import (
    ALL_COUNTIES,
    build_consecutive_month_pairs,
    build_graph_comparisons,
    compare_edge_tables,
    compare_node_tables,
    select_county_comparison_pages,
    data_quality_flag,
    jaccard_similarity,
    normalized_l1_change,
    validate_graph_comparisons,
    weighted_overlap_min,
    write_comparison_map,
    write_detailed_comparison_html,
    write_overview_html,
)


DRIVER_ID = "internal-driver-hash"


def _nodes() -> pd.DataFrame:
    rows = []
    for month, county, values in (
        ("2023-01", "Alpha County", [(1, 3), (2, 2)]),
        ("2023-02", "Alpha County", [(2, 4), (3, 1)]),
        ("2023-01", "Beta County", [(1, 1)]),
        ("2023-02", "Beta County", [(1, 2)]),
    ):
        monthly_trips = max(weight for _, weight in values)
        for fid, count in values:
            rows.append(
                {
                    "driver_id": DRIVER_ID,
                    "month": month,
                    "county": county,
                    "fid": fid,
                    "trip_use_count": count,
                    "trip_use_share": count / monthly_trips,
                    "name": f"Road {fid}",
                    "highway": "residential",
                    "estimated_speed_limit": 30,
                    "lanes": 2,
                    "FDOT_AADT": 1000,
                    "observed_avg_speed": 25 + fid,
                    "observed_median_speed": 24 + fid,
                }
            )
    return pd.DataFrame(rows)


def _edges() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "driver_id": DRIVER_ID,
                "month": "2023-01",
                "county": "Alpha County",
                "source_fid": 1,
                "target_fid": 2,
                "transition_count": 3,
                "trip_count_using_transition": 2,
                "transition_share_of_month_trips": 2 / 3,
            },
            {
                "driver_id": DRIVER_ID,
                "month": "2023-02",
                "county": "Alpha County",
                "source_fid": 2,
                "target_fid": 3,
                "transition_count": 4,
                "trip_count_using_transition": 3,
                "transition_share_of_month_trips": 0.75,
            },
            {
                "driver_id": DRIVER_ID,
                "month": "2023-01",
                "county": "Beta County",
                "source_fid": 1,
                "target_fid": 1,
                "transition_count": 1,
                "trip_count_using_transition": 1,
                "transition_share_of_month_trips": 1.0,
            },
            {
                "driver_id": DRIVER_ID,
                "month": "2023-02",
                "county": "Beta County",
                "source_fid": 1,
                "target_fid": 1,
                "transition_count": 2,
                "trip_count_using_transition": 2,
                "transition_share_of_month_trips": 1.0,
            },
        ]
    )


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trip_month": month,
                "county": county,
                "observed_month": trips > 0,
                "trip_count": trips,
                "fid_node_count": 0 if trips == 0 else 1,
                "directed_edge_count": 0 if trips == 0 else 1,
            }
            for month, alpha, beta in (
                ("2023-01", 3, 1),
                ("2023-02", 4, 2),
                ("2023-03", 0, 0),
            )
            for county, trips in (
                ("Alpha County", alpha),
                ("Beta County", beta),
            )
        ]
    )


def test_consecutive_month_pairs_fill_calendar() -> None:
    assert build_consecutive_month_pairs(["2023-01", "2023-03"]) == [
        ("2023-01", "2023-02"),
        ("2023-02", "2023-03"),
    ]


def test_node_set_comparison_and_classification() -> None:
    nodes = _nodes()
    detail, summary = compare_node_tables(
        nodes.loc[
            (nodes["month"] == "2023-01")
            & (nodes["county"] == "Alpha County")
        ],
        nodes.loc[
            (nodes["month"] == "2023-02")
            & (nodes["county"] == "Alpha County")
        ],
        driver_id=DRIVER_ID,
        month_a="2023-01",
        month_b="2023-02",
        county="Alpha County",
    )
    assert summary["nodes_a"] == 2
    assert summary["nodes_b"] == 2
    assert summary["shared_nodes"] == 1
    assert summary["added_nodes"] == 1
    assert summary["removed_nodes"] == 1
    assert summary["node_jaccard_similarity"] == 1 / 3
    assert dict(zip(detail["fid"], detail["status"])) == {
        1: "removed",
        2: "shared",
        3: "added",
    }
    shared = detail.loc[detail["fid"] == 2].iloc[0]
    assert shared["trip_use_count_delta"] == 2
    assert shared["observed_avg_speed_a"] == 27
    assert shared["observed_avg_speed_b"] == 27


def test_edge_set_comparison_and_weight_changes() -> None:
    edges = _edges()
    detail, summary = compare_edge_tables(
        edges.loc[
            (edges["month"] == "2023-01")
            & (edges["county"] == "Alpha County")
        ],
        edges.loc[
            (edges["month"] == "2023-02")
            & (edges["county"] == "Alpha County")
        ],
        driver_id=DRIVER_ID,
        month_a="2023-01",
        month_b="2023-02",
        county="Alpha County",
    )
    assert summary["shared_edges"] == 0
    assert summary["added_edges"] == 1
    assert summary["removed_edges"] == 1
    assert summary["edge_jaccard_similarity"] == 0
    assert set(detail["status"]) == {"added", "removed"}
    assert summary["weighted_edge_overlap_min"] == 0
    assert summary["normalized_edge_weight_change"] == 1


def test_weighted_metrics_and_empty_policy() -> None:
    assert weighted_overlap_min([3, 2, 0], [0, 4, 1]) == 2 / 8
    assert normalized_l1_change([3, 2, 0], [0, 4, 1]) == 6 / 10
    assert math.isnan(weighted_overlap_min([], []))
    assert math.isnan(normalized_l1_change([], []))
    assert math.isnan(jaccard_similarity(0, 0))


def test_county_fid_keying_keeps_same_fid_separate() -> None:
    nodes = _nodes()
    detail, summary = compare_node_tables(
        nodes.loc[nodes["month"] == "2023-01"],
        nodes.loc[nodes["month"] == "2023-02"],
        driver_id=DRIVER_ID,
        month_a="2023-01",
        month_b="2023-02",
        county=ALL_COUNTIES,
        key_columns=("county", "fid"),
    )
    assert summary["shared_nodes"] == 2
    assert len(detail.loc[detail["fid"] == 1]) == 2
    assert set(detail.loc[detail["fid"] == 1, "county"]) == {
        "Alpha County",
        "Beta County",
    }


def test_data_quality_flags_and_zero_month_handling() -> None:
    assert data_quality_flag(
        trips_a=0, trips_b=0, nodes_a=0, nodes_b=0, edges_a=0, edges_b=0
    ) == "both_months_no_trips"
    assert data_quality_flag(
        trips_a=4, trips_b=20, nodes_a=2, nodes_b=2, edges_a=1, edges_b=1
    ) == "low_trip_count_month"
    node_detail, edge_detail, summary = build_graph_comparisons(
        _nodes(),
        _edges(),
        _manifest(),
        driver_id=DRIVER_ID,
    )
    march_pairs = summary.loc[summary["month_b"] == "2023-03"]
    assert set(march_pairs["data_quality_flag"]) == {"month_b_no_trips"}
    assert (march_pairs["nodes_b"] == 0).all()
    assert (march_pairs["edges_b"] == 0).all()
    assert not node_detail.empty
    assert not edge_detail.empty


def test_summary_counts_and_metrics_validate() -> None:
    node_detail, edge_detail, summary = build_graph_comparisons(
        _nodes(),
        _edges(),
        _manifest(),
        driver_id=DRIVER_ID,
    )
    validation = validate_graph_comparisons(
        node_detail, edge_detail, summary, _manifest()
    )
    assert validation["passed"]
    assert validation["calendar_month_pairs"] == 2
    assert validation["county_comparisons"] == 4
    bounded = [
        "node_jaccard_similarity",
        "edge_jaccard_similarity",
        "weighted_node_overlap_min",
        "weighted_edge_overlap_min",
        "normalized_node_weight_change",
        "normalized_edge_weight_change",
    ]
    for column in bounded:
        assert summary[column].dropna().between(0, 1).all()


def test_overview_uses_nonclinical_graph_change_language() -> None:
    _, _, summary = build_graph_comparisons(
        _nodes(),
        _edges(),
        _manifest(),
        driver_id=DRIVER_ID,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "overview.html"
        write_overview_html(summary, path)
        document = path.read_text(encoding="utf-8")
    assert "Driver 1003 Month-to-Month Attributed Graph Comparison" in document
    assert "not the final path-choice change metric" in document
    assert "dementia" not in document.lower()


def test_overview_lists_generated_comparison_graph_maps() -> None:
    _, _, summary = build_graph_comparisons(
        _nodes(),
        _edges(),
        _manifest(),
        driver_id=DRIVER_ID,
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        map_path = root / "alpha_2023-01_to_2023-02_comparison_map.html"
        map_path.write_text("<html></html>", encoding="utf-8")
        overview_path = root / "overview.html"
        write_overview_html(
            summary,
            overview_path,
            map_paths={("Alpha County", "2023-01", "2023-02"): map_path},
        )
        document = overview_path.read_text(encoding="utf-8")
    assert "County-specific comparison pages by month pair" in document
    assert "Open comparison" in document
    assert "2023-01 → 2023-02" in document
    assert "Alpha County" in document


def test_detailed_comparison_embeds_map_srcdoc_for_portability() -> None:
    node_detail, edge_detail, summary = build_graph_comparisons(
        _nodes(),
        _edges(),
        _manifest(),
        driver_id=DRIVER_ID,
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        map_path = root / "comparison_map.html"
        map_path.write_text(
            "<!doctype html><html><body><p>embedded map</p></body></html>",
            encoding="utf-8",
        )
        detail_path = root / "detail.html"
        write_detailed_comparison_html(
            node_detail,
            edge_detail,
            summary,
            detail_path,
            month_a="2023-01",
            month_b="2023-02",
            county="Alpha County",
            map_path=map_path,
        )
        document = detail_path.read_text(encoding="utf-8")
    assert "srcdoc=" in document
    assert "embedded map" in document
    assert "src='comparison_map.html'" not in document
    assert 'src="comparison_map.html"' not in document


def test_county_comparison_page_selection_skips_both_empty_rows() -> None:
    _, _, summary = build_graph_comparisons(
        _nodes(),
        _edges(),
        _manifest(),
        driver_id=DRIVER_ID,
    )
    selected = select_county_comparison_pages(summary)
    assert not selected.empty
    assert ((selected["nodes_a"] > 0) | (selected["nodes_b"] > 0)).all()
    assert not (
        (selected["county"] == "Alpha County")
        & (selected["month_a"] == "2023-02")
        & (selected["month_b"] == "2023-03")
        & (selected["nodes_b"] == 0)
    ).empty


def test_comparison_map_converts_projected_geometries_to_leaflet_coordinates() -> None:
    nodes = gpd.GeoDataFrame(
        [
            {
                "month": "2023-08",
                "county": "Broward County",
                "fid": 10,
                "geometry": LineString([(560000, 2890000), (560050, 2890050)]),
            },
            {
                "month": "2023-09",
                "county": "Broward County",
                "fid": 10,
                "geometry": LineString([(560000, 2890000), (560050, 2890050)]),
            },
            {
                "month": "2023-09",
                "county": "Broward County",
                "fid": 11,
                "geometry": LineString([(560050, 2890050), (560100, 2890100)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:26917",
    )
    node_detail = pd.DataFrame(
        [
            {
                "month_a": "2023-08",
                "month_b": "2023-09",
                "county": "Broward County",
                "fid": 10,
                "status": "shared",
                "trip_use_count_a": 2,
                "trip_use_count_b": 3,
                "trip_use_count_delta": 1,
                "road_name": "Projected Road",
                "road_type": "primary",
            },
            {
                "month_a": "2023-08",
                "month_b": "2023-09",
                "county": "Broward County",
                "fid": 11,
                "status": "added",
                "trip_use_count_a": 0,
                "trip_use_count_b": 1,
                "trip_use_count_delta": 1,
                "road_name": "Projected Road 2",
                "road_type": "secondary",
            },
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "comparison_map.html"
        write_comparison_map(
            nodes,
            node_detail,
            path,
            county="Broward County",
            month_a="2023-08",
            month_b="2023-09",
        )
        document = path.read_text(encoding="utf-8")
    assert "fitBounds" in document
    assert "26." in document
    assert "-80." in document
    assert "560000" not in document


def test_comparison_map_explains_zero_baseline_county() -> None:
    nodes = gpd.GeoDataFrame(
        [
            {
                "month": "2023-02",
                "county": "Alpha County",
                "fid": 11,
                "geometry": LineString([(560050, 2890050), (560100, 2890100)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:26917",
    )
    node_detail = pd.DataFrame(
        [
            {
                "month_a": "2023-01",
                "month_b": "2023-02",
                "county": "Alpha County",
                "fid": 11,
                "status": "added",
                "trip_use_count_a": 0,
                "trip_use_count_b": 1,
                "trip_use_count_delta": 1,
                "road_name": "New Road",
                "road_type": "secondary",
            },
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "zero_baseline.html"
        write_comparison_map(
            nodes,
            node_detail,
            path,
            county="Alpha County",
            month_a="2023-01",
            month_b="2023-02",
            summary_row={
                "trips_a": 0,
                "trips_b": 1,
                "nodes_a": 0,
                "nodes_b": 1,
                "shared_nodes": 0,
                "added_nodes": 1,
                "removed_nodes": 0,
                "data_quality_flag": "month_a_no_trips",
            },
        )
        document = path.read_text(encoding="utf-8")
    assert "No observations exist for this county in 2023-01" in document
    assert "All displayed FIDs are newly observed in 2023-02" in document
