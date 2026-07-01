from __future__ import annotations

import math

import pandas as pd

from roadnet.route_choice_change_index import (
    assign_confidence_label,
    assign_interpretation_label,
    build_rcci_summary,
    build_sensitivity_table,
    compute_rcci,
    normalize_weights,
)


def _summary_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "driver_id": "driver",
                "month_a": "2024-01",
                "month_b": "2024-02",
                "county": "Broward County",
                "trips_a": 50,
                "trips_b": 50,
                "nodes_a": 10,
                "nodes_b": 10,
                "edges_a": 9,
                "edges_b": 9,
                "shared_nodes": 10,
                "added_nodes": 0,
                "removed_nodes": 0,
                "shared_edges": 9,
                "added_edges": 0,
                "removed_edges": 0,
                "node_jaccard_similarity": 1.0,
                "edge_jaccard_similarity": 1.0,
                "weighted_node_overlap_min": 1.0,
                "weighted_edge_overlap_min": 1.0,
                "data_quality_flag": "ok",
            },
            {
                "driver_id": "driver",
                "month_a": "2024-02",
                "month_b": "2024-03",
                "county": "Broward County",
                "trips_a": 30,
                "trips_b": 30,
                "nodes_a": 10,
                "nodes_b": 10,
                "edges_a": 9,
                "edges_b": 9,
                "shared_nodes": 0,
                "added_nodes": 10,
                "removed_nodes": 10,
                "shared_edges": 0,
                "added_edges": 9,
                "removed_edges": 9,
                "node_jaccard_similarity": 0.0,
                "edge_jaccard_similarity": 0.0,
                "weighted_node_overlap_min": 0.0,
                "weighted_edge_overlap_min": 0.0,
                "data_quality_flag": "ok",
            },
            {
                "driver_id": "driver",
                "month_a": "2024-03",
                "month_b": "2024-04",
                "county": "Broward County",
                "trips_a": 0,
                "trips_b": 0,
                "nodes_a": 0,
                "nodes_b": 0,
                "edges_a": 0,
                "edges_b": 0,
                "shared_nodes": 0,
                "added_nodes": 0,
                "removed_nodes": 0,
                "shared_edges": 0,
                "added_edges": 0,
                "removed_edges": 0,
                "node_jaccard_similarity": math.nan,
                "edge_jaccard_similarity": math.nan,
                "weighted_node_overlap_min": math.nan,
                "weighted_edge_overlap_min": math.nan,
                "data_quality_flag": "both_months_no_trips",
            },
        ]
    )


def test_weight_normalization() -> None:
    assert normalize_weights(0.5, 0.5) == (0.5, 0.5)
    assert normalize_weights(2, 1) == (2 / 3, 1 / 3)


def test_rcci_bounds_and_extremes() -> None:
    assert compute_rcci(0, 0, node_weight=0.5, edge_weight=0.5) == 0
    assert compute_rcci(1, 1, node_weight=0.5, edge_weight=0.5) == 100
    assert compute_rcci(0.2, 0.8, node_weight=0.5, edge_weight=0.5) == 50


def test_custom_weight_score() -> None:
    score = compute_rcci(0.2, 0.8, node_weight=0.25, edge_weight=0.75)
    assert abs(score - 65) < 1e-9


def test_confidence_rules() -> None:
    assert assign_confidence_label(pd.Series({"trips_a": 0, "trips_b": 20}))[0] == "LOW"
    assert assign_confidence_label(pd.Series({"trips_a": 9, "trips_b": 20}))[0] == "LOW"
    assert assign_confidence_label(pd.Series({"trips_a": 15, "trips_b": 20}))[0] == "MEDIUM"
    assert assign_confidence_label(pd.Series({"trips_a": 30, "trips_b": 70}))[0] == "MEDIUM"
    assert assign_confidence_label(pd.Series({"trips_a": 30, "trips_b": 40}))[0] == "HIGH"
    label, reason = assign_confidence_label(
        pd.Series(
            {
                "trips_a": 30,
                "trips_b": 40,
                "data_quality_flag": "missing_files",
            }
        )
    )
    assert label == "LOW"
    assert reason == "missing_comparison_data"


def test_interpretation_rules() -> None:
    assert (
        assign_interpretation_label(
            pd.Series({"trips_a": 0, "trips_b": 0, "rcci_v1": math.nan})
        )
        == "NO COMPARISON"
    )
    assert (
        assign_interpretation_label(
            pd.Series({"trips_a": 0, "trips_b": 12, "rcci_v1": 100})
        )
        == "ZERO-BASELINE CHANGE"
    )
    assert (
        assign_interpretation_label(
            pd.Series(
                {
                    "trips_a": 8,
                    "trips_b": 12,
                    "confidence_label": "LOW",
                    "rcci_v1": 50,
                }
            )
        )
        == "LOW CONFIDENCE - interpret with trip-count context"
    )
    assert (
        assign_interpretation_label(
            pd.Series({"trips_a": 30, "trips_b": 30, "confidence_label": "HIGH", "rcci_v1": 55})
        )
        == "LOW RELATIVE CHANGE"
    )
    assert (
        assign_interpretation_label(
            pd.Series({"trips_a": 30, "trips_b": 30, "confidence_label": "HIGH", "rcci_v1": 65})
        )
        == "MODERATE RELATIVE CHANGE"
    )
    assert (
        assign_interpretation_label(
            pd.Series({"trips_a": 30, "trips_b": 30, "confidence_label": "HIGH", "rcci_v1": 75})
        )
        == "HIGH RELATIVE CHANGE"
    )
    assert (
        assign_interpretation_label(
            pd.Series({"trips_a": 30, "trips_b": 30, "confidence_label": "HIGH", "rcci_v1": 85})
        )
        == "VERY HIGH RELATIVE CHANGE"
    )


def test_build_summary_and_sensitivity_columns() -> None:
    summary = build_rcci_summary(_summary_rows())
    assert summary["rcci_v1"].iloc[0] == 0
    assert summary["rcci_v1"].iloc[1] == 100
    assert pd.isna(summary["rcci_v1"].iloc[2])
    assert summary["confidence_label"].tolist() == ["HIGH", "HIGH", "LOW"]
    assert summary["interpretation_label"].iloc[2] == "NO COMPARISON"
    assert summary["rcci_v1"].dropna().between(0, 100).all()

    sensitivity = build_sensitivity_table(summary)
    for column in [
        "rcci_balanced_weighted",
        "rcci_edge_heavy_weighted",
        "rcci_balanced_jaccard",
        "rcci_geometric_weighted",
    ]:
        assert column in sensitivity.columns
    assert sensitivity["rcci_balanced_weighted"].iloc[0] == 0
    assert sensitivity["rcci_balanced_weighted"].iloc[1] == 100
