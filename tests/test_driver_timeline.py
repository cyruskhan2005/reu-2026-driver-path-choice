from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd

from roadnet.driver_timeline import (
    build_alias_map,
    build_monthly_summary,
    build_population_index,
    normalize_trip_id,
    parse_fid_sequence,
    route_signature,
    select_population,
    write_timeline_visual,
)


def _timeline() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "driver_alias": "Driver 1",
                "internal_driver_id": "internal-hash-a",
                "driver_id": "internal-hash-a",
                "driver_id_source": "session_dir_parent",
                "trip_month": "2022-01",
                "gps_point_count": 10,
                "matched_fid_count": 3,
                "fid_sequence": "1|2|3",
                "route_signature": "a",
                "start_fid": 1,
                "end_fid": 3,
            },
            {
                "driver_alias": "Driver 1",
                "internal_driver_id": "internal-hash-a",
                "driver_id": "internal-hash-a",
                "driver_id_source": "session_dir_parent",
                "trip_month": "2022-03",
                "gps_point_count": 8,
                "matched_fid_count": 3,
                "fid_sequence": "1|2|4",
                "route_signature": "b",
                "start_fid": 1,
                "end_fid": 4,
            },
        ]
    )


def _mapped_population() -> pd.DataFrame:
    rows = []
    specs = [
        ("bbb", ["2022-01", "2022-02"], 3),
        ("aaa", ["2022-01", "2022-02"], 3),
        ("ccc", ["2022-01", "2022-03"], 2),
    ]
    matched_id = 0
    for internal_id, months, trip_count in specs:
        for index in range(trip_count):
            rows.append(
                {
                    "internal_driver_id": internal_id,
                    "driver_id_source": "session_dir_parent",
                    "trip_month": months[index % len(months)],
                    "county": "Broward County",
                    "opath": f"{index + 1},{index + 2}",
                    "matched_trip_id": matched_id,
                }
            )
            matched_id += 1
    return pd.DataFrame(rows)


def test_parse_fid_sequence_collapses_gps_density_repeats() -> None:
    assert parse_fid_sequence("10,10,11,-1,11,10") == [10, 11, 10]
    assert parse_fid_sequence("10,10,11,-1,11,10", collapse_consecutive=False) == [
        10,
        10,
        11,
        11,
        10,
    ]


def test_normalized_trip_and_signature_are_stable() -> None:
    assert (
        normalize_trip_id("Broward County", "210924", "134055")
        == "broward_county_20210924_134055"
    )
    assert route_signature([10, 11, 10]) == route_signature([10, 11, 10])
    assert route_signature([10, 11, 10]) != route_signature([10, 10, 11])


def test_monthly_summary_fills_missing_month_and_flags_observation() -> None:
    summary = build_monthly_summary(_timeline())
    assert summary["trip_month"].tolist() == ["2022-01", "2022-02", "2022-03"]
    assert summary["observed_month"].tolist() == [True, False, True]
    missing = summary.loc[summary["trip_month"] == "2022-02"].iloc[0]
    assert missing["trip_count"] == 0
    assert missing["total_gps_points"] == 0
    assert missing["total_unique_fids"] == 0


def test_monthly_summary_uses_monthly_fid_union() -> None:
    timeline = pd.concat([_timeline().iloc[[0]], _timeline().iloc[[0]]], ignore_index=True)
    timeline.loc[1, "fid_sequence"] = "1|2|4"
    timeline.loc[1, "route_signature"] = "b"
    timeline.loc[1, "end_fid"] = 4
    summary = build_monthly_summary(timeline).iloc[0]
    assert summary["trip_count"] == 2
    assert summary["total_gps_points"] == 20
    assert summary["total_matched_fids"] == 6
    assert summary["total_unique_fids"] == 4
    assert summary["unique_route_signatures"] == 2


def test_alias_assignment_and_population_ranking_are_deterministic() -> None:
    population = build_population_index(_mapped_population())
    # aaa and bbb tie on all metrics; internal ID is the deterministic tiebreaker.
    assert population["internal_driver_id"].tolist() == ["aaa", "bbb", "ccc"]
    assert population["driver_alias"].tolist() == ["Driver 1", "Driver 2", "Driver 3"]
    alias_map = build_alias_map(population)
    assert alias_map.iloc[0].to_dict()["internal_driver_id"] == "aaa"
    assert alias_map.iloc[0].to_dict()["driver_alias"] == "Driver 1"


def test_population_index_selection_marks_top_n() -> None:
    population = build_population_index(_mapped_population())
    selected, updated = select_population(population, top_n=2)
    assert selected["driver_alias"].tolist() == ["Driver 1", "Driver 2"]
    assert updated["selected_for_top_n"].tolist() == [True, True, False]
    assert set(
        [
            "usable_trip_count",
            "observed_month_count",
            "calendar_month_span",
            "total_unique_fids",
        ]
    ).issubset(updated.columns)


def test_html_uses_alias_title_and_preserves_internal_id_in_details() -> None:
    summary = build_monthly_summary(_timeline())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "timeline.html"
        write_timeline_visual(
            summary,
            path,
            driver_alias="Driver 1",
            internal_driver_id="ca351c04cfabaae40cb77059ef799f4a",
            driver_id_source="session_dir_parent",
            source_files=["matched.csv", "gps.csv"],
        )
        document = path.read_text(encoding="utf-8")
    assert "<title>Driver 1 Route Activity Timeline</title>" in document
    assert "<h1>Driver 1 Route Activity Timeline</h1>" in document
    assert "ca351c04cfabaae40cb77059ef799f4a" in document
    assert "Internal pseudonymous ID" in document
    assert "Number of trips" in document
    assert "Distinct matched FIDs" in document
