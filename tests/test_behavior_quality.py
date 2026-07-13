"""Tests for deterministic trip-quality annotation helpers."""

from __future__ import annotations

import copy
import json
import unittest

import pandas as pd

from roadnet.behavior_quality import (
    merge_quality_flags,
    quality_flag_counts,
    validate_trip_quality,
)


def _trip(trip_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "trip_id": trip_id,
        "origin_cluster_id": "C001",
        "destination_cluster_id": "C002",
        "start_timestamp": f"2024-01-{int(trip_id[1:]) + 1:02d}T12:00:00-05:00",
        "end_timestamp": f"2024-01-{int(trip_id[1:]) + 1:02d}T12:10:00-05:00",
        "start_latitude": 26.0000,
        "start_longitude": -80.0000,
        "end_latitude": 26.0100,
        "end_longitude": -80.0000,
        "trip_duration_seconds": 600.0,
        "matched_fid_sequence": "[1,2,3]",
        "matched_road_name_sequence": '["Sample Road","Lyons Road"]',
        "route_distance_m": 1_250.0,
        "average_speed_mph": 10.0,
        "data_quality_flags": "[]",
    }
    row.update(overrides)
    return row


def _flags(frame: pd.DataFrame, position: int = 0) -> list[str]:
    return json.loads(frame.iloc[position]["data_quality_flags"])


class FlagParsingTests(unittest.TestCase):
    def test_existing_json_and_list_flags_merge_in_deterministic_order(self) -> None:
        merged = merge_quality_flags(
            '["custom_z","cross_county_trip","custom_a"]',
            ["empty_road_sequence", "Custom Z"],
        )
        self.assertEqual(
            merged,
            (
                "empty_road_sequence",
                "cross_county_trip",
                "custom_a",
                "custom_z",
            ),
        )
        self.assertEqual(
            merge_quality_flags(["custom_z", "cross_county_trip"], ["custom_a"]),
            ("cross_county_trip", "custom_a", "custom_z"),
        )


class TripValidationTests(unittest.TestCase):
    def test_nonpositive_duration_is_flagged_without_dropping_rows(self) -> None:
        source = pd.DataFrame(
            [_trip("T1", trip_duration_seconds=0), _trip("T2")],
            index=[11, 7],
        )
        original = source.copy(deep=True)
        result = validate_trip_quality(source)
        self.assertEqual(list(result.index), [11, 7])
        self.assertEqual(len(result), len(source))
        self.assertIn("nonpositive_duration", _flags(result, 0))
        self.assertNotIn("nonpositive_duration", _flags(result, 1))
        pd.testing.assert_frame_equal(source, original)

    def test_missing_and_invalid_endpoints_are_flagged(self) -> None:
        frame = pd.DataFrame(
            [
                _trip("T1", start_latitude=float("nan")),
                _trip("T2", end_latitude=91.0),
                _trip("T3", end_longitude=-181.0),
                _trip("T4"),
            ]
        )
        result = validate_trip_quality(frame)
        for position in range(3):
            self.assertIn("missing_or_invalid_endpoint", _flags(result, position))
        self.assertNotIn("missing_or_invalid_endpoint", _flags(result, 3))

    def test_repeated_timestamps_require_distinct_trip_ids(self) -> None:
        frame = pd.DataFrame(
            [
                _trip(
                    "T1",
                    start_timestamp="2024-01-01T12:00:00-05:00",
                    end_timestamp="2024-01-01T12:10:00-05:00",
                ),
                _trip(
                    "T2",
                    start_timestamp="2024-01-01T17:00:00+00:00",
                    end_timestamp="2024-01-01T12:10:00-05:00",
                ),
                _trip(
                    "T3",
                    start_timestamp="2024-01-03T12:00:00-05:00",
                    end_timestamp="2024-01-03T12:10:00-05:00",
                ),
            ]
        )
        result = validate_trip_quality(frame)
        for position in (0, 1):
            self.assertIn("repeated_start_timestamp", _flags(result, position))
            self.assertIn("repeated_end_timestamp", _flags(result, position))
        self.assertNotIn("repeated_start_timestamp", _flags(result, 2))
        self.assertNotIn("repeated_end_timestamp", _flags(result, 2))

        same_identity = pd.DataFrame(
            [
                _trip("T1", start_timestamp="2024-01-01T12:00:00-05:00"),
                _trip("T1", start_timestamp="2024-01-01T12:00:00-05:00"),
            ]
        )
        same_result = validate_trip_quality(same_identity)
        self.assertNotIn("repeated_start_timestamp", _flags(same_result, 0))
        self.assertNotIn("repeated_start_timestamp", _flags(same_result, 1))

    def test_empty_fid_and_road_sequences_are_flagged(self) -> None:
        frame = pd.DataFrame(
            [
                _trip("T1", matched_fid_sequence="[]"),
                _trip("T2", matched_road_name_sequence=["", None]),
                _trip("T3"),
            ]
        )
        result = validate_trip_quality(frame)
        self.assertIn("empty_fid_sequence", _flags(result, 0))
        self.assertIn("empty_road_sequence", _flags(result, 1))
        self.assertNotIn("empty_fid_sequence", _flags(result, 2))
        self.assertNotIn("empty_road_sequence", _flags(result, 2))

    def test_negative_and_over_limit_speeds_are_flagged(self) -> None:
        frame = pd.DataFrame(
            [
                _trip("T1", average_speed_mph=-0.1),
                _trip("T2", average_speed_mph=100.1),
                _trip("T3", average_speed_mph=100.0),
            ]
        )
        result = validate_trip_quality(frame)
        self.assertIn("implausible_average_speed", _flags(result, 0))
        self.assertIn("implausible_average_speed", _flags(result, 1))
        self.assertNotIn("implausible_average_speed", _flags(result, 2))

    def test_implausible_distance_and_circuity_are_flagged(self) -> None:
        frame = pd.DataFrame(
            [
                _trip("T1", route_distance_m=-1),
                _trip("T2", route_distance_m=250_001),
                _trip("T3", route_distance_m=8_000),
                # Tiny endpoint separations do not create unstable circuity flags.
                _trip(
                    "T4",
                    end_latitude=26.000001,
                    route_distance_m=8_000,
                ),
            ]
        )
        result = validate_trip_quality(frame)
        self.assertIn("implausible_route_distance", _flags(result, 0))
        self.assertIn("implausible_route_distance", _flags(result, 1))
        self.assertIn("implausible_circuity", _flags(result, 2))
        self.assertNotIn("implausible_circuity", _flags(result, 3))

    def test_robust_od_pair_outlier_is_flagged_without_removal(self) -> None:
        distances = [1_000.0, 990.0, 1_010.0, 1_005.0, 5_000.0]
        frame = pd.DataFrame(
            [_trip(f"T{index + 1}", route_distance_m=distance) for index, distance in enumerate(distances)]
        )
        result = validate_trip_quality(frame)
        self.assertEqual(len(result), 5)
        for position in range(4):
            self.assertNotIn("od_pair_distance_outlier", _flags(result, position))
        self.assertIn("od_pair_distance_outlier", _flags(result, 4))

    def test_repeated_runs_and_flag_json_are_deterministic(self) -> None:
        existing = ["custom_z", "cross_county_trip"]
        frame = pd.DataFrame(
            [
                _trip(
                    "T1",
                    trip_duration_seconds=0,
                    matched_road_name_sequence="[]",
                    data_quality_flags=copy.deepcopy(existing),
                )
            ]
        )
        first = validate_trip_quality(frame)
        second = validate_trip_quality(frame)
        self.assertEqual(
            first.iloc[0]["data_quality_flags"],
            second.iloc[0]["data_quality_flags"],
        )
        self.assertEqual(
            _flags(first),
            [
                "nonpositive_duration",
                "empty_road_sequence",
                "cross_county_trip",
                "custom_z",
            ],
        )


class SummaryTests(unittest.TestCase):
    def test_summary_counts_each_flag_once_per_row_in_stable_order(self) -> None:
        frame = pd.DataFrame(
            {
                "data_quality_flags": [
                    '["empty_road_sequence","empty_road_sequence","custom_z"]',
                    ["nonpositive_duration", "custom_z"],
                    "[]",
                ]
            }
        )
        self.assertEqual(
            quality_flag_counts(frame),
            {
                "nonpositive_duration": 1,
                "empty_road_sequence": 1,
                "custom_z": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
