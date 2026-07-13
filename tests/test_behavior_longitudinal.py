"""Regression tests for longitudinal route-family analysis."""

from __future__ import annotations

import unittest

import pandas as pd

from roadnet.behavior_longitudinal import (
    analyze_longitudinal_routes,
    canonicalize_road_name,
    ordered_lcs_similarity,
    route_similarity,
)


def _road_context() -> pd.DataFrame:
    rows = []
    definitions = {
        1: ("North Sample Rd", "primary"),
        2: ("N Sample Road", "primary"),
        3: ("Lyons Rd", "primary"),
        4: ("Coconut Creek Pkwy", "secondary"),
        5: ("State Road 7", "primary"),
        6: ("Atlantic Blvd", "primary"),
        7: ("Sample Road", "primary"),
    }
    for fid, (name, highway) in definitions.items():
        rows.append(
            {
                "county": "Broward",
                "fid": fid,
                "road_name": name,
                "highway": highway,
                "length_m": 350.0,
            }
        )
    return pd.DataFrame(rows)


def _trip(
    trip_id: str,
    month: str,
    day: int,
    sequence: list[int],
    *,
    distance_m: float = 1_050.0,
) -> dict[str, object]:
    return {
        "trip_id": trip_id,
        "origin_cluster_id": "C001",
        "origin_label": "Home area",
        "destination_cluster_id": "C009",
        "destination_label": "Recurring destination",
        "month": month,
        "start_timestamp": f"{month}-{day:02d}T13:00:00-04:00",
        "matched_fid_sequence": sequence,
        "origin_county": "Broward",
        "start_latitude": 26.0000,
        "start_longitude": -80.0000,
        "end_latitude": 26.0100,
        "end_longitude": -80.0000,
        "route_distance_m": distance_m,
        "trip_duration_seconds": 600.0,
    }


def _monthly_mix(
    months: list[str],
    route_a_counts: list[int],
    *,
    trips_per_month: int = 10,
    route_b: list[int] | None = None,
) -> pd.DataFrame:
    rows = []
    route_b = route_b or [4, 5, 6]
    for month_index, (month, route_a_count) in enumerate(
        zip(months, route_a_counts, strict=True)
    ):
        for number in range(trips_per_month):
            sequence = [1, 2, 3] if number < route_a_count else route_b
            rows.append(
                _trip(
                    f"T{month_index:02d}-{number:02d}",
                    month,
                    number + 1,
                    sequence,
                )
            )
    return pd.DataFrame(rows)


def _variable_monthly_mix(
    months: list[str],
    route_a_counts: list[int],
    monthly_trip_counts: list[int],
    *,
    route_b: list[int] | None = None,
) -> pd.DataFrame:
    """Build an OD history whose sparse months can differ from dense months."""

    rows = []
    route_b = route_b or [4, 5, 6]
    for month_index, (month, route_a_count, trip_count) in enumerate(
        zip(months, route_a_counts, monthly_trip_counts, strict=True)
    ):
        for number in range(trip_count):
            rows.append(
                _trip(
                    f"V{month_index:02d}-{number:02d}",
                    month,
                    number + 1,
                    [1, 2, 3] if number < route_a_count else route_b,
                )
            )
    return pd.DataFrame(rows)


class SimilarityTests(unittest.TestCase):
    def test_canonical_names_remove_direction_and_normalize_aliases(self) -> None:
        self.assertEqual(canonicalize_road_name("N Sample Rd."), "Sample Road")
        self.assertEqual(
            canonicalize_road_name("Lyons Rd|Atlantic Blvd"),
            "Atlantic Boulevard / Lyons Road",
        )

    def test_canonical_labels_preserve_doctor_title_and_remove_raw_pipe(self) -> None:
        label = canonicalize_road_name(
            "Dr. Martin Luther King Blvd|North Powerline Rd"
        )
        self.assertEqual(
            label,
            "Dr. Martin Luther King Boulevard / Powerline Road",
        )
        self.assertNotIn("Drive Martin Luther King", label or "")
        self.assertNotIn("|", label or "")

    def test_similarity_uses_order_and_segment_evidence(self) -> None:
        self.assertAlmostEqual(
            ordered_lcs_similarity(["A", "B", "C"], ["A", "X", "C"]),
            2 / 3,
        )
        same = route_similarity(
            ["A", "B"],
            ["A", "B"],
            {("Broward", 1): 1},
            {("Broward", 1): 1},
            {(('Broward', 1), ('Broward', 2)): 1},
            {(('Broward', 1), ('Broward', 2)): 1},
        )
        different = route_similarity(
            ["A", "B"],
            ["X", "Y"],
            {("Broward", 1): 1},
            {("Broward", 5): 1},
            {(('Broward', 1), ('Broward', 2)): 1},
            {(('Broward', 5), ('Broward', 6)): 1},
        )
        self.assertEqual(same["combined_similarity"], 1.0)
        self.assertEqual(different["combined_similarity"], 0.0)


class LongitudinalEngineTests(unittest.TestCase):
    def test_od_summary_contains_required_research_fields(self) -> None:
        trips = _monthly_mix(
            ["2022-01", "2022-02", "2022-03"], [7, 7, 7]
        )
        summary = analyze_longitudinal_routes(trips, _road_context())[
            "od_summary"
        ]
        required = {
            "first_observed_date",
            "last_observed_date",
            "frequency_trend",
            "dominant_time_of_day",
            "typical_duration_seconds",
            "data_sufficiency",
        }
        missing = required - set(summary.columns)
        self.assertFalse(
            missing,
            f"OD summary is missing presentation/research fields: {sorted(missing)}",
        )
        row = summary.iloc[0]
        self.assertEqual(str(row["first_observed_date"]), "2022-01-01")
        self.assertEqual(str(row["last_observed_date"]), "2022-03-10")
        self.assertTrue(str(row["frequency_trend"]).strip())
        self.assertEqual(str(row["dominant_time_of_day"]), "afternoon")
        self.assertAlmostEqual(float(row["typical_duration_seconds"]), 600.0)
        self.assertTrue(str(row["data_sufficiency"]).strip())

    def test_trivial_fid_variation_stays_in_one_family(self) -> None:
        rows = []
        for month_index, month in enumerate(["2022-01", "2022-02"]):
            for number in range(3):
                sequence = [1, 2, 3] if number != 1 else [1, 7, 2, 3]
                rows.append(
                    _trip(
                        f"T{month_index}-{number}", month, number + 1, sequence
                    )
                )
        result = analyze_longitudinal_routes(
            pd.DataFrame(rows), _road_context(), min_month_trips=1
        )
        families = result["route_families"]
        self.assertEqual(len(families), 1)
        self.assertEqual(int(families.iloc[0]["trip_count"]), 6)

    def test_robust_od_distance_filter_excludes_only_long_outlier(self) -> None:
        rows = [
            _trip(f"T{index}", "2022-01" if index < 3 else "2022-02", index % 3 + 1, [1, 2, 3])
            for index in range(6)
        ]
        rows[-1]["route_distance_m"] = 3_000.0
        result = analyze_longitudinal_routes(
            pd.DataFrame(rows), _road_context(), min_month_trips=1
        )
        summary = result["od_summary"].iloc[0]
        self.assertEqual(int(summary["eligible_direct_trip_count"]), 5)
        self.assertEqual(int(summary["excluded_distance_outlier"]), 1)

    def test_monthly_family_counts_and_shares_reconcile(self) -> None:
        trips = _monthly_mix(
            ["2022-01", "2022-02", "2022-03"], [7, 5, 3]
        )
        result = analyze_longitudinal_routes(trips, _road_context())
        monthly = result["route_family_monthly_shares"]
        observed = monthly.loc[monthly["eligible_od_trip_count"] > 0]
        grouped = observed.groupby("month").agg(
            family_trips=("family_trip_count", "sum"),
            denominator=("eligible_od_trip_count", "first"),
            share=("route_share", "sum"),
        )
        self.assertTrue((grouped["family_trips"] == grouped["denominator"]).all())
        self.assertTrue((grouped["share"].round(10) == 1.0).all())

    def test_sustained_change_requires_persistent_early_late_difference(self) -> None:
        months = [f"2022-{month:02d}" for month in range(1, 9)]
        trips = _monthly_mix(months, [8, 8, 8, 8, 2, 2, 2, 2])
        result = analyze_longitudinal_routes(trips, _road_context())
        transitions = result["longitudinal_route_transitions"]
        self.assertFalse(transitions.empty)
        self.assertTrue((transitions["persistence_observed_months"] >= 3).all())
        self.assertTrue(
            (transitions["route_share_change_percentage_points"].abs() >= 20).all()
        )

    def test_persistence_separates_presence_and_consecutive_month_counts(self) -> None:
        # The later family occurs in three adjacent *observed* months, but those
        # observations are two calendar months apart.  Presence, observed-run,
        # and calendar-run counts therefore have distinct meanings.
        months = [
            "2022-01",
            "2022-02",
            "2022-03",
            "2022-04",
            "2022-06",
            "2022-08",
            "2022-10",
            "2022-12",
        ]
        trips = _monthly_mix(months, [8, 8, 8, 8, 2, 2, 2, 10])
        transitions = analyze_longitudinal_routes(trips, _road_context())[
            "longitudinal_route_transitions"
        ]
        self.assertEqual(len(transitions), 1)
        row = transitions.iloc[0]
        required = {
            "presence_observed_months",
            "maximum_consecutive_observed_months",
            "maximum_consecutive_calendar_months",
        }
        missing = required - set(transitions.columns)
        self.assertFalse(
            missing,
            f"Transition persistence fields are missing: {sorted(missing)}",
        )
        self.assertEqual(int(row["presence_observed_months"]), 3)
        self.assertEqual(int(row["maximum_consecutive_observed_months"]), 3)
        self.assertEqual(int(row["maximum_consecutive_calendar_months"]), 1)

    def test_presentation_family_names_are_bounded_and_concise(self) -> None:
        names = [
            "Coconut Creek Pkwy",
            "Dr. Martin Luther King Blvd",
            "Powerline Rd",
            "Pompano Pkwy",
            "Atlantic Blvd",
            "Northwest 31st Ave",
            "Copans Rd",
            "Lyons Rd",
        ]
        context = pd.DataFrame(
            [
                {
                    "county": "Broward",
                    "fid": fid,
                    "road_name": name,
                    "highway": "primary",
                    "length_m": 350.0,
                }
                for fid, name in enumerate(names, start=1)
            ]
        )
        trips = pd.DataFrame(
            [
                _trip(
                    f"L{month_index}-{number}",
                    month,
                    number,
                    list(range(1, len(names) + 1)),
                    distance_m=2_800.0,
                )
                for month_index, month in enumerate(["2022-01", "2022-02"])
                for number in range(1, 4)
            ]
        )
        families = analyze_longitudinal_routes(
            trips, context, min_month_trips=1
        )["route_families"]
        supported = families.loc[~families["is_other"]]
        self.assertFalse(supported.empty)
        for family_name in supported["family_name"].astype(str):
            self.assertLessEqual(
                len(family_name),
                120,
                f"Presentation route-family name is too long: {family_name}",
            )
            self.assertLessEqual(
                family_name.count("→") + 1,
                3,
                f"Presentation route-family name lists too many roads: {family_name}",
            )
            self.assertNotIn("|", family_name)

    def test_transition_story_calls_threshold_month_first_adequate_evidence(self) -> None:
        months = [f"2022-{month:02d}" for month in range(1, 9)]
        # January contains one sparse alternate-route trip among only four OD
        # trips.  May is the first month with adequate alternate-route evidence.
        trips = _variable_monthly_mix(
            months,
            route_a_counts=[3, 10, 10, 10, 8, 2, 2, 2],
            monthly_trip_counts=[4, 10, 10, 10, 10, 10, 10, 10],
        )
        result = analyze_longitudinal_routes(trips, _road_context())
        transitions = result["longitudinal_route_transitions"]
        self.assertEqual(len(transitions), 1)
        row = transitions.iloc[0]
        self.assertEqual(str(row["first_recorded_appearance"]), "2022-01")
        self.assertEqual(str(row["first_alternate_appearance"]), "2022-01")
        self.assertEqual(str(row["first_adequate_evidence_month"]), "2022-05")
        story = str(row["plain_english_story"]).casefold()
        self.assertIn("sparse", story)
        self.assertIn("2022-01", story)
        self.assertIn("first adequate evidence", story)
        self.assertIn("2022-05", story)
        self.assertNotIn("first appeared in 2022-05", story)

    def test_one_month_variant_is_temporary_and_not_sustained(self) -> None:
        months = [f"2022-{month:02d}" for month in range(1, 6)]
        trips = _monthly_mix(months, [10, 10, 6, 10, 10])
        result = analyze_longitudinal_routes(trips, _road_context())
        temporary = result["temporary_route_deviations"]
        self.assertEqual(len(temporary), 1)
        self.assertEqual(temporary.iloc[0]["episode_start_month"], "2022-03")
        self.assertEqual(int(temporary.iloc[0]["episode_observed_months"]), 1)
        self.assertTrue(result["longitudinal_route_transitions"].empty)

    def test_missing_calendar_month_disables_contiguous_rolling_window(self) -> None:
        trips = _monthly_mix(["2022-01", "2022-03", "2022-04"], [8, 5, 2])
        result = analyze_longitudinal_routes(trips, _road_context())
        monthly = result["route_family_monthly_shares"]
        april = monthly.loc[monthly["month"].eq("2022-04")]
        self.assertTrue(april["rolling_window_has_3_observed_months"].all())
        self.assertFalse(april["rolling_window_calendar_contiguous"].any())
        self.assertFalse(april["rolling_window_sufficient"].any())


if __name__ == "__main__":
    unittest.main()
