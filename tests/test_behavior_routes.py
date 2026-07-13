"""Regression tests for defensible adjacent-month OD route stories.

These tests describe the contract needed when the legacy adjacent-month route
summary is used beside the newer direct-route eligibility analysis.  Gross
detours and ambiguous monthly modes must not become confident route stories.
"""

from __future__ import annotations

import unittest

import pandas as pd

from roadnet.behavior_routes import (
    compare_consecutive_od_months,
    compute_dominant_routes,
)


COUNTY = "Broward County"


def _road_context() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    route_definitions = {
        # Short direct route.
        1: ("Direct Road", 100.0),
        2: ("Direct Road", 100.0),
        3: ("Direct Road", 100.0),
        # Deliberately implausible loop for the same clustered OD.
        10: ("Gross Detour Loop", 5_000.0),
        11: ("Gross Detour Loop", 5_000.0),
        12: ("Gross Detour Loop", 5_000.0),
        # Independent route families used by tie and low-share tests.
        20: ("Alternate A", 120.0),
        21: ("Alternate B", 120.0),
        22: ("Alternate C", 120.0),
        23: ("Alternate D", 120.0),
        24: ("Alternate E", 120.0),
        25: ("Alternate F", 120.0),
        # A route to a genuinely different destination cluster.
        30: ("Different Destination Road", 140.0),
    }
    for fid, (name, length) in route_definitions.items():
        rows.append(
            {
                "county": COUNTY,
                "fid": fid,
                "road_name": name,
                "highway": "primary",
                "length_m": length,
                "speed_limit": 35,
                "toll": False,
            }
        )
    return pd.DataFrame(rows)


def _trip(
    *,
    month: str,
    sequence: str,
    destination: str = "D",
    destination_label: str = "Recurring destination",
    direct_route_eligible: bool = True,
) -> dict[str, object]:
    return {
        "origin_cluster_id": "O",
        "origin_label": "Generalized origin",
        "destination_cluster_id": destination,
        "destination_label": destination_label,
        "month": month,
        "matched_fid_sequence": sequence,
        "origin_county": COUNTY,
        "trip_duration_seconds": 600,
        "toll_road_usage": False,
        "direct_route_eligible": direct_route_eligible,
    }


class AdjacentMonthRouteRegressionTests(unittest.TestCase):
    def assert_suppressed_or_insufficient(
        self,
        comparisons: pd.DataFrame,
        *,
        reason: str,
    ) -> None:
        """Accept suppression or an explicit machine-readable insufficiency."""
        if comparisons.empty:
            return

        boolean_markers = (
            "dominance_sufficient",
            "dominant_route_sufficient",
            "route_story_sufficient",
        )
        for column in boolean_markers:
            if column in comparisons:
                if (~comparisons[column].fillna(False).astype(bool)).all():
                    return

        text_markers = (
            "data_sufficiency",
            "dominance_status",
            "comparison_status",
            "suppression_reason",
        )
        accepted_words = ("insufficient", "tie", "ambiguous", "low_dominance")
        for column in text_markers:
            if column in comparisons:
                values = comparisons[column].fillna("").astype(str).str.casefold()
                if values.map(
                    lambda value: any(word in value for word in accepted_words)
                ).all():
                    return

        if "confidence" in comparisons:
            values = comparisons["confidence"].fillna("").astype(str).str.casefold()
            if values.eq("insufficient").all():
                return

        self.fail(
            f"{reason} must be suppressed or explicitly marked insufficient; "
            f"received columns {sorted(comparisons.columns)}"
        )

    def test_ineligible_gross_detour_cannot_become_dominant(self) -> None:
        trips = pd.DataFrame(
            [
                _trip(month="2024-01", sequence="1|2|3"),
                _trip(month="2024-01", sequence="1|2|3"),
                _trip(
                    month="2024-01",
                    sequence="10|11|12",
                    direct_route_eligible=False,
                ),
                _trip(
                    month="2024-01",
                    sequence="10|11|12",
                    direct_route_eligible=False,
                ),
                _trip(
                    month="2024-01",
                    sequence="10|11|12",
                    direct_route_eligible=False,
                ),
            ]
        )

        profile = compute_dominant_routes(trips, _road_context()).iloc[0]

        self.assertEqual(profile["dominant_route"], "Direct Road")
        self.assertEqual(int(profile["dominant_route_frequency"]), 2)

    def test_tied_monthly_mode_is_suppressed_or_marked_insufficient(self) -> None:
        trips = pd.DataFrame(
            [
                *[
                    _trip(month="2024-01", sequence=sequence)
                    for sequence in ("1|2|3", "1|2|3", "20", "20")
                ],
                *[
                    _trip(month="2024-02", sequence=sequence)
                    for sequence in ("1|2|3", "1|2|3", "20", "20")
                ],
            ]
        )

        comparisons = compare_consecutive_od_months(trips, _road_context())
        self.assert_suppressed_or_insufficient(
            comparisons,
            reason="A 2-to-2 tie has no unique dominant route",
        )

    def test_low_dominant_share_is_suppressed_or_marked_insufficient(self) -> None:
        # Each month has a unique mode, but it represents only 2/6 trips.
        trips = pd.DataFrame(
            [
                *[
                    _trip(month="2024-01", sequence=sequence)
                    for sequence in ("1|2|3", "1|2|3", "20", "21", "22", "23")
                ],
                *[
                    _trip(month="2024-02", sequence=sequence)
                    for sequence in ("20", "20", "1|2|3", "21", "22", "23")
                ],
            ]
        )

        profiles = compute_dominant_routes(trips, _road_context())
        self.assertTrue(profiles["dominant_route_share"].eq(2 / 6).all())
        comparisons = compare_consecutive_od_months(profiles)
        self.assert_suppressed_or_insufficient(
            comparisons,
            reason="A one-third monthly mode is too weak for a dominant-route story",
        )

    def test_same_od_route_change_is_not_a_destination_change(self) -> None:
        mixed = pd.DataFrame(
            [
                _trip(month="2024-01", sequence="1|2|3"),
                _trip(month="2024-02", sequence="20"),
                _trip(
                    month="2024-02",
                    sequence="30",
                    destination="E",
                    destination_label="Different destination",
                ),
            ]
        )

        comparisons = compare_consecutive_od_months(mixed, _road_context())

        self.assertEqual(len(comparisons), 1)
        row = comparisons.iloc[0]
        self.assertEqual(row["destination_cluster_id"], "D")
        self.assertEqual(row["change_type"], "same_od_route_change")
        self.assertIn(
            "same clustered origin and destination",
            row["plain_english_story"],
        )

        destination_only = pd.DataFrame(
            [
                _trip(month="2024-01", sequence="1|2|3"),
                _trip(
                    month="2024-02",
                    sequence="30",
                    destination="E",
                    destination_label="Different destination",
                ),
            ]
        )
        self.assertTrue(
            compare_consecutive_od_months(
                destination_only, _road_context()
            ).empty
        )


if __name__ == "__main__":
    unittest.main()
