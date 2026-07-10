"""Synthetic tests for endpoint-cluster membership stability."""

from __future__ import annotations

import json
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from roadnet.behavior_cluster_stability import (
    CLUSTER_STABILITY_COLUMNS,
    analyze_cluster_stability,
)


def _endpoints() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 10.0, 11.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "cluster_id": ["A", "A", "A", "B", "B"],
        }
    )


def _selected_membership(x, _y, *, eps_m: float, min_samples: int):
    del eps_m, min_samples
    return [0 if value < 5 else 1 for value in x]


class ClusterStabilityTests(unittest.TestCase):
    def _analyze(self, clusterer, *, radii=(5.0, 10.0, 15.0)):
        return analyze_cluster_stability(
            _endpoints(),
            candidate_radii_m=radii,
            selected_radius_m=10.0,
            min_samples=2,
            important_endpoint_threshold=3,
            clusterer=clusterer,
        ).set_index("cluster_id")

    def test_stable_membership_and_all_cluster_rows(self) -> None:
        result = self._analyze(_selected_membership)

        self.assertEqual(list(result.index), ["A", "B"])
        self.assertEqual(result.loc["A", "stability_status"], "stable")
        self.assertEqual(result.loc["A", "stable_radius_count"], 2)
        self.assertEqual(result.loc["A", "minimum_dominant_retention"], 1.0)
        self.assertEqual(result.loc["A", "minimum_best_jaccard"], 1.0)
        self.assertTrue(bool(result.loc["A", "is_important_cluster"]))
        self.assertFalse(bool(result.loc["B", "is_important_cluster"]))
        self.assertIn(
            "below_important_endpoint_threshold",
            json.loads(result.loc["B", "data_quality_flags"]),
        )

    def test_split_behavior(self) -> None:
        def clusterer(x, _y, *, eps_m: float, min_samples: int):
            del min_samples
            if eps_m < 10:
                return [0 if value < 1.5 else 2 if value < 5 else 1 for value in x]
            return [0 if value < 5 else 1 for value in x]

        result = self._analyze(clusterer)
        self.assertEqual(result.loc["A", "stability_status"], "split")
        self.assertEqual(result.loc["A", "split_radius_count"], 1)
        self.assertEqual(result.loc["A", "maximum_component_count"], 2)
        self.assertAlmostEqual(
            result.loc["A", "minimum_dominant_retention"], 2 / 3
        )

    def test_merge_behavior(self) -> None:
        def clusterer(x, _y, *, eps_m: float, min_samples: int):
            del min_samples
            if eps_m > 10:
                return [0 for _ in x]
            return [0 if value < 5 else 1 for value in x]

        result = self._analyze(clusterer)
        self.assertEqual(result.loc["A", "stability_status"], "merged")
        self.assertEqual(result.loc["B", "stability_status"], "merged")
        self.assertEqual(result.loc["A", "merged_radius_count"], 1)
        self.assertGreater(
            result.loc["A", "maximum_merge_contamination_share"], 0
        )
        evidence = json.loads(result.loc["A", "stability_evidence_json"])
        merged = next(item for item in evidence["comparisons"] if item["radius_m"] == 15.0)
        self.assertEqual(merged["components"][0]["foreign_selected_clusters"], ["B"])

    def test_noise_behavior(self) -> None:
        def clusterer(x, _y, *, eps_m: float, min_samples: int):
            del min_samples
            labels = [0 if value < 5 else 1 for value in x]
            if eps_m < 10:
                labels[0] = -1
            return labels

        result = self._analyze(clusterer)
        self.assertEqual(result.loc["A", "stability_status"], "noise")
        self.assertEqual(result.loc["A", "noise_radius_count"], 1)
        self.assertAlmostEqual(result.loc["A", "maximum_noise_share"], 1 / 3)
        self.assertIn(
            "noise_sensitive", json.loads(result.loc["A", "data_quality_flags"])
        )

    def test_output_is_deterministic_under_input_and_radius_order(self) -> None:
        first = analyze_cluster_stability(
            _endpoints(),
            candidate_radii_m=[15, 5, 10, 5],
            selected_radius_m=10,
            min_samples=2,
            important_endpoint_threshold=3,
            clusterer=_selected_membership,
        )
        second = analyze_cluster_stability(
            _endpoints().sample(frac=1.0, random_state=42),
            candidate_radii_m=[5, 10, 15],
            selected_radius_m=10,
            min_samples=2,
            important_endpoint_threshold=3,
            clusterer=_selected_membership,
        )
        assert_frame_equal(first, second)

    def test_schema_is_explicit_and_stable(self) -> None:
        result = self._analyze(_selected_membership).reset_index()
        self.assertEqual(tuple(result.columns), CLUSTER_STABILITY_COLUMNS)
        evidence = json.loads(result.loc[0, "stability_evidence_json"])
        self.assertEqual(evidence["comparison_radii_m"], [5.0, 15.0])
        self.assertEqual(result.loc[0, "comparison_radius_count"], 2)


if __name__ == "__main__":
    unittest.main()
