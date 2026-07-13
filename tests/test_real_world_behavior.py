"""Core correctness, cache-budget, and privacy tests for behavior insights."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from roadnet.behavior_report import (
    GeneralizedHomeArea,
    find_html_privacy_violations,
    generate_verification_map,
    inject_real_world_behavior_section,
    render_real_world_behavior_insights,
)
from roadnet.google_places import (
    GoogleAPIError,
    GoogleAPIErrorCategory,
    GoogleMapsClient,
    GoogleRequestBudgetExceeded,
    SuccessfulResponseCache,
)
from roadnet.behavior_activity import (
    build_repeated_trip_chains,
    reconstruct_stays,
    summarize_cluster_stays,
    workplace_plausibility,
)
from roadnet.real_world_behavior import (
    _candidate_record,
    _transition_key_finding,
    assign_location_clusters,
    build_road_class_longitudinal_summary,
    build_recurring_patterns,
    classify_location_roles,
    dbscan_projected,
    generalized_home_point,
    haversine_m,
    select_cluster_radius,
    summarize_location_clusters,
)


class _FakeResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def request(self, method: str, endpoint: str, **kwargs: object) -> _FakeResponse:
        self.calls.append({"method": method, "endpoint": endpoint, **kwargs})
        return _FakeResponse({"status": "OK", "results": []})

    def close(self) -> None:
        pass


class DistanceAndClusteringTests(unittest.TestCase):
    def test_haversine_distance_is_metric_scale(self) -> None:
        distance = haversine_m(26.0, -80.0, 26.001, -80.0)
        self.assertGreater(distance, 110)
        self.assertLess(distance, 112)
        self.assertEqual(haversine_m(26.0, -80.0, 26.0, -80.0), 0.0)

    def test_dbscan_separates_nearby_groups(self) -> None:
        x = [0, 2, 4, 100, 102, 104, 1000]
        y = [0, 1, -1, 0, 1, -1, 1000]
        labels = dbscan_projected(x, y, eps_m=8, min_samples=3)
        self.assertEqual(labels[:3], [0, 0, 0])
        self.assertEqual(labels[3:6], [1, 1, 1])
        self.assertEqual(labels[6], -1)

    def test_radius_selection_evaluates_all_candidates(self) -> None:
        frame = pd.DataFrame(
            {
                "x": [0, 2, 4, 100, 102, 104, 1000, 1002, 1004],
                "y": [0, 1, -1, 0, 1, -1, 0, 1, -1],
                "event_date": ["2024-01-01", "2024-01-02", "2024-01-03"] * 3,
            }
        )
        selection = select_cluster_radius(frame, radii_m=(5, 10, 50, 150), min_samples=3)
        self.assertEqual(set(selection.diagnostics["radius_m"]), {5.0, 10.0, 50.0, 150.0})
        self.assertEqual(int(selection.diagnostics["selected"].sum()), 1)
        assigned = assign_location_clusters(frame.assign(
            trip_id=range(len(frame)),
            endpoint_role="origin",
            latitude=26.0,
            longitude=-80.0,
            event_timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
            county="Broward County",
            month="2024-01",
            event_week="2024-W01",
            hour=12.0,
            is_weekday=True,
        ), selection.radius_m, min_samples=3)
        self.assertFalse(assigned["cluster_id"].isna().any())

    def test_cluster_medoid_is_an_observed_endpoint_nearest_centroid(self) -> None:
        trips = pd.DataFrame(
            [
                {
                    **_trip(
                        f"t{index}",
                        f"s{index}",
                        f"2024-01-0{index}T12:00:00-05:00",
                        f"2024-01-0{index}T12:10:00-05:00",
                        "A",
                        "C002",
                        26.0,
                        -80.0,
                        latitude,
                        -80.1,
                    ),
                    "month": "2024-01",
                    "matched_road_name_sequence": json.dumps(["Test Road"]),
                }
                for index, latitude in enumerate(
                    (26.0000, 26.0001, 26.0100), start=1
                )
            ]
        )
        endpoints = pd.DataFrame(
            {
                "trip_id": ["t1", "t2", "t3"],
                "endpoint_role": ["destination"] * 3,
                "cluster_id": ["C002"] * 3,
                "x": [0.0, 1.0, 100.0],
                "y": [0.0, 0.0, 0.0],
                "latitude": [26.0000, 26.0001, 26.0100],
                "longitude": [-80.1] * 3,
                "county": ["Broward County"] * 3,
                "event_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "event_week": ["2024-W01"] * 3,
                "month": ["2024-01"] * 3,
                "hour": [12.0, 12.0, 12.0],
                "is_weekday": [True, True, True],
            }
        )
        clusters, _ = summarize_location_clusters(
            trips,
            endpoints,
            selected_radius_m=50,
            county_paths={},
        )
        self.assertAlmostEqual(float(clusters.iloc[0]["medoid_lat"]), 26.0001)


class RoadClassSummaryTests(unittest.TestCase):
    def test_road_class_summary_uses_direct_trips_and_normalizes_links(self) -> None:
        rows = []
        sequences = {
            "2022-01": [1, 2],
            "2022-02": [3, 4],
            "2022-03": [5, 6],
        }
        for month, fids in sequences.items():
            for suffix in ("a", "b"):
                rows.append(
                    {
                        "trip_id": f"{month}-{suffix}",
                        "month": month,
                        "direct_route_eligible": True,
                        "origin_cluster_id": "C001",
                        "origin_label": "Generalized home area",
                        "destination_cluster_id": "C002",
                        "destination_label": "Recurring place",
                        "deduplicated_fid_sequence": json.dumps(
                            [{"county": "Broward", "fid": fid} for fid in fids]
                        ),
                    }
                )
        # This out-of-scope trip would dominate the early road mix if the
        # eligibility screen were ignored.
        rows.append(
            {
                "trip_id": "excluded",
                "month": "2022-01",
                "direct_route_eligible": False,
                "origin_cluster_id": "C001",
                "origin_label": "Generalized home area",
                "destination_cluster_id": "C002",
                "destination_label": "Recurring place",
                "deduplicated_fid_sequence": json.dumps(
                    [{"county": "Broward", "fid": 5}]
                ),
            }
        )
        roads = pd.DataFrame(
            [
                {"county": "Broward", "fid": 1, "highway": "motorway_link", "road_length_m": 100},
                {"county": "Broward", "fid": 2, "highway": "primary", "road_length_m": 100},
                {"county": "Broward", "fid": 3, "highway": "secondary", "road_length_m": 100},
                {"county": "Broward", "fid": 4, "highway": "residential", "road_length_m": 100},
                {"county": "Broward", "fid": 5, "highway": "trunk_link", "road_length_m": 100},
                {"county": "Broward", "fid": 6, "highway": "service", "road_length_m": 100},
            ]
        )
        od_summary = pd.DataFrame(
            [
                {
                    "origin_cluster_id": "C001",
                    "origin_label": "Generalized home area",
                    "destination_cluster_id": "C002",
                    "destination_label": "Recurring place",
                    "eligible_direct_trip_count": 6,
                    "eligible_months": 3,
                }
            ]
        )
        summary = build_road_class_longitudinal_summary(
            pd.DataFrame(rows), roads, od_summary
        )
        overall = summary.loc[
            summary["scope"].eq("all_eligible_direct_trips")
        ].set_index("period")
        self.assertEqual(int(overall.loc["early", "eligible_trip_count"]), 2)
        self.assertAlmostEqual(float(overall.loc["early", "motorway_share"]), 0.5)
        self.assertAlmostEqual(float(overall.loc["late", "trunk_share"]), 0.5)
        self.assertAlmostEqual(float(overall.loc["late", "service_share"]), 0.5)
        self.assertFalse((summary["eligible_trip_count"] == 3).any())


class CacheAndBudgetTests(unittest.TestCase):
    def test_cache_key_is_deterministic_and_secret_free(self) -> None:
        left = SuccessfulResponseCache.make_key(
            "places", {"latitude": 26.1, "longitude": -80.1, "radius_m": 50}
        )
        right = SuccessfulResponseCache.make_key(
            "places", {"radius_m": 50, "longitude": -80.1, "latitude": 26.1}
        )
        self.assertEqual(left, right)
        self.assertEqual(len(left), 64)
        self.assertNotIn("26.1", left)

    def test_api_error_sanitizes_key_like_values(self) -> None:
        secret = "AIza" + "A" * 35
        error = GoogleAPIError(
            category=GoogleAPIErrorCategory.AUTHORIZATION,
            api="places",
            message=f"authorization failed for key={secret}",
        )
        self.assertNotIn(secret, str(error))
        self.assertNotIn(secret, json.dumps(error.to_dict()))

    def test_successful_response_is_cached_without_key(self) -> None:
        fake = _FakeSession()
        secret = "unit-test-secret-never-persist"
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"GOOGLE_MAPS_API_KEY": secret}, clear=False
        ):
            client = GoogleMapsClient(
                cache_dir=directory,
                session=fake,  # type: ignore[arg-type]
                request_budget=2,
                min_interval_seconds=0,
                max_retries=0,
            )
            first = client.reverse_geocode(26.1, -80.1)
            second = client.reverse_geocode(26.1, -80.1)
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertTrue(first.retrieved_at_utc)
            self.assertEqual(first.retrieved_at_utc, second.retrieved_at_utc)
            self.assertEqual(client.stats.google_requests, 1)
            self.assertEqual(client.stats.cache_hits, 1)
            cache_text = "".join(
                path.read_text(encoding="utf-8") for path in Path(directory).glob("*.json")
            )
            self.assertNotIn(secret, cache_text)
            self.assertNotIn("maps.googleapis.com", cache_text)

    def test_budget_stops_before_second_request(self) -> None:
        fake = _FakeSession()
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"GOOGLE_MAPS_API_KEY": "unit-test-secret"}, clear=False
        ):
            client = GoogleMapsClient(
                cache_dir=directory,
                session=fake,  # type: ignore[arg-type]
                request_budget=1,
                min_interval_seconds=0,
                max_retries=0,
            )
            client.reverse_geocode(26.1, -80.1)
            with self.assertRaises(GoogleRequestBudgetExceeded):
                client.reverse_geocode(26.2, -80.2)
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(client.stats.google_requests, 1)

    def test_cache_only_mode_refuses_transport_on_cache_miss(self) -> None:
        fake = _FakeSession()
        with tempfile.TemporaryDirectory() as directory:
            client = GoogleMapsClient(
                cache_dir=directory,
                session=fake,  # type: ignore[arg-type]
                request_budget=1,
                min_interval_seconds=0,
                max_retries=0,
                allow_network=False,
            )
            with self.assertRaises(GoogleAPIError) as raised:
                client.reverse_geocode(26.1, -80.1)
            self.assertEqual(raised.exception.category, GoogleAPIErrorCategory.CACHE_MISS)
            self.assertEqual(fake.calls, [])
            self.assertEqual(client.stats.google_requests, 0)


def _trip(
    trip_id: str,
    session_id: str,
    start: str,
    end: str,
    origin: str,
    destination: str,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> dict[str, object]:
    return {
        "trip_id": trip_id,
        "session_id": session_id,
        "start_timestamp": start,
        "end_timestamp": end,
        "origin_cluster_id": origin,
        "destination_cluster_id": destination,
        "origin_label": origin,
        "destination_label": destination,
        "start_latitude": start_lat,
        "start_longitude": start_lon,
        "end_latitude": end_lat,
        "end_longitude": end_lon,
    }


class StayAndChainTests(unittest.TestCase):
    def test_stays_separate_measured_micro_censored_and_chain_breaks(self) -> None:
        trips = pd.DataFrame(
            [
                _trip("t1", "day1", "2024-01-02T12:00:00-05:00", "2024-01-02T12:10:00-05:00", "A", "B", 26.0, -80.0, 26.01, -80.01),
                _trip("t2", "day1", "2024-01-02T12:30:00-05:00", "2024-01-02T12:40:00-05:00", "B", "C", 26.0101, -80.0101, 26.02, -80.02),
                _trip("t3", "day1", "2024-01-02T12:43:00-05:00", "2024-01-02T12:50:00-05:00", "C", "A", 26.0201, -80.0201, 26.0, -80.0),
                _trip("t4", "day2", "2024-01-03T12:00:00-05:00", "2024-01-03T12:10:00-05:00", "A", "D", 26.0001, -80.0001, 26.03, -80.03),
                _trip("t5", "day2", "2024-01-03T12:40:00-05:00", "2024-01-03T12:50:00-05:00", "E", "A", 26.04, -80.04, 26.0, -80.0),
            ]
        )
        stays = reconstruct_stays(trips)
        self.assertEqual(len(stays), len(trips))
        self.assertEqual(
            stays["stay_status"].tolist(),
            [
                "MEASURED_STAY",
                "MICRO_STOP_BOUNDARY",
                "CENSORED_CONTINUITY",
                "CENSORED_CHAIN_BREAK",
                "RIGHT_CENSORED_RECORD_END",
            ],
        )
        self.assertAlmostEqual(float(stays.iloc[0]["dwell_minutes"]), 20.0)
        self.assertTrue(pd.isna(stays.iloc[2]["dwell_minutes"]))
        summary = summarize_cluster_stays(stays).set_index("cluster_id")
        self.assertEqual(int(summary.loc["B", "valid_stay_count"]), 1)
        self.assertEqual(int(summary.loc["C", "micro_stop_boundary_count"]), 1)
        self.assertEqual(int(summary.loc["A", "censored_continuity_count"]), 1)

    def test_short_stops_fail_workplace_gate(self) -> None:
        result = workplace_plausibility(
            {
                "valid_stay_count": 200,
                "median_dwell_minutes": 16,
                "share_over_3_hours": 0.01,
                "weekday_share": 0.82,
                "months_visited": 30,
                "home_connection_count": 100,
            }
        )
        self.assertFalse(result["supported"])
        self.assertFalse(result["gates"]["multi_hour_median"])

    def test_service_day_chains_reconcile_every_trip(self) -> None:
        trips = pd.DataFrame(
            [
                _trip("t1", "day1", "2024-01-02T12:00:00-05:00", "2024-01-02T12:10:00-05:00", "A", "B", 26.0, -80.0, 26.01, -80.01),
                _trip("t2", "day1", "2024-01-02T12:30:00-05:00", "2024-01-02T12:40:00-05:00", "B", "A", 26.0101, -80.0101, 26.0, -80.0),
                _trip("t3", "day2", "2024-01-03T12:00:00-05:00", "2024-01-03T12:10:00-05:00", "A", "B", 26.0, -80.0, 26.01, -80.01),
                _trip("t4", "day2", "2024-01-03T12:30:00-05:00", "2024-01-03T12:40:00-05:00", "B", "A", 26.0101, -80.0101, 26.0, -80.0),
            ]
        )
        repeated, occurrences = build_repeated_trip_chains(trips)
        self.assertEqual(int(occurrences["trip_count"].sum()), len(trips))
        self.assertEqual(len(repeated), 1)
        self.assertEqual(int(repeated.iloc[0]["occurrence_count"]), 2)
        self.assertAlmostEqual(
            float(repeated.iloc[0]["median_intermediate_stop_minutes"]), 20.0
        )
        stop_durations = json.loads(
            repeated.iloc[0]["typical_stop_durations_json"]
        )
        self.assertAlmostEqual(float(stop_durations["B"]), 20.0)

    def test_multi_hour_pattern_passes_workplace_gate(self) -> None:
        result = workplace_plausibility(
            {
                "valid_stay_count": 20,
                "median_dwell_minutes": 420,
                "share_over_3_hours": 0.8,
                "weekday_share": 0.9,
                "months_visited": 12,
                "home_connection_count": 18,
            }
        )
        self.assertTrue(result["supported"])
        self.assertTrue(all(result["gates"].values()))


def _role_cluster(
    cluster_id: str,
    *,
    home: bool = False,
    poi_types: tuple[str, ...] = (),
    primary_type: str = "",
    osm_context: str = "no local OSM context",
) -> dict[str, object]:
    return {
        "cluster_id": cluster_id,
        "privacy_flag": "HOME_SENSITIVE" if home else "NONE",
        "home_score": 0.8 if home else 0.1,
        "generalized_location": "Generalized home area" if home else "Test city",
        "months_visited": 12,
        "censored_overnight_association_count": 20 if home else 0,
        "selected_poi_types": json.dumps(list(poi_types)),
        "selected_poi_category": primary_type,
        "selected_poi_name": "Test School" if "school" in poi_types else "Walmart Money Center",
        "selected_poi_address": "100 Test Road, Test City, FL",
        "selected_poi_distance_m": 35.0,
        "selected_poi_source": "google_places_api_new",
        "selected_poi_google_maps_uri": "https://maps.google.com/?cid=1",
        "selected_poi_latitude": 26.1,
        "selected_poi_longitude": -80.1,
        "alternative_pois_json": "[]",
        "median_dwell_minutes": 20.0,
        "valid_stay_count": 10,
        "share_over_3_hours": 0.0,
        "share_20_to_60_minutes": 0.8,
        "share_1_to_3_hours": 0.0,
        "weekday_share": 0.9,
        "coordinate_spread_m": 25.0,
        "recurring_frequency": "weekly",
        "poi_match_quality": "high",
        "osm_context": osm_context,
        "osm_context_source": "OpenStreetMap local cache",
        "top_endpoint_roads_json": json.dumps({"Test Road": 10}),
        "typical_arrival_hour": 8.0,
        "total_visit_count": 30,
        "reverse_geocoded_address": "100 Test Road, Test City, FL",
        "places_search_attempted": True,
    }


class RoleAndRecurrenceTests(unittest.TestCase):
    def test_school_proximity_alone_fails_but_repeated_chain_supports_candidate(self) -> None:
        clusters = pd.DataFrame(
            [
                _role_cluster("C001", home=True),
                _role_cluster(
                    "C002", poi_types=("school",), primary_type="school"
                ),
            ]
        )
        without_chains = classify_location_roles(clusters)
        unsupported = without_chains.set_index("cluster_id").loc["C002"]
        self.assertEqual(
            unsupported["inferred_role"],
            "school/daycare context without sufficient chain evidence",
        )

        chains = pd.DataFrame(
            [
                {
                    "cluster_sequence_json": json.dumps(["C001", "C002", "C001"]),
                    "occurrence_count": 6,
                    "public_chain": "Home area → Test School → Home area",
                    "typical_stop_durations_json": json.dumps({"C002": 20.0}),
                }
            ]
        )
        supported = classify_location_roles(
            clusters, repeated_chains=chains
        ).set_index("cluster_id").loc["C002"]
        self.assertEqual(supported["inferred_role"], "possible school/daycare stop")
        self.assertIn("6 home-based loop occurrences", supported["trip_chain_patterns"])

    def test_named_complex_suppresses_unsupported_tenant_selection(self) -> None:
        clusters = pd.DataFrame(
            [
                _role_cluster("C001", home=True),
                _role_cluster(
                    "C013",
                    poi_types=("finance",),
                    primary_type="finance",
                    osm_context="Turtle Run Shoppes",
                ),
            ]
        )
        clusters.loc[clusters["cluster_id"].eq("C013"), "poi_match_quality"] = "medium"
        classified = classify_location_roles(clusters).set_index("cluster_id").loc["C013"]
        self.assertTrue(bool(classified["multi_tenant_flag"]))
        self.assertEqual(classified["selected_poi_name"], "Turtle Run Shoppes")
        self.assertEqual(classified["selected_poi_category"], "multi_tenant_complex")
        self.assertEqual(
            classified["specific_candidate_poi_name"], "Walmart Money Center"
        )

    def test_recurring_frequency_counts_arrivals_not_both_endpoints(self) -> None:
        trips = pd.DataFrame(
            [
                {
                    "trip_id": f"t{index}",
                    "month": month,
                    "origin_cluster_id": "C001",
                    "destination_cluster_id": "C002",
                    "start_timestamp": f"{month}-15T11:45:00-05:00",
                    "end_timestamp": f"{month}-15T12:00:00-05:00",
                }
                for index, month in enumerate(
                    ("2024-01", "2024-02", "2024-03", "2024-04"), start=1
                )
            ]
        )
        clusters = pd.DataFrame(
            [
                {
                    "cluster_id": "C002",
                    "privacy_flag": "NONE",
                    "recurring_frequency": "daily",
                    "weekday_share": 1.0,
                    "selected_public_label": "Monthly destination",
                    "selected_poi_name": "Monthly destination",
                    "generalized_location": "Test city",
                    "selected_poi_address": "100 Test Road",
                    "typical_arrival_time": "12:00 PM",
                    "median_dwell_minutes": 30.0,
                    "inferred_role": "shopping/retail area",
                    "role_confidence": "medium",
                    "role_evidence_score": 0.6,
                    "behavioral_evidence": "Four arrivals.",
                    "map_evidence": "Retail context.",
                    "competing_explanation": "Another tenant is possible.",
                    "uncertainty_statement": "Purpose is not confirmed.",
                }
            ]
        )
        result = build_recurring_patterns(trips, clusters)
        self.assertEqual(int(result.iloc[0]["visit_count"]), 4)
        self.assertEqual(result.iloc[0]["visit_frequency"], "monthly")
        self.assertEqual(
            result.iloc[0]["visit_frequency_basis"], "destination_arrivals"
        )

    def test_candidate_record_retains_required_provenance(self) -> None:
        candidate = _candidate_record(
            {
                "displayName": {"text": "Test Place"},
                "primaryType": "store",
                "types": ["store"],
                "formattedAddress": "100 Test Road",
                "location": {"latitude": 26.1, "longitude": -80.1},
                "businessStatus": "OPERATIONAL",
                "googleMapsUri": "https://maps.google.com/?cid=1",
            },
            cluster_lat=26.1001,
            cluster_lon=-80.1001,
            source="google_places_api_new",
            search_radius_m=50,
            retrieved_at_utc="2026-07-10T12:00:00+00:00",
            cache_hit=True,
        )
        self.assertEqual(candidate["source"], "google_places_api_new")
        self.assertEqual(candidate["search_radius_m"], 50)
        self.assertEqual(
            candidate["retrieved_at_utc"], "2026-07-10T12:00:00+00:00"
        )
        self.assertGreater(float(candidate["match_quality_score"]), 0.0)


class PrivacyTests(unittest.TestCase):
    def test_generalized_home_is_shifted_from_private_point(self) -> None:
        private = (26.1234567, -80.7654321)
        public = generalized_home_point(*private)
        self.assertNotEqual(public, private)
        self.assertGreater(haversine_m(*private, *public), 100)

    def test_html_privacy_scan_detects_home_and_key(self) -> None:
        document = "<html>123 Private Lane 26.123456 -80.654321 ?key=hidden</html>"
        issues = find_html_privacy_violations(
            document,
            exact_home_address="123 Private Lane",
            exact_home_coordinates=(26.123456, -80.654321),
            api_key="hidden",
        )
        self.assertIn("exact_home_address", issues)
        self.assertIn("exact_home_coordinates", issues)
        self.assertIn("api_key", issues)
        self.assertIn("api_key_parameter", issues)


class ReportNarrativeTests(unittest.TestCase):
    def test_transition_summary_uses_alternate_family_early_share(self) -> None:
        finding = _transition_key_finding(
            {
                "origin_label": "Home area",
                "destination_label": "Medical offices",
                "later_route_family": "Alternate corridor",
                "baseline_share": 0.60,
                "early_route_share": 0.20,
                "later_share": 2 / 3,
                "trips_before": 15,
                "trips_after": 15,
                "presence_observed_months": 7,
                "maximum_consecutive_observed_months": 2,
            }
        )
        self.assertIn("20% of 15 early trips", finding)
        self.assertNotIn("60% of 15 early trips", finding)

    def test_report_prioritizes_longitudinal_story_and_short_stop_correction(self) -> None:
        insights = {
            "likely_home": {
                "generalized_location": "Generalized neighborhood",
                "evidence": "Repeated nighttime returns and afternoon departures.",
                "confidence": "high",
            },
            "key_findings": [
                {
                    "title": "No workplace identified",
                    "finding": "No place met the multi-hour stay and map-context gates.",
                    "confidence": "high",
                }
            ],
            "likely_routine": {
                "summary": "A repeated home-area → short-stop → home-area loop.",
                "confidence": "medium",
            },
            "activity_role_revisions": [
                {
                    "cluster_id": "C002",
                    "previous_label": "likely workplace",
                    "revised_label": "recurring short commercial stop",
                    "median_dwell_minutes": 16,
                }
            ],
            "important_places": [],
            "longitudinal_route_transitions": [],
            "temporary_route_deviations": [],
            "behavior_timeline": [],
            "route_family_monthly_shares": [],
            "new_or_disappearing_destinations": [],
            "highway_surface_street_summary": {
                "full_period_highway_distance_share": 0.14,
                "full_period_surface_street_distance_share": 0.86,
                "early_window_highway_distance_share": 0.10,
                "late_window_highway_distance_share": 0.17,
                "interpretation": "The monthly series was not monotonic.",
            },
            "limitations": ["Activity purposes are inferred."],
        }
        rendered = render_real_world_behavior_insights(
            insights,
            poi_clusters=pd.DataFrame(),
            recurring_patterns=pd.DataFrame(),
            od_route_changes=pd.DataFrame(),
            map_href=None,
        )
        for heading in (
            "Key findings",
            "Likely home area",
            "Frequently visited named places",
            "Recurring monthly/weekly destinations",
            "Likely routine",
            "Major route-choice changes",
            "New or disappearing destinations",
            "Interactive map",
            "Research limitations",
        ):
            self.assertIn(heading, rendered)
        self.assertIn("frequent does not mean workplace", rendered)
        self.assertIn("median of 16 minutes", rendered)

    def test_recurring_table_uses_current_output_schema_and_formats_dwell(self) -> None:
        rendered = render_real_world_behavior_insights(
            {
                "likely_home": {"generalized_location": "Generalized neighborhood"},
                "limitations": ["Activity purposes are inferred."],
            },
            recurring_patterns=pd.DataFrame(
                [
                    {
                        "named_poi_or_generalized_location": "Named grocery store",
                        "address": "100 Public Road",
                        "first_month": "2022-01",
                        "last_month": "2024-06",
                        "visit_frequency": "weekly",
                        "typical_time": "5:30 PM",
                        "median_dwell_minutes": 16.2,
                        "inferred_activity": "grocery destination",
                        "confidence": "medium",
                        "alternative_interpretation": "A nearby tenant is possible.",
                    }
                ]
            ),
            map_href=None,
        )
        self.assertIn("Named grocery store", rendered)
        self.assertIn("100 Public Road", rendered)
        self.assertIn("16 minutes", rendered)
        self.assertIn("2022-01–2024-06", rendered)

    def test_report_injection_places_behavior_before_technical_snapshot(self) -> None:
        source = """<html><head><style></style></head><body><nav><a href='#executive-summary'>Executive Summary</a><a href='#research-process-overview'>Process</a></nav><main>
        <section id='executive-summary'><h2>RCCI technical snapshot</h2></section>
        <section id='research-process-overview'><h2>Research process</h2></section>
        </main></body></html>"""
        section = "<section id='real-world-driver-behavior-insights'>Behavior</section>"
        rendered = inject_real_world_behavior_section(source, section)
        self.assertLess(rendered.index("Behavior"), rendered.index("RCCI technical snapshot"))
        self.assertNotIn("research-process-overview", rendered)
        self.assertIn("RCCI Technical Snapshot", rendered)
        self.assertNotIn("Executive Summary", rendered)

    def test_privacy_safe_schematic_route_is_not_collapsed_by_public_circle(self) -> None:
        home = GeneralizedHomeArea(
            latitude=26.0,
            longitude=-80.0,
            radius_m=800,
            generalized_location="Generalized neighborhood",
            generalization_method="unit-test generalized point",
        )
        route = pd.DataFrame(
            [
                {
                    "latlon_sequence": [(26.0, -80.0), (26.004, -80.004)],
                    "privacy_safe_geometry": True,
                    "route_label": "Public schematic route",
                }
            ]
        )
        rendered = generate_verification_map(
            pd.DataFrame(),
            generalized_home=home,
            later_preferred_routes=route,
        ).get_root().render()
        self.assertIn("L.polyline", rendered)
        self.assertIn("Public schematic route", rendered)


if __name__ == "__main__":
    unittest.main()
