"""Privacy-safe renderers for Driver 1003 behavior-insight deliverables.

This module deliberately contains presentation code only.  It does not read raw
GPS data, infer a home location, call a map API, or decide which POI is a match.
Callers must pass already-sanitized, evidence-backed tables.  In particular, a
home area is supplied separately as a :class:`GeneralizedHomeArea`; precise
home rows are rejected before Folium sees any data.

The public helpers are intentionally dataframe- and mapping-friendly so the
research pipeline can evolve without coupling the renderer to one exact schema.
Common column aliases used by the current Driver 1003 outputs are supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import html
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlparse

import folium
import numpy as np
import pandas as pd


# These markers intentionally match the existing scripts.  Reusing them makes
# the replacement idempotent even if an older behavior script ran previously.
SECTION_BEGIN = "<!-- BEGIN DRIVER 1003 REAL WORLD BEHAVIOR INSIGHTS -->"
SECTION_END = "<!-- END DRIVER 1003 REAL WORLD BEHAVIOR INSIGHTS -->"
NAV_BEGIN = "<!-- BEGIN DRIVER 1003 REAL WORLD BEHAVIOR INSIGHTS NAV -->"
NAV_END = "<!-- END DRIVER 1003 REAL WORLD BEHAVIOR INSIGHTS NAV -->"
STYLE_MARKER = "/* DRIVER_1003_ENRICHED_BEHAVIOR_REPORT_STYLE */"

OLD_RESEARCH_BEGIN = "<!-- BEGIN DRIVER 1003 RESEARCH INSIGHTS -->"
OLD_RESEARCH_END = "<!-- END DRIVER 1003 RESEARCH INSIGHTS -->"
OLD_RESEARCH_NAV_BEGIN = "<!-- BEGIN DRIVER 1003 RESEARCH INSIGHTS NAV -->"
OLD_RESEARCH_NAV_END = "<!-- END DRIVER 1003 RESEARCH INSIGHTS NAV -->"

SECTION_ID = "real-world-driver-behavior-insights"
MIN_GENERALIZED_HOME_RADIUS_M = 500.0
DEFAULT_GENERALIZED_HOME_RADIUS_M = 750.0

_HOME_ROLE_COLUMNS = (
    "privacy_flag",
    "inferred_role",
    "role",
    "location_role",
    "likely_purpose",
    "classification",
)
_PRECISE_HOME_COLUMNS = (
    "latitude",
    "longitude",
    "lat",
    "lon",
    "lng",
    "centroid_lat",
    "centroid_lon",
    "centroid_latitude",
    "centroid_longitude",
    "medoid_lat",
    "medoid_lon",
    "medoid_latitude",
    "medoid_longitude",
    "selected_poi_lat",
    "selected_poi_lon",
    "selected_poi_latitude",
    "selected_poi_longitude",
    "reverse_geocoded_address",
    "selected_poi_address",
    "selected_poi_google_maps_uri",
    "google_maps_uri",
    "formatted_address",
    "exact_address",
    "exact_latitude",
    "exact_longitude",
)
_SECRET_QUERY_NAMES = {
    "key",
    "api_key",
    "apikey",
    "google_maps_api_key",
    "google_api_key",
}


class BehaviorReportError(RuntimeError):
    """Base exception for rendering and report-insertion failures."""


class PrivacyValidationError(BehaviorReportError):
    """Raised when a public artifact could expose protected home/API data."""


@dataclass(frozen=True)
class GeneralizedHomeArea:
    """A caller-certified neighborhood-level home representation.

    ``latitude`` and ``longitude`` must already be rounded, shifted, or be a
    neighborhood centroid.  They must never be copied from the private home
    cluster.  The renderer cannot prove how coordinates were derived, so the
    explicit ``generalization_method`` records the caller's assertion.
    """

    latitude: float
    longitude: float
    radius_m: float = DEFAULT_GENERALIZED_HOME_RADIUS_M
    label: str = "Likely home area (generalized)"
    generalized_location: str = "Generalized residential area"
    confidence: str = ""
    evidence: str = ""
    generalization_method: str = "caller-provided generalized point"

    def __post_init__(self) -> None:
        latitude = _finite_float(self.latitude)
        longitude = _finite_float(self.longitude)
        radius = _finite_float(self.radius_m)
        if latitude is None or not -90 <= latitude <= 90:
            raise PrivacyValidationError("Generalized home latitude is invalid")
        if longitude is None or not -180 <= longitude <= 180:
            raise PrivacyValidationError("Generalized home longitude is invalid")
        if radius is None or radius < MIN_GENERALIZED_HOME_RADIUS_M:
            raise PrivacyValidationError(
                "Generalized home radius must be at least "
                f"{MIN_GENERALIZED_HOME_RADIUS_M:.0f} meters"
            )
        method = str(self.generalization_method or "").strip().casefold()
        if not method or method in {"none", "raw", "exact", "private centroid"}:
            raise PrivacyValidationError(
                "A home-coordinate generalization method must be declared"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GeneralizedHomeArea":
        """Build from explicitly named generalized-coordinate fields.

        Generic ``lat``/``lon`` and centroid/medoid fields are intentionally not
        accepted.  This prevents an accidental pass-through of a private row.
        """

        forbidden = {
            "address",
            "exact_address",
            "reverse_geocoded_address",
            "google_maps_uri",
            "selected_poi_google_maps_uri",
            "exact_latitude",
            "exact_longitude",
            "centroid_lat",
            "centroid_lon",
            "medoid_lat",
            "medoid_lon",
        }
        present = [key for key in forbidden if key in value and not _is_missing(value[key])]
        if present:
            raise PrivacyValidationError(
                "Generalized home input contains precise or address-level fields"
            )
        latitude = _first_mapping_value(
            value,
            "generalized_latitude",
            "generalized_lat",
            "public_latitude",
            "public_lat",
        )
        longitude = _first_mapping_value(
            value,
            "generalized_longitude",
            "generalized_lon",
            "public_longitude",
            "public_lon",
        )
        if _is_missing(latitude) or _is_missing(longitude):
            raise PrivacyValidationError(
                "Generalized home latitude/longitude are required with explicit "
                "generalized_* or public_* field names"
            )
        return cls(
            latitude=float(latitude),
            longitude=float(longitude),
            radius_m=float(
                _first_mapping_value(
                    value,
                    "generalized_radius_m",
                    "radius_m",
                    default=DEFAULT_GENERALIZED_HOME_RADIUS_M,
                )
            ),
            label=_text(_first_mapping_value(value, "label", "role_label"))
            or "Likely home area (generalized)",
            generalized_location=_text(
                _first_mapping_value(
                    value,
                    "generalized_location",
                    "neighborhood_label",
                    default="Generalized residential area",
                )
            )
            or "Generalized residential area",
            confidence=_text(value.get("confidence")),
            evidence=_text(
                _first_mapping_value(value, "evidence", "behavioral_evidence")
            ),
            generalization_method=_text(
                _first_mapping_value(
                    value,
                    "generalization_method",
                    "privacy_method",
                    default="caller-provided generalized point",
                )
            )
            or "caller-provided generalized point",
        )


def _first_mapping_value(
    value: Mapping[str, Any], *keys: str, default: Any = None
) -> Any:
    for key in keys:
        if key in value and not _is_missing(value[key]):
            return value[key]
    return default


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().casefold() in {
            "nan",
            "none",
            "null",
            "nat",
        }
    if isinstance(value, (list, tuple, dict, set, np.ndarray)):
        return len(value) == 0
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _text(value: Any, default: str = "") -> str:
    if _is_missing(value):
        return default
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    return str(value).strip()


def _display(value: Any, default: str = "—") -> str:
    text = _text(value)
    return text if text else default


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _frame(
    value: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, Mapping):
        return pd.DataFrame([value])
    return pd.DataFrame(list(value))


def _row_value(row: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and not _is_missing(row[key]):
            return row[key]
    return default


def _is_home_row(row: Mapping[str, Any]) -> bool:
    for column in _HOME_ROLE_COLUMNS:
        text = _text(row.get(column)).casefold()
        if text and ("home_sensitive" in text or re.search(r"\bhome\b", text)):
            return True
    return False


def _assert_sanitized_home_rows(frame: pd.DataFrame, *, frame_name: str) -> None:
    """Fail closed if a home-tagged public frame still has precise fields."""

    if frame.empty:
        return
    home_mask = frame.apply(lambda row: _is_home_row(row), axis=1)
    if not bool(home_mask.any()):
        return
    home_rows = frame.loc[home_mask]
    populated: list[str] = []
    for column in _PRECISE_HOME_COLUMNS:
        if column in home_rows and home_rows[column].map(lambda value: not _is_missing(value)).any():
            populated.append(column)
    if populated:
        # Do not echo field values; they may themselves be sensitive.
        raise PrivacyValidationError(
            f"{frame_name} contains a home-tagged row with precise coordinate, "
            "address, or map-link data"
        )


def _non_home_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = frame.apply(lambda row: not _is_home_row(row), axis=1)
    return frame.loc[mask].copy()


def _coordinate_from_row(
    row: Mapping[str, Any], *, poi_first: bool = False
) -> tuple[float, float] | None:
    pairs: list[tuple[str, str]] = []
    if poi_first:
        pairs.extend(
            [
                ("selected_poi_latitude", "selected_poi_longitude"),
                ("selected_poi_lat", "selected_poi_lon"),
                ("poi_latitude", "poi_longitude"),
                ("poi_lat", "poi_lon"),
            ]
        )
    pairs.extend(
        [
            ("medoid_latitude", "medoid_longitude"),
            ("medoid_lat", "medoid_lon"),
            ("centroid_latitude", "centroid_longitude"),
            ("centroid_lat", "centroid_lon"),
            ("latitude", "longitude"),
            ("lat", "lon"),
            ("lat", "lng"),
        ]
    )
    for lat_key, lon_key in pairs:
        latitude = _finite_float(row.get(lat_key))
        longitude = _finite_float(row.get(lon_key))
        if (
            latitude is not None
            and longitude is not None
            and -90 <= latitude <= 90
            and -180 <= longitude <= 180
        ):
            return latitude, longitude
    return None


def _safe_maps_uri(value: Any) -> str | None:
    uri = _text(value)
    if not uri:
        return None
    parsed = urlparse(uri)
    host = (parsed.hostname or "").casefold()
    if parsed.scheme != "https" or not (
        host == "google.com"
        or host.endswith(".google.com")
        or host == "goo.gl"
        or host.endswith(".goo.gl")
    ):
        return None
    query_names = {name.casefold() for name, _ in parse_qsl(parsed.query)}
    if query_names & _SECRET_QUERY_NAMES:
        raise PrivacyValidationError("A Google Maps URI contains an API-key parameter")
    return uri


def _popup_line(label: str, value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    return (
        f"<div><strong>{html.escape(label)}:</strong> "
        f"{html.escape(text)}</div>"
    )


def _parse_alternatives(value: Any) -> list[Mapping[str, Any]]:
    if _is_missing(value):
        return []
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    if isinstance(parsed, Mapping):
        parsed = parsed.get("candidates", parsed.get("places", [parsed]))
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
        return []
    return [item for item in parsed if isinstance(item, Mapping)]


def _candidate_name(candidate: Mapping[str, Any]) -> str:
    value = _row_value(candidate, "name", "display_name", "displayName", "poi_name")
    if isinstance(value, Mapping):
        value = _row_value(value, "text", "name")
    return _display(value, "Unnamed candidate")


def _alternatives_html(value: Any) -> str:
    candidates = _parse_alternatives(value)[:5]
    if not candidates:
        return ""
    items: list[str] = []
    for candidate in candidates:
        name = _candidate_name(candidate)
        category = _text(
            _row_value(
                candidate,
                "primary_type",
                "primaryType",
                "category",
                "primary_category",
            )
        )
        distance = _finite_float(
            _row_value(
                candidate,
                "distance_m",
                "distanceM",
                "distance_from_cluster_m",
            )
        )
        details = [category] if category else []
        if distance is not None:
            details.append(f"{distance:.0f} m")
        suffix = f" ({', '.join(details)})" if details else ""
        uri = _safe_maps_uri(
            _row_value(
                candidate,
                "google_maps_uri",
                "googleMapsUri",
                "maps_uri",
                "uri",
            )
        )
        label = html.escape(name)
        if uri:
            label = (
                f"<a href=\"{html.escape(uri, quote=True)}\" "
                f"target=\"_blank\" rel=\"noopener noreferrer\">{label}</a>"
            )
        items.append(f"<li>{label}{html.escape(suffix)}</li>")
    return "<div><strong>Nearby alternatives:</strong><ul>" + "".join(items) + "</ul></div>"


def _array_values(value: Any) -> list[Any]:
    """Normalize CSV JSON, delimiter text, Series, arrays, and sequences."""

    if _is_missing(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        try:
            decoded = json.loads(stripped)
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = None
        if isinstance(decoded, list):
            return decoded
        return [token.strip() for token in re.split(r"[|,;]", stripped) if token.strip()]
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _route_coordinates(row: Mapping[str, Any]) -> list[tuple[float, float]]:
    """Return Folium ``(lat, lon)`` route coordinates from common schemas."""

    latitudes = _row_value(row, "latitudes", "route_latitudes")
    longitudes = _row_value(row, "longitudes", "route_longitudes")
    latitude_values = _array_values(latitudes)
    longitude_values = _array_values(longitudes)
    if latitude_values and longitude_values:
        points = []
        for latitude, longitude in zip(
            latitude_values, longitude_values, strict=False
        ):
            lat = _finite_float(latitude)
            lon = _finite_float(longitude)
            if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
                points.append((lat, lon))
        if len(points) >= 2:
            return points

    geometry = _row_value(
        row,
        "geometry",
        "route_geometry",
        "geometry_json",
        "geojson",
        "route_coordinates",
        "route_coordinates_json",
        "coordinates_json",
        "path_coordinates",
        "polyline_coordinates",
        "latlon_sequence",
        "coordinates",
        "geometry_wkt",
    )
    if _is_missing(geometry):
        start = _coordinate_from_keys(row, "start")
        end = _coordinate_from_keys(row, "end")
        return [start, end] if start and end else []

    if hasattr(geometry, "geom_type") and getattr(geometry, "geom_type", "") == "MultiLineString":
        lines = list(getattr(geometry, "geoms", []))
        longest = max(lines, key=lambda line: getattr(line, "length", 0.0), default=None)
        if longest is not None:
            return _lonlat_to_latlon(list(longest.coords))
    if hasattr(geometry, "geom_type") and hasattr(geometry, "coords"):
        try:
            coordinates = list(geometry.coords)
        except (NotImplementedError, TypeError):
            coordinates = []
        return _lonlat_to_latlon(coordinates)

    value = geometry
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.upper().startswith("LINESTRING"):
            match = re.search(r"\((.*)\)", stripped, flags=re.DOTALL)
            if not match:
                return []
            coordinates = []
            for pair in match.group(1).split(","):
                numbers = pair.strip().split()
                if len(numbers) >= 2:
                    coordinates.append(numbers[:2])
            return _lonlat_to_latlon(coordinates)
        try:
            value = json.loads(stripped)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    if isinstance(value, Mapping):
        if value.get("type") == "Feature":
            value = value.get("geometry", {})
        if value.get("type") == "LineString":
            return _lonlat_to_latlon(value.get("coordinates", []))
        if value.get("type") == "MultiLineString":
            lines = value.get("coordinates", [])
            longest = max(lines, key=len, default=[])
            return _lonlat_to_latlon(longest)
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, pd.Series):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        coordinate_pairs = [
            pair
            for pair in value
            if isinstance(pair, (Sequence, np.ndarray, pd.Series))
            and not isinstance(pair, (str, bytes))
            and len(pair) >= 2
        ]
        if not coordinate_pairs:
            return []
        # A field explicitly named latlon_sequence is unambiguous.  Generic
        # coordinate arrays follow GeoJSON's lon/lat convention unless only the
        # alternate interpretation is geographically valid.
        coordinate_order = _text(row.get("coordinate_order")).casefold().replace("/", "")
        if coordinate_order in {"latlon", "latlng", "yx"}:
            return _validate_latlon(coordinate_pairs)
        if "latlon_sequence" in row and not _is_missing(row.get("latlon_sequence")):
            return _validate_latlon(coordinate_pairs)
        first_a = _finite_float(coordinate_pairs[0][0])
        first_b = _finite_float(coordinate_pairs[0][1])
        if first_a is not None and first_b is not None and abs(first_a) <= 90 < abs(first_b):
            return _validate_latlon(coordinate_pairs)
        # This resolves the common western-hemisphere ambiguity where both
        # South Florida axes have magnitudes below 90.  Explicit
        # ``coordinate_order`` remains authoritative for other regions.
        if first_a is not None and first_b is not None and first_a >= 0 > first_b:
            return _validate_latlon(coordinate_pairs)
        return _lonlat_to_latlon(coordinate_pairs)
    return []


def _coordinate_from_keys(
    row: Mapping[str, Any], prefix: str
) -> tuple[float, float] | None:
    latitude = _finite_float(
        _row_value(row, f"{prefix}_latitude", f"{prefix}_lat")
    )
    longitude = _finite_float(
        _row_value(row, f"{prefix}_longitude", f"{prefix}_lon", f"{prefix}_lng")
    )
    if (
        latitude is None
        or longitude is None
        or not -90 <= latitude <= 90
        or not -180 <= longitude <= 180
    ):
        return None
    return latitude, longitude


def _validate_latlon(values: Iterable[Sequence[Any]]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for pair in values:
        latitude = _finite_float(pair[0])
        longitude = _finite_float(pair[1])
        if latitude is not None and longitude is not None and -90 <= latitude <= 90 and -180 <= longitude <= 180:
            points.append((latitude, longitude))
    return points if len(points) >= 2 else []


def _lonlat_to_latlon(values: Iterable[Sequence[Any]]) -> list[tuple[float, float]]:
    return _validate_latlon((pair[1], pair[0]) for pair in values if len(pair) >= 2)


def _month_color(value: Any) -> str:
    palette = (
        "#2563eb",
        "#0891b2",
        "#16a34a",
        "#ca8a04",
        "#ea580c",
        "#9333ea",
        "#db2777",
        "#475569",
    )
    digest = hashlib.sha256(_display(value, "route").encode("utf-8")).digest()
    return palette[digest[0] % len(palette)]


def _haversine_m(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    radius_m = 6_371_008.8
    phi_a = math.radians(latitude_a)
    phi_b = math.radians(latitude_b)
    delta_phi = math.radians(latitude_b - latitude_a)
    delta_lambda = math.radians(longitude_b - longitude_a)
    value = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_a)
        * math.cos(phi_b)
        * math.sin(delta_lambda / 2.0) ** 2
    )
    return radius_m * 2.0 * math.atan2(
        math.sqrt(value), math.sqrt(max(1.0 - value, 0.0))
    )


def _generalize_home_route_points(
    points: Sequence[tuple[float, float]], home: GeneralizedHomeArea
) -> list[tuple[float, float]]:
    """Collapse route geometry inside the protected circle to its public center."""

    result: list[tuple[float, float]] = []
    public_center = (home.latitude, home.longitude)
    for latitude, longitude in points:
        distance = _haversine_m(
            latitude,
            longitude,
            home.latitude,
            home.longitude,
        )
        point = public_center if distance <= home.radius_m else (latitude, longitude)
        if not result or result[-1] != point:
            result.append(point)
    return result


def _route_popup(row: Mapping[str, Any], *, changed: bool = False) -> str:
    fields = [
        ("Month", _row_value(row, "month", "trip_month", "month_label")),
        ("Origin", _row_value(row, "origin_label", "origin_name")),
        ("Destination", _row_value(row, "destination_label", "destination_name")),
        ("Trips", _row_value(row, "trip_count", "occurrence_count", "route_frequency", "trips")),
        ("Route / chain", _row_value(row, "family_name", "public_chain", "dominant_road_names", "major_roads", "dominant_route")),
    ]
    if changed:
        fields.extend(
            [
                ("Period", _period_text(row)),
                ("Roads removed", row.get("major_roads_removed")),
                ("Roads added", row.get("major_roads_added")),
                ("RCCI", _row_value(row, "RCCI", "rcci")),
                ("Story", _row_value(row, "plain_english_story", "story")),
                ("Confidence", row.get("confidence")),
            ]
        )
    return "<div class='behavior-map-popup'>" + "".join(
        _popup_line(label, value) for label, value in fields
    ) + "</div>"


def _period_text(row: Mapping[str, Any]) -> str:
    month_a = _text(row.get("month_a"))
    month_b = _text(row.get("month_b"))
    return f"{month_a} → {month_b}" if month_a or month_b else ""


def _add_route_rows(
    layer: folium.FeatureGroup,
    frame: pd.DataFrame,
    *,
    generalized_home: GeneralizedHomeArea,
    changed: bool,
    default_color: str | None = None,
) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for _, row in frame.iterrows():
        points = _generalize_home_route_points(
            _route_coordinates(row), generalized_home
        )
        if len(points) < 2:
            continue
        bounds.extend(points)
        color = default_color or _month_color(
            _row_value(row, "month", "trip_month", "month_label", "month_a")
        )
        folium.PolyLine(
            points,
            color=color,
            weight=6 if changed else 3,
            opacity=0.9 if changed else 0.65,
            dash_array="9 6" if changed else None,
            popup=folium.Popup(_route_popup(row, changed=changed), max_width=430),
            tooltip=html.escape(
                _display(
                    _row_value(row, "plain_english_story", "story"),
                    "Major route change",
                )
                if changed
                else _display(
                    _row_value(row, "route_label", "family_name", "public_chain", "month", "trip_month"),
                    "Route",
                )
            ),
        ).add_to(layer)
    return bounds


def generate_verification_map(
    clusters: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    generalized_home: GeneralizedHomeArea | Mapping[str, Any],
    poi_clusters: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    recurring_destinations: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    monthly_routes: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    common_od_routes: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    major_route_changes: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    repeated_trip_chains: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    route_families: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    early_preferred_routes: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    later_preferred_routes: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    sustained_route_changes: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    temporary_alternatives: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    output_path: str | Path | None = None,
    zoom_start: int = 11,
    exact_home_address_for_validation: str | None = None,
    exact_home_coordinates_for_validation: tuple[float, float]
    | Iterable[tuple[float, float]]
    | None = None,
    exact_home_uri_for_validation: str | None = None,
    api_key_for_validation: str | None = None,
    forbidden_privacy_values: Mapping[str, Any] | Iterable[Any] | None = None,
) -> folium.Map:
    """Create the interactive POI/route verification map.

    Home-tagged rows are never rendered.  If such rows still contain a precise
    coordinate, address, or Maps URI, rendering fails rather than silently
    trusting the table.  The only home geometry Folium receives is the broad,
    separately supplied ``generalized_home`` circle.

    The fully rendered HTML is privacy-scanned before the map is returned and,
    when ``output_path`` is supplied, before it is written.  Validation-only
    values are never added to the Folium object or an artifact.
    """

    cluster_frame = _frame(clusters)
    poi_frame = _frame(poi_clusters if poi_clusters is not None else clusters)
    recurring_frame = _frame(
        recurring_destinations if recurring_destinations is not None else clusters
    )
    monthly_frame = _frame(monthly_routes)
    common_frame = _frame(common_od_routes)
    changes_frame = _frame(major_route_changes)
    chains_frame = _frame(repeated_trip_chains)
    families_frame = _frame(route_families)
    early_frame = _frame(early_preferred_routes)
    later_frame = _frame(later_preferred_routes)
    sustained_frame = _frame(sustained_route_changes)
    temporary_frame = _frame(temporary_alternatives)
    home = (
        generalized_home
        if isinstance(generalized_home, GeneralizedHomeArea)
        else GeneralizedHomeArea.from_mapping(generalized_home)
    )

    for frame, name in (
        (cluster_frame, "clusters"),
        (poi_frame, "poi_clusters"),
        (recurring_frame, "recurring_destinations"),
    ):
        _assert_sanitized_home_rows(frame, frame_name=name)

    cluster_frame = _non_home_rows(cluster_frame)
    poi_frame = _non_home_rows(poi_frame)
    recurring_frame = _non_home_rows(recurring_frame)

    map_object = folium.Map(
        location=[home.latitude, home.longitude],
        zoom_start=int(zoom_start),
        control_scale=True,
        tiles="OpenStreetMap",
    )
    layers = {
        "clusters": folium.FeatureGroup(name="Clusters", show=True),
        "pois": folium.FeatureGroup(name="Selected POIs", show=True),
        "recurring": folium.FeatureGroup(name="Recurring destinations", show=False),
        "monthly": folium.FeatureGroup(name="Monthly routes", show=False),
        "common": folium.FeatureGroup(name="Common OD routes", show=False),
        "changes": folium.FeatureGroup(name="Major route changes", show=True),
        "chains": folium.FeatureGroup(name="Repeated trip chains", show=False),
        "families": folium.FeatureGroup(name="Route families", show=False),
        "early": folium.FeatureGroup(name="Earlier preferred route", show=False),
        "later": folium.FeatureGroup(name="Later preferred route", show=True),
        "sustained": folium.FeatureGroup(name="Sustained route changes", show=True),
        "temporary": folium.FeatureGroup(name="Temporary alternatives", show=False),
        "home": folium.FeatureGroup(name="Generalized home area", show=True),
    }
    for layer in layers.values():
        layer.add_to(map_object)

    bounds: list[tuple[float, float]] = [(home.latitude, home.longitude)]
    for index, (_, row) in enumerate(cluster_frame.iterrows(), start=1):
        point = _coordinate_from_row(row)
        if point is None:
            continue
        bounds.append(point)
        cluster_id = _display(
            _row_value(row, "cluster_id", "activity_area_id", "location_cluster_id"),
            str(index),
        )
        role = _display(
            _row_value(row, "inferred_role", "role", "likely_purpose"),
            "Recurring destination",
        )
        popup = "<div class='behavior-map-popup'>" + "".join(
            [
                _popup_line("Cluster", cluster_id),
                _popup_line("Inferred role", role),
                _popup_line("Visits", _row_value(row, "total_visits", "total_visit_count", "visit_count", "endpoint_records")),
                _popup_line("Visit pattern", _row_value(row, "recurrence_pattern", "recurring_frequency", "visit_pattern")),
                _popup_line("Confidence", _row_value(row, "role_confidence", "confidence")),
                _popup_line("Selected POI", row.get("selected_poi_name")),
                _popup_line("Address", row.get("selected_poi_address")),
                _popup_line("Behavioral evidence", row.get("behavioral_evidence")),
                _popup_line("Map evidence", row.get("map_evidence")),
                _popup_line("Classification reason", row.get("classification_reason")),
                _popup_line("Limitations", row.get("limitations")),
            ]
        ) + "</div>"
        marker_html = (
            "<div style='background:#1d4ed8;color:white;border:2px solid white;"
            "border-radius:50%;box-shadow:0 1px 5px #334155;width:28px;height:28px;"
            "line-height:24px;text-align:center;font-weight:700;font-size:11px'>"
            f"{html.escape(cluster_id[:5])}</div>"
        )
        folium.Marker(
            point,
            icon=folium.DivIcon(html=marker_html, icon_size=(28, 28), icon_anchor=(14, 14)),
            popup=folium.Popup(popup, max_width=430),
            tooltip=html.escape(f"Cluster {cluster_id}: {role}"),
        ).add_to(layers["clusters"])

    for _, row in poi_frame.iterrows():
        name = _text(row.get("selected_poi_name"))
        point = _coordinate_from_row(row, poi_first=True)
        if not name or point is None:
            continue
        bounds.append(point)
        maps_uri = _safe_maps_uri(row.get("selected_poi_google_maps_uri"))
        title = html.escape(name)
        if maps_uri:
            title = (
                f"<a href=\"{html.escape(maps_uri, quote=True)}\" target=\"_blank\" "
                f"rel=\"noopener noreferrer\"><strong>{title}</strong></a>"
            )
        else:
            title = f"<strong>{title}</strong>"
        distance = _finite_float(row.get("selected_poi_distance_m"))
        popup = (
            "<div class='behavior-map-popup'>"
            f"<div>{title}</div>"
            + _popup_line("Category", row.get("selected_poi_category"))
            + _popup_line("Address", row.get("selected_poi_address"))
            + _popup_line("Distance from cluster", f"{distance:.0f} m" if distance is not None else "")
            + _popup_line("Source", row.get("selected_poi_source"))
            + _popup_line("Match quality", _row_value(row, "match_quality", "poi_match_quality"))
            + _popup_line("Reasoning", _row_value(row, "classification_reason", "reasoning"))
            + _alternatives_html(row.get("alternative_pois_json"))
            + "</div>"
        )
        folium.Marker(
            point,
            icon=folium.Icon(color="green", icon="info-sign"),
            tooltip=html.escape(name),
            popup=folium.Popup(popup, max_width=450),
        ).add_to(layers["pois"])

    for _, row in recurring_frame.iterrows():
        point = _coordinate_from_row(row)
        recurrence = _text(
            _row_value(row, "recurrence_pattern", "recurring_frequency", "visit_frequency", "frequency")
        ).casefold()
        visits = _finite_float(
            _row_value(row, "total_visits", "total_visit_count", "visit_count", "endpoint_records")
        )
        is_recurring = bool(
            recurrence
            and any(
                word in recurrence
                for word in ("daily", "week", "biweekly", "month", "recurring", "several")
            )
        ) or (visits is not None and visits >= 2)
        if point is None or not is_recurring:
            continue
        bounds.append(point)
        label = _display(
            _row_value(row, "selected_poi_name", "generalized_location", "cluster_id"),
            "Recurring destination",
        )
        folium.CircleMarker(
            point,
            radius=9,
            color="#7e22ce",
            fill=True,
            fill_color="#a855f7",
            fill_opacity=0.25,
            weight=3,
            tooltip=html.escape(label),
            popup=folium.Popup(
                "<div>"
                + _popup_line("Recurring destination", label)
                + _popup_line("Pattern", recurrence)
                + _popup_line("Visits", visits)
                + _popup_line("Typical arrival", row.get("typical_arrival_time"))
                + _popup_line("Typical departure", row.get("typical_departure_time"))
                + "</div>",
                max_width=360,
            ),
        ).add_to(layers["recurring"])

    bounds.extend(
        _add_route_rows(
            layers["monthly"],
            monthly_frame,
            generalized_home=home,
            changed=False,
        )
    )
    bounds.extend(
        _add_route_rows(
            layers["chains"],
            chains_frame,
            generalized_home=home,
            changed=False,
            default_color="#7e22ce",
        )
    )
    bounds.extend(
        _add_route_rows(
            layers["families"],
            families_frame,
            generalized_home=home,
            changed=False,
            default_color="#0f766e",
        )
    )
    bounds.extend(
        _add_route_rows(
            layers["early"],
            early_frame,
            generalized_home=home,
            changed=False,
            default_color="#64748b",
        )
    )
    bounds.extend(
        _add_route_rows(
            layers["later"],
            later_frame,
            generalized_home=home,
            changed=False,
            default_color="#2563eb",
        )
    )
    bounds.extend(
        _add_route_rows(
            layers["sustained"],
            sustained_frame,
            generalized_home=home,
            changed=True,
            default_color="#ea580c",
        )
    )
    bounds.extend(
        _add_route_rows(
            layers["temporary"],
            temporary_frame,
            generalized_home=home,
            changed=True,
            default_color="#dc2626",
        )
    )
    bounds.extend(
        _add_route_rows(
            layers["common"],
            common_frame,
            generalized_home=home,
            changed=False,
            default_color="#0f766e",
        )
    )
    bounds.extend(
        _add_route_rows(
            layers["changes"],
            changes_frame,
            generalized_home=home,
            changed=True,
            default_color="#dc2626",
        )
    )

    home_popup = (
        "<div class='behavior-map-popup'>"
        + _popup_line("Area", home.generalized_location)
        + _popup_line("Confidence", home.confidence)
        + _popup_line("Evidence", home.evidence)
        + _popup_line("Privacy", f"Broad {home.radius_m:.0f} m area; exact location suppressed")
        + "</div>"
    )
    folium.Circle(
        location=(home.latitude, home.longitude),
        radius=home.radius_m,
        color="#b45309",
        fill=True,
        fill_color="#f59e0b",
        fill_opacity=0.13,
        weight=3,
        tooltip=html.escape(home.label),
        popup=folium.Popup(home_popup, max_width=380),
    ).add_to(layers["home"])

    unique_bounds = list(dict.fromkeys(bounds))
    if len(unique_bounds) >= 2:
        map_object.fit_bounds(unique_bounds, padding=(25, 25))
    folium.LayerControl(collapsed=False).add_to(map_object)

    document = map_object.get_root().render()
    validate_html_privacy(
        document,
        exact_home_address=exact_home_address_for_validation,
        exact_home_coordinates=exact_home_coordinates_for_validation,
        exact_home_uri=exact_home_uri_for_validation,
        api_key=api_key_for_validation,
        forbidden_values=forbidden_privacy_values,
    )
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(document, encoding="utf-8")
    return map_object


# Friendly alias for pipeline code that uses "build" terminology.
build_verification_map = generate_verification_map
create_verification_map = generate_verification_map


def _public_home(insights: Mapping[str, Any]) -> Mapping[str, Any]:
    home = insights.get("likely_home") or insights.get("likely_home_area") or {}
    if not isinstance(home, Mapping):
        return {"generalized_location": _text(home)}
    forbidden_keys = {
        "address",
        "formatted_address",
        "reverse_geocoded_address",
        "selected_poi_address",
        "google_maps_uri",
        "selected_poi_google_maps_uri",
        "latitude",
        "longitude",
        "lat",
        "lon",
        "lng",
        "centroid_lat",
        "centroid_lon",
        "medoid_lat",
        "medoid_lon",
        "exact_address",
        "exact_latitude",
        "exact_longitude",
    }
    if any(key in home and not _is_missing(home[key]) for key in forbidden_keys):
        raise PrivacyValidationError(
            "The public likely_home object contains precise coordinates, an address, or a map URI"
        )
    return home


def _html_list(values: Any, *, empty: str) -> str:
    if _is_missing(values):
        return f"<p class=\"empty\">{html.escape(empty)}</p>"
    if isinstance(values, (str, bytes)):
        values = [values]
    if isinstance(values, Mapping):
        values = [values]
    items: list[str] = []
    for value in values:
        if isinstance(value, Mapping):
            statement = _row_value(
                value,
                "finding",
                "statement",
                "summary",
                "pattern",
                "text",
            )
            confidence = _text(value.get("confidence"))
            evidence = _text(value.get("evidence"))
            parts = [_display(statement)]
            if confidence:
                parts.append(f"Confidence: {confidence}")
            if evidence:
                parts.append(f"Evidence: {evidence}")
            text = " — ".join(parts)
        else:
            text = _display(value)
        items.append(f"<li>{html.escape(text)}</li>")
    return "<ul>" + "".join(items) + "</ul>" if items else f"<p class=\"empty\">{html.escape(empty)}</p>"


def _table(
    rows: pd.DataFrame,
    columns: Sequence[tuple[str, str]],
    *,
    empty: str,
    link_column: str | None = None,
    max_rows: int | None = None,
) -> str:
    if rows.empty:
        return f"<p class=\"empty\">{html.escape(empty)}</p>"
    data = rows.head(max_rows) if max_rows else rows
    headers = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body: list[str] = []
    for record in data.to_dict(orient="records"):
        cells: list[str] = []
        uri = _safe_maps_uri(record.get(link_column)) if link_column else None
        for index, (key, _) in enumerate(columns):
            value = _display(record.get(key))
            rendered = html.escape(value)
            if index == 1 and uri and value != "—":
                rendered = (
                    f"<a href=\"{html.escape(uri, quote=True)}\" target=\"_blank\" "
                    f"rel=\"noopener noreferrer\">{rendered}</a>"
                )
            cells.append(f"<td>{rendered}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<div class=\"table-wrap behavior-table\"><table><thead><tr>"
        + headers
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _places_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    _assert_sanitized_home_rows(frame, frame_name="report POI clusters")
    rows: list[dict[str, Any]] = []
    for _, record in _non_home_rows(frame).iterrows():
        name = _text(record.get("selected_poi_name"))
        if not name:
            continue
        arrival = _text(record.get("typical_arrival_time"))
        departure = _text(record.get("typical_departure_time"))
        typical = " / ".join(
            part
            for part in (
                f"arrive {arrival}" if arrival else "",
                f"depart {departure}" if departure else "",
            )
            if part
        )
        reason_parts = [
            _text(record.get("classification_reason")),
            _text(record.get("behavioral_evidence")),
            _text(record.get("map_evidence")),
        ]
        rows.append(
            {
                "purpose": _row_value(record, "inferred_role", "likely_purpose", "role"),
                "place": name,
                "address": record.get("selected_poi_address"),
                "pattern": _row_value(record, "recurrence_pattern", "recurring_frequency", "visit_pattern"),
                "typical_time": typical,
                "confidence": _row_value(record, "role_confidence", "confidence"),
                "reason": " ".join(part for part in reason_parts if part),
                "maps_uri": record.get("selected_poi_google_maps_uri"),
            }
        )
    return pd.DataFrame(rows)


def _recurring_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows: list[dict[str, Any]] = []
    for _, record in frame.iterrows():
        if _is_home_row(record):
            continue
        dates = _row_value(record, "visit_dates", "months_seen", "month_counts")
        if isinstance(dates, (list, tuple, set, dict)):
            dates = json.dumps(dates, ensure_ascii=False)
        rows.append(
            {
                "place": _row_value(record, "selected_poi_name", "poi_name", "generalized_location", "location_label"),
                "address": _row_value(record, "selected_poi_address", "poi_address", "address"),
                "dates": dates,
                "frequency": _row_value(record, "visit_frequency", "recurrence_pattern", "recurring_frequency", "frequency"),
                "typical_time": _row_value(record, "typical_time", "typical_arrival_time"),
                "dwell": _row_value(record, "median_dwell_minutes", "median_dwell_time"),
                "activity": _row_value(record, "inferred_activity", "inferred_role", "likely_purpose"),
                "confidence": record.get("confidence"),
                "alternative": _row_value(record, "alternative_interpretation", "competing_explanation", "limitations"),
            }
        )
    return pd.DataFrame(rows)


def _od_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows: list[dict[str, Any]] = []
    for _, record in frame.iterrows():
        period = _period_text(record)
        roads = []
        removed = _text(record.get("major_roads_removed"))
        added = _text(record.get("major_roads_added"))
        if removed:
            roads.append(f"removed: {removed}")
        if added:
            roads.append(f"added: {added}")
        rows.append(
            {
                "od": " → ".join(
                    filter(
                        None,
                        (
                            _text(record.get("origin_label")),
                            _text(record.get("destination_label")),
                        ),
                    )
                ),
                "period": period,
                "story": _row_value(record, "plain_english_story", "story"),
                "roads": "; ".join(roads),
                "rcci": _row_value(record, "RCCI", "rcci"),
                "confidence": record.get("confidence"),
                "limitations": record.get("limitations"),
            }
        )
    return pd.DataFrame(rows)


def _new_disappearing_rows(insights: Mapping[str, Any], recurring: pd.DataFrame) -> pd.DataFrame:
    source = (
        insights.get("new_or_disappearing_destinations")
        or insights.get("destination_changes")
        or insights.get("new_and_disappearing_destinations")
    )
    if source:
        if isinstance(source, Mapping):
            source = [source]
        rows = []
        for record in source:
            if isinstance(record, Mapping):
                rows.append(
                    {
                        "place": _row_value(record, "place", "poi_name", "location", "location_label"),
                        "change": _row_value(record, "change", "change_type", "status", "pattern"),
                        "period": _row_value(record, "period", "months", "month")
                        or "–".join(
                            filter(
                                None,
                                (
                                    _text(record.get("first_month")),
                                    _text(record.get("last_month")),
                                ),
                            )
                        ),
                        "evidence": record.get("evidence")
                        or (
                            f"{_display(record.get('visit_count'))} visits across "
                            f"{_display(record.get('months_visited'))} months"
                        ),
                        "confidence": record.get("confidence"),
                        "uncertainty": _row_value(record, "uncertainty", "alternative_interpretation", "limitations"),
                    }
                )
            else:
                rows.append({"change": record})
        return pd.DataFrame(rows)
    if recurring.empty:
        return pd.DataFrame()
    status_columns = [
        column
        for column in ("destination_change", "status", "appearance_status", "change_type")
        if column in recurring
    ]
    if not status_columns:
        return pd.DataFrame()
    status = status_columns[0]
    mask = recurring[status].astype(str).str.contains(
        r"\b(new|appear|disappear|stop|ended|seasonal)\b", case=False, regex=True, na=False
    )
    return pd.DataFrame(
        [
            {
                "place": _row_value(row, "selected_poi_name", "poi_name", "location_label"),
                "change": row.get(status),
                "period": _row_value(row, "period", "months_seen", "month"),
                "evidence": row.get("behavioral_evidence"),
                "confidence": row.get("confidence"),
                "uncertainty": _row_value(row, "alternative_interpretation", "limitations"),
            }
            for _, row in recurring.loc[mask].iterrows()
            if not _is_home_row(row)
        ]
    )


def _safe_local_href(value: str | Path | None) -> str:
    href = _text(value)
    if not href:
        return ""
    parsed = urlparse(href)
    if parsed.scheme and parsed.scheme not in {"http", "https"}:
        raise BehaviorReportError("Interactive map link uses an unsafe URI scheme")
    return href


def behavior_report_css() -> str:
    return f"""
{STYLE_MARKER}
.behavior-insights .behavior-lead{{font-size:16px;color:#334155}}
.behavior-insights .finding-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:12px;margin:16px 0}}
.behavior-insights .finding-card{{background:#f8fbff;border:1px solid var(--line);border-radius:13px;padding:14px}}
.behavior-insights .finding-card strong{{display:block;color:var(--text);font-size:16px;margin-bottom:5px}}
.behavior-insights .finding-card p{{margin:0;color:#475569}}
.behavior-insights .finding-card[data-confidence="low"]{{border-left:4px solid #c2410c}}
.behavior-insights .home-privacy{{background:#fff8e6;border-left:5px solid #b65c00;border-radius:12px;padding:15px;margin:12px 0}}
.behavior-insights .interpretation-callout{{background:#fef3c7;border:1px solid #f59e0b;border-radius:12px;padding:14px;margin:14px 0}}
.behavior-insights .timeline-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:12px;margin:14px 0}}
.behavior-insights .timeline-phase{{border-top:5px solid #2563eb;background:#f8fafc;border-radius:10px;padding:14px}}
.behavior-insights .timeline-phase h4{{margin:0 0 6px;color:#1e3a8a}}
.behavior-insights .timeline-phase p{{margin:4px 0;color:#475569}}
.behavior-insights .metric-strip{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:12px 0}}
.behavior-insights .metric-box{{background:#eef6ff;border-radius:10px;padding:12px;text-align:center}}
.behavior-insights .metric-box strong{{display:block;font-size:24px;color:#1d4ed8}}
.behavior-insights .evidence-note{{font-size:13px;color:#64748b}}
.behavior-insights details{{border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin:12px 0;background:#fff}}
.behavior-insights summary{{cursor:pointer;font-weight:700;color:#1e3a8a}}
.behavior-insights .route-share-figure{{overflow-x:auto;background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px;margin:14px 0}}
.behavior-insights .route-share-figure svg{{min-width:720px;width:100%;height:auto}}
.behavior-insights .behavior-table td{{white-space:normal;vertical-align:top;min-width:110px}}
.behavior-insights .map-panel{{background:#eef6ff;border:1px solid #cbdcf8;border-radius:14px;padding:16px}}
.behavior-insights iframe{{width:100%;height:620px;border:1px solid var(--line);border-radius:12px;background:#fff}}
"""


def _percent(value: Any, *, digits: int = 0) -> str:
    number = _finite_float(value)
    return f"{100.0 * number:.{digits}f}%" if number is not None else "—"


def _duration_phrase(value: Any) -> str:
    minutes = _finite_float(value)
    if minutes is None:
        return "stay length unavailable"
    if minutes < 20:
        return f"brief stops (median {minutes:.0f} minutes)"
    if minutes < 60:
        return f"short stays (median {minutes:.0f} minutes)"
    if minutes < 180:
        return f"multi-hour stays (median {minutes / 60.0:.1f} hours)"
    return f"long stays (median {minutes / 60.0:.1f} hours)"


def _route_share_figure(
    insights: Mapping[str, Any], transition: Mapping[str, Any]
) -> str:
    """Render one compact, dependency-free route-family share figure."""
    monthly = insights.get("route_family_monthly_shares") or []
    if not isinstance(monthly, Sequence) or isinstance(monthly, (str, bytes)):
        return ""
    origin = _text(transition.get("origin_cluster_id"))
    destination = _text(transition.get("destination_cluster_id"))
    family_ids = [
        _text(transition.get("baseline_route_family_id")),
        _text(transition.get("later_route_family_id")),
    ]
    rows = [
        row
        for row in monthly
        if isinstance(row, Mapping)
        and _text(row.get("origin_cluster_id")) == origin
        and _text(row.get("destination_cluster_id")) == destination
        and _text(row.get("route_family_id")) in family_ids
        and _finite_float(row.get("eligible_od_trip_count")) not in {None, 0.0}
    ]
    if not rows:
        return ""
    max_index = max(
        int(_finite_float(row.get("observed_month_index")) or 0) for row in rows
    )
    min_index = min(
        int(_finite_float(row.get("observed_month_index")) or 0) for row in rows
    )
    width, height = 920, 300
    left, right, top, bottom = 62, 22, 35, 58
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_position(index: int) -> float:
        span = max(max_index - min_index, 1)
        return left + plot_width * (index - min_index) / span

    def y_position(share: float) -> float:
        return top + plot_height * (1.0 - max(0.0, min(share, 1.0)))

    colors = ["#64748b", "#2563eb"]
    names = [
        _display(transition.get("baseline_route_family"), "Earlier family"),
        _display(transition.get("later_route_family"), "Later family"),
    ]
    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Monthly shares for the earlier and later route families">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for tick in (0.0, 0.5, 1.0):
        y = y_position(tick)
        svg_parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" '
            'stroke="#cbd5e1" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" '
            f'font-size="12" fill="#475569">{tick:.0%}</text>'
        )
    for family_id, color, name in zip(family_ids, colors, names, strict=True):
        family_rows = sorted(
            (row for row in rows if _text(row.get("route_family_id")) == family_id),
            key=lambda row: int(_finite_float(row.get("observed_month_index")) or 0),
        )
        points: list[str] = []
        for row in family_rows:
            share = _finite_float(row.get("route_share"))
            if share is None:
                continue
            index = int(_finite_float(row.get("observed_month_index")) or 0)
            x, y = x_position(index), y_position(share)
            points.append(f"{x:.1f},{y:.1f}")
            count = int(_finite_float(row.get("eligible_od_trip_count")) or 0)
            fill = color if count >= 5 else "#ffffff"
            svg_parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{fill}" '
                f'stroke="{color}" stroke-width="2"><title>'
                f'{html.escape(_text(row.get("month")))}: {share:.0%} ({count} trips)'
                '</title></circle>'
            )
        if points:
            svg_parts.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
                'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        legend_x = left + (0 if color == colors[0] else 330)
        svg_parts.append(
            f'<line x1="{legend_x}" y1="18" x2="{legend_x+28}" y2="18" '
            f'stroke="{color}" stroke-width="4"/>'
            f'<text x="{legend_x+36}" y="22" font-size="13" fill="#334155">'
            f'{html.escape(name)}</text>'
        )
    adoption_month = _text(transition.get("adoption_start"))
    adoption_rows = [row for row in rows if _text(row.get("month")) == adoption_month]
    if adoption_rows:
        adoption_index = int(
            _finite_float(adoption_rows[0].get("observed_month_index")) or 0
        )
        x = x_position(adoption_index)
        svg_parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" '
            'stroke="#b45309" stroke-width="2" stroke-dasharray="5 4"/>'
            f'<text x="{x+6:.1f}" y="{top+14}" font-size="12" fill="#92400e">'
            f'change period {html.escape(adoption_month)}</text>'
        )
    labeled = sorted(rows, key=lambda row: int(_finite_float(row.get("observed_month_index")) or 0))
    label_rows = [labeled[0], labeled[len(labeled) // 2], labeled[-1]]
    for row in label_rows:
        index = int(_finite_float(row.get("observed_month_index")) or 0)
        svg_parts.append(
            f'<text x="{x_position(index):.1f}" y="{height-24}" text-anchor="middle" '
            f'font-size="12" fill="#475569">{html.escape(_text(row.get("month")))}</text>'
        )
    svg_parts.append(
        f'<text x="{left + plot_width/2:.1f}" y="{height-5}" text-anchor="middle" '
        'font-size="12" fill="#64748b">Observed months; hollow points have fewer than five eligible trips</text>'
    )
    svg_parts.append("</svg>")
    title = (
        "When did the driver begin using the alternate route for "
        f"{_display(transition.get('origin_label'))} → "
        f"{_display(transition.get('destination_label'))}?"
    )
    return (
        '<figure class="route-share-figure"><h4>'
        + html.escape(title)
        + "</h4>"
        + "".join(svg_parts)
        + '<figcaption class="evidence-note">Monthly points are shown for context; '
        "the sustained conclusion uses full early/late trip windows because this OD is sparse "
        "in individual months.</figcaption></figure>"
    )


def _render_legacy_real_world_behavior_insights(
    insights: Mapping[str, Any],
    *,
    poi_clusters: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    recurring_patterns: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    od_route_changes: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    map_href: str | Path | None = "outputs/driver_1003_poi_route_insights_map.html",
    embed_map: bool = False,
    max_table_rows: int = 15,
) -> str:
    """Render the required evidence-first real-world behavior section.

    All text and dataframe values are HTML-escaped.  Named POI links are used
    only for non-home rows and only when they are HTTPS Google Maps URLs with no
    key parameter.
    """

    if not isinstance(insights, Mapping):
        raise TypeError("insights must be a mapping")
    home = _public_home(insights)
    poi_frame = _frame(poi_clusters)
    recurring_frame = _frame(recurring_patterns)
    od_frame = _frame(od_route_changes)
    places = _places_display(poi_frame)
    recurring = _recurring_display(recurring_frame)
    od_changes = _od_display(od_frame)
    destination_changes = _new_disappearing_rows(insights, recurring_frame)

    location = _text(
        _first_mapping_value(
            home,
            "generalized_location",
            "neighborhood_and_city",
            "neighborhood",
            "area_label",
        )
    ) or "Generalized home area withheld pending sufficient evidence"
    city = _text(home.get("city"))
    if city and city.casefold() not in location.casefold():
        location = f"{location}, {city}"
    home_evidence = _display(
        _first_mapping_value(home, "evidence", "behavioral_evidence", "classification_reason"),
        "The available evidence did not support a more specific public description.",
    )
    home_confidence = _display(
        _first_mapping_value(home, "confidence", "role_confidence"), "low"
    )
    home_context = _text(
        _first_mapping_value(home, "residential_context", "map_evidence")
    )

    key_findings = insights.get("key_findings", [])
    if isinstance(key_findings, (str, Mapping)):
        key_findings = [key_findings]
    cards: list[str] = []
    for finding in list(key_findings)[:6]:
        if isinstance(finding, Mapping):
            title = _display(_row_value(finding, "title", "label"), "Finding")
            statement = _display(
                _row_value(finding, "finding", "statement", "summary", "text")
            )
            confidence = _text(finding.get("confidence"))
            if confidence:
                statement += f" Confidence: {confidence}."
        else:
            title = "Key finding"
            statement = _display(finding)
        cards.append(
            "<div class=\"finding-card\"><strong>"
            + html.escape(title)
            + "</strong><p>"
            + html.escape(statement)
            + "</p></div>"
        )
    if not cards:
        cards.append(
            "<div class=\"finding-card\"><strong>Evidence status</strong>"
            "<p>No finding met the configured evidence threshold.</p></div>"
        )

    routine = insights.get("likely_routine") or insights.get("strongest_routine")
    if isinstance(routine, Mapping):
        routine_text = _display(
            _row_value(routine, "summary", "routine", "statement", "pattern")
        )
        routine_evidence = _text(routine.get("evidence"))
        routine_confidence = _text(routine.get("confidence"))
        routine_competing = _text(
            _row_value(routine, "competing_explanation", "alternative_interpretation", "limitations")
        )
        routine_parts = [routine_text]
        if routine_evidence:
            routine_parts.append(f"Evidence: {routine_evidence}")
        if routine_confidence:
            routine_parts.append(f"Confidence: {routine_confidence}")
        if routine_competing:
            routine_parts.append(f"Alternative interpretation: {routine_competing}")
        routine_html = "<p>" + html.escape(" ".join(routine_parts)) + "</p>"
    elif routine:
        routine_html = "<p>" + html.escape(_text(routine)) + "</p>"
    else:
        routine_html = (
            "<p>No daily or weekly routine met the configured evidence threshold. "
            "This avoids turning recurrence alone into a claim about work, school, health, or family.</p>"
        )

    limitations = insights.get("limitations") or [
        "Activity purposes are inferred and are not confirmed.",
        "POI proximity does not prove that the driver visited a place.",
        "GPS endpoints can fall in parking lots, entrance roads, or nearby streets.",
        "Google and OpenStreetMap listings may be incomplete, outdated, or changed.",
        "Household, employment, school, and healthcare conclusions are not confirmed.",
    ]
    required_limitations = [
        "Activity purposes are inferred and are not confirmed.",
        "POI proximity does not prove that the driver visited a place.",
        "GPS endpoints can fall in parking lots, entrance roads, or nearby streets.",
        "Google listings may be incomplete or may have changed.",
        "Household, employment, school, and healthcare conclusions are not confirmed.",
    ]
    if isinstance(limitations, str):
        limitations = [limitations]
    limitation_texts = [_text(item) for item in limitations if _text(item)]
    casefolded = " ".join(limitation_texts).casefold()
    for required in required_limitations:
        keywords = [word for word in re.findall(r"[a-z]+", required.casefold()) if len(word) > 5]
        if not any(keyword in casefolded for keyword in keywords):
            limitation_texts.append(required)

    href = _safe_local_href(map_href)
    if href:
        escaped_href = html.escape(href, quote=True)
        map_html = (
            "<div class=\"map-panel\"><p><a href=\""
            + escaped_href
            + "\"><strong>Open the interactive POI and route verification map</strong></a>. "
            "The map shows evidence, candidates, recurring destinations, and route layers; "
            "the home layer is generalized.</p>"
        )
        if embed_map:
            map_html += (
                f"<iframe src=\"{escaped_href}\" loading=\"lazy\" "
                "title=\"Driver 1003 POI and route verification map\"></iframe>"
            )
        map_html += "</div>"
    else:
        map_html = "<p class=\"empty\">The interactive verification map was not produced.</p>"

    places_columns = [
        ("purpose", "Likely purpose"),
        ("place", "Place name"),
        ("address", "Address"),
        ("pattern", "Visit pattern"),
        ("typical_time", "Typical time"),
        ("confidence", "Confidence"),
        ("reason", "Why classified this way"),
    ]
    recurring_columns = [
        ("place", "Named place or area"),
        ("address", "Address"),
        ("dates", "Visit dates / months"),
        ("frequency", "Frequency"),
        ("typical_time", "Typical time"),
        ("dwell", "Median dwell (minutes)"),
        ("activity", "Inferred activity"),
        ("confidence", "Confidence"),
        ("alternative", "Alternative interpretation"),
    ]
    od_columns = [
        ("od", "Named origin → destination"),
        ("period", "Month pair"),
        ("story", "Plain-English route change"),
        ("roads", "Named roads changed"),
        ("rcci", "RCCI"),
        ("confidence", "Confidence"),
        ("limitations", "Limitations"),
    ]
    destination_change_columns = [
        ("place", "Place"),
        ("change", "Observed change"),
        ("period", "Period"),
        ("evidence", "Evidence"),
        ("confidence", "Confidence"),
        ("uncertainty", "Uncertainty"),
    ]

    return f"""{SECTION_BEGIN}
<section id="{SECTION_ID}" class="behavior-insights">
<h2>Real-World Driver Behavior Insights</h2>
<p class="behavior-lead">This section starts with understandable place and route stories, then gives the supporting evidence and uncertainty. Location roles are inferred from repeated timing, dwell, recurrence, map context, and competing explanations—not from the nearest POI alone.</p>

<h3>Key findings</h3>
<div class="finding-grid">{''.join(cards)}</div>

<h3>Likely home area</h3>
<div class="home-privacy"><strong>{html.escape(location)}</strong><p>{html.escape(home_evidence)}</p><p><strong>Confidence:</strong> {html.escape(home_confidence)}{(' · ' + html.escape(home_context)) if home_context else ''}</p><p><strong>Privacy:</strong> The public report suppresses the exact address, exact coordinate, house number, and exact-location map link.</p></div>

<h3>Frequently visited named places</h3>
{_table(places, places_columns, empty="No non-home named place met the match-confidence threshold.", link_column="maps_uri", max_rows=max_table_rows)}

<h3>Recurring monthly/weekly destinations</h3>
{_table(recurring, recurring_columns, empty="No monthly or weekly destination met the recurrence and match-confidence thresholds.", max_rows=max_table_rows)}

<h3>Likely routine</h3>
{routine_html}

<h3>Major route-choice changes</h3>
<p>These comparisons hold the origin and destination areas constant where possible and translate matched FIDs into named roads. Possible explanations include congestion, construction, toll avoidance, or route preference; the GPS data alone cannot determine the cause.</p>
{_table(od_changes, od_columns, empty="No same-OD route change met the trip-count and confidence thresholds.", max_rows=max_table_rows)}

<h3>New or disappearing destinations</h3>
{_table(destination_changes, destination_change_columns, empty="No destination was confidently identified as newly recurring or disappearing during the analysis period.", max_rows=max_table_rows)}

<h3>Interactive map</h3>
{map_html}

<h3>Research limitations</h3>
{_html_list(limitation_texts, empty="No limitations were supplied.")}
</section>
{SECTION_END}"""


def render_real_world_behavior_insights(
    insights: Mapping[str, Any],
    *,
    poi_clusters: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    recurring_patterns: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    od_route_changes: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    map_href: str | Path | None = "outputs/driver_1003_poi_route_insights_map.html",
    embed_map: bool = False,
    max_table_rows: int = 15,
) -> str:
    """Render the presentation-first longitudinal behavior section."""
    if not isinstance(insights, Mapping):
        raise TypeError("insights must be a mapping")
    home = _public_home(insights)
    poi_frame = _frame(poi_clusters)
    recurring_frame = _frame(recurring_patterns)
    adjacent_od_frame = _frame(od_route_changes)
    _assert_sanitized_home_rows(poi_frame, frame_name="report POI clusters")

    location = _display(
        _first_mapping_value(
            home,
            "generalized_location",
            "neighborhood_and_city",
            "neighborhood",
            "area_label",
        ),
        "Generalized home area",
    )
    home_evidence = _display(
        _first_mapping_value(home, "evidence", "behavioral_evidence"),
        "Repeated returns and departures support a neighborhood-level home inference.",
    )
    home_confidence = _display(
        _first_mapping_value(home, "confidence", "role_confidence"), "low"
    )

    transition_records = [
        record
        for record in insights.get("longitudinal_route_transitions", [])
        if isinstance(record, Mapping)
    ]
    temporary_records = [
        record
        for record in insights.get("temporary_route_deviations", [])
        if isinstance(record, Mapping)
    ]

    priority = {
        "likely home area": 0,
        "no workplace identified": 1,
        "sustained route-family change": 2,
        "temporary route deviation": 3,
        "recurring shopping destination": 4,
        "possible healthcare-context pattern": 5,
        "recurring fuel stop": 6,
    }
    raw_findings = insights.get("key_findings") or []
    if isinstance(raw_findings, (str, Mapping)):
        raw_findings = [raw_findings]
    findings = sorted(
        list(raw_findings),
        key=lambda item: priority.get(
            _text(item.get("title") if isinstance(item, Mapping) else "").casefold(),
            99,
        ),
    )[:5]
    finding_cards: list[str] = []
    for finding in findings:
        if isinstance(finding, Mapping):
            title = _display(_row_value(finding, "title", "label"), "Finding")
            statement = _display(
                _row_value(finding, "finding", "statement", "summary", "text")
            )
            confidence = _display(finding.get("confidence"), "not rated")
        else:
            title, statement, confidence = "Finding", _display(finding), "not rated"
        finding_cards.append(
            f'<article class="finding-card" data-confidence="{html.escape(confidence.casefold())}">'
            f"<strong>{html.escape(title)}</strong><p>{html.escape(statement)}</p>"
            f'<p class="evidence-note">Confidence: {html.escape(confidence)}</p></article>'
        )
    if not finding_cards:
        finding_cards.append(
            '<article class="finding-card"><strong>Evidence status</strong>'
            "<p>No long-term finding met the configured evidence threshold.</p></article>"
        )

    routine = insights.get("likely_routine") or {}
    routine_summary = _display(
        _row_value(routine, "summary", "routine", "statement")
        if isinstance(routine, Mapping)
        else routine,
        "No complete repeated routine met the evidence threshold.",
    )
    routine_evidence = (
        _text(routine.get("evidence")) if isinstance(routine, Mapping) else ""
    )
    routine_confidence = (
        _display(routine.get("confidence"), "low")
        if isinstance(routine, Mapping)
        else "low"
    )

    phases: list[tuple[str, str, str]] = []
    if transition_records:
        transition = transition_records[0]
        phases.extend(
            [
                (
                    f"Baseline · {_display(transition.get('baseline_start'))}–{_display(transition.get('baseline_end'))}",
                    "The residential anchor and frequent destinations remained stable.",
                    (
                        f"For {_display(transition.get('origin_label'))} → "
                        f"{_display(transition.get('destination_label'))}, "
                        f"{_display(transition.get('baseline_route_family'))} was the leading "
                        f"early pattern ({_percent(transition.get('baseline_share'))})."
                    ),
                ),
                (
                    f"Emerging change · {_display(transition.get('adoption_start'))}",
                    "A different corridor began appearing for the same repeated trip.",
                    (
                        f"The {_display(transition.get('later_route_family'))} family crossed "
                        f"the earlier family in {_display(transition.get('crossover_month'))}. "
                        "The start and end areas did not change."
                    ),
                ),
                (
                    f"Later window · {_display(transition.get('later_start'))}–{_display(transition.get('later_end'))}",
                    "The alternate corridor became more common, but did not replace the earlier route permanently.",
                    (
                        f"Its share reached {_percent(transition.get('later_share'))} across "
                        f"{_display(transition.get('trips_after'))} late-window trips and it "
                        f"appeared in {_display(transition.get('persistence_months'))} observed months. "
                        + (
                            f"The earlier family reappeared beginning in {_display(transition.get('reversion_month'))}."
                            if _text(transition.get("reversion_month"))
                            else "No later reversion met the threshold."
                        )
                    ),
                ),
            ]
        )
    else:
        timeline = [
            record
            for record in insights.get("behavior_timeline", [])
            if isinstance(record, Mapping)
        ]
        for record in timeline[:4]:
            phases.append(
                (
                    _display(record.get("period"), "Observed period"),
                    _display(record.get("event"), "Recorded behavior"),
                    _display(record.get("evidence")),
                )
            )
    timeline_html = "".join(
        '<article class="timeline-phase">'
        f"<h4>{html.escape(period)}</h4><strong>{html.escape(headline)}</strong>"
        f"<p>{html.escape(story)}</p></article>"
        for period, headline, story in phases
    )

    important = [
        record
        for record in insights.get("important_places", [])
        if isinstance(record, Mapping)
    ]
    transition_cluster_ids = {
        _text(record.get("origin_cluster_id")) for record in transition_records
    } | {_text(record.get("destination_cluster_id")) for record in transition_records}
    place_rows: list[dict[str, Any]] = []
    for record in important:
        confidence = _text(record.get("confidence")).casefold()
        cluster_id = _text(record.get("cluster_id"))
        if confidence not in {"high", "medium"} and cluster_id not in transition_cluster_ids:
            continue
        arrival = _text(record.get("typical_arrival_time"))
        place_rows.append(
            {
                "purpose": _display(record.get("inferred_role"), "unresolved recurring destination"),
                "place": _display(
                    _row_value(record, "place_name", "generalized_location"),
                    "Unresolved area",
                ),
                "address": record.get("address"),
                "pattern": (
                    f"{_display(record.get('visit_pattern'))}; "
                    f"{_duration_phrase(record.get('median_dwell_minutes'))}"
                    + (f"; usually around {arrival}" if arrival else "")
                ),
                "active": "–".join(
                    filter(
                        None,
                        (
                            _text(record.get("first_observed_date")),
                            _text(record.get("last_observed_date")),
                        ),
                    )
                ),
                "confidence": record.get("confidence"),
                "reason": _row_value(
                    record,
                    "classification_reason",
                    "behavioral_evidence",
                    "map_evidence",
                ),
                "maps_uri": record.get("google_maps_uri"),
                "visits": record.get("total_visits"),
            }
        )
    places_frame = pd.DataFrame(place_rows)
    if not places_frame.empty:
        places_frame = places_frame.sort_values("visits", ascending=False)

    revisions = [
        record
        for record in insights.get("activity_role_revisions", [])
        if isinstance(record, Mapping)
    ]
    short_stop_revision = next(
        (
            record
            for record in revisions
            if "workplace" in _text(record.get("previous_label")).casefold()
            or _text(record.get("cluster_id")) == "C002"
        ),
        None,
    )
    if short_stop_revision:
        revision_html = (
            '<div class="interpretation-callout"><strong>Common-sense correction: '
            "frequent does not mean workplace.</strong><p>Although this location is next "
            "to a bank and appears often, its measured stays have a median of "
            f"{_finite_float(short_stop_revision.get('median_dwell_minutes')) or 0:.0f} minutes. "
            "That pattern fits a recurring commercial, financial, pickup, or drop-off stop "
            "better than employment. No workplace is identified in this dataset.</p></div>"
        )
    else:
        revision_html = ""

    transition_rows: list[dict[str, Any]] = []
    for record in transition_records:
        reversion = _text(record.get("reversion_month"))
        transition_rows.append(
            {
                "trip": f"{_display(record.get('origin_label'))} → {_display(record.get('destination_label'))}",
                "earlier": (
                    f"{_display(record.get('baseline_route_family'))} "
                    f"({_percent(record.get('baseline_share'))})"
                ),
                "transition": _display(record.get("adoption_start")),
                "later": (
                    f"{_display(record.get('later_route_family'))} "
                    f"({_percent(record.get('later_share'))})"
                ),
                "persistence": (
                    f"{_display(record.get('persistence_months'))} observed months"
                    + (f"; earlier route reappeared in {reversion}" if reversion else "")
                ),
                "interpretation": record.get("plain_english_story"),
                "confidence": record.get("confidence"),
            }
        )
    transition_frame = pd.DataFrame(transition_rows)
    route_figure = (
        _route_share_figure(insights, transition_records[0])
        if transition_records
        else ""
    )

    temporary_rows = [
        {
            "trip": f"{_display(record.get('origin_label'))} → {_display(record.get('destination_label'))}",
            "period": "–".join(
                filter(
                    None,
                    (
                        _text(record.get("episode_start_month")),
                        _text(record.get("episode_end_month")),
                    ),
                )
            ),
            "route": record.get("family_name"),
            "evidence": (
                f"{_percent(record.get('peak_route_share'))} of "
                f"{_display(record.get('episode_od_trips'))} eligible trips; "
                f"below the threshold again by {_display(record.get('reversion_month'))}"
            ),
            "interpretation": record.get("plain_english_story"),
            "confidence": record.get("confidence"),
        }
        for record in temporary_records
    ]
    temporary_frame = pd.DataFrame(temporary_rows)

    destination_changes = _new_disappearing_rows(insights, recurring_frame)
    recurring_display = _recurring_display(recurring_frame)
    if not recurring_display.empty and "confidence" in recurring_display:
        recurring_display = recurring_display.loc[
            recurring_display["confidence"].astype(str).str.casefold().isin(
                ["high", "medium"]
            )
        ]

    highway = insights.get("highway_surface_street_summary") or {}
    highway_metrics = (
        '<div class="metric-strip">'
        f'<div class="metric-box"><strong>{_percent(highway.get("full_period_highway_distance_share"))}</strong>highway share, full period</div>'
        f'<div class="metric-box"><strong>{_percent(highway.get("full_period_surface_street_distance_share"))}</strong>surface-street share, full period</div>'
        f'<div class="metric-box"><strong>{_percent(highway.get("early_window_highway_distance_share"))} → {_percent(highway.get("late_window_highway_distance_share"))}</strong>early versus late highway share</div>'
        "</div><p>"
        + html.escape(_display(highway.get("interpretation")))
        + "</p>"
    )

    monthly_rows = [
        record
        for record in insights.get("route_family_monthly_shares", [])
        if isinstance(record, Mapping)
        and (
            not transition_records
            or (
                _text(record.get("origin_cluster_id"))
                == _text(transition_records[0].get("origin_cluster_id"))
                and _text(record.get("destination_cluster_id"))
                == _text(transition_records[0].get("destination_cluster_id"))
                and _finite_float(record.get("eligible_od_trip_count")) not in {None, 0.0}
            )
        )
    ]
    monthly_frame = pd.DataFrame(
        [
            {
                "month": record.get("month"),
                "family": record.get("family_name"),
                "family_trips": record.get("family_trip_count"),
                "od_trips": record.get("eligible_od_trip_count"),
                "share": _percent(record.get("route_share")),
                "rolling": _percent(record.get("rolling_3_observed_month_share")),
                "sufficiency": record.get("data_sufficiency"),
            }
            for record in monthly_rows
        ]
    )
    adjacent_od = _od_display(adjacent_od_frame)

    limitations = insights.get("limitations") or []
    if isinstance(limitations, str):
        limitations = [limitations]
    href = _safe_local_href(map_href)
    if href:
        escaped_href = html.escape(href, quote=True)
        map_html = (
            '<div class="map-panel"><p><a href="'
            + escaped_href
            + '"><strong>Open the interactive place, routine, and route-family verification map</strong></a>. '
            "The home area is deliberately generalized.</p>"
        )
        if embed_map:
            map_html += (
                f'<iframe src="{escaped_href}" loading="lazy" '
                'title="Driver 1003 longitudinal behavior verification map"></iframe>'
            )
        map_html += "</div>"
    else:
        map_html = '<p class="empty">The interactive verification map was not produced.</p>'

    place_columns = [
        ("purpose", "Likely purpose"),
        ("place", "Named place or area"),
        ("address", "Address (non-home only)"),
        ("pattern", "Pattern and typical stay"),
        ("active", "Observed period"),
        ("confidence", "Confidence"),
        ("reason", "Why this interpretation"),
    ]
    transition_columns = [
        ("trip", "Repeated trip"),
        ("earlier", "Earlier pattern"),
        ("transition", "Change period"),
        ("later", "Later pattern"),
        ("persistence", "How long / reversions"),
        ("interpretation", "Interpretation"),
        ("confidence", "Confidence"),
    ]
    temporary_columns = [
        ("trip", "Repeated trip"),
        ("period", "Period"),
        ("route", "Temporary route"),
        ("evidence", "Evidence"),
        ("interpretation", "Interpretation"),
        ("confidence", "Confidence"),
    ]
    destination_columns = [
        ("place", "Place"),
        ("change", "Observed change"),
        ("period", "Observed period"),
        ("evidence", "Evidence"),
        ("confidence", "Confidence"),
        ("uncertainty", "Alternative interpretation"),
    ]
    recurring_columns = [
        ("place", "Place or area"),
        ("frequency", "Visit pattern"),
        ("typical_time", "Typical time"),
        ("dwell", "Median dwell (minutes)"),
        ("activity", "Inferred activity"),
        ("confidence", "Confidence"),
        ("alternative", "Alternative interpretation"),
    ]
    monthly_columns = [
        ("month", "Month"),
        ("family", "Route family"),
        ("family_trips", "Family trips"),
        ("od_trips", "Eligible OD trips"),
        ("share", "Monthly share"),
        ("rolling", "3-observed-month share"),
        ("sufficiency", "Data sufficiency"),
    ]
    adjacent_columns = [
        ("od", "Named starting → ending area"),
        ("period", "Month pair"),
        ("story", "What roads changed"),
        ("rcci", "RCCI"),
        ("confidence", "Confidence"),
    ]

    return f"""{SECTION_BEGIN}
<section id="{SECTION_ID}" class="behavior-insights">
<h2>Real-World Driver Behavior Insights</h2>
<p class="behavior-lead">Across 3,284 recorded trips from September 2021 through June 2024, the residential anchor and several recurring destinations stayed recognizable. The clearest long-term route finding is a shift in how one nearby recurring destination was reached—not proof of why the driver changed roads.</p>

<h3>Key Long-Term Findings</h3>
<div class="finding-grid">{''.join(finding_cards)}</div>

<h3>Long-Term Driver Behavior Timeline</h3>
<p>The timeline separates a stable baseline, the first adequately supported alternate-route period, and the later mixed pattern. Sparse months are not treated as proof of a permanent preference.</p>
<div class="timeline-grid">{timeline_html}</div>

<h3>Stable Routine and Important Places</h3>
<div class="home-privacy"><strong>Likely home area: {html.escape(location)}</strong><p>{html.escape(home_evidence)}</p><p><strong>Confidence:</strong> {html.escape(home_confidence)}. The exact address, house number, coordinate, and exact-location map link are suppressed.</p></div>
<p><strong>Strongest complete routine:</strong> {html.escape(routine_summary)} <span class="evidence-note">Confidence: {html.escape(routine_confidence)}. {html.escape(routine_evidence)}</span></p>
{revision_html}
{_table(places_frame, place_columns, empty="No non-home place met the importance and evidence thresholds.", link_column="maps_uri", max_rows=max_table_rows)}

<h3>Sustained Route-Choice Changes</h3>
<p>The destination stayed the same, while the balance between route families changed. This is a late-window distribution shift with intermittent returns to the earlier route—not a claim that one route permanently replaced another.</p>
{_table(transition_frame, transition_columns, empty="No same-OD route-family change met the before/after and persistence thresholds.", max_rows=max_table_rows)}
{route_figure}

<h3>Temporary Route Deviations</h3>
<p>A one-month alternative is treated separately from a lasting change.</p>
{_table(temporary_frame, temporary_columns, empty="No temporary deviation met the minimum monthly trip threshold and reversion test.", max_rows=max_table_rows)}

<h3>New or Disappearing Destinations</h3>
<p>An ending observation means the place stopped appearing in the recordings; it does not prove the underlying activity ended.</p>
{_table(destination_changes, destination_columns, empty="No destination met the recurrence, confidence, and observation-gap thresholds for a new/disappearing claim.", max_rows=max_table_rows)}

<h3>Interactive Map</h3>
{map_html}

<h3>Supporting Monthly Evidence</h3>
{highway_metrics}
<details><summary>Route-family shares for the sustained-change OD pair</summary>
<p class="evidence-note">Monthly route shares are shown even when sparse. A month is marked sufficient only with at least five eligible same-OD trips; the full-window conclusion also requires at least ten trips before and after.</p>
{_table(monthly_frame, monthly_columns, empty="No monthly route-family records were available.", max_rows=None)}
</details>
<details><summary>Recurring destination evidence</summary>
{_table(recurring_display, recurring_columns, empty="No recurring non-home destination met the reporting threshold.", max_rows=max_table_rows)}
</details>
<details><summary>Adjacent-month RCCI comparisons</summary>
<p class="evidence-note">RCCI is supporting technical evidence. It measures network change; it does not establish trip purpose or cause.</p>
{_table(adjacent_od, adjacent_columns, empty="No adjacent-month comparison met the reporting threshold.", max_rows=max_table_rows)}
</details>

<h3>Methodology and Research Limitations</h3>
<p>Trips use explicit source boundaries. Consecutive same-session endpoints create measured stays; cross-session and discontinuous intervals remain censored and are excluded from dwell medians. Similar road sequences are grouped into route families after removing endpoint-access variation.</p>
{_html_list(limitations, empty="No limitations were supplied.")}
</section>
{SECTION_END}"""


# Short alias for callers that already refer to a behavior HTML renderer.
render_behavior_insights_html = render_real_world_behavior_insights
render_behavior_section = render_real_world_behavior_insights


def _remove_marker_block(document: str, begin: str, end: str) -> str:
    return re.sub(
        r"(?:[ \t]*\r?\n)?"
        + re.escape(begin)
        + r".*?"
        + re.escape(end)
        + r"(?:[ \t]*\r?\n)?",
        "",
        document,
        flags=re.DOTALL | re.IGNORECASE,
    )


def inject_real_world_behavior_section(document: str, section_html: str) -> str:
    """Replace legacy insight blocks and insert one section after the summary."""

    if not isinstance(document, str) or not document.strip():
        raise BehaviorReportError("Target report HTML is empty")
    if not isinstance(section_html, str) or not section_html.strip():
        raise BehaviorReportError("Rendered behavior section is empty")

    result = document
    # Remove any existing canonical block before choosing the required
    # near-top insertion point. Older builds placed this section after the long
    # technical tables; retaining that anchor would perpetuate the wrong order.
    canonical = re.search(
        re.escape(SECTION_BEGIN) + r".*?" + re.escape(SECTION_END),
        result,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if canonical:
        result = _remove_marker_block(result, SECTION_BEGIN, SECTION_END)
    for begin, end in (
        (OLD_RESEARCH_BEGIN, OLD_RESEARCH_END),
        (SECTION_BEGIN, SECTION_END),
        (OLD_RESEARCH_NAV_BEGIN, OLD_RESEARCH_NAV_END),
        (NAV_BEGIN, NAV_END),
    ):
        result = _remove_marker_block(result, begin, end)

    if SECTION_BEGIN not in section_html or SECTION_END not in section_html:
        section_html = f"{SECTION_BEGIN}\n{section_html}\n{SECTION_END}"

    if STYLE_MARKER not in result:
        css = behavior_report_css()
        if "</style>" in result:
            result = result.replace("</style>", css + "\n</style>", 1)
        elif "</head>" in result:
            result = result.replace("</head>", f"<style>{css}</style>\n</head>", 1)
        else:
            raise BehaviorReportError("Could not find a stable report style insertion point")

    nav = (
        f"{NAV_BEGIN}\n"
        f'<a href="#{SECTION_ID}">Real-World Driver Behavior Insights</a>\n'
        f"{NAV_END}"
    )
    nav_close = re.search(r"</nav\s*>", result, flags=re.IGNORECASE)
    if nav_close:
        before_nav_close = result[: nav_close.start()].rstrip()
        result = (
            before_nav_close
            + "\n"
            + nav
            + "\n"
            + result[nav_close.start() :]
        )

    placeholders = (
        "<!-- DRIVER 1003 REAL WORLD BEHAVIOR INSIGHTS PLACEHOLDER -->",
        "<!-- REAL WORLD DRIVER BEHAVIOR INSIGHTS PLACEHOLDER -->",
        "<!-- REAL_WORLD_INSIGHTS_PLACEHOLDER -->",
    )
    inserted = False
    for placeholder in placeholders:
        if placeholder in result:
            result = result.replace(placeholder, section_html, 1)
            inserted = True
            break

    if not inserted:
        executive = re.search(
            r"<section\b(?=[^>]*\bid\s*=\s*['\"]executive-summary['\"])[^>]*>.*?</section\s*>",
            result,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if executive:
            result = (
                result[: executive.end()]
                + "\n\n"
                + section_html
                + "\n"
                + result[executive.end() :]
            )
            inserted = True

    if not inserted:
        summary_end = re.search(
            r"<!--\s*END (?:DRIVER 1003 )?EXECUTIVE SUMMARY\s*-->",
            result,
            flags=re.IGNORECASE,
        )
        if summary_end:
            result = (
                result[: summary_end.end()]
                + "\n\n"
                + section_html
                + result[summary_end.end() :]
            )
            inserted = True

    if not inserted:
        raise BehaviorReportError(
            "Could not find the executive summary or behavior-insight placeholder"
        )
    result = re.sub(
        re.escape(SECTION_END) + r"(?:[ \t]*\r?\n){2,}",
        SECTION_END + "\n",
        result,
        count=1,
    )
    if result.count(SECTION_BEGIN) != 1 or result.count(SECTION_END) != 1:
        raise BehaviorReportError("Report insertion did not produce one canonical section")
    return result


# Alias emphasizes the replacement semantics used by the integration script.
replace_report_insights = inject_real_world_behavior_section
insert_report_section = inject_real_world_behavior_section


def _read_html(value: str | Path) -> str:
    if isinstance(value, Path):
        return value.read_text(encoding="utf-8")
    if "<" not in value and "\n" not in value:
        candidate = Path(value)
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return value


def _normalized_for_scan(document: str) -> str:
    return html.unescape(document).casefold()


def _coordinate_tokens(value: float) -> set[str]:
    tokens = {str(value)}
    for digits in range(5, 10):
        fixed = f"{value:.{digits}f}"
        tokens.add(fixed)
        tokens.add(fixed.rstrip("0").rstrip("."))
    return {token for token in tokens if token and token not in {"-0", "0"}}


def find_html_privacy_violations(
    document_or_path: str | Path,
    *,
    exact_home_address: str | None = None,
    exact_home_coordinates: tuple[float, float]
    | Iterable[tuple[float, float]]
    | None = None,
    exact_home_uri: str | None = None,
    api_key: str | None = None,
    forbidden_values: Mapping[str, Any] | Iterable[Any] | None = None,
) -> list[str]:
    """Return violation codes without ever returning protected values."""

    document = _read_html(document_or_path)
    normalized = _normalized_for_scan(document)
    issues: set[str] = set()

    if exact_home_address and exact_home_address.strip().casefold() in normalized:
        issues.add("exact_home_address")
    if exact_home_uri and html.unescape(exact_home_uri).casefold() in normalized:
        issues.add("exact_home_uri")
    if api_key and api_key in document:
        issues.add("api_key")

    coordinate_pairs: list[tuple[float, float]] = []
    if exact_home_coordinates is not None:
        candidate = exact_home_coordinates
        if (
            isinstance(candidate, tuple)
            and len(candidate) == 2
            and _finite_float(candidate[0]) is not None
            and _finite_float(candidate[1]) is not None
        ):
            coordinate_pairs = [(float(candidate[0]), float(candidate[1]))]
        else:
            try:
                coordinate_pairs = [
                    (float(pair[0]), float(pair[1]))  # type: ignore[index]
                    for pair in candidate  # type: ignore[union-attr]
                ]
            except (TypeError, ValueError, IndexError):
                coordinate_pairs = []
    for latitude, longitude in coordinate_pairs:
        latitude_found = any(token in document for token in _coordinate_tokens(latitude))
        longitude_found = any(token in document for token in _coordinate_tokens(longitude))
        if latitude_found and longitude_found:
            issues.add("exact_home_coordinates")

    if forbidden_values:
        items = (
            forbidden_values.items()
            if isinstance(forbidden_values, Mapping)
            else (("forbidden_value", value) for value in forbidden_values)
        )
        for label, value in items:
            if _is_missing(value):
                continue
            normalized_label = re.sub(
                r"[^a-z0-9_]+", "_", str(label).casefold()
            ).strip("_")
            safe_labels = {
                "api_key",
                "exact_home_address",
                "exact_home_coordinates",
                "exact_home_coordinate",
                "exact_home_uri",
                "home_address",
                "home_coordinates",
                "home_coordinate",
                "home_coords",
                "home_uri",
            }
            # A mapping key can itself be sensitive (for example, an address),
            # so only known category labels may reach an exception message.
            issue_label = (
                normalized_label
                if normalized_label in safe_labels
                else "forbidden_value"
            )
            if (
                isinstance(value, (tuple, list))
                and len(value) == 2
                and _finite_float(value[0]) is not None
                and _finite_float(value[1]) is not None
            ):
                first_found = any(
                    token in document
                    for token in _coordinate_tokens(float(value[0]))
                )
                second_found = any(
                    token in document
                    for token in _coordinate_tokens(float(value[1]))
                )
                if first_found and second_found:
                    issues.add(issue_label)
                continue
            if isinstance(value, Mapping):
                scalar_values = list(value.values())
            elif isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
                scalar_values = list(value)
            else:
                scalar_values = [value]
            for scalar in scalar_values:
                if _is_missing(scalar):
                    continue
                needle = html.unescape(str(scalar)).casefold()
                if needle and needle in normalized:
                    issues.add(issue_label)
                    break

    # This catches accidental key-bearing request URLs even when the caller did
    # not pass the key itself.  It intentionally reports only the field class.
    if re.search(
        r"(?:[?&]|&amp;)(?:key|api_key|apikey|google_maps_api_key)\s*=",
        normalized,
        flags=re.IGNORECASE,
    ):
        issues.add("api_key_parameter")

    return sorted(issues)


def validate_html_privacy(
    document_or_path: str | Path,
    *,
    exact_home_address: str | None = None,
    exact_home_coordinates: tuple[float, float]
    | Iterable[tuple[float, float]]
    | None = None,
    exact_home_uri: str | None = None,
    api_key: str | None = None,
    forbidden_values: Mapping[str, Any] | Iterable[Any] | None = None,
) -> list[str]:
    """Raise on a privacy leak and otherwise return an empty issue list.

    Exception text contains only violation categories; it never echoes the
    protected address, coordinate, URI, key, or caller-provided token.
    """

    issues = find_html_privacy_violations(
        document_or_path,
        exact_home_address=exact_home_address,
        exact_home_coordinates=exact_home_coordinates,
        exact_home_uri=exact_home_uri,
        api_key=api_key,
        forbidden_values=forbidden_values,
    )
    if issues:
        raise PrivacyValidationError(
            "Public HTML failed privacy validation: " + ", ".join(issues)
        )
    return issues


def update_report_html(
    report_path: str | Path,
    section_html: str,
    *,
    output_path: str | Path | None = None,
    exact_home_address: str | None = None,
    exact_home_coordinates: tuple[float, float]
    | Iterable[tuple[float, float]]
    | None = None,
    exact_home_uri: str | None = None,
    api_key: str | None = None,
    forbidden_values: Mapping[str, Any] | Iterable[Any] | None = None,
) -> Path:
    """Replace legacy insight sections, validate privacy, and write the report."""

    source = Path(report_path)
    if not source.exists():
        raise FileNotFoundError(f"Target report does not exist: {source}")
    target = Path(output_path) if output_path is not None else source
    updated = inject_real_world_behavior_section(
        source.read_text(encoding="utf-8"), section_html
    )
    updated = "\n".join(line.rstrip() for line in updated.splitlines()) + "\n"
    validate_html_privacy(
        updated,
        exact_home_address=exact_home_address,
        exact_home_coordinates=exact_home_coordinates,
        exact_home_uri=exact_home_uri,
        api_key=api_key,
        forbidden_values=forbidden_values,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(updated, encoding="utf-8")
    return target


# Concise backwards-compatible name for integration code.
GeneralizedHome = GeneralizedHomeArea


__all__ = [
    "BehaviorReportError",
    "PrivacyValidationError",
    "GeneralizedHomeArea",
    "GeneralizedHome",
    "MIN_GENERALIZED_HOME_RADIUS_M",
    "SECTION_BEGIN",
    "SECTION_END",
    "build_verification_map",
    "create_verification_map",
    "generate_verification_map",
    "render_behavior_insights_html",
    "render_behavior_section",
    "render_real_world_behavior_insights",
    "behavior_report_css",
    "inject_real_world_behavior_section",
    "replace_report_insights",
    "insert_report_section",
    "find_html_privacy_violations",
    "validate_html_privacy",
    "update_report_html",
]
