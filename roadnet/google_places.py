"""Server-side Google Geocoding and Places (New) access.

This module intentionally has no browser-facing integration.  It reads the
Google Maps credential only from ``GOOGLE_MAPS_API_KEY`` at request time and
never places that credential in an object representation, return value, cache
file, log message, or persisted request URL/header.

The caller must supply a cache directory that is excluded from version
control.  Cache file names are SHA-256 digests of non-secret, normalized
request parameters.  Cache files contain only a sanitized successful response
payload and its source label; request URLs, query parameters, and headers are
not cached.  (``googleMapsUri`` is an explicitly requested Places response
field, not a request URL.)

Typical usage::

    client = GoogleMapsClient(cache_dir=Path(".cache/google_places"),
                              request_budget=50)
    address = client.reverse_geocode(26.12, -80.14)
    nearby = client.search_nearby_staged(26.12, -80.14)
    print(client.request_stats.to_dict())

``requests.Session`` can be injected for tests.  Importing or constructing the
client never performs a network request.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import requests


GOOGLE_MAPS_API_KEY_ENV = "GOOGLE_MAPS_API_KEY"
GEOCODING_ENDPOINT = "https://maps.googleapis.com/maps/api/geocode/json"
PLACES_NEARBY_ENDPOINT = "https://places.googleapis.com/v1/places:searchNearby"

# This is deliberately fixed.  Callers cannot broaden the Places response.
PLACES_FIELD_MASK = (
    "places.displayName,"
    "places.formattedAddress,"
    "places.location,"
    "places.primaryType,"
    "places.types,"
    "places.businessStatus,"
    "places.googleMapsUri"
)
PLACES_RESPONSE_FIELDS = frozenset(
    {
        "displayName",
        "formattedAddress",
        "location",
        "primaryType",
        "types",
        "businessStatus",
        "googleMapsUri",
    }
)

DEFAULT_SEARCH_RADII_M = (50, 100, 250, 500)
MAX_BILLABLE_REQUESTS = 100
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MIN_INTERVAL_SECONDS = 0.1
DEFAULT_MAX_RETRIES = 2
_CACHE_VERSION = 1
_RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_URL_RE = re.compile(r"https?://[^\s\]\[<>{}\"']+", re.IGNORECASE)
_GOOGLE_KEY_RE = re.compile(r"AIza[0-9A-Za-z_-]{20,}")
_KEY_ASSIGNMENT_RE = re.compile(
    r"(?i)(?P<label>(?:api[_ -]?key|key|x-goog-api-key)\s*[=:]\s*)"
    r"(?P<value>[^\s,;&]+)"
)
_SAFE_STATUS_RE = re.compile(r"[^A-Z0-9_.-]+")
_SENSITIVE_MAPPING_KEYS = frozenset(
    {
        "key",
        "api_key",
        "apikey",
        "x-goog-api-key",
        "authorization",
        "headers",
        "request_headers",
        "request_url",
    }
)


class GoogleAPIErrorCategory(str, Enum):
    """Stable, JSON-friendly categories for Google request failures."""

    MISSING_KEY = "missing_key"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NETWORK = "network"
    TIMEOUT = "timeout"
    AUTHORIZATION = "authorization"
    API_DISABLED = "api_disabled"
    BILLING = "billing"
    QUOTA = "quota"
    RATE_LIMIT = "rate_limit"
    INVALID_REQUEST = "invalid_request"
    MALFORMED_RESPONSE = "malformed_response"
    CACHE_MISS = "cache_miss"
    SERVER = "server"
    UNKNOWN = "unknown"


class GoogleAPIError(RuntimeError):
    """A sanitized Google API failure safe to show in a CLI or report.

    The exception deliberately does not retain a response, prepared request,
    headers, query parameters, or API key.  ``to_dict`` therefore cannot leak
    credential-bearing transport details.
    """

    def __init__(
        self,
        *,
        category: GoogleAPIErrorCategory,
        api: str,
        message: str,
        http_status: int | None = None,
        api_status: str | None = None,
        retryable: bool = False,
    ) -> None:
        safe_message = _sanitize_error_message(message)
        super().__init__(safe_message)
        self.category = category
        self.api = api
        self.http_status = http_status
        self.api_status = _sanitize_api_status(api_status)
        self.retryable = bool(retryable)

    def to_dict(self) -> dict[str, Any]:
        """Return a transport-detail-free representation for structured logs."""

        return {
            "category": self.category.value,
            "api": self.api,
            "http_status": self.http_status,
            "api_status": self.api_status,
            "message": str(self),
            "retryable": self.retryable,
        }


class MissingGoogleMapsAPIKey(GoogleAPIError):
    """Raised on a cache miss when ``GOOGLE_MAPS_API_KEY`` is unavailable."""


class GoogleRequestBudgetExceeded(GoogleAPIError):
    """Raised before a request that would exceed the configured hard budget."""


@dataclass(frozen=True)
class GoogleAPIResponse:
    """Successful response plus non-secret provenance metadata."""

    source: str
    payload: dict[str, Any]
    cache_hit: bool
    http_status: int = 200
    retrieved_at_utc: str = ""

    @property
    def data(self) -> dict[str, Any]:
        """Alias for ``payload`` for callers that prefer a generic name."""

        return self.payload

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable response wrapper."""

        return {
            "source": self.source,
            "cache_hit": self.cache_hit,
            "http_status": self.http_status,
            "retrieved_at_utc": self.retrieved_at_utc,
            "data": self.payload,
        }


@dataclass(frozen=True)
class NearbySearchStage:
    """One radius attempted by :meth:`GoogleMapsClient.search_nearby_staged`."""

    radius_m: int
    response: GoogleAPIResponse
    place_count: int
    stop_condition_met: bool


@dataclass(frozen=True)
class StagedNearbySearchResult:
    """Combined, de-duplicated candidates from an increasing-radius search."""

    source: str
    stages: tuple[NearbySearchStage, ...]
    candidates: tuple[dict[str, Any], ...]
    stopped_radius_m: int | None
    stop_reason: str

    @property
    def cache_hits(self) -> int:
        """Number of stages served from the local cache."""

        return sum(1 for stage in self.stages if stage.response.cache_hit)


@dataclass(frozen=True)
class RequestStats:
    """Immutable request-accounting snapshot for one client/run."""

    request_budget: int
    google_requests: int
    geocoding_requests: int
    places_requests: int
    successful_google_responses: int
    failed_google_responses: int
    retries: int
    cache_hits: int
    cache_read_errors: int
    cache_write_errors: int
    sources_used: tuple[str, ...]
    source_result_counts: dict[str, int]
    error_categories: dict[str, int]

    @property
    def requests_remaining(self) -> int:
        """Requests still available before the hard budget is reached."""

        return max(0, self.request_budget - self.google_requests)

    @property
    def request_count(self) -> int:
        """Compatibility alias for the total actual Google HTTP attempts."""

        return self.google_requests

    @property
    def sources(self) -> tuple[str, ...]:
        """Compatibility alias for :attr:`sources_used`."""

        return self.sources_used

    @property
    def errors(self) -> dict[str, int]:
        """Compatibility alias for categorized error counts."""

        return dict(self.error_categories)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe counters without any credential or transport data."""

        return {
            "request_budget": self.request_budget,
            "request_count": self.request_count,
            "google_requests": self.google_requests,
            "requests_remaining": self.requests_remaining,
            "geocoding_requests": self.geocoding_requests,
            "places_requests": self.places_requests,
            "successful_google_responses": self.successful_google_responses,
            "failed_google_responses": self.failed_google_responses,
            "retries": self.retries,
            "cache_hits": self.cache_hits,
            "cache_read_errors": self.cache_read_errors,
            "cache_write_errors": self.cache_write_errors,
            "sources_used": list(self.sources_used),
            "sources": list(self.sources),
            "source_result_counts": dict(self.source_result_counts),
            "error_categories": dict(self.error_categories),
            "errors": self.errors,
        }


class _MutableStats:
    """Lock-protected mutable storage backing :class:`RequestStats`."""

    def __init__(self) -> None:
        self.google_requests = 0
        self.geocoding_requests = 0
        self.places_requests = 0
        self.successful_google_responses = 0
        self.failed_google_responses = 0
        self.retries = 0
        self.cache_hits = 0
        self.cache_read_errors = 0
        self.cache_write_errors = 0
        self.source_result_counts: Counter[str] = Counter()
        self.error_categories: Counter[str] = Counter()


class SuccessfulResponseCache:
    """Small atomic JSON cache for successful, sanitized API payloads.

    Parameters
    ----------
    directory:
        Caller-supplied private cache directory.  The caller is responsible for
        ensuring this directory is ignored by Git.  File names contain only a
        source slug and a SHA-256 digest of normalized, non-secret parameters.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    @staticmethod
    def make_key(namespace: str, request_identity: Mapping[str, Any]) -> str:
        """Hash a non-secret request identity into a deterministic cache key."""

        canonical = json.dumps(
            {"namespace": namespace, "request": request_identity},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _path(self, namespace: str, key: str) -> Path:
        safe_namespace = re.sub(r"[^a-z0-9_-]+", "_", namespace.lower()).strip("_")
        if not safe_namespace:
            safe_namespace = "google"
        return self.directory / f"{safe_namespace}_{key}.json"

    def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        """Return a cached payload, or ``None`` for missing/invalid cache data."""

        entry = self.get_with_metadata(namespace, key)
        return entry[0] if entry is not None else None

    def get_with_metadata(
        self,
        namespace: str,
        key: str,
    ) -> tuple[dict[str, Any], str] | None:
        """Return a cached payload and a non-secret retrieval timestamp.

        Version-1 cache files created before timestamps were recorded remain
        valid.  Their file modification time is used as the best available
        retrieval-time provenance instead of invalidating a successful cache
        and repeating a billable request.
        """

        path = self._path(namespace, key)
        try:
            with path.open("r", encoding="utf-8") as handle:
                envelope = json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise CacheReadError from None

        if not isinstance(envelope, dict):
            raise CacheReadError
        if envelope.get("cache_version") != _CACHE_VERSION:
            return None
        if envelope.get("source") != namespace:
            raise CacheReadError
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise CacheReadError
        retrieved_at = str(envelope.get("retrieved_at_utc") or "").strip()
        if not retrieved_at:
            try:
                retrieved_at = datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(timespec="seconds")
            except OSError:
                retrieved_at = "unknown"
        return payload, retrieved_at

    def put(
        self,
        namespace: str,
        key: str,
        payload: Mapping[str, Any],
        *,
        retrieved_at_utc: str | None = None,
    ) -> None:
        """Atomically persist a sanitized successful response payload."""

        path = self._path(namespace, key)
        envelope = {
            "cache_version": _CACHE_VERSION,
            "source": namespace,
            "retrieved_at_utc": retrieved_at_utc
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "payload": payload,
        }
        encoded = json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        temp_name = f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        temp_path = self.directory / temp_name
        try:
            descriptor = os.open(
                temp_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except (OSError, TypeError, ValueError):
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise CacheWriteError from None


class CacheReadError(RuntimeError):
    """Internal signal that a cache file could not be safely read."""


class CacheWriteError(RuntimeError):
    """Internal signal that a cache file could not be safely written."""


StrongMatchPredicate = Callable[[Sequence[Mapping[str, Any]], int], bool]


class GoogleMapsClient:
    """Budgeted, cached client for reverse geocoding and nearby Places.

    Parameters
    ----------
    cache_dir:
        Required caller-supplied directory that must be ignored by version
        control.  Exact cluster coordinates and Google response data can be
        sensitive even though the API key is never cached.
    session:
        Optional injected ``requests.Session``-compatible object.  Useful for
        deterministic tests.  The caller retains ownership of an injected
        session; :meth:`close` closes only a session created by this client.
    request_budget:
        Maximum actual HTTP attempts (including retries) for this client/run.
        It defaults to, and may never exceed, 100.  Cache hits consume no
        budget.  A request is refused *before* it could exceed the budget.
    timeout_seconds:
        Per-request timeout passed to ``requests``.
    min_interval_seconds:
        Minimum spacing between HTTP attempt start times.
    max_retries:
        Retries after the first attempt for transient network, rate-limit, and
        server failures.  Every retry consumes request budget.
    sleeper, clock:
        Injectable timing functions for fast deterministic tests.
    """

    GEOCODING_SOURCE = "google_geocoding_api"
    PLACES_SOURCE = "google_places_api_new"

    def __init__(
        self,
        *,
        cache_dir: str | os.PathLike[str],
        session: requests.Session | None = None,
        request_budget: int = MAX_BILLABLE_REQUESTS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        allow_network: bool = True,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(request_budget, bool) or not isinstance(request_budget, int):
            raise TypeError("request_budget must be an integer")
        if not 1 <= request_budget <= MAX_BILLABLE_REQUESTS:
            raise ValueError(
                f"request_budget must be between 1 and {MAX_BILLABLE_REQUESTS}"
            )
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        if not math.isfinite(min_interval_seconds) or min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be finite and non-negative")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise TypeError("max_retries must be an integer")
        if not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")

        self._cache = SuccessfulResponseCache(cache_dir)
        self._session = session if session is not None else requests.Session()
        self._owns_session = session is None
        self._request_budget = request_budget
        self._timeout_seconds = float(timeout_seconds)
        self._min_interval_seconds = float(min_interval_seconds)
        self._max_retries = max_retries
        self._allow_network = bool(allow_network)
        self._sleep = sleeper
        self._clock = clock
        self._last_request_started: float | None = None
        self._stats = _MutableStats()
        self._stats_lock = threading.Lock()
        self._request_start_lock = threading.Lock()

    def __repr__(self) -> str:
        """Return a representation containing counters but never credentials."""

        stats = self.request_stats
        return (
            f"{type(self).__name__}(request_budget={stats.request_budget}, "
            f"google_requests={stats.google_requests}, cache_hits={stats.cache_hits})"
        )

    def __enter__(self) -> "GoogleMapsClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the internally-created session, if any."""

        if self._owns_session:
            self._session.close()

    @property
    def api_key_detected(self) -> bool:
        """Whether the required environment variable currently has a value."""

        return google_maps_api_key_detected()

    @property
    def request_stats(self) -> RequestStats:
        """Return a thread-safe immutable snapshot of all request counters."""

        with self._stats_lock:
            return RequestStats(
                request_budget=self._request_budget,
                google_requests=self._stats.google_requests,
                geocoding_requests=self._stats.geocoding_requests,
                places_requests=self._stats.places_requests,
                successful_google_responses=self._stats.successful_google_responses,
                failed_google_responses=self._stats.failed_google_responses,
                retries=self._stats.retries,
                cache_hits=self._stats.cache_hits,
                cache_read_errors=self._stats.cache_read_errors,
                cache_write_errors=self._stats.cache_write_errors,
                sources_used=tuple(sorted(self._stats.source_result_counts)),
                source_result_counts=dict(self._stats.source_result_counts),
                error_categories=dict(self._stats.error_categories),
            )

    @property
    def stats(self) -> RequestStats:
        """Short alias for :attr:`request_stats`."""

        return self.request_stats

    def reverse_geocode(
        self,
        latitude: float,
        longitude: float,
        *,
        language: str | None = None,
    ) -> GoogleAPIResponse:
        """Reverse geocode one normalized cluster representative coordinate.

        A cache hit is returned even when no API key is currently configured.
        The method treats Geocoding ``OK`` and ``ZERO_RESULTS`` as successful,
        cacheable responses.  Other API statuses raise :class:`GoogleAPIError`.
        """

        lat, lon = _normalize_coordinates(latitude, longitude)
        normalized_language = _normalize_language(language)
        identity: dict[str, Any] = {"latitude": lat, "longitude": lon}
        if normalized_language is not None:
            identity["language"] = normalized_language
        return self._cached_request(
            source=self.GEOCODING_SOURCE,
            api="geocoding",
            request_identity=identity,
            method="GET",
            endpoint=GEOCODING_ENDPOINT,
            params_factory=lambda key: {
                "latlng": f"{lat:.7f},{lon:.7f}",
                "key": key,
                **(
                    {"language": normalized_language}
                    if normalized_language is not None
                    else {}
                ),
            },
            json_body=None,
            headers_factory=None,
        )

    def search_nearby(
        self,
        latitude: float,
        longitude: float,
        radius_m: int | float,
        *,
        max_results: int = 10,
        included_types: Sequence[str] | None = None,
        rank_preference: str = "DISTANCE",
    ) -> GoogleAPIResponse:
        """Call Places API (New) ``searchNearby`` for one cluster and radius.

        The response field mask is fixed to the seven approved place fields;
        callers cannot request additional data.  Up to 20 candidates may be
        requested, and successful empty results are cached.
        """

        lat, lon = _normalize_coordinates(latitude, longitude)
        radius = _normalize_radius(radius_m)
        if isinstance(max_results, bool) or not isinstance(max_results, int):
            raise TypeError("max_results must be an integer")
        if not 1 <= max_results <= 20:
            raise ValueError("max_results must be between 1 and 20")
        normalized_types = _normalize_place_types(included_types)
        rank = str(rank_preference).strip().upper()
        if rank not in {"DISTANCE", "POPULARITY"}:
            raise ValueError("rank_preference must be DISTANCE or POPULARITY")

        identity: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "radius_m": radius,
            "max_results": max_results,
            "rank_preference": rank,
        }
        if normalized_types:
            identity["included_types"] = list(normalized_types)

        body: dict[str, Any] = {
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": lat, "longitude": lon},
                    "radius": radius,
                }
            },
            "maxResultCount": max_results,
            "rankPreference": rank,
        }
        if normalized_types:
            body["includedTypes"] = list(normalized_types)

        return self._cached_request(
            source=self.PLACES_SOURCE,
            api="places",
            request_identity=identity,
            method="POST",
            endpoint=PLACES_NEARBY_ENDPOINT,
            params_factory=None,
            json_body=body,
            headers_factory=lambda key: {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": key,
                "X-Goog-FieldMask": PLACES_FIELD_MASK,
            },
        )

    def search_nearby_staged(
        self,
        latitude: float,
        longitude: float,
        *,
        radii_m: Sequence[int | float] = DEFAULT_SEARCH_RADII_M,
        max_results: int = 10,
        included_types: Sequence[str] | None = None,
        rank_preference: str = "DISTANCE",
        strong_match: StrongMatchPredicate | None = None,
    ) -> StagedNearbySearchResult:
        """Search at increasing radii and stop when a match condition succeeds.

        ``radii_m`` defaults to 50, 100, 250, and 500 metres and must be
        strictly increasing.  ``strong_match`` receives all de-duplicated
        candidates accumulated so far and the current radius.  This lets the
        analysis layer apply behavioral and distance evidence rather than
        making the API client equate "closest" with "correct".  Without a
        callback, the transport-level default stops at the first radius that
        returns at least one candidate.
        """

        radii = tuple(_normalize_radius(radius) for radius in radii_m)
        if not radii:
            raise ValueError("radii_m must contain at least one radius")
        if any(right <= left for left, right in zip(radii, radii[1:])):
            raise ValueError("radii_m must be strictly increasing")

        stages: list[NearbySearchStage] = []
        candidates: list[dict[str, Any]] = []
        seen_candidates: set[str] = set()
        stopped_radius: int | None = None
        stop_reason = "radii_exhausted"

        for radius in radii:
            response = self.search_nearby(
                latitude,
                longitude,
                radius,
                max_results=max_results,
                included_types=included_types,
                rank_preference=rank_preference,
            )
            stage_places = response.payload.get("places", [])
            for place in stage_places:
                if not isinstance(place, dict):
                    continue
                candidate_key = _place_candidate_key(place)
                if candidate_key not in seen_candidates:
                    seen_candidates.add(candidate_key)
                    candidates.append(place)

            if strong_match is None:
                should_stop = bool(stage_places)
                matched_reason = "candidates_found"
            else:
                should_stop = bool(strong_match(tuple(candidates), radius))
                matched_reason = "strong_match"
            stages.append(
                NearbySearchStage(
                    radius_m=radius,
                    response=response,
                    place_count=len(stage_places),
                    stop_condition_met=should_stop,
                )
            )
            if should_stop:
                stopped_radius = radius
                stop_reason = matched_reason
                break

        return StagedNearbySearchResult(
            source=self.PLACES_SOURCE,
            stages=tuple(stages),
            candidates=tuple(candidates),
            stopped_radius_m=stopped_radius,
            stop_reason=stop_reason,
        )

    def nearby_search(
        self,
        latitude: float,
        longitude: float,
        radius_m: int | float,
        **kwargs: Any,
    ) -> GoogleAPIResponse:
        """Verb-order alias for :meth:`search_nearby`."""

        return self.search_nearby(latitude, longitude, radius_m, **kwargs)

    def staged_nearby_search(
        self,
        latitude: float,
        longitude: float,
        *,
        radii: Sequence[int | float] = DEFAULT_SEARCH_RADII_M,
        **kwargs: Any,
    ) -> StagedNearbySearchResult:
        """Alias accepting ``radii=`` for :meth:`search_nearby_staged`."""

        return self.search_nearby_staged(
            latitude,
            longitude,
            radii_m=radii,
            **kwargs,
        )

    def _cached_request(
        self,
        *,
        source: str,
        api: str,
        request_identity: Mapping[str, Any],
        method: str,
        endpoint: str,
        params_factory: Callable[[str], dict[str, Any]] | None,
        json_body: Mapping[str, Any] | None,
        headers_factory: Callable[[str], dict[str, str]] | None,
    ) -> GoogleAPIResponse:
        cache_key = SuccessfulResponseCache.make_key(source, request_identity)
        try:
            cached = self._cache.get_with_metadata(source, cache_key)
        except CacheReadError:
            with self._stats_lock:
                self._stats.cache_read_errors += 1
            cached = None

        if cached is not None and _is_success_payload(api, cached[0]):
            with self._stats_lock:
                self._stats.cache_hits += 1
                self._stats.source_result_counts[source] += 1
            return GoogleAPIResponse(
                source=source,
                payload=cached[0],
                cache_hit=True,
                http_status=200,
                retrieved_at_utc=cached[1],
            )

        if not self._allow_network:
            error = GoogleAPIError(
                category=GoogleAPIErrorCategory.CACHE_MISS,
                api=api,
                message=(
                    f"No cached Google {api} response is available and network "
                    "requests are disabled for this run."
                ),
                retryable=False,
            )
            self._record_error(error, response_failed=False)
            raise error

        api_key = self._read_api_key(api)
        params = params_factory(api_key) if params_factory is not None else None
        headers = headers_factory(api_key) if headers_factory is not None else None

        response = self._request_with_retries(
            api=api,
            method=method,
            endpoint=endpoint,
            params=params,
            json_body=json_body,
            headers=headers,
            api_key=api_key,
        )
        payload = _sanitize_response_payload(api, response.payload, api_key)
        retrieved_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            self._cache.put(
                source,
                cache_key,
                payload,
                retrieved_at_utc=retrieved_at_utc,
            )
        except CacheWriteError:
            with self._stats_lock:
                self._stats.cache_write_errors += 1

        with self._stats_lock:
            self._stats.source_result_counts[source] += 1
        return GoogleAPIResponse(
            source=source,
            payload=payload,
            cache_hit=False,
            http_status=response.http_status,
            retrieved_at_utc=retrieved_at_utc,
        )

    def _read_api_key(self, api: str) -> str:
        value = os.environ.get(GOOGLE_MAPS_API_KEY_ENV)
        if value is None or not value.strip():
            error = MissingGoogleMapsAPIKey(
                category=GoogleAPIErrorCategory.MISSING_KEY,
                api=api,
                message=(
                    f"{GOOGLE_MAPS_API_KEY_ENV} is not set; Google {api} "
                    "requests are unavailable."
                ),
                retryable=False,
            )
            self._record_error(error, response_failed=False)
            raise error
        return value.strip()

    def _request_with_retries(
        self,
        *,
        api: str,
        method: str,
        endpoint: str,
        params: Mapping[str, Any] | None,
        json_body: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
        api_key: str,
    ) -> "_HTTPPayload":
        last_error: GoogleAPIError | None = None
        for attempt in range(self._max_retries + 1):
            retry_after: float | None = None
            self._reserve_request(api)
            try:
                response = self._session.request(
                    method,
                    endpoint,
                    params=params,
                    json=json_body,
                    headers=headers,
                    timeout=self._timeout_seconds,
                )
            except requests.Timeout:
                error = GoogleAPIError(
                    category=GoogleAPIErrorCategory.TIMEOUT,
                    api=api,
                    message=f"Google {api} request timed out.",
                    retryable=True,
                )
                self._record_error(error, response_failed=True)
            except requests.RequestException as exc:
                error = GoogleAPIError(
                    category=GoogleAPIErrorCategory.NETWORK,
                    api=api,
                    message=(
                        f"Network error while calling Google {api} "
                        f"({type(exc).__name__})."
                    ),
                    retryable=True,
                )
                self._record_error(error, response_failed=True)
            except (TimeoutError, ConnectionError, OSError) as exc:
                error = GoogleAPIError(
                    category=GoogleAPIErrorCategory.NETWORK,
                    api=api,
                    message=(
                        f"Network error while calling Google {api} "
                        f"({type(exc).__name__})."
                    ),
                    retryable=True,
                )
                self._record_error(error, response_failed=True)
            else:
                http_status = _safe_http_status(response)
                payload = _read_json_object(response)
                if payload is not None and _is_success_response(
                    api, http_status, payload
                ):
                    with self._stats_lock:
                        self._stats.successful_google_responses += 1
                    return _HTTPPayload(payload=payload, http_status=http_status)

                error = _error_from_response(
                    api, http_status, payload, api_key=api_key
                )
                self._record_error(error, response_failed=True)
                if error.retryable:
                    retry_after = _safe_retry_after_seconds(response)
                else:
                    retry_after = None

            last_error = error
            if not error.retryable or attempt >= self._max_retries:
                raise error
            with self._stats_lock:
                self._stats.retries += 1
            self._sleep(
                retry_after
                if retry_after is not None
                else min(8.0, 0.5 * (2**attempt))
            )

        # The loop always returns or raises, but retain a safe defensive guard.
        if last_error is not None:
            raise last_error
        raise GoogleAPIError(
            category=GoogleAPIErrorCategory.UNKNOWN,
            api=api,
            message=f"Google {api} request failed for an unknown reason.",
        )

    def _reserve_request(self, api: str) -> None:
        """Rate-limit and reserve one budget unit immediately before transport."""

        with self._request_start_lock:
            with self._stats_lock:
                if self._stats.google_requests >= self._request_budget:
                    error = GoogleRequestBudgetExceeded(
                        category=GoogleAPIErrorCategory.BUDGET_EXHAUSTED,
                        api=api,
                        message=(
                            "Google API request budget exhausted; no request was sent."
                        ),
                        retryable=False,
                    )
                    self._stats.error_categories[error.category.value] += 1
                    raise error

            now = self._clock()
            if self._last_request_started is not None:
                wait_seconds = self._min_interval_seconds - (
                    now - self._last_request_started
                )
                if wait_seconds > 0:
                    self._sleep(wait_seconds)
                    now = self._clock()

            with self._stats_lock:
                # The start lock ensures the count cannot change between checks.
                self._stats.google_requests += 1
                if api == "geocoding":
                    self._stats.geocoding_requests += 1
                elif api == "places":
                    self._stats.places_requests += 1
            self._last_request_started = now

    def _record_error(
        self, error: GoogleAPIError, *, response_failed: bool
    ) -> None:
        with self._stats_lock:
            self._stats.error_categories[error.category.value] += 1
            if response_failed:
                self._stats.failed_google_responses += 1


# The alias keeps the client easy to discover by either service name.
GooglePlacesClient = GoogleMapsClient


@dataclass(frozen=True)
class _HTTPPayload:
    payload: dict[str, Any]
    http_status: int


def google_maps_api_key_detected() -> bool:
    """Return only whether ``GOOGLE_MAPS_API_KEY`` is non-empty."""

    value = os.environ.get(GOOGLE_MAPS_API_KEY_ENV)
    return bool(value and value.strip())


def _normalize_coordinates(latitude: float, longitude: float) -> tuple[float, float]:
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        raise ValueError("latitude and longitude must be numeric") from None
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise ValueError("latitude and longitude must be finite")
    if not -90.0 <= lat <= 90.0:
        raise ValueError("latitude must be between -90 and 90")
    if not -180.0 <= lon <= 180.0:
        raise ValueError("longitude must be between -180 and 180")
    # Seven decimals is sub-decimetre precision and stabilizes equivalent keys.
    lat = round(lat, 7)
    lon = round(lon, 7)
    return (0.0 if lat == -0.0 else lat, 0.0 if lon == -0.0 else lon)


def _normalize_radius(radius_m: int | float) -> int:
    try:
        radius_float = float(radius_m)
    except (TypeError, ValueError):
        raise ValueError("radius_m must be numeric") from None
    if not math.isfinite(radius_float) or not radius_float.is_integer():
        raise ValueError("radius_m must be a whole number of metres")
    radius = int(radius_float)
    if not 1 <= radius <= 50_000:
        raise ValueError("radius_m must be between 1 and 50000 metres")
    return radius


def _normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = str(language).strip()
    if not normalized:
        return None
    if len(normalized) > 35 or not re.fullmatch(r"[A-Za-z0-9_-]+", normalized):
        raise ValueError("language contains unsupported characters")
    return normalized


def _normalize_place_types(types: Sequence[str] | None) -> tuple[str, ...]:
    if types is None:
        return ()
    if isinstance(types, (str, bytes)):
        raise TypeError("included_types must be a sequence of place type strings")
    normalized: set[str] = set()
    for value in types:
        place_type = str(value).strip().lower()
        if not place_type or not re.fullmatch(r"[a-z0-9_]+", place_type):
            raise ValueError(f"invalid Places type: {value!r}")
        normalized.add(place_type)
    if len(normalized) > 50:
        raise ValueError("included_types may contain at most 50 unique values")
    return tuple(sorted(normalized))


def _place_candidate_key(place: Mapping[str, Any]) -> str:
    display_name = place.get("displayName")
    if isinstance(display_name, dict):
        display_name = display_name.get("text")
    location = place.get("location")
    if not isinstance(location, dict):
        location = {}
    identity = {
        "name": display_name,
        "address": place.get("formattedAddress"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
    }
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_http_status(response: Any) -> int:
    try:
        return int(response.status_code)
    except (AttributeError, TypeError, ValueError):
        return 0


def _read_json_object(response: Any) -> dict[str, Any] | None:
    try:
        payload = response.json()
    except (ValueError, TypeError, AttributeError):
        return None
    return payload if isinstance(payload, dict) else None


def _is_success_response(
    api: str, http_status: int, payload: Mapping[str, Any]
) -> bool:
    if not 200 <= http_status < 300:
        return False
    return _is_success_payload(api, payload)


def _is_success_payload(api: str, payload: Mapping[str, Any]) -> bool:
    if api == "geocoding":
        return payload.get("status") in {"OK", "ZERO_RESULTS"}
    if api == "places":
        places = payload.get("places", [])
        return "error" not in payload and isinstance(places, list)
    return False


def _error_from_response(
    api: str,
    http_status: int,
    payload: Mapping[str, Any] | None,
    *,
    api_key: str,
) -> GoogleAPIError:
    if payload is None:
        category = (
            GoogleAPIErrorCategory.SERVER
            if http_status in _RETRYABLE_HTTP_STATUSES
            else GoogleAPIErrorCategory.MALFORMED_RESPONSE
        )
        return GoogleAPIError(
            category=category,
            api=api,
            http_status=http_status or None,
            message=(
                f"Google {api} returned an unreadable JSON response "
                f"(HTTP {http_status or 'unknown'})."
            ),
            retryable=http_status in _RETRYABLE_HTTP_STATUSES,
        )

    api_status: str | None = None
    message = ""
    error_object = payload.get("error")
    if isinstance(error_object, dict):
        raw_status = error_object.get("status")
        if raw_status is not None:
            api_status = str(raw_status)
        raw_message = error_object.get("message")
        if raw_message is not None:
            message = str(raw_message)
    else:
        raw_status = payload.get("status")
        if raw_status is not None:
            api_status = str(raw_status)
        raw_message = payload.get("error_message")
        if raw_message is not None:
            message = str(raw_message)

    # Google normally does not echo a credential, but guarantee that even an
    # unusual proxy/test response cannot place the environment secret in the
    # returned exception.
    if api_key:
        message = message.replace(api_key, "[REDACTED]")

    category = _categorize_error(http_status, api_status, message)
    if not message:
        message = (
            f"Google {api} request failed "
            f"(HTTP {http_status or 'unknown'}, status {api_status or 'unknown'})."
        )
    retryable = (
        http_status in _RETRYABLE_HTTP_STATUSES
        or (api_status or "").upper() in {"UNKNOWN_ERROR", "INTERNAL", "UNAVAILABLE"}
    )
    return GoogleAPIError(
        category=category,
        api=api,
        http_status=http_status or None,
        api_status=api_status,
        message=message,
        retryable=retryable,
    )


def _categorize_error(
    http_status: int, api_status: str | None, message: str
) -> GoogleAPIErrorCategory:
    status = (api_status or "").upper()
    lowered = message.lower()
    if "billing" in lowered:
        return GoogleAPIErrorCategory.BILLING
    if any(
        phrase in lowered
        for phrase in (
            "api has not been used",
            "api is not enabled",
            "api is disabled",
            "enable it by visiting",
            "access not configured",
        )
    ):
        return GoogleAPIErrorCategory.API_DISABLED
    if status in {"OVER_DAILY_LIMIT", "OVER_QUERY_LIMIT", "RESOURCE_EXHAUSTED"}:
        return GoogleAPIErrorCategory.QUOTA
    if "quota" in lowered or "daily limit" in lowered:
        return GoogleAPIErrorCategory.QUOTA
    if http_status == 429:
        return GoogleAPIErrorCategory.RATE_LIMIT
    if http_status == 401 or status in {"UNAUTHENTICATED"}:
        return GoogleAPIErrorCategory.AUTHORIZATION
    if http_status == 403 or status in {"PERMISSION_DENIED", "REQUEST_DENIED"}:
        return GoogleAPIErrorCategory.AUTHORIZATION
    if http_status == 400 or status in {"INVALID_ARGUMENT", "INVALID_REQUEST"}:
        return GoogleAPIErrorCategory.INVALID_REQUEST
    if http_status >= 500 or status in {"UNKNOWN_ERROR", "INTERNAL", "UNAVAILABLE"}:
        return GoogleAPIErrorCategory.SERVER
    return GoogleAPIErrorCategory.UNKNOWN


def _safe_retry_after_seconds(response: Any) -> float | None:
    """Read, but never retain, a bounded Retry-After response header."""

    try:
        raw_value = response.headers.get("Retry-After")
        value = float(raw_value)
    except (AttributeError, TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return min(60.0, value)


def _sanitize_api_status(status: str | None) -> str | None:
    if status is None:
        return None
    sanitized = _SAFE_STATUS_RE.sub("_", str(status).upper()).strip("_")
    return sanitized[:80] or None


def _sanitize_error_message(message: str) -> str:
    """Remove URLs and credential-like values from a bounded error message."""

    text = str(message)
    text = _GOOGLE_KEY_RE.sub("[REDACTED]", text)
    text = _KEY_ASSIGNMENT_RE.sub(r"\g<label>[REDACTED]", text)
    text = _URL_RE.sub("[URL omitted]", text)
    text = " ".join(text.split())
    return text[:500] if text else "Google API request failed."


def _sanitize_response_payload(
    api: str, payload: Mapping[str, Any], api_key: str
) -> dict[str, Any]:
    """Copy a successful response while defensively removing transport secrets."""

    if api == "places":
        raw_places = payload.get("places", [])
        places: list[dict[str, Any]] = []
        for raw_place in raw_places:
            if not isinstance(raw_place, dict):
                continue
            filtered = {
                key: value
                for key, value in raw_place.items()
                if key in PLACES_RESPONSE_FIELDS
            }
            places.append(_redact_secrets(filtered, api_key))
        return {"places": places}
    sanitized = _redact_secrets(dict(payload), api_key)
    return sanitized if isinstance(sanitized, dict) else {}


def _redact_secrets(value: Any, api_key: str) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if str(key).lower() in _SENSITIVE_MAPPING_KEYS:
                continue
            result[str(key)] = _redact_secrets(child, api_key)
        return result
    if isinstance(value, list):
        return [_redact_secrets(child, api_key) for child in value]
    if isinstance(value, tuple):
        return [_redact_secrets(child, api_key) for child in value]
    if isinstance(value, str):
        sanitized = value.replace(api_key, "[REDACTED]") if api_key else value
        sanitized = _GOOGLE_KEY_RE.sub("[REDACTED]", sanitized)
        sanitized = _KEY_ASSIGNMENT_RE.sub(r"\g<label>[REDACTED]", sanitized)
        return sanitized
    return value


__all__ = [
    "DEFAULT_SEARCH_RADII_M",
    "GEOCODING_ENDPOINT",
    "GOOGLE_MAPS_API_KEY_ENV",
    "GoogleAPIError",
    "GoogleAPIErrorCategory",
    "GoogleAPIResponse",
    "GoogleMapsClient",
    "GooglePlacesClient",
    "GoogleRequestBudgetExceeded",
    "MAX_BILLABLE_REQUESTS",
    "MissingGoogleMapsAPIKey",
    "NearbySearchStage",
    "PLACES_FIELD_MASK",
    "PLACES_NEARBY_ENDPOINT",
    "RequestStats",
    "StagedNearbySearchResult",
    "SuccessfulResponseCache",
    "google_maps_api_key_detected",
]
