"""Deterministic membership-stability diagnostics for endpoint clusters.

The main clustering pipeline selects one radius and assigns public cluster IDs.
This module audits those selected memberships against nearby candidate radii
without changing the selected clustering.  A clustering implementation is
injected so the helper remains independent of DBSCAN, HDBSCAN, or any specific
third-party package.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
import json
import math

import numpy as np
import pandas as pd


Clusterer = Callable[..., Sequence[object]]


CLUSTER_STABILITY_COLUMNS = (
    "cluster_id",
    "selected_radius_m",
    "min_samples",
    "selected_endpoint_count",
    "important_endpoint_threshold",
    "is_important_cluster",
    "candidate_radii_json",
    "comparison_radius_count",
    "stability_status",
    "stable_radius_count",
    "split_radius_count",
    "merged_radius_count",
    "noise_radius_count",
    "minimum_dominant_retention",
    "mean_dominant_retention",
    "minimum_best_jaccard",
    "mean_best_jaccard",
    "maximum_noise_share",
    "maximum_merge_contamination_share",
    "maximum_component_count",
    "stability_evidence_json",
    "data_quality_flags",
)


class ClusterStabilityError(ValueError):
    """Raised when cluster-stability inputs cannot be reconciled safely."""


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _is_noise_label(value: object) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, np.integer)) and int(value) == -1:
        return True
    if isinstance(value, (float, np.floating)) and float(value) == -1.0:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in {
            "",
            "-1",
            "<na>",
            "nan",
            "noise",
            "none",
            "null",
            "unclustered",
        }
    return False


def _hashable_label(value: object, *, radius_m: float) -> Hashable:
    try:
        hash(value)
    except TypeError as exc:
        raise ClusterStabilityError(
            f"Clusterer returned an unhashable label at radius {radius_m:g} m"
        ) from exc
    return value  # type: ignore[return-value]


def _label_sort_key(value: object) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _label_json_value(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _cluster_display(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _finite_positive(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ClusterStabilityError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result <= 0:
        raise ClusterStabilityError(f"{label} must be finite and positive")
    return result


def _safe_mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _aggregate_status(*, split: bool, merged: bool, noise: bool) -> str:
    behaviors: list[str] = []
    if split:
        behaviors.append("split")
    if merged:
        behaviors.append("merged")
    if noise:
        behaviors.append("noise")
    return "+".join(behaviors) if behaviors else "stable"


def analyze_cluster_stability(
    endpoints: pd.DataFrame,
    *,
    candidate_radii_m: Sequence[float],
    selected_radius_m: float,
    min_samples: int,
    clusterer: Clusterer,
    important_endpoint_threshold: int = 20,
    x_column: str = "x",
    y_column: str = "y",
    cluster_column: str = "cluster_id",
) -> pd.DataFrame:
    """Compare selected endpoint-cluster memberships at candidate radii.

    Parameters
    ----------
    endpoints:
        One row per endpoint with projected ``x``/``y`` coordinates and the
        authoritative cluster ID selected by the main analysis.  Noise rows
        such as ``UNCLUSTERED`` or ``-1`` influence alternate memberships but
        do not receive their own output row.
    candidate_radii_m:
        Candidate radii to audit.  The selected radius is excluded from the
        alternate comparisons because ``cluster_column`` is authoritative.
    selected_radius_m:
        Radius used to produce the supplied selected memberships.
    min_samples:
        Minimum-sample argument passed unchanged to ``clusterer``.
    clusterer:
        Callable compatible with
        ``clusterer(x, y, eps_m=<radius>, min_samples=<value>)``.  It must
        return one label per endpoint; ``-1`` denotes noise.
    important_endpoint_threshold:
        Marks important clusters without filtering smaller clusters from the
        returned audit table.

    Returns
    -------
    pandas.DataFrame
        Exactly one deterministic row for every non-noise selected cluster.
        ``stability_evidence_json`` contains radius-level membership evidence;
        summary fields expose retention, overlap, split, merge, and noise
        sensitivity directly.
    """

    required = {x_column, y_column, cluster_column}
    missing = sorted(required - set(endpoints.columns))
    if missing:
        raise ClusterStabilityError(
            f"Endpoint table is missing required columns: {missing}"
        )
    if not isinstance(min_samples, (int, np.integer)) or int(min_samples) <= 0:
        raise ClusterStabilityError("min_samples must be a positive integer")
    if (
        not isinstance(important_endpoint_threshold, (int, np.integer))
        or int(important_endpoint_threshold) <= 0
    ):
        raise ClusterStabilityError(
            "important_endpoint_threshold must be a positive integer"
        )

    selected_radius = _finite_positive(
        selected_radius_m, label="selected_radius_m"
    )
    radii = sorted(
        {
            _finite_positive(radius, label="candidate radius")
            for radius in candidate_radii_m
        }
    )
    comparison_radii = [
        radius
        for radius in radii
        if not math.isclose(radius, selected_radius, rel_tol=0.0, abs_tol=1e-9)
    ]

    work = endpoints[[x_column, y_column, cluster_column]].copy().reset_index(
        drop=True
    )
    x = pd.to_numeric(work[x_column], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(work[y_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ClusterStabilityError("Endpoint coordinates must be finite numbers")

    selected_values = work[cluster_column].tolist()
    selected_members: dict[Hashable, set[int]] = {}
    selected_display: dict[Hashable, str] = {}
    display_owner: dict[str, Hashable] = {}
    endpoint_selected_label: list[Hashable | None] = []
    for index, value in enumerate(selected_values):
        if _is_noise_label(value):
            endpoint_selected_label.append(None)
            continue
        label = _hashable_label(value, radius_m=selected_radius)
        display = _cluster_display(value)
        owner = display_owner.get(display)
        if owner is not None and owner != label:
            raise ClusterStabilityError(
                "Distinct selected labels have the same string representation"
            )
        display_owner[display] = label
        selected_display[label] = display
        selected_members.setdefault(label, set()).add(index)
        endpoint_selected_label.append(label)

    alternate_labels: dict[float, tuple[Hashable | None, ...]] = {}
    for radius in comparison_radii:
        try:
            raw_labels = list(
                clusterer(
                    x.copy(),
                    y.copy(),
                    eps_m=float(radius),
                    min_samples=int(min_samples),
                )
            )
        except ClusterStabilityError:
            raise
        except Exception as exc:
            raise ClusterStabilityError(
                f"Clusterer failed at radius {radius:g} m"
            ) from exc
        if len(raw_labels) != len(work):
            raise ClusterStabilityError(
                f"Clusterer returned {len(raw_labels)} labels for {len(work)} "
                f"endpoints at radius {radius:g} m"
            )
        normalized: list[Hashable | None] = []
        for value in raw_labels:
            normalized.append(
                None
                if _is_noise_label(value)
                else _hashable_label(value, radius_m=radius)
            )
        alternate_labels[radius] = tuple(normalized)

    rows: list[dict[str, object]] = []
    ordered_clusters = sorted(
        selected_members,
        key=lambda label: (selected_display[label], _label_sort_key(label)),
    )
    selected_radius_listed = any(
        math.isclose(radius, selected_radius, rel_tol=0.0, abs_tol=1e-9)
        for radius in radii
    )

    for selected_label in ordered_clusters:
        members = selected_members[selected_label]
        selected_count = len(members)
        comparisons: list[dict[str, object]] = []
        retention_values: list[float] = []
        jaccard_values: list[float] = []
        noise_values: list[float] = []
        contamination_values: list[float] = []
        component_counts: list[int] = []
        stable_count = 0
        split_count = 0
        merged_count = 0
        noise_count = 0

        for radius in comparison_radii:
            labels = alternate_labels[radius]
            alternative_members: dict[Hashable, set[int]] = {}
            for endpoint_index, alternative_label in enumerate(labels):
                if alternative_label is not None:
                    alternative_members.setdefault(alternative_label, set()).add(
                        endpoint_index
                    )

            component_records: list[dict[str, object]] = []
            for alternative_label in sorted(
                alternative_members, key=_label_sort_key
            ):
                alternative_set = alternative_members[alternative_label]
                intersection = members & alternative_set
                if not intersection:
                    continue
                outside = alternative_set - members
                union = members | alternative_set
                foreign_selected = sorted(
                    {
                        selected_display[value]
                        for endpoint_index in outside
                        if (value := endpoint_selected_label[endpoint_index])
                        is not None
                        and value != selected_label
                    }
                )
                component_records.append(
                    {
                        "alternative_label": _label_json_value(alternative_label),
                        "intersection_count": len(intersection),
                        "retention": len(intersection) / selected_count,
                        "jaccard": len(intersection) / len(union),
                        "alternative_cluster_count": len(alternative_set),
                        "outside_selected_cluster_count": len(outside),
                        "merge_contamination_share": (
                            len(outside) / len(alternative_set)
                        ),
                        "foreign_selected_clusters": foreign_selected,
                    }
                )

            component_records.sort(
                key=lambda record: (
                    -int(record["intersection_count"]),
                    -float(record["jaccard"]),
                    _label_sort_key(record["alternative_label"]),
                )
            )
            selected_noise_count = sum(
                labels[endpoint_index] is None for endpoint_index in members
            )
            selected_noise_share = selected_noise_count / selected_count
            component_count = len(component_records)
            observed_split = component_count > 1
            observed_merge = any(
                int(record["outside_selected_cluster_count"]) > 0
                for record in component_records
            )
            observed_noise = selected_noise_count > 0
            radius_status = _aggregate_status(
                split=observed_split,
                merged=observed_merge,
                noise=observed_noise,
            )
            if radius_status == "stable":
                stable_count += 1
            if observed_split:
                split_count += 1
            if observed_merge:
                merged_count += 1
            if observed_noise:
                noise_count += 1

            dominant_retention = (
                float(component_records[0]["retention"])
                if component_records
                else 0.0
            )
            best_jaccard = max(
                (float(record["jaccard"]) for record in component_records),
                default=0.0,
            )
            merge_contamination = max(
                (
                    float(record["merge_contamination_share"])
                    for record in component_records
                ),
                default=0.0,
            )
            retention_values.append(dominant_retention)
            jaccard_values.append(best_jaccard)
            noise_values.append(selected_noise_share)
            contamination_values.append(merge_contamination)
            component_counts.append(component_count)
            comparisons.append(
                {
                    "radius_m": radius,
                    "status": radius_status,
                    "selected_endpoint_count": selected_count,
                    "component_count": component_count,
                    "dominant_retention": dominant_retention,
                    "best_jaccard": best_jaccard,
                    "noise_count": selected_noise_count,
                    "noise_share": selected_noise_share,
                    "maximum_merge_contamination_share": merge_contamination,
                    "components": component_records,
                }
            )

        observed_split = split_count > 0
        observed_merge = merged_count > 0
        observed_noise = noise_count > 0
        stability_status = (
            _aggregate_status(
                split=observed_split,
                merged=observed_merge,
                noise=observed_noise,
            )
            if comparison_radii
            else "not_evaluated"
        )
        minimum_retention = min(retention_values, default=float("nan"))
        minimum_jaccard = min(jaccard_values, default=float("nan"))
        maximum_noise = max(noise_values, default=0.0)
        maximum_contamination = max(contamination_values, default=0.0)

        flags: list[str] = []
        if not selected_radius_listed:
            flags.append("selected_radius_not_in_candidate_list")
        if not comparison_radii:
            flags.append("no_alternative_radii")
        elif not (
            any(radius < selected_radius for radius in comparison_radii)
            and any(radius > selected_radius for radius in comparison_radii)
        ):
            flags.append("single_sided_radius_evaluation")
        if selected_count < int(min_samples):
            flags.append("selected_cluster_smaller_than_min_samples")
        if selected_count < int(important_endpoint_threshold):
            flags.append("below_important_endpoint_threshold")
        if observed_split:
            flags.append("split_sensitive")
        if observed_merge:
            flags.append("merge_sensitive")
        if observed_noise:
            flags.append("noise_sensitive")
        if retention_values and minimum_retention < 0.8:
            flags.append("low_member_retention")
        if jaccard_values and minimum_jaccard < 0.5:
            flags.append("low_membership_overlap")

        evidence = {
            "selected_radius_m": selected_radius,
            "candidate_radii_m": radii,
            "comparison_radii_m": comparison_radii,
            "comparisons": comparisons,
        }
        rows.append(
            {
                "cluster_id": selected_display[selected_label],
                "selected_radius_m": selected_radius,
                "min_samples": int(min_samples),
                "selected_endpoint_count": selected_count,
                "important_endpoint_threshold": int(
                    important_endpoint_threshold
                ),
                "is_important_cluster": selected_count
                >= int(important_endpoint_threshold),
                "candidate_radii_json": _json(radii),
                "comparison_radius_count": len(comparison_radii),
                "stability_status": stability_status,
                "stable_radius_count": stable_count,
                "split_radius_count": split_count,
                "merged_radius_count": merged_count,
                "noise_radius_count": noise_count,
                "minimum_dominant_retention": minimum_retention,
                "mean_dominant_retention": _safe_mean(retention_values),
                "minimum_best_jaccard": minimum_jaccard,
                "mean_best_jaccard": _safe_mean(jaccard_values),
                "maximum_noise_share": maximum_noise,
                "maximum_merge_contamination_share": maximum_contamination,
                "maximum_component_count": max(component_counts, default=0),
                "stability_evidence_json": _json(evidence),
                "data_quality_flags": _json(flags),
            }
        )

    return pd.DataFrame(rows, columns=CLUSTER_STABILITY_COLUMNS)


__all__ = [
    "CLUSTER_STABILITY_COLUMNS",
    "ClusterStabilityError",
    "analyze_cluster_stability",
]
