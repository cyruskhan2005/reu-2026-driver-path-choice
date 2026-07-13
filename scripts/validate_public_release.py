#!/usr/bin/env python3
"""Offline validation for a curated public artifact directory.

The validator never calls a network service.  It checks the publication
boundary, manifest hashes, local links, common credential signatures, known
private artifact names, and structured home-privacy fields.  Exact private
values can be supplied in a local ignored JSON file; findings report labels and
paths only, never the protected values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlparse
import zipfile


DEFAULT_PUBLIC_ROOT = Path("outputs/public")
DEFAULT_MANIFEST_NAME = "manifest.json"
MAX_TEXT_SCAN_BYTES = 64 * 1024 * 1024
MANIFEST_SCHEMA_VERSION = 1

PRIVATE_FILENAME_PARTS = (
    "driver_1003_trip_summary",
    "driver_1003_location_clusters",
    "driver_1003_poi_enriched_clusters",
    "driver_1003_recurring_poi_patterns",
    "driver_1003_od_route_change_insights",
    "raw_gps",
    "matched_gps",
    "google_cache",
    "api_cache",
    "exact_home",
    "home_sensitive",
    ".inspect.",
)
PRIVATE_SUFFIXES = {
    ".env",
    ".key",
    ".pem",
    ".p12",
    ".pfx",
    ".parquet",
    ".jsonl",
    ".ndjson",
    ".gpkg",
    ".shp",
    ".shx",
    ".dbf",
}
TEXT_SUFFIXES = {
    ".css",
    ".csv",
    ".geojson",
    ".htm",
    ".html",
    ".js",
    ".json",
    ".md",
    ".svg",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
OFFICE_SUFFIXES = {".docx", ".pptx", ".xlsx"}
VISUAL_REVIEW_SUFFIXES = {
    ".gif",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".pptx",
    ".svg",
    ".webp",
}
HOME_CONTEXT_KEYS = {
    "likely_home",
    "likely_home_area",
    "home_candidate",
    "home_area",
}
FORBIDDEN_HOME_KEYS = {
    "address",
    "exact_address",
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
    "exact_latitude",
    "exact_longitude",
}
SECRET_PATTERNS = (
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{20,}")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[0-9A-Za-z]{20,}")),
    (
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "api_key_query_parameter",
        re.compile(
            r"(?:[?&]|&amp;)(?:key|api_key|apikey|google_maps_api_key)\s*=",
            re.IGNORECASE,
        ),
    ),
    (
        "credential_transport_field",
        re.compile(r"(?:GOOGLE_MAPS_API_KEY|X-Goog-Api-Key)", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class ValidationIssue:
    """A value-free release finding suitable for CI output."""

    severity: str
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }


@dataclass(frozen=True)
class ForbiddenRule:
    """Protected value groups; each group requires one alternative to match."""

    label: str
    groups: tuple[tuple[str, ...], ...]

    def matches(self, normalized_document: str) -> bool:
        return bool(self.groups) and all(
            any(alternative in normalized_document for alternative in group)
            for group in self.groups
        )


class ReferenceParser(HTMLParser):
    """Collect local references and element IDs without executing HTML."""

    REFERENCE_ATTRIBUTES = {
        "a": ("href",),
        "iframe": ("src",),
        "img": ("src",),
        "link": ("href",),
        "script": ("src",),
        "source": ("src", "srcset"),
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[tuple[str, str, str]] = []
        self.ids: set[str] = set()
        self.has_embedded_data_image = False

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = {name.casefold(): value or "" for name, value in attrs}
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        for attribute in self.REFERENCE_ATTRIBUTES.get(tag.casefold(), ()):
            value = attributes.get(attribute, "").strip()
            if value:
                self.references.append((tag.casefold(), attribute, value))
                if value.casefold().startswith("data:image/"):
                    self.has_embedded_data_image = True


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.name or "."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_present(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().casefold() not in {
            "nan",
            "none",
            "null",
        }
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _coordinate_alternatives(value: Any) -> tuple[str, ...]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ()
    if not math.isfinite(number):
        return ()
    alternatives = {str(number).casefold()}
    for digits in range(5, 11):
        fixed = f"{number:.{digits}f}"
        alternatives.add(fixed.casefold())
        alternatives.add(fixed.rstrip("0").rstrip(".").casefold())
    return tuple(sorted(item for item in alternatives if item))


def _safe_rule_label(label: str) -> str:
    """Map caller labels to a fixed vocabulary so labels cannot leak values."""

    normalized = re.sub(r"[^a-z0-9_]+", "_", label.casefold()).strip("_")
    canonical_labels = (
        "exact_home_coordinates",
        "exact_home_address",
        "exact_home_uri",
        "api_key",
        "google_maps_api_key",
        "private_coordinate",
        "private_address",
        "private_uri",
    )
    for canonical in canonical_labels:
        if canonical in normalized:
            return canonical
    return "forbidden_value"


def _forbidden_rules_from_mapping(
    value: Mapping[str, Any], prefix: str = ""
) -> list[ForbiddenRule]:
    rules: list[ForbiddenRule] = []
    for key, item in value.items():
        label = f"{prefix}.{key}" if prefix else str(key)
        normalized_label = re.sub(r"[^a-z0-9_.-]+", "_", label.casefold())
        safe_label = _safe_rule_label(normalized_label)
        if isinstance(item, Mapping):
            rules.extend(_forbidden_rules_from_mapping(item, label))
            continue
        if (
            "coordinate" in normalized_label
            and isinstance(item, Sequence)
            and not isinstance(item, (str, bytes))
            and len(item) == 2
        ):
            groups = tuple(
                group for group in (_coordinate_alternatives(part) for part in item) if group
            )
            if len(groups) == 2:
                rules.append(ForbiddenRule(safe_label, groups))
            continue
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for index, part in enumerate(item):
                if _is_present(part):
                    rules.append(
                        ForbiddenRule(
                            safe_label,
                            ((unescape(str(part)).casefold(),),),
                        )
                    )
            continue
        if _is_present(item):
            text = unescape(str(item)).strip().casefold()
            if text:
                rules.append(ForbiddenRule(safe_label, ((text,),)))
    return rules


def load_forbidden_rules(path: Path | None) -> list[ForbiddenRule]:
    """Load local private validation values without returning their content."""

    rules: list[ForbiddenRule] = []
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Forbidden-values file must contain a JSON object")
        rules.extend(_forbidden_rules_from_mapping(payload))
    environment_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    if len(environment_key.strip()) >= 8:
        rules.append(
            ForbiddenRule(
                "google_maps_api_key_environment",
                ((environment_key.strip().casefold(),),),
            )
        )
    return rules


def _read_scannable_text(path: Path) -> str | None:
    """Read text or Office XML without invoking document applications."""

    suffix = path.suffix.casefold()
    if suffix in TEXT_SUFFIXES:
        if path.stat().st_size > MAX_TEXT_SCAN_BYTES:
            raise ValueError("Text artifact exceeds the configured scan-size limit")
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix in OFFICE_SUFFIXES:
        chunks: list[str] = []
        total = 0
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if not name.casefold().endswith((".xml", ".rels", ".json", ".txt")):
                    continue
                data = archive.read(name)
                total += len(data)
                if total > MAX_TEXT_SCAN_BYTES:
                    raise ValueError("Office artifact exceeds the configured scan-size limit")
                chunks.append(data.decode("utf-8", errors="replace"))
        return "\n".join(chunks)
    return None


def _json_privacy_issues(
    value: Any,
    *,
    path: str,
    home_context: bool = False,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if isinstance(value, Mapping):
        privacy_flag = str(value.get("privacy_flag", "")).casefold()
        local_home_context = home_context or "home_sensitive" in privacy_flag
        for key, child in value.items():
            normalized_key = str(key).casefold()
            child_home_context = local_home_context or normalized_key in HOME_CONTEXT_KEYS
            if (
                child_home_context
                and normalized_key in FORBIDDEN_HOME_KEYS
                and _is_present(child)
            ):
                issues.append(
                    ValidationIssue(
                        "error",
                        "precise_home_field",
                        path,
                        "A public JSON home object contains a precise address, coordinate, or map-link field.",
                    )
                )
            issues.extend(
                _json_privacy_issues(
                    child,
                    path=path,
                    home_context=child_home_context,
                )
            )
    elif isinstance(value, list):
        for child in value:
            issues.extend(
                _json_privacy_issues(child, path=path, home_context=home_context)
            )
    return issues


def _reference_issues(path: Path, root: Path, document: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    relative = _display_path(path, root)
    parser = ReferenceParser()
    try:
        parser.feed(document)
    except Exception:
        return [
            ValidationIssue(
                "error", "invalid_html", relative, "HTML parsing failed."
            )
        ]
    if parser.has_embedded_data_image:
        issues.append(
            ValidationIssue(
                "warning",
                "manual_visual_review_required",
                relative,
                "Embedded images require a manual privacy review.",
            )
        )

    root_resolved = root.resolve()
    for tag, attribute, raw_reference in parser.references:
        references = (
            [part.strip().split(" ", 1)[0] for part in raw_reference.split(",")]
            if attribute == "srcset"
            else [raw_reference.strip()]
        )
        for reference in references:
            if not reference:
                continue
            if reference.startswith("#"):
                anchor = unquote(reference[1:])
                if anchor and anchor not in parser.ids:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "broken_internal_anchor",
                            relative,
                            "An internal HTML anchor has no matching target.",
                        )
                    )
                continue
            parsed = urlparse(reference)
            scheme = parsed.scheme.casefold()
            if scheme in {"https", "mailto", "tel", "data"}:
                continue
            if scheme or parsed.netloc:
                issues.append(
                    ValidationIssue(
                        "error",
                        "unsafe_or_insecure_reference",
                        relative,
                        f"A {tag}[{attribute}] reference uses a disallowed URI scheme.",
                    )
                )
                continue
            local_part = Path(unquote(parsed.path)) if parsed.path else path.name
            candidate = (path.parent / local_part).resolve()
            try:
                candidate.relative_to(root_resolved)
            except ValueError:
                issues.append(
                    ValidationIssue(
                        "error",
                        "reference_outside_public_root",
                        relative,
                        "A local HTML reference escapes the curated public directory.",
                    )
                )
                continue
            if not candidate.exists():
                issues.append(
                    ValidationIssue(
                        "error",
                        "missing_local_reference",
                        relative,
                        "A local HTML reference target is missing.",
                    )
                )
                continue
            if parsed.fragment and candidate.suffix.casefold() in {".html", ".htm"}:
                target = ReferenceParser()
                target.feed(candidate.read_text(encoding="utf-8", errors="replace"))
                if unquote(parsed.fragment) not in target.ids:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "broken_linked_anchor",
                            relative,
                            "A linked HTML fragment has no matching target.",
                        )
                    )
    return issues


def _verify_manifest(
    root: Path, manifest_path: Path, artifact_paths: Sequence[Path]
) -> list[ValidationIssue]:
    relative_manifest = _display_path(manifest_path, root)
    issues: list[ValidationIssue] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return [
            ValidationIssue(
                "error", "invalid_manifest", relative_manifest, "Manifest JSON is unreadable."
            )
        ]
    if not isinstance(manifest, Mapping):
        return [
            ValidationIssue(
                "error", "invalid_manifest", relative_manifest, "Manifest root must be an object."
            )
        ]
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        issues.append(
            ValidationIssue(
                "error",
                "manifest_schema",
                relative_manifest,
                "Manifest schema_version is unsupported.",
            )
        )
    if manifest.get("artifact_policy") != "public-reviewed":
        issues.append(
            ValidationIssue(
                "error",
                "manifest_policy",
                relative_manifest,
                "Manifest artifact_policy must be public-reviewed.",
            )
        )
    revision = manifest.get("source_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        issues.append(
            ValidationIssue(
                "error",
                "manifest_source_revision",
                relative_manifest,
                "Manifest source_revision must be a full lowercase Git object ID.",
            )
        )
    entries = manifest.get("files")
    if not isinstance(entries, list):
        issues.append(
            ValidationIssue(
                "error", "manifest_files", relative_manifest, "Manifest files must be a list."
            )
        )
        return issues

    actual = {
        path.relative_to(root).as_posix(): path
        for path in artifact_paths
        if path.resolve() != manifest_path.resolve()
    }
    listed: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            issues.append(
                ValidationIssue(
                    "error", "manifest_entry", relative_manifest, "Manifest contains an invalid file entry."
                )
            )
            continue
        relative = entry["path"]
        pure = Path(relative)
        if pure.is_absolute() or ".." in pure.parts or relative in listed:
            issues.append(
                ValidationIssue(
                    "error", "manifest_path", relative_manifest, "Manifest contains an unsafe or duplicate path."
                )
            )
            continue
        listed.add(relative)
        target = actual.get(relative)
        if target is None:
            issues.append(
                ValidationIssue(
                    "error", "manifest_missing_file", relative_manifest, "Manifest lists a file not present in the public directory."
                )
            )
            continue
        if entry.get("size_bytes") != target.stat().st_size:
            issues.append(
                ValidationIssue(
                    "error", "manifest_size_mismatch", relative, "Manifest byte size does not match the artifact."
                )
            )
        if entry.get("sha256") != _sha256(target):
            issues.append(
                ValidationIssue(
                    "error", "manifest_hash_mismatch", relative, "Manifest SHA-256 does not match the artifact."
                )
            )
        expected_media = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if entry.get("media_type") != expected_media:
            issues.append(
                ValidationIssue(
                    "error", "manifest_media_type", relative, "Manifest media type does not match the artifact name."
                )
            )
        if entry.get("classification") != "public-reviewed":
            issues.append(
                ValidationIssue(
                    "error", "manifest_classification", relative, "Manifest artifact classification is not public-reviewed."
                )
            )
    for relative in sorted(set(actual) - listed):
        issues.append(
            ValidationIssue(
                "error", "manifest_unlisted_file", relative, "Public artifact is missing from the manifest."
            )
        )
    summary = manifest.get("summary")
    expected_total = sum(path.stat().st_size for path in actual.values())
    if not isinstance(summary, Mapping):
        issues.append(
            ValidationIssue(
                "error", "manifest_summary", relative_manifest, "Manifest summary is missing or invalid."
            )
        )
    else:
        if summary.get("file_count") != len(actual):
            issues.append(
                ValidationIssue(
                    "error", "manifest_file_count", relative_manifest, "Manifest file count does not match the public directory."
                )
            )
        if summary.get("total_bytes") != expected_total:
            issues.append(
                ValidationIssue(
                    "error", "manifest_total_bytes", relative_manifest, "Manifest total byte count does not match the public directory."
                )
            )
    return issues


def validate_public_tree(
    root: Path,
    *,
    manifest_path: Path | None = None,
    require_manifest: bool = False,
    allow_empty: bool = False,
    forbidden_values_path: Path | None = None,
    require_private_validation: bool = False,
) -> tuple[list[ValidationIssue], int]:
    """Validate a public tree and return findings plus artifact count."""

    root = root.resolve()
    if not root.exists():
        if allow_empty:
            return [], 0
        return [
            ValidationIssue(
                "error", "missing_public_root", ".", "Curated public directory does not exist."
            )
        ], 0
    if not root.is_dir():
        return [
            ValidationIssue(
                "error", "invalid_public_root", ".", "Curated public path is not a directory."
            )
        ], 0

    issues: list[ValidationIssue] = []
    try:
        rules = load_forbidden_rules(forbidden_values_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return [
            ValidationIssue(
                "error", "invalid_forbidden_values", ".", "Private validation-values file is unreadable or invalid."
            )
        ], 0
    file_rules = rules
    if require_private_validation and forbidden_values_path is None:
        issues.append(
            ValidationIssue(
                "error",
                "private_validation_required",
                ".",
                "A local forbidden-values file is required for release validation.",
            )
        )

    manifest = (manifest_path or root / DEFAULT_MANIFEST_NAME).resolve()
    try:
        manifest.relative_to(root)
    except ValueError:
        issues.append(
            ValidationIssue(
                "error",
                "manifest_outside_public_root",
                manifest.name,
                "Manifest must be stored inside the curated public directory.",
            )
        )
    paths: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = _display_path(path, root)
        if path.is_symlink():
            issues.append(
                ValidationIssue(
                    "error", "symlink_not_allowed", relative, "Public directories may not contain symlinks."
                )
            )
            continue
        if path.is_file():
            paths.append(path)

    artifacts = [path for path in paths if path.resolve() != manifest]
    if not artifacts and not allow_empty:
        issues.append(
            ValidationIssue(
                "error", "empty_public_root", ".", "Curated public directory contains no artifacts."
            )
        )

    for path in paths:
        relative = _display_path(path, root)
        lower_name = relative.casefold()
        is_manifest = path.resolve() == manifest
        if not is_manifest and any(part in lower_name for part in PRIVATE_FILENAME_PARTS):
            issues.append(
                ValidationIssue(
                    "error", "private_artifact_name", relative, "Artifact name is reserved for private or restricted output."
                )
            )
        if not is_manifest and (
            path.suffix.casefold() in PRIVATE_SUFFIXES
            or path.name.casefold().startswith(".env")
            or path.name.startswith(".")
        ):
            issues.append(
                ValidationIssue(
                    "error", "private_artifact_type", relative, "Artifact type is not allowed in the public directory."
                )
            )
        if not is_manifest and path.suffix.casefold() in VISUAL_REVIEW_SUFFIXES:
            issues.append(
                ValidationIssue(
                    "warning", "manual_visual_review_required", relative, "Visual artifact requires documented manual privacy review."
                )
            )
        try:
            document = _read_scannable_text(path)
        except (OSError, UnicodeError, ValueError, zipfile.BadZipFile):
            issues.append(
                ValidationIssue(
                    "error", "artifact_scan_failed", relative, "Artifact could not be scanned safely."
                )
            )
            continue
        if document is None:
            continue
        normalized = unescape(document).casefold()
        for code, pattern in SECRET_PATTERNS:
            if pattern.search(document):
                issues.append(
                    ValidationIssue(
                        "error", code, relative, "Artifact contains credential-like material."
                    )
                )
        for rule in file_rules:
            if rule.matches(normalized):
                issues.append(
                    ValidationIssue(
                        "error",
                        "forbidden_private_value",
                        relative,
                        f"Artifact contains protected validation value labeled {rule.label}.",
                    )
                )
        if path.suffix.casefold() == ".json":
            try:
                payload = json.loads(document)
            except json.JSONDecodeError:
                issues.append(
                    ValidationIssue(
                        "error", "invalid_json", relative, "Public JSON artifact is invalid."
                    )
                )
            else:
                issues.extend(_json_privacy_issues(payload, path=relative))
        if path.suffix.casefold() in {".html", ".htm"}:
            issues.extend(_reference_issues(path, root, document))

    if manifest.exists():
        issues.extend(_verify_manifest(root, manifest, paths))
    elif require_manifest and artifacts:
        issues.append(
            ValidationIssue(
                "error", "missing_manifest", DEFAULT_MANIFEST_NAME, "Public artifact manifest is required."
            )
        )
    return issues, len(artifacts)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public-dir",
        type=Path,
        default=DEFAULT_PUBLIC_ROOT,
        help="Curated public artifact directory (default: outputs/public)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest path (default: <public-dir>/manifest.json)",
    )
    parser.add_argument(
        "--require-manifest",
        action="store_true",
        help="Fail when a nonempty public directory has no manifest",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow a missing or empty public directory for pre-release CI",
    )
    parser.add_argument(
        "--forbidden-values-file",
        type=Path,
        default=None,
        help="Ignored local JSON file containing exact values that must not appear",
    )
    parser.add_argument(
        "--require-private-validation",
        action="store_true",
        help="Require --forbidden-values-file for a release-candidate run",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Treat manual-review warnings as failures",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON result containing only codes, paths, and safe messages",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    issues, artifact_count = validate_public_tree(
        args.public_dir,
        manifest_path=args.manifest,
        require_manifest=args.require_manifest,
        allow_empty=args.allow_empty,
        forbidden_values_path=args.forbidden_values_file,
        require_private_validation=args.require_private_validation,
    )
    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    failed = bool(errors) or (args.fail_on_warnings and bool(warnings))
    if args.json:
        print(
            json.dumps(
                {
                    "status": "failed" if failed else "passed",
                    "artifact_count": artifact_count,
                    "error_count": len(errors),
                    "warning_count": len(warnings),
                    "issues": [issue.to_dict() for issue in issues],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            f"Public release validation: {artifact_count} artifacts, "
            f"{len(errors)} errors, {len(warnings)} warnings"
        )
        for issue in issues:
            print(
                f"{issue.severity.upper()} {issue.code} [{issue.path}]: "
                f"{issue.message}"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
