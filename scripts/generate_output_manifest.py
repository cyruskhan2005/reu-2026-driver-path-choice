#!/usr/bin/env python3
"""Generate or verify a deterministic manifest for curated public artifacts.

The manifest contains only repository-relative artifact paths, byte sizes,
SHA-256 digests, and media types.  It deliberately omits timestamps, host
paths, usernames, environment values, and source-data details.  Re-running the
tool over unchanged files at the same source revision produces identical JSON.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Sequence


SCHEMA_VERSION = 1
DEFAULT_PUBLIC_ROOT = Path("outputs/public")
DEFAULT_MANIFEST_NAME = "manifest.json"
DEFAULT_EXCLUDES = (".DS_Store", "Thumbs.db")
_STABLE_MEDIA_TYPES = {
    ".csv": "text/csv",
    ".html": "text/html",
    ".htm": "text/html",
    ".json": "application/json",
    ".md": "text/markdown",
}


class ManifestError(RuntimeError):
    """Raised when a public artifact tree cannot be manifested safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _media_type(path: Path) -> str:
    """Return a platform-independent media type for release manifests."""

    return _STABLE_MEDIA_TYPES.get(
        path.suffix.casefold(),
        mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    )


def _source_revision(repository_root: Path) -> str:
    """Return the current Git object ID without exposing remote information."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    revision = result.stdout.strip().lower()
    if len(revision) == 40 and all(character in "0123456789abcdef" for character in revision):
        return revision
    return "unknown"


def _is_excluded(relative_path: str, patterns: Sequence[str]) -> bool:
    name = Path(relative_path).name
    return any(
        fnmatch.fnmatchcase(relative_path, pattern)
        or fnmatch.fnmatchcase(name, pattern)
        for pattern in patterns
    )


def iter_artifact_paths(
    root: Path,
    *,
    manifest_path: Path | None = None,
    excludes: Sequence[str] = DEFAULT_EXCLUDES,
) -> list[Path]:
    """Return sorted regular files and reject symlinks in the public tree."""

    root = root.resolve()
    if not root.exists():
        raise ManifestError("Public artifact directory does not exist")
    if not root.is_dir():
        raise ManifestError("Public artifact root is not a directory")

    manifest_resolved = manifest_path.resolve() if manifest_path is not None else None
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ManifestError(
                "Public artifact trees may not contain symlinks: "
                + path.relative_to(root).as_posix()
            )
        if not path.is_file():
            continue
        if manifest_resolved is not None and path.resolve() == manifest_resolved:
            continue
        relative = path.relative_to(root).as_posix()
        if _is_excluded(relative, excludes):
            continue
        files.append(path)
    return files


def build_manifest(
    root: Path,
    *,
    manifest_path: Path | None = None,
    source_revision: str | None = None,
    repository_root: Path = Path("."),
    excludes: Sequence[str] = DEFAULT_EXCLUDES,
    allow_empty: bool = False,
) -> dict[str, object]:
    """Build a deterministic JSON-compatible public artifact manifest."""

    root = root.resolve()
    files = iter_artifact_paths(
        root,
        manifest_path=manifest_path,
        excludes=excludes,
    )
    if not files and not allow_empty:
        raise ManifestError("Public artifact directory contains no releasable files")

    entries: list[dict[str, object]] = []
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        media_type = _media_type(path)
        entries.append(
            {
                "path": relative,
                "size_bytes": size,
                "sha256": _sha256(path),
                "media_type": media_type,
                "classification": "public-reviewed",
            }
        )
        total_bytes += size

    revision = source_revision or _source_revision(repository_root.resolve())
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_policy": "public-reviewed",
        "source_revision": revision,
        "generator": "scripts/generate_output_manifest.py",
        "files": entries,
        "summary": {
            "file_count": len(entries),
            "total_bytes": total_bytes,
        },
    }


def serialize_manifest(manifest: dict[str, object]) -> str:
    """Serialize with stable key and list ordering."""

    return json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def write_manifest(path: Path, document: str) -> None:
    """Atomically write a manifest without embedding temporary host paths."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(document, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_PUBLIC_ROOT,
        help="Curated public artifact directory (default: outputs/public)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Manifest path (default: <root>/manifest.json)",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
        help="Repository used only to determine the source Git revision",
    )
    parser.add_argument(
        "--source-revision",
        default=None,
        help="Explicit source revision for reproducible exported builds",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional filename or relative-path glob to exclude (repeatable)",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Permit an empty public artifact directory",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that the existing manifest is current without writing",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root
    output = args.output or root / DEFAULT_MANIFEST_NAME
    excludes = tuple(DEFAULT_EXCLUDES) + tuple(args.exclude)
    try:
        source_revision = args.source_revision
        if args.check and source_revision is None and output.exists():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                recorded = existing.get("source_revision")
                if isinstance(recorded, str):
                    source_revision = recorded
        manifest = build_manifest(
            root,
            manifest_path=output,
            source_revision=source_revision,
            repository_root=args.repository_root,
            excludes=excludes,
            allow_empty=args.allow_empty,
        )
        document = serialize_manifest(manifest)
        if args.check:
            if not output.exists():
                raise ManifestError(f"Manifest does not exist: {output}")
            if output.read_text(encoding="utf-8") != document:
                raise ManifestError("Manifest is stale; regenerate it before release")
            print(
                f"Manifest is current: {len(manifest['files'])} public artifacts"
            )
            return 0
        write_manifest(output, document)
        print(f"Wrote manifest for {len(manifest['files'])} public artifacts: {output}")
        return 0
    except (ManifestError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
