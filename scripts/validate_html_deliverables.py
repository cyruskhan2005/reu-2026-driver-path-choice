#!/usr/bin/env python3
"""Validate local references in generated HTML deliverables."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


LOCAL_REF_ATTRS = {
    "a": ("href",),
    "link": ("href",),
    "script": ("src",),
    "img": ("src",),
    "iframe": ("src",),
    "source": ("src", "srcset"),
}


@dataclass(frozen=True)
class HtmlIssue:
    html_path: Path
    message: str


class HtmlReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[tuple[str, str, str]] = []
        self.ids: set[str] = set()
        self.has_styles = False
        self.has_leaflet_map = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name.lower(): value or "" for name, value in attrs}
        if "id" in attr_map:
            self.ids.add(attr_map["id"])
        if tag == "style":
            self.has_styles = True
        if tag == "div" and attr_map.get("id", "").startswith("map"):
            self.has_leaflet_map = True
        for attr in LOCAL_REF_ATTRS.get(tag, ()):
            value = attr_map.get(attr)
            if value:
                self.references.append((tag, attr, value))

    def handle_data(self, data: str) -> None:
        if "L.map(" in data:
            self.has_leaflet_map = True


def _is_external(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {
        "http",
        "https",
        "mailto",
        "tel",
        "data",
        "javascript",
    }


def _local_path_and_fragment(value: str) -> tuple[Path | None, str | None]:
    value = value.strip()
    if not value or _is_external(value):
        return None, None
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc:
        return None, parsed.fragment or None
    if parsed.path:
        return Path(unquote(parsed.path)), parsed.fragment or None
    return None, parsed.fragment or None


def _srcset_paths(value: str) -> list[str]:
    paths: list[str] = []
    for candidate in value.split(","):
        first = candidate.strip().split(" ", 1)[0]
        if first:
            paths.append(first)
    return paths


def _parse_html(path: Path) -> HtmlReferenceParser:
    parser = HtmlReferenceParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return parser


def validate_html(root: Path) -> list[HtmlIssue]:
    issues: list[HtmlIssue] = []
    html_paths = sorted(root.rglob("*.html"))
    if not html_paths:
        return [HtmlIssue(root, "No HTML files found")]

    parsed_cache: dict[Path, HtmlReferenceParser] = {}
    root_resolved = root.resolve()
    for html_path in html_paths:
        parser = _parse_html(html_path)
        parsed_cache[html_path.resolve()] = parser
        if not parser.has_styles:
            issues.append(HtmlIssue(html_path, "No inline <style> block found"))
        if "map" in html_path.name.lower() and not parser.has_leaflet_map:
            issues.append(HtmlIssue(html_path, "Map-like filename has no detected map container"))

        for tag, attr, value in parser.references:
            values = _srcset_paths(value) if attr == "srcset" else [value]
            for raw_ref in values:
                local_path, fragment = _local_path_and_fragment(raw_ref)
                if local_path is None:
                    if raw_ref.strip().startswith("#"):
                        anchor = raw_ref.strip()[1:]
                        if anchor and anchor not in parser.ids:
                            issues.append(
                                HtmlIssue(html_path, f"Broken internal anchor #{anchor}")
                            )
                    continue
                candidate = (html_path.parent / local_path).resolve()
                if not candidate.exists():
                    issues.append(
                        HtmlIssue(html_path, f"Missing local {tag}[{attr}] target: {raw_ref}")
                    )
                    continue
                if fragment and candidate.suffix.lower() == ".html":
                    target_parser = parsed_cache.get(candidate)
                    if target_parser is None:
                        target_parser = _parse_html(candidate)
                        parsed_cache[candidate] = target_parser
                    if fragment not in target_parser.ids:
                        try:
                            display = candidate.relative_to(root_resolved)
                        except ValueError:
                            display = candidate
                        issues.append(
                            HtmlIssue(
                                html_path,
                                f"Missing anchor #{fragment} in {display}",
                            )
                        )
    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate local links/assets in generated HTML deliverables."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="deliverables/driver_1003",
        help="Deliverable root to scan",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    issues = validate_html(root)
    html_count = len(list(root.rglob("*.html"))) if root.exists() else 0
    print(f"Validated {html_count} HTML files under {root}")
    if not issues:
        print("No local-reference issues found")
        return 0
    print("Issues:")
    for issue in issues:
        print(f"  {issue.html_path}: {issue.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
