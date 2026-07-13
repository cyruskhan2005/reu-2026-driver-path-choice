#!/usr/bin/env python3
"""Build the self-contained, privacy-reviewed Driver 1003 public package."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC_DIR = ROOT / "outputs" / "public"
REPORT_SOURCE = (
    ROOT
    / "deliverables"
    / "driver_1003"
    / "route_choice_change_index"
    / "visuals"
    / "driver_1003_route_choice_change_index_report.html"
)
MAP_SOURCE = ROOT / "outputs" / "driver_1003_poi_route_insights_map.html"
JSON_SOURCE = ROOT / "outputs" / "driver_1003_real_world_behavior_insights.json"
REPORT_NAME = "driver_1003_route_choice_change_index_report.html"
MAP_NAME = "driver_1003_poi_route_insights_map.html"
JSON_NAME = "driver_1003_real_world_behavior_insights.json"

_REFERENCE = re.compile(
    r"\s+(?P<attribute>href|src)=(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
    re.IGNORECASE,
)


def make_report_self_contained(document: str) -> str:
    """Keep safe external links and the local map; remove escaping local links."""

    def replace(match: re.Match[str]) -> str:
        attribute = match.group("attribute")
        quote = match.group("quote")
        value = match.group("value").strip()
        lowered = value.casefold()
        if value.startswith("#") or lowered.startswith(
            ("https://", "mailto:", "tel:", "data:")
        ):
            return match.group(0)
        if Path(value.split("#", 1)[0]).name == MAP_NAME:
            return f" {attribute}={quote}{MAP_NAME}{quote}"
        return ""

    result = _REFERENCE.sub(replace, document)
    if MAP_NAME not in result:
        raise ValueError("Packaged report lost its verification-map reference")
    return result


def package(public_dir: Path = DEFAULT_PUBLIC_DIR) -> list[Path]:
    sources = (REPORT_SOURCE, MAP_SOURCE, JSON_SOURCE)
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required generated Driver 1003 artifacts are missing")
    public_dir.mkdir(parents=True, exist_ok=True)
    report_target = public_dir / REPORT_NAME
    report_target.write_text(
        make_report_self_contained(REPORT_SOURCE.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    map_target = public_dir / MAP_NAME
    json_target = public_dir / JSON_NAME
    map_document = MAP_SOURCE.read_text(encoding="utf-8")
    map_target.write_text(
        "\n".join(line.rstrip() for line in map_document.splitlines()) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(JSON_SOURCE, json_target)
    return [report_target, map_target, json_target]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC_DIR)
    args = parser.parse_args()
    packaged = package(args.public_dir.resolve())
    print(f"Packaged {len(packaged)} Driver 1003 public artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
