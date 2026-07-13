"""Offline unit tests for public-release hygiene tooling."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "scripts" / filename
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses and other runtime helpers expect the module to be registered.
    import sys

    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


manifest_tool = _load_script(
    "generate_output_manifest_tested", "generate_output_manifest.py"
)
validator = _load_script("validate_public_release_tested", "validate_public_release.py")
packager = _load_script(
    "package_driver_1003_public_release_tested",
    "package_driver_1003_public_release.py",
)


class ManifestTests(unittest.TestCase):
    def test_manifest_is_deterministic_and_excludes_itself(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "report.json").write_text('{"status":"public"}\n', encoding="utf-8")
            output = root / "manifest.json"
            first = manifest_tool.serialize_manifest(
                manifest_tool.build_manifest(
                    root,
                    manifest_path=output,
                    source_revision="a" * 40,
                )
            )
            manifest_tool.write_manifest(output, first)
            second = manifest_tool.serialize_manifest(
                manifest_tool.build_manifest(
                    root,
                    manifest_path=output,
                    source_revision="a" * 40,
                )
            )
            self.assertEqual(first, second)
            payload = json.loads(first)
            self.assertEqual([entry["path"] for entry in payload["files"]], ["report.json"])

    def test_check_preserves_recorded_artifact_source_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "report.json").write_text("{}\n", encoding="utf-8")
            output = root / "manifest.json"
            revision = "c" * 40
            document = manifest_tool.serialize_manifest(
                manifest_tool.build_manifest(
                    root, manifest_path=output, source_revision=revision
                )
            )
            manifest_tool.write_manifest(output, document)
            self.assertEqual(
                manifest_tool.main(
                    ["--root", str(root), "--output", str(output), "--check"]
                ),
                0,
            )

    def test_manifest_and_local_links_validate_offline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "map.html").write_text(
                '<!doctype html><html><body id="map">map</body></html>',
                encoding="utf-8",
            )
            (root / "report.html").write_text(
                '<!doctype html><html><body><a href="map.html#map">map</a></body></html>',
                encoding="utf-8",
            )
            output = root / "manifest.json"
            document = manifest_tool.serialize_manifest(
                manifest_tool.build_manifest(
                    root,
                    manifest_path=output,
                    source_revision="b" * 40,
                )
            )
            manifest_tool.write_manifest(output, document)
            issues, count = validator.validate_public_tree(
                root,
                manifest_path=output,
                require_manifest=True,
            )
            self.assertEqual(count, 2)
            self.assertEqual([issue for issue in issues if issue.severity == "error"], [])


class ValidationTests(unittest.TestCase):
    def test_secret_is_reported_without_echoing_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected = "AIza" + "A" * 32
            (root / "report.html").write_text(protected, encoding="utf-8")
            issues, _ = validator.validate_public_tree(root)
            rendered = json.dumps([issue.to_dict() for issue in issues])
            self.assertIn("google_api_key", rendered)
            self.assertNotIn(protected, rendered)

    def test_forbidden_coordinate_pair_uses_label_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = root.parent / (root.name + "_private.json")
            values.write_text(
                json.dumps({"exact_home_coordinates": [26.1234567, -80.7654321]}),
                encoding="utf-8",
            )
            (root / "report.html").write_text(
                "<html><body>[26.1234567, -80.7654321]</body></html>",
                encoding="utf-8",
            )
            try:
                issues, _ = validator.validate_public_tree(
                    root, forbidden_values_path=values
                )
            finally:
                values.unlink(missing_ok=True)
            rendered = json.dumps([issue.to_dict() for issue in issues])
            self.assertIn("forbidden_private_value", rendered)
            self.assertIn("exact_home_coordinates", rendered)
            self.assertNotIn("26.1234567", rendered)
            self.assertNotIn("-80.7654321", rendered)

    def test_private_filename_and_escaping_link_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "driver_1003_trip_summary.csv").write_text(
                "trip_id\n1\n", encoding="utf-8"
            )
            (root / "report.html").write_text(
                '<html><body><a href="../private-map.html">map</a></body></html>',
                encoding="utf-8",
            )
            issues, _ = validator.validate_public_tree(root)
            codes = {issue.code for issue in issues}
            self.assertIn("private_artifact_name", codes)
            self.assertIn("reference_outside_public_root", codes)


class PackagingTests(unittest.TestCase):
    def test_report_packaging_keeps_map_and_safe_external_links_only(self) -> None:
        source = """<html><body>
        <a href="../../private/report.html">private</a>
        <a href="https://example.com/place">place</a>
        <a href="#section">section</a>
        <a href="../../outputs/driver_1003_poi_route_insights_map.html">map</a>
        <div id="section"></div></body></html>"""
        result = packager.make_report_self_contained(source)
        self.assertNotIn("../../private/report.html", result)
        self.assertIn('href="https://example.com/place"', result)
        self.assertIn('href="#section"', result)
        self.assertIn(
            'href="driver_1003_poi_route_insights_map.html"', result
        )


if __name__ == "__main__":
    unittest.main()
