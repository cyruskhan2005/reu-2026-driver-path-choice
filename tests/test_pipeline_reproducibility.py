"""Regression tests for the macOS/FMM reproducibility configuration layer."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main, mock

import geopandas as gpd

from roadnet.cli.run import load_config, main as run_main
from roadnet.config import CountyConfig, PipelineConfig, WGS84
from roadnet.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
PINNED_FMM_SHA = "19ef34e1f57ff2f2484231aa0d01dfffea986ec1"


class FMMConfigurationTests(TestCase):
    def test_top_level_fmm_bin_is_inherited_and_county_override_wins(self) -> None:
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                f"""
output_dir: {tmp}/outputs
mapillary_enabled: false
fmm_bin: /shared/bin/fmm
counties:
  - name: Shared County
    place_query: Shared County, Florida, USA
  - name: Override County
    place_query: Override County, Florida, USA
    fmm_bin: /county/bin/fmm
""",
                encoding="utf-8",
            )
            cfg = load_config(str(config_path))

        self.assertEqual(cfg.fmm_bin, "/shared/bin/fmm")
        self.assertEqual(cfg.fmm_bin_for(cfg.counties[0]), "/shared/bin/fmm")
        self.assertEqual(cfg.fmm_bin_for(cfg.counties[1]), "/county/bin/fmm")

    def test_missing_fmm_configuration_preserves_path_resolved_default(self) -> None:
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                f"""
output_dir: {tmp}/outputs
mapillary_enabled: false
counties:
  - name: Default County
    place_query: Default County, Florida, USA
""",
                encoding="utf-8",
            )
            cfg = load_config(str(config_path))

        self.assertEqual(cfg.fmm_bin, "fmm")
        self.assertEqual(cfg.counties[0].fmm_bin, "fmm")

    def test_programmatic_config_uses_shared_default_but_preserves_explicit_fmm(self) -> None:
        inherited = CountyConfig(name="Inherited", place_query="Inherited")
        explicit = CountyConfig(
            name="Explicit",
            place_query="Explicit",
            fmm_bin="fmm",
        )
        cfg = PipelineConfig(
            output_dir=Path("outputs"),
            mly_token="",
            fdot_gdb=None,
            counties=[inherited, explicit],
            fmm_bin="/shared/bin/fmm",
            mapillary_enabled=False,
        )

        self.assertEqual(cfg.fmm_bin_for(inherited), "/shared/bin/fmm")
        self.assertEqual(cfg.fmm_bin_for(explicit), "fmm")


class MapillaryDisabledModeTests(TestCase):
    def _pipeline(self, tmp: str, **overrides) -> tuple[Pipeline, CountyConfig]:
        county = CountyConfig(name="Test County", place_query="Test County")
        values = {
            "output_dir": Path(tmp) / "outputs",
            "mly_token": "",
            "fdot_gdb": None,
            "counties": [county],
            "mapillary_enabled": False,
        }
        values.update(overrides)
        return Pipeline(PipelineConfig(**values)), county

    def test_disabled_mode_makes_no_mapillary_calls_and_returns_edge_copy(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline, county = self._pipeline(tmp)
            edges = gpd.GeoDataFrame(
                {"u": [1], "v": [2]},
                geometry=gpd.GeoSeries.from_wkt(["LINESTRING (0 0, 1 1)"], crs=WGS84),
                crs=WGS84,
            )
            empty = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=WGS84), crs=WGS84)
            with (
                mock.patch("roadnet.pipeline.fetch_signs") as fetch,
                mock.patch("roadnet.pipeline.attach_signs_to_edges") as attach,
            ):
                result = pipeline._enrich_with_mapillary(
                    county=county,
                    nodes=empty,
                    edges=edges,
                    landuse=empty,
                    bounds=(-80.2, 25.7, -80.1, 25.8),
                    out_dir=Path(tmp),
                )

        fetch.assert_not_called()
        attach.assert_not_called()
        self.assertIsNot(result, edges)
        self.assertTrue(result.equals(edges))

    def test_enabled_mode_rejects_empty_token_before_any_request(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline, county = self._pipeline(tmp, mapillary_enabled=True)
            empty = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=WGS84), crs=WGS84)
            with mock.patch("roadnet.pipeline.fetch_signs") as fetch:
                with self.assertRaisesRegex(ValueError, "mly_token is empty"):
                    pipeline._enrich_with_mapillary(
                        county=county,
                        nodes=empty,
                        edges=empty,
                        landuse=empty,
                        bounds=(-80.2, 25.7, -80.1, 25.8),
                        out_dir=Path(tmp),
                    )
        fetch.assert_not_called()

    def test_skip_mly_retains_cache_reuse_semantics_when_enabled(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline, county = self._pipeline(
                tmp,
                mapillary_enabled=True,
                mly_token="test-token",
                skip_mly=True,
                skip_conflation=True,
            )
            empty = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=WGS84), crs=WGS84)
            expected = empty.copy()
            with (
                mock.patch("roadnet.pipeline.fetch_signs", return_value=empty) as fetch,
                mock.patch(
                    "roadnet.pipeline.attach_signs_to_edges", return_value=expected
                ) as attach,
            ):
                result = pipeline._enrich_with_mapillary(
                    county=county,
                    nodes=empty,
                    edges=empty,
                    landuse=empty,
                    bounds=(-80.2, 25.7, -80.1, 25.8),
                    out_dir=Path(tmp),
                )

            self.assertIs(result, expected)
            self.assertTrue(fetch.call_args.kwargs["skip_if_exists"])
            self.assertTrue(attach.call_args.kwargs["skip_if_exists"])

    def test_cli_disable_flag_overrides_enabled_yaml(self) -> None:
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                f"""
output_dir: {tmp}/outputs
mapillary_enabled: true
mly_token: placeholder
counties: []
""",
                encoding="utf-8",
            )
            argv = ["roadnet-run", str(config_path), "--disable-mapillary"]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("roadnet.cli.run.Pipeline") as pipeline_class,
            ):
                run_main()

        configured = pipeline_class.call_args.args[0]
        self.assertFalse(configured.mapillary_enabled)
        pipeline_class.return_value.run.assert_called_once_with(counties=None)


class ReproducibilityArtifactTests(TestCase):
    def test_shell_scripts_have_help_and_strict_mode(self) -> None:
        for relative in (
            "scripts/build_fmm_macos.sh",
            "scripts/bootstrap_macos.sh",
            "scripts/run_full_pipeline_macos.sh",
            "scripts/run_fmm_only_macos.sh",
            "scripts/generate_driver_1003_report.sh",
        ):
            script = ROOT / relative
            source = script.read_text(encoding="utf-8")
            self.assertIn("set -euo pipefail", source)
            completed = subprocess.run(
                ["bash", str(script), "--help"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Usage:", completed.stdout)

    def test_build_artifacts_pin_and_validate_native_linkage(self) -> None:
        build_script = (ROOT / "scripts/build_fmm_macos.sh").read_text(encoding="utf-8")
        patch = (ROOT / "patches/fmm-macos-arm64.patch").read_text(encoding="utf-8")

        self.assertIn(f'FMM_SHA="{PINNED_FMM_SHA}"', build_script)
        self.assertIn("LINKER:-undefined,dynamic_lookup", patch)
        self.assertIn("@loader_path/libFMMLIB.dylib", build_script)
        self.assertIn("unexpectedly links libpython", build_script)
        self.assertIn("CMAKE_OSX_ARCHITECTURES=arm64", build_script)
        self.assertIn("while IFS= read -r rpath", build_script)
        self.assertNotIn("for rpath in $(", build_script)

    def test_wrappers_pin_report_roots_and_complete_public_packaging(self) -> None:
        report = (ROOT / "scripts/generate_driver_1003_report.sh").read_text(
            encoding="utf-8"
        )
        fmm_only = (ROOT / "scripts/run_fmm_only_macos.sh").read_text(
            encoding="utf-8"
        )
        for option in ("--graph-root", "--manifest", "--output-dir"):
            self.assertIn(option, report)
        self.assertIn('CACHE_DIR="${REPO_ROOT}/cache/google_maps"', report)
        self.assertIn("package_driver_1003_public_release.py", report)
        self.assertIn("--require-manifest", report)
        self.assertIn("for suffix in shp shx dbf prj cpg", fmm_only)

    def test_example_config_is_credential_free_and_mapillary_disabled(self) -> None:
        cfg = load_config(str(ROOT / "config.example.yaml"))
        self.assertFalse(cfg.mapillary_enabled)
        self.assertEqual(cfg.mly_token, "")

    def test_macos_environment_uses_documented_pipeline_name(self) -> None:
        environment = (ROOT / "environment-macos.yml").read_text(encoding="utf-8")
        self.assertIn("name: pipeline", environment)
        bootstrap = (ROOT / "scripts/bootstrap_macos.sh").read_text(encoding="utf-8")
        self.assertIn('ENV_NAME="pipeline"', bootstrap)

    def test_output_verifier_compiles_and_has_stage_help(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/verify_pipeline_outputs.py"), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--stage", completed.stdout)


if __name__ == "__main__":
    main()
