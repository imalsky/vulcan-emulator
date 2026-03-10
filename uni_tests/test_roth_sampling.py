#!/usr/bin/env python3
"""Unit tests for Roth PT-grid ingestion and integration."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_utils import ConfigValidationError, load_and_validate_config
from preprocess import PreprocessError, validate_existing_raw_reuse
from provenance import raw_generation_config_sha256, stable_config_sha256
from roth_sampling import (
    RothSamplingError,
    derive_planet_mass_jup,
    parse_roth_filename,
    sample_roth_profiles,
)
from sampling import build_run_specs

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


def _load_base_config() -> dict:
    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_roth_file(
    path: Path,
    *,
    columns: list[tuple[float, float, list[float], list[float]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("64,32,53\n\n")
        for lon_deg, lat_deg, pressure_bar, temperature_k in columns:
            for level_idx, (pressure_value, temperature_value) in enumerate(
                zip(pressure_bar, temperature_k, strict=True),
                start=1,
            ):
                handle.write(
                    f"{level_idx},{lon_deg},{lat_deg},{pressure_value:.6e},{temperature_value:.6f},0,0,0,0,0,0\n"
                )
            handle.write("\n")


def _write_raw_run(path: Path, *, run_id: int, source_kind: str = "analytic") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["run_id"] = run_id
        inputs = handle.create_group("inputs")
        inputs.create_dataset("pressure_bar", data=np.array([1.0, 0.1], dtype=np.float64))
        inputs.create_dataset("temperature_k", data=np.array([900.0, 800.0], dtype=np.float64))
        inputs.create_dataset("kzz_cm2_s", data=np.array([1.0e9, 1.0e9], dtype=np.float64))
        inputs.create_dataset("state_species", data=np.asarray(["H2"], dtype=str_dtype))
        inputs.create_dataset("output_species", data=np.asarray(["H2"], dtype=str_dtype))
        globals_group = handle.create_group("globals")
        globals_group.create_dataset("gravity_cm_s2", data=np.float64(1.0e3))
        globals_group.create_dataset("metallicity_log10", data=np.float64(0.0))
        globals_group.create_dataset("c_to_o", data=np.float64(0.55))
        trajectory = handle.create_group("trajectory")
        trajectory.create_dataset("time_s", data=np.array([0.0, 10.0], dtype=np.float64))
        trajectory.create_dataset("ymix_state", data=np.ones((2, 2, 1), dtype=np.float64))
        trajectory.create_dataset("ymix_output", data=np.ones((2, 2, 1), dtype=np.float64))
        sampler = handle.create_group("sampler")
        sampler.create_dataset("source_kind", data=np.asarray(source_kind, dtype=str_dtype))


class RothSamplingTests(unittest.TestCase):
    """Regression tests for Roth grid parsing, filtering, and run-spec integration."""

    def test_parse_roth_filename_and_mass_derivation(self) -> None:
        parsed = parse_roth_filename(
            "PTprofiles-Teq_1800-LogMet_0.7-LogDrag_5-Mstar_0.8-Rp_1.3-logG_1.3-TiOVO_false.dat"
        )
        assert parsed is not None
        self.assertEqual(parsed["Teq"], 1800.0)
        self.assertEqual(parsed["LogMet"], 0.7)
        self.assertEqual(parsed["LogDrag"], 5.0)
        self.assertEqual(parsed["Mstar"], 0.8)
        self.assertEqual(parsed["Rp"], 1.3)
        self.assertEqual(parsed["logG"], 1.3)
        self.assertFalse(bool(parsed["TiOVO"]))
        mass_jup = derive_planet_mass_jup(rp_rjup=1.3, logg_cgs=1.3)
        self.assertAlmostEqual(mass_jup, 0.013604103302233179)

    def test_roth_config_allows_enabled_without_local_grid_for_non_generation_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_roth_cfg_load_") as tmpdir_name:
            root = Path(tmpdir_name)
            config = _load_base_config()
            config["roth_sampler"]["enabled"] = True
            config["roth_sampler"]["num_profiles"] = 1
            config["roth_sampler"]["data_glob"] = "roth-grid/PTprofiles/*.dat"
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with mock.patch.dict(os.environ, {"VULCAN_EMULATOR_PROJECT_ROOT": str(root)}):
                loaded = load_and_validate_config(config_path)
        self.assertTrue(bool(loaded["roth_sampler"]["enabled"]))
        self.assertEqual(int(loaded["roth_sampler"]["num_profiles"]), 1)

    def test_roth_config_rejects_disabled_with_positive_num_profiles(self) -> None:
        config = _load_base_config()
        config["roth_sampler"]["enabled"] = False
        config["roth_sampler"]["num_profiles"] = 1
        with tempfile.TemporaryDirectory(prefix="ve_roth_cfg_disabled_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                ConfigValidationError,
                "num_profiles must be 0 when roth_sampler.enabled is false",
            ):
                load_and_validate_config(config_path)

    def test_sample_roth_profiles_rejects_values_not_present_in_grid(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_roth_cfg_") as tmpdir_name:
            root = Path(tmpdir_name)
            roth_root = root / "roth-grid" / "PTprofiles"
            _write_roth_file(
                roth_root / "PTprofiles-Teq_1000-LogMet_0.0-LogDrag_0-Mstar_0.8-Rp_1.3-logG_0.8-TiOVO_false.dat",
                columns=[(-10.0, -10.0, [1.0e-5, 1.0e-3], [700.0, 900.0])],
            )
            config = _load_base_config()
            config["roth_sampler"]["enabled"] = True
            config["roth_sampler"]["num_profiles"] = 1
            config["roth_sampler"]["data_glob"] = "roth-grid/PTprofiles/*.dat"
            config["roth_sampler"]["filters"]["Teq"] = [999.0]
            config["tp_sampler"]["pressure_grid"]["nz"] = 6
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with mock.patch.dict(os.environ, {"VULCAN_EMULATOR_PROJECT_ROOT": str(root)}):
                loaded = load_and_validate_config(config_path)
                with self.assertRaisesRegex(RothSamplingError, "not present in the Roth grid"):
                    sample_roth_profiles(
                        loaded,
                        target_pressure_bar=np.logspace(1.0, -8.0, 6, dtype=np.float64),
                        rng=np.random.default_rng(0),
                    )

    def test_sample_roth_profiles_skips_invalid_singleton_columns_and_interpolates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_roth_sample_") as tmpdir_name:
            root = Path(tmpdir_name)
            roth_root = root / "roth-grid" / "PTprofiles"
            _write_roth_file(
                roth_root / "PTprofiles-Teq_1000-LogMet_0.0-LogDrag_0-Mstar_0.8-Rp_1.3-logG_0.8-TiOVO_false.dat",
                columns=[
                    (-10.0, -10.0, [1.0e-6, 1.0e-4, 1.0e-2], [650.0, 900.0, 1200.0]),
                    (32.0, 53.0, [1.0e-5], [700.0]),
                ],
            )
            _write_roth_file(
                roth_root / "PTprofiles-Teq_1200-LogMet_0.7-LogDrag_3-Mstar_1.1-Rp_1.3-logG_1.3-TiOVO_true.dat",
                columns=[
                    (20.0, 40.0, [5.0e-6, 5.0e-4, 5.0e-2], [700.0, 1000.0, 1400.0]),
                ],
            )
            config = _load_base_config()
            config["roth_sampler"]["enabled"] = True
            config["roth_sampler"]["num_profiles"] = 2
            config["roth_sampler"]["data_glob"] = "roth-grid/PTprofiles/*.dat"
            config["roth_sampler"]["filters"]["Teq"] = [1000.0, 1200.0]
            config["roth_sampler"]["filters"]["LogMet"] = [0.0, 0.7]
            config["roth_sampler"]["filters"]["LogDrag"] = [0.0, 3.0]
            config["roth_sampler"]["filters"]["Mstar"] = [0.8, 1.1]
            config["roth_sampler"]["filters"]["Rp"] = [1.3]
            config["roth_sampler"]["filters"]["logG"] = [0.8, 1.3]
            config["roth_sampler"]["filters"]["TiOVO"] = [False, True]
            config["tp_sampler"]["pressure_grid"]["nz"] = 8
            config["tp_sampler"]["pressure_grid"]["p_top_bar"] = 1.0e-8
            config["tp_sampler"]["pressure_grid"]["p_bottom_bar"] = 1.0e1
            with mock.patch.dict(os.environ, {"VULCAN_EMULATOR_PROJECT_ROOT": str(root)}):
                rng = np.random.default_rng(43)
                profiles = sample_roth_profiles(
                    config,
                    target_pressure_bar=np.logspace(1.0, -8.0, 8, dtype=np.float64),
                    rng=rng,
                )

        self.assertEqual(len(profiles), 2)
        self.assertEqual({(profile.lon_deg, profile.lat_deg) for profile in profiles}, {(-10.0, -10.0), (20.0, 40.0)})
        for profile in profiles:
            self.assertEqual(profile.interpolated_temperature_k.shape, (8,))
            self.assertTrue(np.all(np.isfinite(profile.interpolated_temperature_k)))
            self.assertTrue(profile.extrapolated_top)
            self.assertTrue(profile.extrapolated_bottom)

    def test_sample_roth_profiles_skips_columns_invalid_only_after_extrapolation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_roth_extrap_") as tmpdir_name:
            root = Path(tmpdir_name)
            roth_root = root / "roth-grid" / "PTprofiles"
            _write_roth_file(
                roth_root / "PTprofiles-Teq_1000-LogMet_0.0-LogDrag_0-Mstar_0.8-Rp_1.3-logG_0.8-TiOVO_false.dat",
                columns=[
                    (-10.0, -10.0, [1.0e-6, 1.0e-5, 1.0e-4], [100.0, 80.0, 60.0]),
                    (20.0, 20.0, [1.0e-6, 1.0e-5, 1.0e-4], [700.0, 900.0, 1100.0]),
                ],
            )
            config = _load_base_config()
            config["roth_sampler"]["enabled"] = True
            config["roth_sampler"]["num_profiles"] = 1
            config["roth_sampler"]["data_glob"] = "roth-grid/PTprofiles/*.dat"
            config["roth_sampler"]["filters"]["Teq"] = [1000.0]
            config["roth_sampler"]["filters"]["LogMet"] = [0.0]
            config["roth_sampler"]["filters"]["LogDrag"] = [0.0]
            config["roth_sampler"]["filters"]["Mstar"] = [0.8]
            config["roth_sampler"]["filters"]["Rp"] = [1.3]
            config["roth_sampler"]["filters"]["logG"] = [0.8]
            config["roth_sampler"]["filters"]["TiOVO"] = [False]
            config["tp_sampler"]["pressure_grid"]["nz"] = 8
            config["tp_sampler"]["pressure_grid"]["p_top_bar"] = 1.0e-8
            config["tp_sampler"]["pressure_grid"]["p_bottom_bar"] = 1.0e1
            with mock.patch.dict(os.environ, {"VULCAN_EMULATOR_PROJECT_ROOT": str(root)}):
                profiles = sample_roth_profiles(
                    config,
                    target_pressure_bar=np.logspace(1.0, -8.0, 8, dtype=np.float64),
                    rng=np.random.default_rng(0),
                )

        self.assertEqual(len(profiles), 1)
        self.assertEqual((profiles[0].lon_deg, profiles[0].lat_deg), (20.0, 20.0))
        self.assertTrue(np.all(profiles[0].interpolated_temperature_k > 0.0))

    def test_sample_roth_profiles_skips_columns_above_temperature_ceiling(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_roth_hot_") as tmpdir_name:
            root = Path(tmpdir_name)
            roth_root = root / "roth-grid" / "PTprofiles"
            _write_roth_file(
                roth_root / "PTprofiles-Teq_1000-LogMet_0.0-LogDrag_0-Mstar_0.8-Rp_1.3-logG_0.8-TiOVO_false.dat",
                columns=[
                    (-10.0, -10.0, [1.0e-6, 1.0e-5, 1.0e-4], [1600.0, 2000.0, 2400.0]),
                    (20.0, 20.0, [1.0e-6, 1.0e-5, 1.0e-4], [700.0, 900.0, 1100.0]),
                ],
            )
            config = _load_base_config()
            config["roth_sampler"]["enabled"] = True
            config["roth_sampler"]["num_profiles"] = 1
            config["roth_sampler"]["data_glob"] = "roth-grid/PTprofiles/*.dat"
            config["roth_sampler"]["filters"]["Teq"] = [1000.0]
            config["roth_sampler"]["filters"]["LogMet"] = [0.0]
            config["roth_sampler"]["filters"]["LogDrag"] = [0.0]
            config["roth_sampler"]["filters"]["Mstar"] = [0.8]
            config["roth_sampler"]["filters"]["Rp"] = [1.3]
            config["roth_sampler"]["filters"]["logG"] = [0.8]
            config["roth_sampler"]["filters"]["TiOVO"] = [False]
            config["tp_sampler"]["pressure_grid"]["nz"] = 8
            config["tp_sampler"]["pressure_grid"]["p_top_bar"] = 1.0e-8
            config["tp_sampler"]["pressure_grid"]["p_bottom_bar"] = 1.0e1
            with mock.patch.dict(os.environ, {"VULCAN_EMULATOR_PROJECT_ROOT": str(root)}):
                profiles = sample_roth_profiles(
                    config,
                    target_pressure_bar=np.logspace(1.0, -8.0, 8, dtype=np.float64),
                    rng=np.random.default_rng(0),
                )

        self.assertEqual(len(profiles), 1)
        self.assertEqual((profiles[0].lon_deg, profiles[0].lat_deg), (20.0, 20.0))
        self.assertLessEqual(float(np.max(profiles[0].interpolated_temperature_k)), 2500.0)

    def test_build_run_specs_preserves_analytic_prefix_when_roth_enabled(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_roth_runspec_") as tmpdir_name:
            root = Path(tmpdir_name)
            roth_root = root / "roth-grid" / "PTprofiles"
            _write_roth_file(
                roth_root / "PTprofiles-Teq_1000-LogMet_0.0-LogDrag_0-Mstar_0.8-Rp_1.3-logG_0.8-TiOVO_false.dat",
                columns=[(-10.0, -10.0, [1.0e-6, 1.0e-4, 1.0e-2], [650.0, 900.0, 1200.0])],
            )
            analytic_only = _load_base_config()
            analytic_only["generation"]["num_runs"] = 2
            analytic_only["roth_sampler"]["enabled"] = False
            analytic_only["roth_sampler"]["num_profiles"] = 0
            analytic_only["tp_sampler"]["pressure_grid"]["nz"] = 6

            mixed = deepcopy(analytic_only)
            mixed["roth_sampler"]["enabled"] = True
            mixed["roth_sampler"]["num_profiles"] = 1
            mixed["roth_sampler"]["data_glob"] = "roth-grid/PTprofiles/*.dat"
            mixed["roth_sampler"]["filters"]["Teq"] = [1000.0]
            mixed["roth_sampler"]["filters"]["LogMet"] = [0.0]
            mixed["roth_sampler"]["filters"]["LogDrag"] = [0.0]
            mixed["roth_sampler"]["filters"]["Mstar"] = [0.8]
            mixed["roth_sampler"]["filters"]["Rp"] = [1.3]
            mixed["roth_sampler"]["filters"]["logG"] = [0.8]
            mixed["roth_sampler"]["filters"]["TiOVO"] = [False]
            with mock.patch.dict(os.environ, {"VULCAN_EMULATOR_PROJECT_ROOT": str(root)}):
                analytic_specs = build_run_specs(analytic_only)
                mixed_specs = build_run_specs(mixed)

        self.assertEqual(len(analytic_specs), 2)
        self.assertEqual(len(mixed_specs), 3)
        for analytic, mixed_prefix in zip(analytic_specs, mixed_specs[:2], strict=True):
            self.assertEqual(analytic.run_id, mixed_prefix.run_id)
            np.testing.assert_allclose(analytic.pressure_bar, mixed_prefix.pressure_bar)
            np.testing.assert_allclose(analytic.temperature_k, mixed_prefix.temperature_k)
            np.testing.assert_allclose(analytic.kzz_cm2_s, mixed_prefix.kzz_cm2_s)
            self.assertEqual(analytic.gravity_cm_s2, mixed_prefix.gravity_cm_s2)
            self.assertEqual(analytic.metallicity_log10, mixed_prefix.metallicity_log10)
            self.assertEqual(analytic.c_to_o, mixed_prefix.c_to_o)
        self.assertEqual(mixed_specs[-1].source_tag, "roth")
        self.assertEqual(mixed_specs[-1].run_id, 2)

    def test_validate_existing_raw_reuse_rejects_roth_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_roth_reuse_") as tmpdir_name:
            root = Path(tmpdir_name)
            data_root = root / "data"
            raw_root = data_root / "raw"
            run_path = raw_root / "run_000000.h5"
            _write_raw_run(run_path, run_id=0)

            config = _load_base_config()
            config["generation"]["manifest_filename"] = "dataset_manifest.json"
            config["roth_sampler"]["enabled"] = True
            config["roth_sampler"]["num_profiles"] = 1
            config["roth_sampler"]["data_glob"] = "roth-grid/PTprofiles/*.dat"

            manifest_path = data_root / "dataset_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "num_runs": 1,
                        "run_files": ["data/raw/run_000000.h5"],
                        "all_raw_run_files": ["data/raw/run_000000.h5"],
                        "split": {"train": [0], "val": [], "test": []},
                        "state_species": ["H2"],
                        "output_species": ["H2"],
                        "source_counts": {"analytic": 1},
                        "raw_generation_config_sha256": raw_generation_config_sha256(config),
                    }
                ),
                encoding="utf-8",
            )
            paths = SimpleNamespace(root=root, data_root=data_root)
            validate_existing_raw_reuse(config, paths, run_files=[run_path])

            changed = deepcopy(config)
            changed["roth_sampler"]["num_profiles"] = 2
            with self.assertRaisesRegex(PreprocessError, "do not match the current raw-generation config"):
                validate_existing_raw_reuse(changed, paths, run_files=[run_path])

    def test_validate_existing_raw_reuse_rejects_unexpected_extra_run_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_roth_reuse_extra_") as tmpdir_name:
            root = Path(tmpdir_name)
            data_root = root / "data"
            raw_root = data_root / "raw"
            run_path = raw_root / "run_000000.h5"
            extra_run_path = raw_root / "run_000001.h5"
            _write_raw_run(run_path, run_id=0)
            _write_raw_run(extra_run_path, run_id=1)

            config = _load_base_config()
            config["generation"]["manifest_filename"] = "dataset_manifest.json"
            manifest_path = data_root / "dataset_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "num_runs": 1,
                        "run_files": ["data/raw/run_000000.h5"],
                        "all_raw_run_files": ["data/raw/run_000000.h5"],
                        "split": {"train": [0], "val": [], "test": []},
                        "state_species": ["H2"],
                        "output_species": ["H2"],
                        "source_counts": {"analytic": 1},
                        "raw_generation_config_sha256": raw_generation_config_sha256(config),
                    }
                ),
                encoding="utf-8",
            )
            paths = SimpleNamespace(root=root, data_root=data_root)
            with self.assertRaisesRegex(PreprocessError, "unexpected raw files"):
                validate_existing_raw_reuse(
                    config,
                    paths,
                    run_files=sorted(raw_root.glob("run_*.h5")),
                )

    def test_validate_existing_raw_reuse_rejects_source_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_roth_reuse_counts_") as tmpdir_name:
            root = Path(tmpdir_name)
            data_root = root / "data"
            raw_root = data_root / "raw"
            run_path = raw_root / "run_000000.h5"
            _write_raw_run(run_path, run_id=0, source_kind="analytic")

            config = _load_base_config()
            config["generation"]["manifest_filename"] = "dataset_manifest.json"
            manifest_path = data_root / "dataset_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "num_runs": 1,
                        "run_files": ["data/raw/run_000000.h5"],
                        "all_raw_run_files": ["data/raw/run_000000.h5"],
                        "split": {"train": [0], "val": [], "test": []},
                        "state_species": ["H2"],
                        "output_species": ["H2"],
                        "source_counts": {"roth": 1},
                        "raw_generation_config_sha256": raw_generation_config_sha256(config),
                    }
                ),
                encoding="utf-8",
            )
            paths = SimpleNamespace(root=root, data_root=data_root)
            with self.assertRaisesRegex(PreprocessError, "source counts do not match"):
                validate_existing_raw_reuse(
                    config,
                    paths,
                    run_files=[run_path],
                )

    def test_validate_existing_raw_reuse_requires_all_raw_run_files_for_extra_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_roth_reuse_legacy_") as tmpdir_name:
            root = Path(tmpdir_name)
            data_root = root / "data"
            raw_root = data_root / "raw"
            run_path = raw_root / "run_000000.h5"
            extra_run_path = raw_root / "run_000001.h5"
            _write_raw_run(run_path, run_id=0)
            _write_raw_run(extra_run_path, run_id=1)

            config = _load_base_config()
            config["generation"]["manifest_filename"] = "dataset_manifest.json"
            manifest_path = data_root / "dataset_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "num_runs": 1,
                        "run_files": ["data/raw/run_000000.h5"],
                        "split": {"train": [0], "val": [], "test": []},
                        "state_species": ["H2"],
                        "output_species": ["H2"],
                        "source_counts": {"analytic": 1},
                        "raw_generation_config_sha256": raw_generation_config_sha256(config),
                    }
                ),
                encoding="utf-8",
            )
            paths = SimpleNamespace(root=root, data_root=data_root)
            with self.assertRaisesRegex(PreprocessError, "missing 'all_raw_run_files'"):
                validate_existing_raw_reuse(
                    config,
                    paths,
                    run_files=sorted(raw_root.glob("run_*.h5")),
                )

    def test_processed_config_hash_changes_when_roth_sampler_changes(self) -> None:
        config = _load_base_config()
        baseline = stable_config_sha256(config)
        config["roth_sampler"]["num_profiles"] = 3
        self.assertNotEqual(stable_config_sha256(config), baseline)


if __name__ == "__main__":
    unittest.main()
