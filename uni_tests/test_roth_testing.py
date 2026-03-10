#!/usr/bin/env python3
"""Regression tests for the standalone Roth testing plot helper."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_roth_testing_module():
    module_path = PROJECT_ROOT / "roth" / "roth_testing.py"
    spec = importlib.util.spec_from_file_location("roth_testing", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_roth_file(
    path: Path,
    *,
    lon_deg: float,
    lat_deg: float,
    pressure_bar: list[float],
    temperature_k: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("64,32,53\n\n")
        for level_idx, (pressure_value, temperature_value) in enumerate(
            zip(pressure_bar, temperature_k, strict=True),
            start=1,
        ):
            handle.write(
                f"{level_idx},{lon_deg},{lat_deg},{pressure_value:.6e},{temperature_value:.6f},0,0,0,0,0,0\n"
            )


class RothTestingScriptTests(unittest.TestCase):
    """Ensure the lightweight Roth testing plot helper stays usable."""

    def test_roth_testing_ignores_large_configured_num_profiles_for_small_plot(self) -> None:
        module = _load_roth_testing_module()
        with tempfile.TemporaryDirectory(prefix="ve_roth_testing_") as tmpdir_name:
            root = Path(tmpdir_name)
            config = json.loads((PROJECT_ROOT / "config" / "config.json").read_text(encoding="utf-8"))
            config["roth_sampler"]["enabled"] = True
            config["roth_sampler"]["num_profiles"] = 1000
            config["roth_sampler"]["data_glob"] = "roth/roth-grid/PTprofiles/*.dat"
            config["roth_sampler"]["filters"]["Teq"] = [1000.0]
            config["roth_sampler"]["filters"]["LogMet"] = [0.0]
            config["roth_sampler"]["filters"]["LogDrag"] = [0.0]
            config["roth_sampler"]["filters"]["Mstar"] = [0.8]
            config["roth_sampler"]["filters"]["Rp"] = [1.3]
            config["roth_sampler"]["filters"]["logG"] = [0.8]
            config["roth_sampler"]["filters"]["TiOVO"] = [False]
            config["tp_sampler"]["pressure_grid"]["nz"] = 6

            config_path = root / "config" / "config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config), encoding="utf-8")

            _write_roth_file(
                root
                / "roth"
                / "roth-grid"
                / "PTprofiles"
                / "PTprofiles-Teq_1000-LogMet_0.0-LogDrag_0-Mstar_0.8-Rp_1.3-logG_0.8-TiOVO_false.dat",
                lon_deg=-10.0,
                lat_deg=-20.0,
                pressure_bar=[1.0e-6, 1.0e-4, 1.0e-2],
                temperature_k=[700.0, 950.0, 1200.0],
            )
            _write_roth_file(
                root
                / "roth"
                / "roth-grid"
                / "PTprofiles"
                / "PTprofiles-Teq_1000-LogMet_0.0-LogDrag_0-Mstar_0.8-Rp_1.3-logG_0.8-TiOVO_false_copy.dat",
                lon_deg=15.0,
                lat_deg=25.0,
                pressure_bar=[1.0e-6, 1.0e-4, 1.0e-2],
                temperature_k=[800.0, 1000.0, 1300.0],
            )

            output_path = root / "logs" / "roth_testing.png"
            argv = [
                "roth_testing.py",
                "--config",
                "config/config.json",
                "--count",
                "2",
                "--debug",
                "--output",
                "logs/roth_testing.png",
            ]
            with mock.patch.dict(os.environ, {"VULCAN_EMULATOR_PROJECT_ROOT": str(root)}):
                with mock.patch.object(sys, "argv", argv):
                    module.main()

            self.assertTrue(output_path.is_file())

    def test_roth_testing_falls_back_to_extra_columns_when_unique_files_are_insufficient(self) -> None:
        module = _load_roth_testing_module()
        with tempfile.TemporaryDirectory(prefix="ve_roth_testing_extra_cols_") as tmpdir_name:
            root = Path(tmpdir_name)
            config = json.loads((PROJECT_ROOT / "config" / "config.json").read_text(encoding="utf-8"))
            config["roth_sampler"]["enabled"] = True
            config["roth_sampler"]["num_profiles"] = 1000
            config["roth_sampler"]["data_glob"] = "roth/roth-grid/PTprofiles/*.dat"
            config["roth_sampler"]["filters"]["Teq"] = [1000.0]
            config["roth_sampler"]["filters"]["LogMet"] = [0.0]
            config["roth_sampler"]["filters"]["LogDrag"] = [0.0]
            config["roth_sampler"]["filters"]["Mstar"] = [0.8]
            config["roth_sampler"]["filters"]["Rp"] = [1.3]
            config["roth_sampler"]["filters"]["logG"] = [0.8]
            config["roth_sampler"]["filters"]["TiOVO"] = [False]
            config["tp_sampler"]["pressure_grid"]["nz"] = 6

            file_path = (
                root
                / "roth"
                / "roth-grid"
                / "PTprofiles"
                / "PTprofiles-Teq_1000-LogMet_0.0-LogDrag_0-Mstar_0.8-Rp_1.3-logG_0.8-TiOVO_false.dat"
            )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("w", encoding="utf-8") as handle:
                handle.write("64,32,53\n\n")
                for level_idx, (pressure_value, temperature_value) in enumerate(
                    zip([1.0e-6, 1.0e-4, 1.0e-2], [700.0, 950.0, 1200.0], strict=True),
                    start=1,
                ):
                    handle.write(
                        f"{level_idx},-10.0,-20.0,{pressure_value:.6e},{temperature_value:.6f},0,0,0,0,0,0\n"
                    )
                handle.write("\n")
                for level_idx, (pressure_value, temperature_value) in enumerate(
                    zip([1.0e-6, 1.0e-4, 1.0e-2], [800.0, 1000.0, 1300.0], strict=True),
                    start=1,
                ):
                    handle.write(
                        f"{level_idx},15.0,25.0,{pressure_value:.6e},{temperature_value:.6f},0,0,0,0,0,0\n"
                    )

            with mock.patch.dict(os.environ, {"VULCAN_EMULATOR_PROJECT_ROOT": str(root)}):
                pressure_bar, profiles, stats = module._select_extrapolated_profiles(
                    config=config,
                    count=2,
                    debug=False,
                )

        self.assertEqual(pressure_bar.shape, (6,))
        self.assertEqual(len(profiles), 2)
        self.assertEqual(stats.selected_unique_files, 1)
        self.assertEqual({(profile.lon_deg, profile.lat_deg) for profile in profiles}, {(-10.0, -20.0), (15.0, 25.0)})


if __name__ == "__main__":
    unittest.main()
