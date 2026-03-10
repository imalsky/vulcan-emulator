#!/usr/bin/env python3
"""Unit tests for local helper scripts."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_training_progression_module():
    module_path = PROJECT_ROOT / "extras" / "training_progression.py"
    spec = importlib.util.spec_from_file_location("training_progression", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_plot_roth_profile_module():
    module_path = PROJECT_ROOT / "extras" / "plot_roth_profile.py"
    spec = importlib.util.spec_from_file_location("plot_roth_profile", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
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


class ExtrasContractTests(unittest.TestCase):
    """Regression tests for local utility-script contract drift."""

    def test_training_progression_reads_current_training_log_header(self) -> None:
        module = _load_training_progression_module()
        with tempfile.TemporaryDirectory(prefix="ve_training_progress_") as tmpdir_name:
            log_path = Path(tmpdir_name) / "training_log.csv"
            with log_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "epoch",
                        "train_combined",
                        "val_combined",
                        "train_mse",
                        "val_mse",
                        "train_mae_log10",
                        "val_mae_log10",
                        "lr",
                        "epoch_seconds",
                        "elapsed_seconds",
                    ]
                )
                writer.writerow([1, 1.0, 1.1, 0.1, 0.2, 0.3, 0.4, 1.0e-3, 10.0, 10.0])

            epochs, train_mse, val_mse, train_mae_log10, val_mae_log10, lrs = module._load_log(log_path)

        self.assertEqual(epochs, [1])
        self.assertEqual(train_mse, [0.1])
        self.assertEqual(val_mse, [0.2])
        self.assertEqual(train_mae_log10, [0.3])
        self.assertEqual(val_mae_log10, [0.4])
        self.assertEqual(lrs, [1.0e-3])

    def test_plot_roth_profile_generates_one_png(self) -> None:
        module = _load_plot_roth_profile_module()
        with tempfile.TemporaryDirectory(prefix="ve_roth_plot_") as tmpdir_name:
            root = Path(tmpdir_name)
            config = json.loads((PROJECT_ROOT / "config" / "config.json").read_text(encoding="utf-8"))
            config["roth_sampler"]["data_glob"] = "roth-grid/PTprofiles/*.dat"
            config["roth_sampler"]["filters"]["Teq"] = [1000.0]
            config["roth_sampler"]["filters"]["LogMet"] = [0.0]
            config["roth_sampler"]["filters"]["LogDrag"] = [0.0]
            config["roth_sampler"]["filters"]["Mstar"] = [0.8]
            config["roth_sampler"]["filters"]["Rp"] = [1.3]
            config["roth_sampler"]["filters"]["logG"] = [0.8]
            config["roth_sampler"]["filters"]["TiOVO"] = [False]
            config["tp_sampler"]["pressure_grid"]["nz"] = 6
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            _write_roth_file(
                root
                / "roth-grid"
                / "PTprofiles"
                / "PTprofiles-Teq_1000-LogMet_0.0-LogDrag_0-Mstar_0.8-Rp_1.3-logG_0.8-TiOVO_false.dat",
                lon_deg=-10.0,
                lat_deg=-20.0,
                pressure_bar=[1.0e-6, 1.0e-4, 1.0e-2],
                temperature_k=[700.0, 950.0, 1200.0],
            )
            output_path = root / "roth_plot.png"
            argv = [
                "plot_roth_profile.py",
                "--config",
                str(config_path),
                "--output",
                str(output_path),
            ]
            with mock.patch.dict(os.environ, {"VULCAN_EMULATOR_PROJECT_ROOT": str(root)}):
                with mock.patch.object(sys, "argv", argv):
                    module.main()
            self.assertTrue(output_path.is_file())

    def test_plot_roth_profile_uses_env_project_root_for_relative_paths(self) -> None:
        module = _load_plot_roth_profile_module()
        with tempfile.TemporaryDirectory(prefix="ve_roth_plot_env_") as tmpdir_name:
            root = Path(tmpdir_name)
            config = json.loads((PROJECT_ROOT / "config" / "config.json").read_text(encoding="utf-8"))
            config["roth_sampler"]["data_glob"] = "roth-grid/PTprofiles/*.dat"
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
                / "roth-grid"
                / "PTprofiles"
                / "PTprofiles-Teq_1000-LogMet_0.0-LogDrag_0-Mstar_0.8-Rp_1.3-logG_0.8-TiOVO_false.dat",
                lon_deg=-10.0,
                lat_deg=-20.0,
                pressure_bar=[1.0e-6, 1.0e-4, 1.0e-2],
                temperature_k=[700.0, 950.0, 1200.0],
            )
            argv = [
                "plot_roth_profile.py",
                "--config",
                "config/config.json",
                "--output",
                "logs/roth_plot.png",
            ]
            with mock.patch.dict(os.environ, {"VULCAN_EMULATOR_PROJECT_ROOT": str(root)}):
                with mock.patch.object(sys, "argv", argv):
                    module.main()
            self.assertTrue((root / "logs" / "roth_plot.png").is_file())


if __name__ == "__main__":
    unittest.main()
