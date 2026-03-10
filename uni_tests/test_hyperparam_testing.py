#!/usr/bin/env python3
"""Regression tests for the Optuna hyperparameter search helper."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import hyperparam_testing

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


def _load_base_config() -> dict:
    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_temp_config(root: Path) -> Path:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps(_load_base_config()), encoding="utf-8")
    return config_path


def _write_training_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(hyperparam_testing.TRIAL_LOG_HEADER)
        writer.writerow([1, 0.1, 0.2, 0.01, 0.02, 0.03, 0.04, 1.0e-4, 1.0, 1.0])


def _write_trial_artifacts(run_dir: Path, config: dict, objective: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_training_log(run_dir / "training_log.csv")
    (run_dir / "last.pt").write_bytes(b"last")
    torch.save({"config": config, "model_state": {}}, run_dir / "best.pt")
    for filename, payload in (
        ("metrics.json", {"best_val_combined_loss": objective}),
        ("data_contract.json", {"sequence_length": 1}),
        ("normalization_metadata.json", {"targets": {"ymix": {"std": [1.0], "method": "log-standard"}}}),
        ("processed_fingerprint.json", {"version": 2}),
    ):
        with (run_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)


class HyperparamTestingTests(unittest.TestCase):
    """Ensure the hyperparameter search runner keeps its output contract."""

    def test_search_promotes_one_best_checkpoint_and_saves_full_config(self) -> None:
        objectives = iter([0.30, 0.10, 0.20])

        def fake_run_training(config: dict, paths, _precision) -> None:
            self.assertEqual(config["paths"]["logs_root"], "models/hyperparam_testing/logs")
            self.assertEqual(config["paths"]["models_root"], "models/hyperparam_testing/_scratch")
            self.assertEqual(config["training"]["epochs"], 20)
            self.assertEqual(
                config["training"]["live_sampling"]["eval_pairs_per_run"],
                _load_base_config()["training"]["live_sampling"]["eval_pairs_per_run"],
            )
            self.assertIn(config["training"]["batch_size"], [128, 256])
            self.assertIn(
                config["training"]["model"]["d_model"],
                [256, 384, 512, 768],
            )
            self.assertEqual(
                config["training"]["model"]["d_model"] % config["training"]["model"]["nhead"],
                0,
            )
            self.assertIn(config["training"]["model"]["num_layers"], [4, 6, 8])
            self.assertIn(
                config["training"]["live_sampling"]["train_pairs_per_run_per_epoch"],
                [500, 1000, 1500],
            )
            progress_log = paths.logs_root / f"training_progress_{config['training']['output_folder']}.log"
            progress_log.parent.mkdir(parents=True, exist_ok=True)
            progress_log.write_text("progress\n", encoding="utf-8")
            run_dir = paths.models_root / str(config["training"]["output_folder"])
            _write_trial_artifacts(run_dir, config, next(objectives))

        with tempfile.TemporaryDirectory(prefix="ve_hyperparam_") as tmpdir_name:
            root = Path(tmpdir_name)
            config_path = _write_temp_config(root)
            with mock.patch.dict(
                hyperparam_testing.os.environ,
                {"VULCAN_EMULATOR_PROJECT_ROOT": str(root)},
                clear=False,
            ):
                with mock.patch.object(hyperparam_testing, "run_training", side_effect=fake_run_training):
                    result = hyperparam_testing.run_hyperparameter_search(
                        config_path=config_path,
                        trials=3,
                        trial_epochs=20,
                        overwrite=False,
                    )

            output_root = root / "models" / "hyperparam_testing"
            best_model_dir = output_root / "best_model"
            best_config_path = output_root / "best_config.json"
            self.assertEqual(result["best_trial_number"], 1)
            self.assertEqual(sorted(path.name for path in (output_root / "logs").glob("trial_*_summary.json")), [
                "trial_000_summary.json",
                "trial_001_summary.json",
                "trial_002_summary.json",
            ])
            self.assertEqual(sorted(path.name for path in (output_root / "logs").glob("trial_*.log")), [
                "trial_000.log",
                "trial_001.log",
                "trial_002.log",
            ])
            self.assertEqual(sorted(path.name for path in (output_root / "logs").glob("trial_*_training_log.csv")), [
                "trial_000_training_log.csv",
                "trial_001_training_log.csv",
                "trial_002_training_log.csv",
            ])
            self.assertTrue(best_model_dir.is_dir())
            self.assertTrue(best_config_path.is_file())
            self.assertFalse((best_model_dir / "last.pt").exists())
            self.assertFalse((output_root / "_scratch").exists())
            self.assertEqual([path.name for path in output_root.rglob("best.pt")], ["best.pt"])
            self.assertEqual(list((output_root / "logs").glob("training_progress_*.log")), [])

            for training_log_path in sorted((output_root / "logs").glob("trial_*_training_log.csv")):
                rows = training_log_path.read_text(encoding="utf-8").strip().splitlines()
                self.assertEqual(len(rows), 2, training_log_path.name)

            best_config = json.loads(best_config_path.read_text(encoding="utf-8"))
            self.assertEqual(best_config["paths"]["models_root"], "models")
            self.assertEqual(best_config["training"]["output_folder"], "hyperparam_testing/best_model")

            checkpoint = torch.load(best_model_dir / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(
                checkpoint["config"]["training"]["output_folder"],
                "hyperparam_testing/best_model",
            )
            self.assertEqual(checkpoint["config"]["paths"]["models_root"], "models")

    def test_search_records_failed_trials_and_continues(self) -> None:
        call_counter = {"count": 0}

        def fake_run_training(config: dict, paths, _precision) -> None:
            call_counter["count"] += 1
            progress_log = paths.logs_root / f"training_progress_{config['training']['output_folder']}.log"
            progress_log.parent.mkdir(parents=True, exist_ok=True)
            progress_log.write_text("progress\n", encoding="utf-8")
            run_dir = paths.models_root / str(config["training"]["output_folder"])
            if call_counter["count"] == 1:
                run_dir.mkdir(parents=True, exist_ok=True)
                _write_training_log(run_dir / "training_log.csv")
                raise RuntimeError("synthetic failure")
            _write_trial_artifacts(run_dir, config, 0.15)

        with tempfile.TemporaryDirectory(prefix="ve_hyperparam_fail_") as tmpdir_name:
            root = Path(tmpdir_name)
            config_path = _write_temp_config(root)
            with mock.patch.dict(
                hyperparam_testing.os.environ,
                {"VULCAN_EMULATOR_PROJECT_ROOT": str(root)},
                clear=False,
            ):
                with mock.patch.object(hyperparam_testing, "run_training", side_effect=fake_run_training):
                    hyperparam_testing.run_hyperparameter_search(
                        config_path=config_path,
                        trials=2,
                        trial_epochs=20,
                        overwrite=False,
                    )

            logs_root = root / "models" / "hyperparam_testing" / "logs"
            failed_summary = json.loads((logs_root / "trial_000_summary.json").read_text(encoding="utf-8"))
            succeeded_summary = json.loads((logs_root / "trial_001_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(failed_summary["status"], "failed")
            self.assertEqual(failed_summary["error_type"], "RuntimeError")
            self.assertEqual(succeeded_summary["status"], "completed")
            self.assertTrue((root / "models" / "hyperparam_testing" / "best_config.json").is_file())
            self.assertEqual(list(logs_root.glob("training_progress_*.log")), [])
            self.assertEqual(
                len((logs_root / "trial_000_training_log.csv").read_text(encoding="utf-8").strip().splitlines()),
                2,
            )


if __name__ == "__main__":
    unittest.main()
