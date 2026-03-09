#!/usr/bin/env python3
"""Fast unit tests for config loading and precision resolution."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_utils import ConfigValidationError, load_and_validate_config, resolve_precision

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


def _load_base_config() -> dict[str, Any]:
    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class ConfigUtilsTests(unittest.TestCase):
    """Unit tests for config parsing and precision policy validation."""

    def test_shipped_config_loads_and_resolves_precision(self) -> None:
        config = load_and_validate_config(BASE_CONFIG_PATH)
        precision = resolve_precision(config)
        self.assertEqual(precision.input_dtype, torch.float32)
        self.assertEqual(precision.stats_dtype, torch.float32)
        self.assertEqual(precision.model_dtype, torch.float32)
        self.assertEqual(precision.forward_dtype, torch.float32)
        self.assertEqual(precision.loss_dtype, torch.float32)
        self.assertEqual(precision.optimizer_state_dtype, torch.float32)
        self.assertIsNone(precision.amp_dtype)
        self.assertFalse(precision.use_amp)

    def test_resolve_precision_rejects_amp_when_device_is_not_cuda(self) -> None:
        config = deepcopy(_load_base_config())
        config["training"]["device"] = "cpu"
        config["training"]["use_amp"] = True
        config["precision"]["amp_autocast_dtype"] = "float16"
        with self.assertRaisesRegex(
            ConfigValidationError,
            "training.use_amp=true requires training.device='cuda'",
        ):
            resolve_precision(config)

    def test_load_and_validate_config_rejects_absolute_logs_path(self) -> None:
        config = deepcopy(_load_base_config())
        config["paths"]["logs_root"] = "/tmp/absolute_logs"
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                ConfigValidationError,
                "paths.logs_root must be relative",
            ):
                load_and_validate_config(config_path)

    def test_load_and_validate_config_requires_explicit_processed_root(self) -> None:
        config = deepcopy(_load_base_config())
        config["paths"].pop("processed_root")
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigValidationError, "Missing required keys in paths"):
                load_and_validate_config(config_path)

    def test_load_and_validate_config_rejects_legacy_failure_policy_key(self) -> None:
        config = deepcopy(_load_base_config())
        config["generation"]["failure_policy"] = "continue_on_error"
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigValidationError, "Unexpected keys in generation"):
                load_and_validate_config(config_path)

    def test_shipped_config_has_explicit_flat_data_paths(self) -> None:
        config = load_and_validate_config(BASE_CONFIG_PATH)
        self.assertEqual(config["paths"]["raw_root"], "data/raw")
        self.assertEqual(config["paths"]["processed_root"], "data/processed")

    def test_load_and_validate_config_rejects_unexpected_generation_keys(self) -> None:
        config = deepcopy(_load_base_config())
        config["generation"]["runs_root"] = "data/raw/runs"
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigValidationError, "Unexpected keys in generation"):
                load_and_validate_config(config_path)

    def test_load_and_validate_config_requires_live_sampling_section(self) -> None:
        config = deepcopy(_load_base_config())
        config["training"].pop("live_sampling")
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigValidationError, "Missing required keys in training"):
                load_and_validate_config(config_path)

    def test_load_and_validate_config_rejects_non_cuda_training_device(self) -> None:
        config = deepcopy(_load_base_config())
        config["training"]["device"] = "cpu"
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                ConfigValidationError,
                "training.device must be 'cuda' for live-sampling training",
            ):
                load_and_validate_config(config_path)

    def test_load_and_validate_config_rejects_legacy_trajectory_sampling_key(self) -> None:
        config = deepcopy(_load_base_config())
        config["trajectory_sampling"]["pairs_per_run"] = 100
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigValidationError, "Unexpected keys in trajectory_sampling"):
                load_and_validate_config(config_path)

    def test_load_and_validate_config_rejects_legacy_rollout_eval_points_key(self) -> None:
        config = deepcopy(_load_base_config())
        config["trajectory_sampling"]["rollout_eval_points"] = 8
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigValidationError, "Unexpected keys in trajectory_sampling"):
                load_and_validate_config(config_path)

    def test_load_and_validate_config_rejects_legacy_generation_shard_size_key(self) -> None:
        config = deepcopy(_load_base_config())
        config["generation"]["shard_size"] = 4096
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigValidationError, "Unexpected keys in generation"):
                load_and_validate_config(config_path)

    def test_load_and_validate_config_rejects_single_snapshot_cap(self) -> None:
        config = deepcopy(_load_base_config())
        config["generation"]["max_trajectory_snapshots"] = 1
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                ConfigValidationError,
                "max_trajectory_snapshots must be 0 or >= 2",
            ):
                load_and_validate_config(config_path)

    def test_load_and_validate_config_rejects_non_log_standard_anchor_normalization(self) -> None:
        config = deepcopy(_load_base_config())
        config["normalization"]["sequence_methods"]["anchor_ymix"] = "standard"
        config["normalization"]["target_method"] = "standard"
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                ConfigValidationError,
                "anchor_ymix must be 'log-standard'",
            ):
                load_and_validate_config(config_path)

    def test_load_and_validate_config_rejects_non_log_standard_target_normalization(self) -> None:
        config = deepcopy(_load_base_config())
        config["normalization"]["target_method"] = "none"
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(
                ConfigValidationError,
                "target_method must be 'log-standard'",
            ):
                load_and_validate_config(config_path)

    def test_load_and_validate_config_rejects_dropout_above_one(self) -> None:
        config = deepcopy(_load_base_config())
        config["training"]["model"]["dropout"] = 1.5
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigValidationError, "training.model.dropout must be in \\[0, 1\\]"):
                load_and_validate_config(config_path)

    def test_load_and_validate_config_requires_training_loss(self) -> None:
        config = deepcopy(_load_base_config())
        config["training"].pop("loss")
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigValidationError, "Missing required keys in training"):
                load_and_validate_config(config_path)

    def test_load_and_validate_config_rejects_negative_loss_weight(self) -> None:
        config = deepcopy(_load_base_config())
        config["training"]["loss"]["lambda_phys"] = -1.0
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigValidationError, "training.loss.lambda_phys must be >= 0"):
                load_and_validate_config(config_path)

    def test_load_and_validate_config_rejects_unexpected_training_model_keys(self) -> None:
        config = deepcopy(_load_base_config())
        config["training"]["model"]["legacy_width"] = 123
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigValidationError, "Unexpected keys in training.model"):
                load_and_validate_config(config_path)

    def test_load_and_validate_config_rejects_legacy_training_loader_keys(self) -> None:
        config = deepcopy(_load_base_config())
        config["training"]["gpu_preload"] = False
        config["training"]["num_workers"] = 0
        config["training"]["data_loading"] = {"mode": "ram"}
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigValidationError, "Unexpected keys in training"):
                load_and_validate_config(config_path)


if __name__ == "__main__":
    unittest.main()
