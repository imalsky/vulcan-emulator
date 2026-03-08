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

BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "tiny_train_smoke.json"


def _load_base_config() -> dict[str, Any]:
    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class ConfigUtilsTests(unittest.TestCase):
    """Unit tests for config parsing and precision policy validation."""

    def test_shipped_smoke_config_loads_and_resolves_precision(self) -> None:
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

    def test_load_and_validate_config_accepts_continue_on_error_failure_policy(self) -> None:
        config = deepcopy(_load_base_config())
        config["generation"]["failure_policy"] = "continue_on_error"
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            loaded = load_and_validate_config(config_path)

        self.assertEqual(loaded["generation"]["failure_policy"], "continue_on_error")

    def test_load_and_validate_config_allows_generation_without_runs_root(self) -> None:
        config = deepcopy(_load_base_config())
        config["generation"].pop("runs_root", None)
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            loaded = load_and_validate_config(config_path)

        self.assertNotIn("runs_root", loaded["generation"])

    def test_load_and_validate_config_rejects_deprecated_runs_root(self) -> None:
        config = deepcopy(_load_base_config())
        config["generation"]["runs_root"] = "data/raw/runs"
        with tempfile.TemporaryDirectory(prefix="ve_cfg_unit_") as tmpdir_name:
            config_path = Path(tmpdir_name) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigValidationError, "generation.runs_root is no longer supported"):
                load_and_validate_config(config_path)


if __name__ == "__main__":
    unittest.main()
