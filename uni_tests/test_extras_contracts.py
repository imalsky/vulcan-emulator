#!/usr/bin/env python3
"""Unit tests for local helper scripts."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_training_progression_module():
    module_path = PROJECT_ROOT / "extras" / "training_progression.py"
    spec = importlib.util.spec_from_file_location("training_progression", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


if __name__ == "__main__":
    unittest.main()
