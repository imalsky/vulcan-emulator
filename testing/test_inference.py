#!/usr/bin/env python3
"""Inference tests for the physical-space predictor wrapper."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

# Prevent duplicate OpenMP runtime aborts before importing torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import iter_split_shards
from inference import (
    PhysicalSpaceStandaloneModel,
    VulcanPredictor,
    load_physical_space_model,
    physical_inputs_from_processed_arrays,
)
from model import VulcanTransformer

RUN_DIR = PROJECT_ROOT / "models" / "tiny_e2e_smoke"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "smoke" / "processed"


class InferenceTests(unittest.TestCase):
    """End-to-end inference tests for the standalone predictor wrapper."""

    def test_standalone_wrapper_exports_with_physical_inputs(self) -> None:
        model = VulcanTransformer(
            input_dim=5,
            global_dim=4,
            target_dim=2,
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            dropout=0.0,
            film_clamp=10.0,
            output_head_divisor=2,
            max_sequence_length=8,
        ).eval()
        normalization_metadata = {
            "epsilon": 1e-30,
            "sequence": {
                "pressure_bar": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "temperature_k": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "kzz_cm2_s": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "initial_ymix": {"method": "none", "mean": [0.0, 0.0], "std": [1.0, 1.0], "min": [0.0, 0.0], "max": [1.0, 1.0]},
            },
            "globals": {
                "gravity_cm_s2": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "metallicity_log10": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "c_to_o": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "log10_time_s": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
            },
            "targets": {
                "ymix": {"method": "none", "mean": [0.0, 0.0], "std": [1.0, 1.0], "min": [0.0, 0.0], "max": [1.0, 1.0]},
            },
        }
        data_contract = {
            "sequence_length": 4,
            "input_dim": 5,
            "global_dim": 4,
            "target_dim": 2,
            "sequence_feature_order": [
                "pressure_bar",
                "temperature_k",
                "kzz_cm2_s",
                "initial_ymix:H2",
                "initial_ymix:He",
            ],
            "global_feature_order": ["gravity_cm_s2", "metallicity_log10", "c_to_o", "log10_time_s"],
            "target_species_order": ["H2", "He"],
            "normalization_fingerprint": "synthetic",
        }
        wrapper = PhysicalSpaceStandaloneModel(
            model=model,
            normalization_metadata=normalization_metadata,
            data_contract=data_contract,
        ).eval()
        example_inputs = (
            torch.full((1, 4), 1.0, dtype=torch.float32),
            torch.full((1, 4), 1000.0, dtype=torch.float32),
            torch.full((1, 4), 1.0e9, dtype=torch.float32),
            torch.full((1, 4, 2), 0.5, dtype=torch.float32),
            torch.tensor([1000.0], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0.5], dtype=torch.float32),
            torch.tensor([1.0e3], dtype=torch.float32),
        )

        with torch.inference_mode():
            exported = torch.export.export(wrapper, args=example_inputs, strict=True)
            reference = wrapper(*example_inputs)
        loaded = exported.module()
        with torch.inference_mode():
            candidate = loaded(*example_inputs)
        torch.testing.assert_close(reference, candidate, rtol=1e-4, atol=1e-5)

    def test_predictor_returns_finite_physical_outputs(self) -> None:
        if not (RUN_DIR / "data_contract.json").is_file():
            self.skipTest("Run testing/smoke_pipeline.py to create the tiny_e2e_smoke artifacts first.")

        model, normalization_metadata, data_contract = load_physical_space_model(RUN_DIR)
        seq, glb, _tgt = next(iter(iter_split_shards(processed_root=PROCESSED_ROOT, split="test")))
        example = physical_inputs_from_processed_arrays(
            sequence_inputs=seq[0],
            global_inputs=glb[0],
            normalization_metadata=normalization_metadata,
            data_contract=data_contract,
        )
        predictor = VulcanPredictor.from_run_dir(RUN_DIR)
        prediction = predictor.predict(
            pressure_bar=example["pressure_bar"],
            temperature_k=example["temperature_k"],
            kzz_cm2_s=example["kzz_cm2_s"],
            initial_ymix=example["initial_ymix"],
            gravity_cm_s2=example["gravity_cm_s2"],
            metallicity_log10=example["metallicity_log10"],
            c_to_o=example["c_to_o"],
            time_s=example["time_s"],
        )

        self.assertEqual(prediction.shape, (seq.shape[1], data_contract["target_dim"]))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(len(model.target_species), data_contract["target_dim"])


if __name__ == "__main__":
    unittest.main()
