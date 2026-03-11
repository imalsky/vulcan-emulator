#!/usr/bin/env python3
"""Inference tests for the physical-space predictor wrapper."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import (
    PhysicalSpaceStandaloneModel,
    load_physical_space_model,
)
from model import VulcanTransitionTransformer


class InferenceTests(unittest.TestCase):
    """End-to-end inference tests for the standalone predictor wrapper."""

    def test_checkpoint_can_rebuild_wrapper_without_sidecar_json(self) -> None:
        model = VulcanTransitionTransformer(
            state_dim=2,
            output_dim=2,
            output_from_state_indices=[0, 1],
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            dropout=0.0,
            film_clamp=10.0,
            output_head_divisor=2,
            max_sequence_length=8,
            conditioning_hidden_dim=8,
        ).eval()
        normalization_metadata = {
            "epsilon": 1e-30,
            "sequence": {
                "pressure_bar": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "temperature_k": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "kzz_cm2_s": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "anchor_ymix": {"method": "none", "mean": [0.0, 0.0], "std": [1.0, 1.0], "min": [0.0, 0.0], "max": [1.0, 1.0]},
            },
            "globals": {
                "gravity_cm_s2": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "metallicity_log10": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "c_to_o": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "log10_dt_s": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
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
            "state_dim": 2,
            "sequence_feature_order": [
                "pressure_bar",
                "temperature_k",
                "kzz_cm2_s",
                "anchor_ymix:H2",
                "anchor_ymix:He",
            ],
            "global_feature_order": ["gravity_cm_s2", "metallicity_log10", "c_to_o", "log10_dt_s"],
            "state_species_order": ["H2", "He"],
            "output_species_order": ["H2", "He"],
            "output_from_state_indices": [0, 1],
            "normalization_fingerprint": "synthetic",
        }
        config = {
            "training": {
                "model": {
                    "d_model": 8,
                    "nhead": 2,
                    "num_layers": 1,
                    "dim_feedforward": 16,
                    "dropout": 0.0,
                    "film_clamp": 10.0,
                    "output_head_divisor": 2,
                    "max_sequence_length": 8,
                    "conditioning_hidden_dim": 8,
                },
            },
            "precision": {
                "model_dtype": "float32",
                "forward_dtype": "float32",
            },
            "physics_toggles": {
                "use_eddy_diffusion": True,
                "use_molecular_diffusion": True,
                "use_upwind_molecular_diffusion": False,
                "use_boundary_conditions": False,
                "use_condensation": False,
                "use_settling": False,
                "use_initial_cold_trap": True,
                "use_sat_surface_h2o": True,
                "use_lowT_limit_rates": False,
                "use_adaptive_rtol": True,
            },
            "vulcan_runtime": {
                "atm_base": "H2",
            },
        }

        with tempfile.TemporaryDirectory(prefix="ve_inference_checkpoint_") as tmpdir_name:
            run_dir = Path(tmpdir_name)
            torch.save(
                {
                    "config": config,
                    "model_state": model.state_dict(),
                    "data_contract": data_contract,
                    "normalization_metadata": normalization_metadata,
                },
                run_dir / "best.pt",
            )

            wrapper, loaded_norm, loaded_contract = load_physical_space_model(run_dir)

        self.assertIsInstance(wrapper, PhysicalSpaceStandaloneModel)
        self.assertEqual(loaded_contract["sequence_length"], 4)
        self.assertEqual(loaded_norm["targets"]["ymix"]["method"], "none")
        prediction = wrapper(
            torch.full((1, 4), 1.0, dtype=torch.float32),
            torch.full((1, 4), 1000.0, dtype=torch.float32),
            torch.full((1, 4), 1.0e9, dtype=torch.float32),
            torch.full((1, 4, 2), 0.5, dtype=torch.float32),
            torch.tensor([1000.0], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0.5], dtype=torch.float32),
            torch.tensor([1.0e3], dtype=torch.float32),
        )
        self.assertEqual(tuple(prediction.shape), (1, 4, 2))

    def test_standalone_wrapper_exports_with_physical_inputs(self) -> None:
        model = VulcanTransitionTransformer(
            state_dim=2,
            output_dim=2,
            output_from_state_indices=[0, 1],
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            dropout=0.0,
            film_clamp=10.0,
            output_head_divisor=2,
            max_sequence_length=8,
            conditioning_hidden_dim=8,
        ).eval()
        normalization_metadata = {
            "epsilon": 1e-30,
            "sequence": {
                "pressure_bar": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "temperature_k": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "kzz_cm2_s": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "anchor_ymix": {"method": "none", "mean": [0.0, 0.0], "std": [1.0, 1.0], "min": [0.0, 0.0], "max": [1.0, 1.0]},
            },
            "globals": {
                "gravity_cm_s2": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "metallicity_log10": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "c_to_o": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "log10_dt_s": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
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
            "state_dim": 2,
            "sequence_feature_order": [
                "pressure_bar",
                "temperature_k",
                "kzz_cm2_s",
                "anchor_ymix:H2",
                "anchor_ymix:He",
            ],
            "global_feature_order": ["gravity_cm_s2", "metallicity_log10", "c_to_o", "log10_dt_s"],
            "state_species_order": ["H2", "He"],
            "output_species_order": ["H2", "He"],
            "output_from_state_indices": [0, 1],
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
        with tempfile.TemporaryDirectory(prefix="ve_exported_wrapper_") as tmpdir_name:
            export_path = Path(tmpdir_name) / "standalone_model.pt2"
            torch.export.save(exported, str(export_path))
            loaded = torch.export.load(str(export_path)).module()
            with torch.inference_mode():
                candidate = loaded(*example_inputs)
            torch.testing.assert_close(reference, candidate, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
