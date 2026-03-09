#!/usr/bin/env python3
"""Unit tests for physical-space transition inference contracts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import PhysicalSpaceStandaloneModel
from model import VulcanTransitionTransformer


def _build_wrapper() -> PhysicalSpaceStandaloneModel:
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
            "anchor_ymix": {"method": "none", "mean": [0.0, 0.0], "std": [1.0, 1.0], "min": [0.0, 0.0], "max": [1.0, 1.0]}
        },
        "globals": {
            "gravity_cm_s2": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
            "metallicity_log10": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
            "c_to_o": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
            "log10_dt_s": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]}
        },
        "targets": {
            "ymix": {"method": "none", "mean": [0.0, 0.0], "std": [1.0, 1.0], "min": [0.0, 0.0], "max": [1.0, 1.0]}
        }
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
        "global_feature_order": [
            "gravity_cm_s2",
            "metallicity_log10",
            "c_to_o",
            "log10_dt_s",
        ],
        "state_species_order": ["H2", "He"],
        "output_species_order": ["H2", "He"],
        "output_from_state_indices": [0, 1],
        "normalization_fingerprint": "synthetic",
    }
    return PhysicalSpaceStandaloneModel(
        model=model,
        normalization_metadata=normalization_metadata,
        data_contract=data_contract,
    ).eval()


class InferenceContractTests(unittest.TestCase):
    """Unit tests for physical-space wrapper input validation."""

    def test_wrapper_rejects_nonpositive_dt_conditioning(self) -> None:
        wrapper = _build_wrapper()
        with self.assertRaisesRegex(ValueError, "dt_s must be strictly positive"):
            wrapper(
                torch.full((1, 4), 1.0, dtype=torch.float32),
                torch.full((1, 4), 1000.0, dtype=torch.float32),
                torch.full((1, 4), 1.0e9, dtype=torch.float32),
                torch.full((1, 4, 2), 0.5, dtype=torch.float32),
                torch.tensor([1000.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([0.5], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
            )

    def test_wrapper_rejects_sequence_length_mismatch(self) -> None:
        wrapper = _build_wrapper()
        with self.assertRaisesRegex(ValueError, "Expected sequence length 4"):
            wrapper(
                torch.full((1, 3), 1.0, dtype=torch.float32),
                torch.full((1, 3), 1000.0, dtype=torch.float32),
                torch.full((1, 3), 1.0e9, dtype=torch.float32),
                torch.full((1, 3, 2), 0.5, dtype=torch.float32),
                torch.tensor([1000.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([0.5], dtype=torch.float32),
                torch.tensor([1.0], dtype=torch.float32),
            )

    def test_wrapper_accepts_additional_conditioning_globals(self) -> None:
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
            num_globals=5,
        ).eval()
        normalization_metadata = {
            "epsilon": 1e-30,
            "sequence": {
                "pressure_bar": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "temperature_k": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "kzz_cm2_s": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "anchor_ymix": {"method": "none", "mean": [0.0, 0.0], "std": [1.0, 1.0], "min": [0.0, 0.0], "max": [1.0, 1.0]}
            },
            "globals": {
                "gravity_cm_s2": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "metallicity_log10": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "c_to_o": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "log10_dt_s": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
                "use_condensation": {"method": "none", "mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]},
            },
            "targets": {
                "ymix": {"method": "none", "mean": [0.0, 0.0], "std": [1.0, 1.0], "min": [0.0, 0.0], "max": [1.0, 1.0]}
            }
        }
        data_contract = {
            "sequence_length": 4,
            "input_dim": 5,
            "global_dim": 5,
            "target_dim": 2,
            "state_dim": 2,
            "sequence_feature_order": [
                "pressure_bar",
                "temperature_k",
                "kzz_cm2_s",
                "anchor_ymix:H2",
                "anchor_ymix:He",
            ],
            "global_feature_order": [
                "gravity_cm_s2",
                "metallicity_log10",
                "c_to_o",
                "log10_dt_s",
                "use_condensation",
            ],
            "state_species_order": ["H2", "He"],
            "output_species_order": ["H2", "He"],
            "output_from_state_indices": [0, 1],
            "normalization_fingerprint": "synthetic",
        }
        wrapper = PhysicalSpaceStandaloneModel(
            model=model,
            normalization_metadata=normalization_metadata,
            data_contract=data_contract,
            default_global_inputs={"use_condensation": 1.0},
        ).eval()
        prediction = wrapper(
            torch.full((1, 4), 1.0, dtype=torch.float32),
            torch.full((1, 4), 1000.0, dtype=torch.float32),
            torch.full((1, 4), 1.0e9, dtype=torch.float32),
            torch.full((1, 4, 2), 0.5, dtype=torch.float32),
            torch.tensor([1000.0], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            torch.tensor([0.5], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
        )
        self.assertEqual(tuple(prediction.shape), (1, 4, 2))


if __name__ == "__main__":
    unittest.main()
