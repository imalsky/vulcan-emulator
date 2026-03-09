#!/usr/bin/env python3
"""Fast unit tests for processed trajectory splits and live-sampling contracts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import DataLoadingError, load_processed_split_arrays, load_split_metadata
from live_sampling import ProcessedTrajectoryStore


def _write_processed_split(split_dir: Path) -> tuple[dict, dict]:
    static_inputs = np.array(
        [
            [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]],
            [[3.0, 30.0, 300.0], [4.0, 40.0, 400.0]],
        ],
        dtype=np.float32,
    )
    state_ymix = np.array(
        [
            [
                [[0.10, 0.20], [0.11, 0.21]],
                [[0.12, 0.22], [0.13, 0.23]],
                [[0.14, 0.24], [0.15, 0.25]],
                [[0.16, 0.26], [0.17, 0.27]],
            ],
            [
                [[0.30, 0.40], [0.31, 0.41]],
                [[0.32, 0.42], [0.33, 0.43]],
                [[0.34, 0.44], [0.35, 0.45]],
                [[0.36, 0.46], [0.37, 0.47]],
            ],
        ],
        dtype=np.float32,
    )
    global_inputs = np.array(
        [
            [1.0, 0.0, 0.55],
            [2.0, 0.1, 0.65],
        ],
        dtype=np.float32,
    )
    time_s = np.array(
        [
            [0.0, 10.0, 20.0, 40.0],
            [0.0, 5.0, 15.0, 25.0],
        ],
        dtype=np.float64,
    )
    valid_steps_mask = np.ones((2, 4), dtype=bool)
    run_ids = np.array([11, 22], dtype=np.int64)

    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "static_inputs.npy", static_inputs, allow_pickle=False)
    np.save(split_dir / "state_ymix.npy", state_ymix, allow_pickle=False)
    np.save(split_dir / "global_inputs.npy", global_inputs, allow_pickle=False)
    np.save(split_dir / "time_s.npy", time_s, allow_pickle=False)
    np.save(split_dir / "valid_steps_mask.npy", valid_steps_mask, allow_pickle=False)
    np.save(split_dir / "run_ids.npy", run_ids, allow_pickle=False)

    metadata = {
        "split": "train",
        "num_runs": 2,
        "max_steps": 4,
        "total_valid_candidates": 12,
        "sequence_length": 2,
        "input_dim": 5,
        "global_dim": 4,
        "global_static_dim": 3,
        "dt_feature_index": 3,
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
        "global_static_feature_order": [
            "gravity_cm_s2",
            "metallicity_log10",
            "c_to_o",
        ],
        "state_species_order": ["H2", "He"],
        "output_species_order": ["H2", "He"],
        "output_from_state_indices": [0, 1],
        "normalization_fingerprint": "a" * 64,
        "sampling_mode": "log_uniform_all_pairs",
        "dt_sampling_min_s": 5.0,
        "dt_sampling_max_s": 40.0,
        "min_future_saved_steps": 1,
        "dt_min_s": 5.0,
        "dt_max_s": 40.0,
    }
    (split_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    normalization_metadata = {
        "globals": {
            "log10_dt_s": {
                "method": "standard",
                "mean": [1.0],
                "std": [0.5],
                "min": [0.0],
                "max": [2.0],
            }
        }
    }
    return metadata, normalization_metadata


def _live_sampling_config() -> dict:
    return {
        "trajectory_sampling": {
            "dt_min_s": 5.0,
            "dt_max_s": 40.0,
            "min_future_saved_steps": 1,
        }
    }


class DataLoaderContractTests(unittest.TestCase):
    """Unit tests for processed split metadata validation and live batch assembly."""

    def test_load_split_metadata_rejects_feature_order_length_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_loader_meta_") as tmpdir_name:
            split_dir = Path(tmpdir_name) / "train"
            metadata, _norm = _write_processed_split(split_dir)
            metadata["global_static_dim"] = 4
            (split_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(
                DataLoadingError,
                "Invalid global static feature order length",
            ):
                load_split_metadata(split_dir)

    def test_load_processed_split_arrays_returns_expected_shapes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_loader_arrays_") as tmpdir_name:
            split_dir = Path(tmpdir_name) / "train"
            metadata, _norm = _write_processed_split(split_dir)
            arrays = load_processed_split_arrays(split_dir)
            self.assertEqual(arrays.static_inputs.shape, (2, 2, 3))
            self.assertEqual(arrays.state_ymix.shape, (2, 4, 2, 2))
            self.assertEqual(arrays.global_inputs.shape, (2, 3))
            self.assertEqual(arrays.time_s.shape, (2, 4))
            self.assertEqual(arrays.valid_steps_mask.shape, (2, 4))
            self.assertEqual(arrays.run_ids.shape, (2,))
            self.assertEqual(arrays.metadata["total_valid_candidates"], metadata["total_valid_candidates"])

    def test_fixed_pair_selection_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_loader_fixed_") as tmpdir_name:
            split_dir = Path(tmpdir_name) / "train"
            _metadata, normalization_metadata = _write_processed_split(split_dir)
            store = ProcessedTrajectoryStore.from_split_dir(
                split_dir=split_dir,
                normalization_metadata=normalization_metadata,
                config=_live_sampling_config(),
                device=torch.device("cpu"),
                tensor_dtype=torch.float32,
            )
            first = store.select_fixed_candidate_indices(pairs_per_run=2, seed=1234)
            second = store.select_fixed_candidate_indices(pairs_per_run=2, seed=1234)
            torch.testing.assert_close(first, second)

    def test_train_pair_selection_changes_across_seeds(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_loader_train_") as tmpdir_name:
            split_dir = Path(tmpdir_name) / "train"
            _metadata, normalization_metadata = _write_processed_split(split_dir)
            store = ProcessedTrajectoryStore.from_split_dir(
                split_dir=split_dir,
                normalization_metadata=normalization_metadata,
                config=_live_sampling_config(),
                device=torch.device("cpu"),
                tensor_dtype=torch.float32,
            )
            first = store.select_train_candidate_indices(pairs_per_run=2, seed=1)
            second = store.select_train_candidate_indices(pairs_per_run=2, seed=2)
            self.assertFalse(torch.equal(first, second))

    def test_build_batch_matches_output_subset_and_global_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_loader_batch_") as tmpdir_name:
            split_dir = Path(tmpdir_name) / "train"
            _metadata, normalization_metadata = _write_processed_split(split_dir)
            store = ProcessedTrajectoryStore.from_split_dir(
                split_dir=split_dir,
                normalization_metadata=normalization_metadata,
                config=_live_sampling_config(),
                device=torch.device("cpu"),
                tensor_dtype=torch.float32,
            )
            selected = store.select_fixed_candidate_indices(pairs_per_run=2, seed=7)
            seq, glb, tgt, mask, dt = store.build_batch(selected[:2])
            self.assertEqual(tuple(seq.shape), (2, 2, 5))
            self.assertEqual(tuple(glb.shape), (2, 4))
            self.assertEqual(tuple(tgt.shape), (2, 2, 2))
            self.assertEqual(tuple(mask.shape), (2, 2))
            self.assertEqual(tuple(dt.shape), (2,))
            self.assertTrue(torch.equal(mask, torch.zeros_like(mask)))

            run_idx = store.candidates.run_index[selected[:2]]
            target_idx = store.candidates.target_index[selected[:2]]
            expected_tgt = store.state_ymix[run_idx, target_idx]
            torch.testing.assert_close(tgt, expected_tgt)
            torch.testing.assert_close(glb[:, -1], store.candidates.normalized_log10_dt_s[selected[:2]])


if __name__ == "__main__":
    unittest.main()
