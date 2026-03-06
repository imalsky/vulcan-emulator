#!/usr/bin/env python3
"""Fast unit tests for processed-split metadata and loader contracts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import DataLoadingConfig, DataLoadingError, build_training_loader, load_split_metadata


def _write_processed_split(split_dir: Path) -> dict[str, Any]:
    """Write a small valid processed split for loader unit tests."""
    sequence_shards = [
        np.array(
            [
                [[1.0, 10.0, 100.0, 0.1], [2.0, 20.0, 200.0, 0.2]],
                [[3.0, 30.0, 300.0, 0.3], [4.0, 40.0, 400.0, 0.4]],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [[5.0, 50.0, 500.0, 0.5], [6.0, 60.0, 600.0, 0.6]],
            ],
            dtype=np.float32,
        ),
    ]
    global_shards = [
        np.array([[1.0, 0.1, 0.5], [2.0, 0.2, 0.6]], dtype=np.float32),
        np.array([[3.0, 0.3, 0.7]], dtype=np.float32),
    ]
    target_shards = [
        np.array(
            [
                [[0.01, 0.02], [0.03, 0.04]],
                [[0.05, 0.06], [0.07, 0.08]],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [[0.09, 0.10], [0.11, 0.12]],
            ],
            dtype=np.float32,
        ),
    ]

    seq_dir = split_dir / "sequence_inputs"
    glb_dir = split_dir / "globals"
    tgt_dir = split_dir / "targets"
    for directory in (seq_dir, glb_dir, tgt_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for shard_idx, (seq, glb, tgt) in enumerate(
        zip(sequence_shards, global_shards, target_shards, strict=True)
    ):
        np.save(seq_dir / f"shard_{shard_idx:05d}.npy", seq, allow_pickle=False)
        np.save(glb_dir / f"shard_{shard_idx:05d}.npy", glb, allow_pickle=False)
        np.save(tgt_dir / f"shard_{shard_idx:05d}.npy", tgt, allow_pickle=False)

    metadata = {
        "split": "train",
        "num_shards": 2,
        "sequence_length": 2,
        "input_dim": 4,
        "global_dim": 3,
        "target_dim": 2,
        "total_samples": 3,
        "sequence_feature_order": [
            "pressure_bar",
            "temperature_k",
            "kzz_cm2_s",
            "initial_ymix:H2",
        ],
        "global_feature_order": [
            "gravity_cm_s2",
            "metallicity_log10",
            "c_to_o",
        ],
        "target_species_order": ["H2", "He"],
        "normalization_fingerprint": "a" * 64,
    }
    (split_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def _loading_config(mode: str) -> DataLoadingConfig:
    """Build a small deterministic loading policy for unit tests."""
    return DataLoadingConfig(
        mode=mode,
        max_cached_shards=2,
        large_shard_mmap_bytes=1,
        ram_safety_fraction=0.5,
        copy_mmap_slices=True,
        use_device_prefetch=False,
    )


class DataLoaderContractTests(unittest.TestCase):
    """Unit tests for processed-split metadata validation and batch contracts."""

    def test_load_split_metadata_rejects_feature_order_length_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_loader_meta_") as tmpdir_name:
            split_dir = Path(tmpdir_name) / "train"
            metadata = _write_processed_split(split_dir)
            metadata["input_dim"] = 5
            (split_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(
                DataLoadingError,
                "Invalid sequence feature order length",
            ):
                load_split_metadata(split_dir)

    def test_build_training_loader_ram_emits_expected_batch_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_loader_ram_") as tmpdir_name:
            split_dir = Path(tmpdir_name) / "train"
            metadata = _write_processed_split(split_dir)

            loader, loaded_metadata = build_training_loader(
                split_dir=split_dir,
                batch_size=2,
                shuffle=False,
                device=torch.device("cpu"),
                preload_to_device=False,
                num_workers=0,
                input_dtype=torch.float32,
                target_dtype=torch.float64,
                loading=_loading_config("ram"),
            )

            seq, glb, tgt, mask = next(iter(loader))
            self.assertEqual(loaded_metadata["total_samples"], metadata["total_samples"])
            self.assertEqual(tuple(seq.shape), (2, 2, 4))
            self.assertEqual(tuple(glb.shape), (2, 3))
            self.assertEqual(tuple(tgt.shape), (2, 2, 2))
            self.assertEqual(tuple(mask.shape), (2, 2))
            self.assertEqual(seq.dtype, torch.float32)
            self.assertEqual(glb.dtype, torch.float32)
            self.assertEqual(tgt.dtype, torch.float64)
            self.assertEqual(mask.dtype, torch.bool)

    def test_build_training_loader_disk_emits_expected_batch_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ve_loader_disk_") as tmpdir_name:
            split_dir = Path(tmpdir_name) / "train"
            _write_processed_split(split_dir)

            loader, _metadata = build_training_loader(
                split_dir=split_dir,
                batch_size=2,
                shuffle=False,
                device=torch.device("cpu"),
                preload_to_device=False,
                num_workers=0,
                input_dtype=torch.float32,
                target_dtype=torch.float32,
                loading=_loading_config("disk"),
            )

            total_samples = 0
            for seq, glb, tgt, mask in loader:
                total_samples += int(seq.shape[0])
                self.assertEqual(tuple(seq.shape[1:]), (2, 4))
                self.assertEqual(tuple(glb.shape[1:]), (3,))
                self.assertEqual(tuple(tgt.shape[1:]), (2, 2))
                self.assertEqual(tuple(mask.shape[1:]), (2,))
                self.assertEqual(mask.dtype, torch.bool)

            self.assertEqual(total_samples, 3)


if __name__ == "__main__":
    unittest.main()
