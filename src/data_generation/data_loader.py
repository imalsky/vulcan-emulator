"""Processed dataset loading, split management, and batch assembly.

This module sits between the on-disk processed tensors (produced by
``preprocess.py``) and the training loop (``trainer.py``). It provides:

* a single ``ProcessedSplit`` dataclass for any chemistry/model combination
* split loading helpers for the processed train/val/test partitions
* index-based batch assembly with optional spectrum inputs
* epoch batch iteration for the trainer
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProcessedSplit:
    """Processed split with normalized tensors and optional spectrum inputs.

    Array shapes:

    * ``sequence_inputs``  — ``(num_runs, nz, sequence_dim)``
    * ``target_outputs``   — ``(num_runs, nz, target_dim)``
    * ``global_inputs``    — ``(num_runs, global_dim)``
    * ``spectrum_inputs``  — ``(num_runs, spectrum_dim)`` when present
    """

    name: str
    sequence_inputs: np.ndarray
    target_outputs: np.ndarray
    global_inputs: np.ndarray
    spectrum_inputs: np.ndarray | None
    run_ids: list[str]
    metadata: dict[str, Any]

    @property
    def num_runs(self) -> int:
        """Return the number of runs stored in this split."""
        return int(self.sequence_inputs.shape[0])

    @property
    def has_spectrum_inputs(self) -> bool:
        """Return whether this split includes spectrum conditioning arrays."""
        return self.spectrum_inputs is not None


def load_processed_split(split_dir: str | Path) -> ProcessedSplit:
    """Load one processed split from disk."""
    split_path = Path(split_dir)
    metadata = json.loads((split_path / "metadata.json").read_text(encoding="utf-8"))
    run_ids = json.loads((split_path / "run_ids.json").read_text(encoding="utf-8"))
    spectrum_path = split_path / "spectrum_inputs.npy"
    return ProcessedSplit(
        name=split_path.name,
        sequence_inputs=np.load(split_path / "sequence_inputs.npy"),
        target_outputs=np.load(split_path / "target_outputs.npy"),
        global_inputs=np.load(split_path / "global_inputs.npy"),
        spectrum_inputs=np.load(spectrum_path) if spectrum_path.exists() else None,
        run_ids=list(run_ids),
        metadata=metadata,
    )


def load_processed_dataset(
    processed_root: str | Path,
) -> tuple[dict[str, ProcessedSplit], dict[str, Any], dict[str, Any]]:
    """Load all processed splits plus normalization and contract metadata."""
    root = Path(processed_root)
    splits = {
        name: load_processed_split(root / name)
        for name in ("train", "val", "test")
        if (root / name / "metadata.json").exists()
    }
    normalization = json.loads((root / "normalization.json").read_text(encoding="utf-8"))
    contract = json.loads((root / "data_contract.json").read_text(encoding="utf-8"))
    return splits, normalization, contract


def build_batch(
    split: ProcessedSplit,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build a batch for any chemistry/model combination."""
    idx = np.asarray(indices, dtype=np.int32)
    batch = {
        "sequence": split.sequence_inputs[idx].astype(np.float32),
        "global_inputs": split.global_inputs[idx].astype(np.float32),
        "target": split.target_outputs[idx].astype(np.float32),
    }
    if split.spectrum_inputs is not None:
        batch["spectrum_inputs"] = split.spectrum_inputs[idx].astype(np.float32)
    return batch


def iter_batches(
    split: ProcessedSplit,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> list[dict[str, np.ndarray]]:
    """Return shuffled batches for one epoch of training."""
    indices = np.arange(split.num_runs, dtype=np.int32)
    rng.shuffle(indices)
    batches: list[dict[str, np.ndarray]] = []
    for start in range(0, indices.size, int(batch_size)):
        stop = min(start + int(batch_size), indices.size)
        batches.append(build_batch(split, indices[start:stop]))
    return batches
