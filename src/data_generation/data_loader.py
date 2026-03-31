"""Processed dataset loading, split management, and batch assembly.

This module sits between the on-disk processed tensors (produced by
``preprocess.py``) and the training loop (``trainer.py``).  It provides:

* **Split dataclasses** — ``FullVulcanSplit`` and ``EquilibriumSplit`` —
  that hold all NumPy arrays for one data partition in memory.
* **Batch assembly** — ``build_full_vulcan_batch`` and
  ``build_equilibrium_batch`` construct index-based batches.
* **Iteration helpers** — ``iter_full_vulcan_batches`` and
  ``iter_equilibrium_batches`` chunk run indices into fixed-size
  batches suitable for one gradient step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FullVulcanSplit:
    """Processed full-VULCAN split with normalized final-state arrays.

    All arrays are already normalized by the training-set statistics
    stored in ``normalization.json``.  Array shapes:

    * ``sequence_inputs``  — ``(num_runs, nz, 3)`` (P, T, Kzz)
    * ``target_outputs``   — ``(num_runs, nz, target_dim)``
    * ``global_inputs``    — ``(num_runs, global_dim)``
    * ``spectrum_inputs``  — ``(num_runs, spectrum_dim)``
    """
    name: str
    sequence_inputs: np.ndarray
    target_outputs: np.ndarray
    global_inputs: np.ndarray
    spectrum_inputs: np.ndarray
    run_ids: list[str]
    metadata: dict[str, Any]

    @property
    def num_runs(self) -> int:
        """Return the number of runs stored in this split."""
        return int(self.sequence_inputs.shape[0])


@dataclass(frozen=True)
class EquilibriumSplit:
    """Processed split for equilibrium models (no trajectory/spectrum).

    Simpler than ``FullVulcanSplit`` — only pressure/temperature
    sequence inputs, equilibrium mixing-ratio targets, and global
    conditioning scalars.
    """
    name: str
    sequence_inputs: np.ndarray   # [num_runs, nz, 2] (P, T)
    target_outputs: np.ndarray    # [num_runs, nz, target_dim]
    global_inputs: np.ndarray     # [num_runs, global_dim]
    run_ids: list[str]
    metadata: dict[str, Any]

    @property
    def num_runs(self) -> int:
        """Return the number of runs stored in this split."""
        return int(self.sequence_inputs.shape[0])


def load_full_vulcan_split(split_dir: str | Path) -> FullVulcanSplit:
    """Load one processed full-VULCAN split from disk."""
    split_path = Path(split_dir)
    metadata = json.loads((split_path / "metadata.json").read_text(encoding="utf-8"))
    run_ids = json.loads((split_path / "run_ids.json").read_text(encoding="utf-8"))
    return FullVulcanSplit(
        name=split_path.name,
        sequence_inputs=np.load(split_path / "sequence_inputs.npy"),
        target_outputs=np.load(split_path / "target_outputs.npy"),
        global_inputs=np.load(split_path / "global_inputs.npy"),
        spectrum_inputs=np.load(split_path / "spectrum_inputs.npy"),
        run_ids=list(run_ids),
        metadata=metadata,
    )


def load_equilibrium_split(split_dir: str | Path) -> EquilibriumSplit:
    """Load a processed equilibrium split (no trajectory/spectrum arrays)."""
    split_path = Path(split_dir)
    metadata = json.loads((split_path / "metadata.json").read_text(encoding="utf-8"))
    run_ids = json.loads((split_path / "run_ids.json").read_text(encoding="utf-8"))
    return EquilibriumSplit(
        name=split_path.name,
        sequence_inputs=np.load(split_path / "sequence_inputs.npy"),
        target_outputs=np.load(split_path / "target_outputs.npy"),
        global_inputs=np.load(split_path / "global_inputs.npy"),
        run_ids=list(run_ids),
        metadata=metadata,
    )


def load_full_vulcan_dataset(
    processed_root: str | Path,
) -> tuple[dict[str, FullVulcanSplit], dict[str, Any], dict[str, Any]]:
    """Load all processed full-VULCAN splits plus normalization and contract metadata."""
    root = Path(processed_root)
    splits = {
        name: load_full_vulcan_split(root / name)
        for name in ("train", "val", "test")
        if (root / name / "metadata.json").exists()
    }
    normalization = json.loads((root / "normalization.json").read_text(encoding="utf-8"))
    contract = json.loads((root / "data_contract.json").read_text(encoding="utf-8"))
    return splits, normalization, contract


def load_equilibrium_dataset(
    processed_root: str | Path,
) -> tuple[dict[str, EquilibriumSplit], dict[str, Any], dict[str, Any]]:
    """Load all equilibrium splits, normalization, and data contract."""
    root = Path(processed_root)
    splits = {
        name: load_equilibrium_split(root / name)
        for name in ("train", "val", "test")
        if (root / name / "metadata.json").exists()
    }
    normalization = json.loads((root / "normalization.json").read_text(encoding="utf-8"))
    contract = json.loads((root / "data_contract.json").read_text(encoding="utf-8"))
    return splits, normalization, contract


def build_full_vulcan_batch(
    split: FullVulcanSplit,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build a batch for the full-VULCAN model."""
    idx = np.asarray(indices, dtype=np.int32)
    return {
        "sequence": split.sequence_inputs[idx].astype(np.float32),
        "global_inputs": split.global_inputs[idx].astype(np.float32),
        "spectrum_inputs": split.spectrum_inputs[idx].astype(np.float32),
        "target": split.target_outputs[idx].astype(np.float32),
    }


def build_equilibrium_batch(
    split: EquilibriumSplit,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build a batch for the equilibrium model (no anchor/dt/spectrum)."""
    idx = np.asarray(indices, dtype=np.int32)
    return {
        "sequence": split.sequence_inputs[idx].astype(np.float32),
        "global_inputs": split.global_inputs[idx].astype(np.float32),
        "target": split.target_outputs[idx].astype(np.float32),
    }


def iter_full_vulcan_batches(
    split: FullVulcanSplit,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> list[dict[str, np.ndarray]]:
    """Return shuffled batches for one epoch of full-VULCAN training.

    Run indices are shuffled in-place using ``rng`` before chunking,
    so each epoch sees a different batch composition.
    """
    indices = np.arange(split.num_runs, dtype=np.int32)
    rng.shuffle(indices)
    batches: list[dict[str, np.ndarray]] = []
    for start in range(0, indices.size, int(batch_size)):
        stop = min(start + int(batch_size), indices.size)
        batches.append(build_full_vulcan_batch(split, indices[start:stop]))
    return batches


def iter_equilibrium_batches(
    split: EquilibriumSplit,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> list[dict[str, np.ndarray]]:
    """Return shuffled batches for one epoch of equilibrium training.

    Run indices are shuffled in-place using ``rng`` before chunking,
    so each epoch sees a different batch composition.
    """
    indices = np.arange(split.num_runs, dtype=np.int32)
    rng.shuffle(indices)
    batches: list[dict[str, np.ndarray]] = []
    for start in range(0, indices.size, int(batch_size)):
        stop = min(start + int(batch_size), indices.size)
        batches.append(build_equilibrium_batch(split, indices[start:stop]))
    return batches
