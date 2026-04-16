"""Processed dataset loading, split management, and batch assembly.

This module sits between the on-disk processed tensors (produced by
``preprocess.py``) and the training loop (``trainer.py``). It provides:

* a single ``ProcessedSplit`` dataclass for any chemistry/model combination
* split loading helpers for the processed train/val/test partitions
* index-based batch assembly
* epoch batch iteration for the trainer
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..constants import PROCESSED_INFO_DIRNAME


def processed_info_dir(processed_root: str | Path) -> Path:
    """Return the shared metadata directory adjacent to raw and processed."""
    return Path(processed_root).parent / PROCESSED_INFO_DIRNAME


@dataclass(frozen=True)
class ProcessedSplit:
    """Processed split with normalized tensors.

    Array shapes (padded to ``max_num_levels``):

    * ``sequence_inputs``  — ``(num_runs, max_nz, sequence_dim)``
    * ``target_outputs``   — ``(num_runs, max_nz, target_dim)``
    * ``global_inputs``    — ``(num_runs, global_dim)``
    * ``valid_mask``       — ``(num_runs, max_nz)`` bool
    * ``position_coord``   — ``(num_runs, max_nz)`` float32 in [0, 1]
    """

    name: str
    sequence_inputs: np.ndarray
    target_outputs: np.ndarray
    global_inputs: np.ndarray
    valid_mask: np.ndarray
    position_coord: np.ndarray
    run_ids: list[str]
    metadata: dict[str, Any]

    @property
    def num_runs(self: "ProcessedSplit") -> int:
        """Return the number of runs stored in this split."""
        return int(self.sequence_inputs.shape[0])


def load_processed_split(split_dir: str | Path) -> ProcessedSplit:
    """Load one processed dataset split from its on-disk tensors."""
    split_path = Path(split_dir)
    metadata = json.loads((split_path / "metadata.json").read_text(encoding="utf-8"))
    run_ids = json.loads((split_path / "run_ids.json").read_text(encoding="utf-8"))

    return ProcessedSplit(
        name=split_path.name,
        sequence_inputs=np.load(split_path / "sequence_inputs.npy"),
        target_outputs=np.load(split_path / "target_outputs.npy"),
        global_inputs=np.load(split_path / "global_inputs.npy"),
        valid_mask=np.load(split_path / "valid_mask.npy"),
        position_coord=np.load(split_path / "position_coord.npy"),
        run_ids=list(run_ids),
        metadata=metadata,
    )


def load_processed_dataset(
    processed_root: str | Path,
) -> tuple[dict[str, ProcessedSplit], dict[str, Any], dict[str, Any]]:
    """Load the processed train/val/test splits plus shared metadata."""
    root = Path(processed_root)
    info_dir = processed_info_dir(root)
    splits = {
        name: load_processed_split(root / name)
        for name in ("train", "val", "test")
        if (root / name / "metadata.json").exists()
    }
    normalization = json.loads((info_dir / "normalization.json").read_text(encoding="utf-8"))
    contract = json.loads((info_dir / "data_contract.json").read_text(encoding="utf-8"))
    return splits, normalization, contract


def build_batch(
    split: ProcessedSplit,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assemble one training batch from a processed split."""
    idx = np.asarray(indices, dtype=np.int32)
    batch = {
        "sequence": split.sequence_inputs[idx].astype(np.float32),
        "global_inputs": split.global_inputs[idx].astype(np.float32),
        "target": split.target_outputs[idx].astype(np.float32),
        "valid_mask": split.valid_mask[idx].astype(bool),
        "position_coord": split.position_coord[idx].astype(np.float32),
    }
    return batch


def iter_batches(
    split: ProcessedSplit,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield one shuffled epoch of mini-batches for a split."""
    indices = np.arange(split.num_runs, dtype=np.int32)
    rng.shuffle(indices)
    step = int(batch_size)
    for start in range(0, indices.size, step):
        yield build_batch(split, indices[start : start + step])
