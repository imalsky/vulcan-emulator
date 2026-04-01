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


PROCESSED_INFO_DIRNAME = "info"


def processed_info_dir(processed_root: str | Path) -> Path:
    """Return the shared metadata directory for a processed dataset.

    Parameters
    ----------
    processed_root : str or Path
        Root directory of the processed dataset layout.

    Returns
    -------
    Path
        Directory reserved for shared processed-dataset JSON metadata such as
        ``normalization.json`` and ``data_contract.json``.
    """
    return Path(processed_root) / PROCESSED_INFO_DIRNAME


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
        """Return the number of runs stored in this split.

        Returns
        -------
        int
            Size of the leading run dimension shared by the stored arrays.
        """
        return int(self.sequence_inputs.shape[0])

    @property
    def has_spectrum_inputs(self) -> bool:
        """Return whether this split includes stellar-spectrum conditioning.

        Returns
        -------
        bool
            ``True`` when ``spectrum_inputs`` is present with shape
            ``(num_runs, spectrum_dim)``.
        """
        return self.spectrum_inputs is not None


def load_processed_split(split_dir: str | Path) -> ProcessedSplit:
    """Load one processed dataset split from its on-disk tensors.

    Parameters
    ----------
    split_dir : str or Path
        Directory containing ``sequence_inputs.npy``, ``target_outputs.npy``,
        ``global_inputs.npy``, ``run_ids.json``, and ``metadata.json``.
        ``spectrum_inputs.npy`` is optional.

    Returns
    -------
    ProcessedSplit
        In-memory wrapper around the normalized split tensors and their
        associated metadata.
    """
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
    """Load the processed train/val/test splits plus shared metadata.

    Parameters
    ----------
    processed_root : str or Path
        Processed dataset root containing split subdirectories together with
        ``info/normalization.json`` and ``info/data_contract.json``.

    Returns
    -------
    tuple[dict[str, ProcessedSplit], dict[str, Any], dict[str, Any]]
        Mapping of available split names to ``ProcessedSplit`` objects, the
        normalization payload, and the data-contract metadata that defines the
        tensor ordering.
    """
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
    """Assemble one training batch from a processed split.

    Parameters
    ----------
    split : ProcessedSplit
        Source split containing normalized arrays with leading dimension
        ``num_runs``.
    indices : np.ndarray
        Integer run indices selecting the rows to gather from the split.

    Returns
    -------
    dict[str, np.ndarray]
        Batch dictionary with ``sequence`` of shape
        ``(batch, nz, sequence_dim)``, ``global_inputs`` of shape
        ``(batch, global_dim)``, ``target`` of shape
        ``(batch, nz, target_dim)``, and optional ``spectrum_inputs`` of shape
        ``(batch, spectrum_dim)``.
    """
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
    """Materialize one shuffled epoch of mini-batches for a split.

    Parameters
    ----------
    split : ProcessedSplit
        Normalized split to iterate over.
    batch_size : int
        Maximum number of runs per batch.
    rng : np.random.Generator
        Random generator used to shuffle run indices reproducibly.

    Returns
    -------
    list[dict[str, np.ndarray]]
        Ordered list of batch dictionaries produced by ``build_batch``. The
        final batch may be smaller than ``batch_size``.
    """
    indices = np.arange(split.num_runs, dtype=np.int32)
    rng.shuffle(indices)
    batches: list[dict[str, np.ndarray]] = []
    for start in range(0, indices.size, int(batch_size)):
        stop = min(start + int(batch_size), indices.size)
        batches.append(build_batch(split, indices[start:stop]))
    return batches
