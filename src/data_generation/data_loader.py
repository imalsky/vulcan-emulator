"""Processed dataset loading, split management, and batch assembly.

This module sits between the on-disk processed tensors (produced by
``preprocess.py``) and the training loop (``trainer.py``). It provides:

* a single ``ProcessedSplit`` dataclass for any chemistry/model combination
* split loading helpers for the processed train/val/test partitions
* index-based batch assembly with optional variable-length spectrum inputs
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
    """Return the shared metadata directory for a processed dataset."""
    return Path(processed_root) / PROCESSED_INFO_DIRNAME


@dataclass(frozen=True)
class ProcessedSplit:
    """Processed split with normalized tensors and optional spectrum tokens.

    Array shapes:

    * ``sequence_inputs``                 — ``(num_runs, nz, sequence_dim)``
    * ``target_outputs``                  — ``(num_runs, nz, target_dim)``
    * ``global_inputs``                   — ``(num_runs, global_dim)``
    * ``spectrum_wavelengths_nm``         — ``(num_runs, spectrum_max_tokens)``
    * ``spectrum_fluxes_erg_cm2_s_nm``    — ``(num_runs, spectrum_max_tokens)``
    * ``spectrum_mask``                   — ``(num_runs, spectrum_max_tokens)``
    """

    name: str
    sequence_inputs: np.ndarray
    target_outputs: np.ndarray
    global_inputs: np.ndarray
    spectrum_wavelengths_nm: np.ndarray | None
    spectrum_fluxes_erg_cm2_s_nm: np.ndarray | None
    spectrum_mask: np.ndarray | None
    run_ids: list[str]
    metadata: dict[str, Any]

    @property
    def num_runs(self) -> int:
        """Return the number of runs stored in this split."""
        return int(self.sequence_inputs.shape[0])

    @property
    def has_spectrum_inputs(self) -> bool:
        """Return whether this split includes stellar-spectrum conditioning."""
        return (
            self.spectrum_wavelengths_nm is not None
            and self.spectrum_fluxes_erg_cm2_s_nm is not None
            and self.spectrum_mask is not None
        )


def load_processed_split(split_dir: str | Path) -> ProcessedSplit:
    """Load one processed dataset split from its on-disk tensors."""
    split_path = Path(split_dir)
    metadata = json.loads((split_path / "metadata.json").read_text(encoding="utf-8"))
    run_ids = json.loads((split_path / "run_ids.json").read_text(encoding="utf-8"))

    wavelength_path = split_path / "spectrum_wavelengths_nm.npy"
    flux_path = split_path / "spectrum_fluxes_erg_cm2_s_nm.npy"
    mask_path = split_path / "spectrum_mask.npy"

    has_spectrum = wavelength_path.exists() and flux_path.exists() and mask_path.exists()
    return ProcessedSplit(
        name=split_path.name,
        sequence_inputs=np.load(split_path / "sequence_inputs.npy"),
        target_outputs=np.load(split_path / "target_outputs.npy"),
        global_inputs=np.load(split_path / "global_inputs.npy"),
        spectrum_wavelengths_nm=np.load(wavelength_path) if has_spectrum else None,
        spectrum_fluxes_erg_cm2_s_nm=np.load(flux_path) if has_spectrum else None,
        spectrum_mask=np.load(mask_path) if has_spectrum else None,
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
    }
    if split.has_spectrum_inputs:
        batch["spectrum_wavelengths_nm"] = split.spectrum_wavelengths_nm[idx].astype(np.float32)
        batch["spectrum_fluxes_erg_cm2_s_nm"] = split.spectrum_fluxes_erg_cm2_s_nm[idx].astype(
            np.float32
        )
        batch["spectrum_mask"] = split.spectrum_mask[idx].astype(bool)
    return batch


def iter_batches(
    split: ProcessedSplit,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> list[dict[str, np.ndarray]]:
    """Materialize one shuffled epoch of mini-batches for a split."""
    indices = np.arange(split.num_runs, dtype=np.int32)
    rng.shuffle(indices)
    batches: list[dict[str, np.ndarray]] = []
    for start in range(0, indices.size, int(batch_size)):
        stop = min(start + int(batch_size), indices.size)
        batches.append(build_batch(split, indices[start:stop]))
    return batches
