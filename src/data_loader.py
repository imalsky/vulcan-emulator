from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .transition_sampling import CandidateTable


@dataclass(frozen=True)
class ProcessedSplit:
    name: str
    sequence_inputs: np.ndarray
    state_trajectories: np.ndarray
    target_outputs: np.ndarray
    global_inputs: np.ndarray
    spectrum_inputs: np.ndarray
    time_s: np.ndarray
    valid_steps_mask: np.ndarray
    run_ids: list[str]
    metadata: dict[str, Any]

    @property
    def num_runs(self) -> int:
        return int(self.sequence_inputs.shape[0])


@dataclass(frozen=True)
class EquilibriumSplit:
    """Processed split for equilibrium models (no trajectory/spectrum)."""
    name: str
    sequence_inputs: np.ndarray   # [num_runs, nz, 2] (P, T)
    target_outputs: np.ndarray    # [num_runs, nz, target_dim]
    global_inputs: np.ndarray     # [num_runs, global_dim]
    run_ids: list[str]
    metadata: dict[str, Any]

    @property
    def num_runs(self) -> int:
        return int(self.sequence_inputs.shape[0])


def load_processed_split(split_dir: str | Path) -> ProcessedSplit:
    split_path = Path(split_dir)
    metadata = json.loads((split_path / "metadata.json").read_text(encoding="utf-8"))
    run_ids = json.loads((split_path / "run_ids.json").read_text(encoding="utf-8"))
    return ProcessedSplit(
        name=split_path.name,
        sequence_inputs=np.load(split_path / "sequence_inputs.npy"),
        state_trajectories=np.load(split_path / "state_trajectories.npy"),
        target_outputs=np.load(split_path / "target_outputs.npy"),
        global_inputs=np.load(split_path / "global_inputs.npy"),
        spectrum_inputs=np.load(split_path / "spectrum_inputs.npy"),
        time_s=np.load(split_path / "time_s.npy"),
        valid_steps_mask=np.load(split_path / "valid_steps_mask.npy"),
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


def load_processed_dataset(processed_root: str | Path) -> tuple[dict[str, ProcessedSplit], dict[str, Any], dict[str, Any]]:
    root = Path(processed_root)
    splits = {
        name: load_processed_split(root / name)
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


def build_batch_from_rows(
    split: ProcessedSplit,
    candidate_table: CandidateTable,
    row_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.asarray(row_indices, dtype=np.int32)
    run_idx = candidate_table.run_index[rows]
    anchor_idx = candidate_table.anchor_index[rows]
    target_idx = candidate_table.target_index[rows]
    static_inputs = split.sequence_inputs[run_idx]
    anchor_state = split.state_trajectories[run_idx, anchor_idx]
    target_state = split.target_outputs[run_idx, target_idx]
    sequence = np.concatenate([static_inputs, anchor_state], axis=-1)
    global_static = split.global_inputs[run_idx]
    dt_feature_index = int(split.metadata["dt_feature_index"])
    dt_values = candidate_table.normalized_log10_dt_s[rows][:, None]
    globals_full = np.concatenate(
        [
            global_static[:, :dt_feature_index],
            dt_values,
            global_static[:, dt_feature_index:],
        ],
        axis=1,
    )
    return {
        "sequence": sequence.astype(np.float32),
        "global_inputs": globals_full.astype(np.float32),
        "spectrum_inputs": split.spectrum_inputs[run_idx].astype(np.float32),
        "target": target_state.astype(np.float32),
        "dt_s": candidate_table.dt_s[rows].astype(np.float32),
        "row_indices": rows,
    }


def iter_batches(
    split: ProcessedSplit,
    candidate_table: CandidateTable,
    row_indices: np.ndarray,
    *,
    batch_size: int,
) -> list[dict[str, np.ndarray]]:
    rows = np.asarray(row_indices, dtype=np.int32)
    if rows.size == 0:
        return []
    batches: list[dict[str, np.ndarray]] = []
    for start in range(0, rows.size, int(batch_size)):
        stop = min(start + int(batch_size), rows.size)
        batches.append(build_batch_from_rows(split, candidate_table, rows[start:stop]))
    return batches


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


def iter_equilibrium_batches(
    split: EquilibriumSplit,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> list[dict[str, np.ndarray]]:
    """Yield shuffled batches for equilibrium training."""
    indices = np.arange(split.num_runs, dtype=np.int32)
    rng.shuffle(indices)
    batches: list[dict[str, np.ndarray]] = []
    for start in range(0, indices.size, int(batch_size)):
        stop = min(start + int(batch_size), indices.size)
        batches.append(build_equilibrium_batch(split, indices[start:stop]))
    return batches
