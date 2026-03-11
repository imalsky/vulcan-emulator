"""Processed normalized-trajectory split loading utilities.

Loads the padded ``.npy`` arrays and ``metadata.json`` produced by ``--gen``
for one train/val/test split, validating shapes, finiteness, and metadata
contracts before returning a :class:`ProcessedSplitArrays` bundle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class DataLoadingError(RuntimeError):
    """Raised when processed-split loading contracts are violated."""


@dataclass(frozen=True)
class ProcessedSplitArrays:
    """One processed split stored as padded normalized trajectories."""

    static_inputs: np.ndarray
    state_ymix: np.ndarray
    global_inputs: np.ndarray
    time_s: np.ndarray
    valid_steps_mask: np.ndarray
    run_ids: np.ndarray
    metadata: dict[str, Any]


def load_split_metadata(split_dir: Path) -> dict[str, Any]:
    """Load and validate one processed split metadata file."""
    metadata_path = split_dir / "metadata.json"
    if not metadata_path.is_file():
        raise DataLoadingError(f"Missing split metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise DataLoadingError(f"Invalid split metadata payload: {metadata_path}")

    positive_int_keys = {
        "num_runs",
        "max_steps",
        "total_valid_candidates",
        "sequence_length",
        "input_dim",
        "global_dim",
        "global_static_dim",
        "target_dim",
        "state_dim",
    }
    non_negative_int_keys = {"dt_feature_index"}
    list_keys = {
        "sequence_feature_order",
        "global_feature_order",
        "global_static_feature_order",
        "state_species_order",
        "output_species_order",
        "output_from_state_indices",
    }
    required_keys = (
        positive_int_keys
        | non_negative_int_keys
        | list_keys
        | {
            "split",
            "normalization_fingerprint",
            "sampling_mode",
            "dt_sampling_min_s",
            "dt_sampling_max_s",
            "min_future_saved_steps",
            "dt_min_s",
            "dt_max_s",
        }
    )
    missing = required_keys - metadata.keys()
    if missing:
        raise DataLoadingError(f"Missing required fields {sorted(missing)} in {metadata_path}.")

    for key in positive_int_keys:
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise DataLoadingError(f"Metadata field '{key}' must be a positive integer.")
    for key in non_negative_int_keys:
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DataLoadingError(f"Metadata field '{key}' must be a non-negative integer.")

    for key in list_keys - {"output_from_state_indices"}:
        value = metadata[key]
        if not isinstance(value, list) or not value:
            raise DataLoadingError(f"Metadata field '{key}' must be a non-empty list.")
        if any((not isinstance(item, str) or not item.strip()) for item in value):
            raise DataLoadingError(f"Metadata field '{key}' contains an invalid entry.")

    output_from_state_indices = metadata["output_from_state_indices"]
    if not isinstance(output_from_state_indices, list) or not output_from_state_indices:
        raise DataLoadingError("Metadata field 'output_from_state_indices' must be a non-empty list.")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in output_from_state_indices):
        raise DataLoadingError("output_from_state_indices must contain integers.")

    fingerprint = metadata["normalization_fingerprint"]
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise DataLoadingError("normalization_fingerprint must be a 64-character sha256 hex string.")
    try:
        int(fingerprint, 16)
    except ValueError as exc:
        raise DataLoadingError("normalization_fingerprint must be hexadecimal.") from exc

    if len(metadata["sequence_feature_order"]) != int(metadata["input_dim"]):
        raise DataLoadingError("Invalid sequence feature order length in split metadata.")
    if len(metadata["global_feature_order"]) != int(metadata["global_dim"]):
        raise DataLoadingError("Invalid global feature order length in split metadata.")
    if len(metadata["global_static_feature_order"]) != int(metadata["global_static_dim"]):
        raise DataLoadingError("Invalid global static feature order length in split metadata.")
    if len(metadata["state_species_order"]) != int(metadata["state_dim"]):
        raise DataLoadingError("Invalid state species order length in split metadata.")
    if len(metadata["output_species_order"]) != int(metadata["target_dim"]):
        raise DataLoadingError("Invalid output species order length in split metadata.")
    if len(output_from_state_indices) != int(metadata["target_dim"]):
        raise DataLoadingError("Invalid output_from_state_indices length in split metadata.")
    if any(index < 0 or index >= int(metadata["state_dim"]) for index in output_from_state_indices):
        raise DataLoadingError("output_from_state_indices contains an out-of-range state index.")
    if int(metadata["input_dim"]) != 3 + int(metadata["state_dim"]):
        raise DataLoadingError("input_dim must equal 3 + state_dim.")
    if int(metadata["global_static_dim"]) != int(metadata["global_dim"]) - 1:
        raise DataLoadingError("global_static_dim must equal global_dim - 1.")
    if int(metadata["dt_feature_index"]) >= int(metadata["global_dim"]):
        raise DataLoadingError("dt_feature_index must be < global_dim.")
    if metadata["global_feature_order"][int(metadata["dt_feature_index"])] != "log10_dt_s":
        raise DataLoadingError("dt_feature_index must point at 'log10_dt_s'.")
    expected_static_order = [
        name for name in metadata["global_feature_order"] if name != "log10_dt_s"
    ]
    if metadata["global_static_feature_order"] != expected_static_order:
        raise DataLoadingError(
            "global_static_feature_order must equal global_feature_order without 'log10_dt_s'."
        )
    if float(metadata["dt_min_s"]) <= 0.0 or float(metadata["dt_max_s"]) < float(metadata["dt_min_s"]):
        raise DataLoadingError("Invalid dt range recorded in split metadata.")
    return metadata


def load_processed_split_arrays(split_dir: Path) -> ProcessedSplitArrays:
    """Load one processed split eagerly into host memory and validate its shapes."""
    metadata = load_split_metadata(split_dir)
    required_paths = {
        "static_inputs": split_dir / "static_inputs.npy",
        "state_ymix": split_dir / "state_ymix.npy",
        "global_inputs": split_dir / "global_inputs.npy",
        "time_s": split_dir / "time_s.npy",
        "valid_steps_mask": split_dir / "valid_steps_mask.npy",
        "run_ids": split_dir / "run_ids.npy",
    }
    for path in required_paths.values():
        if not path.is_file():
            raise DataLoadingError(f"Missing processed split array: {path}")

    static_inputs = np.load(required_paths["static_inputs"], allow_pickle=False)
    state_ymix = np.load(required_paths["state_ymix"], allow_pickle=False)
    global_inputs = np.load(required_paths["global_inputs"], allow_pickle=False)
    time_s = np.load(required_paths["time_s"], allow_pickle=False)
    valid_steps_mask = np.load(required_paths["valid_steps_mask"], allow_pickle=False)
    run_ids = np.load(required_paths["run_ids"], allow_pickle=False)

    num_runs = int(metadata["num_runs"])
    max_steps = int(metadata["max_steps"])
    sequence_length = int(metadata["sequence_length"])
    state_dim = int(metadata["state_dim"])
    global_static_dim = int(metadata["global_static_dim"])

    if static_inputs.shape != (num_runs, sequence_length, 3):
        raise DataLoadingError(f"Unexpected static_inputs shape in {split_dir}: {static_inputs.shape}.")
    if state_ymix.shape != (num_runs, max_steps, sequence_length, state_dim):
        raise DataLoadingError(f"Unexpected state_ymix shape in {split_dir}: {state_ymix.shape}.")
    if global_inputs.shape != (num_runs, global_static_dim):
        raise DataLoadingError(f"Unexpected global_inputs shape in {split_dir}: {global_inputs.shape}.")
    if time_s.shape != (num_runs, max_steps):
        raise DataLoadingError(f"Unexpected time_s shape in {split_dir}: {time_s.shape}.")
    if valid_steps_mask.shape != (num_runs, max_steps):
        raise DataLoadingError(
            f"Unexpected valid_steps_mask shape in {split_dir}: {valid_steps_mask.shape}."
        )
    if run_ids.shape != (num_runs,):
        raise DataLoadingError(f"Unexpected run_ids shape in {split_dir}: {run_ids.shape}.")

    if np.any(~np.isfinite(static_inputs)) or np.any(~np.isfinite(state_ymix)) or np.any(~np.isfinite(global_inputs)):
        raise DataLoadingError(f"Non-finite normalized inputs found in {split_dir}.")
    if np.any(~np.isfinite(time_s)):
        raise DataLoadingError(f"Non-finite times found in {split_dir}.")
    if valid_steps_mask.dtype != np.bool_:
        valid_steps_mask = np.asarray(valid_steps_mask, dtype=bool)
    if run_ids.dtype.kind not in {"i", "u"}:
        raise DataLoadingError("run_ids.npy must contain integer values.")

    valid_counts = valid_steps_mask.sum(axis=1, dtype=np.int64)
    if np.any(valid_counts < 2):
        raise DataLoadingError("Every processed run must contain at least two valid saved steps.")

    # Padded time positions must stay at zero so candidate building can ignore them cleanly.
    padded_time_values = time_s[~valid_steps_mask]
    if padded_time_values.size > 0 and not np.allclose(padded_time_values, 0.0):
        raise DataLoadingError("Padded time positions must be zero-filled.")

    return ProcessedSplitArrays(
        static_inputs=np.asarray(static_inputs, dtype=np.float32),
        state_ymix=np.asarray(state_ymix, dtype=np.float32),
        global_inputs=np.asarray(global_inputs, dtype=np.float32),
        time_s=np.asarray(time_s, dtype=np.float64),
        valid_steps_mask=np.asarray(valid_steps_mask, dtype=bool),
        run_ids=np.asarray(run_ids, dtype=np.int64),
        metadata=metadata,
    )
