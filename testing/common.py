#!/usr/bin/env python3
"""Shared helpers for vulcan-emulator testing scripts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# Match src/main.py so standalone test utilities don't abort on duplicate OpenMP runtimes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import VulcanTransformer

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON file and require UTF-8 decoding.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON object as `dict[str, Any]`.
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def torch_dtype_from_name(name: str) -> torch.dtype:
    """Resolve one serialized dtype name to the matching Torch dtype.

    Args:
        name: Serialized dtype string such as `"float32"` or `"bfloat16"`.

    Returns:
        Matching `torch.dtype`.
    """
    lowered = str(name).lower()
    if lowered not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype name: {name}")
    return _DTYPE_MAP[lowered]


def load_checkpoint(run_dir: Path, checkpoint_name: str) -> dict[str, Any]:
    """Load one training checkpoint from a run directory.

    Args:
        run_dir: Model run directory.
        checkpoint_name: Checkpoint filename inside `run_dir`.

    Returns:
        Parsed checkpoint dictionary loaded with `torch.load`.
    """
    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Invalid checkpoint structure: {checkpoint_path}")
    return checkpoint


def load_split_metadata(processed_root: Path, split: str) -> dict[str, Any]:
    """Load processed-split metadata for one named dataset partition.

    Args:
        processed_root: Root processed-data directory.
        split: Split name such as `"train"`, `"val"`, or `"test"`.

    Returns:
        Parsed split metadata dictionary.
    """
    split_dir = processed_root / split
    metadata_path = split_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing split metadata: {metadata_path}")
    return load_json(metadata_path)


def iter_split_shards(
    processed_root: Path,
    split: str,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield `(sequence, globals, targets)` arrays for each shard in one split.

    Args:
        processed_root: Root processed-data directory.
        split: Split name such as `"train"`, `"val"`, or `"test"`.

    Yields:
        Tuples of NumPy arrays with shapes `[samples, nz, input_dim]`,
        `[samples, global_dim]`, and `[samples, nz, target_dim]`.
    """
    split_dir = processed_root / split
    metadata = load_split_metadata(processed_root=processed_root, split=split)
    num_shards = int(metadata["num_shards"])
    seq_dir = split_dir / "sequence_inputs"
    glb_dir = split_dir / "globals"
    tgt_dir = split_dir / "targets"

    for shard_idx in range(num_shards):
        seq = np.load(seq_dir / f"shard_{shard_idx:05d}.npy", allow_pickle=False)
        glb = np.load(glb_dir / f"shard_{shard_idx:05d}.npy", allow_pickle=False)
        tgt = np.load(tgt_dir / f"shard_{shard_idx:05d}.npy", allow_pickle=False)
        yield seq, glb, tgt


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    split_metadata: dict[str, Any],
    device: torch.device,
) -> tuple[VulcanTransformer, torch.dtype]:
    """Rebuild the trained model and return it with the forward-pass dtype.

    Args:
        checkpoint: Loaded checkpoint dictionary containing config and weights.
        split_metadata: Processed split metadata describing input/output dimensions.
        device: Torch device on which the model should be materialized.

    Returns:
        Tuple `(model, forward_dtype)` where `model` is an eval-mode `VulcanTransformer`.
    """
    config = checkpoint["config"]
    model_cfg = config["training"]["model"]
    precision_cfg = config["precision"]

    model = VulcanTransformer(
        input_dim=int(split_metadata["input_dim"]),
        global_dim=int(split_metadata["global_dim"]),
        target_dim=int(split_metadata["target_dim"]),
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        num_layers=int(model_cfg["num_layers"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        dropout=float(model_cfg["dropout"]),
        film_clamp=float(model_cfg["film_clamp"]),
        output_head_divisor=int(model_cfg["output_head_divisor"]),
        max_sequence_length=int(model_cfg["max_sequence_length"]),
    ).to(device=device, dtype=torch_dtype_from_name(str(precision_cfg["model_dtype"])))

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    forward_dtype = torch_dtype_from_name(str(precision_cfg["forward_dtype"]))
    return model, forward_dtype


def denormalize(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """Invert one normalized NumPy array using serialized normalization stats.

    Args:
        values: Normalized NumPy array of any broadcast-compatible shape.
        stats: Serialized normalization-stat dictionary.

    Returns:
        Physical-space NumPy array with the same shape as `values` and dtype `float64`.
    """
    method = str(stats["method"])
    data = np.asarray(values, dtype=np.float64)
    if method == "none":
        return data
    if method == "standard":
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        return data * std + mean
    if method == "log-standard":
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        return np.power(10.0, data * std + mean)
    if method == "log-min-max":
        lower = np.asarray(stats["min"], dtype=np.float64)
        upper = np.asarray(stats["max"], dtype=np.float64)
        width = np.where((upper - lower) > 0.0, upper - lower, 1.0)
        return np.power(10.0, data * width + lower)
    raise ValueError(f"Unsupported normalization method: {method}")
