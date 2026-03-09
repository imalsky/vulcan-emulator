#!/usr/bin/env python3
"""Shared helpers for repo scripts and local test utilities."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_utils import load_and_validate_config
from model import VulcanTransitionTransformer
from path_utils import resolve_paths

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_run_dir(*, config_path: Path, run_dir: Path | None) -> tuple[Path, Path | None]:
    """Resolve one run directory either explicitly or from a validated config."""
    if run_dir is not None:
        candidate = run_dir if run_dir.is_absolute() else (PROJECT_ROOT / run_dir)
        return candidate.resolve(), None

    resolved_config = config_path if config_path.is_absolute() else (PROJECT_ROOT / config_path)
    resolved_config = resolved_config.resolve()
    config = load_and_validate_config(resolved_config)
    paths = resolve_paths(config)
    resolved_run_dir = (paths.models_root / str(config["training"]["output_folder"])).resolve()
    return resolved_run_dir, resolved_config


def torch_dtype_from_name(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype name: {name}")
    return _DTYPE_MAP[lowered]


def load_checkpoint(run_dir: Path, checkpoint_name: str) -> dict[str, Any]:
    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Invalid checkpoint structure: {checkpoint_path}")
    return checkpoint


def resolve_processed_root_from_checkpoint(run_dir: Path, checkpoint_name: str) -> Path:
    """Resolve processed-root path from one saved checkpoint config."""
    checkpoint = load_checkpoint(run_dir=run_dir, checkpoint_name=checkpoint_name)
    return (PROJECT_ROOT / str(checkpoint["config"]["paths"]["processed_root"])).resolve()


def load_split_metadata(processed_root: Path, split: str) -> dict[str, Any]:
    split_dir = processed_root / split
    metadata_path = split_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing split metadata: {metadata_path}")
    return load_json(metadata_path)


def iter_split_shards(
    processed_root: Path,
    split: str,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    split_dir = processed_root / split
    metadata = load_split_metadata(processed_root=processed_root, split=split)
    num_shards = int(metadata["num_shards"])
    seq_dir = split_dir / "sequence_inputs"
    glb_dir = split_dir / "globals"
    tgt_dir = split_dir / "targets"
    dt_dir = split_dir / "dt_s"

    for shard_idx in range(num_shards):
        seq = np.load(seq_dir / f"shard_{shard_idx:05d}.npy", allow_pickle=False)
        glb = np.load(glb_dir / f"shard_{shard_idx:05d}.npy", allow_pickle=False)
        tgt = np.load(tgt_dir / f"shard_{shard_idx:05d}.npy", allow_pickle=False)
        dt = np.load(dt_dir / f"shard_{shard_idx:05d}.npy", allow_pickle=False)
        yield seq, glb, tgt, dt


def load_split_sample(
    *,
    processed_root: Path,
    split: str,
    sample_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load one sample by global index from a processed split."""
    if sample_index < 0:
        raise ValueError("sample_index must be >= 0.")

    remaining = int(sample_index)
    for seq, glb, tgt, dt in iter_split_shards(processed_root=processed_root, split=split):
        shard_size = int(seq.shape[0])
        if remaining < shard_size:
            return (
                np.asarray(seq[remaining], dtype=np.float64),
                np.asarray(glb[remaining], dtype=np.float64),
                np.asarray(tgt[remaining], dtype=np.float64),
                float(np.asarray(dt[remaining], dtype=np.float64)),
            )
        remaining -= shard_size
    raise IndexError(f"Sample index {sample_index} is out of range for split '{split}'.")


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    split_metadata: dict[str, Any],
    device: torch.device,
) -> tuple[VulcanTransitionTransformer, torch.dtype]:
    config = checkpoint["config"]
    model_cfg = config["training"]["model"]
    precision_cfg = config["precision"]

    model = VulcanTransitionTransformer(
        state_dim=int(split_metadata["state_dim"]),
        output_dim=int(split_metadata["target_dim"]),
        output_from_state_indices=list(split_metadata["output_from_state_indices"]),
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        num_layers=int(model_cfg["num_layers"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        dropout=float(model_cfg["dropout"]),
        film_clamp=float(model_cfg["film_clamp"]),
        output_head_divisor=int(model_cfg["output_head_divisor"]),
        max_sequence_length=int(model_cfg["max_sequence_length"]),
        conditioning_hidden_dim=int(model_cfg["conditioning_hidden_dim"]),
    ).to(device=device, dtype=torch_dtype_from_name(str(precision_cfg["model_dtype"])))

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    forward_dtype = torch_dtype_from_name(str(precision_cfg["forward_dtype"]))
    return model, forward_dtype


def denormalize(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
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
