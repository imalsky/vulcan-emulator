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
from data_loader import load_split_metadata as _load_validated_split_metadata
from live_sampling import ProcessedTrajectoryStore
from model import VulcanTransitionTransformer
from path_utils import resolve_paths

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON file and require an object payload."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}, found {type(payload).__name__}.")
    return payload


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
    """Resolve one configured torch dtype name."""
    lowered = str(name).lower()
    if lowered not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype name: {name}")
    return _DTYPE_MAP[lowered]


def load_checkpoint(run_dir: Path, checkpoint_name: str) -> dict[str, Any]:
    """Load one saved training checkpoint from disk."""
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
    """Load one processed split metadata file."""
    return _load_validated_split_metadata(processed_root / split)


def _fixed_eval_seed(*, split: str, base_seed: int) -> int:
    """Derive the deterministic fixed-eval seed used by trainer and scripts."""
    return int(base_seed) + {"train": 0, "val": 1, "test": 2}.get(split, 0)


def _fixed_pair_selection(
    *,
    processed_root: Path,
    split: str,
    config: dict[str, Any],
    normalization_metadata: dict[str, Any],
    pairs_per_run: int | None = None,
    seed: int | None = None,
) -> tuple[ProcessedTrajectoryStore, torch.Tensor]:
    """Build one CPU trajectory store and fixed-eval candidate index table."""
    resolved_pairs_per_run = (
        int(pairs_per_run)
        if pairs_per_run is not None
        else int(config["training"]["live_sampling"]["eval_pairs_per_run"])
    )
    resolved_seed = (
        int(seed)
        if seed is not None
        else _fixed_eval_seed(split=split, base_seed=int(config["training"]["seed"]))
    )
    store = ProcessedTrajectoryStore.from_split_dir(
        split_dir=processed_root / split,
        normalization_metadata=normalization_metadata,
        config=config,
        device=torch.device("cpu"),
        tensor_dtype=torch.float32,
    )
    selected = store.select_fixed_candidate_indices(
        pairs_per_run=resolved_pairs_per_run,
        seed=resolved_seed,
    )
    return store, selected


def count_fixed_split_samples(
    *,
    processed_root: Path,
    split: str,
    config: dict[str, Any],
    normalization_metadata: dict[str, Any],
    pairs_per_run: int | None = None,
    seed: int | None = None,
) -> int:
    """Return the deterministic fixed-eval sample count for one split."""
    _store, selected = _fixed_pair_selection(
        processed_root=processed_root,
        split=split,
        config=config,
        normalization_metadata=normalization_metadata,
        pairs_per_run=pairs_per_run,
        seed=seed,
    )
    return int(selected.numel())


def iter_fixed_split_batches(
    *,
    processed_root: Path,
    split: str,
    config: dict[str, Any],
    normalization_metadata: dict[str, Any],
    batch_size: int,
    pairs_per_run: int | None = None,
    seed: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Iterate deterministic fixed-eval pairs as numpy batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    store, selected = _fixed_pair_selection(
        processed_root=processed_root,
        split=split,
        config=config,
        normalization_metadata=normalization_metadata,
        pairs_per_run=pairs_per_run,
        seed=seed,
    )
    for start in range(0, int(selected.numel()), batch_size):
        end = min(start + batch_size, int(selected.numel()))
        seq, glb, tgt, _mask, dt = store.build_batch(selected[start:end])
        yield (
            seq.detach().cpu().numpy(),
            glb.detach().cpu().numpy(),
            tgt.detach().cpu().numpy(),
            dt.detach().cpu().numpy(),
        )


def load_fixed_split_sample(
    *,
    processed_root: Path,
    split: str,
    config: dict[str, Any],
    normalization_metadata: dict[str, Any],
    sample_index: int,
    pairs_per_run: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load one deterministic fixed-eval pair by global sample index."""
    if sample_index < 0:
        raise ValueError("sample_index must be >= 0.")
    store, selected = _fixed_pair_selection(
        processed_root=processed_root,
        split=split,
        config=config,
        normalization_metadata=normalization_metadata,
        pairs_per_run=pairs_per_run,
        seed=seed,
    )
    if sample_index >= int(selected.numel()):
        raise IndexError(f"Sample index {sample_index} is out of range for split '{split}'.")
    seq, glb, tgt, _mask, dt = store.build_batch(selected[sample_index : sample_index + 1])
    return (
        np.asarray(seq[0].detach().cpu().numpy(), dtype=np.float64),
        np.asarray(glb[0].detach().cpu().numpy(), dtype=np.float64),
        np.asarray(tgt[0].detach().cpu().numpy(), dtype=np.float64),
        float(dt[0].item()),
    )


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    split_metadata: dict[str, Any],
    device: torch.device,
) -> tuple[VulcanTransitionTransformer, torch.dtype]:
    """Rebuild one trained transition model and its forward dtype."""
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
        num_globals=int(split_metadata["global_dim"]),
    ).to(device=device, dtype=torch_dtype_from_name(str(precision_cfg["model_dtype"])))

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    forward_dtype = torch_dtype_from_name(str(precision_cfg["forward_dtype"]))
    return model, forward_dtype


def denormalize(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """Invert one normalized array back to physical space using stored stats."""
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
