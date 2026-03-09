"""Training pipeline for the VULCAN transition emulator.

Implements the full ``--train`` lifecycle:

1. **Artifact validation**: Verifies processed data provenance via SHA-256
   fingerprints before training begins.
2. **Data loading**: Configurable RAM/disk/auto modes with optional CUDA
   prefetching for GPU-resident training.
3. **Optimization**: AdamW with explicit bias/norm no-decay groups, linear
   warmup followed by cosine decay to ``min_lr``, gradient clipping, and
   optional mixed-precision (AMP) training via GradScaler.
4. **Evaluation**: Masked MSE/MAE over validation/test splits with optional
   dt-binned breakdowns showing performance across timescales.
5. **Rollout evaluation**: Autoregressive multi-jump composition on held-out
   raw trajectories, validating that single-step accuracy composes well.
6. **Checkpointing**: Saves ``best.pt`` (lowest val MSE) and ``last.pt``,
   plus inference artifacts (normalization metadata, data contract).
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import re
import shutil
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config_utils import PrecisionConfig, resolve_conditioning_inputs
from data_loader import DataLoadingConfig, DevicePrefetchLoader, build_training_loader
from inference import PhysicalSpaceStandaloneModel, VulcanPredictor
from model import VulcanTransitionTransformer
from preprocess import load_raw_run_file
from transition_sampling import TransitionSamplingError, build_rollout_indices
from provenance import PROCESSED_FINGERPRINT_FILENAME, validate_processed_artifacts

logger = logging.getLogger(__name__)


class TrainingError(RuntimeError):
    """Raised when training contract checks fail."""


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_name: str) -> torch.device:
    """Resolve the configured training device with explicit availability checks."""
    lowered = str(device_name).lower()
    if lowered == "cuda":
        if not torch.cuda.is_available():
            raise TrainingError("training.device='cuda' requested but CUDA is unavailable.")
        return torch.device("cuda")
    if lowered == "mps":
        if not torch.backends.mps.is_available():
            raise TrainingError("training.device='mps' requested but MPS is unavailable.")
        return torch.device("mps")
    if lowered == "cpu":
        return torch.device("cpu")
    raise TrainingError(f"Unsupported training.device: {device_name}")


def _configure_cpu_threads(device: torch.device) -> None:
    """Use deterministic single-thread CPU execution to avoid backend oversubscription stalls."""
    if device.type != "cpu":
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _format_progress_line(
    *,
    epoch: int,
    epochs: int,
    train_mse: float,
    val_mse: float,
    val_mae: float,
    lr: float,
    epoch_seconds: float,
    elapsed_seconds: float,
) -> str:
    """Render one fixed-width training-progress line for the logs directory."""
    return (
        f"{epoch:6d}/{epochs:<6d}"
        f" {train_mse:15.8e}"
        f" {val_mse:15.8e}"
        f" {val_mae:15.8e}"
        f" {lr:15.8e}"
        f" {epoch_seconds:15.8e}"
        f" {elapsed_seconds:15.8e}"
    )


def _build_loader(
    *,
    split_dir: Path,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    preload_to_device: bool,
    num_workers: int,
    input_dtype: torch.dtype,
    target_dtype: torch.dtype,
    loading: DataLoadingConfig,
) -> tuple[DataLoader | DevicePrefetchLoader, dict[str, Any]]:
    """Build one split loader and normalize loader errors into ``TrainingError``."""
    try:
        loader, metadata = build_training_loader(
            split_dir=split_dir,
            batch_size=batch_size,
            shuffle=shuffle,
            device=device,
            preload_to_device=preload_to_device,
            num_workers=num_workers,
            input_dtype=input_dtype,
            target_dtype=target_dtype,
            loading=loading,
        )
    except Exception as exc:
        raise TrainingError(str(exc)) from exc
    return loader, metadata


def _build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Construct AdamW with explicit no-decay handling for biases and norms."""
    norm_types = (
        nn.LayerNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()

    for _module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            if param_name.endswith("bias") or isinstance(module, norm_types):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    if not decay_params and not no_decay_params:
        raise TrainingError("No trainable parameters available for optimizer construction.")

    param_groups: list[dict[str, Any]] = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return torch.optim.AdamW(param_groups, lr=learning_rate)


def _cast_optimizer_state(optimizer: torch.optim.Optimizer, dtype: torch.dtype) -> None:
    """Cast floating-point optimizer state tensors to the configured dtype."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and torch.is_floating_point(value) and value.dtype != dtype:
                state[key] = value.to(dtype=dtype)


def _unpack_batch(
    batch: tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Require the 5-tensor transition batch contract."""
    if len(batch) != 5:
        raise TrainingError(
            f"Unexpected batch structure with {len(batch)} tensors; expected 5 (seq, glb, tgt, mask, dt)."
        )
    seq, glb, tgt, padding_mask, dt = batch
    return seq, glb, tgt, padding_mask.bool(), dt


def _loader_prefetches_to_device(loader: Any) -> bool:
    """Return whether the loader already yields tensors on the target device."""
    return bool(getattr(loader, "prefetches_to_device", False))


def _validate_padding_mask(seq: torch.Tensor, padding_mask: torch.Tensor) -> None:
    """Validate the boolean ``[batch, seq]`` padding-mask contract."""
    if padding_mask.dtype != torch.bool:
        raise TrainingError("padding_mask must use bool dtype (True = padding position).")
    if padding_mask.ndim != 2:
        raise TrainingError(
            f"padding_mask must be rank-2 [batch, seq], got rank {padding_mask.ndim}."
        )
    expected_shape = (seq.shape[0], seq.shape[1])
    if tuple(padding_mask.shape) != expected_shape:
        raise TrainingError(
            "padding_mask shape mismatch: expected "
            f"{expected_shape}, got {tuple(padding_mask.shape)}."
        )
    if int((~padding_mask).sum(dtype=torch.int64).item()) <= 0:
        raise TrainingError("Encountered all-padding batch (no valid sequence positions).")


def _compute_error_summaries(
    diff: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total and samplewise squared/absolute errors under the mask contract."""
    if padding_mask is not None:
        valid_steps = ~padding_mask
        valid_mask = valid_steps.unsqueeze(-1)
        sq_error = (diff * diff).masked_fill(~valid_mask, 0.0)
        abs_error = diff.abs().masked_fill(~valid_mask, 0.0)
        sample_sq = sq_error.sum(dim=(1, 2))
        sample_abs = abs_error.sum(dim=(1, 2))
        sample_elements = valid_steps.sum(dim=1, dtype=torch.int64) * diff.shape[-1]
        total_elements = sample_elements.sum()
    else:
        sq_error = diff * diff
        abs_error = diff.abs()
        sample_sq = sq_error.reshape(diff.shape[0], -1).sum(dim=1)
        sample_abs = abs_error.reshape(diff.shape[0], -1).sum(dim=1)
        sample_elements = torch.full(
            (diff.shape[0],),
            diff.shape[1] * diff.shape[2],
            dtype=torch.int64,
            device=diff.device,
        )
        total_elements = torch.tensor(diff.numel(), device=diff.device, dtype=torch.int64)
    return (
        sq_error.sum(),
        abs_error.sum(),
        total_elements,
        sample_sq,
        sample_abs,
        sample_elements,
    )


def _make_dt_bin_edges(min_s: float, max_s: float, *, num_bins: int = 6) -> np.ndarray:
    """Construct deterministic log-spaced dt bins for evaluation summaries."""
    lower = float(min_s)
    upper = float(max_s)
    if lower <= 0.0 or upper <= 0.0:
        raise TrainingError("dt bin bounds must be strictly positive.")
    if upper < lower:
        raise TrainingError("dt bin bounds must satisfy min <= max.")
    if math.isclose(lower, upper):
        upper = lower * (1.0 + 1.0e-6)
    edges = np.logspace(np.log10(lower), np.log10(upper), num_bins + 1, dtype=np.float64)
    edges[-1] = np.nextafter(edges[-1], np.float64(np.inf))
    return edges


def _evaluate(
    *,
    model: nn.Module,
    loader: Any,
    device: torch.device,
    preload_to_device: bool,
    forward_dtype: torch.dtype,
    loss_dtype: torch.dtype,
    stats_dtype: torch.dtype,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
    dt_bin_edges: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate one split and optionally accumulate dt-binned metrics."""
    model.eval()
    loader_prefetches = _loader_prefetches_to_device(loader)
    total_sq_error = torch.zeros((), device=device, dtype=stats_dtype)
    total_abs_error = torch.zeros((), device=device, dtype=stats_dtype)
    total_elements = torch.zeros((), device=device, dtype=torch.int64)

    edges_tensor: torch.Tensor | None = None
    bin_sq: torch.Tensor | None = None
    bin_abs: torch.Tensor | None = None
    bin_elements: torch.Tensor | None = None
    bin_counts: torch.Tensor | None = None
    if dt_bin_edges is not None:
        edges_tensor = torch.as_tensor(dt_bin_edges, dtype=torch.float64, device=device)
        num_bins = int(edges_tensor.numel() - 1)
        bin_sq = torch.zeros((num_bins,), device=device, dtype=stats_dtype)
        bin_abs = torch.zeros((num_bins,), device=device, dtype=stats_dtype)
        bin_elements = torch.zeros((num_bins,), device=device, dtype=torch.int64)
        bin_counts = torch.zeros((num_bins,), device=device, dtype=torch.int64)

    with torch.no_grad():
        for batch in loader:
            seq, glb, tgt, padding_mask, dt = _unpack_batch(batch)
            if not preload_to_device and not loader_prefetches:
                seq = seq.to(device=device, non_blocking=True)
                glb = glb.to(device=device, non_blocking=True)
                tgt = tgt.to(device=device, non_blocking=True)
                dt = dt.to(device=device, non_blocking=True)
                if padding_mask is not None:
                    padding_mask = padding_mask.to(device=device, non_blocking=True)

            seq = seq.to(dtype=forward_dtype)
            glb = glb.to(dtype=forward_dtype)
            tgt = tgt.to(dtype=loss_dtype)
            dt = dt.to(dtype=loss_dtype)
            if padding_mask is not None and padding_mask.device != device:
                padding_mask = padding_mask.to(device=device, non_blocking=True)

            if padding_mask is not None:
                _validate_padding_mask(seq, padding_mask)

            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                pred = model(seq, glb, padding_mask=padding_mask)

            diff = pred.to(dtype=loss_dtype) - tgt
            (
                sq_error_sum,
                abs_error_sum,
                batch_elements,
                sample_sq,
                sample_abs,
                sample_elements,
            ) = _compute_error_summaries(diff, padding_mask)

            total_sq_error += sq_error_sum.detach().to(dtype=stats_dtype)
            total_abs_error += abs_error_sum.detach().to(dtype=stats_dtype)
            total_elements += batch_elements.detach().to(dtype=torch.int64)

            if edges_tensor is not None and bin_sq is not None and bin_abs is not None and bin_elements is not None and bin_counts is not None:
                bucket = torch.bucketize(
                    dt.detach().to(dtype=torch.float64),
                    edges_tensor[1:-1],
                    right=False,
                )
                bin_sq.index_add_(0, bucket, sample_sq.detach().to(dtype=stats_dtype))
                bin_abs.index_add_(0, bucket, sample_abs.detach().to(dtype=stats_dtype))
                bin_elements.index_add_(0, bucket, sample_elements.detach().to(dtype=torch.int64))
                bin_counts.index_add_(
                    0,
                    bucket,
                    torch.ones_like(bucket, dtype=torch.int64, device=device),
                )

    if int(total_elements.item()) <= 0:
        raise TrainingError("Encountered empty loader during evaluation.")

    metrics: dict[str, Any] = {
        "mse": float(
            (total_sq_error / total_elements.to(dtype=stats_dtype).clamp_min(1.0)).item()
        ),
        "mae": float(
            (total_abs_error / total_elements.to(dtype=stats_dtype).clamp_min(1.0)).item()
        ),
    }

    if (
        dt_bin_edges is not None
        and bin_sq is not None
        and bin_abs is not None
        and bin_elements is not None
        and bin_counts is not None
    ):
        dt_bins: list[dict[str, Any]] = []
        for idx in range(len(dt_bin_edges) - 1):
            count = int(bin_counts[idx].item())
            elements = int(bin_elements[idx].item())
            mse: float | None = None
            mae: float | None = None
            if elements > 0:
                mse = float((bin_sq[idx] / max(elements, 1)).item())
                mae = float((bin_abs[idx] / max(elements, 1)).item())
            dt_bins.append(
                {
                    "lower_s": float(dt_bin_edges[idx]),
                    "upper_s": float(dt_bin_edges[idx + 1]),
                    "count": count,
                    "elements": elements,
                    "mse": mse,
                    "mae": mae,
                }
            )
        metrics["dt_bins"] = dt_bins
    return metrics


def _persist_inference_artifacts(
    *,
    run_dir: Path,
    normalization_metadata: dict[str, Any],
    data_contract: dict[str, Any],
    fingerprint_path: Path,
) -> None:
    """Write the files required by standalone inference utilities."""
    with (run_dir / "data_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(data_contract, handle, indent=2)
    with (run_dir / "normalization_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(normalization_metadata, handle, indent=2)
    shutil.copy2(fingerprint_path, run_dir / PROCESSED_FINGERPRINT_FILENAME)


def validate_processed_split_contract(
    *,
    config: dict[str, Any],
    train_meta: dict[str, Any],
    val_meta: dict[str, Any],
    test_meta: dict[str, Any],
    expected_norm_fingerprint: str,
) -> None:
    """Validate that processed split metadata matches the transition-model contract."""
    shape_tuple = (
        train_meta["sequence_length"],
        train_meta["input_dim"],
        train_meta["global_dim"],
        train_meta["target_dim"],
        train_meta["state_dim"],
    )
    for meta in (val_meta, test_meta):
        other = (
            meta["sequence_length"],
            meta["input_dim"],
            meta["global_dim"],
            meta["target_dim"],
            meta["state_dim"],
        )
        if other != shape_tuple:
            raise TrainingError("Processed split shape mismatch across train/val/test.")

    expected_state_species = list(config["data_spec"]["state_species"])
    expected_output_species = list(config["data_spec"]["output_species"])
    expected_sequence_order = [
        "pressure_bar",
        "temperature_k",
        "kzz_cm2_s",
        *[f"anchor_ymix:{species_name}" for species_name in expected_state_species],
    ]
    expected_global_order = list(config["data_spec"]["required_global_inputs"])
    expected_output_from_state = [expected_state_species.index(species) for species in expected_output_species]

    if train_meta["sequence_feature_order"] != expected_sequence_order:
        raise TrainingError(
            "Processed sequence feature order does not match configured data_spec/order contract."
        )
    if train_meta["global_feature_order"] != expected_global_order:
        raise TrainingError(
            "Processed global feature order does not match configured data_spec/order contract."
        )
    if train_meta["state_species_order"] != expected_state_species:
        raise TrainingError(
            "Processed state species order does not match data_spec.state_species."
        )
    if train_meta["output_species_order"] != expected_output_species:
        raise TrainingError(
            "Processed output species order does not match data_spec.output_species."
        )
    if train_meta["output_from_state_indices"] != expected_output_from_state:
        raise TrainingError(
            "Processed output_from_state_indices does not match the configured species subset mapping."
        )
    if train_meta["normalization_fingerprint"] != expected_norm_fingerprint:
        raise TrainingError(
            "Processed metadata normalization fingerprint does not match normalization_metadata.json."
        )

    for meta in (val_meta, test_meta):
        for key in (
            "sequence_feature_order",
            "global_feature_order",
            "state_species_order",
            "output_species_order",
            "output_from_state_indices",
            "normalization_fingerprint",
        ):
            if meta[key] != train_meta[key]:
                raise TrainingError(f"Processed split field '{key}' mismatch across train/val/test.")
        if float(meta["dt_min_s"]) <= 0.0 or float(meta["dt_max_s"]) < float(meta["dt_min_s"]):
            raise TrainingError("Invalid dt range recorded in processed split metadata.")


def _build_data_contract(train_meta: dict[str, Any]) -> dict[str, Any]:
    """Build the standalone inference contract from train-split metadata."""
    return {
        "sequence_length": int(train_meta["sequence_length"]),
        "input_dim": int(train_meta["input_dim"]),
        "global_dim": int(train_meta["global_dim"]),
        "target_dim": int(train_meta["target_dim"]),
        "state_dim": int(train_meta["state_dim"]),
        "sequence_feature_order": list(train_meta["sequence_feature_order"]),
        "global_feature_order": list(train_meta["global_feature_order"]),
        "state_species_order": list(train_meta["state_species_order"]),
        "output_species_order": list(train_meta["output_species_order"]),
        "output_from_state_indices": list(train_meta["output_from_state_indices"]),
        "normalization_fingerprint": str(train_meta["normalization_fingerprint"]),
    }


def _parse_run_id_from_path(path: Path) -> int:
    """Extract the integer run id from a canonical raw-run filename."""
    match = re.search(r"run_(\d+)$", path.stem)
    if match is None:
        raise TrainingError(f"Unable to parse run id from raw run path: {path}")
    return int(match.group(1))


def _load_split_assignments(split_path: Path) -> dict[str, list[int]]:
    """Load the serialized train/val/test run split mapping."""
    with split_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TrainingError(f"Invalid split file structure: {split_path}")
    result: dict[str, list[int]] = {}
    for key in ("train", "val", "test"):
        values = payload.get(key)
        if not isinstance(values, list):
            raise TrainingError(f"Invalid split entry '{key}' in {split_path}")
        result[key] = [int(value) for value in values]
    return result


def _rollout_eval_indices(
    *,
    times_s: np.ndarray,
    rollout_eval_points: int,
) -> np.ndarray:
    """Build a deterministic rollout path on one saved trajectory."""
    if rollout_eval_points < 2:
        raise TrainingError("Rollout evaluation requires at least two checkpoint points.")
    try:
        return build_rollout_indices(
            times_s=times_s,
            max_points=rollout_eval_points,
        )
    except TransitionSamplingError as exc:
        raise TrainingError(f"Failed to construct rollout path: {exc}") from exc


def _compute_rollout_metrics(
    *,
    model: VulcanTransitionTransformer,
    normalization_metadata: dict[str, Any],
    data_contract: dict[str, Any],
    raw_run_files: list[Path],
    split_path: Path,
    rollout_eval_points: int,
    config: dict[str, Any],
    device: torch.device,
    forward_dtype: torch.dtype,
) -> dict[str, Any]:
    """Validate autoregressive multi-jump composition on held-out raw runs."""
    state_species = list(data_contract["state_species_order"])
    output_species = list(data_contract["output_species_order"])
    if state_species != output_species:
        return {
            "skipped": True,
            "reason": "rollout requires output_species == state_species",
        }

    split_assignments = _load_split_assignments(split_path)
    test_ids = set(split_assignments["test"])
    if not test_ids:
        return {"skipped": True, "reason": "test split contains no raw runs"}

    run_file_by_id = {_parse_run_id_from_path(path): path for path in raw_run_files}
    wrapper = PhysicalSpaceStandaloneModel(
        model=model,
        normalization_metadata=normalization_metadata,
        data_contract=data_contract,
    ).to(device=device, dtype=forward_dtype)
    wrapper.eval()
    predictor = VulcanPredictor(wrapper, device=device)

    total_sq_error = 0.0
    total_abs_error = 0.0
    total_elements = 0
    total_steps = 0

    for run_id in sorted(test_ids):
        run_file = run_file_by_id.get(run_id)
        if run_file is None:
            raise TrainingError(f"Missing raw run file for test run id {run_id}.")
        raw = load_raw_run_file(
            run_file,
            state_species=state_species,
            output_species=output_species,
        )
        static_globals = resolve_conditioning_inputs(
            raw_global_inputs=raw.global_inputs,
            config=config,
            required_global_inputs=list(data_contract["global_feature_order"]),
        )
        try:
            eval_indices = _rollout_eval_indices(
                times_s=raw.time_s,
                rollout_eval_points=rollout_eval_points,
            )
        except TrainingError:
            continue
        if eval_indices.size < 2:
            continue
        current_state = np.asarray(raw.ymix_state[eval_indices[0]], dtype=np.float64)
        prev_index = int(eval_indices[0])

        for next_index in eval_indices[1:]:
            next_index_int = int(next_index)
            dt_s = float(raw.time_s[next_index_int] - raw.time_s[prev_index])
            prediction = predictor.predict(
                pressure_bar=raw.pressure_bar,
                temperature_k=raw.temperature_k,
                kzz_cm2_s=raw.kzz_cm2_s,
                anchor_ymix=current_state,
                gravity_cm_s2=static_globals["gravity_cm_s2"],
                metallicity_log10=static_globals["metallicity_log10"],
                c_to_o=static_globals["c_to_o"],
                dt_s=dt_s,
                extra_global_inputs={
                    key: value
                    for key, value in static_globals.items()
                    if key not in {"gravity_cm_s2", "metallicity_log10", "c_to_o"}
                },
            )
            target = np.asarray(raw.ymix_output[next_index_int], dtype=np.float64)
            diff = np.asarray(prediction, dtype=np.float64) - target
            total_sq_error += float(np.sum(diff * diff))
            total_abs_error += float(np.sum(np.abs(diff)))
            total_elements += int(diff.size)
            total_steps += 1
            current_state = np.asarray(prediction, dtype=np.float64)
            prev_index = next_index_int

    if total_elements <= 0 or total_steps <= 0:
        return {"skipped": True, "reason": "no rollout transitions were evaluated"}

    return {
        "skipped": False,
        "num_steps": total_steps,
        "mse": total_sq_error / total_elements,
        "mae": total_abs_error / total_elements,
    }


def run_training(config: dict[str, Any], paths: Any, precision: PrecisionConfig) -> None:
    """Execute ``--train`` on processed transition shards."""
    training = config["training"]
    _seed_everything(int(training["seed"]))

    try:
        artifact_info = validate_processed_artifacts(config=config, paths=paths)
    except Exception as exc:
        raise TrainingError(str(exc)) from exc

    device = _resolve_device(training["device"])
    _configure_cpu_threads(device)
    preload = bool(training["gpu_preload"])
    batch_size = int(training["batch_size"])
    num_workers = int(training["num_workers"])
    loading_cfg = training["data_loading"]
    data_loading = DataLoadingConfig(
        mode=str(loading_cfg["mode"]).lower(),
        max_cached_shards=int(loading_cfg["max_cached_shards"]),
        large_shard_mmap_bytes=int(loading_cfg["large_shard_mmap_bytes"]),
        ram_safety_fraction=float(loading_cfg["ram_safety_fraction"]),
        copy_mmap_slices=bool(loading_cfg["copy_mmap_slices"]),
        use_device_prefetch=bool(loading_cfg["use_device_prefetch"]),
    )

    train_dir = paths.processed_root / "train"
    val_dir = paths.processed_root / "val"
    test_dir = paths.processed_root / "test"
    norm_path = artifact_info["normalization_path"]
    if not norm_path.is_file():
        raise TrainingError(f"Missing normalization metadata: {norm_path}. Run --gen before --train.")

    with norm_path.open("r", encoding="utf-8") as handle:
        normalization_metadata = json.load(handle)
    expected_norm_fingerprint = sha256(
        json.dumps(normalization_metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    logger.info("Loading processed datasets...")
    train_loader, train_meta = _build_loader(
        split_dir=train_dir,
        batch_size=batch_size,
        shuffle=True,
        device=device,
        preload_to_device=preload,
        num_workers=num_workers,
        input_dtype=precision.input_dtype,
        target_dtype=precision.loss_dtype,
        loading=data_loading,
    )
    val_loader, val_meta = _build_loader(
        split_dir=val_dir,
        batch_size=batch_size,
        shuffle=False,
        device=device,
        preload_to_device=preload,
        num_workers=num_workers,
        input_dtype=precision.input_dtype,
        target_dtype=precision.loss_dtype,
        loading=data_loading,
    )
    test_loader, test_meta = _build_loader(
        split_dir=test_dir,
        batch_size=batch_size,
        shuffle=False,
        device=device,
        preload_to_device=preload,
        num_workers=num_workers,
        input_dtype=precision.input_dtype,
        target_dtype=precision.loss_dtype,
        loading=data_loading,
    )

    validate_processed_split_contract(
        config=config,
        train_meta=train_meta,
        val_meta=val_meta,
        test_meta=test_meta,
        expected_norm_fingerprint=expected_norm_fingerprint,
    )

    model_cfg = training["model"]
    if int(train_meta["sequence_length"]) > int(model_cfg["max_sequence_length"]):
        raise TrainingError(
            "Processed sequence length exceeds training.model.max_sequence_length: "
            f"{train_meta['sequence_length']} > {model_cfg['max_sequence_length']}"
        )

    model = VulcanTransitionTransformer(
        state_dim=int(train_meta["state_dim"]),
        output_dim=int(train_meta["target_dim"]),
        output_from_state_indices=list(train_meta["output_from_state_indices"]),
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        num_layers=int(model_cfg["num_layers"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        dropout=float(model_cfg["dropout"]),
        film_clamp=float(model_cfg["film_clamp"]),
        output_head_divisor=int(model_cfg["output_head_divisor"]),
        max_sequence_length=int(model_cfg["max_sequence_length"]),
        conditioning_hidden_dim=int(model_cfg["conditioning_hidden_dim"]),
        num_globals=int(train_meta["global_dim"]),
    ).to(device=device, dtype=precision.model_dtype)

    optimizer = _build_optimizer(
        model=model,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    epochs = int(training["epochs"])
    warmup_epochs = int(training["warmup_epochs"])
    min_lr = float(training["min_lr"])
    base_lr = float(training["learning_rate"])
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = warmup_epochs * len(train_loader)
    min_factor = min_lr / base_lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps and warmup_steps > 0:
            return max((step + 1) / warmup_steps, min_factor)
        remain = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / remain, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler_enabled = precision.use_amp and precision.amp_dtype == torch.float16
    scaler = GradScaler(enabled=scaler_enabled)

    run_dir = paths.models_root / str(training["output_folder"])
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    data_contract = _build_data_contract(train_meta)
    _persist_inference_artifacts(
        run_dir=run_dir,
        normalization_metadata=normalization_metadata,
        data_contract=data_contract,
        fingerprint_path=artifact_info["fingerprint_path"],
    )

    history_path = run_dir / "training_log.csv"
    progress_log_path = paths.logs_root / f"training_progress_{training['output_folder']}.log"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["epoch", "train_mse", "val_mse", "val_mae", "lr", "epoch_seconds", "elapsed_seconds"]
        )
    with progress_log_path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"{'epoch':>13} {'train_mse':>15} {'val_mse':>15} {'val_mae':>15} "
            f"{'lr':>15} {'epoch_s':>15} {'elapsed_s':>15}\n"
        )

    best_val_mse = float("inf")
    best_epoch = 0
    global_step = 0
    gradient_clip = float(training["gradient_clip"])
    train_loader_prefetches = _loader_prefetches_to_device(train_loader)
    training_start = time.perf_counter()

    logger.info("Starting training for %d epochs", epochs)
    logger.info("Detailed training progress log: %s", progress_log_path)
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_sq_error = torch.zeros((), device=device, dtype=precision.stats_dtype)
        train_elements = torch.zeros((), device=device, dtype=torch.int64)

        for batch in train_loader:
            seq, glb, tgt, padding_mask, _dt = _unpack_batch(batch)
            if not preload and not train_loader_prefetches:
                seq = seq.to(device=device, non_blocking=True)
                glb = glb.to(device=device, non_blocking=True)
                tgt = tgt.to(device=device, non_blocking=True)
                if padding_mask is not None:
                    padding_mask = padding_mask.to(device=device, non_blocking=True)

            seq = seq.to(dtype=precision.forward_dtype)
            glb = glb.to(dtype=precision.forward_dtype)
            tgt = tgt.to(dtype=precision.loss_dtype)
            if padding_mask is not None and padding_mask.device != device:
                padding_mask = padding_mask.to(device=device, non_blocking=True)
            if padding_mask is not None:
                _validate_padding_mask(seq, padding_mask)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=precision.use_amp, dtype=precision.amp_dtype):
                pred = model(seq, glb, padding_mask=padding_mask)
                diff = pred.to(dtype=precision.loss_dtype) - tgt
                sq_error_sum, _abs_error_sum, batch_elements, _sample_sq, _sample_abs, _sample_elements = _compute_error_summaries(
                    diff,
                    padding_mask,
                )
                loss = sq_error_sum / batch_elements.to(dtype=precision.loss_dtype).clamp_min(1.0)

            if not torch.isfinite(loss):
                raise TrainingError("Encountered non-finite training loss.")

            if scaler_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

            if precision.optimizer_state_dtype != precision.model_dtype:
                _cast_optimizer_state(optimizer, precision.optimizer_state_dtype)

            scheduler.step()
            global_step += 1
            train_sq_error += sq_error_sum.detach().to(dtype=precision.stats_dtype)
            train_elements += batch_elements.detach().to(dtype=torch.int64)

        if int(train_elements.item()) <= 0:
            raise TrainingError("Empty train loader encountered.")

        train_mse = float(
            (train_sq_error / train_elements.to(dtype=precision.stats_dtype).clamp_min(1.0)).item()
        )
        val_metrics = _evaluate(
            model=model,
            loader=val_loader,
            device=device,
            preload_to_device=preload,
            forward_dtype=precision.forward_dtype,
            loss_dtype=precision.loss_dtype,
            stats_dtype=precision.stats_dtype,
            use_amp=precision.use_amp,
            amp_dtype=precision.amp_dtype,
        )
        lr_current = float(optimizer.param_groups[0]["lr"])
        epoch_seconds = float(time.perf_counter() - epoch_start)
        elapsed_seconds = float(time.perf_counter() - training_start)
        progress_line = _format_progress_line(
            epoch=epoch,
            epochs=epochs,
            train_mse=train_mse,
            val_mse=float(val_metrics["mse"]),
            val_mae=float(val_metrics["mae"]),
            lr=lr_current,
            epoch_seconds=epoch_seconds,
            elapsed_seconds=elapsed_seconds,
        )
        logger.info("%s", progress_line)

        with history_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    epoch,
                    train_mse,
                    val_metrics["mse"],
                    val_metrics["mae"],
                    lr_current,
                    epoch_seconds,
                    elapsed_seconds,
                ]
            )
        with progress_log_path.open("a", encoding="utf-8") as handle:
            handle.write(progress_line + "\n")

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler_enabled else None,
            "config": config,
            "data_contract": data_contract,
            "normalization_fingerprint": expected_norm_fingerprint,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if val_metrics["mse"] < best_val_mse:
            best_val_mse = float(val_metrics["mse"])
            best_epoch = epoch
            torch.save(checkpoint, run_dir / "best.pt")

    best_ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    val_dt_edges = _make_dt_bin_edges(float(val_meta["dt_min_s"]), float(val_meta["dt_max_s"]))
    test_dt_edges = _make_dt_bin_edges(float(test_meta["dt_min_s"]), float(test_meta["dt_max_s"]))
    final_val_metrics = _evaluate(
        model=model,
        loader=val_loader,
        device=device,
        preload_to_device=preload,
        forward_dtype=precision.forward_dtype,
        loss_dtype=precision.loss_dtype,
        stats_dtype=precision.stats_dtype,
        use_amp=precision.use_amp,
        amp_dtype=precision.amp_dtype,
        dt_bin_edges=val_dt_edges,
    )
    final_test_metrics = _evaluate(
        model=model,
        loader=test_loader,
        device=device,
        preload_to_device=preload,
        forward_dtype=precision.forward_dtype,
        loss_dtype=precision.loss_dtype,
        stats_dtype=precision.stats_dtype,
        use_amp=precision.use_amp,
        amp_dtype=precision.amp_dtype,
        dt_bin_edges=test_dt_edges,
    )
    rollout_metrics = _compute_rollout_metrics(
        model=model,
        normalization_metadata=normalization_metadata,
        data_contract=data_contract,
        raw_run_files=artifact_info["raw_run_files"],
        split_path=artifact_info["split_path"],
        rollout_eval_points=int(config["trajectory_sampling"]["rollout_eval_points"]),
        config=config,
        device=device,
        forward_dtype=precision.forward_dtype,
    )

    metrics = {
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "val": final_val_metrics,
        "test": final_test_metrics,
        "rollout": rollout_metrics,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    logger.info("Training complete. Final metrics: %s", metrics)
