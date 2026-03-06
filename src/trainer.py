"""Training pipeline for vulcan-emulator."""

from __future__ import annotations

import csv
import json
import logging
import math
import random
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config_utils import PrecisionConfig
from data_loader import DataLoadingConfig, DevicePrefetchLoader, build_training_loader
from model import VulcanTransformer
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
    """Build one split loader and normalize loader errors into `TrainingError`.

    Args:
        split_dir: Processed split directory.
        batch_size: Number of samples per batch.
        shuffle: Whether sample order should be shuffled.
        device: Target Torch device.
        preload_to_device: Whether the full split should be preloaded to `device`.
        num_workers: Host-side DataLoader worker count.
        input_dtype: Torch dtype for sequence/global tensors.
        target_dtype: Torch dtype for target tensors.
        loading: Data-loading policy controlling RAM/disk/prefetch behavior.

    Returns:
        Tuple `(loader, metadata)` where `loader` yields `(seq, glb, tgt, mask)` batches and
        `metadata` is the validated processed-split metadata dictionary.
    """
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
            if (
                torch.is_tensor(value)
                and torch.is_floating_point(value)
                and value.dtype != dtype
            ):
                state[key] = value.to(dtype=dtype)


def _unpack_batch(
    batch: tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Support both 3-tensor and 4-tensor batch contracts."""
    if len(batch) == 3:
        seq, glb, tgt = batch
        return seq, glb, tgt, None
    if len(batch) == 4:
        seq, glb, tgt, padding_mask = batch
        return seq, glb, tgt, padding_mask.bool()
    raise TrainingError(
        f"Unexpected batch structure with {len(batch)} tensors; expected 3 or 4."
    )


def _loader_prefetches_to_device(loader: Any) -> bool:
    """Return whether the loader already yields tensors on the target device."""
    return bool(getattr(loader, "prefetches_to_device", False))


def _validate_padding_mask(seq: torch.Tensor, padding_mask: torch.Tensor) -> None:
    """Validate the boolean `[batch, seq]` padding-mask contract."""
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
    valid_steps = ~padding_mask
    if int(valid_steps.sum(dtype=torch.int64).item()) <= 0:
        raise TrainingError("Encountered all-padding batch (no valid sequence positions).")


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
) -> dict[str, float]:
    """Evaluate one split and accumulate metrics in the configured stats dtype.

    Args:
        model: Model being evaluated.
        loader: Batch iterator yielding either 3-tensor or 4-tensor batches.
        device: Device on which evaluation runs.
        preload_to_device: Whether batches are already preloaded to `device`.
        forward_dtype: Dtype used for model inputs.
        loss_dtype: Dtype used for loss/error calculations.
        stats_dtype: Dtype used for metric accumulation.
        use_amp: Whether autocast should be enabled.
        amp_dtype: Autocast dtype when AMP is enabled.

    Returns:
        Dictionary with scalar float metrics `{"mse": ..., "mae": ...}`.
    """
    model.eval()
    loader_prefetches = _loader_prefetches_to_device(loader)
    total_sq_error = torch.zeros((), device=device, dtype=stats_dtype)
    total_abs_error = torch.zeros((), device=device, dtype=stats_dtype)
    total_elements = torch.zeros((), device=device, dtype=torch.int64)

    with torch.no_grad():
        for batch in loader:
            seq, glb, tgt, padding_mask = _unpack_batch(batch)
            if not preload_to_device and not loader_prefetches:
                seq = seq.to(device=device, non_blocking=True)
                glb = glb.to(device=device, non_blocking=True)
                tgt = tgt.to(device=device, non_blocking=True)
                if padding_mask is not None:
                    padding_mask = padding_mask.to(device=device, non_blocking=True)

            seq = seq.to(dtype=forward_dtype)
            glb = glb.to(dtype=forward_dtype)
            tgt = tgt.to(dtype=loss_dtype)
            if padding_mask is not None and padding_mask.device != device:
                padding_mask = padding_mask.to(device=device, non_blocking=True)

            if padding_mask is not None:
                _validate_padding_mask(seq, padding_mask)

            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                pred = model(seq, glb, padding_mask=padding_mask)

            diff = pred.to(dtype=loss_dtype) - tgt
            if padding_mask is not None:
                valid_steps = ~padding_mask
                valid_mask = valid_steps.unsqueeze(-1)
                sq_error = (diff * diff).masked_fill(~valid_mask, 0.0)
                abs_error = diff.abs().masked_fill(~valid_mask, 0.0)
                batch_elements = valid_steps.sum(dtype=torch.int64) * diff.shape[-1]
            else:
                sq_error = diff * diff
                abs_error = diff.abs()
                batch_elements = torch.tensor(diff.numel(), device=device, dtype=torch.int64)

            total_sq_error += sq_error.sum().detach().to(dtype=stats_dtype)
            total_abs_error += abs_error.sum().detach().to(dtype=stats_dtype)
            total_elements += batch_elements

    if int(total_elements.item()) <= 0:
        raise TrainingError("Empty evaluation loader encountered.")

    denom = total_elements.to(dtype=stats_dtype).clamp_min(1.0)
    return {
        "mse": float((total_sq_error / denom).item()),
        "mae": float((total_abs_error / denom).item()),
    }


def _persist_inference_artifacts(
    *,
    run_dir: Path,
    norm_stats: dict[str, Any],
    train_meta: dict[str, Any],
    fingerprint_path: Path,
) -> None:
    """Write the artifacts required for standalone inference and provenance checks."""
    data_contract = {
        "sequence_length": int(train_meta["sequence_length"]),
        "input_dim": int(train_meta["input_dim"]),
        "global_dim": int(train_meta["global_dim"]),
        "target_dim": int(train_meta["target_dim"]),
        "sequence_feature_order": list(train_meta["sequence_feature_order"]),
        "global_feature_order": list(train_meta["global_feature_order"]),
        "target_species_order": list(train_meta["target_species_order"]),
        "normalization_fingerprint": str(train_meta["normalization_fingerprint"]),
    }

    with (run_dir / "data_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(data_contract, handle, indent=2)
    with (run_dir / "normalization_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(norm_stats, handle, indent=2)
    shutil.copy2(fingerprint_path, run_dir / PROCESSED_FINGERPRINT_FILENAME)


def validate_processed_split_contract(
    *,
    config: dict[str, Any],
    train_meta: dict[str, Any],
    val_meta: dict[str, Any],
    test_meta: dict[str, Any],
    expected_norm_fingerprint: str,
) -> None:
    """Validate that processed split metadata matches the configured v1 contract.

    Args:
        config: Fully validated project configuration dictionary.
        train_meta: Metadata dictionary for the train split.
        val_meta: Metadata dictionary for the validation split.
        test_meta: Metadata dictionary for the test split.
        expected_norm_fingerprint: SHA-256 fingerprint expected for normalization metadata.
    """
    shape_tuple = (
        train_meta["sequence_length"],
        train_meta["input_dim"],
        train_meta["global_dim"],
        train_meta["target_dim"],
    )
    for meta in (val_meta, test_meta):
        other = (
            meta["sequence_length"],
            meta["input_dim"],
            meta["global_dim"],
            meta["target_dim"],
        )
        if other != shape_tuple:
            raise TrainingError("Processed split shape mismatch across train/val/test.")

    expected_species = list(config["data_spec"]["target_species"])
    expected_sequence_order = [
        "pressure_bar",
        "temperature_k",
        "kzz_cm2_s",
        *[f"initial_ymix:{sp}" for sp in expected_species],
    ]
    expected_global_order = ["gravity_cm_s2", "metallicity_log10", "c_to_o", "log10_time_s"]

    if train_meta["sequence_feature_order"] != expected_sequence_order:
        raise TrainingError(
            "Processed sequence feature order does not match configured data_spec/order contract."
        )
    if train_meta["global_feature_order"] != expected_global_order:
        raise TrainingError(
            "Processed global feature order does not match configured data_spec/order contract."
        )
    if train_meta["target_species_order"] != expected_species:
        raise TrainingError(
            "Processed target species order does not match data_spec.target_species."
        )
    if train_meta["normalization_fingerprint"] != expected_norm_fingerprint:
        raise TrainingError(
            "Processed metadata normalization fingerprint does not match "
            "normalization_metadata.json."
        )

    for meta in (val_meta, test_meta):
        if meta["sequence_feature_order"] != train_meta["sequence_feature_order"]:
            raise TrainingError(
                "Processed split sequence feature order mismatch across train/val/test."
            )
        if meta["global_feature_order"] != train_meta["global_feature_order"]:
            raise TrainingError(
                "Processed split global feature order mismatch across train/val/test."
            )
        if meta["target_species_order"] != train_meta["target_species_order"]:
            raise TrainingError(
                "Processed split target species order mismatch across train/val/test."
            )
        if meta["normalization_fingerprint"] != train_meta["normalization_fingerprint"]:
            raise TrainingError(
                "Processed split normalization fingerprint mismatch across train/val/test."
            )


def run_training(config: dict[str, Any], paths: Any, precision: PrecisionConfig) -> None:
    """Execute `--train` on processed shards.

    Args:
        config: Fully validated project configuration dictionary.
        paths: Resolved path bundle with processed-data and model-output locations.
        precision: Resolved precision policy controlling model, loss, and stats dtypes.
    """
    training = config["training"]
    _seed_everything(int(training["seed"]))

    try:
        artifact_info = validate_processed_artifacts(config=config, paths=paths)
    except Exception as exc:
        raise TrainingError(str(exc)) from exc

    device = _resolve_device(training["device"])
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
        raise TrainingError(
            f"Missing normalization metadata: {norm_path}. Run --gen before --train."
        )
    try:
        with norm_path.open("r", encoding="utf-8") as handle:
            norm_stats = json.load(handle)
    except json.JSONDecodeError as exc:
        raise TrainingError(f"Invalid JSON in normalization metadata: {norm_path}") from exc
    expected_norm_fingerprint = sha256(
        json.dumps(norm_stats, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    for split_dir in (train_dir, val_dir, test_dir):
        if not split_dir.is_dir():
            raise TrainingError(
                f"Missing processed split directory: {split_dir}. Run --gen before --train."
            )

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
    model = VulcanTransformer(
        input_dim=int(train_meta["input_dim"]),
        global_dim=int(train_meta["global_dim"]),
        target_dim=int(train_meta["target_dim"]),
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        num_layers=int(model_cfg["num_layers"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        dropout=float(model_cfg["dropout"]),
        film_clamp=float(model_cfg["film_clamp"]),
        output_head_divisor=int(model_cfg["output_head_divisor"]),
        max_sequence_length=int(model_cfg["max_sequence_length"]),
    ).to(device=device, dtype=precision.model_dtype)

    if int(train_meta["sequence_length"]) > int(model_cfg["max_sequence_length"]):
        raise TrainingError(
            "Processed sequence length exceeds training.model.max_sequence_length: "
            f"{train_meta['sequence_length']} > {model_cfg['max_sequence_length']}"
        )

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
    run_dir.mkdir(parents=True, exist_ok=True)
    _persist_inference_artifacts(
        run_dir=run_dir,
        norm_stats=norm_stats,
        train_meta=train_meta,
        fingerprint_path=artifact_info["fingerprint_path"],
    )

    history_path = run_dir / "training_log.csv"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_mse", "val_mse", "val_mae", "lr"])

    best_val = float("inf")
    global_step = 0

    logger.info("Starting training for %d epochs", epochs)
    train_loader_prefetches = _loader_prefetches_to_device(train_loader)
    for epoch in range(1, epochs + 1):
        model.train()
        train_sq_error = torch.zeros((), device=device, dtype=precision.stats_dtype)
        train_elements = torch.zeros((), device=device, dtype=torch.int64)

        for batch in train_loader:
            seq, glb, tgt, padding_mask = _unpack_batch(batch)
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

            with autocast(
                device_type=device.type, enabled=precision.use_amp, dtype=precision.amp_dtype
            ):
                pred = model(seq, glb, padding_mask=padding_mask)
                diff = pred.to(dtype=precision.loss_dtype) - tgt
                if padding_mask is not None:
                    valid_steps = ~padding_mask
                    valid_mask = valid_steps.unsqueeze(-1)
                    sq_error = (diff * diff).masked_fill(~valid_mask, 0.0)
                    sq_error_sum = sq_error.sum()
                    batch_elements = valid_steps.sum(dtype=torch.int64) * diff.shape[-1]
                else:
                    sq_error = diff * diff
                    sq_error_sum = sq_error.sum()
                    batch_elements = torch.tensor(diff.numel(), device=device, dtype=torch.int64)

                loss = sq_error_sum / batch_elements.to(dtype=precision.loss_dtype).clamp_min(1.0)

            if not torch.isfinite(loss):
                raise TrainingError("Encountered non-finite training loss.")

            if scaler_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
                optimizer.step()

            # Cast optimizer state only when model and optimizer dtypes differ
            # (e.g., mixed-precision training). When they match this is a no-op.
            if precision.optimizer_state_dtype != precision.model_dtype:
                _cast_optimizer_state(optimizer, precision.optimizer_state_dtype)

            scheduler.step()
            global_step += 1

            train_sq_error += sq_error_sum.detach().to(dtype=precision.stats_dtype)
            train_elements += batch_elements

        if int(train_elements.item()) <= 0:
            raise TrainingError("Empty train loader encountered.")

        train_mse = float(
            (
                train_sq_error
                / train_elements.to(dtype=precision.stats_dtype).clamp_min(1.0)
            ).item()
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
        logger.info(
            "Epoch %d/%d | train_mse=%.6e | val_mse=%.6e | val_mae=%.6e | lr=%.3e",
            epoch,
            epochs,
            train_mse,
            val_metrics["mse"],
            val_metrics["mae"],
            lr_current,
        )

        with history_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([epoch, train_mse, val_metrics["mse"], val_metrics["mae"], lr_current])

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": config,
        }
        torch.save(checkpoint, run_dir / "last.pt")

        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            torch.save(checkpoint, run_dir / "best.pt")

    # weights_only=False is safe here: checkpoints are self-generated by this pipeline.
    best_ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])

    test_metrics = _evaluate(
        model=model,
        loader=test_loader,
        device=device,
        preload_to_device=preload,
        forward_dtype=precision.forward_dtype,
        loss_dtype=precision.loss_dtype,
        stats_dtype=precision.stats_dtype,
        use_amp=precision.use_amp,
        amp_dtype=precision.amp_dtype,
    )

    metrics = {
        "best_val_mse": best_val,
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    logger.info("Training complete. Test metrics: %s", metrics)
