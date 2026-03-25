from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .numpy_compat import patch_numpy_asarray_copy

patch_numpy_asarray_copy()

import jax
import jax.numpy as jnp
import numpy as np

from .config_utils import effective_transition_sampling, is_equilibrium
from .data_loader import (
    EquilibriumSplit,
    iter_batches,
    iter_equilibrium_batches,
    load_equilibrium_dataset,
    load_processed_dataset,
)
from .export_jax import export_checkpoint_payload
from .jax_model import apply_model
from .live_sampling import sample_eval_rows, sample_train_rows
from .logging_utils import get_logger
from .model import count_parameters, initialize_model
from .path_utils import ensure_dir, resolve_path
from .preprocess import PROCESSED_DATA_VERSION, preprocess_raw_dataset
from .vulcan_runner import generate_raw_dataset
from .transition_sampling import build_candidate_table

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class TrainingArtifacts:
    checkpoint_path: Path
    export_root: Path
    history_path: Path
    metrics_path: Path


def _tree_global_norm(tree: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def _clip_tree(tree: Any, max_norm: float) -> Any:
    norm = _tree_global_norm(tree)
    scale = jnp.minimum(1.0, float(max_norm) / jnp.maximum(norm, 1.0e-12))
    return jax.tree_util.tree_map(lambda x: x * scale, tree)


def _init_adamw_state(params: Any) -> dict[str, Any]:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.asarray(0, dtype=jnp.int32)}


def _adamw_update(
    params: Any,
    grads: Any,
    state: dict[str, Any],
    *,
    learning_rate: jax.Array,
    weight_decay: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1.0e-8,
) -> tuple[Any, dict[str, Any]]:
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m_, g_: beta1 * m_ + (1.0 - beta1) * g_, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v_, g_: beta2 * v_ + (1.0 - beta2) * (g_ * g_), state["v"], grads)
    bias1 = 1.0 - jnp.power(jnp.asarray(beta1, dtype=jnp.float32), t.astype(jnp.float32))
    bias2 = 1.0 - jnp.power(jnp.asarray(beta2, dtype=jnp.float32), t.astype(jnp.float32))
    m_hat = jax.tree_util.tree_map(lambda x: x / bias1, m)
    v_hat = jax.tree_util.tree_map(lambda x: x / bias2, v)
    new_params = jax.tree_util.tree_map(
        lambda p, m_h, v_h: p - learning_rate * (m_h / (jnp.sqrt(v_h) + eps) + weight_decay * p),
        params,
        m_hat,
        v_hat,
    )
    return new_params, {"m": m, "v": v, "t": t}


def _learning_rate_schedule(
    *,
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(max(warmup_steps, 1))
    if total_steps <= warmup_steps:
        return base_lr
    progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _state_stats(normalization: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
    mean = jnp.asarray(normalization["target"]["mean"], dtype=jnp.float32)
    std = jnp.asarray(normalization["target"]["std"], dtype=jnp.float32)
    return mean, std


def make_train_eval_functions(
    *,
    dims,
    normalization: dict[str, Any],
    loss_cfg: dict[str, float],
    gradient_clip: float,
    weight_decay: float,
):
    target_mean, target_std = _state_stats(normalization)

    @jax.jit
    def train_step(params, opt_state, batch, learning_rate):
        def loss_fn(model_params):
            pred, aux = apply_model(
                model_params,
                batch["sequence"],
                batch["global_inputs"],
                batch["spectrum_inputs"],
                dims,
            )
            mse_norm = jnp.mean((pred - batch["target"]) ** 2)
            pred_log10 = pred * target_std + target_mean
            target_log10 = batch["target"] * target_std + target_mean
            mse_log10 = jnp.mean((pred_log10 - target_log10) ** 2)
            spectrum_loss = jnp.asarray(0.0, dtype=jnp.float32)
            reconstruction = aux["spectrum_reconstruction"]
            if reconstruction is not None:
                spectrum_loss = jnp.mean((reconstruction - batch["spectrum_inputs"]) ** 2)
            total = (
                float(loss_cfg["lambda_z"]) * mse_norm
                + float(loss_cfg["lambda_phys"]) * mse_log10
                + float(loss_cfg["lambda_spectrum"]) * spectrum_loss
            )
            metrics = {
                "combined_loss": total,
                "mse_norm": mse_norm,
                "mse_log10": mse_log10,
                "spectrum_recon_mse": spectrum_loss,
            }
            return total, metrics

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = _clip_tree(grads, gradient_clip)
        new_params, new_opt_state = _adamw_update(
            params,
            grads,
            opt_state,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        return new_params, new_opt_state, metrics

    @jax.jit
    def eval_step(params, batch):
        pred, aux = apply_model(
            params,
            batch["sequence"],
            batch["global_inputs"],
            batch["spectrum_inputs"],
            dims,
        )
        mse_norm = jnp.mean((pred - batch["target"]) ** 2)
        pred_log10 = pred * target_std + target_mean
        target_log10 = batch["target"] * target_std + target_mean
        mse_log10 = jnp.mean((pred_log10 - target_log10) ** 2)
        spectrum_loss = jnp.asarray(0.0, dtype=jnp.float32)
        reconstruction = aux["spectrum_reconstruction"]
        if reconstruction is not None:
            spectrum_loss = jnp.mean((reconstruction - batch["spectrum_inputs"]) ** 2)
        total = (
            float(loss_cfg["lambda_z"]) * mse_norm
            + float(loss_cfg["lambda_phys"]) * mse_log10
            + float(loss_cfg["lambda_spectrum"]) * spectrum_loss
        )
        return {
            "combined_loss": total,
            "mse_norm": mse_norm,
            "mse_log10": mse_log10,
            "spectrum_recon_mse": spectrum_loss,
        }

    return train_step, eval_step


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {
            "combined_loss": float("nan"),
            "mse_norm": float("nan"),
            "mse_log10": float("nan"),
            "spectrum_recon_mse": float("nan"),
        }
    keys = metrics[0].keys()
    return {
        key: float(np.mean([metric[key] for metric in metrics]))
        for key in keys
    }


def _ensure_processed(config: dict[str, Any], *, project_root: Path) -> Path:
    processed_root = resolve_path(config["paths"]["processed_root"], project_root)
    contract_path = processed_root / "data_contract.json"
    equilibrium = is_equilibrium(config)
    if contract_path.exists():
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            version_ok = int(contract.get("processed_data_version", -1)) == PROCESSED_DATA_VERSION
            if equilibrium:
                mode_ok = contract.get("model_type") == "equilibrium"
                required_files_ok = all(
                    (processed_root / split / "target_outputs.npy").exists()
                    for split in ("train", "val", "test")
                )
            else:
                mode_ok = str(contract.get("target_mode", "")).lower() == str(
                    config["generation"]["target_mode"]
                ).lower()
                required_files_ok = all(
                    (processed_root / split / "state_trajectories.npy").exists()
                    for split in ("train", "val", "test")
                )
            if version_ok and mode_ok and required_files_ok:
                return processed_root
        except (OSError, ValueError, TypeError):
            pass

    raw_root = resolve_path(config["paths"]["raw_root"], project_root)
    raw_runs_dir = raw_root / "runs"
    raw_run_files = sorted(raw_runs_dir.glob("run_*.h5")) if raw_runs_dir.exists() else []
    if not raw_run_files:
        generate_raw_dataset(config, project_root=project_root)

    preprocess_raw_dataset(config, project_root=project_root)
    return processed_root


def _checkpoint_payload(
    *,
    params: Any,
    dims: Any,
    config: dict[str, Any],
    normalization: dict[str, Any],
    data_contract: dict[str, Any],
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "params": jax.tree_util.tree_map(np.asarray, params),
        "model_dimensions": dims.to_dict(),
        "config": config,
        "normalization": normalization,
        "data_contract": data_contract,
        "metrics": metrics,
        "history": history,
    }


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_equilibrium_train_eval_functions(
    *,
    dims,
    normalization: dict[str, Any],
    loss_cfg: dict[str, float],
    gradient_clip: float,
    weight_decay: float,
):
    """Build JIT-compiled train/eval functions for the equilibrium model."""
    target_mean, target_std = _state_stats(normalization)
    dummy_spectrum = jnp.zeros((1, 0), dtype=jnp.float32)

    @jax.jit
    def train_step(params, opt_state, batch, learning_rate):
        def loss_fn(model_params):
            batch_size = batch["sequence"].shape[0]
            spectrum = jnp.zeros((batch_size, 0), dtype=jnp.float32)
            pred, _aux = apply_model(
                model_params, batch["sequence"], batch["global_inputs"], spectrum, dims,
            )
            mse_norm = jnp.mean((pred - batch["target"]) ** 2)
            pred_log10 = pred * target_std + target_mean
            target_log10 = batch["target"] * target_std + target_mean
            mse_log10 = jnp.mean((pred_log10 - target_log10) ** 2)
            total = (
                float(loss_cfg["lambda_z"]) * mse_norm
                + float(loss_cfg["lambda_phys"]) * mse_log10
            )
            metrics = {
                "combined_loss": total,
                "mse_norm": mse_norm,
                "mse_log10": mse_log10,
            }
            return total, metrics

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = _clip_tree(grads, gradient_clip)
        new_params, new_opt_state = _adamw_update(
            params, grads, opt_state,
            learning_rate=learning_rate, weight_decay=weight_decay,
        )
        return new_params, new_opt_state, metrics

    @jax.jit
    def eval_step(params, batch):
        batch_size = batch["sequence"].shape[0]
        spectrum = jnp.zeros((batch_size, 0), dtype=jnp.float32)
        pred, _aux = apply_model(
            params, batch["sequence"], batch["global_inputs"], spectrum, dims,
        )
        mse_norm = jnp.mean((pred - batch["target"]) ** 2)
        pred_log10 = pred * target_std + target_mean
        target_log10 = batch["target"] * target_std + target_mean
        mse_log10 = jnp.mean((pred_log10 - target_log10) ** 2)
        total = (
            float(loss_cfg["lambda_z"]) * mse_norm
            + float(loss_cfg["lambda_phys"]) * mse_log10
        )
        return {
            "combined_loss": total,
            "mse_norm": mse_norm,
            "mse_log10": mse_log10,
        }

    return train_step, eval_step


def train_equilibrium_model(
    config: dict[str, Any],
    *,
    project_root: Path,
) -> TrainingArtifacts:
    """Train the equilibrium chemistry emulator."""
    processed_root = _ensure_processed(config, project_root=project_root)
    splits, normalization, contract = load_equilibrium_dataset(processed_root)

    train_split = splits["train"]
    val_split = splits["val"]
    test_split = splits.get("test", val_split)

    dims, params = initialize_model(config, contract, seed=int(config["training"]["seed"]))
    LOGGER.info("Initialized equilibrium model with %d parameters.", count_parameters(params))
    opt_state = _init_adamw_state(params)
    train_step, eval_step = make_equilibrium_train_eval_functions(
        dims=dims,
        normalization=normalization,
        loss_cfg=config["training"]["loss"],
        gradient_clip=float(config["training"]["gradient_clip"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    checkpoints_root = resolve_path(config["paths"]["checkpoints_root"], project_root)
    export_root = resolve_path(config["paths"]["jax_export_root"], project_root)
    ensure_dir(checkpoints_root)
    ensure_dir(export_root)

    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    history: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_val = float("inf")
    global_step = 0

    # Estimate total steps for LR schedule.
    steps_per_epoch = max(1, (train_split.num_runs + batch_size - 1) // batch_size)
    warmup_steps = int(config["training"]["warmup_epochs"]) * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    rng = np.random.default_rng(int(config["training"]["seed"]))

    for epoch in range(epochs):
        train_batches = iter_equilibrium_batches(train_split, batch_size=batch_size, rng=rng)
        train_metrics_epoch: list[dict[str, float]] = []

        for batch in train_batches:
            lr = _learning_rate_schedule(
                step=global_step, total_steps=total_steps,
                base_lr=float(config["training"]["learning_rate"]),
                min_lr=float(config["training"]["min_lr"]),
                warmup_steps=warmup_steps,
            )
            device_batch = {key: jnp.asarray(value) for key, value in batch.items()}
            params, opt_state, metrics = train_step(
                params, opt_state, device_batch, jnp.asarray(lr, dtype=jnp.float32),
            )
            train_metrics_epoch.append({key: float(value) for key, value in metrics.items()})
            global_step += 1

        # Validation.
        val_batches = iter_equilibrium_batches(val_split, batch_size=batch_size, rng=rng)
        val_metrics_epoch: list[dict[str, float]] = []
        for batch in val_batches:
            device_batch = {key: jnp.asarray(value) for key, value in batch.items()}
            metrics = eval_step(params, device_batch)
            val_metrics_epoch.append({key: float(value) for key, value in metrics.items()})

        train_summary = _mean_metrics(train_metrics_epoch)
        val_summary = _mean_metrics(val_metrics_epoch)
        record = {"epoch": epoch + 1, "train": train_summary, "val": val_summary}
        history.append(record)
        LOGGER.info(
            "Epoch %d/%d train=%.6e val=%.6e",
            epoch + 1, epochs, train_summary["combined_loss"], val_summary["combined_loss"],
        )

        current_payload = _checkpoint_payload(
            params=params, dims=dims, config=config,
            normalization=normalization, data_contract=contract,
            metrics={"epoch": epoch + 1, "train": train_summary, "val": val_summary},
            history=history,
        )
        _write_checkpoint(checkpoints_root / "last.pt", current_payload)
        if val_summary["combined_loss"] < best_val:
            best_val = val_summary["combined_loss"]
            best_payload = current_payload
            _write_checkpoint(checkpoints_root / "best.pt", current_payload)

    if best_payload is None:
        raise RuntimeError("Training completed without producing a best checkpoint.")

    # Test evaluation.
    best_params = jax.tree_util.tree_map(jnp.asarray, best_payload["params"])
    test_batches = iter_equilibrium_batches(test_split, batch_size=batch_size, rng=rng)
    test_metrics_epoch: list[dict[str, float]] = []
    for batch in test_batches:
        device_batch = {key: jnp.asarray(value) for key, value in batch.items()}
        metrics = eval_step(best_params, device_batch)
        test_metrics_epoch.append({key: float(value) for key, value in metrics.items()})

    final_metrics = {
        "best_val_combined_loss": float(best_val),
        "test": _mean_metrics(test_metrics_epoch),
        "num_train_runs": train_split.num_runs,
        "num_val_runs": val_split.num_runs,
        "num_test_runs": test_split.num_runs,
        "parameter_count": int(count_parameters(best_params)),
    }
    (checkpoints_root / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8",
    )
    (checkpoints_root / "metrics.json").write_text(
        json.dumps(final_metrics, indent=2) + "\n", encoding="utf-8",
    )
    export_checkpoint_payload(best_payload, export_root)

    return TrainingArtifacts(
        checkpoint_path=checkpoints_root / "best.pt",
        export_root=export_root,
        history_path=checkpoints_root / "history.json",
        metrics_path=checkpoints_root / "metrics.json",
    )


def train_model(
    config: dict[str, Any],
    *,
    project_root: Path,
) -> TrainingArtifacts:
    if is_equilibrium(config):
        return train_equilibrium_model(config, project_root=project_root)
    processed_root = _ensure_processed(config, project_root=project_root)
    splits, normalization, contract = load_processed_dataset(processed_root)

    train_split = splits["train"]
    val_split = splits["val"]
    test_split = splits["test"]

    dt_stats = {
        "mean": float(normalization["log10_dt_s"]["mean"][0]),
        "std": float(normalization["log10_dt_s"]["std"][0]),
    }
    transition_sampling = effective_transition_sampling(config)
    candidate_common = {
        "dt_min_s": float(transition_sampling["dt_min_s"]),
        "dt_max_s": float(transition_sampling["dt_max_s"]),
        "min_future_saved_steps": int(transition_sampling["min_future_saved_steps"]),
        "log10_dt_stats": dt_stats,
    }
    train_candidates = build_candidate_table(train_split.time_s, train_split.valid_steps_mask, **candidate_common)
    val_candidates = build_candidate_table(val_split.time_s, val_split.valid_steps_mask, **candidate_common)
    test_candidates = build_candidate_table(test_split.time_s, test_split.valid_steps_mask, **candidate_common)

    dims, params = initialize_model(config, contract, seed=int(config["training"]["seed"]))
    LOGGER.info("Initialized model with %d parameters.", count_parameters(params))
    opt_state = _init_adamw_state(params)
    train_step, eval_step = make_train_eval_functions(
        dims=dims,
        normalization=normalization,
        loss_cfg=config["training"]["loss"],
        gradient_clip=float(config["training"]["gradient_clip"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    checkpoints_root = resolve_path(config["paths"]["checkpoints_root"], project_root)
    export_root = resolve_path(config["paths"]["jax_export_root"], project_root)
    ensure_dir(checkpoints_root)
    ensure_dir(export_root)

    eval_pairs = int(config["training"]["live_sampling"]["eval_pairs_per_run"])
    val_rows = sample_eval_rows(
        val_candidates,
        pairs_per_run=eval_pairs,
        num_logdt_bins=int(transition_sampling["num_logdt_bins"]),
        seed=int(config["training"]["seed"]) + 1,
    )
    test_rows = sample_eval_rows(
        test_candidates,
        pairs_per_run=eval_pairs,
        num_logdt_bins=int(transition_sampling["num_logdt_bins"]),
        seed=int(config["training"]["seed"]) + 2,
    )

    history: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_val = float("inf")
    global_step = 0

    for epoch in range(int(config["training"]["epochs"])):
        train_rows = sample_train_rows(
            train_candidates,
            pairs_per_run=int(config["training"]["live_sampling"]["train_pairs_per_run_per_epoch"]),
            num_logdt_bins=int(transition_sampling["num_logdt_bins"]),
            seed=int(config["training"]["seed"]),
            epoch=epoch,
        )
        train_batches = iter_batches(
            train_split,
            train_candidates,
            train_rows,
            batch_size=int(config["training"]["batch_size"]),
        )
        steps_per_epoch = max(len(train_batches), 1)
        warmup_steps = int(config["training"]["warmup_epochs"]) * steps_per_epoch
        total_steps = int(config["training"]["epochs"]) * steps_per_epoch
        train_metrics_epoch: list[dict[str, float]] = []

        for batch in train_batches:
            lr = _learning_rate_schedule(
                step=global_step,
                total_steps=total_steps,
                base_lr=float(config["training"]["learning_rate"]),
                min_lr=float(config["training"]["min_lr"]),
                warmup_steps=warmup_steps,
            )
            device_batch = {
                key: jnp.asarray(value)
                for key, value in batch.items()
                if key != "row_indices"
            }
            params, opt_state, metrics = train_step(
                params,
                opt_state,
                device_batch,
                jnp.asarray(lr, dtype=jnp.float32),
            )
            train_metrics_epoch.append({key: float(value) for key, value in metrics.items()})
            global_step += 1

        val_metrics_epoch: list[dict[str, float]] = []
        for batch in iter_batches(val_split, val_candidates, val_rows, batch_size=int(config["training"]["batch_size"])):
            device_batch = {
                key: jnp.asarray(value)
                for key, value in batch.items()
                if key != "row_indices"
            }
            metrics = eval_step(params, device_batch)
            val_metrics_epoch.append({key: float(value) for key, value in metrics.items()})

        train_summary = _mean_metrics(train_metrics_epoch)
        val_summary = _mean_metrics(val_metrics_epoch)
        record = {
            "epoch": epoch + 1,
            "train": train_summary,
            "val": val_summary,
        }
        history.append(record)
        LOGGER.info(
            "Epoch %d/%d train=%.6e val=%.6e",
            epoch + 1,
            int(config["training"]["epochs"]),
            train_summary["combined_loss"],
            val_summary["combined_loss"],
        )

        current_payload = _checkpoint_payload(
            params=params,
            dims=dims,
            config=config,
            normalization=normalization,
            data_contract=contract,
            metrics={
                "epoch": epoch + 1,
                "train": train_summary,
                "val": val_summary,
            },
            history=history,
        )
        _write_checkpoint(checkpoints_root / "last.pt", current_payload)

        if val_summary["combined_loss"] < best_val:
            best_val = val_summary["combined_loss"]
            best_payload = current_payload
            _write_checkpoint(checkpoints_root / "best.pt", current_payload)

    if best_payload is None:
        raise RuntimeError("Training completed without producing a best checkpoint.")

    test_metrics_epoch: list[dict[str, float]] = []
    best_params = jax.tree_util.tree_map(jnp.asarray, best_payload["params"])
    for batch in iter_batches(test_split, test_candidates, test_rows, batch_size=int(config["training"]["batch_size"])):
        device_batch = {
            key: jnp.asarray(value)
            for key, value in batch.items()
            if key != "row_indices"
        }
        metrics = eval_step(best_params, device_batch)
        test_metrics_epoch.append({key: float(value) for key, value in metrics.items()})
    final_metrics = {
        "best_val_combined_loss": float(best_val),
        "test": _mean_metrics(test_metrics_epoch),
        "num_train_candidates": int(train_candidates.num_rows()),
        "num_val_candidates": int(val_candidates.num_rows()),
        "num_test_candidates": int(test_candidates.num_rows()),
        "parameter_count": int(count_parameters(best_params)),
    }
    (checkpoints_root / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (checkpoints_root / "metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n", encoding="utf-8")
    export_checkpoint_payload(best_payload, export_root)

    return TrainingArtifacts(
        checkpoint_path=checkpoints_root / "best.pt",
        export_root=export_root,
        history_path=checkpoints_root / "history.json",
        metrics_path=checkpoints_root / "metrics.json",
    )
