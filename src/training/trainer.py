"""Training loops for all chemistry/model combinations.

This module provides the complete training pipeline for the emulator matrix:

- **FastChem / MLP**
- **FastChem / Transformer**
- **VULCAN / MLP**
- **VULCAN / Transformer**

Both paths share the same optimizer (AdamW with decoupled weight decay),
gradient clipping (global L2 norm), and checkpointing logic (save best by
validation combined loss). Learning-rate scheduling keeps linear warmup for
all runs, defaults to reduce-on-plateau after warmup, and retains the
pre-existing cosine-annealing path when explicitly requested in the config.

The combined loss is ``lambda_z * MSE_norm + lambda_phys * MSE_log10
[+ lambda_spectrum * spectrum_recon_MSE]``.  The normalized-space MSE is the
primary gradient signal; the log10-space MSE is a physical-scale diagnostic
that improves low-abundance species accuracy.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..data_generation.data_loader import (
    iter_batches,
    load_processed_dataset,
    processed_info_dir,
)
from ..data_generation.generation import generate_raw_dataset
from ..data_generation.preprocess import PROCESSED_DATA_VERSION, preprocess_raw_dataset
from ..models.jax_model import (
    apply_mlp,
    apply_transformer_model,
    count_parameters,
    initialize_model,
)
from ..utils.config import get_chemistry_type, get_model_type, uses_mlp
from ..utils.helpers import ensure_dir, get_logger, resolve_path

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class TrainingArtifacts:
    """Key output paths produced by a training run."""

    checkpoint_path: Path
    history_path: Path
    metrics_path: Path


@dataclass(frozen=True)
class ReduceOnPlateauState:
    """Track post-warmup reduce-on-plateau learning-rate state."""

    current_lr: float
    best_metric: float | None
    bad_epochs: int


def _tree_global_norm(tree: Any) -> jax.Array:
    """Compute the global L2 norm of a nested JAX parameter or gradient tree.

    Parameters
    ----------
    tree : Any
        Nested JAX pytree whose leaves are numeric arrays.

    Returns
    -------
    jax.Array
        Scalar ``float32`` array containing the global L2 norm across all
        leaves.
    """
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def _clip_tree(tree: Any, max_norm: float) -> Any:
    """Clip all leaves in a tree so the global L2 norm does not exceed *max_norm*.

    Parameters
    ----------
    tree : Any
        Nested JAX pytree whose leaves are numeric arrays.
    max_norm : float
        Maximum allowed global L2 norm.

    Returns
    -------
    Any
        Tree with the same structure as ``tree`` after global-norm clipping.
    """
    norm = _tree_global_norm(tree)
    scale = jnp.minimum(1.0, float(max_norm) / jnp.maximum(norm, 1.0e-12))
    return jax.tree_util.tree_map(lambda x: x * scale, tree)


def _init_adamw_state(params: Any) -> dict[str, Any]:
    """Initialize AdamW optimizer state: zero first-moment (m), second-moment (v), and step counter.

    Parameters
    ----------
    params : Any
        Parameter pytree whose structure defines the optimizer moment trees.

    Returns
    -------
    dict[str, Any]
        Optimizer state dictionary containing zero-valued ``m`` and ``v``
        trees plus the scalar step counter ``t``.
    """
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
    """Apply one AdamW parameter update step (Loshchilov & Hutter, 2019).

    AdamW decouples weight decay from the adaptive gradient step:
        p_new = p - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)

    Parameters
    ----------
    params : Any
        Current parameter tree.
    grads : Any
        Gradient tree (same structure as *params*).
    state : dict
        Optimizer state with keys ``"m"``, ``"v"``, ``"t"``.
    learning_rate : jax.Array
        Current learning rate (scalar).
    weight_decay : float
        L2 weight decay coefficient.

    Returns
    -------
    tuple[Any, dict]
        ``(new_params, new_state)``.
    """
    t = state["t"] + 1
    # Update biased first-moment (mean) and second-moment (variance) estimates.
    m = jax.tree_util.tree_map(lambda m_, g_: beta1 * m_ + (1.0 - beta1) * g_, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v_, g_: beta2 * v_ + (1.0 - beta2) * (g_ * g_), state["v"], grads)
    # Bias correction for the exponential moving averages.
    bias1 = 1.0 - jnp.power(jnp.asarray(beta1, dtype=jnp.float32), t.astype(jnp.float32))
    bias2 = 1.0 - jnp.power(jnp.asarray(beta2, dtype=jnp.float32), t.astype(jnp.float32))
    m_hat = jax.tree_util.tree_map(lambda x: x / bias1, m)
    v_hat = jax.tree_util.tree_map(lambda x: x / bias2, v)
    # AdamW: decoupled weight decay applied directly to params (Loshchilov & Hutter, 2019).
    new_params = jax.tree_util.tree_map(
        lambda p, m_h, v_h: p - learning_rate * (m_h / (jnp.sqrt(v_h) + eps) + weight_decay * p),
        params,
        m_hat,
        v_hat,
    )
    return new_params, {"m": m, "v": v, "t": t}


def _warmup_learning_rate(
    *,
    step: int,
    base_lr: float,
    warmup_steps: int,
) -> float | None:
    """Return the linear-warmup learning rate for one optimizer step.

    Parameters
    ----------
    step : int
        Zero-based global optimizer step.
    base_lr : float
        Peak learning rate reached at the end of warmup.
    warmup_steps : int
        Number of warmup steps. A value of zero disables warmup.

    Returns
    -------
    float or None
        Warmup learning rate for the step, or ``None`` once warmup is over.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(max(warmup_steps, 1))
    return None


def _cosine_learning_rate_schedule(
    *,
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    """Compute the warmup-plus-cosine-annealing learning rate for one step.

    During warmup (step < warmup_steps), the learning rate increases linearly
    from 0 to base_lr.  After warmup, it decays following a cosine curve
    from base_lr to min_lr over the remaining steps.

    Parameters
    ----------
    step : int
        Current global training step.
    total_steps : int
        Total number of training steps across all epochs.
    base_lr : float
        Peak learning rate (reached at end of warmup).
    min_lr : float
        Minimum learning rate (reached at end of training).
    warmup_steps : int
        Number of linear warmup steps.

    Returns
    -------
    float
        Learning rate for the current step.
    """
    warmup_lr = _warmup_learning_rate(
        step=step,
        base_lr=base_lr,
        warmup_steps=warmup_steps,
    )
    if warmup_lr is not None:
        return warmup_lr
    if total_steps <= warmup_steps:
        return base_lr
    # Cosine annealing from base_lr down to min_lr over the remaining steps.
    progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _init_reduce_on_plateau_state(base_lr: float) -> ReduceOnPlateauState:
    """Create the initial scheduler state for reduce-on-plateau learning-rate decay.

    Parameters
    ----------
    base_lr : float
        Starting learning rate before any plateau-triggered reductions.

    Returns
    -------
    ReduceOnPlateauState
        Scheduler state with the current learning rate set to ``base_lr`` and
        no recorded best validation metric yet.
    """
    return ReduceOnPlateauState(
        current_lr=base_lr,
        best_metric=None,
        bad_epochs=0,
    )


def _update_reduce_on_plateau(
    state: ReduceOnPlateauState,
    metric: float,
    *,
    factor: float,
    patience: int,
    threshold: float,
    min_lr: float,
) -> ReduceOnPlateauState:
    """Apply one epoch-end reduce-on-plateau scheduler update.

    Parameters
    ----------
    state : ReduceOnPlateauState
        Current scheduler state.
    metric : float
        Validation metric to monitor, where lower is better.
    factor : float
        Multiplicative learning-rate decay factor.
    patience : int
        Number of non-improving epochs to tolerate before decaying.
    threshold : float
        Minimum absolute improvement required to reset the bad-epoch counter.
    min_lr : float
        Lower bound on the scheduled learning rate.

    Returns
    -------
    ReduceOnPlateauState
        Updated scheduler state after considering the current validation
        metric.
    """
    if state.best_metric is None or metric < (state.best_metric - threshold):
        return ReduceOnPlateauState(
            current_lr=state.current_lr,
            best_metric=metric,
            bad_epochs=0,
        )

    bad_epochs = state.bad_epochs + 1
    if bad_epochs <= patience or state.current_lr <= min_lr:
        return ReduceOnPlateauState(
            current_lr=state.current_lr,
            best_metric=state.best_metric,
            bad_epochs=bad_epochs,
        )

    return ReduceOnPlateauState(
        current_lr=max(min_lr, state.current_lr * factor),
        best_metric=state.best_metric,
        bad_epochs=0,
    )


def _scheduled_learning_rate(
    *,
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    scheduler: dict[str, Any],
    plateau_state: ReduceOnPlateauState | None,
) -> float:
    """Compute the active learning rate for the current training step.

    Parameters
    ----------
    step : int
        Current global optimizer step.
    total_steps : int
        Total planned optimizer steps across the full run.
    base_lr : float
        Peak learning rate.
    min_lr : float
        Minimum learning rate used by cosine / plateau decay.
    warmup_steps : int
        Number of linear warmup steps.
    scheduler : dict[str, Any]
        Validated scheduler config.
    plateau_state : ReduceOnPlateauState or None
        Current reduce-on-plateau state when that scheduler is active.

    Returns
    -------
    float
        Learning rate to apply at this step.
    """
    warmup_lr = _warmup_learning_rate(
        step=step,
        base_lr=base_lr,
        warmup_steps=warmup_steps,
    )
    if warmup_lr is not None:
        return warmup_lr

    if scheduler["name"] == "cosine":
        return _cosine_learning_rate_schedule(
            step=step,
            total_steps=total_steps,
            base_lr=base_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
        )
    if scheduler["name"] == "reduce_on_plateau":
        return base_lr if plateau_state is None else plateau_state.current_lr
    raise ValueError(f"Unsupported scheduler: {scheduler['name']!r}")


def _maybe_update_plateau_scheduler(
    *,
    scheduler: dict[str, Any],
    plateau_state: ReduceOnPlateauState | None,
    metric: float,
    global_step: int,
    warmup_steps: int,
    min_lr: float,
) -> ReduceOnPlateauState | None:
    """Update reduce-on-plateau state after validation when applicable.

    Parameters
    ----------
    scheduler : dict[str, Any]
        Validated scheduler config.
    plateau_state : ReduceOnPlateauState or None
        Current plateau scheduler state.
    metric : float
        Validation metric used for plateau detection.
    global_step : int
        Global optimizer step reached at the end of the epoch.
    warmup_steps : int
        Number of warmup steps that must finish before plateau updates start.
    min_lr : float
        Lower learning-rate bound.

    Returns
    -------
    ReduceOnPlateauState or None
        Updated plateau state, or the original state when the scheduler does
        not require an update.
    """
    if scheduler["name"] != "reduce_on_plateau" or plateau_state is None:
        return plateau_state
    if global_step < warmup_steps:
        return plateau_state
    return _update_reduce_on_plateau(
        plateau_state,
        metric,
        factor=float(scheduler["factor"]),
        patience=int(scheduler["patience"]),
        threshold=float(scheduler["threshold"]),
        min_lr=min_lr,
    )


def _state_stats(normalization: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
    """Load target normalization statistics as JAX arrays for loss computation.

    Parameters
    ----------
    normalization : dict[str, Any]
        Normalization payload containing a ``"target"`` block with ``"mean"``
        and ``"std"`` arrays.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Target means and standard deviations converted to ``float32`` JAX
        arrays.
    """
    mean = jnp.asarray(normalization["target"]["mean"], dtype=jnp.float32)
    std = jnp.asarray(normalization["target"]["std"], dtype=jnp.float32)
    return mean, std


def make_transformer_train_eval_functions(
    *,
    dims,
    normalization: dict[str, Any],
    loss_cfg: dict[str, float],
    gradient_clip: float,
    weight_decay: float,
):
    """Build JIT-compiled train/eval functions for the Transformer model.

    Parameters
    ----------
    dims : TransformerDimensions
        Transformer architecture dimensions.
    normalization : dict[str, Any]
        Normalization payload used to recover target-space log10 statistics.
    loss_cfg : dict[str, float]
        Loss weights for normalized-space, physical-space, and optional
        spectrum-reconstruction losses.
    gradient_clip : float
        Global L2 gradient-clip threshold.
    weight_decay : float
        AdamW decoupled weight-decay coefficient.

    Returns
    -------
    tuple[callable, callable]
        ``(train_step, eval_step)`` closures that consume normalized batch
        dictionaries.
    """
    target_mean, target_std = _state_stats(normalization)

    @jax.jit
    def train_step(params, opt_state, batch, learning_rate, dropout_key):
        """Run one Transformer optimizer step on a normalized batch.

        Parameters
        ----------
        params : Any
            Current Transformer parameter tree.
        opt_state : dict[str, Any]
            AdamW optimizer state.
        batch : dict[str, jax.Array]
            Normalized batch containing ``sequence``, ``global_inputs``,
            ``target``, and optional ``spectrum_inputs``.
        learning_rate : jax.Array
            Scalar learning rate for this step.
        dropout_key : jax.Array
            PRNG key used for hidden-layer dropout.

        Returns
        -------
        tuple[Any, dict[str, Any], dict[str, jax.Array]]
            Updated parameter tree, updated optimizer state, and scalar
            training metrics.
        """
        def loss_fn(model_params):
            """Compute weighted Transformer losses and scalar metrics.

            Parameters
            ----------
            model_params : Any
                Transformer parameter tree.

            Returns
            -------
            tuple[jax.Array, dict[str, jax.Array]]
                Combined loss plus a metrics mapping containing normalized,
                log10-physical, and optional spectrum-reconstruction terms.
            """
            pred, aux = apply_transformer_model(
                model_params,
                batch["sequence"],
                batch["global_inputs"],
                batch.get("spectrum_inputs"),
                dims,
                dropout_key=dropout_key,
                training=True,
            )
            # MSE in normalized space (the primary training signal).
            mse_norm = jnp.mean((pred - batch["target"]) ** 2)
            # MSE in log10 mixing-ratio space (physical-scale diagnostic).
            pred_log10 = pred * target_std + target_mean
            target_log10 = batch["target"] * target_std + target_mean
            mse_log10 = jnp.mean((pred_log10 - target_log10) ** 2)
            # Spectrum autoencoder reconstruction loss (zero if encoder is not autoencoder).
            spectrum_loss = jnp.asarray(0.0, dtype=jnp.float32)
            reconstruction = aux["spectrum_reconstruction"]
            spectrum_inputs = batch.get("spectrum_inputs")
            if reconstruction is not None and spectrum_inputs is not None:
                spectrum_loss = jnp.mean((reconstruction - spectrum_inputs) ** 2)
            # Weighted combination of the three loss components.
            total = (
                float(loss_cfg["lambda_z"]) * mse_norm
                + float(loss_cfg["lambda_phys"]) * mse_log10
                + float(loss_cfg.get("lambda_spectrum", 0.0)) * spectrum_loss
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
        """Evaluate the Transformer model on one normalized batch.

        Parameters
        ----------
        params : Any
            Transformer parameter tree.
        batch : dict[str, jax.Array]
            Normalized validation or test batch.

        Returns
        -------
        dict[str, jax.Array]
            Scalar evaluation metrics for the batch.
        """
        pred, aux = apply_transformer_model(
            params,
            batch["sequence"],
            batch["global_inputs"],
            batch.get("spectrum_inputs"),
            dims,
        )
        mse_norm = jnp.mean((pred - batch["target"]) ** 2)
        pred_log10 = pred * target_std + target_mean
        target_log10 = batch["target"] * target_std + target_mean
        mse_log10 = jnp.mean((pred_log10 - target_log10) ** 2)
        spectrum_loss = jnp.asarray(0.0, dtype=jnp.float32)
        reconstruction = aux["spectrum_reconstruction"]
        spectrum_inputs = batch.get("spectrum_inputs")
        if reconstruction is not None and spectrum_inputs is not None:
            spectrum_loss = jnp.mean((reconstruction - spectrum_inputs) ** 2)
        total = (
            float(loss_cfg["lambda_z"]) * mse_norm
            + float(loss_cfg["lambda_phys"]) * mse_log10
            + float(loss_cfg.get("lambda_spectrum", 0.0)) * spectrum_loss
        )
        return {
            "combined_loss": total,
            "mse_norm": mse_norm,
            "mse_log10": mse_log10,
            "spectrum_recon_mse": spectrum_loss,
        }

    return train_step, eval_step


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of scalar metric dictionaries by key.

    Parameters
    ----------
    metrics : list[dict[str, float]]
        Per-batch metric dictionaries with identical scalar keys.

    Returns
    -------
    dict[str, float]
        Mean metric values across the supplied list, or ``NaN`` defaults when
        the list is empty.
    """
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


def _feature_order_matches(contract: dict[str, Any], config: dict[str, Any]) -> bool:
    """Return whether processed feature orders still match the active config.

    Parameters
    ----------
    contract : dict[str, Any]
        Stored processed-data contract.
    config : dict[str, Any]
        Active validated config.

    Returns
    -------
    bool
        ``True`` when both sequence and global feature orders still match.
    """
    expected_sequence = list(config["data_spec"]["sequence_static_feature_order"])
    expected_globals = list(config["data_spec"]["global_static_feature_order"])
    return (
        list(contract.get("sequence_static_feature_order", [])) == expected_sequence
        and list(contract.get("global_static_feature_order", [])) == expected_globals
    )


def _epoch_table_header() -> str:
    """Build the fixed-width header row for the live epoch progress table.

    Returns
    -------
    str
        Header text labeling the epoch, train loss, validation loss, learning
        rate, and elapsed time columns.
    """
    return f"{'Epoch':>9}  {'Train Loss':>12}  {'Val Loss':>12}  {'LR':>12}  {'Time':>8}"


def _epoch_table_rule() -> str:
    """Build a separator row matching the live epoch progress table width.

    Returns
    -------
    str
        Dashed separator string aligned to the same column widths as
        :func:`_epoch_table_header`.
    """
    return f"{'-' * 9}  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 8}"


def _format_epoch_row(
    *,
    epoch: int,
    epochs: int,
    train_loss: float,
    val_loss: float,
    learning_rate: float,
    epoch_seconds: float,
) -> str:
    """Format one plain-text epoch summary row.

    Parameters
    ----------
    epoch : int
        One-based epoch index.
    epochs : int
        Total number of configured epochs.
    train_loss : float
        Mean training combined loss for the epoch.
    val_loss : float
        Mean validation combined loss for the epoch.
    learning_rate : float
        Learning rate used at the end of the epoch.
    epoch_seconds : float
        Wall-clock duration of the epoch.

    Returns
    -------
    str
        Fixed-width table row for live logging.
    """
    epoch_label = f"{epoch}/{epochs}"
    return (
        f"{epoch_label:>9}  "
        f"{train_loss:>12.4e}  "
        f"{val_loss:>12.4e}  "
        f"{learning_rate:>12.4e}  "
        f"{epoch_seconds:>7.3f}s"
    )


def _should_early_stop(no_improvement_epochs: int, *, patience: int) -> bool:
    """Decide whether early stopping should terminate training.

    Parameters
    ----------
    no_improvement_epochs : int
        Number of consecutive epochs without validation improvement.
    patience : int
        Allowed number of non-improving epochs before stopping.

    Returns
    -------
    bool
        ``True`` when the non-improvement count has reached or exceeded the
        configured patience.
    """
    return no_improvement_epochs >= patience


def _emit_epoch_table_line(line: str) -> None:
    """Write one epoch-table line to stdout and the optional live log file.

    Parameters
    ----------
    line : str
        Fully formatted line to emit.
    """
    print(line, flush=True)
    live_log_path = os.environ.get("VULCAN_LIVE_LOG_PATH", "").strip()
    if not live_log_path:
        return
    with Path(live_log_path).expanduser().resolve().open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")
        handle.flush()


def _ensure_processed(config: dict[str, Any], *, project_root: Path) -> Path:
    """Ensure the processed dataset exists and matches the active config.

    Parameters
    ----------
    config : dict[str, Any]
        Validated config describing the required chemistry, model, and feature
        order contract.
    project_root : Path
        Repository root used to resolve raw and processed paths.

    Returns
    -------
    Path
        Path to a processed dataset compatible with the active config.
    """
    processed_root = resolve_path(config["paths"]["processed_root"], project_root)
    contract_path = processed_info_dir(processed_root) / "data_contract.json"
    if contract_path.exists():
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            version_ok = int(contract.get("processed_data_version", -1)) == PROCESSED_DATA_VERSION
            chemistry_ok = str(contract.get("chemistry_type", "")).lower() == get_chemistry_type(config)
            model_ok = str(contract.get("model_type", "")).lower() == get_model_type(config)
            feature_order_ok = _feature_order_matches(contract, config)
            required_files_ok = all(
                (processed_root / split / "target_outputs.npy").exists()
                for split in ("train", "val", "test")
            )
            if version_ok and chemistry_ok and model_ok and feature_order_ok and required_files_ok:
                LOGGER.info("Using existing processed dataset at %s", processed_root)
                return processed_root
        except (OSError, ValueError, TypeError):
            pass

    raw_root = resolve_path(config["paths"]["raw_root"], project_root)
    raw_runs_dir = raw_root / "runs"
    consolidated_exists = (raw_root / "runs.h5").exists()
    raw_run_files = sorted(raw_runs_dir.glob("run_*.h5")) if raw_runs_dir.exists() else []
    if not raw_run_files and not consolidated_exists:
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
    """Package model state and training metadata into a checkpoint payload.

    Parameters
    ----------
    params : Any
        Current model parameter tree.
    dims : Any
        Model-dimension dataclass.
    config : dict[str, Any]
        Validated config to embed in the checkpoint.
    normalization : dict[str, Any]
        Normalization payload for downstream export and inference.
    data_contract : dict[str, Any]
        Processed-data contract describing tensor ordering.
    metrics : dict[str, Any]
        Current summary metrics to store alongside the checkpoint.
    history : list[dict[str, Any]]
        Full epoch history accumulated so far.

    Returns
    -------
    dict[str, Any]
        Pickle-serializable checkpoint dictionary.
    """
    config_payload = {
        key: value
        for key, value in config.items()
        if key != "_roth_profile_cache"
    }
    return {
        "params": jax.tree_util.tree_map(np.asarray, params),
        "model_dimensions": dims.to_dict(),
        "config": config_payload,
        "normalization": normalization,
        "data_contract": data_contract,
        "metrics": metrics,
        "history": history,
    }


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Serialize one training checkpoint payload to disk with pickle.

    Parameters
    ----------
    path : Path
        Output checkpoint path such as ``best.pt`` or ``last.pt``.
    payload : dict[str, Any]
        Checkpoint dictionary containing params, optimizer state, metrics,
        normalization, and data-contract metadata.

    Returns
    -------
    None
        The checkpoint payload is written to ``path`` using the highest pickle
        protocol.
    """
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_mlp_train_eval_functions(
    *,
    dims,
    normalization: dict[str, Any],
    loss_cfg: dict[str, float],
    gradient_clip: float,
    weight_decay: float,
):
    """Build JIT-compiled train/eval functions for the FiLM-MLP.

    Parameters
    ----------
    dims : MLPDimensions
        MLP architecture dimensions.
    normalization : dict[str, Any]
        Normalization payload used to recover target-space log10 statistics.
    loss_cfg : dict[str, float]
        Loss weights for normalized-space, physical-space, and optional
        spectrum-reconstruction losses.
    gradient_clip : float
        Global L2 gradient-clip threshold.
    weight_decay : float
        AdamW decoupled weight-decay coefficient.

    Returns
    -------
    tuple[callable, callable]
        ``(train_step, eval_step)`` closures that consume normalized batch
        dictionaries.
    """
    target_mean, target_std = _state_stats(normalization)

    @jax.jit
    def train_step(params, opt_state, batch, learning_rate, dropout_key):
        """Run one FiLM-MLP optimizer step on a normalized batch.

        Parameters
        ----------
        params : Any
            Current MLP parameter tree.
        opt_state : dict[str, Any]
            AdamW optimizer state.
        batch : dict[str, jax.Array]
            Normalized batch containing ``sequence``, ``global_inputs``,
            ``target``, and optional ``spectrum_inputs``.
        learning_rate : jax.Array
            Scalar learning rate for this step.
        dropout_key : jax.Array
            PRNG key used for hidden-layer dropout.

        Returns
        -------
        tuple[Any, dict[str, Any], dict[str, jax.Array]]
            Updated parameter tree, updated optimizer state, and scalar
            training metrics.
        """
        def loss_fn(model_params):
            """Compute weighted FiLM-MLP losses and scalar metrics.

            Parameters
            ----------
            model_params : Any
                MLP parameter tree.

            Returns
            -------
            tuple[jax.Array, dict[str, jax.Array]]
                Combined loss plus a metrics mapping containing normalized,
                log10-physical, and optional spectrum-reconstruction terms.
            """
            pred, aux = apply_mlp(
                model_params,
                batch["sequence"],
                batch["global_inputs"],
                dims,
                batch.get("spectrum_inputs"),
                dropout_key=dropout_key,
                training=True,
            )
            mse_norm = jnp.mean((pred - batch["target"]) ** 2)
            pred_log10 = pred * target_std + target_mean
            target_log10 = batch["target"] * target_std + target_mean
            mse_log10 = jnp.mean((pred_log10 - target_log10) ** 2)
            spectrum_loss = jnp.asarray(0.0, dtype=jnp.float32)
            reconstruction = aux["spectrum_reconstruction"]
            spectrum_inputs = batch.get("spectrum_inputs")
            if reconstruction is not None and spectrum_inputs is not None:
                spectrum_loss = jnp.mean((reconstruction - spectrum_inputs) ** 2)
            total = (
                float(loss_cfg["lambda_z"]) * mse_norm
                + float(loss_cfg["lambda_phys"]) * mse_log10
                + float(loss_cfg.get("lambda_spectrum", 0.0)) * spectrum_loss
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
            params, grads, opt_state,
            learning_rate=learning_rate, weight_decay=weight_decay,
        )
        return new_params, new_opt_state, metrics

    @jax.jit
    def eval_step(params, batch):
        """Evaluate the FiLM-MLP on one normalized batch.

        Parameters
        ----------
        params : Any
            MLP parameter tree.
        batch : dict[str, jax.Array]
            Normalized validation or test batch.

        Returns
        -------
        dict[str, jax.Array]
            Scalar evaluation metrics for the batch.
        """
        pred, aux = apply_mlp(
            params,
            batch["sequence"],
            batch["global_inputs"],
            dims,
            batch.get("spectrum_inputs"),
        )
        mse_norm = jnp.mean((pred - batch["target"]) ** 2)
        pred_log10 = pred * target_std + target_mean
        target_log10 = batch["target"] * target_std + target_mean
        mse_log10 = jnp.mean((pred_log10 - target_log10) ** 2)
        spectrum_loss = jnp.asarray(0.0, dtype=jnp.float32)
        reconstruction = aux["spectrum_reconstruction"]
        spectrum_inputs = batch.get("spectrum_inputs")
        if reconstruction is not None and spectrum_inputs is not None:
            spectrum_loss = jnp.mean((reconstruction - spectrum_inputs) ** 2)
        total = (
            float(loss_cfg["lambda_z"]) * mse_norm
            + float(loss_cfg["lambda_phys"]) * mse_log10
            + float(loss_cfg.get("lambda_spectrum", 0.0)) * spectrum_loss
        )
        return {
            "combined_loss": total,
            "mse_norm": mse_norm,
            "mse_log10": mse_log10,
            "spectrum_recon_mse": spectrum_loss,
        }

    return train_step, eval_step


def train_model(
    config: dict[str, Any],
    *,
    project_root: Path,
) -> TrainingArtifacts:
    """Train the active chemistry/model combination end to end.

    Parameters
    ----------
    config : dict[str, Any]
        Validated config describing data paths, model hyperparameters, loss
        weights, and training schedule.
    project_root : Path
        Repository root used to resolve data and checkpoint paths.

    Returns
    -------
    TrainingArtifacts
        Paths to the best checkpoint, epoch history, and final metrics files.
    """
    processed_root = _ensure_processed(config, project_root=project_root)
    splits, normalization, contract = load_processed_dataset(processed_root)

    train_split = splits["train"]
    val_split = splits["val"]
    test_split = splits.get("test", val_split)

    dims, params = initialize_model(config, contract, seed=int(config["training"]["seed"]))
    LOGGER.info(
        "Initialized %s/%s model with %d parameters.",
        get_chemistry_type(config),
        get_model_type(config),
        count_parameters(params),
    )
    opt_state = _init_adamw_state(params)
    if uses_mlp(config):
        train_step, eval_step = make_mlp_train_eval_functions(
            dims=dims,
            normalization=normalization,
            loss_cfg=config["training"]["loss"],
            gradient_clip=float(config["training"]["gradient_clip"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
    else:
        train_step, eval_step = make_transformer_train_eval_functions(
            dims=dims,
            normalization=normalization,
            loss_cfg=config["training"]["loss"],
            gradient_clip=float(config["training"]["gradient_clip"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )

    checkpoints_root = resolve_path(config["paths"]["checkpoints_root"], project_root)
    ensure_dir(checkpoints_root)

    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    history: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_val = float("inf")
    best_epoch = 0
    no_improvement_epochs = 0
    global_step = 0
    scheduler = config["training"]["scheduler"]
    early_stopping_patience = int(config["training"]["early_stopping_patience"])
    base_lr = float(config["training"]["learning_rate"])
    min_lr = float(config["training"]["min_lr"])
    plateau_state = (
        _init_reduce_on_plateau_state(base_lr)
        if scheduler["name"] == "reduce_on_plateau"
        else None
    )

    steps_per_epoch = max(1, (train_split.num_runs + batch_size - 1) // batch_size)
    warmup_steps = int(config["training"]["warmup_epochs"]) * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    LOGGER.info(
        "Training %s/%s model from %s | runs train/val/test=%d/%d/%d | steps/epoch=%d",
        get_chemistry_type(config),
        get_model_type(config),
        processed_root,
        train_split.num_runs,
        val_split.num_runs,
        test_split.num_runs,
        steps_per_epoch,
    )
    _emit_epoch_table_line(_epoch_table_header())
    _emit_epoch_table_line(_epoch_table_rule())

    rng = np.random.default_rng(int(config["training"]["seed"]))
    dropout_rng = jax.random.PRNGKey(int(config["training"]["seed"]))

    for epoch in range(epochs):
        epoch_t0 = time.monotonic()
        train_batches = iter_batches(train_split, batch_size=batch_size, rng=rng)
        train_metrics_epoch: list[dict[str, float]] = []
        train_steps = 0

        for batch in train_batches:
            lr = _scheduled_learning_rate(
                step=global_step,
                total_steps=total_steps,
                base_lr=base_lr,
                min_lr=min_lr,
                warmup_steps=warmup_steps,
                scheduler=scheduler,
                plateau_state=plateau_state,
            )
            device_batch = {key: jnp.asarray(value) for key, value in batch.items()}
            dropout_rng, step_dropout_key = jax.random.split(dropout_rng)
            params, opt_state, metrics = train_step(
                params,
                opt_state,
                device_batch,
                jnp.asarray(lr, dtype=jnp.float32),
                step_dropout_key,
            )
            train_metrics_epoch.append({key: float(value) for key, value in metrics.items()})
            global_step += 1
            train_steps += 1

        val_batches = iter_batches(val_split, batch_size=batch_size, rng=rng)
        val_metrics_epoch: list[dict[str, float]] = []
        val_steps = 0
        for batch in val_batches:
            device_batch = {key: jnp.asarray(value) for key, value in batch.items()}
            metrics = eval_step(params, device_batch)
            val_metrics_epoch.append({key: float(value) for key, value in metrics.items()})
            val_steps += 1

        epoch_dt = time.monotonic() - epoch_t0
        train_summary = _mean_metrics(train_metrics_epoch)
        val_summary = _mean_metrics(val_metrics_epoch)
        plateau_state = _maybe_update_plateau_scheduler(
            scheduler=scheduler,
            plateau_state=plateau_state,
            metric=val_summary["combined_loss"],
            global_step=global_step,
            warmup_steps=warmup_steps,
            min_lr=min_lr,
        )
        current_lr = float(
            plateau_state.current_lr
            if plateau_state is not None and scheduler["name"] == "reduce_on_plateau"
            else lr
        )
        record = {
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "epoch_seconds": float(epoch_dt),
            "train_steps": train_steps,
            "val_steps": val_steps,
            "train": train_summary,
            "val": val_summary,
        }
        history.append(record)
        _emit_epoch_table_line(
            _format_epoch_row(
                epoch=epoch + 1,
                epochs=epochs,
                train_loss=train_summary["combined_loss"],
                val_loss=val_summary["combined_loss"],
                learning_rate=current_lr,
                epoch_seconds=epoch_dt,
            )
        )

        current_payload = _checkpoint_payload(
            params=params,
            dims=dims,
            config=config,
            normalization=normalization,
            data_contract=contract,
            metrics={"epoch": epoch + 1, "train": train_summary, "val": val_summary},
            history=history,
        )
        _write_checkpoint(checkpoints_root / "last.pt", current_payload)
        if val_summary["combined_loss"] < best_val:
            best_val = val_summary["combined_loss"]
            best_epoch = epoch + 1
            no_improvement_epochs = 0
            best_payload = current_payload
            _write_checkpoint(checkpoints_root / "best.pt", current_payload)
        else:
            no_improvement_epochs += 1
            if _should_early_stop(no_improvement_epochs, patience=early_stopping_patience):
                LOGGER.info(
                    "Early stopping at epoch %d after %d epochs without validation improvement. Best epoch: %d.",
                    epoch + 1,
                    no_improvement_epochs,
                    best_epoch,
                )
                break

    if best_payload is None:
        raise RuntimeError("Training completed without producing a best checkpoint.")

    best_params = jax.tree_util.tree_map(jnp.asarray, best_payload["params"])
    test_batches = iter_batches(test_split, batch_size=batch_size, rng=rng)
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
    return TrainingArtifacts(
        checkpoint_path=checkpoints_root / "best.pt",
        history_path=checkpoints_root / "history.json",
        metrics_path=checkpoints_root / "metrics.json",
    )
