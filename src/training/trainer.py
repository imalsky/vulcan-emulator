"""Training loop for the FiLM-conditioned Transformer emulator.

Supports FastChem and VULCAN (condensation) chemistry types with the
Transformer architecture.

Uses AdamW with decoupled weight decay, global L2 gradient clipping, and
checkpointing (save best by validation combined loss). Learning-rate
scheduling keeps linear warmup for all runs, defaults to reduce-on-plateau
after warmup, and retains the pre-existing cosine-annealing path when
explicitly requested in the config.

When ``training.ema.enabled`` is true, an exponential-moving-average
shadow of the parameters is maintained and used for all validation, test,
and exported weights; the raw optimizer state is kept only for training
continuation.

The physical-space loss term is selected by ``training.loss.type``:

* ``"mae"``  — combined loss is
  ``lambda_z * MSE_norm + lambda_log10_mae * MAE_log10``.
  ``MAE_log10`` is mean absolute error of the log-ratio residual
  ``r = log10(pred_VMR / target_VMR)``. Constant gradient at all scales;
  optimizes median log-fractional error.
* ``"huber"`` — combined loss is
  ``lambda_z * MSE_norm + lambda_log10_huber * Huber_log10``.
  ``Huber_log10`` is the Huber loss on the same residual, with transition
  point ``huber_delta_log10`` (dex). Quadratic in ``|r| <= delta``
  (smooth gradient on already-accurate predictions), linear beyond
  (outlier-robust).

In either case, the physical-space term is reported in metrics under the
unified key ``log10_loss``.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..data_generation.data_loader import (
    iter_batches,
    load_processed_dataset,
    processed_info_dir,
)
from ..data_generation.generation import generate_raw_dataset
from ..data_generation.preprocess import preprocess_raw_dataset
from ..models.export_bundle import _flatten_params, _unflatten_params
from ..models.jax_model import (
    TransformerDimensions,
    apply_transformer_model,
    count_parameters,
    initialize_model,
)
from ..utils.config import get_chemistry_type, get_model_type
from ..utils.helpers import LIVE_LOG_ENV, ensure_dir, get_logger, resolve_path

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class TrainingArtifacts:
    """Key output paths produced by a training run."""

    run_root: Path
    config_path: Path
    history_path: Path
    metadata_path: Path
    params_best_path: Path
    params_last_path: Path


def _build_optimizer(
    *, gradient_clip: float, weight_decay: float
) -> optax.GradientTransformation:
    """Construct the AdamW optimizer with global-norm gradient clipping.

    The returned transformation produces updates in the *gradient* sign;
    ``train_step`` scales them by ``-learning_rate`` before
    ``optax.apply_updates``. Splitting the LR scaling out this way lets the
    epoch-level scheduler (warmup + reduce-on-plateau, or cosine) control the
    per-step LR without rebuilding the optimizer.
    """
    return optax.chain(
        optax.clip_by_global_norm(gradient_clip),
        optax.scale_by_adam(),
        optax.add_decayed_weights(weight_decay),
    )


@jax.jit
def _ema_update(ema: Any, params: Any, decay: jax.Array) -> Any:
    """Exponential-moving-average update of a shadow parameter tree.

    ``ema_new = decay * ema + (1 - decay) * params``. Applied every optimizer
    step; the shadow tree is the one exported and used for validation/test.
    """
    return jax.tree_util.tree_map(
        lambda e, p: decay * e + (1.0 - decay) * p, ema, params
    )


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
    from 0 to base_lr.  After warmup, decay follows
    :func:`optax.cosine_decay_schedule` from ``base_lr`` down to ``min_lr``
    over the remaining steps.
    """
    warmup_lr = _warmup_learning_rate(
        step=step,
        base_lr=base_lr,
        warmup_steps=warmup_steps,
    )
    if warmup_lr is not None:
        return warmup_lr
    decay_steps = max(total_steps - warmup_steps, 1)
    alpha = (min_lr / base_lr) if base_lr > 0.0 else 0.0
    schedule = optax.cosine_decay_schedule(
        init_value=base_lr, decay_steps=decay_steps, alpha=alpha,
    )
    return float(schedule(step - warmup_steps))


def _scheduled_learning_rate(
    *,
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    scheduler: dict[str, Any],
) -> float:
    """Compute the active learning rate for the current training step.

    Returns the linear-warmup value while ``step < warmup_steps``. After
    warmup, dispatches on ``scheduler["name"]``: ``"cosine"`` returns the
    warmup+cosine-annealed value; ``"reduce_on_plateau"`` returns the
    constant peak LR — the plateau-driven scale factor is maintained
    separately via :func:`optax.contrib.reduce_on_plateau` and multiplied
    into the update inside ``train_step``.
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
        return base_lr
    raise ValueError(f"Unsupported scheduler: {scheduler['name']!r}")


def _build_plateau_transform(
    scheduler: dict[str, Any],
    *,
    base_lr: float,
    min_lr: float,
) -> optax.GradientTransformationExtraArgs:
    """Build the :func:`optax.contrib.reduce_on_plateau` state machine.

    Patience is interpreted in *epochs* (matching the public config
    contract); the transform is called once per validation epoch, so
    optax's per-``update`` counter coincides with epochs. ``atol`` is the
    absolute improvement floor (``rtol`` is disabled), and ``min_scale``
    lower-bounds the scale at ``min_lr / base_lr``.

    The ``+ 1`` on ``patience`` preserves the legacy semantics where
    ``bad_epochs > patience`` triggered a reduction (first reduction
    after ``patience + 1`` non-improving epochs); optax triggers at
    ``plateau_count >= patience`` — one epoch earlier — so the offset
    restores bit-for-bit compatibility.
    """
    min_scale = (min_lr / base_lr) if base_lr > 0.0 else 0.0
    return optax.contrib.reduce_on_plateau(
        factor=float(scheduler["factor"]),
        patience=int(scheduler["patience"]) + 1,
        atol=float(scheduler["threshold"]),
        rtol=0.0,
        min_scale=float(min_scale),
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


def _huber_log10_loss(
    pred_log10: jax.Array,
    target_log10: jax.Array,
    mask: jax.Array,
    mask_sum: jax.Array,
    *,
    delta: float,
) -> jax.Array:
    """Mask-weighted Huber loss in log10 mixing-ratio space.

    The residual is ``r = pred_log10 - target_log10 = log10(pred_VMR /
    target_VMR)``. Uses the scaled form with continuous first derivative
    at the transition point:

        L(r) = 0.5 * r^2 / delta   if |r| <= delta
               |r| - 0.5 * delta   otherwise

    This is ``optax.huber_loss / delta``: the unscaled Huber returns
    ``0.5 r^2`` in the quadratic regime and ``delta * |r| - 0.5 delta^2``
    outside it, so dividing by ``delta`` recovers the scaled form whose
    tail matches ``|r|`` (comparable to log10-MAE within ``delta/2``).
    """
    elem = optax.huber_loss(pred_log10, target_log10, delta=delta) / delta
    return jnp.sum(elem * mask) / mask_sum


def _mae_log10_loss(
    pred_log10: jax.Array,
    target_log10: jax.Array,
    mask: jax.Array,
    mask_sum: jax.Array,
) -> jax.Array:
    """Mask-weighted MAE of the log-ratio residual ``log10(pred_VMR/target_VMR)``."""
    return jnp.sum(jnp.abs(pred_log10 - target_log10) * mask) / mask_sum


def make_transformer_train_eval_functions(
    *,
    dims: TransformerDimensions,
    normalization: dict[str, Any],
    loss_cfg: dict[str, float],
    gradient_clip: float,
    weight_decay: float,
) -> tuple[
    optax.GradientTransformation,
    Callable[[Any, Any, dict[str, jax.Array], jax.Array, jax.Array], tuple[Any, Any, dict[str, jax.Array]]],
    Callable[[Any, dict[str, jax.Array]], dict[str, jax.Array]],
]:
    """Build the optax optimizer plus JIT-compiled train/eval steps.

    Parameters
    ----------
    dims : TransformerDimensions
        Transformer architecture dimensions.
    normalization : dict[str, Any]
        Normalization payload used to recover target-space log10 statistics.
    loss_cfg : dict[str, float]
        Loss weights plus selector ``type`` (``"mae"`` or ``"huber"``).
        MAE requires ``lambda_z`` and ``lambda_log10_mae``; Huber requires
        ``lambda_z``, ``lambda_log10_huber``, and ``huber_delta_log10``
        (transition point in dex).
    gradient_clip : float
        Global L2 gradient-clip threshold.
    weight_decay : float
        AdamW decoupled weight-decay coefficient.

    Returns
    -------
    tuple[optax.GradientTransformation, callable, callable]
        ``(optimizer, train_step, eval_step)``. Caller is responsible for
        ``opt_state = optimizer.init(params)`` and threading ``opt_state``
        through successive ``train_step`` calls.
    """
    optimizer = _build_optimizer(
        gradient_clip=gradient_clip, weight_decay=weight_decay
    )
    target_mean, target_std = _state_stats(normalization)
    lambda_z = float(loss_cfg["lambda_z"])
    loss_type = loss_cfg["type"]
    if loss_type == "huber":
        huber_delta = float(loss_cfg["huber_delta_log10"])
        lambda_log10 = float(loss_cfg["lambda_log10_huber"])

        def _log10_loss(pred_log10, target_log10, mask, mask_sum):
            return _huber_log10_loss(
                pred_log10, target_log10, mask, mask_sum, delta=huber_delta
            )
    else:  # "mae" — validated upstream
        lambda_log10 = float(loss_cfg["lambda_log10_mae"])
        _log10_loss = _mae_log10_loss

    @jax.jit
    def train_step(
        params: Any,
        opt_state: Any,
        batch: dict[str, jax.Array],
        learning_rate: jax.Array,
        plateau_scale: jax.Array,
        dropout_key: jax.Array,
    ) -> tuple[Any, Any, dict[str, jax.Array]]:
        """Run one Transformer optimizer step on a normalized batch.

        ``plateau_scale`` is the multiplicative factor produced by the
        external :func:`optax.contrib.reduce_on_plateau` state machine; it
        is ``1.0`` when the active scheduler is cosine (no plateau decay
        applied) and drops toward the configured ``min_lr / base_lr`` when
        validation stagnates.
        """
        def loss_fn(model_params: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
            """Compute weighted Transformer losses and scalar metrics.

            Parameters
            ----------
            model_params : Any
                Transformer parameter tree.

            Returns
            -------
            tuple[jax.Array, dict[str, jax.Array]]
                Combined loss plus a metrics mapping containing normalized
                and log10-physical terms.
            """
            pred, _aux = apply_transformer_model(
                model_params,
                batch["sequence"],
                batch["global_inputs"],
                dims,
                position_coord=batch["position_coord"],
                attention_mask=batch["valid_mask"],
                dropout_key=dropout_key,
                training=True,
            )
            mask = batch["valid_mask"].astype(pred.dtype)[..., None]
            mask_sum = jnp.maximum(jnp.sum(mask) * pred.shape[-1], 1.0)
            mse_norm = jnp.sum(((pred - batch["target"]) ** 2) * mask) / mask_sum
            pred_log10 = pred * target_std + target_mean
            target_log10 = batch["target"] * target_std + target_mean
            log10_loss = _log10_loss(pred_log10, target_log10, mask, mask_sum)
            total = lambda_z * mse_norm + lambda_log10 * log10_loss
            metrics = {
                "combined_loss": total,
                "mse_norm": mse_norm,
                "log10_loss": log10_loss,
            }
            return total, metrics

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        # ``_build_optimizer`` produces updates in the gradient sign; scale by
        # ``-learning_rate * plateau_scale`` so per-step LR scheduling
        # (warmup + cosine) and the optax plateau state machine both stay
        # outside the optimizer's closure-captured hyperparameters.
        scale = -learning_rate * plateau_scale
        updates = jax.tree_util.tree_map(lambda u: scale * u, updates)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, metrics

    @jax.jit
    def eval_step(params: Any, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
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
        pred, _aux = apply_transformer_model(
            params,
            batch["sequence"],
            batch["global_inputs"],
            dims,
            position_coord=batch["position_coord"],
            attention_mask=batch["valid_mask"],
        )
        mask = batch["valid_mask"].astype(pred.dtype)[..., None]
        mask_sum = jnp.maximum(jnp.sum(mask) * pred.shape[-1], 1.0)
        mse_norm = jnp.sum(((pred - batch["target"]) ** 2) * mask) / mask_sum
        pred_log10 = pred * target_std + target_mean
        target_log10 = batch["target"] * target_std + target_mean
        log10_loss = _log10_loss(pred_log10, target_log10, mask, mask_sum)
        total = lambda_z * mse_norm + lambda_log10 * log10_loss
        return {
            "combined_loss": total,
            "mse_norm": mse_norm,
            "log10_loss": log10_loss,
        }

    return optimizer, train_step, eval_step


def _mean_metrics(
    metrics: list[dict[str, float]],
    weights: list[float] | None = None,
) -> dict[str, float]:
    """Average a list of scalar metric dictionaries by key.

    When ``weights`` is provided, entries are averaged by weighted mean. Each
    weight should equal the number of valid supervision targets that went into
    the corresponding per-batch metric (i.e. ``valid_mask.sum()``). Without
    weights, a plain arithmetic mean is returned — which biases the epoch
    summary when the last batch is smaller than the rest.

    Parameters
    ----------
    metrics : list[dict[str, float]]
        Per-batch metric dictionaries with identical scalar keys.
    weights : list[float] or None, optional
        Per-batch weights matching ``metrics``. ``None`` falls back to
        arithmetic mean.

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
            "log10_loss": float("nan"),
        }
    keys = metrics[0].keys()
    if weights is None:
        return {
            key: float(np.mean([metric[key] for metric in metrics]))
            for key in keys
        }
    if len(weights) != len(metrics):
        raise ValueError(
            f"weights length {len(weights)} does not match metrics length {len(metrics)}"
        )
    w = np.asarray(weights, dtype=np.float64)
    total_w = float(w.sum())
    if total_w <= 0.0:
        return {key: float("nan") for key in keys}
    return {
        key: float(np.sum(w * np.asarray([metric[key] for metric in metrics], dtype=np.float64)) / total_w)
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
    live_log_path = os.environ.get(LIVE_LOG_ENV, "").strip()
    if not live_log_path:
        return
    with Path(live_log_path).expanduser().resolve().open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")
        handle.flush()


def ensure_processed(config: dict[str, Any], *, project_root: Path) -> Path:
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
        contract: dict[str, Any] | None = None
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning(
                "Processed contract %s could not be read (%s: %s); regenerating processed dataset.",
                contract_path, type(exc).__name__, exc,
            )
        if contract is not None:
            try:
                chemistry_ok = str(contract.get("chemistry_type", "")).lower() == get_chemistry_type(config)
                model_ok = str(contract.get("model_type", "")).lower() == get_model_type(config)
                feature_order_ok = _feature_order_matches(contract, config)
                required_files_ok = all(
                    (processed_root / split / "target_outputs.npy").exists()
                    for split in ("train", "val", "test")
                )
            except (TypeError, ValueError) as exc:
                LOGGER.warning(
                    "Processed contract %s is malformed (%s: %s); regenerating processed dataset.",
                    contract_path, type(exc).__name__, exc,
                )
            else:
                if chemistry_ok and model_ok and feature_order_ok and required_files_ok:
                    LOGGER.info("Using existing processed dataset at %s", processed_root)
                    return processed_root

    raw_root = resolve_path(config["paths"]["raw_root"], project_root)
    if not (raw_root / "runs.h5").exists():
        generate_raw_dataset(config, project_root=project_root)

    preprocess_raw_dataset(config, project_root=project_root)
    return processed_root


def _strip_runtime_annotations(obj: Any) -> Any:
    """Remove ``_project_root`` entries from every nested dict.

    ``_project_root`` is a :class:`pathlib.Path` injected into configs at
    CLI/tuning entry points to thread the repo root through the pipeline;
    it is runtime-only state that must not leak into on-disk checkpoint
    metadata.
    """
    if isinstance(obj, dict):
        return {
            key: _strip_runtime_annotations(value)
            for key, value in obj.items()
            if key != "_project_root"
        }
    if isinstance(obj, list):
        return [_strip_runtime_annotations(value) for value in obj]
    return obj


CONFIG_FILENAME = "config.json"
METADATA_FILENAME = "metadata.json"
HISTORY_FILENAME = "history.csv"
PARAMS_BEST_FILENAME = "params_best.npz"
PARAMS_LAST_FILENAME = "params_last.npz"


def _write_params_npz(path: Path, params: Any) -> None:
    """Write a JAX parameter pytree to a flat NPZ archive under ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **_flatten_params(params))


def _write_config(run_root: Path, config: dict[str, Any]) -> None:
    """Write the input config (with runtime annotations stripped) to config.json."""
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / CONFIG_FILENAME).write_text(
        json.dumps(_strip_runtime_annotations(config), indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _write_metadata(run_root: Path, metadata: dict[str, Any]) -> None:
    """Write run-level metadata (model dims, normalization, contract, final metrics)."""
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _write_history_csv(run_root: Path, history: list[dict[str, Any]]) -> None:
    """Write per-epoch training history as a flat CSV (one row per epoch).

    Nested ``train`` / ``val`` metric dicts are flattened to ``train_<key>``
    and ``val_<key>`` columns so the file is directly plottable in pandas.
    """
    run_root.mkdir(parents=True, exist_ok=True)
    rows = [_flatten_history_record(record) for record in history]
    field_names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                field_names.append(key)
    with (run_root / HISTORY_FILENAME).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _flatten_history_record(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten one history entry's nested train/val metric dicts to a flat row."""
    flat: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}_{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat


def _read_checkpoint(
    run_root: Path,
    which: Literal["best", "last"] = "best",
) -> dict[str, Any]:
    """Load a training run's params plus config and metadata.

    ``run_root`` is the run directory containing ``config.json``,
    ``metadata.json``, and ``params_{best,last}.npz``. ``which`` selects
    which params file to load.
    """
    resolved = run_root.resolve()
    params_file = PARAMS_BEST_FILENAME if which == "best" else PARAMS_LAST_FILENAME
    with np.load(resolved / params_file) as archive:
        flat = {key: archive[key] for key in archive.files}
    params = _unflatten_params(flat)
    config = json.loads((resolved / CONFIG_FILENAME).read_text(encoding="utf-8"))
    metadata = json.loads((resolved / METADATA_FILENAME).read_text(encoding="utf-8"))
    return {"params": params, "config": config, **metadata}


def train_model(
    config: dict[str, Any],
    *,
    project_root: Path,
    preloaded: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None,
    on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
) -> TrainingArtifacts:
    """Train the active chemistry/model combination end to end.

    Parameters
    ----------
    config : dict[str, Any]
        Validated config describing data paths, model hyperparameters, loss
        weights, and training schedule.
    project_root : Path
        Repository root used to resolve data and checkpoint paths.
    preloaded : tuple or None, optional
        Optional ``(splits, normalization, contract)`` tuple matching the
        output of ``load_processed_dataset``. When provided, the processed
        dataset is not re-read from disk, which is useful for hyperparameter
        sweeps that share a dataset across many trials.
    on_epoch_end : callable or None, optional
        Optional callback ``(epoch_1_indexed, val_summary)`` invoked after
        each epoch's validation pass and checkpoint write. The callback may
        raise to abort training early (e.g. for Optuna pruning).

    Returns
    -------
    TrainingArtifacts
        Paths to the best checkpoint, epoch history, and final metrics files.
    """
    if preloaded is None:
        processed_root = ensure_processed(config, project_root=project_root)
        splits, normalization, contract = load_processed_dataset(processed_root)
    else:
        processed_root = Path("<preloaded>")
        splits, normalization, contract = preloaded

    train_split = splits["train"]
    val_split = splits["val"]
    test_split = splits["test"]

    dims, params = initialize_model(config, contract, seed=int(config["training"]["seed"]))
    LOGGER.info(
        "Initialized %s/%s model with %d parameters.",
        get_chemistry_type(config),
        get_model_type(config),
        count_parameters(params),
    )
    # ``config["training"]["ema"]`` is populated (with an ``enabled=False``
    # default if the user omitted the section) by the config validator, so
    # this branch can assume the key exists with the canonical shape.
    ema_cfg = config["training"]["ema"]
    ema_enabled = bool(ema_cfg["enabled"])
    ema_decay = float(ema_cfg["decay"])
    ema_params = jax.tree_util.tree_map(lambda x: x, params) if ema_enabled else None
    ema_decay_scalar = jnp.asarray(ema_decay, dtype=jnp.float32) if ema_enabled else None
    if ema_enabled:
        LOGGER.info("EMA enabled with decay=%.6f.", ema_decay)
    optimizer, train_step, eval_step = make_transformer_train_eval_functions(
        dims=dims,
        normalization=normalization,
        loss_cfg=config["training"]["loss"],
        gradient_clip=float(config["training"]["gradient_clip"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    opt_state = optimizer.init(params)

    checkpoints_root = resolve_path(config["paths"]["checkpoints_root"], project_root)
    ensure_dir(checkpoints_root)
    _write_config(checkpoints_root, config)

    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    history: list[dict[str, Any]] = []
    best_params_np: Any = None
    best_val = float("inf")
    best_epoch = 0
    no_improvement_epochs = 0
    global_step = 0
    scheduler = config["training"]["scheduler"]
    early_stopping_patience = int(config["training"]["early_stopping_patience"])
    base_lr = float(config["training"]["learning_rate"])
    min_lr = float(config["training"]["min_lr"])
    plateau_transform: optax.GradientTransformationExtraArgs | None = None
    plateau_state: Any = None
    if scheduler["name"] == "reduce_on_plateau":
        plateau_transform = _build_plateau_transform(
            scheduler, base_lr=base_lr, min_lr=min_lr,
        )
        plateau_state = plateau_transform.init(params)

    # Training drops the partial last batch (see ``iter_batches`` call below)
    # so ``steps_per_epoch`` is a floor division, not a ceil. This must match
    # the actual loop or warmup/cosine schedules drift.
    steps_per_epoch = max(1, train_split.num_runs // batch_size)
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
        train_batches = iter_batches(
            train_split, batch_size=batch_size, rng=rng, drop_last=True,
        )
        train_metrics_device: list[dict[str, jax.Array]] = []
        train_weights_epoch: list[float] = []
        train_steps = 0

        plateau_scale = (
            float(plateau_state.scale) if plateau_state is not None else 1.0
        )
        plateau_scale_jax = jnp.asarray(plateau_scale, dtype=jnp.float32)
        lr = base_lr
        for batch in train_batches:
            lr = _scheduled_learning_rate(
                step=global_step,
                total_steps=total_steps,
                base_lr=base_lr,
                min_lr=min_lr,
                warmup_steps=warmup_steps,
                scheduler=scheduler,
            )
            batch_weight = float(np.asarray(batch["valid_mask"]).sum())
            device_batch = {key: jnp.asarray(value) for key, value in batch.items()}
            dropout_rng, step_dropout_key = jax.random.split(dropout_rng)
            params, opt_state, metrics = train_step(
                params,
                opt_state,
                device_batch,
                jnp.asarray(lr, dtype=jnp.float32),
                plateau_scale_jax,
                step_dropout_key,
            )
            if ema_enabled:
                ema_params = _ema_update(ema_params, params, ema_decay_scalar)
            train_metrics_device.append(metrics)
            train_weights_epoch.append(batch_weight)
            global_step += 1
            train_steps += 1
        # One host sync per epoch instead of per step. Per-step ``float()``
        # serializes training and turns any deferred GPU error (e.g. a
        # CUDA_ERROR_STREAM_CAPTURE_INVALIDATED in the captured graph) into
        # a useless traceback inside the ``float()`` call.
        train_metrics_epoch: list[dict[str, float]] = [
            {k: float(v) for k, v in m.items()}
            for m in jax.device_get(train_metrics_device)
        ]

        eval_params = ema_params if ema_enabled else params
        val_batches = iter_batches(val_split, batch_size=batch_size, rng=rng)
        val_metrics_device: list[dict[str, jax.Array]] = []
        val_weights_epoch: list[float] = []
        val_steps = 0
        for batch in val_batches:
            batch_weight = float(np.asarray(batch["valid_mask"]).sum())
            device_batch = {key: jnp.asarray(value) for key, value in batch.items()}
            metrics = eval_step(eval_params, device_batch)
            val_metrics_device.append(metrics)
            val_weights_epoch.append(batch_weight)
            val_steps += 1
        val_metrics_epoch: list[dict[str, float]] = [
            {k: float(v) for k, v in m.items()}
            for m in jax.device_get(val_metrics_device)
        ]

        epoch_dt = time.monotonic() - epoch_t0
        train_summary = _mean_metrics(train_metrics_epoch, train_weights_epoch)
        val_summary = _mean_metrics(val_metrics_epoch, val_weights_epoch)
        if plateau_transform is not None and global_step >= warmup_steps:
            plateau_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
            _, plateau_state = plateau_transform.update(
                plateau_grads,
                plateau_state,
                params,
                value=jnp.asarray(val_summary["combined_loss"], dtype=jnp.float32),
            )
        current_lr = (
            lr * float(plateau_state.scale)
            if plateau_state is not None
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

        current_params_np = jax.tree_util.tree_map(np.asarray, eval_params)
        _write_params_npz(checkpoints_root / PARAMS_LAST_FILENAME, current_params_np)
        _write_history_csv(checkpoints_root, history)
        if val_summary["combined_loss"] < best_val:
            best_val = val_summary["combined_loss"]
            best_epoch = epoch + 1
            no_improvement_epochs = 0
            best_params_np = current_params_np
            _write_params_npz(checkpoints_root / PARAMS_BEST_FILENAME, current_params_np)
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

        if on_epoch_end is not None:
            on_epoch_end(epoch + 1, dict(val_summary))

    if best_params_np is None:
        raise RuntimeError("Training completed without producing a best checkpoint.")

    best_params = jax.tree_util.tree_map(jnp.asarray, best_params_np)
    test_batches = iter_batches(test_split, batch_size=batch_size, rng=rng)
    test_metrics_device: list[dict[str, jax.Array]] = []
    test_weights_epoch: list[float] = []
    for batch in test_batches:
        batch_weight = float(np.asarray(batch["valid_mask"]).sum())
        device_batch = {key: jnp.asarray(value) for key, value in batch.items()}
        metrics = eval_step(best_params, device_batch)
        test_metrics_device.append(metrics)
        test_weights_epoch.append(batch_weight)
    test_metrics_epoch: list[dict[str, float]] = [
        {k: float(v) for k, v in m.items()}
        for m in jax.device_get(test_metrics_device)
    ]

    final_metrics = {
        "best_val_combined_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "test": _mean_metrics(test_metrics_epoch, test_weights_epoch),
        "num_train_runs": train_split.num_runs,
        "num_val_runs": val_split.num_runs,
        "num_test_runs": test_split.num_runs,
        "parameter_count": int(count_parameters(best_params)),
    }
    _write_history_csv(checkpoints_root, history)
    _write_metadata(
        checkpoints_root,
        {
            "model_dimensions": dims.to_dict(),
            "normalization": normalization,
            "data_contract": contract,
            "final_metrics": final_metrics,
        },
    )
    return TrainingArtifacts(
        run_root=checkpoints_root,
        config_path=checkpoints_root / CONFIG_FILENAME,
        history_path=checkpoints_root / HISTORY_FILENAME,
        metadata_path=checkpoints_root / METADATA_FILENAME,
        params_best_path=checkpoints_root / PARAMS_BEST_FILENAME,
        params_last_path=checkpoints_root / PARAMS_LAST_FILENAME,
    )
