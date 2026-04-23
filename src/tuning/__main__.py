"""Optuna hyperparameter sweep for the chemistry emulator Transformer.

Runs ``n_trials`` independent training runs, each for ``epochs_per_trial``
epochs, and minimizes validation ``combined_loss``. The processed dataset is
loaded once at study startup and shared with every trial via
``train_model(..., preloaded=...)``. Per-epoch pruning uses
``optuna.pruners.MedianPruner`` via a callback plumbed through
``train_model(..., on_epoch_end=...)``.

Search space is restricted to architectural and regularization choices
(sizing, activation, norm, FFN type, QK-norm, FiLM init, dropout,
weight decay). Learning-rate, optimizer, batch size, and loss weights are
kept fixed at the base-config values.

Invocation::

    python -m src.tuning --config config/fastchem_no_condensation.json \\
        --trials 100 --epochs 100
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import jax

import optuna

from ..constants import (
    TUNING_EARLY_STOP_MAX_PATIENCE,
    TUNING_EARLY_STOP_MIN_PATIENCE,
    TUNING_EARLY_STOP_PATIENCE_DIVISOR,
    TUNING_EMA_DEFAULT_DECAY,
    TUNING_TPE_SEED,
    TUNING_WEIGHT_DECAY_RANGE,
)
from ..data_generation.data_loader import load_processed_dataset
from ..training.trainer import _ensure_processed, train_model
from ..utils.config import _validate_transformer_model_config, load_and_validate_config
from ..utils.helpers import ensure_dir, get_logger, resolve_path, resolve_project_root

LOGGER = get_logger(__name__)

STUDY_NAME = "vulcan_emulator_arch_v2"

_D_MODEL_CANDIDATES = (192, 256, 384)
_NHEAD_CANDIDATES = (4, 6, 8, 12, 16)

# Enumerate valid ``(d_model, nhead)`` pairs as a single categorical. Using one
# parameter name across all trials (instead of ``f"nhead_for_d{d_model}"``)
# keeps the Optuna dashboard and cross-trial analysis clean, and enforces the
# ``d_model % nhead == 0`` constraint by construction.
_ARCH_CHOICES: tuple[tuple[int, int], ...] = tuple(
    (d, h)
    for d in _D_MODEL_CANDIDATES
    for h in _NHEAD_CANDIDATES
    if d % h == 0
)
_ARCH_LABELS: tuple[str, ...] = tuple(f"d{d}_h{h}" for d, h in _ARCH_CHOICES)
_ARCH_BY_LABEL: dict[str, tuple[int, int]] = dict(zip(_ARCH_LABELS, _ARCH_CHOICES))


def _arch_label(d_model: int, nhead: int) -> str:
    """Return the ``d{d_model}_h{nhead}`` categorical label for a valid pair."""
    label = f"d{d_model}_h{nhead}"
    if label not in _ARCH_BY_LABEL:
        raise ValueError(
            f"(d_model={d_model}, nhead={nhead}) is not in _ARCH_CHOICES."
        )
    return label


# Formulation knobs held fixed in v2 (sweep v1 agreed these are neutral-or-good
# across the top completed trials, so we spend budget on sizing + regularization
# instead). activation=silu, ffn_type=swiglu, use_qk_norm=true,
# zero_init_film=true. Overridden *in the base config*, not per-trial.
_FIXED_FORMULATION = {
    "activation": "silu",
    "ffn_type": "swiglu",
    "use_qk_norm": True,
    "zero_init_film": True,
}

# Incumbent architecture enqueued as trial #0 so every sweep has a known
# benchmark. Under _FIXED_FORMULATION the sampled config is the 2026-04-17
# "hybrid" arch (2026-04-16 sizing + 2026-04-17 sweep formulation wins); if
# the hybrid run ships, its numbers become the bar any new trial has to clear.
INCUMBENT_PARAMS = {
    "d_model_nhead": _arch_label(256, 8),
    "num_layers": 6,
    "dim_ff_mult": 4.0,
    "conditioning_hidden_dim": 256,
    "output_head_divisor": 2,
    "norm_type": "layernorm",
    "dropout_rate": 0.0,
    "weight_decay": 1e-4,
    "ema_enabled": True,
    "loss_type": "mae",
}


def _sample_trial_overrides(trial: optuna.Trial) -> dict[str, Any]:
    """Sample architectural + regularization hyperparameters for one trial.

    v2 search space: sizing and regularization only. Formulation knobs that
    sweep v1 showed were consistently good (silu / swiglu / qk_norm /
    zero_init_film) are fixed in the base config; see ``_FIXED_FORMULATION``.

    Constraints (``d_model % nhead == 0``, ``dim_feedforward >= d_model``)
    are enforced by construction here; the final config is also re-run
    through :func:`_validate_transformer_model_config` before training.
    """
    arch_label = trial.suggest_categorical("d_model_nhead", list(_ARCH_LABELS))
    d_model, nhead = _ARCH_BY_LABEL[arch_label]
    num_layers = trial.suggest_int("num_layers", 4, 8)
    dim_ff_mult = trial.suggest_float("dim_ff_mult", 2.5, 4.0)
    conditioning_hidden_dim = trial.suggest_categorical(
        "conditioning_hidden_dim", [128, 256, 512, 1024]
    )
    output_head_divisor = trial.suggest_categorical("output_head_divisor", [1, 2])
    norm_type = trial.suggest_categorical("norm_type", ["layernorm", "rmsnorm"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.1)
    weight_decay = trial.suggest_float(
        "weight_decay",
        TUNING_WEIGHT_DECAY_RANGE[0],
        TUNING_WEIGHT_DECAY_RANGE[1],
        log=True,
    )
    ema_enabled = trial.suggest_categorical("ema_enabled", [True, False])

    loss_type = trial.suggest_categorical("loss_type", ["mae", "huber"])
    if loss_type == "huber":
        huber_delta_log10 = trial.suggest_float(
            "huber_delta_log10", 0.02, 0.3, log=True
        )
    else:
        huber_delta_log10 = None

    dim_feedforward = int(round(d_model * dim_ff_mult))
    if dim_feedforward < d_model:
        dim_feedforward = d_model

    return {
        "d_model": int(d_model),
        "nhead": int(nhead),
        "num_layers": int(num_layers),
        "dim_feedforward": int(dim_feedforward),
        "conditioning_hidden_dim": int(conditioning_hidden_dim),
        "output_head_divisor": int(output_head_divisor),
        "norm_type": norm_type,
        "dropout_rate": float(dropout_rate),
        "weight_decay": float(weight_decay),
        "ema_enabled": bool(ema_enabled),
        "loss_type": loss_type,
        "huber_delta_log10": (
            float(huber_delta_log10) if huber_delta_log10 is not None else None
        ),
    }


def _apply_overrides(
    base_config: dict[str, Any],
    overrides: dict[str, Any],
    *,
    epochs: int,
    checkpoints_root: Path,
) -> dict[str, Any]:
    """Return a deep copy of ``base_config`` mutated with the trial overrides."""
    cfg = copy.deepcopy(base_config)

    model = cfg["model"]
    for key in (
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "conditioning_hidden_dim",
        "output_head_divisor",
        "norm_type",
        "dropout_rate",
    ):
        model[key] = overrides[key]
    for key, value in _FIXED_FORMULATION.items():
        model[key] = value

    cfg["model"] = _validate_transformer_model_config(model, "model")
    cfg["training"]["model"] = dict(cfg["model"])

    training = cfg["training"]
    training["weight_decay"] = overrides["weight_decay"]
    training["epochs"] = int(epochs)
    training["early_stopping_patience"] = max(
        TUNING_EARLY_STOP_MIN_PATIENCE,
        min(
            TUNING_EARLY_STOP_MAX_PATIENCE,
            int(epochs) // TUNING_EARLY_STOP_PATIENCE_DIVISOR,
        ),
    )
    # The config validator guarantees ``training["ema"]`` exists with the
    # canonical ``{"enabled": bool, "decay": float}`` shape. A trial that
    # enables EMA on top of a base config that had it disabled inherits
    # ``decay == 0.0`` (the validator's unset default), which is invalid once
    # ``enabled`` flips to ``True`` — seed a sensible decay in that case.
    ema_cfg = training["ema"]
    ema_cfg["enabled"] = bool(overrides["ema_enabled"])
    if ema_cfg["enabled"] and not (0.0 < ema_cfg["decay"] < 1.0):
        ema_cfg["decay"] = TUNING_EMA_DEFAULT_DECAY

    loss_cfg = training["loss"]
    loss_cfg["type"] = overrides["loss_type"]
    if overrides["loss_type"] == "huber":
        loss_cfg["lambda_log10_huber"] = float(
            loss_cfg.get("lambda_log10_huber", loss_cfg.get("lambda_log10_mae", 0.25))
        )
        loss_cfg["huber_delta_log10"] = float(overrides["huber_delta_log10"])
        loss_cfg.pop("lambda_log10_mae", None)
    else:  # mae
        loss_cfg["lambda_log10_mae"] = float(
            loss_cfg.get("lambda_log10_mae", loss_cfg.get("lambda_log10_huber", 0.25))
        )
        loss_cfg.pop("lambda_log10_huber", None)
        loss_cfg.pop("huber_delta_log10", None)

    cfg["paths"]["checkpoints_root"] = str(checkpoints_root)
    return cfg


def _study_root(base_config: dict[str, Any], project_root: Path) -> Path:
    """Resolve the ``<checkpoints_root>/optuna`` directory for the active config."""
    base_ckpt_root = resolve_path(base_config["paths"]["checkpoints_root"], project_root)
    return base_ckpt_root / "optuna"


_CSV_FIELDS = (
    "epoch",
    "combined_loss",
    "mse_norm",
    "log10_loss",
)


def _make_callback(trial: optuna.Trial, csv_path: Path):
    """Build an ``on_epoch_end`` callback that reports, logs to CSV, and prunes."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    handle = csv_path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
    writer.writeheader()
    handle.flush()

    def _callback(epoch: int, val_summary: dict[str, float]) -> None:
        value = float(val_summary["combined_loss"])
        row = {"epoch": epoch}
        for key in _CSV_FIELDS[1:]:
            row[key] = f"{float(val_summary[key]):.8g}" if key in val_summary else ""
        writer.writerow(row)
        handle.flush()
        trial.report(value, step=epoch)
        if trial.should_prune():
            handle.close()
            raise optuna.TrialPruned()

    _callback.close = handle.close  # type: ignore[attr-defined]
    return _callback


def _make_objective(
    base_config: dict[str, Any],
    *,
    project_root: Path,
    preloaded: tuple[dict, dict, dict],
    study_root: Path,
    epochs: int,
):
    """Build the Optuna objective closure."""

    def _objective(trial: optuna.Trial) -> float:
        overrides = _sample_trial_overrides(trial)
        trial_ckpt = study_root / f"trial_{trial.number:04d}"
        ensure_dir(trial_ckpt)
        trial_config = _apply_overrides(
            base_config,
            overrides,
            epochs=epochs,
            checkpoints_root=trial_ckpt,
        )
        trial_config["_project_root"] = project_root

        LOGGER.info(
            "Trial %04d overrides: %s",
            trial.number,
            json.dumps(
                {k: overrides[k] for k in overrides if k != "weight_decay"}
                | {"weight_decay": f"{overrides['weight_decay']:.2e}"}
            ),
        )

        csv_path = trial_ckpt / "val_metrics.csv"
        callback = _make_callback(trial, csv_path)
        try:
            artifacts = train_model(
                trial_config,
                project_root=project_root,
                preloaded=preloaded,
                on_epoch_end=callback,
            )
        finally:
            callback.close()
        metrics = json.loads(Path(artifacts.metrics_path).read_text())
        best_val = float(metrics["best_val_combined_loss"])
        LOGGER.info(
            "Trial %04d best_val_combined_loss=%.6f | csv=%s",
            trial.number,
            best_val,
            csv_path,
        )
        return best_val

    return _objective


def _copy_best_checkpoint(study: optuna.Study, study_root: Path) -> Path | None:
    """Copy the best trial's best checkpoint into ``<study_root>/best_overall/``.

    Checkpoints are now Orbax directories; copy the whole tree.
    """
    try:
        best_number = int(study.best_trial.number)
    except (ValueError, RuntimeError):
        return None
    src = study_root / f"trial_{best_number:04d}" / "best"
    if not src.exists():
        LOGGER.warning("Best trial %d has no best checkpoint at %s", best_number, src)
        return None
    dst_dir = study_root / "best_overall"
    ensure_dir(dst_dir)
    dst = dst_dir / "best"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter sweep for the chemistry emulator.",
    )
    parser.add_argument(
        "--config",
        default="config/fastchem_no_condensation.json",
        help="Path to the base configuration JSON file.",
    )
    parser.add_argument("--trials", type=int, default=60, help="Number of Optuna trials.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Maximum training epochs per trial (HyperbandPruner max_resource; "
        "overrides config.training.epochs).",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=30,
        help="Minimum epochs a trial runs before it can be pruned "
        "(HyperbandPruner min_resource).",
    )
    parser.add_argument(
        "--reduction-factor",
        type=int,
        default=3,
        help="HyperbandPruner reduction factor (bracket aggressiveness).",
    )
    parser.add_argument(
        "--skip-incumbent",
        action="store_true",
        help="Skip enqueueing INCUMBENT_PARAMS as trial #0 (by default the "
        "incumbent arch is always included as a baseline).",
    )
    parser.add_argument(
        "--study-name",
        default=STUDY_NAME,
        help="Optuna study name (used as the RDB study key).",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL. Defaults to SQLite under <checkpoints_root>/optuna/.",
    )
    args = parser.parse_args(argv)

    project_root = resolve_project_root(Path(__file__).resolve())
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    base_config = load_and_validate_config(config_path)
    base_config["_project_root"] = project_root

    study_root = _study_root(base_config, project_root)
    ensure_dir(study_root)

    processed_root = _ensure_processed(base_config, project_root=project_root)
    LOGGER.info("Loading processed dataset once from %s", processed_root)
    splits, normalization, contract = load_processed_dataset(processed_root)

    storage = args.storage or f"sqlite:///{study_root / 'study.db'}"
    LOGGER.info("Optuna storage: %s | study: %s", storage, args.study_name)

    sampler = optuna.samplers.TPESampler(
        seed=TUNING_TPE_SEED, multivariate=True, group=True
    )
    # HyperbandPruner funnels compute to survivors so winners are evaluated at
    # real training length. MedianPruner (sweep v1) killed ~88% of trials before
    # epoch 30 and biased selection toward fast-early-learners.
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=int(args.min_epochs),
        max_resource=int(args.epochs),
        reduction_factor=int(args.reduction_factor),
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # Always benchmark against the known-good pre-sweep architecture: enqueue it
    # as the first trial. Skipped if already present in a resumed study, or
    # when --skip-incumbent is set.
    if not args.skip_incumbent:
        existing_params = {
            frozenset(t.params.items())
            for t in study.get_trials(deepcopy=False, states=None)
        }
        if frozenset(INCUMBENT_PARAMS.items()) not in existing_params:
            study.enqueue_trial(INCUMBENT_PARAMS)
            LOGGER.info("Enqueued incumbent arch as baseline trial: %s", INCUMBENT_PARAMS)
        else:
            LOGGER.info("Incumbent arch already present in study; not re-enqueueing.")

    objective = _make_objective(
        base_config,
        project_root=project_root,
        preloaded=(splits, normalization, contract),
        study_root=study_root,
        epochs=args.epochs,
    )

    def _between_trials(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        LOGGER.info(
            "Trial %04d state=%s value=%s",
            trial.number,
            trial.state.name,
            f"{trial.value:.6f}" if trial.value is not None else "n/a",
        )
        gc.collect()
        try:
            jax.clear_caches()
        except AttributeError:
            pass

    study.optimize(
        objective,
        n_trials=args.trials,
        catch=(RuntimeError, ValueError),
        callbacks=[_between_trials],
        gc_after_trial=True,
    )

    best_params_path = study_root / "best_params.json"
    best_params_path.write_text(
        json.dumps(
            {
                "best_value": float(study.best_value),
                "best_trial_number": int(study.best_trial.number),
                "best_params": study.best_params,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Best value %.6f written to %s", study.best_value, best_params_path)

    copied = _copy_best_checkpoint(study, study_root)
    if copied is not None:
        LOGGER.info("Copied best checkpoint to %s", copied)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
