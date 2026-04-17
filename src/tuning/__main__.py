"""Optuna hyperparameter sweep for the VULCAN emulator Transformer.

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

    python -m src.tuning --config config/vulcan_no_condensation.json \\
        --trials 100 --epochs 100
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import jax

import optuna

from ..data_generation.data_loader import load_processed_dataset
from ..training.trainer import _ensure_processed, train_model
from ..utils.config import _validate_transformer_model_config, load_and_validate_config
from ..utils.helpers import ensure_dir, get_logger, resolve_path, resolve_project_root

LOGGER = get_logger(__name__)

STUDY_NAME = "vulcan_emulator_arch_v1"

_NHEAD_CANDIDATES = (4, 6, 8, 12, 16)


def _valid_nheads(d_model: int) -> list[int]:
    """Return the nhead options that evenly divide ``d_model``."""
    return [h for h in _NHEAD_CANDIDATES if d_model % h == 0]


def _sample_trial_overrides(trial: optuna.Trial) -> dict[str, Any]:
    """Sample architectural + regularization hyperparameters for one trial.

    Returns a flat dict ready to be merged into a copy of the base config.
    All constraints (``d_model % nhead == 0``, ``dim_feedforward >= d_model``)
    are enforced by construction here; the final config is also re-run
    through :func:`_validate_transformer_model_config` before training.
    """
    d_model = trial.suggest_categorical("d_model", [128, 192, 256, 384, 512])
    nhead = trial.suggest_categorical(f"nhead_for_d{d_model}", _valid_nheads(d_model))
    num_layers = trial.suggest_int("num_layers", 3, 8)
    dim_ff_mult = trial.suggest_float("dim_ff_mult", 2.0, 4.0)
    conditioning_hidden_dim = trial.suggest_categorical(
        "conditioning_hidden_dim", [128, 256, 512, 1024]
    )
    activation = trial.suggest_categorical("activation", ["silu", "gelu", "relu", "elu"])
    norm_type = trial.suggest_categorical("norm_type", ["layernorm", "rmsnorm"])
    use_qk_norm = trial.suggest_categorical("use_qk_norm", [True, False])
    ffn_type = trial.suggest_categorical("ffn_type", ["dense", "swiglu"])
    zero_init_film = trial.suggest_categorical("zero_init_film", [True, False])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.20)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    dim_feedforward = int(round(d_model * dim_ff_mult))
    if dim_feedforward < d_model:
        dim_feedforward = d_model

    return {
        "d_model": int(d_model),
        "nhead": int(nhead),
        "num_layers": int(num_layers),
        "dim_feedforward": int(dim_feedforward),
        "conditioning_hidden_dim": int(conditioning_hidden_dim),
        "activation": activation,
        "norm_type": norm_type,
        "use_qk_norm": bool(use_qk_norm),
        "ffn_type": ffn_type,
        "zero_init_film": bool(zero_init_film),
        "dropout_rate": float(dropout_rate),
        "weight_decay": float(weight_decay),
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
        "activation",
        "norm_type",
        "use_qk_norm",
        "ffn_type",
        "zero_init_film",
        "dropout_rate",
    ):
        model[key] = overrides[key]

    cfg["model"] = _validate_transformer_model_config(model, "model")
    cfg["training"]["model"] = dict(cfg["model"])

    training = cfg["training"]
    training["weight_decay"] = overrides["weight_decay"]
    training["epochs"] = int(epochs)
    training["early_stopping_patience"] = max(5, min(20, int(epochs) // 5))

    cfg["paths"]["checkpoints_root"] = str(checkpoints_root)
    return cfg


def _study_root(base_config: dict[str, Any], project_root: Path) -> Path:
    """Resolve the ``<checkpoints_root>/optuna`` directory for the active config."""
    base_ckpt_root = resolve_path(base_config["paths"]["checkpoints_root"], project_root)
    return base_ckpt_root / "optuna"


def _make_callback(trial: optuna.Trial):
    """Build an ``on_epoch_end`` callback that reports val loss and prunes."""

    def _callback(epoch: int, val_summary: dict[str, float]) -> None:
        value = float(val_summary["combined_loss"])
        trial.report(value, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

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

        artifacts = train_model(
            trial_config,
            project_root=project_root,
            preloaded=preloaded,
            on_epoch_end=_make_callback(trial),
        )
        metrics = json.loads(Path(artifacts.metrics_path).read_text())
        best_val = float(metrics["best_val_combined_loss"])
        LOGGER.info("Trial %04d best_val_combined_loss=%.6f", trial.number, best_val)
        return best_val

    return _objective


def _copy_best_checkpoint(study: optuna.Study, study_root: Path) -> Path | None:
    """Copy the best trial's best.pt into ``<study_root>/best_overall/``."""
    try:
        best_number = int(study.best_trial.number)
    except (ValueError, RuntimeError):
        return None
    src = study_root / f"trial_{best_number:04d}" / "best.pt"
    if not src.exists():
        LOGGER.warning("Best trial %d has no best.pt at %s", best_number, src)
        return None
    dst_dir = study_root / "best_overall"
    ensure_dir(dst_dir)
    dst = dst_dir / "best.pt"
    shutil.copy2(src, dst)
    return dst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter sweep for the VULCAN emulator.",
    )
    parser.add_argument(
        "--config",
        default="config/vulcan_no_condensation.json",
        help="Path to the base configuration JSON file.",
    )
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs per trial (overrides config.training.epochs).",
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

    sampler = optuna.samplers.TPESampler(seed=123, multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=1,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

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
