#!/usr/bin/env python3
"""Run a small self-contained Optuna study over training-only hyperparameters.

Searches over architecture presets, optimizer settings, and data-sampling
budgets while reusing existing processed data artifacts.  Each trial runs
a full ``run_training()`` call in a temporary scratch directory.  The
winning trial is promoted into ``models/hyperparam_testing/best_model/``
with rewritten paths and reformatted numeric outputs.

This script never calls ``--gen`` and never modifies processed data.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import optuna
import torch

from config_utils import load_and_validate_config, resolve_precision
from logging_utils import setup_logging
from path_utils import ensure_runtime_dirs, resolve_paths
from trainer import run_training

logger = logging.getLogger(__name__)

DEFAULT_TRIALS = 12
DEFAULT_TRIAL_EPOCHS = 20
HYPERPARAM_ROOT_NAME = "hyperparam_testing"
BEST_MODEL_OUTPUT_FOLDER = f"{HYPERPARAM_ROOT_NAME}/best_model"
TRIAL_LOG_HEADER = [
    "epoch",
    "train_combined",
    "val_combined",
    "train_mse",
    "val_mse",
    "train_mae_log10",
    "val_mae_log10",
    "lr",
    "epoch_seconds",
    "elapsed_seconds",
]
ARCHITECTURE_PRESETS: dict[str, tuple[int, int]] = {
    "256x4": (256, 4),
    "384x6": (384, 6),
    "512x8": (512, 8),
    "768x12": (768, 12),
}
PROMOTED_ARTIFACTS = (
    "best.pt",
    "training_log.csv",
    "metrics.json",
    "data_contract.json",
    "normalization_metadata.json",
    "processed_fingerprint.json",
)
FLOAT_SIG_FIGS = 4


def _format_floats(obj: Any) -> Any:
    """Recursively format all floats in a JSON-serializable object to scientific notation.

    Walks dicts, lists, and tuples, converting every ``float`` to a string
    representation with ``FLOAT_SIG_FIGS`` significant figures.  The formatted
    strings are wrapped so that ``json.dumps`` emits them unquoted by using a
    custom encoder's ``default`` hook on sentinel objects.
    """
    if isinstance(obj, float):
        return _FormattedFloat(obj)
    if isinstance(obj, dict):
        return {key: _format_floats(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_format_floats(item) for item in obj]
    return obj


class _FormattedFloat:
    """Sentinel that carries a pre-formatted float string for JSON output."""

    __slots__ = ("text",)

    def __init__(self, value: float) -> None:
        self.text = f"{value:.{FLOAT_SIG_FIGS - 1}e}"


class _ScientificNotationJSONEncoder(json.JSONEncoder):
    """JSON encoder that renders floats in scientific notation with fixed sig figs.

    Works by pre-formatting the payload via :func:`_format_floats` before
    encoding, avoiding reliance on CPython-private ``json.encoder`` internals.
    """

    def encode(self, o: Any) -> str:
        formatted = _format_floats(o)
        indent_size = self.indent if isinstance(self.indent, int) else 2
        return self._encode_recursive(formatted, depth=0, indent_size=indent_size)

    def iterencode(self, o: Any, _one_shot: bool = False):
        return iter([self.encode(o)])

    def _encode_recursive(self, obj: Any, *, depth: int, indent_size: int) -> str:
        """Serialize one object, rendering ``_FormattedFloat`` values unquoted."""
        if isinstance(obj, _FormattedFloat):
            return obj.text
        if obj is None:
            return "null"
        if isinstance(obj, bool):
            return "true" if obj else "false"
        if isinstance(obj, int):
            return str(obj)
        if isinstance(obj, float):
            return f"{obj:.{FLOAT_SIG_FIGS - 1}e}"
        if isinstance(obj, str):
            return json.dumps(obj, ensure_ascii=self.ensure_ascii)
        if isinstance(obj, dict):
            items = sorted(obj.items()) if self.sort_keys else obj.items()
            entries = [
                f"{json.dumps(str(key), ensure_ascii=self.ensure_ascii)}: "
                f"{self._encode_recursive(value, depth=depth + 1, indent_size=indent_size)}"
                for key, value in items
            ]
            return self._format_compound("{", entries, "}", depth=depth, indent_size=indent_size)
        if isinstance(obj, (list, tuple)):
            entries = [
                self._encode_recursive(item, depth=depth + 1, indent_size=indent_size)
                for item in obj
            ]
            return self._format_compound("[", entries, "]", depth=depth, indent_size=indent_size)
        return json.dumps(obj, ensure_ascii=self.ensure_ascii)

    def _format_compound(
        self,
        open_br: str,
        entries: list[str],
        close_br: str,
        *,
        depth: int,
        indent_size: int,
    ) -> str:
        """Format a dict or list with depth-aware indentation."""
        if self.indent is not None:
            if not entries:
                return f"{open_br}{close_br}"
            inner_indent = " " * (indent_size * (depth + 1))
            outer_indent = " " * (indent_size * depth)
            inner = ",\n".join(f"{inner_indent}{entry}" for entry in entries)
            return f"{open_br}\n{inner}\n{outer_indent}{close_br}"
        if not entries:
            return f"{open_br}{close_br}"
        return f"{open_br}{', '.join(entries)}{close_br}"


def _positive_int(value: str) -> int:
    """Parse one CLI integer and require it to be positive."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _parse_args() -> argparse.Namespace:
    """Parse the hyperparameter search CLI."""
    parser = argparse.ArgumentParser(
        description="Run a small Optuna study over training-only hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.json"),
        help="Relative or absolute path to the JSON config file.",
    )
    parser.add_argument(
        "--trials",
        type=_positive_int,
        default=DEFAULT_TRIALS,
        help="Number of Optuna trials to run.",
    )
    parser.add_argument(
        "--trial-epochs",
        type=_positive_int,
        default=DEFAULT_TRIAL_EPOCHS,
        help="Epoch budget for each trial.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing models/hyperparam_testing directory before starting.",
    )
    return parser.parse_args()


def _project_root() -> Path:
    """Resolve the project root in the same env-aware way as the main CLI."""
    env_root = os.environ.get("VULCAN_EMULATOR_PROJECT_ROOT")
    if env_root:
        return Path(os.path.normpath(env_root))
    return Path(__file__).resolve().parent.parent


def _resolve_config_path(path: Path) -> Path:
    """Resolve the config path relative to the project root when needed."""
    return path if path.is_absolute() else _project_root() / path


def _trial_name(trial_number: int) -> str:
    """Return the canonical trial folder/file stem."""
    return f"trial_{trial_number:03d}"


def _trial_output_folder(trial_number: int) -> str:
    """Build the training.output_folder value for one trial scratch run."""
    return _trial_name(trial_number)


def _logs_root_relative(base_config: dict[str, Any]) -> str:
    """Build the relative logs root used by all hyperparameter trials."""
    return str(Path(base_config["paths"]["models_root"]) / HYPERPARAM_ROOT_NAME / "logs")


def _trial_models_root_relative(base_config: dict[str, Any]) -> str:
    """Build the relative models root used for scratch trial checkpoints."""
    return str(Path(base_config["paths"]["models_root"]) / HYPERPARAM_ROOT_NAME / "_scratch")


def _hyperparam_root(base_config: dict[str, Any]) -> Path:
    """Resolve the top-level hyperparameter output directory."""
    base_paths = resolve_paths(base_config)
    return base_paths.models_root / HYPERPARAM_ROOT_NAME


def _prepare_output_root(root: Path, *, overwrite: bool) -> None:
    """Create the hyperparameter output layout, optionally replacing old results."""
    if root.exists():
        if not overwrite:
            raise RuntimeError(
                f"Output directory already exists: {root}. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(root)
    (root / "_scratch").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)


def _cleanup_trial_dir(path: Path) -> None:
    """Best-effort removal for one scratch trial directory."""
    if path.exists():
        shutil.rmtree(path)


def _cleanup_cuda() -> None:
    """Release Python and CUDA memory between trials."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _classify_failure(exc: BaseException) -> str:
    """Classify one failed trial for summary logging."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return "oom"
    if "out of memory" in str(exc).lower():
        return "oom"
    return "failed"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON object with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            cls=_ScientificNotationJSONEncoder,
        )


def _rewrite_json_file(path: Path) -> None:
    """Rewrite one existing JSON artifact with the scientific-notation encoder."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}, found {type(payload).__name__}.")
    _write_json(path, payload)


def _rewrite_training_log_csv(path: Path) -> None:
    """Rewrite one copied training log with scientific notation and four sig figs."""
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise RuntimeError(f"Missing CSV header in {path}.")
        rows = list(reader)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted: dict[str, str] = {}
            for key, value in row.items():
                if value is None or value == "":
                    formatted[key] = ""
                elif key == "epoch":
                    formatted[key] = str(int(float(value)))
                else:
                    formatted[key] = f"{float(value):.{FLOAT_SIG_FIGS - 1}e}"
            writer.writerow(formatted)


def _ensure_trial_training_log(path: Path) -> None:
    """Ensure each trial leaves behind a training log CSV path."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(TRIAL_LOG_HEADER) + "\n")


def _copy_trial_training_log(
    *,
    trial_run_dir: Path,
    destination: Path,
) -> None:
    """Copy the per-epoch training CSV out of one scratch run before cleanup."""
    source = trial_run_dir / "training_log.csv"
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        _rewrite_training_log_csv(destination)
    else:
        _ensure_trial_training_log(destination)


def _load_metrics(path: Path) -> dict[str, Any]:
    """Load the trainer metrics JSON for one completed trial."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}, found {type(payload).__name__}.")
    return payload


def _trial_progress_log_path(*, output_root: Path, trial_number: int) -> Path:
    """Return the trainer progress-log path produced for one trial."""
    return output_root / "logs" / f"training_progress_{_trial_name(trial_number)}.log"


def _sample_trial_config(
    *,
    base_config: dict[str, Any],
    trial: optuna.trial.Trial,
    trial_epochs: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Clone the base config and apply one Optuna-sampled hyperparameter set.

    Search space:

    - **architecture**: ``{256x4, 384x6, 512x8, 768x12}`` (d_model x nhead)
    - **num_layers**: ``{4, 6, 8}``
    - **dim_feedforward_multiplier**: ``{2, 4}`` (times d_model)
    - **conditioning_hidden_multiplier**: ``{1, 2}`` (times d_model)
    - **batch_size**: ``{128, 256}``
    - **learning_rate**: log-uniform in ``[3e-5, 3e-4]``
    - **weight_decay**: log-uniform in ``[1e-6, 1e-3]``
    - **dropout**: ``{0.0, 0.05, 0.10, 0.15}``
    - **train_pairs_per_run_per_epoch**: ``{500, 1000, 1500}``

    Returns the mutated config and a flat dict of derived hyperparameter values
    for logging.
    """
    config = deepcopy(base_config)
    training = config["training"]
    model_cfg = training["model"]

    architecture = trial.suggest_categorical("architecture", list(ARCHITECTURE_PRESETS))
    d_model, nhead = ARCHITECTURE_PRESETS[architecture]
    dim_feedforward_multiplier = trial.suggest_categorical(
        "dim_feedforward_multiplier",
        [2, 4],
    )
    conditioning_hidden_multiplier = trial.suggest_categorical(
        "conditioning_hidden_multiplier",
        [1, 2],
    )

    training["epochs"] = int(trial_epochs)
    training["warmup_epochs"] = min(int(training["warmup_epochs"]), int(trial_epochs))
    training["batch_size"] = int(trial.suggest_categorical("batch_size", [128, 256]))
    training["learning_rate"] = float(trial.suggest_float("learning_rate", 3.0e-5, 3.0e-4, log=True))
    training["weight_decay"] = float(trial.suggest_float("weight_decay", 1.0e-6, 1.0e-3, log=True))
    training["output_folder"] = _trial_output_folder(trial.number)
    training["live_sampling"]["train_pairs_per_run_per_epoch"] = int(
        trial.suggest_categorical(
            "train_pairs_per_run_per_epoch",
            [500, 1000, 1500],
        )
    )

    model_cfg["d_model"] = int(d_model)
    model_cfg["nhead"] = int(nhead)
    model_cfg["num_layers"] = int(trial.suggest_categorical("num_layers", [4, 6, 8]))
    model_cfg["dim_feedforward"] = int(d_model * dim_feedforward_multiplier)
    model_cfg["conditioning_hidden_dim"] = int(d_model * conditioning_hidden_multiplier)
    model_cfg["dropout"] = float(trial.suggest_categorical("dropout", [0.0, 0.05, 0.10, 0.15]))

    config["paths"]["logs_root"] = _logs_root_relative(base_config)
    config["paths"]["models_root"] = _trial_models_root_relative(base_config)

    derived = {
        "architecture": architecture,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": model_cfg["num_layers"],
        "dim_feedforward": model_cfg["dim_feedforward"],
        "conditioning_hidden_dim": model_cfg["conditioning_hidden_dim"],
        "dropout": model_cfg["dropout"],
        "batch_size": training["batch_size"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "train_pairs_per_run_per_epoch": training["live_sampling"]["train_pairs_per_run_per_epoch"],
        "epochs": training["epochs"],
        "warmup_epochs": training["warmup_epochs"],
    }
    return config, derived


def _promote_best_trial(
    *,
    best_config: dict[str, Any],
    base_config: dict[str, Any],
    best_trial_dir: Path,
    output_root: Path,
) -> None:
    """Promote one winning scratch trial into the final best_model directory.

    Copies ``PROMOTED_ARTIFACTS``, rewrites the checkpoint's embedded config
    to point at the permanent output folder, and reformats all JSON/CSV
    artifacts with scientific notation.
    """
    best_model_dir = output_root / "best_model"
    if best_model_dir.exists():
        shutil.rmtree(best_model_dir)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    for filename in PROMOTED_ARTIFACTS:
        source = best_trial_dir / filename
        if not source.is_file():
            raise RuntimeError(f"Missing expected winning-trial artifact: {source}")
        shutil.copy2(source, best_model_dir / filename)

    promoted_config = deepcopy(best_config)
    promoted_config["paths"]["models_root"] = str(base_config["paths"]["models_root"])
    promoted_config["training"]["output_folder"] = BEST_MODEL_OUTPUT_FOLDER

    best_checkpoint_path = best_model_dir / "best.pt"
    checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Invalid checkpoint structure in {best_checkpoint_path}.")
    checkpoint["config"] = promoted_config
    torch.save(checkpoint, best_checkpoint_path)

    _rewrite_training_log_csv(best_model_dir / "training_log.csv")
    for json_name in (
        "metrics.json",
        "data_contract.json",
        "normalization_metadata.json",
        "processed_fingerprint.json",
    ):
        _rewrite_json_file(best_model_dir / json_name)
    _write_json(output_root / "best_config.json", promoted_config)


def run_hyperparameter_search(
    *,
    config_path: Path,
    trials: int,
    trial_epochs: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Run the Optuna study and return the winning-trial summary.

    Lifecycle per trial:

    1. Sample hyperparameters via TPE.
    2. Deep-copy and mutate the base config.
    3. Run full ``run_training()`` in a scratch directory.
    4. Record the best-val-combined-loss as the Optuna objective.
    5. Keep the current-best trial's scratch dir; clean up losers.

    After all trials, the winning trial's artifacts are promoted into
    ``models/hyperparam_testing/best_model/`` and the scratch root is removed.
    """
    resolved_config_path = _resolve_config_path(config_path)
    base_config = load_and_validate_config(resolved_config_path)
    output_root = _hyperparam_root(base_config)
    _prepare_output_root(output_root, overwrite=overwrite)

    sampler = optuna.samplers.TPESampler(seed=int(base_config["training"]["seed"]))
    study = optuna.create_study(direction="minimize", sampler=sampler)

    best_summary: dict[str, Any] | None = None
    best_config: dict[str, Any] | None = None
    best_trial_dir: Path | None = None

    for _ in range(trials):
        trial = study.ask()
        trial_recorded = False
        trial_name = _trial_name(trial.number)
        trial_log_path = output_root / "logs" / f"{trial_name}.log"
        trial_training_log_path = output_root / "logs" / f"{trial_name}_training_log.csv"
        trial_summary_path = output_root / "logs" / f"{trial_name}_summary.json"
        trial_progress_log_path = _trial_progress_log_path(
            output_root=output_root,
            trial_number=trial.number,
        )

        trial_config, derived = _sample_trial_config(
            base_config=base_config,
            trial=trial,
            trial_epochs=trial_epochs,
        )
        precision = resolve_precision(trial_config)
        trial_paths = resolve_paths(trial_config)
        ensure_runtime_dirs(trial_paths)
        trial_run_dir = trial_paths.models_root / str(trial_config["training"]["output_folder"])

        setup_logging(trial_log_path)
        logger.info("Starting hyperparameter trial %d/%d (%s)", trial.number + 1, trials, trial_name)
        logger.info("Using config: %s", resolved_config_path)
        logger.info("Sampled hyperparameters: %s", derived)

        summary: dict[str, Any] = {
            "trial_number": int(trial.number),
            "trial_name": trial_name,
            "status": "running",
            "params": dict(trial.params),
            "derived_hyperparameters": derived,
            "config": trial_config,
            "trial_log": str(trial_log_path),
            "training_log": str(trial_training_log_path),
        }

        try:
            run_training(trial_config, trial_paths, precision)
            metrics = _load_metrics(trial_run_dir / "metrics.json")
            objective = float(metrics["best_val_combined_loss"])

            summary["status"] = "completed"
            summary["best_val_combined_loss"] = objective
            summary["metrics"] = metrics

            if best_summary is None or objective < float(best_summary["best_val_combined_loss"]):
                if best_trial_dir is not None and best_trial_dir != trial_run_dir:
                    _cleanup_trial_dir(best_trial_dir)
                best_summary = deepcopy(summary)
                best_config = deepcopy(trial_config)
                best_trial_dir = trial_run_dir
            else:
                _copy_trial_training_log(
                    trial_run_dir=trial_run_dir,
                    destination=trial_training_log_path,
                )
                _cleanup_trial_dir(trial_run_dir)

            logger.info(
                "Completed trial %s with best_val_combined_loss=%.8e",
                trial_name,
                objective,
            )
            study.tell(trial, objective)
            trial_recorded = True

        except Exception as exc:
            if not trial_recorded:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            summary["status"] = _classify_failure(exc)
            summary["error_type"] = type(exc).__name__
            summary["error_message"] = str(exc)
            logger.exception("Trial %s failed: %s", trial_name, exc)
            _copy_trial_training_log(
                trial_run_dir=trial_run_dir,
                destination=trial_training_log_path,
            )
            _cleanup_trial_dir(trial_run_dir)

        finally:
            if not trial_training_log_path.exists():
                _copy_trial_training_log(
                    trial_run_dir=trial_run_dir,
                    destination=trial_training_log_path,
                )
            _write_json(trial_summary_path, summary)
            if trial_progress_log_path.exists():
                trial_progress_log_path.unlink()
            _cleanup_cuda()

    if best_summary is None or best_config is None or best_trial_dir is None:
        raise RuntimeError("All hyperparameter trials failed; no best model was produced.")

    _promote_best_trial(
        best_config=best_config,
        base_config=base_config,
        best_trial_dir=best_trial_dir,
        output_root=output_root,
    )
    scratch_root = output_root / "_scratch"
    if scratch_root.exists():
        shutil.rmtree(scratch_root)

    final_summary = {
        "best_trial_number": int(best_summary["trial_number"]),
        "best_val_combined_loss": float(best_summary["best_val_combined_loss"]),
        "best_model_dir": str(output_root / "best_model"),
        "best_config_path": str(output_root / "best_config.json"),
    }
    logger.info("Hyperparameter search complete: %s", final_summary)
    return final_summary


def main() -> int:
    """Run the hyperparameter study from the command line."""
    args = _parse_args()
    setup_logging()
    try:
        run_hyperparameter_search(
            config_path=args.config,
            trials=args.trials,
            trial_epochs=args.trial_epochs,
            overwrite=args.overwrite,
        )
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except Exception as exc:
        logger.error("Hyperparameter search failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
