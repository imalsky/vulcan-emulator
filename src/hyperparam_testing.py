from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .path_utils import ensure_dir
from .trainer import train_model


def _candidate_configs(base_config: dict[str, Any]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    presets = [
        {"d_model": 32, "nhead": 4, "num_layers": 2, "dim_feedforward": 64, "conditioning_hidden_dim": 64},
        {"d_model": 48, "nhead": 4, "num_layers": 2, "dim_feedforward": 96, "conditioning_hidden_dim": 96},
        {"d_model": 64, "nhead": 4, "num_layers": 3, "dim_feedforward": 128, "conditioning_hidden_dim": 128},
    ]
    for preset in presets:
        cfg = copy.deepcopy(base_config)
        cfg["training"]["model"].update(preset)
        variants.append(cfg)
    return variants


def run_hyperparam_search(
    config: dict[str, Any],
    *,
    project_root: Path,
    trial_epochs: int = 1,
) -> dict[str, Any]:
    search_root = ensure_dir(project_root / "models" / "hyperparam_testing")
    logs_root = ensure_dir(search_root / "logs")
    best_score = float("inf")
    best_result: dict[str, Any] | None = None

    for trial_index, trial_config in enumerate(_candidate_configs(config)):
        trial_cfg = copy.deepcopy(trial_config)
        trial_cfg["training"]["epochs"] = int(trial_epochs)
        trial_cfg["paths"]["checkpoints_root"] = str(search_root / f"_trial_{trial_index:03d}")
        trial_cfg["paths"]["jax_export_root"] = str(search_root / f"_trial_{trial_index:03d}" / "jax")
        artifacts = train_model(trial_cfg, project_root=project_root)
        metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
        result = {
            "trial_index": trial_index,
            "metrics": metrics,
            "model": trial_cfg["training"]["model"],
        }
        (logs_root / f"trial_{trial_index:03d}.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        score = float(metrics["best_val_combined_loss"])
        if score < best_score:
            best_score = score
            best_result = result

    if best_result is None:
        raise RuntimeError("Hyperparameter search produced no valid trials.")
    (search_root / "best_config.json").write_text(
        json.dumps(best_result, indent=2) + "\n",
        encoding="utf-8",
    )
    return best_result
