#!/usr/bin/env python3
"""Main CLI for vulcan-emulator.

Supported actions:
- --gen: generate raw VULCAN trajectories and processed normalized shards
- --train: train surrogate model from processed data
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Prevent host-specific OpenMP duplicate-runtime aborts before importing torch.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config_utils import ConfigValidationError, load_and_validate_config, resolve_precision
from logging_utils import setup_logging
from path_utils import PathValidationError, ensure_runtime_dirs, resolve_paths
from preprocess import PreprocessError, run_generation_and_preprocess
from trainer import TrainingError, run_training
from vulcan_runner import (
    VulcanRuntimeError,
    preflight_vulcan_source,
    resolve_boundary_conditions,
    validate_species_available,
)

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse the constrained CLI surface for the emulator pipeline."""
    parser = argparse.ArgumentParser(
        description="VULCAN surrogate pipeline (fail-fast).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.json"),
        help="Relative path to JSON config file.",
    )

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--gen", action="store_true", help="Generate raw + processed datasets.")
    action.add_argument("--train", action="store_true", help="Train from processed datasets.")
    return parser.parse_args()


def _enforce_nn_environment() -> None:
    """Require the expected conda environment before any heavy imports or IO."""
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    expected_env = os.environ.get("VULCAN_EMULATOR_CONDA_ENV", "nn")
    if env_name != expected_env:
        raise RuntimeError(
            f"This project must run inside conda env '{expected_env}'. "
            f"Use: conda run -n {expected_env} python src/main.py ..."
        )


def _resolve_config_path(arg_path: Path) -> Path:
    """Resolve the user-provided config path relative to the project root."""
    if arg_path.is_absolute():
        raise RuntimeError("Config path must be relative.")
    project_root_override = os.environ.get("VULCAN_EMULATOR_PROJECT_ROOT")
    if project_root_override:
        project_root = Path(os.path.normpath(project_root_override))
    else:
        project_root = Path(__file__).resolve().parent.parent
    return Path(os.path.normpath(str(project_root / arg_path)))


def main() -> int:
    """Execute the requested pipeline action and return a process exit code."""
    args = _parse_args()

    try:
        _enforce_nn_environment()

        config_path = _resolve_config_path(args.config)
        config = load_and_validate_config(config_path)
        precision = resolve_precision(config)

        paths = resolve_paths(config)
        ensure_runtime_dirs(paths)

        action_name = "gen" if args.gen else "train"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = paths.logs_root / f"{action_name}_{timestamp}.log"
        setup_logging(log_file)

        logger.info("Loaded config: %s", config_path)
        logger.info("Action: --%s", action_name)

        # Persist exact runtime config for reproducibility
        snapshot_path = paths.logs_root / f"runtime_config_{action_name}_{timestamp}.json"
        with snapshot_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

        if args.gen:
            boundary_conditions = resolve_boundary_conditions(config, paths.vulcan_source)
            validate_species_available(
                paths.vulcan_source,
                state_species=tuple(config["data_spec"]["state_species"]),
                output_species=tuple(config["data_spec"]["output_species"]),
            )
            preflight_vulcan_source(
                paths.vulcan_source,
                boundary_conditions=boundary_conditions,
                use_transport=bool(config["physics_toggles"]["use_transport"]),
                use_condensation_optional=bool(
                    config["physics_toggles"]["use_condensation_optional"]
                ),
                timeout_seconds=min(int(config["generation"]["run_timeout_seconds"]), 120),
            )
            run_generation_and_preprocess(
                config,
                paths,
                precision=precision,
                boundary_conditions=boundary_conditions,
            )
        else:
            run_training(config, paths, precision)

        logger.info("Action --%s completed successfully.", action_name)
        return 0

    except (
        RuntimeError,
        FileNotFoundError,
        ConfigValidationError,
        PathValidationError,
        PreprocessError,
        VulcanRuntimeError,
        TrainingError,
    ) as exc:
        logger.error("Fatal error: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except Exception as exc:
        logger.exception("Unhandled exception: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
