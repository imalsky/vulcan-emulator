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
from preprocess import (
    PreprocessError,
    build_worker_settings,
    discover_existing_raw_run_files,
    run_generation_and_preprocess,
)
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
    """Require the expected conda environment before any heavy imports or IO.

    The expected environment name is read from the ``VULCAN_EMULATOR_CONDA_ENV``
    environment variable.  If unset, it defaults to ``nn`` (the local dev env).
    On HPC, ``run.pbs`` exports this variable after activating the cluster env.
    """
    expected = os.environ.get("VULCAN_EMULATOR_CONDA_ENV", "nn")
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    if env_name != expected:
        raise RuntimeError(
            f"This project must run inside conda env '{expected}'. "
            f"Use: conda run -n {expected} python src/main.py ... "
            "(or set VULCAN_EMULATOR_CONDA_ENV to override)."
        )


def _resolve_config_path(arg_path: Path) -> Path:
    """Resolve the user-provided config path relative to the project root.

    Uses ``VULCAN_EMULATOR_PROJECT_ROOT`` when set (HPC), otherwise derives
    the root from this file's location (``src/..``).  Path joining uses
    ``os.path.normpath`` instead of ``.resolve()`` so that symlink-based
    canonical-path rewriting on HPC does not break sibling-directory references.
    """
    if arg_path.is_absolute():
        raise RuntimeError("Config path must be relative.")
    env_root = os.environ.get("VULCAN_EMULATOR_PROJECT_ROOT")
    if env_root:
        project_root = Path(os.path.normpath(env_root))
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
            existing_raw_run_files = discover_existing_raw_run_files(paths)
            boundary_conditions = None
            if existing_raw_run_files:
                logger.info(
                    "Detected %d existing raw run files; skipping VULCAN preflight and generation.",
                    len(existing_raw_run_files),
                )
            else:
                boundary_conditions = resolve_boundary_conditions(config, paths.vulcan_source)
                state_species = list(config["data_spec"]["state_species"])
                output_species = list(config["data_spec"]["output_species"])
                validate_species_available(
                    paths.vulcan_source,
                    state_species=tuple(state_species),
                    output_species=tuple(output_species),
                )
                settings = build_worker_settings(
                    config,
                    paths,
                    boundary_conditions=boundary_conditions,
                    state_species=state_species,
                    output_species=output_species,
                )
                preflight_vulcan_source(
                    paths.vulcan_source,
                    settings=settings,
                    timeout_seconds=int(config["generation"]["run_timeout_seconds"]),
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
