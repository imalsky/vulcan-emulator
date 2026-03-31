"""CLI entry point dispatching the three pipeline stages.

Invoked via ``python -m src.utils --config <path> --stage <stage>``.
The three stages form a sequential pipeline:

1. **generation** — sample atmospheric parameters and produce raw
   HDF5 runs (calls ``generation.generate_raw_dataset``).
2. **normalization** — fit normalization on the training split and
   convert raw HDF5 to processed NumPy tensors
   (calls ``preprocess.preprocess_raw_dataset``).
3. **training** — train the JAX model on the processed data
   (calls ``trainer.train_model``).

Each stage prints a JSON summary of produced artefacts to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_and_validate_config
from .helpers import get_logger, resolve_project_root
from ..data_generation.migrate import migrate_from_config
from ..data_generation.preprocess import preprocess_raw_dataset
from ..data_generation.generation import generate_raw_dataset
from ..training.trainer import train_model

LOGGER = get_logger(__name__)


def _load_config(config_path: str | Path, project_root: Path) -> dict[str, Any]:
    """Resolve, validate, and annotate the requested config file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root / path
    config = load_and_validate_config(path)
    # Persist the resolved project root for downstream path resolution.
    config["_project_root"] = str(project_root.resolve())
    return config


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, resolve the project root, and dispatch the requested stage.

    Returns 0 on success.  Raises ``ValueError`` for unrecognised stages.
    """
    parser = argparse.ArgumentParser(description="Photochemical VULCAN surrogate pipeline.")
    parser.add_argument(
        "--config",
        default="config/equilibrium_only_config.json",
        help="Path to a configuration JSON file relative to the project root.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("generation", "normalization", "training", "migrate"),
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="For migrate: report what would happen without making changes.",
    )

    args = parser.parse_args(argv)
    project_root = resolve_project_root(Path(__file__).resolve())
    config = _load_config(args.config, project_root)

    LOGGER.info("Starting stage: %s", args.stage)

    if args.stage == "generation":
        artifact = generate_raw_dataset(config, project_root=project_root)
        LOGGER.info("Generation complete: %d runs", len(artifact.run_files))
        print(
            json.dumps(
                {
                    "raw_root": str(artifact.raw_root),
                    "num_runs": len(artifact.run_files),
                    "manifest_path": str(artifact.manifest_path) if artifact.manifest_path is not None else None,
                    "coverage_path": str(artifact.coverage_path) if artifact.coverage_path is not None else None,
                },
                indent=2,
            )
        )
        return 0
    if args.stage == "normalization":
        artifact = preprocess_raw_dataset(config, project_root=project_root)
        LOGGER.info("Normalization complete")
        print(json.dumps(artifact, indent=2))
        return 0
    if args.stage == "training":
        artifact = train_model(config, project_root=project_root)
        LOGGER.info("Training complete: %s", artifact.checkpoint_path)
        print(
            json.dumps(
                {
                    "checkpoint_path": str(artifact.checkpoint_path),
                    "history_path": str(artifact.history_path),
                    "metrics_path": str(artifact.metrics_path),
                },
                indent=2,
            )
        )
        return 0
    if args.stage == "migrate":
        result = migrate_from_config(config, project_root=project_root, dry_run=args.dry_run)
        if result is not None:
            LOGGER.info("Migration complete: %s", result)
            print(json.dumps({"consolidated_path": str(result)}, indent=2))
        else:
            LOGGER.info("Nothing to migrate.")
            print(json.dumps({"consolidated_path": None}, indent=2))
        return 0
    raise ValueError(f"Unhandled stage: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
