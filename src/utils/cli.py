"""CLI entry point dispatching the emulator stages.

Invoked via ``python -m src.utils --config <path> --stage <stage>``.
The CLI exposes four sequential pipeline stages:

1. **generation** — sample atmospheric parameters and produce raw
   HDF5 runs (calls ``generation.generate_raw_dataset``).
2. **normalization** — fit normalization on the training split and
   convert raw HDF5 to processed NumPy tensors
   (calls ``preprocess.preprocess_raw_dataset``).
3. **training** — train the JAX model on the processed data
   (calls ``trainer.train_model``).
4. **export** — convert the best checkpoint to a portable NPZ bundle
   (calls ``export_bundle.export_checkpoint_to_npz``).

Each stage prints a JSON summary of produced artefacts to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..data_generation.generation import generate_raw_dataset
from ..data_generation.preprocess import preprocess_raw_dataset
from ..models.export_bundle import export_checkpoint_to_npz
from ..training.trainer import train_model
from .config import load_and_validate_config
from .helpers import get_logger, resolve_project_root

LOGGER = get_logger(__name__)


def _load_config(config_path: str | Path, project_root: Path) -> dict[str, Any]:
    """Resolve, validate, and annotate the requested config file.

    Parameters
    ----------
    config_path : str or Path
        User-supplied config path, absolute or relative to ``project_root``.
    project_root : Path
        Repository root used for path resolution and stored for downstream
        helpers.

    Returns
    -------
    dict[str, Any]
        Validated config dictionary annotated with ``_project_root``.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root / path
    config = load_and_validate_config(path)
    # Persist the resolved project root for downstream path resolution.
    config["_project_root"] = str(project_root.resolve())
    return config


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, resolve the project root, and dispatch the requested stage.

    Parameters
    ----------
    argv : list[str] or None, optional
        Optional CLI argument vector. When ``None``, arguments are read from
        ``sys.argv``.

    Returns
    -------
    int
        Process-style exit code for the selected stage.
    """
    parser = argparse.ArgumentParser(description="Photochemical VULCAN surrogate pipeline.")
    parser.add_argument(
        "--config",
        default="config/fastchem_mlp_config.json",
        help="Path to a configuration JSON file relative to the project root.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("generation", "normalization", "training", "export"),
        help="Pipeline stage to execute.",
    )

    args = parser.parse_args(argv)
    project_root = resolve_project_root(Path(__file__).resolve())
    config = _load_config(args.config, project_root)

    LOGGER.info("Starting stage: %s", args.stage)

    if args.stage == "generation":
        artifact = generate_raw_dataset(config, project_root=project_root)
        LOGGER.info("Generation complete: %d runs", len(artifact.run_ids))
        print(
            json.dumps(
                {
                    "run_root": str(artifact.run_root),
                    "num_runs": len(artifact.run_ids),
                    "manifest_path": str(artifact.manifest_path) if artifact.manifest_path is not None else None,
                    "coverage_path": str(artifact.coverage_path) if artifact.coverage_path is not None else None,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    if args.stage == "normalization":
        artifact = preprocess_raw_dataset(config, project_root=project_root)
        LOGGER.info("Normalization complete")
        print(json.dumps(artifact, indent=2), flush=True)
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
            ),
            flush=True,
        )
        return 0
    if args.stage == "export":
        checkpoints_root = Path(config["paths"]["checkpoints_root"])
        if not checkpoints_root.is_absolute():
            checkpoints_root = project_root / checkpoints_root
        best_checkpoint = checkpoints_root / "best.pt"
        if not best_checkpoint.exists():
            LOGGER.error("No best checkpoint found at %s", best_checkpoint)
            return 1
        bundle_path = export_checkpoint_to_npz(best_checkpoint)
        LOGGER.info("Export complete: %s", bundle_path)
        print(json.dumps({"bundle_path": str(bundle_path)}, indent=2), flush=True)
        return 0
    raise ValueError(f"Unhandled stage: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
