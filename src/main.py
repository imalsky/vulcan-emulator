from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config_utils import load_and_validate_config
from .hyperparam_testing import run_hyperparam_search
from .path_utils import resolve_project_root
from .preprocess import preprocess_raw_dataset
from .trainer import train_model
from .vulcan_runner import generate_raw_dataset


def _load_config(config_path: str | Path, project_root: Path) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root / path
    return load_and_validate_config(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Photochemical VULCAN surrogate pipeline.")
    parser.add_argument(
        "--config",
        default="config/config.json",
        help="Path to a configuration JSON file relative to the project root.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("gen", help="Generate raw runs.")
    subparsers.add_parser("preprocess", help="Preprocess raw HDF5 runs.")
    subparsers.add_parser("train", help="Train the JAX surrogate and export the best model.")
    subparsers.add_parser("hyperparam", help="Run a tiny built-in hyperparameter search.")
    subparsers.add_parser("show-config", help="Print the validated config.")

    args = parser.parse_args(argv)
    project_root = resolve_project_root(Path(__file__).resolve())
    config = _load_config(args.config, project_root)

    if args.command == "gen":
        artifact = generate_raw_dataset(config, project_root=project_root)
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
    if args.command == "preprocess":
        artifact = preprocess_raw_dataset(config, project_root=project_root)
        print(json.dumps(artifact, indent=2))
        return 0
    if args.command == "train":
        artifact = train_model(config, project_root=project_root)
        print(
            json.dumps(
                {
                    "checkpoint_path": str(artifact.checkpoint_path),
                    "export_root": str(artifact.export_root),
                    "history_path": str(artifact.history_path),
                    "metrics_path": str(artifact.metrics_path),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "hyperparam":
        result = run_hyperparam_search(config, project_root=project_root)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "show-config":
        print(json.dumps(config, indent=2))
        return 0
    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
