#!/usr/bin/env python3
"""Read one trained VULCAN emulator run and print a compact summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_utils import load_and_validate_config
from inference import load_physical_space_model
from path_utils import resolve_paths


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for reading one trained run."""
    parser = argparse.ArgumentParser(description="Read one trained VULCAN emulator run.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.json"),
        help="Config file used to infer the default run directory.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional explicit run directory. Overrides the config-derived location.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best.pt",
        help="Checkpoint filename inside the run directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device used to load the model.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the summary as JSON.",
    )
    return parser.parse_args()


def _resolve_config_path(config_path: Path) -> Path:
    """Resolve a config path relative to the project root when needed."""
    if config_path.is_absolute():
        return config_path
    return (PROJECT_ROOT / config_path).resolve()


def _resolve_run_dir(args: argparse.Namespace) -> tuple[Path, Path | None]:
    """Resolve the run directory either explicitly or from the config."""
    if args.run_dir is not None:
        run_dir = args.run_dir if args.run_dir.is_absolute() else (PROJECT_ROOT / args.run_dir)
        return run_dir.resolve(), None

    config_path = _resolve_config_path(args.config)
    config = load_and_validate_config(config_path)
    paths = resolve_paths(config)
    run_dir = (paths.models_root / str(config["training"]["output_folder"])).resolve()
    return run_dir, config_path


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    """Load one JSON file when it exists, otherwise return ``None``."""
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {path}.")
    return payload


def _build_summary(
    *,
    run_dir: Path,
    config_path: Path | None,
    checkpoint_name: str,
    device: str,
) -> dict[str, Any]:
    """Load the trained artifacts and build a compact summary."""
    checkpoint_path = run_dir / checkpoint_name
    model, normalization_metadata, data_contract = load_physical_space_model(
        run_dir,
        checkpoint_name=checkpoint_name,
        device=device,
    )
    metrics = _load_json_if_present(run_dir / "metrics.json")

    return {
        "config_path": str(config_path) if config_path is not None else None,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "sequence_length": int(model.sequence_length),
        "state_dim": int(model.state_dim),
        "target_dim": int(model.target_dim),
        "state_species": list(model.state_species),
        "target_species": list(model.target_species),
        "target_normalization": str(normalization_metadata["targets"]["ymix"]["method"]),
        "metrics": metrics,
        "files": {
            "data_contract": str(run_dir / "data_contract.json"),
            "normalization_metadata": str(run_dir / "normalization_metadata.json"),
            "metrics": str(run_dir / "metrics.json"),
        },
        "data_contract": {
            "sequence_feature_order": list(data_contract["sequence_feature_order"]),
            "global_feature_order": list(data_contract["global_feature_order"]),
            "output_from_state_indices": list(data_contract["output_from_state_indices"]),
        },
    }


def _print_human_summary(summary: dict[str, Any]) -> None:
    """Print a compact readable summary."""
    print(f"Run dir: {summary['run_dir']}")
    print(f"Checkpoint: {summary['checkpoint']}")
    if summary["config_path"] is not None:
        print(f"Config: {summary['config_path']}")
    print(f"Device: {summary['device']}")
    print(
        "Shape: "
        f"sequence_length={summary['sequence_length']} "
        f"state_dim={summary['state_dim']} "
        f"target_dim={summary['target_dim']}"
    )
    print(f"State species: {', '.join(summary['state_species'])}")
    print(f"Target species: {', '.join(summary['target_species'])}")
    print(f"Target normalization: {summary['target_normalization']}")

    metrics = summary["metrics"]
    if metrics is None:
        print("Metrics: missing")
    else:
        print("Metrics:")
        for key in ("best_epoch", "best_val_mse"):
            if key in metrics:
                print(f"  {key}: {metrics[key]}")
        for split_name in ("val", "test", "rollout"):
            if split_name in metrics:
                print(f"  {split_name}: {json.dumps(metrics[split_name], sort_keys=True)}")


def main() -> int:
    """Read one run and print its summary."""
    args = _parse_args()
    run_dir, config_path = _resolve_run_dir(args)
    summary = _build_summary(
        run_dir=run_dir,
        config_path=config_path,
        checkpoint_name=args.checkpoint,
        device=args.device,
    )
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_human_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
