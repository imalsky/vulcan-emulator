"""CLI entry point dispatching the emulator stages.

Invoked via ``python -m src.utils --config <path> --stage <stage>``.
The CLI exposes the sequential pipeline stages:

1. **generation** — sample atmospheric parameters and produce raw
   HDF5 runs (calls ``generation.generate_raw_dataset``). Optionally
   sharded across a SLURM job array via ``--shard-id`` and
   ``--num-shards``; each shard writes its own per-shard chunks subdir
   and skips the final merge.
2. **merge_shards** — when generation was sharded, this single-node stage
   consolidates every shard's chunks into ``raw/runs.h5`` and writes the
   unified manifest + sampling-coverage. Skipped for unsharded generation.
3. **normalization** — fit normalization on the training split and
   convert raw HDF5 to processed NumPy tensors
   (calls ``preprocess.preprocess_raw_dataset``).
4. **training** — train the JAX model on the processed data
   (calls ``trainer.train_model``).
5. **export** — convert the best checkpoint to a portable NPZ bundle
   (calls ``export_bundle.export_checkpoint_to_npz``).

Each stage prints a JSON summary of produced artefacts to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..data_generation.generation import generate_raw_dataset, merge_shards_stage
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
    config["_project_root"] = project_root.resolve()
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
    parser = argparse.ArgumentParser(description="FastChem/VULCAN chemistry emulator pipeline.")
    parser.add_argument(
        "--config",
        default="config/fastchem.json",
        help="Path to a configuration JSON file relative to the project root.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("generation", "merge_shards", "normalization", "training", "export"),
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help=(
            "Generation only: zero-based shard index when running as part of a "
            "SLURM job array. Must be paired with --num-shards. The shard owns "
            "the half-open run-index slice [shard_id*N//K, (shard_id+1)*N//K)."
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help=(
            "Generation/merge_shards: total number of shards in the job array. "
            "Required for both --stage generation with --shard-id and for "
            "--stage merge_shards."
        ),
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=None,
        help=(
            "Generation only: directory to use for per-run HDF5 staging and "
            "VULCAN/FastChem worker-tree copies. Intended for node-local "
            "scratch like $SLURM_TMPDIR. The shared-FS chunks_sNN/ directory "
            "is unaffected (chunks always persist on shared FS so they "
            "survive a SLURM kill)."
        ),
    )

    args = parser.parse_args(argv)
    project_root = resolve_project_root(Path(__file__).resolve())
    config = _load_config(args.config, project_root)

    if args.stage == "generation":
        if (args.shard_id is None) != (args.num_shards is None):
            parser.error(
                "--shard-id and --num-shards must be passed together (or both omitted)."
            )
        if args.shard_id is not None and not (0 <= args.shard_id < args.num_shards):
            parser.error(
                f"--shard-id={args.shard_id} out of range for --num-shards={args.num_shards}."
            )
    elif args.stage == "merge_shards":
        if args.num_shards is None:
            parser.error("--num-shards is required for --stage merge_shards.")
        if args.shard_id is not None:
            parser.error("--shard-id is not valid for --stage merge_shards.")
        if args.staging_root is not None:
            parser.error("--staging-root is not valid for --stage merge_shards.")
    else:
        if args.shard_id is not None or args.staging_root is not None:
            parser.error(
                f"--shard-id and --staging-root are only valid for --stage generation, "
                f"not --stage {args.stage}."
            )
        if args.num_shards is not None:
            parser.error(
                f"--num-shards is only valid for --stage generation or merge_shards, "
                f"not --stage {args.stage}."
            )

    LOGGER.info("Starting stage: %s", args.stage)

    if args.stage == "generation":
        artifact = generate_raw_dataset(
            config,
            project_root=project_root,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            staging_root=args.staging_root,
        )
        LOGGER.info("Generation complete: %d runs", len(artifact.run_ids))
        print(
            json.dumps(
                {
                    "run_root": str(artifact.run_root),
                    "num_runs": len(artifact.run_ids),
                    "manifest_path": str(artifact.manifest_path) if artifact.manifest_path is not None else None,
                    "coverage_path": str(artifact.coverage_path) if artifact.coverage_path is not None else None,
                    "shard_id": args.shard_id,
                    "num_shards": args.num_shards,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    if args.stage == "merge_shards":
        artifact = merge_shards_stage(
            config, project_root=project_root, num_shards=args.num_shards,
        )
        LOGGER.info("merge_shards complete: %d runs", len(artifact.run_ids))
        print(
            json.dumps(
                {
                    "runs_h5_path": str(artifact.consolidated_path),
                    "num_runs": len(artifact.run_ids),
                    "manifest_path": str(artifact.manifest_path) if artifact.manifest_path is not None else None,
                    "coverage_path": str(artifact.coverage_path) if artifact.coverage_path is not None else None,
                    "merged_shard_count": args.num_shards,
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
        LOGGER.info("Training complete: %s", artifact.run_root)
        print(
            json.dumps(
                {
                    "run_root": str(artifact.run_root),
                    "config_path": str(artifact.config_path),
                    "history_path": str(artifact.history_path),
                    "metadata_path": str(artifact.metadata_path),
                    "params_best_path": str(artifact.params_best_path),
                    "params_last_path": str(artifact.params_last_path),
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
        params_best = checkpoints_root / "params_best.npz"
        if not params_best.exists():
            LOGGER.error("No best params file found at %s", params_best)
            return 1
        bundle_path = export_checkpoint_to_npz(
            checkpoints_root,
            which="best",
            output_path=checkpoints_root / "best_exported.npz",
        )
        LOGGER.info("Export complete: %s", bundle_path)
        print(json.dumps({"bundle_path": str(bundle_path)}, indent=2), flush=True)
        return 0
    raise ValueError(f"Unhandled stage: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
