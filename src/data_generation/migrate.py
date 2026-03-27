"""Migrate per-file raw runs into a single consolidated HDF5 file.

Usage via CLI::

    python -m src.main migrate [--dry-run]

This reads all ``run_*.h5`` files under ``{raw_root}/runs/``, copies them
into ``{raw_root}/runs.h5`` as top-level groups, and removes the originals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.helpers import get_logger, resolve_path
from .generation import consolidate_runs_to_single_hdf5

LOGGER = get_logger(__name__)


def migrate_per_file_to_consolidated(
    raw_root: Path,
    *,
    dry_run: bool = False,
) -> Path | None:
    """Consolidate existing per-file runs into ``runs.h5``.

    Returns the consolidated path on success, or ``None`` if there is
    nothing to migrate.
    """
    runs_dir = raw_root / "runs"
    run_files = sorted(runs_dir.glob("run_*.h5"))
    consolidated_path = raw_root / "runs.h5"

    if not run_files:
        LOGGER.info("No per-file runs found under %s — nothing to migrate.", runs_dir)
        return None
    if consolidated_path.exists():
        raise RuntimeError(
            f"{consolidated_path} already exists. Remove it first or use generation.overwrite=true."
        )

    LOGGER.info(
        "%sMigrating %d per-file runs to %s",
        "[DRY RUN] " if dry_run else "",
        len(run_files),
        consolidated_path,
    )

    if dry_run:
        return None

    return consolidate_runs_to_single_hdf5(
        run_files, consolidated_path, delete_originals=True,
    )


def migrate_from_config(
    config: dict[str, Any],
    *,
    project_root: Path,
    dry_run: bool = False,
) -> Path | None:
    """Resolve ``raw_root`` from *config* and run the migration."""
    raw_root = resolve_path(config["paths"]["raw_root"], project_root)
    return migrate_per_file_to_consolidated(raw_root, dry_run=dry_run)
