"""Path resolution and project layout utilities.

Supports two modes of project-root resolution:

- **Local dev**: Derives the root from ``__file__`` (``src/..``), uses
  ``Path.resolve()`` for canonical absolute paths.
- **HPC / batch**: Reads ``VULCAN_EMULATOR_PROJECT_ROOT`` and optionally
  ``VULCAN_EMULATOR_VULCAN_SOURCE``, joins paths with ``os.path.normpath``
  instead of ``.resolve()`` so that symlink-based canonical-path rewriting
  on cluster filesystems does not break sibling-directory references.

Environment variables consulted:

- ``VULCAN_EMULATOR_PROJECT_ROOT``: Override the project root directory.
- ``VULCAN_EMULATOR_VULCAN_SOURCE``: Override the VULCAN source tree path
  (takes precedence over ``paths.vulcan_source_path`` in config).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved absolute paths for all pipeline directories."""

    root: Path
    vulcan_source: Path
    data_root: Path
    raw_root: Path
    processed_root: Path
    models_root: Path
    logs_root: Path


class PathValidationError(ValueError):
    """Raised when a configured path violates project path rules."""


def _ensure_relative(value: str, field_name: str) -> Path:
    """Parse one configured path string and reject absolute values."""
    path = Path(value)
    if path.is_absolute():
        raise PathValidationError(
            f"Config path '{field_name}' must be relative, got absolute path: {value}"
        )
    return path


def _project_root() -> Path:
    """Return the project root, honoring ``VULCAN_EMULATOR_PROJECT_ROOT``.

    When the environment variable is set (HPC / batch mode), its value is
    normalized but *not* resolved through symlinks.  Otherwise, the root is
    derived from this file's location (``src/..``) using ``Path.resolve()``.
    """
    env_root = os.environ.get("VULCAN_EMULATOR_PROJECT_ROOT")
    if env_root:
        return Path(os.path.normpath(env_root))
    return Path(__file__).resolve().parent.parent


def _override_vulcan_source() -> Path | None:
    """Return an explicit VULCAN source override, or ``None``.

    When ``VULCAN_EMULATOR_VULCAN_SOURCE`` is set (typically by ``run.pbs``),
    that path is used verbatim instead of the config-relative
    ``paths.vulcan_source_path``.
    """
    env_source = os.environ.get("VULCAN_EMULATOR_VULCAN_SOURCE")
    if env_source:
        return Path(os.path.normpath(env_source))
    return None


def _join_project_path(root: Path, relative: Path) -> Path:
    """Join *root* and *relative* without resolving through symlinks.

    Uses ``os.path.normpath`` to clean ``..`` segments without following
    symlinks, which is critical on HPC filesystems where ``Path.resolve()``
    rewrites the prefix to a canonical mount point that differs from the
    working-directory path exported by PBS.
    """
    return Path(os.path.normpath(str(root / relative)))


def resolve_paths(config: dict[str, Any]) -> ProjectPaths:
    """Resolve and validate all configured project-relative paths.

    All configured paths are relative to the project root.  The root is
    determined by ``_project_root()`` (env var or ``__file__``).  If
    ``VULCAN_EMULATOR_VULCAN_SOURCE`` is set, it overrides the configured
    ``paths.vulcan_source_path``.

    Path joining avoids ``.resolve()`` so that HPC symlink rewriting does
    not break sibling-directory references.
    """
    root = _project_root()
    paths_cfg = config["paths"]

    project_root_cfg = _ensure_relative(str(paths_cfg["project_root"]), "paths.project_root")
    if project_root_cfg != Path("."):
        raise PathValidationError(
            "paths.project_root must be '.' to enforce root-relative contract."
        )

    # VULCAN source: environment override takes precedence over config.
    override = _override_vulcan_source()
    if override is not None:
        vulcan_source = override
    else:
        vulcan_source = _join_project_path(
            root,
            _ensure_relative(str(paths_cfg["vulcan_source_path"]), "paths.vulcan_source_path"),
        )

    data_root = _join_project_path(
        root, _ensure_relative(str(paths_cfg["data_root"]), "paths.data_root")
    )
    models_root = _join_project_path(
        root, _ensure_relative(str(paths_cfg["models_root"]), "paths.models_root")
    )
    logs_root = _join_project_path(
        root, _ensure_relative(str(paths_cfg["logs_root"]), "paths.logs_root")
    )

    raw_root = _join_project_path(
        root, _ensure_relative(str(paths_cfg["raw_root"]), "paths.raw_root")
    )
    processed_root = _join_project_path(
        root,
        _ensure_relative(str(paths_cfg["processed_root"]), "paths.processed_root"),
    )

    return ProjectPaths(
        root=root,
        vulcan_source=vulcan_source,
        data_root=data_root,
        raw_root=raw_root,
        processed_root=processed_root,
        models_root=models_root,
        logs_root=logs_root,
    )


def ensure_runtime_dirs(paths: ProjectPaths) -> None:
    """Create required runtime directories if they do not already exist."""
    for directory in (
        paths.data_root,
        paths.raw_root,
        paths.processed_root,
        paths.models_root,
        paths.logs_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)
