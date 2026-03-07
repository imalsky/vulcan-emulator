"""Path resolution and project layout utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved project paths used by pipeline actions."""

    root: Path
    vulcan_source: Path
    data_root: Path
    raw_root: Path
    processed_root: Path
    models_root: Path
    logs_root: Path


class PathValidationError(ValueError):
    """Raised when a configured path violates project path rules."""


def _project_root() -> Path:
    """Return the runtime project root, honoring an explicit environment override."""
    override = os.environ.get("VULCAN_EMULATOR_PROJECT_ROOT")
    if override:
        return Path(os.path.normpath(override))
    return Path(__file__).resolve().parent.parent


def _join_project_path(root: Path, value: str, field_name: str) -> Path:
    """Join one relative config path to the project root without resolving symlinks."""
    relative = _ensure_relative(value, field_name)
    return Path(os.path.normpath(str(root / relative)))


def _ensure_relative(value: str, field_name: str) -> Path:
    """Parse one configured path and reject absolute values."""
    path = Path(value)
    if path.is_absolute():
        raise PathValidationError(
            f"Config path '{field_name}' must be relative, got absolute path: {value}"
        )
    return path


def resolve_paths(config: dict[str, Any]) -> ProjectPaths:
    """Resolve and validate all configured project-relative paths.

    All configured paths are relative to project root. Project root is derived from
    the current file location (`src/..`) and must match config path conventions.
    """
    root = _project_root()
    paths_cfg = config["paths"]

    project_root_cfg = _ensure_relative(str(paths_cfg["project_root"]), "paths.project_root")
    if project_root_cfg != Path("."):
        raise PathValidationError(
            "paths.project_root must be '.' to enforce root-relative contract."
        )

    vulcan_source = _join_project_path(
        root,
        str(paths_cfg["vulcan_source_path"]),
        "paths.vulcan_source_path",
    )
    data_root = _join_project_path(root, str(paths_cfg["data_root"]), "paths.data_root")
    models_root = _join_project_path(root, str(paths_cfg["models_root"]), "paths.models_root")
    logs_root = _join_project_path(root, str(paths_cfg["logs_root"]), "paths.logs_root")

    raw_root = data_root / "raw"
    processed_root = data_root / "processed"

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
    """Create required runtime directories."""
    for directory in (
        paths.data_root,
        paths.raw_root,
        paths.processed_root,
        paths.models_root,
        paths.logs_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)
