"""Path resolution and project layout utilities."""

from __future__ import annotations

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
    root = Path(__file__).resolve().parent.parent
    paths_cfg = config["paths"]

    project_root_cfg = _ensure_relative(str(paths_cfg["project_root"]), "paths.project_root")
    if project_root_cfg != Path("."):
        raise PathValidationError(
            "paths.project_root must be '.' to enforce root-relative contract."
        )

    vulcan_source = (
        root / _ensure_relative(str(paths_cfg["vulcan_source_path"]), "paths.vulcan_source_path")
    ).resolve()
    data_root = (root / _ensure_relative(str(paths_cfg["data_root"]), "paths.data_root")).resolve()
    models_root = (
        root / _ensure_relative(str(paths_cfg["models_root"]), "paths.models_root")
    ).resolve()
    logs_root = (root / _ensure_relative(str(paths_cfg["logs_root"]), "paths.logs_root")).resolve()

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
