"""Shared utilities: logging setup, path resolution, and directory management.

Small, dependency-free helpers used across every module in the
emulator.  Keeps I/O and logging concerns out of the domain logic.
"""

from __future__ import annotations

import logging
from pathlib import Path


PROJECT_MARKERS = ("pyproject.toml", "spec.md")


_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_file_logging(project_root: Path) -> None:
    """Add a shared file handler writing to ``logs/pipeline.log``."""
    root_logger = logging.getLogger()
    # Skip if a file handler is already attached.
    if any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        return
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a simple deterministic formatter."""
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    return logger


def ensure_dir(path: Path) -> Path:
    """Create a directory path if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_project_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` until the repository root markers are found.

    Looks for ``pyproject.toml`` and ``spec.md`` as co-located root
    indicators.  Raises ``FileNotFoundError`` if neither the start
    directory nor any of its parents contain both markers.
    """
    cursor = (start or Path.cwd()).resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for candidate in (cursor, *cursor.parents):
        if all((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    raise FileNotFoundError("Could not resolve project root from the current path.")


def resolve_path(path_like: str | Path, project_root: Path) -> Path:
    """Resolve a config-style path relative to the project root.

    Absolute paths are returned unchanged; relative paths are joined
    to ``project_root`` and resolved.
    """
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()
