"""Shared helpers for logging, path resolution, and directory creation.

The emulator uses these helpers to keep filesystem and logging concerns
out of the data-generation, preprocessing, and training modules.
"""

from __future__ import annotations

import logging
from pathlib import Path


PROJECT_MARKERS = ("pyproject.toml", "spec.md")


_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_NOISY_LOGGER_NAMES = (
    "absl",
    "jax",
    "jax._src.xla_bridge",
    "jaxlib",
)


def _configure_console_logging() -> None:
    """Configure console logging once and suppress noisy backend discovery logs."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    # Keep project INFO logs while muting third-party device diagnostics.
    for logger_name in _NOISY_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger after applying the shared console configuration."""
    _configure_console_logging()
    return logging.getLogger(name)


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
