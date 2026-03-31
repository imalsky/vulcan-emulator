"""Shared helpers for logging, path resolution, and directory creation.

The emulator uses these helpers to keep filesystem and logging concerns
out of the data-generation, preprocessing, and training modules.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys


PROJECT_MARKERS = ("pyproject.toml", "spec.md")


_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_NOISY_LOGGER_NAMES = (
    "absl",
    "jax",
    "jax._src.xla_bridge",
    "jaxlib",
)
_LIVE_LOG_ENV = "VULCAN_LIVE_LOG_PATH"


def _configure_standard_streams() -> None:
    """Force line-buffered standard streams when the runtime supports it."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True, write_through=True)


def _configure_live_file_logging(root_logger: logging.Logger) -> None:
    """Attach one live file handler when requested by the launcher environment."""
    live_log_path = os.environ.get(_LIVE_LOG_ENV, "").strip()
    if not live_log_path:
        return
    resolved_path = Path(live_log_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            base_filename = getattr(handler, "baseFilename", "")
            if base_filename and Path(base_filename).resolve() == resolved_path:
                return
    file_handler = logging.FileHandler(resolved_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root_logger.addHandler(file_handler)


def _configure_console_logging() -> None:
    """Configure console logging once and suppress noisy backend discovery logs."""
    _configure_standard_streams()
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, stream=sys.stdout)
    else:
        root_logger.setLevel(logging.INFO)
    _configure_live_file_logging(root_logger)
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
