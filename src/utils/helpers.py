"""Shared helpers for logging, path resolution, and directory creation.

The emulator uses these helpers to keep filesystem and logging concerns
out of the data-generation, preprocessing, and training modules.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

PROJECT_MARKERS = (
    ("pyproject.toml", "src"),
    ("pyproject.toml", "spec.md"),
)


_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_NOISY_LOGGER_NAMES = (
    "absl",
    "jax",
    "jax._src.xla_bridge",
    "jaxlib",
)
_LIVE_LOG_ENV = "VULCAN_LIVE_LOG_PATH"


def _configure_standard_streams() -> None:
    """Enable line-buffered stdout and stderr when the runtime supports it.

    Parameters
    ----------
    None
        Stream reconfiguration operates on the process-global standard streams.

    Returns
    -------
    None
        Standard streams are reconfigured in place to flush promptly during
        long-running jobs and PBS log capture.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True, write_through=True)


def _configure_live_file_logging(root_logger: logging.Logger) -> None:
    """Attach a live file handler when the launcher requests one.

    Parameters
    ----------
    root_logger : logging.Logger
        Root logger that may receive a shared file handler pointing at the
        live log path.

    Returns
    -------
    None
        The logger is updated in place when live-file logging is requested.
    """
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
    """Configure shared console logging and suppress noisy third-party logs.

    This function is idempotent across repeated logger requests and keeps the
    project's INFO-level logs visible while muting backend discovery noise.

    Parameters
    ----------
    None
        Logging is configured through process-global logger state.

    Returns
    -------
    None
        Root logging handlers and third-party logger levels are normalized in
        place.
    """
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
    """Return a logger after applying the project's shared logging setup.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` from the calling module.

    Returns
    -------
    logging.Logger
        Logger configured to share the project's console and optional live-file
        handlers.
    """
    _configure_console_logging()
    return logging.getLogger(name)


def ensure_dir(path: Path) -> Path:
    """Create a directory tree if needed and return the resolved path object.

    Parameters
    ----------
    path : Path
        Directory path that should exist after the call.

    Returns
    -------
    Path
        Same ``path`` object after ensuring the directory exists.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_project_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` until the repository root markers are found.

    Looks for a supported set of co-located root indicators. The primary
    marker set is ``pyproject.toml`` plus the top-level ``src`` directory;
    ``spec.md`` is accepted as an additional legacy marker. Raises
    ``FileNotFoundError`` if neither the start directory nor any of its
    parents contain a supported marker set.

    Parameters
    ----------
    start : Path or None, optional
        Starting filesystem path. Files are resolved to their parent
        directory; ``None`` starts from the current working directory.

    Returns
    -------
    Path
        Resolved repository root containing the configured project markers.
    """
    cursor = (start or Path.cwd()).resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for candidate in (cursor, *cursor.parents):
        for marker_group in PROJECT_MARKERS:
            if all((candidate / marker).exists() for marker in marker_group):
                return candidate
    raise FileNotFoundError("Could not resolve project root from the current path.")


def resolve_path(path_like: str | Path, project_root: Path) -> Path:
    """Resolve a config-style path relative to the project root.

    Absolute paths are returned unchanged; relative paths are joined
    to ``project_root`` and resolved.

    Parameters
    ----------
    path_like : str or Path
        Candidate path from config or caller input.
    project_root : Path
        Repository root used to resolve relative paths.

    Returns
    -------
    Path
        Absolute resolved filesystem path.
    """
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()
