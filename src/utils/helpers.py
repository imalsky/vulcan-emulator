"""Shared helpers for logging, path resolution, and directory creation.

The emulator uses these helpers to keep filesystem and logging concerns
out of the data-generation, preprocessing, and training modules.
"""

from __future__ import annotations

import logging
import logging.config
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
LIVE_LOG_ENV = "VULCAN_LIVE_LOG_PATH"
_PROJECT_ROOT_ENV = "VULCAN_PROJECT_ROOT"
_LOGGING_CONFIGURED = False


def _resolve_live_log_path() -> Path | None:
    """Return the resolved live-log path when the launcher requested one."""
    live_log_path = os.environ.get(LIVE_LOG_ENV, "").strip()
    if not live_log_path:
        return None
    resolved_path = Path(live_log_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_path


def _configure_logging() -> None:
    """One-shot project logging setup via :mod:`logging.config`.

    Idempotent: subsequent calls return immediately. Flushes stdout/stderr
    line-buffered, attaches an INFO-level stream handler on the root logger,
    optionally attaches a file handler at ``$VULCAN_LIVE_LOG_PATH``, and
    mutes noisy third-party device-discovery loggers.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True, write_through=True)

    handlers: dict[str, dict[str, object]] = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
            "level": "INFO",
        },
    }
    root_handlers = ["console"]
    live_path = _resolve_live_log_path()
    if live_path is not None:
        handlers["live_file"] = {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": str(live_path),
            "mode": "a",
            "encoding": "utf-8",
            "level": "INFO",
        }
        root_handlers.append("live_file")

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": {"format": _LOG_FORMAT}},
        "handlers": handlers,
        "loggers": {name: {"level": "WARNING"} for name in _NOISY_LOGGER_NAMES},
        "root": {"level": "INFO", "handlers": root_handlers},
    })
    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger after applying the project's shared logging setup."""
    _configure_logging()
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

    Looks for any supported pair of co-located root indicators
    (``PROJECT_MARKERS``): either ``pyproject.toml`` + ``src/`` or
    ``pyproject.toml`` + ``spec.md``. Both pairs are first-class; the
    ``spec.md`` pair lets detached source trees without a top-level ``src``
    directory still identify the project root. When the runtime exports
    ``VULCAN_PROJECT_ROOT``, that location is trusted first so batch
    launchers can pin the working tree explicitly. When ``start`` points
    into a detached source tree, the current working directory is used as a
    fallback search origin. Raises ``FileNotFoundError`` if neither location
    contains a supported marker set.

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
    configured_root = os.environ.get(_PROJECT_ROOT_ENV, "").strip()
    if configured_root:
        candidate = Path(configured_root).expanduser().resolve()
        if (candidate / "src").is_dir():
            return candidate

    start_paths = [start] if start is not None else []
    start_paths.append(Path.cwd())

    seen: set[Path] = set()
    for origin in start_paths:
        cursor = origin.resolve()
        if cursor.is_file():
            cursor = cursor.parent
        if cursor in seen:
            continue
        seen.add(cursor)
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
