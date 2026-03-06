"""Logging helpers for vulcan-emulator."""

from __future__ import annotations

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"


def setup_logging(log_file: Path | None = None, *, level: int = logging.INFO) -> None:
    """Configure process-wide logging.

    Args:
        log_file: Optional destination file. Parent directory must exist or be creatable.
        level: Root logger level.
    """
    root = logging.getLogger()
    while root.handlers:
        handler = root.handlers.pop()
        handler.close()

    root.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
