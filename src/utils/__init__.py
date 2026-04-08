"""CLI, configuration, and shared utilities."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """Dispatch the package CLI while keeping import side effects minimal.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument vector to forward to ``src.utils.cli.main``. When
        ``None``, the CLI reads arguments from ``sys.argv``.

    Returns
    -------
    int
        Process-style exit status returned by the CLI entry point.
    """
    from .cli import main as _main
    return _main(argv)

__all__ = ["main"]
