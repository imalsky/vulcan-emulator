from __future__ import annotations

"""CLI, configuration, and shared utilities."""

from .numpy_compat import patch_numpy_asarray_copy as _patch
_patch()
del _patch

def main(argv: list[str] | None = None) -> int:
    """Lazily import and dispatch the CLI entry point."""
    from .cli import main as _main
    return _main(argv)

__all__ = ["main"]
