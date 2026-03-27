from __future__ import annotations

"""CLI, configuration, and shared utilities."""

from .numpy_compat import patch_numpy_asarray_copy as _patch
_patch()
del _patch

from .cli import main

__all__ = ["main"]
