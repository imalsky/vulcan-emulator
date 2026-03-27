"""NumPy 2.0 compatibility: backfill the copy= kwarg on np.asarray for older releases.

This module can be removed once the project drops support for NumPy < 2.0.
At that point, remove all ``patch_numpy_asarray_copy()`` call sites in the
extras/ scripts and in ``src/utils/__init__.py``.
"""

from __future__ import annotations

import numpy as np


def patch_numpy_asarray_copy() -> None:
    """Allow code paths that pass ``copy=...`` to ``np.asarray`` on older NumPy.

    NumPy 2.0 added a ``copy`` keyword to ``np.asarray``.  JAX and
    other downstream libraries may pass it unconditionally, causing a
    ``TypeError`` on NumPy < 2.0.  This monkey-patch silently drops
    the ``copy`` kwarg on older releases so the call succeeds.
    """
    try:
        np.asarray(0, copy=None)
        return
    except TypeError:
        pass

    original_asarray = np.asarray

    def _compat_asarray(a, dtype=None, order=None, *, like=None, copy=None):
        """Backfill the NumPy 2 ``copy=...`` signature on older releases."""
        del copy
        kwargs = {}
        if order is not None:
            kwargs["order"] = order
        if like is not None:
            kwargs["like"] = like
        return original_asarray(a, dtype=dtype, **kwargs)

    np.asarray = _compat_asarray
