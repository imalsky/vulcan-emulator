from __future__ import annotations

import numpy as np


def patch_numpy_asarray_copy() -> None:
    """Allow code paths that pass ``copy=...`` to ``np.asarray`` on older NumPy."""
    try:
        np.asarray(0, copy=None)
        return
    except TypeError:
        pass

    original_asarray = np.asarray

    def _compat_asarray(a, dtype=None, order=None, *, like=None, copy=None):
        del copy
        kwargs = {}
        if order is not None:
            kwargs["order"] = order
        if like is not None:
            kwargs["like"] = like
        return original_asarray(a, dtype=dtype, **kwargs)

    np.asarray = _compat_asarray
