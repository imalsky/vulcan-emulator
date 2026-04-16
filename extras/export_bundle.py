"""Standalone checkpoint → NPZ bundle exporter.

Minimal CLI around ``src.models.export_bundle.export_checkpoint_to_npz`` that
skips the config pipeline and takes the checkpoint path directly.

Bundled in here is a NumPy 2.x → 1.x pickle compatibility shim: training
usually runs with NumPy 2.x, but the local ``vulcan`` conda env is pinned to
1.26 by exojax. A custom ``Unpickler.find_class`` rewrites the NumPy 2.x
module paths on the fly so 1.26 can decode the checkpoint without an env
change.

Usage
-----
    python extras/export_bundle.py --checkpoint path/to/best.pt
    python extras/export_bundle.py --checkpoint best.pt --output custom.npz
"""

from __future__ import annotations

import argparse
import io
import pickle
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.models.export_bundle import export_checkpoint_payload


# NumPy 2.0 moved internals from ``numpy.core`` to ``numpy._core``. Pickle
# streams written under NumPy 2.x therefore reference ``numpy._core.*`` and
# NumPy 1.26 can't resolve them. Both module trees contain the same public
# symbols, so rewriting the path at unpickle time is safe for the objects
# stored in training checkpoints (arrays, dtypes, plain dicts, strings).
_NUMPY_MODULE_REWRITES = {
    "numpy._core": "numpy.core",
    "numpy._core.numeric": "numpy.core.numeric",
    "numpy._core.multiarray": "numpy.core.multiarray",
    "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
    "numpy._core.umath": "numpy.core.umath",
    "numpy._core._methods": "numpy.core._methods",
    "numpy._core.fromnumeric": "numpy.core.fromnumeric",
    "numpy._core.arrayprint": "numpy.core.arrayprint",
    "numpy._core.numerictypes": "numpy.core.numerictypes",
    "numpy._core.shape_base": "numpy.core.shape_base",
    "numpy._core.overrides": "numpy.core.overrides",
}


class _Numpy2CompatUnpickler(pickle.Unpickler):
    """Unpickler that redirects NumPy 2.x internal module paths onto 1.x."""

    def find_class(self, module: str, name: str) -> Any:
        module = _NUMPY_MODULE_REWRITES.get(module, module)
        return super().find_class(module, name)


def _load_checkpoint_compat(checkpoint_path: Path) -> dict[str, Any]:
    """Load a training pickle, tolerating NumPy 2.x-written checkpoints."""
    with checkpoint_path.open("rb") as handle:
        raw = handle.read()
    try:
        return _Numpy2CompatUnpickler(io.BytesIO(raw)).load()
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "") or ""
        if not missing.startswith("numpy._core"):
            raise
        raise RuntimeError(
            f"Checkpoint references {missing!r} which is not covered by the "
            "numpy 2.x → 1.x shim. Add it to _NUMPY_MODULE_REWRITES in "
            "extras/export_bundle.py."
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a training checkpoint to a portable NPZ bundle.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to the training checkpoint (e.g. models/<run>/best.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination NPZ path. Defaults to <checkpoint_stem>_exported.npz alongside the checkpoint.",
    )
    args = parser.parse_args(argv)

    if not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    payload = _load_checkpoint_compat(args.checkpoint)
    destination = args.output if args.output is not None else args.checkpoint.with_name(
        f"{args.checkpoint.stem}_exported.npz"
    )
    bundle_path = export_checkpoint_payload(payload, destination)
    print(f"wrote {bundle_path}  (numpy {np.__version__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
