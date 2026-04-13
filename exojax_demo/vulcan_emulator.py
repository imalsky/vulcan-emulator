"""Thin standalone wrapper around the embedded emulator implementation.

The exported bundle already embeds the self-contained inference module source
under ``meta/vulcan_emulator_src``. This file provides a normal import surface
for the demo scripts so they do not repeat dynamic module-loading boilerplate.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = DEMO_DIR / "bundle" / "best_exported.npz"
_EMBEDDED_MODULE_NAME = "_embedded_vulcan_emulator"

SOLAR_ABUNDANCES = {
    "He_H": 8.38e-2,
    "C_H": 2.95e-4,
    "O_H": 5.37e-4,
    "N_H": 7.08e-5,
    "S_H": 1.41e-5,
}


def _load_embedded_module() -> types.ModuleType:
    """Load the embedded inference implementation exactly once."""
    existing = sys.modules.get(_EMBEDDED_MODULE_NAME)
    if existing is not None:
        return existing

    with np.load(BUNDLE_PATH, allow_pickle=False) as bundle:
        source = bytes(bundle["meta/vulcan_emulator_src"]).decode()

    module = types.ModuleType(_EMBEDDED_MODULE_NAME)
    module.__file__ = str(BUNDLE_PATH)
    sys.modules[_EMBEDDED_MODULE_NAME] = module
    exec(compile(source, str(BUNDLE_PATH), "exec"), module.__dict__)
    return module


_IMPL = _load_embedded_module()

ExportedModel = _IMPL.ExportedModel
FASTCHEM_GLOBAL_ORDER = list(_IMPL.FASTCHEM_GLOBAL_LABELS)
load_model = _IMPL.load_model
make_fastchem_vmr_fn = _IMPL.make_fastchem_vmr_fn

__all__ = [
    "BUNDLE_PATH",
    "DEMO_DIR",
    "ExportedModel",
    "FASTCHEM_GLOBAL_ORDER",
    "SOLAR_ABUNDANCES",
    "load_model",
    "make_fastchem_vmr_fn",
]
