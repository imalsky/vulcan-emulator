"""Public inference API for VULCAN emulator exported bundles.

This is the single module that notebooks and external consumers (ExoJAX,
retrieval frameworks) should import from. Everything here is a thin
re-export of the canonical implementations in ``src/models``:

- forward pass — ``apply_transformer_model`` (``src/models/transformer.py``)
- bundle I/O + physical-unit prediction — ``ExportedModel``, ``load_model``
  (``src/models/export_bundle.py``)
- ExoJAX-ready callables — ``make_fastchem_vmr_fn``, ``make_exogibbs_vmr_fn``,
  ``make_vulcan_vmr_fn``
  (``src/models/exojax_api.py``)

Usage
-----
    from src.models.standalone_inference import load_model, make_fastchem_vmr_fn

    model = load_model("best_exported.npz")
    vmr   = model.predict_fastchem_profile(pressure_bar, temperature_k,
                                           global_inputs=global_inputs)
    vmr_fn, species = make_fastchem_vmr_fn(model)   # ExoJAX top-to-bottom API

All of ``jax.grad``, ``jax.jacfwd``, ``jax.jacrev``, ``jax.jvp``, ``jax.vmap``,
and ``jax.jit`` compose through the predict methods and the vmr functions.
Consumers must have the ``vulcan-emulator`` package on ``sys.path``; the
exported bundle is weights + JSON metadata, not a self-contained forward pass.
"""

from __future__ import annotations

from ..constants import FASTCHEM_GLOBAL_LABELS, VULCAN_GLOBAL_LABELS
from .exojax_api import make_exogibbs_vmr_fn, make_fastchem_vmr_fn, make_vulcan_vmr_fn
from .export_bundle import ExportedJAXModel as ExportedModel
from .export_bundle import load_exported_model as load_model
from .jax_model import TransformerDimensions, apply_transformer_model
from .pt_profiles import guillot_temperature

__all__ = [
    "ExportedModel",
    "FASTCHEM_GLOBAL_LABELS",
    "TransformerDimensions",
    "VULCAN_GLOBAL_LABELS",
    "apply_transformer_model",
    "guillot_temperature",
    "load_model",
    "make_fastchem_vmr_fn",
    "make_exogibbs_vmr_fn",
    "make_vulcan_vmr_fn",
]
