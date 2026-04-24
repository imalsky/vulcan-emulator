"""Model definitions and initialization.

Sub-modules
-----------
layers
    Shared neural-network primitives (linear, LayerNorm, dropout, spectrum encoder).
transformer
    FiLM-conditioned Transformer architecture.
jax_model
    Model construction helpers (``build_model_dimensions``,
    ``initialize_model``, ``count_parameters``).
standalone_inference
    Public inference API — the single module notebooks and external
    consumers (ExoJAX, retrieval frameworks) should import from. Thin
    re-export surface over ``export_bundle`` and ``exojax_api``.
export_bundle
    Bundle export utilities and physical-units inference wrapper.
exojax_api
    ExoJAX integration wrappers.
"""

from __future__ import annotations
