"""Model definitions and initialization.

Sub-modules
-----------
layers
    Shared neural-network primitives (linear, LayerNorm, dropout, spectrum encoder).
mlp
    FiLM-conditioned MLP architecture.
transformer
    FiLM-conditioned Transformer architecture.
jax_model
    Backward-compatible facade re-exporting both architectures plus
    model construction helpers (``build_model_dimensions``, ``initialize_model``).
standalone_inference
    Self-contained inference module embedded in exported bundles.
export_bundle
    Bundle export utilities and physical-units inference wrapper.
exojax_api
    ExoJAX integration wrappers.
"""

from __future__ import annotations
