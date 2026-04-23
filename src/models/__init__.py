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
    Backward-compatibility shim that re-exports ``load_model``,
    ``ExportedModel``, and the ExoJAX wrapper factories under their
    pre-consolidation names. New code should import from ``export_bundle``
    or ``exojax_api`` directly.
export_bundle
    Bundle export utilities and physical-units inference wrapper.
exojax_api
    ExoJAX integration wrappers.
"""

from __future__ import annotations
