"""JAX model definitions for the FastChem and VULCAN emulators.

Implements a FiLM-conditioned Transformer architecture:

    1. Project global_inputs to per-layer FiLM parameters (gamma, beta).
    2. Project per-level sequence to d_model + sinusoidal positional encoding.
    3. Per block:
       a. Pre-norm (ln1) -> multi-head self-attention -> dropout -> residual add
       b. FiLM: x = x * (1 + gamma) + beta
       c. Pre-norm (ln_ffn) -> FFN (up-project, act, dropout, down-project) -> residual add
    4. Output head: LayerNorm -> act -> bottleneck -> final projection.

All operations are pure JAX and compatible with ``jax.grad`` / ``jax.jvp``.

This module is a convenience aggregation point.  The actual implementations
live in :mod:`~.layers` and :mod:`~.transformer`.
"""

from __future__ import annotations

from typing import Any

import jax

# Re-export shared layers so existing ``from .jax_model import ...`` keeps working.
from .layers import (  # noqa: F401
    _apply_dropout,
    _init_layer_norm,
    _init_linear,
    _layer_norm,
    _linear,
    _multihead_attention,
    _multihead_attention_qkv,
    _resolve_activation,
    sinusoidal_position_encoding,
)

# Re-export Transformer architecture.
from .transformer import (  # noqa: F401
    TransformerDimensions,
    apply_transformer_model,
    init_transformer_params,
)


# ---------------------------------------------------------------------------
# Model construction helpers
# ---------------------------------------------------------------------------


def build_model_dimensions(
    config: dict, contract: dict
) -> TransformerDimensions:
    """Build the dimension dataclass for the active chemistry/model contract."""
    try:
        model_cfg = config["training"]["model"]
    except KeyError as exc:
        raise KeyError(
            "config['training']['model'] is missing; ensure the config was passed "
            "through load_and_validate_config(), which materializes this alias."
        ) from exc
    sequence_dim = int(
        contract.get(
            "sequence_dim",
            len(contract.get("sequence_static_feature_order", [])),
        )
    )
    global_dim = int(
        contract.get(
            "global_dim",
            len(contract.get("global_static_feature_order", [])),
        )
    )
    target_dim = int(
        contract.get(
            "target_dim",
            len(contract.get("output_species_order", [])),
        )
    )

    return TransformerDimensions(
        sequence_dim=sequence_dim,
        global_dim=global_dim,
        target_dim=target_dim,
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        num_layers=int(model_cfg["num_layers"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        conditioning_hidden_dim=int(model_cfg["conditioning_hidden_dim"]),
        film_clamp=float(model_cfg["film_clamp"]),
        output_head_divisor=int(model_cfg["output_head_divisor"]),
        activation=str(model_cfg.get("activation", "gelu")).lower(),
        dropout_rate=float(model_cfg.get("dropout_rate", 0.0)),
    )


def initialize_model(
    config: dict, contract: dict, *, seed: int
) -> tuple[TransformerDimensions, dict]:
    """Initialize model dimensions and parameters from the validated config.

    Parameters
    ----------
    config : dict
        Validated pipeline config.
    contract : dict
        Processed-data contract defining the tensor dimensions.
    seed : int
        Random seed used for JAX parameter initialization.

    Returns
    -------
    tuple[TransformerDimensions, dict]
        Model-dimension dataclass and the initialized parameter tree.
    """
    dims = build_model_dimensions(config, contract)
    key = jax.random.PRNGKey(int(seed))
    params = init_transformer_params(key, dims)
    return dims, params


def count_parameters(params: dict) -> int:
    """Count the total number of scalar parameters in a nested JAX parameter tree.

    Parameters
    ----------
    params : dict
        Nested parameter tree whose leaves are JAX arrays.

    Returns
    -------
    int
        Total number of scalar entries across all leaves in the parameter
        tree.
    """
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(int(leaf.size) for leaf in leaves))
