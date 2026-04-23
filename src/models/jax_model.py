"""Model construction helpers: build dims / init params / count params.

The Transformer architecture itself lives in :mod:`~.transformer`; the
shared numeric primitives (linear, LayerNorm, RMSNorm, attention) live in
:mod:`~.layers`. This module only wraps ``config + contract`` into a
``TransformerDimensions`` dataclass and allocates its parameter tree.
"""

from __future__ import annotations

from typing import Any

import jax

from .transformer import (
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
        activation=str(model_cfg["activation"]).lower(),
        dropout_rate=float(model_cfg["dropout_rate"]),
        norm_type=str(model_cfg["norm_type"]).lower(),
        use_qk_norm=bool(model_cfg["use_qk_norm"]),
        ffn_type=str(model_cfg["ffn_type"]).lower(),
        zero_init_film=bool(model_cfg["zero_init_film"]),
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
