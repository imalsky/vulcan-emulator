from __future__ import annotations

from dataclasses import asdict

import jax

from .jax_model import ModelDimensions, init_model_params


def build_model_dimensions(config: dict, contract: dict) -> ModelDimensions:
    model_cfg = config["training"]["model"]
    spectrum_cfg = config["stellar_spectrum"]
    return ModelDimensions(
        sequence_dim=int(contract["sequence_dim"]),
        global_dim=len(contract["global_feature_order"]),
        spectrum_dim=int(contract["spectrum_dim"]),
        target_dim=int(contract["target_dim"]),
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        num_layers=int(model_cfg["num_layers"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        conditioning_hidden_dim=int(model_cfg["conditioning_hidden_dim"]),
        film_clamp=float(model_cfg["film_clamp"]),
        output_head_divisor=int(model_cfg["output_head_divisor"]),
        spectrum_latent_dim=int(spectrum_cfg["latent_dim"]),
        spectrum_hidden_dim=int(spectrum_cfg["hidden_dim"]),
        spectrum_encoder_mode=str(spectrum_cfg["encoder_mode"]).lower(),
    )


def initialize_model(config: dict, contract: dict, *, seed: int) -> tuple[ModelDimensions, dict]:
    dims = build_model_dimensions(config, contract)
    params = init_model_params(jax.random.PRNGKey(int(seed)), dims)
    return dims, params


def count_parameters(params: dict) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(int(leaf.size) for leaf in leaves))
