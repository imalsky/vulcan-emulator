from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from src.utils.config import load_and_validate_config
from src.models.jax_model import apply_mlp, initialize_model


def test_equilibrium_model_uses_configured_activation():
    root = Path(__file__).resolve().parents[1]
    config = load_and_validate_config(root / "config" / "fastchem_mlp_config.json")
    config["training"]["model"]["activation"] = "relu"
    config["training"]["model"]["dropout_rate"] = 0.2
    contract = {
        "sequence_dim": 2,
        "global_dim": len(config["data_spec"]["global_static_feature_order"]),
        "target_dim": len(config["data_spec"]["output_species"]),
    }
    dims, params = initialize_model(config, contract, seed=7)
    pred, aux = apply_mlp(
        params,
        jnp.ones((2, 5, 2), dtype=jnp.float32),
        jnp.ones((2, contract["global_dim"]), dtype=jnp.float32),
        dims,
    )
    assert dims.activation == "relu"
    assert dims.dropout_rate == 0.2
    assert pred.shape == (2, 5, contract["target_dim"])
    assert aux["spectrum_reconstruction"] is None
    assert aux["spectrum_latent"].shape == (2, 0)
