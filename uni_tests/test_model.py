"""Tests for model forward pass, autodiff, and training checkpoint generation."""

from __future__ import annotations

import copy
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.data_generation.data_loader import build_full_vulcan_batch, load_full_vulcan_dataset
from src.data_generation.preprocess import preprocess_raw_dataset
from src.data_generation.generation import generate_synthetic_raw_runs
from src.models.jax_model import (
    EquilibriumMLPDimensions,
    apply_equilibrium_mlp,
    apply_model,
    init_equilibrium_mlp_params,
    initialize_model,
)
from src.training.trainer import train_model


def _prepare_batch(tiny_config):
    generate_synthetic_raw_runs(tiny_config, project_root=tiny_config["_project_root"])
    preprocess_raw_dataset(tiny_config, project_root=tiny_config["_project_root"])
    splits, normalization, contract = load_full_vulcan_dataset(tiny_config["paths"]["processed_root"])
    train = splits["train"]
    indices = np.arange(min(2, train.num_runs))
    batch = build_full_vulcan_batch(train, indices)
    return batch, normalization, contract


def test_jax_forward_grad_and_jvp(tiny_config):
    batch, _, contract = _prepare_batch(tiny_config)
    dims, params = initialize_model(tiny_config, contract, seed=3)

    sequence = jnp.asarray(batch["sequence"])
    globals_ = jnp.asarray(batch["global_inputs"])
    spectrum = jnp.asarray(batch["spectrum_inputs"])

    pred, aux = apply_model(params, sequence, globals_, spectrum, dims)
    assert pred.shape == batch["target"].shape
    assert aux["spectrum_latent"].shape[-1] == tiny_config["stellar_spectrum"]["latent_dim"]

    def scalar_fn(seq):
        out, _ = apply_model(params, seq, globals_, spectrum, dims)
        return jnp.sum(out)

    grad = jax.grad(scalar_fn)(sequence)
    assert grad.shape == sequence.shape

    value, tangent = jax.jvp(scalar_fn, (sequence,), (jnp.ones_like(sequence),))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(tangent))


def test_training_checkpoint_smoke(tiny_config):
    artifacts = train_model(tiny_config, project_root=tiny_config["_project_root"])
    assert artifacts.checkpoint_path.exists()
    assert artifacts.history_path.exists()
    assert artifacts.metrics_path.exists()

    with artifacts.checkpoint_path.open("rb") as f:
        payload = pickle.load(f)
    assert "params" in payload
    assert "model_dimensions" in payload
    assert "normalization" in payload
    assert "data_contract" in payload
    assert "config" in payload


def test_training_checkpoint_smoke_with_cosine_scheduler(tiny_config):
    tiny_config["training"]["scheduler"] = {"name": "cosine"}
    artifacts = train_model(tiny_config, project_root=tiny_config["_project_root"])
    assert artifacts.checkpoint_path.exists()


@pytest.mark.parametrize(
    "activation",
    ["relu", "gelu", "silu", "tanh", "elu", "selu", "softplus", "leaky_relu"],
)
def test_supported_activations_run_forward_passes(tiny_config, activation):
    eq_dims = EquilibriumMLPDimensions(
        sequence_dim=2,
        global_dim=5,
        target_dim=3,
        d_hidden=8,
        num_hidden_layers=2,
        conditioning_hidden_dim=6,
        film_clamp=1.5,
        activation=activation,
    )
    eq_params = init_equilibrium_mlp_params(jax.random.PRNGKey(0), eq_dims)
    eq_sequence = jnp.ones((2, 4, 2), dtype=jnp.float32)
    eq_globals = jnp.ones((2, 5), dtype=jnp.float32)
    eq_pred, _ = apply_equilibrium_mlp(eq_params, eq_sequence, eq_globals, eq_dims)
    assert eq_pred.shape == (2, 4, 3)

    config = copy.deepcopy(tiny_config)
    config["training"]["model"]["activation"] = activation
    config["full_vulcan"]["model"]["activation"] = activation
    batch, _, contract = _prepare_batch(config)
    dims, params = initialize_model(config, contract, seed=3)
    pred, _ = apply_model(
        params,
        jnp.asarray(batch["sequence"]),
        jnp.asarray(batch["global_inputs"]),
        jnp.asarray(batch["spectrum_inputs"]),
        dims,
    )
    assert pred.shape == batch["target"].shape
