"""Tests for model forward pass, autodiff, and training checkpoint generation."""

from __future__ import annotations

import copy
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.data_generation.data_loader import build_batch, load_processed_dataset
from src.data_generation.generation import generate_synthetic_raw_runs
from src.data_generation.preprocess import preprocess_raw_dataset
from src.models.jax_model import (
    MLPDimensions,
    TransformerDimensions,
    apply_mlp,
    apply_transformer_model,
    init_mlp_params,
    init_transformer_params,
    initialize_model,
)
from src.training.trainer import train_model


def _prepare_batch(tiny_config: dict) -> tuple[dict[str, np.ndarray], dict, dict]:
    generate_synthetic_raw_runs(tiny_config, project_root=tiny_config["_project_root"])
    preprocess_raw_dataset(tiny_config, project_root=tiny_config["_project_root"])
    splits, normalization, contract = load_processed_dataset(tiny_config["paths"]["processed_root"])
    train = splits["train"]
    indices = np.arange(min(2, train.num_runs))
    batch = build_batch(train, indices)
    return batch, normalization, contract


def test_transformer_forward_grad_and_jvp(tiny_config):
    batch, _, contract = _prepare_batch(tiny_config)
    dims, params = initialize_model(tiny_config, contract, seed=3)

    sequence = jnp.asarray(batch["sequence"])
    globals_ = jnp.asarray(batch["global_inputs"])
    spectrum = jnp.asarray(batch["spectrum_inputs"])

    pred, aux = apply_transformer_model(params, sequence, globals_, spectrum, dims)
    assert pred.shape == batch["target"].shape
    assert aux["spectrum_latent"].shape[-1] == tiny_config["stellar_spectrum"]["latent_dim"]

    def scalar_fn(seq: jax.Array) -> jax.Array:
        out, _ = apply_transformer_model(params, seq, globals_, spectrum, dims)
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


def test_dropout_is_stochastic_only_in_training_mode(tiny_config):
    fastchem_dims = MLPDimensions(
        sequence_dim=2,
        global_dim=5,
        spectrum_dim=0,
        spectrum_latent_dim=0,
        spectrum_hidden_dim=0,
        spectrum_encoder_mode="none",
        target_dim=3,
        d_hidden=16,
        num_hidden_layers=2,
        conditioning_hidden_dim=8,
        film_clamp=1.5,
        activation="leaky_relu",
        dropout_rate=0.5,
    )
    fastchem_params = init_mlp_params(jax.random.PRNGKey(0), fastchem_dims)
    fastchem_sequence = jnp.ones((2, 6, 2), dtype=jnp.float32)
    fastchem_globals = jnp.ones((2, 5), dtype=jnp.float32)
    eval_a, _ = apply_mlp(fastchem_params, fastchem_sequence, fastchem_globals, fastchem_dims)
    eval_b, _ = apply_mlp(
        fastchem_params,
        fastchem_sequence,
        fastchem_globals,
        fastchem_dims,
        dropout_key=jax.random.PRNGKey(1),
        training=False,
    )
    train_a, _ = apply_mlp(
        fastchem_params,
        fastchem_sequence,
        fastchem_globals,
        fastchem_dims,
        dropout_key=jax.random.PRNGKey(2),
        training=True,
    )
    train_b, _ = apply_mlp(
        fastchem_params,
        fastchem_sequence,
        fastchem_globals,
        fastchem_dims,
        dropout_key=jax.random.PRNGKey(3),
        training=True,
    )
    assert np.allclose(np.asarray(eval_a), np.asarray(eval_b))
    assert not np.allclose(np.asarray(train_a), np.asarray(train_b))

    config = copy.deepcopy(tiny_config)
    config["training"]["model"]["dropout_rate"] = 0.5
    config["model"]["dropout_rate"] = 0.5
    batch, _, contract = _prepare_batch(config)
    dims, params = initialize_model(config, contract, seed=3)
    sequence = jnp.asarray(batch["sequence"])
    globals_ = jnp.asarray(batch["global_inputs"])
    spectrum = jnp.asarray(batch["spectrum_inputs"])
    eval_vulcan_a, _ = apply_transformer_model(params, sequence, globals_, spectrum, dims)
    eval_vulcan_b, _ = apply_transformer_model(
        params,
        sequence,
        globals_,
        spectrum,
        dims,
        dropout_key=jax.random.PRNGKey(4),
        training=False,
    )
    train_vulcan_a, _ = apply_transformer_model(
        params,
        sequence,
        globals_,
        spectrum,
        dims,
        dropout_key=jax.random.PRNGKey(5),
        training=True,
    )
    train_vulcan_b, _ = apply_transformer_model(
        params,
        sequence,
        globals_,
        spectrum,
        dims,
        dropout_key=jax.random.PRNGKey(6),
        training=True,
    )
    assert np.allclose(np.asarray(eval_vulcan_a), np.asarray(eval_vulcan_b))
    assert not np.allclose(np.asarray(train_vulcan_a), np.asarray(train_vulcan_b))


@pytest.mark.parametrize(
    "activation",
    ["relu", "gelu", "silu", "tanh", "elu", "selu", "softplus", "leaky_relu"],
)
def test_supported_activations_run_forward_passes(tiny_config, activation: str):
    fastchem_dims = MLPDimensions(
        sequence_dim=2,
        global_dim=5,
        spectrum_dim=0,
        spectrum_latent_dim=0,
        spectrum_hidden_dim=0,
        spectrum_encoder_mode="none",
        target_dim=3,
        d_hidden=8,
        num_hidden_layers=2,
        conditioning_hidden_dim=6,
        film_clamp=1.5,
        activation=activation,
    )
    fastchem_params = init_mlp_params(jax.random.PRNGKey(0), fastchem_dims)
    fastchem_pred, _ = apply_mlp(
        fastchem_params,
        jnp.ones((2, 4, 2), dtype=jnp.float32),
        jnp.ones((2, 5), dtype=jnp.float32),
        fastchem_dims,
    )
    assert fastchem_pred.shape == (2, 4, 3)

    config = copy.deepcopy(tiny_config)
    config["training"]["model"]["activation"] = activation
    config["model"]["activation"] = activation
    batch, _, contract = _prepare_batch(config)
    dims, params = initialize_model(config, contract, seed=3)
    pred, _ = apply_transformer_model(
        params,
        jnp.asarray(batch["sequence"]),
        jnp.asarray(batch["global_inputs"]),
        jnp.asarray(batch["spectrum_inputs"]),
        dims,
    )
    assert pred.shape == batch["target"].shape


def test_transformer_dimensions_round_trip():
    dims = TransformerDimensions(
        sequence_dim=3,
        global_dim=21,
        spectrum_dim=8,
        target_dim=4,
        d_model=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=32,
        conditioning_hidden_dim=12,
        film_clamp=2.0,
        output_head_divisor=2,
        spectrum_latent_dim=4,
        spectrum_hidden_dim=8,
        spectrum_encoder_mode="linear",
        activation="gelu",
        dropout_rate=0.1,
    )
    restored = TransformerDimensions.from_dict(dims.to_dict())
    assert restored == dims


def test_mlp_dimensions_round_trip():
    dims = MLPDimensions(
        sequence_dim=2,
        global_dim=5,
        spectrum_dim=0,
        spectrum_latent_dim=0,
        spectrum_hidden_dim=0,
        spectrum_encoder_mode="none",
        target_dim=3,
        d_hidden=16,
        num_hidden_layers=3,
        conditioning_hidden_dim=8,
        film_clamp=1.5,
        activation="gelu",
        dropout_rate=0.1,
    )
    restored = MLPDimensions.from_dict(dims.to_dict())
    assert restored == dims


def test_mlp_params_contain_layer_norm_at_every_layer():
    dims = MLPDimensions(
        sequence_dim=2,
        global_dim=5,
        spectrum_dim=0,
        spectrum_latent_dim=0,
        spectrum_hidden_dim=0,
        spectrum_encoder_mode="none",
        target_dim=3,
        d_hidden=16,
        num_hidden_layers=4,
        conditioning_hidden_dim=8,
        film_clamp=1.5,
        activation="gelu",
    )
    params = init_mlp_params(jax.random.PRNGKey(0), dims)
    for i, layer in enumerate(params["layers"]):
        assert "ln" in layer, f"Layer {i} missing 'ln' (LayerNorm params)"
        assert layer["ln"]["scale"].shape == (16,)
        assert layer["ln"]["bias"].shape == (16,)


def test_transformer_params_contain_ln_film_at_every_layer():
    dims = TransformerDimensions(
        sequence_dim=2,
        global_dim=5,
        spectrum_dim=0,
        target_dim=3,
        d_model=16,
        nhead=4,
        num_layers=3,
        dim_feedforward=32,
        conditioning_hidden_dim=8,
        film_clamp=1.5,
        output_head_divisor=2,
        spectrum_latent_dim=0,
        spectrum_hidden_dim=0,
        spectrum_encoder_mode="none",
        activation="gelu",
    )
    params = init_transformer_params(jax.random.PRNGKey(0), dims)
    for i, layer in enumerate(params["layers"]):
        assert "ln_film" in layer, f"Layer {i} missing 'ln_film' (post-FiLM LayerNorm)"
        assert layer["ln_film"]["scale"].shape == (16,)
        assert layer["ln_film"]["bias"].shape == (16,)


def test_mlp_residual_connections_enable_gradient_flow():
    """Verify that a deep MLP with residual connections has non-vanishing gradients."""
    dims = MLPDimensions(
        sequence_dim=2,
        global_dim=5,
        spectrum_dim=0,
        spectrum_latent_dim=0,
        spectrum_hidden_dim=0,
        spectrum_encoder_mode="none",
        target_dim=3,
        d_hidden=16,
        num_hidden_layers=6,
        conditioning_hidden_dim=8,
        film_clamp=1.5,
        activation="gelu",
    )
    params = init_mlp_params(jax.random.PRNGKey(0), dims)
    sequence = jnp.ones((2, 4, 2), dtype=jnp.float32)
    globals_ = jnp.ones((2, 5), dtype=jnp.float32)

    def scalar_fn(seq):
        pred, _ = apply_mlp(params, seq, globals_, dims)
        return jnp.sum(pred)

    grad = jax.grad(scalar_fn)(sequence)
    grad_norm = float(jnp.sqrt(jnp.sum(grad ** 2)))
    assert grad_norm > 1e-6, f"Gradient norm too small ({grad_norm}), residual connections may not be working"
