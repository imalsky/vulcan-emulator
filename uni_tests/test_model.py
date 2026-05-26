"""Tests for model forward pass, autodiff, and training checkpoint generation."""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.data_generation.data_loader import build_batch, load_processed_dataset
from src.data_generation.preprocess import preprocess_raw_dataset
from src.models.jax_model import (
    TransformerDimensions,
    apply_transformer_model,
    init_transformer_params,
    initialize_model,
)
from src.training.trainer import _read_checkpoint, train_model

from synthetic_fixture import generate_synthetic_raw_runs


def _prepare_batch(tiny_config: dict) -> tuple[dict[str, np.ndarray], dict, dict]:
    generate_synthetic_raw_runs(tiny_config, project_root=tiny_config["_project_root"])
    preprocess_raw_dataset(tiny_config, project_root=tiny_config["_project_root"])
    splits, normalization, contract = load_processed_dataset(tiny_config["paths"]["processed_root"])
    train = splits["train"]
    batch = build_batch(train, np.arange(min(2, train.num_runs)))
    return batch, normalization, contract


def _fastchem_transformer_dims() -> TransformerDimensions:
    return TransformerDimensions(
        sequence_dim=2,
        global_dim=5,
        target_dim=3,
        d_model=16,
        nhead=4,
        num_layers=3,
        dim_feedforward=32,
        conditioning_hidden_dim=8,
        film_clamp=1.5,
        output_head_divisor=2,
        activation="gelu",
        dropout_rate=0.0,
        norm_type="layernorm",
        use_qk_norm=False,
        ffn_type="dense",
        zero_init_film=False,
    )


def test_transformer_forward_grad_and_jvp(tiny_config):
    batch, _, contract = _prepare_batch(tiny_config)
    dims, params = initialize_model(tiny_config, contract, seed=3)

    sequence = jnp.asarray(batch["sequence"])
    globals_ = jnp.asarray(batch["global_inputs"])
    position_coord = jnp.asarray(batch["position_coord"])
    valid_mask = jnp.asarray(batch["valid_mask"])

    pred, aux = apply_transformer_model(
        params,
        sequence,
        globals_,
        dims,
        position_coord=position_coord,
        attention_mask=valid_mask,
    )
    assert pred.shape == batch["target"].shape

    def scalar_fn(seq: jax.Array) -> jax.Array:
        out, _ = apply_transformer_model(
            params,
            seq,
            globals_,
            dims,
            position_coord=position_coord,
            attention_mask=valid_mask,
        )
        return jnp.sum(out)

    grad = jax.grad(scalar_fn)(sequence)
    assert grad.shape == sequence.shape

    value, tangent = jax.jvp(scalar_fn, (sequence,), (jnp.ones_like(sequence),))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(tangent))


def test_training_checkpoint_smoke(tiny_config):
    generate_synthetic_raw_runs(tiny_config, project_root=tiny_config["_project_root"])
    artifacts = train_model(tiny_config, project_root=tiny_config["_project_root"])
    assert artifacts.run_root.exists()
    assert artifacts.config_path.exists()
    assert artifacts.history_path.exists()
    assert artifacts.metadata_path.exists()
    assert artifacts.params_best_path.exists()
    assert artifacts.params_last_path.exists()

    payload = _read_checkpoint(artifacts.run_root, which="best")
    assert "params" in payload
    assert "model_dimensions" in payload
    assert "normalization" in payload
    assert "data_contract" in payload
    assert "config" in payload


def test_training_checkpoint_smoke_with_cosine_scheduler(tiny_config):
    tiny_config["training"]["scheduler"] = {"name": "cosine"}
    generate_synthetic_raw_runs(tiny_config, project_root=tiny_config["_project_root"])
    artifacts = train_model(tiny_config, project_root=tiny_config["_project_root"])
    assert artifacts.run_root.exists()
    assert artifacts.params_best_path.exists()


def test_dropout_is_stochastic_only_in_training_mode(tiny_config):
    config = copy.deepcopy(tiny_config)
    config["model"]["dropout_rate"] = 0.5
    batch, _, contract = _prepare_batch(config)
    dims, params = initialize_model(config, contract, seed=3)
    sequence = jnp.asarray(batch["sequence"])
    globals_ = jnp.asarray(batch["global_inputs"])
    position_coord = jnp.asarray(batch["position_coord"])
    valid_mask = jnp.asarray(batch["valid_mask"])
    fwd_kwargs = {"position_coord": position_coord, "attention_mask": valid_mask}
    eval_a, _ = apply_transformer_model(params, sequence, globals_, dims, **fwd_kwargs)
    eval_b, _ = apply_transformer_model(
        params,
        sequence,
        globals_,
        dims,
        **fwd_kwargs,
        dropout_key=jax.random.PRNGKey(4),
        training=False,
    )
    train_a, _ = apply_transformer_model(
        params,
        sequence,
        globals_,
        dims,
        **fwd_kwargs,
        dropout_key=jax.random.PRNGKey(5),
        training=True,
    )
    train_b, _ = apply_transformer_model(
        params,
        sequence,
        globals_,
        dims,
        **fwd_kwargs,
        dropout_key=jax.random.PRNGKey(6),
        training=True,
    )
    assert np.allclose(np.asarray(eval_a), np.asarray(eval_b))
    assert not np.allclose(np.asarray(train_a), np.asarray(train_b))


@pytest.mark.parametrize(
    "activation",
    ["relu", "gelu", "silu", "tanh", "elu", "selu", "softplus", "leaky_relu"],
)
def test_supported_activations_run_forward_passes(tiny_config, activation: str):
    config = copy.deepcopy(tiny_config)
    config["model"]["activation"] = activation
    batch, _, contract = _prepare_batch(config)
    dims, params = initialize_model(config, contract, seed=3)
    pred, _ = apply_transformer_model(
        params,
        jnp.asarray(batch["sequence"]),
        jnp.asarray(batch["global_inputs"]),
        dims,
        position_coord=jnp.asarray(batch["position_coord"]),
        attention_mask=jnp.asarray(batch["valid_mask"]),
    )
    assert pred.shape == batch["target"].shape


def test_transformer_dimensions_round_trip():
    dims = TransformerDimensions(
        sequence_dim=3,
        global_dim=21,
        target_dim=4,
        d_model=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=32,
        conditioning_hidden_dim=12,
        film_clamp=2.0,
        output_head_divisor=2,
        activation="gelu",
        dropout_rate=0.1,
        norm_type="layernorm",
        use_qk_norm=False,
        ffn_type="dense",
        zero_init_film=False,
    )
    restored = TransformerDimensions.from_dict(dims.to_dict())
    assert restored == dims


def test_transformer_params_contain_ln_ffn_at_every_layer():
    dims = _fastchem_transformer_dims()
    params = init_transformer_params(jax.random.PRNGKey(0), dims)
    for layer in params["layers"]:
        assert "ln_ffn" in layer
        assert layer["ln_ffn"]["scale"].shape == (16,)
        assert layer["ln_ffn"]["bias"].shape == (16,)
