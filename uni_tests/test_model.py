"""Tests for model forward pass, autodiff, and training checkpoint generation."""

from __future__ import annotations

import copy
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from src.utils.config import effective_transition_sampling
from src.data_generation.data_loader import build_batch_from_rows, load_processed_dataset
from src.data_generation.preprocess import preprocess_raw_dataset
from src.data_generation.transition_sampling import build_candidate_table
from src.data_generation.generation import generate_synthetic_raw_runs
from src.models.jax_model import apply_model, initialize_model
from src.training.trainer import train_model


def _prepare_batch(tiny_config):
    generate_synthetic_raw_runs(tiny_config, project_root=tiny_config["_project_root"])
    preprocess_raw_dataset(tiny_config, project_root=tiny_config["_project_root"])
    splits, normalization, contract = load_processed_dataset(tiny_config["paths"]["processed_root"])
    train = splits["train"]
    transition_sampling = effective_transition_sampling(tiny_config)
    candidate_table = build_candidate_table(
        train.time_s,
        train.valid_steps_mask,
        dt_min_s=float(transition_sampling["dt_min_s"]),
        dt_max_s=float(transition_sampling["dt_max_s"]),
        min_future_saved_steps=int(transition_sampling["min_future_saved_steps"]),
        log10_dt_stats={
            "mean": float(normalization["log10_dt_s"]["mean"][0]),
            "std": float(normalization["log10_dt_s"]["std"][0]),
        },
    )
    rows = candidate_table.rows_for_run(0)[:2]
    batch = build_batch_from_rows(train, candidate_table, rows)
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
