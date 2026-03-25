from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np

from src.config_utils import effective_transition_sampling
from src.data_loader import build_batch_from_rows, load_processed_dataset
from src.export_jax import load_export_bundle
from src.inference import load_physical_space_model
from src.jax_model import apply_model
from src.model import initialize_model
from src.preprocess import preprocess_raw_dataset
from src.trainer import train_model
from src.transition_sampling import build_candidate_table
from src.vulcan_runner import generate_synthetic_raw_runs


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


def test_training_export_and_inference_smoke(tiny_config):
    artifacts = train_model(tiny_config, project_root=tiny_config["_project_root"])
    assert artifacts.checkpoint_path.exists()
    assert artifacts.export_root.exists()

    bundle = load_export_bundle(artifacts.export_root)
    assert "params" in bundle

    model = load_physical_space_model(artifacts.export_root)
    pressure_bar = np.logspace(2.0, -7.0, tiny_config["sampling"]["num_levels"])
    temperature_k = np.linspace(900.0, 1200.0, tiny_config["sampling"]["num_levels"])
    kzz = np.full(tiny_config["sampling"]["num_levels"], 1.0e8)
    rng = np.random.default_rng(0)
    ymix = rng.uniform(1.0e-12, 1.0e-3, size=(tiny_config["sampling"]["num_levels"], len(tiny_config["data_spec"]["state_species"])))
    ymix /= ymix.sum(axis=1, keepdims=True)
    global_inputs = {
        name: 0.0 for name in model.contract["global_static_feature_order"]
    }
    global_inputs["gravity_cm_s2"] = 1000.0
    global_inputs["metallicity_log10"] = 0.0
    global_inputs["c_to_o"] = 0.55
    global_inputs["use_photochemistry"] = 1.0
    global_inputs["use_eddy_diffusion"] = 1.0
    global_inputs["atm_base_H2"] = 1.0
    spectrum = np.linspace(1.0, 2.0, tiny_config["stellar_spectrum"]["num_bins"], dtype=np.float32)
    pred = model.predict(
        pressure_bar=pressure_bar,
        temperature_K=temperature_k,
        eddy_diffusion_cm2_s=kzz,
        ymix_state=ymix,
        global_inputs=global_inputs,
        log10_dt_s=3.0,
        spectrum_inputs=spectrum,
    )
    assert pred.shape == ymix.shape
    assert np.all(np.isfinite(pred))
