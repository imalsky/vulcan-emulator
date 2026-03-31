from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from src.data_generation.preprocess import apply_block, apply_mixed_block, inverse_block
from src.models.export_bundle import (
    export_checkpoint_payload,
    load_exported_model,
)
from src.models.jax_model import (
    EquilibriumMLPDimensions,
    ModelDimensions,
    apply_equilibrium_mlp,
    apply_model,
    init_equilibrium_mlp_params,
    init_model_params,
)

ELEMENT_ORDER = ["He_H", "C_H", "O_H", "N_H", "S_H"]


def _sequence_static_numpy(static_inputs: np.ndarray, blocks: list[dict]) -> np.ndarray:
    """Apply per-column sequence normalization with the repository contract."""
    return np.concatenate(
        [apply_block(static_inputs[:, idx : idx + 1], block) for idx, block in enumerate(blocks)],
        axis=-1,
    )


def test_exported_equilibrium_bundle_predicts_from_physical_inputs(tmp_path):
    dims = EquilibriumMLPDimensions(
        sequence_dim=2,
        global_dim=len(ELEMENT_ORDER),
        target_dim=2,
        d_hidden=8,
        num_hidden_layers=2,
        conditioning_hidden_dim=6,
        film_clamp=1.5,
        activation="silu",
    )
    params = init_equilibrium_mlp_params(jax.random.PRNGKey(0), dims)
    normalization = {
        "sequence_static": {
            "feature_order": ["pressure_bar", "temperature_k"],
            "blocks": [
                {"method": "log-standard", "mean": [-2.0], "std": [1.5], "floor": 1.0e-30},
                {"method": "standard", "mean": [1100.0], "std": [250.0]},
            ],
        },
        "global_static": {
            "method": "mixed",
            "methods": ["log-standard"] * len(ELEMENT_ORDER),
            "mean": [-1.1, -3.5, -3.3, -4.2, -4.8],
            "std": [0.1, 0.2, 0.2, 0.2, 0.2],
            "floor": [1.0e-30] * len(ELEMENT_ORDER),
        },
        "target": {
            "method": "log-standard",
            "mean": [-5.0, -6.0],
            "std": [1.2, 0.8],
            "floor": 1.0e-30,
        },
    }
    contract = {
        "model_type": "equilibrium",
        "global_static_feature_order": list(ELEMENT_ORDER),
        "element_input_order": list(ELEMENT_ORDER),
        "output_species_order": ["H2O", "CO"],
    }
    payload = {
        "params": jax.tree_util.tree_map(np.asarray, params),
        "model_dimensions": dims.to_dict(),
        "normalization": normalization,
        "data_contract": contract,
        "config": {"task": {"kind": "equilibrium_only"}},
    }
    bundle_path = export_checkpoint_payload(payload, tmp_path / "equilibrium_export.npz")
    bundle = load_exported_model(bundle_path)

    pressure_bar = np.array([100.0, 10.0, 1.0, 0.1], dtype=np.float32)
    temperature_k = np.array([1450.0, 1300.0, 1050.0, 900.0], dtype=np.float32)
    global_inputs = {
        "He_H": 8.38e-2,
        "C_H": 3.40e-4,
        "O_H": 5.10e-4,
        "N_H": 8.50e-5,
        "S_H": 1.60e-5,
    }

    static_inputs = np.stack([pressure_bar, temperature_k], axis=-1)
    sequence_norm = _sequence_static_numpy(static_inputs, normalization["sequence_static"]["blocks"])
    globals_norm = apply_mixed_block(
        np.array([[global_inputs[name] for name in ELEMENT_ORDER]], dtype=np.float64),
        normalization["global_static"],
    )
    expected_norm, _ = apply_equilibrium_mlp(
        params,
        jnp.asarray(sequence_norm[None, :, :], dtype=jnp.float32),
        jnp.asarray(globals_norm, dtype=jnp.float32),
        dims,
    )
    expected_physical = inverse_block(np.asarray(expected_norm[0]), normalization["target"])
    expected_log10 = (
        np.asarray(expected_norm[0], dtype=np.float64)
        * np.asarray(normalization["target"]["std"], dtype=np.float64)
        + np.asarray(normalization["target"]["mean"], dtype=np.float64)
    )

    predicted_physical = bundle.predict_equilibrium_profile(
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        global_inputs=global_inputs,
    )
    predicted_log10 = bundle.predict_equilibrium_profile(
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        global_inputs=global_inputs,
        return_log10=True,
    )

    np.testing.assert_allclose(np.asarray(predicted_physical), expected_physical, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(predicted_log10), expected_log10, rtol=1.0e-5, atol=1.0e-6)


def test_exported_transition_bundle_predicts_from_physical_inputs(tmp_path):
    dims = ModelDimensions(
        sequence_dim=5,
        global_dim=len(ELEMENT_ORDER) + 2,
        spectrum_dim=4,
        target_dim=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        conditioning_hidden_dim=8,
        film_clamp=1.5,
        output_head_divisor=2,
        spectrum_latent_dim=2,
        spectrum_hidden_dim=4,
        spectrum_encoder_mode="linear",
        activation="gelu",
    )
    params = init_model_params(jax.random.PRNGKey(1), dims)
    normalization = {
        "sequence_static": {
            "feature_order": ["pressure_bar", "temperature_k", "kzz_cm2_s"],
            "blocks": [
                {"method": "log-standard", "mean": [-2.0], "std": [1.8], "floor": 1.0e-30},
                {"method": "standard", "mean": [1200.0], "std": [300.0]},
                {"method": "log-standard", "mean": [8.0], "std": [0.5], "floor": 1.0e-30},
            ],
        },
        "state": {
            "method": "log-standard",
            "mean": [-4.0, -5.0],
            "std": [0.8, 1.1],
            "floor": 1.0e-30,
        },
        "target": {
            "method": "log-standard",
            "mean": [-6.0, -7.0],
            "std": [1.0, 0.7],
            "floor": 1.0e-30,
        },
        "global_static": {
            "method": "mixed",
            "methods": ["log-standard"] * (len(ELEMENT_ORDER) + 1),
            "mean": [3.0, -1.1, -3.5, -3.3, -4.2, -4.8],
            "std": [0.2, 0.1, 0.2, 0.2, 0.2, 0.2],
            "floor": [1.0e-30] * (len(ELEMENT_ORDER) + 1),
        },
        "log10_dt_s": {
            "method": "standard",
            "mean": [3.0],
            "std": [0.5],
        },
        "spectrum": {
            "method": "log-standard",
            "mean": [1.0, 1.2, 1.1, 0.9],
            "std": [0.3, 0.25, 0.35, 0.4],
            "floor": 1.0e-30,
        },
    }
    contract = {
        "target_mode": "trajectory",
        "global_static_feature_order": ["gravity_cm_s2", *ELEMENT_ORDER],
        "dt_feature_index": len(ELEMENT_ORDER) + 1,
        "spectrum_dim": 4,
        "element_input_order": list(ELEMENT_ORDER),
        "output_species_order": ["H2O", "CO"],
    }
    payload = {
        "params": jax.tree_util.tree_map(np.asarray, params),
        "model_dimensions": dims.to_dict(),
        "normalization": normalization,
        "data_contract": contract,
        "config": {"task": {"kind": "full_vulcan"}},
    }
    bundle_path = export_checkpoint_payload(payload, tmp_path / "transition_export.npz")
    bundle = load_exported_model(bundle_path)

    pressure_bar = np.array([100.0, 10.0, 1.0], dtype=np.float32)
    temperature_k = np.array([1500.0, 1200.0, 950.0], dtype=np.float32)
    kzz_cm2_s = np.array([1.0e8, 1.0e8, 1.0e8], dtype=np.float32)
    anchor_state = np.array(
        [
            [1.0e-4, 2.0e-6],
            [8.0e-5, 1.5e-6],
            [5.0e-5, 1.0e-6],
        ],
        dtype=np.float32,
    )
    global_inputs = {
        "gravity_cm_s2": 900.0,
        "He_H": 8.38e-2,
        "C_H": 3.40e-4,
        "O_H": 5.10e-4,
        "N_H": 8.50e-5,
        "S_H": 1.60e-5,
    }
    spectrum_flux = np.array([15.0, 20.0, 18.0, 12.0], dtype=np.float32)
    dt_s = 1.0e4

    static_inputs = np.stack([pressure_bar, temperature_k, kzz_cm2_s], axis=-1)
    static_norm = _sequence_static_numpy(static_inputs, normalization["sequence_static"]["blocks"])
    state_norm = apply_block(anchor_state, normalization["state"])
    sequence = np.concatenate([static_norm, state_norm], axis=-1)
    globals_static = apply_mixed_block(
        np.array(
            [[global_inputs["gravity_cm_s2"], *[global_inputs[name] for name in ELEMENT_ORDER]]],
            dtype=np.float64,
        ),
        normalization["global_static"],
    )[0]
    dt_feature = (
        (np.log10(dt_s) - float(normalization["log10_dt_s"]["mean"][0]))
        / float(normalization["log10_dt_s"]["std"][0])
    )
    dt_feature_index = int(contract["dt_feature_index"])
    globals_full = np.concatenate(
        [
            globals_static[:dt_feature_index],
            np.array([dt_feature]),
            globals_static[dt_feature_index:],
        ],
        axis=0,
    )
    spectrum_norm = apply_block(spectrum_flux[None, :], normalization["spectrum"])
    expected_norm, _ = apply_model(
        params,
        jnp.asarray(sequence[None, :, :], dtype=jnp.float32),
        jnp.asarray(globals_full[None, :], dtype=jnp.float32),
        jnp.asarray(spectrum_norm, dtype=jnp.float32),
        dims,
    )
    expected_physical = inverse_block(np.asarray(expected_norm[0]), normalization["target"])
    expected_log10 = (
        np.asarray(expected_norm[0], dtype=np.float64)
        * np.asarray(normalization["target"]["std"], dtype=np.float64)
        + np.asarray(normalization["target"]["mean"], dtype=np.float64)
    )

    predicted_physical = bundle.predict_transition_profile(
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        kzz_cm2_s=kzz_cm2_s,
        anchor_state=anchor_state,
        global_inputs=global_inputs,
        spectrum_flux=spectrum_flux,
        dt_s=dt_s,
    )
    predicted_log10 = bundle.predict_transition_profile(
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        kzz_cm2_s=kzz_cm2_s,
        anchor_state=anchor_state,
        global_inputs=global_inputs,
        spectrum_flux=spectrum_flux,
        dt_s=dt_s,
        return_log10=True,
    )

    np.testing.assert_allclose(np.asarray(predicted_physical), expected_physical, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(predicted_log10), expected_log10, rtol=1.0e-5, atol=1.0e-6)
    assert bundle.export_format == "jax_physical_bundle"
    assert bundle.export_version == 1
