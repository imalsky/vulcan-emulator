from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.data_generation.preprocess import apply_block, apply_mixed_block, inverse_block
from src.data_generation.spectrum import pack_spectrum_tokens
from src.models.export_bundle import export_checkpoint_payload, load_exported_model
from src.models.jax_model import (
    MLPDimensions,
    TransformerDimensions,
    apply_mlp,
    apply_transformer_model,
    init_mlp_params,
    init_transformer_params,
)
from src.utils.config import DEFAULT_REQUIRED_GLOBAL_INPUTS

FASTCHEM_GLOBAL_ORDER = ["He_H", "C_H", "O_H", "N_H", "S_H"]
VULCAN_GLOBAL_ORDER = list(DEFAULT_REQUIRED_GLOBAL_INPUTS)


def _element_globals() -> dict[str, float]:
    return {
        "He_H": 8.38e-2,
        "C_H": 3.60e-4,
        "O_H": 5.37e-4,
        "N_H": 8.50e-5,
        "S_H": 1.80e-5,
    }


def _vulcan_globals() -> dict[str, float]:
    return {
        "gravity_cm_s2": 900.0,
        "planet_radius_cm": 9.0e9,
        **_element_globals(),
        "r_star_rsun": 0.939,
        "semi_major_axis_au": 0.04858,
        "zenith_angle_deg": 48.0,
        "diurnal_factor": 1.0,
        "use_photochemistry": 0.0,
        "use_ion_chemistry": 0.0,
        "use_eddy_diffusion": 1.0,
        "use_molecular_diffusion": 0.0,
        "use_upwind_molecular_diffusion": 0.0,
        "use_boundary_conditions": 0.0,
        "use_condensation": 0.0,
        "use_settling": 0.0,
        "use_initial_cold_trap": 0.0,
        "use_sat_surface_h2o": 0.0,
        "atm_base_H2": 1.0,
        "atm_base_N2": 0.0,
        "atm_base_O2": 0.0,
        "atm_base_CO2": 0.0,
        "atm_base_H2O": 0.0,
    }


def _sequence_static_numpy(static_inputs: np.ndarray, blocks: list[dict]) -> np.ndarray:
    return np.concatenate(
        [apply_block(static_inputs[:, idx : idx + 1], block) for idx, block in enumerate(blocks)],
        axis=-1,
    )


def _fastchem_normalization() -> dict:
    return {
        "sequence_static": {
            "feature_order": ["pressure_bar", "temperature_k"],
            "blocks": [
                {"method": "log-standard", "mean": [-2.0], "std": [1.5], "floor": 1.0e-30},
                {"method": "standard", "mean": [1100.0], "std": [250.0]},
            ],
        },
        "global_static": {
            "method": "mixed",
            "methods": ["standard", "log-standard", "log-standard", "log-standard", "log-standard"],
            "mean": [8.0e-2, -3.5, -3.2, -4.2, -4.8],
            "std": [1.0e-2, 0.2, 0.2, 0.2, 0.2],
            "floor": [None, 1.0e-30, 1.0e-30, 1.0e-30, 1.0e-30],
        },
        "target": {
            "method": "log-standard",
            "mean": [-5.0, -6.0],
            "std": [1.2, 0.8],
            "floor": 1.0e-30,
        },
    }


def _vulcan_normalization() -> dict:
    return {
        "sequence_static": {
            "feature_order": ["pressure_bar", "temperature_k", "kzz_cm2_s"],
            "blocks": [
                {"method": "log-standard", "mean": [-2.0], "std": [1.8], "floor": 1.0e-30},
                {"method": "standard", "mean": [1200.0], "std": [300.0]},
                {"method": "log-standard", "mean": [8.0], "std": [0.5], "floor": 1.0e-30},
            ],
        },
        "target": {
            "method": "log-standard",
            "mean": [-6.0, -7.0],
            "std": [1.0, 0.7],
            "floor": 1.0e-30,
        },
        "global_static": {
            "method": "mixed",
            "methods": [
                "log-standard",
                "log-standard",
                "standard",
                "log-standard",
                "log-standard",
                "log-standard",
                "log-standard",
            ]
            + ["none"] * (len(VULCAN_GLOBAL_ORDER) - 7),
            "mean": [3.0, 10.0, 8.0e-2, -3.5, -3.2, -4.2, -4.8] + [0.0] * (len(VULCAN_GLOBAL_ORDER) - 7),
            "std": [0.2, 0.15, 1.0e-2, 0.2, 0.2, 0.2, 0.2] + [1.0] * (len(VULCAN_GLOBAL_ORDER) - 7),
            "floor": [1.0e-30, 1.0e-30, None, 1.0e-30, 1.0e-30, 1.0e-30, 1.0e-30]
            + [None] * (len(VULCAN_GLOBAL_ORDER) - 7),
        },
        "spectrum_processing": {
            "floor": 1.0e-30,
            "wavelength_min_nm": 400.0,
            "wavelength_max_nm": 410.0,
            "max_tokens": 8,
        },
    }


def _fastchem_mlp_dims() -> MLPDimensions:
    return MLPDimensions(
        sequence_dim=2,
        global_dim=len(FASTCHEM_GLOBAL_ORDER),
        spectrum_max_tokens=0,
        spectrum_latent_dim=0,
        spectrum_hidden_dim=0,
        spectrum_num_latents=0,
        spectrum_num_layers=0,
        spectrum_num_heads=1,
        spectrum_fourier_features=0,
        spectrum_encoder_mode="none",
        spectrum_floor=1.0e-30,
        target_dim=2,
        d_hidden=8,
        num_hidden_layers=2,
        conditioning_hidden_dim=6,
        film_clamp=1.5,
        activation="silu",
    )


def _fastchem_transformer_dims() -> TransformerDimensions:
    return TransformerDimensions(
        sequence_dim=2,
        global_dim=len(FASTCHEM_GLOBAL_ORDER),
        spectrum_max_tokens=0,
        target_dim=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        conditioning_hidden_dim=8,
        film_clamp=1.5,
        output_head_divisor=2,
        spectrum_latent_dim=0,
        spectrum_hidden_dim=0,
        spectrum_num_latents=0,
        spectrum_num_layers=0,
        spectrum_num_heads=1,
        spectrum_fourier_features=0,
        spectrum_encoder_mode="none",
        spectrum_floor=1.0e-30,
        activation="gelu",
    )


def _vulcan_mlp_dims() -> MLPDimensions:
    return MLPDimensions(
        sequence_dim=3,
        global_dim=len(VULCAN_GLOBAL_ORDER),
        spectrum_max_tokens=8,
        spectrum_latent_dim=2,
        spectrum_hidden_dim=8,
        spectrum_num_latents=2,
        spectrum_num_layers=1,
        spectrum_num_heads=2,
        spectrum_fourier_features=4,
        spectrum_encoder_mode="perceiver",
        spectrum_floor=1.0e-30,
        target_dim=2,
        d_hidden=8,
        num_hidden_layers=2,
        conditioning_hidden_dim=6,
        film_clamp=1.5,
        activation="silu",
    )


def _vulcan_transformer_dims() -> TransformerDimensions:
    return TransformerDimensions(
        sequence_dim=3,
        global_dim=len(VULCAN_GLOBAL_ORDER),
        spectrum_max_tokens=8,
        target_dim=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        conditioning_hidden_dim=8,
        film_clamp=1.5,
        output_head_divisor=2,
        spectrum_latent_dim=2,
        spectrum_hidden_dim=8,
        spectrum_num_latents=2,
        spectrum_num_layers=1,
        spectrum_num_heads=2,
        spectrum_fourier_features=4,
        spectrum_encoder_mode="perceiver",
        spectrum_floor=1.0e-30,
        activation="gelu",
    )


def _make_fastchem_payload(model_type: str) -> tuple[dict, dict, MLPDimensions | TransformerDimensions]:
    normalization = _fastchem_normalization()
    if model_type == "mlp":
        dims: MLPDimensions | TransformerDimensions = _fastchem_mlp_dims()
        params = init_mlp_params(jax.random.PRNGKey(0), dims)
    else:
        dims = _fastchem_transformer_dims()
        params = init_transformer_params(jax.random.PRNGKey(1), dims)
    payload = {
        "params": jax.tree_util.tree_map(np.asarray, params),
        "model_dimensions": dims.to_dict(),
        "normalization": normalization,
        "data_contract": {
            "chemistry_type": "fastchem",
            "model_type": model_type,
            "global_static_feature_order": list(FASTCHEM_GLOBAL_ORDER),
            "output_species_order": ["H2O", "CO"],
        },
        "config": {"chemistry_type": "fastchem", "model_type": model_type},
    }
    return payload, normalization, dims


def _make_vulcan_payload(model_type: str) -> tuple[dict, dict, MLPDimensions | TransformerDimensions]:
    normalization = _vulcan_normalization()
    if model_type == "mlp":
        dims: MLPDimensions | TransformerDimensions = _vulcan_mlp_dims()
        params = init_mlp_params(jax.random.PRNGKey(2), dims)
    else:
        dims = _vulcan_transformer_dims()
        params = init_transformer_params(jax.random.PRNGKey(3), dims)
    payload = {
        "params": jax.tree_util.tree_map(np.asarray, params),
        "model_dimensions": dims.to_dict(),
        "normalization": normalization,
        "data_contract": {
            "chemistry_type": "vulcan",
            "model_type": model_type,
            "global_static_feature_order": list(VULCAN_GLOBAL_ORDER),
            "spectrum_max_tokens": 8,
            "spectrum_wavelength_min_nm": 400.0,
            "spectrum_wavelength_max_nm": 410.0,
            "output_species_order": ["H2O", "CO"],
        },
        "config": {
            "chemistry_type": "vulcan",
            "model_type": model_type,
            "normalization": {"spectrum_floor": 1.0e-30},
            "stellar_spectrum": {
                "latent_dim": dims.spectrum_latent_dim,
                "hidden_dim": dims.spectrum_hidden_dim,
                "num_latents": dims.spectrum_num_latents,
                "num_layers": dims.spectrum_num_layers,
                "num_heads": dims.spectrum_num_heads,
                "fourier_features": dims.spectrum_fourier_features,
                "encoder_mode": dims.spectrum_encoder_mode,
            },
        },
    }
    return payload, normalization, dims


def _expected_fastchem_prediction(
    dims: MLPDimensions | TransformerDimensions,
    params: dict,
    normalization: dict,
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
    global_inputs: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    static_inputs = np.stack([pressure_bar, temperature_k], axis=-1)
    sequence_norm = _sequence_static_numpy(static_inputs, normalization["sequence_static"]["blocks"])
    globals_norm = apply_mixed_block(
        np.array([[global_inputs[name] for name in FASTCHEM_GLOBAL_ORDER]], dtype=np.float64),
        normalization["global_static"],
    )
    if isinstance(dims, MLPDimensions):
        expected_norm, _ = apply_mlp(
            params,
            jnp.asarray(sequence_norm[None, :, :], dtype=jnp.float32),
            jnp.asarray(globals_norm, dtype=jnp.float32),
            dims,
        )
    else:
        expected_norm, _ = apply_transformer_model(
            params,
            jnp.asarray(sequence_norm[None, :, :], dtype=jnp.float32),
            jnp.asarray(globals_norm, dtype=jnp.float32),
            None,
            None,
            None,
            dims,
        )
    expected_physical = inverse_block(np.asarray(expected_norm[0]), normalization["target"])
    expected_log10 = (
        np.asarray(expected_norm[0], dtype=np.float64)
        * np.asarray(normalization["target"]["std"], dtype=np.float64)
        + np.asarray(normalization["target"]["mean"], dtype=np.float64)
    )
    return expected_physical, expected_log10


def _expected_vulcan_prediction(
    dims: MLPDimensions | TransformerDimensions,
    params: dict,
    normalization: dict,
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
    kzz_cm2_s: np.ndarray,
    global_inputs: dict[str, float],
    spectrum_wavelength_nm: np.ndarray,
    spectrum_flux_erg_cm2_s_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    static_inputs = np.stack([pressure_bar, temperature_k, kzz_cm2_s], axis=-1)
    sequence_norm = _sequence_static_numpy(static_inputs, normalization["sequence_static"]["blocks"])
    globals_norm = apply_mixed_block(
        np.array([[global_inputs[name] for name in VULCAN_GLOBAL_ORDER]], dtype=np.float64),
        normalization["global_static"],
    )
    packed_wavelengths_nm, packed_fluxes, packed_mask = pack_spectrum_tokens(
        spectrum_wavelength_nm,
        spectrum_flux_erg_cm2_s_nm,
        wavelength_min_nm=400.0,
        wavelength_max_nm=410.0,
        max_tokens=dims.spectrum_max_tokens,
    )
    if isinstance(dims, MLPDimensions):
        expected_norm, _ = apply_mlp(
            params,
            jnp.asarray(sequence_norm[None, :, :], dtype=jnp.float32),
            jnp.asarray(globals_norm, dtype=jnp.float32),
            dims,
            jnp.asarray(packed_wavelengths_nm[None, :], dtype=jnp.float32),
            jnp.asarray(packed_fluxes[None, :], dtype=jnp.float32),
            jnp.asarray(packed_mask[None, :], dtype=bool),
        )
    else:
        expected_norm, _ = apply_transformer_model(
            params,
            jnp.asarray(sequence_norm[None, :, :], dtype=jnp.float32),
            jnp.asarray(globals_norm, dtype=jnp.float32),
            jnp.asarray(packed_wavelengths_nm[None, :], dtype=jnp.float32),
            jnp.asarray(packed_fluxes[None, :], dtype=jnp.float32),
            jnp.asarray(packed_mask[None, :], dtype=bool),
            dims,
        )
    expected_physical = inverse_block(np.asarray(expected_norm[0]), normalization["target"])
    expected_log10 = (
        np.asarray(expected_norm[0], dtype=np.float64)
        * np.asarray(normalization["target"]["std"], dtype=np.float64)
        + np.asarray(normalization["target"]["mean"], dtype=np.float64)
    )
    return expected_physical, expected_log10


def test_exported_fastchem_mlp_bundle_predicts_from_physical_inputs(tmp_path):
    payload, normalization, dims = _make_fastchem_payload("mlp")
    bundle = load_exported_model(export_checkpoint_payload(payload, tmp_path / "fastchem_mlp_export.npz"))

    pressure_bar = np.array([100.0, 10.0, 1.0, 0.1], dtype=np.float32)
    temperature_k = np.array([1450.0, 1300.0, 1050.0, 900.0], dtype=np.float32)
    global_inputs = _element_globals()
    expected_physical, expected_log10 = _expected_fastchem_prediction(
        dims,
        payload["params"],
        normalization,
        pressure_bar,
        temperature_k,
        global_inputs,
    )

    predicted_physical = bundle.predict_fastchem_profile(
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        global_inputs=global_inputs,
    )
    predicted_log10 = bundle.predict_fastchem_profile(
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        global_inputs=global_inputs,
        return_log10=True,
    )

    np.testing.assert_allclose(np.asarray(predicted_physical), expected_physical, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(predicted_log10), expected_log10, rtol=1.0e-5, atol=1.0e-6)


def test_exported_fastchem_transformer_bundle_predicts_from_physical_inputs(tmp_path):
    payload, normalization, dims = _make_fastchem_payload("transformer")
    bundle = load_exported_model(export_checkpoint_payload(payload, tmp_path / "fastchem_transformer_export.npz"))

    pressure_bar = np.array([100.0, 10.0, 1.0, 0.1], dtype=np.float32)
    temperature_k = np.array([1450.0, 1300.0, 1050.0, 900.0], dtype=np.float32)
    global_inputs = _element_globals()
    expected_physical, expected_log10 = _expected_fastchem_prediction(
        dims,
        payload["params"],
        normalization,
        pressure_bar,
        temperature_k,
        global_inputs,
    )

    predicted_physical = bundle.predict_fastchem_profile(
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        global_inputs=global_inputs,
    )
    predicted_log10 = bundle.predict_fastchem_profile(
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        global_inputs=global_inputs,
        return_log10=True,
    )

    np.testing.assert_allclose(np.asarray(predicted_physical), expected_physical, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(predicted_log10), expected_log10, rtol=1.0e-5, atol=1.0e-6)


def test_exported_fastchem_compiled_predictor_matches_eager_and_is_cached(tmp_path):
    payload, _, _ = _make_fastchem_payload("transformer")
    bundle = load_exported_model(export_checkpoint_payload(payload, tmp_path / "fastchem_transformer_export.npz"))

    pressure_bar = jnp.asarray([100.0, 10.0, 1.0, 0.1], dtype=jnp.float32)
    temperature_k = jnp.asarray([1450.0, 1300.0, 1050.0, 900.0], dtype=jnp.float32)
    global_inputs = _element_globals()

    compiled_linear = bundle.make_compiled_fastchem_profile_predictor()
    compiled_log10 = bundle.make_compiled_fastchem_profile_predictor(return_log10=True)

    assert compiled_linear is bundle.make_compiled_fastchem_profile_predictor()
    assert compiled_log10 is bundle.make_compiled_fastchem_profile_predictor(return_log10=True)

    eager_linear = bundle.predict_fastchem_profile(
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        global_inputs=global_inputs,
    )
    eager_log10 = bundle.predict_fastchem_profile(
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        global_inputs=global_inputs,
        return_log10=True,
    )
    compiled_linear_output = compiled_linear(pressure_bar, temperature_k, global_inputs)
    compiled_log10_output = compiled_log10(pressure_bar, temperature_k, global_inputs)

    np.testing.assert_allclose(
        np.asarray(compiled_linear_output),
        np.asarray(eager_linear),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(compiled_log10_output),
        np.asarray(eager_log10),
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_exported_vulcan_mlp_bundle_predicts_from_unseen_physical_spectrum(tmp_path):
    payload, normalization, dims = _make_vulcan_payload("mlp")
    bundle = load_exported_model(export_checkpoint_payload(payload, tmp_path / "vulcan_mlp_export.npz"))

    pressure_bar = np.array([100.0, 10.0, 1.0], dtype=np.float32)
    temperature_k = np.array([1500.0, 1200.0, 950.0], dtype=np.float32)
    kzz_cm2_s = np.array([1.0e8, 1.0e8, 1.0e8], dtype=np.float32)
    global_inputs = _vulcan_globals()
    spectrum_wavelength_nm = np.array([401.0, 403.0, 405.0, 407.0, 409.0], dtype=np.float32)
    spectrum_flux = np.array([15.0, 20.0, 18.0, 12.0, 5.0], dtype=np.float32)
    expected_physical, expected_log10 = _expected_vulcan_prediction(
        dims,
        payload["params"],
        normalization,
        pressure_bar,
        temperature_k,
        kzz_cm2_s,
        global_inputs,
        spectrum_wavelength_nm,
        spectrum_flux,
    )

    predicted_physical = bundle.predict_vulcan_profile(
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        kzz_cm2_s=kzz_cm2_s,
        global_inputs=global_inputs,
        spectrum_wavelength_nm=spectrum_wavelength_nm,
        spectrum_flux_erg_cm2_s_nm=spectrum_flux,
    )
    predicted_log10 = bundle.predict_vulcan_profile(
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        kzz_cm2_s=kzz_cm2_s,
        global_inputs=global_inputs,
        spectrum_wavelength_nm=spectrum_wavelength_nm,
        spectrum_flux_erg_cm2_s_nm=spectrum_flux,
        return_log10=True,
    )

    np.testing.assert_allclose(np.asarray(predicted_physical), expected_physical, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(predicted_log10), expected_log10, rtol=1.0e-5, atol=1.0e-6)


def test_exported_vulcan_transformer_bundle_predicts_from_unseen_physical_spectrum(tmp_path):
    payload, normalization, dims = _make_vulcan_payload("transformer")
    bundle = load_exported_model(export_checkpoint_payload(payload, tmp_path / "vulcan_transformer_export.npz"))

    pressure_bar = np.array([100.0, 10.0, 1.0], dtype=np.float32)
    temperature_k = np.array([1500.0, 1200.0, 950.0], dtype=np.float32)
    kzz_cm2_s = np.array([1.0e8, 1.0e8, 1.0e8], dtype=np.float32)
    global_inputs = _vulcan_globals()
    spectrum_wavelength_nm = np.array([401.0, 403.0, 405.0, 407.0, 409.0], dtype=np.float32)
    spectrum_flux = np.array([15.0, 20.0, 18.0, 12.0, 5.0], dtype=np.float32)
    expected_physical, expected_log10 = _expected_vulcan_prediction(
        dims,
        payload["params"],
        normalization,
        pressure_bar,
        temperature_k,
        kzz_cm2_s,
        global_inputs,
        spectrum_wavelength_nm,
        spectrum_flux,
    )

    predicted_physical = bundle.predict_vulcan_profile(
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        kzz_cm2_s=kzz_cm2_s,
        global_inputs=global_inputs,
        spectrum_wavelength_nm=spectrum_wavelength_nm,
        spectrum_flux_erg_cm2_s_nm=spectrum_flux,
    )
    predicted_log10 = bundle.predict_vulcan_profile(
        pressure_bar=pressure_bar,
        temperature_k=temperature_k,
        kzz_cm2_s=kzz_cm2_s,
        global_inputs=global_inputs,
        spectrum_wavelength_nm=spectrum_wavelength_nm,
        spectrum_flux_erg_cm2_s_nm=spectrum_flux,
        return_log10=True,
    )

    np.testing.assert_allclose(np.asarray(predicted_physical), expected_physical, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(predicted_log10), expected_log10, rtol=1.0e-5, atol=1.0e-6)
    assert bundle.export_format == "jax_physical_bundle"
    assert bundle.export_version == 4


def test_export_loader_rejects_legacy_task_based_bundle(tmp_path):
    dims = _fastchem_mlp_dims()
    params = init_mlp_params(jax.random.PRNGKey(11), dims)
    payload = {
        "params": jax.tree_util.tree_map(np.asarray, params),
        "model_dimensions": dims.to_dict(),
        "normalization": _fastchem_normalization(),
        "data_contract": {
            "chemistry_type": "fastchem",
            "model_type": "mlp",
            "global_static_feature_order": list(FASTCHEM_GLOBAL_ORDER),
            "output_species_order": ["H2O", "CO"],
        },
        "config": {"chemistry_type": "fastchem", "model_type": "mlp"},
    }
    bundle_path = export_checkpoint_payload(payload, tmp_path / "legacy_export.npz")
    with np.load(Path(bundle_path), allow_pickle=False) as arrays:
        rewritten = {
            name: arrays[name]
            for name in arrays.files
            if name not in {"meta/chemistry_type", "meta/model_type"}
        }
        legacy_contract = dict(payload["data_contract"])
        legacy_contract.pop("chemistry_type")
        legacy_contract.pop("model_type")
        rewritten["meta/data_contract"] = np.array(json.dumps(legacy_contract))
        rewritten["meta/config"] = np.array(json.dumps({"task": {"kind": "equilibrium_only"}}))
    legacy_path = tmp_path / "legacy_export_missing_metadata.npz"
    np.savez(legacy_path, **rewritten)

    with pytest.raises(ValueError, match="legacy task-based contract"):
        load_exported_model(legacy_path)
