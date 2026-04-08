from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.models.exojax_api import make_fastchem_vmr_fn, make_vulcan_vmr_fn
from src.models.export_bundle import export_checkpoint_payload, load_exported_model
from src.models.jax_model import (
    MLPDimensions,
    TransformerDimensions,
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


def _vulcan_global_dict() -> dict[str, float]:
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


def _make_fastchem_bundle(tmp_path) -> object:
    dims = TransformerDimensions(
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
    params = init_transformer_params(jax.random.PRNGKey(42), dims)
    payload = {
        "params": jax.tree_util.tree_map(np.asarray, params),
        "model_dimensions": dims.to_dict(),
        "normalization": {
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
        },
        "data_contract": {
            "chemistry_type": "fastchem",
            "model_type": "transformer",
            "global_static_feature_order": list(FASTCHEM_GLOBAL_ORDER),
            "output_species_order": ["H2O", "CO"],
        },
        "config": {"chemistry_type": "fastchem", "model_type": "transformer"},
    }
    bundle_path = export_checkpoint_payload(payload, tmp_path / "fastchem_bundle.npz")
    return load_exported_model(bundle_path)


def _make_vulcan_bundle(tmp_path) -> object:
    dims = MLPDimensions(
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
    params = init_mlp_params(jax.random.PRNGKey(7), dims)
    payload = {
        "params": jax.tree_util.tree_map(np.asarray, params),
        "model_dimensions": dims.to_dict(),
        "normalization": {
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
        },
        "data_contract": {
            "chemistry_type": "vulcan",
            "model_type": "mlp",
            "global_static_feature_order": list(VULCAN_GLOBAL_ORDER),
            "spectrum_max_tokens": 8,
            "spectrum_wavelength_min_nm": 400.0,
            "spectrum_wavelength_max_nm": 410.0,
            "output_species_order": ["H2O", "CO"],
        },
        "config": {
            "chemistry_type": "vulcan",
            "model_type": "mlp",
            "normalization": {"spectrum_floor": 1.0e-30},
            "stellar_spectrum": {
                "latent_dim": 2,
                "hidden_dim": 8,
                "num_latents": 2,
                "num_layers": 1,
                "num_heads": 2,
                "fourier_features": 4,
                "encoder_mode": "perceiver",
            },
        },
    }
    bundle_path = export_checkpoint_payload(payload, tmp_path / "vulcan_bundle.npz")
    return load_exported_model(bundle_path)


def _make_legacy_vulcan_bundle(tmp_path) -> object:
    bundle = _make_vulcan_bundle(tmp_path)
    legacy_payload = {
        "params": jax.tree_util.tree_map(np.asarray, bundle.params),
        "model_dimensions": bundle.dims.to_dict(),
        "normalization": bundle.normalization,
        "data_contract": {
            "chemistry_type": "vulcan",
            "model_type": "mlp",
            "global_static_feature_order": ["gravity_cm_s2", "He_H", "C_H", "O_H", "N_H", "S_H"],
            "spectrum_max_tokens": 8,
            "spectrum_wavelength_min_nm": 400.0,
            "spectrum_wavelength_max_nm": 410.0,
            "output_species_order": ["H2O", "CO"],
        },
        "config": bundle.config,
    }
    bundle_path = export_checkpoint_payload(legacy_payload, tmp_path / "legacy_vulcan_bundle.npz")
    return load_exported_model(bundle_path)


def test_fastchem_vmr_fn_output_shape_and_labels(tmp_path):
    bundle = _make_fastchem_bundle(tmp_path)
    vmr_fn, species = make_fastchem_vmr_fn(bundle)

    nz = 6
    internal_t = jnp.ones(nz, dtype=jnp.float32) * 1200.0
    internal_p = jnp.logspace(2, -3, nz, dtype=jnp.float32)
    public_t = internal_t[::-1]
    public_p = internal_p[::-1]
    gravity = jnp.full((nz,), 2500.0, dtype=jnp.float32)

    vmr = vmr_fn(public_t, public_p, _element_globals(), gravity)

    assert vmr.shape == (nz, 2)
    assert species == ["H2O", "CO"]


def test_fastchem_vmr_fn_matches_bundle_after_level_reversal(tmp_path):
    bundle = _make_fastchem_bundle(tmp_path)
    vmr_fn, _ = make_fastchem_vmr_fn(bundle)

    internal_p = np.array([100.0, 10.0, 1.0, 0.1], dtype=np.float32)
    internal_t = np.array([1500.0, 1300.0, 1100.0, 900.0], dtype=np.float32)
    public_p = jnp.asarray(internal_p[::-1])
    public_t = jnp.asarray(internal_t[::-1])
    gravity = jnp.full(public_p.shape, 2500.0, dtype=jnp.float32)

    vmr_api = np.asarray(vmr_fn(public_t, public_p, _element_globals(), gravity))
    vmr_ref = np.asarray(
        bundle.predict_fastchem_profile(
            pressure_bar=internal_p,
            temperature_k=internal_t,
            global_inputs=_element_globals(),
        )
    )[::-1]

    np.testing.assert_allclose(vmr_api, vmr_ref, rtol=1.0e-5, atol=1.0e-7)


def test_fastchem_vmr_fn_jit_and_vjp(tmp_path):
    bundle = _make_fastchem_bundle(tmp_path)
    vmr_fn, _ = make_fastchem_vmr_fn(bundle)

    nz = 5
    temperatures = jnp.linspace(900.0, 1400.0, nz, dtype=jnp.float32)
    pressures = jnp.logspace(-2, 2, nz, dtype=jnp.float32)
    gravity = jnp.full((nz,), 2200.0, dtype=jnp.float32)
    global_inputs = jnp.asarray([_element_globals()[name] for name in FASTCHEM_GLOBAL_ORDER], dtype=jnp.float32)

    eager = vmr_fn(temperatures, pressures, global_inputs, gravity)
    compiled = jax.jit(vmr_fn)(temperatures, pressures, global_inputs, gravity)
    np.testing.assert_allclose(np.asarray(eager), np.asarray(compiled), atol=1.0e-6)

    grad_t = jax.grad(lambda temp: jnp.sum(vmr_fn(temp, pressures, global_inputs, gravity)))(temperatures)
    assert grad_t.shape == temperatures.shape

    vmr_val, vjp_fn = jax.vjp(vmr_fn, temperatures, pressures, global_inputs, gravity)
    g_t, g_p, g_global, g_g = vjp_fn(jnp.ones_like(vmr_val))
    assert g_t.shape == temperatures.shape
    assert g_p.shape == pressures.shape
    assert g_global.shape == global_inputs.shape
    assert g_g.shape == gravity.shape


def test_vulcan_vmr_fn_matches_bundle_after_level_reversal(tmp_path):
    bundle = _make_vulcan_bundle(tmp_path)
    vmr_fn, species = make_vulcan_vmr_fn(bundle)

    internal_p = np.array([100.0, 10.0, 1.0], dtype=np.float32)
    internal_t = np.array([1500.0, 1200.0, 950.0], dtype=np.float32)
    internal_kzz = np.array([1.0e8, 1.0e8, 1.0e8], dtype=np.float32)
    public_p = jnp.asarray(internal_p[::-1])
    public_t = jnp.asarray(internal_t[::-1])
    public_kzz = jnp.asarray(internal_kzz[::-1])
    spectrum_wavelength_nm = jnp.asarray([401.0, 403.0, 405.0, 407.0, 409.0], dtype=jnp.float32)
    spectrum_flux = jnp.asarray([15.0, 20.0, 18.0, 12.0, 5.0], dtype=jnp.float32)

    vmr_api = np.asarray(
        vmr_fn(public_t, public_p, public_kzz, _vulcan_global_dict(), spectrum_wavelength_nm, spectrum_flux)
    )
    vmr_ref = np.asarray(
        bundle.predict_vulcan_profile(
            pressure_bar=internal_p,
            temperature_k=internal_t,
            kzz_cm2_s=internal_kzz,
            global_inputs=_vulcan_global_dict(),
            spectrum_wavelength_nm=spectrum_wavelength_nm,
            spectrum_flux_erg_cm2_s_nm=spectrum_flux,
        )
    )[::-1]

    assert species == ["H2O", "CO"]
    np.testing.assert_allclose(vmr_api, vmr_ref, rtol=1.0e-5, atol=1.0e-7)


def test_vulcan_vmr_fn_jit_and_vjp(tmp_path):
    bundle = _make_vulcan_bundle(tmp_path)
    vmr_fn, _ = make_vulcan_vmr_fn(bundle)

    nz = 4
    temperatures = jnp.linspace(900.0, 1400.0, nz, dtype=jnp.float32)
    pressures = jnp.logspace(-2, 2, nz, dtype=jnp.float32)
    kzz = jnp.full((nz,), 1.0e8, dtype=jnp.float32)
    spectrum_wavelength_nm = jnp.asarray([401.0, 403.0, 405.0, 407.0, 409.0], dtype=jnp.float32)
    spectrum_flux = jnp.asarray([15.0, 20.0, 18.0, 12.0, 5.0], dtype=jnp.float32)
    global_inputs = jnp.asarray([_vulcan_global_dict()[name] for name in VULCAN_GLOBAL_ORDER], dtype=jnp.float32)

    eager = vmr_fn(temperatures, pressures, kzz, global_inputs, spectrum_wavelength_nm, spectrum_flux)
    compiled = jax.jit(vmr_fn)(
        temperatures,
        pressures,
        kzz,
        global_inputs,
        spectrum_wavelength_nm,
        spectrum_flux,
    )
    np.testing.assert_allclose(np.asarray(eager), np.asarray(compiled), atol=1.0e-6)

    grad_t = jax.grad(
        lambda temp: jnp.sum(vmr_fn(temp, pressures, kzz, global_inputs, spectrum_wavelength_nm, spectrum_flux))
    )(temperatures)
    assert grad_t.shape == temperatures.shape

    vmr_val, vjp_fn = jax.vjp(
        vmr_fn,
        temperatures,
        pressures,
        kzz,
        global_inputs,
        spectrum_wavelength_nm,
        spectrum_flux,
    )
    g_t, g_p, g_kzz, g_global, g_wavelength, g_flux = vjp_fn(jnp.ones_like(vmr_val))
    assert g_t.shape == temperatures.shape
    assert g_p.shape == pressures.shape
    assert g_kzz.shape == kzz.shape
    assert g_global.shape == global_inputs.shape
    assert g_wavelength.shape == spectrum_wavelength_nm.shape
    assert g_flux.shape == spectrum_flux.shape


def test_vulcan_vmr_fn_rejects_nonconstant_gravity_in_eager_mode(tmp_path):
    bundle = _make_vulcan_bundle(tmp_path)
    vmr_fn, _ = make_vulcan_vmr_fn(bundle)

    nz = 4
    temperatures = jnp.linspace(900.0, 1400.0, nz, dtype=jnp.float32)
    pressures = jnp.logspace(-2, 2, nz, dtype=jnp.float32)
    kzz = jnp.full((nz,), 1.0e8, dtype=jnp.float32)
    spectrum_wavelength_nm = jnp.asarray([401.0, 403.0, 405.0, 407.0, 409.0], dtype=jnp.float32)
    spectrum_flux = jnp.asarray([15.0, 20.0, 18.0, 12.0, 5.0], dtype=jnp.float32)
    global_inputs = _vulcan_global_dict()
    global_inputs["gravity_cm_s2"] = jnp.asarray([850.0, 850.0, 900.0, 850.0], dtype=jnp.float32)

    with pytest.raises(ValueError, match="gravity_cm_s2"):
        vmr_fn(temperatures, pressures, kzz, global_inputs, spectrum_wavelength_nm, spectrum_flux)


def test_vulcan_vmr_fn_rejects_nonconstant_planet_radius_in_eager_mode(tmp_path):
    bundle = _make_vulcan_bundle(tmp_path)
    vmr_fn, _ = make_vulcan_vmr_fn(bundle)

    nz = 4
    temperatures = jnp.linspace(900.0, 1400.0, nz, dtype=jnp.float32)
    pressures = jnp.logspace(-2, 2, nz, dtype=jnp.float32)
    kzz = jnp.full((nz,), 1.0e8, dtype=jnp.float32)
    spectrum_wavelength_nm = jnp.asarray([401.0, 403.0, 405.0, 407.0, 409.0], dtype=jnp.float32)
    spectrum_flux = jnp.asarray([15.0, 20.0, 18.0, 12.0, 5.0], dtype=jnp.float32)
    global_inputs = _vulcan_global_dict()
    global_inputs["planet_radius_cm"] = jnp.asarray([9.0e9, 9.0e9, 9.1e9, 9.0e9], dtype=jnp.float32)

    with pytest.raises(ValueError, match="planet_radius_cm"):
        vmr_fn(temperatures, pressures, kzz, global_inputs, spectrum_wavelength_nm, spectrum_flux)


def test_vulcan_wrapper_rejects_legacy_elemental_bundle(tmp_path):
    legacy_bundle = _make_legacy_vulcan_bundle(tmp_path)
    with pytest.raises(ValueError, match="runtime-knob contract"):
        make_vulcan_vmr_fn(legacy_bundle)


def test_exojax_factories_reject_wrong_bundle_type(tmp_path):
    fastchem_bundle = _make_fastchem_bundle(tmp_path)
    vulcan_bundle = _make_vulcan_bundle(tmp_path)

    with pytest.raises(ValueError, match="fastchem"):
        make_fastchem_vmr_fn(vulcan_bundle)
    with pytest.raises(ValueError, match="vulcan"):
        make_vulcan_vmr_fn(fastchem_bundle)
