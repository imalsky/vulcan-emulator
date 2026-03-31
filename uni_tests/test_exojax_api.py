from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from src.models.export_bundle import export_checkpoint_payload, load_exported_model
from src.models.jax_model import (
    EquilibriumMLPDimensions,
    ModelDimensions,
    init_equilibrium_mlp_params,
    init_model_params,
)
from src.models.exojax_api import (
    make_equilibrium_vmr_fn,
    make_full_vulcan_vmr_fn,
)

ELEMENT_ORDER = ["He_H", "C_H", "O_H", "N_H", "S_H"]


def _element_profile(nz: int) -> jax.Array:
    """Build a column-constant FastChem-native elemental-abundance profile."""
    values = jnp.asarray([8.38e-2, 3.40e-4, 5.10e-4, 8.50e-5, 1.60e-5], dtype=jnp.float32)
    return jnp.repeat(values[None, :], nz, axis=0)


def _element_dict() -> dict[str, float]:
    """Return the same elemental composition as a dict keyed by element label."""
    return {
        "He_H": 8.38e-2,
        "C_H": 3.40e-4,
        "O_H": 5.10e-4,
        "N_H": 8.50e-5,
        "S_H": 1.60e-5,
    }


def _make_equilibrium_bundle(tmp_path):
    """Build and export a tiny synthetic equilibrium bundle for testing."""
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
    params = init_equilibrium_mlp_params(jax.random.PRNGKey(42), dims)
    normalization = {
        "sequence_static": {
            "feature_order": ["pressure_bar", "temperature_k"],
            "blocks": [
                {"method": "log-standard", "mean": [-2.0], "std": [1.5], "floor": 1e-30},
                {"method": "standard", "mean": [1100.0], "std": [250.0]},
            ],
        },
        "global_static": {
            "method": "mixed",
            "methods": ["log-standard"] * len(ELEMENT_ORDER),
            "mean": [-1.1, -3.5, -3.3, -4.2, -4.8],
            "std": [0.1, 0.2, 0.2, 0.2, 0.2],
            "floor": [1e-30] * len(ELEMENT_ORDER),
        },
        "target": {
            "method": "log-standard",
            "mean": [-5.0, -6.0],
            "std": [1.2, 0.8],
            "floor": 1e-30,
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
    bundle_path = export_checkpoint_payload(payload, tmp_path / "eq_bundle.npz")
    return load_exported_model(bundle_path)


def _make_full_vulcan_bundle(tmp_path):
    """Build and export a tiny synthetic full-VULCAN bundle for testing."""
    global_static_order = [
        "gravity_cm_s2",
        *ELEMENT_ORDER,
        "use_photochemistry",
        "atm_base_H2",
    ]
    dims = ModelDimensions(
        sequence_dim=3,
        global_dim=len(global_static_order),
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
    params = init_model_params(jax.random.PRNGKey(7), dims)
    normalization = {
        "sequence_static": {
            "feature_order": ["pressure_bar", "temperature_k", "kzz_cm2_s"],
            "blocks": [
                {"method": "log-standard", "mean": [-2.0], "std": [1.8], "floor": 1e-30},
                {"method": "standard", "mean": [1200.0], "std": [300.0]},
                {"method": "log-standard", "mean": [8.0], "std": [0.5], "floor": 1e-30},
            ],
        },
        "target": {"method": "log-standard", "mean": [-6.0, -7.0], "std": [1.0, 0.7], "floor": 1e-30},
        "global_static": {
            "method": "mixed",
            "methods": ["log-standard"] * 6 + ["none", "none"],
            "mean": [3.0, -1.1, -3.5, -3.3, -4.2, -4.8, 0.0, 0.0],
            "std": [0.2, 0.1, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0],
            "floor": [1e-30] * 6 + [None, None],
        },
        "spectrum": {
            "method": "log-standard",
            "mean": [1.0, 1.2, 1.1, 0.9],
            "std": [0.3, 0.25, 0.35, 0.4],
            "floor": 1e-30,
        },
    }
    contract = {
        "global_static_feature_order": global_static_order,
        "spectrum_dim": 4,
        "element_input_order": list(ELEMENT_ORDER),
        "output_species_order": ["H2O", "CO"],
    }
    payload = {
        "params": jax.tree_util.tree_map(np.asarray, params),
        "model_dimensions": dims.to_dict(),
        "normalization": normalization,
        "data_contract": contract,
        "config": {
            "task": {"kind": "full_vulcan"},
            "physics_toggles": {
                "use_photochemistry": True,
                "use_ion_chemistry": False,
                "use_eddy_diffusion": False,
                "use_molecular_diffusion": False,
                "use_upwind_molecular_diffusion": False,
                "use_boundary_conditions": False,
                "use_condensation": False,
                "use_settling": False,
                "use_initial_cold_trap": False,
                "use_sat_surface_h2o": False,
            },
            "vulcan_runtime": {"atm_base": "H2"},
        },
    }
    bundle_path = export_checkpoint_payload(payload, tmp_path / "fv_bundle.npz")
    return load_exported_model(bundle_path)


def test_equilibrium_vmr_fn_output_shape_and_labels(tmp_path):
    bundle = _make_equilibrium_bundle(tmp_path)
    vmr_fn, species = make_equilibrium_vmr_fn(bundle)

    nz = 6
    internal_t = jnp.ones(nz, dtype=jnp.float32) * 1200.0
    internal_p = jnp.logspace(2, -3, nz, dtype=jnp.float32)
    public_t = internal_t[::-1]
    public_p = internal_p[::-1]
    element_profile = _element_profile(nz)
    gravity = jnp.full((nz,), 2500.0, dtype=jnp.float32)

    vmr = vmr_fn(public_t, public_p, element_profile, gravity)

    assert vmr.shape == (nz, 2)
    assert species == ["H2O", "CO"]


def test_equilibrium_vmr_fn_matches_bundle_after_level_reversal(tmp_path):
    bundle = _make_equilibrium_bundle(tmp_path)
    vmr_fn, _ = make_equilibrium_vmr_fn(bundle)

    internal_p = np.array([100.0, 10.0, 1.0, 0.1], dtype=np.float32)
    internal_t = np.array([1500.0, 1300.0, 1100.0, 900.0], dtype=np.float32)
    public_p = jnp.asarray(internal_p[::-1])
    public_t = jnp.asarray(internal_t[::-1])
    element_profile = _element_profile(public_p.shape[0])
    gravity = jnp.full(public_p.shape, 2500.0, dtype=jnp.float32)

    vmr_api = np.asarray(vmr_fn(public_t, public_p, element_profile, gravity))
    vmr_ref = np.asarray(
        bundle.predict_equilibrium_profile(
            pressure_bar=internal_p,
            temperature_k=internal_t,
            global_inputs=_element_dict(),
        )
    )[::-1]

    np.testing.assert_allclose(vmr_api, vmr_ref, rtol=1e-5, atol=1e-7)


def test_equilibrium_vmr_fn_jit_and_vjp(tmp_path):
    bundle = _make_equilibrium_bundle(tmp_path)
    vmr_fn, _ = make_equilibrium_vmr_fn(bundle)

    nz = 5
    T = jnp.linspace(900.0, 1400.0, nz, dtype=jnp.float32)
    P = jnp.logspace(-2, 2, nz, dtype=jnp.float32)
    elemental = _element_profile(nz)
    gravity = jnp.full((nz,), 2200.0, dtype=jnp.float32)

    eager = vmr_fn(T, P, elemental, gravity)
    compiled = jax.jit(vmr_fn)(T, P, elemental, gravity)
    np.testing.assert_allclose(np.asarray(eager), np.asarray(compiled), atol=1e-6)

    grad_t = jax.grad(lambda temp: jnp.sum(vmr_fn(temp, P, elemental, gravity)))(T)
    assert grad_t.shape == T.shape

    batched_t = jnp.stack([T, T + 10.0], axis=0)
    batched_p = jnp.stack([P, P], axis=0)
    batched_elemental = jnp.stack([elemental, elemental], axis=0)
    batched_gravity = jnp.stack([gravity, gravity], axis=0)
    vmapped = jax.vmap(vmr_fn)(batched_t, batched_p, batched_elemental, batched_gravity)
    assert vmapped.shape == (2, nz, 2)

    vmr_val, vjp_fn = jax.vjp(vmr_fn, T, P, elemental, gravity)
    g_t, g_p, g_elem, g_g = vjp_fn(jnp.ones_like(vmr_val))

    assert g_t.shape == T.shape
    assert g_p.shape == P.shape
    assert g_elem.shape == elemental.shape
    assert g_g.shape == gravity.shape


def test_equilibrium_vmr_fn_rejects_nonconstant_profiles_in_eager_mode(tmp_path):
    import pytest

    bundle = _make_equilibrium_bundle(tmp_path)
    vmr_fn, _ = make_equilibrium_vmr_fn(bundle)

    nz = 4
    T = jnp.linspace(900.0, 1200.0, nz, dtype=jnp.float32)
    P = jnp.logspace(-2, 2, nz, dtype=jnp.float32)
    elemental = _element_profile(nz).at[1, 0].set(9.0e-2)
    gravity = jnp.full((nz,), 2200.0, dtype=jnp.float32)

    with pytest.raises(ValueError, match="elemental_abundances_x_h"):
        vmr_fn(T, P, elemental, gravity)


def test_full_vulcan_vmr_fn_matches_bundle_after_level_reversal(tmp_path):
    bundle = _make_full_vulcan_bundle(tmp_path)
    vmr_fn, species = make_full_vulcan_vmr_fn(bundle)

    internal_p = np.array([100.0, 10.0, 1.0], dtype=np.float32)
    internal_t = np.array([1500.0, 1200.0, 950.0], dtype=np.float32)
    internal_kzz = np.array([1.0e8, 1.0e8, 1.0e8], dtype=np.float32)
    public_p = jnp.asarray(internal_p[::-1])
    public_t = jnp.asarray(internal_t[::-1])
    public_kzz = jnp.asarray(internal_kzz[::-1])
    elemental = _element_profile(public_p.shape[0])
    gravity = jnp.full(public_p.shape, 900.0, dtype=jnp.float32)
    spectrum_flux = jnp.asarray([15.0, 20.0, 18.0, 12.0], dtype=jnp.float32)

    vmr_api = np.asarray(
        vmr_fn(
            public_t,
            public_p,
            elemental,
            public_kzz,
            gravity,
            spectrum_flux,
        )
    )
    vmr_ref = np.asarray(
        bundle.predict_full_vulcan_profile(
            pressure_bar=internal_p,
            temperature_k=internal_t,
            kzz_cm2_s=internal_kzz,
            global_inputs={
                "gravity_cm_s2": 900.0,
                **_element_dict(),
                "use_photochemistry": 1.0,
                "atm_base_H2": 1.0,
            },
            spectrum_flux=spectrum_flux,
        )
    )[::-1]

    assert species == ["H2O", "CO"]
    np.testing.assert_allclose(vmr_api, vmr_ref, rtol=1e-5, atol=1e-7)


def test_full_vulcan_vmr_fn_jit_and_vjp(tmp_path):
    bundle = _make_full_vulcan_bundle(tmp_path)
    vmr_fn, _ = make_full_vulcan_vmr_fn(bundle)

    nz = 4
    T = jnp.linspace(900.0, 1400.0, nz, dtype=jnp.float32)
    P = jnp.logspace(-2, 2, nz, dtype=jnp.float32)
    Kzz = jnp.full((nz,), 1.0e8, dtype=jnp.float32)
    elemental = _element_profile(nz)
    gravity = jnp.full((nz,), 850.0, dtype=jnp.float32)
    spectrum_flux = jnp.asarray([15.0, 20.0, 18.0, 12.0], dtype=jnp.float32)

    eager = vmr_fn(T, P, elemental, Kzz, gravity, spectrum_flux)
    compiled = jax.jit(vmr_fn)(T, P, elemental, Kzz, gravity, spectrum_flux)
    np.testing.assert_allclose(np.asarray(eager), np.asarray(compiled), atol=1e-6)

    grad_t = jax.grad(
        lambda temp: jnp.sum(vmr_fn(temp, P, elemental, Kzz, gravity, spectrum_flux))
    )(T)
    assert grad_t.shape == T.shape

    batched_t = jnp.stack([T, T + 5.0], axis=0)
    batched_p = jnp.stack([P, P], axis=0)
    batched_elemental = jnp.stack([elemental, elemental], axis=0)
    batched_kzz = jnp.stack([Kzz, Kzz], axis=0)
    batched_gravity = jnp.stack([gravity, gravity], axis=0)
    batched_spectrum = jnp.stack([spectrum_flux, spectrum_flux], axis=0)
    vmapped = jax.vmap(vmr_fn)(
        batched_t,
        batched_p,
        batched_elemental,
        batched_kzz,
        batched_gravity,
        batched_spectrum,
    )
    assert vmapped.shape == (2, nz, 2)

    vmr_val, vjp_fn = jax.vjp(
        vmr_fn,
        T,
        P,
        elemental,
        Kzz,
        gravity,
        spectrum_flux,
    )
    g_t, g_p, g_elem, g_kzz, g_g, g_spectrum = vjp_fn(jnp.ones_like(vmr_val))

    assert g_t.shape == T.shape
    assert g_p.shape == P.shape
    assert g_elem.shape == elemental.shape
    assert g_kzz.shape == Kzz.shape
    assert g_g.shape == gravity.shape
    assert g_spectrum.shape == spectrum_flux.shape


def test_full_vulcan_vmr_fn_rejects_nonconstant_gravity_in_eager_mode(tmp_path):
    import pytest

    bundle = _make_full_vulcan_bundle(tmp_path)
    vmr_fn, _ = make_full_vulcan_vmr_fn(bundle)

    nz = 4
    T = jnp.linspace(900.0, 1400.0, nz, dtype=jnp.float32)
    P = jnp.logspace(-2, 2, nz, dtype=jnp.float32)
    Kzz = jnp.full((nz,), 1.0e8, dtype=jnp.float32)
    elemental = _element_profile(nz)
    gravity = jnp.asarray([850.0, 850.0, 900.0, 850.0], dtype=jnp.float32)
    spectrum_flux = jnp.asarray([15.0, 20.0, 18.0, 12.0], dtype=jnp.float32)

    with pytest.raises(ValueError, match="gravity_cm_s2"):
        vmr_fn(T, P, elemental, Kzz, gravity, spectrum_flux)


def test_exojax_v2_factories_reject_wrong_bundle_type(tmp_path):
    import pytest

    eq_bundle = _make_equilibrium_bundle(tmp_path)
    fv_bundle = _make_full_vulcan_bundle(tmp_path)

    with pytest.raises(ValueError, match="equilibrium"):
        make_equilibrium_vmr_fn(fv_bundle)
    with pytest.raises(ValueError, match="full_vulcan"):
        make_full_vulcan_vmr_fn(eq_bundle)


def test_old_ratio_helpers_are_not_exported():
    import src.models.exojax_api as exojax_api

    assert not hasattr(exojax_api, "abundances_to_model_params")
    assert not hasattr(exojax_api, "SOLAR_O_H")
    assert not hasattr(exojax_api, "SOLAR_C_H")
    assert not hasattr(exojax_api, "SOLAR_S_H")
