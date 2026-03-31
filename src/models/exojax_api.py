"""ExoJAX-facing JAX wrappers for exported VULCAN emulator bundles.

Two branch-specific factories are provided:

    make_equilibrium_vmr_fn(bundle)
    make_full_vulcan_vmr_fn(bundle)

Both return pure JAX callables plus species labels. The public API uses
top-to-bottom level order and column-global chemistry scalars
(``metallicity_log10``, ``c_to_o``, ``s_to_o``). The raw-generation pipeline
still materializes explicit elemental-abundance profiles internally for
FastChem and VULCAN, but those internal solver inputs are not part of the
supported ExoJAX inference contract.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..utils.config import (
    DEFAULT_REQUIRED_GLOBAL_INPUTS,
    EQUILIBRIUM_CONDITIONING_INPUT_ORDER,
    static_conditioning_defaults,
)
from .export_bundle import ExportedJAXModel

EQUILIBRIUM_GLOBAL_LABELS = list(EQUILIBRIUM_CONDITIONING_INPUT_ORDER)
FULL_VULCAN_GLOBAL_LABELS = list(DEFAULT_REQUIRED_GLOBAL_INPUTS)


def _validate_equilibrium_bundle_global_order(bundle: ExportedJAXModel) -> None:
    """Reject equilibrium bundles that do not match the ratio-global contract."""
    feature_order = list(bundle.data_contract.get("global_static_feature_order", []))
    if feature_order != EQUILIBRIUM_GLOBAL_LABELS:
        raise ValueError(
            "Equilibrium bundle global_static_feature_order must be "
            f"{EQUILIBRIUM_GLOBAL_LABELS}; got {feature_order}. Re-preprocess and retrain "
            "the equilibrium model with the updated ratio-global chemistry contract."
        )


def _validate_full_vulcan_bundle_global_order(bundle: ExportedJAXModel) -> None:
    """Reject full-VULCAN bundles that still use the legacy elemental conditioning contract."""
    feature_order = list(bundle.data_contract.get("global_static_feature_order", []))
    if feature_order != FULL_VULCAN_GLOBAL_LABELS:
        raise ValueError(
            "Full-VULCAN bundle global_static_feature_order must be "
            f"{FULL_VULCAN_GLOBAL_LABELS}; got {feature_order}. Re-preprocess and retrain "
            "the full-VULCAN model with the updated ratio-global chemistry contract."
        )


def _maybe_require_column_constant(values: Any, *, name: str) -> None:
    """Reject varying profiles in eager mode for inputs that remain column-constant."""
    if isinstance(values, jax.core.Tracer):
        return
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        is_constant = np.allclose(arr, arr[0], rtol=1.0e-8, atol=0.0)
    elif arr.ndim == 2:
        is_constant = np.allclose(arr, arr[0][None, :], rtol=1.0e-8, atol=0.0)
    else:
        return
    if not is_constant:
        raise ValueError(f"{name} must be vertically constant in the current ExoJAX contract.")


def _scalar_input(value: jax.Array | float, *, name: str) -> jax.Array:
    """Normalize one scalar chemistry input to a rank-0 JAX array."""
    scalar = jnp.asarray(value, dtype=jnp.float32)
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be a scalar input, got shape {tuple(scalar.shape)}.")
    return scalar


def _global_vector(
    *,
    feature_order: list[str],
    values: dict[str, Any],
) -> jax.Array:
    """Assemble one ordered global feature vector from named scalar values."""
    missing = [name for name in feature_order if name not in values]
    if missing:
        raise ValueError(f"Missing required conditioning values for features: {missing}.")
    return jnp.stack(
        [jnp.asarray(values[name], dtype=jnp.float32) for name in feature_order],
        axis=0,
    )


def _full_vulcan_static_defaults(bundle: ExportedJAXModel) -> dict[str, float]:
    """Resolve static full-VULCAN conditioning inputs baked into the bundle config."""
    feature_order = list(bundle.data_contract["global_static_feature_order"])
    needs_defaults = any(
        name.startswith("use_") or name.startswith("atm_base_")
        for name in feature_order
    )
    if not needs_defaults:
        return {}
    if "physics_toggles" not in bundle.config or "vulcan_runtime" not in bundle.config:
        raise ValueError(
            "Full-VULCAN bundle config is missing physics/base defaults required by "
            "the ExoJAX wrapper."
        )
    return static_conditioning_defaults(bundle.config)


def make_equilibrium_vmr_fn(
    bundle: ExportedJAXModel,
) -> tuple[Any, list[str]]:
    """Create a differentiable equilibrium-profile wrapper for ExoJAX."""
    if not bundle.is_equilibrium:
        raise ValueError("make_equilibrium_vmr_fn requires an equilibrium bundle.")

    _validate_equilibrium_bundle_global_order(bundle)
    species_labels = list(bundle.data_contract["output_species_order"])
    feature_order = list(bundle.data_contract["global_static_feature_order"])

    def vmr_fn(
        temperatures_k: jax.Array,  # shape: (nz,), top -> bottom
        pressures_bar: jax.Array,  # shape: (nz,), top -> bottom
        metallicity_log10: jax.Array,  # scalar
        c_to_o: jax.Array,  # scalar
        s_to_o: jax.Array,  # scalar
        gravity_cm_s2: jax.Array,  # shape: (nz,), top -> bottom
    ) -> jax.Array:
        """Return linear VMRs with shape ``(nz, n_species)`` in top-to-bottom order."""
        temperatures = jnp.asarray(temperatures_k, dtype=jnp.float32)
        pressures = jnp.asarray(pressures_bar, dtype=jnp.float32)
        gravity = jnp.asarray(gravity_cm_s2, dtype=jnp.float32)

        if temperatures.ndim != 1 or pressures.ndim != 1 or gravity.ndim != 1:
            raise ValueError("temperatures_k, pressures_bar, and gravity_cm_s2 must be 1-D arrays.")
        if temperatures.shape != pressures.shape or temperatures.shape != gravity.shape:
            raise ValueError("temperatures_k, pressures_bar, and gravity_cm_s2 must share the same shape.")

        internal_temperatures = temperatures[::-1]
        internal_pressures = pressures[::-1]
        global_inputs = _global_vector(
            feature_order=feature_order,
            values={
                "metallicity_log10": _scalar_input(metallicity_log10, name="metallicity_log10"),
                "c_to_o": _scalar_input(c_to_o, name="c_to_o"),
                "s_to_o": _scalar_input(s_to_o, name="s_to_o"),
            },
        )
        vmr_internal = bundle.predict_equilibrium_profile(
            pressure_bar=internal_pressures,
            temperature_k=internal_temperatures,
            global_inputs=global_inputs,
        )
        # Keep gravity in the public signature for interface compatibility.
        del gravity
        return vmr_internal[::-1, :]

    return vmr_fn, species_labels


def make_full_vulcan_vmr_fn(
    bundle: ExportedJAXModel,
) -> tuple[Any, list[str]]:
    """Create a differentiable full-VULCAN-profile wrapper for ExoJAX."""
    if bundle.is_equilibrium:
        raise ValueError("make_full_vulcan_vmr_fn requires a full_vulcan bundle.")

    _validate_full_vulcan_bundle_global_order(bundle)
    species_labels = list(bundle.data_contract["output_species_order"])
    feature_order = list(bundle.data_contract["global_static_feature_order"])
    static_defaults = _full_vulcan_static_defaults(bundle)

    def vmr_fn(
        temperatures_k: jax.Array,  # shape: (nz,), top -> bottom
        pressures_bar: jax.Array,  # shape: (nz,), top -> bottom
        kzz_cm2_s: jax.Array,  # shape: (nz,), top -> bottom
        metallicity_log10: jax.Array,  # scalar
        c_to_o: jax.Array,  # scalar
        s_to_o: jax.Array,  # scalar
        gravity_cm_s2: jax.Array,  # shape: (nz,), top -> bottom
        spectrum_flux: jax.Array,  # shape: (spectrum_dim,)
    ) -> jax.Array:
        """Return linear VMRs with shape ``(nz, n_species)`` in top-to-bottom order."""
        _maybe_require_column_constant(gravity_cm_s2, name="gravity_cm_s2")

        temperatures = jnp.asarray(temperatures_k, dtype=jnp.float32)
        pressures = jnp.asarray(pressures_bar, dtype=jnp.float32)
        kzz = jnp.asarray(kzz_cm2_s, dtype=jnp.float32)
        gravity = jnp.asarray(gravity_cm_s2, dtype=jnp.float32)

        if temperatures.ndim != 1 or pressures.ndim != 1 or kzz.ndim != 1 or gravity.ndim != 1:
            raise ValueError(
                "temperatures_k, pressures_bar, kzz_cm2_s, and gravity_cm_s2 must be 1-D arrays."
            )
        if not (temperatures.shape == pressures.shape == kzz.shape == gravity.shape):
            raise ValueError(
                "temperatures_k, pressures_bar, kzz_cm2_s, and gravity_cm_s2 must share the same shape."
            )

        internal_temperatures = temperatures[::-1]
        internal_pressures = pressures[::-1]
        internal_kzz = kzz[::-1]
        internal_gravity = gravity[::-1]

        global_values: dict[str, Any] = dict(static_defaults)
        global_values.update(
            {
                "gravity_cm_s2": internal_gravity[0],
                "metallicity_log10": _scalar_input(
                    metallicity_log10,
                    name="metallicity_log10",
                ),
                "c_to_o": _scalar_input(c_to_o, name="c_to_o"),
                "s_to_o": _scalar_input(s_to_o, name="s_to_o"),
            }
        )
        global_inputs = _global_vector(feature_order=feature_order, values=global_values)
        vmr_internal = bundle.predict_full_vulcan_profile(
            pressure_bar=internal_pressures,
            temperature_k=internal_temperatures,
            kzz_cm2_s=internal_kzz,
            global_inputs=global_inputs,
            spectrum_flux=spectrum_flux,
        )
        return vmr_internal[::-1, :]

    return vmr_fn, species_labels
