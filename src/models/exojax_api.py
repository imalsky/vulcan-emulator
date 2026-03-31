"""ExoJAX-facing JAX wrappers for the VULCAN emulator bundles.

Two branch-specific factories are provided:

    make_equilibrium_vmr_fn(bundle)
    make_transition_vmr_fn(bundle)

Both return pure JAX callables plus species labels.  The public API uses
top-to-bottom level order and FastChem-native hydrogen-normalized number
abundances (``n_X / n_H``).  The currently supported chemistry/gravity
manifold is column-constant, so eager calls reject vertically varying
elemental-abundance and gravity profiles.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import jax.numpy as jnp

from ..utils.config import ELEMENT_INPUT_ORDER, static_conditioning_defaults
from .export_bundle import ExportedJAXModel

ELEMENT_LABELS = list(ELEMENT_INPUT_ORDER)


def _validate_bundle_element_order(bundle: ExportedJAXModel) -> None:
    """Reject bundles whose stored element order disagrees with the fixed API contract."""
    labels = bundle.data_contract.get("element_input_order")
    if labels is None:
        return
    if not isinstance(labels, list) or not labels:
        raise ValueError(
            "Export bundle is missing data_contract['element_input_order']; "
            "re-export the model with the ExoJAX v2 data contract."
        )
    normalized = [str(label) for label in labels]
    if normalized != ELEMENT_LABELS:
        raise ValueError(
            "Export bundle element_input_order does not match the fixed ExoJAX "
            f"contract {ELEMENT_LABELS}; got {normalized}."
        )


def _maybe_require_column_constant(values: Any, *, name: str) -> None:
    """Reject varying profiles in eager mode for the current constant-profile contract."""
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


def _transition_static_defaults(bundle: ExportedJAXModel) -> dict[str, float]:
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
            "Transition bundle config is missing physics/base defaults required by "
            "the ExoJAX v2 wrapper."
        )
    return static_conditioning_defaults(bundle.config)


def make_equilibrium_vmr_fn(
    bundle: ExportedJAXModel,
) -> tuple[Any, list[str]]:
    """Create the ExoJAX v2 equilibrium interface for one equilibrium bundle."""
    if not bundle.is_equilibrium:
        raise ValueError("make_equilibrium_vmr_fn requires an equilibrium bundle.")

    _validate_bundle_element_order(bundle)
    species_labels = list(bundle.data_contract["output_species_order"])
    feature_order = list(bundle.data_contract["global_static_feature_order"])

    def vmr_fn(
        temperatures_k: jax.Array,           # shape: (nz,), top -> bottom
        pressures_bar: jax.Array,            # shape: (nz,), top -> bottom
        elemental_abundances_x_h: jax.Array, # shape: (nz, n_elements), top -> bottom
        gravity_cm_s2: jax.Array,            # shape: (nz,), top -> bottom
    ) -> jax.Array:
        """Return linear VMRs in shape ``(nz, n_species)`` using top-to-bottom order."""
        _maybe_require_column_constant(
            elemental_abundances_x_h,
            name="elemental_abundances_x_h",
        )
        _maybe_require_column_constant(
            gravity_cm_s2,
            name="gravity_cm_s2",
        )
        temperatures = jnp.asarray(temperatures_k, dtype=jnp.float32)
        pressures = jnp.asarray(pressures_bar, dtype=jnp.float32)
        elemental = jnp.asarray(elemental_abundances_x_h, dtype=jnp.float32)
        gravity = jnp.asarray(gravity_cm_s2, dtype=jnp.float32)

        if temperatures.ndim != 1 or pressures.ndim != 1 or gravity.ndim != 1:
            raise ValueError("temperatures_k, pressures_bar, and gravity_cm_s2 must be 1-D arrays.")
        if temperatures.shape != pressures.shape or temperatures.shape != gravity.shape:
            raise ValueError("temperatures_k, pressures_bar, and gravity_cm_s2 must share the same shape.")
        if elemental.ndim != 2 or elemental.shape[0] != temperatures.shape[0]:
            raise ValueError(
                "elemental_abundances_x_h must have shape (nz, n_elements) with nz matching the profiles."
            )
        if elemental.shape[1] != len(ELEMENT_LABELS):
            raise ValueError(
                f"elemental_abundances_x_h must have {len(ELEMENT_LABELS)} columns, got {elemental.shape[1]}."
            )

        # Internal models were trained on bottom-to-top ordering.
        internal_temperatures = temperatures[::-1]
        internal_pressures = pressures[::-1]
        internal_elemental = elemental[::-1, :]

        # Current equilibrium models are conditioned only on column chemistry.
        element_vector = internal_elemental[0, :]
        global_inputs = _global_vector(
            feature_order=feature_order,
            values={name: element_vector[idx] for idx, name in enumerate(ELEMENT_LABELS)},
        )
        vmr_internal = bundle.predict_equilibrium_profile(
            pressure_bar=internal_pressures,
            temperature_k=internal_temperatures,
            global_inputs=global_inputs,
        )
        # Keep gravity in the public signature; current equilibrium bundles do not use it.
        del gravity
        return vmr_internal[::-1, :]

    return vmr_fn, species_labels


def make_transition_vmr_fn(
    bundle: ExportedJAXModel,
) -> tuple[Any, list[str]]:
    """Create the ExoJAX v2 transition/full-VULCAN interface for one transition bundle."""
    if bundle.is_equilibrium:
        raise ValueError("make_transition_vmr_fn requires a transition bundle.")

    _validate_bundle_element_order(bundle)
    species_labels = list(bundle.data_contract["output_species_order"])
    feature_order = list(bundle.data_contract["global_static_feature_order"])
    static_defaults = _transition_static_defaults(bundle)

    def vmr_fn(
        temperatures_k: jax.Array,           # shape: (nz,), top -> bottom
        pressures_bar: jax.Array,            # shape: (nz,), top -> bottom
        elemental_abundances_x_h: jax.Array, # shape: (nz, n_elements), top -> bottom
        kzz_cm2_s: jax.Array,                # shape: (nz,), top -> bottom
        gravity_cm_s2: jax.Array,            # shape: (nz,), top -> bottom
        anchor_state: jax.Array,             # shape: (nz, state_dim), top -> bottom
        spectrum_flux: jax.Array,            # shape: (spectrum_dim,)
        dt_s: jax.Array,                     # shape: (), scalar
    ) -> jax.Array:
        """Return linear VMRs in shape ``(nz, n_species)`` using top-to-bottom order."""
        _maybe_require_column_constant(
            elemental_abundances_x_h,
            name="elemental_abundances_x_h",
        )
        _maybe_require_column_constant(
            gravity_cm_s2,
            name="gravity_cm_s2",
        )
        temperatures = jnp.asarray(temperatures_k, dtype=jnp.float32)
        pressures = jnp.asarray(pressures_bar, dtype=jnp.float32)
        elemental = jnp.asarray(elemental_abundances_x_h, dtype=jnp.float32)
        kzz = jnp.asarray(kzz_cm2_s, dtype=jnp.float32)
        gravity = jnp.asarray(gravity_cm_s2, dtype=jnp.float32)
        state = jnp.asarray(anchor_state, dtype=jnp.float32)

        if temperatures.ndim != 1 or pressures.ndim != 1 or kzz.ndim != 1 or gravity.ndim != 1:
            raise ValueError(
                "temperatures_k, pressures_bar, kzz_cm2_s, and gravity_cm_s2 must be 1-D arrays."
            )
        if not (
            temperatures.shape == pressures.shape == kzz.shape == gravity.shape
        ):
            raise ValueError(
                "temperatures_k, pressures_bar, kzz_cm2_s, and gravity_cm_s2 must share the same shape."
            )
        if elemental.ndim != 2 or elemental.shape[0] != temperatures.shape[0]:
            raise ValueError(
                "elemental_abundances_x_h must have shape (nz, n_elements) with nz matching the profiles."
            )
        if elemental.shape[1] != len(ELEMENT_LABELS):
            raise ValueError(
                f"elemental_abundances_x_h must have {len(ELEMENT_LABELS)} columns, got {elemental.shape[1]}."
            )
        if state.ndim != 2 or state.shape[0] != temperatures.shape[0]:
            raise ValueError("anchor_state must have shape (nz, state_dim).")

        internal_temperatures = temperatures[::-1]
        internal_pressures = pressures[::-1]
        internal_elemental = elemental[::-1, :]
        internal_kzz = kzz[::-1]
        internal_gravity = gravity[::-1]
        internal_state = state[::-1, :]

        element_vector = internal_elemental[0, :]
        gravity_value = internal_gravity[0]
        global_values: dict[str, Any] = dict(static_defaults)
        global_values["gravity_cm_s2"] = gravity_value
        for idx, name in enumerate(ELEMENT_LABELS):
            global_values[name] = element_vector[idx]

        global_inputs = _global_vector(feature_order=feature_order, values=global_values)
        vmr_internal = bundle.predict_transition_profile(
            pressure_bar=internal_pressures,
            temperature_k=internal_temperatures,
            kzz_cm2_s=internal_kzz,
            anchor_state=internal_state,
            global_inputs=global_inputs,
            spectrum_flux=spectrum_flux,
            dt_s=dt_s,
        )
        return vmr_internal[::-1, :]

    return vmr_fn, species_labels
