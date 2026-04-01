"""ExoJAX-facing JAX wrappers for exported emulator bundles.

Two chemistry-specific factories are provided:

    make_fastchem_vmr_fn(bundle)
    make_vulcan_vmr_fn(bundle)

Both return pure JAX callables plus species labels. The public API uses
top-to-bottom level order and the same named ``global_inputs`` contract used
by ``ExportedJAXModel.predict_*_profile()``. FastChem bundles expect
profile-global elemental abundances in fixed ``X/H`` order, while VULCAN
bundles expect gravity, the same elemental globals, curated science knobs,
and atmosphere-base one-hot flags.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..utils.config import DEFAULT_REQUIRED_GLOBAL_INPUTS, FASTCHEM_CONDITIONING_INPUT_ORDER
from .export_bundle import ExportedJAXModel

FASTCHEM_GLOBAL_LABELS = list(FASTCHEM_CONDITIONING_INPUT_ORDER)
VULCAN_GLOBAL_LABELS = list(DEFAULT_REQUIRED_GLOBAL_INPUTS)

def _validate_fastchem_bundle_global_order(bundle: ExportedJAXModel) -> None:
    """Reject FastChem bundles that use an outdated global-input ordering.

    Parameters
    ----------
    bundle : ExportedJAXModel
        Loaded export bundle whose ``data_contract`` should match the ExoJAX
        FastChem elemental-conditioning contract.

    Returns
    -------
    None
        The function returns silently when the bundle matches the current
        FastChem global-input contract.
    """
    feature_order = list(bundle.data_contract.get("global_static_feature_order", []))
    if feature_order != FASTCHEM_GLOBAL_LABELS:
        raise ValueError(
            "FastChem bundle global_static_feature_order must be "
            f"{FASTCHEM_GLOBAL_LABELS}; got {feature_order}. Re-preprocess and retrain "
            "the fastchem model with the updated elemental-conditioning contract."
        )


def _validate_vulcan_bundle_global_order(bundle: ExportedJAXModel) -> None:
    """Reject VULCAN bundles that use an outdated global-input ordering.

    Parameters
    ----------
    bundle : ExportedJAXModel
        Loaded export bundle whose ``data_contract`` should match the ExoJAX
        VULCAN runtime-conditioning contract.

    Returns
    -------
    None
        The function returns silently when the bundle matches the current
        VULCAN global-input contract.
    """
    feature_order = list(bundle.data_contract.get("global_static_feature_order", []))
    if feature_order != VULCAN_GLOBAL_LABELS:
        raise ValueError(
            "VULCAN bundle global_static_feature_order must be "
            f"{VULCAN_GLOBAL_LABELS}; got {feature_order}. Re-preprocess and retrain "
            "the vulcan model with the updated elemental-conditioning/runtime-knob contract."
        )


def _maybe_require_column_constant(values: Any, *, name: str) -> None:
    """Reject eager inputs that violate the column-constant ExoJAX contract.

    Parameters
    ----------
    values : Any
        Candidate scalar, 1-D profile, or 2-D stacked profile input.
    name : str
        Human-readable field name used in validation errors.

    Returns
    -------
    None
        The function returns silently when eager inputs are vertically
        constant or when tracing prevents eager validation.
    """
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


def _require_matching_level_shapes(
    *,
    temperatures: jax.Array,
    pressures: jax.Array,
    name: str = "temperatures_k and pressures_bar",
) -> None:
    """Validate that paired ExoJAX level-grid inputs share one 1-D grid.

    Parameters
    ----------
    temperatures : jax.Array
        Temperature profile with shape ``(nz,)``.
    pressures : jax.Array
        Pressure profile with shape ``(nz,)``.
    name : str, default="temperatures_k and pressures_bar"
        Field label used in validation errors.

    Returns
    -------
    None
        The function returns silently when both arrays are one-dimensional and
        share the same shape.
    """
    if temperatures.ndim != 1 or pressures.ndim != 1:
        raise ValueError(f"{name} must be 1-D arrays.")
    if temperatures.shape != pressures.shape:
        raise ValueError(f"{name} must share the same shape.")


def make_fastchem_vmr_fn(
    bundle: ExportedJAXModel,
) -> tuple[Any, list[str]]:
    """Create an ExoJAX-compatible FastChem profile inference callable.

    Parameters
    ----------
    bundle : ExportedJAXModel
        Exported FastChem emulator bundle with physical-unit preprocessing
        embedded in the wrapper.

    Returns
    -------
    tuple[Any, list[str]]
        Callable ``vmr_fn`` plus the ordered output-species labels. The
        callable returns linear VMR predictions with shape ``(nz, n_species)``
        in top-to-bottom level order.
    """
    if not bundle.uses_fastchem:
        raise ValueError("make_fastchem_vmr_fn requires a fastchem bundle.")

    _validate_fastchem_bundle_global_order(bundle)
    species_labels = list(bundle.data_contract["output_species_order"])

    def vmr_fn(
        temperatures_k: jax.Array,  # shape: (nz,), top -> bottom
        pressures_bar: jax.Array,  # shape: (nz,), top -> bottom
        global_inputs: dict[str, Any] | jax.Array,  # X/H globals
        gravity_cm_s2: jax.Array | None = None,  # optional shape: (nz,), top -> bottom
    ) -> jax.Array:
        """Predict one FastChem composition profile in ExoJAX ordering.

        Parameters
        ----------
        temperatures_k : jax.Array
            Temperature profile in Kelvin with shape ``(nz,)`` ordered from
            top to bottom.
        pressures_bar : jax.Array
            Pressure profile in bar with shape ``(nz,)`` ordered from top to
            bottom.
        global_inputs : dict[str, Any] or jax.Array
            Elemental conditioning inputs matching
            ``FASTCHEM_GLOBAL_LABELS``.
        gravity_cm_s2 : jax.Array or None, optional
            Optional gravity profile. It is validated for shape compatibility
            but not consumed by the current FastChem bundle contract.

        Returns
        -------
        jax.Array
            Linear VMR array with shape ``(nz, n_species)`` in top-to-bottom
            order.
        """
        temperatures = jnp.asarray(temperatures_k, dtype=jnp.float32)
        pressures = jnp.asarray(pressures_bar, dtype=jnp.float32)
        _require_matching_level_shapes(temperatures=temperatures, pressures=pressures)
        if gravity_cm_s2 is not None:
            gravity = jnp.asarray(gravity_cm_s2, dtype=jnp.float32)
            if gravity.ndim != 1 or gravity.shape != temperatures.shape:
                raise ValueError("gravity_cm_s2 must be a 1-D array sharing the PT grid shape.")
            del gravity

        internal_temperatures = temperatures[::-1]
        internal_pressures = pressures[::-1]
        vmr_internal = bundle.predict_fastchem_profile(
            pressure_bar=internal_pressures,
            temperature_k=internal_temperatures,
            global_inputs=global_inputs,
        )
        return vmr_internal[::-1, :]

    return vmr_fn, species_labels


def make_vulcan_vmr_fn(
    bundle: ExportedJAXModel,
) -> tuple[Any, list[str]]:
    """Create an ExoJAX-compatible VULCAN profile inference callable.

    Parameters
    ----------
    bundle : ExportedJAXModel
        Exported VULCAN emulator bundle with embedded physical-unit
        preprocessing.

    Returns
    -------
    tuple[Any, list[str]]
        Callable ``vmr_fn`` plus the ordered output-species labels. The
        callable returns linear VMR predictions with shape ``(nz, n_species)``
        in top-to-bottom level order.
    """
    if not bundle.uses_vulcan_chemistry:
        raise ValueError("make_vulcan_vmr_fn requires a vulcan bundle.")

    _validate_vulcan_bundle_global_order(bundle)
    species_labels = list(bundle.data_contract["output_species_order"])

    def vmr_fn(
        temperatures_k: jax.Array,  # shape: (nz,), top -> bottom
        pressures_bar: jax.Array,  # shape: (nz,), top -> bottom
        kzz_cm2_s: jax.Array,  # shape: (nz,), top -> bottom
        global_inputs: dict[str, Any] | jax.Array,
        spectrum_flux: jax.Array,  # shape: (spectrum_dim,)
    ) -> jax.Array:
        """Predict one VULCAN composition profile in ExoJAX ordering.

        Parameters
        ----------
        temperatures_k : jax.Array
            Temperature profile in Kelvin with shape ``(nz,)`` ordered from
            top to bottom.
        pressures_bar : jax.Array
            Pressure profile in bar with shape ``(nz,)`` ordered from top to
            bottom.
        kzz_cm2_s : jax.Array
            Eddy-diffusion profile in ``cm^2 s^-1`` with shape ``(nz,)``.
        global_inputs : dict[str, Any] or jax.Array
            Global conditioning inputs matching ``VULCAN_GLOBAL_LABELS``.
        spectrum_flux : jax.Array
            Stellar spectrum sampled on the bundle's fixed wavelength grid,
            shape ``(spectrum_dim,)``.

        Returns
        -------
        jax.Array
            Linear VMR array with shape ``(nz, n_species)`` in top-to-bottom
            order.
        """
        temperatures = jnp.asarray(temperatures_k, dtype=jnp.float32)
        pressures = jnp.asarray(pressures_bar, dtype=jnp.float32)
        kzz = jnp.asarray(kzz_cm2_s, dtype=jnp.float32)
        _require_matching_level_shapes(temperatures=temperatures, pressures=pressures)
        if kzz.ndim != 1 or kzz.shape != temperatures.shape:
            raise ValueError("kzz_cm2_s must be a 1-D array sharing the PT grid shape.")
        if isinstance(global_inputs, dict) and "gravity_cm_s2" in global_inputs:
            _maybe_require_column_constant(global_inputs["gravity_cm_s2"], name="gravity_cm_s2")

        internal_temperatures = temperatures[::-1]
        internal_pressures = pressures[::-1]
        internal_kzz = kzz[::-1]
        vmr_internal = bundle.predict_vulcan_profile(
            pressure_bar=internal_pressures,
            temperature_k=internal_temperatures,
            kzz_cm2_s=internal_kzz,
            global_inputs=global_inputs,
            spectrum_flux=spectrum_flux,
        )
        return vmr_internal[::-1, :]

    return vmr_fn, species_labels
