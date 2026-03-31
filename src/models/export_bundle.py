"""Portable JAX export helpers with embedded physical-unit preprocessing.

This module provides two main capabilities:

1. **Checkpoint export**: Convert a training checkpoint (pickle) into a
   portable NPZ bundle (``jax_physical_bundle`` format, version 1).  The
   bundle embeds model parameters, dimensions, normalization metadata,
   data contract, and config — everything needed for standalone inference.

2. **Physical-unit inference**: ``ExportedJAXModel`` wraps the exported
   bundle and provides ``predict_equilibrium_profile()`` and
   ``predict_transition_profile()`` methods that accept raw physical
   inputs (pressure in bar, temperature in K, etc.), normalize them
   internally, run the JAX forward pass, and return predictions in
   physical space (mixing ratios or log10 mixing ratios).

The NPZ layout uses ``params/<dotted.key>`` for model weight arrays and
``meta/<name>`` for JSON-encoded metadata strings.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .jax_model import (
    EquilibriumMLPDimensions,
    ModelDimensions,
    apply_equilibrium_mlp,
    apply_model,
)


EXPORT_FORMAT = "jax_physical_bundle"
EXPORT_VERSION = 1


def _flatten_params(tree: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten a nested param tree into ``{dotted.key: ndarray}`` pairs."""
    flat: dict[str, np.ndarray] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            flat.update(_flatten_params(value, prefix=f"{prefix}{key}."))
        return flat
    if isinstance(tree, list):
        for idx, value in enumerate(tree):
            flat.update(_flatten_params(value, prefix=f"{prefix}{idx}."))
        return flat
    flat[prefix.rstrip(".")] = np.asarray(tree)
    return flat


def _convert_numeric_dicts(node: Any) -> Any:
    """Convert dicts keyed by contiguous integer strings back into Python lists."""
    if isinstance(node, dict):
        converted = {key: _convert_numeric_dicts(value) for key, value in node.items()}
        if converted and all(key.isdigit() for key in converted):
            indices = sorted(int(key) for key in converted)
            if indices == list(range(len(indices))):
                return [converted[str(idx)] for idx in indices]
        return converted
    return node


def _unflatten_params(flat_params: dict[str, np.ndarray]) -> Any:
    """Reconstruct the nested param tree from dotted keys."""
    root: dict[str, Any] = {}
    for key, value in flat_params.items():
        cursor = root
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return _convert_numeric_dicts(root)


def _restore_block_transform_space_jax(
    x: jax.Array,
    block: dict[str, Any],
) -> jax.Array:
    """Undo only the affine part of a normalization block."""
    arr = jnp.asarray(x, dtype=jnp.float32)
    method = str(block["method"]).lower()
    if method == "none":
        return arr
    mean = jnp.asarray(block["mean"], dtype=arr.dtype)
    std = jnp.asarray(block["std"], dtype=arr.dtype)
    if method in {"standard", "log-standard"}:
        return arr * std + mean
    raise ValueError(f"Unsupported normalization method: {method}")


def apply_block_jax(x: jax.Array | np.ndarray, block: dict[str, Any]) -> jax.Array:
    """Apply one normalization block in JAX."""
    arr = jnp.asarray(x, dtype=jnp.float32)
    method = str(block["method"]).lower()
    if method == "none":
        return arr
    mean = jnp.asarray(block["mean"], dtype=arr.dtype)
    std = jnp.asarray(block["std"], dtype=arr.dtype)
    if method == "standard":
        return (arr - mean) / std
    if method == "log-standard":
        floor = jnp.asarray(float(block["floor"]), dtype=arr.dtype)
        return (jnp.log10(jnp.maximum(arr, floor)) - mean) / std
    raise ValueError(f"Unsupported normalization method: {method}")


def inverse_block_jax(x: jax.Array | np.ndarray, block: dict[str, Any]) -> jax.Array:
    """Invert one normalization block back to physical space in JAX."""
    restored = _restore_block_transform_space_jax(jnp.asarray(x, dtype=jnp.float32), block)
    if str(block["method"]).lower() == "log-standard":
        return jnp.power(10.0, restored)
    return restored


def apply_mixed_block_jax(x: jax.Array | np.ndarray, block: dict[str, Any]) -> jax.Array:
    """Apply a mixed per-feature normalization block in JAX."""
    arr = jnp.asarray(x, dtype=jnp.float32)
    outputs: list[jax.Array] = []
    for idx, method in enumerate(block["methods"]):
        column = arr[..., idx]
        if method == "none":
            outputs.append(column)
            continue
        mean = jnp.asarray(float(block["mean"][idx]), dtype=arr.dtype)
        std = jnp.asarray(float(block["std"][idx]), dtype=arr.dtype)
        if method == "standard":
            outputs.append((column - mean) / std)
            continue
        if method == "log-standard":
            floor = jnp.asarray(float(block["floor"][idx]), dtype=arr.dtype)
            outputs.append((jnp.log10(jnp.maximum(column, floor)) - mean) / std)
            continue
        raise ValueError(f"Unsupported mixed normalization method: {method}")
    return jnp.stack(outputs, axis=-1)


def _apply_sequence_static_block_jax(
    sequence_static: jax.Array | np.ndarray,
    payload: dict[str, Any],
) -> jax.Array:
    """Normalize sequence-static features block by block in JAX."""
    arr = jnp.asarray(sequence_static, dtype=jnp.float32)
    parts = [
        apply_block_jax(arr[..., idx : idx + 1], block)
        for idx, block in enumerate(payload["blocks"])
    ]
    return jnp.concatenate(parts, axis=-1)


def _ordered_feature_vector(
    values: dict[str, float] | jax.Array | np.ndarray,
    feature_order: list[str],
    *,
    field_name: str,
) -> jax.Array:
    """Resolve an ordered feature vector from a dict or a 1-D array."""
    if isinstance(values, dict):
        missing = [name for name in feature_order if name not in values]
        if missing:
            raise ValueError(f"{field_name} is missing required keys: {missing}.")
        return jnp.asarray([values[name] for name in feature_order], dtype=jnp.float32)

    arr = jnp.asarray(values, dtype=jnp.float32)
    if arr.ndim != 1 or arr.shape[0] != len(feature_order):
        raise ValueError(
            f"{field_name} must have shape ({len(feature_order)},), got {tuple(arr.shape)}."
        )
    return arr


def _normalize_log10_dt_s(dt_s: float | jax.Array | np.ndarray, block: dict[str, Any]) -> jax.Array:
    """Normalize a physical timestep by taking log10 before applying the fitted affine block."""
    dt = jnp.asarray(dt_s, dtype=jnp.float32)
    if dt.ndim != 0:
        raise ValueError(f"dt_s must be a scalar, got shape {tuple(dt.shape)}.")
    mean = jnp.asarray(float(block["mean"][0]), dtype=dt.dtype)
    std = jnp.asarray(float(block["std"][0]), dtype=dt.dtype)
    return (jnp.log10(jnp.maximum(dt, 1.0e-30)) - mean) / std


def export_checkpoint_payload(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Write one checkpoint payload to a portable NPZ bundle."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flat_params = _flatten_params(payload["params"])
    metadata = {
        "export_format": EXPORT_FORMAT,
        "export_version": str(EXPORT_VERSION),
        "model_dimensions": json.dumps(payload["model_dimensions"]),
        "normalization": json.dumps(payload["normalization"]),
        "data_contract": json.dumps(payload["data_contract"]),
        "config": json.dumps(payload["config"]),
    }
    np.savez(
        destination,
        **{f"params/{key}": value for key, value in flat_params.items()},
        **{f"meta/{key}": np.array(value) for key, value in metadata.items()},
    )
    return destination


def export_checkpoint_to_npz(
    checkpoint_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Load a training checkpoint and export it as a portable NPZ bundle.

    The bundle includes all model parameters, architecture dimensions,
    normalization metadata, data contract, and config — everything needed
    for standalone inference without the training codebase.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the pickle checkpoint (``best.pt`` or ``last.pt``).
    output_path : str, Path, or None
        Destination path for the NPZ file.  If None, defaults to
        ``<checkpoint_stem>_exported.npz`` in the same directory.

    Returns
    -------
    Path
        Path to the written NPZ bundle.
    """
    checkpoint = Path(checkpoint_path)
    with checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    destination = Path(output_path) if output_path is not None else checkpoint.with_name(
        f"{checkpoint.stem}_exported.npz"
    )
    return export_checkpoint_payload(payload, destination)


def _resolve_device(device: str | jax.Device | None) -> jax.Device | None:
    """Resolve an optional device specifier to a concrete JAX device."""
    if device is None:
        return None
    if isinstance(device, str):
        return jax.devices(device)[0]
    return device


def _parse_export_metadata(arrays: np.lib.npyio.NpzFile) -> tuple[str, int, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Parse the JSON metadata payload from an exported NPZ bundle."""
    export_format = str(arrays["meta/export_format"].item()) if "meta/export_format" in arrays else "legacy_npz"
    export_version = int(str(arrays["meta/export_version"].item())) if "meta/export_version" in arrays else 0
    model_dimensions = json.loads(str(arrays["meta/model_dimensions"].item()))
    normalization = json.loads(str(arrays["meta/normalization"].item()))
    data_contract = json.loads(str(arrays["meta/data_contract"].item()))
    config = json.loads(str(arrays["meta/config"].item()))
    return export_format, export_version, model_dimensions, normalization, data_contract, config


@dataclass(frozen=True)
class ExportedJAXModel:
    """Portable JAX model bundle with physical-units inference helpers."""

    params: Any
    dims: ModelDimensions | EquilibriumMLPDimensions
    normalization: dict[str, Any]
    data_contract: dict[str, Any]
    config: dict[str, Any]
    export_format: str
    export_version: int

    @property
    def is_equilibrium(self) -> bool:
        """Return whether this bundle wraps the equilibrium MLP."""
        return isinstance(self.dims, EquilibriumMLPDimensions)

    def predict_equilibrium_profile(
        self,
        *,
        pressure_bar: jax.Array | np.ndarray,
        temperature_k: jax.Array | np.ndarray,
        global_inputs: dict[str, float] | jax.Array | np.ndarray,
        return_log10: bool = False,
    ) -> jax.Array:
        """Run equilibrium inference directly from physical-unit inputs.

        Handles all normalization internally: sequence-static features are
        normalized per-column (log-standard for pressure, standard for
        temperature), global inputs are normalized via the mixed block,
        and predictions are inverse-normalized back to physical mixing
        ratios.

        Parameters
        ----------
        pressure_bar : array-like
            Pressure grid in bar, shape ``(nz,)``.
        temperature_k : array-like
            Temperature profile in Kelvin, shape ``(nz,)``.
        global_inputs : dict or array-like
            Global conditioning scalars.  If a dict, keys must match
            ``data_contract["global_static_feature_order"]`` (e.g.,
            ``{"He_H": 8.38e-2, "C_H": 2.95e-4, "O_H": 5.37e-4, "N_H": 7.08e-5, "S_H": 1.41e-5}``).
            If an array, must have shape ``(global_dim,)`` in the correct order.
        return_log10 : bool
            If True, return log10 mixing ratios instead of linear.

        Returns
        -------
        jax.Array
            Predicted mixing ratios, shape ``(nz, target_dim)``.
        """
        if not self.is_equilibrium:
            raise ValueError("predict_equilibrium_profile requires an equilibrium export bundle.")

        pressure = jnp.asarray(pressure_bar, dtype=jnp.float32)
        temperature = jnp.asarray(temperature_k, dtype=jnp.float32)
        if pressure.ndim != 1 or temperature.ndim != 1 or pressure.shape != temperature.shape:
            raise ValueError("pressure_bar and temperature_k must be 1-D arrays with identical shapes.")

        static_inputs = jnp.stack([pressure, temperature], axis=-1)  # shape: (nz, 2)
        sequence = _apply_sequence_static_block_jax(
            static_inputs,
            self.normalization["sequence_static"],
        )[None, :, :]  # shape: (1, nz, 2)
        global_vector = _ordered_feature_vector(
            global_inputs,
            list(self.data_contract["global_static_feature_order"]),
            field_name="global_inputs",
        )
        globals_norm = apply_mixed_block_jax(
            global_vector[None, :],
            self.normalization["global_static"],
        )  # shape: (1, global_dim)

        pred_norm, _ = apply_equilibrium_mlp(self.params, sequence, globals_norm, self.dims)
        pred_norm = pred_norm[0]
        if return_log10:
            return _restore_block_transform_space_jax(pred_norm, self.normalization["target"])
        return inverse_block_jax(pred_norm, self.normalization["target"])

    def predict_transition_profile(
        self,
        *,
        pressure_bar: jax.Array | np.ndarray,
        temperature_k: jax.Array | np.ndarray,
        kzz_cm2_s: jax.Array | np.ndarray,
        anchor_state: jax.Array | np.ndarray,
        global_inputs: dict[str, float] | jax.Array | np.ndarray,
        spectrum_flux: jax.Array | np.ndarray,
        dt_s: float | jax.Array | np.ndarray,
        return_log10: bool = False,
    ) -> jax.Array:
        """Run transition-model inference directly from physical-unit inputs.

        Bakes in all normalization: sequence-static features (pressure,
        temperature, Kzz), anchor mixing-ratio state, global conditioning
        (including log10(dt_s) insertion), and stellar spectrum normalization.
        Predictions are inverse-normalized back to physical mixing ratios.

        Parameters
        ----------
        pressure_bar : array-like
            Pressure grid in bar, shape ``(nz,)``.
        temperature_k : array-like
            Temperature profile in Kelvin, shape ``(nz,)``.
        kzz_cm2_s : array-like
            Eddy diffusion coefficient in cm^2/s, shape ``(nz,)``.
        anchor_state : array-like
            Anchor mixing ratios in physical space, shape ``(nz, state_dim)``.
        global_inputs : dict or array-like
            Global conditioning scalars (gravity, FastChem-native
            hydrogen-normalized elemental abundances ``n_X / n_H``, plus any
            physics toggles and atmosphere-base flags).
        spectrum_flux : array-like
            Stellar spectrum flux values, shape ``(spectrum_dim,)``.
        dt_s : float or scalar array
            Physical timestep in seconds (anchor to target).
        return_log10 : bool
            If True, return log10 mixing ratios instead of linear.

        Returns
        -------
        jax.Array
            Predicted mixing ratios, shape ``(nz, target_dim)``.
        """
        if self.is_equilibrium:
            raise ValueError("predict_transition_profile requires a transition export bundle.")

        pressure = jnp.asarray(pressure_bar, dtype=jnp.float32)
        temperature = jnp.asarray(temperature_k, dtype=jnp.float32)
        kzz = jnp.asarray(kzz_cm2_s, dtype=jnp.float32)
        state = jnp.asarray(anchor_state, dtype=jnp.float32)
        spectrum = jnp.asarray(spectrum_flux, dtype=jnp.float32)
        if pressure.ndim != 1 or temperature.ndim != 1 or kzz.ndim != 1:
            raise ValueError("pressure_bar, temperature_k, and kzz_cm2_s must be 1-D arrays.")
        if not (pressure.shape == temperature.shape == kzz.shape):
            raise ValueError("pressure_bar, temperature_k, and kzz_cm2_s must share the same shape.")
        if state.ndim != 2 or state.shape[0] != pressure.shape[0]:
            raise ValueError("anchor_state must have shape (nz, state_dim).")
        if spectrum.ndim != 1 or spectrum.shape[0] != int(self.data_contract["spectrum_dim"]):
            raise ValueError(
                "spectrum_flux must be a 1-D array with length "
                f"{int(self.data_contract['spectrum_dim'])}."
            )

        static_inputs = jnp.stack([pressure, temperature, kzz], axis=-1)  # shape: (nz, 3)
        static_norm = _apply_sequence_static_block_jax(
            static_inputs,
            self.normalization["sequence_static"],
        )
        state_norm = apply_block_jax(state, self.normalization["state"])
        sequence = jnp.concatenate([static_norm, state_norm], axis=-1)[None, :, :]

        global_vector = _ordered_feature_vector(
            global_inputs,
            list(self.data_contract["global_static_feature_order"]),
            field_name="global_inputs",
        )
        static_globals = apply_mixed_block_jax(
            global_vector,
            self.normalization["global_static"],
        )
        dt_feature = _normalize_log10_dt_s(dt_s, self.normalization["log10_dt_s"])[None]
        dt_index = int(self.data_contract["dt_feature_index"])
        globals_full = jnp.concatenate(
            [static_globals[:dt_index], dt_feature, static_globals[dt_index:]],
            axis=0,
        )[None, :]

        spectrum_norm = apply_block_jax(spectrum[None, :], self.normalization["spectrum"])
        pred_norm, _ = apply_model(
            self.params,
            sequence,
            globals_full,
            spectrum_norm,
            self.dims,
        )
        pred_norm = pred_norm[0]
        if return_log10:
            return _restore_block_transform_space_jax(pred_norm, self.normalization["target"])
        return inverse_block_jax(pred_norm, self.normalization["target"])


def load_exported_model(
    bundle_path: str | Path,
    *,
    device: str | jax.Device | None = None,
) -> ExportedJAXModel:
    """Load an exported NPZ bundle and return an ``ExportedJAXModel`` instance.

    Automatically detects whether the bundle contains an equilibrium MLP or
    a transition Transformer based on the presence of ``d_hidden`` in the
    serialized model dimensions.

    Parameters
    ----------
    bundle_path : str or Path
        Path to the ``.npz`` bundle file.
    device : str, jax.Device, or None
        Target JAX device (e.g., ``"cpu"``, ``"gpu"``).  If None, uses
        the JAX default placement.

    Returns
    -------
    ExportedJAXModel
        Ready-to-use model with ``predict_equilibrium_profile()`` or
        ``predict_transition_profile()`` methods.
    """
    bundle = Path(bundle_path)
    with np.load(bundle, allow_pickle=False) as arrays:
        export_format, export_version, model_dimensions, normalization, data_contract, config = _parse_export_metadata(
            arrays
        )
        flat_params = {
            name.split("/", 1)[1]: np.asarray(arrays[name])
            for name in arrays.files
            if name.startswith("params/")
        }

    params = _unflatten_params(flat_params)
    target_device = _resolve_device(device)
    params = jax.tree_util.tree_map(
        lambda x: jax.device_put(jnp.asarray(x), target_device) if target_device is not None else jnp.asarray(x),
        params,
    )
    dims: ModelDimensions | EquilibriumMLPDimensions
    if "d_hidden" in model_dimensions:
        dims = EquilibriumMLPDimensions.from_dict(model_dimensions)
    else:
        dims = ModelDimensions.from_dict(model_dimensions)
    return ExportedJAXModel(
        params=params,
        dims=dims,
        normalization=normalization,
        data_contract=data_contract,
        config=config,
        export_format=export_format,
        export_version=export_version,
    )
