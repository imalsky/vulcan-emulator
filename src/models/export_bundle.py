"""Portable JAX export helpers with embedded physical-unit preprocessing.

This module provides two main capabilities:

1. **Checkpoint export**: Convert a training checkpoint (pickle) into a
   portable NPZ bundle (``jax_physical_bundle`` format, version 1).  The
   bundle embeds model parameters, dimensions, normalization metadata,
   data contract, and config — everything needed for standalone inference.

2. **Physical-unit inference**: ``ExportedJAXModel`` wraps the exported
   bundle and provides ``predict_fastchem_profile()`` and
   ``predict_vulcan_profile()`` methods that accept raw physical
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
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..data_generation.spectrum import pack_spectrum_tokens
from .jax_model import (
    MLPDimensions,
    TransformerDimensions,
    apply_mlp,
    apply_transformer_model,
)

EXPORT_FORMAT = "jax_physical_bundle"
EXPORT_VERSION = 4


def _flatten_params(tree: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten a nested parameter tree into dotted-key arrays.

    Parameters
    ----------
    tree : Any
        Nested dict/list/array parameter structure.
    prefix : str, default=""
        Dotted-key prefix accumulated during recursion.

    Returns
    -------
    dict[str, np.ndarray]
        Flat mapping from dotted parameter names to NumPy arrays suitable for
        NPZ serialization.
    """
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
    """Convert integer-keyed dict nodes back into Python lists.

    Parameters
    ----------
    node : Any
        Nested structure produced while rebuilding a flattened parameter tree.

    Returns
    -------
    Any
        Structure with contiguous ``{"0": ..., "1": ...}`` mappings restored
        to list form.
    """
    if isinstance(node, dict):
        converted = {key: _convert_numeric_dicts(value) for key, value in node.items()}
        if converted and all(key.isdigit() for key in converted):
            indices = sorted(int(key) for key in converted)
            if indices == list(range(len(indices))):
                return [converted[str(idx)] for idx in indices]
        return converted
    return node


def _unflatten_params(flat_params: dict[str, np.ndarray]) -> Any:
    """Reconstruct the nested parameter tree from dotted NPZ keys.

    Parameters
    ----------
    flat_params : dict[str, np.ndarray]
        Flat mapping produced by ``_flatten_params``.

    Returns
    -------
    Any
        Nested parameter tree matching the original dict/list structure.
    """
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
    """Undo only the affine part of a normalization block in JAX.

    Parameters
    ----------
    x : jax.Array
        Normalized values in model space.
    block : dict[str, Any]
        Normalization block containing ``method`` plus fitted statistics.
        Standard and log-standard blocks use ``mean`` and ``std``; log-minmax
        blocks use ``log10_min`` and ``span``.

    Returns
    -------
    jax.Array
        Values restored to pre-affine transform space. For log-standard
        blocks this means log10 space, not linear physical space.
    """
    arr = jnp.asarray(x, dtype=jnp.float32)
    method = str(block["method"]).lower()
    if method == "none":
        return arr
    if method == "log-minmax":
        log10_min = jnp.asarray(block["log10_min"], dtype=arr.dtype)
        span = jnp.asarray(block["span"], dtype=arr.dtype)
        return arr * span + log10_min
    mean = jnp.asarray(block["mean"], dtype=arr.dtype)
    std = jnp.asarray(block["std"], dtype=arr.dtype)
    if method in {"standard", "log-standard"}:
        return arr * std + mean
    raise ValueError(f"Unsupported normalization method: {method}")


def apply_block_jax(x: jax.Array | np.ndarray, block: dict[str, Any]) -> jax.Array:
    """Apply one normalization block using JAX arrays.

    Parameters
    ----------
    x : jax.Array or np.ndarray
        Input values whose last dimension matches the normalization block.
    block : dict[str, Any]
        Normalization block with method metadata and fitted statistics.

    Returns
    -------
    jax.Array
        Normalized values in model space.
    """
    arr = jnp.asarray(x, dtype=jnp.float32)
    method = str(block["method"]).lower()
    if method == "none":
        return arr
    if method == "log-minmax":
        floor = jnp.asarray(float(block["floor"]), dtype=arr.dtype)
        log10_min = jnp.asarray(block["log10_min"], dtype=arr.dtype)
        span = jnp.asarray(block["span"], dtype=arr.dtype)
        return (jnp.log10(jnp.maximum(arr, floor)) - log10_min) / span
    mean = jnp.asarray(block["mean"], dtype=arr.dtype)
    std = jnp.asarray(block["std"], dtype=arr.dtype)
    if method == "standard":
        return (arr - mean) / std
    if method == "log-standard":
        floor = jnp.asarray(float(block["floor"]), dtype=arr.dtype)
        return (jnp.log10(jnp.maximum(arr, floor)) - mean) / std
    raise ValueError(f"Unsupported normalization method: {method}")


def inverse_block_jax(x: jax.Array | np.ndarray, block: dict[str, Any]) -> jax.Array:
    """Invert one normalization block from model space back to physical units.

    Parameters
    ----------
    x : jax.Array or np.ndarray
        Normalized array whose last dimension matches the supplied
        normalization block.
    block : dict[str, Any]
        Normalization metadata containing the method name plus any per-feature
        statistics needed to undo the transform.

    Returns
    -------
    jax.Array
        Array with the same shape as ``x`` expressed in the original physical
        units.
    """
    restored = _restore_block_transform_space_jax(jnp.asarray(x, dtype=jnp.float32), block)
    if str(block["method"]).lower() in {"log-standard", "log-minmax"}:
        return jnp.power(10.0, restored)
    return restored


def apply_mixed_block_jax(x: jax.Array | np.ndarray, block: dict[str, Any]) -> jax.Array:
    """Apply a mixed per-feature normalization block in JAX.

    Parameters
    ----------
    x : jax.Array or np.ndarray
        Input array whose last dimension enumerates feature columns.
    block : dict[str, Any]
        Mixed normalization payload with per-column methods, means, stds, and
        optional floors.

    Returns
    -------
    jax.Array
        Normalized array with the same shape as ``x``.
    """
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
        if method == "log-minmax":
            # In the mixed-block format, _fit_mixed_block stores log10_min
            # as "mean" and span as "std", so the formula is identical to
            # log-standard despite the different semantic meaning.
            floor = jnp.asarray(float(block["floor"][idx]), dtype=arr.dtype)
            outputs.append((jnp.log10(jnp.maximum(column, floor)) - mean) / std)
            continue
        raise ValueError(f"Unsupported mixed normalization method: {method}")
    return jnp.stack(outputs, axis=-1)


def _apply_sequence_static_block_jax(
    sequence_static: jax.Array | np.ndarray,
    payload: dict[str, Any],
) -> jax.Array:
    """Normalize sequence-static features block by block in JAX.

    Parameters
    ----------
    sequence_static : jax.Array or np.ndarray
        Sequence feature tensor whose last dimension follows the payload's
        block order.
    payload : dict[str, Any]
        Sequence-static normalization payload containing one block per
        feature.

    Returns
    -------
    jax.Array
        Normalized sequence tensor with the same shape as ``sequence_static``.
    """
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
    """Resolve one ordered feature vector from dict or array inputs.

    Parameters
    ----------
    values : dict[str, float] or array-like
        Either a name-indexed mapping or a pre-ordered 1-D feature vector.
    feature_order : list[str]
        Required feature order for the downstream model contract.
    field_name : str
        Human-readable field name used in validation errors.

    Returns
    -------
    jax.Array
        One-dimensional feature vector in the requested order with shape
        ``(len(feature_order),)``.
    """
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


def _is_traced_array(value: Any) -> bool:
    """Return whether an input is a JAX tracer inside a transformed context."""
    return isinstance(value, jax.core.Tracer)


def _pack_native_spectrum_tokens_jax(
    wavelength_nm: jax.Array,
    flux_erg_cm2_s_nm: jax.Array,
    *,
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    max_tokens: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pad an already-valid native-grid spectrum without leaving JAX.

    This helper is intentionally narrow: it does not sanitize, clip, sort, or
    rebin traced inputs. The caller must provide a one-dimensional, strictly
    increasing native spectrum whose wavelength coverage already lies within
    ``[wavelength_min_nm, wavelength_max_nm]`` and whose length is at most
    ``max_tokens``.

    Parameters
    ----------
    wavelength_nm : jax.Array
        1-D wavelength grid in nanometers, strictly increasing.
    flux_erg_cm2_s_nm : jax.Array
        1-D flux array in erg cm-2 s-1 nm-1, same length as ``wavelength_nm``.
    wavelength_min_nm : float
        Minimum allowed wavelength (consumed by the caller for validation;
        unused here).
    wavelength_max_nm : float
        Maximum allowed wavelength (consumed by the caller for validation;
        unused here).
    max_tokens : int
        Fixed token-sequence length for the model; shorter inputs are
        zero-padded on the right.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        ``(padded_wavelength, padded_flux, mask)`` each with shape
        ``(max_tokens,)``.  The boolean mask is ``True`` for real tokens.
    """
    del wavelength_min_nm, wavelength_max_nm
    wavelength = jnp.asarray(wavelength_nm, dtype=jnp.float32)
    flux = jnp.asarray(flux_erg_cm2_s_nm, dtype=jnp.float32)
    if wavelength.ndim != 1 or flux.ndim != 1:
        raise ValueError("JAX-native VULCAN spectra must be 1-D arrays.")
    if wavelength.shape != flux.shape:
        raise ValueError("JAX-native VULCAN wavelength and flux arrays must share the same shape.")
    num_tokens = int(wavelength.shape[0])
    if num_tokens < 2:
        raise ValueError("JAX-native VULCAN spectra must contain at least two samples.")
    if num_tokens > int(max_tokens):
        raise ValueError(
            "JAX-native VULCAN spectra must satisfy len(spectrum) <= spectrum_max_tokens."
        )
    pad_width = int(max_tokens) - num_tokens
    padded_wavelength = jnp.pad(wavelength, (0, pad_width))
    padded_flux = jnp.pad(flux, (0, pad_width))
    mask = jnp.arange(int(max_tokens)) < num_tokens
    return padded_wavelength, padded_flux, mask


def _predict_fastchem_profile_impl(
    *,
    params: Any,
    dims: TransformerDimensions | MLPDimensions,
    normalization: dict[str, Any],
    data_contract: dict[str, Any],
    model_type: str,
    pressure_bar: jax.Array | np.ndarray,
    temperature_k: jax.Array | np.ndarray,
    global_inputs: dict[str, float] | jax.Array | np.ndarray,
    return_log10: bool,
) -> jax.Array:
    """Run FastChem inference from physical inputs using shared bundle logic.

    Normalizes the physical inputs, runs the forward pass through the
    selected model architecture, and inverse-normalizes the predictions
    back to physical space.

    Parameters
    ----------
    params : Any
        Nested JAX parameter tree for the model.
    dims : TransformerDimensions or MLPDimensions
        Architecture dimensionality constants.
    normalization : dict[str, Any]
        Full normalization payload (sequence_static, global_static, target).
    data_contract : dict[str, Any]
        Data contract specifying feature orders and target shape.
    model_type : str
        Either ``"mlp"`` or ``"transformer"``.
    pressure_bar : jax.Array or np.ndarray
        Pressure grid in bar, shape ``(nz,)``.
    temperature_k : jax.Array or np.ndarray
        Temperature profile in Kelvin, shape ``(nz,)``.
    global_inputs : dict[str, float] or array-like
        Global conditioning scalars (elemental abundances).
    return_log10 : bool
        If True, return log10 mixing ratios instead of linear.

    Returns
    -------
    jax.Array
        Predicted mixing ratios, shape ``(nz, target_dim)``.
    """
    pressure = jnp.asarray(pressure_bar, dtype=jnp.float32)
    temperature = jnp.asarray(temperature_k, dtype=jnp.float32)
    if pressure.ndim != 1 or temperature.ndim != 1 or pressure.shape != temperature.shape:
        raise ValueError("pressure_bar and temperature_k must be 1-D arrays with identical shapes.")

    static_inputs = jnp.stack([pressure, temperature], axis=-1)  # shape: (nz, 2)
    sequence = _apply_sequence_static_block_jax(
        static_inputs,
        normalization["sequence_static"],
    )[None, :, :]  # shape: (1, nz, 2)
    global_vector = _ordered_feature_vector(
        global_inputs,
        list(data_contract["global_static_feature_order"]),
        field_name="global_inputs",
    )
    globals_norm = apply_mixed_block_jax(
        global_vector[None, :],
        normalization["global_static"],
    )  # shape: (1, global_dim)

    if model_type == "mlp":
        pred_norm, _ = apply_mlp(
            params,
            sequence,
            globals_norm,
            dims,
            None,
            None,
            None,
        )
    else:
        pred_norm, _ = apply_transformer_model(
            params,
            sequence,
            globals_norm,
            None,
            None,
            None,
            dims,
        )
    pred_norm = pred_norm[0]
    if return_log10:
        return _restore_block_transform_space_jax(pred_norm, normalization["target"])
    return inverse_block_jax(pred_norm, normalization["target"])


def export_checkpoint_payload(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Write a training checkpoint payload to a portable NPZ bundle.

    Parameters
    ----------
    payload : dict[str, Any]
        Checkpoint dictionary containing model params, dimensions,
        normalization, data contract, and config.
    output_path : str or Path
        Destination ``.npz`` file.

    Returns
    -------
    Path
        Path to the written export bundle.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flat_params = _flatten_params(payload["params"])
    metadata = {
        "export_format": EXPORT_FORMAT,
        "export_version": str(EXPORT_VERSION),
        "chemistry_type": str(payload["config"]["chemistry_type"]),
        "model_type": str(payload["config"]["model_type"]),
        "model_dimensions": json.dumps(payload["model_dimensions"]),
        "normalization": json.dumps(payload["normalization"]),
        "data_contract": json.dumps(payload["data_contract"]),
        "config": json.dumps(payload["config"]),
    }
    standalone_src = (Path(__file__).parent / "standalone_inference.py").read_text(encoding="utf-8")
    np.savez(
        destination,
        **{f"params/{key}": value for key, value in flat_params.items()},
        **{f"meta/{key}": np.array(value) for key, value in metadata.items()},
        **{"meta/vulcan_emulator_src": np.frombuffer(standalone_src.encode("utf-8"), dtype=np.uint8)},
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
    """Normalize an optional device specifier into a concrete JAX device.

    Parameters
    ----------
    device : str, jax.Device, or None
        Requested device expressed either as ``None``, a platform name such as
        ``"cpu"`` or ``"gpu"``, or an already resolved ``jax.Device``.

    Returns
    -------
    jax.Device or None
        Concrete device handle suitable for ``jax.device_put``, or ``None`` to
        keep JAX's default placement.
    """
    if device is None:
        return None
    if isinstance(device, str):
        return jax.devices(device)[0]
    return device


def _parse_export_metadata(
    arrays: np.lib.npyio.NpzFile,
) -> tuple[str, int, str, str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Parse bundle metadata from an exported NPZ file handle.

    Parameters
    ----------
    arrays : np.lib.npyio.NpzFile
        Open NPZ bundle containing ``meta/*`` JSON payloads.

    Returns
    -------
    tuple[str, int, str, str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]
        Export format name, export version, chemistry type, model type, model
        dimensions, normalization payload, data contract, and config.
    """
    export_format = str(arrays["meta/export_format"].item()) if "meta/export_format" in arrays else "legacy_npz"
    export_version = int(str(arrays["meta/export_version"].item())) if "meta/export_version" in arrays else 0
    chemistry_type = str(arrays["meta/chemistry_type"].item()) if "meta/chemistry_type" in arrays else ""
    model_type = str(arrays["meta/model_type"].item()) if "meta/model_type" in arrays else ""
    model_dimensions = json.loads(str(arrays["meta/model_dimensions"].item()))
    normalization = json.loads(str(arrays["meta/normalization"].item()))
    data_contract = json.loads(str(arrays["meta/data_contract"].item()))
    config = json.loads(str(arrays["meta/config"].item()))
    return (
        export_format,
        export_version,
        chemistry_type,
        model_type,
        model_dimensions,
        normalization,
        data_contract,
        config,
    )


@dataclass(frozen=True)
class ExportedJAXModel:
    """Portable JAX model bundle with physical-units inference helpers."""

    params: Any
    dims: TransformerDimensions | MLPDimensions
    normalization: dict[str, Any]
    data_contract: dict[str, Any]
    config: dict[str, Any]
    export_format: str
    export_version: int
    chemistry_type: str
    model_type: str

    @property
    def uses_fastchem(self: "ExportedJAXModel") -> bool:
        """Report whether the exported bundle predicts FastChem chemistry outputs.

        Returns
        -------
        bool
            ``True`` when ``chemistry_type`` equals ``"fastchem"``.
        """
        return self.chemistry_type == "fastchem"

    @property
    def uses_vulcan_chemistry(self: "ExportedJAXModel") -> bool:
        """Report whether the exported bundle predicts VULCAN chemistry outputs.

        Returns
        -------
        bool
            ``True`` when ``chemistry_type`` equals ``"vulcan"``.
        """
        return self.chemistry_type == "vulcan"

    @property
    def uses_mlp(self: "ExportedJAXModel") -> bool:
        """Report whether the exported bundle wraps the FiLM-conditioned MLP.

        Returns
        -------
        bool
            ``True`` when ``model_type`` equals ``"mlp"``.
        """
        return self.model_type == "mlp"

    @property
    def uses_transformer(self: "ExportedJAXModel") -> bool:
        """Report whether the exported bundle wraps the FiLM Transformer model.

        Returns
        -------
        bool
            ``True`` when ``model_type`` equals ``"transformer"``.
        """
        return self.model_type == "transformer"

    @cached_property
    def _compiled_fastchem_profile_predictor(
        self: "ExportedJAXModel",
    ) -> Callable[[Any, Any, Any], jax.Array]:
        """Cache a compiled FastChem predictor for repeated linear-space calls."""
        return jax.jit(
            lambda pressure_bar, temperature_k, global_inputs: _predict_fastchem_profile_impl(
                params=self.params,
                dims=self.dims,
                normalization=self.normalization,
                data_contract=self.data_contract,
                model_type=self.model_type,
                pressure_bar=pressure_bar,
                temperature_k=temperature_k,
                global_inputs=global_inputs,
                return_log10=False,
            )
        )

    @cached_property
    def _compiled_fastchem_profile_predictor_log10(
        self: "ExportedJAXModel",
    ) -> Callable[[Any, Any, Any], jax.Array]:
        """Cache a compiled FastChem predictor for repeated log10-space calls."""
        return jax.jit(
            lambda pressure_bar, temperature_k, global_inputs: _predict_fastchem_profile_impl(
                params=self.params,
                dims=self.dims,
                normalization=self.normalization,
                data_contract=self.data_contract,
                model_type=self.model_type,
                pressure_bar=pressure_bar,
                temperature_k=temperature_k,
                global_inputs=global_inputs,
                return_log10=True,
            )
        )

    def make_compiled_fastchem_profile_predictor(
        self: "ExportedJAXModel",
        *,
        return_log10: bool = False,
    ) -> Callable[[Any, Any, Any], jax.Array]:
        """Return a cached JIT-compiled FastChem predictor for repeated calls.

        This keeps ``predict_fastchem_profile()`` eager, which avoids compile
        latency for one-off scripts, while exposing an explicitly compiled path
        for repeated inference workloads.

        Parameters
        ----------
        return_log10 : bool
            If True, the returned callable produces log10 mixing ratios;
            otherwise linear mixing ratios.

        Returns
        -------
        Callable[[array-like, array-like, array-like], jax.Array]
            JIT-compiled function with signature
            ``(pressure_bar, temperature_k, global_inputs) -> predictions``.
        """
        if not self.uses_fastchem:
            raise ValueError("make_compiled_fastchem_profile_predictor requires a fastchem export bundle.")
        if return_log10:
            return self._compiled_fastchem_profile_predictor_log10
        return self._compiled_fastchem_profile_predictor

    def predict_fastchem_profile(
        self: "ExportedJAXModel",
        *,
        pressure_bar: jax.Array | np.ndarray,
        temperature_k: jax.Array | np.ndarray,
        global_inputs: dict[str, float] | jax.Array | np.ndarray,
        return_log10: bool = False,
    ) -> jax.Array:
        """Run FastChem inference directly from physical-unit inputs.

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
            ``{"He_H": 8.38e-2, "C_H": 3.6e-4, "O_H": 5.37e-4, "N_H": 8.5e-5, "S_H": 1.8e-5}``).
            If an array, must have shape ``(global_dim,)`` in the correct order.
        return_log10 : bool
            If True, return log10 mixing ratios instead of linear.

        Returns
        -------
        jax.Array
            Predicted mixing ratios, shape ``(nz, target_dim)``.
        """
        if not self.uses_fastchem:
            raise ValueError("predict_fastchem_profile requires a fastchem export bundle.")
        return _predict_fastchem_profile_impl(
            params=self.params,
            dims=self.dims,
            normalization=self.normalization,
            data_contract=self.data_contract,
            model_type=self.model_type,
            pressure_bar=pressure_bar,
            temperature_k=temperature_k,
            global_inputs=global_inputs,
            return_log10=return_log10,
        )

    def predict_vulcan_profile(
        self: "ExportedJAXModel",
        *,
        pressure_bar: jax.Array | np.ndarray,
        temperature_k: jax.Array | np.ndarray,
        kzz_cm2_s: jax.Array | np.ndarray,
        global_inputs: dict[str, float] | jax.Array | np.ndarray,
        spectrum_wavelength_nm: jax.Array | np.ndarray,
        spectrum_flux_erg_cm2_s_nm: jax.Array | np.ndarray,
        return_log10: bool = False,
    ) -> jax.Array:
        """Run VULCAN inference directly from physical-unit inputs.

        Eager inputs may use arbitrary valid wavelength grids and are sanitized
        and packed through the NumPy preprocessing path. JAX-traced spectrum
        inputs must already be valid native-grid spectra with static
        ``len(spectrum) <= spectrum_max_tokens``; in that case the method pads
        them inside JAX without sorting, clipping, or rebinning.

        Parameters
        ----------
        pressure_bar : array-like
            Pressure grid in bar, shape ``(nz,)``.
        temperature_k : array-like
            Temperature profile in Kelvin, shape ``(nz,)``.
        kzz_cm2_s : array-like
            Vertical eddy diffusion coefficient in cm2 s-1, shape ``(nz,)``.
        global_inputs : dict or array-like
            Global conditioning scalars.  If a dict, keys must match
            ``data_contract["global_static_feature_order"]``.
            If an array, must have shape ``(global_dim,)`` in the correct order.
        spectrum_wavelength_nm : array-like
            Stellar spectrum wavelength grid in nanometers.
        spectrum_flux_erg_cm2_s_nm : array-like
            Stellar spectrum flux in erg cm-2 s-1 nm-1, same length as
            ``spectrum_wavelength_nm``.
        return_log10 : bool
            If True, return log10 mixing ratios instead of linear.

        Returns
        -------
        jax.Array
            Predicted mixing ratios, shape ``(nz, target_dim)``.
        """
        if not self.uses_vulcan_chemistry:
            raise ValueError("predict_vulcan_profile requires a vulcan export bundle.")

        pressure = jnp.asarray(pressure_bar, dtype=jnp.float32)
        temperature = jnp.asarray(temperature_k, dtype=jnp.float32)
        kzz = jnp.asarray(kzz_cm2_s, dtype=jnp.float32)
        if pressure.ndim != 1 or temperature.ndim != 1 or kzz.ndim != 1:
            raise ValueError("pressure_bar, temperature_k, and kzz_cm2_s must be 1-D arrays.")
        if not (pressure.shape == temperature.shape == kzz.shape):
            raise ValueError("pressure_bar, temperature_k, and kzz_cm2_s must share the same shape.")

        static_inputs = jnp.stack([pressure, temperature, kzz], axis=-1)
        sequence = _apply_sequence_static_block_jax(
            static_inputs,
            self.normalization["sequence_static"],
        )[None, :, :]

        global_vector = _ordered_feature_vector(
            global_inputs,
            list(self.data_contract["global_static_feature_order"]),
            field_name="global_inputs",
        )
        globals_norm = apply_mixed_block_jax(
            global_vector[None, :],
            self.normalization["global_static"],
        )

        if _is_traced_array(spectrum_wavelength_nm) or _is_traced_array(spectrum_flux_erg_cm2_s_nm):
            packed_wavelengths_nm, packed_fluxes, packed_mask = _pack_native_spectrum_tokens_jax(
                spectrum_wavelength_nm,
                spectrum_flux_erg_cm2_s_nm,
                wavelength_min_nm=float(self.data_contract["spectrum_wavelength_min_nm"]),
                wavelength_max_nm=float(self.data_contract["spectrum_wavelength_max_nm"]),
                max_tokens=int(self.data_contract["spectrum_max_tokens"]),
            )
            spectrum_wavelengths = packed_wavelengths_nm[None, :]
            spectrum_fluxes = packed_fluxes[None, :]
            spectrum_mask = packed_mask[None, :]
        else:
            wavelength_np = np.asarray(spectrum_wavelength_nm, dtype=np.float64).reshape(-1)
            flux_np = np.asarray(spectrum_flux_erg_cm2_s_nm, dtype=np.float64).reshape(-1)
            if wavelength_np.size != flux_np.size:
                raise ValueError(
                    "spectrum_wavelength_nm and spectrum_flux_erg_cm2_s_nm must have the same length."
                )
            packed_wavelengths_nm, packed_fluxes, packed_mask = pack_spectrum_tokens(
                wavelength_np,
                flux_np,
                wavelength_min_nm=float(self.data_contract["spectrum_wavelength_min_nm"]),
                wavelength_max_nm=float(self.data_contract["spectrum_wavelength_max_nm"]),
                max_tokens=int(self.data_contract["spectrum_max_tokens"]),
                dbin1_nm=float(self.data_contract.get("spectrum_dbin1_nm", 0.1)),
                dbin2_nm=float(self.data_contract.get("spectrum_dbin2_nm", 2.0)),
                dbin_12trans_nm=float(self.data_contract.get("spectrum_dbin_12trans_nm", 240.0)),
            )
            spectrum_wavelengths = jnp.asarray(packed_wavelengths_nm[None, :], dtype=jnp.float32)
            spectrum_fluxes = jnp.asarray(packed_fluxes[None, :], dtype=jnp.float32)
            spectrum_mask = jnp.asarray(packed_mask[None, :], dtype=bool)

        if self.uses_mlp:
            pred_norm, _ = apply_mlp(
                self.params,
                sequence,
                globals_norm,
                self.dims,
                spectrum_wavelengths,
                spectrum_fluxes,
                spectrum_mask,
            )
        else:
            pred_norm, _ = apply_transformer_model(
                self.params,
                sequence,
                globals_norm,
                spectrum_wavelengths,
                spectrum_fluxes,
                spectrum_mask,
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

    The bundle stores explicit chemistry/model metadata, so loading does not
    infer architecture from dimension keys.

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
        Ready-to-use model with ``predict_fastchem_profile()`` or
        ``predict_vulcan_profile()`` methods.
    """
    bundle = Path(bundle_path)
    with np.load(bundle, allow_pickle=False) as arrays:
        (
            export_format,
            export_version,
            chemistry_type,
            model_type,
            model_dimensions,
            normalization,
            data_contract,
            config,
        ) = _parse_export_metadata(arrays)
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
    chemistry_type = chemistry_type or str(data_contract.get("chemistry_type", "")).lower()
    model_type = model_type or str(data_contract.get("model_type", "")).lower()
    if chemistry_type not in {"fastchem", "vulcan"} or model_type not in {"mlp", "transformer"}:
        raise ValueError(
            "This export bundle uses the legacy task-based contract. Re-export a model trained "
            "with the chemistry_type × model_type config surface."
        )
    dims: TransformerDimensions | MLPDimensions
    if model_type == "mlp":
        dims = MLPDimensions.from_dict(model_dimensions)
    elif model_type == "transformer":
        dims = TransformerDimensions.from_dict(model_dimensions)
    else:
        raise ValueError(f"Unsupported exported model_type: {model_type!r}.")
    return ExportedJAXModel(
        params=params,
        dims=dims,
        normalization=normalization,
        data_contract=data_contract,
        config=config,
        export_format=export_format,
        export_version=export_version,
        chemistry_type=chemistry_type,
        model_type=model_type,
    )
