"""Portable JAX export helpers with embedded physical-unit preprocessing.

This module provides two main capabilities:

1. **Checkpoint export**: Convert a training run directory into a portable
   NPZ bundle that embeds model parameters, dimensions, normalization
   metadata, data contract, and config — everything needed for inference.
   Bundles are identified by their structural metadata; stale bundles
   whose structure no longer matches the current loader fail with a
   ``KeyError`` on missing metadata and must be re-exported.

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
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..constants import (
    NEAR_CONSTANT_STD_THRESHOLD,
    NORM_SPAN_FLOOR,
)
from .jax_model import (
    TransformerDimensions,
    apply_transformer_model,
)


def _flatten_params(tree: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten a nested parameter tree into dotted-key arrays."""
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
    """Restore contiguous ``{"0": ..., "1": ...}`` mappings to list form."""
    if isinstance(node, dict):
        converted = {key: _convert_numeric_dicts(value) for key, value in node.items()}
        if converted and all(key.isdigit() for key in converted):
            indices = sorted(int(key) for key in converted)
            if indices == list(range(len(indices))):
                return [converted[str(idx)] for idx in indices]
        return converted
    return node


def _unflatten_params(flat_params: dict[str, np.ndarray]) -> Any:
    """Reconstruct the nested parameter tree from dotted NPZ keys."""
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
    """Normalize sequence-static features in one vectorized pass.

    Each feature column is normalized by the corresponding block in
    ``payload["blocks"]``. Supported methods reduce to a common form
    ``y = (x_work - offset) / scale`` where ``x_work`` is either ``x`` or
    ``log10(max(x, floor))`` depending on the method.

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
    blocks = payload["blocks"]

    def _scalar(value: Any) -> float:
        return float(np.asarray(value).reshape(-1)[0])

    offsets: list[float] = []
    scales: list[float] = []
    floors: list[float] = []
    use_log: list[bool] = []
    for block in blocks:
        method = str(block["method"]).lower()
        if method == "none":
            offsets.append(0.0)
            scales.append(1.0)
            floors.append(1.0)
            use_log.append(False)
        elif method == "standard":
            offsets.append(_scalar(block["mean"]))
            scales.append(_scalar(block["std"]))
            floors.append(1.0)
            use_log.append(False)
        elif method == "log-standard":
            offsets.append(_scalar(block["mean"]))
            scales.append(_scalar(block["std"]))
            floors.append(_scalar(block["floor"]))
            use_log.append(True)
        elif method == "log-minmax":
            offsets.append(_scalar(block["log10_min"]))
            scales.append(_scalar(block["span"]))
            floors.append(_scalar(block["floor"]))
            use_log.append(True)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    offset_arr = jnp.asarray(offsets, dtype=arr.dtype)
    scale_arr = jnp.asarray(scales, dtype=arr.dtype)
    floor_arr = jnp.asarray(floors, dtype=arr.dtype)
    use_log_arr = jnp.asarray(use_log, dtype=jnp.bool_)
    x_work = jnp.where(use_log_arr, jnp.log10(jnp.maximum(arr, floor_arr)), arr)
    return (x_work - offset_arr) / scale_arr


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


def _validate_near_constant_standard_globals(
    values: dict[str, float] | jax.Array | np.ndarray,
    feature_order: list[str],
    normalization_block: dict[str, Any],
    *,
    field_name: str,
    std_threshold: float = NEAR_CONSTANT_STD_THRESHOLD,
    atol: float = NEAR_CONSTANT_STD_THRESHOLD,
) -> None:
    """Reject eager inputs that change fixed training-time global features."""
    methods = list(normalization_block.get("methods", []))
    means = list(normalization_block.get("mean", []))
    stds = list(normalization_block.get("std", []))
    if not methods or len(methods) != len(feature_order):
        return

    if isinstance(values, dict):
        missing = [name for name in feature_order if name not in values]
        if missing:
            return
        if any(_is_traced_array(values[name]) for name in feature_order):
            return
        vector = np.asarray([values[name] for name in feature_order], dtype=np.float64)
    else:
        if _is_traced_array(values):
            return
        vector = np.asarray(values, dtype=np.float64)
        if vector.ndim != 1 or vector.shape[0] != len(feature_order):
            return

    for idx, name in enumerate(feature_order):
        if str(methods[idx]).lower() != "standard":
            continue
        std = float(stds[idx])
        if std > std_threshold:
            continue
        mean = float(means[idx])
        value = float(vector[idx])
        if np.isclose(value, mean, atol=atol, rtol=0.0):
            continue
        raise ValueError(
            f"{field_name}[{name}] must equal {mean:.8g} within {atol:.1e} "
            f"because this exported model was trained with {name} fixed; got {value:.8g}."
        )


def _is_traced_array(value: Any) -> bool:
    """Return whether an input is a JAX tracer inside a transformed context."""
    return isinstance(value, jax.core.Tracer)


def _log10_pressure_union_bounds(
    data_contract: dict[str, Any],
) -> tuple[float, float]:
    """Return the training-range log10(P in bar) lower/upper bounds."""
    lo, hi = data_contract["log10_pressure_bar_union_range"]
    return float(lo), float(hi)


def _validate_pressure_grid(
    pressure_bar: jax.Array | np.ndarray,
    data_contract: dict[str, Any],
) -> None:
    """Reject out-of-range pressure grids before the jitted forward runs.

    Runs eagerly on concrete arrays; traced inputs inside ``jax.jit`` are
    skipped so the check never appears in the JAX graph.
    """
    if _is_traced_array(pressure_bar):
        return
    arr = np.asarray(pressure_bar, dtype=np.float64)
    if arr.ndim != 1:
        return
    nz = int(arr.shape[0])
    nl_range = data_contract.get("num_levels_range")
    if nl_range is not None:
        nl_lo, nl_hi = int(nl_range[0]), int(nl_range[1])
        if not (nl_lo <= nz <= nl_hi):
            raise ValueError(
                f"pressure_bar length {nz} is outside the training num_levels range "
                f"[{nl_lo}, {nl_hi}]."
            )
    if "log10_pressure_bar_union_range" not in data_contract:
        return
    if np.any(arr <= 0):
        raise ValueError("pressure_bar must contain strictly positive values.")
    log_p = np.log10(arr)
    lo, hi = _log10_pressure_union_bounds(data_contract)
    if log_p.min() < lo - 1.0e-9 or log_p.max() > hi + 1.0e-9:
        raise ValueError(
            f"pressure_bar range [{arr.min():.3e}, {arr.max():.3e}] bar is outside "
            f"the training pressure range [{10.0 ** lo:.3e}, {10.0 ** hi:.3e}] bar."
        )


def _position_coord_from_pressure_jax(
    pressure_bar: jax.Array,
    data_contract: dict[str, Any],
) -> jax.Array:
    """Build a normalized log10(P) position coordinate for the continuous PE."""
    lo, hi = _log10_pressure_union_bounds(data_contract)
    span = jnp.asarray(max(hi - lo, NORM_SPAN_FLOOR), dtype=jnp.float32)
    log_p = jnp.log10(jnp.asarray(pressure_bar, dtype=jnp.float32))
    return (log_p - jnp.asarray(lo, dtype=jnp.float32)) / span


def _predict_fastchem_profile_impl(
    *,
    params: Any,
    dims: TransformerDimensions,
    normalization: dict[str, Any],
    data_contract: dict[str, Any],
    pressure_bar: jax.Array | np.ndarray,
    temperature_k: jax.Array | np.ndarray,
    global_inputs: dict[str, float] | jax.Array | np.ndarray,
    return_log10: bool,
) -> jax.Array:
    """Run FastChem inference from physical inputs using shared bundle logic.

    Parameters
    ----------
    params : Any
        Nested JAX parameter tree for the model.
    dims : TransformerDimensions
        Architecture dimensionality constants.
    normalization : dict[str, Any]
        Full normalization payload (sequence_static, global_static, target).
    data_contract : dict[str, Any]
        Data contract specifying feature orders and target shape.
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

    position_coord = _position_coord_from_pressure_jax(pressure, data_contract)[None, :]
    attention_mask = jnp.ones(sequence.shape[:2], dtype=jnp.bool_)
    pred_norm, _ = apply_transformer_model(
        params,
        sequence,
        globals_norm,
        dims,
        position_coord=position_coord,
        attention_mask=attention_mask,
    )
    pred_norm = pred_norm[0]
    if return_log10:
        return _restore_block_transform_space_jax(pred_norm, normalization["target"])
    return inverse_block_jax(pred_norm, normalization["target"])


def _predict_vulcan_profile_impl(
    *,
    params: Any,
    dims: TransformerDimensions,
    normalization: dict[str, Any],
    data_contract: dict[str, Any],
    pressure_bar: jax.Array | np.ndarray,
    temperature_k: jax.Array | np.ndarray,
    kzz_cm2_s: jax.Array | np.ndarray,
    global_inputs: dict[str, float] | jax.Array | np.ndarray,
    return_log10: bool,
) -> jax.Array:
    """Run VULCAN inference from physical inputs using shared bundle logic.

    Parameters
    ----------
    params : Any
        Nested JAX parameter tree for the model.
    dims : TransformerDimensions
        Architecture dimensionality constants.
    normalization : dict[str, Any]
        Full normalization payload (sequence_static, global_static, target).
    data_contract : dict[str, Any]
        Data contract specifying feature orders and target shape.
    pressure_bar : jax.Array or np.ndarray
        Pressure grid in bar, shape ``(nz,)``.
    temperature_k : jax.Array or np.ndarray
        Temperature profile in Kelvin, shape ``(nz,)``.
    kzz_cm2_s : jax.Array or np.ndarray
        Eddy diffusion profile in cm^2 s^-1, shape ``(nz,)``.
    global_inputs : dict[str, float] or array-like
        Global conditioning scalars (planetary, elemental, stellar, toggles).
    return_log10 : bool
        If True, return log10 mixing ratios instead of linear.

    Returns
    -------
    jax.Array
        Predicted mixing ratios, shape ``(nz, target_dim)``.
    """
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
        normalization["sequence_static"],
    )[None, :, :]
    global_vector = _ordered_feature_vector(
        global_inputs,
        list(data_contract["global_static_feature_order"]),
        field_name="global_inputs",
    )
    globals_norm = apply_mixed_block_jax(
        global_vector[None, :],
        normalization["global_static"],
    )

    position_coord = _position_coord_from_pressure_jax(pressure, data_contract)[None, :]
    attention_mask = jnp.ones(sequence.shape[:2], dtype=jnp.bool_)
    pred_norm, _ = apply_transformer_model(
        params,
        sequence,
        globals_norm,
        dims,
        position_coord=position_coord,
        attention_mask=attention_mask,
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
        "chemistry_type": str(payload["config"]["chemistry_type"]),
        "model_type": str(payload["config"]["model_type"]),
        "model_dimensions": json.dumps(payload["model_dimensions"], default=str),
        "normalization": json.dumps(payload["normalization"], default=str),
        "data_contract": json.dumps(payload["data_contract"], default=str),
        "config": json.dumps(payload["config"], default=str),
    }
    np.savez(
        destination,
        **{f"params/{key}": value for key, value in flat_params.items()},
        **{f"meta/{key}": np.array(value) for key, value in metadata.items()},
    )
    return destination


def load_run_payload(
    run_root: str | Path,
    *,
    which: Literal["best", "last"] = "best",
) -> dict[str, Any]:
    """Load a training run's params + metadata + config from the flat layout.

    Expects ``run_root`` to contain ``config.json``, ``metadata.json``, and
    ``params_{best,last}.npz`` (written by the trainer).
    """
    root = Path(run_root).resolve()
    params_file = f"params_{which}.npz"
    with np.load(root / params_file) as archive:
        flat = {key: archive[key] for key in archive.files}
    params = _unflatten_params(flat)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    return {"params": params, "config": config, **metadata}


def export_checkpoint_to_npz(
    run_root: str | Path,
    *,
    which: Literal["best", "last"] = "best",
    output_path: str | Path | None = None,
) -> Path:
    """Load a training run and export it as a portable NPZ bundle.

    The bundle includes all model parameters, architecture dimensions,
    normalization metadata, data contract, and config — everything needed
    for standalone inference without the training codebase.

    Parameters
    ----------
    run_root : str or Path
        Path to the run directory (e.g. ``models/<run_name>/``) containing
        ``config.json``, ``metadata.json``, and ``params_{best,last}.npz``.
    which : {"best", "last"}
        Which params file to export.
    output_path : str, Path, or None
        Destination path for the NPZ file. If None, writes to
        ``<run_root>/<which>_exported.npz``.

    Returns
    -------
    Path
        Path to the written NPZ bundle.
    """
    root = Path(run_root).resolve()
    payload = load_run_payload(root, which=which)
    destination = (
        Path(output_path) if output_path is not None else root / f"{which}_exported.npz"
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
) -> tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Parse bundle metadata from an exported NPZ file handle."""
    chemistry_type = str(arrays["meta/chemistry_type"].item())
    model_type = str(arrays["meta/model_type"].item())
    model_dimensions = json.loads(str(arrays["meta/model_dimensions"].item()))
    normalization = json.loads(str(arrays["meta/normalization"].item()))
    data_contract = json.loads(str(arrays["meta/data_contract"].item()))
    config = json.loads(str(arrays["meta/config"].item()))
    return (
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
    dims: TransformerDimensions
    normalization: dict[str, Any]
    data_contract: dict[str, Any]
    config: dict[str, Any]
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
    def uses_transformer(self: "ExportedJAXModel") -> bool:
        """Report whether the exported bundle wraps the FiLM Transformer model.

        Returns
        -------
        bool
            ``True`` when ``model_type`` equals ``"transformer"``.
        """
        return self.model_type == "transformer"

    @property
    def species(self: "ExportedJAXModel") -> list[str]:
        """Return the ordered species labels predicted by the bundled model."""
        return list(self.data_contract["output_species_order"])

    @property
    def fixed_globals(self: "ExportedJAXModel") -> dict[str, float]:
        """Return global inputs held constant during training.

        Returns a dict mapping feature names to their required constant
        values.  Features whose training-set standard deviation fell below
        ``NEAR_CONSTANT_STD_THRESHOLD`` are considered fixed; passing a
        different value will raise ``ValueError`` at prediction time.
        """
        feature_order = list(self.data_contract.get("global_static_feature_order", []))
        norm = self.normalization.get("global_static", {})
        methods = list(norm.get("methods", []))
        means = list(norm.get("mean", []))
        stds = list(norm.get("std", []))
        result: dict[str, float] = {}
        for idx, name in enumerate(feature_order):
            if idx >= len(methods):
                break
            if (
                str(methods[idx]).lower() == "standard"
                and float(stds[idx]) < NEAR_CONSTANT_STD_THRESHOLD
            ):
                result[name] = float(means[idx])
        return result

    def species_index(self: "ExportedJAXModel", name: str) -> int:
        """Return the column index for a named output species."""
        labels = list(self.data_contract["output_species_order"])
        try:
            return labels.index(name)
        except ValueError:
            raise ValueError(
                f"Species '{name}' not found. Available: {labels}"
            ) from None

    @cached_property
    def _compiled_fastchem_predictors(
        self: "ExportedJAXModel",
    ) -> dict[bool, Callable[[Any, Any, Any], jax.Array]]:
        """Cache compiled FastChem predictors keyed by the log10 flag."""
        predictors: dict[bool, Callable[[Any, Any, Any], jax.Array]] = {}
        for return_log10 in (False, True):
            compiled = jax.jit(
                lambda pressure_bar, temperature_k, global_inputs, _flag=return_log10: _predict_fastchem_profile_impl(
                    params=self.params,
                    dims=self.dims,
                    normalization=self.normalization,
                    data_contract=self.data_contract,
                    pressure_bar=pressure_bar,
                    temperature_k=temperature_k,
                    global_inputs=global_inputs,
                    return_log10=_flag,
                )
            )

            def predictor(
                pressure_bar: Any,
                temperature_k: Any,
                global_inputs: Any,
                _compiled: Callable[[Any, Any, Any], jax.Array] = compiled,
            ) -> jax.Array:
                _validate_pressure_grid(pressure_bar, self.data_contract)
                _validate_near_constant_standard_globals(
                    global_inputs,
                    list(self.data_contract["global_static_feature_order"]),
                    self.normalization["global_static"],
                    field_name="global_inputs",
                )
                return _compiled(pressure_bar, temperature_k, global_inputs)

            predictors[return_log10] = predictor
        return predictors

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
        return self._compiled_fastchem_predictors[bool(return_log10)]

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
            ``{"He_H": 7.84e-2, "C_H": 3.3e-4, "O_H": 4.9e-4, "N_H": 7.8e-5, "S_H": 1.6e-5}``).
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
        _validate_pressure_grid(pressure_bar, self.data_contract)
        _validate_near_constant_standard_globals(
            global_inputs,
            list(self.data_contract["global_static_feature_order"]),
            self.normalization["global_static"],
            field_name="global_inputs",
        )
        return _predict_fastchem_profile_impl(
            params=self.params,
            dims=self.dims,
            normalization=self.normalization,
            data_contract=self.data_contract,
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
        return_log10: bool = False,
    ) -> jax.Array:
        """Run VULCAN inference directly from physical-unit inputs.

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
        return_log10 : bool
            If True, return log10 mixing ratios instead of linear.

        Returns
        -------
        jax.Array
            Predicted mixing ratios, shape ``(nz, target_dim)``.
        """
        if not self.uses_vulcan_chemistry:
            raise ValueError("predict_vulcan_profile requires a vulcan export bundle.")
        _validate_pressure_grid(pressure_bar, self.data_contract)
        _validate_near_constant_standard_globals(
            global_inputs,
            list(self.data_contract["global_static_feature_order"]),
            self.normalization["global_static"],
            field_name="global_inputs",
        )
        return _predict_vulcan_profile_impl(
            params=self.params,
            dims=self.dims,
            normalization=self.normalization,
            data_contract=self.data_contract,
            pressure_bar=pressure_bar,
            temperature_k=temperature_k,
            kzz_cm2_s=kzz_cm2_s,
            global_inputs=global_inputs,
            return_log10=return_log10,
        )

    # ------------------------------------------------------------------
    # Short-name aliases kept for backward compatibility with callers that
    # used the pre-consolidation ``standalone_inference.ExportedModel`` API.
    # ------------------------------------------------------------------

    def predict_fastchem(
        self: "ExportedJAXModel",
        pressure_bar: jax.Array | np.ndarray,
        temperature_k: jax.Array | np.ndarray,
        global_inputs: dict[str, float] | jax.Array | np.ndarray,
        *,
        return_log10: bool = False,
    ) -> jax.Array:
        """Alias for :meth:`predict_fastchem_profile` with positional inputs."""
        return self.predict_fastchem_profile(
            pressure_bar=pressure_bar,
            temperature_k=temperature_k,
            global_inputs=global_inputs,
            return_log10=return_log10,
        )

    def predict_vulcan(
        self: "ExportedJAXModel",
        pressure_bar: jax.Array | np.ndarray,
        temperature_k: jax.Array | np.ndarray,
        kzz_cm2_s: jax.Array | np.ndarray,
        global_inputs: dict[str, float] | jax.Array | np.ndarray,
        *,
        return_log10: bool = False,
    ) -> jax.Array:
        """Alias for :meth:`predict_vulcan_profile` with positional inputs."""
        return self.predict_vulcan_profile(
            pressure_bar=pressure_bar,
            temperature_k=temperature_k,
            kzz_cm2_s=kzz_cm2_s,
            global_inputs=global_inputs,
            return_log10=return_log10,
        )

    def make_compiled_fastchem_predictor(
        self: "ExportedJAXModel",
        *,
        return_log10: bool = False,
    ) -> Callable[[Any, Any, Any], jax.Array]:
        """Alias for :meth:`make_compiled_fastchem_profile_predictor`."""
        return self.make_compiled_fastchem_profile_predictor(return_log10=return_log10)


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
    if chemistry_type not in {"fastchem", "vulcan"} or model_type != "transformer":
        raise ValueError(
            f"Unsupported chemistry_type={chemistry_type!r} or model_type={model_type!r}."
        )
    dims = TransformerDimensions.from_dict(model_dimensions)
    return ExportedJAXModel(
        params=params,
        dims=dims,
        normalization=normalization,
        data_contract=data_contract,
        config=config,
        chemistry_type=chemistry_type,
        model_type=model_type,
    )
