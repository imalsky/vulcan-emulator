"""Standalone JAX inference module for VULCAN and FastChem emulator bundles.

Self-contained: imports only JAX, NumPy, and the standard library.
Embedded inside every exported .npz bundle under ``meta/vulcan_emulator_src``.

NOTE: This module intentionally duplicates constants and primitives from
``src/constants.py``, ``src/models/layers.py``, ``src/models/transformer.py``,
and ``src/models/export_bundle.py`` so that bundles load without the training
codebase installed. When changing any of those upstream modules, mirror the
change here. Do NOT add imports from the ``src/`` tree into this file.

Public API
----------
    model = load_model("best_exported.npz")
    vmr   = model.predict_fastchem(pressure_bar, temperature_k, global_inputs)
    vmr   = model.predict_vulcan(pressure_bar, temperature_k, kzz_cm2_s,
                                  global_inputs)
    vmr_fn, species = make_fastchem_vmr_fn(model)   # ExoJAX top-to-bottom API
    vmr_fn, species = make_vulcan_vmr_fn(model)      # ExoJAX top-to-bottom API

Loading from a bundle
---------------------
    import sys, types, numpy as np
    _src = bytes(np.load("bundle.npz", allow_pickle=False)["meta/vulcan_emulator_src"]).decode()
    _mod = types.ModuleType("vulcan_emulator")
    sys.modules["vulcan_emulator"] = _mod   # required for @dataclass type resolution
    exec(compile(_src, "vulcan_emulator", "exec"), _mod.__dict__)
    load_model = _mod.load_model
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Section 2: Constants
# ---------------------------------------------------------------------------

FASTCHEM_GLOBAL_LABELS: list[str] = ["He_H", "C_H", "O_H", "N_H", "S_H"]

# ---------------------------------------------------------------------------
# Section 3: JAX model dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransformerDimensions:
    """All dimensionality and architecture hyper-parameters for the surrogate."""

    sequence_dim: int
    global_dim: int
    target_dim: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    conditioning_hidden_dim: int
    film_clamp: float
    output_head_divisor: int
    activation: str = "gelu"
    dropout_rate: float = 0.0

    def to_dict(self: "TransformerDimensions") -> dict[str, Any]:
        """Serialize the dataclass fields into a plain Python mapping."""
        return asdict(self)

    @classmethod
    def from_dict(
        cls: type["TransformerDimensions"],
        payload: dict[str, Any],
    ) -> "TransformerDimensions":
        """Rebuild one dimensions dataclass from serialized metadata."""
        return cls(**payload)


# ---------------------------------------------------------------------------
# Section 4: JAX helper functions
# ---------------------------------------------------------------------------

_SINUSOIDAL_BASE_WAVELENGTH = 10_000.0
_POSITION_SCALE = 64.0


def _linear(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    """Apply a dense affine transform while preserving leading batch axes."""
    leading_shape = x.shape[:-1]
    x_2d = x.reshape((-1, x.shape[-1]))
    y_2d = x_2d @ params["weight"] + params["bias"]
    return y_2d.reshape(leading_shape + (params["bias"].shape[0],))


def _apply_dropout(
    x: jax.Array,
    *,
    rate: float,
    key: jax.Array | None,
    training: bool,
) -> jax.Array:
    """Apply inverted-dropout masking during training and pass through otherwise."""
    if not training or rate <= 0.0 or key is None:
        return x
    keep_prob = 1.0 - float(rate)
    mask = jax.random.bernoulli(key, p=keep_prob, shape=x.shape)
    return jnp.where(mask, x / keep_prob, jnp.zeros_like(x))


def _layer_norm(params: dict[str, jax.Array], x: jax.Array, eps: float = 1.0e-5) -> jax.Array:
    """Apply learned affine layer normalization across the last dimension."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    normalized = (x - mean) / jnp.sqrt(var + eps)
    return normalized * params["scale"] + params["bias"]


def sinusoidal_position_encoding_continuous(
    position: jax.Array,
    dim: int,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Continuous Vaswani-style PE over a real-valued coordinate."""
    scaled = position.astype(dtype) * jnp.asarray(_POSITION_SCALE, dtype=dtype)
    index = jnp.arange(dim, dtype=dtype)
    angle_rate = 1.0 / jnp.power(
        _SINUSOIDAL_BASE_WAVELENGTH,
        (2.0 * jnp.floor(index / 2.0)) / float(dim),
    )
    angle = scaled[..., None] * angle_rate
    even_mask = (jnp.arange(dim) % 2) == 0
    return jnp.where(even_mask, jnp.sin(angle), jnp.cos(angle))


def _resolve_activation(name: str) -> Callable[[jax.Array], jax.Array]:
    """Resolve an activation name to the corresponding JAX callable."""
    activations = {
        "elu": jax.nn.elu,
        "gelu": jax.nn.gelu,
        "leaky_relu": jax.nn.leaky_relu,
        "relu": jax.nn.relu,
        "selu": jax.nn.selu,
        "silu": jax.nn.silu,
        "softplus": jax.nn.softplus,
        "tanh": jnp.tanh,
    }
    try:
        return activations[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


def _multihead_attention_qkv(
    q_params: dict[str, jax.Array],
    k_params: dict[str, jax.Array],
    v_params: dict[str, jax.Array],
    o_params: dict[str, jax.Array],
    query: jax.Array,
    source: jax.Array,
    *,
    nhead: int,
    source_mask: jax.Array | None = None,
) -> jax.Array:
    """Scaled dot-product attention from ``query`` tokens over ``source`` tokens."""
    batch_size, query_len, d_model = query.shape
    source_len = source.shape[1]
    head_dim = d_model // nhead

    q = _linear(q_params, query).reshape(batch_size, query_len, nhead, head_dim).transpose(0, 2, 1, 3)
    k = _linear(k_params, source).reshape(batch_size, source_len, nhead, head_dim).transpose(0, 2, 1, 3)
    v = _linear(v_params, source).reshape(batch_size, source_len, nhead, head_dim).transpose(0, 2, 1, 3)

    scale = 1.0 / math.sqrt(float(head_dim))
    logits = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
    if source_mask is not None:
        expanded_mask = source_mask[:, None, None, :]
        logits = jnp.where(expanded_mask, logits, jnp.full_like(logits, -1.0e30))
    weights = jax.nn.softmax(logits, axis=-1)
    if source_mask is not None:
        expanded_mask = source_mask[:, None, None, :].astype(weights.dtype)
        weights = weights * expanded_mask
        weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1.0e-8)
    attended = jnp.einsum("bhij,bhjd->bhid", weights, v)
    attended = attended.transpose(0, 2, 1, 3).reshape(batch_size, query_len, d_model)
    return _linear(o_params, attended)


def _multihead_attention(
    params: dict[str, jax.Array],
    x: jax.Array,
    *,
    nhead: int,
    key_mask: jax.Array | None = None,
) -> jax.Array:
    """Convenience wrapper for bidirectional self-attention."""
    return _multihead_attention_qkv(
        params["q"],
        params["k"],
        params["v"],
        params["o"],
        x,
        x,
        nhead=nhead,
        source_mask=key_mask,
    )


def apply_transformer_model(
    params: dict[str, Any],
    sequence_inputs: jax.Array,
    global_inputs: jax.Array,
    dims: TransformerDimensions,
    *,
    position_coord: jax.Array,
    attention_mask: jax.Array | None = None,
    dropout_key: jax.Array | None = None,
    training: bool = False,
) -> tuple[jax.Array, dict[str, jax.Array | None]]:
    """Run the Transformer forward pass in normalized space."""
    if sequence_inputs.ndim != 3:
        raise ValueError("sequence_inputs must have shape [batch, nz, feature_dim].")
    if global_inputs.ndim != 2:
        raise ValueError("global_inputs must have shape [batch, global_dim].")
    if position_coord.ndim != 2:
        raise ValueError("position_coord must have shape [batch, nz].")
    if position_coord.shape[0] != sequence_inputs.shape[0] or position_coord.shape[1] != sequence_inputs.shape[1]:
        raise ValueError("position_coord must match sequence_inputs in (batch, nz).")
    if attention_mask is not None:
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, nz].")
        if attention_mask.shape[0] != sequence_inputs.shape[0] or attention_mask.shape[1] != sequence_inputs.shape[1]:
            raise ValueError("attention_mask must match sequence_inputs in (batch, nz).")
        attention_mask = attention_mask.astype(jnp.bool_)

    act = _resolve_activation(dims.activation)
    _keys_per_layer = 2
    layer_dropout_keys: list[tuple[jax.Array | None, ...]] = [
        tuple(None for _ in range(_keys_per_layer)) for _ in range(dims.num_layers)
    ]
    output_dropout_key: jax.Array | None = None
    if training and dims.dropout_rate > 0.0 and dropout_key is not None:
        total_dropout_keys = (dims.num_layers * _keys_per_layer) + 1
        dropout_keys = iter(jax.random.split(dropout_key, total_dropout_keys))
        layer_dropout_keys = [
            tuple(next(dropout_keys) for _ in range(_keys_per_layer))
            for _ in range(dims.num_layers)
        ]
        output_dropout_key = next(dropout_keys)

    context = act(_linear(params["context_in"], global_inputs))
    film = _linear(params["context_out"], context)
    film = film.reshape(sequence_inputs.shape[0], dims.num_layers, 2, dims.d_model)

    x = _linear(params["sequence_in"], sequence_inputs)
    x = x + sinusoidal_position_encoding_continuous(position_coord, dims.d_model, x.dtype)

    for layer_index, (layer, layer_keys) in enumerate(zip(params["layers"], layer_dropout_keys)):
        attn_dropout_key = layer_keys[0]
        ff_dropout_key = layer_keys[1]

        gamma = jnp.clip(film[:, layer_index, 0], -dims.film_clamp, dims.film_clamp)
        beta = jnp.clip(film[:, layer_index, 1], -dims.film_clamp, dims.film_clamp)

        # (a) Self-attention
        x_norm = _layer_norm(layer["ln1"], x)
        attn = _multihead_attention(layer, x_norm, nhead=dims.nhead, key_mask=attention_mask)
        attn = _apply_dropout(
            attn,
            rate=dims.dropout_rate,
            key=attn_dropout_key,
            training=training,
        )
        x = x + attn

        # (b) FiLM conditioning
        x = x * (1.0 + gamma[:, None, :]) + beta[:, None, :]

        # (c) FFN
        ff_in = _layer_norm(layer["ln_ffn"], x)
        ff_hidden = act(_linear(layer["ff1"], ff_in))
        ff_hidden = _apply_dropout(
            ff_hidden,
            rate=dims.dropout_rate,
            key=ff_dropout_key,
            training=training,
        )
        ff = _linear(layer["ff2"], ff_hidden)
        x = x + ff

    x = _layer_norm(params["out_norm"], x)
    x = act(_linear(params["out_head_hidden"], x))
    x = _apply_dropout(
        x,
        rate=dims.dropout_rate,
        key=output_dropout_key,
        training=training,
    )
    pred = _linear(params["out_head_final"], x)
    return pred, {}


# ---------------------------------------------------------------------------
# Section 5: Normalization utilities
# ---------------------------------------------------------------------------

def _restore_block_transform_space_jax(
    x: jax.Array,
    block: dict[str, Any],
) -> jax.Array:
    """Undo only the affine part of a normalization block in JAX."""
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
    """Apply one normalization block using JAX arrays."""
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
    """Invert one normalization block from model space back to physical units."""
    restored = _restore_block_transform_space_jax(jnp.asarray(x, dtype=jnp.float32), block)
    if str(block["method"]).lower() in {"log-standard", "log-minmax"}:
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
        if method == "log-minmax":
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
    """Resolve one ordered feature vector from dict or array inputs."""
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
    std_threshold: float = 1.0e-6,
    atol: float = 1.0e-6,
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
    """Return the training-range log10(P in bar) bounds recorded at export time."""
    lo, hi = data_contract["log10_pressure_bar_union_range"]
    return float(lo), float(hi)


def _num_levels_training_range(data_contract: dict[str, Any]) -> tuple[int, int]:
    """Return the training-time ``num_levels`` range."""
    lo, hi = data_contract["num_levels_range"]
    return int(lo), int(hi)


def _position_coord_from_pressure_jax(
    pressure_bar: jax.Array,
    data_contract: dict[str, Any],
) -> jax.Array:
    """Build a normalized log10(P) position coordinate for the continuous PE."""
    lo, hi = _log10_pressure_union_bounds(data_contract)
    span = jnp.asarray(max(hi - lo, 1.0e-12), dtype=jnp.float32)
    log_p = jnp.log10(jnp.asarray(pressure_bar, dtype=jnp.float32))
    return (log_p - jnp.asarray(lo, dtype=jnp.float32)) / span


def _validate_pressure_grid(
    pressure_bar: jax.Array | np.ndarray,
    data_contract: dict[str, Any],
) -> None:
    """Reject out-of-range pressure grids before invoking the jitted forward.

    Runs eagerly on concrete arrays; traced inputs (inside ``jax.jit``) are
    skipped so the check never appears in the JAX graph.
    """
    if _is_traced_array(pressure_bar):
        return
    arr = np.asarray(pressure_bar, dtype=np.float64)
    if arr.ndim != 1:
        return
    nz = int(arr.shape[0])
    nl_lo, nl_hi = _num_levels_training_range(data_contract)
    if not (nl_lo <= nz <= nl_hi):
        raise ValueError(
            f"pressure_bar length {nz} is outside the training num_levels range "
            f"[{nl_lo}, {nl_hi}]."
        )
    if np.any(arr <= 0):
        raise ValueError("pressure_bar must contain strictly positive values.")
    log_p = np.log10(arr)
    lo, hi = _log10_pressure_union_bounds(data_contract)
    if log_p.min() < lo - 1.0e-9 or log_p.max() > hi + 1.0e-9:
        raise ValueError(
            f"pressure_bar range [{arr.min():.3e}, {arr.max():.3e}] bar is outside "
            f"the training pressure range [{10.0 ** lo:.3e}, {10.0 ** hi:.3e}] bar."
        )


# ---------------------------------------------------------------------------
# Section 6: Parameter tree utilities
# ---------------------------------------------------------------------------

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
    """Convert integer-keyed dict nodes back into Python lists."""
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


# ---------------------------------------------------------------------------
# Section 7: Bundle loading utilities
# ---------------------------------------------------------------------------

def _resolve_device(device: str | jax.Device | None) -> jax.Device | None:
    """Normalize an optional device specifier into a concrete JAX device."""
    if device is None:
        return None
    if isinstance(device, str):
        return jax.devices(device)[0]
    return device


def _parse_export_metadata(
    arrays: np.lib.npyio.NpzFile,
) -> tuple[str, int, str, str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Parse bundle metadata from an exported NPZ file handle."""
    export_format = str(arrays["meta/export_format"].item())
    export_version = int(str(arrays["meta/export_version"].item()))
    chemistry_type = str(arrays["meta/chemistry_type"].item())
    model_type = str(arrays["meta/model_type"].item())
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


# ---------------------------------------------------------------------------
# Section 8: FastChem inference implementation
# ---------------------------------------------------------------------------

def _predict_fastchem_impl(
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
    """Run FastChem inference from physical inputs using shared bundle logic."""
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


# ---------------------------------------------------------------------------
# Section 9: ExportedModel class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExportedModel:
    """Portable JAX model bundle with physical-units inference helpers."""

    params: Any
    dims: TransformerDimensions
    normalization: dict[str, Any]
    data_contract: dict[str, Any]
    config: dict[str, Any]
    export_format: str
    export_version: int
    chemistry_type: str
    model_type: str

    @property
    def uses_fastchem(self: "ExportedModel") -> bool:
        """Report whether the exported bundle predicts FastChem chemistry outputs."""
        return self.chemistry_type == "fastchem"

    @property
    def uses_vulcan_chemistry(self: "ExportedModel") -> bool:
        """Report whether the exported bundle predicts VULCAN chemistry outputs."""
        return self.chemistry_type == "vulcan"

    @property
    def uses_transformer(self: "ExportedModel") -> bool:
        """Report whether the exported bundle wraps the FiLM-conditioned Transformer."""
        return self.model_type == "transformer"

    @property
    def species(self: "ExportedModel") -> list[str]:
        """Return the ordered species labels predicted by the bundled model."""
        return list(self.data_contract["output_species_order"])

    @property
    def fixed_globals(self: "ExportedModel") -> dict[str, float]:
        """Return global inputs held constant during training.

        Returns a dict mapping feature names to their required constant
        values.  Features whose training-set standard deviation fell below
        ``1e-6`` are considered fixed; passing a different value will raise
        ``ValueError`` at prediction time.
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
            if str(methods[idx]).lower() == "standard" and float(stds[idx]) < 1.0e-6:
                result[name] = float(means[idx])
        return result

    def species_index(self: "ExportedModel", name: str) -> int:
        """Return the column index for a named output species."""
        labels = list(self.data_contract["output_species_order"])
        try:
            return labels.index(name)
        except ValueError:
            raise ValueError(
                f"Species '{name}' not found. Available: {labels}"
            ) from None

    @cached_property
    def _compiled_fastchem_predictor(
        self: "ExportedModel",
    ) -> Callable[[Any, Any, Any], jax.Array]:
        """Cache a compiled FastChem predictor for repeated linear-space calls."""
        compiled = jax.jit(
            lambda pressure_bar, temperature_k, global_inputs: _predict_fastchem_impl(
                params=self.params,
                dims=self.dims,
                normalization=self.normalization,
                data_contract=self.data_contract,
                pressure_bar=pressure_bar,
                temperature_k=temperature_k,
                global_inputs=global_inputs,
                return_log10=False,
            )
        )

        def predictor(pressure_bar: Any, temperature_k: Any, global_inputs: Any) -> jax.Array:
            _validate_pressure_grid(pressure_bar, self.data_contract)
            _validate_near_constant_standard_globals(
                global_inputs,
                list(self.data_contract["global_static_feature_order"]),
                self.normalization["global_static"],
                field_name="global_inputs",
            )
            return compiled(pressure_bar, temperature_k, global_inputs)

        return predictor

    @cached_property
    def _compiled_fastchem_predictor_log10(
        self: "ExportedModel",
    ) -> Callable[[Any, Any, Any], jax.Array]:
        """Cache a compiled FastChem predictor for repeated log10-space calls."""
        compiled = jax.jit(
            lambda pressure_bar, temperature_k, global_inputs: _predict_fastchem_impl(
                params=self.params,
                dims=self.dims,
                normalization=self.normalization,
                data_contract=self.data_contract,
                pressure_bar=pressure_bar,
                temperature_k=temperature_k,
                global_inputs=global_inputs,
                return_log10=True,
            )
        )

        def predictor(pressure_bar: Any, temperature_k: Any, global_inputs: Any) -> jax.Array:
            _validate_pressure_grid(pressure_bar, self.data_contract)
            _validate_near_constant_standard_globals(
                global_inputs,
                list(self.data_contract["global_static_feature_order"]),
                self.normalization["global_static"],
                field_name="global_inputs",
            )
            return compiled(pressure_bar, temperature_k, global_inputs)

        return predictor

    def make_compiled_fastchem_predictor(
        self: "ExportedModel",
        *,
        return_log10: bool = False,
    ) -> Callable[[Any, Any, Any], jax.Array]:
        """Return a cached JIT-compiled FastChem predictor for repeated calls."""
        if not self.uses_fastchem:
            raise ValueError("make_compiled_fastchem_predictor requires a fastchem export bundle.")
        if return_log10:
            return self._compiled_fastchem_predictor_log10
        return self._compiled_fastchem_predictor

    def predict_fastchem(
        self: "ExportedModel",
        pressure_bar: jax.Array | np.ndarray,
        temperature_k: jax.Array | np.ndarray,
        global_inputs: dict[str, float] | jax.Array | np.ndarray,
        *,
        return_log10: bool = False,
    ) -> jax.Array:
        """Run FastChem inference directly from physical-unit inputs.

        Parameters
        ----------
        pressure_bar : array-like
            Pressure grid in bar, shape ``(nz,)``.
        temperature_k : array-like
            Temperature profile in Kelvin, shape ``(nz,)``.
        global_inputs : dict or array-like
            Global conditioning scalars. Elemental entries (``He_H``,
            ``C_H``, ``O_H``, ``N_H``, ``S_H``) are hydrogen-normalized
            number fractions ``n_X / n_H`` (ratio of element-X atoms to
            H atoms, matching VULCAN's ``O_H``, ``C_H`` convention) — not
            mass fractions, not molecular volume fractions.
        return_log10 : bool
            If True, return log10 mixing ratios instead of linear.

        Returns
        -------
        jax.Array
            Predicted mixing ratios, shape ``(nz, target_dim)``.
        """
        if not self.uses_fastchem:
            raise ValueError("predict_fastchem requires a fastchem export bundle.")
        _validate_pressure_grid(pressure_bar, self.data_contract)
        _validate_near_constant_standard_globals(
            global_inputs,
            list(self.data_contract["global_static_feature_order"]),
            self.normalization["global_static"],
            field_name="global_inputs",
        )
        return _predict_fastchem_impl(
            params=self.params,
            dims=self.dims,
            normalization=self.normalization,
            data_contract=self.data_contract,
            pressure_bar=pressure_bar,
            temperature_k=temperature_k,
            global_inputs=global_inputs,
            return_log10=return_log10,
        )

    def predict_vulcan(
        self: "ExportedModel",
        pressure_bar: jax.Array | np.ndarray,
        temperature_k: jax.Array | np.ndarray,
        kzz_cm2_s: jax.Array | np.ndarray,
        global_inputs: dict[str, float] | jax.Array | np.ndarray,
        *,
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
            Global conditioning scalars. Elemental entries (``He_H``,
            ``C_H``, ``O_H``, ``N_H``, ``S_H``) are hydrogen-normalized
            number fractions ``n_X / n_H`` (ratio of element-X atoms to
            H atoms, matching VULCAN's ``O_H``, ``C_H`` convention) — not
            mass fractions, not molecular volume fractions.
        return_log10 : bool
            If True, return log10 mixing ratios instead of linear.

        Returns
        -------
        jax.Array
            Predicted mixing ratios, shape ``(nz, target_dim)``.
        """
        if not self.uses_vulcan_chemistry:
            raise ValueError("predict_vulcan requires a vulcan export bundle.")
        _validate_pressure_grid(pressure_bar, self.data_contract)
        _validate_near_constant_standard_globals(
            global_inputs,
            list(self.data_contract["global_static_feature_order"]),
            self.normalization["global_static"],
            field_name="global_inputs",
        )

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

        position_coord = _position_coord_from_pressure_jax(pressure, self.data_contract)[None, :]
        attention_mask = jnp.ones(sequence.shape[:2], dtype=jnp.bool_)
        pred_norm, _ = apply_transformer_model(
            self.params,
            sequence,
            globals_norm,
            self.dims,
            position_coord=position_coord,
            attention_mask=attention_mask,
        )
        pred_norm = pred_norm[0]
        if return_log10:
            return _restore_block_transform_space_jax(pred_norm, self.normalization["target"])
        return inverse_block_jax(pred_norm, self.normalization["target"])


# ---------------------------------------------------------------------------
# Section 10: load_model function
# ---------------------------------------------------------------------------

def load_model(
    bundle_path: str | Path,
    *,
    device: str | jax.Device | None = None,
) -> ExportedModel:
    """Load an exported NPZ bundle and place parameters onto the requested device."""
    bundle = Path(bundle_path)
    with np.load(bundle, allow_pickle=False) as arrays:
        (export_format, export_version, chemistry_type, model_type,
         model_dimensions, normalization, data_contract, config) = _parse_export_metadata(arrays)
        flat_params = {
            name.split("/", 1)[1]: np.asarray(arrays[name])
            for name in arrays.files if name.startswith("params/")
        }
    params = _unflatten_params(flat_params)
    target_device = _resolve_device(device)
    if target_device is not None:
        params = jax.tree_util.tree_map(lambda x: jax.device_put(jnp.asarray(x), target_device), params)
    else:
        params = jax.tree_util.tree_map(jnp.asarray, params)
    chemistry_type = chemistry_type or str(data_contract.get("chemistry_type", "")).lower()
    model_type = model_type or str(data_contract.get("model_type", "")).lower()
    if chemistry_type not in {"fastchem", "vulcan"} or model_type != "transformer":
        raise ValueError(
            f"Unsupported chemistry_type={chemistry_type!r} or model_type={model_type!r}."
        )
    dims = TransformerDimensions.from_dict(model_dimensions)
    return ExportedModel(
        params=params, dims=dims, normalization=normalization,
        data_contract=data_contract, config=config,
        export_format=export_format, export_version=export_version,
        chemistry_type=chemistry_type, model_type=model_type,
    )


# ---------------------------------------------------------------------------
# Section 11: ExoJAX wrappers
# ---------------------------------------------------------------------------

def make_fastchem_vmr_fn(
    model: ExportedModel,
    *,
    pressure_order: str = "top_to_bottom",
) -> tuple[Any, list[str]]:
    """Create an ExoJAX-compatible FastChem profile inference callable.

    Parameters
    ----------
    model : ExportedModel
        Exported FastChem emulator bundle.
    pressure_order : ``"top_to_bottom"`` or ``"bottom_to_top"``
        Level ordering convention.  Default ``"top_to_bottom"`` means
        index 0 is the top of the atmosphere (lowest pressure).

    Returns
    -------
    tuple[Any, list[str]]
        Callable ``vmr_fn`` plus ordered species labels.
    """
    if pressure_order not in ("top_to_bottom", "bottom_to_top"):
        raise ValueError(
            f"pressure_order must be 'top_to_bottom' or 'bottom_to_top', got {pressure_order!r}"
        )
    if not model.uses_fastchem:
        raise ValueError("make_fastchem_vmr_fn requires a fastchem bundle.")

    feature_order = list(model.data_contract.get("global_static_feature_order", []))
    if feature_order != FASTCHEM_GLOBAL_LABELS:
        raise ValueError(
            "FastChem bundle global_static_feature_order must be "
            f"{FASTCHEM_GLOBAL_LABELS}; got {feature_order}."
        )

    compiled_predict = model.make_compiled_fastchem_predictor()
    _flip = pressure_order == "top_to_bottom"

    def vmr_fn(
        temperatures_k: jax.Array,
        pressures_bar: jax.Array,
        global_inputs: dict[str, Any] | jax.Array,
        gravity_cm_s2: jax.Array | None = None,
    ) -> jax.Array:
        """Predict FastChem VMRs on the configured pressure grid."""
        temperatures = jnp.asarray(temperatures_k, dtype=jnp.float32)
        pressures = jnp.asarray(pressures_bar, dtype=jnp.float32)
        if temperatures.ndim != 1 or pressures.ndim != 1:
            raise ValueError("temperatures_k and pressures_bar must be 1-D arrays.")
        if temperatures.shape != pressures.shape:
            raise ValueError("temperatures_k and pressures_bar must share the same shape.")
        if gravity_cm_s2 is not None:
            gravity = jnp.asarray(gravity_cm_s2, dtype=jnp.float32)
            if gravity.ndim != 1 or gravity.shape != temperatures.shape:
                raise ValueError("gravity_cm_s2 must be a 1-D array sharing the PT grid shape.")
            del gravity

        internal_temperatures = temperatures[::-1] if _flip else temperatures
        internal_pressures = pressures[::-1] if _flip else pressures
        vmr_internal = compiled_predict(internal_pressures, internal_temperatures, global_inputs)
        return vmr_internal[::-1, :] if _flip else vmr_internal

    return vmr_fn, list(model.data_contract["output_species_order"])


def make_vulcan_vmr_fn(
    model: ExportedModel,
    *,
    pressure_order: str = "top_to_bottom",
) -> tuple[Any, list[str]]:
    """Create an ExoJAX-compatible VULCAN profile inference callable.

    Parameters
    ----------
    model : ExportedModel
        Exported VULCAN emulator bundle.
    pressure_order : ``"top_to_bottom"`` or ``"bottom_to_top"``
        Level ordering convention.  Default ``"top_to_bottom"`` means
        index 0 is the top of the atmosphere (lowest pressure).

    Returns
    -------
    tuple[Any, list[str]]
        Callable ``vmr_fn`` plus ordered species labels.
    """
    if pressure_order not in ("top_to_bottom", "bottom_to_top"):
        raise ValueError(
            f"pressure_order must be 'top_to_bottom' or 'bottom_to_top', got {pressure_order!r}"
        )
    if not model.uses_vulcan_chemistry:
        raise ValueError("make_vulcan_vmr_fn requires a vulcan bundle.")

    _flip = pressure_order == "top_to_bottom"

    def vmr_fn(
        temperatures_k: jax.Array,
        pressures_bar: jax.Array,
        kzz_cm2_s: jax.Array,
        global_inputs: dict[str, Any] | jax.Array,
    ) -> jax.Array:
        """Predict VULCAN VMRs on the configured pressure grid."""
        temperatures = jnp.asarray(temperatures_k, dtype=jnp.float32)
        pressures = jnp.asarray(pressures_bar, dtype=jnp.float32)
        kzz = jnp.asarray(kzz_cm2_s, dtype=jnp.float32)
        if temperatures.ndim != 1 or pressures.ndim != 1:
            raise ValueError("temperatures_k and pressures_bar must be 1-D arrays.")
        if temperatures.shape != pressures.shape:
            raise ValueError("temperatures_k and pressures_bar must share the same shape.")
        if kzz.ndim != 1 or kzz.shape != temperatures.shape:
            raise ValueError("kzz_cm2_s must be a 1-D array sharing the PT grid shape.")

        internal_temperatures = temperatures[::-1] if _flip else temperatures
        internal_pressures = pressures[::-1] if _flip else pressures
        internal_kzz = kzz[::-1] if _flip else kzz
        vmr_internal = model.predict_vulcan(
            internal_pressures,
            internal_temperatures,
            internal_kzz,
            global_inputs,
        )
        return vmr_internal[::-1, :] if _flip else vmr_internal

    return vmr_fn, list(model.data_contract["output_species_order"])
