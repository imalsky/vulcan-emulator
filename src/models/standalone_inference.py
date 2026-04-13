"""Standalone JAX inference module for VULCAN and FastChem emulator bundles.

Self-contained: imports only JAX, NumPy, and the standard library.
Embedded inside every exported .npz bundle under ``meta/vulcan_emulator_src``.

Public API
----------
    model = load_model("best_exported.npz")
    vmr   = model.predict_fastchem(pressure_bar, temperature_k, global_inputs)
    vmr   = model.predict_vulcan(pressure_bar, temperature_k, kzz_cm2_s,
                                  global_inputs, spectrum_wavelength_nm,
                                  spectrum_flux_erg_cm2_s_nm)
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
    spectrum_max_tokens: int
    target_dim: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    conditioning_hidden_dim: int
    film_clamp: float
    output_head_divisor: int
    spectrum_latent_dim: int
    spectrum_hidden_dim: int
    spectrum_num_latents: int
    spectrum_num_layers: int
    spectrum_num_heads: int
    spectrum_fourier_features: int
    spectrum_encoder_mode: str
    spectrum_floor: float = 1.0e-30
    activation: str = "gelu"
    dropout_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TransformerDimensions":
        return cls(**payload)


@dataclass(frozen=True)
class MLPDimensions:
    """Dimensionality and architecture hyper-parameters for the FiLM MLP."""

    sequence_dim: int
    global_dim: int
    spectrum_max_tokens: int
    spectrum_latent_dim: int
    spectrum_hidden_dim: int
    spectrum_num_latents: int
    spectrum_num_layers: int
    spectrum_num_heads: int
    spectrum_fourier_features: int
    spectrum_encoder_mode: str
    spectrum_floor: float
    target_dim: int
    d_hidden: int
    num_hidden_layers: int
    conditioning_hidden_dim: int
    film_clamp: float
    activation: str
    dropout_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MLPDimensions":
        return cls(**payload)


# ---------------------------------------------------------------------------
# Section 4: JAX helper functions
# ---------------------------------------------------------------------------

_SINUSOIDAL_BASE_WAVELENGTH = 10_000.0


def _linear(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
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
    if not training or rate <= 0.0 or key is None:
        return x
    keep_prob = 1.0 - float(rate)
    mask = jax.random.bernoulli(key, p=keep_prob, shape=x.shape)
    return jnp.where(mask, x / keep_prob, jnp.zeros_like(x))


def _layer_norm(params: dict[str, jax.Array], x: jax.Array, eps: float = 1.0e-5) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    normalized = (x - mean) / jnp.sqrt(var + eps)
    return normalized * params["scale"] + params["bias"]


def sinusoidal_position_encoding(length: int, dim: int, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    """Compute fixed sinusoidal positional encoding (Vaswani et al., 2017)."""
    position = jnp.arange(length, dtype=dtype)[:, None]
    index = jnp.arange(dim, dtype=dtype)[None, :]
    angle_rate = 1.0 / jnp.power(
        _SINUSOIDAL_BASE_WAVELENGTH,
        (2.0 * jnp.floor(index / 2.0)) / float(dim),
    )
    angle = position * angle_rate
    return jnp.where((jnp.arange(dim) % 2)[None, :] == 0, jnp.sin(angle), jnp.cos(angle))


def _masked_mean(
    x: jax.Array,
    mask: jax.Array,
    *,
    axis: int,
    keepdims: bool = False,
) -> jax.Array:
    weights = mask.astype(x.dtype)
    while weights.ndim < x.ndim:
        weights = weights[..., None]
    numerator = jnp.sum(x * weights, axis=axis, keepdims=keepdims)
    denominator = jnp.maximum(jnp.sum(weights, axis=axis, keepdims=keepdims), 1.0)
    return numerator / denominator


def _fourier_encode(values: jax.Array, *, num_features: int) -> jax.Array:
    if int(num_features) <= 0:
        return values[..., None]
    frequencies = jnp.power(
        2.0,
        jnp.arange(int(num_features), dtype=values.dtype),
    )
    phase = values[..., None] * frequencies * jnp.pi
    return jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)


def _resolve_activation(name: str):
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


def _prepare_spectrum_tokens(
    wavelengths_nm: jax.Array,
    fluxes_erg_cm2_s_nm: jax.Array,
    mask: jax.Array,
    dims: TransformerDimensions | MLPDimensions,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    valid_mask = mask.astype(bool)
    dtype = fluxes_erg_cm2_s_nm.dtype
    safe_wavelengths = jnp.where(
        valid_mask,
        jnp.maximum(wavelengths_nm, 1.0e-6),
        jnp.ones_like(wavelengths_nm),
    )
    safe_fluxes = jnp.where(
        valid_mask,
        jnp.maximum(fluxes_erg_cm2_s_nm, dims.spectrum_floor),
        jnp.full_like(fluxes_erg_cm2_s_nm, dims.spectrum_floor),
    )
    log_wavelength = jnp.log10(safe_wavelengths)
    log_flux = jnp.log10(safe_fluxes)

    log_flux_mean = _masked_mean(log_flux, valid_mask, axis=1, keepdims=True)
    log_flux_var = _masked_mean(
        (log_flux - log_flux_mean) ** 2,
        valid_mask,
        axis=1,
        keepdims=True,
    )
    log_flux_std = jnp.sqrt(jnp.maximum(log_flux_var, 1.0e-6))
    normalized_log_flux = (log_flux - log_flux_mean) / log_flux_std

    wavelength_step = jnp.concatenate(
        [
            log_wavelength[:, 1:] - log_wavelength[:, :-1],
            jnp.zeros_like(log_wavelength[:, :1]),
        ],
        axis=1,
    )
    wavelength_step = jnp.where(valid_mask, wavelength_step, jnp.zeros_like(wavelength_step))

    pair_mask = valid_mask[:, :-1] & valid_mask[:, 1:]
    pair_widths = jnp.maximum(safe_wavelengths[:, 1:] - safe_wavelengths[:, :-1], 0.0)
    pair_integrand = 0.5 * (safe_fluxes[:, 1:] + safe_fluxes[:, :-1]) * pair_widths
    integrated_flux = jnp.sum(pair_integrand * pair_mask.astype(dtype), axis=1, keepdims=True)

    inf_fill = jnp.full_like(log_wavelength, jnp.inf)
    ninf_fill = jnp.full_like(log_wavelength, -jnp.inf)
    log_wavelength_min = jnp.min(jnp.where(valid_mask, log_wavelength, inf_fill), axis=1, keepdims=True)
    log_wavelength_max = jnp.max(jnp.where(valid_mask, log_wavelength, ninf_fill), axis=1, keepdims=True)
    coverage_fraction = jnp.mean(valid_mask.astype(dtype), axis=1, keepdims=True)

    summary = jnp.concatenate(
        [
            log_flux_mean,
            log_flux_std,
            jnp.log10(jnp.maximum(integrated_flux, dims.spectrum_floor)),
            coverage_fraction,
            log_wavelength_min,
            log_wavelength_max,
        ],
        axis=-1,
    )

    wavelength_embedding = _fourier_encode(
        log_wavelength,
        num_features=int(dims.spectrum_fourier_features),
    )
    token_features = jnp.concatenate(
        [
            log_wavelength[..., None],
            wavelength_step[..., None],
            normalized_log_flux[..., None],
            wavelength_embedding,
        ],
        axis=-1,
    )
    token_features = token_features * valid_mask[..., None].astype(dtype)
    return token_features, summary, valid_mask


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


def _multihead_attention(params: dict[str, jax.Array], x: jax.Array, *, nhead: int) -> jax.Array:
    """Convenience wrapper for bidirectional self-attention."""
    return _multihead_attention_qkv(
        params["q"],
        params["k"],
        params["v"],
        params["o"],
        x,
        x,
        nhead=nhead,
    )


def _encode_spectrum(
    params: dict[str, Any],
    spectrum_wavelengths_nm: jax.Array | None,
    spectrum_fluxes_erg_cm2_s_nm: jax.Array | None,
    spectrum_mask: jax.Array | None,
    dims: TransformerDimensions | MLPDimensions,
    *,
    batch_size: int | None = None,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
    """Compress a variable-length native-grid stellar spectrum into a latent vector."""
    mode = dims.spectrum_encoder_mode
    if mode == "none":
        resolved_batch = batch_size
        if resolved_batch is None:
            if spectrum_fluxes_erg_cm2_s_nm is None:
                raise ValueError("batch_size must be provided when the spectrum encoder is disabled.")
            resolved_batch = int(spectrum_fluxes_erg_cm2_s_nm.shape[0])
        latent = jnp.zeros((resolved_batch, dims.spectrum_latent_dim), dtype=dtype)
        return latent, None, None

    if (
        spectrum_wavelengths_nm is None
        or spectrum_fluxes_erg_cm2_s_nm is None
        or spectrum_mask is None
    ):
        raise ValueError(
            "spectrum_wavelengths_nm, spectrum_fluxes_erg_cm2_s_nm, and spectrum_mask "
            "must all be provided when the spectrum encoder is enabled."
        )
    if mode != "perceiver":
        raise ValueError(f"Unsupported spectrum encoder mode: {mode}")

    act = _resolve_activation(dims.activation)
    token_features, summary, valid_mask = _prepare_spectrum_tokens(
        spectrum_wavelengths_nm,
        spectrum_fluxes_erg_cm2_s_nm,
        spectrum_mask,
        dims,
    )
    tokens = act(_layer_norm(params["token_ln"], _linear(params["token_in"], token_features)))
    latents = jnp.broadcast_to(
        params["latents"][None, :, :],
        (tokens.shape[0], params["latents"].shape[0], params["latents"].shape[1]),
    )
    cross_input = _layer_norm(params["cross_ln"], latents)
    latents = latents + _multihead_attention_qkv(
        params["cross_q"],
        params["cross_k"],
        params["cross_v"],
        params["cross_o"],
        cross_input,
        tokens,
        nhead=dims.spectrum_num_heads,
        source_mask=valid_mask,
    )

    for block in params["latent_blocks"]:
        ln1 = _layer_norm(block["ln1"], latents)
        latents = latents + _multihead_attention_qkv(
            block["q"],
            block["k"],
            block["v"],
            block["o"],
            ln1,
            ln1,
            nhead=dims.spectrum_num_heads,
        )
        ff_in = _layer_norm(block["ln2"], latents)
        latents = latents + _linear(block["ff2"], act(_linear(block["ff1"], ff_in)))

    normed_latents = _layer_norm(params["pool_ln"], latents)
    pooled = jnp.mean(normed_latents, axis=1)
    fused = jnp.concatenate([pooled, summary.astype(pooled.dtype)], axis=-1)
    hidden = act(_linear(params["out_hidden"], fused))
    latent = _linear(params["out_out"], hidden)
    return latent, normed_latents, None


def apply_mlp(
    params: dict[str, Any],
    sequence_inputs: jax.Array,
    global_inputs: jax.Array,
    dims: MLPDimensions,
    spectrum_wavelengths_nm: jax.Array | None = None,
    spectrum_fluxes_erg_cm2_s_nm: jax.Array | None = None,
    spectrum_mask: jax.Array | None = None,
    *,
    dropout_key: jax.Array | None = None,
    training: bool = False,
) -> tuple[jax.Array, dict]:
    """Forward pass for the FiLM-MLP."""
    if sequence_inputs.ndim != 3:
        raise ValueError("sequence_inputs must have shape [batch, nz, feature_dim].")
    if global_inputs.ndim != 2:
        raise ValueError("global_inputs must have shape [batch, global_dim].")
    if dims.spectrum_max_tokens > 0 and dims.spectrum_encoder_mode != "none":
        if (
            spectrum_wavelengths_nm is None
            or spectrum_fluxes_erg_cm2_s_nm is None
            or spectrum_mask is None
            or spectrum_wavelengths_nm.ndim != 2
            or spectrum_fluxes_erg_cm2_s_nm.ndim != 2
            or spectrum_mask.ndim != 2
        ):
            raise ValueError(
                "spectrum_wavelengths_nm, spectrum_fluxes_erg_cm2_s_nm, and spectrum_mask "
                "must have shape [batch, spectrum_max_tokens] when enabled."
            )

    activation = _resolve_activation(dims.activation)
    layer_dropout_keys = [None] * dims.num_hidden_layers
    if training and dims.dropout_rate > 0.0 and dropout_key is not None:
        layer_dropout_keys = list(jax.random.split(dropout_key, dims.num_hidden_layers))

    latent, _spectrum_latent_tokens, reconstruction = _encode_spectrum(
        params.get("spectrum_encoder", {}),
        spectrum_wavelengths_nm,
        spectrum_fluxes_erg_cm2_s_nm,
        spectrum_mask,
        dims,
        batch_size=int(global_inputs.shape[0]),
        dtype=global_inputs.dtype,
    )
    context_inputs = jnp.concatenate([global_inputs, latent], axis=-1)
    context = activation(_linear(params["context_in"], context_inputs))
    film = _linear(params["context_out"], context)
    film = film.reshape(global_inputs.shape[0], dims.num_hidden_layers, 2, dims.d_hidden)

    x = sequence_inputs
    for layer_index, (layer, layer_key) in enumerate(zip(params["layers"], layer_dropout_keys)):
        residual = x if layer_index >= 1 else None
        x = _linear(layer["linear"], x)
        x = _layer_norm(layer["ln"], x)
        gamma = jnp.clip(film[:, layer_index, 0], -dims.film_clamp, dims.film_clamp)
        beta = jnp.clip(film[:, layer_index, 1], -dims.film_clamp, dims.film_clamp)
        x = x * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        x = activation(x)
        x = _apply_dropout(
            x,
            rate=dims.dropout_rate,
            key=layer_key,
            training=training,
        )
        if residual is not None:
            x = residual + x

    pred = _linear(params["output"], x)
    return pred, {
        "spectrum_reconstruction": reconstruction,
        "spectrum_latent": latent,
    }


def apply_transformer_model(
    params: dict[str, Any],
    sequence_inputs: jax.Array,
    global_inputs: jax.Array,
    spectrum_wavelengths_nm: jax.Array | None,
    spectrum_fluxes_erg_cm2_s_nm: jax.Array | None,
    spectrum_mask: jax.Array | None,
    dims: TransformerDimensions,
    *,
    dropout_key: jax.Array | None = None,
    training: bool = False,
) -> tuple[jax.Array, dict[str, jax.Array | None]]:
    """Run the Transformer forward pass in normalized space."""
    if sequence_inputs.ndim != 3:
        raise ValueError("sequence_inputs must have shape [batch, nz, feature_dim].")
    if global_inputs.ndim != 2:
        raise ValueError("global_inputs must have shape [batch, global_dim].")
    if dims.spectrum_max_tokens > 0 and dims.spectrum_encoder_mode != "none":
        if (
            spectrum_wavelengths_nm is None
            or spectrum_fluxes_erg_cm2_s_nm is None
            or spectrum_mask is None
            or spectrum_wavelengths_nm.ndim != 2
            or spectrum_fluxes_erg_cm2_s_nm.ndim != 2
            or spectrum_mask.ndim != 2
        ):
            raise ValueError(
                "spectrum_wavelengths_nm, spectrum_fluxes_erg_cm2_s_nm, and spectrum_mask "
                "must have shape [batch, spectrum_max_tokens] when enabled."
            )

    act = _resolve_activation(dims.activation)
    _has_cross_attn = dims.spectrum_encoder_mode != "none"
    _keys_per_layer = 3 if _has_cross_attn else 2
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

    latent, spectrum_latent_tokens, reconstruction = _encode_spectrum(
        params.get("spectrum_encoder", {}),
        spectrum_wavelengths_nm,
        spectrum_fluxes_erg_cm2_s_nm,
        spectrum_mask,
        dims,
        batch_size=int(global_inputs.shape[0]),
        dtype=global_inputs.dtype,
    )
    context = jnp.concatenate([global_inputs, latent], axis=-1)
    context = act(_linear(params["context_in"], context))
    film = _linear(params["context_out"], context)
    film = film.reshape(sequence_inputs.shape[0], dims.num_layers, 2, dims.d_model)

    x = _linear(params["sequence_in"], sequence_inputs)
    x = x + sinusoidal_position_encoding(sequence_inputs.shape[1], dims.d_model, x.dtype)[None, :, :]

    for layer_index, (layer, layer_keys) in enumerate(zip(params["layers"], layer_dropout_keys)):
        attn_dropout_key = layer_keys[0]
        cross_dropout_key = layer_keys[1] if _has_cross_attn else None
        ff_dropout_key = layer_keys[-1]

        gamma = jnp.clip(film[:, layer_index, 0], -dims.film_clamp, dims.film_clamp)
        beta = jnp.clip(film[:, layer_index, 1], -dims.film_clamp, dims.film_clamp)

        # (a) Self-attention
        x_norm = _layer_norm(layer["ln1"], x)
        attn = _multihead_attention(layer, x_norm, nhead=dims.nhead)
        attn = _apply_dropout(
            attn,
            rate=dims.dropout_rate,
            key=attn_dropout_key,
            training=training,
        )
        x = x + attn

        # (b) Cross-attention to spectrum latent tokens (VULCAN only)
        if spectrum_latent_tokens is not None and "ln_cross" in layer:
            cross_norm = _layer_norm(layer["ln_cross"], x)
            cross_attn = _multihead_attention_qkv(
                layer["cross_q"],
                layer["cross_k"],
                layer["cross_v"],
                layer["cross_o"],
                cross_norm,
                spectrum_latent_tokens,
                nhead=dims.nhead,
            )
            cross_attn = _apply_dropout(
                cross_attn,
                rate=dims.dropout_rate,
                key=cross_dropout_key,
                training=training,
            )
            x = x + cross_attn

        # (c) FiLM conditioning
        x = x * (1.0 + gamma[:, None, :]) + beta[:, None, :]

        # (d) FFN
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
    aux = {
        "spectrum_reconstruction": reconstruction,
        "spectrum_latent": latent,
    }
    return pred, aux


# ---------------------------------------------------------------------------
# Section 5: Spectrum packing utilities
# ---------------------------------------------------------------------------

def _coalesce_duplicate_wavelengths(
    wavelength_nm: np.ndarray,
    flux_erg_cm2_s_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average duplicate wavelength samples onto a unique strictly increasing grid."""
    unique_wavelengths, inverse = np.unique(wavelength_nm, return_inverse=True)
    summed_flux = np.zeros_like(unique_wavelengths, dtype=np.float64)
    counts = np.zeros_like(unique_wavelengths, dtype=np.float64)
    np.add.at(summed_flux, inverse, flux_erg_cm2_s_nm)
    np.add.at(counts, inverse, 1.0)
    return unique_wavelengths, summed_flux / np.maximum(counts, 1.0)


def sanitize_spectrum_arrays(
    wavelength_nm: np.ndarray,
    flux_erg_cm2_s_nm: np.ndarray,
    *,
    wavelength_min_nm: float | None = None,
    wavelength_max_nm: float | None = None,
    min_points: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort, de-duplicate, clip, and validate raw spectrum arrays."""
    wavelength = np.asarray(wavelength_nm, dtype=np.float64).reshape(-1)
    flux = np.asarray(flux_erg_cm2_s_nm, dtype=np.float64).reshape(-1)
    if wavelength.size != flux.size:
        raise ValueError("Wavelength and flux arrays must have the same length.")
    mask = np.isfinite(wavelength) & np.isfinite(flux)
    wavelength = wavelength[mask]
    flux = np.maximum(flux[mask], 0.0)
    if wavelength_min_nm is not None:
        mask = wavelength >= float(wavelength_min_nm)
        wavelength = wavelength[mask]
        flux = flux[mask]
    if wavelength_max_nm is not None:
        mask = wavelength <= float(wavelength_max_nm)
        wavelength = wavelength[mask]
        flux = flux[mask]
    if wavelength.size < int(min_points):
        raise ValueError(
            "Spectrum does not contain enough valid samples after clipping to the configured "
            "wavelength interval."
        )
    order = np.argsort(wavelength, kind="mergesort")
    wavelength = wavelength[order]
    flux = flux[order]
    wavelength, flux = _coalesce_duplicate_wavelengths(wavelength, flux)
    if wavelength.size < int(min_points):
        raise ValueError("Spectrum must contain at least two unique wavelength samples.")
    if not np.all(np.diff(wavelength) > 0.0):
        raise ValueError("Sanitized wavelength grid must be strictly increasing.")
    return wavelength.astype(np.float64), flux.astype(np.float64)


def vulcan_wavelength_bins(
    wavelength_min_nm: float = 2.0,
    wavelength_max_nm: float = 700.0,
    dbin1_nm: float = 0.1,
    dbin2_nm: float = 2.0,
    dbin_12trans_nm: float = 240.0,
) -> np.ndarray:
    """Construct VULCAN's non-uniform wavelength bin-centre array."""
    wmin = float(wavelength_min_nm)
    wmax = float(wavelength_max_nm)
    trans = float(dbin_12trans_nm)
    d1 = float(dbin1_nm)
    d2 = float(dbin2_nm)
    if trans >= wmin and trans <= wmax:
        bins = np.concatenate((
            np.arange(wmin, trans, d1),
            np.arange(trans, wmax, d2),
        ))
    elif trans < wmin:
        bins = np.arange(wmin, wmax, d2)
    else:
        bins = np.arange(wmin, wmax, d1)
    return bins.astype(np.float64)


def _flux_conserving_rebin(
    wavelength_nm: np.ndarray,
    flux_erg_cm2_s_nm: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Rebin a spectrum onto arbitrary bin edges with flux conservation."""
    w = wavelength_nm.astype(np.float64)
    f = flux_erg_cm2_s_nm.astype(np.float64)
    edges = np.asarray(edges, dtype=np.float64)

    edge_flux = np.interp(edges, w, f)
    merged_w = np.concatenate([w, edges])
    merged_f = np.concatenate([f, edge_flux])
    order = np.argsort(merged_w, kind="stable")
    merged_w = merged_w[order]
    merged_f = merged_f[order]
    keep = np.concatenate(([True], np.diff(merged_w) > 0.0))
    merged_w = merged_w[keep]
    merged_f = merged_f[keep]

    cum = np.empty(len(merged_w), dtype=np.float64)
    cum[0] = 0.0
    cum[1:] = np.cumsum(0.5 * (merged_f[:-1] + merged_f[1:]) * np.diff(merged_w))

    edge_idx = np.searchsorted(merged_w, edges)
    edge_idx = np.clip(edge_idx, 0, len(cum) - 1)
    bin_integrals = cum[edge_idx[1:]] - cum[edge_idx[:-1]]

    bin_widths = np.maximum(edges[1:] - edges[:-1], 1.0e-12)
    return bin_integrals / bin_widths


def _vulcan_bin_edges(
    centers: np.ndarray,
    dbin1_nm: float,
    dbin2_nm: float,
    dbin_12trans_nm: float,
) -> np.ndarray:
    """Compute contiguous bin edges from VULCAN bin centres."""
    n = len(centers)
    edges = np.empty(n + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    uv_mask = centers < dbin_12trans_nm
    first_width = dbin1_nm if uv_mask[0] else dbin2_nm
    last_width = dbin1_nm if uv_mask[-1] else dbin2_nm
    edges[0] = centers[0] - 0.5 * first_width
    edges[-1] = centers[-1] + 0.5 * last_width
    return edges


def pack_spectrum_tokens(
    wavelength_nm: np.ndarray,
    flux_erg_cm2_s_nm: np.ndarray,
    *,
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    max_tokens: int,
    dbin1_nm: float = 0.1,
    dbin2_nm: float = 2.0,
    dbin_12trans_nm: float = 240.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack an arbitrary spectrum into VULCAN's native wavelength grid."""
    clean_wavelengths, clean_flux = sanitize_spectrum_arrays(
        wavelength_nm,
        flux_erg_cm2_s_nm,
        wavelength_min_nm=wavelength_min_nm,
        wavelength_max_nm=wavelength_max_nm,
        min_points=2,
    )
    centers = vulcan_wavelength_bins(
        wavelength_min_nm=wavelength_min_nm,
        wavelength_max_nm=wavelength_max_nm,
        dbin1_nm=dbin1_nm,
        dbin2_nm=dbin2_nm,
        dbin_12trans_nm=dbin_12trans_nm,
    )
    edges = _vulcan_bin_edges(centers, dbin1_nm, dbin2_nm, dbin_12trans_nm)
    rebinned_flux = _flux_conserving_rebin(clean_wavelengths, clean_flux, edges)
    valid_count = len(centers)
    if valid_count > int(max_tokens):
        raise ValueError(
            f"VULCAN bin grid produces {valid_count} bins but max_tokens is "
            f"{max_tokens}. Increase max_tokens to at least {valid_count}."
        )
    padded_wavelengths = np.zeros((int(max_tokens),), dtype=np.float32)
    padded_flux = np.zeros((int(max_tokens),), dtype=np.float32)
    padded_mask = np.zeros((int(max_tokens),), dtype=bool)
    padded_wavelengths[:valid_count] = centers.astype(np.float32)
    padded_flux[:valid_count] = rebinned_flux.astype(np.float32)
    padded_mask[:valid_count] = True
    return padded_wavelengths, padded_flux, padded_mask


# ---------------------------------------------------------------------------
# Section 6: Normalization utilities
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


def _is_traced_array(value: Any) -> bool:
    """Return whether an input is a JAX tracer inside a transformed context."""
    return isinstance(value, jax.core.Tracer)


# ---------------------------------------------------------------------------
# Section 7: Parameter tree utilities
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
# Section 8: Bundle loading utilities
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


# ---------------------------------------------------------------------------
# Section 9: VULCAN JAX-native spectrum packing
# ---------------------------------------------------------------------------

def _pack_native_spectrum_tokens_jax(
    wavelength_nm: jax.Array,
    flux_erg_cm2_s_nm: jax.Array,
    *,
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    max_tokens: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pad an already-valid native-grid spectrum without leaving JAX."""
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


# ---------------------------------------------------------------------------
# Section 10: FastChem inference implementation
# ---------------------------------------------------------------------------

def _predict_fastchem_impl(
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


# ---------------------------------------------------------------------------
# Section 11: ExportedModel class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExportedModel:
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
    def uses_fastchem(self) -> bool:
        return self.chemistry_type == "fastchem"

    @property
    def uses_vulcan_chemistry(self) -> bool:
        return self.chemistry_type == "vulcan"

    @property
    def uses_mlp(self) -> bool:
        return self.model_type == "mlp"

    @property
    def uses_transformer(self) -> bool:
        return self.model_type == "transformer"

    @property
    def species(self) -> list[str]:
        return list(self.data_contract["output_species_order"])

    @cached_property
    def _compiled_fastchem_predictor(self) -> Callable[[Any, Any, Any], jax.Array]:
        """Cache a compiled FastChem predictor for repeated linear-space calls."""
        return jax.jit(
            lambda pressure_bar, temperature_k, global_inputs: _predict_fastchem_impl(
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
    def _compiled_fastchem_predictor_log10(self) -> Callable[[Any, Any, Any], jax.Array]:
        """Cache a compiled FastChem predictor for repeated log10-space calls."""
        return jax.jit(
            lambda pressure_bar, temperature_k, global_inputs: _predict_fastchem_impl(
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

    def make_compiled_fastchem_predictor(
        self,
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
        self,
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
            Global conditioning scalars.
        return_log10 : bool
            If True, return log10 mixing ratios instead of linear.

        Returns
        -------
        jax.Array
            Predicted mixing ratios, shape ``(nz, target_dim)``.
        """
        if not self.uses_fastchem:
            raise ValueError("predict_fastchem requires a fastchem export bundle.")
        return _predict_fastchem_impl(
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

    def predict_vulcan(
        self,
        pressure_bar: jax.Array | np.ndarray,
        temperature_k: jax.Array | np.ndarray,
        kzz_cm2_s: jax.Array | np.ndarray,
        global_inputs: dict[str, float] | jax.Array | np.ndarray,
        spectrum_wavelength_nm: jax.Array | np.ndarray,
        spectrum_flux_erg_cm2_s_nm: jax.Array | np.ndarray,
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
            Global conditioning scalars.
        spectrum_wavelength_nm : array-like
            Stellar spectrum wavelength grid in nanometers.
        spectrum_flux_erg_cm2_s_nm : array-like
            Stellar spectrum flux in erg cm-2 s-1 nm-1.
        return_log10 : bool
            If True, return log10 mixing ratios instead of linear.

        Returns
        -------
        jax.Array
            Predicted mixing ratios, shape ``(nz, target_dim)``.
        """
        if not self.uses_vulcan_chemistry:
            raise ValueError("predict_vulcan requires a vulcan export bundle.")

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


# ---------------------------------------------------------------------------
# Section 12: load_model function
# ---------------------------------------------------------------------------

def load_model(bundle_path: str | Path, *, device=None) -> ExportedModel:
    """Load an exported NPZ bundle."""
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
    if chemistry_type not in {"fastchem", "vulcan"} or model_type not in {"mlp", "transformer"}:
        raise ValueError("Legacy bundle format — re-export with the current pipeline.")
    dims: TransformerDimensions | MLPDimensions
    if model_type == "mlp":
        dims = MLPDimensions.from_dict(model_dimensions)
    else:
        dims = TransformerDimensions.from_dict(model_dimensions)
    return ExportedModel(
        params=params, dims=dims, normalization=normalization,
        data_contract=data_contract, config=config,
        export_format=export_format, export_version=export_version,
        chemistry_type=chemistry_type, model_type=model_type,
    )


# ---------------------------------------------------------------------------
# Section 13: ExoJAX wrappers
# ---------------------------------------------------------------------------

def make_fastchem_vmr_fn(model: ExportedModel) -> tuple[Any, list[str]]:
    """Create an ExoJAX-compatible FastChem profile inference callable.

    Parameters
    ----------
    model : ExportedModel
        Exported FastChem emulator bundle.

    Returns
    -------
    tuple[Any, list[str]]
        Callable ``vmr_fn`` plus ordered species labels. The callable returns
        linear VMR predictions with shape ``(nz, n_species)`` in top-to-bottom
        level order.
    """
    if not model.uses_fastchem:
        raise ValueError("make_fastchem_vmr_fn requires a fastchem bundle.")

    feature_order = list(model.data_contract.get("global_static_feature_order", []))
    if feature_order != FASTCHEM_GLOBAL_LABELS:
        raise ValueError(
            "FastChem bundle global_static_feature_order must be "
            f"{FASTCHEM_GLOBAL_LABELS}; got {feature_order}."
        )

    compiled_predict = model.make_compiled_fastchem_predictor()

    def vmr_fn(
        temperatures_k: jax.Array,
        pressures_bar: jax.Array,
        global_inputs: dict[str, Any] | jax.Array,
        gravity_cm_s2: jax.Array | None = None,
    ) -> jax.Array:
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

        internal_temperatures = temperatures[::-1]
        internal_pressures = pressures[::-1]
        vmr_internal = compiled_predict(internal_pressures, internal_temperatures, global_inputs)
        return vmr_internal[::-1, :]

    return vmr_fn, list(model.data_contract["output_species_order"])


def make_vulcan_vmr_fn(model: ExportedModel) -> tuple[Any, list[str]]:
    """Create an ExoJAX-compatible VULCAN profile inference callable.

    Parameters
    ----------
    model : ExportedModel
        Exported VULCAN emulator bundle.

    Returns
    -------
    tuple[Any, list[str]]
        Callable ``vmr_fn`` plus ordered species labels. The callable returns
        linear VMR predictions with shape ``(nz, n_species)`` in top-to-bottom
        level order.
    """
    if not model.uses_vulcan_chemistry:
        raise ValueError("make_vulcan_vmr_fn requires a vulcan bundle.")

    def vmr_fn(
        temperatures_k: jax.Array,
        pressures_bar: jax.Array,
        kzz_cm2_s: jax.Array,
        global_inputs: dict[str, Any] | jax.Array,
        spectrum_wavelength_nm: jax.Array,
        spectrum_flux_erg_cm2_s_nm: jax.Array,
    ) -> jax.Array:
        temperatures = jnp.asarray(temperatures_k, dtype=jnp.float32)
        pressures = jnp.asarray(pressures_bar, dtype=jnp.float32)
        kzz = jnp.asarray(kzz_cm2_s, dtype=jnp.float32)
        if temperatures.ndim != 1 or pressures.ndim != 1:
            raise ValueError("temperatures_k and pressures_bar must be 1-D arrays.")
        if temperatures.shape != pressures.shape:
            raise ValueError("temperatures_k and pressures_bar must share the same shape.")
        if kzz.ndim != 1 or kzz.shape != temperatures.shape:
            raise ValueError("kzz_cm2_s must be a 1-D array sharing the PT grid shape.")

        internal_temperatures = temperatures[::-1]
        internal_pressures = pressures[::-1]
        internal_kzz = kzz[::-1]
        vmr_internal = model.predict_vulcan(
            internal_pressures,
            internal_temperatures,
            internal_kzz,
            global_inputs,
            spectrum_wavelength_nm,
            spectrum_flux_erg_cm2_s_nm,
        )
        return vmr_internal[::-1, :]

    return vmr_fn, list(model.data_contract["output_species_order"])
