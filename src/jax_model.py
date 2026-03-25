"""Transformer-based surrogate model for 1-D photochemical trajectories.

Implements a FiLM-conditioned transformer that maps an atmospheric column
(pressure, temperature, Kzz, and anchor mixing ratios) to future mixing
ratios, conditioned on global scalars (gravity, metallicity, C/O, physics
toggles) and a stellar spectrum latent vector.

The architecture:
    1. Project the per-level sequence (P, T, Kzz, ymix) to d_model.
    2. Add sinusoidal positional encoding over the vertical grid.
    3. Encode the stellar spectrum into a latent vector.
    4. Concatenate [global_scalars, spectrum_latent] and project to
       per-layer FiLM parameters (gamma, beta).
    5. Run L transformer blocks with pre-norm multi-head self-attention,
       FiLM modulation, and GELU feed-forward sub-layers.
    6. Project the final representation to target mixing ratios.

All operations are pure JAX and compatible with ``jax.grad`` / ``jax.jvp``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .numpy_compat import patch_numpy_asarray_copy

patch_numpy_asarray_copy()

import jax
import jax.numpy as jnp

# Base wavelength used in the standard sinusoidal positional encoding
# (Vaswani et al., 2017).
_SINUSOIDAL_BASE_WAVELENGTH = 10_000.0


@dataclass(frozen=True)
class ModelDimensions:
    """All dimensionality and architecture hyper-parameters for the surrogate.

    Fields
    ------
    sequence_dim : int
        Per-level input width (3 static + state_dim anchor species).
    global_dim : int
        Width of the full global conditioning vector (static + dt).
    spectrum_dim : int
        Number of wavelength bins in the input spectrum.
    target_dim : int
        Number of output species per level.
    d_model : int
        Hidden width of the transformer backbone.
    nhead : int
        Number of attention heads (must divide d_model).
    num_layers : int
        Number of transformer blocks.
    dim_feedforward : int
        Width of each block's feed-forward sub-layer.
    conditioning_hidden_dim : int
        Hidden width of the FiLM conditioning MLP.
    film_clamp : float
        Symmetric clamp applied to FiLM gamma / beta.
    output_head_divisor : int
        The output MLP bottleneck is d_model // output_head_divisor.
    spectrum_latent_dim : int
        Dimension of the spectrum encoder's latent vector.
    spectrum_hidden_dim : int
        Hidden width inside the spectrum autoencoder.
    spectrum_encoder_mode : str
        One of ``"autoencoder"``, ``"linear"``, or ``"none"``.
    """

    sequence_dim: int
    global_dim: int
    spectrum_dim: int
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
    spectrum_encoder_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelDimensions":
        return cls(**payload)


def _init_linear(key: jax.Array, in_dim: int, out_dim: int) -> dict[str, jax.Array]:
    """Xavier-uniform initialisation for a single dense layer."""
    limit = math.sqrt(6.0 / float(in_dim + out_dim))
    weight = jax.random.uniform(key, shape=(in_dim, out_dim), minval=-limit, maxval=limit)
    bias = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"weight": weight.astype(jnp.float32), "bias": bias}


def _linear(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    """Dense affine transform: x @ W + b."""
    return jnp.einsum("...i,io->...o", x, params["weight"]) + params["bias"]


def _init_layer_norm(dim: int) -> dict[str, jax.Array]:
    """Initialise LayerNorm parameters (scale=1, bias=0)."""
    return {
        "scale": jnp.ones((dim,), dtype=jnp.float32),
        "bias": jnp.zeros((dim,), dtype=jnp.float32),
    }


def _layer_norm(params: dict[str, jax.Array], x: jax.Array, eps: float = 1.0e-5) -> jax.Array:
    """Layer normalisation over the last axis."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    normalized = (x - mean) / jnp.sqrt(var + eps)
    return normalized * params["scale"] + params["bias"]


def sinusoidal_position_encoding(length: int, dim: int, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    """Fixed sinusoidal positional encoding over the vertical grid."""
    position = jnp.arange(length, dtype=dtype)[:, None]
    index = jnp.arange(dim, dtype=dtype)[None, :]
    angle_rate = 1.0 / jnp.power(
        _SINUSOIDAL_BASE_WAVELENGTH,
        (2.0 * jnp.floor(index / 2.0)) / float(dim),
    )
    angle = position * angle_rate
    return jnp.where((jnp.arange(dim) % 2)[None, :] == 0, jnp.sin(angle), jnp.cos(angle))


def init_model_params(key: jax.Array, dims: ModelDimensions) -> dict[str, Any]:
    """Allocate and Xavier-initialise all model parameters."""
    if dims.spectrum_encoder_mode == "autoencoder":
        spectrum_key_count = 4
    elif dims.spectrum_encoder_mode == "linear":
        spectrum_key_count = 1
    else:
        spectrum_key_count = 0
    total_key_count = 5 + spectrum_key_count + dims.num_layers * 6
    keys = iter(jax.random.split(key, total_key_count))
    params: dict[str, Any] = {
        "sequence_in": _init_linear(next(keys), dims.sequence_dim, dims.d_model),
        "context_in": _init_linear(next(keys), dims.global_dim + dims.spectrum_latent_dim, dims.conditioning_hidden_dim),
        "context_out": _init_linear(next(keys), dims.conditioning_hidden_dim, dims.num_layers * 2 * dims.d_model),
        "out_norm": _init_layer_norm(dims.d_model),
        "out_head_hidden": _init_linear(next(keys), dims.d_model, max(1, dims.d_model // dims.output_head_divisor)),
        "out_head_final": _init_linear(next(keys), max(1, dims.d_model // dims.output_head_divisor), dims.target_dim),
    }

    if dims.spectrum_encoder_mode == "autoencoder":
        params["spectrum_encoder"] = {
            "encoder_1": _init_linear(next(keys), dims.spectrum_dim, dims.spectrum_hidden_dim),
            "encoder_2": _init_linear(next(keys), dims.spectrum_hidden_dim, dims.spectrum_latent_dim),
            "decoder_1": _init_linear(next(keys), dims.spectrum_latent_dim, dims.spectrum_hidden_dim),
            "decoder_2": _init_linear(next(keys), dims.spectrum_hidden_dim, dims.spectrum_dim),
        }
    elif dims.spectrum_encoder_mode == "linear":
        params["spectrum_encoder"] = {
            "encoder_1": _init_linear(next(keys), dims.spectrum_dim, dims.spectrum_latent_dim),
        }
    else:
        params["spectrum_encoder"] = {}

    layers: list[dict[str, Any]] = []
    for _ in range(dims.num_layers):
        layers.append(
            {
                "ln1": _init_layer_norm(dims.d_model),
                "q": _init_linear(next(keys), dims.d_model, dims.d_model),
                "k": _init_linear(next(keys), dims.d_model, dims.d_model),
                "v": _init_linear(next(keys), dims.d_model, dims.d_model),
                "o": _init_linear(next(keys), dims.d_model, dims.d_model),
                "ln2": _init_layer_norm(dims.d_model),
                "ff1": _init_linear(next(keys), dims.d_model, dims.dim_feedforward),
                "ff2": _init_linear(next(keys), dims.dim_feedforward, dims.d_model),
            }
        )
    params["layers"] = layers
    return params


def _multihead_attention(params: dict[str, jax.Array], x: jax.Array, *, nhead: int) -> jax.Array:
    """Standard scaled-dot-product multi-head self-attention."""
    batch_size, nlevel, d_model = x.shape
    head_dim = d_model // nhead
    q = _linear(params["q"], x).reshape(batch_size, nlevel, nhead, head_dim).transpose(0, 2, 1, 3)
    k = _linear(params["k"], x).reshape(batch_size, nlevel, nhead, head_dim).transpose(0, 2, 1, 3)
    v = _linear(params["v"], x).reshape(batch_size, nlevel, nhead, head_dim).transpose(0, 2, 1, 3)
    scale = 1.0 / math.sqrt(float(head_dim))
    logits = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
    weights = jax.nn.softmax(logits, axis=-1)
    attended = jnp.einsum("bhij,bhjd->bhid", weights, v)
    attended = attended.transpose(0, 2, 1, 3).reshape(batch_size, nlevel, d_model)
    return _linear(params["o"], attended)


def _encode_spectrum(
    params: dict[str, Any],
    spectrum_inputs: jax.Array,
    dims: ModelDimensions,
) -> tuple[jax.Array, jax.Array | None]:
    """Compress the stellar spectrum into a latent vector.

    Returns (latent, reconstruction) where reconstruction is non-None only
    in autoencoder mode.
    """
    mode = dims.spectrum_encoder_mode
    if mode == "none":
        latent = jnp.zeros((spectrum_inputs.shape[0], dims.spectrum_latent_dim), dtype=spectrum_inputs.dtype)
        return latent, None
    if mode == "linear":
        latent = _linear(params["encoder_1"], spectrum_inputs)
        return latent, None
    if mode == "autoencoder":
        hidden = jax.nn.gelu(_linear(params["encoder_1"], spectrum_inputs))
        latent = _linear(params["encoder_2"], hidden)
        recon_hidden = jax.nn.gelu(_linear(params["decoder_1"], latent))
        reconstruction = _linear(params["decoder_2"], recon_hidden)
        return latent, reconstruction
    raise ValueError(f"Unsupported spectrum encoder mode: {mode}")


def apply_model(
    params: dict[str, Any],
    sequence_inputs: jax.Array,
    global_inputs: jax.Array,
    spectrum_inputs: jax.Array,
    dims: ModelDimensions,
) -> tuple[jax.Array, dict[str, jax.Array | None]]:
    if sequence_inputs.ndim != 3:
        raise ValueError("sequence_inputs must have shape [batch, nz, feature_dim].")
    if global_inputs.ndim != 2:
        raise ValueError("global_inputs must have shape [batch, global_dim].")
    if spectrum_inputs.ndim != 2:
        raise ValueError("spectrum_inputs must have shape [batch, spectrum_dim].")

    latent, reconstruction = _encode_spectrum(params["spectrum_encoder"], spectrum_inputs, dims)
    context = jnp.concatenate([global_inputs, latent], axis=-1)
    context = jax.nn.gelu(_linear(params["context_in"], context))
    film = _linear(params["context_out"], context)
    film = film.reshape(sequence_inputs.shape[0], dims.num_layers, 2, dims.d_model)

    x = _linear(params["sequence_in"], sequence_inputs)
    x = x + sinusoidal_position_encoding(sequence_inputs.shape[1], dims.d_model, x.dtype)[None, :, :]
    for layer_index, layer in enumerate(params["layers"]):
        gamma = jnp.clip(film[:, layer_index, 0], -dims.film_clamp, dims.film_clamp)
        beta = jnp.clip(film[:, layer_index, 1], -dims.film_clamp, dims.film_clamp)
        x_norm = _layer_norm(layer["ln1"], x)
        attn = _multihead_attention(layer, x_norm, nhead=dims.nhead)
        x = x + attn
        x = x * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        ff_in = _layer_norm(layer["ln2"], x)
        ff_hidden = jax.nn.gelu(_linear(layer["ff1"], ff_in))
        ff = _linear(layer["ff2"], ff_hidden)
        x = x + ff

    x = _layer_norm(params["out_norm"], x)
    x = jax.nn.gelu(_linear(params["out_head_hidden"], x))
    pred = _linear(params["out_head_final"], x)
    aux = {
        "spectrum_reconstruction": reconstruction,
        "spectrum_latent": latent,
    }
    return pred, aux
