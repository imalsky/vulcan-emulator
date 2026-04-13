"""JAX model definitions for the FastChem and VULCAN emulators.

Implements two FiLM-conditioned architectures:

**FiLM-conditioned MLP** (per-level, shared weights across vertical grid):
    1. Encode the stellar spectrum into a latent vector (VULCAN only).
    2. Concatenate [global_inputs, spectrum_latent] and project through a
       conditioning MLP to produce per-layer FiLM gamma/beta.
    3. Per-level loop (shared weights across nz):
       - Layer 0: Linear(sequence_dim → d_hidden) → LayerNorm → FiLM → act → dropout
       - Layer i≥1: Linear → LayerNorm → FiLM → act → dropout → residual add
    4. Output: Linear(d_hidden → target_dim).

**FiLM-conditioned Transformer**:
    1. Encode the stellar spectrum via Perceiver into (a) a mean-pooled
       latent vector for FiLM conditioning and (b) un-pooled latent tokens
       for per-block cross-attention (VULCAN only).
    2. Concatenate [global_inputs, spectrum_latent] and project to
       per-layer FiLM parameters (gamma, beta).
    3. Project per-level sequence to d_model + sinusoidal positional encoding.
    4. Per block:
       a. Pre-norm (ln1) → multi-head self-attention → dropout → residual add
       b. Pre-norm (ln_cross) → cross-attention(Q=sequence, KV=spectrum latent tokens)
          → dropout → residual add  [VULCAN only; skipped when no spectrum]
       c. FiLM: x = x * (1 + gamma) + beta
       d. Pre-norm (ln_ffn) → FFN (up-project, act, dropout, down-project) → residual add
    5. Output head: LayerNorm → act → bottleneck → final projection.

All operations are pure JAX and compatible with ``jax.grad`` / ``jax.jvp``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

# Base wavelength used in the standard sinusoidal positional encoding
# (Vaswani et al., 2017).
_SINUSOIDAL_BASE_WAVELENGTH = 10_000.0



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


def _init_linear(key: jax.Array, in_dim: int, out_dim: int) -> dict[str, jax.Array]:
    """Initialize a single dense (affine) layer with Xavier-uniform weights.

    The weight matrix is sampled uniformly from [-limit, +limit] where
    ``limit = sqrt(6 / (in_dim + out_dim))``.  Biases are initialized to zero.

    Parameters
    ----------
    key : jax.Array
        PRNG key for random initialization.
    in_dim : int
        Number of input features.
    out_dim : int
        Number of output features.

    Returns
    -------
    dict[str, jax.Array]
        ``{"weight": (in_dim, out_dim), "bias": (out_dim,)}`` in float32.
    """
    limit = math.sqrt(6.0 / float(in_dim + out_dim))
    weight = jax.random.uniform(key, shape=(in_dim, out_dim), minval=-limit, maxval=limit)
    bias = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"weight": weight.astype(jnp.float32), "bias": bias}


def _linear(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    """Apply a dense affine transform: ``output = x @ W + b``.

    Flattens any leading batch dimensions so XLA can lower the operation to a
    single 2-D GEMM on accelerators, then restores the original shape.

    Parameters
    ----------
    params : dict[str, jax.Array]
        ``{"weight": (in_dim, out_dim), "bias": (out_dim,)}``.
    x : jax.Array
        Input tensor with last dimension equal to ``in_dim``.

    Returns
    -------
    jax.Array
        Output tensor with last dimension replaced by ``out_dim``.
    """
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
    """Apply inverted dropout to a hidden activation tensor.

    Parameters
    ----------
    x : jax.Array
        Input activation tensor of any shape.
    rate : float
        Dropout probability in ``[0, 1)``.
    key : jax.Array or None
        PRNG key used to sample the dropout mask.
    training : bool
        Whether dropout should be enabled for this forward pass.

    Returns
    -------
    jax.Array
        Activation tensor with the same shape as ``x``. When dropout is
        active, surviving activations are scaled by ``1 / (1 - rate)``.
    """
    if not training or rate <= 0.0 or key is None:
        return x
    keep_prob = 1.0 - float(rate)
    mask = jax.random.bernoulli(key, p=keep_prob, shape=x.shape)
    return jnp.where(mask, x / keep_prob, jnp.zeros_like(x))


def _init_layer_norm(dim: int) -> dict[str, jax.Array]:
    """Initialize LayerNorm learnable parameters: scale=1, bias=0.

    Parameters
    ----------
    dim : int
        Feature dimension (last axis of the normalized tensor).

    Returns
    -------
    dict[str, jax.Array]
        ``{"scale": (dim,), "bias": (dim,)}`` in float32.
    """
    return {
        "scale": jnp.ones((dim,), dtype=jnp.float32),
        "bias": jnp.zeros((dim,), dtype=jnp.float32),
    }


def _layer_norm(params: dict[str, jax.Array], x: jax.Array, eps: float = 1.0e-5) -> jax.Array:
    """Apply layer normalization (Ba et al., 2016) over the last axis.

    Computes ``y = (x - mean) / sqrt(var + eps) * scale + bias`` where mean
    and variance are computed per-token over the feature dimension.

    Parameters
    ----------
    params : dict[str, jax.Array]
        ``{"scale": (dim,), "bias": (dim,)}``.
    x : jax.Array
        Input tensor of any shape; normalization is over the last axis.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    jax.Array
        Normalized tensor, same shape as *x*.
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    normalized = (x - mean) / jnp.sqrt(var + eps)
    return normalized * params["scale"] + params["bias"]


def sinusoidal_position_encoding(length: int, dim: int, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    """Compute fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Even indices use sin, odd indices use cos:
        PE(pos, 2i)   = sin(pos / 10000^(2i/dim))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))

    Parameters
    ----------
    length : int
        Number of positions (atmospheric levels).
    dim : int
        Encoding dimension (should equal ``d_model``).
    dtype : jnp.dtype
        Output dtype.

    Returns
    -------
    jax.Array
        Positional encoding matrix of shape ``(length, dim)``.
    """
    position = jnp.arange(length, dtype=dtype)[:, None]
    index = jnp.arange(dim, dtype=dtype)[None, :]
    angle_rate = 1.0 / jnp.power(
        _SINUSOIDAL_BASE_WAVELENGTH,
        (2.0 * jnp.floor(index / 2.0)) / float(dim),
    )
    angle = position * angle_rate
    return jnp.where((jnp.arange(dim) % 2)[None, :] == 0, jnp.sin(angle), jnp.cos(angle))



def _spectrum_encoder_key_count(dims: TransformerDimensions | "MLPDimensions") -> int:
    """Return the number of random-init keys required by the spectrum encoder."""
    if dims.spectrum_encoder_mode == "perceiver":
        return 8 + int(dims.spectrum_num_layers) * 6
    return 0


def _masked_mean(
    x: jax.Array,
    mask: jax.Array,
    *,
    axis: int,
    keepdims: bool = False,
) -> jax.Array:
    """Compute a mask-aware mean over one axis."""
    weights = mask.astype(x.dtype)
    while weights.ndim < x.ndim:
        weights = weights[..., None]
    numerator = jnp.sum(x * weights, axis=axis, keepdims=keepdims)
    denominator = jnp.maximum(jnp.sum(weights, axis=axis, keepdims=keepdims), 1.0)
    return numerator / denominator


def _fourier_encode(values: jax.Array, *, num_features: int) -> jax.Array:
    """Return sin/cos Fourier features for a continuous scalar coordinate."""
    if int(num_features) <= 0:
        return values[..., None]
    frequencies = jnp.power(
        2.0,
        jnp.arange(int(num_features), dtype=values.dtype),
    )
    phase = values[..., None] * frequencies * jnp.pi
    return jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)


def _prepare_spectrum_tokens(
    wavelengths_nm: jax.Array,
    fluxes_erg_cm2_s_nm: jax.Array,
    mask: jax.Array,
    dims: TransformerDimensions | "MLPDimensions",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build wavelength-aware token features and spectrum-level summary scalars."""
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


def _init_spectrum_encoder_params(
    key_iter: Any,
    *,
    spectrum_hidden_dim: int,
    spectrum_latent_dim: int,
    spectrum_num_latents: int,
    spectrum_num_layers: int,
    spectrum_fourier_features: int,
    spectrum_encoder_mode: str,
) -> dict[str, Any]:
    """Allocate parameters for the wavelength-aware latent-bottleneck spectrum encoder."""
    if spectrum_encoder_mode != "perceiver":
        return {}

    token_feature_dim = 2 * int(spectrum_fourier_features) + 3
    summary_dim = 6
    params: dict[str, Any] = {
        "token_in": _init_linear(next(key_iter), token_feature_dim, spectrum_hidden_dim),
        "token_ln": _init_layer_norm(spectrum_hidden_dim),
        "cross_ln": _init_layer_norm(spectrum_hidden_dim),
        "cross_q": _init_linear(next(key_iter), spectrum_hidden_dim, spectrum_hidden_dim),
        "cross_k": _init_linear(next(key_iter), spectrum_hidden_dim, spectrum_hidden_dim),
        "cross_v": _init_linear(next(key_iter), spectrum_hidden_dim, spectrum_hidden_dim),
        "cross_o": _init_linear(next(key_iter), spectrum_hidden_dim, spectrum_hidden_dim),
        "latents": (
            0.02
            * jax.random.normal(
                next(key_iter),
                shape=(int(spectrum_num_latents), int(spectrum_hidden_dim)),
                dtype=jnp.float32,
            )
        ),
        "pool_ln": _init_layer_norm(spectrum_hidden_dim),
        "out_hidden": _init_linear(
            next(key_iter),
            spectrum_hidden_dim + summary_dim,
            spectrum_hidden_dim,
        ),
        "out_out": _init_linear(next(key_iter), spectrum_hidden_dim, spectrum_latent_dim),
    }
    latent_blocks: list[dict[str, Any]] = []
    for _ in range(int(spectrum_num_layers)):
        latent_blocks.append(
            {
                "ln1": _init_layer_norm(spectrum_hidden_dim),
                "q": _init_linear(next(key_iter), spectrum_hidden_dim, spectrum_hidden_dim),
                "k": _init_linear(next(key_iter), spectrum_hidden_dim, spectrum_hidden_dim),
                "v": _init_linear(next(key_iter), spectrum_hidden_dim, spectrum_hidden_dim),
                "o": _init_linear(next(key_iter), spectrum_hidden_dim, spectrum_hidden_dim),
                "ln2": _init_layer_norm(spectrum_hidden_dim),
                "ff1": _init_linear(next(key_iter), spectrum_hidden_dim, spectrum_hidden_dim * 4),
                "ff2": _init_linear(next(key_iter), spectrum_hidden_dim * 4, spectrum_hidden_dim),
            }
        )
    params["latent_blocks"] = latent_blocks
    return params


def init_transformer_params(key: jax.Array, dims: TransformerDimensions) -> dict[str, Any]:
    """Allocate and Xavier-initialize all Transformer parameters.

    Splits the PRNG key into enough sub-keys for every linear layer and
    LayerNorm in the model.  The key budget is:
    - 5 top-level: sequence_in, context_in, context_out, out_head_hidden, out_head_final
    - spectrum_key_count: keys for spectrum encoder (Perceiver)
    - 6 per transformer layer: Q, K, V, O projections + 2 FFN layers
    - 4 per transformer layer (VULCAN only): cross-attention Q, K, V, O to spectrum latents

    Parameters
    ----------
    key : jax.Array
        Root PRNG key.
    dims : TransformerDimensions
        Architecture hyperparameters.

    Returns
    -------
    dict[str, Any]
        Nested parameter tree ready for ``apply_model``.
    """
    spectrum_key_count = _spectrum_encoder_key_count(dims)
    cross_attn_keys_per_layer = 4 if dims.spectrum_encoder_mode != "none" else 0
    total_key_count = 5 + spectrum_key_count + dims.num_layers * (6 + cross_attn_keys_per_layer)
    keys = iter(jax.random.split(key, total_key_count))
    params: dict[str, Any] = {
        "sequence_in": _init_linear(next(keys), dims.sequence_dim, dims.d_model),
        "context_in": _init_linear(next(keys), dims.global_dim + dims.spectrum_latent_dim, dims.conditioning_hidden_dim),
        "context_out": _init_linear(next(keys), dims.conditioning_hidden_dim, dims.num_layers * 2 * dims.d_model),
        "out_norm": _init_layer_norm(dims.d_model),
        "out_head_hidden": _init_linear(next(keys), dims.d_model, max(1, dims.d_model // dims.output_head_divisor)),
        "out_head_final": _init_linear(next(keys), max(1, dims.d_model // dims.output_head_divisor), dims.target_dim),
    }
    params["spectrum_encoder"] = _init_spectrum_encoder_params(
        keys,
        spectrum_hidden_dim=dims.spectrum_hidden_dim,
        spectrum_latent_dim=dims.spectrum_latent_dim,
        spectrum_num_latents=dims.spectrum_num_latents,
        spectrum_num_layers=dims.spectrum_num_layers,
        spectrum_fourier_features=dims.spectrum_fourier_features,
        spectrum_encoder_mode=dims.spectrum_encoder_mode,
    )

    layers: list[dict[str, Any]] = []
    for _ in range(dims.num_layers):
        layer_params: dict[str, Any] = {
            "ln1": _init_layer_norm(dims.d_model),
            "q": _init_linear(next(keys), dims.d_model, dims.d_model),
            "k": _init_linear(next(keys), dims.d_model, dims.d_model),
            "v": _init_linear(next(keys), dims.d_model, dims.d_model),
            "o": _init_linear(next(keys), dims.d_model, dims.d_model),
            "ln_ffn": _init_layer_norm(dims.d_model),
            "ff1": _init_linear(next(keys), dims.d_model, dims.dim_feedforward),
            "ff2": _init_linear(next(keys), dims.dim_feedforward, dims.d_model),
        }
        if dims.spectrum_encoder_mode != "none":
            layer_params["ln_cross"] = _init_layer_norm(dims.d_model)
            layer_params["cross_q"] = _init_linear(next(keys), dims.d_model, dims.d_model)
            layer_params["cross_k"] = _init_linear(next(keys), dims.spectrum_hidden_dim, dims.d_model)
            layer_params["cross_v"] = _init_linear(next(keys), dims.spectrum_hidden_dim, dims.d_model)
            layer_params["cross_o"] = _init_linear(next(keys), dims.d_model, dims.d_model)
        layers.append(layer_params)
    params["layers"] = layers
    return params



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
    dims: TransformerDimensions | "MLPDimensions",
    *,
    batch_size: int | None = None,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
    """Compress a variable-length native-grid stellar spectrum into a latent vector.

    Returns
    -------
    latent : jax.Array
        Mean-pooled latent vector ``(batch, spectrum_latent_dim)`` for FiLM conditioning.
    latent_tokens : jax.Array or None
        Un-pooled Perceiver latent tokens ``(batch, num_latents, spectrum_hidden_dim)``
        for cross-attention in the transformer.  ``None`` when encoder is disabled.
    reconstruction : jax.Array or None
        Reserved for future spectrum reconstruction (always ``None``).
    """
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
        latent_norm = _layer_norm(block["ln1"], latents)
        latents = latents + _multihead_attention_qkv(
            block["q"],
            block["k"],
            block["v"],
            block["o"],
            latent_norm,
            latent_norm,
            nhead=dims.spectrum_num_heads,
        )
        ff_in = _layer_norm(block["ln2"], latents)
        ff_hidden = act(_linear(block["ff1"], ff_in))
        latents = latents + _linear(block["ff2"], ff_hidden)

    normed_latents = _layer_norm(params["pool_ln"], latents)
    pooled = jnp.mean(normed_latents, axis=1)
    fused = jnp.concatenate([pooled, summary.astype(pooled.dtype)], axis=-1)
    hidden = act(_linear(params["out_hidden"], fused))
    latent = _linear(params["out_out"], hidden)
    return latent, normed_latents, None



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

    def to_dict(self: "MLPDimensions") -> dict[str, Any]:
        """Serialize the dataclass fields into a plain Python mapping."""
        return asdict(self)

    @classmethod
    def from_dict(
        cls: type["MLPDimensions"],
        payload: dict[str, Any],
    ) -> "MLPDimensions":
        """Rebuild one dimensions dataclass from serialized metadata."""
        return cls(**payload)


def init_mlp_params(
    key: jax.Array, dims: MLPDimensions
) -> dict[str, Any]:
    """Allocate and Xavier-initialize all FiLM-MLP parameters.

    Parameters
    ----------
    key : jax.Array
        Root PRNG key for parameter initialization.
    dims : MLPDimensions
        Architecture hyperparameters defining layer widths and encoder mode.

    Returns
    -------
    dict[str, Any]
        Nested parameter tree for the FiLM-conditioned MLP.
    """
    spectrum_key_count = _spectrum_encoder_key_count(dims)
    total_keys = 2 + spectrum_key_count + dims.num_hidden_layers + 1
    keys = iter(jax.random.split(key, total_keys))
    params: dict[str, Any] = {
        "context_in": _init_linear(
            next(keys), dims.global_dim + dims.spectrum_latent_dim, dims.conditioning_hidden_dim
        ),
        "context_out": _init_linear(
            next(keys),
            dims.conditioning_hidden_dim,
            dims.num_hidden_layers * 2 * dims.d_hidden,
        ),
    }
    params["spectrum_encoder"] = _init_spectrum_encoder_params(
        keys,
        spectrum_hidden_dim=dims.spectrum_hidden_dim,
        spectrum_latent_dim=dims.spectrum_latent_dim,
        spectrum_num_latents=dims.spectrum_num_latents,
        spectrum_num_layers=dims.spectrum_num_layers,
        spectrum_fourier_features=dims.spectrum_fourier_features,
        spectrum_encoder_mode=dims.spectrum_encoder_mode,
    )
    layers: list[dict[str, Any]] = []
    in_dim = dims.sequence_dim
    for _ in range(dims.num_hidden_layers):
        layers.append({
            "linear": _init_linear(next(keys), in_dim, dims.d_hidden),
            "ln": _init_layer_norm(dims.d_hidden),
        })
        in_dim = dims.d_hidden
    params["layers"] = layers
    params["output"] = _init_linear(next(keys), dims.d_hidden, dims.target_dim)
    return params


def _resolve_activation(name: str) -> Callable[[jax.Array], jax.Array]:
    """Resolve an activation name to the corresponding JAX callable.

    Parameters
    ----------
    name : str
        Activation identifier stored in the validated config.

    Returns
    -------
    callable
        JAX-compatible activation function.
    """
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
# Model construction helpers (merged from model.py)
# ---------------------------------------------------------------------------




def build_model_dimensions(
    config: dict, contract: dict
) -> TransformerDimensions | MLPDimensions:
    """Build the dimension dataclass for the active chemistry/model contract."""
    from ..utils.config import uses_fastchem, uses_mlp

    model_cfg = config["training"]["model"]
    sequence_dim = int(
        contract.get(
            "sequence_dim",
            len(contract.get("sequence_static_feature_order", [])),
        )
    )
    global_dim = int(
        contract.get(
            "global_dim",
            len(contract.get("global_static_feature_order", [])),
        )
    )
    target_dim = int(
        contract.get(
            "target_dim",
            len(contract.get("output_species_order", [])),
        )
    )

    if uses_fastchem(config):
        spectrum_max_tokens = 0
        spectrum_latent_dim = 0
        spectrum_hidden_dim = 0
        spectrum_num_latents = 0
        spectrum_num_layers = 0
        spectrum_num_heads = 1
        spectrum_fourier_features = 0
        spectrum_encoder_mode = "none"
        spectrum_floor = float(config["normalization"].get("spectrum_floor", 1.0e-30))
    else:
        spectrum_cfg = config["stellar_spectrum"]
        spectrum_max_tokens = int(contract.get("spectrum_max_tokens", 0))
        spectrum_latent_dim = int(spectrum_cfg["latent_dim"])
        spectrum_hidden_dim = int(spectrum_cfg["hidden_dim"])
        spectrum_num_latents = int(spectrum_cfg["num_latents"])
        spectrum_num_layers = int(spectrum_cfg["num_layers"])
        spectrum_num_heads = int(spectrum_cfg["num_heads"])
        spectrum_fourier_features = int(spectrum_cfg["fourier_features"])
        spectrum_encoder_mode = str(spectrum_cfg["encoder_mode"]).lower()
        spectrum_floor = float(config["normalization"]["spectrum_floor"])

    if uses_mlp(config):
        return MLPDimensions(
            sequence_dim=sequence_dim,
            global_dim=global_dim,
            spectrum_max_tokens=spectrum_max_tokens,
            spectrum_latent_dim=spectrum_latent_dim,
            spectrum_hidden_dim=spectrum_hidden_dim,
            spectrum_num_latents=spectrum_num_latents,
            spectrum_num_layers=spectrum_num_layers,
            spectrum_num_heads=spectrum_num_heads,
            spectrum_fourier_features=spectrum_fourier_features,
            spectrum_encoder_mode=spectrum_encoder_mode,
            spectrum_floor=spectrum_floor,
            target_dim=target_dim,
            d_hidden=int(model_cfg["d_hidden"]),
            num_hidden_layers=int(model_cfg["num_hidden_layers"]),
            conditioning_hidden_dim=int(model_cfg["conditioning_hidden_dim"]),
            film_clamp=float(model_cfg["film_clamp"]),
            activation=str(model_cfg["activation"]).lower(),
            dropout_rate=float(model_cfg.get("dropout_rate", 0.05)),
        )

    return TransformerDimensions(
        sequence_dim=sequence_dim,
        global_dim=global_dim,
        spectrum_max_tokens=spectrum_max_tokens,
        target_dim=target_dim,
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        num_layers=int(model_cfg["num_layers"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        conditioning_hidden_dim=int(model_cfg["conditioning_hidden_dim"]),
        film_clamp=float(model_cfg["film_clamp"]),
        output_head_divisor=int(model_cfg["output_head_divisor"]),
        spectrum_latent_dim=spectrum_latent_dim,
        spectrum_hidden_dim=spectrum_hidden_dim,
        spectrum_num_latents=spectrum_num_latents,
        spectrum_num_layers=spectrum_num_layers,
        spectrum_num_heads=spectrum_num_heads,
        spectrum_fourier_features=spectrum_fourier_features,
        spectrum_encoder_mode=spectrum_encoder_mode,
        spectrum_floor=spectrum_floor,
        activation=str(model_cfg.get("activation", "leaky_relu")).lower(),
        dropout_rate=float(model_cfg.get("dropout_rate", 0.05)),
    )


def initialize_model(
    config: dict, contract: dict, *, seed: int
) -> tuple[TransformerDimensions | MLPDimensions, dict]:
    """Initialize model dimensions and parameters from the validated config.

    Parameters
    ----------
    config : dict
        Validated pipeline config.
    contract : dict
        Processed-data contract defining the tensor dimensions.
    seed : int
        Random seed used for JAX parameter initialization.

    Returns
    -------
    tuple[TransformerDimensions | MLPDimensions, dict]
        Model-dimension dataclass and the initialized parameter tree.
    """
    dims = build_model_dimensions(config, contract)
    key = jax.random.PRNGKey(int(seed))
    if isinstance(dims, MLPDimensions):
        params = init_mlp_params(key, dims)
    else:
        params = init_transformer_params(key, dims)
    return dims, params


def count_parameters(params: dict) -> int:
    """Count the total number of scalar parameters in a nested JAX parameter tree.

    Parameters
    ----------
    params : dict
        Nested parameter tree whose leaves are JAX arrays.

    Returns
    -------
    int
        Total number of scalar entries across all leaves in the parameter
        tree.
    """
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(int(leaf.size) for leaf in leaves))
