"""Shared neural-network primitives for the FastChem and VULCAN emulators.

Contains low-level building blocks (linear layers, LayerNorm, dropout,
positional encoding, activation resolution, and multi-head attention).

All operations are pure JAX and compatible with ``jax.grad`` / ``jax.jvp``.
"""

from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp

from ..constants import _POSITION_SCALE, _SINUSOIDAL_BASE_WAVELENGTH, ATTN_MASK_NEG_INF

# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------


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


def _init_rms_norm(dim: int) -> dict[str, jax.Array]:
    """Initialize RMSNorm learnable parameters: scale=1 only (no bias)."""
    return {"scale": jnp.ones((dim,), dtype=jnp.float32)}


def _rms_norm(params: dict[str, jax.Array], x: jax.Array, eps: float = 1.0e-6) -> jax.Array:
    """Apply RMSNorm (Zhang & Sennrich, 2019) over the last axis.

    Computes ``y = x / sqrt(mean(x^2) + eps) * scale``. Drops the mean
    subtraction and bias of LayerNorm; ~10% faster at equivalent quality.
    """
    rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * params["scale"]


def _init_norm(dim: int, norm_type: str) -> dict[str, jax.Array]:
    """Initialize a normalization block chosen by ``norm_type``."""
    if norm_type == "layernorm":
        return _init_layer_norm(dim)
    if norm_type == "rmsnorm":
        return _init_rms_norm(dim)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def _apply_norm(
    params: dict[str, jax.Array],
    x: jax.Array,
    norm_type: str,
) -> jax.Array:
    """Apply the normalization block selected by ``norm_type``."""
    if norm_type == "layernorm":
        return _layer_norm(params, x)
    if norm_type == "rmsnorm":
        return _rms_norm(params, x)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def sinusoidal_position_encoding_continuous(
    position: jax.Array,
    dim: int,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Continuous sinusoidal positional encoding over a real-valued coordinate.

    Accepts any real-valued ``position`` array (e.g. normalized
    log10-pressure) and returns the same Vaswani-style sin/cos bands as the
    legacy fixed-index helper, but evaluated at the provided positions rather
    than integer indices.
    This is the PE used for the variable-grid emulator: two columns with different ``nz`` but the
    same physical pressures produce identical PE at those pressures.

    The raw position is multiplied by ``_POSITION_SCALE`` so that the
    lowest frequency band spans roughly that many "steps" across the
    training range, keeping the frequency content similar to the legacy
    fixed-index encoding the model was tuned against.

    Parameters
    ----------
    position : jax.Array
        Position coordinate array of shape ``(..., nz)``, typically in
        ``[0, 1]`` representing normalized log10(P).
    dim : int
        Encoding dimension (should equal ``d_model``).
    dtype : jnp.dtype
        Output dtype.

    Returns
    -------
    jax.Array
        Positional encoding tensor of shape ``(..., nz, dim)``.

    Notes
    -----
    Smooth w.r.t. ``position``; ``jax.grad`` flows through.
    """
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


# ---------------------------------------------------------------------------
# Multi-head attention
# ---------------------------------------------------------------------------


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
    q_norm: dict[str, jax.Array] | None = None,
    k_norm: dict[str, jax.Array] | None = None,
) -> jax.Array:
    """Scaled dot-product attention from ``query`` tokens over ``source`` tokens.

    When ``q_norm`` / ``k_norm`` are provided, Q and K are RMSNorm'd along
    the per-head feature axis before the dot product (QK-Norm, Henry et al.
    2020; adopted in Gemini/DeepSeek stacks to prevent attention-logit
    explosion).
    """
    batch_size, query_len, d_model = query.shape
    source_len = source.shape[1]
    head_dim = d_model // nhead

    q = _linear(q_params, query).reshape(batch_size, query_len, nhead, head_dim).transpose(0, 2, 1, 3)
    k = _linear(k_params, source).reshape(batch_size, source_len, nhead, head_dim).transpose(0, 2, 1, 3)
    v = _linear(v_params, source).reshape(batch_size, source_len, nhead, head_dim).transpose(0, 2, 1, 3)

    if q_norm is not None:
        q = _rms_norm(q_norm, q)
    if k_norm is not None:
        k = _rms_norm(k_norm, k)

    scale = 1.0 / math.sqrt(float(head_dim))
    logits = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
    if source_mask is not None:
        expanded_mask = source_mask[:, None, None, :]
        logits = jnp.where(expanded_mask, logits, jnp.full_like(logits, ATTN_MASK_NEG_INF))
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
    """Convenience wrapper for bidirectional self-attention.

    ``key_mask`` (bool, shape ``(batch, nz)``) — when provided, attention
    scores at padded key positions are driven to ``-inf`` before softmax
    so padded tokens cannot contribute to any query's output.

    QK-Norm is applied when the layer params tree contains ``q_norm`` and
    ``k_norm`` entries (added by ``init_transformer_params`` when the
    ``use_qk_norm`` flag is on).
    """
    return _multihead_attention_qkv(
        params["q"],
        params["k"],
        params["v"],
        params["o"],
        x,
        x,
        nhead=nhead,
        source_mask=key_mask,
        q_norm=params.get("q_norm"),
        k_norm=params.get("k_norm"),
    )
