"""FiLM-conditioned Transformer architecture for the FastChem and VULCAN emulators.

Architecture:
    1. Project global_inputs to per-layer FiLM parameters (gamma, beta).
    2. Project per-level sequence to d_model + sinusoidal positional encoding.
    3. Per block:
       a. Pre-norm (ln1) -> multi-head self-attention -> dropout -> residual add
       b. FiLM: x = x * (1 + gamma) + beta
       c. Pre-norm (ln_ffn) -> FFN (up-project, act, dropout, down-project) -> residual add
    4. Output head: LayerNorm -> act -> bottleneck -> final projection.

All operations are pure JAX and compatible with ``jax.grad`` / ``jax.jvp``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .layers import (
    _apply_dropout,
    _init_layer_norm,
    _init_linear,
    _layer_norm,
    _linear,
    _multihead_attention,
    _resolve_activation,
    sinusoidal_position_encoding_continuous,
)


@dataclass(frozen=True)
class TransformerDimensions:
    """All dimensionality and architecture hyper-parameters for the surrogate.

    ``film_clamp`` bounds the per-layer FiLM gamma/beta magnitudes. Typical
    safe values are ``[5.0, 20.0]`` in float32; values above ~65 risk pushing
    activations outside the float16 dynamic range (relevant if mixed-precision
    training is ever enabled).
    """

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


def init_transformer_params(key: jax.Array, dims: TransformerDimensions) -> dict[str, Any]:
    """Allocate and Xavier-initialize all Transformer parameters.

    Splits the PRNG key into enough sub-keys for every linear layer and
    LayerNorm in the model.  The key budget is:
    - 5 top-level: sequence_in, context_in, context_out, out_head_hidden, out_head_final
    - 6 per transformer layer: Q, K, V, O projections + 2 FFN layers

    Parameters
    ----------
    key : jax.Array
        Root PRNG key.
    dims : TransformerDimensions
        Architecture hyperparameters.

    Returns
    -------
    dict[str, Any]
        Nested parameter tree ready for ``apply_transformer_model``.
    """
    total_key_count = 5 + dims.num_layers * 6
    keys = iter(jax.random.split(key, total_key_count))
    params: dict[str, Any] = {
        "sequence_in": _init_linear(next(keys), dims.sequence_dim, dims.d_model),
        "context_in": _init_linear(next(keys), dims.global_dim, dims.conditioning_hidden_dim),
        "context_out": _init_linear(next(keys), dims.conditioning_hidden_dim, dims.num_layers * 2 * dims.d_model),
        "out_norm": _init_layer_norm(dims.d_model),
        "out_head_hidden": _init_linear(next(keys), dims.d_model, max(1, dims.d_model // dims.output_head_divisor)),
        "out_head_final": _init_linear(next(keys), max(1, dims.d_model // dims.output_head_divisor), dims.target_dim),
    }

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
        layers.append(layer_params)
    params["layers"] = layers
    return params


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
    """Run the Transformer forward pass in normalized space.

    Parameters
    ----------
    position_coord : jax.Array
        Per-level position coordinate of shape ``(batch, nz)``, typically
        normalized log10(pressure) in ``[0, 1]``. Used to build the
        continuous sinusoidal positional encoding.
    attention_mask : jax.Array or None
        Optional per-level validity mask of shape ``(batch, nz)`` (bool
        or {0, 1}). Padded positions (False / 0) are excluded from every
        layer's attention key set. When ``None``, every position is
        treated as valid.
    """
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
    _keys_per_layer = 2  # attn + ffn
    layer_dropout_keys: list[tuple[jax.Array | None, ...]] = [
        (None,) * _keys_per_layer for _ in range(dims.num_layers)
    ]
    output_dropout_key: jax.Array | None = None
    if training and dims.dropout_rate > 0.0 and dropout_key is not None:
        total_dropout_keys = (dims.num_layers * _keys_per_layer) + 1
        dropout_keys = jax.random.split(dropout_key, total_dropout_keys)
        if dropout_keys.shape[0] != total_dropout_keys:
            raise RuntimeError(
                f"Expected {total_dropout_keys} dropout keys, got {dropout_keys.shape[0]}."
            )
        layer_dropout_keys = [
            tuple(
                dropout_keys[layer_idx * _keys_per_layer + within_layer]
                for within_layer in range(_keys_per_layer)
            )
            for layer_idx in range(dims.num_layers)
        ]
        output_dropout_key = dropout_keys[dims.num_layers * _keys_per_layer]

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
        attn = _multihead_attention(
            layer, x_norm, nhead=dims.nhead, key_mask=attention_mask,
        )
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
