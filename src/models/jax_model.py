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
    activation : str
        Hidden activation applied throughout the model (``"gelu"``, ``"relu"``,
        or ``"silu"``).
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
    activation: str = "gelu"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dimensions dataclass to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelDimensions":
        """Reconstruct dimensions from a serialized dictionary."""
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


def init_model_params(key: jax.Array, dims: ModelDimensions) -> dict[str, Any]:
    """Allocate and Xavier-initialize all full-VULCAN Transformer parameters.

    Splits the PRNG key into enough sub-keys for every linear layer and
    LayerNorm in the model.  The key budget is:
    - 5 top-level: sequence_in, context_in, context_out, out_head_hidden, out_head_final
    - spectrum_key_count: 4 for autoencoder, 1 for linear, 0 for none
    - 6 per transformer layer: Q, K, V, O projections + 2 FFN layers

    Parameters
    ----------
    key : jax.Array
        Root PRNG key.
    dims : ModelDimensions
        Architecture hyperparameters.

    Returns
    -------
    dict[str, Any]
        Nested parameter tree ready for ``apply_model``.
    """
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
    """Scaled-dot-product multi-head self-attention (Vaswani et al., 2017).

    Computes Q, K, V projections, splits into ``nhead`` heads, applies
    ``softmax(Q K^T / sqrt(d_k)) V``, concatenates heads, and projects
    back to ``d_model``.  No masking is applied (bidirectional attention).

    Parameters
    ----------
    params : dict[str, jax.Array]
        Must contain ``"q"``, ``"k"``, ``"v"``, ``"o"`` linear layer params.
    x : jax.Array
        Input tensor of shape ``(batch, nlevel, d_model)``.
    nhead : int
        Number of attention heads.

    Returns
    -------
    jax.Array
        Output tensor of shape ``(batch, nlevel, d_model)``.
    """
    batch_size, nlevel, d_model = x.shape
    head_dim = d_model // nhead

    # Project to Q, K, V and reshape to [batch, heads, levels, head_dim].
    q = _linear(params["q"], x).reshape(batch_size, nlevel, nhead, head_dim).transpose(0, 2, 1, 3)
    k = _linear(params["k"], x).reshape(batch_size, nlevel, nhead, head_dim).transpose(0, 2, 1, 3)
    v = _linear(params["v"], x).reshape(batch_size, nlevel, nhead, head_dim).transpose(0, 2, 1, 3)

    # Scaled dot-product attention.
    scale = 1.0 / math.sqrt(float(head_dim))
    logits = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
    weights = jax.nn.softmax(logits, axis=-1)
    attended = jnp.einsum("bhij,bhjd->bhid", weights, v)

    # Concatenate heads and project back to d_model.
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
        act = _resolve_activation(dims.activation)
        hidden = act(_linear(params["encoder_1"], spectrum_inputs))
        latent = _linear(params["encoder_2"], hidden)
        recon_hidden = act(_linear(params["decoder_1"], latent))
        reconstruction = _linear(params["decoder_2"], recon_hidden)
        return latent, reconstruction
    raise ValueError(f"Unsupported spectrum encoder mode: {mode}")


@dataclass(frozen=True)
class EquilibriumMLPDimensions:
    """Dimensionality and architecture hyper-parameters for the equilibrium MLP.

    The equilibrium model uses a per-level MLP (shared weights across the
    vertical grid) with FiLM conditioning from global scalars.  No positional
    encoding or self-attention — thermochemical equilibrium is local.

    Fields
    ------
    sequence_dim : int
        Per-level input width (P, T).
    global_dim : int
        Width of the global conditioning vector (metallicity, C/O, S/O).
    target_dim : int
        Number of output species per level.
    d_hidden : int
        Hidden width of the per-level MLP.
    num_hidden_layers : int
        Number of hidden layers in the per-level MLP.
    conditioning_hidden_dim : int
        Hidden width of the FiLM conditioning MLP.
    film_clamp : float
        Symmetric clamp applied to FiLM gamma / beta.
    activation : str
        Hidden activation applied in the context and per-level MLP blocks.
    """

    sequence_dim: int
    global_dim: int
    target_dim: int
    d_hidden: int
    num_hidden_layers: int
    conditioning_hidden_dim: int
    film_clamp: float
    activation: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the equilibrium dimensions dataclass to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EquilibriumMLPDimensions":
        """Reconstruct equilibrium dimensions from a serialized dictionary."""
        return cls(**payload)


def init_equilibrium_mlp_params(
    key: jax.Array, dims: EquilibriumMLPDimensions
) -> dict[str, Any]:
    """Allocate and Xavier-initialise all equilibrium MLP parameters."""
    total_keys = 2 + dims.num_hidden_layers + 1  # context_in, context_out, hidden layers, output
    keys = iter(jax.random.split(key, total_keys))
    params: dict[str, Any] = {
        "context_in": _init_linear(
            next(keys), dims.global_dim, dims.conditioning_hidden_dim
        ),
        "context_out": _init_linear(
            next(keys),
            dims.conditioning_hidden_dim,
            dims.num_hidden_layers * 2 * dims.d_hidden,
        ),
    }
    layers: list[dict[str, Any]] = []
    in_dim = dims.sequence_dim
    for _ in range(dims.num_hidden_layers):
        layers.append({"linear": _init_linear(next(keys), in_dim, dims.d_hidden)})
        in_dim = dims.d_hidden
    params["layers"] = layers
    params["output"] = _init_linear(next(keys), dims.d_hidden, dims.target_dim)
    return params


def _resolve_activation(name: str):
    """Return the JAX activation function for the given name."""
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


def apply_equilibrium_mlp(
    params: dict[str, Any],
    sequence_inputs: jax.Array,
    global_inputs: jax.Array,
    dims: EquilibriumMLPDimensions,
) -> tuple[jax.Array, dict]:
    """Forward pass for the equilibrium FiLM-MLP.

    Parameters
    ----------
    params : dict
        Parameter tree from :func:`init_equilibrium_mlp_params`.
    sequence_inputs : jax.Array
        Per-level features ``[batch, nz, sequence_dim]`` (P, T).
    global_inputs : jax.Array
        Global conditioning ``[batch, global_dim]``.
    dims : EquilibriumMLPDimensions
        Architecture dimensions.

    Returns
    -------
    pred : jax.Array
        ``[batch, nz, target_dim]`` predictions in normalised space.
    aux : dict
        Empty auxiliary dict (kept for API symmetry with the transformer).
    """
    if sequence_inputs.ndim != 3:
        raise ValueError("sequence_inputs must have shape [batch, nz, feature_dim].")
    if global_inputs.ndim != 2:
        raise ValueError("global_inputs must have shape [batch, global_dim].")

    activation = _resolve_activation(dims.activation)

    # Conditioning: globals → per-layer FiLM parameters.
    context = activation(_linear(params["context_in"], global_inputs))
    film = _linear(params["context_out"], context)
    film = film.reshape(
        global_inputs.shape[0], dims.num_hidden_layers, 2, dims.d_hidden
    )

    # Per-level MLP with FiLM modulation at each hidden layer.
    x = sequence_inputs
    for layer_index, layer in enumerate(params["layers"]):
        x = _linear(layer["linear"], x)
        gamma = jnp.clip(
            film[:, layer_index, 0], -dims.film_clamp, dims.film_clamp
        )
        beta = jnp.clip(
            film[:, layer_index, 1], -dims.film_clamp, dims.film_clamp
        )
        x = x * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        x = activation(x)

    pred = _linear(params["output"], x)
    return pred, {}


def apply_model(
    params: dict[str, Any],
    sequence_inputs: jax.Array,
    global_inputs: jax.Array,
    spectrum_inputs: jax.Array,
    dims: ModelDimensions,
) -> tuple[jax.Array, dict[str, jax.Array | None]]:
    """Run the full-VULCAN Transformer forward pass in normalized space.

    Architecture flow:
    1. Encode the stellar spectrum into a latent vector.
    2. Concatenate [global_inputs, spectrum_latent] and project through
       the FiLM conditioning MLP to produce per-layer gamma/beta.
    3. Project per-level sequence inputs to d_model and add sinusoidal
       positional encoding.
    4. Run L pre-norm Transformer blocks.  In each block:
       a. LayerNorm -> multi-head self-attention -> residual
       b. FiLM modulation: x = x * (1 + gamma) + beta
       c. LayerNorm -> FFN (up-project, activation, down-project) -> residual
    5. Output head: LayerNorm -> activation -> bottleneck -> final projection.

    Parameters
    ----------
    params : dict
        Nested parameter tree from ``init_model_params``.
    sequence_inputs : jax.Array
        Per-level features ``[batch, nz, sequence_dim]``.
    global_inputs : jax.Array
        Global conditioning ``[batch, global_dim]``.
    spectrum_inputs : jax.Array
        Stellar spectrum ``[batch, spectrum_dim]``.
    dims : ModelDimensions
        Architecture hyperparameters.

    Returns
    -------
    tuple[jax.Array, dict]
        ``(pred, aux)`` where pred has shape ``[batch, nz, target_dim]``
        and aux contains ``spectrum_reconstruction`` and ``spectrum_latent``.
    """
    if sequence_inputs.ndim != 3:
        raise ValueError("sequence_inputs must have shape [batch, nz, feature_dim].")
    if global_inputs.ndim != 2:
        raise ValueError("global_inputs must have shape [batch, global_dim].")
    if spectrum_inputs.ndim != 2:
        raise ValueError("spectrum_inputs must have shape [batch, spectrum_dim].")

    act = _resolve_activation(dims.activation)

    # Encode the stellar spectrum and combine with global scalars for FiLM.
    latent, reconstruction = _encode_spectrum(params["spectrum_encoder"], spectrum_inputs, dims)
    context = jnp.concatenate([global_inputs, latent], axis=-1)
    context = act(_linear(params["context_in"], context))
    # Produce per-layer FiLM parameters: [batch, num_layers, 2 (gamma/beta), d_model].
    film = _linear(params["context_out"], context)
    film = film.reshape(sequence_inputs.shape[0], dims.num_layers, 2, dims.d_model)

    # Project per-level inputs and add fixed positional encoding.
    x = _linear(params["sequence_in"], sequence_inputs)
    x = x + sinusoidal_position_encoding(sequence_inputs.shape[1], dims.d_model, x.dtype)[None, :, :]

    # Pre-norm transformer blocks with FiLM modulation after attention.
    for layer_index, layer in enumerate(params["layers"]):
        gamma = jnp.clip(film[:, layer_index, 0], -dims.film_clamp, dims.film_clamp)
        beta = jnp.clip(film[:, layer_index, 1], -dims.film_clamp, dims.film_clamp)
        x_norm = _layer_norm(layer["ln1"], x)
        attn = _multihead_attention(layer, x_norm, nhead=dims.nhead)
        x = x + attn  # Residual connection from attention.
        x = x * (1.0 + gamma[:, None, :]) + beta[:, None, :]  # FiLM conditioning.
        ff_in = _layer_norm(layer["ln2"], x)
        ff_hidden = act(_linear(layer["ff1"], ff_in))
        ff = _linear(layer["ff2"], ff_hidden)
        x = x + ff  # Residual connection from feed-forward.

    # Output head: layer-norm → bottleneck → final projection.
    x = _layer_norm(params["out_norm"], x)
    x = act(_linear(params["out_head_hidden"], x))
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
) -> ModelDimensions | EquilibriumMLPDimensions:
    """Build the model-dimension dataclass for the active model type."""
    from ..utils.config import is_equilibrium

    model_cfg = config["training"]["model"]
    if is_equilibrium(config):
        return EquilibriumMLPDimensions(
            sequence_dim=int(contract["sequence_dim"]),
            global_dim=int(contract["global_dim"]),
            target_dim=int(contract["target_dim"]),
            d_hidden=int(model_cfg["d_hidden"]),
            num_hidden_layers=int(model_cfg["num_hidden_layers"]),
            conditioning_hidden_dim=int(model_cfg["conditioning_hidden_dim"]),
            film_clamp=float(model_cfg["film_clamp"]),
            activation=str(model_cfg["activation"]).lower(),
        )
    spectrum_cfg = config["stellar_spectrum"]
    return ModelDimensions(
        sequence_dim=int(contract["sequence_dim"]),
        global_dim=len(contract["global_feature_order"]),
        spectrum_dim=int(contract["spectrum_dim"]),
        target_dim=int(contract["target_dim"]),
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        num_layers=int(model_cfg["num_layers"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        conditioning_hidden_dim=int(model_cfg["conditioning_hidden_dim"]),
        film_clamp=float(model_cfg["film_clamp"]),
        output_head_divisor=int(model_cfg["output_head_divisor"]),
        spectrum_latent_dim=int(spectrum_cfg["latent_dim"]),
        spectrum_hidden_dim=int(spectrum_cfg["hidden_dim"]),
        spectrum_encoder_mode=str(spectrum_cfg["encoder_mode"]).lower(),
        activation=str(model_cfg.get("activation", "gelu")).lower(),
    )


def initialize_model(
    config: dict, contract: dict, *, seed: int
) -> tuple[ModelDimensions | EquilibriumMLPDimensions, dict]:
    """Initialize model dimensions and parameters from the validated config."""
    dims = build_model_dimensions(config, contract)
    key = jax.random.PRNGKey(int(seed))
    if isinstance(dims, EquilibriumMLPDimensions):
        params = init_equilibrium_mlp_params(key, dims)
    else:
        params = init_model_params(key, dims)
    return dims, params


def count_parameters(params: dict) -> int:
    """Count the total number of scalar parameters in a nested JAX parameter tree.

    Flattens the tree to its leaf arrays and sums their sizes.  Useful for
    logging model complexity.
    """
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(int(leaf.size) for leaf in leaves))
