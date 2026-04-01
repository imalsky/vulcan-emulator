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
    1. Encode the stellar spectrum into a latent vector (VULCAN only).
    2. Concatenate [global_inputs, spectrum_latent] and project to
       per-layer FiLM parameters (gamma, beta).
    3. Project per-level sequence to d_model + sinusoidal positional encoding.
    4. Per block:
       a. Pre-norm (ln1) → multi-head self-attention → dropout → residual add
       b. FiLM: x = x * (1 + gamma) + beta
       c. Post-FiLM norm (ln_film) → re-stabilize residual stream
       d. Pre-norm (ln2) → FFN (up-project, act, dropout, down-project) → residual add
    5. Output head: LayerNorm → act → bottleneck → final projection.

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
class TransformerDimensions:
    """All dimensionality and architecture hyper-parameters for the surrogate.

    Fields
    ------
    sequence_dim : int
        Per-level input width (pressure, temperature, Kzz for full-VULCAN).
    global_dim : int
        Width of the full global conditioning vector.
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
        Hidden activation applied throughout the model.
    dropout_rate : float
        Dropout probability applied to hidden activations during training.
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
    dropout_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize transformer dimension metadata into plain Python types.

        Returns
        -------
        dict[str, Any]
            Dictionary of scalar architecture hyperparameters suitable for
            checkpointing and JSON-compatible metadata payloads.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TransformerDimensions":
        """Reconstruct transformer dimensions from serialized metadata.

        Parameters
        ----------
        payload : dict[str, Any]
            Dictionary containing the fields required by
            ``TransformerDimensions``.

        Returns
        -------
        TransformerDimensions
            Dataclass instance rebuilt from the serialized payload.
        """
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


def _init_spectrum_encoder_params(
    key_iter: Any,
    *,
    spectrum_dim: int,
    spectrum_hidden_dim: int,
    spectrum_latent_dim: int,
    spectrum_encoder_mode: str,
) -> dict[str, Any]:
    """Allocate spectrum-encoder parameters for the selected encoder mode.

    Parameters
    ----------
    key_iter : Any
        Iterator yielding JAX PRNG keys.
    spectrum_dim : int
        Input spectrum width.
    spectrum_hidden_dim : int
        Hidden width used by the autoencoder variant.
    spectrum_latent_dim : int
        Output latent width exposed to the main model.
    spectrum_encoder_mode : str
        Encoder mode: ``"autoencoder"``, ``"linear"``, or ``"none"``.

    Returns
    -------
    dict[str, Any]
        Nested parameter tree for the selected encoder variant.
    """
    if spectrum_encoder_mode == "autoencoder":
        return {
            "encoder_1": _init_linear(next(key_iter), spectrum_dim, spectrum_hidden_dim),
            "encoder_2": _init_linear(next(key_iter), spectrum_hidden_dim, spectrum_latent_dim),
            "decoder_1": _init_linear(next(key_iter), spectrum_latent_dim, spectrum_hidden_dim),
            "decoder_2": _init_linear(next(key_iter), spectrum_hidden_dim, spectrum_dim),
        }
    if spectrum_encoder_mode == "linear":
        return {
            "encoder_1": _init_linear(next(key_iter), spectrum_dim, spectrum_latent_dim),
        }
    return {}


def init_transformer_params(key: jax.Array, dims: TransformerDimensions) -> dict[str, Any]:
    """Allocate and Xavier-initialize all Transformer parameters.

    Splits the PRNG key into enough sub-keys for every linear layer and
    LayerNorm in the model.  The key budget is:
    - 5 top-level: sequence_in, context_in, context_out, out_head_hidden, out_head_final
    - spectrum_key_count: 4 for autoencoder, 1 for linear, 0 for none
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
    params["spectrum_encoder"] = _init_spectrum_encoder_params(
        keys,
        spectrum_dim=dims.spectrum_dim,
        spectrum_hidden_dim=dims.spectrum_hidden_dim,
        spectrum_latent_dim=dims.spectrum_latent_dim,
        spectrum_encoder_mode=dims.spectrum_encoder_mode,
    )

    layers: list[dict[str, Any]] = []
    for _ in range(dims.num_layers):
        layers.append(
            {
                "ln1": _init_layer_norm(dims.d_model),
                "q": _init_linear(next(keys), dims.d_model, dims.d_model),
                "k": _init_linear(next(keys), dims.d_model, dims.d_model),
                "v": _init_linear(next(keys), dims.d_model, dims.d_model),
                "o": _init_linear(next(keys), dims.d_model, dims.d_model),
                "ln_film": _init_layer_norm(dims.d_model),
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
    spectrum_inputs: jax.Array | None,
    dims: TransformerDimensions | "MLPDimensions",
    *,
    dropout_keys: tuple[jax.Array | None, jax.Array | None] | None = None,
    training: bool = False,
    batch_size: int | None = None,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array, jax.Array | None]:
    """Compress the stellar spectrum into a latent vector.

    Parameters
    ----------
    params : dict[str, Any]
        Spectrum-encoder parameter subtree.
    spectrum_inputs : jax.Array or None
        Batched spectrum tensor with shape ``(batch, spectrum_dim)`` when the
        encoder is enabled.
    dims : TransformerDimensions or MLPDimensions
        Model dimensions containing the spectrum encoder configuration.
    dropout_keys : tuple[jax.Array | None, jax.Array | None] or None, optional
        Optional encoder and decoder dropout keys used in autoencoder mode.
    training : bool, default=False
        Whether dropout should be active.
    batch_size : int or None, optional
        Explicit batch size used when ``spectrum_encoder_mode == "none"``.
    dtype : jnp.dtype, default=jnp.float32
        Output dtype for synthesized latent vectors.

    Returns
    -------
    tuple[jax.Array, jax.Array | None]
        ``(latent, reconstruction)`` where ``latent`` has shape
        ``(batch, spectrum_latent_dim)`` and ``reconstruction`` is non-``None``
        only in autoencoder mode.
    """
    mode = dims.spectrum_encoder_mode
    if mode == "none":
        resolved_batch = batch_size if batch_size is not None else int(spectrum_inputs.shape[0])
        latent = jnp.zeros((resolved_batch, dims.spectrum_latent_dim), dtype=dtype)
        return latent, None
    if spectrum_inputs is None:
        raise ValueError("spectrum_inputs must be provided when the spectrum encoder is enabled.")
    if mode == "linear":
        latent = _linear(params["encoder_1"], spectrum_inputs)
        return latent, None
    if mode == "autoencoder":
        act = _resolve_activation(dims.activation)
        encoder_dropout_key = None if dropout_keys is None else dropout_keys[0]
        decoder_dropout_key = None if dropout_keys is None else dropout_keys[1]
        hidden = act(_linear(params["encoder_1"], spectrum_inputs))
        hidden = _apply_dropout(
            hidden,
            rate=dims.dropout_rate,
            key=encoder_dropout_key,
            training=training,
        )
        latent = _linear(params["encoder_2"], hidden)
        recon_hidden = act(_linear(params["decoder_1"], latent))
        recon_hidden = _apply_dropout(
            recon_hidden,
            rate=dims.dropout_rate,
            key=decoder_dropout_key,
            training=training,
        )
        reconstruction = _linear(params["decoder_2"], recon_hidden)
        return latent, reconstruction
    raise ValueError(f"Unsupported spectrum encoder mode: {mode}")


@dataclass(frozen=True)
class MLPDimensions:
    """Dimensionality and architecture hyper-parameters for the FiLM MLP.

    The MLP uses a per-level backbone (shared weights across the vertical
    grid) with FiLM conditioning from global scalars and, when enabled, a
    stellar-spectrum latent vector.

    Fields
    ------
    sequence_dim : int
        Per-level input width (P, T).
    global_dim : int
        Width of the global conditioning vector (derived equilibrium chemistry globals).
    spectrum_dim : int
        Number of wavelength bins in the input spectrum. Zero when unused.
    spectrum_latent_dim : int
        Dimension of the spectrum encoder latent vector. Zero when unused.
    spectrum_hidden_dim : int
        Hidden width inside the spectrum autoencoder.
    spectrum_encoder_mode : str
        One of ``"autoencoder"``, ``"linear"``, or ``"none"``.
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
    dropout_rate : float
        Dropout probability applied to hidden activations during training.
    """

    sequence_dim: int
    global_dim: int
    spectrum_dim: int
    spectrum_latent_dim: int
    spectrum_hidden_dim: int
    spectrum_encoder_mode: str
    target_dim: int
    d_hidden: int
    num_hidden_layers: int
    conditioning_hidden_dim: int
    film_clamp: float
    activation: str
    dropout_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize MLP dimension metadata into plain Python types.

        Returns
        -------
        dict[str, Any]
            Dictionary of scalar architecture hyperparameters suitable for
            checkpointing and exported metadata.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MLPDimensions":
        """Reconstruct MLP dimensions from serialized metadata.

        Parameters
        ----------
        payload : dict[str, Any]
            Dictionary containing the fields required by ``MLPDimensions``.

        Returns
        -------
        MLPDimensions
            Dataclass instance rebuilt from the serialized payload.
        """
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
    if dims.spectrum_encoder_mode == "autoencoder":
        spectrum_key_count = 4
    elif dims.spectrum_encoder_mode == "linear":
        spectrum_key_count = 1
    else:
        spectrum_key_count = 0
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
        spectrum_dim=dims.spectrum_dim,
        spectrum_hidden_dim=dims.spectrum_hidden_dim,
        spectrum_latent_dim=dims.spectrum_latent_dim,
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


def _resolve_activation(name: str):
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
    spectrum_inputs: jax.Array | None = None,
    *,
    dropout_key: jax.Array | None = None,
    training: bool = False,
) -> tuple[jax.Array, dict]:
    """Forward pass for the FiLM-MLP.

    Architecture flow:
    1. Encode the stellar spectrum into a latent (or zeros if disabled).
    2. [global_inputs, spectrum_latent] → conditioning MLP → per-layer FiLM.
    3. Per-level loop with shared weights across the vertical grid:
       - Linear → LayerNorm → FiLM(gamma, beta) → activation → dropout
       - Residual add from layer index ≥ 1 (where dims match at d_hidden).
    4. Linear output projection → target_dim.

    Parameters
    ----------
    params : dict
        Parameter tree from :func:`init_mlp_params`.
    sequence_inputs : jax.Array
        Per-level features ``[batch, nz, sequence_dim]`` (P, T).
    global_inputs : jax.Array
        Global conditioning ``[batch, global_dim]``.
    dims : MLPDimensions
        Architecture dimensions.
    spectrum_inputs : jax.Array or None
        Optional stellar spectrum ``[batch, spectrum_dim]``.
    dropout_key : jax.Array or None
        PRNG key used for hidden-layer dropout during training.
    training : bool
        Whether to enable dropout.

    Returns
    -------
    pred : jax.Array
        ``[batch, nz, target_dim]`` predictions in normalised space.
    aux : dict
        Auxiliary dict containing ``spectrum_reconstruction`` and ``spectrum_latent``.
    """
    if sequence_inputs.ndim != 3:
        raise ValueError("sequence_inputs must have shape [batch, nz, feature_dim].")
    if global_inputs.ndim != 2:
        raise ValueError("global_inputs must have shape [batch, global_dim].")
    if dims.spectrum_dim > 0:
        if spectrum_inputs is None or spectrum_inputs.ndim != 2:
            raise ValueError("spectrum_inputs must have shape [batch, spectrum_dim] when enabled.")
    elif spectrum_inputs is not None and spectrum_inputs.ndim != 2:
        raise ValueError("spectrum_inputs must have shape [batch, spectrum_dim].")

    activation = _resolve_activation(dims.activation)
    spectrum_dropout_keys: tuple[jax.Array | None, jax.Array | None] | None = None
    layer_dropout_keys = [None] * dims.num_hidden_layers
    if training and dims.dropout_rate > 0.0 and dropout_key is not None:
        total_dropout_keys = dims.num_hidden_layers + 2
        dropout_keys = iter(jax.random.split(dropout_key, total_dropout_keys))
        spectrum_dropout_keys = (next(dropout_keys), next(dropout_keys))
        layer_dropout_keys = [next(dropout_keys) for _ in range(dims.num_hidden_layers)]

    latent, reconstruction = _encode_spectrum(
        params.get("spectrum_encoder", {}),
        spectrum_inputs,
        dims,
        dropout_keys=spectrum_dropout_keys,
        training=training,
        batch_size=int(global_inputs.shape[0]),
        dtype=global_inputs.dtype,
    )
    context_inputs = jnp.concatenate([global_inputs, latent], axis=-1)
    context = activation(_linear(params["context_in"], context_inputs))
    film = _linear(params["context_out"], context)
    film = film.reshape(
        global_inputs.shape[0], dims.num_hidden_layers, 2, dims.d_hidden
    )

    # Per-level MLP with LayerNorm, FiLM modulation, and residual connections.
    x = sequence_inputs
    for layer_index, (layer, layer_key) in enumerate(zip(params["layers"], layer_dropout_keys)):
        residual = x if layer_index >= 1 else None
        x = _linear(layer["linear"], x)
        x = _layer_norm(layer["ln"], x)
        gamma = jnp.clip(
            film[:, layer_index, 0], -dims.film_clamp, dims.film_clamp
        )
        beta = jnp.clip(
            film[:, layer_index, 1], -dims.film_clamp, dims.film_clamp
        )
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
    spectrum_inputs: jax.Array | None,
    dims: TransformerDimensions,
    *,
    dropout_key: jax.Array | None = None,
    training: bool = False,
) -> tuple[jax.Array, dict[str, jax.Array | None]]:
    """Run the Transformer forward pass in normalized space.

    Architecture flow:
    1. Encode the stellar spectrum into a latent vector.
    2. Concatenate [global_inputs, spectrum_latent] and project through
       the FiLM conditioning MLP to produce per-layer gamma/beta.
    3. Project per-level sequence inputs to d_model and add sinusoidal
       positional encoding.
    4. Run L pre-norm Transformer blocks.  In each block:
       a. LayerNorm (ln1) -> multi-head self-attention -> dropout -> residual add
       b. FiLM modulation: x = x * (1 + gamma) + beta
       c. Post-FiLM LayerNorm (ln_film) -> re-stabilize residual stream
       d. LayerNorm (ln2) -> FFN (up-project, activation, dropout, down-project) -> residual add
    5. Output head: LayerNorm -> activation -> bottleneck -> final projection.

    Parameters
    ----------
    params : dict
        Nested parameter tree from ``init_transformer_params``.
    sequence_inputs : jax.Array
        Per-level features ``[batch, nz, sequence_dim]``.
    global_inputs : jax.Array
        Global conditioning ``[batch, global_dim]``.
    spectrum_inputs : jax.Array or None
        Stellar spectrum ``[batch, spectrum_dim]`` when enabled.
    dims : TransformerDimensions
        Architecture hyperparameters.
    dropout_key : jax.Array or None
        PRNG key used for hidden-activation dropout during training.
    training : bool
        Whether to enable dropout.

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
    if dims.spectrum_dim > 0:
        if spectrum_inputs is None or spectrum_inputs.ndim != 2:
            raise ValueError("spectrum_inputs must have shape [batch, spectrum_dim] when enabled.")
    elif spectrum_inputs is not None and spectrum_inputs.ndim != 2:
        raise ValueError("spectrum_inputs must have shape [batch, spectrum_dim].")

    act = _resolve_activation(dims.activation)
    spectrum_dropout_keys: tuple[jax.Array | None, jax.Array | None] | None = None
    layer_dropout_keys: list[tuple[jax.Array | None, jax.Array | None]] = [
        (None, None)
        for _ in range(dims.num_layers)
    ]
    output_dropout_key: jax.Array | None = None
    if training and dims.dropout_rate > 0.0 and dropout_key is not None:
        total_dropout_keys = 2 + (dims.num_layers * 2) + 1
        dropout_keys = iter(jax.random.split(dropout_key, total_dropout_keys))
        spectrum_dropout_keys = (next(dropout_keys), next(dropout_keys))
        layer_dropout_keys = [
            (next(dropout_keys), next(dropout_keys))
            for _ in range(dims.num_layers)
        ]
        output_dropout_key = next(dropout_keys)

    # Encode the stellar spectrum and combine with global scalars for FiLM.
    latent, reconstruction = _encode_spectrum(
        params.get("spectrum_encoder", {}),
        spectrum_inputs,
        dims,
        dropout_keys=spectrum_dropout_keys,
        training=training,
        batch_size=int(global_inputs.shape[0]),
        dtype=global_inputs.dtype,
    )
    context = jnp.concatenate([global_inputs, latent], axis=-1)
    context = act(_linear(params["context_in"], context))
    # Produce per-layer FiLM parameters: [batch, num_layers, 2 (gamma/beta), d_model].
    film = _linear(params["context_out"], context)
    film = film.reshape(sequence_inputs.shape[0], dims.num_layers, 2, dims.d_model)

    # Project per-level inputs and add fixed positional encoding.
    x = _linear(params["sequence_in"], sequence_inputs)
    x = x + sinusoidal_position_encoding(sequence_inputs.shape[1], dims.d_model, x.dtype)[None, :, :]

    # Pre-norm transformer blocks with FiLM modulation after attention.
    for layer_index, (layer, dropout_keys) in enumerate(zip(params["layers"], layer_dropout_keys)):
        attn_dropout_key, ff_dropout_key = dropout_keys
        gamma = jnp.clip(film[:, layer_index, 0], -dims.film_clamp, dims.film_clamp)
        beta = jnp.clip(film[:, layer_index, 1], -dims.film_clamp, dims.film_clamp)
        x_norm = _layer_norm(layer["ln1"], x)
        attn = _multihead_attention(layer, x_norm, nhead=dims.nhead)
        attn = _apply_dropout(
            attn,
            rate=dims.dropout_rate,
            key=attn_dropout_key,
            training=training,
        )
        x = x + attn  # Residual connection from attention.
        x = x * (1.0 + gamma[:, None, :]) + beta[:, None, :]  # FiLM conditioning.
        x = _layer_norm(layer["ln_film"], x)  # Re-stabilize after FiLM.
        ff_in = _layer_norm(layer["ln2"], x)
        ff_hidden = act(_linear(layer["ff1"], ff_in))
        ff_hidden = _apply_dropout(
            ff_hidden,
            rate=dims.dropout_rate,
            key=ff_dropout_key,
            training=training,
        )
        ff = _linear(layer["ff2"], ff_hidden)
        x = x + ff  # Residual connection from feed-forward.

    # Output head: layer-norm → bottleneck → final projection.
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
    """Build the dimension dataclass for the active chemistry/model contract.

    Parameters
    ----------
    config : dict
        Validated pipeline config containing architecture hyperparameters and
        spectrum settings.
    contract : dict
        Processed-data contract defining sequence, global, target, and
        spectrum dimensions.

    Returns
    -------
    TransformerDimensions or MLPDimensions
        Dimension dataclass matching the selected model family and chemistry
        mode.
    """
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
    if uses_mlp(config):
        if uses_fastchem(config):
            spectrum_dim = 0
            spectrum_latent_dim = 0
            spectrum_hidden_dim = 0
            spectrum_encoder_mode = "none"
        else:
            spectrum_cfg = config["stellar_spectrum"]
            spectrum_dim = int(contract.get("spectrum_dim", 0))
            spectrum_latent_dim = int(spectrum_cfg["latent_dim"])
            spectrum_hidden_dim = int(spectrum_cfg["hidden_dim"])
            spectrum_encoder_mode = str(spectrum_cfg["encoder_mode"]).lower()
        return MLPDimensions(
            sequence_dim=sequence_dim,
            global_dim=global_dim,
            spectrum_dim=spectrum_dim,
            spectrum_latent_dim=spectrum_latent_dim,
            spectrum_hidden_dim=spectrum_hidden_dim,
            spectrum_encoder_mode=spectrum_encoder_mode,
            target_dim=target_dim,
            d_hidden=int(model_cfg["d_hidden"]),
            num_hidden_layers=int(model_cfg["num_hidden_layers"]),
            conditioning_hidden_dim=int(model_cfg["conditioning_hidden_dim"]),
            film_clamp=float(model_cfg["film_clamp"]),
            activation=str(model_cfg["activation"]).lower(),
            dropout_rate=float(model_cfg.get("dropout_rate", 0.05)),
        )
    if uses_fastchem(config):
        spectrum_dim = 0
        spectrum_latent_dim = 0
        spectrum_hidden_dim = 0
        spectrum_encoder_mode = "none"
    else:
        spectrum_cfg = config["stellar_spectrum"]
        spectrum_dim = int(contract.get("spectrum_dim", 0))
        spectrum_latent_dim = int(spectrum_cfg["latent_dim"])
        spectrum_hidden_dim = int(spectrum_cfg["hidden_dim"])
        spectrum_encoder_mode = str(spectrum_cfg["encoder_mode"]).lower()
    return TransformerDimensions(
        sequence_dim=sequence_dim,
        global_dim=global_dim,
        spectrum_dim=spectrum_dim,
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
        spectrum_encoder_mode=spectrum_encoder_mode,
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
