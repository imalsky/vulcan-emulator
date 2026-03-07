"""Encoder-only transformer for irregular-time VULCAN state transitions.

Architecture: Transformer Encoder + FiLM (Feature-wise Linear Modulation).

Data flow::

    Sequence inputs [batch, nz, 3+state_dim]   Global inputs [batch, 4]
            |                                    (g, [M/H], C/O, log10_dt)
      Profile projection (P,T,Kzz -> d_model)         |
      + State projection (ymix -> d_model)       MLP -> d_model
            |                                          |
      LayerNorm + Sinusoidal PE                        |
            |                                          |
      Initial FiLM conditioning  <---------------------+
            |                                          |
      N x [ TransformerEncoderLayer                    |
            + per-block FiLM ]   <---------------------+
            |
      Output head (d_model -> hidden -> target_dim)
            |
      + anchor_subset (residual skip)
            |
      Predictions [batch, nz, target_dim]

The residual skip connection means the model predicts the *change* (delta)
from the initial anchor state in normalized space, making short-dt predictions
nearly free (delta ~ 0) and concentrating model capacity on learning the
chemistry evolution.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding over the vertical atmospheric grid.

    Uses the Vaswani et al. 2017 formulation with sin/cos pairs.  The encoding
    table is precomputed once and stored as a non-persistent buffer so that
    positional information is injected without any learnable parameters.

    Args:
        d_model: Model embedding dimension (must be positive and even).
        max_len: Maximum supported sequence length (number of pressure levels).
    """

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        if d_model <= 0 or d_model % 2 != 0:
            raise ValueError("d_model must be a positive even integer.")
        if max_len <= 0:
            raise ValueError("max_len must be > 0.")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        table = torch.zeros(max_len, d_model, dtype=torch.float32)
        table[:, 0::2] = torch.sin(position * div_term)
        table[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("table", table.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = int(x.shape[1])
        return x + self.table[:, :seq_len].to(device=x.device, dtype=x.dtype)


class FiLMLayer(nn.Module):
    """Identity-biased feature-wise linear modulation from conditioning vectors.

    Applies ``(1 + gamma) * x + beta`` where gamma and beta are projected from
    a conditioning vector.  Initialized to identity (gamma=0, beta=0) so the
    layer is a no-op at initialization.  Tanh clamping on gamma/beta prevents
    early-training instabilities from large modulation values.

    Args:
        condition_dim: Dimensionality of the conditioning input.
        d_model: Dimensionality of the sequence features to modulate.
        clamp: Maximum absolute value for gamma and beta (applied via tanh).
    """

    def __init__(self, condition_dim: int, d_model: int, clamp: float) -> None:
        super().__init__()
        if clamp <= 0.0:
            raise ValueError("FiLM clamp must be > 0.")
        self.proj = nn.Linear(condition_dim, 2 * d_model)
        self.register_buffer("clamp", torch.tensor(float(clamp), dtype=torch.float32), persistent=False)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        film = self.proj(condition)
        gamma, beta = torch.chunk(film, 2, dim=-1)
        clamp = self.clamp.to(device=x.device, dtype=x.dtype)
        gamma = clamp * torch.tanh(gamma / clamp)
        beta = clamp * torch.tanh(beta / clamp)
        return (1.0 + gamma.unsqueeze(1)) * x + beta.unsqueeze(1)


class ConditioningProjector(nn.Module):
    """Project global conditioning scalars into one dense conditioning vector.

    All four global inputs (gravity, metallicity, C/O, log10_dt) are processed
    through a single two-layer MLP, then refined through a residual MLP with
    LayerNorm.  The output is a ``[batch, d_model]`` conditioning vector
    consumed by FiLM layers throughout the transformer stack.

    The time dimension (log10_dt) is already pre-transformed to log10 space
    by the normalization pipeline, so no additional Fourier expansion is needed;
    the MLP can learn to respond to the normalized scalar directly.

    Args:
        d_model: Output conditioning dimension.
        hidden_dim: Hidden size for the projection MLP.
        num_globals: Number of global conditioning inputs (default 4).
    """

    def __init__(self, *, d_model: int, hidden_dim: int, num_globals: int = 4) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0.")
        self.num_globals = num_globals
        self.global_mlp = nn.Sequential(
            nn.Linear(num_globals, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.xavier_uniform_(self.global_mlp[0].weight, gain=0.5)
        nn.init.zeros_(self.global_mlp[0].bias)
        nn.init.xavier_uniform_(self.global_mlp[2].weight, gain=0.5)
        nn.init.zeros_(self.global_mlp[2].bias)
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)
        nn.init.zeros_(self.out[3].weight)
        nn.init.zeros_(self.out[3].bias)

    def forward(self, conditioning_inputs: Tensor) -> Tensor:
        if conditioning_inputs.ndim != 2 or conditioning_inputs.shape[1] != self.num_globals:
            raise ValueError(
                f"conditioning_inputs must have shape [batch, {self.num_globals}]."
            )
        combined = self.global_mlp(conditioning_inputs)
        return combined + self.out(combined)


class VulcanTransitionTransformer(nn.Module):
    """Encoder-only transformer that predicts the future chemistry state from anchor state + dt.

    Given an atmospheric profile (pressure, temperature, Kzz) and an initial
    chemistry state (anchor ymix), predicts the mixing-ratio profiles at a
    future time ``dt`` conditioned through FiLM.

    Key design decisions:

    - **Separate profile/state projections**: The static atmospheric structure
      (P, T, Kzz) and the mutable chemistry state (ymix) are projected
      independently then summed, allowing each to learn distinct features.
    - **Residual delta prediction**: The output head predicts a correction
      (delta) added to the anchor subset, so if chemistry is unchanged the
      model outputs identity with delta=0.
    - **Bidirectional attention**: No causal mask because all pressure levels
      are physically coupled through vertical transport.
    - **Pre-norm architecture**: ``norm_first=True`` for more stable training
      gradients, especially with FiLM conditioning.

    Args:
        state_dim: Number of state species channels in the input.
        output_dim: Number of output species channels to predict.
        output_from_state_indices: Maps each output channel to a state index
            (used for the residual skip connection).
        d_model: Transformer embedding dimension.
        nhead: Number of self-attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Hidden dimension in the feedforward sub-layers.
        dropout: Dropout probability (0.0 = no dropout).
        film_clamp: Tanh clamping bound for FiLM gamma/beta values.
        output_head_divisor: ``d_model // divisor`` sets the output MLP hidden size.
        max_sequence_length: Maximum number of pressure levels for positional encoding.
        conditioning_hidden_dim: Hidden dimension for the conditioning projector.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        output_dim: int,
        output_from_state_indices: Sequence[int],
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        film_clamp: float,
        output_head_divisor: int,
        max_sequence_length: int,
        conditioning_hidden_dim: int,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or output_dim <= 0:
            raise ValueError("state_dim and output_dim must be > 0.")
        if len(output_from_state_indices) != output_dim:
            raise ValueError("output_from_state_indices length must equal output_dim.")
        if any(index < 0 or index >= state_dim for index in output_from_state_indices):
            raise ValueError("output_from_state_indices contains an out-of-range state index.")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if output_head_divisor <= 0:
            raise ValueError("output_head_divisor must be > 0.")

        self.static_input_dim = 3
        self.state_dim = int(state_dim)
        self.output_dim = int(output_dim)
        self.register_buffer(
            "output_from_state_indices",
            torch.tensor(list(output_from_state_indices), dtype=torch.long),
            persistent=False,
        )

        self.profile_proj = nn.Linear(self.static_input_dim, d_model)
        self.state_proj = nn.Linear(self.state_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.posenc = PositionalEncoding(d_model=d_model, max_len=max_sequence_length)
        self.conditioner = ConditioningProjector(
            d_model=d_model,
            hidden_dim=conditioning_hidden_dim,
        )
        self.initial_film = FiLMLayer(d_model, d_model, clamp=film_clamp)
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.block_films = nn.ModuleList([FiLMLayer(d_model, d_model, clamp=film_clamp) for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(d_model)

        hidden = max(d_model // output_head_divisor, output_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )

        nn.init.xavier_uniform_(self.profile_proj.weight, gain=0.5)
        nn.init.zeros_(self.profile_proj.bias)
        nn.init.xavier_uniform_(self.state_proj.weight, gain=0.5)
        nn.init.zeros_(self.state_proj.bias)
        nn.init.xavier_uniform_(self.delta_head[0].weight, gain=0.5)
        nn.init.zeros_(self.delta_head[0].bias)
        nn.init.zeros_(self.delta_head[2].weight)
        nn.init.zeros_(self.delta_head[2].bias)

    def forward(
        self,
        sequence_inputs: Tensor,
        conditioning_inputs: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        if sequence_inputs.ndim != 3:
            raise ValueError("sequence_inputs must have shape [batch, nz, input_dim].")
        expected_input_dim = self.static_input_dim + self.state_dim
        if int(sequence_inputs.shape[-1]) != expected_input_dim:
            raise ValueError(
                f"Expected sequence_inputs last dimension {expected_input_dim}, got {int(sequence_inputs.shape[-1])}."
            )
        if padding_mask is not None:
            if padding_mask.dtype != torch.bool:
                raise ValueError("padding_mask must use bool dtype.")
            expected_shape = (sequence_inputs.shape[0], sequence_inputs.shape[1])
            if tuple(padding_mask.shape) != expected_shape:
                raise ValueError(
                    f"padding_mask shape mismatch: expected {expected_shape}, got {tuple(padding_mask.shape)}."
                )

        # Split sequence inputs into static atmospheric profiles and mutable chemistry state.
        static_inputs = sequence_inputs[..., : self.static_input_dim]   # [batch, nz, 3] = P, T, Kzz
        state_inputs = sequence_inputs[..., self.static_input_dim :]    # [batch, nz, state_dim] = ymix

        # Extract the output-species subset of the anchor state for the residual skip.
        anchor_subset = torch.index_select(state_inputs, dim=-1, index=self.output_from_state_indices)

        # Independent projections for profiles and state, combined additively.
        x = self.profile_proj(static_inputs) + self.state_proj(state_inputs)
        x = self.input_norm(x)
        x = self.posenc(x)

        # Build the global conditioning vector from [gravity, metallicity, C/O, log10_dt].
        condition = self.conditioner(conditioning_inputs)

        # Apply initial FiLM modulation before the encoder stack.
        x = self.initial_film(x, condition)

        # N transformer encoder layers, each followed by per-block FiLM conditioning.
        for layer, film in zip(self.encoder_layers, self.block_films, strict=True):
            x = layer(x, src_key_padding_mask=padding_mask)
            x = film(x, condition)

        x = self.output_norm(x)

        # Output head predicts a delta correction; residual skip adds the anchor.
        delta = self.delta_head(x)
        return anchor_subset + delta
