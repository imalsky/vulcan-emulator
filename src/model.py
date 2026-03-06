"""Transformer + FiLM regression model for chemistry trajectories."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        """Precompute the sinusoidal table up to the configured max length."""
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal positional encoding.")
        if max_len <= 0:
            raise ValueError("max_len must be > 0")
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encodings to an input sequence batch."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(dtype=x.dtype, device=x.device)


class FiLMLayer(nn.Module):
    """Feature-wise linear modulation from global conditions."""

    def __init__(self, global_dim: int, d_model: int, clamp: float) -> None:
        """Project global features into per-channel FiLM scale and shift terms."""
        super().__init__()
        if clamp <= 0.0:
            raise ValueError("FiLM clamp must be > 0")
        self.proj = nn.Linear(global_dim, 2 * d_model)
        self.register_buffer(
            "clamp", torch.tensor(float(clamp), dtype=torch.float32), persistent=False
        )

        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, globals_tensor: Tensor) -> Tensor:
        """Condition one sequence representation on one batch of global inputs."""
        film = self.proj(globals_tensor)
        gamma, beta = torch.chunk(film, 2, dim=-1)
        clamp = self.clamp.to(dtype=x.dtype, device=x.device)
        gamma = torch.clamp(gamma, -clamp, clamp)
        beta = torch.clamp(beta, -clamp, clamp)
        return (1.0 + gamma.unsqueeze(1)) * x + beta.unsqueeze(1)


class VulcanTransformer(nn.Module):
    """Encoder-only transformer with FiLM conditioning."""

    def __init__(
        self,
        *,
        input_dim: int,
        global_dim: int,
        target_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        film_clamp: float,
        output_head_divisor: int,
        max_sequence_length: int,
    ) -> None:
        """Build the encoder stack and FiLM-conditioned regression head."""
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if output_head_divisor <= 0:
            raise ValueError("output_head_divisor must be > 0.")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be > 0.")

        self.input_proj = nn.Linear(input_dim, d_model)
        self.posenc = PositionalEncoding(d_model, max_len=max_sequence_length)
        self.initial_film = FiLMLayer(global_dim, d_model, clamp=film_clamp)

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
        self.block_films = nn.ModuleList(
            [FiLMLayer(global_dim, d_model, clamp=film_clamp) for _ in range(num_layers)]
        )

        hidden = max(d_model // output_head_divisor, target_dim)
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, target_dim),
        )

    def forward(
        self,
        sequence_inputs: Tensor,
        global_inputs: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict normalized target profiles for one batch of sequence inputs."""
        if padding_mask is not None:
            if padding_mask.dtype != torch.bool:
                raise ValueError("padding_mask must use bool dtype.")
            expected = (sequence_inputs.shape[0], sequence_inputs.shape[1])
            if tuple(padding_mask.shape) != expected:
                raise ValueError(
                    "padding_mask shape mismatch: expected "
                    f"{expected}, got {tuple(padding_mask.shape)}."
                )

        x = self.input_proj(sequence_inputs)
        x = self.posenc(x)
        x = self.initial_film(x, global_inputs)

        for layer, film in zip(self.encoder_layers, self.block_films, strict=True):
            x = layer(x, src_key_padding_mask=padding_mask)
            x = film(x, global_inputs)

        return self.head(x)
