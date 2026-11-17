"""Continuous-time temporal attention used in DyGFormer.

Time differences `Δt` between events are encoded as Bochner-style harmonic
features (Xu et al., 2020) and added to the key/query inner product. This is
the temporal-graph analogue of positional encodings.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class TimeEncoder(nn.Module):
    """Bochner harmonic time encoding: φ(Δt) = cos(ω · Δt + φ)."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "time-encoding dimension must be even"
        self.dim = dim
        # Log-spaced frequencies.
        freqs = torch.exp(-math.log(10_000.0) * torch.arange(0, dim, 2).float() / dim)
        self.register_buffer("freqs", freqs)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """dt: (...,) → (..., dim)."""
        x = dt.unsqueeze(-1) * self.freqs  # (..., dim/2)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class TemporalAttention(nn.Module):
    """Multi-head attention with additive time-delta bias."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1,
                 time_dim: int = 64):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.time_enc = TimeEncoder(time_dim)
        self.time_proj = nn.Linear(time_dim, n_heads, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                dt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        q: (B, Lq, d_model)   k,v: (B, Lk, d_model)   dt: (B, Lq, Lk)
        mask: (B, Lq, Lk) bool — True where attention should be SUPPRESSED.
        """
        b, lq, _ = q.shape
        lk = k.size(1)
        Q = self.q_proj(q).view(b, lq, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(k).view(b, lk, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(v).view(b, lk, self.n_heads, self.d_head).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, Lq, Lk)
        # Additive time bias broadcast across heads.
        time_bias = self.time_proj(self.time_enc(dt))                # (B, Lq, Lk, H)
        scores = scores + time_bias.permute(0, 3, 1, 2)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))

        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ V                                                # (B, H, Lq, d_head)
        out = out.transpose(1, 2).contiguous().view(b, lq, self.d_model)
        return self.o_proj(out)
