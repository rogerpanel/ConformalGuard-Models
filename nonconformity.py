"""Nonconformity scores.

The paper uses the **negative log-probability of the safe action** as its
canonical nonconformity score:

    s(G_<t, y_t) = − log p_θ(y_t | h_θ(G_<t))

where h_θ is the DyGFormer-HGT embedding and p_θ a softmax classifier over the
candidate-action vocabulary V. We also expose a margin-based score for
ablations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class NonconformityScore(ABC):
    """Abstract score s(features, label) ∈ R, larger = more nonconforming."""

    @abstractmethod
    def __call__(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def all_candidates(self, logits: torch.Tensor) -> torch.Tensor:
        """Return s(.) for every candidate label — shape (B, V)."""


class NegLogProbScore(NonconformityScore):
    """Canonical paper score: s = − log softmax(logits)[y]."""

    def __call__(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_p = torch.log_softmax(logits, dim=-1)
        return -log_p.gather(-1, y.view(-1, 1)).squeeze(-1)

    def all_candidates(self, logits: torch.Tensor) -> torch.Tensor:
        return -torch.log_softmax(logits, dim=-1)


class MarginScore(NonconformityScore):
    """Hinge-style: s = max_{j≠y} logit_j − logit_y."""

    def __call__(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        true_logit = logits.gather(-1, y.view(-1, 1)).squeeze(-1)
        masked = logits.clone()
        masked.scatter_(-1, y.view(-1, 1), float("-inf"))
        other = masked.max(dim=-1).values
        return other - true_logit

    def all_candidates(self, logits: torch.Tensor) -> torch.Tensor:
        v = logits.size(-1)
        b = logits.size(0)
        scores = torch.zeros(b, v, device=logits.device)
        for j in range(v):
            mask = torch.full_like(logits, float("-inf"))
            mask[..., j] = logits[..., j]
            true_logit = logits[..., j]
            other = logits.clone()
            other[..., j] = float("-inf")
            scores[..., j] = other.max(dim=-1).values - true_logit
        return scores


class SafetyHead(nn.Module):
    """Linear classifier from h_θ(G_<t) to action-vocabulary logits."""

    def __init__(self, d_model: int = 256, vocab_size: int = 64):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)
