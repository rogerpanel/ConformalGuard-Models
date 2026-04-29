"""Training losses.

The base loss is a class-balanced cross-entropy on the 12-way
{safe, 11 violation} prediction. Section 5.2 of the paper additionally adds a
*coverage-aware* term that penalizes scores above the calibration threshold
for the true label, encouraging tighter prediction sets without violating the
exchangeability assumptions of split-conformal calibration (which only sees
held-out data).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def safety_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    coverage_weight: float = 0.1,
    threshold_estimate: float | None = None,
) -> torch.Tensor:
    """Cross-entropy + soft coverage regularizer.

    logits             : (B, C)
    targets            : (B,)
    class_weights      : (C,) optional inverse-frequency weights.
    coverage_weight    : strength of the coverage regularizer.
    threshold_estimate : exponential-moving-average τ̂ used during training.
    """
    ce = F.cross_entropy(logits, targets, weight=class_weights, reduction="mean")
    if threshold_estimate is None or coverage_weight == 0.0:
        return ce
    log_p = torch.log_softmax(logits, dim=-1)
    s_true = -log_p.gather(-1, targets.view(-1, 1)).squeeze(-1)
    # Penalize the *positive* part of (s_true − τ̂): scores that would push the
    # true label out of the prediction set.
    cov_pen = F.softplus(s_true - threshold_estimate).mean()
    return ce + coverage_weight * cov_pen
