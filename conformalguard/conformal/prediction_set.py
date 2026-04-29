"""Prediction-set wrapper used by the deployment layer.

A `PredictionSet` is the user-facing object returned at inference time. It
contains the candidate safe-action ids, their scores, the calibration
threshold, and a verdict that the deployment server / RobustIDPS plugin can
act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import torch


class Verdict(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REVIEW = "review"   # routed to the SOC analyst review queue


@dataclass
class PredictionSet:
    candidates: list[int]
    scores: list[float]
    threshold: float
    proposed_action: int | None = None
    verdict: Verdict = Verdict.ALLOW
    width: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.width = len(self.candidates)

    @classmethod
    def from_logits(
        cls,
        logits: torch.Tensor,
        threshold: float,
        nonconformity_all_candidates: torch.Tensor,
        proposed_action: int | None = None,
    ) -> "PredictionSet":
        scores = nonconformity_all_candidates.detach().cpu().numpy()
        in_set = (scores <= threshold)
        cand = [int(i) for i in range(scores.shape[0]) if in_set[i]]
        sc = [float(scores[i]) for i in cand]

        if proposed_action is None:
            verdict = Verdict.ALLOW
        elif proposed_action in cand:
            verdict = Verdict.ALLOW
        else:
            # Fall-back to REVIEW vs BLOCK heuristic (Sec. 6.2).
            verdict = Verdict.BLOCK if min(scores) < threshold else Verdict.REVIEW

        return cls(
            candidates=cand,
            scores=sc,
            threshold=threshold,
            proposed_action=proposed_action,
            verdict=verdict,
        )

    def contains(self, action_id: int) -> bool:
        return action_id in self.candidates

    def to_dict(self) -> dict:
        return {
            "candidates": self.candidates,
            "scores":     self.scores,
            "threshold":  self.threshold,
            "proposed_action": self.proposed_action,
            "verdict":    self.verdict.value,
            "width":      self.width,
            "metadata":   self.metadata,
        }


def average_width(sets: Iterable[PredictionSet]) -> float:
    sets = list(sets)
    if not sets:
        return 0.0
    return sum(s.width for s in sets) / len(sets)
