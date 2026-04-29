"""Split-conformal calibration.

Algorithm 1 (paper, Sec. 4.2):

    Inputs:
      D_cal = {(G_<t^(i), y_t^(i))}_{i=1..n}    calibration traces,
      α ∈ (0, 1)                                 target miscoverage,
      s(.,.)                                     nonconformity score.

    Steps:
      1. Compute s_i = s(h_θ(G_<t^(i)), y_t^(i)) for all i.
      2. Set τ = Quantile_{⌈(n+1)(1−α)⌉/n}(s_1, …, s_n).
      3. At inference, return C_α(G_<t) = { y ∈ V : s(h_θ(G_<t), y) ≤ τ }.

Theorem 1 (Sec. 4.3) gives the marginal coverage guarantee:

    P[ y_t ∈ C_α(G_<t) ] ≥ 1 − α.

Under exchangeability + a continuity assumption on s,
    P[ y_t ∈ C_α(G_<t) ] ≤ 1 − α + 1/(n + 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from conformalguard.conformal.nonconformity import NegLogProbScore, NonconformityScore


@dataclass
class CalibrationResult:
    threshold: float
    n_calibration: int
    alpha: float
    quantile_level: float
    score_name: str

    def to_dict(self) -> dict:
        return {
            "threshold":     self.threshold,
            "n_calibration": self.n_calibration,
            "alpha":         self.alpha,
            "quantile_level": self.quantile_level,
            "score_name":    self.score_name,
        }


class ConformalCalibrator:
    """Split-conformal calibrator.

    Parameters
    ----------
    alpha : float
        Target miscoverage (paper default 0.05 ⇒ 95% coverage).
    score : NonconformityScore
        Nonconformity score — defaults to NegLogProbScore as in the paper.
    """

    def __init__(self, alpha: float = 0.05, score: NonconformityScore | None = None):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.score = score or NegLogProbScore()
        self.threshold_: float | None = None
        self.n_: int | None = None

    # -- calibration -------------------------------------------------------

    def calibrate(self, logits: torch.Tensor, y: torch.Tensor) -> CalibrationResult:
        """Fit the threshold τ from a calibration set.

        logits : (n_cal, V) — raw classifier outputs on calibration traces.
        y      : (n_cal,)   — ground-truth safe action ids.
        """
        if logits.dim() != 2 or y.dim() != 1:
            raise ValueError("expected logits (n, V) and y (n,)")
        n = logits.size(0)
        if n != y.size(0):
            raise ValueError("n_cal mismatch between logits and y")

        scores = self.score(logits, y).detach().cpu().numpy()
        # Finite-sample quantile level (Vovk/Lei): ⌈(n+1)(1-α)⌉ / n.
        q_level = math.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        tau = float(np.quantile(scores, q_level, method="higher"))

        self.threshold_ = tau
        self.n_ = n
        return CalibrationResult(
            threshold=tau,
            n_calibration=n,
            alpha=self.alpha,
            quantile_level=q_level,
            score_name=type(self.score).__name__,
        )

    # -- inference --------------------------------------------------------

    def threshold(self) -> float:
        if self.threshold_ is None:
            raise RuntimeError("Calibrator has not been fitted. Call .calibrate(...).")
        return self.threshold_

    def predict_set(self, logits: torch.Tensor) -> list[list[int]]:
        """Return the prediction set — list of label-ids per row.

        Action `y` is included iff s(logits, y) ≤ τ.
        """
        tau = self.threshold()
        all_scores = self.score.all_candidates(logits)            # (B, V)
        in_set = (all_scores <= tau).cpu().numpy()
        return [list(np.flatnonzero(row)) for row in in_set]

    # -- I/O --------------------------------------------------------------

    def state_dict(self) -> dict:
        if self.threshold_ is None:
            raise RuntimeError("Cannot serialize unfit calibrator.")
        return {
            "alpha": self.alpha,
            "threshold": self.threshold_,
            "n": self.n_,
            "score_name": type(self.score).__name__,
        }

    def load_state_dict(self, state: dict) -> "ConformalCalibrator":
        self.alpha = state["alpha"]
        self.threshold_ = state["threshold"]
        self.n_ = state["n"]
        return self
