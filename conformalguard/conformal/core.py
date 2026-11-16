"""NumPy-only conformal primitives.

Everything the guarantees in the paper rest on lives in this file and has no
dependency on the encoder, so it can be audited and unit-tested in isolation.

Conventions
-----------
* A *score* is a nonconformity score ``s = NCS(G(t), a)``; larger = more
  anomalous.  The guard allows a proposed action iff ``s <= q``.
* ``conformal_quantile(scores, alpha)`` is the split-conformal threshold: the
  ``ceil((n + 1)(1 - alpha))``-th smallest calibration score, ``+inf`` when
  that rank exceeds ``n``.
* Calibration units are *traces*.  ``one_step_per_trace`` draws a single step
  uniformly at random from each trace, which makes the calibration scores
  i.i.d. whenever the traces are, so Proposition 1 holds exactly and no
  within-trace dependence correction is needed.  ``trace_max_scores`` gives
  the trace-level variant (probability that a workflow is *ever* interrupted).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Order-statistic thresholds
# ---------------------------------------------------------------------------

def conformal_rank(n: int, alpha: float) -> int:
    """Rank ``k = ceil((n + 1)(1 - alpha))`` of the split-conformal threshold."""
    # The small epsilon guards against (n + 1)(1 - alpha) landing a hair above
    # an integer through floating-point error, e.g. 2001 * 0.95.
    return int(math.ceil((n + 1) * (1.0 - alpha) - 1e-12))


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Split-conformal threshold.

    Returns the ``k``-th smallest score with ``k = ceil((n+1)(1-alpha))``;
    ``+inf`` if ``k > n`` (the set is the whole action space) and ``-inf`` if
    ``k <= 0`` (the set is empty).  Under exchangeability of the ``n``
    calibration scores and the test score,
    ``P(s_test <= q) >= 1 - alpha``, and ``<= 1 - alpha + 1/(n+1)`` when
    scores are almost surely distinct.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = s.size
    k = conformal_rank(n, alpha)
    if k > n:
        return math.inf
    if k <= 0:
        return -math.inf
    return float(s[k - 1])


def lower_conformal_quantile(scores: Sequence[float], beta: float) -> float:
    """Lower threshold for the violation class.

    Returns the ``k``-th smallest score with ``k = floor(beta (m + 1))``;
    ``-inf`` if ``k == 0``.  If the ``m`` calibration scores come from
    *violating* steps and the test violating step is exchangeable with them,
    ``P(s_test < lam) <= beta``: auto-allowing only when ``s < lam`` misses
    at most a ``beta`` fraction of in-distribution violations.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))
    m = s.size
    k = int(math.floor(beta * (m + 1) + 1e-12))
    if k <= 0:
        return -math.inf
    return float(s[min(k, m) - 1])


# ---------------------------------------------------------------------------
# Calibration-unit construction
# ---------------------------------------------------------------------------

def one_step_per_trace(
    trace_scores: Sequence[Sequence[float]],
    rng: np.random.Generator,
) -> np.ndarray:
    """One uniformly random step score per trace (traces with no step skipped).

    ``trace_scores[i]`` holds the scores of the eligible steps of trace ``i``
    (for Proposition 1: the steps whose proposed action is labelled safe).
    """
    out = []
    for steps in trace_scores:
        steps = np.asarray(steps, dtype=np.float64)
        if steps.size:
            out.append(steps[rng.integers(steps.size)])
    return np.asarray(out, dtype=np.float64)


def trace_max_scores(trace_scores: Sequence[Sequence[float]]) -> np.ndarray:
    """Per-trace maximum score (trace-level calibration unit)."""
    return np.asarray(
        [np.max(np.asarray(s, dtype=np.float64)) for s in trace_scores if len(s)],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Split conformal on the safe class
# ---------------------------------------------------------------------------

@dataclass
class SplitConformal:
    """Static split-conformal guard threshold (Proposition 1)."""

    alpha: float = 0.05
    threshold_: float | None = None
    n_: int = 0

    def fit(self, safe_scores: Sequence[float]) -> "SplitConformal":
        self.threshold_ = conformal_quantile(safe_scores, self.alpha)
        self.n_ = int(np.asarray(safe_scores).size)
        return self

    def allow(self, score: float) -> bool:
        if self.threshold_ is None:
            raise RuntimeError("call fit() first")
        return bool(score <= self.threshold_)

    def coverage_bounds(self) -> tuple[float, float]:
        """(lower, upper) of Proposition 1; upper needs a.s. distinct scores."""
        return 1.0 - self.alpha, 1.0 - self.alpha + 1.0 / (self.n_ + 1.0)


# ---------------------------------------------------------------------------
# Two-sided (label-conditional) guard: ALLOW / ESCALATE / BLOCK
# ---------------------------------------------------------------------------

ALLOW, ESCALATE, BLOCK = "allow", "escalate", "block"


@dataclass
class TwoSidedConformal:
    """Label-conditional conformal guard with three verdicts.

    ``q_safe`` is calibrated on safe steps at level ``alpha`` and ``lam_viol``
    on violating steps at level ``beta``:

        s >  q_safe              -> BLOCK      (P(safe step blocked)        <= alpha)
        s <  lam_viol, s<=q_safe -> ALLOW      (P(violating step allowed)   <= beta)
        otherwise                -> ESCALATE   (held for review; rate is empirical)

    The second bound holds only for violations exchangeable with the
    calibration violations (known attack families); it says nothing about an
    adversary who adapts to the scorer.
    """

    alpha: float = 0.05
    beta: float = 0.05
    q_safe_: float | None = None
    lam_viol_: float | None = None

    def fit(self, safe_scores: Sequence[float],
            violating_scores: Sequence[float]) -> "TwoSidedConformal":
        self.q_safe_ = conformal_quantile(safe_scores, self.alpha)
        self.lam_viol_ = lower_conformal_quantile(violating_scores, self.beta)
        return self

    def verdict(self, score: float) -> str:
        if self.q_safe_ is None or self.lam_viol_ is None:
            raise RuntimeError("call fit() first")
        if score > self.q_safe_:
            return BLOCK
        if score < self.lam_viol_:
            return ALLOW
        return ESCALATE
