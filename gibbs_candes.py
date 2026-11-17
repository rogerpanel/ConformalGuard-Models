"""Adaptive conformal inference (Gibbs & Candes, 2021), exactly as in Algorithm 1.

The working level is updated on every labelled step,

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t),      err_t in {0, 1},

and the threshold is *re-read* from the calibration scores at the new level,

    q_t = Quantile({s_i}; 1 - alpha_t),

with ``q_t = +inf`` when ``alpha_t <= 0`` (nothing is blocked, so ``err_t = 0``)
and ``q_t = -inf`` when ``alpha_t >= 1`` (everything is blocked, ``err_t = 1``).
There is **no clipping or projection of alpha_t**: the self-correcting excursion
outside ``[0, 1]`` is what makes the bound of Proposition 2 hold, and clipping
breaks it.

Proposition 2 (deterministic, any data sequence).  With feedback for step ``s``
arriving at most ``D`` labelled steps late (``D = 0`` is the original ACI),

    | (1/T) sum_{t<=T} err_t - alpha |
        <=  (max(alpha_1, 1 - alpha_1) + gamma (D + 1)) / (gamma T)  +  D / T.

``AdaptiveConformal.bound`` returns the right-hand side so that every run can
be checked against it.

An earlier version of this file moved the threshold through a linear surrogate
and clipped alpha_t to ``[1e-3, 0.5]``.  That variant does not satisfy
Proposition 2 and has been removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from conformalguard.conformal.core import conformal_quantile


def aci_bound(T: int, gamma: float, alpha_init: float, max_delay: int = 0) -> float:
    """Right-hand side of Proposition 2 after ``T`` labelled steps."""
    if T <= 0:
        return float("inf")
    return ((max(alpha_init, 1.0 - alpha_init) + gamma * (max_delay + 1))
            / (gamma * T) + max_delay / T)


@dataclass
class AdaptiveConformal:
    """Stateful ACI controller over a fixed set of calibration scores."""

    cal_scores: Sequence[float]
    alpha: float = 0.05                 # target miscoverage
    gamma: float = 0.05                 # step size
    alpha_init: float | None = None     # alpha_1; defaults to the target
    alpha_t: float = field(init=False)
    n_updates: int = field(init=False, default=0)
    n_errors: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        self._sorted = np.sort(np.asarray(self.cal_scores, dtype=np.float64))
        if self.alpha_init is None:
            self.alpha_init = self.alpha
        self.alpha_t = float(self.alpha_init)

    # -- decision ----------------------------------------------------------

    def threshold(self) -> float:
        """q_t: the calibration quantile re-read at the current working level."""
        return conformal_quantile(self._sorted, self.alpha_t)

    def allow(self, score: float) -> bool:
        return bool(score <= self.threshold())

    # -- feedback ----------------------------------------------------------

    def update(self, err: int) -> float:
        """Apply one label: ``err = 1`` iff a safe proposed action was blocked."""
        if err not in (0, 1):
            raise ValueError("err must be 0 or 1")
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - err)
        self.n_updates += 1
        self.n_errors += int(err)
        return self.alpha_t

    # -- diagnostics -------------------------------------------------------

    def empirical_miscoverage(self) -> float:
        return self.n_errors / max(1, self.n_updates)

    def bound(self, T: int | None = None, max_delay: int = 0) -> float:
        return aci_bound(T if T is not None else self.n_updates,
                         self.gamma, float(self.alpha_init), max_delay)


def run_stream(
    cal_scores: Sequence[float],
    stream_scores: Sequence[float],
    alpha: float = 0.05,
    gamma: float = 0.05,
    delays: Sequence[int] | None = None,
    adaptive: bool = True,
) -> dict[str, np.ndarray]:
    """Replay a stream of safe-step scores through the static or adaptive guard.

    ``stream_scores[t]`` is the score of the proposed action at the ``t``-th
    labelled-safe step.  ``delays[t]`` (optional) is the number of later steps
    after which the label of step ``t`` becomes available; with ``delays=None``
    feedback is immediate.

    Returns per-step arrays: ``err``, ``alpha_t``, ``q_t`` and the running
    mean ``cum_err``.  With ``adaptive=False`` the threshold stays at the
    split-conformal quantile and ``alpha_t`` is constant.
    """
    scores = np.asarray(stream_scores, dtype=np.float64)
    T = scores.size
    ctl = AdaptiveConformal(cal_scores, alpha=alpha, gamma=gamma)
    arrivals: dict[int, list[int]] = {}
    err = np.zeros(T, dtype=np.int64)
    alpha_t = np.zeros(T, dtype=np.float64)
    q_t = np.zeros(T, dtype=np.float64)
    for t in range(T):
        alpha_t[t] = ctl.alpha_t
        q_t[t] = ctl.threshold()
        err[t] = int(scores[t] > q_t[t])
        if adaptive:
            d = 0 if delays is None else int(delays[t])
            arrivals.setdefault(t + d, []).append(int(err[t]))
            for e in arrivals.pop(t, []):
                ctl.update(e)
    cum_err = np.cumsum(err) / np.arange(1, T + 1)
    return {"err": err, "alpha_t": alpha_t, "q_t": q_t, "cum_err": cum_err}


# Backwards-compatible name used by earlier scripts.
GibbsCandesController = AdaptiveConformal
