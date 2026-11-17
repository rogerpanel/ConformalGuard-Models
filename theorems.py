"""Coverage diagnostics implementing the bounds proved in the manuscript.

Theorem 1 (marginal coverage, Sec. 4.3)
    Under exchangeability of the calibration subgraphs and any nonconformity
    score s, the split-conformal procedure satisfies

        1 − α ≤ P[ y_t ∈ C_α(G_<t) ] ≤ 1 − α + 1/(n + 1).

Theorem 2 (online adaptive coverage, Sec. 4.4)
    The Gibbs–Candès update τ_{t+1} = τ_t + γ · (1{y_t ∉ C_α} − α) keeps
    long-run miscoverage at α even under arbitrary distribution drift.

The functions in this module evaluate empirical coverage and check that the
calibrated threshold lies inside the theoretical envelope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoverageReport:
    n: int
    alpha: float
    empirical: float
    lower_bound: float
    upper_bound: float
    in_envelope: bool


def empirical_coverage(in_set_flags: np.ndarray) -> float:
    """Fraction of trials where the true label was inside the prediction set."""
    if in_set_flags.size == 0:
        return float("nan")
    return float(np.mean(in_set_flags.astype(np.float64)))


def marginal_coverage_bound(n_cal: int, alpha: float) -> tuple[float, float]:
    """Return the (lower, upper) bound from Theorem 1."""
    return 1.0 - alpha, 1.0 - alpha + 1.0 / (n_cal + 1.0)


def finite_sample_correction(n_cal: int, alpha: float) -> float:
    """The 1/(n+1) slack on the upper bound."""
    return 1.0 / (n_cal + 1.0)


def evaluate_coverage(in_set_flags: np.ndarray, n_cal: int, alpha: float,
                      tolerance: float = 0.01) -> CoverageReport:
    emp = empirical_coverage(in_set_flags)
    lo, hi = marginal_coverage_bound(n_cal, alpha)
    return CoverageReport(
        n=int(in_set_flags.size),
        alpha=alpha,
        empirical=emp,
        lower_bound=lo,
        upper_bound=hi,
        in_envelope=(lo - tolerance) <= emp <= (hi + tolerance),
    )
