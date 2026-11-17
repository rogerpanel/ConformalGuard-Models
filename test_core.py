"""Tests for the NumPy conformal primitives (Proposition 1 and its variants)."""

from __future__ import annotations

import math

import numpy as np

from conformalguard.conformal.core import (
    ALLOW,
    BLOCK,
    ESCALATE,
    SplitConformal,
    TwoSidedConformal,
    conformal_quantile,
    conformal_rank,
    lower_conformal_quantile,
    one_step_per_trace,
    trace_max_scores,
)


def test_rank_and_edge_cases():
    assert conformal_rank(2000, 0.05) == 1901          # ceil(2001 * 0.95)
    s = np.arange(1, 11, dtype=float)                   # 1..10
    assert conformal_quantile(s, 0.2) == 9.0            # ceil(11 * 0.8) = 9
    assert conformal_quantile(s, 0.05) == math.inf      # ceil(11 * .95) = 11 > 10
    assert conformal_quantile(s, 1.0) == -math.inf      # empty set
    assert lower_conformal_quantile(s, 0.05) == -math.inf
    assert lower_conformal_quantile(s, 0.2) == 2.0      # floor(11 * 0.2) = 2


def test_marginal_coverage_within_proposition_1_envelope():
    """Monte Carlo over the joint draw of calibration and test points."""
    rng = np.random.default_rng(0)
    n, alpha, reps = 200, 0.1, 20_000
    covered = 0
    for _ in range(reps):
        s = rng.exponential(size=n + 1)
        covered += s[-1] <= conformal_quantile(s[:-1], alpha)
    cov = covered / reps
    lo, hi = 1 - alpha, 1 - alpha + 1 / (n + 1)
    se = math.sqrt(0.1 * 0.9 / reps)
    assert lo - 4 * se <= cov <= hi + 4 * se, cov


def _traces(rng, n_traces):
    """Traces with strong within-trace dependence and random lengths."""
    out = []
    for _ in range(n_traces):
        k = int(rng.integers(5, 51))
        level = rng.normal(0.0, 1.0)                    # shared by all steps
        out.append(level + 0.3 * rng.normal(size=k))
    return out


def test_one_step_per_trace_is_exact_despite_dependence():
    rng = np.random.default_rng(1)
    alpha, reps, n = 0.1, 4000, 100
    covered = 0
    for _ in range(reps):
        cal = one_step_per_trace(_traces(rng, n), rng)
        test = one_step_per_trace(_traces(rng, 1), rng)[0]
        covered += test <= conformal_quantile(cal, alpha)
    cov = covered / reps
    se = math.sqrt(alpha * (1 - alpha) / reps)
    assert 1 - alpha - 4 * se <= cov <= 1 - alpha + 1 / (n + 1) + 4 * se, cov


def test_trace_level_threshold_bounds_any_block_in_a_trace():
    rng = np.random.default_rng(2)
    alpha, reps, n = 0.1, 4000, 100
    interrupted = 0
    for _ in range(reps):
        q = conformal_quantile(trace_max_scores(_traces(rng, n)), alpha)
        interrupted += bool(np.any(_traces(rng, 1)[0] > q))
    rate = interrupted / reps
    assert rate <= alpha + 4 * math.sqrt(alpha * (1 - alpha) / reps), rate


def test_two_sided_guard_controls_both_error_rates():
    rng = np.random.default_rng(3)
    alpha = beta = 0.1
    reps = 4000
    safe_blocked = viol_allowed = 0
    for _ in range(reps):
        g = TwoSidedConformal(alpha, beta).fit(rng.normal(0, 1, 300),
                                               rng.normal(2, 1, 60))
        safe_blocked += g.verdict(rng.normal(0, 1)) == BLOCK
        viol_allowed += g.verdict(rng.normal(2, 1)) == ALLOW
    tol = 4 * math.sqrt(0.1 * 0.9 / reps)
    assert safe_blocked / reps <= alpha + tol
    assert viol_allowed / reps <= beta + tol


def test_two_sided_verdict_regions():
    g = TwoSidedConformal(0.5, 0.5)
    g.q_safe_, g.lam_viol_ = 2.0, 1.0
    assert g.verdict(0.5) == ALLOW
    assert g.verdict(1.5) == ESCALATE
    assert g.verdict(2.5) == BLOCK


def test_split_conformal_wrapper():
    sc = SplitConformal(alpha=0.2).fit(np.arange(1, 11, dtype=float))
    assert sc.allow(9.0) and not sc.allow(9.5)
    assert sc.coverage_bounds() == (0.8, 0.8 + 1 / 11)
