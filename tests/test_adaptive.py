"""Tests for adaptive conformal inference (Proposition 2) and the drift monitor.

The ACI bound is deterministic: it must hold on *every* sequence, including
adversarial ones, at *every* horizon T.  These tests check exactly that.
"""

from __future__ import annotations

import numpy as np
import pytest

from conformalguard.adaptive.drift_monitor import DriftMonitor
from conformalguard.adaptive.gibbs_candes import (
    AdaptiveConformal,
    aci_bound,
    run_stream,
)


def _assert_bound(cum_err, alpha, gamma, max_delay=0):
    T = np.arange(1, cum_err.size + 1)
    rhs = np.array([aci_bound(int(t), gamma, alpha, max_delay) for t in T])
    gap = np.abs(cum_err - alpha)
    worst = int(np.argmax(gap - rhs))
    assert np.all(gap <= rhs + 1e-12), (
        f"bound violated at T={worst + 1}: gap={gap[worst]:.5f} rhs={rhs[worst]:.5f}")


@pytest.mark.parametrize("gamma", [0.005, 0.05, 0.2])
def test_bound_holds_under_abrupt_shift(gamma):
    rng = np.random.default_rng(0)
    cal = rng.normal(0.0, 1.0, 2000)
    stream = np.concatenate([rng.normal(0.0, 1.0, 15_000),
                             rng.normal(1.5, 1.3, 35_000)])
    out = run_stream(cal, stream, alpha=0.05, gamma=gamma)
    _assert_bound(out["cum_err"], 0.05, gamma)
    # The static threshold has no such guarantee and visibly drifts.
    static = run_stream(cal, stream, alpha=0.05, gamma=gamma, adaptive=False)
    assert static["cum_err"][-1] > 0.15


def test_bound_holds_on_adversarial_sequence():
    """Scores chosen after seeing the threshold: err whenever it is possible."""
    rng = np.random.default_rng(1)
    cal = rng.normal(0.0, 1.0, 500)
    ctl = AdaptiveConformal(cal, alpha=0.1, gamma=0.05)
    errs = []
    for _ in range(20_000):
        q = ctl.threshold()
        score = q + 1.0 if np.isfinite(q) else 0.0   # exceed q unless q = +inf
        err = int(score > q)
        errs.append(err)
        ctl.update(err)
    cum = np.cumsum(errs) / np.arange(1, len(errs) + 1)
    _assert_bound(cum, 0.1, 0.05)
    assert abs(cum[-1] - 0.1) < 1e-2


@pytest.mark.parametrize("max_delay", [1, 20, 50])
def test_bound_holds_with_delayed_feedback(max_delay):
    rng = np.random.default_rng(2)
    cal = rng.normal(0.0, 1.0, 2000)
    stream = np.concatenate([rng.normal(0.0, 1.0, 5_000),
                             rng.normal(2.0, 1.0, 15_000)])
    delays = rng.integers(0, max_delay + 1, stream.size)
    out = run_stream(cal, stream, alpha=0.05, gamma=0.05, delays=delays)
    _assert_bound(out["cum_err"], 0.05, 0.05, max_delay=max_delay)


def test_alpha_stays_in_gibbs_candes_interval():
    rng = np.random.default_rng(3)
    cal = rng.normal(0.0, 1.0, 1000)
    out = run_stream(cal, rng.normal(3.0, 1.0, 5_000), alpha=0.05, gamma=0.05)
    assert out["alpha_t"].min() >= -0.05 - 1e-12
    assert out["alpha_t"].max() <= 1.05 + 1e-12


def test_paper_fig2_reading_is_outside_the_bound():
    """Regression for the reviewer's check: with alpha = alpha_1 = gamma = 0.05
    the cumulative error at T = 16,000 must lie in [4.875 %, 5.125 %]."""
    rhs = aci_bound(16_000, gamma=0.05, alpha_init=0.05)
    assert rhs == pytest.approx(20 / 16_000)
    assert abs(0.08 - 0.05) > rhs


def test_drift_monitor_triggers_on_shift():
    rng = np.random.default_rng(0)
    dm = DriftMonitor(rng.normal(0, 1, 5000), window=512, alpha_drift=0.01)
    out = None
    for x in rng.normal(3, 1, 700):
        out = dm.observe(float(x))
    assert out.triggered, out
