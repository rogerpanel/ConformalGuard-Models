"""Tests for the Gibbs-Candès controller and drift monitor."""

from __future__ import annotations

import numpy as np

from conformalguard.adaptive.drift_monitor import DriftMonitor
from conformalguard.adaptive.gibbs_candes import GibbsCandesController


def test_long_run_miscoverage_converges():
    rng = np.random.default_rng(42)
    ctl = GibbsCandesController(target_alpha=0.10, gamma=0.05)
    # True miscoverage = 0.10 — controller should track it.
    for _ in range(20_000):
        err = int(rng.random() < 0.10)
        ctl.update(err)
    assert abs(ctl.empirical_miscoverage() - 0.10) < 0.01


def test_alpha_clipped():
    ctl = GibbsCandesController(target_alpha=0.05, gamma=0.5,
                                alpha_min=0.01, alpha_max=0.5)
    for _ in range(100):
        ctl.update(1)
    assert ctl.alpha <= 0.5 + 1e-9


def test_drift_monitor_triggers_on_shift():
    cal_scores = np.random.RandomState(0).normal(0, 1, 5000)
    dm = DriftMonitor(cal_scores, window=512, alpha_drift=0.01)
    # Inject a clear shift: mean +3.
    out = None
    for _ in range(700):
        out = dm.observe(np.random.RandomState(1).normal(3, 1, 1)[0])
    assert out.triggered, out
