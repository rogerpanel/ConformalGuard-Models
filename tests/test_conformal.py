"""Tests for split-conformal calibration and theorem bounds."""

from __future__ import annotations

import numpy as np
import torch

from conformalguard.conformal.calibration import ConformalCalibrator
from conformalguard.conformal.nonconformity import NegLogProbScore
from conformalguard.conformal.theorems import (
    evaluate_coverage,
    marginal_coverage_bound,
)


def _synthetic_logits(n: int, c: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, c, (n,), generator=g)
    base = torch.randn(n, c, generator=g) * 0.4
    base[torch.arange(n), y] += 2.5
    return base, y


def test_quantile_level_correctness():
    logits, y = _synthetic_logits(2000, 12)
    cal = ConformalCalibrator(alpha=0.05)
    res = cal.calibrate(logits, y)
    assert res.quantile_level == np.ceil((2000 + 1) * 0.95) / 2000


def test_marginal_coverage_in_envelope():
    logits, y = _synthetic_logits(2000, 12)
    cal = ConformalCalibrator(alpha=0.10)
    cal.calibrate(logits, y)
    test_logits, test_y = _synthetic_logits(2000, 12, seed=1)
    sets = cal.predict_set(test_logits)
    in_set = np.array([int(int(test_y[i]) in s) for i, s in enumerate(sets)])
    cov = evaluate_coverage(in_set, n_cal=2000, alpha=0.10, tolerance=0.02)
    assert cov.in_envelope, cov


def test_neg_log_prob_score_matches_loss():
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 1.0, 0.0]])
    y = torch.tensor([2, 0])
    s = NegLogProbScore()(logits, y)
    expected = -torch.log_softmax(logits, dim=-1).gather(-1, y.view(-1, 1)).squeeze(-1)
    assert torch.allclose(s, expected)


def test_marginal_bound_endpoints():
    lo, hi = marginal_coverage_bound(n_cal=99, alpha=0.05)
    assert lo == 0.95
    assert abs(hi - (0.95 + 1 / 100)) < 1e-9
