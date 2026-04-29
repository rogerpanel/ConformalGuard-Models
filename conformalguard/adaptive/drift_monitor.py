"""Distribution-drift monitor.

Tracks rolling statistics on nonconformity scores and triggers a
"recalibration warning" when the empirical CDF on a sliding window deviates
from the calibration distribution under a Kolmogorov–Smirnov test.

This is the same pattern used by RobustIDPS for the SurrogateIDS continual-
learning module — we share the API with `integrated_ai_ids/core/drift.py`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class DriftSignal:
    ks_statistic: float
    p_value: float
    triggered: bool
    window_size: int


class DriftMonitor:
    def __init__(self, calibration_scores: np.ndarray,
                 window: int = 512, alpha_drift: float = 0.01):
        self.cal_scores = np.sort(np.asarray(calibration_scores, dtype=np.float64))
        self.window = window
        self.alpha_drift = alpha_drift
        self._buf: deque[float] = deque(maxlen=window)

    def observe(self, score: float) -> DriftSignal:
        self._buf.append(float(score))
        if len(self._buf) < max(32, self.window // 4):
            return DriftSignal(0.0, 1.0, False, len(self._buf))

        win = np.sort(np.fromiter(self._buf, dtype=np.float64))
        # Two-sample KS — manual implementation to avoid scipy import here.
        all_v = np.concatenate([self.cal_scores, win])
        cdf_cal = np.searchsorted(self.cal_scores, all_v, side="right") / self.cal_scores.size
        cdf_win = np.searchsorted(win, all_v, side="right") / win.size
        d = float(np.max(np.abs(cdf_cal - cdf_win)))

        n1, n2 = self.cal_scores.size, win.size
        en = np.sqrt(n1 * n2 / (n1 + n2))
        # Asymptotic KS p-value approximation.
        lam = (en + 0.12 + 0.11 / en) * d
        p_val = 2.0 * np.exp(-2.0 * lam ** 2)
        p_val = float(min(1.0, max(0.0, p_val)))
        triggered = p_val < self.alpha_drift
        return DriftSignal(d, p_val, triggered, len(self._buf))
