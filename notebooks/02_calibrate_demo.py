"""Demonstrate Algorithm 1 (split-conformal calibration) end-to-end on
synthetic logits — useful for unit-level intuition.
"""

from __future__ import annotations

import torch

from conformalguard.conformal.calibration import ConformalCalibrator


def main():
    torch.manual_seed(0)
    n_cal, n_test, c = 2000, 1000, 12
    y_cal = torch.randint(0, c, (n_cal,))
    logits_cal = torch.randn(n_cal, c) * 0.5
    logits_cal[torch.arange(n_cal), y_cal] += 2.0

    cal = ConformalCalibrator(alpha=0.05)
    res = cal.calibrate(logits_cal, y_cal)
    print(f"calibrated tau={res.threshold:.3f}, q={res.quantile_level:.4f}")

    y_test = torch.randint(0, c, (n_test,))
    logits_test = torch.randn(n_test, c) * 0.5
    logits_test[torch.arange(n_test), y_test] += 2.0
    sets = cal.predict_set(logits_test)
    in_set = sum(int(int(y_test[i]) in s) for i, s in enumerate(sets))
    avg_w = sum(len(s) for s in sets) / n_test
    print(f"empirical coverage = {in_set / n_test:.3f}  avg width = {avg_w:.2f}")


if __name__ == "__main__":
    main()
