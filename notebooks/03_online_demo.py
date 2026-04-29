"""Demo: Gibbs-Candès controller on a piece-wise stationary stream."""

from __future__ import annotations

import numpy as np

from conformalguard.adaptive.gibbs_candes import GibbsCandesController


def main():
    rng = np.random.default_rng(0)
    ctl = GibbsCandesController(target_alpha=0.05, gamma=0.05)
    base = 1.0
    err_log = []
    for t in range(20_000):
        # change-point at 10k.
        true_p = 0.05 if t < 10_000 else 0.20
        err = int(rng.random() < true_p)
        ctl.update(err)
        err_log.append(err)
    print("final alpha:", round(ctl.alpha, 3))
    print("empirical miscoverage:", round(np.mean(err_log), 3))


if __name__ == "__main__":
    main()
