"""Streaming evaluation: Gibbs-Candès controller under shift regimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from conformalguard.adaptive.gibbs_candes import GibbsCandesController
from conformalguard.utils.config import load_config
from conformalguard.utils.logging import get_logger
from conformalguard.utils.seed import set_seed


def _simulate_episode(regime: str, n_steps: int, rng: np.random.Generator):
    """Synthetic stream of per-step nonconformity scores + ground truth set membership.

    Each regime shapes the score distribution differently so we can test the
    controller's ability to maintain target miscoverage.
    """
    if regime == "stationary":
        scores = rng.normal(loc=1.0, scale=0.4, size=n_steps)
    elif regime == "mild_drift":
        drift = np.linspace(0, 0.4, n_steps)
        scores = rng.normal(loc=1.0 + drift, scale=0.4)
    elif regime == "abrupt_shift":
        scores = rng.normal(loc=1.0, scale=0.4, size=n_steps)
        scores[n_steps // 2:] += 1.2
    elif regime == "recovery":
        drift = np.concatenate([np.linspace(0.0, 1.0, n_steps // 2),
                                np.linspace(1.0, 0.0, n_steps - n_steps // 2)])
        scores = rng.normal(loc=1.0 + drift, scale=0.4)
    else:
        raise ValueError(regime)
    return scores


def main():
    log = get_logger("conformalguard.online")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/adaptive.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(1234)

    target_alpha = cfg["adaptive"]["target_alpha"]
    gamma        = cfg["adaptive"]["gamma"]
    base_tau     = 1.20  # representative threshold from calibration

    rng = np.random.default_rng(7)
    out = []
    for regime in cfg["streaming"]["shift_schedule"]:
        ctl = GibbsCandesController(target_alpha=target_alpha, gamma=gamma)
        miscover = []
        for ep in range(cfg["streaming"]["n_episodes"]):
            scores = _simulate_episode(regime,
                                       cfg["streaming"]["steps_per_episode"], rng)
            for s in scores:
                tau_eff = base_tau + ctl.adjustment(base_tau)
                err = int(s > tau_eff)
                ctl.update(err)
                miscover.append(err)
        emp = float(np.mean(miscover))
        log.info(f"regime={regime:14s} emp_miscoverage={emp:.4f}  "
                 f"final_alpha={ctl.alpha:.4f}")
        out.append({"regime": regime,
                    "empirical_miscoverage": emp,
                    "final_alpha": ctl.alpha})

    Path("runs/conformalguard").mkdir(parents=True, exist_ok=True)
    Path("runs/conformalguard/online.json").write_text(json.dumps(out, indent=2))
    log.info("Wrote runs/conformalguard/online.json")


if __name__ == "__main__":
    main()
