"""Controlled simulations for the two conformal guarantees.  SYNTHETIC SCORES ONLY.

These simulations do not use agent traces, the encoder or any benchmark.  They
draw nonconformity scores from stated distributions and run the exact
calibration and update rules of the paper on them, so they check the
*procedures* (Propositions 1 and 2), not the detector.

    python scripts/sim_checks.py                       # prints both tables
    python scripts/sim_checks.py --figure fig.pdf      # also draws the figure

(A) Adaptive conformal inference under an abrupt shift with delayed labels.
    Calibration: n = 2000 scores ~ N(0, 1).
    Stream: 50,000 labelled-safe steps; N(0, 1) for t <= 15,000, then
    N(0.45, 1.25^2).  The label of each step arrives after a delay drawn
    uniformly from {0, ..., 50} steps (or is immediate, D = 0).  alpha = 0.05,
    alpha_1 = alpha, step size gamma in {0.05, 0.005}.

(B) Calibration unit under within-trace dependence.
    A trace has K ~ Uniform{5..50} steps; step scores are b + 0.5 eps with a
    trace effect b ~ N((27.5 - K) / 13, 1), so short traces score higher.
    Target: a uniformly random step of a fresh trace is covered w.p. >= 1 - alpha.
    Compared: one random step per calibration trace (Proposition 1) against
    pooling every step of every calibration trace.
"""

from __future__ import annotations

import argparse

import numpy as np

from conformalguard.adaptive.gibbs_candes import aci_bound, run_stream
from conformalguard.conformal.core import conformal_quantile

SHIFT_AT, T_TOTAL, MAX_DELAY, N_CAL = 15_000, 50_000, 50, 2000


def shift_stream(seed: int):
    rng = np.random.default_rng(seed)
    cal = rng.normal(0.0, 1.0, N_CAL)
    stream = np.concatenate([rng.normal(0.0, 1.0, SHIFT_AT),
                             rng.normal(0.45, 1.25, T_TOTAL - SHIFT_AT)])
    delays = rng.integers(0, MAX_DELAY + 1, T_TOTAL)
    return cal, stream, delays


def _roll(x: np.ndarray, w: int = 1000) -> np.ndarray:
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    return (c[w:] - c[:-w]) / w                      # window ending at step t >= w


def sim_a(seeds: int, alpha: float = 0.05) -> None:
    print(f"(A) ACI under an abrupt shift at T = {SHIFT_AT:,}; {seeds} seeds, means.  "
          "'open' = share of steps with\n    an infinite threshold (the guard blocks nothing "
          "on those steps).  All values in % except the band (pp).")
    print("    gamma   delay D | open pre  open post | cum.err 16k  cum.err 50k | "
          "peak 1k-window | band 16k  band 50k | Prop.2 held")
    static_rows = []
    for gamma in (0.05, 0.005):
        for D in (0, MAX_DELAY):
            rows = []
            for seed in range(seeds):
                cal, stream, delays = shift_stream(seed)
                aci = run_stream(cal, stream, alpha, gamma, delays=delays if D else None)
                steps = np.arange(1, T_TOTAL + 1)
                band = np.array([aci_bound(int(t), gamma, alpha, D) for t in steps])
                rows.append([
                    100 * np.mean(np.isinf(aci["q_t"][:SHIFT_AT])),
                    100 * np.mean(np.isinf(aci["q_t"][SHIFT_AT:])),
                    100 * aci["cum_err"][15_999], 100 * aci["cum_err"][-1],
                    100 * _roll(aci["err"])[SHIFT_AT - 999:].max(),
                    float(np.all(np.abs(aci["cum_err"] - alpha) <= band + 1e-12)),
                ])
                if gamma == 0.05 and D == 0:
                    st = run_stream(cal, stream, alpha, gamma, adaptive=False)
                    static_rows.append([100 * st["cum_err"][15_999], 100 * st["cum_err"][-1],
                                        100 * st["err"][SHIFT_AT:].mean()])
            m = np.mean(rows, 0)
            print(f"    {gamma:<7} {D:<7} | {m[0]:8.1f}  {m[1]:9.1f} | {m[2]:11.2f}  {m[3]:11.2f} | "
                  f"{m[4]:14.2f} | {100 * aci_bound(16_000, gamma, alpha, D):8.2f}  "
                  f"{100 * aci_bound(50_000, gamma, alpha, D):8.2f} | "
                  f"{int(sum(r[-1] for r in rows))}/{seeds}")
    m = np.mean(static_rows, 0)
    print(f"    static threshold | {0.0:8.1f}  {0.0:9.1f} | {m[0]:11.2f}  {m[1]:11.2f} | "
          f"post-shift error rate {m[2]:.2f}")


def figure(out: str, seed: int = 0, alpha: float = 0.05) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cal, stream, delays = shift_stream(seed)
    st = run_stream(cal, stream, alpha, 0.05, adaptive=False)
    fast = run_stream(cal, stream, alpha, 0.05, delays=delays)
    slow = run_stream(cal, stream, alpha, 0.005, delays=delays)
    k = np.arange(1, T_TOTAL + 1) / 1000.0
    band = np.array([aci_bound(int(t), 0.05, alpha, MAX_DELAY) for t in range(1, T_TOTAL + 1)])
    red, orange, blue, grey = "#c0392b", "#d98c1f", "#2c5aa0", "#8a8a8a"
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.4, 4.2), sharex=True)
    a1.fill_between(k, 100 * (alpha - band), 100 * (alpha + band), color=red, alpha=0.15, lw=0)
    a1.plot(k, 100 * st["cum_err"], color=blue, lw=1.3, label="static")
    a1.plot(k, 100 * slow["cum_err"], color=orange, lw=1.3, label=r"ACI, $\gamma=0.005$")
    a1.plot(k, 100 * fast["cum_err"], color=red, lw=1.3, label=r"ACI, $\gamma=0.05$")
    a1.axhline(100 * alpha, color=grey, lw=0.7, ls="--")
    a1.set_ylim(0, 15)
    a1.set_ylabel("cumulative false-block rate (%)")
    a1.legend(frameon=False, fontsize=7, loc="upper left")
    kk = k[999:]
    a2.plot(kk, 100 * _roll(np.isinf(slow["q_t"])), color=orange, lw=1.0)
    a2.plot(kk, 100 * _roll(np.isinf(fast["q_t"])), color=red, lw=1.0)
    a2.set_ylim(0, 100)
    a2.set_ylabel("steps with guard open (%)\n(last 1,000 steps)")
    a2.set_xlabel("labelled safe step $T$ (thousands)")
    for ax in (a1, a2):
        ax.axvline(SHIFT_AT / 1000.0, color=grey, lw=0.8, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.yaxis.label.set_size(7.5)
    a2.xaxis.label.set_size(8)
    a1.text(0.99, 0.04, "controlled simulation, synthetic scores", transform=a1.transAxes,
            ha="right", va="bottom", fontsize=6, color=grey)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    print(f"wrote {out}")


def _traces(rng, n):
    K = rng.integers(5, 51, n)
    b = rng.normal((27.5 - K) / 13.0, 1.0)
    return K, b


def sim_b(reps: int, n: int = 2000, alpha: float = 0.05, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    cov_one = cov_pool = 0
    for _ in range(reps):
        K, b = _traces(rng, n)
        one = b + 0.5 * rng.normal(size=n)                       # one random step per trace
        pooled = np.repeat(b, K) + 0.5 * rng.normal(size=int(K.sum()))
        _, bt = _traces(rng, 1)
        test = bt[0] + 0.5 * rng.normal()                        # random step of a fresh trace
        cov_one += test <= conformal_quantile(one, alpha)
        cov_pool += test <= conformal_quantile(pooled, alpha)
    se = (alpha * (1 - alpha) / reps) ** 0.5
    print(f"\n(B) Calibration unit, target coverage {100 * (1 - alpha):.0f}%, n = {n} traces, "
          f"{reps} repetitions (s.e. {100 * se:.2f} pp)")
    print(f"    one random step per trace (Prop. 1)   {100 * cov_one / reps:6.2f}%")
    print(f"    all steps pooled                      {100 * cov_pool / reps:6.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--reps", type=int, default=20_000)
    ap.add_argument("--figure", help="also write the simulation figure to this path (.pdf/.png)")
    a = ap.parse_args()
    sim_a(a.seeds)
    sim_b(a.reps)
    if a.figure:
        figure(a.figure)
