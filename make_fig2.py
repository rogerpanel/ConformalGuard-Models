"""Regenerate Figure 2 (coverage under shift) from logged sequences.

Every curve in the figure is computed here from two inputs:

  --cal-scores   .npy, the n_cal calibration scores (one safe step per trace)
  --stream       .csv with one row per labelled-safe step of the deployment
                 stream, in arrival order, with columns
                     score   NCS of the proposed action
                     delay   (optional) labelled steps until the label arrives

and written back out, step by step, to ``<out>_sequences.csv`` so the figure
can be audited: err_t, the running mean, the working level alpha_t, the
threshold q_t, a rolling-window error, and the Proposition 2 band.

The script *asserts* Proposition 2 at every horizon T before it draws
anything.  A cumulative-error curve that leaves the band cannot be the output
of Algorithm 1 and the script refuses to plot it.

``--demo`` runs the same code on a synthetic stream (Gaussian scores with a
mean shift) purely to exercise the pipeline.  Demo figures are watermarked
and must not be used in the paper.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from conformalguard.adaptive.gibbs_candes import aci_bound, run_stream


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    out = np.empty(x.size)
    for t in range(x.size):
        lo = max(0, t + 1 - w)
        out[t] = (c[t + 1] - c[lo]) / (t + 1 - lo)
    return out


def _demo(seed: int, shift_at: int, T: int):
    rng = np.random.default_rng(seed)
    cal = rng.normal(0.0, 1.0, 2000)
    stream = np.concatenate([rng.normal(0.0, 1.0, shift_at),
                             rng.normal(0.45, 1.25, T - shift_at)])
    delays = rng.integers(0, 51, T)          # labels arrive at trace completion
    return cal, stream, delays


def _load_stream(path: Path):
    scores, delays = [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            scores.append(float(row["score"]))
            delays.append(int(row.get("delay", 0) or 0))
    return np.asarray(scores), np.asarray(delays, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cal-scores", type=Path)
    ap.add_argument("--stream", type=Path)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--shift-at", type=int, default=15_000,
                    help="step at which the shift was injected (drawn as a marker)")
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("runs/fig2_coverage"))
    args = ap.parse_args()

    if args.demo:
        cal, stream, delays = _demo(args.seed, args.shift_at, 50_000)
    elif args.cal_scores and args.stream:
        cal = np.load(args.cal_scores)
        stream, delays = _load_stream(args.stream)
    else:
        ap.error("give --cal-scores and --stream, or --demo")

    D = int(delays.max()) if delays.size else 0
    T = stream.size
    steps = np.arange(1, T + 1)

    aci = run_stream(cal, stream, args.alpha, args.gamma, delays=delays)
    static = run_stream(cal, stream, args.alpha, args.gamma, adaptive=False)
    band = np.array([aci_bound(int(t), args.gamma, args.alpha, D) for t in steps])

    gap = np.abs(aci["cum_err"] - args.alpha)
    if not np.all(gap <= band + 1e-12):
        t_bad = int(np.argmax(gap - band)) + 1
        raise SystemExit(
            f"Proposition 2 violated at T={t_bad}: |cum_err - alpha|="
            f"{gap[t_bad - 1]:.5f} > {band[t_bad - 1]:.5f}. "
            "The update rule is not the one in Algorithm 1; not plotting.")

    roll_aci = _rolling_mean(aci["err"], args.window)
    roll_static = _rolling_mean(static["err"], args.window)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seq_path = args.out.with_name(args.out.name + "_sequences.csv")
    with seq_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "score", "delay", "err_static", "cum_err_static",
                    "alpha_t", "q_t", "err_aci", "cum_err_aci",
                    f"rolling{args.window}_aci", f"rolling{args.window}_static",
                    "prop2_band_halfwidth"])
        for i in range(T):
            w.writerow([i + 1, f"{stream[i]:.6g}", int(delays[i]),
                        int(static["err"][i]), f"{static['cum_err'][i]:.6f}",
                        f"{aci['alpha_t'][i]:.6f}", f"{aci['q_t'][i]:.6g}",
                        int(aci["err"][i]), f"{aci['cum_err'][i]:.6f}",
                        f"{roll_aci[i]:.6f}", f"{roll_static[i]:.6f}",
                        f"{band[i]:.6f}"])

    def at(arr, t):
        return float(arr[min(t, T) - 1])

    print(f"n_cal={cal.size}  T={T}  alpha={args.alpha}  gamma={args.gamma}  "
          f"max feedback delay D={D}")
    for t in sorted({args.shift_at, args.shift_at + 1000, 25_000, T}):
        if t <= T:
            print(f"  T={t:>6}: cum_err ACI={at(aci['cum_err'], t):.4f} "
                  f"(band +/-{at(band, t):.4f})  static={at(static['cum_err'], t):.4f}  "
                  f"rolling ACI={at(roll_aci, t):.4f}  alpha_t={at(aci['alpha_t'], t):+.4f}")
    print(f"  peak rolling-{args.window} ACI error after shift: "
          f"{roll_aci[min(args.shift_at, T - 1):].max():.4f}")
    print(f"wrote {seq_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; sequences written, figure skipped")
        return

    ink, red, blue, grey = "#1a1a1a", "#c0392b", "#2c5aa0", "#8a8a8a"
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.6, 4.4), sharex=True,
                                 gridspec_kw={"height_ratios": [1.15, 1]})
    k = steps / 1000.0
    a1.fill_between(k, args.alpha - band, args.alpha + band, color=red, alpha=0.15,
                    lw=0, label="Prop. 2 band")
    a1.plot(k, static["cum_err"], color=blue, lw=1.4, label="static $\\hat q$")
    a1.plot(k, aci["cum_err"], color=red, lw=1.4, label="adaptive (ACI)")
    a1.axhline(args.alpha, color=grey, lw=0.8, ls="--")
    a1.set_ylabel("cumulative error $\\overline{\\mathrm{err}}_T$")
    a1.set_ylim(0, max(0.15, float(static["cum_err"][T // 10:].max()) * 1.1))
    a1.legend(frameon=False, fontsize=7, loc="lower right", ncol=3)

    a2.plot(k, roll_static, color=blue, lw=1.0)
    a2.plot(k, roll_aci, color=red, lw=1.0)
    a2.axhline(args.alpha, color=grey, lw=0.8, ls="--")
    a2.set_ylabel(f"rolling error (last {args.window})")
    a2.set_xlabel("labelled safe step $T$ (thousands)")
    for ax in (a1, a2):
        if args.shift_at < T:
            ax.axvline(args.shift_at / 1000.0, color=grey, lw=0.8, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7, colors=ink)
        ax.yaxis.label.set_size(8)
    a2.xaxis.label.set_size(8)
    if args.demo:
        fig.text(0.5, 0.5, "SYNTHETIC DEMO\nnot a result", color=grey, alpha=0.35,
                 fontsize=22, ha="center", va="center", rotation=20)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = args.out.with_suffix("." + ext)
        fig.savefig(p, dpi=200)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
