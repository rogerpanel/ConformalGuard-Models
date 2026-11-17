"""Calibrate judges' scores on the multi-agent sessions and compare them at a matched false-block budget.

    python scripts/mas_guard_eval.py --sessions results/mas_kaggle_v2/claude_sessions.jsonl.gz results/mas_kaggle_v2/openai_sessions.jsonl.gz \
        --scores claude=runs/mas_judge_claude/scores.jsonl qwen25=runs/mas_judge_qwen25/scores.jsonl \
        --prices claude=4,20,0.2,5 --trained results/mas_kaggle_v2/results.json --out runs/mas_guard_eval

For each score file: the sessions it covers are split at random into calibration and test halves
(no training), the threshold is set by the session-level rule (Corollary 1) and by the step-level
rule (Proposition 1), and the report gives the false-block rate, the share of violating steps
blocked, the median latency per decision, tokens per decision and, with --prices
(input,output[,cache_read,cache_write] in USD per million tokens, yours to supply), the cost per
thousand decisions.  --trained adds the trained scorers' numbers from scripts/mas_run.py for
reference; those come from 50/25/25 splits of all sessions, so the split protocol differs.
A step without a score (a refused or unparseable reply) counts as blocked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformalguard.mas import convert as C   # noqa: E402
from conformalguard.mas import judge as J     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--scores", nargs="+", required=True, help="name=path")
    ap.add_argument("--prices", nargs="*", default=[], help="name=input,output,cache_read,cache_write (USD per million tokens)")
    ap.add_argument("--trained", default="", help="results.json of scripts/mas_run.py")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--out", default="runs/mas_guard_eval")
    args = ap.parse_args()
    by_seed = {s["seed"]: s for s in C.load_sessions(args.sessions)}
    prices = {k: ([float(x) for x in v.split(",")] + [0.0, 0.0])[:4] for k, v in (p.split("=", 1) for p in args.prices)}
    res: dict = {}
    for spec in args.scores:
        name, path = spec.split("=", 1)
        rows = J.read_scores(path)
        sessions, scores = [], []
        for seed in sorted({k[0] for k in rows}):
            sc = J.session_scores(by_seed[seed], rows) if seed in by_seed else None
            if sc is not None:
                sessions.append(by_seed[seed]); scores.append(sc)
        r = J.evaluate(sessions, scores, reps=args.reps)
        vals = list(rows.values())
        secs = np.array([v["seconds"] for v in vals if v.get("seconds") is not None])
        tok = lambda k: float(np.mean([v.get(k) or 0 for v in vals]))
        r.update(guard=vals[0].get("guard") if vals else name, decisions=len(vals),
                 seconds_median=float(np.median(secs)) if secs.size else None, seconds_p90=float(np.quantile(secs, 0.9)) if secs.size else None,
                 tokens_per_decision={k: tok(k) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")},
                 served_by_other_model=sum(v.get("served_by") not in (None, v.get("guard")) for v in vals),
                 refused=sum(v.get("stop_reason") == "refusal" for v in vals))
        if name in prices:
            pi, po, pr, pw = prices[name]; t = r["tokens_per_decision"]
            r["usd_per_1000_decisions"] = 1000 * (t["input_tokens"] * pi + t["output_tokens"] * po + t["cache_read_input_tokens"] * pr + t["cache_creation_input_tokens"] * pw) / 1e6
            all_in = t["input_tokens"] + t["cache_read_input_tokens"] + t["cache_creation_input_tokens"]
            r["usd_per_1000_decisions_uncached"] = 1000 * (all_in * pi + t["output_tokens"] * po) / 1e6
        res[name] = r
    f = lambda r, k: f"{100 * r[k]['mean']:.1f} ± {100 * r[k]['sd']:.1f}"
    md = ["# Judges as guards on the multi-agent sessions", "",
          f"{args.reps} random halves of the judged sessions (calibration / test), no training. The judge is given the policy and the "
          "monitor's metadata. FBR and Block in %, mean ± sd.", "",
          "### Session-level rule (Corollary 1): a violation-free session has a blocked step with probability <= alpha", "",
          "| guard | sessions | violating steps | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | Block@.10, one task | Block@.10, concurrent |", "|---|---|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        md.append(f"| {n} ({r['guard']}) | {r['n_sessions']} | {r['n_violating_steps']} | {f(r, 'fbr_session@0.05')} | {f(r, 'block_session@0.05')} | "
                  f"{f(r, 'fbr_session@0.10')} | {f(r, 'block_session@0.10')} | {f(r, 'block_session_single@0.10')} | {f(r, 'block_session_concurrent@0.10')} |")
    md += ["", "### Step-level rule (Proposition 1), alpha = 0.10, and operating figures", "",
           "| guard | FBR@.10 | Block@.10 | median latency per decision | 90th percentile | tokens per decision (in / out / cache read / cache write) | USD per 1,000 decisions as billed | at list price, no caching | unscored steps | served by another model |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        t = r["tokens_per_decision"]
        md.append(f"| {n} | {f(r, 'fbr@0.10')} | {f(r, 'block@0.10')} | "
                  + (f"{r['seconds_median']:.2f} s | {r['seconds_p90']:.2f} s" if r["seconds_median"] is not None else "n/a | n/a")
                  + f" | {t['input_tokens']:.0f} / {t['output_tokens']:.0f} / {t['cache_read_input_tokens']:.0f} / {t['cache_creation_input_tokens']:.0f} | "
                  + (f"{r['usd_per_1000_decisions']:.2f} | {r['usd_per_1000_decisions_uncached']:.2f}" if "usd_per_1000_decisions" in r else "n/a | n/a")
                  + f" | {r['unscored_steps']} | {r['served_by_other_model']} |")
    if args.trained:
        A = json.loads(Path(args.trained).read_text())["aggregate"]["random"]
        md += ["", "### Trained scorers, for reference (scripts/mas_run.py: 50/25/25 splits of all sessions, session-level rule)", "",
               "| scorer | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | Block@.10, one task | Block@.10, concurrent |", "|---|---|---|---|---|---|---|"]
        for n in ("step", "seq", "graph"):
            md.append(f"| {n} | " + " | ".join(f"{100 * A[n][k]['mean']:.1f} ± {100 * A[n][k]['sd']:.1f}" for k in
                      ("fbr_session@0.05", "block_session@0.05", "fbr_session@0.10", "block_session@0.10", "block_session_single@0.10", "block_session_concurrent@0.10")) + " |")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "guard_eval.json").write_text(json.dumps(res, indent=1))
    (out / "guard_eval.md").write_text("\n".join(md) + "\n")
    print("\n".join(md)); print(f"\nwrote {out / 'guard_eval.json'} and {out / 'guard_eval.md'}")


if __name__ == "__main__":
    main()
