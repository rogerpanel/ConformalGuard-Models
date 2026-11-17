"""Step-level false-block guarantee (Proposition 1) on R-Judge's injection records.

    git clone https://github.com/Lordog/R-Judge
    git clone https://github.com/uiuc-kang-lab/InjecAgent
    python scripts/rjudge_steps.py --rjudge-dir R-Judge --injecagent-dir InjecAgent --features st --out runs/rjudge_steps

Step labels are derived by code from InjecAgent's attacker-tool list (see
conformalguard/rjudge/steps.py).  Scorers are trained with a per-step loss.  Reported per
split: the false-block rate of a random safe step (guaranteed <= alpha with one calibration
step per trace), the same with all calibration steps pooled, and how often the violating
step is blocked.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformalguard.rjudge import experiment as E          # noqa: E402
from conformalguard.rjudge import steps as S               # noqa: E402
from conformalguard.rjudge.data import load_rjudge, rjudge_commit  # noqa: E402
from conformalguard.rjudge.features import node_features  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rjudge-dir", required=True)
    ap.add_argument("--injecagent-dir", required=True)
    ap.add_argument("--features", choices=("lsa", "st"), default="lsa")
    ap.add_argument("--st-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--models", default="step,seq,graph")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/rjudge_steps")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.epochs = 3, 6

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    all_traces = load_rjudge(args.rjudge_dir)
    all_feats = node_features(all_traces, args.features, args.st_model, cache_dir=out / "cache")
    attackers = S.attacker_tools(args.injecagent_dir)
    keep, targets = S.labelled_subset(all_traces, attackers)
    traces = [all_traces[i] for i in keep]; feats = [all_feats[i] for i in keep]
    n_inj = sum(t.attack_type == "injection" for t in all_traces)
    n_viol = sum(int((y == S.VIOL).any()) for y in targets)
    n_safe_steps = int(sum((y == S.SAFE).sum() for y in targets))
    data = {"injection_records": n_inj, "labelled": len(traces), "left_out": n_inj - len(traces),
            "with_violating_step": n_viol, "safe_steps": n_safe_steps, "attacker_tools": len(attackers)}
    print(json.dumps(data))
    names = [m for m in args.models.split(",") if m]
    rows, t0 = [], time.time()
    for r in range(args.reps):
        rng = np.random.default_rng([args.seed, zlib.crc32(b"steps"), r])
        parts = E.split_random(traces, rng)
        rows.append(S.run_split(traces, feats, targets, parts, args.seed * 100_003 + r, args.device, names,
                                dict(epochs=args.epochs)))
        g = rows[-1][names[-1]]
        print(f"split {r + 1}/{args.reps}  {names[-1]}: fbr@.10={g['fbr@0.10']:.3f} "
              f"block_at@.10={g['block_at@0.10']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    agg = E.aggregate(rows)
    f = lambda n, k: f"{100 * agg[n][k]['mean']:.1f} ± {100 * agg[n][k]['sd']:.1f}"
    md = [f"# Step-level guarantee on R-Judge injection records ({'QUICK SMOKE RUN, not a result' if args.quick else 'full run'})", "",
          f"- {data['labelled']} of {data['injection_records']} injection records labelled by code "
          f"({data['left_out']} left out: label and tool evidence disagree); {data['with_violating_step']} have a "
          f"violating step; {data['safe_steps']} safe steps",
          f"- features: `{args.features}`" + (f" (`{args.st_model}`)" if args.features == "st" else "")
          + f"; {len(rows)} random splits 50/25/25; per-step loss",
          "- FBR: a random safe step of a test trace is blocked. Block at: the violating step is blocked. "
          "Block by: some step up to and including it is blocked.", "",
          "### One calibration step per trace (Proposition 1: FBR <= alpha)", "",
          "| scorer | step AUROC | FBR@.05 | Block at@.05 | FBR@.10 | Block at@.10 | Block by@.10 | FBR@.20 | Block at@.20 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        md.append(f"| {n} | {agg[n]['auroc_step']['mean']:.3f} | {f(n, 'fbr@0.05')} | {f(n, 'block_at@0.05')} | "
                  f"{f(n, 'fbr@0.10')} | {f(n, 'block_at@0.10')} | {f(n, 'block_by@0.10')} | {f(n, 'fbr@0.20')} | {f(n, 'block_at@0.20')} |")
    md += ["", "### All calibration steps pooled (no exact guarantee), alpha = 0.10", "",
           "| scorer | FBR, random safe step | FBR, averaged over safe steps | Block at |", "|---|---|---|---|"]
    for n in names:
        md.append(f"| {n} | {f(n, 'fbr_pooled@0.10')} | {f(n, 'fbr_allsteps_pooled@0.10')} | {f(n, 'block_at_pooled@0.10')} |")
    results = {"per_split": rows, "aggregate": agg,
               "config": {**vars(args), "data": data, "rjudge_commit": rjudge_commit(args.rjudge_dir),
                          "injecagent_commit": rjudge_commit(args.injecagent_dir), "torch": torch.__version__,
                          "quick_smoke_run": bool(args.quick)}}
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md)); print(f"\nwrote {out / 'results.json'} and {out / 'summary.md'}")


if __name__ == "__main__":
    main()
