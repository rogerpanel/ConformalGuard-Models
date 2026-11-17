"""Real-data experiment on R-Judge: conformal false-block control and scorer comparison.

    git clone https://github.com/Lordog/R-Judge
    python scripts/rjudge_run.py --rjudge-dir R-Judge --features st --reps 100 --out runs/rjudge

Writes <out>/results.json (every per-split number, the aggregate, the paired
comparisons and the full configuration) and <out>/summary.md (the tables).
See conformalguard/rjudge/experiment.py for the protocol.

--features lsa   TF-IDF + SVD features; CPU only, no model download
--features st    frozen sentence-transformers encoder (default all-MiniLM-L6-v2)
--quick          3 splits, 6 epochs: a smoke test, not a result
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformalguard.rjudge import experiment as E                      # noqa: E402
from conformalguard.rjudge.data import (load_published_judge, load_rjudge,  # noqa: E402
                                        rjudge_commit, summary)
from conformalguard.rjudge.features import node_features               # noqa: E402
from conformalguard.rjudge.models import build, n_params               # noqa: E402


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _fmt(agg, name, key, pct=True):
    m = agg.get(name, {}).get(key)
    if m is None:
        return "n/a"
    s = 100 if pct else 1
    return f"{s * m['mean']:.1f} ± {s * m['sd']:.1f}" if pct else f"{m['mean']:.3f} ± {m['sd']:.3f}"


def table(agg, names, title, n_splits) -> str:
    out = [f"### {title}", "",
           f"Mean ± sd over {n_splits} splits. FBR and Block in %. Target FBR = alpha.", "",
           "| scorer | AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |",
           "|---|---|---|---|---|---|---|---|"]
    for n in names:
        if n not in agg:
            continue
        out.append(f"| {n} | {_fmt(agg, n, 'auroc', False)} | " + " | ".join(
            f"{_fmt(agg, n, f'fbr@{a}')} | {_fmt(agg, n, f'block@{a}')}" for a in ("0.05", "0.10", "0.20")) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rjudge-dir", required=True)
    ap.add_argument("--features", choices=("lsa", "st"), default="lsa")
    ap.add_argument("--st-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--reps", type=int, default=50, help="random and grouped splits each")
    ap.add_argument("--loco-reps", type=int, default=5, help="splits per held-out category")
    ap.add_argument("--models", default="step,seq,graph,graph_untyped,logreg_pooled,rule_last_tool")
    ap.add_argument("--protocols", default="random,grouped,loco",
                    help="any of random,grouped,loco,recal")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/rjudge")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.loco_reps, args.epochs = 3, 1, 6

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    traces = load_rjudge(args.rjudge_dir)
    feats = node_features(traces, args.features, args.st_model, cache_dir=out / "cache")
    names = [m for m in args.models.split(",") if m]
    protocols = [p for p in args.protocols.split(",") if p]
    train_kw = dict(epochs=args.epochs, d=args.dim, layers=args.layers)
    data = summary(traces)
    print(json.dumps(data, indent=1))
    params = {m: n_params(build(m, feats[0].shape[1], d=args.dim, layers=args.layers))
              for m in names if m in E.NEURAL}
    print("parameters:", params, "| device:", args.device, "| features:", args.features, feats[0].shape[1])

    results: dict = {"per_split": {}, "aggregate": {}, "paired": {}}
    t_start = time.time()

    def run(protocol, key, make_parts, n):
        rows = []
        for r in range(n):
            rng = np.random.default_rng([args.seed, zlib.crc32(key.encode()), r])   # stable across runs
            parts = make_parts(rng)
            rows.append(E.run_split(traces, feats, parts, seed=args.seed * 100_003 + r,
                                    device=args.device, model_names=names, train_kw=train_kw))
            g = rows[-1].get("graph") or next(iter(rows[-1].values()))
            print(f"[{protocol}:{key}] split {r + 1}/{n}  "
                  f"auroc={g['auroc']:.3f} fbr@.10={g['fbr@0.10']:.3f} block@.10={g['block@0.10']:.3f}  "
                  f"({time.time() - t_start:.0f}s)", flush=True)
        results["per_split"][key] = rows
        results["aggregate"][key] = E.aggregate(rows)
        # Keep what is finished: an interrupted session then loses at most one protocol.
        (out / "results_partial.json").write_text(json.dumps(results, indent=1))
        return rows

    md = [f"# R-Judge experiment ({'QUICK SMOKE RUN, not a result' if args.quick else 'full run'})", "",
          f"- records: {data['records']} ({data['unsafe']} unsafe, {data['safe']} safe), "
          f"{data['prompt_groups']} distinct user prompts, {data['decision_steps']} decision steps",
          f"- features: `{args.features}`" + (f" (`{args.st_model}`)" if args.features == "st" else "")
          + f", dimension {feats[0].shape[1]}",
          f"- scorer parameters: {params}",
          "- record score = maximum step logit; threshold = conformal quantile of the safe "
          "calibration records; split 50/25/25", ""]

    if "random" in protocols:
        rows = run("random", "random", lambda rng: E.split_random(traces, rng), args.reps)
        md.append(table(results["aggregate"]["random"], names,
                        "Random splits (exchangeable: the FBR guarantee applies)", len(rows)))
        for a, b in (("graph", "seq"), ("graph", "step"), ("seq", "step"), ("graph", "graph_untyped"), ("graph", "logreg_pooled")):
            if a in names and b in names:
                for metric in ("auroc", "block@0.10"):
                    results["paired"][f"random:{a}-{b}:{metric}"] = E.paired(rows, a, b, metric)
    if "random" in protocols:
        agg = results["aggregate"]["random"]
        cats = sorted({t.category for t in traces})
        md += ["Category-conditional (Mondrian) calibration on the random splits, alpha = 0.10: FBR (%) per",
               "category with one pooled threshold, and with one threshold per category.", "",
               "| scorer | " + " | ".join(f"{c} pooled / per-cat" for c in cats) + " | all: FBR per-cat | all: Block per-cat |",
               "|---|" + "---|" * (len(cats) + 2)]
        for n in names:
            if n in agg and "fbr_mond@0.10" in agg[n]:
                cells = [f"{_fmt(agg, n, f'fbr_cat:{c}@0.10')} / {_fmt(agg, n, f'fbr_mond_cat:{c}@0.10')}" for c in cats]
                md.append(f"| {n} | " + " | ".join(cells) + f" | {_fmt(agg, n, 'fbr_mond@0.10')} | {_fmt(agg, n, 'block_mond@0.10')} |")
        md.append("")
    if "grouped" in protocols:
        rows = run("grouped", "grouped", lambda rng: E.split_grouped(traces, rng), args.reps)
        md.append(table(results["aggregate"]["grouped"], names,
                        "Prompt-grouped splits (no user prompt shared between train and test)", len(rows)))
        for a, b in (("graph", "seq"), ("graph", "step"), ("seq", "step"), ("graph", "graph_untyped"), ("graph", "logreg_pooled")):
            if a in names and b in names:
                for metric in ("auroc", "block@0.10"):
                    results["paired"][f"grouped:{a}-{b}:{metric}"] = E.paired(rows, a, b, metric)
        agg = results["aggregate"]["grouped"]
        md += ["Calibration unit on the grouped splits: FBR (%) at alpha = 0.10 with the threshold set",
               "from all safe calibration records (pooled) and from one random safe record per",
               "calibration prompt group, evaluated on one random record per test group.", "",
               "| scorer | FBR pooled | FBR one per group | Block one per group |", "|---|---|---|---|"]
        md += [f"| {n} | {_fmt(agg, n, 'fbr@0.10')} | {_fmt(agg, n, 'fbr_g1@0.10')} | {_fmt(agg, n, 'block_g1@0.10')} |"
               for n in names if n in agg]
        md.append("")
    if "loco" in protocols:
        for cat in sorted({t.category for t in traces}):
            rows = run("loco", f"loco:{cat}",
                       lambda rng, cat=cat: E.split_loco(traces, rng, cat), args.loco_reps)
            md.append(table(results["aggregate"][f"loco:{cat}"], names,
                            f"Leave-one-category-out, test on {cat} (distribution shift: no guarantee)",
                            len(rows)))

    if "recal" in protocols:
        md += ["### In-domain recalibration: train on four categories, calibrate and test on halves of the fifth", "",
               f"Mean ± sd over {args.loco_reps} splits per category, alpha = 0.10. Calibration and test records are",
               "exchangeable, so the guarantee applies. 'finite' is the share of splits in which the category had",
               "enough safe calibration records for a finite threshold (otherwise nothing is blocked).", "",
               "| held-out category | scorer | FBR@.10 | Block@.10 | finite | safe cal records |", "|---|---|---|---|---|---|"]
        for cat in sorted({t.category for t in traces}):
            run("recal", f"recal:{cat}", lambda rng, cat=cat: E.split_recal(traces, rng, cat), args.loco_reps)
            agg = results["aggregate"][f"recal:{cat}"]
            for n in names:
                if n in agg:
                    md.append(f"| {cat} | {n} | {_fmt(agg, n, 'fbr@0.10')} | {_fmt(agg, n, 'block@0.10')} | "
                              f"{_fmt(agg, n, 'q_finite@0.10')} | {agg[n]['n_cal_safe']['mean']:.0f} |")
        md.append("")

    if results["paired"]:
        md += ["### Paired comparisons (a minus b, per split; exact sign test)", "",
               "| protocol | a - b | metric | mean diff | a better / b better / ties | p |", "|---|---|---|---|---|---|"]
        for k, v in results["paired"].items():
            proto, pair, metric = k.split(":")
            scale = 100 if metric.startswith("block") else 1
            md.append(f"| {proto} | {pair} | {metric} | {scale * v['mean_diff']:+.2f}"
                      f"{' pp' if scale == 100 else ''} | {v['a_better']} / {v['b_better']} / {v['ties']} "
                      f"| {v['sign_test_p']:.3g} |")
        md.append("")

    # Third-party LLM-judge predictions shipped with R-Judge (not re-run here).
    judge = load_published_judge(args.rjudge_dir)
    if judge:
        y = np.array([t.label for t in traces]); p = np.array([judge.get(t.id, -1) for t in traces])
        atk = np.array([t.attack_type for t in traces])
        flag = p == 1                                  # unparseable outputs count as not flagged
        jr = {"model": "Meta-Llama-3.1-8B-Instruct (R-Judge authors' released predictions)",
              "n": int(y.size), "unparseable": int((p == -1).sum()),
              "block": float(flag[y == 1].mean()), "fbr": float(flag[y == 0].mean())}
        for at in ("injection", "unintended"):
            jr[f"block_{at}"] = float(flag[(y == 1) & (atk == at)].mean())
            jr[f"fbr_{at}"] = float(flag[(y == 0) & (atk == at)].mean())
        results["published_llm_judge"] = jr
        md += ["### LLM judge released with R-Judge (all records; fixed operating point, no calibration)", "",
               f"{jr['model']}: Block {100 * jr['block']:.1f}%, FBR {100 * jr['fbr']:.1f}% "
               f"(injection: {100 * jr['block_injection']:.1f} / {100 * jr['fbr_injection']:.1f}; "
               f"unintended: {100 * jr['block_unintended']:.1f} / {100 * jr['fbr_unintended']:.1f}); "
               f"{jr['unparseable']} unparseable outputs counted as not flagged.", ""]
    if "rule_last_tool" in names and "random" in results["aggregate"]:
        r = results["aggregate"]["random"]["rule_last_tool"]
        md += ["### Dataset artefact", "",
               f"The rule \"block a record that ends on a tool call\" (it sees the end of the record, "
               f"which a runtime guard does not) gives Block {100 * r['block@rule']['mean']:.1f}% at "
               f"FBR {100 * r['fbr@rule']['mean']:.1f}% on the test parts of the random splits.", ""]

    results["config"] = {**vars(args), "data": data, "parameters": params,
                         "feature_dim": int(feats[0].shape[1]), "alphas": list(E.ALPHAS),
                         "split_fractions": [0.5, 0.25, 0.25], "code_commit": _git_head(),
                         "rjudge_commit": rjudge_commit(args.rjudge_dir),
                         "python": platform.python_version(), "torch": torch.__version__,
                         "numpy": np.__version__, "wall_seconds": round(time.time() - t_start, 1),
                         "quick_smoke_run": bool(args.quick)}
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "results_partial.json").unlink(missing_ok=True)
    (out / "summary.md").write_text("\n".join(md))
    print("\n" + "\n".join(md))
    print(f"\nwrote {out / 'results.json'} and {out / 'summary.md'}")


if __name__ == "__main__":
    main()
