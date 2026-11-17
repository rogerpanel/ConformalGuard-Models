"""Multi-agent trace experiment: step-level conformal guard on simulated team sessions.

    python scripts/mas_generate_claude.py --n 400 --out runs/mas_claude
    python scripts/mas_generate_openai.py --model <id> --n 400 --out runs/mas_openai
    python scripts/mas_run.py --sessions runs/mas_claude/sessions.jsonl runs/mas_openai/sessions.jsonl \
        --features st --out runs/mas_exp

Step labels come from code (conformalguard/mas/labels.py).  Protocols:

  random   sessions are split 50/25/25 at random, stratified by whether they contain a
           violation.  Calibration and test sessions are exchangeable, so the step-level
           false-block guarantee (Proposition 1) applies.
  cross    only with sessions from two or more agent models: train and calibrate on the
           sessions of one model, test on those of another.  A distribution shift; no
           guarantee applies and the false-block rate is reported to show what happens.

--quick runs three splits with six epochs to check the installation; its output is marked as
a smoke run and is not a result.
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
from conformalguard.mas import convert as C          # noqa: E402
from conformalguard.mas import experiment as X       # noqa: E402
from conformalguard.mas import labels as L           # noqa: E402
from conformalguard.rjudge import experiment as E    # noqa: E402
from conformalguard.rjudge.features import node_features  # noqa: E402


def code_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", required=True, help="one or more sessions.jsonl files")
    ap.add_argument("--features", choices=("lsa", "st"), default="lsa")
    ap.add_argument("--st-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--protocols", default="random,cross")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--cross-reps", type=int, default=10)
    ap.add_argument("--models", default="step,seq,graph,graph_nocites,graph_untyped")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=256, help="position embeddings (nodes per session)")
    ap.add_argument("--no-thread-feature", action="store_true",
                    help="do not give scorers the task-thread slot of each node")
    ap.add_argument("--keep-refused", action="store_true",
                    help="keep sessions in which the provider refused an agent step (dropped by default)")
    ap.add_argument("--sim-version", type=int, default=0,
                    help="use only sessions of this simulator version (default: the newest one present)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/mas_exp")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.cross_reps, args.epochs = 3, 1, 6

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sessions = C.load_sessions(args.sessions)
    loaded = len(sessions)
    version = args.sim_version or max(int(s.get("sim_version", 1)) for s in sessions)
    sessions = [s for s in sessions if int(s.get("sim_version", 1)) == version]
    other_version = loaded - len(sessions)
    with_refusal = sum(int(s.get("refusals", 0)) > 0 for s in sessions)
    if not args.keep_refused:
        sessions = [s for s in sessions if int(s.get("refusals", 0)) == 0]
    data = C.summary(sessions)
    data.update(sessions_loaded=loaded, sim_version=version, dropped_other_sim_version=other_version,
                sessions_with_a_refused_step=with_refusal, refused_sessions_dropped=not args.keep_refused)
    print(f"{loaded} sessions loaded; simulator version {version}; {other_version} of another version left out; "
          f"{with_refusal} with a provider-refused step {'kept' if args.keep_refused else 'left out'}")
    traces, metas, targets, infos = [], [], [], []
    for i, s in enumerate(sessions):
        tr, meta, y, info = C.session_to_trace(s, i, thread_feature=not args.no_thread_feature)
        traces.append(tr); metas.append(meta); targets.append(y); infos.append(info)
    text = node_features(traces, args.features, args.st_model, cache_dir=out / "cache")
    feats = [np.concatenate([t, m], axis=1) for t, m in zip(text, metas)]
    sizes = np.array([t.n_nodes for t in traces])
    data.update(nodes_mean=float(sizes.mean()), nodes_max=int(sizes.max()),
                edges_by_relation={r: int(sum(e[2] == k for t in traces for e in t.edges))
                                   for r, k in C.REL.items()})
    print(json.dumps(data, indent=1))
    if sizes.max() > args.max_len:
        print(f"warning: {int((sizes > args.max_len).sum())} sessions have more than --max-len {args.max_len} nodes; "
              f"their later nodes share one position embedding")
    if data["violating_steps"] < 20:
        print("warning: fewer than 20 violating steps; block rates will be very noisy")
    names = [m for m in args.models.split(",") if m]
    kw = dict(epochs=args.epochs, layers=args.layers, max_len=args.max_len)
    protocols = [p for p in args.protocols.split(",") if p]
    results: dict = {"per_split": {}, "aggregate": {}, "paired": {}}
    t0 = time.time()

    if "random" in protocols:
        rows = []
        for r in range(args.reps):
            rng = np.random.default_rng([args.seed, zlib.crc32(b"mas-random"), r])
            parts = E.split_random(traces, rng)
            rows.append(X.run_split(traces, feats, targets, infos, parts, args.seed * 100_003 + r, args.device, names, kw))
            g = rows[-1][names[-1]]
            print(f"random {r + 1}/{args.reps}  {names[-1]}: fbr@.10={g['fbr@0.10']:.3f} block@.10={g['block@0.10']:.3f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
            (out / "results_partial.json").write_text(json.dumps({"random": rows}))
        results["per_split"]["random"] = rows
        results["aggregate"]["random"] = E.aggregate(rows)
        pairs = [(a, b) for a, b in (("graph", "seq"), ("graph", "graph_nocites"), ("graph", "graph_untyped"), ("seq", "step"))
                 if a in names and b in names]
        results["paired"]["random"] = {f"{a}-{b}": {m: E.paired(rows, a, b, m) for m in
                                                    ("block_session@0.05", "block_session@0.10", "block_session_concurrent@0.10",
                                                     "block_session_single@0.10", "block_session_restricted_exfiltration@0.10",
                                                     "block@0.10", "auroc_step")}
                                       for a, b in pairs}

    agent_models = sorted({t.category for t in traces})
    if "cross" in protocols and len(agent_models) >= 2:
        for src in agent_models:
            for dst in agent_models:
                if src == dst:
                    continue
                rows = []
                a_idx = np.array([i for i, t in enumerate(traces) if t.category == src])
                b_idx = np.array([i for i, t in enumerate(traces) if t.category == dst])
                for r in range(args.cross_reps):
                    rng = np.random.default_rng([args.seed, zlib.crc32(f"mas-cross-{src}-{dst}".encode()), r])
                    perm = rng.permutation(a_idx)
                    cut = int(round(2 * perm.size / 3))
                    parts = (np.sort(perm[:cut]), np.sort(perm[cut:]), b_idx)
                    rows.append(X.run_split(traces, feats, targets, infos, parts, args.seed * 100_003 + r, args.device, names, kw))
                    print(f"cross {src} -> {dst} {r + 1}/{args.cross_reps}  ({time.time() - t0:.0f}s)", flush=True)
                results["per_split"][f"cross:{src}->{dst}"] = rows
                results["aggregate"][f"cross:{src}->{dst}"] = E.aggregate(rows)

    def cell(agg, n, k):
        v = agg.get(n, {}).get(k)
        return f"{100 * v['mean']:.1f} ± {100 * v['sd']:.1f}" if v else "n/a"

    md = [f"# Multi-agent trace experiment ({'QUICK SMOKE RUN, not a result' if args.quick else 'full run'})", "",
          f"- {data['sessions']} sessions ({data['concurrent_sessions']} with concurrent tasks), agent models: {data['by_agent_model']}",
          f"- {data['steps']} agent steps: {data['safe_steps']} safe, {data['violating_steps']} violating "
          f"({data['violations_by_kind']}); {data['sessions_with_violation']} sessions contain a violation",
          f"- {data['tasks_with_injection']} of {data['tasks']} tasks had an injected instruction planted",
          f"- simulator version {data['sim_version']}; {data['sessions_loaded']} sessions loaded, {data['dropped_other_sim_version']} of another "
          f"simulator version left out, {data['sessions_with_a_refused_step']} with a step refused by the model provider "
          f"({'kept' if args.keep_refused else 'left out'})",
          f"- nodes per session: mean {data['nodes_mean']:.1f}, max {data['nodes_max']}; labels are code-derived",
          f"- features: `{args.features}`" + (f" (`{args.st_model}`)" if args.features == "st" else "")
          + f" plus {C.META_DIM} metadata dimensions"
          + (" (no task-thread slot)" if args.no_thread_feature else "") + f"; {args.layers} layers; per-step loss", ""]
    if "random" in results["aggregate"]:
        agg = results["aggregate"]["random"]
        md += [f"## Random splits ({len(results['per_split']['random'])} splits, 50/25/25)", "",
               "### Step-level rule (Proposition 1): a random safe step is blocked with probability <= alpha", "",
               "FBR: a random safe step of a test session is blocked. Block: a violating step is blocked when proposed. "
               "Most steps are plainly harmless, so this budget is easy to meet; the session-level table below is the stricter test.", "",
               "| scorer | step AUROC | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |", "|---|---|---|---|---|---|---|---|"]
        for n in names:
            au = agg[n].get("auroc_step")
            md.append(f"| {n} | {au['mean']:.3f} | " if au else f"| {n} | n/a | ")
            md[-1] += " | ".join(cell(agg, n, k) for k in ("fbr@0.05", "block@0.05", "fbr@0.10", "block@0.10", "fbr@0.20", "block@0.20")) + " |"
        md += ["", "### Session-level rule (Corollary 1): a violation-free session has a blocked step with probability <= alpha", "",
               "FBR: share of violation-free test sessions with at least one blocked step. Block: a violating step is blocked when proposed.", "",
               "| scorer | FBR@.05 | Block@.05 | FBR@.10 | Block@.10 | FBR@.20 | Block@.20 |", "|---|---|---|---|---|---|---|"]
        for n in names:
            md.append(f"| {n} | " + " | ".join(cell(agg, n, k) for k in ("fbr_session@0.05", "block_session@0.05", "fbr_session@0.10",
                                                                         "block_session@0.10", "fbr_session@0.20", "block_session@0.20")) + " |")
        md += ["", "### By concurrency, session-level rule, alpha = 0.10 (conditional rates; the guarantee is marginal)", "",
               "| scorer | FBR, one task | FBR, concurrent tasks | Block, one task | Block, concurrent tasks |", "|---|---|---|---|---|"]
        for n in names:
            md.append(f"| {n} | " + " | ".join(cell(agg, n, k) for k in ("fbr_session_single@0.10", "fbr_session_concurrent@0.10",
                                                                         "block_session_single@0.10", "block_session_concurrent@0.10")) + " |")
        md += ["", "### Block by kind of violation, alpha = 0.10 (step-level rule / session-level rule)", "",
               "| scorer | " + " | ".join(L.KINDS) + " |", "|---|" + "---|" * len(L.KINDS)]
        for n in names:
            md.append(f"| {n} | " + " | ".join(f"{cell(agg, n, f'block_{k}@0.10')} / {cell(agg, n, f'block_session_{k}@0.10')}"
                                                for k in L.KINDS) + " |")
        md += ["", "### Paired differences over splits (percentage points; exact sign test)", "",
               "| comparison | metric | mean difference | splits better / worse | p |", "|---|---|---|---|---|"]
        for pair, ms in results["paired"]["random"].items():
            for m, p in ms.items():
                if not np.isnan(p["mean_diff"]):
                    scale = 1 if m == "auroc_step" else 100
                    md.append(f"| {pair} | {m} | {scale * p['mean_diff']:+.{3 if scale == 1 else 1}f} | {p['a_better']} / {p['b_better']} | {p['sign_test_p']:.3g} |")
    for key, agg in results["aggregate"].items():
        if key.startswith("cross:"):
            md += ["", f"## {key} ({len(results['per_split'][key])} runs): distribution shift, no guarantee", "",
                   "| scorer | step FBR@.10 | step Block@.10 | session FBR@.10 | session Block@.10 | step AUROC |", "|---|---|---|---|---|---|"]
            for n in names:
                au = agg[n].get("auroc_step")
                md.append(f"| {n} | " + " | ".join(cell(agg, n, k) for k in ("fbr@0.10", "block@0.10", "fbr_session@0.10", "block_session@0.10"))
                          + (f" | {au['mean']:.3f} |" if au else " | n/a |"))
    results["config"] = {**vars(args), "data": data, "code_commit": code_commit(), "torch": torch.__version__,
                         "quick_smoke_run": bool(args.quick)}
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md)); print(f"\nwrote {out / 'results.json'} and {out / 'summary.md'}")


if __name__ == "__main__":
    main()
