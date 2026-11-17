"""Closed-loop study with a Claude model playing the agents (Anthropic SDK): the same simulated
workspaces are run without and with the guard in the loop.

    python scripts/mas_closed_loop_claude.py \
        --train-sessions results/mas_kaggle_v2/claude_sessions.jsonl.gz results/mas_kaggle_v2/openai_sessions.jsonl.gz \
        --n 150 --alpha 0.10 --out runs/closed_loop

Steps: (1) train the graph scorer on half of the logged sessions and take its session-level
conformal threshold from the other half; (2) generate --n sessions on new workspaces without the
guard; (3) generate the same workspaces with the guard scoring every proposed action before it
takes effect (a blocked action is returned to the agent, which may propose another up to twice,
after which that task thread stops); (4) report task success, violations prevented, tasks that
still succeed after a block, and the rate of false interruptions.

Everything is resumable: the trained guard is saved to <out>/guard.pt and reused, and finished
sessions are kept.  Request handling is shared with scripts/mas_generate_claude.py.  The guard
and the report are unit-tested with scripted agents; the live path has not been run by the author.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anthropic
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mas_generate_claude as GC                       # noqa: E402
from conformalguard.mas import closed_loop as CL       # noqa: E402
from conformalguard.mas import convert as C            # noqa: E402
from conformalguard.mas import generate as G           # noqa: E402
from conformalguard.mas import guard as GD             # noqa: E402
from conformalguard.rjudge import models as M          # noqa: E402


def load_checkpoint(path: Path, device: str) -> dict:
    """The saved guard (weights, threshold, settings).  It is a file this script wrote itself."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:                                         # older PyTorch without the argument
        return torch.load(path, map_location=device)


def load_or_train_guard(args, out: Path) -> GD.GraphGuard:
    encoder = GD.STEncoder(args.st_model, device=args.device)
    ckpt = out / "guard.pt"
    if ckpt.exists():
        saved = load_checkpoint(ckpt, args.device)
        info = saved["info"]
        model = M.build(info["model"], saved["feat_dim"], d=info.get("d", 128), layers=info["layers"], max_len=info["max_len"]).to(args.device)
        model.load_state_dict(saved["state"])
        print(f"reusing the guard saved in {ckpt}: {info}")
        return GD.GraphGuard(model, encoder, info["threshold"], args.device, True, info)
    sessions = [s for s in C.load_sessions(args.train_sessions)
                if int(s.get("sim_version", 1)) == 2 and not int(s.get("refusals", 0))]
    print(f"training the guard on {len(sessions)} logged sessions (half for training, half for the threshold)")
    guard = GD.train_guard(sessions, encoder, alpha=args.alpha, seed=args.seed, device=args.device, epochs=args.epochs)
    feat_dim = guard.model.embed.proj.in_features
    torch.save({"state": guard.model.state_dict(), "info": guard.info, "feat_dim": feat_dim}, ckpt)
    print(f"guard trained: {guard.info}")
    return guard


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-sessions", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=150, help="workspaces; each is run once without and once with the guard")
    ap.add_argument("--seed-start", type=int, default=200000, help="first workspace seed (must not overlap the training sessions)")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--backup-model", default="")
    ap.add_argument("--effort", default="low", choices=("low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--no-fallbacks", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--st-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--report-only", action="store_true", help="write the report from the sessions already generated")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    keep = lambda arm: [s for s in C.load_sessions(out / arm / "sessions.jsonl") if not int(s.get("refusals", 0))] if (out / arm / "sessions.jsonl").exists() else []
    if args.report_only:
        saved = load_checkpoint(out / "guard.pt", "cpu")
        model = json.loads((out / "config.json").read_text())["chosen"]["model"] if (out / "config.json").exists() else args.model
        results, md = CL.report(keep("no_guard"), keep("guard"), {**saved["info"], "agent_model": model, "max_reproposals": 2},
                                note=" (report from the sessions generated so far)")
        (out / "results.json").write_text(json.dumps(results, indent=1)); (out / "summary.md").write_text(md)
        print(md)
        return
    guard = load_or_train_guard(args, out)
    client = anthropic.Anthropic(max_retries=6)
    cfg_path = out / "config.json"
    if cfg_path.exists():
        chosen = json.loads(cfg_path.read_text())["chosen"]
    else:
        cand = lambda model, schema: {"model": model, "schema": schema, "fallbacks": not args.no_fallbacks and GC.supports_fallbacks(model)}
        candidates = [cand(args.model, True), cand(args.model, False)] + ([cand(args.backup_model, True)] if args.backup_model else [])
        chosen, report = GC.probe(client, candidates, args.effort)
        if chosen is None:
            print("STOPPED: the provider did not accept any request configuration; nothing was generated.")
            sys.exit(2)
        cfg_path.write_text(json.dumps({"chosen": chosen, "probe": report}, indent=1))
    make_policy = lambda: GC.ClaudePolicy(client, chosen["model"], args.effort, chosen["fallbacks"], chosen["schema"])
    seeds = range(args.seed_start, args.seed_start + args.n)
    extra = {"agent_model": chosen["model"], "provider": "anthropic", "effort": args.effort}
    stopped = False
    for arm, g in (("no_guard", None), ("guard", guard)):
        print(f"--- arm: {arm}")
        counts = G.run_many(make_policy, seeds, out / arm / "sessions.jsonl", args.workers, {**extra, "arm": arm}, guard=g)
        print(f"{arm}: {counts['written']} sessions written this run, {counts['failed']} failed (rerun to retry)")
        print("usage:", G.usage_report(out / arm / "sessions.jsonl"))
        stopped = stopped or counts["stopped"]
    info = {**guard.info, "agent_model": chosen["model"], "max_reproposals": 2}
    results, md = CL.report(keep("no_guard"), keep("guard"), info)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "summary.md").write_text(md)
    print("\n" + md)
    if stopped:
        sys.exit(2)


if __name__ == "__main__":
    main()
