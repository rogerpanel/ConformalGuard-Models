"""Score R-Judge records with a Claude model as the guard (Anthropic SDK).

    export ANTHROPIC_API_KEY=...          # or use `ant auth login`
    python scripts/rjudge_judge_claude.py --rjudge-dir R-Judge --mode causal --out runs/guard_claude_causal
    python scripts/rjudge_judge_claude.py --rjudge-dir R-Judge --mode full   --out runs/guard_claude_full

Writes <out>/scores.jsonl (appended as results arrive; an interrupted run resumes).
Each call asks for a 0-100 risk rating as structured output.  Calibrate the file
with scripts/rjudge_guard_eval.py.

Notes
- Refusals: R-Judge records describe harmful scenarios, so a request can be declined
  by safety classifiers.  Server-side fallbacks are enabled (`fallbacks="default"`), so
  a declined request is re-run on a fallback model in the same call; a row records
  which model served it.  Pass --no-fallbacks to disable.  A request that is still
  refused gets score null and is treated as "blocked" by the evaluator.
- Prompt caching: the transcript up to the proposed action is marked as a cache
  breakpoint and the steps of one record are scored in order, so a long record's
  later steps can read the earlier prefix from cache.  Most R-Judge records are below
  the minimum cacheable length, so expect few cache hits.
- This script has unit tests for its request construction and response handling
  with a stand-in client.  It has not been run against the live API in the
  environment where it was written.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformalguard.rjudge import guards as G  # noqa: E402

SCHEMA = {
    "type": "object",
    "properties": {"risk": {"type": "integer"}, "reason": {"type": "string"}},
    "required": ["risk", "reason"],
    "additionalProperties": False,
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def build_request(rec: dict, step: int, model: str, effort: str, fallbacks: bool) -> dict:
    mode = "full" if step < 0 else "causal"
    blocks = [{"type": "text", "text": b} for b in G.transcript_blocks(rec, None if step < 0 else step)]
    blocks[0]["text"] = "<transcript>\n" + blocks[0]["text"]
    blocks[-1]["text"] += "\n</transcript>"
    blocks[-1]["cache_control"] = {"type": "ephemeral"}          # stable prefix ends here
    blocks.append({"type": "text", "text": f"{G.QUESTION[mode]} {G.RATING_FORMAT}"})
    req = dict(
        model=model,
        max_tokens=8000,
        system=G.INSTRUCTIONS,
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": blocks}],
    )
    if fallbacks:
        req.update(betas=[FALLBACK_BETA], fallbacks="default")
    return req


def read_response(resp) -> dict:
    """Score and bookkeeping from a Messages response."""
    row = {"score": None, "served_by": getattr(resp, "model", None), "stop_reason": resp.stop_reason}
    usage = getattr(resp, "usage", None)
    if usage is not None:
        row["input_tokens"] = getattr(usage, "input_tokens", None)
        row["output_tokens"] = getattr(usage, "output_tokens", None)
        row["cache_read_input_tokens"] = getattr(usage, "cache_read_input_tokens", None)
    if resp.stop_reason == "refusal":
        details = getattr(resp, "stop_details", None)
        row["refusal_category"] = getattr(details, "category", None)
        return row
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if text is None:
        return row
    try:
        data = json.loads(text)
        risk = int(data["risk"])
        row["score"] = float(min(100, max(0, risk)))
        row["reason"] = str(data.get("reason", ""))[:300]
    except (ValueError, KeyError, TypeError):
        row["score"] = G.parse_rating(text)
    return row


def call(client, req: dict):
    if "betas" in req:
        return client.beta.messages.create(**req)
    return client.messages.create(**req)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rjudge-dir", required=True)
    ap.add_argument("--mode", choices=("full", "causal"), default="causal")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium", choices=("low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--no-fallbacks", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="records scored in parallel")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N records (testing)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = G.load_records(args.rjudge_dir)
    if args.limit:
        recs = recs[:args.limit]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / "scores.jsonl"
    done = G.read_done(path)
    client = anthropic.Anthropic(max_retries=5)
    lock = threading.Lock()
    counts = {"ok": 0, "failed": 0}

    def score_record(rec: dict) -> None:
        steps = [-1] if args.mode == "full" else list(range(G.n_steps(rec)))
        for step in steps:                                   # in order, so later steps can hit the cache
            if (int(rec["id"]), step) in done:
                continue
            try:
                resp = call(client, build_request(rec, step, args.model, args.effort, not args.no_fallbacks))
            except anthropic.RateLimitError as e:
                print(f"rate limited on record {rec['id']} step {step}: {e.message}", flush=True)
                counts["failed"] += 1
                continue
            except anthropic.APIStatusError as e:
                print(f"API error {e.status_code} on record {rec['id']} step {step}: {e.message}", flush=True)
                counts["failed"] += 1
                continue
            except anthropic.APIConnectionError as e:
                print(f"connection error on record {rec['id']} step {step}: {e}", flush=True)
                counts["failed"] += 1
                continue
            row = {"id": int(rec["id"]), "step": step, "guard": args.model, "mode": args.mode, **read_response(resp)}
            with lock:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["ok"] += 1
                if counts["ok"] % 50 == 0:
                    print(f"  {counts['ok']} scored", flush=True)

    todo = len(G.units(recs, args.mode)) - sum(1 for r, s in G.units(recs, args.mode) if (int(r["id"]), s) in done)
    print(f"{args.model}, mode {args.mode}: {len(done)} already scored, {todo} to go")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(score_record, recs))
    rows = G.read_done(path)
    refused = sum(1 for r in rows.values() if r.get("stop_reason") == "refusal")
    fell_back = sum(1 for r in rows.values() if r.get("served_by") not in (None, args.model))
    print(f"done: {len(rows)} rows in {path}; {counts['failed']} calls failed this run (rerun to retry); "
          f"{refused} refused, {fell_back} served by a fallback model, "
          f"{sum(1 for r in rows.values() if r['score'] is None)} without a score")


if __name__ == "__main__":
    main()
