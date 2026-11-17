"""Score the proposed actions of multi-agent sessions with a Claude model as the judge (Anthropic SDK).

    python scripts/mas_judge_claude.py --sessions results/mas_kaggle_v2/claude_sessions.jsonl.gz \
        results/mas_kaggle_v2/openai_sessions.jsonl.gz --n-sessions 300 --out runs/mas_judge_claude

The judge is given the policy and the session log up to each proposed action, with the monitor's
metadata (conformalguard/mas/judge.py), and returns a 0-100 rating as structured output.  Each row
of <out>/scores.jsonl records the rating, the wall-clock latency of the request, token usage and
which model served it.  An interrupted run resumes.  Calibrate with scripts/mas_guard_eval.py.

Server-side refusal fallbacks are on for models that support them.  The steps of one session are
scored in order with a cache breakpoint on the log, so later steps read the earlier log from cache.
Latency is measured around the API call with the stated number of parallel sessions; it includes
the network and any retries.  Request construction and response handling are unit-tested with a
stand-in client; the script was run live once with the whole log in a single block, which wrote the cache at every step and never read it (see results/followup_kaggle/README.md); the per-line blocks have not been run live.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformalguard.mas import convert as C   # noqa: E402
from conformalguard.mas import judge as J     # noqa: E402

SCHEMA = {"type": "object", "properties": {"risk": {"type": "integer"}}, "required": ["risk"], "additionalProperties": False}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
NO_SERVER_FALLBACK = ("haiku", "claude-opus-4", "claude-sonnet-4")


def build_request(session: dict, event_index: int, model: str, effort: str, fallbacks: bool) -> dict:
    before, proposed = J.transcript(session, event_index)
    # One block per log line: a cache entry is looked up at block boundaries, so the log of an earlier step
    # must end on a boundary of this request for its cached prefix to be read.
    blocks = [{"type": "text", "text": ("<log>\n" if k == 0 else "") + line + "\n"} for k, line in enumerate(before)]
    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    blocks.append({"type": "text", "text": proposed + "\n</log>\n\n" + f"{J.QUESTION} {J.RATING_FORMAT}"})
    req = dict(model=model, max_tokens=4000, system=J.INSTRUCTIONS,
               output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
               messages=[{"role": "user", "content": blocks}])
    if fallbacks:
        req.update(betas=[FALLBACK_BETA], fallbacks="default")
    return req


def read_response(resp) -> dict:
    row = {"score": None, "served_by": getattr(resp, "model", None), "stop_reason": resp.stop_reason}
    usage = getattr(resp, "usage", None)
    for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        row[k] = int(getattr(usage, k, 0) or 0)
    if resp.stop_reason == "refusal":
        row["refusal_category"] = str(getattr(getattr(resp, "stop_details", None), "category", None) or "unspecified")
        return row
    text = next((b.text for b in resp.content if b.type == "text"), None)
    try:
        row["score"] = float(min(100, max(0, int(json.loads(text)["risk"]))))
    except (ValueError, KeyError, TypeError):
        pass
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--n-sessions", type=int, default=300, help="judge a random sample of this many sessions (0: all)")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="low", choices=("low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--no-fallbacks", action="store_true")
    ap.add_argument("--workers", type=int, default=6, help="sessions judged in parallel")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sessions = [s for s in C.load_sessions(args.sessions) if int(s.get("sim_version", 1)) == 2 and not int(s.get("refusals", 0))]
    sessions = J.sample_sessions(sessions, args.n_sessions, args.sample_seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / "scores.jsonl"
    done = J.read_scores(path)
    fallbacks = not args.no_fallbacks and not any(k in args.model for k in NO_SERVER_FALLBACK)
    client = anthropic.Anthropic(max_retries=5)
    lock = threading.Lock()
    counts = {"ok": 0, "failed": 0}
    todo = sum((s["seed"], i) not in done for s in sessions for i in J.units(s))
    print(f"{args.model}: {len(sessions)} sessions, {len(done)} steps already scored, {todo} to go", flush=True)

    def judge_session(s: dict) -> None:
        for i in J.units(s):                                  # in order, so later steps read the log from cache
            if (s["seed"], i) in done:
                continue
            req = build_request(s, i, args.model, args.effort, fallbacks)
            t0 = time.perf_counter()
            try:
                resp = client.beta.messages.create(**req) if "betas" in req else client.messages.create(**req)
            except anthropic.RateLimitError as e:
                print(f"rate limited on session {s['seed']} step {i}: {e.message}", flush=True); counts["failed"] += 1; continue
            except anthropic.APIStatusError as e:
                print(f"API error {e.status_code} on session {s['seed']} step {i}: {e.message}", flush=True); counts["failed"] += 1; continue
            except anthropic.APIConnectionError as e:
                print(f"connection error on session {s['seed']} step {i}: {e}", flush=True); counts["failed"] += 1; continue
            row = {"seed": s["seed"], "event": i, "guard": args.model, "seconds": time.perf_counter() - t0, **read_response(resp)}
            with lock:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                counts["ok"] += 1
                if counts["ok"] % 200 == 0:
                    print(f"  {counts['ok']} steps scored", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(judge_session, sessions))
    rows = list(J.read_scores(path).values())
    print(f"done: {len(rows)} rows in {path}; {counts['failed']} calls failed this run (rerun to retry); "
          f"{sum(r.get('stop_reason') == 'refusal' for r in rows)} refused, "
          f"{sum(r.get('served_by') not in (None, args.model) for r in rows)} served by a fallback model, "
          f"{sum(r['score'] is None for r in rows)} without a score")


if __name__ == "__main__":
    main()
