"""Score the proposed actions of multi-agent sessions with a chat model served through the OpenAI API
or an OpenAI-compatible service such as OpenRouter (OpenAI SDK).

    export OPENROUTER_API_KEY=...
    python scripts/mas_judge_openai.py --sessions results/mas_kaggle_v2/claude_sessions.jsonl.gz results/mas_kaggle_v2/openai_sessions.jsonl.gz \
        --n-sessions 200 --model deepseek/deepseek-chat --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \
        --out runs/mas_judge_deepseek

Same prompt and policy as scripts/mas_judge_claude.py; the judge answers with an integer from 0 to
100.  Each row of <out>/scores.jsonl records the rating, the wall-clock latency of the request and
token usage.  An interrupted run resumes.  Calibrate with scripts/mas_guard_eval.py.  Request
construction and response handling are unit-tested with a stand-in client; not run live by the author.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformalguard.mas import convert as C        # noqa: E402
from conformalguard.mas import judge as J          # noqa: E402


def build_request(session: dict, event_index: int, model: str) -> dict:
    return dict(model=model, messages=[{"role": "system", "content": J.INSTRUCTIONS},
                                       {"role": "user", "content": J.user_prompt(session, event_index, J.RATING_FORMAT)}])


def read_response(resp) -> dict:
    message = resp.choices[0].message
    usage = getattr(resp, "usage", None)
    row = {"score": None, "served_by": getattr(resp, "model", None),
           "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0), "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0)}
    if getattr(message, "refusal", None):
        row["stop_reason"] = "refusal"
        return row
    row["score"] = J.leading_rating(message.content)
    row["raw"] = (message.content or "")[:120]
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--n-sessions", type=int, default=200)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="", help="an OpenAI-compatible endpoint, e.g. https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="environment variable that holds the key")
    ap.add_argument("--workers", type=int, default=6, help="sessions judged in parallel")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sessions = [s for s in C.load_sessions(args.sessions) if int(s.get("sim_version", 1)) == 2 and not int(s.get("refusals", 0))]
    sessions = J.sample_sessions(sessions, args.n_sessions, args.sample_seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / "scores.jsonl"
    done = J.read_scores(path)
    client = OpenAI(max_retries=5, api_key=os.environ.get(args.api_key_env), **({"base_url": args.base_url} if args.base_url else {}))
    lock = threading.Lock()
    counts = {"ok": 0, "failed": 0}
    todo = sum((s["seed"], i) not in done for s in sessions for i in J.units(s))
    print(f"{args.model}: {len(sessions)} sessions, {len(done)} steps already scored, {todo} to go", flush=True)

    def judge_session(s: dict) -> None:
        for i in J.units(s):
            if (s["seed"], i) in done:
                continue
            t0 = time.perf_counter()
            try:
                resp = client.chat.completions.create(**build_request(s, i, args.model))
            except openai.RateLimitError as e:
                print(f"rate limited on session {s['seed']} step {i}: {e.message}", flush=True); counts["failed"] += 1; continue
            except openai.APIStatusError as e:
                print(f"API error {e.status_code} on session {s['seed']} step {i}: {e.message}", flush=True); counts["failed"] += 1; continue
            except openai.APIConnectionError as e:
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
          f"{sum(r['score'] is None for r in rows)} without a score")


if __name__ == "__main__":
    main()
