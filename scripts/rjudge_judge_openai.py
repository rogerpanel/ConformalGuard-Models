"""Score R-Judge records with an OpenAI chat model as the guard (OpenAI SDK).

    export OPENAI_API_KEY=...
    python scripts/rjudge_judge_openai.py --rjudge-dir R-Judge --model <model-id> --mode causal --out runs/guard_openai_causal

--score rating   (default) asks for a 0-100 risk rating; works with any chat model.
--score logprob  asks for one letter (A safe / B unsafe) and uses log p(B) - log p(A) from the
                 token log-probabilities.  Needs a model that returns logprobs.

Writes <out>/scores.jsonl; an interrupted run resumes.  Calibrate with
scripts/rjudge_guard_eval.py.  Request construction and response handling are unit-tested
with a stand-in client; the script has not been run against the live API in the environment
where it was written.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformalguard.rjudge import guards as G  # noqa: E402


def build_request(rec: dict, step: int, model: str, score: str) -> dict:
    fmt = G.RATING_FORMAT if score == "rating" else G.LETTER_FORMAT
    req = dict(model=model, messages=[{"role": "system", "content": G.INSTRUCTIONS},
                                      {"role": "user", "content": G.user_prompt(rec, step, fmt)}])
    if score == "logprob":
        req.update(logprobs=True, top_logprobs=20, max_completion_tokens=4)
    return req


def read_response(resp, score: str) -> dict:
    choice = resp.choices[0]
    text = choice.message.content or ""
    row = {"score": None, "served_by": getattr(resp, "model", None), "raw": text[:300]}
    if score == "rating":
        row["score"] = G.parse_rating(text)
        return row
    lp = {"A": None, "B": None}
    content = getattr(getattr(choice, "logprobs", None), "content", None) or []
    if content:
        for cand in content[0].top_logprobs:
            t = cand.token.strip().upper()
            if t in lp and (lp[t] is None or cand.logprob > lp[t]):
                lp[t] = cand.logprob
    if lp["A"] is not None or lp["B"] is not None:
        floor = min(v for v in lp.values() if v is not None) - 5.0     # a letter outside the top-20
        row["score"] = float((lp["B"] if lp["B"] is not None else floor)
                             - (lp["A"] if lp["A"] is not None else floor))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rjudge-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=("full", "causal"), default="causal")
    ap.add_argument("--score", choices=("rating", "logprob"), default="rating")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = G.load_records(args.rjudge_dir)
    if args.limit:
        recs = recs[:args.limit]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / "scores.jsonl"
    done = G.read_done(path)
    todo = [(r, s) for r, s in G.units(recs, args.mode) if (int(r["id"]), s) not in done]
    print(f"{args.model}, mode {args.mode}, score {args.score}: {len(done)} already scored, {len(todo)} to go")
    client = OpenAI(max_retries=5)
    lock = threading.Lock()
    n = {"ok": 0, "failed": 0}

    def one(unit) -> None:
        rec, step = unit
        try:
            resp = client.chat.completions.create(**build_request(rec, step, args.model, args.score))
        except Exception as e:                                # noqa: BLE001  (reported, retried on rerun)
            n["failed"] += 1
            print(f"error on record {rec['id']} step {step}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            time.sleep(1.0)
            return
        row = {"id": int(rec["id"]), "step": step, "guard": args.model, "mode": args.mode,
               **read_response(resp, args.score)}
        with lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n["ok"] += 1
            if n["ok"] % 50 == 0:
                print(f"  {n['ok']}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, todo))
    rows = G.read_done(path)
    print(f"done: {len(rows)} rows in {path}; {n['failed']} calls failed this run (rerun to retry); "
          f"{sum(1 for r in rows.values() if r['score'] is None)} without a score")


if __name__ == "__main__":
    main()
