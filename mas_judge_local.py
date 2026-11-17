"""Score the proposed actions of multi-agent sessions with a local open-weight chat model as the judge.

    python scripts/mas_judge_local.py --sessions results/mas_kaggle_v2/claude_sessions.jsonl.gz \
        results/mas_kaggle_v2/openai_sessions.jsonl.gz --n-sessions 300 --model Qwen/Qwen2.5-7B-Instruct --out runs/mas_judge_qwen25

Same prompt and policy as scripts/mas_judge_claude.py, with a one-letter answer; the score is
log p(B: violates) - log p(A: complies) for the next token.  Each row records the latency of the
forward pass.  Use the same --n-sessions and --sample-seed as for the other judges so that all are
scored on the same sessions.  Needs a GPU and the model weights; not run by the author.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rjudge_judge_local as RL                # noqa: E402
from conformalguard.mas import convert as C    # noqa: E402
from conformalguard.mas import judge as J      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--n-sessions", type=int, default=300)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-input-tokens", type=int, default=6000)
    ap.add_argument("--chat-kwargs", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    chat_kwargs = json.loads(args.chat_kwargs) if args.chat_kwargs else None

    sessions = [s for s in C.load_sessions(args.sessions) if int(s.get("sim_version", 1)) == 2 and not int(s.get("refusals", 0))]
    sessions = J.sample_sessions(sessions, args.n_sessions, args.sample_seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / "scores.jsonl"
    done = J.read_scores(path)
    todo = [(s, i) for s in sessions for i in J.units(s) if (s["seed"], i) not in done]
    print(f"{args.model}: {len(sessions)} sessions, {len(done)} steps already scored, {len(todo)} to go", flush=True)
    if not todo:
        return
    tok, model = RL.load_model(args.model, args.load_4bit)
    a_ids, b_ids = RL.letter_ids(tok)
    t0 = time.time()
    with path.open("a", encoding="utf-8") as fh:
        for k, (s, i) in enumerate(todo, 1):
            messages = [{"role": "system", "content": J.INSTRUCTIONS}, {"role": "user", "content": J.user_prompt(s, i, J.LETTER_FORMAT)}]
            t1 = time.perf_counter()
            score = RL.letter_score(tok, model, messages, a_ids, b_ids, args.max_input_tokens, chat_kwargs)
            fh.write(json.dumps({"seed": s["seed"], "event": i, "guard": args.model, "score": score,
                                 "seconds": time.perf_counter() - t1}) + "\n")
            fh.flush()
            if k <= 3 or k % 200 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"  {k}/{len(todo)}  {el:.0f}s elapsed, about {el / k * (len(todo) - k) / 60:.0f} min left", flush=True)
    print(f"done: {path}")


if __name__ == "__main__":
    main()
