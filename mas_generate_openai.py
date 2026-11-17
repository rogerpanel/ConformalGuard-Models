"""Generate multi-agent sessions with an OpenAI chat model playing every agent (OpenAI SDK).

    export OPENAI_API_KEY=...
    python scripts/mas_generate_openai.py --model <model-id> --n 400 --seed-start 100000 --out runs/mas_openai

Same simulation, prompts and output file format as scripts/mas_generate_claude.py.  Use a
different --seed-start for each agent model so that no two sessions share a workspace.
Replies are requested in JSON mode (pass --no-json-mode for a model that does not support it).
Request construction and reply handling are unit-tested with a stand-in client; the script has
not been run against the live API in the environment where it was written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import openai
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformalguard.mas import generate as G  # noqa: E402


def build_request(system: str, turns: list[dict], model: str, json_mode: bool) -> dict:
    req = dict(model=model, messages=[{"role": "system", "content": system}]
               + [{"role": t["role"], "content": t["content"]} for t in turns])
    if json_mode:
        req["response_format"] = {"type": "json_object"}
    return req


def read_action(resp) -> tuple[dict, bool]:
    """(action, refused) from a chat completion."""
    message = resp.choices[0].message
    if getattr(message, "refusal", None):
        return dict(G.DECLINED), True
    return G.parse_action(message.content), False


class OpenAIPolicy:
    def __init__(self, client, model: str, json_mode: bool):
        self.client, self.model, self.json_mode = client, model, json_mode
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self.refusals = 0
        self.refusal_categories: dict[str, int] = {}
        self.served_by: set[str] = set()

    def act(self, system: str, turns: list[dict]) -> dict:
        try:
            resp = self.client.chat.completions.create(**build_request(system, turns, self.model, self.json_mode))
        except openai.BadRequestError as e:
            raise RuntimeError(f"request rejected (400): {e.message}") from e
        except openai.RateLimitError as e:
            raise RuntimeError(f"rate limited after retries: {e.message}") from e
        except openai.APIStatusError as e:
            raise RuntimeError(f"API error {e.status_code}: {e.message}") from e
        except openai.APIConnectionError as e:
            raise RuntimeError(f"connection error: {e}") from e
        self.usage["calls"] += 1
        usage = getattr(resp, "usage", None)
        self.usage["input_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.usage["output_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
        if getattr(resp, "model", None):
            self.served_by.add(resp.model)
        action, refused = read_action(resp)
        if refused:
            self.refusals += 1
            self.refusal_categories["refusal"] = self.refusal_categories.get("refusal", 0) + 1
        return action


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=400, help="number of sessions")
    ap.add_argument("--seed-start", type=int, default=100000, help="first workspace seed")
    ap.add_argument("--no-json-mode", action="store_true")
    ap.add_argument("--workers", type=int, default=8, help="sessions generated in parallel")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    client = OpenAI(max_retries=6)
    path = Path(args.out) / "sessions.jsonl"
    counts = G.run_many(lambda: OpenAIPolicy(client, args.model, not args.no_json_mode),
                        range(args.seed_start, args.seed_start + args.n), path, args.workers,
                        {"agent_model": args.model, "provider": "openai"})
    print(f"done: {counts['written']} sessions written this run, {counts['failed']} failed (rerun to retry); file {path}")
    print("usage:", G.usage_report(path))
    if counts["stopped"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
