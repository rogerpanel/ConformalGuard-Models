"""Generate multi-agent sessions with a Claude model playing every agent (Anthropic SDK).

    export ANTHROPIC_API_KEY=...          # or use `ant auth login`
    python scripts/mas_generate_claude.py --n 400 --out runs/mas_claude

Each session is one simulated workspace (conformalguard/mas/world.py) in which a coordinator,
a researcher and an acting agent work on one to three concurrent tasks with mock tools.
Nothing has a real side effect.  Finished sessions are appended to <out>/sessions.jsonl; an
interrupted run resumes, and a session whose calls failed is retried on the next run.
Labels are computed later, by code, from the event log (conformalguard/mas/labels.py).

Notes
- Every agent step is one Messages request with the agent's own history.  The reply is a
  JSON action (call a tool, hand over, or finish), requested as structured output.
- The action has no free-text field besides the message itself.  Two earlier formats, which
  asked each agent for a "thought" and then for a "note for the activity log", had almost
  every step declined in the refusal category `reasoning_extraction` (a scratchpad or running
  log filled in before the answer is a documented trigger).  That category has no server-side
  fallback, so the fix is in the request.
- Before generating, the script probes request configurations on a few throw-away sessions
  and uses the first one the provider accepts: the chosen model with structured output, the
  same model with JSON asked for in the prompt only, then --backup-model.  The choice is
  printed, saved in <out>/config.json and reused when the run resumes.  --no-probe skips this.
- Refusals for other reasons: some workspaces contain an injected instruction, so a request
  can be declined by safety classifiers.  Server-side fallbacks are on (`fallbacks="default"`)
  for models that support them, so such a request is re-run on a fallback model in the same
  call; each session records which models served it, how many steps were refused and in
  which category.  Generation stops if more than a quarter of the steps are refused.
- Prompt caching: the last turn of each request is a cache breakpoint, so an agent's later
  steps read its earlier history from cache once the history is long enough to be cached.
- Request construction, reply handling and the probe are unit-tested with stand-in clients.
  The author cannot call the live API; see docs/MAS_EXPERIMENT.md for what live runs showed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformalguard.mas import generate as G  # noqa: E402
from conformalguard.mas.sim import run_session  # noqa: E402
from conformalguard.mas.world import make_world  # noqa: E402

ARG_NAMES = ("doc_id", "name", "to", "subject", "body", "channel", "text", "url", "content",
             "with_user", "user", "resource")
SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["tool", "message", "finish"]},
        "tool": {"type": "string"},
        "args": {"type": "object", "properties": {n: {"type": "string"} for n in ARG_NAMES},
                 "required": list(ARG_NAMES), "additionalProperties": False},
        "to": {"type": "string"},
        "content": {"type": "string"},
    },
    "required": ["kind", "tool", "args", "to", "content"],
    "additionalProperties": False,
}
ARGS_NOTE = ("\nIn \"args\", fill in only the arguments of the tool you call and leave every other "
             "field an empty string.")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
NO_SERVER_FALLBACK = ("haiku", "claude-opus-4", "claude-sonnet-4", "claude-sonnet-5")   # prefixes / substrings
PROBE_SEED = 9_000_000          # probe workspaces are never written to the session file


def supports_fallbacks(model: str) -> bool:
    """Server-side refusal fallbacks exist for the models that run safety classifiers, except Haiku."""
    return not any(k in model for k in NO_SERVER_FALLBACK) or model.startswith("claude-sonnet-5-5")


def build_request(system: str, turns: list[dict], model: str, effort: str, fallbacks: bool, schema: bool) -> dict:
    messages = [{"role": t["role"], "content": t["content"]} for t in turns]
    messages[-1] = {"role": messages[-1]["role"],
                    "content": [{"type": "text", "text": messages[-1]["content"],
                                 "cache_control": {"type": "ephemeral"}}]}
    output_config: dict = {"effort": effort}
    if schema:
        output_config["format"] = {"type": "json_schema", "schema": SCHEMA}
    req = dict(model=model, max_tokens=8000, system=system + (ARGS_NOTE if schema else ""),
               output_config=output_config, messages=messages)
    if fallbacks:
        req.update(betas=[FALLBACK_BETA], fallbacks="default")
    return req


def read_action(resp) -> tuple[dict, str | None]:
    """(action, refusal category or None) from a Messages response."""
    if resp.stop_reason == "refusal":
        return dict(G.DECLINED), str(getattr(getattr(resp, "stop_details", None), "category", None) or "unspecified")
    text = next((b.text for b in resp.content if b.type == "text"), None)
    return G.parse_action(text), None


class ClaudePolicy:
    """One agent step = one request.  Counters belong to the session the policy is used for."""

    def __init__(self, client, model: str, effort: str, fallbacks: bool, schema: bool):
        self.client, self.model, self.effort = client, model, effort
        self.fallbacks, self.schema = fallbacks, schema
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0}
        self.refusals = 0
        self.refusal_categories: dict[str, int] = {}
        self.served_by: set[str] = set()

    def act(self, system: str, turns: list[dict]) -> dict:
        req = build_request(system, turns, self.model, self.effort, self.fallbacks, self.schema)
        try:
            resp = self.client.beta.messages.create(**req) if "betas" in req else self.client.messages.create(**req)
        except anthropic.BadRequestError as e:
            raise RuntimeError(f"request rejected (400): {e.message}") from e
        except anthropic.RateLimitError as e:
            raise RuntimeError(f"rate limited after retries: {e.message}") from e
        except anthropic.APIStatusError as e:
            raise RuntimeError(f"API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise RuntimeError(f"connection error: {e}") from e
        self.usage["calls"] += 1
        usage = getattr(resp, "usage", None)
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            self.usage[k] += int(getattr(usage, k, 0) or 0)
        if getattr(resp, "model", None):
            self.served_by.add(resp.model)
        action, category = read_action(resp)
        if category is not None:
            self.refusals += 1
            self.refusal_categories[category] = self.refusal_categories.get(category, 0) + 1
        return action


def probe(client, candidates: list[dict], effort: str, sessions: int = 3) -> tuple[dict | None, list[dict]]:
    """Try each request configuration on a few throw-away sessions; return the first one the provider
    accepts (at most 5% of agent steps refused, no request error) and a report on every one tried."""
    report = []
    for cand in candidates:
        calls = refused = 0
        cats: dict[str, int] = {}
        error = None
        for k in range(sessions):
            policy = ClaudePolicy(client, cand["model"], effort, cand["fallbacks"], cand["schema"])
            try:
                session = run_session(make_world(PROBE_SEED + k), policy)
            except RuntimeError as e:
                error = str(e)[:300]
                break
            calls += sum(e["type"] == "action" for e in session["events"])
            refused += policy.refusals
            for c, v in policy.refusal_categories.items():
                cats[c] = cats.get(c, 0) + v
            if calls >= 8 and refused > 0.25 * calls:          # plainly not accepted: stop spending on it
                break
        ok = error is None and calls > 0 and refused <= 0.05 * calls
        report.append({**cand, "agent_steps": calls, "refused": refused, "categories": cats, "error": error, "accepted": ok})
        print(f"probe: model {cand['model']}, structured output {'on' if cand['schema'] else 'off'}: "
              f"{refused} of {calls} agent steps refused {cats or ''}" + (f"; error: {error}" if error else "")
              + (" -> accepted" if ok else " -> not usable"), flush=True)
        if ok:
            return cand, report
    return None, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400, help="number of sessions")
    ap.add_argument("--seed-start", type=int, default=0, help="first workspace seed")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--backup-model", default="", help="model to use if the provider declines --model's requests")
    ap.add_argument("--effort", default="low", choices=("low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--no-fallbacks", action="store_true")
    ap.add_argument("--no-schema", action="store_true", help="ask for JSON in the prompt only")
    ap.add_argument("--no-probe", action="store_true", help="use --model as given without probing")
    ap.add_argument("--workers", type=int, default=8, help="sessions generated in parallel")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def cand(model: str, schema: bool) -> dict:
        return {"model": model, "schema": schema, "fallbacks": not args.no_fallbacks and supports_fallbacks(model)}

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path, cfg_path = out / "sessions.jsonl", out / "config.json"
    client = anthropic.Anthropic(max_retries=6)
    if cfg_path.exists():
        chosen = json.loads(cfg_path.read_text())["chosen"]
        print(f"using the configuration saved in {cfg_path}: {chosen}")
    elif args.no_probe:
        chosen = cand(args.model, not args.no_schema)
    else:
        candidates = ([cand(args.model, True)] if not args.no_schema else []) + [cand(args.model, False)]
        if args.backup_model:
            candidates.append(cand(args.backup_model, True))
        chosen, report = probe(client, candidates, args.effort)
        if chosen is None:
            print("STOPPED: the provider did not accept any request configuration; no sessions were generated. "
                  "The probe lines above give the refusal categories and errors.")
            sys.exit(2)
        cfg_path.write_text(json.dumps({"chosen": chosen, "probe": report}, indent=1))
    if chosen["model"] != args.model:
        print(f"NOTE: sessions are generated with the backup model {chosen['model']}, not {args.model}")
    counts = G.run_many(lambda: ClaudePolicy(client, chosen["model"], args.effort, chosen["fallbacks"], chosen["schema"]),
                        range(args.seed_start, args.seed_start + args.n), path, args.workers,
                        {"agent_model": chosen["model"], "provider": "anthropic", "effort": args.effort,
                         "structured_output": chosen["schema"]})
    print(f"done: {counts['written']} sessions written this run, {counts['failed']} failed (rerun to retry); file {path}")
    print("usage:", G.usage_report(path))
    if counts["stopped"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
