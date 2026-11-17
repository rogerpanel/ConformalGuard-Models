"""Provider-independent parts of session generation: reply parsing and the session runner.

The calls to a model live in scripts/mas_generate_claude.py and
scripts/mas_generate_openai.py (one SDK per file).
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from conformalguard.mas.sim import run_session
from conformalguard.mas.world import make_world

MIN_STEPS_BEFORE_STOP, MAX_REFUSED_SHARE = 20, 0.25      # stop generating if the provider refuses this much
DECLINED = {"kind": "finish", "tool": "", "args": {}, "to": "", "content": "(declined)"}


def parse_action(text: str | None) -> dict:
    """The JSON object in a model reply; a reply without one ends the agent's part."""
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                if isinstance(obj.get("args"), dict):
                    obj["args"] = {k: v for k, v in obj["args"].items() if v not in ("", None)}
                return obj
        except ValueError:
            pass
    return {"kind": "finish", "tool": "", "args": {}, "to": "", "content": text[:500]}


def done_seeds(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {json.loads(line)["seed"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def run_many(make_policy, seeds, path: Path, workers: int, extra: dict, guard=None) -> dict:
    """Generate one session per seed with ``make_policy()`` (a fresh policy per session, so its
    counters belong to that session) and append each finished session to ``path``.
    A session whose model calls fail is not written; rerun to retry it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seeds if s not in done_seeds(path)]
    lock = threading.Lock()
    counts = {"written": 0, "failed": 0, "calls": 0, "refused": 0, "stopped": False}
    print(f"{len(seeds) - len(todo)} sessions already in {path}, {len(todo)} to go", flush=True)

    def one(seed: int) -> None:
        if counts["stopped"]:
            return
        policy = make_policy()
        try:
            session = run_session(make_world(seed), policy, guard=guard)
        except Exception as e:                          # the policy re-raises API errors after its retries
            with lock:
                counts["failed"] += 1
                print(f"session {seed} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
            return
        session.update(extra, usage=dict(getattr(policy, "usage", {})), refusals=int(getattr(policy, "refusals", 0)),
                       refusal_categories=dict(getattr(policy, "refusal_categories", {})),
                       served_by=sorted(getattr(policy, "served_by", set())))
        with lock:
            counts["calls"] += sum(e["type"] == "action" for e in session["events"])
            counts["refused"] += session["refusals"]
            if counts["calls"] >= MIN_STEPS_BEFORE_STOP and counts["refused"] > MAX_REFUSED_SHARE * counts["calls"]:
                if not counts["stopped"]:
                    print(f"STOPPED: {counts['refused']} of {counts['calls']} agent steps were refused by the provider "
                          f"({session['refusal_categories'] or 'no category given'}). Sessions like that are not usable, "
                          f"so no further sessions are generated.", flush=True)
                counts["stopped"] = True
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(session, ensure_ascii=False) + "\n")
            counts["written"] += 1
            if counts["written"] % 20 == 0:
                print(f"  {counts['written']} sessions written", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
    return counts


def usage_report(path: Path) -> str:
    """Token totals over the sessions in a file, for cost estimates."""
    total: dict[str, int] = {}
    n = refusals = 0
    cats: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        if line.strip():
            s = json.loads(line)
            n += 1
            refusals += int(s.get("refusals", 0))
            for k, v in (s.get("refusal_categories") or {}).items():
                cats[k] = cats.get(k, 0) + int(v)
            for k, v in (s.get("usage") or {}).items():
                total[k] = total.get(k, 0) + int(v)
    if not n:
        return "no sessions yet"
    per = ", ".join(f"{k} {v / n:,.0f}" for k, v in total.items())
    return f"{n} sessions; per session: {per}; refused steps in total: {refusals}" + (f" {cats}" if cats else "")
