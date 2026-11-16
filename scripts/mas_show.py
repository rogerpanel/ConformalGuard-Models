"""Print what is in one or more session files: counts, labels, and one session as a transcript.

    python scripts/mas_show.py runs/mas_claude/sessions.jsonl [--session 0]

Use it to read a few generated sessions before training on them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformalguard.mas import convert as C   # noqa: E402
from conformalguard.mas import labels as L    # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--session", type=int, default=None, help="index of the session to print (default: the first with a violation)")
    ap.add_argument("--width", type=int, default=150)
    args = ap.parse_args()
    sessions = C.load_sessions(args.sessions)
    print(json.dumps(C.summary(sessions), indent=1))
    if not sessions:
        return
    k = args.session
    if k is None:
        k = next((i for i, s in enumerate(sessions) if any(x["y"] == L.VIOL for x in L.label_session(s))), 0)
    s = sessions[k]
    labs = {x["event"]: x for x in L.label_session(s)}
    cut = lambda t: " ".join(str(t).split())[:args.width]
    print(f"\n--- session {k}: seed {s['seed']}, {s['n_threads']} task(s), agent model {s.get('agent_model', 'unknown')}, "
          f"usage {s.get('usage', {})}")
    for i, ev in enumerate(s["events"]):
        if ev["type"] == "user":
            print(f"[{ev['thread']}] USER  {cut(ev['text'])}")
        elif ev["type"] == "action":
            lab = labs[i]
            mark = {L.VIOL: "  <-- VIOLATION: " + ", ".join(lab["kinds"]), L.IGNORE: "  (after a violation: ignored)"}.get(lab["y"], "")
            what = f"{ev['tool']} {json.dumps(ev['args'], ensure_ascii=False)}" if ev["kind"] == "tool" else f"{ev['kind']}: {ev['content']}"
            print(f"[{ev['thread']}] {ev['agent']:<15} {cut(what)}{mark}")
        else:
            print(f"[{ev['thread']}]     -> {cut(ev['text'])}  {ev.get('meta') or ''}")


if __name__ == "__main__":
    main()
