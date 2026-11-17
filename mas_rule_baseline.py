"""How well does a hand-written provenance rule reproduce the labels of the multi-agent sessions?

    python scripts/mas_rule_baseline.py --sessions runs/mas_claude/sessions.jsonl runs/mas_openai/sessions.jsonl

The rule flags a step if its tool call names an external destination and cites a restricted
memory entry (conformalguard/mas/experiment.py, provenance_rule).  It needs no training.  Report
it next to the learned scorers: it shows how much of the task is decided by the provenance edges
and resource metadata that the graph scorer is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformalguard.mas import convert as C          # noqa: E402
from conformalguard.mas import experiment as X       # noqa: E402
from conformalguard.mas import labels as L           # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    c: dict = defaultdict(lambda: defaultdict(int))
    for i, s in enumerate(C.load_sessions(args.sessions)):
        if int(s.get("refusals", 0)):
            continue
        tr, meta, y, info = C.session_to_trace(s, i)
        fire = X.provenance_rule(tr, meta) > 0
        for group in ("all", str(s.get("agent_model", "unknown"))):
            g = c[group]
            g["sessions"] += 1
            g["violating_steps"] += int((y == L.VIOL).sum())
            g["violating_steps_flagged"] += int((fire & (y == L.VIOL)).sum())
            g["safe_steps"] += int((y == L.SAFE).sum())
            g["safe_steps_flagged"] += int((fire & (y == L.SAFE)).sum())
            clean = not (y == L.VIOL).any()
            g["violation_free_sessions"] += clean
            g["violation_free_sessions_flagged"] += bool(clean and fire[y == L.SAFE].any())
            for k in ((y == L.VIOL) & ~fire).nonzero()[0]:
                g[f"missed_{info[k]['kinds'][0]}"] += 1
    out = {k: dict(v) for k, v in c.items()}
    for k, g in out.items():
        print(f"{k}: {g['sessions']} sessions; rule flags {g['violating_steps_flagged']} of {g['violating_steps']} violating steps, "
              f"{g['safe_steps_flagged']} of {g['safe_steps']} safe steps, "
              f"{g['violation_free_sessions_flagged']} of {g['violation_free_sessions']} violation-free sessions"
              + "".join(f"; missed {m[7:]}: {n}" for m, n in g.items() if m.startswith("missed_")))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
