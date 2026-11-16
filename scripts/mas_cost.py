"""Estimate what generating N sessions will cost, from the measured token usage of sessions already generated.

    python scripts/mas_cost.py runs_trial/mas_claude/sessions.jsonl --n 600 --in-price 4 --out-price 20 \
        --cache-read-price 0.2 --cache-write-price 5

Prices are in USD per million tokens and are yours to supply: check the provider's pricing page.
The estimate scales the mean usage per session; with a handful of trial sessions it is rough.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformalguard.mas import convert as C   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions")
    ap.add_argument("--n", type=int, required=True, help="number of sessions planned")
    ap.add_argument("--in-price", type=float, required=True)
    ap.add_argument("--out-price", type=float, required=True)
    ap.add_argument("--cache-read-price", type=float, default=None, help="default: the input price")
    ap.add_argument("--cache-write-price", type=float, default=None, help="default: the input price")
    args = ap.parse_args()
    sessions = C.load_sessions(args.sessions)
    if not sessions:
        print("no sessions to measure"); return
    k = len(sessions)
    mean = lambda key: sum(int((s.get("usage") or {}).get(key, 0)) for s in sessions) / k
    price = {"input_tokens": args.in_price, "output_tokens": args.out_price,
             "cache_read_input_tokens": args.in_price if args.cache_read_price is None else args.cache_read_price,
             "cache_creation_input_tokens": args.in_price if args.cache_write_price is None else args.cache_write_price}
    per = sum(mean(key) * p / 1e6 for key, p in price.items())
    steps = sum(sum(e["type"] == "action" for e in s["events"]) for s in sessions) / k
    print(f"measured on {k} sessions ({sessions[0].get('agent_model', 'unknown')}): {steps:.1f} agent steps, "
          + ", ".join(f"{key} {mean(key):,.0f}" for key in price if mean(key)) + " per session")
    print(f"estimated cost: ${per:.3f} per session, about ${per * args.n:,.0f} for {args.n} sessions "
          f"(rough: based on {k} sessions; the probe adds a few sessions' worth)")


if __name__ == "__main__":
    main()
