"""Materialize AgentChain-26 (curated archive if available, synthetic fallback).

Usage:
    python scripts/download_datasets.py [--root PATH] [--n N]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from conformalguard.data.agentchain26 import AgentChain26
from conformalguard.data.external import EXTERNAL_DATASETS
from conformalguard.utils.logging import get_logger


def main():
    log = get_logger("conformalguard.download")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get(
        "CONFORMALGUARD_DATA",
        str(Path.home() / ".cache" / "conformalguard" / "agentchain26")))
    ap.add_argument("--n_train", type=int, default=9920)
    ap.add_argument("--n_val",   type=int, default=1240)
    ap.add_argument("--n_test",  type=int, default=1240)
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    log.info(f"Materializing AgentChain-26 to {root}")

    for split, n in [("train", args.n_train),
                      ("val",   args.n_val),
                      ("test",  args.n_test)]:
        out = root / f"{split}.jsonl"
        log.info(f"  -> {split}: {n} traces -> {out}")
        AgentChain26(root=root, split=split, synthetic_if_missing=True).save_jsonl(out, n=n)

    log.info("Done. External datasets you may also download separately:")
    for d in EXTERNAL_DATASETS:
        log.info(f"  {d.name:18s}  {d.url}")


if __name__ == "__main__":
    main()
