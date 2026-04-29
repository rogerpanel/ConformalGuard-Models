"""Notebook (script form): explore AgentChain-26 marginal statistics.

Usage:  python notebooks/01_explore_dataset.py
"""

from __future__ import annotations

from collections import Counter

from conformalguard.data.agentchain26 import AgentChain26
from conformalguard.violations.classes import id_to_label


def main():
    n = 1000
    traces = AgentChain26(split="train").materialize(n)
    print(f"|train sample| = {len(traces)}")

    lengths = [t.graph.stats()["n_nodes"] for t in traces]
    labels = Counter(t.label for t in traces)
    platforms = Counter(t.platform for t in traces)

    print(f"avg trace size  : {sum(lengths)/len(lengths):.1f} nodes")
    print(f"min / max size  : {min(lengths)} / {max(lengths)}")
    print(f"label histogram :")
    for k, v in sorted(labels.items()):
        print(f"  {k:2d} {id_to_label(k).name:30s} {v:5d}")
    print(f"platform mix    : {dict(platforms)}")


if __name__ == "__main__":
    main()
