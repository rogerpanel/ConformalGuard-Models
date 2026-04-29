"""AgentChain-26 dataset.

AgentChain-26 (paper Table 1) consists of 12,400 multi-agent execution traces
collected across AutoGen, MetaGPT, LangGraph and CrewAI, manually-labeled
with one of the 12 categories (`safe` + 11 critical violation classes).

A curated release will be hosted at:
    https://github.com/rogerpanel/CV/releases/tag/agentchain26-v1
    DOI: 10.5281/zenodo.<TBA>

If the curated archive is not present locally, this loader transparently
falls back to the deterministic synthetic generator
(`SyntheticTraceGenerator`) which reproduces the marginal statistics of the
benchmark (length distribution, platform mix, violation rate).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from conformalguard.data.synthetic import SyntheticConfig, SyntheticTraceGenerator
from conformalguard.graph.execution_graph import ExecutionGraph


@dataclass
class AgentChainTrace:
    trace_id: str
    graph: ExecutionGraph
    label: int
    platform: str = "synthetic"

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "label": self.label,
            "platform": self.platform,
            "graph": self.graph.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentChainTrace":
        return cls(
            trace_id=d["trace_id"],
            label=int(d["label"]),
            platform=d.get("platform", "unknown"),
            graph=ExecutionGraph.from_dict(d["graph"]),
        )


_DEFAULT_ROOT = Path(os.environ.get(
    "CONFORMALGUARD_DATA",
    Path.home() / ".cache" / "conformalguard" / "agentchain26",
))


class AgentChain26:
    """Loader / iterator over the AgentChain-26 corpus."""

    def __init__(self, root: str | Path | None = None,
                 split: str = "train", synthetic_if_missing: bool = True,
                 seed: int = 1234):
        self.root = Path(root or _DEFAULT_ROOT)
        self.split = split
        self.synthetic_if_missing = synthetic_if_missing
        self.seed = seed
        self._manifest: list[dict] | None = None

    def manifest_path(self) -> Path:
        return self.root / f"{self.split}.jsonl"

    def is_present(self) -> bool:
        return self.manifest_path().exists()

    def __iter__(self) -> Iterator[AgentChainTrace]:
        if self.is_present():
            yield from self._iter_from_disk()
        elif self.synthetic_if_missing:
            yield from self._iter_synthetic()
        else:
            raise FileNotFoundError(
                f"AgentChain-26 not found at {self.root}. "
                "Run scripts/download_datasets.py or set synthetic_if_missing=True."
            )

    # -- backends ---------------------------------------------------------

    def _iter_from_disk(self) -> Iterator[AgentChainTrace]:
        with self.manifest_path().open() as f:
            for line in f:
                if line.strip():
                    yield AgentChainTrace.from_dict(json.loads(line))

    def _iter_synthetic(self) -> Iterator[AgentChainTrace]:
        # Splits: 80/10/10 train/val/test, deterministic.
        cfg = SyntheticConfig(seed=self.seed)
        n = cfg.n_traces
        cuts = {"train": (0, int(0.8 * n)),
                "val":   (int(0.8 * n), int(0.9 * n)),
                "test":  (int(0.9 * n), n)}
        lo, hi = cuts.get(self.split, (0, n))
        gen = SyntheticTraceGenerator(cfg)
        for i, (graph, label) in enumerate(gen.generate_dataset()):
            if lo <= i < hi:
                platform = graph.trace_id.split("-")[0] if graph.trace_id else "synthetic"
                yield AgentChainTrace(
                    trace_id=graph.trace_id or f"syn-{i:06d}",
                    graph=graph,
                    label=label,
                    platform=platform,
                )

    # -- materialization --------------------------------------------------

    def materialize(self, n: int | None = None) -> list[AgentChainTrace]:
        out: list[AgentChainTrace] = []
        for i, t in enumerate(self):
            if n is not None and i >= n:
                break
            out.append(t)
        return out

    def save_jsonl(self, path: str | Path, n: int | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for i, trace in enumerate(self):
                if n is not None and i >= n:
                    break
                f.write(json.dumps(trace.to_dict()) + "\n")
