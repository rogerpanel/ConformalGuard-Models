"""Synthetic AgentChain-26 generator.

When the curated AgentChain-26 release is not present locally, we fall back
to a procedural generator that reproduces the same trace-level statistics
reported in Table 1 of the paper:

    n_traces        : 12,400
    platform mix    : AutoGen 30 % / MetaGPT 30 % / LangGraph 25 % / CrewAI 15 %
    trace length    : 5–50 steps,   mean ≈ 19.4
    violation rate  : 11 critical classes, ~7 % overall

The generator is deterministic given a seed, so calibration results match
across machines for unit-test purposes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from conformalguard.graph.builder import GraphBuilder
from conformalguard.graph.execution_graph import ExecutionGraph
from conformalguard.violations.classes import VIOLATION_CLASSES


PLATFORMS = [
    ("autogen",  0.30),
    ("metagpt",  0.30),
    ("langgraph", 0.25),
    ("crewai",   0.15),
]


@dataclass
class SyntheticConfig:
    n_traces: int = 12_400
    min_steps: int = 5
    max_steps: int = 50
    target_mean_steps: float = 19.4
    violation_rate: float = 0.07
    seed: int = 1234


class SyntheticTraceGenerator:
    def __init__(self, cfg: SyntheticConfig | None = None):
        self.cfg = cfg or SyntheticConfig()
        self._rng = random.Random(self.cfg.seed)
        self._np = np.random.default_rng(self.cfg.seed)

    # -- per-trace generation ---------------------------------------------

    def _sample_platform(self) -> str:
        x = self._rng.random()
        cum = 0.0
        for name, p in PLATFORMS:
            cum += p
            if x < cum:
                return name
        return PLATFORMS[-1][0]

    def _sample_length(self) -> int:
        # Negative binomial fit to mean ≈ 19.4 inside [5, 50].
        x = int(self._np.poisson(self.cfg.target_mean_steps))
        return max(self.cfg.min_steps, min(self.cfg.max_steps, x))

    def generate_trace(self, trace_id: str | None = None) -> tuple[ExecutionGraph, int]:
        b = GraphBuilder(trace_id=trace_id)
        platform = self._sample_platform()
        n_steps = self._sample_length()
        is_violation = self._rng.random() < self.cfg.violation_rate

        coordinator = f"{platform}/coordinator"
        b.ingest({"kind": "agent", "id": coordinator, "t": 0.0,
                  "payload": {"role": "coordinator", "platform": platform}})

        last_msg_id: str | None = None
        t = 0.0
        for step in range(n_steps):
            t += 0.05 + self._rng.random() * 0.2
            agent_id = f"{platform}/agent-{step % 4}"
            b.ingest({"kind": "agent", "id": agent_id, "t": t,
                      "payload": {"step": step}})

            # tool call
            tool_id = f"tool/{platform}-{self._rng.choice(['search','code','db','http','vector'])}-{step}"
            b.ingest({"kind": "tool", "id": tool_id, "t": t,
                      "payload": {"args_len": self._rng.randint(8, 256)}})
            b.ingest({"kind": "invoke", "src": agent_id, "dst": tool_id, "t": t})

            # memory read/write
            mem_id = f"mem/{step}"
            b.ingest({"kind": "memory", "id": mem_id, "t": t,
                      "payload": {"chars": self._rng.randint(64, 1024)}})
            b.ingest({"kind": "write", "src": agent_id, "dst": mem_id, "t": t})
            if step > 0:
                b.ingest({"kind": "read", "src": agent_id, "dst": f"mem/{step-1}", "t": t})

            # message
            msg_id = f"msg/{step}"
            b.ingest({"kind": "message", "id": msg_id, "t": t,
                      "payload": {"sender": agent_id}})
            b.ingest({"kind": "reply", "src": agent_id, "dst": msg_id, "t": t})
            if last_msg_id is not None:
                b.ingest({"kind": "reply", "src": msg_id, "dst": last_msg_id, "t": t})
            last_msg_id = msg_id

            # delegation between agents
            if step % 3 == 2:
                next_agent = f"{platform}/agent-{(step + 1) % 4}"
                b.ingest({"kind": "agent", "id": next_agent, "t": t})
                b.ingest({"kind": "delegate", "src": agent_id, "dst": next_agent, "t": t})

        # Choose label.
        if is_violation:
            # uniformly across the 11 critical classes
            label = self._rng.randint(1, len(VIOLATION_CLASSES) - 1)
        else:
            label = 0

        graph = b.finalize()
        return graph, label

    # -- batch generation -------------------------------------------------

    def generate_dataset(self):
        for i in range(self.cfg.n_traces):
            yield self.generate_trace(trace_id=f"syn-{i:06d}")
