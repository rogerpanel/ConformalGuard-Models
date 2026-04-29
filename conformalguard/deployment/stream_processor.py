"""High-level streaming wrapper.

Wraps `ConformalGuardRuntime` to operate on a *live* execution graph that is
being built incrementally by an instrumentation hook. The stream processor:

    1. Subscribes to events from the GraphBuilder.
    2. After every agent-invocation timestamp, runs `runtime.evaluate_step`.
    3. Forwards the verdict to the AnalystRouter.

This is the object the RobustIDPS API calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from conformalguard.deployment.guard import ConformalGuardRuntime, GuardDecision
from conformalguard.deployment.routing import AnalystRouter
from conformalguard.graph.builder import GraphBuilder


@dataclass
class StepRecord:
    step_t: float
    proposed_action: int
    decision: GuardDecision


class StreamProcessor:
    def __init__(self, runtime: ConformalGuardRuntime,
                 router: AnalystRouter | None = None):
        self.runtime = runtime
        self.router = router or AnalystRouter()
        self.builder = GraphBuilder()
        self._records: list[StepRecord] = []

    def ingest(self, event: dict) -> None:
        self.builder.ingest(event)

    def step(self, step_t: float, proposed_action: int,
             top_violation_id: int = 0) -> StepRecord:
        graph = self.builder.graph
        decision = self.runtime.evaluate_step(graph, step_t, proposed_action)
        self.router.route(
            trace_id=graph.trace_id,
            step_t=step_t,
            pset=decision.prediction_set,
            top_violation_id=top_violation_id,
            drift=decision.drift,
        )
        rec = StepRecord(step_t=step_t, proposed_action=proposed_action, decision=decision)
        self._records.append(rec)
        return rec

    def history(self) -> Iterable[StepRecord]:
        return list(self._records)
