"""Analyst review-queue routing.

When ConformalGuard returns `verdict = REVIEW`, the trace is added to the SOC
analyst queue used by RobustIDPS.ai. The routing policy is a small RL
controller (Sec. 6.2 of the paper); for cold-start deployments we use a
priority heuristic combining:

    priority = w1 · severity + w2 · prediction_set_width + w3 · drift_score

with weights {0.6, 0.2, 0.2}. Items above `priority_threshold` are escalated
to the on-call queue; everything else accumulates in the review backlog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from conformalguard.conformal.prediction_set import PredictionSet
from conformalguard.violations.classes import id_to_label


class RoutingDecision(str, Enum):
    AUTO_BLOCK = "auto_block"
    ESCALATE_ONCALL = "escalate_oncall"
    REVIEW_QUEUE = "review_queue"
    ALLOW = "allow"


_SEVERITY_WEIGHT = {"low": 0.0, "medium": 0.4, "high": 0.7, "critical": 1.0}


@dataclass
class QueueItem:
    trace_id: str
    step_t: float
    pset: PredictionSet
    severity: str
    drift: float
    priority: float
    decision: RoutingDecision


@dataclass
class AnalystRouter:
    w_severity: float = 0.6
    w_width: float = 0.2
    w_drift: float = 0.2
    priority_escalate: float = 0.7
    priority_block: float = 0.9
    queue: List[QueueItem] = field(default_factory=list)

    def route(self, trace_id: str, step_t: float, pset: PredictionSet,
              top_violation_id: int, drift: float = 0.0) -> QueueItem:
        sev = id_to_label(top_violation_id).severity
        sev_w = _SEVERITY_WEIGHT.get(sev, 0.5)
        # Width is normalized — a width of 1 (single safe action) is best.
        width_norm = min(1.0, max(0.0, (pset.width - 1) / 10.0))
        priority = (
            self.w_severity * sev_w
            + self.w_width * width_norm
            + self.w_drift * min(1.0, drift)
        )

        if pset.verdict.value == "block" and priority >= self.priority_block:
            decision = RoutingDecision.AUTO_BLOCK
        elif priority >= self.priority_escalate:
            decision = RoutingDecision.ESCALATE_ONCALL
        elif pset.verdict.value == "review":
            decision = RoutingDecision.REVIEW_QUEUE
        else:
            decision = RoutingDecision.ALLOW

        item = QueueItem(
            trace_id=trace_id,
            step_t=step_t,
            pset=pset,
            severity=sev,
            drift=drift,
            priority=priority,
            decision=decision,
        )
        self.queue.append(item)
        return item

    def pending(self, decision: RoutingDecision | None = None) -> list[QueueItem]:
        if decision is None:
            return list(self.queue)
        return [q for q in self.queue if q.decision == decision]
