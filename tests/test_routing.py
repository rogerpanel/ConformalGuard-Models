"""Tests for analyst routing decisions."""

from __future__ import annotations

from conformalguard.conformal.prediction_set import PredictionSet, Verdict
from conformalguard.deployment.routing import AnalystRouter, RoutingDecision


def _pset(verdict: Verdict, width: int = 3) -> PredictionSet:
    return PredictionSet(
        candidates=list(range(width)),
        scores=[0.1] * width,
        threshold=1.0,
        verdict=verdict,
    )


def test_critical_block_escalates():
    r = AnalystRouter()
    item = r.route("trace1", 0.0,
                   _pset(Verdict.BLOCK, width=8),
                   top_violation_id=1,    # pii_exfiltration → critical
                   drift=0.5)
    assert item.decision == RoutingDecision.AUTO_BLOCK


def test_low_severity_review_goes_to_queue():
    r = AnalystRouter()
    item = r.route("t2", 1.0, _pset(Verdict.REVIEW, width=2),
                   top_violation_id=8, drift=0.1)  # high severity but width small
    assert item.decision in (RoutingDecision.REVIEW_QUEUE,
                              RoutingDecision.ESCALATE_ONCALL)


def test_allow_passes_through():
    r = AnalystRouter()
    item = r.route("t3", 0.0, _pset(Verdict.ALLOW, width=1),
                   top_violation_id=0, drift=0.0)
    assert item.decision == RoutingDecision.ALLOW
