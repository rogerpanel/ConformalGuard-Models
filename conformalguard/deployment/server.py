"""FastAPI server exposing ConformalGuard to RobustIDPS.ai.

Wired up to the `agent-layer` plug-in inside RobustIDPS — see
`integrated_ai_ids/plugins/conformalguard.py` (companion repo).

Endpoints
---------
POST /trace/event     ingest a single hook event
POST /trace/step      run guard.evaluate_step on the current trace
GET  /queue           list outstanding analyst-review items
GET  /healthz         liveness probe
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from conformalguard.deployment.guard import ConformalGuardRuntime
from conformalguard.deployment.routing import AnalystRouter
from conformalguard.deployment.stream_processor import StreamProcessor


app = FastAPI(title="ConformalGuard", version="1.0.0")
_runtime: ConformalGuardRuntime | None = None
_processor: StreamProcessor | None = None


class TraceEvent(BaseModel):
    kind: str
    id: str | None = None
    src: str | None = None
    dst: str | None = None
    t: float | None = None
    payload: dict | None = None


class StepRequest(BaseModel):
    step_t: float
    proposed_action: int
    top_violation_id: int = 0


def _bootstrap():
    global _runtime, _processor
    if _runtime is not None:
        return
    ckpt = os.environ.get("CONFORMALGUARD_CKPT", "runs/conformalguard/model.pt")
    if not Path(ckpt).exists():
        raise RuntimeError(f"checkpoint {ckpt} missing — set CONFORMALGUARD_CKPT")
    _runtime = ConformalGuardRuntime.load(ckpt)
    _processor = StreamProcessor(_runtime, AnalystRouter())


@app.post("/trace/event")
def trace_event(ev: TraceEvent):
    _bootstrap()
    payload = ev.dict(exclude_none=True)
    try:
        _processor.ingest(payload)  # type: ignore[union-attr]
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "n_nodes": len(_processor.builder.graph)}  # type: ignore[union-attr]


@app.post("/trace/step")
def trace_step(req: StepRequest):
    _bootstrap()
    rec = _processor.step(req.step_t, req.proposed_action, req.top_violation_id)  # type: ignore[union-attr]
    return {
        "verdict": rec.decision.verdict.value,
        "prediction_set": rec.decision.prediction_set.to_dict(),
        "latency_ms": rec.decision.latency_ms,
        "alpha_eff": rec.decision.alpha_eff,
        "drift": rec.decision.drift,
    }


@app.get("/queue")
def get_queue():
    _bootstrap()
    return [
        {
            "trace_id": q.trace_id,
            "step_t":   q.step_t,
            "decision": q.decision.value,
            "priority": q.priority,
            "severity": q.severity,
            "width":    q.pset.width,
        }
        for q in _processor.router.queue  # type: ignore[union-attr]
    ]


@app.get("/healthz")
def healthz():
    return {"ok": True, "version": app.version}
