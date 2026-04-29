"""LangGraph integration example."""

from __future__ import annotations

import os

from conformalguard.deployment.guard import ConformalGuardRuntime
from conformalguard.deployment.routing import AnalystRouter
from conformalguard.deployment.stream_processor import StreamProcessor
from conformalguard.graph.instrumentation import LangGraphHook


def main():
    ckpt = os.environ.get("CONFORMALGUARD_CKPT", "runs/conformalguard/model.pt")
    if not os.path.exists(ckpt):
        print(f"Missing checkpoint at {ckpt}.")
        return

    runtime = ConformalGuardRuntime.load(ckpt)
    proc = StreamProcessor(runtime, AnalystRouter())
    LangGraphHook(builder=proc.builder).attach()

    try:
        from langgraph.graph import StateGraph
    except ImportError:
        print("Install langgraph to run this example.")
        return

    g = StateGraph(dict)
    g.add_node("planner", lambda s: {"plan": "do x"})
    g.add_node("executor", lambda s: {"result": "ok"})
    g.add_edge("planner", "executor")
    g.set_entry_point("planner")
    g.set_finish_point("executor")
    app = g.compile()

    for chunk in app.stream({"input": "hi"}):
        print("stream chunk:", list(chunk.keys()))

    rec = proc.step(step_t=1e18, proposed_action=0)
    print("verdict:", rec.decision.verdict.value)


if __name__ == "__main__":
    main()
