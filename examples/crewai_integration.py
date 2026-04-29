"""CrewAI integration example."""

from __future__ import annotations

import os

from conformalguard.deployment.guard import ConformalGuardRuntime
from conformalguard.deployment.routing import AnalystRouter
from conformalguard.deployment.stream_processor import StreamProcessor
from conformalguard.graph.instrumentation import CrewAIHook


def main():
    ckpt = os.environ.get("CONFORMALGUARD_CKPT", "runs/conformalguard/model.pt")
    if not os.path.exists(ckpt):
        print(f"Missing checkpoint at {ckpt}.")
        return

    runtime = ConformalGuardRuntime.load(ckpt)
    proc = StreamProcessor(runtime, AnalystRouter())
    CrewAIHook(builder=proc.builder).attach()

    try:
        from crewai import Agent, Crew, Task
    except ImportError:
        print("Install crewai to run this example.")
        return

    a = Agent(role="researcher", goal="find facts", backstory="careful")
    t = Task(description="What is 2+2?", agent=a, expected_output="4")
    Crew(agents=[a], tasks=[t]).kickoff()

    rec = proc.step(step_t=1e18, proposed_action=0)
    print("verdict:", rec.decision.verdict.value,
          "width:", rec.decision.prediction_set.width)


if __name__ == "__main__":
    main()
