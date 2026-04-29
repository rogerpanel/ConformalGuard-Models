"""End-to-end AutoGen integration example.

Wraps an AutoGen ConversableAgent pair, attaches the ConformalGuard hook,
and prints the verdict for each step.

Run:
    pip install pyautogen
    python examples/autogen_integration.py
"""

from __future__ import annotations

import os

from conformalguard.deployment.guard import ConformalGuardRuntime
from conformalguard.deployment.routing import AnalystRouter
from conformalguard.deployment.stream_processor import StreamProcessor
from conformalguard.graph.instrumentation import AutoGenHook


def main():
    ckpt = os.environ.get("CONFORMALGUARD_CKPT", "runs/conformalguard/model.pt")
    if not os.path.exists(ckpt):
        print(f"Missing checkpoint at {ckpt}. Train first via scripts/train.py.")
        return

    runtime = ConformalGuardRuntime.load(ckpt)
    router = AnalystRouter()
    proc = StreamProcessor(runtime, router)

    hook = AutoGenHook(builder=proc.builder)
    hook.attach()

    try:
        import autogen
    except ImportError:
        print("Install pyautogen to run this example.")
        return

    config_list = [{"model": "gpt-4o", "api_key": os.environ.get("OPENAI_API_KEY")}]
    coder = autogen.ConversableAgent(
        "coder", llm_config={"config_list": config_list},
        system_message="You write Python code."
    )
    critic = autogen.ConversableAgent(
        "critic", llm_config={"config_list": config_list},
        system_message="You critique code for safety issues."
    )

    coder.initiate_chat(critic, message="Write a hello-world script.", max_turns=2)

    # Step the runtime once at the end of the conversation.
    rec = proc.step(step_t=1e18, proposed_action=0, top_violation_id=0)
    print("verdict:", rec.decision.verdict.value,
          "width:", rec.decision.prediction_set.width,
          "latency_ms:", round(rec.decision.latency_ms, 2))


if __name__ == "__main__":
    main()
