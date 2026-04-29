"""MetaGPT integration example."""

from __future__ import annotations

import asyncio
import os

from conformalguard.deployment.guard import ConformalGuardRuntime
from conformalguard.deployment.routing import AnalystRouter
from conformalguard.deployment.stream_processor import StreamProcessor
from conformalguard.graph.instrumentation import MetaGPTHook


async def main():
    ckpt = os.environ.get("CONFORMALGUARD_CKPT", "runs/conformalguard/model.pt")
    if not os.path.exists(ckpt):
        print(f"Missing checkpoint at {ckpt}.")
        return

    runtime = ConformalGuardRuntime.load(ckpt)
    proc = StreamProcessor(runtime, AnalystRouter())
    MetaGPTHook(builder=proc.builder).attach()

    try:
        from metagpt.actions import Action
    except ImportError:
        print("Install metagpt to run this example.")
        return

    class WriteSpec(Action):  # pragma: no cover — MetaGPT-specific
        async def run(self, requirement: str):
            return f"spec for {requirement}"

    await WriteSpec().run("hello world")
    rec = proc.step(step_t=1e18, proposed_action=0)
    print("verdict:", rec.decision.verdict.value)


if __name__ == "__main__":
    asyncio.run(main())
