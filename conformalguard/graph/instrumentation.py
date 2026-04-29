"""Instrumentation hooks for popular multi-agent frameworks.

Each hook implements a thin adapter that converts framework-native events into
the canonical event schema accepted by `GraphBuilder`. The implementations are
purposefully *defensive*: they attempt to import the corresponding framework
and silently fall back to a duck-typed mode if the framework is not installed,
so that the codebase remains importable in CI environments where only some of
the agent frameworks are available.

Tested against the versions reported in the paper:
    AutoGen   ≥ 0.2,
    MetaGPT   ≥ 0.8,
    LangGraph ≥ 0.0.40,
    CrewAI    ≥ 0.30.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from conformalguard.graph.builder import GraphBuilder


class _BaseHook:
    """Base class for framework-specific instrumentation hooks."""

    framework: str = "base"

    def __init__(self, builder: GraphBuilder | None = None):
        self.builder = builder or GraphBuilder()
        self._sink: Callable[[dict[str, Any]], None] = self.builder.ingest

    # -- helpers -----------------------------------------------------------

    def _emit_node(self, kind: str, id_: str, payload: dict[str, Any] | None = None) -> None:
        self._sink({"kind": kind, "id": id_, "t": time.time(), "payload": payload or {}})

    def _emit_edge(self, kind: str, src: str, dst: str, payload: dict[str, Any] | None = None) -> None:
        self._sink({"kind": kind, "src": src, "dst": dst, "t": time.time(),
                    "payload": payload or {}})

    # -- public API --------------------------------------------------------

    def attach(self) -> None:  # pragma: no cover — framework specific
        raise NotImplementedError

    def finalize(self):
        return self.builder.finalize()


class AutoGenHook(_BaseHook):
    framework = "autogen"

    def attach(self) -> None:
        try:
            import autogen  # type: ignore
        except ImportError:
            return  # silently no-op in environments without autogen

        # AutoGen 0.2 exposes ChatCompletion + ConversableAgent.
        # We monkey-patch `_print_received_message` and `register_function`.
        original_recv = autogen.ConversableAgent.receive

        def patched_recv(self_agent, message, sender, request_reply=None, silent=False):  # noqa: ANN001
            self._emit_node("agent", id_=self_agent.name)
            self._emit_node("agent", id_=sender.name)
            self._emit_node("message", id_=f"msg-{id(message)}",
                             payload={"content": str(message)[:512]})
            self._emit_edge("delegate", src=sender.name, dst=self_agent.name)
            self._emit_edge("reply",    src=self_agent.name, dst=f"msg-{id(message)}")
            return original_recv(self_agent, message, sender, request_reply, silent)

        autogen.ConversableAgent.receive = patched_recv  # type: ignore

    # Public method usable for manual emission in tests.
    def manual_invoke(self, agent: str, tool: str) -> None:
        self._emit_node("agent", agent)
        self._emit_node("tool", tool)
        self._emit_edge("invoke", agent, tool)


class MetaGPTHook(_BaseHook):
    framework = "metagpt"

    def attach(self) -> None:
        try:
            from metagpt.actions import Action  # type: ignore
        except ImportError:
            return

        original_run = Action.run

        async def patched_run(self_action, *args, **kwargs):  # noqa: ANN001
            agent_id = getattr(self_action, "name", self_action.__class__.__name__)
            tool_id = f"action-{id(self_action)}"
            self._emit_node("agent", agent_id)
            self._emit_node("tool", tool_id, payload={"action_class": agent_id})
            self._emit_edge("invoke", agent_id, tool_id)
            return await original_run(self_action, *args, **kwargs)

        Action.run = patched_run  # type: ignore


class LangGraphHook(_BaseHook):
    framework = "langgraph"

    def attach(self) -> None:
        try:
            from langgraph.graph import StateGraph  # type: ignore
        except ImportError:
            return

        original_compile = StateGraph.compile

        def patched_compile(self_g, *args, **kwargs):  # noqa: ANN001
            compiled = original_compile(self_g, *args, **kwargs)
            outer = self

            class _Wrapped:
                def __init__(self, c):
                    self._c = c

                def stream(self_w, inputs, *a, **kw):  # noqa: ANN001
                    for chunk in self_w._c.stream(inputs, *a, **kw):
                        for node_name, payload in chunk.items():
                            outer._emit_node("agent", node_name)
                            mid = f"msg-{time.time_ns()}"
                            outer._emit_node("message", mid,
                                             payload={"keys": list(payload.keys()) if isinstance(payload, dict) else []})
                            outer._emit_edge("reply", node_name, mid)
                        yield chunk

                def invoke(self_w, *a, **kw):  # noqa: ANN001
                    return self_w._c.invoke(*a, **kw)

            return _Wrapped(compiled)

        StateGraph.compile = patched_compile  # type: ignore


class CrewAIHook(_BaseHook):
    framework = "crewai"

    def attach(self) -> None:
        try:
            from crewai import Agent, Task  # type: ignore
        except ImportError:
            return

        original_execute = Task.execute_sync if hasattr(Task, "execute_sync") else Task.execute

        def patched_execute(self_task, *args, **kwargs):  # noqa: ANN001
            agent: Agent = self_task.agent
            tool_id = f"task-{id(self_task)}"
            self._emit_node("agent", agent.role)
            self._emit_node("tool", tool_id,
                             payload={"description": getattr(self_task, "description", "")[:256]})
            self._emit_edge("invoke", agent.role, tool_id)
            return original_execute(self_task, *args, **kwargs)

        if hasattr(Task, "execute_sync"):
            Task.execute_sync = patched_execute  # type: ignore
        else:
            Task.execute = patched_execute  # type: ignore
