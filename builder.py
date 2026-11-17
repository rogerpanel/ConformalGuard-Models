"""Real-time graph builder.

`GraphBuilder` accepts a stream of generic events emitted by an instrumentation
hook (AutoGen / MetaGPT / LangGraph / CrewAI / custom) and incrementally
materializes an `ExecutionGraph`. Events are typed `dict`s with the minimal
fields below; the hook adapter is responsible for normalizing whatever the
underlying framework emits.

Required event schema
---------------------
{
    "kind":       one of {"agent", "tool", "memory", "message",
                          "invoke", "delegate", "read", "write",
                          "reply", "cite"},
    "id":         stable identifier (string),
    "src":        for edge events — source node id,
    "dst":        for edge events — destination node id,
    "t":          float, seconds; if absent, time.time() is used,
    "payload":    optional dict with framework-specific data.
}
"""

from __future__ import annotations

import time
from typing import Any, Iterable

from conformalguard.graph.execution_graph import (
    Edge,
    EdgeType,
    ExecutionGraph,
    Node,
    NodeType,
)


_NODE_KIND_MAP = {
    "agent":   NodeType.AGENT_INVOCATION,
    "tool":    NodeType.TOOL_CALL,
    "memory":  NodeType.MEMORY_ENTRY,
    "message": NodeType.MESSAGE,
}

_EDGE_KIND_MAP = {
    "invoke":   EdgeType.INVOKES,
    "delegate": EdgeType.DELEGATES,
    "read":     EdgeType.READS,
    "write":    EdgeType.WRITES,
    "reply":    EdgeType.REPLIES,
    "cite":     EdgeType.CITES,
}


class GraphBuilder:
    """Incrementally consume hook events into an `ExecutionGraph`."""

    def __init__(self, trace_id: str | None = None):
        self.graph = ExecutionGraph(trace_id=trace_id)

    def ingest(self, event: dict[str, Any]) -> None:
        kind = event["kind"]
        ts = float(event.get("t", time.time()))
        if kind in _NODE_KIND_MAP:
            self.graph.add_node(Node(
                node_id=str(event["id"]),
                node_type=_NODE_KIND_MAP[kind],
                timestamp=ts,
                payload=event.get("payload", {}),
            ))
        elif kind in _EDGE_KIND_MAP:
            self.graph.add_edge(Edge(
                src=str(event["src"]),
                dst=str(event["dst"]),
                edge_type=_EDGE_KIND_MAP[kind],
                timestamp=ts,
                weight=float(event.get("weight", 1.0)),
                payload=event.get("payload", {}),
            ))
        else:
            raise ValueError(f"Unknown event kind: {kind!r}")

    def ingest_many(self, events: Iterable[dict[str, Any]]) -> None:
        for e in events:
            self.ingest(e)

    def finalize(self) -> ExecutionGraph:
        return self.graph
