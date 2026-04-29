"""Heterogeneous Continuous-Time Dynamic Execution Graph.

The graph is the central object in ConformalGuard. We follow the paper's
formalization (Sec. 3.1):

    G_t = (V_t, E_t, X_t, T_t, R_t)

where
    V_t      vertices observed up to time t,
    E_t      directed timestamped edges,
    X_t      heterogeneous node features,
    T_t      timestamps in continuous time,
    R_t      relation/edge type labels.

There are FOUR node types and SIX edge types as defined in the paper.
The implementation is intentionally framework-agnostic — `GraphBuilder` and
the framework hooks in `instrumentation.py` translate AutoGen / MetaGPT /
LangGraph / CrewAI traces into this representation.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class NodeType(str, Enum):
    """Four heterogeneous node types in a multi-agent execution graph."""

    AGENT_INVOCATION = "agent_invocation"
    TOOL_CALL = "tool_call"
    MEMORY_ENTRY = "memory_entry"
    MESSAGE = "message"


class EdgeType(str, Enum):
    """Six relation types describing inter-node interactions."""

    INVOKES = "invokes"
    READS = "reads"
    WRITES = "writes"
    DELEGATES = "delegates"
    REPLIES = "replies"
    CITES = "cites"


@dataclass
class Node:
    node_id: str
    node_type: NodeType
    timestamp: float
    features: np.ndarray | None = None  # raw heterogeneous attributes
    payload: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new(node_type: NodeType, payload: dict[str, Any] | None = None) -> "Node":
        return Node(
            node_id=str(uuid.uuid4()),
            node_type=node_type,
            timestamp=time.time(),
            payload=payload or {},
        )


@dataclass
class Edge:
    src: str
    dst: str
    edge_type: EdgeType
    timestamp: float
    weight: float = 1.0
    payload: dict[str, Any] = field(default_factory=dict)


class ExecutionGraph:
    """Append-only heterogeneous CT-DG.

    The graph is materialized incrementally as the multi-agent system runs.
    Subgraph snapshots `subgraph_at(t)` are used for both training and the
    online conformal prediction step.
    """

    # Allowed (src_type, edge_type, dst_type) triples — the schema the encoder
    # expects. Anything not in this table is dropped and counted as malformed.
    SCHEMA: set[tuple[NodeType, EdgeType, NodeType]] = {
        (NodeType.AGENT_INVOCATION, EdgeType.INVOKES, NodeType.TOOL_CALL),
        (NodeType.AGENT_INVOCATION, EdgeType.DELEGATES, NodeType.AGENT_INVOCATION),
        (NodeType.AGENT_INVOCATION, EdgeType.WRITES, NodeType.MEMORY_ENTRY),
        (NodeType.AGENT_INVOCATION, EdgeType.READS, NodeType.MEMORY_ENTRY),
        (NodeType.AGENT_INVOCATION, EdgeType.REPLIES, NodeType.MESSAGE),
        (NodeType.AGENT_INVOCATION, EdgeType.CITES, NodeType.MEMORY_ENTRY),
        (NodeType.MESSAGE, EdgeType.REPLIES, NodeType.MESSAGE),
        (NodeType.TOOL_CALL, EdgeType.WRITES, NodeType.MEMORY_ENTRY),
    }

    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._adj: dict[str, list[int]] = {}  # node_id -> indices in self.edges
        self._malformed: int = 0

    # -- mutation -----------------------------------------------------------

    def add_node(self, node: Node) -> str:
        if node.node_id in self.nodes:
            return node.node_id
        self.nodes[node.node_id] = node
        self._adj.setdefault(node.node_id, [])
        return node.node_id

    def add_edge(self, edge: Edge) -> bool:
        if edge.src not in self.nodes or edge.dst not in self.nodes:
            self._malformed += 1
            return False
        triple = (
            self.nodes[edge.src].node_type,
            edge.edge_type,
            self.nodes[edge.dst].node_type,
        )
        if triple not in self.SCHEMA:
            self._malformed += 1
            return False
        self.edges.append(edge)
        idx = len(self.edges) - 1
        self._adj[edge.src].append(idx)
        self._adj[edge.dst].append(idx)
        return True

    # -- views --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def subgraph_at(self, t: float) -> "ExecutionGraph":
        """Causal subgraph: every node and edge with timestamp ≤ t.

        This is the object used as input to the encoder at step t — the paper's
        G_<t. The returned graph shares no mutable state with self.
        """
        sub = ExecutionGraph(trace_id=f"{self.trace_id}@{t:.6f}")
        for nid, node in self.nodes.items():
            if node.timestamp <= t:
                sub.add_node(node)
        for e in self.edges:
            if e.timestamp <= t and e.src in sub.nodes and e.dst in sub.nodes:
                sub.add_edge(e)
        return sub

    def step_timestamps(self) -> list[float]:
        """Sorted unique timestamps of agent-invocation events."""
        ts = sorted({n.timestamp for n in self.nodes.values()
                     if n.node_type == NodeType.AGENT_INVOCATION})
        return ts

    def neighbors(self, node_id: str) -> list[Edge]:
        return [self.edges[i] for i in self._adj.get(node_id, [])]

    # -- (de)serialization -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "nodes": [
                {
                    "id": n.node_id,
                    "type": n.node_type.value,
                    "t": n.timestamp,
                    "payload": n.payload,
                } for n in self.nodes.values()
            ],
            "edges": [
                {
                    "src": e.src,
                    "dst": e.dst,
                    "type": e.edge_type.value,
                    "t": e.timestamp,
                    "w": e.weight,
                    "payload": e.payload,
                } for e in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExecutionGraph":
        g = cls(trace_id=d.get("trace_id"))
        for n in d["nodes"]:
            g.add_node(Node(
                node_id=n["id"],
                node_type=NodeType(n["type"]),
                timestamp=float(n["t"]),
                payload=n.get("payload", {}),
            ))
        for e in d["edges"]:
            g.add_edge(Edge(
                src=e["src"],
                dst=e["dst"],
                edge_type=EdgeType(e["type"]),
                timestamp=float(e["t"]),
                weight=float(e.get("w", 1.0)),
                payload=e.get("payload", {}),
            ))
        return g

    def stats(self) -> dict[str, int]:
        nt_count = {nt.value: 0 for nt in NodeType}
        et_count = {et.value: 0 for et in EdgeType}
        for n in self.nodes.values():
            nt_count[n.node_type.value] += 1
        for e in self.edges:
            et_count[e.edge_type.value] += 1
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "malformed": self._malformed,
            **nt_count,
            **et_count,
        }
