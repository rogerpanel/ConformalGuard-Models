"""Tests for the heterogeneous CT-DG."""

from __future__ import annotations

import time

import pytest

from conformalguard.graph.builder import GraphBuilder
from conformalguard.graph.execution_graph import (
    Edge,
    EdgeType,
    ExecutionGraph,
    Node,
    NodeType,
)


def test_schema_enforced():
    g = ExecutionGraph()
    g.add_node(Node(node_id="a", node_type=NodeType.AGENT_INVOCATION, timestamp=0.0))
    g.add_node(Node(node_id="t", node_type=NodeType.TOOL_CALL, timestamp=0.0))
    assert g.add_edge(Edge("a", "t", EdgeType.INVOKES, 0.0))
    # Bogus triple → rejected.
    assert not g.add_edge(Edge("a", "t", EdgeType.READS, 0.0))
    assert g.stats()["malformed"] == 1


def test_subgraph_at():
    b = GraphBuilder()
    t0 = time.time()
    b.ingest({"kind": "agent", "id": "a", "t": t0})
    b.ingest({"kind": "tool",  "id": "t", "t": t0 + 1})
    b.ingest({"kind": "invoke", "src": "a", "dst": "t", "t": t0 + 2})
    g = b.finalize()
    assert len(g) == 2
    assert g.num_edges == 1
    sub = g.subgraph_at(t0 + 0.5)
    assert len(sub) == 1                 # only the agent
    assert sub.num_edges == 0


def test_serialization_roundtrip():
    b = GraphBuilder()
    b.ingest({"kind": "agent",   "id": "a",  "t": 0.0})
    b.ingest({"kind": "memory",  "id": "m",  "t": 0.1})
    b.ingest({"kind": "write",   "src": "a", "dst": "m", "t": 0.2})
    g = b.finalize()
    g2 = ExecutionGraph.from_dict(g.to_dict())
    assert g2.stats()["n_nodes"] == g.stats()["n_nodes"]
    assert g2.stats()["n_edges"] == g.stats()["n_edges"]


def test_unknown_event_kind_raises():
    b = GraphBuilder()
    with pytest.raises(ValueError):
        b.ingest({"kind": "bogus", "id": "x"})
