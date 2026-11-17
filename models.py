"""Step scorers compared on R-Judge.

Every scorer maps a batch of traces to one logit per *decision step* (one per
agent message): the score of the proposed action given what the guard can see
when the action is proposed.  All are causal: the logit of step k never
depends on anything that happens after the action of step k is proposed.

``step``           the current step only (thought + proposed action).  No history.
``seq``            a Transformer with a causal mask over the nodes in time order.
                   Same node features, node-type and position embeddings as the
                   graph model; no edges.
``graph``          relation-aware attention (the HGT layer of the paper) over the
                   typed execution graph.
``graph_untyped``  the same graph and layers with a single node type and a single
                   relation, i.e. topology without types.

The trace-level score used for calibration and blocking is the maximum step
logit (the guard blocks a trace at the first step whose score exceeds the
threshold).  Training uses that same maximum with a binary cross-entropy on the
record label (multiple-instance learning), because R-Judge labels records, not
steps.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from conformalguard.encoder.relation_attention import RelationAwareAttention
from conformalguard.rjudge.data import NODE_TYPES, RELATIONS, Trace

MAX_LEN = 64      # default number of position embeddings; the longest R-Judge trace has 37 nodes


def collate(traces: list[Trace], feats: list[np.ndarray], device: str = "cpu") -> dict:
    """Disjoint union of the traces' graphs plus the indices every scorer needs."""
    xs, ntype, pos, gid, src, dst, rel = [], [], [], [], [], [], []
    agent_idx, action_idx, step_graph, labels = [], [], [], []
    off = 0
    for g, (tr, f) in enumerate(zip(traces, feats)):
        n = tr.n_nodes
        xs.append(f)
        ntype += tr.node_type
        pos += list(range(n))
        gid += [g] * n
        for s, d, r in tr.edges:
            src.append(off + s); dst.append(off + d); rel.append(r)
        for a, act in tr.steps:
            agent_idx.append(off + a); action_idx.append(off + act); step_graph.append(g)
        labels.append(tr.label)
        off += n
    t = lambda a, dt=torch.long: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
    return {
        "x": torch.as_tensor(np.concatenate(xs), dtype=torch.float32, device=device),
        "ntype": t(ntype), "pos": t(pos), "gid": t(gid),
        "edge_index": t([src, dst]) if src else torch.zeros(2, 0, dtype=torch.long, device=device),
        "etype": t(rel), "agent_idx": t(agent_idx), "action_idx": t(action_idx),
        "step_graph": t(step_graph), "y": t(labels, torch.float32), "n_graphs": len(traces),
    }


def trace_max(step_logits: torch.Tensor, step_graph: torch.Tensor, n_graphs: int) -> torch.Tensor:
    out = torch.full((n_graphs,), -1e9, device=step_logits.device, dtype=step_logits.dtype)
    return out.scatter_reduce(0, step_graph, step_logits, reduce="amax", include_self=True)


class _Embed(nn.Module):
    def __init__(self, feat_dim: int, d: int, dropout: float, use_types: bool = True, max_len: int = MAX_LEN):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d)
        self.type_emb = nn.Embedding(len(NODE_TYPES), d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.use_types = use_types
        self.drop = nn.Dropout(dropout)

    def forward(self, b: dict) -> torch.Tensor:
        h = self.proj(b["x"]) + self.pos_emb(b["pos"].clamp_max(self.pos_emb.num_embeddings - 1))
        if self.use_types:
            h = h + self.type_emb(b["ntype"])
        return self.drop(h)


def _head(d_in: int, d: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))


class StepScorer(nn.Module):
    """Per-step features only: the agent's thought and its proposed action."""

    def __init__(self, feat_dim: int, d: int = 128, dropout: float = 0.2, **_):
        super().__init__()
        self.head = _head(2 * feat_dim, d, dropout)

    def forward(self, b: dict) -> torch.Tensor:
        x = torch.cat([b["x"][b["agent_idx"]], b["x"][b["action_idx"]]], dim=-1)
        return self.head(x).squeeze(-1)


class SeqScorer(nn.Module):
    """Causal Transformer over the time-ordered node sequence (no edges)."""

    def __init__(self, feat_dim: int, d: int = 128, layers: int = 3, heads: int = 4,
                 dropout: float = 0.2, max_len: int = MAX_LEN, **_):
        super().__init__()
        self.embed = _Embed(feat_dim, d, dropout, max_len=max_len)
        # Feed-forward width 6d gives the same parameter count as the graph scorer.
        layer = nn.TransformerEncoderLayer(d, heads, 6 * d, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = _head(d, d, dropout)

    def forward(self, b: dict) -> torch.Tensor:
        h = self.embed(b)
        B, L, d = b["n_graphs"], int(b["pos"].max()) + 1, h.size(-1)
        seq = h.new_zeros(B, L, d)
        seq[b["gid"], b["pos"]] = h
        pad = torch.ones(B, L, dtype=torch.bool, device=h.device)
        pad[b["gid"], b["pos"]] = False
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=h.device), diagonal=1)
        out = self.enc(seq, mask=causal, src_key_padding_mask=pad)
        a = b["action_idx"]
        return self.head(out[b["gid"][a], b["pos"][a]]).squeeze(-1)


class GraphScorer(nn.Module):
    """Relation-aware attention over the typed execution graph."""

    def __init__(self, feat_dim: int, d: int = 128, layers: int = 3, heads: int = 4,
                 dropout: float = 0.2, typed: bool = True, max_len: int = MAX_LEN, **_):
        super().__init__()
        self.typed = typed
        self.embed = _Embed(feat_dim, d, dropout, use_types=typed, max_len=max_len)
        n_nt = len(NODE_TYPES) if typed else 1
        n_et = len(RELATIONS) if typed else 1
        self.layers = nn.ModuleList(
            [RelationAwareAttention(d, n_nt, n_et, n_heads=heads, dropout=dropout)
             for _ in range(layers)])
        self.head = _head(d, d, dropout)

    def forward(self, b: dict) -> torch.Tensor:
        h = self.embed(b)
        nt = b["ntype"] if self.typed else torch.zeros_like(b["ntype"])
        et = b["etype"] if self.typed else torch.zeros_like(b["etype"])
        for layer in self.layers:
            h = layer(h, nt, b["edge_index"], et)
        return self.head(h[b["action_idx"]]).squeeze(-1)


def build(name: str, feat_dim: int, d: int = 128, layers: int = 3, dropout: float = 0.2,
          max_len: int = MAX_LEN) -> nn.Module:
    kw = dict(feat_dim=feat_dim, d=d, layers=layers, dropout=dropout, max_len=max_len)
    if name == "step":
        return StepScorer(**kw)
    if name == "seq":
        return SeqScorer(**kw)
    if name == "graph":
        return GraphScorer(typed=True, **kw)
    if name == "graph_untyped":
        return GraphScorer(typed=False, **kw)
    raise ValueError(name)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
