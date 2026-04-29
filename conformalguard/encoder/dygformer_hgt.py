"""DyGFormer-HGT encoder.

Three-layer heterogeneous dynamic graph neural network combining:
  * HGT (Hu et al., 2020) — relation-aware attention over heterogeneous nodes,
  * DyGFormer (Yu et al., 2023) — neighbor-co-occurrence / temporal attention.

The encoder produces a per-step subgraph embedding `h_t ∈ R^{d_model}` that is
consumed by the conformal prediction head.

Hyper-parameters from the paper (Sec. 5):
    d_model = 256, n_heads = 8, n_layers = 3, dropout = 0.1, time_dim = 64.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from conformalguard.encoder.relation_attention import RelationAwareAttention
from conformalguard.encoder.temporal_attention import TemporalAttention, TimeEncoder
from conformalguard.graph.execution_graph import EdgeType, ExecutionGraph, NodeType


@dataclass
class EncoderConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 3
    dropout: float = 0.1
    time_dim: int = 64
    n_node_types: int = len(NodeType)   # 4
    n_edge_types: int = len(EdgeType)   # 6
    feature_dim: int = 128              # raw node feature dimensionality
    pool: str = "mean+gru"              # how subgraphs are pooled into h_t


class _MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or d_out
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):  # noqa: ANN001
        return self.net(x)


class DyGFormerHGT(nn.Module):
    """The full encoder: feature -> 3× (HGT + Temporal) -> pooled embedding."""

    def __init__(self, cfg: EncoderConfig | None = None):
        super().__init__()
        cfg = cfg or EncoderConfig()
        self.cfg = cfg

        self.feature_proj = _MLP(cfg.feature_dim, cfg.d_model, dropout=cfg.dropout)
        self.time_enc = TimeEncoder(cfg.time_dim)
        self.time_proj = nn.Linear(cfg.time_dim, cfg.d_model)
        self.type_emb = nn.Embedding(cfg.n_node_types, cfg.d_model)

        self.hgt_layers = nn.ModuleList([
            RelationAwareAttention(
                d_model=cfg.d_model,
                n_node_types=cfg.n_node_types,
                n_edge_types=cfg.n_edge_types,
                n_heads=cfg.n_heads,
                dropout=cfg.dropout,
            ) for _ in range(cfg.n_layers)
        ])
        self.temporal_layers = nn.ModuleList([
            TemporalAttention(d_model=cfg.d_model, n_heads=cfg.n_heads,
                              dropout=cfg.dropout, time_dim=cfg.time_dim)
            for _ in range(cfg.n_layers)
        ])

        if cfg.pool.startswith("mean+gru"):
            self.pool_gru = nn.GRU(cfg.d_model, cfg.d_model, batch_first=True)

        self.norm = nn.LayerNorm(cfg.d_model)

    # --------------------------------------------------------------------
    # Tensorization: ExecutionGraph -> tensors
    # --------------------------------------------------------------------
    def tensorize(self, graph: ExecutionGraph) -> dict[str, torch.Tensor]:
        nodes = list(graph.nodes.values())
        node_id_to_idx = {n.node_id: i for i, n in enumerate(nodes)}
        n = len(nodes)

        if n == 0:
            return self._empty_batch()

        feat = np.zeros((n, self.cfg.feature_dim), dtype=np.float32)
        nt = np.zeros(n, dtype=np.int64)
        ts = np.zeros(n, dtype=np.float32)
        nt_to_id = {t: i for i, t in enumerate(NodeType)}
        for i, node in enumerate(nodes):
            nt[i] = nt_to_id[node.node_type]
            ts[i] = node.timestamp
            if node.features is not None:
                d = min(self.cfg.feature_dim, node.features.shape[0])
                feat[i, :d] = node.features[:d]
            else:
                # Hash text payload to a deterministic feature vector.
                content = str(node.payload.get("content", node.node_id))
                rng = np.random.default_rng(abs(hash(content)) % (2**32))
                feat[i] = rng.standard_normal(self.cfg.feature_dim).astype(np.float32) * 0.1

        et_to_id = {t: i for i, t in enumerate(EdgeType)}
        if graph.edges:
            edge_index = np.array(
                [[node_id_to_idx[e.src], node_id_to_idx[e.dst]] for e in graph.edges],
                dtype=np.int64).T
            et = np.array([et_to_id[e.edge_type] for e in graph.edges], dtype=np.int64)
            edge_t = np.array([e.timestamp for e in graph.edges], dtype=np.float32)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            et = np.zeros((0,), dtype=np.int64)
            edge_t = np.zeros((0,), dtype=np.float32)

        return {
            "x":          torch.from_numpy(feat),
            "nt":         torch.from_numpy(nt),
            "ts":         torch.from_numpy(ts),
            "edge_index": torch.from_numpy(edge_index),
            "et":         torch.from_numpy(et),
            "edge_t":     torch.from_numpy(edge_t),
        }

    def _empty_batch(self) -> dict[str, torch.Tensor]:
        return {
            "x":          torch.zeros(0, self.cfg.feature_dim),
            "nt":         torch.zeros(0, dtype=torch.long),
            "ts":         torch.zeros(0),
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
            "et":         torch.zeros(0, dtype=torch.long),
            "edge_t":     torch.zeros(0),
        }

    # --------------------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------------------
    def forward(self, batch: dict[str, torch.Tensor],
                step_t: float | None = None) -> torch.Tensor:
        """Encode a (sub)graph into a single d_model-dimensional embedding."""
        x = batch["x"]
        nt = batch["nt"]
        ts = batch["ts"]
        edge_index = batch["edge_index"]
        et = batch["et"]

        if x.numel() == 0:
            return torch.zeros(self.cfg.d_model, device=x.device)

        # Features + node-type + time embeddings.
        h = self.feature_proj(x) + self.type_emb(nt)
        if step_t is None:
            step_t = float(ts.max().item())
        dt = (step_t - ts).clamp_min(0.0)
        h = h + self.time_proj(self.time_enc(dt))

        # 3 alternating HGT + Temporal blocks.
        for hgt, temp in zip(self.hgt_layers, self.temporal_layers):
            h = hgt(h, nt, edge_index, et)
            # Temporal self-attention over the node sequence (ordered by ts).
            order = torch.argsort(ts)
            h_seq = h[order].unsqueeze(0)             # (1, N, d)
            t_seq = ts[order].unsqueeze(0)            # (1, N)
            dt_mat = (t_seq.unsqueeze(2) - t_seq.unsqueeze(1)).clamp_min(0.0)
            causal = dt_mat < 0  # never True given clamp; placeholder
            h_seq = h_seq + temp(h_seq, h_seq, h_seq, dt_mat, mask=causal)
            inv = torch.argsort(order)
            h = h_seq.squeeze(0)[inv]

        # Pooling: mean over nodes, then optional 1-step GRU update with the
        # most recent agent-invocation embedding.
        h_mean = h.mean(dim=0)
        if self.cfg.pool.startswith("mean+gru"):
            agent_mask = nt == list(NodeType).index(NodeType.AGENT_INVOCATION)
            if agent_mask.any():
                latest_idx = int(ts.masked_fill(~agent_mask, float("-inf")).argmax())
                h_latest = h[latest_idx]
            else:
                h_latest = h_mean
            inp = h_latest.view(1, 1, -1)
            hidden = h_mean.view(1, 1, -1)
            _, hidden = self.pool_gru(inp, hidden)
            return self.norm(hidden.view(-1))
        return self.norm(h_mean)

    # --------------------------------------------------------------------
    # Convenience: encode straight from an ExecutionGraph
    # --------------------------------------------------------------------
    @torch.no_grad()
    def encode_graph(self, graph: ExecutionGraph,
                     step_t: float | None = None) -> torch.Tensor:
        batch = self.tensorize(graph)
        device = next(self.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        return self.forward(batch, step_t=step_t)
