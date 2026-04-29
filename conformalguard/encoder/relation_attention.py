"""Heterogeneous Graph Transformer (HGT) relation-aware attention.

Implements the per-relation attention mechanism from Hu et al., 2020 with the
schema defined in `ExecutionGraph.SCHEMA`. Each (src_type, edge_type, dst_type)
triple owns its own (Q, K, V, message) projections.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class RelationAwareAttention(nn.Module):
    """One HGT layer over the heterogeneous CT-DG.

    Given node features `H` (N, d), node types `nt` (N,), edge index `(2, E)`,
    edge types `et` (E,), and src/dst type ids `src_t, dst_t` (E,), produce
    updated `H'` (N, d).
    """

    def __init__(self, d_model: int, n_node_types: int, n_edge_types: int,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_nt = n_node_types
        self.n_et = n_edge_types

        # Per-node-type linear projections.
        self.q = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_node_types)])
        self.k = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_node_types)])
        self.v = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_node_types)])
        self.a_lin = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_node_types)])

        # Relation-specific transforms (one per edge type).
        self.W_att = nn.Parameter(torch.empty(n_edge_types, n_heads, self.d_head, self.d_head))
        self.W_msg = nn.Parameter(torch.empty(n_edge_types, n_heads, self.d_head, self.d_head))
        nn.init.xavier_uniform_(self.W_att)
        nn.init.xavier_uniform_(self.W_msg)

        # Per-edge-type prior (learnable scalar μ) and skip α per node type.
        self.mu = nn.Parameter(torch.ones(n_edge_types))
        self.alpha = nn.Parameter(torch.ones(n_node_types))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, nt: torch.Tensor,
                edge_index: torch.Tensor, et: torch.Tensor) -> torch.Tensor:
        n = h.size(0)
        device = h.device
        # Per-node-type projections — vectorize via masking.
        q = torch.zeros(n, self.d_model, device=device)
        k = torch.zeros(n, self.d_model, device=device)
        v = torch.zeros(n, self.d_model, device=device)
        for t in range(self.n_nt):
            mask = nt == t
            if mask.any():
                q[mask] = self.q[t](h[mask])
                k[mask] = self.k[t](h[mask])
                v[mask] = self.v[t](h[mask])
        q = q.view(n, self.n_heads, self.d_head)
        k = k.view(n, self.n_heads, self.d_head)
        v = v.view(n, self.n_heads, self.d_head)

        src, dst = edge_index[0], edge_index[1]
        e = src.size(0)
        if e == 0:
            return self.norm(h)

        # Relation-specific transform of K and V.
        W_att_e = self.W_att[et]                        # (E, H, dh, dh)
        W_msg_e = self.W_msg[et]
        k_src = torch.einsum("ehd,ehdo->eho", k[src], W_att_e)
        v_src = torch.einsum("ehd,ehdo->eho", v[src], W_msg_e)
        q_dst = q[dst]                                  # (E, H, dh)

        # Attention logits: <q_dst, k_src> / sqrt(d) * μ_relation
        scores = (q_dst * k_src).sum(-1) / math.sqrt(self.d_head)  # (E, H)
        scores = scores * self.mu[et].unsqueeze(-1)

        # Softmax over incoming edges per destination node, per head.
        # Stable segmented softmax.
        scores = scores - scores.amax()
        exp = scores.exp()
        denom = torch.zeros(n, self.n_heads, device=device).index_add_(0, dst, exp)
        denom = denom.clamp_min(1e-9)
        attn = exp / denom[dst]
        attn = self.dropout(attn)

        # Aggregate messages.
        msg = attn.unsqueeze(-1) * v_src                # (E, H, dh)
        agg = torch.zeros(n, self.n_heads, self.d_head, device=device)
        agg.index_add_(0, dst, msg)
        agg = agg.reshape(n, self.d_model)

        # Per-destination-type linear + residual with α gate.
        out = torch.zeros_like(h)
        for t in range(self.n_nt):
            mask = nt == t
            if mask.any():
                out[mask] = self.a_lin[t](agg[mask])

        gate = torch.sigmoid(self.alpha[nt]).unsqueeze(-1)
        return self.norm(gate * out + (1.0 - gate) * h)
