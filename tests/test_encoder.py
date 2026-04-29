"""Smoke tests for the DyGFormer-HGT encoder."""

from __future__ import annotations

import torch

from conformalguard.data.synthetic import SyntheticConfig, SyntheticTraceGenerator
from conformalguard.encoder.dygformer_hgt import DyGFormerHGT, EncoderConfig


def test_encoder_forward_shapes():
    cfg = EncoderConfig(d_model=64, n_heads=4, n_layers=2, time_dim=16, feature_dim=32)
    enc = DyGFormerHGT(cfg)
    gen = SyntheticTraceGenerator(SyntheticConfig(n_traces=4, seed=0))
    for graph, _ in gen.generate_dataset():
        h = enc.encode_graph(graph)
        assert h.shape == (cfg.d_model,)


def test_encoder_handles_empty_graph():
    cfg = EncoderConfig(d_model=32, n_heads=4, n_layers=1, time_dim=8, feature_dim=16)
    enc = DyGFormerHGT(cfg)
    from conformalguard.graph.execution_graph import ExecutionGraph
    h = enc.encode_graph(ExecutionGraph())
    assert torch.all(h == 0)
