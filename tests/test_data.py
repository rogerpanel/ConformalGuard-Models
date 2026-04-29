"""Tests for the AgentChain-26 / synthetic loader."""

from __future__ import annotations

from conformalguard.data.agentchain26 import AgentChain26
from conformalguard.data.external import EXTERNAL_DATASETS, find


def test_synthetic_split_sizes():
    ds = AgentChain26(split="train", synthetic_if_missing=True)
    n = sum(1 for _ in ds.materialize(50))
    assert n == 50


def test_external_registry_complete():
    assert any(d.name == "ToolEmu" for d in EXTERNAL_DATASETS)
    d = find("RobustIDPS-PCAPs")
    assert d.url.startswith("https://doi.org/")
