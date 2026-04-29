"""Console-script entry points (declared in pyproject.toml)."""

from __future__ import annotations

import sys
from pathlib import Path


def _load_yaml(path: str):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def train():
    from conformalguard.training.trainer import Trainer, TrainerConfig
    from conformalguard.encoder.dygformer_hgt import EncoderConfig
    from conformalguard.data.agentchain26 import AgentChain26
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/encoder.yaml"
    cfg = _load_yaml(cfg_path)
    enc = EncoderConfig(**cfg.get("encoder", {}))
    tr = TrainerConfig(encoder=enc, **{k: v for k, v in cfg.get("trainer", {}).items()})
    trainer = Trainer(tr)
    train_set = AgentChain26(split="train").materialize(cfg.get("limit_train"))
    val_set = AgentChain26(split="val").materialize(cfg.get("limit_val", 200))
    trainer.fit(train_set, val_set)


def calibrate():
    print("Run scripts/calibrate.py — see docs/REPRODUCIBILITY.md.")


def evaluate():
    print("Run scripts/evaluate.py — see docs/REPRODUCIBILITY.md.")


def serve():
    import uvicorn
    uvicorn.run("conformalguard.deployment.server:app", host="0.0.0.0", port=8088)
