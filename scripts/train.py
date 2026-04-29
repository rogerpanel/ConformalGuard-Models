"""Train DyGFormer-HGT + safety head on AgentChain-26."""

from __future__ import annotations

import argparse

from conformalguard.data.agentchain26 import AgentChain26
from conformalguard.encoder.dygformer_hgt import EncoderConfig
from conformalguard.training.trainer import Trainer, TrainerConfig
from conformalguard.utils.config import load_config
from conformalguard.utils.logging import get_logger
from conformalguard.utils.seed import set_seed


def main():
    log = get_logger("conformalguard.train")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/encoder.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    seed = cfg.get("seed", 1234)
    set_seed(seed)

    enc = EncoderConfig(**cfg.get("encoder", {}))
    tr_cfg = cfg.get("trainer", {})
    tr = TrainerConfig(encoder=enc, **tr_cfg)
    log.info(f"Training on device={tr.device}, seed={seed}, lr={tr.lr}")

    trainer = Trainer(tr)
    train_set = AgentChain26(split="train").materialize(cfg.get("limit_train", 9920))
    val_set   = AgentChain26(split="val").materialize(cfg.get("limit_val", 1240))
    log.info(f"|train|={len(train_set)}  |val|={len(val_set)}")
    out = trainer.fit(train_set, val_set)
    log.info(f"Final epoch loss: {out['history'][-1]['train_loss']:.4f}")


if __name__ == "__main__":
    main()
