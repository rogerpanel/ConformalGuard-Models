"""Run split-conformal calibration on a held-out validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from conformalguard.conformal.calibration import ConformalCalibrator
from conformalguard.conformal.nonconformity import NegLogProbScore, SafetyHead
from conformalguard.data.agentchain26 import AgentChain26
from conformalguard.encoder.dygformer_hgt import DyGFormerHGT, EncoderConfig
from conformalguard.utils.config import load_config
from conformalguard.utils.logging import get_logger
from conformalguard.utils.seed import set_seed
from conformalguard.violations.classes import VIOLATION_CLASSES


def main():
    log = get_logger("conformalguard.calibrate")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/conformal.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 1234))

    ckpt_path = cfg["input"]["checkpoint"]
    log.info(f"Loading checkpoint {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    enc_cfg = EncoderConfig(**ckpt["cfg"]["encoder"])
    encoder = DyGFormerHGT(enc_cfg); encoder.load_state_dict(ckpt["encoder"])
    head = SafetyHead(d_model=enc_cfg.d_model, vocab_size=len(VIOLATION_CLASSES))
    head.load_state_dict(ckpt["head"])
    encoder.eval(); head.eval()

    n_cal = int(cfg["calibration"]["n_calibration"])
    alpha = float(cfg["calibration"]["alpha"])
    cal = AgentChain26(split=cfg["dataset"]["split"]).materialize(n_cal)
    log.info(f"|cal|={len(cal)} alpha={alpha}")

    with torch.no_grad():
        embs = torch.stack([encoder(encoder.tensorize(t.graph)) for t in cal])
        logits = head(embs)
    y = torch.tensor([t.label for t in cal], dtype=torch.long)

    calibrator = ConformalCalibrator(alpha=alpha, score=NegLogProbScore())
    result = calibrator.calibrate(logits, y)
    log.info(f"τ = {result.threshold:.4f}  (q-level {result.quantile_level:.4f})")

    out = Path(cfg["output"]["threshold_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2))
    log.info(f"Wrote {out}")

    scores = NegLogProbScore()(logits, y).numpy()
    np.save(cfg["output"]["scores_path"], scores)
    log.info(f"Wrote {cfg['output']['scores_path']}")

    # Persist calibrator state into the checkpoint for the runtime to load.
    ckpt["calibrator"] = calibrator.state_dict()
    torch.save(ckpt, ckpt_path)
    log.info("Updated checkpoint with calibrator state.")


if __name__ == "__main__":
    main()
