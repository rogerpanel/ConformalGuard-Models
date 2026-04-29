"""Evaluate ConformalGuard against the 7 baselines from Table 2."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from conformalguard.conformal.nonconformity import NegLogProbScore, SafetyHead
from conformalguard.conformal.theorems import evaluate_coverage
from conformalguard.data.agentchain26 import AgentChain26
from conformalguard.deployment.guard import ConformalGuardRuntime
from conformalguard.encoder.dygformer_hgt import DyGFormerHGT, EncoderConfig
from conformalguard.evaluation.baselines import BASELINES
from conformalguard.evaluation.metrics import block_metrics, latency_summary
from conformalguard.utils.config import load_config
from conformalguard.utils.logging import get_logger
from conformalguard.utils.seed import set_seed
from conformalguard.violations.classes import VIOLATION_CLASSES


def main():
    log = get_logger("conformalguard.evaluate")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 1234))

    ckpt_path = cfg["evaluation"].get("checkpoint",
                                      "runs/conformalguard/model.pt")
    runtime = ConformalGuardRuntime.load(ckpt_path)

    n_test = int(cfg["dataset"]["limit_test"])
    test = AgentChain26(split="test").materialize(n_test)
    log.info(f"|test|={len(test)}")

    in_set_flags = []
    widths = []
    blocks = []
    truths = []
    latencies = []

    for t in test:
        ts = max((n.timestamp for n in t.graph.nodes.values()), default=0.0)
        proposed = t.label  # in-distribution: proposed = ground truth
        d = runtime.evaluate_step(t.graph, ts, proposed_action=proposed)
        in_set_flags.append(int(d.prediction_set.contains(proposed)))
        widths.append(d.prediction_set.width)
        blocks.append(int(d.verdict.value == "block"))
        truths.append(int(t.label != 0))
        latencies.append(d.latency_ms)

    flags = np.asarray(in_set_flags)
    cov = evaluate_coverage(flags, n_cal=cfg["dataset"]["n_calibration"],
                             alpha=cfg["conformal"]["alpha"])
    bm = block_metrics(np.asarray(truths), np.asarray(blocks))
    lat = latency_summary(np.asarray(latencies))

    log.info(f"empirical coverage = {cov.empirical:.3f} "
             f"[{cov.lower_bound:.3f}, {cov.upper_bound:.3f}] "
             f"in-envelope={cov.in_envelope}")
    log.info(f"block_rate_critical={bm.block_rate_critical:.3f} "
             f"false_block={bm.false_block_rate:.3f} "
             f"F1={bm.f1:.3f}")
    log.info(f"latency mean/p95 = {lat.mean_ms:.2f} / {lat.p95_ms:.2f} ms")

    # Baselines.
    baseline_rows = []
    batch = [{"text": " ".join(str(x) for x in t.graph.stats().items())}
             for t in test]
    for b in BASELINES:
        t0 = time.perf_counter()
        pred = b.predict(batch)
        dt = (time.perf_counter() - t0) * 1000.0 / max(1, len(batch))
        m = block_metrics(np.asarray(truths), np.asarray(pred, dtype=np.int64))
        baseline_rows.append({"name": b.name, "block_critical": m.block_rate_critical,
                              "false_block": m.false_block_rate, "f1": m.f1,
                              "latency_ms": dt})
        log.info(f"  {b.name:24s} block={m.block_rate_critical:.3f} "
                 f"false={m.false_block_rate:.3f} f1={m.f1:.3f}")

    report = {
        "alpha": cfg["conformal"]["alpha"],
        "coverage": cov.__dict__,
        "block_metrics": bm.__dict__,
        "latency": lat.__dict__,
        "average_set_width": float(np.mean(widths)),
        "n_test": len(test),
        "baselines": baseline_rows,
        "n_violation_classes": len(VIOLATION_CLASSES),
    }
    out = Path(cfg["evaluation"]["report_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
