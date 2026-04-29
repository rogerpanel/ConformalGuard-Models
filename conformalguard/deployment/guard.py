"""Production runtime — the object instantiated by the RobustIDPS plug-in.

Brings together:
  * `DyGFormerHGT`            — encoder
  * `SafetyHead`              — vocabulary classifier
  * `ConformalCalibrator`     — split-conformal threshold
  * `GibbsCandesController`   — online adaptive coverage
  * `DriftMonitor`            — KS-based drift early warning
  * `AnalystRouter`           — routes flagged steps to the SOC review queue
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from conformalguard.adaptive.drift_monitor import DriftMonitor
from conformalguard.adaptive.gibbs_candes import GibbsCandesController
from conformalguard.conformal.calibration import ConformalCalibrator
from conformalguard.conformal.nonconformity import NegLogProbScore, SafetyHead
from conformalguard.conformal.prediction_set import PredictionSet, Verdict
from conformalguard.encoder.dygformer_hgt import DyGFormerHGT, EncoderConfig
from conformalguard.graph.execution_graph import ExecutionGraph
from conformalguard.violations.classes import VIOLATION_CLASSES


@dataclass
class GuardDecision:
    verdict: Verdict
    prediction_set: PredictionSet
    proposed_action: int
    latency_ms: float
    drift: float
    alpha_eff: float


class ConformalGuardRuntime:
    """End-to-end inference pipeline."""

    def __init__(
        self,
        encoder: DyGFormerHGT,
        head: SafetyHead,
        calibrator: ConformalCalibrator,
        adaptive: GibbsCandesController | None = None,
        drift_monitor: DriftMonitor | None = None,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device).eval()
        self.head = head.to(self.device).eval()
        self.calibrator = calibrator
        self.adaptive = adaptive or GibbsCandesController()
        self.drift_monitor = drift_monitor
        self.score = NegLogProbScore()

    # --------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_step(
        self,
        graph: ExecutionGraph,
        step_t: float,
        proposed_action: int,
    ) -> GuardDecision:
        t0 = time.perf_counter()
        sub = graph.subgraph_at(step_t)
        h = self.encoder.encode_graph(sub, step_t=step_t).unsqueeze(0)
        logits = self.head(h)
        all_scores = self.score.all_candidates(logits).squeeze(0)

        # Adjust threshold by adaptive controller.
        base_tau = self.calibrator.threshold()
        tau_eff = base_tau + self.adaptive.adjustment(base_tau)
        pset = PredictionSet.from_logits(
            logits.squeeze(0),
            threshold=tau_eff,
            nonconformity_all_candidates=all_scores,
            proposed_action=proposed_action,
        )
        latency = (time.perf_counter() - t0) * 1000.0

        # Update online state.
        true_in_set = int(pset.contains(proposed_action))
        self.adaptive.update(error_indicator=1 - true_in_set)
        drift = 0.0
        if self.drift_monitor is not None:
            drift = self.drift_monitor.observe(float(all_scores[proposed_action])).ks_statistic

        return GuardDecision(
            verdict=pset.verdict,
            prediction_set=pset,
            proposed_action=proposed_action,
            latency_ms=latency,
            drift=drift,
            alpha_eff=self.adaptive.alpha,
        )

    # --------------------------------------------------------------------
    @classmethod
    def load(cls, ckpt_path: str | Path) -> "ConformalGuardRuntime":
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = EncoderConfig(**ckpt["cfg"]["encoder"])
        encoder = DyGFormerHGT(cfg); encoder.load_state_dict(ckpt["encoder"])
        head = SafetyHead(d_model=cfg.d_model, vocab_size=len(VIOLATION_CLASSES))
        head.load_state_dict(ckpt["head"])
        calibrator = ConformalCalibrator()
        cal_state = ckpt.get("calibrator")
        if cal_state:
            calibrator.load_state_dict(cal_state)
        return cls(encoder=encoder, head=head, calibrator=calibrator)
