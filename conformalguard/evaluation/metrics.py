"""Evaluation metrics matching Sec. 5 of the paper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BlockMetrics:
    block_rate_critical: float
    false_block_rate: float
    precision: float
    recall: float
    f1: float


def block_metrics(true_violation: np.ndarray,
                  predicted_block: np.ndarray) -> BlockMetrics:
    """Per-step block-vs-allow metrics.

    true_violation    : (N,) ∈ {0,1}, 1 if step is a critical violation.
    predicted_block   : (N,) ∈ {0,1}, 1 if ConformalGuard blocked the step.
    """
    tp = float(np.sum((predicted_block == 1) & (true_violation == 1)))
    fp = float(np.sum((predicted_block == 1) & (true_violation == 0)))
    fn = float(np.sum((predicted_block == 0) & (true_violation == 1)))
    tn = float(np.sum((predicted_block == 0) & (true_violation == 0)))
    block_rate_crit = tp / max(1.0, tp + fn)
    false_block = fp / max(1.0, fp + tn)
    prec = tp / max(1.0, tp + fp)
    rec = tp / max(1.0, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return BlockMetrics(block_rate_crit, false_block, prec, rec, f1)


@dataclass
class CoverageMetrics:
    empirical_coverage: float
    target_coverage: float
    avg_set_width: float
    miscoverage_gap: float


def coverage_metrics(in_set_flags: np.ndarray,
                     widths: np.ndarray,
                     alpha: float) -> CoverageMetrics:
    emp = float(np.mean(in_set_flags))
    avg_w = float(np.mean(widths))
    return CoverageMetrics(
        empirical_coverage=emp,
        target_coverage=1 - alpha,
        avg_set_width=avg_w,
        miscoverage_gap=emp - (1 - alpha),
    )


@dataclass
class LatencySummary:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


def latency_summary(latencies_ms: np.ndarray) -> LatencySummary:
    return LatencySummary(
        mean_ms=float(np.mean(latencies_ms)),
        p50_ms=float(np.percentile(latencies_ms, 50)),
        p95_ms=float(np.percentile(latencies_ms, 95)),
        p99_ms=float(np.percentile(latencies_ms, 99)),
    )
