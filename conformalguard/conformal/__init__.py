from importlib import import_module

from conformalguard.conformal.core import (
    SplitConformal,
    TwoSidedConformal,
    conformal_quantile,
    lower_conformal_quantile,
    one_step_per_trace,
    trace_max_scores,
)
from conformalguard.conformal.theorems import (
    empirical_coverage,
    finite_sample_correction,
    marginal_coverage_bound,
)

_LAZY = {  # torch-dependent
    "ConformalCalibrator": "conformalguard.conformal.calibration",
    "NegLogProbScore": "conformalguard.conformal.nonconformity",
    "MarginScore": "conformalguard.conformal.nonconformity",
    "NonconformityScore": "conformalguard.conformal.nonconformity",
    "PredictionSet": "conformalguard.conformal.prediction_set",
}

__all__ = [
    "SplitConformal", "TwoSidedConformal", "conformal_quantile",
    "lower_conformal_quantile", "one_step_per_trace", "trace_max_scores",
    "empirical_coverage", "finite_sample_correction", "marginal_coverage_bound",
    *_LAZY,
]


def __getattr__(name):
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(name)
