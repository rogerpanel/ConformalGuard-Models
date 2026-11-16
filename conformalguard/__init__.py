"""ConformalGuard: distribution-free false-block guarantees for multi-agent LLM systems.

Reference implementation.  Heavy (torch) submodules are imported lazily so the
conformal primitives in ``conformalguard.conformal.core`` and
``conformalguard.adaptive.gibbs_candes`` can be used and tested with NumPy only.
"""

from importlib import import_module

__version__ = "1.1.0"

_LAZY = {
    "ExecutionGraph": "conformalguard.graph.execution_graph",
    "NodeType": "conformalguard.graph.execution_graph",
    "EdgeType": "conformalguard.graph.execution_graph",
    "GraphBuilder": "conformalguard.graph.builder",
    "DyGFormerHGT": "conformalguard.encoder.dygformer_hgt",
    "ConformalCalibrator": "conformalguard.conformal.calibration",
    "PredictionSet": "conformalguard.conformal.prediction_set",
    "SplitConformal": "conformalguard.conformal.core",
    "TwoSidedConformal": "conformalguard.conformal.core",
    "AdaptiveConformal": "conformalguard.adaptive.gibbs_candes",
    "GibbsCandesController": "conformalguard.adaptive.gibbs_candes",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'conformalguard' has no attribute {name!r}")
