"""ConformalGuard: Distribution-Free Safety Certification for Multi-Agent LLM Systems.

Reference implementation accompanying Anaedevha & Trofimov (2026).
"""

from conformalguard.graph.execution_graph import ExecutionGraph, NodeType, EdgeType
from conformalguard.graph.builder import GraphBuilder
from conformalguard.encoder.dygformer_hgt import DyGFormerHGT
from conformalguard.conformal.calibration import ConformalCalibrator
from conformalguard.conformal.prediction_set import PredictionSet
from conformalguard.adaptive.gibbs_candes import GibbsCandesController

__version__ = "1.0.0"

__all__ = [
    "ExecutionGraph",
    "NodeType",
    "EdgeType",
    "GraphBuilder",
    "DyGFormerHGT",
    "ConformalCalibrator",
    "PredictionSet",
    "GibbsCandesController",
]
