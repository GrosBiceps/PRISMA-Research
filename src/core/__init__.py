"""
src/core/__init__.py — Exports publics de la couche core.
"""

from .transformers import DataTransformer
from .normalizers import DataNormalizer
from .gating import PreGating
from .auto_gating import AutoGating
from .clustering import (
    FlowSOMClusterer,
    find_optimal_clusters_stability,
    phase1_silhouette_on_codebook,
    phase2_bootstrap_stability,
    phase3_composite_selection,
)
from .metaclustering import find_optimal_clusters
from .interfaces import INodeProcessor, IAnalysisStrategy, IExportStrategy

__all__ = [
    "DataTransformer",
    "DataNormalizer",
    "PreGating",
    "AutoGating",
    "FlowSOMClusterer",
    "find_optimal_clusters",
    "find_optimal_clusters_stability",
    "phase1_silhouette_on_codebook",
    "phase2_bootstrap_stability",
    "phase3_composite_selection",
    "INodeProcessor",
    "IAnalysisStrategy",
    "IExportStrategy",
]
