"""
strategies/umap_strategy.py — Stratégie UMAP avec fallback CPU.

GPU (cuML) utilisé si disponible, sinon umap-learn CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from prisma.core.registry import StrategyRegistry
from prisma.strategies.base import DimReducParams

logger = logging.getLogger(__name__)

try:
    from cuml.manifold import UMAP as cuUMAP
    _CUML_AVAILABLE = True
    logger.info("cuML UMAP available — GPU acceleration active")
except ImportError:
    _CUML_AVAILABLE = False

try:
    import umap as umap_learn
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False
    logger.warning("umap-learn not installed — UMAPStrategy unavailable")


@dataclass
class UMAPParams(DimReducParams):
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"


@StrategyRegistry.register_dimreduc("umap")
class UMAPStrategy:
    """UMAP : GPU (cuML) si dispo, sinon umap-learn CPU."""

    name: str = "umap"

    def fit_transform(self, data: np.ndarray, params: DimReducParams) -> np.ndarray:
        if not _UMAP_AVAILABLE and not _CUML_AVAILABLE:
            raise ImportError("Neither umap-learn nor cuml is installed.")

        umap_params = params if isinstance(params, UMAPParams) else UMAPParams(
            n_components=params.n_components,
            seed=params.seed,
            n_jobs=params.n_jobs,
        )

        logger.info(
            "UMAP fit_transform: %d cells × %d features, n_neighbors=%d, min_dist=%.2f",
            data.shape[0], data.shape[1],
            umap_params.n_neighbors, umap_params.min_dist,
        )

        if _CUML_AVAILABLE:
            reducer = cuUMAP(
                n_components=umap_params.n_components,
                n_neighbors=umap_params.n_neighbors,
                min_dist=umap_params.min_dist,
                random_state=umap_params.seed,
            )
        else:
            reducer = umap_learn.UMAP(
                n_components=umap_params.n_components,
                n_neighbors=umap_params.n_neighbors,
                min_dist=umap_params.min_dist,
                metric=umap_params.metric,
                random_state=umap_params.seed,
                n_jobs=umap_params.n_jobs,
            )

        embedding: np.ndarray = reducer.fit_transform(data)
        logger.info("UMAP done: embedding shape %s", embedding.shape)
        return embedding
