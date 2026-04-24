"""
strategies/tsne_strategy.py — t-SNE via openTSNE (CPU multithread).

Fallback sklearn TSNE si openTSNE absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from prisma.core.registry import StrategyRegistry
from prisma.strategies.base import DimReducParams

logger = logging.getLogger(__name__)

try:
    from openTSNE import TSNE as openTSNE
    _OPENTSNE_AVAILABLE = True
except ImportError:
    _OPENTSNE_AVAILABLE = False
    logger.warning("openTSNE not installed — falling back to sklearn TSNE")

try:
    from sklearn.manifold import TSNE as skTSNE
    _SKLEARN_TSNE_AVAILABLE = True
except ImportError:
    _SKLEARN_TSNE_AVAILABLE = False


@dataclass
class TSNEParams(DimReducParams):
    perplexity: float = 30.0
    n_iter: int = 1000
    learning_rate: float = 200.0


@StrategyRegistry.register_dimreduc("tsne")
class TSNEStrategy:
    """t-SNE : openTSNE si dispo, sinon sklearn."""

    name: str = "tsne"

    def fit_transform(self, data: np.ndarray, params: DimReducParams) -> np.ndarray:
        if not _OPENTSNE_AVAILABLE and not _SKLEARN_TSNE_AVAILABLE:
            raise ImportError("Neither openTSNE nor sklearn is installed.")

        tsne_params = params if isinstance(params, TSNEParams) else TSNEParams(
            n_components=params.n_components,
            seed=params.seed,
        )

        logger.info(
            "t-SNE fit_transform: %d cells × %d features, perplexity=%.1f",
            data.shape[0], data.shape[1], tsne_params.perplexity,
        )

        if _OPENTSNE_AVAILABLE:
            reducer = openTSNE(
                n_components=tsne_params.n_components,
                perplexity=tsne_params.perplexity,
                n_iter=tsne_params.n_iter,
                learning_rate=tsne_params.learning_rate,
                random_state=tsne_params.seed,
                n_jobs=tsne_params.n_jobs,
            )
            embedding = np.array(reducer.fit(data))
        else:
            reducer = skTSNE(
                n_components=tsne_params.n_components,
                perplexity=tsne_params.perplexity,
                n_iter=tsne_params.n_iter,
                learning_rate=tsne_params.learning_rate,
                random_state=tsne_params.seed,
            )
            embedding = reducer.fit_transform(data)

        logger.info("t-SNE done: embedding shape %s", embedding.shape)
        return embedding
