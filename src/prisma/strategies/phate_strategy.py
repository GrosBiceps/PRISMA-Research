"""
strategies/phate_strategy.py — PHATE avec fallback CPU.

PHATE est utilisé comme stratégie de réduction dimensionnelle modulaire.
Si le package phate est absent, un fallback PCA CPU conserve le pipeline
opérationnel sans dépendance optionnelle bloquante.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from prisma.core.registry import StrategyRegistry
from prisma.strategies.base import DimReducParams

logger = logging.getLogger(__name__)

try:
    import phate as _phate  # type: ignore

    _PHATE_AVAILABLE = True
except ImportError:
    _phate = None
    _PHATE_AVAILABLE = False
    logger.warning("phate not installed — PHATEStrategy will use CPU fallback.")

try:
    from sklearn.decomposition import PCA as _PCA

    _SKLEARN_AVAILABLE = True
except ImportError:
    _PCA = None
    _SKLEARN_AVAILABLE = False


@dataclass
class PHATEParams(DimReducParams):
    """Hyperparamètres PHATE strictement typés."""

    decay: int = 40
    knn: int = 10


@StrategyRegistry.register_dimreduc("phate")
class PHATEStrategy:
    """Réduction dimensionnelle PHATE avec fallback CPU explicite."""

    name: str = "phate"

    def fit_transform(self, data: np.ndarray, params: DimReducParams) -> np.ndarray:
        phate_params = self._coerce_params(params)

        logger.info(
            "PHATE fit_transform: %d cells × %d features, knn=%d, decay=%d",
            data.shape[0],
            data.shape[1],
            phate_params.knn,
            phate_params.decay,
        )

        if _PHATE_AVAILABLE and _phate is not None:
            reducer = _phate.PHATE(
                n_components=phate_params.n_components,
                knn=phate_params.knn,
                decay=phate_params.decay,
                random_state=phate_params.seed,
                n_jobs=phate_params.n_jobs,
                verbose=False,
            )
            embedding = np.asarray(reducer.fit_transform(data), dtype=np.float32)
            logger.info("PHATE done: embedding shape %s", embedding.shape)
            return embedding

        embedding = self._fallback_embedding(data, phate_params)
        logger.info("PHATE fallback done: embedding shape %s", embedding.shape)
        return embedding

    @staticmethod
    def _coerce_params(params: DimReducParams) -> PHATEParams:
        if isinstance(params, PHATEParams):
            return params
        return PHATEParams(
            n_components=params.n_components,
            seed=params.seed,
            n_jobs=params.n_jobs,
            extra=dict(params.extra),
        )

    @staticmethod
    def _fallback_embedding(data: np.ndarray, params: PHATEParams) -> np.ndarray:
        centered = np.asarray(data, dtype=np.float32)
        centered = centered - np.nanmean(centered, axis=0, keepdims=True)

        if _SKLEARN_AVAILABLE and _PCA is not None:
            reducer = _PCA(n_components=params.n_components, random_state=params.seed)
            return np.asarray(reducer.fit_transform(centered), dtype=np.float32)

        # Fallback ultime: projection SVD pure NumPy, toujours CPU.
        u, s, _ = np.linalg.svd(np.nan_to_num(centered, nan=0.0), full_matrices=False)
        embedding = (u[:, : params.n_components] * s[: params.n_components]).astype(np.float32)
        if embedding.shape[1] < params.n_components:
            pad = np.zeros(
                (embedding.shape[0], params.n_components - embedding.shape[1]), dtype=np.float32
            )
            embedding = np.hstack([embedding, pad])
        return embedding
