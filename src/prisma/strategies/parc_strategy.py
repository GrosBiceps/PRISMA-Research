"""
strategies/parc_strategy.py — PARC via parc ou sklearn fallback.

Backend 1 : parc (pip install parc) — ultra-rapide HNSW + Leiden, millions de cellules
Backend 2 : sklearn AgglomerativeClustering (fallback)
"""

from __future__ import annotations

import logging

import numpy as np

from prisma.core.registry import StrategyRegistry
from prisma.strategies.base import ClusterParams

logger = logging.getLogger(__name__)

try:
    import parc as _parc_lib
    _PARC_AVAILABLE = True
    logger.info("parc disponible")
except ImportError:
    _parc_lib = None
    _PARC_AVAILABLE = False
    logger.warning("parc non installé — fallback sklearn actif")

try:
    from sklearn.cluster import AgglomerativeClustering as _Agg
    _SKLEARN_AVAILABLE = True
except ImportError:
    _Agg = None
    _SKLEARN_AVAILABLE = False


@StrategyRegistry.register_clustering("parc")
class PARCStrategy:
    """Clustering PARC : HNSW graph + Leiden community detection."""

    name: str = "parc"

    def fit_predict(self, data: np.ndarray, params: ClusterParams) -> np.ndarray:
        extra = dict(params.extra or {})
        seed = int(params.seed)

        logger.info(
            "PARC fit_predict: %d cells × %d features",
            data.shape[0], data.shape[1],
        )

        if _PARC_AVAILABLE and _parc_lib is not None:
            try:
                p = _parc_lib.PARC(
                    data,
                    dist_std_local=float(extra.get("dist_std_local", 2.0)),
                    jac_std_global=float(extra.get("jac_std_global", 0.15)),
                    knn=int(extra.get("knn", 30)),
                    n_iter_leiden=int(extra.get("n_iter_leiden", 5)),
                    random_seed=seed,
                    verbose=False,
                )
                p.run_PARC()
                labels = np.asarray(p.labels, dtype=np.int32)
                logger.info("PARC done: %d clusters", len(np.unique(labels)))
                return labels
            except Exception as exc:
                logger.warning("PARC failed (%s) — fallback sklearn", exc)

        return self._fallback(data, params)

    def _fallback(self, data: np.ndarray, params: ClusterParams) -> np.ndarray:
        if not _SKLEARN_AVAILABLE or _Agg is None:
            raise ImportError("parc et sklearn non disponibles.")

        n_clusters = int(params.n_clusters or 20)
        logger.warning("PARC fallback AgglomerativeClustering k=%d", n_clusters)
        labels = _Agg(n_clusters=n_clusters, linkage="ward").fit_predict(data)
        return np.asarray(labels, dtype=np.int32)
