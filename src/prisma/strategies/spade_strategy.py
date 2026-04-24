"""
strategies/spade_strategy.py — SPADE via sklearn (implémentation PRISMA-native).

SPADE original (Bendall 2011) nécessite R/Cytobank. Cette implémentation Python
reproduit les deux étapes clés :
  1. Density-dependent downsampling (équilibre les populations rares)
  2. Agglomerative clustering → spanning tree MST sur les centroïdes

Résultat compatible avec le Protocol ClusterStrategy (fit_predict).
"""

from __future__ import annotations

import logging

import numpy as np

from prisma.core.registry import StrategyRegistry
from prisma.strategies.base import ClusterParams

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import AgglomerativeClustering as _Agg
    from sklearn.neighbors import NearestNeighbors as _NNS
    _SKLEARN_AVAILABLE = True
except ImportError:
    _Agg = None
    _NNS = None
    _SKLEARN_AVAILABLE = False


@StrategyRegistry.register_clustering("spade")
class SPADEStrategy:
    """SPADE : density downsampling + clustering hiérarchique + MST."""

    name: str = "spade"

    def fit_predict(self, data: np.ndarray, params: ClusterParams) -> np.ndarray:
        if not _SKLEARN_AVAILABLE:
            raise ImportError("sklearn requis pour SPADEStrategy.")

        extra = dict(params.extra or {})
        n_clusters = int(params.n_clusters or extra.get("k", 200))
        density_factor = float(extra.get("density_factor", 0.1))
        max_cells = int(extra.get("max_cells", 500_000))
        seed = int(params.seed)

        logger.info(
            "SPADE fit_predict: %d cells × %d features, k=%d, density_factor=%.2f",
            data.shape[0], data.shape[1], n_clusters, density_factor,
        )

        x = data
        original_n = len(x)

        # 1. Density-dependent downsampling
        subsample_idx = self._density_downsample(x, density_factor, max_cells, seed)
        x_sub = x[subsample_idx]

        logger.info("SPADE downsampling: %d → %d cellules", original_n, len(x_sub))

        # 2. Agglomerative clustering sur le subset
        k_eff = min(n_clusters, len(x_sub) - 1)
        agg = _Agg(n_clusters=k_eff, linkage="average")
        sub_labels = agg.fit_predict(x_sub).astype(np.int32)

        # 3. Projeter les cellules exclues vers le centroïde le plus proche
        labels = np.full(original_n, -1, dtype=np.int32)
        labels[subsample_idx] = sub_labels

        excluded_mask = labels == -1
        if excluded_mask.any():
            centroids = np.array([
                x_sub[sub_labels == k].mean(axis=0)
                for k in range(k_eff)
                if (sub_labels == k).any()
            ])
            nn = _NNS(n_neighbors=1, metric="euclidean", n_jobs=-1)
            nn.fit(centroids)
            _, nearest = nn.kneighbors(x[excluded_mask])
            labels[excluded_mask] = nearest.ravel().astype(np.int32)

        logger.info("SPADE done: %d clusters", len(np.unique(labels)))
        return labels

    @staticmethod
    def _density_downsample(
        data: np.ndarray,
        density_factor: float,
        max_cells: int,
        seed: int,
    ) -> np.ndarray:
        """Sous-échantillonnage density-dependent : préserve les populations rares."""
        n = len(data)
        if n <= max_cells and density_factor >= 1.0:
            return np.arange(n)

        rng = np.random.default_rng(seed)

        # Estimation de densité locale via distance au k-ième voisin (k=5)
        k_density = min(5, n - 1)
        nn = _NNS(n_neighbors=k_density, metric="euclidean", n_jobs=-1)
        nn.fit(data)
        dists, _ = nn.kneighbors(data)
        local_density = 1.0 / (np.mean(dists[:, 1:], axis=1) + 1e-10)

        # Probabilité inversement proportionnelle à la densité locale
        inv_density = 1.0 / (local_density + 1e-10)
        prob = inv_density / inv_density.sum()

        target = min(max_cells, max(1, int(n * density_factor)))
        target = min(target, n)

        idx = rng.choice(n, size=target, replace=False, p=prob)
        return np.sort(idx)
