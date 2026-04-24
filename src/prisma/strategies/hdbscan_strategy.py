"""
strategies/hdbscan_strategy.py — HDBSCAN avec routage GPU intelligent.

Priorité backend :
  1. cuML HDBSCAN (GPU NVIDIA RAPIDS) — O(N) sur GPU, idéal >100k cellules
  2. hdbscan (C++ CPU)                — rapide, gère bruit (label -1)
  3. sklearn HDBSCAN (CPU)            — sklearn ≥1.3
  4. sklearn AgglomerativeClustering  — fallback ultime

Option allow_noise=False : réassigne les cellules bruit (label -1) au cluster
le plus proche via KNN 1-voisin (sklearn CPU, toujours disponible).
"""

from __future__ import annotations

import logging

import numpy as np

from prisma.core.registry import StrategyRegistry
from prisma.core.gpu_context import GPUContext
from prisma.strategies.base import ClusterParams

logger = logging.getLogger(__name__)

# ── Backend 1 : cuML GPU ────────────────────────────────────────────────────
try:
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    _CUML_AVAILABLE = True
    logger.info("[HDBSCAN] cuML GPU disponible — backend RAPIDS actif")
except ImportError:
    cuHDBSCAN = None
    _CUML_AVAILABLE = False

# ── Backend 2 : hdbscan C++ CPU ─────────────────────────────────────────────
try:
    import hdbscan as _hdbscan_lib
    _HDBSCAN_LIB_AVAILABLE = True
    if not _CUML_AVAILABLE:
        logger.info("[HDBSCAN] hdbscan (C++) disponible")
except ImportError:
    _hdbscan_lib = None
    _HDBSCAN_LIB_AVAILABLE = False

# ── Backend 3 : sklearn HDBSCAN CPU ─────────────────────────────────────────
try:
    from sklearn.cluster import HDBSCAN as _skHDBSCAN
    _SKLEARN_HDBSCAN_AVAILABLE = True
except ImportError:
    _skHDBSCAN = None
    _SKLEARN_HDBSCAN_AVAILABLE = False

# ── Backend 4 : fallback AgglomerativeClustering ────────────────────────────
try:
    from sklearn.cluster import AgglomerativeClustering as _Agg
    _AGG_AVAILABLE = True
except ImportError:
    _Agg = None
    _AGG_AVAILABLE = False


@StrategyRegistry.register_clustering("hdbscan")
class HDBSCANStrategy:
    """
    Clustering HDBSCAN basé sur la densité hiérarchique.

    Routage automatique : cuML GPU > hdbscan C++ > sklearn > AgglomerativeClustering.
    """

    name: str = "hdbscan"

    def fit_predict(self, data: np.ndarray, params: ClusterParams) -> np.ndarray:
        extra = dict(params.extra or {})
        min_cluster_size = int(extra.get("min_cluster_size", 50))
        min_samples = int(extra.get("min_samples", 10))
        metric = str(extra.get("metric", "euclidean"))
        allow_noise = bool(extra.get("allow_noise", True))

        logger.info(
            "[HDBSCAN] fit_predict: %d cells × %d features, min_cluster_size=%d",
            data.shape[0], data.shape[1], min_cluster_size,
        )

        labels = self._run(data, min_cluster_size, min_samples, metric)

        if not allow_noise and -1 in labels:
            labels = self._reassign_noise(data, labels)

        n_clusters = len(set(labels) - {-1})
        n_noise = int((labels == -1).sum())
        logger.info(
            "[HDBSCAN] done: %d clusters, %d cellules bruit (label -1)",
            n_clusters, n_noise,
        )
        return labels.astype(np.int32)

    # ── Backends privés ───────────────────────────────────────────────────────

    def _run(
        self,
        data: np.ndarray,
        min_cluster_size: int,
        min_samples: int,
        metric: str,
    ) -> np.ndarray:
        # ── GPU cuML ────────────────────────────────────────────────────────
        if _CUML_AVAILABLE and cuHDBSCAN is not None and not GPUContext.use_gpu():
            logger.info("[HDBSCAN] Exécution forcée sur CPU (Version de référence) demandée par l'utilisateur.")
        elif _CUML_AVAILABLE and cuHDBSCAN is not None:
            try:
                logger.info("[HDBSCAN] Accélération GPU (cuML RAPIDS) — %d cellules", len(data))
                model = cuHDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    # cuML HDBSCAN ne supporte que euclidean nativement
                    metric="euclidean",
                    prediction_data=False,
                )
                labels = np.asarray(model.fit_predict(data), dtype=np.int32)
                logger.info("[HDBSCAN] GPU done")
                return labels
            except Exception as exc:
                logger.warning("[HDBSCAN] cuML échoué (%s) — fallback CPU", exc)

        # ── hdbscan C++ CPU ─────────────────────────────────────────────────
        if _HDBSCAN_LIB_AVAILABLE and _hdbscan_lib is not None:
            logger.info("[HDBSCAN] CPU hdbscan (C++) — %d cellules", len(data))
            model = _hdbscan_lib.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                core_dist_n_jobs=-1,
            )
            return np.asarray(model.fit_predict(data), dtype=np.int32)

        # ── sklearn HDBSCAN CPU ──────────────────────────────────────────────
        if _SKLEARN_HDBSCAN_AVAILABLE and _skHDBSCAN is not None:
            logger.warning("[HDBSCAN] GPU/hdbscan non disponible — sklearn CPU")
            model = _skHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                n_jobs=-1,
            )
            return np.asarray(model.fit_predict(data), dtype=np.int32)

        # ── AgglomerativeClustering fallback ultime ──────────────────────────
        if _AGG_AVAILABLE and _Agg is not None:
            n_clusters = max(2, len(data) // max(1, min_cluster_size))
            logger.warning(
                "[HDBSCAN] HDBSCAN totalement absent — AgglomerativeClustering k=%d",
                n_clusters,
            )
            return np.asarray(
                _Agg(n_clusters=n_clusters, linkage="ward").fit_predict(data),
                dtype=np.int32,
            )

        raise ImportError(
            "Aucun backend HDBSCAN disponible. "
            "Installez cuml (GPU) ou hdbscan: pip install hdbscan"
        )

    @staticmethod
    def _reassign_noise(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Réassigne les cellules bruit (label -1) au cluster voisin le plus proche."""
        noise_mask = labels == -1
        if not noise_mask.any() or not (~noise_mask).any():
            return labels
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
            nn.fit(data[~noise_mask])
            _, idx = nn.kneighbors(data[noise_mask])
            non_noise_labels = labels[~noise_mask]
            labels = labels.copy()
            labels[noise_mask] = non_noise_labels[idx.ravel()]
        except Exception as exc:
            logger.warning("[HDBSCAN] Réassignation bruit échouée (%s) — label -1 conservé", exc)
        return labels
