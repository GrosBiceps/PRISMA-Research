"""
strategies/hdbscan_strategy.py — HDBSCAN avec routage GPU intelligent.

Priorité backend (GPU autorisé) :
  1. cuML HDBSCAN      (Linux + NVIDIA RAPIDS) — O(N) sur GPU
  2. PyTorch kNN GPU   (Windows/Linux + CUDA)  — kNN GPU + hdbscan/sklearn CPU
  3. hdbscan C++ CPU                           — rapide, gère bruit (label -1)
  4. sklearn HDBSCAN   (CPU)                   — sklearn ≥1.3
  5. AgglomerativeClustering                   — fallback ultime

Si GPUContext.use_gpu() == False → backends 3/4/5 (référence exacte).

Stratégie PyTorch (backend 2) :
  Calcule les k voisins sur GPU via torch_knn_indices(), construit la matrice
  de distances pré-calculée, puis l'injecte dans hdbscan/sklearn comme
  matrice precomputed — évite le calcul de distances O(N²) sur CPU.
"""

from __future__ import annotations

import logging

import numpy as np

from prisma.core.registry import StrategyRegistry
from prisma.core.gpu_context import GPUContext
from prisma.strategies.base import ClusterParams

logger = logging.getLogger(__name__)

# ── Backend 1 : cuML GPU (Linux/RAPIDS) ─────────────────────────────────────
try:
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    _CUML_AVAILABLE = True
    logger.info("[HDBSCAN] cuML GPU disponible (RAPIDS)")
except ImportError:
    cuHDBSCAN = None
    _CUML_AVAILABLE = False

# ── Backend 2 : PyTorch kNN GPU ──────────────────────────────────────────────
try:
    from prisma.core.torch_utils import torch_cuda_available, torch_knn_indices
    _TORCH_CUDA = torch_cuda_available()
    if _TORCH_CUDA and not _CUML_AVAILABLE:
        logger.info("[HDBSCAN] PyTorch CUDA disponible — kNN GPU actif")
except (ImportError, OSError):
    _TORCH_CUDA = False

# ── Backend 3 : hdbscan C++ CPU ─────────────────────────────────────────────
try:
    import hdbscan as _hdbscan_lib
    _HDBSCAN_LIB_AVAILABLE = True
except ImportError:
    _hdbscan_lib = None
    _HDBSCAN_LIB_AVAILABLE = False

# ── Backend 4 : sklearn HDBSCAN CPU ─────────────────────────────────────────
try:
    from sklearn.cluster import HDBSCAN as _skHDBSCAN
    _SKLEARN_HDBSCAN_AVAILABLE = True
except ImportError:
    _skHDBSCAN = None
    _SKLEARN_HDBSCAN_AVAILABLE = False

# ── Backend 5 : AgglomerativeClustering fallback ─────────────────────────────
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

    Routage : cuML GPU > PyTorch kNN GPU > hdbscan C++ > sklearn > AgglomerativeClustering.
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

    # ── Dispatch principal ────────────────────────────────────────────────────

    def _run(
        self,
        data: np.ndarray,
        min_cluster_size: int,
        min_samples: int,
        metric: str,
    ) -> np.ndarray:
        use_gpu = GPUContext.use_gpu()

        # ── Backend 1 : cuML GPU ─────────────────────────────────────────────
        if _CUML_AVAILABLE and cuHDBSCAN is not None and use_gpu:
            try:
                logger.info("[HDBSCAN] Accélération GPU (cuML RAPIDS) — %d cellules", len(data))
                model = cuHDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric="euclidean",
                    prediction_data=False,
                )
                labels = np.asarray(model.fit_predict(data), dtype=np.int32)
                logger.info("[HDBSCAN] cuML GPU done")
                return labels
            except Exception as exc:
                logger.warning("[HDBSCAN] cuML échoué (%s) — fallback", exc)

        if not use_gpu and _CUML_AVAILABLE:
            logger.info("[HDBSCAN] Exécution forcée sur CPU (Version de référence) demandée par l'utilisateur.")

        # ── Backend 2 : PyTorch kNN GPU + hdbscan/sklearn CPU ───────────────
        if _TORCH_CUDA and use_gpu:
            try:
                return self._run_torch_knn(data, min_cluster_size, min_samples)
            except Exception as exc:
                logger.warning("[HDBSCAN] PyTorch kNN GPU échoué (%s) — fallback CPU", exc)

        # ── Backend 3 : hdbscan C++ CPU ──────────────────────────────────────
        if _HDBSCAN_LIB_AVAILABLE and _hdbscan_lib is not None:
            logger.info("[HDBSCAN] CPU hdbscan (C++) — %d cellules", len(data))
            model = _hdbscan_lib.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                core_dist_n_jobs=-1,
            )
            return np.asarray(model.fit_predict(data), dtype=np.int32)

        # ── Backend 4 : sklearn HDBSCAN CPU ──────────────────────────────────
        if _SKLEARN_HDBSCAN_AVAILABLE and _skHDBSCAN is not None:
            logger.warning("[HDBSCAN] GPU/hdbscan non disponible — sklearn CPU")
            model = _skHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                n_jobs=-1,
            )
            return np.asarray(model.fit_predict(data), dtype=np.int32)

        # ── Backend 5 : AgglomerativeClustering fallback ultime ──────────────
        if _AGG_AVAILABLE and _Agg is not None:
            n_clusters = max(2, len(data) // max(1, min_cluster_size))
            logger.warning(
                "[HDBSCAN] Fallback AgglomerativeClustering k=%d", n_clusters
            )
            return np.asarray(
                _Agg(n_clusters=n_clusters, linkage="ward").fit_predict(data),
                dtype=np.int32,
            )

        raise ImportError(
            "Aucun backend HDBSCAN disponible. "
            "Installez hdbscan: pip install hdbscan"
        )

    @staticmethod
    def _run_torch_knn(
        data: np.ndarray,
        min_cluster_size: int,
        min_samples: int,
    ) -> np.ndarray:
        """
        kNN GPU PyTorch → matrice distances precomputed → hdbscan/sklearn CPU.

        Réduit la complexité O(N²) du calcul de distances de hdbscan
        en précalculant les k voisins sur GPU.
        """
        import scipy.sparse

        n = len(data)
        k = min(min_samples * 5, n - 1)
        logger.info(
            "[HDBSCAN] PyTorch CUDA kNN GPU (k=%d) + hdbscan CPU — %d cellules", k, n
        )

        # kNN sur GPU → distances sparses pour hdbscan precomputed
        indices = torch_knn_indices(data, k=k)   # (n, k)
        rows = np.repeat(np.arange(n), k)
        cols = indices.ravel()
        dists_vals = np.sqrt(np.maximum(0.0,
            np.sum((data[rows] - data[cols]) ** 2, axis=1)
        )).astype(np.float32)

        dist_matrix = scipy.sparse.csr_matrix(
            (dists_vals, (rows, cols)), shape=(n, n)
        )
        # Symétriser
        dist_matrix = (dist_matrix + dist_matrix.T) / 2

        if _HDBSCAN_LIB_AVAILABLE and _hdbscan_lib is not None:
            model = _hdbscan_lib.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="precomputed",
                core_dist_n_jobs=-1,
            )
            labels = np.asarray(model.fit_predict(dist_matrix.toarray()), dtype=np.int32)
        elif _SKLEARN_HDBSCAN_AVAILABLE and _skHDBSCAN is not None:
            model = _skHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="precomputed",
                n_jobs=-1,
            )
            labels = np.asarray(model.fit_predict(dist_matrix.toarray()), dtype=np.int32)
        else:
            raise ImportError("hdbscan ou sklearn requis pour le backend PyTorch kNN")

        logger.info("[HDBSCAN] PyTorch kNN done: %d clusters", len(set(labels) - {-1}))
        return labels

    @staticmethod
    def _reassign_noise(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
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
