"""
strategies/phenograph_strategy.py — PhenoGraph avec routage GPU intelligent.

Priorité backend :
  1. Full GPU (cuML KNN + cuGraph Louvain) — pipeline RAPIDS entier sur GPU
  2. phenograph CPU                        — package officiel, Leiden/Louvain
  3. sklearn KNN + AgglomerativeClustering — fallback universel

Architecture GPU "Full-GPU PhenoGraph" :
  ① cuml.neighbors.NearestNeighbors → graphe k-NN construit sur GPU
  ② cudf/scipy → matrice d'adjacence sparse symétrisée
  ③ cugraph.community.louvain       → détection communautés sur GPU

Note cuGraph : louvain retourne (parts_df, modularity). parts_df colonnes :
  "vertex" (int) et "partition" (int).
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

from prisma.core.registry import StrategyRegistry
from prisma.strategies.base import ClusterParams

logger = logging.getLogger(__name__)

# ── Backend GPU : cuML + cuGraph ─────────────────────────────────────────────
try:
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    _CUML_NN_AVAILABLE = True
    logger.info("[PhenoGraph] cuML NearestNeighbors GPU disponible")
except ImportError:
    cuNearestNeighbors = None
    _CUML_NN_AVAILABLE = False

try:
    import cugraph
    _CUGRAPH_AVAILABLE = True
    if _CUML_NN_AVAILABLE:
        logger.info("[PhenoGraph] cuGraph GPU disponible — Full-GPU PhenoGraph actif")
except ImportError:
    cugraph = None
    _CUGRAPH_AVAILABLE = False
    if _CUML_NN_AVAILABLE:
        logger.warning("[PhenoGraph] cuML dispo mais cuGraph absent — fallback CPU")

_GPU_FULL_AVAILABLE = _CUML_NN_AVAILABLE and _CUGRAPH_AVAILABLE

# ── Backend CPU : phenograph ─────────────────────────────────────────────────
try:
    import phenograph as _phenograph
    _PHENOGRAPH_AVAILABLE = True
    if not _GPU_FULL_AVAILABLE:
        logger.info("[PhenoGraph] phenograph CPU disponible")
except ImportError:
    _phenograph = None
    _PHENOGRAPH_AVAILABLE = False
    if not _GPU_FULL_AVAILABLE:
        logger.warning("[PhenoGraph] phenograph absent — fallback sklearn actif")

# ── Fallback sklearn ─────────────────────────────────────────────────────────
try:
    from sklearn.neighbors import NearestNeighbors as _skNN
    from sklearn.cluster import AgglomerativeClustering as _Agg
    _SKLEARN_AVAILABLE = True
except ImportError:
    _skNN = _Agg = None
    _SKLEARN_AVAILABLE = False


@StrategyRegistry.register_clustering("phenograph")
class PhenoGraphStrategy:
    """
    PhenoGraph : graphe k-NN + détection de communautés Louvain/Leiden.

    Routage automatique : Full-GPU (cuML+cuGraph) > phenograph CPU > sklearn fallback.
    """

    name: str = "phenograph"

    def fit_predict(self, data: np.ndarray, params: ClusterParams) -> np.ndarray:
        extra = dict(params.extra or {})
        k = int(extra.get("k", 30))
        seed = int(params.seed)

        logger.info(
            "[PhenoGraph] fit_predict: %d cells × %d features, k=%d",
            data.shape[0], data.shape[1], k,
        )

        if _GPU_FULL_AVAILABLE:
            try:
                return self._run_gpu(data, k, seed)
            except Exception as exc:
                logger.warning("[PhenoGraph] Full-GPU échoué (%s) — fallback CPU", exc)

        if _PHENOGRAPH_AVAILABLE and _phenograph is not None:
            return self._run_phenograph(data, k, seed)

        return self._run_sklearn_fallback(data, params)

    # ── Backend Full-GPU ──────────────────────────────────────────────────────

    @staticmethod
    def _run_gpu(data: np.ndarray, k: int, seed: int) -> np.ndarray:
        import cudf

        n = len(data)
        k_safe = min(k, n - 1)
        logger.info(
            "[PhenoGraph] Full-GPU (cuML KNN + cuGraph Louvain) — %d cellules, k=%d", n, k_safe
        )

        # ① KNN sur GPU
        nn_model = cuNearestNeighbors(n_neighbors=k_safe, metric="euclidean")
        nn_model.fit(data)
        distances, indices = nn_model.kneighbors(data)

        # Convertir indices GPU → numpy
        indices_np = np.asarray(indices)   # (n, k_safe)
        distances_np = np.asarray(distances)  # (n, k_safe) — non utilisé pour Louvain

        # ② Construction matrice d'adjacence sparse symétrique (CPU scipy)
        rows = np.repeat(np.arange(n), k_safe)
        cols = indices_np.ravel()
        weights = np.ones(len(rows), dtype=np.float32)

        adj = sp.coo_matrix((weights, (rows, cols)), shape=(n, n))
        adj = adj + adj.T                    # symétrisation
        adj = (adj > 0).astype(np.float32)   # binaire non-pondéré

        # ③ cuGraph Louvain — requiert cudf.DataFrame (src, dst, weight)
        coo = adj.tocoo()
        edge_df = cudf.DataFrame({
            "src": coo.row.astype(np.int32),
            "dst": coo.col.astype(np.int32),
            "weight": coo.data.astype(np.float32),
        })
        G = cugraph.Graph()
        G.from_cudf_edgelist(edge_df, source="src", destination="dst", edge_attr="weight")

        parts_df, modularity = cugraph.community.louvain(G)
        parts_np = parts_df.sort_values("vertex").to_pandas()
        labels = np.asarray(parts_np["partition"].values, dtype=np.int32)

        n_clusters = len(np.unique(labels))
        logger.info(
            "[PhenoGraph] Full-GPU done: %d clusters, modularity=%.4f", n_clusters, modularity
        )
        return labels

    # ── Backend phenograph CPU ────────────────────────────────────────────────

    @staticmethod
    def _run_phenograph(data: np.ndarray, k: int, seed: int) -> np.ndarray:
        logger.warning("[PhenoGraph] GPU non disponible — phenograph CPU (k=%d)", k)
        communities, _, _ = _phenograph.cluster(
            data,
            k=k,
            seed=seed,
            n_jobs=-1,
        )
        labels = np.asarray(communities, dtype=np.int32)
        labels[labels < 0] = 0   # outliers phenograph → cluster 0
        logger.info("[PhenoGraph] CPU done: %d clusters", len(np.unique(labels)))
        return labels

    # ── Fallback sklearn ──────────────────────────────────────────────────────

    @staticmethod
    def _run_sklearn_fallback(data: np.ndarray, params: ClusterParams) -> np.ndarray:
        if not _SKLEARN_AVAILABLE or _Agg is None:
            raise ImportError(
                "Aucun backend PhenoGraph disponible. "
                "Installez cuml+cugraph (GPU) ou phenograph: pip install phenograph"
            )
        extra = dict(params.extra or {})
        k = int(extra.get("k", 30))
        n_clusters = int(params.n_clusters or 20)

        logger.warning(
            "[PhenoGraph] Fallback sklearn: KNN k=%d + AgglomerativeClustering k=%d",
            k, n_clusters,
        )
        nn = _skNN(n_neighbors=min(k, len(data) - 1), metric="euclidean", n_jobs=-1)
        nn.fit(data)
        labels = _Agg(n_clusters=n_clusters, linkage="ward").fit_predict(data).astype(np.int32)
        logger.info("[PhenoGraph] Fallback done: %d clusters", len(np.unique(labels)))
        return labels
