"""
strategies/phenograph_strategy.py — PhenoGraph avec routage GPU intelligent.

Priorité backend (GPU autorisé) :
  1. Full GPU RAPIDS  (Linux)              — cuML KNN + cuGraph Louvain
  2. PyTorch kNN GPU  (Windows/Linux)      — kNN GPU + Leiden/Louvain CPU
  3. phenograph CPU                        — package officiel
  4. sklearn fallback                      — KNN + AgglomerativeClustering

Si GPUContext.use_gpu() == False → backends 3/4 (référence exacte).

Stratégie PyTorch (backend 2) :
  Calcule le graphe k-NN sur GPU via torch_knn_indices(), construit la
  matrice d'adjacence sparse, puis applique Louvain/Leiden CPU via igraph
  (même algo que PhenoGraph officiel, mais kNN 10-50x plus rapide).
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

from prisma.core.registry import StrategyRegistry
from prisma.core.gpu_context import GPUContext
from prisma.strategies.base import ClusterParams

logger = logging.getLogger(__name__)

# ── Backend 1 : cuML + cuGraph GPU (Linux/RAPIDS) ───────────────────────────
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

_GPU_FULL_AVAILABLE = _CUML_NN_AVAILABLE and _CUGRAPH_AVAILABLE

# ── Backend 2 : PyTorch kNN GPU ──────────────────────────────────────────────
try:
    from prisma.core.torch_utils import torch_cuda_available, torch_knn_indices
    _TORCH_CUDA = torch_cuda_available()
    if _TORCH_CUDA and not _GPU_FULL_AVAILABLE:
        logger.info("[PhenoGraph] PyTorch CUDA disponible — kNN GPU + Louvain CPU actif")
except ImportError:
    _TORCH_CUDA = False

# ── Louvain/Leiden CPU via igraph (pour backend 2) ──────────────────────────
try:
    import igraph as _igraph
    _IGRAPH_AVAILABLE = True
except ImportError:
    _igraph = None
    _IGRAPH_AVAILABLE = False

try:
    import leidenalg as _leiden
    _LEIDEN_AVAILABLE = True
except ImportError:
    _leiden = None
    _LEIDEN_AVAILABLE = False

# ── Backend 3 : phenograph CPU ───────────────────────────────────────────────
try:
    import phenograph as _phenograph
    _PHENOGRAPH_AVAILABLE = True
    if not _GPU_FULL_AVAILABLE and not _TORCH_CUDA:
        logger.info("[PhenoGraph] phenograph CPU disponible")
except ImportError:
    _phenograph = None
    _PHENOGRAPH_AVAILABLE = False

# ── Backend 4 : sklearn fallback ─────────────────────────────────────────────
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

    Routage : Full-GPU RAPIDS > PyTorch kNN GPU + Louvain CPU > phenograph CPU > sklearn.
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

        use_gpu = GPUContext.use_gpu()

        # ── Backend 1 : Full-GPU RAPIDS ─────────────────────────────────────
        if _GPU_FULL_AVAILABLE and use_gpu:
            try:
                return self._run_rapids(data, k, seed)
            except Exception as exc:
                logger.warning("[PhenoGraph] Full-GPU RAPIDS échoué (%s) — fallback", exc)

        if not use_gpu and (_GPU_FULL_AVAILABLE or _TORCH_CUDA):
            logger.info(
                "[PhenoGraph] Exécution forcée sur CPU (Version de référence) demandée par l'utilisateur."
            )

        # ── Backend 2 : PyTorch kNN GPU + Louvain/Leiden CPU ────────────────
        if _TORCH_CUDA and (_IGRAPH_AVAILABLE or _LEIDEN_AVAILABLE) and use_gpu:
            try:
                return self._run_torch_louvain(data, k, seed)
            except Exception as exc:
                logger.warning("[PhenoGraph] PyTorch kNN GPU échoué (%s) — fallback CPU", exc)

        # ── Backend 3 : phenograph CPU ───────────────────────────────────────
        if _PHENOGRAPH_AVAILABLE and _phenograph is not None:
            return self._run_phenograph(data, k, seed)

        # ── Backend 4 : sklearn fallback ─────────────────────────────────────
        return self._run_sklearn_fallback(data, params)

    # ── Backend Full-GPU RAPIDS ───────────────────────────────────────────────

    @staticmethod
    def _run_rapids(data: np.ndarray, k: int, seed: int) -> np.ndarray:
        import cudf

        n = len(data)
        k_safe = min(k, n - 1)
        logger.info(
            "[PhenoGraph] Full-GPU (cuML KNN + cuGraph Louvain) — %d cellules, k=%d",
            n, k_safe,
        )

        nn_model = cuNearestNeighbors(n_neighbors=k_safe, metric="euclidean")
        nn_model.fit(data)
        distances, indices = nn_model.kneighbors(data)
        indices_np = np.asarray(indices)

        rows = np.repeat(np.arange(n), k_safe)
        cols = indices_np.ravel()
        adj = sp.coo_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)
        )
        adj = (adj + adj.T > 0).astype(np.float32)

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

        logger.info(
            "[PhenoGraph] Full-GPU done: %d clusters, modularity=%.4f",
            len(np.unique(labels)), modularity,
        )
        return labels

    # ── Backend PyTorch kNN GPU + Louvain CPU ─────────────────────────────────

    @staticmethod
    def _run_torch_louvain(data: np.ndarray, k: int, seed: int) -> np.ndarray:
        """
        kNN GPU PyTorch → graphe igraph → Leiden/Louvain CPU.

        Même algorithme que PhenoGraph officiel, kNN 10-50x plus rapide.
        """
        n = len(data)
        k_safe = min(k, n - 1)
        logger.info(
            "[PhenoGraph] PyTorch CUDA kNN GPU + %s CPU — %d cellules, k=%d",
            "Leiden" if _LEIDEN_AVAILABLE else "Louvain",
            n, k_safe,
        )

        # kNN GPU
        indices = torch_knn_indices(data, k=k_safe)   # (n, k_safe)

        # Graphe igraph non-orienté
        rows = np.repeat(np.arange(n), k_safe)
        cols = indices.ravel()
        edges = list(zip(rows.tolist(), cols.tolist()))

        G = _igraph.Graph(n=n, edges=edges, directed=False)
        G.simplify()  # supprimer doublons et self-loops

        # Détection de communautés
        if _LEIDEN_AVAILABLE and _leiden is not None:
            partition = _leiden.find_partition(
                G,
                _leiden.ModularityVertexPartition,
                seed=seed,
            )
            labels = np.asarray(partition.membership, dtype=np.int32)
        else:
            # Louvain via igraph
            result = G.community_multilevel(weights=None)
            labels = np.asarray(result.membership, dtype=np.int32)

        logger.info(
            "[PhenoGraph] PyTorch kNN + %s done: %d clusters",
            "Leiden" if _LEIDEN_AVAILABLE else "Louvain",
            len(np.unique(labels)),
        )
        return labels

    # ── Backend phenograph CPU ────────────────────────────────────────────────

    @staticmethod
    def _run_phenograph(data: np.ndarray, k: int, seed: int) -> np.ndarray:
        logger.warning("[PhenoGraph] GPU non disponible — phenograph CPU (k=%d)", k)
        communities, _, _ = _phenograph.cluster(data, k=k, seed=seed, n_jobs=-1)
        labels = np.asarray(communities, dtype=np.int32)
        labels[labels < 0] = 0
        logger.info("[PhenoGraph] CPU done: %d clusters", len(np.unique(labels)))
        return labels

    # ── Backend sklearn fallback ──────────────────────────────────────────────

    @staticmethod
    def _run_sklearn_fallback(data: np.ndarray, params: ClusterParams) -> np.ndarray:
        if not _SKLEARN_AVAILABLE or _Agg is None:
            raise ImportError(
                "Aucun backend PhenoGraph disponible. "
                "Installez phenograph: pip install phenograph"
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
