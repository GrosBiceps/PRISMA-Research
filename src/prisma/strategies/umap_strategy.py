"""
strategies/umap_strategy.py — UMAP avec routage GPU intelligent.

Priorité backend (GPU autorisé) :
  1. cuML UMAP      (Linux + NVIDIA RAPIDS)  — embedding complet sur GPU
  2. PyTorch kNN    (Windows/Linux + CUDA)   — kNN GPU + layout umap-learn CPU
  3. umap-learn CPU (universel)              — implémentation de référence

Si GPUContext.use_gpu() == False → backend 3 directement (référence exacte).

Stratégie PyTorch (backend 2) :
  Calcule les k voisins sur GPU via torch_knn_indices(), construit le graphe
  kNN, puis confie le layout UMAP à umap-learn en lui injectant le graphe
  précompilé — évite la partie O(N²) du kNN sur CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from prisma.core.registry import StrategyRegistry
from prisma.core.gpu_context import GPUContext
from prisma.strategies.base import DimReducParams

logger = logging.getLogger(__name__)

# ── Backend 1 : cuML GPU (Linux/RAPIDS) ─────────────────────────────────────
try:
    from cuml.manifold import UMAP as cuUMAP
    _CUML_AVAILABLE = True
    logger.info("[UMAP] cuML GPU disponible (RAPIDS)")
except ImportError:
    cuUMAP = None
    _CUML_AVAILABLE = False

# ── Backend 2 : PyTorch kNN GPU ──────────────────────────────────────────────
try:
    from prisma.core.torch_utils import torch_cuda_available, torch_knn_indices
    _TORCH_CUDA = torch_cuda_available()
    if _TORCH_CUDA and not _CUML_AVAILABLE:
        logger.info("[UMAP] PyTorch CUDA disponible — kNN GPU actif")
except ImportError:
    _TORCH_CUDA = False

# ── Backend 3 : umap-learn CPU ───────────────────────────────────────────────
try:
    import umap as umap_learn
    _UMAP_AVAILABLE = True
except ImportError:
    umap_learn = None
    _UMAP_AVAILABLE = False
    logger.warning("[UMAP] umap-learn absent — UMAPStrategy indisponible")


@dataclass
class UMAPParams(DimReducParams):
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"


@StrategyRegistry.register_dimreduc("umap")
class UMAPStrategy:
    """
    UMAP avec routage GPU intelligent.

    Ordre : cuML GPU > PyTorch kNN GPU + umap-learn layout > umap-learn CPU.
    """

    name: str = "umap"

    def fit_transform(self, data: np.ndarray, params: DimReducParams) -> np.ndarray:
        if not _UMAP_AVAILABLE and not _CUML_AVAILABLE:
            raise ImportError(
                "Aucun backend UMAP disponible. "
                "Installez umap-learn: pip install umap-learn"
            )

        p = (
            params if isinstance(params, UMAPParams)
            else UMAPParams(
                n_components=params.n_components,
                seed=params.seed,
                n_jobs=params.n_jobs,
            )
        )

        logger.info(
            "[UMAP] fit_transform: %d cells × %d features, n_neighbors=%d, min_dist=%.2f",
            data.shape[0], data.shape[1], p.n_neighbors, p.min_dist,
        )

        use_gpu = GPUContext.use_gpu()

        # ── Backend 1 : cuML (RAPIDS, Linux) ────────────────────────────────
        if _CUML_AVAILABLE and use_gpu:
            return self._run_cuml(data, p)

        if not use_gpu:
            logger.info("[UMAP] Exécution forcée sur CPU (Version de référence) demandée par l'utilisateur.")
            return self._run_cpu(data, p)

        # ── Backend 2 : PyTorch kNN GPU + umap-learn layout ─────────────────
        if _TORCH_CUDA and _UMAP_AVAILABLE:
            try:
                return self._run_torch_knn(data, p)
            except Exception as exc:
                logger.warning("[UMAP] PyTorch kNN GPU échoué (%s) — fallback CPU", exc)

        # ── Backend 3 : umap-learn CPU ───────────────────────────────────────
        logger.warning("[UMAP] GPU non disponible. Exécution sur CPU (référence).")
        return self._run_cpu(data, p)

    # ── Backends privés ───────────────────────────────────────────────────────

    @staticmethod
    def _run_cuml(data: np.ndarray, p: UMAPParams) -> np.ndarray:
        logger.info("[UMAP] Accélération GPU (cuML RAPIDS) — %d cellules", len(data))
        model = cuUMAP(
            n_components=p.n_components,
            n_neighbors=p.n_neighbors,
            min_dist=p.min_dist,
            random_state=p.seed,
        )
        emb = np.asarray(model.fit_transform(data), dtype=np.float32)
        logger.info("[UMAP] cuML done: shape %s", emb.shape)
        return emb

    @staticmethod
    def _run_torch_knn(data: np.ndarray, p: UMAPParams) -> np.ndarray:
        logger.info(
            "[UMAP] PyTorch CUDA kNN GPU + umap-learn layout — %d cellules", len(data)
        )
        import scipy.sparse

        # kNN sur GPU → graphe précompilé pour umap-learn
        k = p.n_neighbors
        indices = torch_knn_indices(data, k=k)           # (n, k)
        n = len(data)

        # Construction matrice kNN sparse (symétrique)
        rows = np.repeat(np.arange(n), k)
        cols = indices.ravel()
        vals = np.ones(len(rows), dtype=np.float32)
        knn_graph = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
        knn_graph = (knn_graph + knn_graph.T)
        knn_graph.data[:] = 1.0

        # umap-learn accepte precomputed_knn comme (indices, distances)
        # On passe juste les indices — distances uniformes (1.0)
        dists = np.ones_like(indices, dtype=np.float32)

        reducer = umap_learn.UMAP(
            n_components=p.n_components,
            n_neighbors=k,
            min_dist=p.min_dist,
            metric="precomputed",
            precomputed_knn=(indices, dists),
            random_state=p.seed,
            n_jobs=p.n_jobs,
        )
        emb = np.asarray(reducer.fit_transform(data), dtype=np.float32)
        logger.info("[UMAP] PyTorch kNN done: shape %s", emb.shape)
        return emb

    @staticmethod
    def _run_cpu(data: np.ndarray, p: UMAPParams) -> np.ndarray:
        reducer = umap_learn.UMAP(
            n_components=p.n_components,
            n_neighbors=p.n_neighbors,
            min_dist=p.min_dist,
            metric=p.metric,
            random_state=p.seed,
            n_jobs=p.n_jobs,
        )
        emb = np.asarray(reducer.fit_transform(data), dtype=np.float32)
        logger.info("[UMAP] CPU done: shape %s", emb.shape)
        return emb
