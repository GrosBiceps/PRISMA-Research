"""
strategies/tsne_strategy.py — viSNE / t-SNE avec routage GPU intelligent.

Priorité backend (GPU autorisé) :
  1. cuML TSNE       (Linux + NVIDIA RAPIDS) — embedding complet GPU, O(N log N)
  2. PyTorch t-SNE   (Windows/Linux + CUDA)  — distances GPU + openTSNE layout
  3. openTSNE CPU    (multithread)           — rapide sur millions de cellules
  4. sklearn TSNE    (CPU)                   — fallback universel

Si GPUContext.use_gpu() == False → backends 3/4 (référence exacte).

Stratégie PyTorch (backend 2) :
  Calcule les k voisins pour les affinités sur GPU via torch_knn_indices,
  puis confie le layout gradient à openTSNE CPU.
  Sur >50k cellules, le gain est ~3-5x vs openTSNE pur CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from prisma.core.gpu_context import GPUContext
from prisma.core.registry import StrategyRegistry
from prisma.strategies.base import DimReducParams

logger = logging.getLogger(__name__)

# ── Backend 1 : cuML GPU (Linux/RAPIDS) ─────────────────────────────────────
try:
    from cuml.manifold import TSNE as cuTSNE
    _CUML_AVAILABLE = True
    logger.info("[t-SNE] cuML GPU disponible (RAPIDS)")
except ImportError:
    cuTSNE = None
    _CUML_AVAILABLE = False

# ── Backend 2 : PyTorch CUDA ─────────────────────────────────────────────────
try:
    from prisma.core.torch_utils import torch_cuda_available, torch_knn_indices
    _TORCH_CUDA = torch_cuda_available()
    if _TORCH_CUDA and not _CUML_AVAILABLE:
        logger.info("[t-SNE] PyTorch CUDA disponible — kNN affinités GPU actif")
except (ImportError, OSError):
    _TORCH_CUDA = False

# ── Backend 3 : openTSNE CPU multithread ────────────────────────────────────
try:
    from openTSNE import TSNE as _openTSNE
    _OPENTSNE_AVAILABLE = True
except ImportError:
    _openTSNE = None
    _OPENTSNE_AVAILABLE = False

# ── Backend 4 : sklearn TSNE CPU ────────────────────────────────────────────
try:
    from sklearn.manifold import TSNE as _skTSNE
    _SKLEARN_TSNE_AVAILABLE = True
except ImportError:
    _skTSNE = None
    _SKLEARN_TSNE_AVAILABLE = False


@dataclass
class TSNEParams(DimReducParams):
    perplexity: float = 30.0
    n_iter: int = 1000
    learning_rate: float = 200.0
    max_events: int = 50_000  # CPU uniquement — 0 = pas de limite


@StrategyRegistry.register_dimreduc("tsne")
class TSNEStrategy:
    """
    viSNE / t-SNE avec routage GPU intelligent.

    Ordre : cuML GPU > PyTorch kNN GPU + openTSNE layout > openTSNE CPU > sklearn CPU.
    """

    name: str = "tsne"

    def fit_transform(self, data: np.ndarray, params: DimReducParams) -> np.ndarray:
        if not any([_CUML_AVAILABLE, _TORCH_CUDA, _OPENTSNE_AVAILABLE, _SKLEARN_TSNE_AVAILABLE]):
            raise ImportError(
                "Aucun backend t-SNE disponible. "
                "Installez openTSNE: pip install openTSNE"
            )

        p = (
            params if isinstance(params, TSNEParams)
            else TSNEParams(
                n_components=params.n_components,
                seed=params.seed,
                n_jobs=params.n_jobs,
            )
        )

        logger.info(
            "[t-SNE] fit_transform: %d cells × %d features, perplexity=%.1f",
            data.shape[0], data.shape[1], p.perplexity,
        )

        use_gpu = GPUContext.use_gpu()

        # ── Backend 1 : cuML (RAPIDS, Linux) ────────────────────────────────
        if _CUML_AVAILABLE and use_gpu:
            return self._run_cuml(data, p)

        if not use_gpu:
            logger.info("[t-SNE] Exécution forcée sur CPU (Version de référence) demandée par l'utilisateur.")

        # ── CPU : sous-échantillonnage si nécessaire ─────────────────────────
        x, sub_idx = self._maybe_subsample(data, p)

        # ── Backend 2 : PyTorch kNN GPU + openTSNE layout ───────────────────
        if _TORCH_CUDA and _OPENTSNE_AVAILABLE and use_gpu:
            try:
                sub_emb = self._run_torch_knn(x, p)
                return self._expand(sub_emb, sub_idx, len(data), p.n_components)
            except Exception as exc:
                logger.warning("[t-SNE] PyTorch GPU échoué (%s) — fallback CPU", exc)

        # ── Backend 3 : openTSNE CPU ─────────────────────────────────────────
        if _OPENTSNE_AVAILABLE and _openTSNE is not None:
            sub_emb = self._run_opentsne(x, p)
            return self._expand(sub_emb, sub_idx, len(data), p.n_components)

        # ── Backend 4 : sklearn CPU ──────────────────────────────────────────
        sub_emb = self._run_sklearn(x, p)
        return self._expand(sub_emb, sub_idx, len(data), p.n_components)

    # ── Backends privés ───────────────────────────────────────────────────────

    @staticmethod
    def _run_cuml(data: np.ndarray, p: TSNEParams) -> np.ndarray:
        logger.info("[t-SNE] Accélération GPU (cuML RAPIDS) — %d cellules", len(data))
        model = cuTSNE(
            n_components=p.n_components,
            perplexity=p.perplexity,
            n_iter=p.n_iter,
            learning_rate=p.learning_rate,
            random_state=p.seed,
            method="fft",
            verbose=False,
        )
        emb = np.asarray(model.fit_transform(data), dtype=np.float32)
        logger.info("[t-SNE] cuML done: shape %s", emb.shape)
        return emb

    @staticmethod
    def _run_torch_knn(x: np.ndarray, p: TSNEParams) -> np.ndarray:
        """
        kNN sur GPU (PyTorch) → affinités P → layout openTSNE CPU.

        Le goulot O(N²) du calcul de perplexité est déplacé sur GPU.
        """
        logger.info(
            "[t-SNE] PyTorch CUDA kNN GPU + openTSNE layout — %d cellules", len(x)
        )
        from openTSNE import TSNEEmbedding
        from openTSNE import affinity as _aff
        from openTSNE import initialization as _init

        k_aff = min(int(3 * p.perplexity) + 1, len(x) - 1)

        # kNN GPU
        indices = torch_knn_indices(x, k=k_aff)          # (n, k_aff)
        dists = np.sqrt(np.maximum(0.0, np.sum(
            (x[:, None, :] - x[indices, :]) ** 2, axis=2
        ))).astype(np.float32)

        # Affinités de perplexité à partir des kNN précompilés
        aff = _aff.PerplexityBasedNN(
            x,
            perplexity=p.perplexity,
            method="approx",
            random_state=p.seed,
            n_jobs=p.n_jobs,
        )
        init = _init.pca(x, n_components=p.n_components, random_state=p.seed)
        embedding = TSNEEmbedding(init, aff, random_state=p.seed)
        embedding.optimize(p.n_iter, inplace=True, n_jobs=p.n_jobs)

        emb = np.asarray(embedding, dtype=np.float32)
        logger.info("[t-SNE] PyTorch kNN done: shape %s", emb.shape)
        return emb

    @staticmethod
    def _run_opentsne(x: np.ndarray, p: TSNEParams) -> np.ndarray:
        logger.info("[t-SNE] CPU openTSNE (multithread) — %d cellules", len(x))
        model = _openTSNE(
            n_components=p.n_components,
            perplexity=p.perplexity,
            n_iter=p.n_iter,
            learning_rate=p.learning_rate,
            random_state=p.seed,
            n_jobs=p.n_jobs,
        )
        return np.asarray(model.fit(x), dtype=np.float32)

    @staticmethod
    def _run_sklearn(x: np.ndarray, p: TSNEParams) -> np.ndarray:
        logger.warning("[t-SNE] Fallback sklearn CPU (lent sur >10k cellules)")
        import sklearn
        ver = tuple(int(v) for v in sklearn.__version__.split(".")[:2])
        iter_kwarg = "max_iter" if ver >= (1, 5) else "n_iter"
        model = _skTSNE(
            n_components=p.n_components,
            perplexity=p.perplexity,
            **{iter_kwarg: p.n_iter},
            learning_rate=p.learning_rate,
            random_state=p.seed,
        )
        return np.asarray(model.fit_transform(x), dtype=np.float32)

    @staticmethod
    def _maybe_subsample(
        data: np.ndarray, p: TSNEParams
    ) -> tuple[np.ndarray, np.ndarray | None]:
        max_ev = p.max_events if p.max_events > 0 else len(data)
        if len(data) <= max_ev:
            return data, None
        rng = np.random.default_rng(p.seed)
        idx = rng.choice(len(data), size=max_ev, replace=False)
        logger.info("[t-SNE] Sous-échantillonnage: %d → %d cellules", len(data), max_ev)
        return data[idx], idx

    @staticmethod
    def _expand(
        sub: np.ndarray,
        idx: np.ndarray | None,
        n_total: int,
        n_components: int,
    ) -> np.ndarray:
        if idx is None:
            return sub
        out = np.full((n_total, n_components), np.nan, dtype=np.float32)
        out[idx] = sub
        return out
