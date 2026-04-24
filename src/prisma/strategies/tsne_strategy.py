"""
strategies/tsne_strategy.py — viSNE / t-SNE avec routage GPU intelligent.

Priorité backend :
  1. cuML TSNE (GPU NVIDIA RAPIDS) — le plus rapide sur grandes cohortes
  2. openTSNE (CPU multithread)   — rapide sur millions de cellules
  3. sklearn TSNE (CPU)           — fallback universel

La limite max_events s'applique aux backends CPU uniquement (cuML gère les
grands datasets nativement). Si subsample_idx est utilisé, les cellules non
sélectionnées sont remplies avec NaN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from prisma.core.registry import StrategyRegistry
from prisma.core.gpu_context import GPUContext
from prisma.strategies.base import DimReducParams

logger = logging.getLogger(__name__)

# ── Backend 1 : cuML GPU ────────────────────────────────────────────────────
try:
    from cuml.manifold import TSNE as cuTSNE
    _CUML_AVAILABLE = True
    logger.info("[t-SNE] cuML GPU disponible — backend RAPIDS actif")
except ImportError:
    cuTSNE = None
    _CUML_AVAILABLE = False

# ── Backend 2 : openTSNE CPU multithread ────────────────────────────────────
try:
    from openTSNE import TSNE as _openTSNE
    _OPENTSNE_AVAILABLE = True
except ImportError:
    _openTSNE = None
    _OPENTSNE_AVAILABLE = False
    if not _CUML_AVAILABLE:
        logger.warning("[t-SNE] openTSNE absent — backend sklearn CPU actif")

# ── Backend 3 : sklearn TSNE CPU ────────────────────────────────────────────
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
    viSNE / t-SNE avec routage GPU → CPU intelligent.

    Ordre de priorité : cuML GPU > openTSNE CPU > sklearn CPU.
    """

    name: str = "tsne"

    def fit_transform(self, data: np.ndarray, params: DimReducParams) -> np.ndarray:
        if not any([_CUML_AVAILABLE, _OPENTSNE_AVAILABLE, _SKLEARN_TSNE_AVAILABLE]):
            raise ImportError(
                "Aucun backend t-SNE disponible. "
                "Installez cuml (GPU) ou openTSNE (CPU): pip install openTSNE"
            )

        tsne_params = (
            params if isinstance(params, TSNEParams)
            else TSNEParams(
                n_components=params.n_components,
                seed=params.seed,
                n_jobs=params.n_jobs,
            )
        )

        logger.info(
            "[t-SNE] fit_transform: %d cells × %d features, perplexity=%.1f",
            data.shape[0], data.shape[1], tsne_params.perplexity,
        )

        # GPU cuML : pas de limite max_events, pas de subsample
        if _CUML_AVAILABLE and cuTSNE is not None and GPUContext.use_gpu():
            return self._run_cuml(data, tsne_params)
        if _CUML_AVAILABLE and not GPUContext.use_gpu():
            logger.info("[t-SNE] Exécution forcée sur CPU (Version de référence) demandée par l'utilisateur.")

        # CPU : sous-échantillonnage si nécessaire
        x, subsample_idx = self._maybe_subsample(data, tsne_params)

        if _OPENTSNE_AVAILABLE and _openTSNE is not None:
            sub_embedding = self._run_opentsne(x, tsne_params)
        else:
            sub_embedding = self._run_sklearn(x, tsne_params)

        embedding = self._expand(sub_embedding, subsample_idx, len(data), tsne_params.n_components)
        logger.info("[t-SNE] done: embedding shape %s", embedding.shape)
        return embedding

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
            method="fft",          # cuML FMM/FFT — O(N log N)
            verbose=False,
        )
        embedding = np.asarray(model.fit_transform(data), dtype=np.float32)
        logger.info("[t-SNE] GPU done: shape %s", embedding.shape)
        return embedding

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
        logger.warning("[t-SNE] GPU/openTSNE non disponible — sklearn CPU (lent sur >10k cellules)")
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
        logger.info(
            "[t-SNE] Sous-échantillonnage CPU: %d → %d cellules", len(data), max_ev
        )
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
