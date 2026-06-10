"""
analysis/backends.py — Factory CPU/GPU pour la réduction dimensionnelle.

Tente cuML (GPU RAPIDS) en premier, fallback umap-learn CPU.
Logging du backend effectivement utilisé.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disponibilité des backends
# ---------------------------------------------------------------------------

try:
    import cuml  # noqa: F401
    _CUML_AVAILABLE = True
    logger.info("backend: cuML (GPU) disponible")
except ImportError:
    _CUML_AVAILABLE = False

try:
    import umap as _umap_learn  # noqa: F401
    _UMAP_CPU_AVAILABLE = True
except ImportError:
    _UMAP_CPU_AVAILABLE = False

try:
    from openTSNE import TSNE as _openTSNE  # noqa: F401
    _OPENTSNE_AVAILABLE = True
except ImportError:
    _OPENTSNE_AVAILABLE = False

try:
    from sklearn.manifold import TSNE as _skTSNE  # noqa: F401
    _SKLEARN_TSNE_AVAILABLE = True
except ImportError:
    _SKLEARN_TSNE_AVAILABLE = False


# ---------------------------------------------------------------------------
# UMAP factory
# ---------------------------------------------------------------------------

def build_umap(
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Any:
    """
    Retourne un objet UMAP prêt à l'emploi (interface sklearn : fit_transform).

    Priorité : cuML (GPU) > umap-learn (CPU).

    Raises:
        ImportError: Ni cuml ni umap-learn installé.
    """
    if _CUML_AVAILABLE:
        from cuml.manifold import UMAP as cuUMAP
        logger.info("UMAP backend: cuML GPU")
        return cuUMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )

    if _UMAP_CPU_AVAILABLE:
        import umap
        logger.info("UMAP backend: umap-learn CPU")
        return umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    raise ImportError(
        "Aucun backend UMAP disponible. "
        "Installez umap-learn (CPU) ou cuml (GPU RAPIDS)."
    )


# ---------------------------------------------------------------------------
# t-SNE factory
# ---------------------------------------------------------------------------

def build_tsne(
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    learning_rate: float = 200.0,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Any:
    """
    Retourne un objet t-SNE (openTSNE ou sklearn).

    Priorité : openTSNE (multithread) > sklearn TSNE.

    Raises:
        ImportError: Ni openTSNE ni sklearn installé.
    """
    if _OPENTSNE_AVAILABLE:
        from openTSNE import TSNE as openTSNE
        logger.info("t-SNE backend: openTSNE")
        return openTSNE(
            n_components=n_components,
            perplexity=perplexity,
            n_iter=n_iter,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    if _SKLEARN_TSNE_AVAILABLE:
        import sklearn
        from sklearn.manifold import TSNE as skTSNE
        logger.info("t-SNE backend: sklearn (fallback)")
        # sklearn ≥ 1.4 renomme n_iter → max_iter
        _sklearn_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        _iter_kwarg = "max_iter" if _sklearn_ver >= (1, 4) else "n_iter"
        return skTSNE(
            n_components=n_components,
            perplexity=perplexity,
            **{_iter_kwarg: n_iter},
            learning_rate=learning_rate,
            random_state=random_state,
        )

    raise ImportError(
        "Aucun backend t-SNE disponible. "
        "Installez openTSNE ou scikit-learn."
    )


# ---------------------------------------------------------------------------
# Rapport de disponibilité (utile au démarrage du pipeline)
# ---------------------------------------------------------------------------

def backend_report() -> dict:
    """Retourne un dict résumant les backends disponibles."""
    return {
        "umap_gpu_cuml": _CUML_AVAILABLE,
        "umap_cpu": _UMAP_CPU_AVAILABLE,
        "tsne_opentsne": _OPENTSNE_AVAILABLE,
        "tsne_sklearn": _SKLEARN_TSNE_AVAILABLE,
    }
