"""
analysis/advanced_strategies.py — Stratégies avancées cohorte (Experiment-level).

Contient:
- HarmonyStrategy: correction batch sur matrice concaténée.
- PHATEStrategy: embedding trajectoire continu.
- PhenoGraphStrategy: clustering populations rares (avec fallback robuste).
- AdvancedCohortExecutor: exécution séquentielle orientée Experiment.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from prisma.core.models_legacy.experiment import Experiment
from .batch import (
    CohortMatrix,
    HarmonyParams,
    build_cohort_matrix,
    inject_cluster_labels,
    inject_corrected_channels,
    inject_embedding,
    run_harmony,
    store_experiment_mapping,
)
from .trajectory import PHATEParams, run_phate_embedding

logger = logging.getLogger(__name__)

try:
    import phenograph  # type: ignore

    _PHENOGRAPH_AVAILABLE = True
except ImportError:
    _PHENOGRAPH_AVAILABLE = False

try:
    import scanpy as sc  # type: ignore
    import anndata as ad  # type: ignore

    _SCANPY_AVAILABLE = True
except ImportError:
    _SCANPY_AVAILABLE = False


@runtime_checkable
class IExperimentStrategy(Protocol):
    """Contrat minimal pour une stratégie appliquée au niveau Experiment."""

    name: str

    def run_experiment(self, experiment: Experiment, **kwargs: Any) -> Experiment: ...


@dataclass
class CohortBaseParams:
    """Paramètres communs des stratégies cohorte."""

    channels: Optional[List[str]] = None
    source_masks: Optional[List[str]] = None
    batch_key: str = "batch"


@dataclass
class HarmonyStrategyParams(CohortBaseParams):
    """Paramètres de HarmonyStrategy."""

    theta: float = 2.0
    lambda_: float = 1.0
    sigma: float = 0.1
    nclust: Optional[int] = None
    max_iter_harmony: int = 10
    max_iter_kmeans: int = 20
    random_state: int = 42
    store_corrected_channels: bool = True
    corrected_channel_prefix: str = "harmony_"


@dataclass
class PHATEStrategyParams(CohortBaseParams):
    """Paramètres de PHATEStrategy."""

    embedding_name: str = "phate_2d"
    n_components: int = 2
    knn: int = 10
    decay: int = 40
    t: str | int = "auto"
    gamma: float = 1.0
    random_state: int = 42
    use_harmony_if_available: bool = True


@dataclass
class PhenoGraphStrategyParams(CohortBaseParams):
    """Paramètres de PhenoGraphStrategy."""

    clustering_name: str = "phenograph"
    k: int = 30
    random_state: int = 42
    use_harmony_if_available: bool = True
    min_rare_size: int = 30


class HarmonyStrategy:
    """Alignement batch Harmony sur données concaténées de cohorte."""

    name: str = "harmony"

    def run_experiment(self, experiment: Experiment, **kwargs: Any) -> Experiment:
        p = kwargs.get("params", HarmonyStrategyParams())
        if not isinstance(p, HarmonyStrategyParams):
            p = HarmonyStrategyParams(**p) if isinstance(p, dict) else HarmonyStrategyParams()

        cohort = build_cohort_matrix(
            experiment,
            channels=p.channels,
            source_masks=p.source_masks,
            batch_key=p.batch_key,
        )

        X_corr, harmony_meta = run_harmony(
            cohort.X,
            cohort.batch_labels,
            HarmonyParams(
                channels=p.channels,
                source_masks=p.source_masks,
                batch_key=p.batch_key,
                theta=p.theta,
                lambda_=p.lambda_,
                sigma=p.sigma,
                nclust=p.nclust,
                max_iter_harmony=p.max_iter_harmony,
                max_iter_kmeans=p.max_iter_kmeans,
                random_state=p.random_state,
            ),
            feature_names=cohort.channels,
        )

        if p.store_corrected_channels:
            inject_corrected_channels(
                experiment,
                cohort,
                X_corr,
                channel_prefix=p.corrected_channel_prefix,
            )

        store_experiment_mapping(
            experiment,
            "harmony",
            {
                "backend": harmony_meta.get("backend", "unknown"),
                "channels": cohort.channels,
                "n_rows": int(cohort.X.shape[0]),
                "n_features": int(cohort.X.shape[1]),
                "batch_key": p.batch_key,
                "unique_batches": np.unique(cohort.batch_labels).astype(str).tolist(),
                "corrected_matrix": X_corr,
                "row_sample_ids": cohort.row_sample_ids,
                "row_event_indices": cohort.row_event_indices,
            },
        )

        logger.info(
            "[HarmonyStrategy] cohort corrected: %d rows x %d features (%s).",
            X_corr.shape[0],
            X_corr.shape[1],
            harmony_meta.get("backend", "unknown"),
        )
        return experiment


class PHATEStrategy:
    """Embedding PHATE sur cohorte avec mapping retour par sample."""

    name: str = "phate"

    def run_experiment(self, experiment: Experiment, **kwargs: Any) -> Experiment:
        p = kwargs.get("params", PHATEStrategyParams())
        if not isinstance(p, PHATEStrategyParams):
            p = PHATEStrategyParams(**p) if isinstance(p, dict) else PHATEStrategyParams()

        cohort = build_cohort_matrix(
            experiment,
            channels=p.channels,
            source_masks=p.source_masks,
            batch_key=p.batch_key,
        )

        x_input = cohort.X
        harmony_payload = getattr(experiment, "analysis_results", {}).get("harmony")
        if p.use_harmony_if_available and isinstance(harmony_payload, dict):
            x_harmony = harmony_payload.get("corrected_matrix")
            same_mapping = (
                harmony_payload.get("n_rows") == cohort.X.shape[0]
                and np.array_equal(harmony_payload.get("row_sample_ids"), cohort.row_sample_ids)
                and np.array_equal(
                    harmony_payload.get("row_event_indices"), cohort.row_event_indices
                )
            )
            if isinstance(x_harmony, np.ndarray) and same_mapping:
                x_input = x_harmony
                logger.info("[PHATE] using Harmony-corrected matrix as input.")

        emb, meta = run_phate_embedding(
            x_input,
            PHATEParams(
                n_components=p.n_components,
                knn=p.knn,
                decay=p.decay,
                t=p.t,
                gamma=p.gamma,
                random_state=p.random_state,
            ),
        )

        inject_embedding(experiment, cohort, emb, p.embedding_name)

        store_experiment_mapping(
            experiment,
            "phate",
            {
                "backend": meta.get("backend", "unknown"),
                "embedding_name": p.embedding_name,
                "shape": tuple(int(v) for v in emb.shape),
                "channels": cohort.channels,
            },
        )

        logger.info(
            "[PHATE] embedding '%s' generated with backend=%s, shape=%s.",
            p.embedding_name,
            meta.get("backend", "unknown"),
            emb.shape,
        )
        return experiment


def _run_phenograph(
    X: np.ndarray,
    k: int,
    random_state: int,
) -> tuple[np.ndarray, str, Dict[str, Any]]:
    """Exécute PhenoGraph ou fallback documenté."""
    if _PHENOGRAPH_AVAILABLE:
        communities, graph, q = phenograph.cluster(
            X,
            k=k,
            seed=random_state,
        )
        labels = np.asarray(communities, dtype=np.int32)
        return labels, "phenograph", {"modularity_q": float(q)}

    if _SCANPY_AVAILABLE:
        try:
            adata = ad.AnnData(X=np.asarray(X, dtype=np.float32))
            sc.pp.neighbors(
                adata,
                n_neighbors=max(2, min(k, X.shape[0] - 1)),
                use_rep="X",
            )
            sc.tl.leiden(adata, key_added="leiden", random_state=random_state)
            cat = adata.obs["leiden"].astype("category")
            labels = cat.cat.codes.to_numpy(dtype=np.int32)
            return labels, "scanpy_leiden", {"n_clusters": int(np.unique(labels).size)}
        except Exception as exc:  # noqa: BLE001
            logger.warning("[PhenoGraph] scanpy/leiden unavailable (%s), fallback sklearn.", exc)

    # Fallback pur sklearn: graphe kNN + agglomératif contraint
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph

    n_neighbors = max(2, min(k, X.shape[0] - 1))
    connectivity = kneighbors_graph(
        X,
        n_neighbors=n_neighbors,
        mode="connectivity",
        include_self=False,
    )
    target_cluster_size = 300
    n_clusters = int(np.clip(X.shape[0] // target_cluster_size, 2, 120))

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
        connectivity=connectivity,
    )
    labels = model.fit_predict(X).astype(np.int32)
    return labels, "knn_agglomerative", {"n_clusters": int(np.unique(labels).size)}


class PhenoGraphStrategy:
    """Détection de sous-populations rares via PhenoGraph ou fallback."""

    name: str = "phenograph"

    def run_experiment(self, experiment: Experiment, **kwargs: Any) -> Experiment:
        p = kwargs.get("params", PhenoGraphStrategyParams())
        if not isinstance(p, PhenoGraphStrategyParams):
            p = PhenoGraphStrategyParams(**p) if isinstance(p, dict) else PhenoGraphStrategyParams()

        cohort = build_cohort_matrix(
            experiment,
            channels=p.channels,
            source_masks=p.source_masks,
            batch_key=p.batch_key,
        )

        x_input = cohort.X
        harmony_payload = getattr(experiment, "analysis_results", {}).get("harmony")
        if p.use_harmony_if_available and isinstance(harmony_payload, dict):
            x_harmony = harmony_payload.get("corrected_matrix")
            same_mapping = (
                harmony_payload.get("n_rows") == cohort.X.shape[0]
                and np.array_equal(harmony_payload.get("row_sample_ids"), cohort.row_sample_ids)
                and np.array_equal(
                    harmony_payload.get("row_event_indices"), cohort.row_event_indices
                )
            )
            if isinstance(x_harmony, np.ndarray) and same_mapping:
                x_input = x_harmony
                logger.info("[PhenoGraph] using Harmony-corrected matrix as input.")

        labels, backend, extras = _run_phenograph(
            x_input,
            k=p.k,
            random_state=p.random_state,
        )

        # Reindex labels to dense [0..K-1] for stabilité aval
        _, dense = np.unique(labels, return_inverse=True)
        dense_labels = dense.astype(np.int32)

        inject_cluster_labels(experiment, cohort, dense_labels, p.clustering_name)

        counts = np.bincount(dense_labels)
        rare_cluster_ids = [int(i) for i, c in enumerate(counts) if int(c) <= int(p.min_rare_size)]

        store_experiment_mapping(
            experiment,
            "phenograph",
            {
                "backend": backend,
                "label_name": p.clustering_name,
                "k": int(p.k),
                "n_clusters": int(len(counts)),
                "cluster_sizes": {int(i): int(c) for i, c in enumerate(counts)},
                "rare_cluster_ids": rare_cluster_ids,
                **extras,
            },
        )

        logger.info(
            "[PhenoGraph] labels '%s' generated (%s), n_clusters=%d, rares=%d.",
            p.clustering_name,
            backend,
            len(counts),
            len(rare_cluster_ids),
        )
        return experiment


class AdvancedCohortExecutor:
    """Exécuteur séquentiel de stratégies Experiment-level."""

    def __init__(self) -> None:
        self._strategies: List[IExperimentStrategy] = []

    def register(self, strategy: IExperimentStrategy) -> "AdvancedCohortExecutor":
        self._strategies.append(strategy)
        logger.debug("Advanced strategy registered: %s", strategy.name)
        return self

    def run(self, experiment: Experiment, **strategy_kwargs: Any) -> Experiment:
        if not self._strategies:
            logger.warning("AdvancedCohortExecutor: no strategy registered.")
            return experiment

        for strategy in self._strategies:
            key = f"{strategy.name}_params"
            kwargs: Dict[str, Any] = {}
            if key in strategy_kwargs:
                kwargs["params"] = strategy_kwargs[key]
            logger.info("[AdvancedExecutor] running '%s'...", strategy.name)
            experiment = strategy.run_experiment(experiment, **kwargs)

        return experiment

    @property
    def strategy_names(self) -> List[str]:
        return [s.name for s in self._strategies]


def _as_dict(value: Any) -> Dict[str, Any]:
    """Normalise une entrée config vers dict (sinon dict vide)."""
    return value if isinstance(value, dict) else {}


def _build_dataclass_params(default_obj: Any, override: Dict[str, Any]) -> Any:
    """Construit un objet params typé en appliquant un override filtré."""
    base = asdict(default_obj)
    for key, val in override.items():
        if key in base:
            base[key] = val
    return type(default_obj)(**base)


def build_advanced_executor_from_config(
    config: Any,
) -> Tuple[AdvancedCohortExecutor, Dict[str, Any]]:
    """
    Construit l'exécuteur avancé depuis la section YAML advanced_cohort_analysis.

    Le parsing est tolérant: sections absentes => stratégie non enregistrée.
    """
    extra = getattr(config, "_extra", {}) if config is not None else {}
    section = _as_dict(extra.get("advanced_cohort_analysis", {}))

    if not section.get("enabled", False):
        return AdvancedCohortExecutor(), {}

    shared: Dict[str, Any] = {}
    for k in ("channels", "source_masks", "batch_key"):
        if k in section:
            shared[k] = section[k]

    executor = AdvancedCohortExecutor()
    strategy_kwargs: Dict[str, Any] = {}

    harmony_cfg = _as_dict(section.get("harmony", {}))
    if harmony_cfg.get("enabled", True):
        params = _build_dataclass_params(
            HarmonyStrategyParams(),
            {**shared, **harmony_cfg},
        )
        executor.register(HarmonyStrategy())
        strategy_kwargs["harmony_params"] = params

    phate_cfg = _as_dict(section.get("phate", {}))
    if phate_cfg.get("enabled", False):
        params = _build_dataclass_params(
            PHATEStrategyParams(),
            {**shared, **phate_cfg},
        )
        executor.register(PHATEStrategy())
        strategy_kwargs["phate_params"] = params

    phenograph_cfg = _as_dict(section.get("phenograph", {}))
    if phenograph_cfg.get("enabled", False):
        params = _build_dataclass_params(
            PhenoGraphStrategyParams(),
            {**shared, **phenograph_cfg},
        )
        executor.register(PhenoGraphStrategy())
        strategy_kwargs["phenograph_params"] = params

    return executor, strategy_kwargs


def run_advanced_cohort_from_config(experiment: Experiment, config: Any) -> Experiment:
    """Exécute les stratégies avancées activées dans la config YAML."""
    executor, kwargs = build_advanced_executor_from_config(config)
    if not kwargs:
        logger.info("[AdvancedExecutor] advanced_cohort_analysis disabled or empty.")
        return experiment

    logger.info(
        "[AdvancedExecutor] config-driven execution: %s",
        executor.strategy_names,
    )
    return executor.run(experiment, **kwargs)
