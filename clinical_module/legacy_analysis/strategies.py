"""
analysis/strategies.py — Moteur d'analyse haute dimension avec Strategy Pattern.

Hiérarchie :
  IAnalysisStrategy (Protocol)
  └── BaseAnalysisStrategy (ABC, optionnel — accès Sample + downsampling)
      ├── UMAPStrategy        → embedding UMAP (GPU/CPU) → Sample.add_embedding()
      ├── TSNEStrategy        → embedding t-SNE             → Sample.add_embedding()
      └── FlowSOMStrategy     → clustering SOM + méta       → Sample.add_cluster_labels()

ResearchPipelineExecutor orchestre l'application séquentielle des stratégies
sur tous les échantillons d'une expérience.

Extensible : implémenter IAnalysisStrategy et enregistrer via
ResearchPipelineExecutor.register().
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np

from prisma.analysis.downsampling import expand_to_full, random_downsample_df
from prisma.core.models_legacy.experiment import Experiment
from prisma.core.models_legacy.sample import Sample

from .backends import build_tsne, build_umap

logger = logging.getLogger(__name__)

_DOWNSAMPLE_THRESHOLD = 100_000


# ---------------------------------------------------------------------------
# IAnalysisStrategy — Interface (Protocol)
# ---------------------------------------------------------------------------


@runtime_checkable
class IAnalysisStrategy(Protocol):
    """
    Interface commune pour toute stratégie d'analyse haute dimension.

    Une stratégie reçoit un Sample, produit un embedding ou des labels,
    et les stocke directement dans le Sample via ses méthodes dédiées.
    """

    name: str

    def run(self, sample: Sample, **kwargs: Any) -> Sample:
        """
        Exécute l'analyse sur sample et retourne le même sample modifié.

        Args:
            sample: Échantillon cytométrique avec événements pré-gatés.
            **kwargs: Paramètres spécifiques à la stratégie.

        Returns:
            Le même Sample enrichi (embeddings / cluster_assignments).
        """
        ...


# ---------------------------------------------------------------------------
# BaseAnalysisStrategy — Classe de base concrète (optionnelle)
# ---------------------------------------------------------------------------


class BaseAnalysisStrategy(ABC):
    """
    Base concrète offrant :
    - sélection de canaux
    - récupération des données pré-gatées
    - downsampling si > _DOWNSAMPLE_THRESHOLD événements
    - seed reproductible
    """

    name: str = "base"

    def _prepare_data(
        self,
        sample: Sample,
        channels: Optional[List[str]],
        seed: int,
        max_events: int = _DOWNSAMPLE_THRESHOLD,
    ):
        """
        Extrait et sous-échantillonne les données pré-gatées.

        Returns:
            Tuple (data_np, downsample_result_or_None, used_channels, gate_indices)

            gate_indices: indices dans sample.events des cellules retenues par
            le gating. Permet de réindexer l'embedding vers N_events complet.
            None si aucun masque actif (toutes les cellules retenues).
        """
        used_channels = channels or sample.channels

        # Calcul des indices de gating pour pouvoir réexpanser vers N_events
        if sample.masks:
            combined = np.ones(sample.n_events, dtype=bool)
            for m in sample.masks.values():
                combined &= m
            gate_indices: Optional[np.ndarray] = np.where(combined)[0]
        else:
            gate_indices = None

        df = sample.get_pre_gated_data(channels=used_channels)
        n = len(df)

        if n == 0:
            raise ValueError(
                f"[{self.name}] Sample '{sample.sample_id}' : 0 événements après gating."
            )

        if n > max_events:
            ds = random_downsample_df(df, max_events=max_events, seed=seed)
            return ds.data.astype(np.float32), ds, used_channels, gate_indices

        data = df.to_numpy(dtype=np.float32)
        return data, None, used_channels, gate_indices

    def _expand_to_full_events(
        self,
        values: np.ndarray,
        gate_indices: Optional[np.ndarray],
        n_events: int,
        fill_value: float = np.nan,
    ) -> np.ndarray:
        """
        Projette values (calculés sur cellules pré-gatées) vers N_events complet.

        Si gate_indices est None, toutes les cellules étaient incluses → retourne
        values tel quel (longueur déjà correcte).
        """
        if gate_indices is None:
            return values

        if values.ndim == 1:
            full = np.full(n_events, fill_value, dtype=values.dtype)
            full[gate_indices] = values
        else:
            full = np.full((n_events, values.shape[1]), fill_value, dtype=values.dtype)
            full[gate_indices] = values
        return full

    @abstractmethod
    def run(self, sample: Sample, **kwargs: Any) -> Sample:
        """Implémentation obligatoire dans chaque sous-classe."""
        ...


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------


@dataclass
class UMAPStrategyParams:
    channels: Optional[List[str]] = None
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    seed: int = 42
    n_jobs: int = -1
    embedding_name: str = "umap_2d"
    max_events: int = _DOWNSAMPLE_THRESHOLD


@dataclass
class TSNEStrategyParams:
    channels: Optional[List[str]] = None
    n_components: int = 2
    perplexity: float = 30.0
    n_iter: int = 1000
    learning_rate: float = 200.0
    seed: int = 42
    n_jobs: int = -1
    embedding_name: str = "tsne_2d"
    max_events: int = _DOWNSAMPLE_THRESHOLD


@dataclass
class FlowSOMStrategyParams:
    channels: Optional[List[str]] = None
    xdim: int = 10
    ydim: int = 10
    n_metaclusters: int = 20
    rlen: int = 10
    mst: int = 1
    alpha: tuple = (0.05, 0.01)
    mad_allowed: int = 4
    seed: int = 42
    som_label: str = "flowsom_nodes"
    meta_label: str = "flowsom_meta"
    compute_centers: bool = True
    store_codes: bool = True
    store_mfi: bool = True
    max_events: int = _DOWNSAMPLE_THRESHOLD
    # Consensus metaclustering params (backend saeyslab ConsensusCluster)
    consensus_H: int = 100
    consensus_resample: float = 0.9
    consensus_linkage: str = "average"


@dataclass
class PreGatingParams:
    """Paramètres de la stratégie de pré-gating explicite."""

    source_masks: Optional[List[str]] = None
    output_mask_name: str = "pre_gating"
    require_masks: bool = True
    min_pass_fraction: float = 0.01


# ---------------------------------------------------------------------------
# UMAPStrategy
# ---------------------------------------------------------------------------


class UMAPStrategy(BaseAnalysisStrategy):
    """
    Embedding UMAP sur données pré-gatées.

    - Sélection de canaux configurable.
    - Downsampling reproductible si > max_events.
    - GPU (cuML) si disponible, sinon umap-learn CPU.
    - Résultat stocké via sample.add_embedding(embedding_name, …).
    - Si downsampling : les événements non sélectionnés reçoivent NaN.
    """

    name: str = "umap"

    def run(self, sample: Sample, **kwargs: Any) -> Sample:
        p = kwargs.get("params", UMAPStrategyParams())
        if not isinstance(p, UMAPStrategyParams):
            p = UMAPStrategyParams(**p) if isinstance(p, dict) else UMAPStrategyParams()

        data, ds, channels, gate_idx = self._prepare_data(sample, p.channels, p.seed, p.max_events)

        logger.info(
            "[UMAP] '%s': %d × %d canaux (n_neighbors=%d, min_dist=%.2f)",
            sample.sample_id,
            data.shape[0],
            data.shape[1],
            p.n_neighbors,
            p.min_dist,
        )

        reducer = build_umap(
            n_components=p.n_components,
            n_neighbors=p.n_neighbors,
            min_dist=p.min_dist,
            metric=p.metric,
            random_state=p.seed,
            n_jobs=p.n_jobs,
        )
        embedding = np.array(reducer.fit_transform(data), dtype=np.float32)

        # Expansion downsampling d'abord (sur les N_gated), puis gating (vers N_events)
        if ds is not None and ds.was_downsampled:
            embedding = expand_to_full(embedding, ds, fill_value=np.nan)

        embedding = self._expand_to_full_events(embedding, gate_idx, sample.n_events)

        sample.add_embedding(p.embedding_name, embedding)
        logger.info(
            "[UMAP] '%s': embedding '%s' stocké %s",
            sample.sample_id,
            p.embedding_name,
            embedding.shape,
        )
        return sample


# ---------------------------------------------------------------------------
# TSNEStrategy
# ---------------------------------------------------------------------------


class TSNEStrategy(BaseAnalysisStrategy):
    """
    Embedding t-SNE sur données pré-gatées.

    - openTSNE si disponible (multithread), sinon sklearn.
    - Downsampling reproductible si > max_events.
    - Résultat stocké via sample.add_embedding(embedding_name, …).
    - Les événements non sélectionnés reçoivent NaN si downsampled.
    """

    name: str = "tsne"

    def run(self, sample: Sample, **kwargs: Any) -> Sample:
        p = kwargs.get("params", TSNEStrategyParams())
        if not isinstance(p, TSNEStrategyParams):
            p = TSNEStrategyParams(**p) if isinstance(p, dict) else TSNEStrategyParams()

        data, ds, channels, gate_idx = self._prepare_data(sample, p.channels, p.seed, p.max_events)

        logger.info(
            "[t-SNE] '%s': %d × %d canaux (perplexity=%.1f, n_iter=%d)",
            sample.sample_id,
            data.shape[0],
            data.shape[1],
            p.perplexity,
            p.n_iter,
        )

        reducer = build_tsne(
            n_components=p.n_components,
            perplexity=p.perplexity,
            n_iter=p.n_iter,
            learning_rate=p.learning_rate,
            random_state=p.seed,
            n_jobs=p.n_jobs,
        )

        # openTSNE retourne un objet TSNEEmbedding, pas un ndarray
        raw = (
            reducer.fit(data)
            if hasattr(reducer, "fit") and not hasattr(reducer, "fit_transform")
            else reducer.fit_transform(data)
        )
        embedding = np.array(raw, dtype=np.float32)

        if ds is not None and ds.was_downsampled:
            embedding = expand_to_full(embedding, ds, fill_value=np.nan)

        embedding = self._expand_to_full_events(embedding, gate_idx, sample.n_events)

        sample.add_embedding(p.embedding_name, embedding)
        logger.info(
            "[t-SNE] '%s': embedding '%s' stocké %s",
            sample.sample_id,
            p.embedding_name,
            embedding.shape,
        )
        return sample


# ---------------------------------------------------------------------------
# FlowSOM backend detection
# ---------------------------------------------------------------------------

try:
    import flowsom as _flowsom_lib

    _FLOWSOM_AVAILABLE = True
    logger.info("flowsom (saeyslab) disponible — backend natif actif")
except ImportError:
    _FLOWSOM_AVAILABLE = False
    logger.warning("flowsom non installé — fallback MiniBatchKMeans+Agglomerative")


# ---------------------------------------------------------------------------
# FlowSOMStrategy — saeyslab natif + proxy sklearn fallback
# ---------------------------------------------------------------------------


class FlowSOMStrategy(BaseAnalysisStrategy):
    """
    Clustering FlowSOM sur données pré-gatées.

    Backend prioritaire : saeyslab flowsom (FlowSOM + ConsensusCluster).
    Fallback si absent      : MiniBatchKMeans (SOM proxy) + AgglomerativeClustering.

    Résultats stockés dans Sample :
      cluster_assignments[som_label]  → labels nœuds SOM  (int32, -1 = exclu)
      cluster_assignments[meta_label] → labels métaclusters (int32, -1 = exclu)

    results["flowsom_codes"]    → ndarray (n_nodes × n_channels) — codebook SOM
    results["flowsom_mfi"]      → dict {metacluster_id: median_expression}
    results["flowsom_counts"]   → dict {metacluster_id: n_cells}
    results["flowsom_backend"]  → "saeyslab" | "proxy"
    results["flowsom_node_meta"] → ndarray (n_nodes,) — quel méta pour chaque nœud
    """

    name: str = "flowsom"

    def __init__(self) -> None:
        self._fsom: Any = None
        self._node_centers_: Optional[np.ndarray] = None
        self._meta_labels_per_node_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # run() public
    # ------------------------------------------------------------------

    def run(self, sample: Sample, **kwargs: Any) -> Sample:
        p = kwargs.get("params", FlowSOMStrategyParams())
        if not isinstance(p, FlowSOMStrategyParams):
            p = FlowSOMStrategyParams(**p) if isinstance(p, dict) else FlowSOMStrategyParams()

        data, ds, channels, gate_idx = self._prepare_data(sample, p.channels, p.seed, p.max_events)

        if _FLOWSOM_AVAILABLE:
            return self._run_saeyslab(sample, data, ds, channels, gate_idx, p)
        return self._run_proxy(sample, data, ds, channels, gate_idx, p)

    # ------------------------------------------------------------------
    # Backend saeyslab flowsom
    # ------------------------------------------------------------------

    def _run_saeyslab(
        self,
        sample: Sample,
        data: np.ndarray,
        ds: Any,
        channels: List[str],
        gate_idx: Optional[np.ndarray],
        p: FlowSOMStrategyParams,
    ) -> Sample:
        import anndata
        import flowsom as fs

        n_cells, n_feat = data.shape
        logger.info(
            "[FlowSOM/saeyslab] '%s': %d × %d, SOM=%dx%d, méta=%d, rlen=%d",
            sample.sample_id,
            n_cells,
            n_feat,
            p.xdim,
            p.ydim,
            p.n_metaclusters,
            p.rlen,
        )

        cols = [f"f{i}" for i in range(n_feat)]
        adata = anndata.AnnData(X=data.astype(np.float32))
        adata.var_names = cols

        fsom = fs.FlowSOM(
            adata,
            cols_to_use=cols,
            xdim=p.xdim,
            ydim=p.ydim,
            n_clusters=p.n_metaclusters,
            rlen=p.rlen,
            mst=p.mst,
            alpha=p.alpha,
            seed=p.seed,
        )
        self._fsom = fsom

        # — Labels nœuds SOM (n_cells_ds,)
        cell_obs = fsom.get_cell_data().obs
        node_labels = np.array(cell_obs["clustering"], dtype=np.int32)
        metacluster_labels = np.array(cell_obs["metaclustering"], dtype=np.int32)

        # — Codebook SOM (n_nodes × n_features)
        cluster_adata = fsom.get_cluster_data()
        codes = cluster_adata.obsm.get("codes", None)
        if codes is not None:
            codes = np.array(codes, dtype=np.float32)

        # — MFI par métacluster (médiane X par métacluster sur données cluster)
        node_meta = np.array(cluster_adata.obs.get("metaclustering", []), dtype=np.int32)

        self._node_centers_ = codes
        self._meta_labels_per_node_ = node_meta

        return self._finalise(
            sample,
            data,
            ds,
            gate_idx,
            p,
            node_labels,
            metacluster_labels,
            codes,
            node_meta,
            backend="saeyslab",
        )

    # ------------------------------------------------------------------
    # Backend proxy (MiniBatchKMeans + ConsensusCluster ou Agglomerative)
    # ------------------------------------------------------------------

    def _run_proxy(
        self,
        sample: Sample,
        data: np.ndarray,
        ds: Any,
        channels: List[str],
        gate_idx: Optional[np.ndarray],
        p: FlowSOMStrategyParams,
    ) -> Sample:
        from sklearn.cluster import MiniBatchKMeans

        n_nodes = p.xdim * p.ydim
        n_cells, n_feat = data.shape
        logger.info(
            "[FlowSOM/proxy] '%s': %d × %d, SOM=%dx%d (%d nœuds), méta=%d",
            sample.sample_id,
            n_cells,
            n_feat,
            p.xdim,
            p.ydim,
            n_nodes,
            p.n_metaclusters,
        )

        # — Étape 1 : SOM via MiniBatchKMeans
        som = MiniBatchKMeans(
            n_clusters=n_nodes,
            random_state=p.seed,
            n_init=3,
            max_iter=300,
        )
        node_labels = som.fit_predict(data).astype(np.int32)
        codes = som.cluster_centers_.astype(np.float32)
        self._node_centers_ = codes

        # — Étape 2 : Métaclustering (ConsensusCluster si possible, sinon Agglomerative)
        n_meta = min(p.n_metaclusters, n_nodes)
        node_meta = self._metacluster_proxy(codes, n_meta, p)
        self._meta_labels_per_node_ = node_meta

        metacluster_labels = node_meta[node_labels]

        return self._finalise(
            sample,
            data,
            ds,
            gate_idx,
            p,
            node_labels,
            metacluster_labels,
            codes,
            node_meta,
            backend="proxy",
        )

    def _metacluster_proxy(
        self,
        codes: np.ndarray,
        n_meta: int,
        p: FlowSOMStrategyParams,
    ) -> np.ndarray:
        """
        Métaclustering sur les codes SOM.

        Tente ConsensusCluster (saeyslab style) si sklearn disponible,
        sinon AgglomerativeClustering standard.
        """
        try:
            from flowsom.models import ConsensusCluster

            logger.info(
                "[FlowSOM/proxy] ConsensusCluster (H=%d, resample=%.2f)",
                p.consensus_H,
                p.consensus_resample,
            )
            cc = ConsensusCluster(
                n_clusters=n_meta,
                H=p.consensus_H,
                resample_proportion=p.consensus_resample,
                linkage=p.consensus_linkage,
            )
            return cc.fit_predict(codes).astype(np.int32)
        except Exception:
            pass

        from sklearn.cluster import AgglomerativeClustering

        logger.info("[FlowSOM/proxy] AgglomerativeClustering (n=%d)", n_meta)
        agg = AgglomerativeClustering(n_clusters=n_meta)
        return agg.fit_predict(codes).astype(np.int32)

    # ------------------------------------------------------------------
    # Finalisation commune : expansion + stockage dans Sample
    # ------------------------------------------------------------------

    def _finalise(
        self,
        sample: Sample,
        data: np.ndarray,
        ds: Any,
        gate_idx: Optional[np.ndarray],
        p: FlowSOMStrategyParams,
        node_labels: np.ndarray,
        metacluster_labels: np.ndarray,
        codes: Optional[np.ndarray],
        node_meta: np.ndarray,
        backend: str,
    ) -> Sample:
        # Sauvegarder les labels bruts (alignés avec data) avant toute expansion
        raw_meta_labels = metacluster_labels.copy()

        # — Expansion downsampling (N_ds → N_gated)
        if ds is not None and ds.was_downsampled:
            node_labels = expand_to_full(
                node_labels.astype(np.float32), ds, fill_value=-1.0
            ).astype(np.int32)
            metacluster_labels = expand_to_full(
                metacluster_labels.astype(np.float32), ds, fill_value=-1.0
            ).astype(np.int32)

        # — Expansion gating (N_gated → N_events)
        node_labels_full = self._expand_to_full_events(
            node_labels, gate_idx, sample.n_events, fill_value=-1.0
        ).astype(np.int32)
        metacluster_labels_full = self._expand_to_full_events(
            metacluster_labels, gate_idx, sample.n_events, fill_value=-1.0
        ).astype(np.int32)

        sample.add_cluster_labels(p.som_label, node_labels_full)
        sample.add_cluster_labels(p.meta_label, metacluster_labels_full)

        sample.results["flowsom_backend"] = backend
        sample.results["flowsom_node_meta"] = node_meta.tolist() if node_meta is not None else []

        if p.store_codes and codes is not None:
            sample.results["flowsom_codes"] = codes.tolist()

        if p.compute_centers:
            # raw_meta_labels et data sont alignés (N cellules avant expansion)
            self._store_metacluster_mfi(sample, data, raw_meta_labels)

        if p.store_mfi:
            counts = {}
            valid_meta_for_counts = raw_meta_labels[raw_meta_labels >= 0]
            for m in np.unique(valid_meta_for_counts):
                counts[int(m)] = int(np.sum(valid_meta_for_counts == m))
            sample.results["flowsom_counts"] = counts

        logger.info(
            "[FlowSOM/%s] '%s': labels '%s' + '%s' stockés | backend=%s",
            backend,
            sample.sample_id,
            p.som_label,
            p.meta_label,
            backend,
        )
        return sample

    def _store_metacluster_mfi(
        self,
        sample: Sample,
        data: np.ndarray,
        metacluster_labels: np.ndarray,
    ) -> None:
        # data et metacluster_labels sont alignés (avant expansion downsampling/gating)
        valid_mask = metacluster_labels >= 0
        valid_data = data[valid_mask]
        valid_meta = metacluster_labels[valid_mask]
        unique = np.unique(valid_meta)
        mfi = {int(m): np.median(valid_data[valid_meta == m], axis=0).tolist() for m in unique}
        sample.results["flowsom_mfi"] = mfi
        logger.debug("[FlowSOM] MFI calculé: %d métaclusters", len(mfi))


# ---------------------------------------------------------------------------
# PreGatingStrategy
# ---------------------------------------------------------------------------


class PreGatingStrategy(BaseAnalysisStrategy):
    """
    Construit un masque de pré-gating explicite à partir d'un ensemble de masques.

    Objectif: rendre l'étape "pré-gating" visible dans la chaîne d'analyse
    (après QC/unmixing), sans changer les APIs des stratégies aval.
    """

    name: str = "pre_gating"

    def run(self, sample: Sample, **kwargs: Any) -> Sample:
        p = kwargs.get("params", PreGatingParams())
        if not isinstance(p, PreGatingParams):
            p = PreGatingParams(**p) if isinstance(p, dict) else PreGatingParams()

        if p.source_masks is None:
            source_masks = list(sample.masks.keys())
        else:
            source_masks = list(p.source_masks)

        if p.require_masks and not source_masks:
            raise ValueError(
                "[PreGating] Aucun masque disponible. Exécuter QC/gating avant pre_gating."
            )

        combined = np.ones(sample.n_events, dtype=bool)
        for mask_name in source_masks:
            if mask_name not in sample.masks:
                raise ValueError(f"[PreGating] Masque source introuvable: '{mask_name}'.")
            combined &= sample.get_mask(mask_name)

        pass_fraction = float(combined.mean())
        if pass_fraction < p.min_pass_fraction:
            logger.warning(
                "[PreGating] '%s': pass_fraction faible %.2f%% (< %.2f%%).",
                sample.sample_id,
                100.0 * pass_fraction,
                100.0 * p.min_pass_fraction,
            )

        sample.set_mask(p.output_mask_name, combined)
        sample.results["pre_gating"] = {
            "source_masks": source_masks,
            "output_mask": p.output_mask_name,
            "n_events": int(sample.n_events),
            "n_pass": int(combined.sum()),
            "n_fail": int(sample.n_events - combined.sum()),
            "pass_fraction": pass_fraction,
        }

        logger.info(
            "[PreGating] '%s': masque '%s' construit depuis %s — %d/%d pass.",
            sample.sample_id,
            p.output_mask_name,
            source_masks,
            int(combined.sum()),
            sample.n_events,
        )
        return sample


# ---------------------------------------------------------------------------
# ResearchPipelineExecutor
# ---------------------------------------------------------------------------


class ResearchPipelineExecutor:
    """
    Orchestre l'application séquentielle de stratégies d'analyse
    sur tous les échantillons d'une liste.

    Usage :
        executor = ResearchPipelineExecutor()
        executor.register(UMAPStrategy())
        executor.register(FlowSOMStrategy())
        samples = executor.run(samples, umap_params=..., flowsom_params=...)

    Chaque stratégie reçoit le sample ET les kwargs spécifiques (nom de la
    stratégie préfixé : 'umap_params', 'tsne_params', 'flowsom_proxy_params').
    """

    def __init__(self) -> None:
        self._strategies: List[IAnalysisStrategy] = []

    def register(self, strategy: IAnalysisStrategy) -> "ResearchPipelineExecutor":
        """Enregistre une stratégie dans l'ordre d'exécution."""
        self._strategies.append(strategy)
        logger.debug("Stratégie enregistrée: %s", strategy.name)
        return self

    def run(
        self,
        samples: List[Sample],
        **strategy_kwargs: Any,
    ) -> List[Sample]:
        """
        Applique toutes les stratégies à chaque sample.

        Args:
            samples: Liste de Sample à analyser.
            **strategy_kwargs: Paramètres nommés par stratégie.
                Ex. umap_params=UMAPStrategyParams(n_neighbors=30)
                    flowsom_params=FlowSOMStrategyParams(n_metaclusters=15)

        Returns:
            La même liste de samples modifiés in-place.
        """
        if not self._strategies:
            logger.warning("ResearchPipelineExecutor: aucune stratégie enregistrée.")
            return samples

        for i, sample in enumerate(samples):
            logger.info(
                "[Executor] Sample %d/%d: '%s'",
                i + 1,
                len(samples),
                sample.sample_id,
            )
            for strategy in self._strategies:
                param_key = f"{strategy.name}_params"
                kwargs = {}
                if param_key in strategy_kwargs:
                    kwargs["params"] = strategy_kwargs[param_key]
                try:
                    sample = strategy.run(sample, **kwargs)
                except Exception as exc:
                    logger.exception(
                        "[Executor] Stratégie '%s' sur '%s' échouée: %s",
                        strategy.name,
                        sample.sample_id,
                        exc,
                    )
                    raise

        return samples

    def run_experiment(
        self,
        experiment: Experiment,
        **strategy_kwargs: Any,
    ) -> Experiment:
        """
        Applique les stratégies sur un Experiment complet.

        Règle d'exécution:
        - si la stratégie expose run_experiment(...), elle est exécutée une fois
          au niveau cohorte.
        - sinon fallback sample-level: run(...) sur chaque sample.
        """
        if not self._strategies:
            logger.warning("ResearchPipelineExecutor: aucune stratégie enregistrée.")
            return experiment

        for strategy in self._strategies:
            param_key = f"{strategy.name}_params"
            kwargs: Dict[str, Any] = {}
            if param_key in strategy_kwargs:
                kwargs["params"] = strategy_kwargs[param_key]

            if hasattr(strategy, "run_experiment"):
                logger.info("[Executor] Experiment-level strategy: '%s'", strategy.name)
                try:
                    experiment = strategy.run_experiment(experiment, **kwargs)
                except Exception as exc:
                    logger.exception(
                        "[Executor] Stratégie cohorte '%s' échouée: %s",
                        strategy.name,
                        exc,
                    )
                    raise
                continue

            for sample in experiment.samples:
                try:
                    strategy.run(sample, **kwargs)
                except Exception as exc:
                    logger.exception(
                        "[Executor] Stratégie '%s' sur '%s' échouée: %s",
                        strategy.name,
                        sample.sample_id,
                        exc,
                    )
                    raise

        return experiment

    @property
    def strategy_names(self) -> List[str]:
        return [s.name for s in self._strategies]
