"""
prisma/strategies/flowsom_like_strategy.py — FlowSOM-like (proxy MVP) avec autométaclustering.

Stratégie pragmatique sans dépendance saeyslab flowsom :
  Étape 1 : MiniBatchKMeans → nœuds SOM proxy (xdim × ydim nœuds)
  Étape 2 : Métaclustering fixe (AgglomerativeClustering [ward] ou KMeans)
            OU autométaclustering 3 phases identique à l'ancien pipeline

Autométaclustering (port exact de flowsom_pipeline_pro/src/core/metaclustering.py) :
  Phase 1 — Silhouette sur codebook MiniBatchKMeans (screening rapide)
  Phase 2 — Bootstrap ARI (stabilité inter-runs sur nœuds bruités)
  Phase 3 — Score composite pondéré (stabilité × w_stab + silhouette × w_sil)

Enregistré dans StrategyRegistry sous "flowsom_like".
Implémente le Protocol ClusterStrategy (fit_predict).

Compatible avec :
  - ResearchPipelineExecutor (run via analysis/flowsom_like_strategy.py)
  - StrategyRegistry (via @register_clustering)
  - Sample.add_cluster_labels() pour le stockage
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from prisma.core.registry import StrategyRegistry
from prisma.strategies.base import ClusterParams

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------

@dataclass
class FlowSOMlikeParams(ClusterParams):
    """
    Paramètres complets de la stratégie FlowSOM-like.

    Tous les paramètres du pipeline legacy (metaclustering.py) sont exposés.
    Si auto_metaclusters=True, ignore n_metaclusters et cherche le k optimal.
    """
    # — SOM proxy
    xdim: int = 10
    ydim: int = 10

    # — Métaclustering fixe
    n_metaclusters: int = 20
    metacluster_method: str = "agglomerative"  # "agglomerative" | "kmeans"

    # — Autométaclustering 3 phases
    auto_metaclusters: bool = False
    min_clusters: int = 5
    max_clusters: int = 35
    n_bootstrap: int = 10
    sample_size_bootstrap: int = 20_000
    min_stability_threshold: float = 0.75
    weight_stability: float = 0.65
    weight_silhouette: float = 0.35

    # — Marqueurs / canaux
    markers: Optional[List[str]] = None

    # — Stockage
    som_label: str = "flowsom_like_nodes"
    meta_label: str = "flowsom_like_meta"
    compute_centers: bool = True
    store_codes: bool = True


# ---------------------------------------------------------------------------
# Résultat
# ---------------------------------------------------------------------------

@dataclass
class FlowSOMlikeResult:
    """
    Résultats complets d'un run FlowSOM-like.

    Attributes:
        node_labels       : ndarray (n_cells,) — nœud SOM par cellule.
        metacluster_labels: ndarray (n_cells,) — métacluster par cellule.
        codes             : ndarray (n_nodes, n_features) — codebook.
        node_meta         : ndarray (n_nodes,) — métacluster par nœud.
        mfi               : dict {meta_id: median_expression} ou None.
        counts            : dict {meta_id: n_cells}.
        n_metaclusters    : k effectivement utilisé.
        auto_selected     : True si autométaclustering.
        composite_scores  : dict {k: composite} si auto, sinon {}.
    """
    node_labels: np.ndarray
    metacluster_labels: np.ndarray
    codes: np.ndarray
    node_meta: np.ndarray
    mfi: Optional[Dict[int, List[float]]]
    counts: Dict[int, int]
    n_metaclusters: int
    auto_selected: bool
    composite_scores: Dict[int, float]


# ---------------------------------------------------------------------------
# Autométaclustering — 3 phases (algorithme legacy)
# ---------------------------------------------------------------------------

def phase1_silhouette_on_codebook(
    codebook: np.ndarray,
    k_range: range,
    seed: int = 42,
) -> Dict[int, float]:
    """
    Phase 1 : Silhouette sur le codebook (screening rapide).

    ~100 nœuds → quasi-instantané. Élimine les k sous-optimaux.

    Returns:
        Dict {k: silhouette_score}.
    """
    scores: Dict[int, float] = {}
    logger.info("Phase 1 — Silhouette codebook (%d nœuds)", len(codebook))

    for k in k_range:
        if k >= len(codebook):
            continue
        try:
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(codebook)
            if len(np.unique(labels)) < 2:
                scores[k] = -1.0
                continue
            score = float(silhouette_score(codebook, labels))
            scores[k] = score
            logger.debug("  k=%d: silhouette=%.3f", k, score)
        except Exception as exc:
            warnings.warn(f"  k={k}: silhouette codebook échoué ({exc})")
            scores[k] = -1.0

    return scores


def phase2_bootstrap_stability(
    X: np.ndarray,
    codebook: np.ndarray,
    k_candidates: List[int],
    n_bootstrap: int = 10,
    sample_size: int = 20_000,
    seed: int = 42,
) -> Dict[int, float]:
    """
    Phase 2 : Stabilité bootstrap ARI.

    Pour chaque k : rémétaclustère le codebook n_bootstrap fois avec bruit
    infinitésimal (surrogate de variabilité SOM sans réentraînement),
    projette les cellules, mesure ARI pairwise entre runs.

    Un k stable → ARI proche de 1.

    Returns:
        Dict {k: mean_ARI}.
    """
    rng = np.random.default_rng(seed)
    n_sample = min(sample_size, X.shape[0])
    eval_idx = rng.choice(X.shape[0], size=n_sample, replace=False)
    X_eval = X[eval_idx]

    logger.info(
        "Phase 2 — Bootstrap ARI (%d runs/k, %d cellules)", n_bootstrap, n_sample
    )

    stability: Dict[int, float] = {}

    for k in k_candidates:
        labels_runs: List[np.ndarray] = []

        for b in range(n_bootstrap):
            try:
                rng_b = np.random.default_rng(seed + 100 + b)
                noise = rng_b.normal(0, 1e-6, codebook.shape).astype(np.float32)
                codes_b = codebook + noise

                node_meta_b = AgglomerativeClustering(
                    n_clusters=k, linkage="ward"
                ).fit_predict(codes_b).astype(np.int32)

                # Projection cellules → nœud le plus proche
                dists = np.linalg.norm(
                    X_eval[:, np.newaxis, :] - codebook[np.newaxis, :, :], axis=2
                )
                node_assign = np.argmin(dists, axis=1)
                labels_runs.append(node_meta_b[node_assign])
            except Exception as exc:
                warnings.warn(f"  k={k}, run {b}: échoué ({exc})")
                continue

        ari_scores: List[float] = []
        for i in range(len(labels_runs)):
            for j in range(i + 1, len(labels_runs)):
                try:
                    ari_scores.append(float(adjusted_rand_score(labels_runs[i], labels_runs[j])))
                except Exception:
                    pass

        score = float(np.mean(ari_scores)) if ari_scores else 0.0
        stability[k] = score
        logger.info(
            "  k=%3d: ARI=%.3f (%d/%d runs OK)", k, score, len(labels_runs), n_bootstrap
        )

    return stability


def phase3_composite_selection(
    silhouette_scores: Dict[int, float],
    stability_scores: Dict[int, float],
    min_stability_threshold: float = 0.75,
    weight_stability: float = 0.65,
    weight_silhouette: float = 0.35,
) -> Tuple[Optional[int], Dict[int, float]]:
    """
    Phase 3 : Score composite pondéré.

    composite = w_stab × ARI + w_sil × silhouette_normalisé[0,1]
    Seulement les k avec ARI ≥ min_stability_threshold sont candidats.

    Returns:
        Tuple (best_k, composite_scores_dict).
    """
    common_k = set(silhouette_scores) & set(stability_scores)
    if not common_k:
        warnings.warn("Aucun k commun — vérifier phases 1 et 2")
        return None, {}

    candidates = {k for k in common_k if stability_scores.get(k, 0) >= min_stability_threshold}
    if not candidates:
        warnings.warn(
            f"Aucun k dépasse seuil stabilité={min_stability_threshold}. "
            "Utilisation du k le plus stable."
        )
        candidates = {max(stability_scores, key=stability_scores.get)}

    sil_vals = [silhouette_scores[k] for k in candidates]
    sil_min, sil_max = min(sil_vals), max(sil_vals)
    sil_range = sil_max - sil_min if sil_max > sil_min else 1.0

    composite: Dict[int, float] = {}
    for k in candidates:
        sil_norm = (silhouette_scores[k] - sil_min) / sil_range
        composite[k] = weight_stability * stability_scores[k] + weight_silhouette * sil_norm

    best_k = max(composite, key=composite.get)

    logger.info(
        "Phase 3 — Composite (stab×%.2f + sil×%.2f)", weight_stability, weight_silhouette
    )
    for k in sorted(composite):
        marker = " ← OPTIMAL" if k == best_k else ""
        logger.info(
            "  k=%d: composite=%.3f (stab=%.3f, sil=%.3f)%s",
            k, composite[k],
            stability_scores.get(k, 0),
            silhouette_scores.get(k, 0),
            marker,
        )

    return best_k, composite


def find_optimal_metaclusters(
    X: np.ndarray,
    codebook: np.ndarray,
    min_clusters: int = 5,
    max_clusters: int = 35,
    n_bootstrap: int = 10,
    sample_size_bootstrap: int = 20_000,
    min_stability_threshold: float = 0.75,
    weight_stability: float = 0.65,
    weight_silhouette: float = 0.35,
    seed: int = 42,
) -> Tuple[int, Dict[int, float]]:
    """
    Trouve le k optimal en 3 phases.

    Port direct de flowsom_pipeline_pro/src/core/metaclustering.py.
    Adapté au codebook MiniBatchKMeans (Phase 2 sans réentraînement SOM).

    Args:
        X       : Données (n_cells × n_features), alignées avec le codebook.
        codebook: Codes SOM proxy (n_nodes × n_features).
        …       : Paramètres des 3 phases.
        seed    : Graine globale reproductible.

    Returns:
        Tuple (best_k, composite_scores).
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn requis")

    max_k = min(max_clusters, len(codebook) - 1)
    k_range = range(min_clusters, max_k + 1)

    logger.info("AUTO-METACLUSTERING: k ∈ [%d, %d]", min_clusters, max_k)

    sil_scores = phase1_silhouette_on_codebook(codebook, k_range, seed=seed)

    top_half = sorted(sil_scores, key=sil_scores.get, reverse=True)
    top_half = top_half[: max(len(top_half) // 2, 3)]
    k_candidates = sorted([k for k in top_half if min_clusters <= k <= max_k])
    logger.info("Phase 1 → %d candidats: %s", len(k_candidates), k_candidates)

    stab_scores = phase2_bootstrap_stability(
        X, codebook, k_candidates,
        n_bootstrap=n_bootstrap,
        sample_size=sample_size_bootstrap,
        seed=seed,
    )

    best_k, composite = phase3_composite_selection(
        sil_scores, stab_scores,
        min_stability_threshold=min_stability_threshold,
        weight_stability=weight_stability,
        weight_silhouette=weight_silhouette,
    )

    if best_k is None:
        default_k = (min_clusters + max_k) // 2
        warnings.warn(f"Aucun k optimal. Fallback k={default_k}.")
        return default_k, {}

    logger.info("k OPTIMAL SÉLECTIONNÉ: %d", best_k)
    return best_k, composite


# ---------------------------------------------------------------------------
# FlowSOMlikeStrategy
# ---------------------------------------------------------------------------

@StrategyRegistry.register_clustering("flowsom_like")
class FlowSOMlikeStrategy:
    """
    Stratégie FlowSOM-like (proxy MVP) pour PRISMA Research.

    Implémente le Protocol ClusterStrategy (fit_predict) ET expose
    les résultats complets via self.result_ après entraînement.

    Autométaclustering activé via FlowSOMlikeParams(auto_metaclusters=True).
    Toutes les phases et paramètres du pipeline legacy sont exposés.

    Usage minimal :
        strategy = FlowSOMlikeStrategy()
        meta_labels = strategy.fit_predict(data, FlowSOMlikeParams())

    Usage autométaclustering :
        params = FlowSOMlikeParams(auto_metaclusters=True, min_clusters=5, max_clusters=20)
        meta_labels = strategy.fit_predict(data, params)
        print("k optimal:", strategy.result_.n_metaclusters)
        print("scores:", strategy.result_.composite_scores)

    Usage dans ResearchPipelineExecutor :
        # Importer depuis analysis/flowsom_like_strategy.py (même algorithme, interface Sample)
    """

    name: str = "flowsom_like"

    def __init__(self) -> None:
        self.result_: Optional[FlowSOMlikeResult] = None

    # ------------------------------------------------------------------
    # Protocol ClusterStrategy
    # ------------------------------------------------------------------

    def fit_predict(self, data: np.ndarray, params: ClusterParams) -> np.ndarray:
        """
        Entraîne le SOM proxy + métaclustère. Retourne labels métaclusters.

        Args:
            data  : (n_cells × n_features) float32/float64.
            params: FlowSOMlikeParams ou ClusterParams générique.

        Returns:
            Labels int32 (n_cells,).
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn requis")

        p = self._coerce_params(params)

        n_nodes = p.xdim * p.ydim
        logger.info(
            "[FlowSOM-like] %d × %d, SOM=%dx%d (%d nœuds), auto=%s",
            data.shape[0], data.shape[1], p.xdim, p.ydim, n_nodes, p.auto_metaclusters,
        )

        # — Étape 1 : SOM proxy
        som = MiniBatchKMeans(
            n_clusters=n_nodes,
            random_state=p.seed,
            n_init=3,
            max_iter=300,
        )
        node_labels = som.fit_predict(data).astype(np.int32)
        codes = som.cluster_centers_.astype(np.float32)

        # — Étape 2 : Métaclustering
        n_meta, composite_scores, auto_selected = self._resolve_metaclusters(data, codes, p)
        node_meta = self._apply_metaclustering(codes, n_meta, p)
        metacluster_labels = node_meta[node_labels].astype(np.int32)

        # — Résultats
        mfi = self._compute_mfi(data, metacluster_labels) if p.compute_centers else None
        counts = self._compute_counts(metacluster_labels)

        self.result_ = FlowSOMlikeResult(
            node_labels=node_labels,
            metacluster_labels=metacluster_labels,
            codes=codes,
            node_meta=node_meta,
            mfi=mfi,
            counts=counts,
            n_metaclusters=n_meta,
            auto_selected=auto_selected,
            composite_scores=composite_scores,
        )

        logger.info(
            "[FlowSOM-like] terminé: %d nœuds → %d métaclusters (auto=%s)",
            n_nodes, n_meta, auto_selected,
        )
        return metacluster_labels

    # ------------------------------------------------------------------
    # Accesseurs post-fit
    # ------------------------------------------------------------------

    def get_node_labels(self) -> np.ndarray:
        """Labels nœuds SOM (n_cells,)."""
        self._check_fitted()
        return self.result_.node_labels

    def get_metacluster_labels(self) -> np.ndarray:
        """Labels métaclusters (n_cells,)."""
        self._check_fitted()
        return self.result_.metacluster_labels

    def get_codes(self) -> np.ndarray:
        """Codebook SOM proxy (n_nodes × n_features)."""
        self._check_fitted()
        return self.result_.codes

    def get_node_metacluster_mapping(self) -> np.ndarray:
        """Pour chaque nœud : quel métacluster (n_nodes,)."""
        self._check_fitted()
        return self.result_.node_meta

    def get_mfi(self) -> Optional[Dict[int, List[float]]]:
        """MFI médian par métacluster. None si compute_centers=False."""
        self._check_fitted()
        return self.result_.mfi

    def get_counts(self) -> Dict[int, int]:
        """n_cells par métacluster."""
        self._check_fitted()
        return self.result_.counts

    def get_percentages(self) -> Dict[int, float]:
        """% cellules par métacluster."""
        self._check_fitted()
        counts = self.result_.counts
        total = sum(counts.values())
        if total == 0:
            return {k: 0.0 for k in counts}
        return {k: v / total * 100.0 for k, v in counts.items()}

    def get_optimal_k(self) -> int:
        """k sélectionné (auto ou fixe)."""
        self._check_fitted()
        return self.result_.n_metaclusters

    def get_composite_scores(self) -> Dict[int, float]:
        """Scores composites de l'autométaclustering. Vide si auto=False."""
        self._check_fitted()
        return self.result_.composite_scores

    def map_new_data(self, data: np.ndarray) -> np.ndarray:
        """
        Projette de nouvelles données sur le SOM existant.

        Distance euclidienne → nœud le plus proche → métacluster.

        Returns:
            Labels métaclusters (n_new_cells,).
        """
        self._check_fitted()
        codes = self.result_.codes
        dists = np.linalg.norm(
            data[:, np.newaxis, :] - codes[np.newaxis, :, :], axis=2
        )
        node_assign = np.argmin(dists, axis=1).astype(np.int32)
        return self.result_.node_meta[node_assign]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_metaclusters(
        self,
        data: np.ndarray,
        codes: np.ndarray,
        p: FlowSOMlikeParams,
    ) -> Tuple[int, Dict[int, float], bool]:
        """Retourne (n_meta, composite_scores, auto_selected)."""
        if p.auto_metaclusters:
            best_k, composite = find_optimal_metaclusters(
                X=data,
                codebook=codes,
                min_clusters=p.min_clusters,
                max_clusters=p.max_clusters,
                n_bootstrap=p.n_bootstrap,
                sample_size_bootstrap=p.sample_size_bootstrap,
                min_stability_threshold=p.min_stability_threshold,
                weight_stability=p.weight_stability,
                weight_silhouette=p.weight_silhouette,
                seed=p.seed,
            )
            return best_k, composite, True

        return min(p.n_metaclusters, len(codes) - 1), {}, False

    def _apply_metaclustering(
        self,
        codes: np.ndarray,
        n_meta: int,
        p: FlowSOMlikeParams,
    ) -> np.ndarray:
        if p.metacluster_method == "kmeans":
            km = KMeans(n_clusters=n_meta, random_state=p.seed, n_init=5)
            return km.fit_predict(codes).astype(np.int32)
        agg = AgglomerativeClustering(n_clusters=n_meta, linkage="ward")
        return agg.fit_predict(codes).astype(np.int32)

    @staticmethod
    def _compute_mfi(data: np.ndarray, labels: np.ndarray) -> Dict[int, List[float]]:
        valid = labels >= 0
        mfi = {}
        for m in np.unique(labels[valid]):
            mfi[int(m)] = np.median(data[(labels == m) & valid], axis=0).tolist()
        return mfi

    @staticmethod
    def _compute_counts(labels: np.ndarray) -> Dict[int, int]:
        valid = labels[labels >= 0]
        return {int(m): int(np.sum(valid == m)) for m in np.unique(valid)}

    @staticmethod
    def _coerce_params(params: ClusterParams) -> FlowSOMlikeParams:
        if isinstance(params, FlowSOMlikeParams):
            return params
        return FlowSOMlikeParams(
            n_clusters=params.n_clusters,
            seed=params.seed,
            n_jobs=params.n_jobs,
        )

    def _check_fitted(self) -> None:
        if self.result_ is None:
            raise RuntimeError(
                "FlowSOMlikeStrategy non entraîné. Appelez fit_predict() d'abord."
            )
