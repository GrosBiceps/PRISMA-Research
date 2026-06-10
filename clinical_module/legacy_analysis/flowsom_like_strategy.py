"""
analysis/flowsom_like_strategy.py — FlowSOM-like (proxy MVP) avec autométaclustering.

Stratégie pragmatique sans dépendance saeyslab flowsom :
  Étape 1 : MiniBatchKMeans → nœuds SOM proxy
  Étape 2 : Autométaclustering optionnel (3 phases) ou métaclustering fixe
            (AgglomerativeClustering ou KMeans)

Autométaclustering (port direct de l'ancien pipeline) :
  Phase 1  — Silhouette sur codebook (screening rapide)
  Phase 2  — Bootstrap ARI (stabilité inter-runs)
  Phase 3  — Score composite pondéré (stabilité × 0.65 + silhouette × 0.35)

Résultats stockés dans Sample :
  cluster_assignments[som_label]  → labels nœuds (int32, -1 = exclu)
  cluster_assignments[meta_label] → labels métaclusters (int32, -1 = exclu)
  results["flowsom_like_mfi"]     → médiane par métacluster (si compute_centers)
  results["flowsom_like_counts"]  → n_cells par métacluster
  results["flowsom_like_codes"]   → codebook SOM (n_nodes × n_features)
  results["flowsom_like_node_meta"] → mapping nœud → métacluster
  results["flowsom_like_auto_k"]  → k sélectionné automatiquement (si auto)
  results["flowsom_like_composite_scores"] → scores composite par k (si auto)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from prisma.analysis.downsampling import expand_to_full
from prisma.core.models_legacy.sample import Sample

from .strategies import _DOWNSAMPLE_THRESHOLD, BaseAnalysisStrategy

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------

@dataclass
class FlowSOMlikeParams:
    """
    Paramètres FlowSOM-like (proxy MiniBatchKMeans).

    Autométaclustering :
        auto_metaclusters=True  → 3 phases (silhouette + ARI + composite)
        auto_metaclusters=False → n_metaclusters fixe
    """
    channels: Optional[List[str]] = None

    # — SOM proxy
    xdim: int = 10
    ydim: int = 10
    seed: int = 42

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

    # — Stockage
    som_label: str = "flowsom_like_nodes"
    meta_label: str = "flowsom_like_meta"
    compute_centers: bool = True
    store_codes: bool = True

    # — Downsampling
    max_events: int = _DOWNSAMPLE_THRESHOLD


# ---------------------------------------------------------------------------
# Autométaclustering — 3 phases (port direct legacy pipeline)
# ---------------------------------------------------------------------------

def _phase1_silhouette_on_codebook(
    codebook: np.ndarray,
    k_range: range,
    seed: int = 42,
) -> Dict[int, float]:
    """
    Phase 1 : Silhouette sur le codebook SOM (screening rapide).

    ~100 nœuds → quasi-instantané. Élimine les k sous-optimaux
    avant la Phase 2 (bootstrap coûteux).

    Returns:
        Dict {k: silhouette_score}.
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn requis")

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


def _phase2_bootstrap_stability_proxy(
    X: np.ndarray,
    codebook: np.ndarray,
    k_candidates: List[int],
    n_bootstrap: int = 10,
    sample_size: int = 20_000,
    seed: int = 42,
) -> Dict[int, float]:
    """
    Phase 2 : Stabilité bootstrap ARI sur données complètes.

    Pour chaque k, on rémétaclustère le codebook n_bootstrap fois
    avec AgglomerativeClustering + différentes permutations d'ordre
    (surrogate de stabilité sans réentraîner le SOM à chaque fois).

    Puis on projette les cellules et mesure l'ARI entre les runs.

    Returns:
        Dict {k: mean_ARI}.
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn requis")

    # Sous-échantillon FIXE pour tous les runs
    rng = np.random.default_rng(seed)
    n_sample = min(sample_size, X.shape[0])
    eval_idx = rng.choice(X.shape[0], size=n_sample, replace=False)
    X_eval = X[eval_idx]

    logger.info(
        "Phase 2 — Bootstrap ARI (%d runs/k, %d cellules)", n_bootstrap, n_sample
    )

    stability_scores: Dict[int, float] = {}

    for k in k_candidates:
        labels_runs: List[np.ndarray] = []

        for b in range(n_bootstrap):
            try:
                # Permute légèrement le codebook (surrogate de variabilité SOM)
                rng_b = np.random.default_rng(seed + 100 + b)
                noise = rng_b.normal(0, 1e-6, codebook.shape).astype(np.float32)
                codebook_b = codebook + noise

                # Métaclustering sur codebook bruité
                agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
                node_meta_b = agg.fit_predict(codebook_b).astype(np.int32)

                # Projection cellules → nœud le plus proche → métacluster
                dists = np.linalg.norm(
                    X_eval[:, np.newaxis, :] - codebook[np.newaxis, :, :], axis=2
                )
                node_assignments = np.argmin(dists, axis=1)
                labels_b = node_meta_b[node_assignments]
                labels_runs.append(labels_b)
            except Exception as exc:
                warnings.warn(f"  k={k}, run {b}: échoué ({exc})")
                continue

        # ARI pairwise entre toutes les paires de runs
        ari_scores: List[float] = []
        for i in range(len(labels_runs)):
            for j in range(i + 1, len(labels_runs)):
                try:
                    ari_scores.append(float(adjusted_rand_score(labels_runs[i], labels_runs[j])))
                except Exception:
                    pass

        stability = float(np.mean(ari_scores)) if ari_scores else 0.0
        stability_scores[k] = stability

        n_valid = len(labels_runs)
        logger.info(
            "  k=%3d: ARI=%.3f (%d/%d runs OK)", k, stability, n_valid, n_bootstrap
        )

    return stability_scores


def _phase3_composite_selection(
    silhouette_scores: Dict[int, float],
    stability_scores: Dict[int, float],
    min_stability_threshold: float = 0.75,
    weight_stability: float = 0.65,
    weight_silhouette: float = 0.35,
) -> Tuple[Optional[int], Dict[int, float]]:
    """
    Phase 3 : Sélection du k optimal par score composite.

    composite = weight_stability × ARI + weight_silhouette × silhouette_normalisé

    Returns:
        Tuple (best_k, composite_scores_dict).
    """
    common_k = set(silhouette_scores) & set(stability_scores)
    if not common_k:
        warnings.warn("Aucun k commun entre silhouette et stabilité")
        return None, {}

    candidates = {k for k in common_k if stability_scores.get(k, 0) >= min_stability_threshold}

    if not candidates:
        warnings.warn(
            f"Aucun k ≥ seuil stabilité ({min_stability_threshold}). "
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
            k, composite[k], stability_scores.get(k, 0), silhouette_scores.get(k, 0), marker,
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
    Trouve le nombre optimal de métaclusters en 3 phases.

    Port direct de l'algorithme du pipeline legacy (metaclustering.py).
    Adapté pour fonctionner sur un codebook MiniBatchKMeans (proxy SOM)
    au lieu du codebook saeyslab FlowSOM.

    Args:
        X: Données pré-gatées (n_cells × n_features).
        codebook: Codes SOM proxy (n_nodes × n_features) — centres MiniBatchKMeans.
        min_clusters: k minimum à tester.
        max_clusters: k maximum à tester.
        n_bootstrap: Runs bootstrap par k candidat.
        sample_size_bootstrap: Cellules par run bootstrap.
        min_stability_threshold: ARI minimum pour être candidat.
        weight_stability: Poids ARI dans score composite.
        weight_silhouette: Poids silhouette dans score composite.
        seed: Graine reproductible.

    Returns:
        Tuple (best_k, composite_scores_dict).
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn requis pour find_optimal_metaclusters")

    k_range = range(min_clusters, max_clusters + 1)
    logger.info("AUTO-METACLUSTERING: k ∈ [%d, %d]", min_clusters, max_clusters)

    # Phase 1 : silhouette sur codebook (screening rapide)
    silhouette_scores = _phase1_silhouette_on_codebook(codebook, k_range, seed=seed)

    # Top 50% des k pour Phase 2
    top_half = sorted(silhouette_scores, key=silhouette_scores.get, reverse=True)
    top_half = top_half[: max(len(top_half) // 2, 3)]
    k_candidates = sorted([k for k in top_half if min_clusters <= k <= max_clusters])

    logger.info("Phase 1 → %d candidats pour Phase 2: %s", len(k_candidates), k_candidates)

    # Phase 2 : bootstrap ARI
    stability_scores = _phase2_bootstrap_stability_proxy(
        X,
        codebook,
        k_candidates,
        n_bootstrap=n_bootstrap,
        sample_size=sample_size_bootstrap,
        seed=seed,
    )

    # Phase 3 : sélection composite
    best_k, composite_scores = _phase3_composite_selection(
        silhouette_scores,
        stability_scores,
        min_stability_threshold=min_stability_threshold,
        weight_stability=weight_stability,
        weight_silhouette=weight_silhouette,
    )

    if best_k is None:
        default_k = (min_clusters + max_clusters) // 2
        warnings.warn(f"Aucun k optimal trouvé. Fallback k={default_k}.")
        return default_k, {}

    logger.info("k OPTIMAL: %d", best_k)
    return best_k, composite_scores


# ---------------------------------------------------------------------------
# FlowSOMlikeStrategy
# ---------------------------------------------------------------------------

class FlowSOMlikeStrategy(BaseAnalysisStrategy):
    """
    Clustering FlowSOM-like (proxy MVP) sans dépendance saeyslab.

    Étape 1 : MiniBatchKMeans → nœuds SOM proxy.
    Étape 2 : Métaclustering fixe (AgglomerativeClustering ou KMeans)
              OU autométaclustering 3 phases si auto_metaclusters=True.

    Stockage dans Sample :
      cluster_assignments[som_label]   → nœuds (int32, -1 = exclu)
      cluster_assignments[meta_label]  → métaclusters (int32, -1 = exclu)
      results["flowsom_like_codes"]    → codebook (n_nodes × n_features)
      results["flowsom_like_node_meta"]→ nœud → métacluster
      results["flowsom_like_mfi"]      → médiane par métacluster
      results["flowsom_like_counts"]   → n_cells par métacluster
      results["flowsom_like_auto_k"]   → k sélectionné si auto
      results["flowsom_like_composite_scores"] → scores composite si auto

    Branchement dans ResearchPipelineExecutor :
        executor.register(FlowSOMlikeStrategy())
        executor.run(samples, flowsom_like_params=FlowSOMlikeParams(auto_metaclusters=True))
    """

    name: str = "flowsom_like"

    def __init__(self) -> None:
        self._codes_: Optional[np.ndarray] = None
        self._node_meta_: Optional[np.ndarray] = None

    def run(self, sample: Sample, **kwargs: Any) -> Sample:
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn requis pour FlowSOMlikeStrategy")

        p = kwargs.get("params", FlowSOMlikeParams())
        if not isinstance(p, FlowSOMlikeParams):
            p = FlowSOMlikeParams(**p) if isinstance(p, dict) else FlowSOMlikeParams()

        data, ds, channels, gate_idx = self._prepare_data(
            sample, p.channels, p.seed, p.max_events
        )

        n_nodes = p.xdim * p.ydim

        logger.info(
            "[FlowSOM-like] '%s': %d × %d, SOM=%dx%d (%d nœuds), auto=%s",
            sample.sample_id, data.shape[0], data.shape[1],
            p.xdim, p.ydim, n_nodes, p.auto_metaclusters,
        )

        # — Étape 1 : SOM proxy MiniBatchKMeans
        som = MiniBatchKMeans(
            n_clusters=n_nodes,
            random_state=p.seed,
            n_init=3,
            max_iter=300,
        )
        node_labels = som.fit_predict(data).astype(np.int32)
        codes = som.cluster_centers_.astype(np.float32)
        self._codes_ = codes

        # — Étape 2 : Métaclustering
        n_meta, composite_scores = self._metacluster(data, codes, p)

        node_meta = self._apply_metaclustering(codes, n_meta, p)
        self._node_meta_ = node_meta

        metacluster_labels = node_meta[node_labels]

        # — Expansion labels (downsampling → gating → N_events)
        raw_meta = metacluster_labels.copy()

        if ds is not None and ds.was_downsampled:
            node_labels = expand_to_full(
                node_labels.astype(np.float32), ds, fill_value=-1.0
            ).astype(np.int32)
            metacluster_labels = expand_to_full(
                metacluster_labels.astype(np.float32), ds, fill_value=-1.0
            ).astype(np.int32)

        node_labels_full = self._expand_to_full_events(
            node_labels, gate_idx, sample.n_events, fill_value=-1.0
        ).astype(np.int32)
        metacluster_labels_full = self._expand_to_full_events(
            metacluster_labels, gate_idx, sample.n_events, fill_value=-1.0
        ).astype(np.int32)

        sample.add_cluster_labels(p.som_label, node_labels_full)
        sample.add_cluster_labels(p.meta_label, metacluster_labels_full)

        # — Résultats analytiques (sur données brutes, avant expansion)
        self._store_results(sample, data, raw_meta, codes, node_meta, n_meta,
                            composite_scores, p)

        logger.info(
            "[FlowSOM-like] '%s': '%s' + '%s' stockés (k=%d)",
            sample.sample_id, p.som_label, p.meta_label, n_meta,
        )
        return sample

    # ------------------------------------------------------------------
    # Métaclustering
    # ------------------------------------------------------------------

    def _metacluster(
        self,
        data: np.ndarray,
        codes: np.ndarray,
        p: FlowSOMlikeParams,
    ) -> Tuple[int, Dict[int, float]]:
        """Retourne (n_meta_choisi, composite_scores)."""
        if p.auto_metaclusters:
            logger.info("[FlowSOM-like] autométaclustering 3 phases")
            best_k, composite = find_optimal_metaclusters(
                X=data,
                codebook=codes,
                min_clusters=p.min_clusters,
                max_clusters=min(p.max_clusters, len(codes) - 1),
                n_bootstrap=p.n_bootstrap,
                sample_size_bootstrap=p.sample_size_bootstrap,
                min_stability_threshold=p.min_stability_threshold,
                weight_stability=p.weight_stability,
                weight_silhouette=p.weight_silhouette,
                seed=p.seed,
            )
            return best_k, composite

        return min(p.n_metaclusters, len(codes) - 1), {}

    def _apply_metaclustering(
        self,
        codes: np.ndarray,
        n_meta: int,
        p: FlowSOMlikeParams,
    ) -> np.ndarray:
        """Applique la méthode de métaclustering choisie sur les codes."""
        if p.metacluster_method == "kmeans":
            km = KMeans(n_clusters=n_meta, random_state=p.seed, n_init=5)
            return km.fit_predict(codes).astype(np.int32)

        # Default : AgglomerativeClustering (ward linkage, comme le pipeline legacy)
        agg = AgglomerativeClustering(n_clusters=n_meta, linkage="ward")
        return agg.fit_predict(codes).astype(np.int32)

    # ------------------------------------------------------------------
    # Stockage résultats
    # ------------------------------------------------------------------

    def _store_results(
        self,
        sample: Sample,
        data: np.ndarray,
        raw_meta: np.ndarray,
        codes: np.ndarray,
        node_meta: np.ndarray,
        n_meta: int,
        composite_scores: Dict[int, float],
        p: FlowSOMlikeParams,
    ) -> None:
        if p.store_codes:
            sample.results["flowsom_like_codes"] = codes.tolist()

        sample.results["flowsom_like_node_meta"] = node_meta.tolist()

        if p.auto_metaclusters:
            sample.results["flowsom_like_auto_k"] = n_meta
            sample.results["flowsom_like_composite_scores"] = {
                str(k): float(v) for k, v in composite_scores.items()
            }

        valid_mask = raw_meta >= 0
        valid_data = data[valid_mask]
        valid_meta = raw_meta[valid_mask]

        counts: Dict[int, int] = {}
        for m in np.unique(valid_meta):
            counts[int(m)] = int(np.sum(valid_meta == m))
        sample.results["flowsom_like_counts"] = counts

        if p.compute_centers:
            mfi: Dict[int, List[float]] = {}
            for m in np.unique(valid_meta):
                mfi[int(m)] = np.median(valid_data[valid_meta == m], axis=0).tolist()
            sample.results["flowsom_like_mfi"] = mfi
