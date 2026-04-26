"""
metaclustering.py — Optimisation automatique du nombre de clusters.

Implémente la stratégie de sélection du k optimal par:
  Phase 1: Screening rapide via silhouette sur le codebook SOM
  Phase 2: Bootstrap ARI (stabilité inter-runs)
  Phase 3: Score composite (stabilité × 0.65 + silhouette × 0.35)

Référence: littérature 2024 — la stabilité est plus pertinente qu'un
score unique (silhouette) pour les données MRD où les blastes rares
nécessitent une résolution fine.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_logger = logging.getLogger("core.metaclustering")

try:
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.cluster import AgglomerativeClustering

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    warnings.warn(
        "scikit-learn requis pour l'auto-clustering: pip install scikit-learn"
    )


def phase1_silhouette_on_codebook(
    codebook: np.ndarray,
    k_range: range,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[int, float]:
    """
    Phase 1 : Screening rapide par silhouette sur le codebook SOM.

    Avantage: Seulement n_nodes points (ex: 100) → quasi-instantané.
    Permet d'éliminer rapidement les k sous-optimaux.

    Args:
        codebook: Centroïdes des nodes SOM, shape (n_nodes, n_markers).
        k_range: Valeurs de k à tester.
        seed: Graine pour reproducibilité.
        verbose: Afficher la progression.

    Returns:
        Dict {k: silhouette_score}.
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn requis pour phase1_silhouette_on_codebook")

    scores: Dict[int, float] = {}

    _logger.info("Phase 1 — Silhouette codebook (%d nodes)", len(codebook))

    for k in k_range:
        if k >= len(codebook):
            continue
        try:
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(
                codebook
            )
            if len(np.unique(labels)) < 2:
                scores[k] = -1.0
                continue
            score = silhouette_score(codebook, labels)
            scores[k] = float(score)
            _logger.debug("  k=%d: silhouette=%.3f", k, score)
        except Exception as e:
            warnings.warn(f"    k={k}: échec silhouette codebook ({e})")
            scores[k] = -1.0

    return scores


def phase2_bootstrap_stability(
    X: np.ndarray,
    k_candidates: List[int],
    n_bootstrap: int = 10,
    sample_size: int = 20_000,
    seed: int = 42,
    xdim: int = 10,
    ydim: int = 10,
    verbose: bool = True,
) -> Dict[int, float]:
    """
    Phase 2 : Stabilité bootstrap par ARI (Adjusted Rand Index).

    Pour chaque k candidat, on effectue n_bootstrap runs FlowSOM
    sur des sous-échantillons différents. La stabilité est mesurée
    par l'ARI moyen entre paires de runs consécutifs.

    Un k stable → les assignations sont reproductibles (ARI proche de 1).

    Args:
        X: Matrice de données complète (n_cells, n_markers).
        k_candidates: Valeurs de k à évaluer.
        n_bootstrap: Nombre de runs bootstrap par k.
        sample_size: Taille de chaque sous-échantillon bootstrap.
        seed: Graine de base.
        xdim: Dimension X grille SOM.
        ydim: Dimension Y grille SOM.
        verbose: Afficher la progression.

    Returns:
        Dict {k: stability_score_moyen}.
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn requis pour phase2_bootstrap_stability")

    try:
        import flowsom as fs
        import anndata as ad
    except ImportError as exc:
        raise ImportError("flowsom requis pour phase2_bootstrap_stability") from exc

    stability_scores: Dict[int, float] = {}

    _logger.info("Phase 2 — Stabilité bootstrap (%d runs/k)", n_bootstrap)

    # Sous-échantillon FIXE pour tous les runs (même cellules, seeds SOM différentes)
    rng_fixed = np.random.default_rng(seed)
    n_sample = min(sample_size, X.shape[0])
    eval_idx = rng_fixed.choice(X.shape[0], size=n_sample, replace=False)
    X_eval = X[eval_idx]

    _logger.info("  Sous-échantillon fixe : %d cellules", n_sample)

    for k in k_candidates:
        labels_all_runs: List[np.ndarray] = []

        for b in range(n_bootstrap):
            try:
                adata = ad.AnnData(X_eval)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=FutureWarning, module="mudata"
                    )
                    warnings.filterwarnings(
                        "ignore", category=FutureWarning, module="flowsom"
                    )
                    fsom_b = fs.FlowSOM(
                        adata,
                        cols_to_use=list(range(X_eval.shape[1])),
                        xdim=xdim,
                        ydim=ydim,
                        n_clusters=k,
                        seed=seed + 100 + b,  # Seed SOM différente à chaque run
                    )

                cell_data = fsom_b.get_cell_data()
                if "metaclustering" in cell_data.obs.columns:
                    labels_b = cell_data.obs["metaclustering"].values
                    labels_all_runs.append(labels_b)
                else:
                    warnings.warn(
                        f"    k={k}, bootstrap {b}: colonne 'metaclustering' absente"
                    )

            except Exception as e:
                warnings.warn(f"    k={k}, bootstrap {b}: échoué ({e})")
                continue

        # ARI pairwise entre toutes les paires de runs
        ari_scores: List[float] = []
        for i in range(len(labels_all_runs)):
            for j in range(i + 1, len(labels_all_runs)):
                try:
                    ari = adjusted_rand_score(labels_all_runs[i], labels_all_runs[j])
                    ari_scores.append(float(ari))
                except Exception:
                    pass

        stability = float(np.mean(ari_scores)) if ari_scores else 0.0
        stability_scores[k] = stability

        if verbose:
            ari_str = f"{stability:.3f}" if ari_scores else "N/A"
            n_valid = len(labels_all_runs)
            print(
                f"    k={k:3d}: stabilité ARI={ari_str} ({n_valid}/{n_bootstrap} runs OK)"
            )

    return stability_scores


def phase3_composite_selection(
    silhouette_scores: Dict[int, float],
    stability_scores: Dict[int, float],
    min_stability_threshold: float = 0.75,
    weight_stability: float = 0.65,
    weight_silhouette: float = 0.35,
    verbose: bool = True,
) -> Tuple[Optional[int], Dict[int, float]]:
    """
    Phase 3 : Sélection du k optimal par score composite.

    Score composite = w_stab × stabilité + w_sil × silhouette_normalisé
    Seuls les k avec stabilité >= min_stability_threshold sont candidats.

    Args:
        silhouette_scores: {k: score} de la Phase 1.
        stability_scores: {k: score} de la Phase 2.
        min_stability_threshold: Stabilité minimum pour être candidat.
        weight_stability: Pondération stabilité (défaut 0.65).
        weight_silhouette: Pondération silhouette normalisé (défaut 0.35).
        verbose: Afficher le classement.

    Returns:
        Tuple (best_k, composite_scores_dict).
    """
    common_k = set(silhouette_scores) & set(stability_scores)
    if not common_k:
        warnings.warn("Aucun k commun entre scores silhouette et stabilité")
        return None, {}

    # Filtrer par seuil de stabilité
    candidates = {
        k for k in common_k if stability_scores.get(k, 0) >= min_stability_threshold
    }

    if not candidates:
        warnings.warn(
            f"Aucun k ne dépasse le seuil de stabilité={min_stability_threshold}. "
            "Utilisation du k le plus stable."
        )
        candidates = {max(stability_scores, key=stability_scores.get)}

    # Normalisation des silhouette scores dans [0, 1]
    sil_vals = [silhouette_scores[k] for k in candidates]
    sil_min, sil_max = min(sil_vals), max(sil_vals)
    sil_range = sil_max - sil_min if sil_max > sil_min else 1.0

    composite: Dict[int, float] = {}
    for k in candidates:
        sil_norm = (silhouette_scores[k] - sil_min) / sil_range
        stab = stability_scores[k]
        composite[k] = weight_stability * stab + weight_silhouette * sil_norm

    best_k = max(composite, key=composite.get)

    _logger.info(
        "Phase 3 — Score composite (stabilité × %.2f + silhouette × %.2f)",
        weight_stability, weight_silhouette,
    )
    for k in sorted(composite):
        marker = " ← OPTIMAL" if k == best_k else ""
        _logger.info(
            "  k=%d: composite=%.3f (stab=%.3f, sil=%.3f)%s",
            k, composite[k], stability_scores.get(k, 0), silhouette_scores.get(k, 0), marker,
        )

    return best_k, composite


def find_optimal_clusters(
    X: np.ndarray,
    codebook: Optional[np.ndarray] = None,
    min_clusters: int = 5,
    max_clusters: int = 35,
    n_bootstrap: int = 10,
    sample_size_bootstrap: int = 20_000,
    min_stability_threshold: float = 0.75,
    weight_stability: float = 0.65,
    weight_silhouette: float = 0.35,
    xdim: int = 10,
    ydim: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> int:
    """
    Trouve le nombre optimal de métaclusters en 3 phases.

    Args:
        X: Matrice de données (n_cells, n_markers).
        codebook: Centroïdes SOM (optionnel — si None, Phase 1 ignorée).
        min_clusters: k minimum à tester.
        max_clusters: k maximum à tester.
        n_bootstrap: Runs bootstrap pour la stabilité.
        sample_size_bootstrap: Cellules par run bootstrap.
        min_stability_threshold: Stabilité minimum pour être candidat.
        weight_stability: Poids stabilité dans le score composite.
        weight_silhouette: Poids silhouette dans le score composite.
        xdim: Dimension X grille SOM.
        ydim: Dimension Y grille SOM.
        seed: Graine aléatoire.
        verbose: Afficher la progression.

    Returns:
        Nombre optimal de métaclusters k.
    """
    k_range = range(min_clusters, max_clusters + 1)

    _logger.info("AUTO-CLUSTERING: recherche k ∈ [%d, %d]", min_clusters, max_clusters)

    # Phase 0 : si codebook non fourni, entraîner un SOM de référence pour l'extraire.
    # Le SOM est entraîné une seule fois (avec k=max_clusters), le codebook extrait,
    # puis re-métaclustèré rapidement pour chaque k — pattern notebook de référence.
    if codebook is None:
        try:
            import flowsom as fs
            import anndata as ad

            n_cells = X.shape[0]
            # Sous-échantillon pour l'entraînement de référence (max 50k, quasi-instantané)
            ref_size = min(50_000, n_cells)
            rng_ref = np.random.default_rng(seed)
            ref_idx = rng_ref.choice(n_cells, size=ref_size, replace=False)
            X_ref = X[ref_idx]

            _logger.info(
                "Phase 0 — Entraînement SOM de référence (%d cellules, k_max=%d)",
                ref_size, max_clusters,
            )

            adata_ref = ad.AnnData(X_ref)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                fsom_ref = fs.FlowSOM(
                    adata_ref,
                    cols_to_use=list(range(X_ref.shape[1])),
                    xdim=xdim,
                    ydim=ydim,
                    n_clusters=max_clusters,
                    seed=seed,
                )

                # Extraire le codebook DANS le bloc de suppression des warnings
                raw_codebook = fsom_ref.get_cluster_data().X
                if hasattr(raw_codebook, "toarray"):
                    raw_codebook = raw_codebook.toarray()
                codebook = np.nan_to_num(
                    np.array(raw_codebook, dtype=np.float32), nan=0.0
                )

            _logger.info(
                "Codebook extrait : %d nodes × %d marqueurs",
                codebook.shape[0], codebook.shape[1],
            )

        except Exception as e:
            _logger.warning("Phase 0 échouée (%s: %s) — Phase 1 ignorée", type(e).__name__, e)
            codebook = None

    # Phase 1 : silhouette sur codebook SOM (screening rapide, tous les k)
    if codebook is not None and len(codebook) >= min_clusters:
        silhouette_scores = phase1_silhouette_on_codebook(
            codebook, k_range, seed=seed, verbose=verbose
        )
        # Ne garder que le top 50% des k pour la Phase 2 (gain de temps)
        top_half = sorted(silhouette_scores, key=silhouette_scores.get, reverse=True)
        top_half = top_half[: max(len(top_half) // 2, 3)]
        k_candidates = sorted(
            [k for k in top_half if min_clusters <= k <= max_clusters]
        )
    else:
        silhouette_scores = {k: 0.5 for k in k_range}
        k_candidates = list(k_range)
        _logger.warning("Codebook non disponible — Phase 1 ignorée, tous k testés")

    # Phase 2 : stabilité bootstrap
    stability_scores = phase2_bootstrap_stability(
        X,
        k_candidates,
        n_bootstrap=n_bootstrap,
        sample_size=sample_size_bootstrap,
        seed=seed,
        xdim=xdim,
        ydim=ydim,
        verbose=verbose,
    )

    # Phase 3 : sélection composite
    best_k, composite_scores = phase3_composite_selection(
        silhouette_scores,
        stability_scores,
        min_stability_threshold=min_stability_threshold,
        weight_stability=weight_stability,
        weight_silhouette=weight_silhouette,
        verbose=verbose,
    )

    if best_k is None:
        default_k = (min_clusters + max_clusters) // 2
        warnings.warn(f"Aucun k optimal trouvé. Utilisation de k={default_k}.")
        return default_k

    _logger.info("k OPTIMAL SELECTIONNE: %d", best_k)
    return best_k
