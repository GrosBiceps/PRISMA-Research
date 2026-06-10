"""
analysis/citrus_strategy.py — Implémentation Python de l'algorithme Citrus.

Citrus (Bruggner et al., 2014) identifie les sous-populations cellulaires
dont l'abondance ou l'expression corrèle avec un endpoint clinique.

Workflow:
  1. Clustering hiérarchique sur cellules combinées de tous les samples
  2. Extraction de features par cluster (abondances + médianes marqueurs)
  3. Modèle régularisé (Lasso/ElasticNet ou SAM) pour identifier clusters stratifiants
  4. Output: clusters significatifs + leur importance

Référence: Bruggner RV et al., PNAS 2014.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from prisma.core.models_legacy.experiment import Experiment
from prisma.core.models_legacy.sample import Sample

logger = logging.getLogger(__name__)

FeatureType = Literal["abundances", "medians", "both"]
ModelType = Literal["glmnet", "sam", "pamr"]
EndpointType = Literal["classification", "continuous"]


# ---------------------------------------------------------------------------
# Dataclasses de configuration et résultats
# ---------------------------------------------------------------------------


@dataclass
class CitrusParams:
    """Paramètres de l'analyse Citrus."""

    clustering_channels: Optional[List[str]] = None
    functional_channels: Optional[List[str]] = None
    n_cells_per_sample: int = 1000
    min_cluster_size_percent: float = 0.05
    feature_type: FeatureType = "abundances"
    model_type: ModelType = "glmnet"
    endpoint_type: EndpointType = "classification"
    linkage_method: str = "average"
    n_cv_folds: int = 5
    random_seed: int = 42


@dataclass
class CitrusCluster:
    """Cluster identifié par Citrus."""

    cluster_id: int
    size: int
    size_percent: float
    feature_importance: float
    mean_abundance: float
    median_expressions: Dict[str, float] = field(default_factory=dict)
    per_sample_abundances: Dict[str, float] = field(default_factory=dict)


@dataclass
class CitrusResult:
    """Résultat complet d'une analyse Citrus."""

    stratifying_clusters: List[CitrusCluster]
    all_cluster_labels: np.ndarray
    feature_matrix: pd.DataFrame
    model_score: float
    model_type: str
    endpoint_type: str
    n_samples: int
    n_cells_total: int
    clustering_channels: List[str]
    params: CitrusParams


# ---------------------------------------------------------------------------
# Fonctions internes
# ---------------------------------------------------------------------------


def _subsample_sample(sample: Sample, n_cells: int, seed: int) -> np.ndarray:
    """Extrait n_cells événements d'un Sample (avec ou sans gating actif)."""
    if sample.events.empty:
        return np.empty((0, 0))

    rng = np.random.default_rng(seed)
    n_available = len(sample.events)
    idx = rng.choice(n_available, size=min(n_cells, n_available), replace=False)
    return sample.events.iloc[idx].values, sample.events.columns.tolist(), idx


def _build_combined_matrix(
    experiment: Experiment,
    channels: List[str],
    n_cells: int,
    seed: int,
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    """
    Concatène les cellules sous-échantillonnées de tous les samples.

    Returns:
        combined: ndarray (N_total × n_channels)
        sample_ids: list de sample_id (longueur N_total)
        sample_indices: dict sample_id → indices dans combined
    """
    matrices = []
    sample_ids = []
    sample_indices: Dict[str, np.ndarray] = {}
    offset = 0

    for sample in experiment.samples:
        if sample.events.empty:
            continue

        available = [c for c in channels if c in sample.events.columns]
        if not available:
            logger.warning("Sample %s: aucun canal citrus disponible", sample.sample_id)
            continue

        result = _subsample_sample(sample, n_cells, seed + hash(sample.sample_id) % 1000)
        data, cols, _ = result

        col_idx = [cols.index(c) for c in available if c in cols]
        if not col_idx:
            continue

        sub = data[:, col_idx]
        n = len(sub)
        matrices.append(sub)
        ids = [sample.sample_id] * n
        sample_ids.extend(ids)
        sample_indices[sample.sample_id] = np.arange(offset, offset + n)
        offset += n

    if not matrices:
        raise ValueError("Aucun sample valide pour Citrus")

    combined = np.vstack(matrices)
    return combined, sample_ids, sample_indices


def _hierarchical_cluster(
    data: np.ndarray,
    method: str = "average",
    min_cluster_size_frac: float = 0.05,
) -> np.ndarray:
    """
    Clustering hiérarchique sur données combinées.
    Coupe l'arbre pour obtenir ~sqrt(N) clusters, filtre par taille minimale.
    """
    n = len(data)
    if n < 10:
        return np.zeros(n, dtype=int)

    logger.info("Clustering hiérarchique sur %d cellules...", n)

    # Subsample pour calcul distance si trop grand (limite mémoire)
    max_for_linkage = 5000
    if n > max_for_linkage:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_for_linkage, replace=False)
        data_small = data[idx]
    else:
        data_small = data
        idx = np.arange(n)

    dist = pdist(data_small, metric="euclidean")
    Z = linkage(dist, method=method)

    # Nombre cible de clusters: sqrt(N)
    n_clusters_target = max(10, int(np.sqrt(n)))
    labels_small = fcluster(Z, t=n_clusters_target, criterion="maxclust")

    if n > max_for_linkage:
        # Propagation par KNN vers cellules non clustérisées
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(data_small, labels_small)
        labels = knn.predict(data)
    else:
        labels = labels_small

    return labels.astype(int)


def _calculate_abundances(
    labels: np.ndarray,
    sample_ids: List[str],
    cluster_ids: np.ndarray,
    experiment_samples: List[Sample],
) -> pd.DataFrame:
    """
    Calcule l'abondance (proportion) de chaque cluster dans chaque sample.

    Returns:
        DataFrame (n_samples × n_clusters), index=sample_id, cols=cluster_id
    """
    sample_id_arr = np.array(sample_ids)
    unique_samples = [s.sample_id for s in experiment_samples if not s.events.empty]

    rows = {}
    for sid in unique_samples:
        mask = sample_id_arr == sid
        if not mask.any():
            rows[sid] = {cid: 0.0 for cid in cluster_ids}
            continue
        n_total = mask.sum()
        sample_labels = labels[mask]
        row = {}
        for cid in cluster_ids:
            row[cid] = (sample_labels == cid).sum() / n_total
        rows[sid] = row

    return pd.DataFrame(rows).T.fillna(0.0)


def _calculate_medians(
    data: np.ndarray,
    labels: np.ndarray,
    sample_ids: List[str],
    cluster_ids: np.ndarray,
    functional_channels: List[str],
    experiment_samples: List[Sample],
) -> pd.DataFrame:
    """
    Calcule la médiane d'expression par cluster par sample.

    Returns:
        DataFrame (n_samples × n_clusters*n_channels)
    """
    sample_id_arr = np.array(sample_ids)
    unique_samples = [s.sample_id for s in experiment_samples if not s.events.empty]
    n_func = len(functional_channels)

    rows = {}
    for sid in unique_samples:
        mask = sample_id_arr == sid
        row = {}
        for cid in cluster_ids:
            cmask = mask & (labels == cid)
            for ch_idx, ch in enumerate(functional_channels):
                col_name = f"cluster{cid}_{ch}_median"
                if cmask.sum() > 0 and ch_idx < data.shape[1]:
                    row[col_name] = float(np.median(data[cmask, ch_idx]))
                else:
                    row[col_name] = float("nan")
        rows[sid] = row

    return pd.DataFrame(rows).T.fillna(0.0)


def _fit_glmnet(
    X: np.ndarray,
    y: np.ndarray,
    endpoint_type: EndpointType,
    n_cv: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    """
    Régression Lasso/LogisticCV pour identifier features importantes.

    Returns:
        coefs: ndarray (n_features,)
        score: CV score
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if endpoint_type == "classification":
        n_classes = len(np.unique(y))
        min_class_count = min(np.bincount(y.astype(int))) if n_classes > 1 else len(y)
        safe_cv = min(n_cv, min_class_count)
        model = LogisticRegressionCV(
            cv=safe_cv,
            penalty="l1",
            solver="saga",
            max_iter=2000,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_scaled, y)
        coefs = model.coef_.ravel()
        score = model.score(X_scaled, y)
    else:
        model = LassoCV(cv=n_cv, max_iter=5000, random_state=seed, n_jobs=-1)
        model.fit(X_scaled, y)
        coefs = model.coef_
        score = model.score(X_scaled, y)

    return coefs, score


def _fit_sam(
    X: np.ndarray,
    y: np.ndarray,
    endpoint_type: EndpointType,
) -> Tuple[np.ndarray, float]:
    """
    SAM simplifié: t-test ou Mann-Whitney par feature, FDR correction.

    Returns:
        scores: ndarray (n_features,) — -log10(p_adjusted)
        score: proportion features significatives (FDR < 0.05)
    """
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    n_features = X.shape[1]
    pvals = np.ones(n_features)

    classes = np.unique(y)
    if len(classes) == 2 and endpoint_type == "classification":
        g1 = X[y == classes[0]]
        g2 = X[y == classes[1]]
        for i in range(n_features):
            _, p = stats.mannwhitneyu(g1[:, i], g2[:, i], alternative="two-sided")
            pvals[i] = p
    else:
        for i in range(n_features):
            _, p = stats.spearmanr(X[:, i], y)
            pvals[i] = p if not np.isnan(p) else 1.0

    try:
        _, pvals_adj, _, _ = multipletests(pvals, method="fdr_bh")
    except ImportError:
        pvals_adj = pvals

    scores = -np.log10(np.clip(pvals_adj, 1e-10, 1.0))
    n_sig = (pvals_adj < 0.05).sum()
    score = n_sig / n_features if n_features > 0 else 0.0

    return scores, score


# ---------------------------------------------------------------------------
# Stratégie principale
# ---------------------------------------------------------------------------


class CitrusStrategy:
    """
    Implémentation Python de l'algorithme Citrus (Bruggner et al., 2014).

    Identifie les clusters cellulaires dont les propriétés (abondance,
    expression) stratifient les groupes cliniques ou corrèlent avec
    un endpoint continu.

    Compatible avec IExperimentStrategy du pipeline PRISMA.
    """

    name = "citrus"

    def __init__(self, params: Optional[CitrusParams] = None) -> None:
        self.params = params or CitrusParams()

    def run_experiment(
        self,
        experiment: Experiment,
        endpoint: np.ndarray,
        endpoint_type: Optional[EndpointType] = None,
        **kwargs: Any,
    ) -> CitrusResult:
        """
        Exécute Citrus sur un Experiment complet.

        Args:
            experiment: Cohorte de samples cytométriques.
            endpoint: Vecteur (n_samples,) — labels (classification) ou
                      valeurs continues. Doit être aligné sur experiment.samples.
            endpoint_type: Override du type d'endpoint (optionnel).

        Returns:
            CitrusResult avec clusters stratifiants et leur importance.
        """
        p = self.params
        ep_type = endpoint_type or p.endpoint_type

        valid_samples = [s for s in experiment.samples if not s.events.empty]
        if len(valid_samples) < 2:
            raise ValueError("Citrus requiert au moins 2 samples avec données")

        if len(endpoint) != len(valid_samples):
            raise ValueError(
                f"endpoint longueur {len(endpoint)} != {len(valid_samples)} samples"
            )

        # 1. Résoudre canaux de clustering
        all_channels = set()
        for s in valid_samples:
            all_channels.update(s.events.columns.tolist())

        clustering_channels = p.clustering_channels or sorted(all_channels)
        clustering_channels = [c for c in clustering_channels if c in all_channels]

        functional_channels = p.functional_channels or clustering_channels

        logger.info(
            "Citrus: %d samples, %d canaux clustering, endpoint=%s",
            len(valid_samples),
            len(clustering_channels),
            ep_type,
        )

        # 2. Construire matrice combinée
        combined, sample_ids, sample_indices = _build_combined_matrix(
            experiment,
            clustering_channels,
            p.n_cells_per_sample,
            p.random_seed,
        )

        # 3. Clustering hiérarchique
        labels = _hierarchical_cluster(
            combined,
            method=p.linkage_method,
            min_cluster_size_frac=p.min_cluster_size_percent,
        )

        unique_labels, counts = np.unique(labels, return_counts=True)
        n_total = len(labels)
        min_size = int(p.min_cluster_size_percent * n_total)

        # Filtrer petits clusters
        valid_clusters = unique_labels[counts >= min_size]
        logger.info(
            "%d clusters totaux, %d après filtrage taille minimale (>= %d cellules)",
            len(unique_labels),
            len(valid_clusters),
            min_size,
        )

        # 4. Extraire features
        if p.feature_type in ("abundances", "both"):
            feat_abundance = _calculate_abundances(
                labels, sample_ids, valid_clusters, valid_samples
            )
        else:
            feat_abundance = pd.DataFrame(index=[s.sample_id for s in valid_samples])

        if p.feature_type in ("medians", "both") and functional_channels:
            func_idx = [
                clustering_channels.index(c)
                for c in functional_channels
                if c in clustering_channels
            ]
            data_func = combined[:, func_idx] if func_idx else combined
            func_ch_names = [clustering_channels[i] for i in func_idx] if func_idx else clustering_channels
            feat_medians = _calculate_medians(
                data_func, labels, sample_ids, valid_clusters,
                func_ch_names, valid_samples
            )
        else:
            feat_medians = pd.DataFrame(index=[s.sample_id for s in valid_samples])

        # Aligner sur valid_samples dans l'ordre
        sample_order = [s.sample_id for s in valid_samples]
        feat_abundance = feat_abundance.reindex(sample_order).fillna(0.0)
        feat_medians = feat_medians.reindex(sample_order).fillna(0.0)

        feature_matrix = pd.concat([feat_abundance, feat_medians], axis=1)
        X = feature_matrix.values.astype(float)
        y = np.array(endpoint)

        if X.shape[1] == 0:
            raise ValueError("Matrice de features vide après extraction")

        # 5. Modèle statistique
        logger.info("Ajustement modèle %s (endpoint=%s)...", p.model_type, ep_type)

        if p.model_type == "sam":
            coefs, model_score = _fit_sam(X, y, ep_type)
        else:
            try:
                coefs, model_score = _fit_glmnet(X, y, ep_type, p.n_cv_folds, p.random_seed)
            except Exception as e:
                logger.warning("glmnet échoué (%s), fallback SAM", e)
                coefs, model_score = _fit_sam(X, y, ep_type)

        # 6. Identifier clusters stratifiants
        feature_cols = feature_matrix.columns.tolist()
        cluster_importance: Dict[int, float] = {cid: 0.0 for cid in valid_clusters}

        for feat_idx, col in enumerate(feature_cols):
            if feat_idx >= len(coefs):
                break
            imp = abs(float(coefs[feat_idx]))
            if imp == 0.0:
                continue
            col_str = str(col)
            for cid in valid_clusters:
                if col_str == str(cid) or col_str.startswith(f"cluster{cid}_"):
                    cluster_importance[cid] = max(cluster_importance.get(cid, 0.0), imp)

        # Construire CitrusCluster pour clusters avec importance > 0
        stratifying = []
        labels_arr = np.array(sample_ids)

        for cid in valid_clusters:
            imp = cluster_importance.get(cid, 0.0)
            if imp == 0.0 and p.model_type != "sam":
                continue

            cid_mask = labels == cid
            size = int(cid_mask.sum())
            size_pct = size / n_total * 100

            # Abondance moyenne inter-samples
            if str(cid) in feat_abundance.columns:
                mean_ab = float(feat_abundance[cid].mean())
            elif cid in feat_abundance.columns:
                mean_ab = float(feat_abundance[cid].mean())
            else:
                mean_ab = size_pct / 100

            # Médianes d'expression
            median_expr = {}
            for ch_idx, ch in enumerate(clustering_channels):
                if ch_idx < combined.shape[1] and cid_mask.any():
                    median_expr[ch] = float(np.median(combined[cid_mask, ch_idx]))

            # Abondances par sample
            per_sample_ab = {}
            for sid in sample_order:
                smask = np.array(sample_ids) == sid
                if smask.any():
                    per_sample_ab[sid] = float((labels[smask] == cid).sum() / smask.sum())
                else:
                    per_sample_ab[sid] = 0.0

            stratifying.append(CitrusCluster(
                cluster_id=int(cid),
                size=size,
                size_percent=size_pct,
                feature_importance=imp,
                mean_abundance=mean_ab,
                median_expressions=median_expr,
                per_sample_abundances=per_sample_ab,
            ))

        # Trier par importance décroissante
        stratifying.sort(key=lambda c: c.feature_importance, reverse=True)

        logger.info(
            "Citrus terminé: %d clusters stratifiants, score CV=%.3f",
            len(stratifying),
            model_score,
        )

        return CitrusResult(
            stratifying_clusters=stratifying,
            all_cluster_labels=labels,
            feature_matrix=feature_matrix,
            model_score=model_score,
            model_type=p.model_type,
            endpoint_type=ep_type,
            n_samples=len(valid_samples),
            n_cells_total=n_total,
            clustering_channels=clustering_channels,
            params=p,
        )
