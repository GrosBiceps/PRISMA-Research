"""
statistics.py — Calculs statistiques pour l'analyse cytométrique comparative.

Fournit les tests non-paramétriques recommandés pour la cytométrie en flux:
  - Mann-Whitney U (comparaison de médianes entre conditions)
  - Kolmogorov-Smirnov (comparaison de distributions)
  - Fold-change MFI entre condition et référence NBM (critère ELN MRD)
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    warnings.warn("scipy requis pour les tests statistiques: pip install scipy")

from config.constants import (
    MRD_LOD,
    MRD_LOQ,
    NBM_FREQ_MAX,
    MRD_FOLD_CHANGE_THRESHOLD,
)
from src.utils.logger import get_logger

_logger = get_logger("analysis.statistics")


def mann_whitney_u(
    group1: np.ndarray,
    group2: np.ndarray,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Test de Mann-Whitney U (non-paramétrique, adapté à la cytométrie).

    Args:
        group1: Valeurs du groupe 1.
        group2: Valeurs du groupe 2.
        alternative: "two-sided", "less", ou "greater".

    Returns:
        Tuple (statistic, p_value).
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy requis: pip install scipy")

    if len(group1) < 2 or len(group2) < 2:
        return float("nan"), float("nan")

    result = _scipy_stats.mannwhitneyu(group1, group2, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


def kolmogorov_smirnov(
    group1: np.ndarray,
    group2: np.ndarray,
) -> Tuple[float, float]:
    """
    Test de Kolmogorov-Smirnov (comparaison de distributions complètes).

    Args:
        group1: Échantillon du groupe 1.
        group2: Échantillon du groupe 2.

    Returns:
        Tuple (statistic, p_value).
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy requis: pip install scipy")

    if len(group1) < 2 or len(group2) < 2:
        return float("nan"), float("nan")

    result = _scipy_stats.ks_2samp(group1, group2)
    return float(result.statistic), float(result.pvalue)


def compute_fold_change(
    mfi_patient: float,
    mfi_nbm: float,
    eps: float = 1e-6,
) -> float:
    """
    Calcule le fold-change MFI patient / NBM.

    Utilisé pour la détection MRD: FC > 1.9× = positif (ELN 2022).

    Args:
        mfi_patient: MFI de la population patient.
        mfi_nbm: MFI de la même population en NBM de référence.
        eps: Epsilon pour éviter div/0.

    Returns:
        Fold-change.
    """
    return float(mfi_patient / max(mfi_nbm, eps))


def assess_mrd_status(
    blast_frequency: float,
    fold_change_vs_nbm: Optional[float] = None,
) -> Dict[str, object]:
    """
    Évalue le statut MRD selon les critères ELN 2022.

    Critères:
      - Positif si fréquence > LOQ (5e-5) ET fold-change > 1.9×
      - Détectable si fréquence entre LOD (9e-5) et LOQ
      - Non détectable si < LOD

    Args:
        blast_frequency: Fréquence des blastes (fraction, pas pourcentage).
        fold_change_vs_nbm: FC vs NBM (optionnel, critère supplémentaire).

    Returns:
        Dict {status, mrd_positive, above_lod, above_loq, fc_positive}.
    """
    above_lod = blast_frequency >= MRD_LOD
    above_loq = blast_frequency >= MRD_LOQ
    fc_positive = (
        fold_change_vs_nbm is not None
        and fold_change_vs_nbm >= MRD_FOLD_CHANGE_THRESHOLD
    )
    nbm_exceeded = blast_frequency >= NBM_FREQ_MAX

    mrd_positive = above_loq and (fold_change_vs_nbm is None or fc_positive)

    if not above_lod:
        status = "MRD_NEGATIVE"
    elif not above_loq:
        status = "MRD_DETECTABLE_BELOW_LOQ"
    elif mrd_positive:
        status = "MRD_POSITIVE"
    else:
        status = "MRD_INDETERMINATE"

    return {
        "status": status,
        "mrd_positive": mrd_positive,
        "blast_frequency": blast_frequency,
        "blast_pct": round(blast_frequency * 100, 4),
        "above_lod": above_lod,
        "above_loq": above_loq,
        "fc_vs_nbm": fold_change_vs_nbm,
        "fc_positive": fc_positive,
        "nbm_frequency_exceeded": nbm_exceeded,
        "lod_threshold": MRD_LOD,
        "loq_threshold": MRD_LOQ,
        "fc_threshold": MRD_FOLD_CHANGE_THRESHOLD,
    }


def compare_conditions_per_cluster(
    df: pd.DataFrame,
    marker_columns: List[str],
    cluster_column: str = "FlowSOM_metacluster",
    condition_column: str = "condition",
    condition1: str = "Sain",
    condition2: str = "Pathologique",
) -> pd.DataFrame:
    """
    Compare statistiquement les conditions pour chaque cluster et marqueur.

    Pour chaque (cluster, marqueur): Mann-Whitney U + fold-change.

    Args:
        df: DataFrame cellulaire.
        marker_columns: Marqueurs à comparer.
        cluster_column: Colonne d'assignation de cluster.
        condition_column: Colonne de condition.
        condition1: Nom de la condition 1 (ex: "Sain").
        condition2: Nom de la condition 2 (ex: "Pathologique").

    Returns:
        DataFrame avec [cluster, marker, stat, p_value, fold_change, significant].
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy requis pour compare_conditions_per_cluster")

    records = []
    p_adj_threshold = 0.05

    for cluster_id in sorted(df[cluster_column].unique()):
        mask_cluster = df[cluster_column] == cluster_id
        df_cluster = df[mask_cluster]

        grp1 = df_cluster[df_cluster[condition_column] == condition1]
        grp2 = df_cluster[df_cluster[condition_column] == condition2]

        if len(grp1) < 5 or len(grp2) < 5:
            continue

        for marker in marker_columns:
            if marker not in df.columns:
                continue

            v1 = grp1[marker].dropna().values
            v2 = grp2[marker].dropna().values

            if len(v1) < 2 or len(v2) < 2:
                continue

            stat, pval = mann_whitney_u(v1, v2)
            median1 = float(np.median(v1))
            median2 = float(np.median(v2))
            fc = compute_fold_change(median2, median1)

            records.append(
                {
                    "cluster": cluster_id,
                    "marker": marker,
                    "n_cond1": len(v1),
                    "n_cond2": len(v2),
                    "median_cond1": round(median1, 4),
                    "median_cond2": round(median2, 4),
                    "fold_change": round(fc, 4),
                    "mw_stat": round(stat, 2),
                    "p_value": round(pval, 6),
                    "significant": pval < p_adj_threshold,
                }
            )

    return pd.DataFrame(records)
