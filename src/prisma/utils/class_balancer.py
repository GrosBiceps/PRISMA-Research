"""
class_balancer.py — Déséquilibre Maîtrisé pour le pool d'entraînement FlowSOM.

Problème : dans les données MRD/AML, les cellules pathologiques (blastes, class==1)
représentent parfois < 1% du pool. FlowSOM, optimisé pour la densité, rate les
clusters rares. Ce module impose un rapport sain/patho contrôlé AVANT le SOM.

Usage:
    from src.prisma.utils.class_balancer import equilibrer_pool_flowsom

    df_balanced = equilibrer_pool_flowsom(
        samples=processed_samples,
        nbm_ids=["NBM_01.fcs", "NBM_02.fcs"],
        balance_conditions=True,
        imbalance_ratio=2.0,   # 2 cellules saines pour 1 blaste
        seed=42,
    )
    # → df_balanced est prêt à être converti en FlowSample ou passé directement au SOM.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from src.prisma.core.models_legacy.sample import FlowSample

_logger = logging.getLogger("utils.class_balancer")


# ──────────────────────────────────────────────────────────────────────────────
# Constante : nom de la colonne de classe dans les données prétraitées
# Doit correspondre à la valeur réelle dans FlowSample.data (colonne "class" ou
# à la convention de condition : "Pathological" / "Healthy").
# On supporte les deux conventions via le paramètre `class_column`.
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULT_CLASS_COL = "class"        # colonne binaire  : 0 = sain, 1 = blaste
_DEFAULT_COND_COL = "condition"     # colonne textuelle : "Sain" / "Pathologique"
_PATHO_LABELS = {"Pathological", "Pathologique", "pathologique", "pathological", "Patho", "patho"}
_HEALTHY_LABELS = {"Healthy", "Sain", "sain", "healthy", "Normal", "normal", "NBM"}


def _to_flat_dataframe(samples: List[FlowSample]) -> pd.DataFrame:
    """
    Concatène tous les FlowSample en un DataFrame plat.

    Ajoute automatiquement les colonnes ``condition`` et ``file_origin``
    depuis les métadonnées de chaque FlowSample.
    """
    frames: List[pd.DataFrame] = []
    for s in samples:
        df = s.data.copy()
        if _DEFAULT_COND_COL not in df.columns:
            df[_DEFAULT_COND_COL] = s.condition
        if "file_origin" not in df.columns:
            df["file_origin"] = s.name
        frames.append(df)

    if not frames:
        raise ValueError("equilibrer_pool_flowsom: aucun FlowSample fourni.")

    return pd.concat(frames, ignore_index=True)


def _detect_patho_mask(df: pd.DataFrame) -> pd.Series:
    """
    Retourne un masque booléen identifiant les cellules pathologiques.

    Priorité :
      1. Colonne ``class`` (binaire int) : class == 1
      2. Colonne ``condition`` (str)     : condition == "Pathological"
    """
    if _DEFAULT_CLASS_COL in df.columns:
        return df[_DEFAULT_CLASS_COL].astype(int) == 1
    if _DEFAULT_COND_COL in df.columns:
        return df[_DEFAULT_COND_COL].astype(str).isin(_PATHO_LABELS)
    raise KeyError(
        "Impossible de détecter les cellules pathologiques : "
        f"ni la colonne '{_DEFAULT_CLASS_COL}' ni '{_DEFAULT_COND_COL}' "
        "n'est présente dans le DataFrame."
    )


def equilibrer_pool_flowsom(
    samples: List[FlowSample],
    nbm_ids: List[str],
    balance_conditions: bool = True,
    imbalance_ratio: float = 1.0,
    seed: int = 42,
    allow_oversampling: bool = False,
) -> pd.DataFrame:
    """
    Crée un pool d'entraînement FlowSOM avec un déséquilibre maîtrisé.

    Le pool résultant contient :
      - **Toutes** les cellules pathologiques (blastes, class==1 ou condition=="Pathological")
        provenant de la moelle pathologique.
      - Un sous-échantillon équitable des cellules saines (class==0) distribué
        uniformément entre les fichiers NBM listés dans ``nbm_ids``.

    Le rapport final est : n_sain = n_patho × imbalance_ratio.

    Args:
        samples:
            Liste des FlowSample prétraités (après gating, transformation,
            normalisation). Doit inclure les échantillons sains (NBM) ET
            les échantillons pathologiques.
        nbm_ids:
            Liste des noms de fichiers NBM (``FlowSample.name``) à utiliser
            comme source de cellules saines. Seuls ces fichiers contribuent
            au pool sain pour garantir un tirage équitable inter-donneur.
        balance_conditions:
            Si ``False``, la fonction retourne le pool non modifié (toutes
            les cellules concaténées). Permet de désactiver le rééquilibrage
            sans changer le code appelant.
        imbalance_ratio:
            Rapport souhaité n_sain / n_patho.
            - 1.0 → 50/50 (équilibre parfait)
            - 2.0 → 2 cellules saines pour 1 blaste
            - 5.0 → 5 cellules saines pour 1 blaste (léger déséquilibre intentionnel)
        seed:
            Graine aléatoire pour la reproductibilité du .sample().
        allow_oversampling:
            Si True, le tirage par fichier NBM utilise replace=True quand les
            cellules disponibles sont insuffisantes pour atteindre le quota.
            Garantit le ratio cible même si le pool NBM est petit.
            Si False (défaut), le tirage est limité aux cellules disponibles
            et un WARNING est émis.

    Returns:
        DataFrame mélangé prêt à être ingéré par FlowSOM.
        Colonnes : tous les marqueurs prétraités + ``condition`` + ``file_origin``
        + (``class`` si présente).

    Raises:
        ValueError: Si aucune cellule pathologique n'est trouvée dans les
            échantillons ou si ``nbm_ids`` est vide quand ``balance_conditions``
            est True.
    """
    # ── Étape 0 : court-circuit si rééquilibrage désactivé ────────────────────
    if not balance_conditions:
        _logger.info(
            "equilibrer_pool_flowsom: balance_conditions=False → pool non modifié."
        )
        df_all = _to_flat_dataframe(samples)
        return df_all.sample(frac=1, random_state=seed).reset_index(drop=True)

    if not nbm_ids:
        raise ValueError(
            "equilibrer_pool_flowsom: nbm_ids ne peut pas être vide "
            "quand balance_conditions=True."
        )

    # ── Étape 1 : mise à plat ─────────────────────────────────────────────────
    df_all = _to_flat_dataframe(samples)

    # ── Étape 2 : séparation patho / sain ────────────────────────────────────
    patho_mask = _detect_patho_mask(df_all)
    nbm_mask = df_all["file_origin"].isin(nbm_ids)

    df_patho = df_all[patho_mask].copy()
    df_sain_nbm = df_all[~patho_mask & nbm_mask].copy()

    n_patho = len(df_patho)
    if n_patho == 0:
        raise ValueError(
            "equilibrer_pool_flowsom: aucune cellule pathologique (class==1 "
            "ou condition=='Pathological') trouvée dans les échantillons fournis. "
            "Vérifiez que vos FCS pathologiques sont bien inclus dans `samples`."
        )
    if len(df_sain_nbm) == 0:
        raise ValueError(
            f"equilibrer_pool_flowsom: aucune cellule saine trouvée dans les "
            f"fichiers NBM spécifiés ({nbm_ids}). "
            "Vérifiez que les noms correspondent à FlowSample.name."
        )

    _logger.info(
        "Pool brut — pathologiques: %d | sains NBM disponibles: %d",
        n_patho,
        len(df_sain_nbm),
    )

    # ── Étape 2b : calcul de la cible saine globale ───────────────────────────
    cible_saine_totale = int(round(n_patho * imbalance_ratio))
    n_nbm_fichiers = len(nbm_ids)
    quota_par_fichier = int(np.ceil(cible_saine_totale / n_nbm_fichiers))

    _logger.info(
        "Cible saine: %d cellules (ratio %.1f×) → quota/fichier NBM: %d (sur %d fichiers)",
        cible_saine_totale,
        imbalance_ratio,
        quota_par_fichier,
        n_nbm_fichiers,
    )

    # ── Étape 3 : tirage équitable par fichier NBM ───────────────────────────
    rng = np.random.default_rng(seed)
    sous_echantillons: List[pd.DataFrame] = []

    for nbm_id in nbm_ids:
        df_nbm_i = df_sain_nbm[df_sain_nbm["file_origin"] == nbm_id]
        n_dispo = len(df_nbm_i)

        if n_dispo == 0:
            _logger.warning(
                "NBM '%s' absent ou vide dans les échantillons — ignoré.", nbm_id
            )
            continue

        rseed = int(rng.integers(0, 2**31))
        if n_dispo >= quota_par_fichier:
            # Cas normal : assez de cellules, tirage sans remplacement
            echantillon = df_nbm_i.sample(n=quota_par_fichier, random_state=rseed)
        elif allow_oversampling:
            # Oversampling activé : tirage avec remplacement pour atteindre le quota
            _logger.info(
                "NBM '%s': oversampling %d → %d cellules (replace=True).",
                nbm_id, n_dispo, quota_par_fichier,
            )
            echantillon = df_nbm_i.sample(
                n=quota_par_fichier, replace=True, random_state=rseed
            )
        else:
            # Tirage limité au disponible, WARNING
            _logger.warning(
                "NBM '%s': seulement %d cellules disponibles (quota=%d) — tirage limité. "
                "Activez allow_oversampling=true pour atteindre le ratio cible.",
                nbm_id, n_dispo, quota_par_fichier,
            )
            echantillon = df_nbm_i.sample(n=n_dispo, random_state=rseed)

        sous_echantillons.append(echantillon)
        _logger.debug("  NBM '%s': %d/%d cellules tirées.", nbm_id, len(echantillon), n_dispo)

    if not sous_echantillons:
        raise ValueError(
            "equilibrer_pool_flowsom: aucun sous-échantillon sain n'a pu être "
            "constitué. Vérifiez que nbm_ids contient des noms valides."
        )

    # ── Étape 4 : reconstruction et mélange ──────────────────────────────────
    df_sain_pool = pd.concat(sous_echantillons, ignore_index=True)
    df_balanced = pd.concat([df_patho, df_sain_pool], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_final_patho = int(patho_mask.values[df_balanced.index if False else slice(None)].sum()
                        if False else df_balanced.pipe(lambda d: _detect_patho_mask(d)).sum())
    n_final_sain = len(df_balanced) - n_final_patho
    ratio_final = n_final_sain / max(n_final_patho, 1)

    _logger.info(
        "Pool équilibré final: %d cellules "
        "(patho=%d | sain=%d | ratio_réel=%.2f×)",
        len(df_balanced),
        n_final_patho,
        n_final_sain,
        ratio_final,
    )

    return df_balanced
