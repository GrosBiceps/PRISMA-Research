"""
patho_fcs_exporter.py — Export FCS restreint à la moelle pathologique avec annotation MRD.

Génère un fichier FCS contenant uniquement les cellules de la moelle pathologique,
enrichi de toutes les métriques FlowSOM (cluster, métacluster, coordonnées xGrid/yGrid,
xNodes/yNodes, size) et des colonnes d'annotation suivantes :

  Is_MRD        : 1.0 = cellule MRD positive, 0.0 = non-MRD
  CD45_Status   : 1.0 = CD45+  (bright), 2.0 = CD45dim, 3.0 = CD45- (neg)
  CD34_Status   : 1.0 = CD34+, 0.0 = CD34-
  Debris_Flag   : 1.0 = débris (FSC/SSC hors-population), 0.0 = cellule valide
  Doublet_Flag  : 1.0 = doublet (FSC-A/FSC-H ratio anormal), 0.0 = singlet

Noms choisis pour être directement lisibles dans Kaluza / FlowJo sans ambiguïté.

La méthode utilisée pour Is_MRD est configurable : "jf" (méthode JF) ou "flo" (méthode Flo).
Par défaut : "flo".

Usage:
    from src.io.patho_fcs_exporter import export_patho_mrd_fcs
    export_patho_mrd_fcs(df_fcs, mrd_result, output_path, mrd_method="flo")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.legacy_clinical.mrd_calculator import MRDResult
from src.io.fcs_writer import export_to_fcs_kaluza
from src.utils.logger import get_logger

_logger = get_logger("io.patho_fcs_exporter")


# ── Seuils heuristiques pour les annotations phénotypiques ───────────────────
# CD45 : percentile 15 → dim, percentile 5 → négatif
_CD45_DIM_PERCENTILE = 15.0
_CD45_NEG_PERCENTILE = 5.0
# CD34 : percentile 85 (valeur haute = positif)
_CD34_POS_PERCENTILE = 85.0
# Doublets : ratio FSC-A / FSC-H > seuil → doublet
_DOUBLET_RATIO_THRESHOLD = 1.20  # FSC-A/FSC-H > 1.20 → probable doublet
# Débris : percentile 5 sur FSC-A (très petite taille)
_DEBRIS_FSC_PERCENTILE = 5.0
_DEBRIS_SSC_PERCENTILE = 5.0  # ET SSC très bas


def _find_marker_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Cherche parmi les colonnes du DataFrame le premier nom qui matche
    une liste de candidats (insensible à la casse, correspondance partielle).
    Retourne None si aucun candidat trouvé.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        cand_l = cand.lower()
        # Correspondance exacte
        if cand_l in cols_lower:
            return cols_lower[cand_l]
        # Correspondance partielle (ex: "cd45 ko" matche "cd45")
        for col_l, col in cols_lower.items():
            if cand_l in col_l or col_l.startswith(cand_l):
                return col
    return None


def _compute_cd45_status(df: pd.DataFrame) -> np.ndarray:
    """
    Calcule CD45_Status pour chaque cellule :
      1.0 → CD45+  (bright : au-dessus du percentile _CD45_DIM_PERCENTILE)
      2.0 → CD45dim (entre percentiles _CD45_NEG_PERCENTILE et _CD45_DIM_PERCENTILE)
      3.0 → CD45-  (négatif : sous le percentile _CD45_NEG_PERCENTILE)
      0.0 → non déterminé (marqueur absent)
    """
    col = _find_marker_col(df, ["CD45", "cd45", "CD45-KO", "CD45KO", "CD45 KO", "CD45-PerCP"])
    if col is None:
        _logger.warning("CD45_Status: colonne CD45 non trouvée — valeur 0.0 (N/D)")
        return np.zeros(len(df), dtype=np.float32)

    vals = df[col].values.astype(np.float64)
    # Calcul des seuils sur les valeurs finies
    finite_mask = np.isfinite(vals)
    if finite_mask.sum() < 10:
        return np.zeros(len(df), dtype=np.float32)

    p_neg = np.percentile(vals[finite_mask], _CD45_NEG_PERCENTILE)
    p_dim = np.percentile(vals[finite_mask], _CD45_DIM_PERCENTILE)

    status = np.where(vals < p_neg, 3.0,          # CD45-
               np.where(vals < p_dim, 2.0,         # CD45dim
                        1.0)).astype(np.float32)   # CD45+

    n_bright = int((status == 1.0).sum())
    n_dim    = int((status == 2.0).sum())
    n_neg    = int((status == 3.0).sum())
    _logger.info(
        "  CD45_Status (%s) : CD45+=%d  CD45dim=%d  CD45-=%d  |  seuils: neg<%.1f dim<%.1f",
        col, n_bright, n_dim, n_neg, p_neg, p_dim,
    )
    return status


def _compute_cd34_status(df: pd.DataFrame) -> np.ndarray:
    """
    Calcule CD34_Status pour chaque cellule :
      1.0 → CD34+  (valeur au-dessus du percentile _CD34_POS_PERCENTILE)
      0.0 → CD34-  (ou marqueur absent)
    """
    col = _find_marker_col(df, ["CD34", "cd34", "CD34-PE", "CD34-FITC", "CD34 PE"])
    if col is None:
        _logger.warning("CD34_Status: colonne CD34 non trouvée — valeur 0.0 (N/D)")
        return np.zeros(len(df), dtype=np.float32)

    vals = df[col].values.astype(np.float64)
    finite_mask = np.isfinite(vals)
    if finite_mask.sum() < 10:
        return np.zeros(len(df), dtype=np.float32)

    p_pos = np.percentile(vals[finite_mask], _CD34_POS_PERCENTILE)
    status = (vals >= p_pos).astype(np.float32)

    n_pos = int(status.sum())
    n_neg = len(status) - n_pos
    _logger.info(
        "  CD34_Status (%s) : CD34+=%d  CD34-=%d  |  seuil: %.1f (p%g)",
        col, n_pos, n_neg, p_pos, _CD34_POS_PERCENTILE,
    )
    return status


def _compute_debris_flag(df: pd.DataFrame) -> np.ndarray:
    """
    Calcule Debris_Flag pour chaque cellule :
      1.0 → débris (FSC-A très bas ET SSC-A très bas)
      0.0 → cellule valide

    Utilise les percentiles bas sur FSC-A et SSC-A.
    Si les colonnes sont absentes, retourne 0.0 pour toutes les cellules.
    """
    fsc_col = _find_marker_col(df, ["FSC-A", "FSC_A", "FSCA", "FSC"])
    ssc_col = _find_marker_col(df, ["SSC-A", "SSC_A", "SSCA", "SSC"])

    if fsc_col is None or ssc_col is None:
        _logger.warning(
            "Debris_Flag: FSC-A (%s) ou SSC-A (%s) non trouvé — valeur 0.0 (N/D)",
            fsc_col, ssc_col,
        )
        return np.zeros(len(df), dtype=np.float32)

    fsc = df[fsc_col].values.astype(np.float64)
    ssc = df[ssc_col].values.astype(np.float64)
    finite_mask = np.isfinite(fsc) & np.isfinite(ssc)

    if finite_mask.sum() < 10:
        return np.zeros(len(df), dtype=np.float32)

    fsc_thresh = np.percentile(fsc[finite_mask], _DEBRIS_FSC_PERCENTILE)
    ssc_thresh = np.percentile(ssc[finite_mask], _DEBRIS_SSC_PERCENTILE)

    # Débris = FSC-A très bas ET SSC-A très bas (coin inférieur gauche du plot FSC/SSC)
    debris_mask = (fsc < fsc_thresh) & (ssc < ssc_thresh)
    flag = debris_mask.astype(np.float32)

    n_debris = int(flag.sum())
    _logger.info(
        "  Debris_Flag : %d débris / %d cellules (FSC<%.0f ET SSC<%.0f)",
        n_debris, len(df), fsc_thresh, ssc_thresh,
    )
    return flag


def _compute_doublet_flag(df: pd.DataFrame) -> np.ndarray:
    """
    Calcule Doublet_Flag pour chaque cellule :
      1.0 → doublet probable (ratio FSC-A / FSC-H > _DOUBLET_RATIO_THRESHOLD)
      0.0 → singlet

    Les doublets ont un FSC-A élevé mais un FSC-H proportionnellement moins élevé,
    ce qui produit un ratio FSC-A/FSC-H > 1 (classiquement > 1.15–1.25).
    Si FSC-H est absent, retourne 0.0 pour toutes les cellules.
    """
    fsc_a_col = _find_marker_col(df, ["FSC-A", "FSC_A", "FSCA"])
    fsc_h_col = _find_marker_col(df, ["FSC-H", "FSC_H", "FSCH"])

    if fsc_a_col is None or fsc_h_col is None:
        _logger.warning(
            "Doublet_Flag: FSC-A (%s) ou FSC-H (%s) non trouvé — valeur 0.0 (N/D)",
            fsc_a_col, fsc_h_col,
        )
        return np.zeros(len(df), dtype=np.float32)

    fsc_a = df[fsc_a_col].values.astype(np.float64)
    fsc_h = df[fsc_h_col].values.astype(np.float64)

    # Éviter division par zéro : remplacer 0 par NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(fsc_h > 0, fsc_a / fsc_h, np.nan)

    doublet_mask = np.where(np.isfinite(ratio), ratio > _DOUBLET_RATIO_THRESHOLD, False)
    flag = doublet_mask.astype(np.float32)

    n_doublets = int(flag.sum())
    _logger.info(
        "  Doublet_Flag : %d doublets / %d cellules (FSC-A/FSC-H > %.2f)",
        n_doublets, len(df), _DOUBLET_RATIO_THRESHOLD,
    )
    return flag


def export_patho_mrd_fcs(
    df_fcs: pd.DataFrame,
    mrd_result: "MRDResult",
    output_path: Path | str,
    mrd_method: str = "flo",
    condition_patho: str = "Pathologique",
) -> bool:
    """
    Exporte un FCS restreint aux cellules de la moelle pathologique avec colonne Is_MRD.

    Le DataFrame ``df_fcs`` doit contenir les colonnes suivantes (produites par
    FlowSOMPipeline._build_fcs_dataframe) :
      - Tous les marqueurs bruts (pré-transformation)
      - FlowSOM_metacluster  (1-based, float32)
      - FlowSOM_cluster      (1-based, float32)
      - xGrid, yGrid         (coordonnées grille SOM avec jitter)
      - xNodes, yNodes       (coordonnées MST avec jitter)
      - size                 (nombre de cellules par nœud SOM)
      - Condition            (string "Sain" / "Pathologique")
      - Condition_Num        (float32  1.0 = Sain, 2.0 = Pathologique)
      - File_Origin          (string)

    La colonne ``Is_MRD`` est ajoutée :
      - 1.0  →  cellule appartenant à un nœud SOM classé MRD selon la méthode choisie
      - 0.0  →  cellule non-MRD

    Seules les colonnes numériques sont écrites dans le FCS (``fcswrite`` requis).
    ``Condition`` et ``File_Origin`` (strings) sont implicitement exclues.

    Args:
        df_fcs:          DataFrame complet produit par _build_fcs_dataframe.
        mrd_result:      Objet MRDResult issu de compute_mrd().
                         Peut être None — dans ce cas l'export est annulé.
        output_path:     Chemin de sortie du fichier .fcs.
        mrd_method:      Méthode Is_MRD : "flo" (défaut) ou "jf".
        condition_patho: Label de la condition pathologique dans la colonne Condition.

    Returns:
        True si succès, False sinon.
    """
    if mrd_result is None:
        _logger.warning("export_patho_mrd_fcs: mrd_result est None — export annulé.")
        return False

    if df_fcs is None or df_fcs.empty:
        _logger.warning("export_patho_mrd_fcs: df_fcs vide ou None — export annulé.")
        return False

    mrd_method = mrd_method.lower().strip()
    if mrd_method not in ("jf", "flo"):
        _logger.warning(
            "export_patho_mrd_fcs: méthode '%s' inconnue — utilisation de 'flo'.",
            mrd_method,
        )
        mrd_method = "flo"

    # ── 1. Filtrer les cellules pathologiques ─────────────────────────────────
    if "Condition" not in df_fcs.columns:
        _logger.error(
            "export_patho_mrd_fcs: colonne 'Condition' absente du df_fcs — export annulé."
        )
        return False

    mask_patho = df_fcs["Condition"] == condition_patho
    n_patho = int(mask_patho.sum())

    if n_patho == 0:
        _logger.warning(
            "export_patho_mrd_fcs: aucune cellule '%s' dans df_fcs — export annulé.",
            condition_patho,
        )
        return False

    _logger.info(
        "export_patho_mrd_fcs: %d cellules pathologiques sélectionnées / %d total",
        n_patho,
        len(df_fcs),
    )

    df_patho = df_fcs[mask_patho].copy()

    # ── 2. Construire le mapping nœud SOM → Is_MRD ───────────────────────────
    # FlowSOM_cluster est 1-based → convertir en 0-based pour rejoindre per_node
    per_node = getattr(mrd_result, "per_node", [])
    if not per_node:
        _logger.warning(
            "export_patho_mrd_fcs: mrd_result.per_node vide — Is_MRD = 0 pour toutes les cellules."
        )
        df_patho["Is_MRD"] = np.float32(0.0)
    else:
        # Choisir le flag selon la méthode demandée, avec fallback robuste:
        # si la méthode choisie ne détecte aucun nœud mais l'autre oui,
        # on bascule automatiquement pour éviter un Is_MRD tout à 0 incohérent.
        n_nodes_jf = int(
            sum(bool(getattr(node, "is_mrd_jf", False)) for node in per_node)
        )
        n_nodes_flo = int(
            sum(bool(getattr(node, "is_mrd_flo", False)) for node in per_node)
        )

        effective_method = mrd_method
        if mrd_method == "flo" and n_nodes_flo == 0 and n_nodes_jf > 0:
            _logger.warning(
                "Is_MRD: méthode FLO vide (0 nœud MRD) mais JF détecte %d nœud(s) — bascule automatique vers JF.",
                n_nodes_jf,
            )
            effective_method = "jf"
        elif mrd_method == "jf" and n_nodes_jf == 0 and n_nodes_flo > 0:
            _logger.warning(
                "Is_MRD: méthode JF vide (0 nœud MRD) mais FLO détecte %d nœud(s) — bascule automatique vers FLO.",
                n_nodes_flo,
            )
            effective_method = "flo"

        flag_attr = "is_mrd_flo" if effective_method == "flo" else "is_mrd_jf"
        node_to_mrd: dict = {
            node.cluster_id: float(getattr(node, flag_attr, False)) for node in per_node
        }

        # FlowSOM_cluster est 1-based → node_id 0-based = cluster - 1
        cluster_col = np.rint(df_patho["FlowSOM_cluster"].values).astype(np.int32) - 1
        cluster_col = np.clip(cluster_col, 0, None)
        n_nodes = int(cluster_col.max()) + 1 if len(cluster_col) > 0 else 0
        lookup = np.array(
            [node_to_mrd.get(nid, 0.0) for nid in range(n_nodes)],
            dtype=np.float32,
        )
        is_mrd_arr = lookup[cluster_col]
        df_patho["Is_MRD"] = is_mrd_arr

        n_mrd_cells = int((is_mrd_arr > 0).sum())
        pct_mrd = 100.0 * n_mrd_cells / n_patho if n_patho > 0 else 0.0
        _logger.info(
            "  Is_MRD (%s) : %d cellules MRD / %d patho (%.2f%%)",
            effective_method.upper(),
            n_mrd_cells,
            n_patho,
            pct_mrd,
        )

    # ── 3. Colonnes d'annotation phénotypique ────────────────────────────────
    # Utilise les données brutes de df_patho (pré-transformation) quand disponibles.
    # Fallback automatique sur les colonnes transformées si besoin.

    _logger.info("  Calcul des colonnes d'annotation phénotypique...")

    # CD45_Status : 1.0=CD45+bright  2.0=CD45dim  3.0=CD45-  0.0=N/D
    df_patho["CD45_Status"] = _compute_cd45_status(df_patho)

    # CD34_Status : 1.0=CD34+  0.0=CD34-  (0.0 si marqueur absent)
    df_patho["CD34_Status"] = _compute_cd34_status(df_patho)

    # Debris_Flag : 1.0=débris (FSC-A bas + SSC-A bas)  0.0=cellule valide
    df_patho["Debris_Flag"] = _compute_debris_flag(df_patho)

    # Doublet_Flag : 1.0=doublet probable (FSC-A/FSC-H > seuil)  0.0=singlet
    df_patho["Doublet_Flag"] = _compute_doublet_flag(df_patho)

    _n_cd45_bright = int((df_patho["CD45_Status"] == 1.0).sum())
    _n_cd45_dim    = int((df_patho["CD45_Status"] == 2.0).sum())
    _n_cd45_neg    = int((df_patho["CD45_Status"] == 3.0).sum())
    _n_cd34_pos    = int((df_patho["CD34_Status"] == 1.0).sum())
    _n_debris      = int((df_patho["Debris_Flag"]  == 1.0).sum())
    _n_doublets    = int((df_patho["Doublet_Flag"] == 1.0).sum())
    _logger.info(
        "  Annotation finale : CD45+=%d CD45dim=%d CD45-=%d | "
        "CD34+=%d | Débris=%d | Doublets=%d",
        _n_cd45_bright, _n_cd45_dim, _n_cd45_neg,
        _n_cd34_pos, _n_debris, _n_doublets,
    )

    # ── 4. Export FCS ─────────────────────────────────────────────────────────
    output_path = Path(output_path)
    ok = export_to_fcs_kaluza(df_patho, output_path)
    if ok:
        _logger.info(
            "FCS pathologique exporté : %s (%d events, %d colonnes)",
            output_path.name,
            n_patho,
            df_patho.select_dtypes(include=[np.number]).shape[1],
        )
    return ok
