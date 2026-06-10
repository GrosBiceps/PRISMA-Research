"""
cluster_distribution_exporter.py — Export de la distribution cellulaire par cluster SOM.

Produit deux fichiers combinés (niveau nœud SOM ET niveau métacluster) :
  1. Un rapport texte lisible (*.txt) avec tableaux ASCII alignés.
  2. Un CSV récapitulatif (*.csv) prêt pour Excel / R / downstream.

Colonnes produites (par cluster, trié % Patho décroissant) :
  cluster_id         — identifiant du nœud SOM (ou métacluster)
  metacluster        — métacluster assigné au nœud (uniquement dans la vue nœud)
  n_total            — nombre total de cellules dans le cluster
  pct_total          — % de l'ensemble des cellules
  n_sain             — nombre de cellules Sain dans le cluster
  pct_sain_in_cluster — % des cellules du cluster qui sont Sain
  pct_sain_of_sain   — % des cellules Sain totales dans ce cluster
  n_patho            — nombre de cellules Pathologiques dans le cluster
  pct_patho_in_cluster — % des cellules du cluster qui sont Pathologiques
  pct_patho_of_patho  — % des cellules Patho totales dans ce cluster

Tri : décroissant sur pct_patho_in_cluster.

Configuration via la section ``export_cluster_distribution`` du YAML :
  enabled: true
  level: "both"           # "node" | "metacluster" | "both"
  sort_by: "pct_patho_in_cluster"   # toute colonne numérique du tableau
  ascending: false
  sain_labels: ["Sain", "Normal", "NBM", "Healthy", "Moelle normale"]
  patho_labels: ["Pathologique", "Patho", "AML", "Disease"]
  txt_enabled: true
  csv_enabled: true
  txt_decimal_places: 1
  csv_decimal_places: 3
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from prisma.utils.logger import get_logger

_logger = get_logger("io.cluster_distribution_exporter")

# ─── Etiquettes par défaut ────────────────────────────────────────────────────
_DEFAULT_SAIN_LABELS = {"sain", "normal", "nbm", "healthy", "moelle normale"}
_DEFAULT_PATHO_LABELS = {"pathologique", "patho", "aml", "disease", "leucémie"}


# =============================================================================
#  Fonction principale
# =============================================================================


def export_cluster_distribution(
    clustering: np.ndarray,
    metaclustering: np.ndarray,
    condition_labels: np.ndarray,
    output_dir: Path | str,
    timestamp: str = "",
    export_cfg: Optional[Dict[str, Any]] = None,
    metacluster_names: Optional[Dict[int, str]] = None,
) -> Dict[str, str]:
    """
    Exporte la distribution Sain/Patho par nœud SOM et/ou par métacluster.

    Args:
        clustering:        Assignation cellule → nœud SOM (int, 0-indexé).
        metaclustering:    Assignation cellule → métacluster (int, 0-indexé).
        condition_labels:  Label de condition par cellule (ex: "Sain", "Pathologique").
        output_dir:        Dossier de sortie.
        timestamp:         Suffixe horodatage pour les noms de fichiers.
        export_cfg:        Paramètres issus du YAML (section export_cluster_distribution).
        metacluster_names: Optionnel — dict {id_mc: "nom"} pour renommage.

    Returns:
        Dict {clé: chemin_fichier} pour les fichiers produits.
    """
    cfg = export_cfg or {}
    if not cfg.get("enabled", True):
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    level = cfg.get("level", "both")  # "node" | "metacluster" | "both"
    sort_col = cfg.get("sort_by", "pct_patho_in_cluster")
    ascending = bool(cfg.get("ascending", False))
    txt_enabled = bool(cfg.get("txt_enabled", True))
    csv_enabled = bool(cfg.get("csv_enabled", True))
    txt_dp = int(cfg.get("txt_decimal_places", 1))
    csv_dp = int(cfg.get("csv_decimal_places", 3))

    # Résoudre les labels Sain / Patho depuis la config
    sain_labels = {
        s.lower() for s in cfg.get("sain_labels", [])
    } or _DEFAULT_SAIN_LABELS
    patho_labels = {
        s.lower() for s in cfg.get("patho_labels", [])
    } or _DEFAULT_PATHO_LABELS

    cond_lower = np.array([str(c).lower() for c in condition_labels])
    is_sain = np.isin(cond_lower, list(sain_labels))
    is_patho = np.isin(cond_lower, list(patho_labels))

    # Si ni sain ni patho reconnus → fallback : la condition la plus fréquente = sain
    if is_sain.sum() == 0 and is_patho.sum() == 0:
        unique_conds, counts = np.unique(cond_lower, return_counts=True)
        if len(unique_conds) >= 2:
            sorted_idx = np.argsort(counts)[::-1]
            is_sain = cond_lower == unique_conds[sorted_idx[0]]
            is_patho = cond_lower == unique_conds[sorted_idx[1]]
            _logger.warning(
                "Labels Sain/Patho non reconnus — fallback: '%s' = Sain, '%s' = Patho",
                unique_conds[sorted_idx[0]],
                unique_conds[sorted_idx[1]],
            )
        else:
            _logger.warning(
                "Une seule condition détectée ('%s') — export distribution ignoré",
                unique_conds[0] if len(unique_conds) > 0 else "?",
            )
            return {}

    n_sain_total = int(is_sain.sum())
    n_patho_total = int(is_patho.sum())
    n_total = len(clustering)

    suffix = f"_{timestamp}" if timestamp else ""
    produced: Dict[str, str] = {}

    # ── Construire les DataFrames des deux niveaux ────────────────────────────
    df_node = None
    df_meta = None

    if level in ("node", "both"):
        df_node = _build_node_distribution(
            clustering,
            metaclustering,
            is_sain,
            is_patho,
            n_sain_total,
            n_patho_total,
            n_total,
            sort_col,
            ascending,
            metacluster_names,
        )

    if level in ("metacluster", "both"):
        df_meta = _build_metacluster_distribution(
            metaclustering,
            is_sain,
            is_patho,
            n_sain_total,
            n_patho_total,
            n_total,
            sort_col,
            ascending,
            metacluster_names,
        )

    # ── Exports texte ─────────────────────────────────────────────────────────
    if txt_enabled:
        txt_path = output_dir / f"cluster_distribution{suffix}.txt"
        _write_txt_report(
            txt_path,
            df_node=df_node if level in ("node", "both") else None,
            df_meta=df_meta if level in ("metacluster", "both") else None,
            n_sain_total=n_sain_total,
            n_patho_total=n_patho_total,
            n_total=n_total,
            decimal_places=txt_dp,
            timestamp=timestamp,
        )
        produced["txt_cluster_distribution"] = str(txt_path)
        _logger.info("Distribution clusters (TXT) -> %s", txt_path.name)

    # ── Exports CSV ───────────────────────────────────────────────────────────
    if csv_enabled:
        if df_node is not None:
            csv_node_path = output_dir / f"cluster_distribution_nodes{suffix}.csv"
            _write_csv(df_node, csv_node_path, dp=csv_dp)
            produced["csv_cluster_distribution_nodes"] = str(csv_node_path)
            _logger.info("Distribution noeuds SOM (CSV) -> %s", csv_node_path.name)

        if df_meta is not None:
            csv_meta_path = (
                output_dir / f"cluster_distribution_metaclusters{suffix}.csv"
            )
            _write_csv(df_meta, csv_meta_path, dp=csv_dp)
            produced["csv_cluster_distribution_meta"] = str(csv_meta_path)
            _logger.info("Distribution metaclusters (CSV) -> %s", csv_meta_path.name)

    return produced


# =============================================================================
#  Constructeurs de tableaux
# =============================================================================


def _build_node_distribution(
    clustering: np.ndarray,
    metaclustering: np.ndarray,
    is_sain: np.ndarray,
    is_patho: np.ndarray,
    n_sain_total: int,
    n_patho_total: int,
    n_total: int,
    sort_col: str,
    ascending: bool,
    metacluster_names: Optional[Dict[int, str]],
) -> pd.DataFrame:
    """Construit la distribution par nœud SOM."""
    rows = []
    unique_nodes = np.unique(clustering)

    for node_id in unique_nodes:
        mask = clustering == node_id
        n_node = int(mask.sum())
        n_s = int((mask & is_sain).sum())
        n_p = int((mask & is_patho).sum())

        # Métacluster majoritaire du nœud
        mc_vals = metaclustering[mask]
        mc_mode = int(np.bincount(mc_vals).argmax()) if len(mc_vals) > 0 else -1
        mc_label = (metacluster_names or {}).get(mc_mode, str(mc_mode + 1))

        rows.append(
            _make_row(
                cluster_id=int(node_id),
                metacluster=mc_label,
                n_total=n_node,
                n_sain=n_s,
                n_patho=n_p,
                n_total_all=n_total,
                n_sain_total=n_sain_total,
                n_patho_total=n_patho_total,
            )
        )

    df = pd.DataFrame(rows)
    return _sort_df(df, sort_col, ascending)


def _build_metacluster_distribution(
    metaclustering: np.ndarray,
    is_sain: np.ndarray,
    is_patho: np.ndarray,
    n_sain_total: int,
    n_patho_total: int,
    n_total: int,
    sort_col: str,
    ascending: bool,
    metacluster_names: Optional[Dict[int, str]],
) -> pd.DataFrame:
    """Construit la distribution par métacluster."""
    rows = []
    unique_mc = np.unique(metaclustering)

    for mc_id in unique_mc:
        mask = metaclustering == mc_id
        n_node = int(mask.sum())
        n_s = int((mask & is_sain).sum())
        n_p = int((mask & is_patho).sum())

        mc_label = (metacluster_names or {}).get(int(mc_id), str(int(mc_id) + 1))

        rows.append(
            _make_row(
                cluster_id=int(mc_id),
                metacluster=mc_label,
                n_total=n_node,
                n_sain=n_s,
                n_patho=n_p,
                n_total_all=n_total,
                n_sain_total=n_sain_total,
                n_patho_total=n_patho_total,
            )
        )

    df = pd.DataFrame(rows)
    # Sur la vue métacluster la colonne "metacluster" == "cluster_id" → supprimer le doublon
    if "metacluster" in df.columns:
        df = df.drop(columns=["metacluster"])
    return _sort_df(df, sort_col, ascending)


def _make_row(
    cluster_id: int,
    metacluster: str,
    n_total: int,
    n_sain: int,
    n_patho: int,
    n_total_all: int,
    n_sain_total: int,
    n_patho_total: int,
) -> Dict[str, Any]:
    pct_total = round(n_total / max(n_total_all, 1) * 100, 6)
    pct_sain_in = round(n_sain / max(n_total, 1) * 100, 6)
    pct_sain_of = round(n_sain / max(n_sain_total, 1) * 100, 6)
    pct_patho_in = round(n_patho / max(n_total, 1) * 100, 6)
    pct_patho_of = round(n_patho / max(n_patho_total, 1) * 100, 6)
    return {
        "cluster_id": cluster_id,
        "metacluster": metacluster,
        "n_total": n_total,
        "pct_total": pct_total,
        "n_sain": n_sain,
        "pct_sain_in_cluster": pct_sain_in,
        "pct_sain_of_sain": pct_sain_of,
        "n_patho": n_patho,
        "pct_patho_in_cluster": pct_patho_in,
        "pct_patho_of_patho": pct_patho_of,
    }


def _sort_df(df: pd.DataFrame, sort_col: str, ascending: bool) -> pd.DataFrame:
    effective_col = sort_col if sort_col in df.columns else "pct_patho_in_cluster"
    if effective_col not in df.columns:
        effective_col = df.select_dtypes("number").columns[0]
    return df.sort_values(effective_col, ascending=ascending).reset_index(drop=True)


# =============================================================================
#  Renderers
# =============================================================================


def _write_csv(df: pd.DataFrame, path: Path, dp: int) -> None:
    """Arrondit les colonnes float au nombre de décimales voulu et sauve."""
    df_out = df.copy()
    for col in df_out.select_dtypes(include="float").columns:
        df_out[col] = df_out[col].round(dp)
    df_out.to_csv(path, index=False, encoding="utf-8-sig")


def _write_txt_report(
    path: Path,
    df_node: Optional[pd.DataFrame],
    df_meta: Optional[pd.DataFrame],
    n_sain_total: int,
    n_patho_total: int,
    n_total: int,
    decimal_places: int,
    timestamp: str,
) -> None:
    """Écrit le rapport texte ASCII lisible dans *path*."""
    lines: List[str] = []

    lines.append("=" * 100)
    lines.append("  DISTRIBUTION CELLULAIRE PAR CLUSTER SOM — FlowSOM Pipeline Pro")
    if timestamp:
        lines.append(f"  Généré le : {timestamp}")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"  Total cellules  : {n_total:>12,}")
    lines.append(
        f"  Sain            : {n_sain_total:>12,}  ({n_sain_total / max(n_total, 1) * 100:.{decimal_places}f}%)"
    )
    lines.append(
        f"  Pathologique    : {n_patho_total:>12,}  ({n_patho_total / max(n_total, 1) * 100:.{decimal_places}f}%)"
    )
    lines.append("")

    if df_meta is not None:
        lines.append("─" * 100)
        lines.append("  NIVEAU METACLUSTER")
        lines.append("  Trié par % Patho dans cluster (décroissant)")
        lines.append("─" * 100)
        lines += _df_to_ascii(df_meta, decimal_places)
        lines.append("")

    if df_node is not None:
        lines.append("─" * 100)
        lines.append("  NIVEAU NOEUD SOM")
        lines.append("  Trié par % Patho dans cluster (décroissant)")
        lines.append("─" * 100)
        lines += _df_to_ascii(df_node, decimal_places)
        lines.append("")

    lines.append("=" * 100)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _df_to_ascii(df: pd.DataFrame, dp: int) -> List[str]:
    """Formate un DataFrame en tableau ASCII avec alignement des colonnes."""
    # Construire les chaînes de cellules
    float_cols = set(df.select_dtypes(include="float").columns)

    def fmt(val: Any, col: str) -> str:
        if col in float_cols:
            return f"{val:.{dp}f}"
        if isinstance(val, (int, np.integer)):
            return f"{val:,}"
        return str(val)

    header = list(df.columns)
    rows_str = [[fmt(row[col], col) for col in header] for _, row in df.iterrows()]

    # Calculer la largeur de chaque colonne
    col_widths = [
        max(len(h), max((len(r[i]) for r in rows_str), default=0))
        for i, h in enumerate(header)
    ]

    sep = "  ".join("-" * w for w in col_widths)
    hdr = "  ".join(h.ljust(w) for h, w in zip(header, col_widths))

    lines = [f"  {hdr}", f"  {sep}"]
    for row in rows_str:
        line = "  ".join(
            (cell.rjust(w) if i > 0 else cell.ljust(w))
            for i, (cell, w) in enumerate(zip(row, col_widths))
        )
        lines.append(f"  {line}")
    return lines
