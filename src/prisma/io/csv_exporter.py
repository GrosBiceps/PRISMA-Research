"""
csv_exporter.py — Export des résultats d'analyse en CSV.

Fournit des fonctions pour exporter les données cellulaires, les statistiques
de clusters et les matrices MFI au format CSV compatible avec les logiciels
de bureautique (Excel) et de statistiques (R, Python downstream).
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from prisma.utils.logger import get_logger

_logger = get_logger("io.csv_exporter")


def export_cells_csv(
    df: pd.DataFrame,
    output_path: Path | str,
    *,
    include_all_columns: bool = True,
    columns: Optional[List[str]] = None,
) -> bool:
    """
    Exporte le DataFrame cellulaire complet en CSV.

    Args:
        df: DataFrame avec les données cellulaires.
        output_path: Chemin du fichier CSV de sortie.
        include_all_columns: Si True, toutes les colonnes sont exportées.
        columns: Si spécifié, seules ces colonnes sont exportées.

    Returns:
        True si succès, False si échec.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        export_df = df[columns] if columns and not include_all_columns else df
        export_df.to_csv(
            output_path, index=False, encoding="utf-8-sig"
        )  # utf-8-sig pour Excel
        _logger.info(
            "CSV export: %d cellules, %d colonnes → %s",
            len(export_df),
            export_df.shape[1],
            output_path.name,
        )
        return True
    except Exception as exc:
        _logger.error("Échec export CSV %s: %s", output_path.name, exc)
        return False


def export_statistics_csv(
    stats_df: pd.DataFrame,
    output_path: Path | str,
) -> bool:
    """
    Exporte le DataFrame de statistiques par cluster en CSV.

    Args:
        stats_df: DataFrame avec les statistiques par métacluster.
        output_path: Chemin du fichier CSV de sortie.

    Returns:
        True si succès, False si échec.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        stats_df.to_csv(output_path, index=True, encoding="utf-8-sig")
        _logger.info(
            "Statistiques export: %d lignes → %s",
            len(stats_df),
            output_path.name,
        )
        return True
    except Exception as exc:
        _logger.error("Échec export statistiques %s: %s", output_path.name, exc)
        return False


def export_mfi_matrix_csv(
    mfi_matrix: pd.DataFrame,
    output_path: Path | str,
) -> bool:
    """
    Exporte la matrice MFI (Median Fluorescence Intensity) par métacluster.

    Args:
        mfi_matrix: DataFrame [n_metaclusters × n_markers] avec les MFI médianes.
        output_path: Chemin du fichier CSV de sortie.

    Returns:
        True si succès, False si échec.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        mfi_matrix.to_csv(output_path, index=True, encoding="utf-8-sig")
        _logger.info(
            "Matrice MFI export: %d clusters × %d marqueurs → %s",
            mfi_matrix.shape[0],
            mfi_matrix.shape[1],
            output_path.name,
        )
        return True
    except Exception as exc:
        _logger.error("Échec export MFI %s: %s", output_path.name, exc)
        return False


def export_per_file_csv(
    df: pd.DataFrame,
    output_dir: Path | str,
    file_origin_column: str = "file_origin",
    timestamp: str = "",
) -> Dict[str, bool]:
    """
    Exporte un CSV séparé pour chaque fichier FCS source.

    Args:
        df: DataFrame complet avec la colonne d'origine.
        output_dir: Dossier de sortie pour les CSV individuels.
        file_origin_column: Nom de la colonne identifiant le fichier source.
        timestamp: Suffixe horodatage pour les noms de fichiers.

    Returns:
        Dict {nom_fichier: succès}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, bool] = {}

    if file_origin_column not in df.columns:
        _logger.warning(
            "Colonne '%s' absente — export par fichier ignoré", file_origin_column
        )
        return results

    for file_name in df[file_origin_column].unique():
        mask = df[file_origin_column] == file_name
        df_file = df[mask].copy()

        safe_name = (
            str(file_name).replace(" ", "_").replace(".fcs", "").replace(".FCS", "")
        )
        suffix = f"_{timestamp}" if timestamp else ""
        csv_path = output_dir / f"{safe_name}{suffix}.csv"

        ok = export_cells_csv(df_file, csv_path)
        results[file_name] = ok

    return results


def compute_cluster_statistics(
    df: pd.DataFrame,
    marker_columns: List[str],
    cluster_column: str = "FlowSOM_metacluster",
) -> pd.DataFrame:
    """
    Calcule des statistiques descriptives par métacluster.

    Calcule: n_cells, pct_total, median par marqueur.

    Args:
        df: DataFrame cellulaire.
        marker_columns: Colonnes à inclure dans les statistiques.
        cluster_column: Colonne d'assignation de cluster.

    Returns:
        DataFrame de statistiques [n_clusters × (2 + n_markers)].
    """
    if cluster_column not in df.columns:
        _logger.error("Colonne '%s' absente du DataFrame", cluster_column)
        return pd.DataFrame()

    n_total = len(df)
    stats_rows = []

    for cluster_id in sorted(df[cluster_column].unique()):
        mask = df[cluster_column] == cluster_id
        df_cluster = df.loc[mask, marker_columns]

        row: Dict = {
            cluster_column: cluster_id,
            "n_cells": int(mask.sum()),
            "pct_total": round(float(mask.sum() / n_total * 100), 3),
        }

        for col in marker_columns:
            if col in df_cluster.columns:
                row[f"median_{col}"] = float(df_cluster[col].median())

        stats_rows.append(row)

    return pd.DataFrame(stats_rows).set_index(cluster_column)


# ─────────────────────────────────────────────────────────────────────────────
#  Extraction de timepoint depuis les noms de fichiers FCS
# ─────────────────────────────────────────────────────────────────────────────

# Regex triés par priorité (les formats les plus explicites d'abord)
_DATE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(\d{4})[._-](\d{2})[._-](\d{2})"), "%Y-%m-%d"),  # 2024-03-15
    (re.compile(r"(\d{2})[._-](\d{2})[._-](\d{4})"), "%d-%m-%Y"),  # 15-03-2024
    (re.compile(r"(\d{2})[._-](\d{2})[._-](\d{2})"), "%d-%m-%y"),  # 15-03-24
    (re.compile(r"(\d{8})"), "%Y%m%d"),  # 20240315
]


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    """
    Extrait une date depuis un nom de fichier FCS.

    Ordre de priorité :
      1. Date dans le nom du fichier (YYYY-MM-DD, DD-MM-YYYY, DD-MM-YY, YYYYMMDD).
      2. Champ $DATE dans les métadonnées internes du fichier FCS.
      3. Date de modification du fichier (st_mtime).

    Args:
        filename: Chemin complet ou nom de fichier FCS.

    Returns:
        datetime si une date valide est trouvée, None sinon.
    """
    # ── 1. Date dans le nom du fichier ────────────────────────────────────────
    stem = Path(filename).stem
    for pattern, fmt in _DATE_PATTERNS:
        match = pattern.search(stem)
        if match:
            date_str = (
                "-".join(match.groups()) if len(match.groups()) > 1 else match.group(1)
            )
            try:
                return datetime.strptime(
                    date_str, fmt.replace(".", "-").replace("_", "-")
                )
            except ValueError:
                continue

    # ── 2. Champ $DATE dans les métadonnées FCS internes ─────────────────────
    fpath = Path(filename)
    if fpath.exists() and fpath.suffix.lower() == ".fcs":
        try:

            with open(fpath, "rb") as _f:
                # En-tête FCS : bytes 10-17 = offset début TEXT, 18-25 = offset fin TEXT
                _f.seek(10)
                _header = _f.read(16).decode("ascii", errors="replace").strip()
                _text_start = int(_header[:8])
                _text_end = int(_header[8:])
                _f.seek(_text_start)
                _text = _f.read(_text_end - _text_start + 1).decode(
                    "latin-1", errors="replace"
                )
            # Le délimiteur est le premier caractère du segment TEXT
            _delim = _text[0]
            _pairs = _text.split(_delim)
            _meta = {}
            for i in range(1, len(_pairs) - 1, 2):
                _meta[_pairs[i].strip().upper()] = _pairs[i + 1].strip()
            _fcs_date = _meta.get("$DATE", "")
            if _fcs_date:
                # Formats FCS courants : DD-Mon-YYYY, DD-MMM-YYYY, YYYY-MM-DD
                for _fmt in ("%d-%b-%Y", "%d-%b-%y", "%Y-%m-%d", "%d/%m/%Y"):
                    try:
                        return datetime.strptime(_fcs_date, _fmt)
                    except ValueError:
                        continue
        except Exception:
            pass

    # ── 3. Date de modification du fichier ───────────────────────────────────
    if fpath.exists():
        try:
            import os
            _mtime = os.path.getmtime(fpath)
            return datetime.fromtimestamp(_mtime)
        except Exception:
            pass

    return None


def add_timepoint_columns(
    df: pd.DataFrame,
    file_origin_column: str = "file_origin",
) -> pd.DataFrame:
    """
    Ajoute les colonnes Timepoint (date ISO) et Timepoint_Num (rang chronologique)
    au DataFrame en extrayant les dates depuis les noms de fichiers.

    Args:
        df: DataFrame cellulaire avec une colonne d'origine fichier.
        file_origin_column: Nom de la colonne contenant le nom du fichier source.

    Returns:
        DataFrame enrichi avec Timepoint et Timepoint_Num.
    """
    if file_origin_column not in df.columns:
        _logger.warning(
            "Colonne '%s' absente — impossible d'extraire les timepoints",
            file_origin_column,
        )
        return df

    result = df.copy()

    # Extraire les dates par fichier unique
    unique_files = result[file_origin_column].unique()
    file_dates: Dict[str, Optional[datetime]] = {}
    for fname in unique_files:
        file_dates[fname] = extract_date_from_filename(str(fname))

    # Colonne Timepoint (date ISO string)
    result["Timepoint"] = result[file_origin_column].map(
        lambda f: (
            file_dates.get(f).strftime("%Y-%m-%d") if file_dates.get(f) else "Unknown"
        )
    )

    # Colonne Timepoint_Num (rang chronologique, 1-indexé)
    dated = {f: d for f, d in file_dates.items() if d is not None}
    sorted_files = sorted(dated.keys(), key=lambda f: dated[f])
    rank_map = {f: i + 1 for i, f in enumerate(sorted_files)}
    result["Timepoint_Num"] = result[file_origin_column].map(
        lambda f: rank_map.get(f, 0)
    )

    n_found = sum(1 for d in file_dates.values() if d is not None)
    _logger.info(
        "Timepoints extraits: %d/%d fichiers avec date détectée",
        n_found,
        len(unique_files),
    )

    return result
