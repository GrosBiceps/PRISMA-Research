"""
json_exporter.py — Export des métadonnées d'analyse en JSON.

Fournit une traçabilité complète de l'analyse: paramètres, données,
résultats de clustering. Format structuré pour archivage clinique.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from prisma.utils.logger import get_logger

_logger = get_logger("io.json_exporter")


class _NumpyEncoder(json.JSONEncoder):
    """Encodeur JSON gérant les types numpy (int64, float32, ndarray)."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def export_analysis_metadata(
    metadata: Dict[str, Any],
    output_path: Path | str,
) -> bool:
    """
    Exporte le dictionnaire de métadonnées de l'analyse en JSON.

    Args:
        metadata: Dictionnaire structuré avec toutes les métadonnées.
        output_path: Chemin du fichier JSON de sortie.

    Returns:
        True si succès, False si échec.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
        _logger.info("Métadonnées JSON exportées: %s", output_path)
        return True
    except Exception as exc:
        _logger.error("Échec export JSON %s: %s", output_path.name, exc)
        return False


def build_analysis_metadata(
    *,
    pipeline_version: str = "FlowSOM_Pipeline_Pro v1.0",
    input_files: List[str],
    config_dict: Dict[str, Any],
    n_cells: int,
    marker_names: List[str],
    used_markers: List[str],
    metaclustering: Optional[np.ndarray] = None,
    n_metaclusters: int = 0,
    cell_data_obs: Optional[pd.DataFrame] = None,
    export_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Construit le dictionnaire complet de métadonnées pour une analyse.

    Args:
        pipeline_version: Identifiant de version du pipeline.
        input_files: Chemins des fichiers FCS chargés.
        config_dict: Configuration utilisée (depuis PipelineConfig.to_dict()).
        n_cells: Nombre total de cellules analysées.
        marker_names: Tous les marqueurs disponibles.
        used_markers: Marqueurs utilisés pour le clustering.
        metaclustering: Assignation par cellule (optionnel).
        n_metaclusters: Nombre de métaclusters.
        cell_data_obs: obs DataFrame d'AnnData (pour résumé par condition/fichier).
        export_paths: Chemins des fichiers exportés.

    Returns:
        Dictionnaire imbriqué structuré.
    """
    timestamp = datetime.now().isoformat()

    metadata: Dict[str, Any] = {
        "analysis_info": {
            "date": timestamp,
            "pipeline_version": pipeline_version,
        },
        "input_files": {
            "total_files": len(input_files),
            "files": input_files,
        },
        "configuration": config_dict,
        "data_summary": {
            "total_cells": n_cells,
            "all_markers": marker_names,
            "markers_used_for_clustering": used_markers,
            "n_markers_used": len(used_markers),
        },
        "export_files": export_paths or {},
    }

    # Résumé par condition et par fichier (si obs disponible)
    if cell_data_obs is not None:
        if "condition" in cell_data_obs.columns:
            metadata["data_summary"]["cells_per_condition"] = {
                str(cond): int((cell_data_obs["condition"] == cond).sum())
                for cond in cell_data_obs["condition"].unique()
            }
        if "file_origin" in cell_data_obs.columns:
            metadata["data_summary"]["cells_per_file"] = {
                str(fname): int((cell_data_obs["file_origin"] == fname).sum())
                for fname in cell_data_obs["file_origin"].unique()
            }

    # Résumé par métacluster
    if metaclustering is not None and n_metaclusters > 0:
        counts = np.bincount(metaclustering.astype(int), minlength=n_metaclusters)
        total = len(metaclustering)
        mc_summary: Dict[str, Dict[str, Any]] = {
            f"MC{i}": {
                "n_cells": int(counts[i]),
                "pct_total": round(int(counts[i]) / total * 100, 3) if total > 0 else 0.0,
            }
            for i in range(n_metaclusters)
        }
        metadata["metacluster_summary"] = mc_summary

    return metadata


def export_gating_log(
    events: List[Dict[str, Any]],
    output_path: Path | str,
) -> bool:
    """
    Exporte un log de gating structuré en JSON.

    Args:
        events: Liste de dicts d'événements (depuis GatingLogger.to_dict()).
        output_path: Chemin du fichier JSON de sortie.

    Returns:
        True si succès, False si échec.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "version": "1.0",
        "date": datetime.now().isoformat(),
        "n_events": len(events),
        "events": events,
    }

    try:
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
        _logger.info("Log de gating exporté: %s (%d events)", output_path, len(events))
        return True
    except Exception as exc:
        _logger.error("Échec export gating log %s: %s", output_path.name, exc)
        return False
