# -*- coding: utf-8 -*-
"""
mrd_adapter.py — Adaptateur MRDResult → structures UI-friendly.

Traduit le MRDResult (domaine scientifique) en dicts simples
consommables par les widgets UI sans couplage direct au domaine.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def adapt_mrd_result(result: Any, method_used: str = "all") -> Dict[str, Any]:
    """
    Extrait les données MRD d'un PipelineResult pour l'UI.

    Args:
        result: PipelineResult (post-pipeline).
        method_used: méthode choisie par l'utilisateur ("all", "jf", "flo", "eln").

    Returns:
        dict avec les clés :
          - patient_info: dict (nom, date, fichier)
          - gauges: list de dict par méthode affichée
          - nodes: list de dict (tous les nœuds MRD positifs)
          - method_used: str
    """
    if result is None or not result.success:
        return _empty_state()

    mrd: Any = getattr(result, "mrd_result", None)
    if mrd is None:
        return _empty_state()

    # ── Infos patient ────────────────────────────────────────────────
    patient_info = {
        "stem": getattr(result, "patho_stem", "") or "",
        "date": getattr(result, "patho_date", "") or "",
        "timestamp": getattr(result, "timestamp", "") or "",
        "n_cells": result.n_cells,
        "n_cells_patho": (
            getattr(mrd, "n_patho_pre_cd45", 0) or getattr(mrd, "total_cells_patho", 0)
        ),
    }

    # ── Gauges par méthode ───────────────────────────────────────────
    gauges: List[Dict[str, Any]] = []

    # ELN n'est jamais affiché dans l'onglet Accueil (trop de faux positifs dans
    # l'usage courant). ELN reste calculé par le pipeline et exporté dans les
    # rapports, mais la gauge UI est réservée à JF et Flo.
    _show_jf = method_used in ("all", "jf")
    _show_flo = method_used in ("all", "flo")
    _show_eln = method_used == "eln"  # uniquement si l'utilisateur choisit explicitement "eln" seul

    if _show_jf:
        gauges.append({
            "method": "JF",
            "pct": round(getattr(mrd, "mrd_pct_jf", 0.0), 4),
            "n_cells": getattr(mrd, "mrd_cells_jf", 0),
            "n_nodes": getattr(mrd, "n_nodes_mrd_jf", 0),
            "positive": getattr(mrd, "mrd_pct_jf", 0.0) > 0,
            "positivity_threshold": None,
        })

    if _show_flo:
        gauges.append({
            "method": "Flo",
            "pct": round(getattr(mrd, "mrd_pct_flo", 0.0), 4),
            "n_cells": getattr(mrd, "mrd_cells_flo", 0),
            "n_nodes": getattr(mrd, "n_nodes_mrd_flo", 0),
            "positive": getattr(mrd, "mrd_pct_flo", 0.0) > 0,
            "positivity_threshold": None,
        })

    if _show_eln:
        cfg = getattr(mrd, "config_snapshot", {})
        eln_cfg = cfg.get("eln_standards", {}) if isinstance(cfg, dict) else {}
        threshold = eln_cfg.get("clinical_positivity_pct", 0.1)
        gauges.append({
            "method": "ELN 2025",
            "pct": round(getattr(mrd, "mrd_pct_eln", 0.0), 4),
            "n_cells": getattr(mrd, "mrd_cells_eln", 0),
            "n_nodes": getattr(mrd, "n_nodes_mrd_eln", 0),
            "positive": getattr(mrd, "eln_positive", False),
            "low_level": getattr(mrd, "eln_low_level", False),
            "positivity_threshold": threshold,
        })

    # ── Nœuds MRD positifs (pour le tableau) ────────────────────────
    nodes: List[Dict[str, Any]] = []
    for node in getattr(mrd, "per_node", []):
        is_any = node.is_mrd_jf or node.is_mrd_flo or node.is_mrd_eln
        if not is_any:
            continue
        nodes.append({
            "node_id": node.cluster_id,
            "n_cells": node.n_cells_total,
            "n_sain": node.n_cells_sain,
            "n_patho": node.n_cells_patho,
            "pct_sain": round(node.pct_sain, 2),
            "pct_patho": round(node.pct_patho, 2),
            "pct_sain_global": round(node.pct_sain_global, 4),
            "is_mrd_jf": node.is_mrd_jf,
            "is_mrd_flo": node.is_mrd_flo,
            "is_mrd_eln": node.is_mrd_eln,
        })

    nodes.sort(key=lambda n: n["pct_patho"], reverse=True)

    return {
        "patient_info": patient_info,
        "gauges": gauges,
        "nodes": nodes,
        "method_used": method_used,
        "has_data": True,
    }


def adapt_all_nodes(mrd: Any) -> List[Dict[str, Any]]:
    """
    Retourne la liste de TOUS les nœuds SOM patients, y compris ceux non
    retenus par JF / Flo / ELN, pour alimenter ExpertFocusDialog.

    Accepte directement un objet MRDResult (self._raw_mrd_result dans HomeTab),
    contrairement à adapt_mrd_result() qui reçoit un PipelineResult.

    Chaque dict contient les mêmes clés que adapt_mrd_result() avec les flags
    is_mrd_jf / is_mrd_flo / is_mrd_eln à False pour les nœuds non sélectionnés.
    """
    if mrd is None:
        return []

    all_nodes: List[Dict[str, Any]] = []
    for node in getattr(mrd, "per_node", []):
        all_nodes.append({
            "node_id":          node.cluster_id,
            "n_cells":          node.n_cells_total,
            "n_sain":           node.n_cells_sain,
            "n_patho":          node.n_cells_patho,
            "pct_sain":         round(node.pct_sain, 2),
            "pct_patho":        round(node.pct_patho, 2),
            "pct_sain_global":  round(node.pct_sain_global, 4),
            "is_mrd_jf":        node.is_mrd_jf,
            "is_mrd_flo":       node.is_mrd_flo,
            "is_mrd_eln":       node.is_mrd_eln,
        })

    all_nodes.sort(key=lambda n: n["pct_patho"], reverse=True)
    return all_nodes


def _empty_state() -> Dict[str, Any]:
    return {
        "patient_info": {},
        "gauges": [],
        "nodes": [],
        "method_used": "all",
        "has_data": False,
    }
