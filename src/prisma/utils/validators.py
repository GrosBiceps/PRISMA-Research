"""
validators.py — Validation des données cytométriques en entrée de pipeline.

Règles de validation basées sur les bonnes pratiques ELN 2022:
- Présence des marqueurs obligatoires
- Détection de données non transformées (valeurs négatives abondantes)
- Détection de manque de compensation ($SPILL)
- Contrôle du nombre minimal de cellules
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import anndata as ad

    _ANNDATA_AVAILABLE = True
except ImportError:
    _ANNDATA_AVAILABLE = False


def check_nan(X: np.ndarray) -> int:
    """
    Vérifie la présence de NaN dans la matrice de données.

    Les NaN dans AnnData.X font planter FlowSOM silencieusement.

    Args:
        X: Matrice (n_cells, n_markers).

    Returns:
        Nombre de NaN détectés.
    """
    n_nan = int(np.isnan(X).sum())
    if n_nan > 0:
        warnings.warn(
            f"VALIDATION: {n_nan} valeurs NaN détectées dans la matrice. "
            "FlowSOM plantera si elles ne sont pas imputées ou supprimées.",
            stacklevel=2,
        )
    return n_nan


def check_min_cells(n_cells: int, min_cells: int = 1000) -> bool:
    """
    Vérifie que le nombre de cellules est suffisant pour l'analyse.

    Args:
        n_cells: Nombre de cellules dans l'échantillon.
        min_cells: Seuil minimum (défaut 1000).

    Returns:
        True si suffisant, False sinon.
    """
    if n_cells < min_cells:
        warnings.warn(
            f"VALIDATION: Seulement {n_cells} cellules (minimum recommandé: {min_cells}). "
            "Les résultats FlowSOM pourraient être instables.",
            stacklevel=2,
        )
        return False
    return True


def check_markers_present(
    available_markers: List[str],
    required_patterns: List[str],
    sample_name: str = "",
) -> Tuple[List[str], List[str]]:
    """
    Vérifie que les marqueurs requis sont présents (par correspondance partielle).

    Args:
        available_markers: Liste des marqueurs disponibles dans le fichier FCS.
        required_patterns: Patterns à rechercher (ex: ["CD45", "CD34", "CD38"]).
        sample_name: Nom de l'échantillon pour les messages d'erreur.

    Returns:
        Tuple (found_markers, missing_patterns).
    """
    found: List[str] = []
    missing: List[str] = []

    for pattern in required_patterns:
        pattern_upper = pattern.upper()
        matching = [m for m in available_markers if pattern_upper in m.upper()]
        if matching:
            found.extend(matching)
        else:
            missing.append(pattern)

    if missing:
        prefix = f"[{sample_name}] " if sample_name else ""
        warnings.warn(
            f"{prefix}VALIDATION: Marqueurs non trouvés: {missing}. "
            "Certaines gates pourraient être ignorées.",
            stacklevel=2,
        )

    return found, missing


def check_transformation_needed(X: np.ndarray, var_names: List[str]) -> bool:
    """
    Détecte si les données semblent non transformées.

    Heuristique: si plus de 5% des valeurs de canaux de fluorescence
    sont très négatives (<-500), les données sont probablement non transformées.

    FSC/SSC/Time sont exclus de ce contrôle.

    Args:
        X: Matrice de données (n_cells, n_markers).
        var_names: Noms des marqueurs (colonnes).

    Returns:
        True si une transformation semble nécessaire, False sinon.
    """
    skip_patterns = {"FSC", "SSC", "TIME", "WIDTH", "AREA", "HEIGHT"}
    fluoro_indices = [
        i
        for i, name in enumerate(var_names)
        if not any(skip in name.upper() for skip in skip_patterns)
    ]

    if not fluoro_indices:
        return False

    X_fluoro = X[:, fluoro_indices]
    pct_negative = float(np.mean(X_fluoro < -500) * 100)

    if pct_negative > 5.0:
        warnings.warn(
            f"VALIDATION: {pct_negative:.1f}% de valeurs < -500 sur les canaux de fluorescence. "
            "Les données semblent non transformées. Appliquer logicle/arcsinh avant FlowSOM.",
            stacklevel=2,
        )
        return True
    return False


def check_compensation(fcs_metadata: Dict[str, Any]) -> bool:
    """
    Vérifie si une matrice de compensation est présente dans les métadonnées FCS.

    Args:
        fcs_metadata: Dictionnaire des métadonnées FCS (header TEXT segment).

    Returns:
        True si $SPILL ou $SPILLOVER présent, False sinon.
    """
    has_spill = any(k.upper() in {"$SPILL", "$SPILLOVER"} for k in fcs_metadata)
    if not has_spill:
        warnings.warn(
            "VALIDATION: Aucune matrice de compensation ($SPILL) trouvée dans le FCS. "
            "Vérifier que la compensation a déjà été appliquée ou appliquer via flowkit.",
            stacklevel=2,
        )
    return has_spill


def check_no_fsc_ssc_in_analysis_markers(marker_names: List[str]) -> bool:
    """
    Vérifie qu'aucun marqueur FSC/SSC/Time n'est inclus dans l'analyse FlowSOM.

    Ces paramètres ne doivent jamais entrer dans la matrice FlowSOM — ils
    biaiseraient le clustering vers la morphologie cellulaire plutôt que
    vers l'immunophénotype.

    Args:
        marker_names: Liste des marqueurs à utiliser pour FlowSOM.

    Returns:
        True si la liste est propre, False si des FSC/SSC/Time sont présents.
    """
    scatter_patterns = {"FSC", "SSC", "TIME", "WIDTH", "AREA", "HEIGHT"}
    contaminants = [
        m for m in marker_names if any(p in m.upper() for p in scatter_patterns)
    ]

    if contaminants:
        warnings.warn(
            f"VALIDATION: Les paramètres suivants ne doivent PAS être utilisés pour FlowSOM: "
            f"{contaminants}. Retirer FSC/SSC/Time de la liste des marqueurs.",
            stacklevel=2,
        )
        return False
    return True


def check_cell_balance(
    n_condition1: int,
    n_condition2: int,
    condition1_name: str = "condition1",
    condition2_name: str = "condition2",
    ratio_threshold: float = 10.0,
) -> bool:
    """
    Vérifie que deux conditions n'ont pas une différence de taille excessive.

    Un déséquilibre > 10× biaiserait le FlowSOM vers la condition plus grande.

    Args:
        n_condition1: Nombre de cellules dans la condition 1.
        n_condition2: Nombre de cellules dans la condition 2.
        condition1_name: Nom de la condition 1.
        condition2_name: Nom de la condition 2.
        ratio_threshold: Ratio max acceptable avant avertissement.

    Returns:
        True si équilibre acceptable, False si déséquilibre detecté.
    """
    if n_condition1 == 0 or n_condition2 == 0:
        return True

    ratio = max(n_condition1, n_condition2) / min(n_condition1, n_condition2)
    if ratio > ratio_threshold:
        larger = condition1_name if n_condition1 > n_condition2 else condition2_name
        warnings.warn(
            f"VALIDATION: Déséquilibre de cellules {ratio:.1f}× entre "
            f"{condition1_name} ({n_condition1}) et {condition2_name} ({n_condition2}). "
            f"La condition '{larger}' domine — sous-échantillonner avant FlowSOM.",
            stacklevel=2,
        )
        return False
    return True


def validate_anndata_for_flowsom(
    adata: "ad.AnnData",
    required_marker_patterns: Optional[List[str]] = None,
    min_cells: int = 1000,
) -> Dict[str, Any]:
    """
    Validation complète d'un AnnData avant analyse FlowSOM.

    Effectue tous les contrôles qualité en une seule passe.

    Args:
        adata: Objet AnnData à valider.
        required_marker_patterns: Patterns de marqueurs à vérifier.
        min_cells: Nombre minimum de cellules.

    Returns:
        Dict avec les résultats de chaque contrôle.
    """
    if not _ANNDATA_AVAILABLE:
        return {"error": "anndata non disponible"}

    results: Dict[str, Any] = {
        "n_cells": adata.n_obs,
        "n_markers": adata.n_vars,
        "checks": {},
    }

    # 1. Vérification NaN
    n_nan = check_nan(adata.X)
    results["checks"]["nan"] = {"n_nan": n_nan, "ok": n_nan == 0}

    # 2. Nombre minimal de cellules
    ok_cells = check_min_cells(adata.n_obs, min_cells=min_cells)
    results["checks"]["min_cells"] = {"n_cells": adata.n_obs, "ok": ok_cells}

    # 3. FSC/SSC dans les marqueurs
    marker_names = list(adata.var_names)
    ok_scatter = check_no_fsc_ssc_in_analysis_markers(marker_names)
    results["checks"]["no_scatter"] = {"ok": ok_scatter}

    # 4. Transformation nécessaire
    needs_transform = check_transformation_needed(adata.X, marker_names)
    results["checks"]["transformation"] = {"needs_transform": needs_transform}

    # 5. Marqueurs requis
    if required_marker_patterns:
        found, missing = check_markers_present(marker_names, required_marker_patterns)
        results["checks"]["required_markers"] = {
            "ok": len(missing) == 0,
            "found": found,
            "missing": missing,
        }

    return results
