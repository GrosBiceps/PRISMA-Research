"""
fcs_writer.py — Export de données vers le format FCS (compatible Kaluza/FlowJo).

Utilise fcswrite pour créer des fichiers FCS avec les colonnes de clustering
ajoutées par FlowSOM (métacluster, cluster, coordonnées grille).

Note: Les données exportées sont les valeurs BRUTES (pré-transformation) pour
compatibilité avec les logiciels de cytométrie clinique (Kaluza, FlowJo).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from prisma.utils.logger import get_logger

_logger = get_logger("io.fcs_writer")

try:
    import fcswrite as _fcswrite

    _FCSWRITE_AVAILABLE = True
except ImportError:
    _FCSWRITE_AVAILABLE = False
    _logger.warning(
        "fcswrite non disponible — export FCS désactivé: pip install fcswrite"
    )


def export_to_fcs(
    df: pd.DataFrame,
    output_path: Path | str,
    *,
    compat_chn_names: bool = True,
) -> bool:
    """
    Exporte un DataFrame en fichier FCS compatible Kaluza/FlowJo.

    Seules les colonnes numériques sont exportées. Les NaN/Inf sont
    remplacés par 0 et 1e6 respectivement pour respecter le format FCS.

    Args:
        df: DataFrame avec les données cellulaires + colonnes de clustering.
        output_path: Chemin du fichier .fcs de sortie.
        compat_chn_names: Nettoyer les noms de canaux pour compatibilité.

    Returns:
        True si succès, False si échec.
    """
    if not _FCSWRITE_AVAILABLE:
        _logger.error("fcswrite requis pour export FCS: pip install fcswrite")
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Sélectionner uniquement les colonnes numériques
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            _logger.error("Aucune colonne numérique dans le DataFrame — export annulé")
            return False

        data = numeric_df.values.astype(np.float32)
        channels = list(numeric_df.columns)

        # Garantir que toutes les valeurs sont finies (FCS ne supporte pas NaN/Inf)
        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=0.0)

        _logger.info(
            "Export FCS: %d events, %d canaux → %s",
            data.shape[0],
            data.shape[1],
            output_path.name,
        )

        _fcswrite.write_fcs(
            str(output_path),
            channels,
            data,
            compat_chn_names=compat_chn_names,
        )
        _logger.info("FCS exporté avec succès: %s", output_path)
        return True

    except Exception as exc:
        _logger.error("Échec export FCS %s: %s", output_path.name, exc)
        return False


def export_to_fcs_kaluza(
    df: pd.DataFrame,
    output_path: Path | str,
) -> bool:
    """
    Export FCS optimisé pour Kaluza avec toutes les coordonnées positives.

    Alias de export_to_fcs() avec paramètres adaptés à Kaluza.

    Args:
        df: DataFrame avec les données cellulaires.
        output_path: Chemin du fichier .fcs de sortie.

    Returns:
        True si succès, False si échec.
    """
    return export_to_fcs(df, output_path, compat_chn_names=True)


def add_clustering_columns(
    df: pd.DataFrame,
    metaclustering: np.ndarray,
    clustering: Optional[np.ndarray] = None,
    x_grid: Optional[np.ndarray] = None,
    y_grid: Optional[np.ndarray] = None,
    grid_coords: Optional[np.ndarray] = None,
    node_sizes: Optional[np.ndarray] = None,
    use_circular_jitter: bool = True,
    max_node_radius: float = 0.45,
    min_node_radius: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Ajoute les colonnes de clustering FlowSOM à un DataFrame cellulaire.

    Si grid_coords et clustering sont fournis, calcule les coordonnées
    xGrid/yGrid avec jitter circulaire style FlowSOM R : le rayon du
    nuage de points dépend de sqrt(node_size / max_size).

    Args:
        df: DataFrame original (n_cells, n_markers).
        metaclustering: Assignation de métacluster (n_cells,).
        clustering: Assignation de cluster SOM node (n_cells,), optionnel.
        x_grid: Coordonnée X grille SOM pré-calculée (n_cells,), optionnel.
        y_grid: Coordonnée Y grille SOM pré-calculée (n_cells,), optionnel.
        grid_coords: Coordonnées grille SOM (n_nodes, 2), optionnel.
        node_sizes: Taille par nœud SOM (n_nodes,), optionnel.
        use_circular_jitter: Utiliser le jitter circulaire FlowSOM R.
        max_node_radius: Rayon max du jitter (proportionnel aux gros nœuds).
        min_node_radius: Rayon min du jitter (pour les petits nœuds).
        seed: Graine aléatoire pour la reproductibilité du jitter.

    Returns:
        DataFrame enrichi avec les colonnes de clustering.
    """
    result = df.copy()
    result["FlowSOM_metacluster"] = metaclustering.astype(np.float32)

    if clustering is not None:
        result["FlowSOM_cluster"] = clustering.astype(np.float32)

    # ── Jitter circulaire style FlowSOM R ─────────────────────────────────────
    if use_circular_jitter and clustering is not None and grid_coords is not None:
        jx, jy = circular_jitter(
            clustering=clustering,
            grid_coords=grid_coords,
            node_sizes=node_sizes,
            max_radius=max_node_radius,
            min_radius=min_node_radius,
            seed=seed,
        )
        result["xGrid"] = jx.astype(np.float32)
        result["yGrid"] = jy.astype(np.float32)
    else:
        if x_grid is not None:
            result["xGrid"] = x_grid.astype(np.float32)
        if y_grid is not None:
            result["yGrid"] = y_grid.astype(np.float32)

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Jitter circulaire — style FlowSOM R
# ─────────────────────────────────────────────────────────────────────────────


def circular_jitter(
    clustering: np.ndarray,
    grid_coords: np.ndarray,
    node_sizes: Optional[np.ndarray] = None,
    max_radius: float = 0.45,
    min_radius: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère un jitter circulaire vectorisé style FlowSOM R pour les coordonnées
    xGrid/yGrid. Le rayon des cercles dépend de sqrt(node_size / max_size).

    Chaque cellule reçoit un angle aléatoire θ ∈ [0, 2π] et un rayon
    proportionnel à la taille de son nœud SOM. La distribution radiale
    utilise sqrt(U) pour une densité uniforme sur le disque.

    Args:
        clustering: Assignation de nœud SOM par cellule (n_cells,).
        grid_coords: Coordonnées de la grille SOM (n_nodes, 2).
        node_sizes: Nombre de cellules par nœud (n_nodes,). Calculé si absent.
        max_radius: Rayon max du jitter (pour les nœuds les plus peuplés).
        min_radius: Rayon min du jitter (pour les petits nœuds).
        seed: Graine aléatoire.

    Returns:
        Tuple (xGrid, yGrid) — coordonnées avec jitter circulaire (n_cells,).
    """
    n_points = len(clustering)
    cluster_ids = clustering.astype(int)

    # Calculer les tailles si non fournies (vectorisé)
    if node_sizes is None:
        n_nodes = grid_coords.shape[0]
        node_sizes = np.bincount(cluster_ids, minlength=n_nodes).astype(np.float32)

    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n_points)
    u = rng.uniform(0, 1, n_points)

    max_size_val = max(float(node_sizes.max()), 1.0)
    size_ratios = np.sqrt(node_sizes[cluster_ids] / max_size_val)
    radii = (min_radius + (max_radius - min_radius) * size_ratios).astype(np.float32)
    r = np.sqrt(u) * radii

    # Coordonnées de base sur la grille (vectorisé)
    xGrid_base = grid_coords[cluster_ids, 0].astype(np.float32)
    yGrid_base = grid_coords[cluster_ids, 1].astype(np.float32)

    # Décaler pour commencer à 1
    xGrid_shifted = xGrid_base - xGrid_base.min() + 1
    yGrid_shifted = yGrid_base - yGrid_base.min() + 1

    # Appliquer le jitter circulaire
    xGrid = xGrid_shifted + (r * np.cos(theta)).astype(np.float32)
    yGrid = yGrid_shifted + (r * np.sin(theta)).astype(np.float32)

    return xGrid, yGrid


def log_exported_columns(df: pd.DataFrame) -> None:
    """
    Affiche un rapport des colonnes numériques qui seront exportées en FCS.

    Args:
        df: DataFrame à inspecter.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    _logger.info("Colonnes FCS à exporter (%d):", len(numeric_df.columns))
    for col in numeric_df.columns:
        col_data = numeric_df[col]
        _logger.info(
            "  %-30s [%10.2f, %10.2f]",
            col,
            float(col_data.min()),
            float(col_data.max()),
        )
