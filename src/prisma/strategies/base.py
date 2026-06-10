"""
strategies/base.py — Protocols pour les stratégies PRISMA Research.

DimReducStrategy : interface réduction dimensionnelle.
ClusterStrategy : interface clustering.

Toutes les implémentations concrètes importent ces protocols
et s'enregistrent via StrategyRegistry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# DimReducStrategy
# ---------------------------------------------------------------------------

@runtime_checkable
class DimReducStrategy(Protocol):
    """
    Interface commune pour UMAP, t-SNE, PHATE, etc.

    Chaque stratégie expose fit_transform() retournant un tableau 2D (n_cells, n_dims).
    """

    name: str

    def fit_transform(
        self,
        data: np.ndarray,
        params: "DimReducParams",
    ) -> np.ndarray:
        """
        Réduit data (n_cells × n_features) en embedding (n_cells × n_dims).

        Args:
            data: Matrice d'expression pré-normalisée.
            params: Hyperparamètres de la stratégie.

        Returns:
            Embedding numpy (n_cells × n_dims).
        """
        ...


# ---------------------------------------------------------------------------
# ClusterStrategy
# ---------------------------------------------------------------------------

@runtime_checkable
class ClusterStrategy(Protocol):
    """
    Interface commune pour PhenoGraph, Leiden, KMeans, FlowSOM.

    Retourne un vecteur d'entiers (n_cells,) de labels de cluster.
    """

    name: str

    def fit_predict(
        self,
        data: np.ndarray,
        params: "ClusterParams",
    ) -> np.ndarray:
        """
        Affecte chaque cellule à un cluster.

        Args:
            data: Matrice (n_cells × n_features) ou embedding réduit.
            params: Hyperparamètres du clustering.

        Returns:
            Labels (n_cells,) dtype int.
        """
        ...


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------

@dataclass
class DimReducParams:
    """Hyperparamètres communs aux stratégies de réduction dimensionnelle."""
    n_components: int = 2
    seed: int = 42
    n_jobs: int = -1
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterParams:
    """Hyperparamètres communs aux stratégies de clustering."""
    n_clusters: Optional[int] = None
    seed: int = 42
    n_jobs: int = -1
    extra: Dict[str, Any] = field(default_factory=dict)
