"""
normalizers.py — Normalisation des données cytométriques.

Appliquée après transformation pour standardiser les plages de valeurs
avant l'entraînement FlowSOM (améliore la convergence du SOM).

Méthodes disponibles:
    - zscore_normalize  : Centrage-réduction par marqueur
    - min_max_normalize : Mise à l'échelle [0, 1] par marqueur
"""

from __future__ import annotations

import numpy as np


class DataNormalizer:
    """
    Normalisation des données de cytométrie.

    Toutes les méthodes sont statiques. La normalisation est appliquée
    colonne par colonne (par marqueur), jamais sur les cellules.
    """

    @staticmethod
    def zscore_normalize(data: np.ndarray) -> np.ndarray:
        """
        Normalisation Z-score par marqueur (colonne).

        Après normalisation: moyenne ≈ 0, écart-type ≈ 1.
        Les marqueurs constants (std = 0) ne sont pas divisés.

        Args:
            data: Matrice (n_cells, n_markers).

        Returns:
            Données normalisées, même shape.
        """
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        std[std == 0] = 1.0  # Éviter division par zéro sur marqueurs constants
        return (data - mean) / std

    @staticmethod
    def min_max_normalize(data: np.ndarray) -> np.ndarray:
        """
        Normalisation Min-Max par marqueur vers [0, 1].

        Les marqueurs constants (range = 0) sont mis à 0.

        Args:
            data: Matrice (n_cells, n_markers).

        Returns:
            Données normalisées dans [0, 1], même shape.
        """
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0  # Marqueurs constants → 0
        return (data - min_val) / range_val

    @staticmethod
    def apply(data: np.ndarray, method: str) -> np.ndarray:
        """
        Applique la normalisation spécifiée.

        Args:
            data: Matrice (n_cells, n_markers).
            method: 'zscore' | 'minmax' | 'none'.

        Returns:
            Données normalisées.

        Raises:
            ValueError: Si la méthode est inconnue.
        """
        method = method.lower()
        if method == "zscore":
            return DataNormalizer.zscore_normalize(data)
        elif method in ("minmax", "min_max"):
            return DataNormalizer.min_max_normalize(data)
        elif method == "none":
            return data.copy() if isinstance(data, np.ndarray) else np.array(data)
        else:
            raise ValueError(
                f"Méthode de normalisation inconnue: '{method}'. "
                "Valeurs acceptées: 'zscore', 'minmax', 'none'."
            )
