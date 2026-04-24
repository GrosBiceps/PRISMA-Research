"""
strategies/spectral_strategy.py — Démixage spectral (cytométrie spectrale).

Utilise scipy.optimize.nnls pour la déconvolution non négative des spectres.
Supporte matrice de référence (spillover/unmixing matrix).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.optimize import nnls

from prisma.core.registry import StrategyRegistry

logger = logging.getLogger(__name__)


@dataclass
class SpectralParams:
    """
    Paramètres de démixage spectral.

    Attributes:
        reference_matrix: Matrice de référence (n_detectors × n_fluorochromes).
        autofluorescence_col: Index de la colonne autofluorescence (-1 = absent).
        clip_negative: Si True, remplace valeurs négatives post-unmixing par 0.
    """
    reference_matrix: Optional[np.ndarray] = None
    autofluorescence_col: int = -1
    clip_negative: bool = True


@StrategyRegistry.register_dimreduc("spectral")
class SpectralUnmixingStrategy:
    """
    Démixage spectral par NNLS (Non-Negative Least Squares).

    Input  : data (n_cells × n_detectors)
    Output : abundance (n_cells × n_fluorochromes)
    """

    name: str = "spectral"

    def __init__(self) -> None:
        self._residuals: Optional[np.ndarray] = None

    def fit_transform(
        self,
        data: np.ndarray,
        params: SpectralParams,  # type: ignore[override]
    ) -> np.ndarray:
        """
        Applique le démixage NNLS cellule par cellule.

        Args:
            data: (n_cells × n_detectors) — intensités brutes spectrales.
            params: SpectralParams avec reference_matrix obligatoire.

        Returns:
            (n_cells × n_fluorochromes) — abondances démixées.
        """
        if params.reference_matrix is None:
            raise ValueError("SpectralParams.reference_matrix must be provided.")

        A = params.reference_matrix  # shape: (n_detectors, n_fluoro)
        n_cells, n_det = data.shape
        n_fluoro = A.shape[1]

        if A.shape[0] != n_det:
            raise ValueError(
                f"reference_matrix rows ({A.shape[0]}) != data columns ({n_det})"
            )

        logger.info(
            "Spectral unmixing: %d cells, %d detectors → %d fluorochromes",
            n_cells, n_det, n_fluoro,
        )

        abundances = np.empty((n_cells, n_fluoro), dtype=np.float64)
        residuals = np.empty(n_cells, dtype=np.float64)

        for i in range(n_cells):
            x, res = nnls(A, data[i])
            abundances[i] = x
            residuals[i] = res

        self._residuals = residuals

        if params.clip_negative:
            abundances = np.clip(abundances, 0, None)

        logger.info(
            "Spectral unmixing done. Mean residual: %.4f", residuals.mean()
        )
        return abundances

    @property
    def residuals(self) -> Optional[np.ndarray]:
        """Résidus NNLS par cellule (n_cells,) — disponible après fit_transform."""
        return self._residuals
