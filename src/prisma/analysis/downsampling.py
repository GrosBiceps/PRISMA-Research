"""
analysis/downsampling.py — Stratégies de sous-échantillonnage reproductibles.

DownsampleResult conserve les indices sélectionnés pour relier
l'embedding aux événements originaux dans le Sample.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Résultat de downsampling
# ---------------------------------------------------------------------------

@dataclass
class DownsampleResult:
    """
    Contient les données sous-échantillonnées et les indices d'origine.

    Attributes:
        data: Sous-ensemble (n_sampled × n_features).
        indices: Indices dans le tableau d'origine (n_sampled,).
        n_original: Nombre d'événements avant downsampling.
        was_downsampled: True si un downsampling a eu lieu.
    """
    data: np.ndarray
    indices: np.ndarray
    n_original: int
    was_downsampled: bool


# ---------------------------------------------------------------------------
# Downsampling aléatoire reproductible
# ---------------------------------------------------------------------------

def random_downsample(
    data: np.ndarray,
    max_events: int,
    seed: int = 42,
) -> DownsampleResult:
    """
    Sous-échantillonne data si len(data) > max_events.

    Args:
        data: Matrice (N × F).
        max_events: Seuil au-delà duquel on sous-échantillonne.
        seed: Graine pour la reproductibilité.

    Returns:
        DownsampleResult avec les données et les indices conservés.
    """
    n = len(data)
    if n <= max_events:
        return DownsampleResult(
            data=data,
            indices=np.arange(n),
            n_original=n,
            was_downsampled=False,
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=max_events, replace=False)
    indices.sort()
    logger.info(
        "Downsampling: %d → %d événements (seed=%d)", n, max_events, seed
    )
    return DownsampleResult(
        data=data[indices],
        indices=indices,
        n_original=n,
        was_downsampled=True,
    )


def random_downsample_df(
    df: pd.DataFrame,
    max_events: int,
    seed: int = 42,
) -> DownsampleResult:
    """Même chose sur un DataFrame pandas. Retourne data en ndarray."""
    arr = df.to_numpy(dtype=np.float32)
    return random_downsample(arr, max_events=max_events, seed=seed)


# ---------------------------------------------------------------------------
# Expansion : scatter vers l'espace original
# ---------------------------------------------------------------------------

def expand_to_full(
    values: np.ndarray,
    result: DownsampleResult,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Projette un vecteur calculé sur les événements downsampleés
    vers un vecteur de taille n_original.

    Les positions non sélectionnées reçoivent fill_value (NaN par défaut).

    Args:
        values: Résultat sur les événements sous-échantillonnés (n_sampled,) ou
                (n_sampled, k) pour les embeddings 2D.
        result: DownsampleResult de la même session de downsampling.
        fill_value: Valeur pour les événements non sélectionnés.

    Returns:
        Array (n_original,) ou (n_original, k).
    """
    if values.ndim == 1:
        full = np.full(result.n_original, fill_value, dtype=values.dtype)
        full[result.indices] = values
    else:
        k = values.shape[1]
        full = np.full((result.n_original, k), fill_value, dtype=values.dtype)
        full[result.indices] = values
    return full
