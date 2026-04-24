"""
analysis/trajectory.py — Outils trajectoires (PHATE + fallback).

Le module reste pur (pas d'UI) et retourne des matrices numpy testables.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import phate  # type: ignore

    _PHATE_AVAILABLE = True
except ImportError:
    _PHATE_AVAILABLE = False


@dataclass
class PHATEParams:
    """Paramètres PHATE pour trajectoires continues."""

    n_components: int = 2
    knn: int = 10
    decay: int = 40
    t: str | int = "auto"
    gamma: float = 1.0
    random_state: int = 42


def _pca_fallback(X: np.ndarray, n_components: int) -> np.ndarray:
    """Fallback déterministe (PCA) si PHATE indisponible."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X).astype(np.float32)


def run_phate_embedding(
    X: np.ndarray,
    params: PHATEParams,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Calcule un embedding trajectoire PHATE avec fallback propre.

    Returns:
        (embedding, metadata)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape={X.shape}.")
    if X.shape[0] < 3:
        raise ValueError("PHATE requires at least 3 events.")
    if params.n_components not in (2, 3):
        raise ValueError("PHATE n_components must be 2 or 3.")

    if not _PHATE_AVAILABLE:
        logger.warning("[PHATE] package missing; fallback to PCA.")
        emb = _pca_fallback(X, params.n_components)
        return emb, {
            "backend": "pca_fallback",
            "shape": tuple(int(v) for v in emb.shape),
        }

    try:
        op = phate.PHATE(
            n_components=params.n_components,
            knn=params.knn,
            decay=params.decay,
            t=params.t,
            gamma=params.gamma,
            random_state=params.random_state,
            verbose=False,
        )
        emb = np.asarray(op.fit_transform(X), dtype=np.float32)
        return emb, {
            "backend": "phate",
            "shape": tuple(int(v) for v in emb.shape),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("[PHATE] failed (%s); fallback to PCA.", exc)
        emb = _pca_fallback(X, params.n_components)
        return emb, {
            "backend": "pca_fallback",
            "reason": "phate_error",
            "error": str(exc),
            "shape": tuple(int(v) for v in emb.shape),
        }
