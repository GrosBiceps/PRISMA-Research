"""
prisma/core/torch_utils.py — Utilitaires GPU PyTorch pour les stratégies.

Fournit :
  - torch_knn_indices()  : kNN exact sur GPU (distances euclidiennes)
  - torch_pairwise_sq()  : matrice distances² sur GPU (petits datasets)
  - TorchDevice.get()    : device cuda/cpu avec détection auto

Ces fonctions sont utilisées comme accélérateur GPU intermédiaire quand
cuML n'est pas disponible (Windows) mais PyTorch CUDA l'est.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Détection PyTorch CUDA au chargement du module
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
    if _torch.cuda.is_available():
        _TORCH_CUDA = True
        logger.info(
            "[torch_utils] PyTorch CUDA disponible — GPU: %s (CUDA %s)",
            _torch.cuda.get_device_name(0),
            _torch.version.cuda,
        )
    else:
        _TORCH_CUDA = False
        logger.info("[torch_utils] PyTorch disponible mais CUDA absent — CPU uniquement")
except ImportError:
    _torch = None
    _TORCH_AVAILABLE = False
    _TORCH_CUDA = False


def torch_available() -> bool:
    """True si PyTorch est installé."""
    return _TORCH_AVAILABLE


def torch_cuda_available() -> bool:
    """True si PyTorch avec CUDA GPU est disponible."""
    return _TORCH_CUDA


def get_device() -> "torch.device":  # type: ignore[name-defined]
    """Retourne cuda:0 si dispo, sinon cpu."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch non installé.")
    return _torch.device("cuda:0" if _TORCH_CUDA else "cpu")


def torch_knn_indices(
    data: np.ndarray,
    k: int,
    batch_size: int = 4096,
    device: Optional["torch.device"] = None,  # type: ignore[name-defined]
) -> np.ndarray:
    """
    kNN exact GPU via PyTorch — retourne indices (n, k).

    Calcul par batch pour éviter OOM sur grandes matrices.
    Utilise la distance euclidienne L2 exacte.

    Args:
        data:       Matrice (n, d) float32.
        k:          Nombre de voisins (sans inclure la cellule elle-même).
        batch_size: Taille des batches pour le calcul de distances.
        device:     torch.device cible (défaut: cuda:0 si dispo sinon cpu).

    Returns:
        indices np.ndarray (n, k) int64.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch requis pour torch_knn_indices.")

    dev = device or get_device()
    n = len(data)
    k_safe = min(k + 1, n)  # +1 car inclut la cellule elle-même

    x = _torch.from_numpy(np.asarray(data, dtype=np.float32)).to(dev)
    all_indices = _torch.empty((n, k_safe - 1), dtype=_torch.long, device="cpu")

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = x[start:end]                        # (B, d)
        # Distance euclidienne² = ||a||² + ||b||² - 2 a·b^T
        sq_a = (batch ** 2).sum(dim=1, keepdim=True)       # (B, 1)
        sq_b = (x ** 2).sum(dim=1, keepdim=True).T         # (1, n)
        dists = sq_a + sq_b - 2.0 * batch @ x.T            # (B, n)
        # Exclure soi-même : mettre inf sur la diagonale du batch
        for i in range(end - start):
            dists[i, start + i] = float("inf")
        # k_safe - 1 voisins (hors soi-même)
        _, idx = _torch.topk(dists, k=k_safe - 1, dim=1, largest=False, sorted=True)
        all_indices[start:end] = idx.cpu()

    return all_indices.numpy()


def torch_pairwise_sq_distances(
    data: np.ndarray,
    device: Optional["torch.device"] = None,  # type: ignore[name-defined]
) -> np.ndarray:
    """
    Matrice complète de distances² (n, n) sur GPU.

    Uniquement pour n ≤ ~8k (limite mémoire GPU typique ~6GB).

    Returns:
        np.ndarray (n, n) float32.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch requis.")

    dev = device or get_device()
    x = _torch.from_numpy(np.asarray(data, dtype=np.float32)).to(dev)
    sq = (x ** 2).sum(dim=1, keepdim=True)
    dists = sq + sq.T - 2.0 * x @ x.T
    dists.clamp_(min=0.0)
    return dists.cpu().numpy()
