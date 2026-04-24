"""
prisma/core/torch_utils.py — Utilitaires GPU PyTorch anti-OOM pour les stratégies.

Fournit :
  torch_cuda_available()          — détection GPU avec logging détaillé
  torch_available()               — True si PyTorch installé (CPU ou GPU)
  get_device()                    — cuda:0 si dispo, sinon cpu
  torch_knn_indices()             — kNN exact chunked (indices uniquement)
  torch_knn()                     — kNN exact chunked (indices + distances)
  torch_pairwise_sq_distances()   — distances² complètes (petits datasets ≤8k)

Stratégie anti-OOM (cytométrie 500k–2M cellules) :
  torch_knn() et torch_knn_indices() découpent les requêtes en chunks.
  Chaque chunk calcule les distances vers TOUTES les cellules de référence,
  extrait les topk, puis libère la mémoire GPU avant le chunk suivant.

  La taille de chunk est auto-adaptée à la VRAM disponible si
  auto_chunk=True (défaut). En cas d'OOM malgré tout, le chunk est divisé
  par 2 jusqu'à réussite ou chunk_size=1 (impossible en pratique).

Formule distances euclidiennes (stable, sans torch.cdist) :
  ||a - b||² = ||a||² + ||b||² - 2·(a @ b^T)
  clamp_(min=0) pour éviter les NaN issus d'imprécisions float32.
"""

from __future__ import annotations

import gc
import logging
import math
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Détection PyTorch + CUDA au chargement ───────────────────────────────────
try:
    import torch as _torch

    _TORCH_AVAILABLE = True

    if _torch.cuda.is_available():
        _TORCH_CUDA = True
        _gpu_name   = _torch.cuda.get_device_name(0)
        _cuda_ver   = _torch.version.cuda
        _vram_total = _torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(
            "[torch_utils] PyTorch CUDA disponible — GPU : %s | CUDA %s | VRAM %.1f GB",
            _gpu_name, _cuda_ver, _vram_total,
        )
    else:
        _TORCH_CUDA   = False
        _gpu_name     = None
        _cuda_ver     = None
        _vram_total   = 0.0
        logger.info("[torch_utils] PyTorch disponible — CUDA absent, mode CPU uniquement")

except ImportError:
    _torch        = None          # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    _TORCH_CUDA   = False
    _gpu_name     = None
    _cuda_ver     = None
    _vram_total   = 0.0


# ── API publique — détection ─────────────────────────────────────────────────

def torch_available() -> bool:
    """True si PyTorch est installé (CPU ou GPU)."""
    return _TORCH_AVAILABLE


def torch_cuda_available() -> bool:
    """True si PyTorch avec accélération CUDA est disponible."""
    return _TORCH_CUDA


def gpu_name() -> Optional[str]:
    """Nom du GPU détecté, ou None."""
    return _gpu_name


def vram_total_gb() -> float:
    """VRAM totale du GPU en GB (0.0 si pas de GPU)."""
    return _vram_total


def get_device() -> "torch.device":  # type: ignore[name-defined]
    """Retourne cuda:0 si CUDA disponible, sinon cpu."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch n'est pas installé.")
    return _torch.device("cuda:0" if _TORCH_CUDA else "cpu")


# ── Calcul de chunk optimal ───────────────────────────────────────────────────

def _optimal_chunk_size(n_ref: int, n_features: int, dtype_bytes: int = 4) -> int:
    """
    Estime la taille de chunk maximale sans dépasser ~70% de la VRAM libre.

    Budget mémoire pour un chunk B :
      - batch en entrée  :  B × d × 4 bytes
      - matrice distances: B × N × 4 bytes   ← dominant
      - topk buffer      :  B × k × 4 bytes  (négligeable)

    On cible 70% de la VRAM libre (headroom pour PyTorch runtime + tenseurs
    permanents). Sur CPU, chunk = 512 (vitesse raisonnable sans explosion RAM).
    """
    if not _TORCH_CUDA:
        return 512

    try:
        free, _ = _torch.cuda.mem_get_info(0)
        free_gb = free / (1024 ** 3)
    except Exception:
        free_gb = max(_vram_total * 0.5, 1.0)

    budget_bytes = free_gb * (1024 ** 3) * 0.70
    # B × N × 4 ≤ budget  →  B ≤ budget / (N × 4)
    chunk = int(budget_bytes / (n_ref * dtype_bytes))
    chunk = max(64, min(chunk, 8192))   # bornes : [64, 8192]
    return chunk


# ── kNN chunked — fonction principale ────────────────────────────────────────

def torch_knn(
    data: np.ndarray,
    k: int,
    chunk_size: Optional[int] = None,
    auto_chunk: bool = True,
    device: Optional["torch.device"] = None,  # type: ignore[name-defined]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    kNN exact GPU chunked — retourne (indices, distances).

    Calcule les distances euclidiennes par chunks sur la dimension des requêtes.
    Chaque chunk est traité indépendamment et libéré avant le suivant,
    garantissant un usage mémoire constant O(chunk × N) quelle que soit N.

    Args:
        data:       Matrice (N, d) — sera convertie en float32.
        k:          Nombre de voisins (hors la cellule elle-même).
        chunk_size: Taille de chunk fixe. Si None et auto_chunk=True,
                    calculée automatiquement selon la VRAM libre.
        auto_chunk: Si True, adapte chunk_size à la VRAM disponible.
        device:     torch.device cible (défaut : cuda:0 ou cpu).

    Returns:
        indices   np.ndarray (N, k) int32   — indices des k voisins
        distances np.ndarray (N, k) float32 — distances euclidiennes (pas²)

    Raises:
        ImportError: PyTorch non installé.
        ValueError:  k ≥ N.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch est requis pour torch_knn(). "
            "Installez-le : pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )

    data = np.asarray(data, dtype=np.float32)
    n, d = data.shape

    if k >= n:
        raise ValueError(f"k={k} doit être < N={n}.")

    dev = device or get_device()

    if chunk_size is None:
        chunk_size = _optimal_chunk_size(n, d) if auto_chunk else 4096

    logger.info(
        "[torch_knn] %d cells × %d features | k=%d | chunk=%d | device=%s",
        n, d, k, chunk_size, dev,
    )

    # Tenseur de référence (toutes les cellules) — reste en GPU pendant tout le calcul
    x_ref = _torch.from_numpy(data).to(dev)           # (N, d)
    sq_ref = (x_ref ** 2).sum(dim=1, keepdim=True).T  # (1, N) — précalculé

    all_indices   = np.empty((n, k), dtype=np.int32)
    all_distances = np.empty((n, k), dtype=np.float32)

    chunk = chunk_size
    start = 0

    while start < n:
        end = min(start + chunk, n)
        success = False

        while not success:
            try:
                batch = x_ref[start:end]                              # (B, d)
                sq_batch = (batch ** 2).sum(dim=1, keepdim=True)     # (B, 1)
                dists_sq = sq_batch + sq_ref - 2.0 * (batch @ x_ref.T)  # (B, N)
                dists_sq.clamp_(min=0.0)

                # Masquer soi-même (diagonale du batch) → inf
                diag_idx = _torch.arange(end - start, device=dev)
                dists_sq[diag_idx, start + diag_idx] = float("inf")

                # topk voisins les plus proches
                top_sq, top_idx = _torch.topk(
                    dists_sq, k=k, dim=1, largest=False, sorted=True
                )

                # Copier vers CPU numpy immédiatement
                all_indices[start:end]   = top_idx.to(_torch.int32).cpu().numpy()
                all_distances[start:end] = _torch.sqrt(top_sq).cpu().numpy()

                # Libération GPU explicite
                del batch, sq_batch, dists_sq, top_sq, top_idx
                if _TORCH_CUDA:
                    _torch.cuda.empty_cache()

                success = True

            except RuntimeError as exc:
                # OOM → réduire chunk de moitié et réessayer
                if "out of memory" in str(exc).lower() and chunk > 1:
                    del batch, sq_batch  # noqa: F821 — peut ne pas exister
                    if _TORCH_CUDA:
                        _torch.cuda.empty_cache()
                    gc.collect()
                    chunk = max(1, chunk // 2)
                    logger.warning(
                        "[torch_knn] OOM GPU — chunk réduit à %d et retry...", chunk
                    )
                    end = min(start + chunk, n)
                else:
                    # Erreur non-OOM ou chunk=1 → re-raise
                    raise

        start = end

    # Libération du tenseur de référence
    del x_ref, sq_ref
    if _TORCH_CUDA:
        _torch.cuda.empty_cache()

    logger.info("[torch_knn] done — indices %s, distances %s", all_indices.shape, all_distances.shape)
    return all_indices, all_distances


def torch_knn_indices(
    data: np.ndarray,
    k: int,
    chunk_size: Optional[int] = None,
    auto_chunk: bool = True,
    device: Optional["torch.device"] = None,  # type: ignore[name-defined]
) -> np.ndarray:
    """
    Raccourci retournant uniquement les indices (N, k) int32.

    Voir torch_knn() pour la documentation complète.
    Utilisé par UMAP et PhenoGraph qui n'ont besoin que des indices.
    """
    indices, _ = torch_knn(data, k=k, chunk_size=chunk_size, auto_chunk=auto_chunk, device=device)
    return indices


# ── Matrice distances² complète (petits datasets) ────────────────────────────

def torch_pairwise_sq_distances(
    data: np.ndarray,
    max_n: int = 8_000,
    device: Optional["torch.device"] = None,  # type: ignore[name-defined]
) -> np.ndarray:
    """
    Matrice complète de distances euclidiennes² (N, N) float32 sur GPU.

    Réservé aux petits datasets (N ≤ max_n) pour éviter OOM.
    Au-dessus de max_n, lève une ValueError pour forcer l'usage de torch_knn().

    Args:
        data:   Matrice (N, d) — float32.
        max_n:  Limite supérieure de N (défaut 8 000 ≈ 256 MB sur RTX 3060).
        device: torch.device cible.

    Returns:
        np.ndarray (N, N) float32.

    Raises:
        ValueError: N > max_n.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch requis pour torch_pairwise_sq_distances().")

    data = np.asarray(data, dtype=np.float32)
    n = len(data)

    if n > max_n:
        raise ValueError(
            f"torch_pairwise_sq_distances: N={n} > max_n={max_n}. "
            f"Utilisez torch_knn() pour les grands datasets."
        )

    dev = device or get_device()
    logger.info("[torch_pairwise_sq] %d × %d sur %s", n, n, dev)

    x = _torch.from_numpy(data).to(dev)
    sq = (x ** 2).sum(dim=1, keepdim=True)
    dists = sq + sq.T - 2.0 * (x @ x.T)
    dists.clamp_(min=0.0)
    result = dists.cpu().numpy()

    del x, sq, dists
    if _TORCH_CUDA:
        _torch.cuda.empty_cache()

    return result
