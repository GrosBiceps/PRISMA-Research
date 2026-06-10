"""
embedding_cache.py — Cache disque LRU pour embeddings coûteux (UMAP, t-SNE, FlowSOM).

Clé de cache = hash blake2b des données + paramètres.
Serialisation pickle compressé (lzma niveau 1 — rapide, ratio ~5x).
Éviction LRU basée sur mtime des fichiers.
"""

from __future__ import annotations

import hashlib
import logging
import lzma
import pickle
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger("prisma.cache.embedding")

CACHE_DIR = Path.home() / ".prisma_cache" / "embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MAX_CACHE_GB = 4.0
MAX_CACHE_BYTES = int(MAX_CACHE_GB * 1024**3)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _hash_key(data: np.ndarray, params: dict) -> str:
    h = hashlib.blake2b(digest_size=20)
    h.update(data.tobytes())
    # Trie les clés pour que l'ordre des params n'affecte pas le hash
    h.update(str(sorted((k, str(v)) for k, v in params.items())).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Éviction LRU
# ---------------------------------------------------------------------------

def _evict_if_needed() -> None:
    files = sorted(CACHE_DIR.glob("*.pkl.xz"), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    while total > MAX_CACHE_BYTES and files:
        oldest = files.pop(0)
        size = oldest.stat().st_size
        oldest.unlink(missing_ok=True)
        total -= size
        log.debug("[cache] Éviction LRU: %s (%.1f MB)", oldest.name, size / 1024**2)


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

def get_or_compute(
    data: np.ndarray,
    params: dict,
    compute_fn: Callable[..., Any],
    tag: str = "embed",
) -> Any:
    """
    Retourne le résultat depuis le cache disque, ou appelle compute_fn et le met en cache.

    Args:
        data: Données source (utilisées pour le hash uniquement).
        params: Paramètres de l'algorithme (dict sérialisable en str).
        compute_fn: Callable(data, **params) → résultat.
        tag: Préfixe lisible pour les fichiers cache (ex: 'umap', 'flowsom').

    Returns:
        Résultat de compute_fn (potentiellement depuis cache).
    """
    key = _hash_key(data, params)
    cache_file = CACHE_DIR / f"{tag}_{key}.pkl.xz"

    if cache_file.exists():
        try:
            with lzma.open(cache_file, "rb") as f:
                result = pickle.load(f)
            # Mise à jour mtime pour LRU
            cache_file.touch()
            log.info("[cache] HIT %s (%s)", tag, key[:8])
            return result
        except Exception as e:
            log.warning("[cache] Lecture corrompue, recalcul: %s", e)
            cache_file.unlink(missing_ok=True)

    log.info("[cache] MISS %s — calcul en cours...", tag)
    t0 = time.perf_counter()
    result = compute_fn(data, **params)
    elapsed = time.perf_counter() - t0
    log.info("[cache] Calcul terminé en %.1fs", elapsed)

    try:
        _evict_if_needed()
        with lzma.open(cache_file, "wb", preset=1) as f:  # preset=1 : rapide
            pickle.dump(result, f)
        log.info("[cache] Écrit: %s (%.1f MB)", cache_file.name,
                 cache_file.stat().st_size / 1024**2)
    except Exception as e:
        log.warning("[cache] Écriture échouée: %s", e)

    return result


def invalidate(data: np.ndarray, params: dict, tag: str = "embed") -> bool:
    """Supprime une entrée du cache. Retourne True si supprimé."""
    key = _hash_key(data, params)
    cache_file = CACHE_DIR / f"{tag}_{key}.pkl.xz"
    if cache_file.exists():
        cache_file.unlink()
        log.info("[cache] Invalidé: %s", cache_file.name)
        return True
    return False


def clear_all() -> int:
    """Vide entièrement le cache. Retourne le nombre de fichiers supprimés."""
    files = list(CACHE_DIR.glob("*.pkl.xz"))
    for f in files:
        f.unlink(missing_ok=True)
    log.info("[cache] Cache vidé: %d fichiers supprimés", len(files))
    return len(files)


def cache_stats() -> dict:
    """Retourne les statistiques du cache."""
    files = list(CACHE_DIR.glob("*.pkl.xz"))
    total_bytes = sum(f.stat().st_size for f in files)
    return {
        "n_entries": len(files),
        "total_mb": total_bytes / 1024**2,
        "max_gb": MAX_CACHE_GB,
        "cache_dir": str(CACHE_DIR),
    }
