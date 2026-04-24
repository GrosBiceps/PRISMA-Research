"""
prisma/core/gpu_context.py — Contexte GPU global thread-safe.

Singleton léger portant le flag use_gpu. Les stratégies le consultent
sans dépendre de PipelineConfig, évitant le couplage config → stratégie.

Usage:
    # Écriture (wizard/pipeline au démarrage)
    from prisma.core.gpu_context import GPUContext
    GPUContext.set(use_gpu=True)

    # Lecture (dans une stratégie)
    if GPUContext.use_gpu():
        ...  # tentative GPU
    else:
        log.info("[ALGO] Exécution forcée sur CPU (Version de référence) demandée par l'utilisateur.")
        ...  # CPU directement
"""

from __future__ import annotations

import threading


class GPUContext:
    """Singleton thread-safe portant le choix GPU/CPU de l'utilisateur."""

    _lock: threading.Lock = threading.Lock()
    _use_gpu: bool = True          # défaut: GPU autorisé

    @classmethod
    def set(cls, use_gpu: bool) -> None:
        """Définit la préférence GPU. Appelé par la GUI avant le lancement."""
        with cls._lock:
            cls._use_gpu = bool(use_gpu)

    @classmethod
    def use_gpu(cls) -> bool:
        """Retourne True si l'accélération GPU est autorisée."""
        with cls._lock:
            return cls._use_gpu
