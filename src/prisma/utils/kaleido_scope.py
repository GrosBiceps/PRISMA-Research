"""kaleido_scope.py — Serveur kaleido persistant pour les exports Plotly → image.

Kaleido ≥ 1.0 expose start_sync_server() / write_fig_sync() qui permettent de
garder un seul processus Chromium actif pour tous les exports d'une session.
Sans ça, chaque fig.write_image() démarre et arrête Chromium (~2-4s par figure).

Utilisation :
    from src.prisma.utils.kaleido_scope import (
        ensure_kaleido_scope, warm_up_kaleido, write_image_fast
    )
    ensure_kaleido_scope()          # idempotent, démarre le serveur une fois
    write_image_fast(fig, path, format="png", width=1800, height=950, scale=2)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger("utils.kaleido_scope")
_scope_initialized: bool = False


def ensure_kaleido_scope() -> bool:
    """Démarre le serveur kaleido persistant si ce n'est pas déjà fait.

    Compatible kaleido ≥ 1.0 (start_sync_server) et kaleido 0.x (pio.kaleido.scope).

    Returns:
        True si le serveur est actif, False sinon.
    """
    global _scope_initialized
    if _scope_initialized:
        return True

    # ── Kaleido ≥ 1.0 ────────────────────────────────────────────────────────
    try:
        import kaleido
        if hasattr(kaleido, "start_sync_server"):
            kaleido.start_sync_server(silence_warnings=True)
            _scope_initialized = True
            _logger.debug("Serveur kaleido 1.x démarré (persistant).")
            return True
    except Exception as exc:
        _logger.debug("kaleido 1.x start_sync_server échoué: %s", exc)

    # ── Kaleido 0.x fallback ─────────────────────────────────────────────────
    try:
        import plotly.io as pio
        scope = getattr(pio, "kaleido", None)
        if scope is not None:
            _ = scope.scope
            _scope_initialized = True
            _logger.debug("Scope kaleido 0.x initialisé (persistant).")
            return True
    except Exception as exc:
        _logger.debug("kaleido 0.x scope échoué: %s", exc)

    return False


def warm_up_kaleido() -> bool:
    """Rend une image vide pour forcer le démarrage de Chromium en avance.

    À appeler une seule fois au début du pipeline, avant les vrais exports.

    Returns:
        True si le warm-up a réussi, False sinon.
    """
    ensure_kaleido_scope()
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        write_image_fast(fig, None, fmt="png", width=10, height=10, scale=1)
        _logger.debug("Kaleido warm-up effectué.")
        return True
    except Exception as exc:
        _logger.debug("Kaleido warm-up échoué (non bloquant): %s", exc)
        return False


def write_image_fast(
    fig: Any,
    path: "str | Path | None",
    fmt: str = "png",
    width: int = 1400,
    height: int = 700,
    scale: float = 2.0,
) -> None:
    """Exporte une figure Plotly en image via le serveur kaleido persistant.

    Réutilise le processus Chromium déjà démarré par ensure_kaleido_scope(),
    évitant le coût de démarrage (~2-4s) à chaque figure.

    Args:
        fig: Figure Plotly (go.Figure).
        path: Chemin de sortie. Si None, l'export est ignoré (utile pour warm-up).
        fmt: Format image ("png", "jpg", "svg", "pdf").
        width: Largeur en pixels.
        height: Hauteur en pixels.
        scale: Facteur d'échelle (DPI effectif).
    """
    if path is None:
        return

    ensure_kaleido_scope()

    # ── Kaleido ≥ 1.0 : write_fig_sync ───────────────────────────────────────
    try:
        import kaleido
        if hasattr(kaleido, "write_fig_sync"):
            from kaleido import LayoutOpts
            opts = LayoutOpts(width=width, height=height, scale=scale)
            kaleido.write_fig_sync(fig, path=str(path), opts=opts)
            return
    except Exception as exc:
        _logger.debug("write_fig_sync échoué, fallback write_image: %s", exc)

    # ── Fallback : fig.write_image standard ──────────────────────────────────
    fig.write_image(str(path), format=fmt, width=width, height=height, scale=scale)
