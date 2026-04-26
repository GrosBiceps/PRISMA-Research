"""
screen_utils.py — Utilitaires d'adaptation à la taille d'écran pour PRISMA.

Usage :
    from gui.screen_utils import maximize_window, fit_dialog_to_screen

    # Fenêtre principale → plein écran natif (barre tâches visible)
    maximize_window(win)

    # Dialog → 90% de l'écran, centré
    fit_dialog_to_screen(dlg, ratio=0.90)
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import QApplication, QWidget


def _primary_screen_geometry() -> QRect:
    """Retourne la géométrie disponible de l'écran primaire (hors barre des tâches)."""
    screen = QApplication.primaryScreen()
    if screen is not None:
        return screen.availableGeometry()
    # Fallback si appelé avant QApplication
    return QRect(0, 0, 1920, 1080)


def maximize_window(win: QWidget) -> None:
    """
    Affiche win en plein écran maximisé (état natif OS — barre des tâches visible).
    Compatible Windows / macOS / Linux.
    Doit être appelé AVANT ou APRÈS win.show() — les deux fonctionnent.
    """
    win.showMaximized()


def fit_dialog_to_screen(
    dlg: QWidget,
    ratio: float = 0.88,
    min_w: int = 900,
    min_h: int = 600,
) -> None:
    """
    Redimensionne dlg à `ratio` de l'écran disponible et le centre.

    Args:
        dlg:   QDialog ou QMainWindow à adapter.
        ratio: Fraction de l'écran (0.88 = 88%).
        min_w: Largeur minimale garantie (pixels).
        min_h: Hauteur minimale garantie (pixels).
    """
    geom = _primary_screen_geometry()
    w = max(int(geom.width() * ratio), min_w)
    h = max(int(geom.height() * ratio), min_h)

    x = geom.x() + (geom.width() - w) // 2
    y = geom.y() + (geom.height() - h) // 2

    dlg.setGeometry(x, y, w, h)
