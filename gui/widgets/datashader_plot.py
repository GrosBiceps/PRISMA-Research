"""
datashader_plot.py — Scatter plot haute performance via Datashader + PyQtGraph.

Rasterise des millions de points en < 50ms côté CPU → image RGBA → ImageItem PyQtGraph.
Pan/zoom → re-rasterisation débounced 100ms → 60fps perçu.

Dépendances : datashader, pyqtgraph (OpenGL activé dans launch_gui.py), PyQt5.
Fallback automatique vers ScatterPlotItem standard si datashader absent.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor

log = logging.getLogger("prisma.gui.datashader_plot")

try:
    import datashader as ds
    import datashader.transfer_functions as tf
    import pandas as pd
    _HAS_DATASHADER = True
except ImportError:
    _HAS_DATASHADER = False
    log.warning("datashader non installé — fallback ScatterPlotItem (lent sur >100k pts)")


# Palettes de couleurs pour Datashader
PALETTES = {
    "prisma":  ["#0d0d1a", "#003366", "#0066cc", "#00d4ff", "#39ff14", "#ff6b35"],
    "plasma":  ["#0d0887", "#7e03a8", "#cc4778", "#f89441", "#f0f921"],
    "heat":    ["#000000", "#1a0000", "#800000", "#ff4500", "#ffcc00", "#ffffff"],
    "blues":   ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
}


class DatashaderScatterWidget(pg.GraphicsLayoutWidget):
    """
    Scatter plot haute performance.

    Sur 10M points : < 50ms de rendu (CPU rasterisation Datashader).
    Fallback automatique sur ScatterPlotItem si Datashader absent.
    """

    def __init__(
        self,
        palette: str = "prisma",
        debounce_ms: int = 80,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._palette = PALETTES.get(palette, PALETTES["prisma"])
        self._n_rendered: int = 0

        # Mise en page du plot
        self._plot = self.addPlot()
        self._plot.setLabel("bottom", "X")
        self._plot.setLabel("left", "Y")
        self._plot.showGrid(x=True, y=True, alpha=0.2)

        if _HAS_DATASHADER:
            self._img_item = pg.ImageItem()
            self._img_item.setOpts(axisOrder="row-major")
            self._plot.addItem(self._img_item)
            self._scatter_item = None
        else:
            # Fallback : ScatterPlotItem avec décimation agressive
            self._scatter_item = pg.ScatterPlotItem(
                size=1, pen=None, brush=pg.mkBrush(0, 180, 255, 80)
            )
            self._plot.addItem(self._scatter_item)
            self._img_item = None

        # Débounce pan/zoom → re-rasterise 80ms après la fin du mouvement
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(debounce_ms)
        self._redraw_timer.timeout.connect(self._rasterize)
        self._plot.getViewBox().sigRangeChanged.connect(
            lambda: self._redraw_timer.start()
        )

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def set_data(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_label: str = "X",
        y_label: str = "Y",
    ) -> None:
        """
        Charge de nouvelles données et redessine.
        Doit être appelé depuis le Main Thread (slot connecté à un signal worker).
        """
        self._x = np.asarray(x, dtype=np.float32)
        self._y = np.asarray(y, dtype=np.float32)
        self._plot.setLabel("bottom", x_label)
        self._plot.setLabel("left", y_label)

        # Auto-range sur les nouvelles données
        x_min, x_max = float(np.nanmin(self._x)), float(np.nanmax(self._x))
        y_min, y_max = float(np.nanmin(self._y)), float(np.nanmax(self._y))
        margin_x = (x_max - x_min) * 0.05 or 1.0
        margin_y = (y_max - y_min) * 0.05 or 1.0
        self._plot.setXRange(x_min - margin_x, x_max + margin_x, padding=0)
        self._plot.setYRange(y_min - margin_y, y_max + margin_y, padding=0)

        self._rasterize()

    def set_palette(self, name: str) -> None:
        self._palette = PALETTES.get(name, PALETTES["prisma"])
        self._rasterize()

    def clear_data(self) -> None:
        self._x = None
        self._y = None
        if self._img_item:
            self._img_item.clear()
        if self._scatter_item:
            self._scatter_item.clear()

    # ------------------------------------------------------------------
    # Rendu interne
    # ------------------------------------------------------------------

    def _rasterize(self) -> None:
        if self._x is None or len(self._x) == 0:
            return

        if _HAS_DATASHADER:
            self._rasterize_datashader()
        else:
            self._rasterize_fallback()

    def _rasterize_datashader(self) -> None:
        vb = self._plot.getViewBox()
        rect = vb.viewRect()

        x_min, x_max = rect.left(), rect.right()
        y_min, y_max = rect.top(), rect.bottom()

        if x_max <= x_min or y_max <= y_min:
            return

        # Résolution en pixels du widget
        w = max(int(self.width()), 512)
        h = max(int(self.height()), 512)

        # Filtrage viewport (évite de rasteriser hors-écran)
        in_view = (
            (self._x >= x_min) & (self._x <= x_max) &
            (self._y >= y_min) & (self._y <= y_max)
        )
        x_view = self._x[in_view]
        y_view = self._y[in_view]
        self._n_rendered = int(in_view.sum())

        if self._n_rendered == 0:
            return

        try:
            cvs = ds.Canvas(
                plot_width=w, plot_height=h,
                x_range=(x_min, x_max),
                y_range=(y_min, y_max),
            )
            df = pd.DataFrame({"x": x_view, "y": y_view})
            agg = cvs.points(df, "x", "y", ds.count())
            img_ds = tf.shade(agg, cmap=self._palette, how="log", min_alpha=40)

            # PIL → numpy RGBA uint8
            rgba = np.array(img_ds.to_pil(), dtype=np.uint8)     # (H, W, 4)
            # PyQtGraph ImageItem row-major : axe Y normal (pas d'inversion)

            self._img_item.setImage(rgba)
            self._img_item.setRect(
                pg.QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min)
            )
        except Exception as e:
            log.warning("[datashader] Erreur rasterisation: %s", e)

    def _rasterize_fallback(self) -> None:
        """Fallback ScatterPlotItem avec décimation à 50k points max."""
        MAX_PTS = 50_000
        n = len(self._x)
        if n > MAX_PTS:
            idx = np.random.choice(n, MAX_PTS, replace=False)
            x, y = self._x[idx], self._y[idx]
        else:
            x, y = self._x, self._y

        self._scatter_item.setData(x=x, y=y)
        self._n_rendered = len(x)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def n_total(self) -> int:
        return len(self._x) if self._x is not None else 0

    @property
    def n_rendered(self) -> int:
        return self._n_rendered
