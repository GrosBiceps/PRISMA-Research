"""
src/gui/viewer/interactive_canvas.py — Canvas de gating interactif PyQtGraph.

Étend pyqtgraph.PlotWidget pour :
  - scatter 2D haute performance (ScatterPlotItem)
  - histogramme 1D (BarGraphItem)
  - density-colored scatter (couleur par densité KDE estimée)
  - dessin interactif : Polygon / Rectangle / Quadrant
  - affichage visuel des gates existantes (ROI overlay)
  - émission de signaux Qt propres vers GatingWorkspace

Architecture :
  DrawMode          — enum des modes interactifs
  InteractiveGatingCanvas — widget principal (PlotWidget étendu)

Signaux émis :
  polygonGateCompleted(name, x_channel, y_channel, vertices)
  rectangleGateCompleted(name, x_channel, y_channel, x_min, x_max, y_min, y_max)
  quadrantGateCompleted(name, x_channel, y_channel, x_threshold, y_threshold)
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import QWidget

from src.utils.logger import get_logger

_logger = get_logger("viewer.interactive_canvas")

# ---------------------------------------------------------------------------
# Configuration rendu
# ---------------------------------------------------------------------------

MAX_SCATTER_PTS: int = 150_000  # Au-delà → sous-échantillonnage
DENSITY_BINS: int = 80  # Résolution estimation densité KDE proxy
_BG_COLOR = "#04070D"
_GRID_COLOR = (40, 50, 70, 120)
_POINT_COLOR = (80, 160, 255, 100)
_GATED_COLOR = (80, 255, 160, 160)
_GATE_OVERLAY_COLOR = (100, 220, 120)
_GATE_DRAWING_COLOR = (255, 200, 50)
_HIST_COLOR = (80, 160, 255, 160)

pg.setConfigOption("background", _BG_COLOR)
pg.setConfigOption("foreground", "#8899AA")


# ---------------------------------------------------------------------------
# Enum des modes interactifs
# ---------------------------------------------------------------------------


class DrawMode(Enum):
    NAVIGATE = auto()
    POLYGON = auto()
    RECTANGLE = auto()
    QUADRANT = auto()


# ---------------------------------------------------------------------------
# Canvas principal
# ---------------------------------------------------------------------------


class InteractiveGatingCanvas(pg.PlotWidget):
    """
    Widget de visualisation et de gating interactif.

    Signaux
    -------
    polygonGateCompleted(str, str, str, list)
        (gate_name, x_channel, y_channel, vertices: list[tuple[float,float]])
    rectangleGateCompleted(str, str, str, float, float, float, float)
        (gate_name, x_channel, y_channel, x_min, x_max, y_min, y_max)
    quadrantGateCompleted(str, str, str, float, float)
        (gate_name, x_channel, y_channel, x_threshold, y_threshold)
    """

    polygonGateCompleted = pyqtSignal(str, str, str, list)
    rectangleGateCompleted = pyqtSignal(str, str, str, float, float, float, float)
    quadrantGateCompleted = pyqtSignal(str, str, str, float, float)
    # Émis quand l'utilisateur déplace une gate existante par drag & drop
    # (gate_id, x_channel, y_channel, new_vertices_or_bounds)
    gateModified = pyqtSignal(str, str, str, list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent=parent)

        self._x_channel: str = ""
        self._y_channel: str = ""
        self._x_label: str = ""
        self._y_label: str = ""
        self._mode: DrawMode = DrawMode.NAVIGATE
        self._next_gate_name: str = "Gate"

        # données courantes
        self._xdata: np.ndarray = np.array([])
        self._ydata: np.ndarray = np.array([])

        # items PyQtGraph permanents
        self._scatter: Optional[pg.ScatterPlotItem] = None
        self._hist_bars: Optional[pg.BarGraphItem] = None
        self._gate_overlays: Dict[str, pg.PlotDataItem] = {}
        # ROI draggables (gate_id → ROI item)
        self._gate_rois: Dict[str, object] = {}
        # Labels texte des gates (gate_id -> TextItem)
        self._gate_labels: Dict[str, pg.TextItem] = {}

        # état dessin polygon
        self._poly_vertices: List[Tuple[float, float]] = []
        self._poly_preview: Optional[pg.PlotDataItem] = None

        # état dessin rectangle 2D
        self._rect_start: Optional[Tuple[float, float]] = None
        self._rect_preview: Optional[pg.LinearRegionItem] = None

        # état dessin quadrant (lignes croisées)
        self._quad_lines: List[pg.InfiniteLine] = []

        # état gate 1D (LinearRegionItem — histogramme uniquement)
        self._region_1d: Optional[pg.LinearRegionItem] = None
        self._region_1d_gate_name: str = "Gate1D"
        self._region_1d_signal_name: Optional[str] = None

        # density image item (ImageItem PyQtGraph)
        self._density_image: Optional[pg.ImageItem] = None
        # items backgate (liste de ScatterPlotItem par sous-population)
        self._backgate_items: List[pg.ScatterPlotItem] = []

        self._setup_plot()

    # ------------------------------------------------------------------
    # Configuration du PlotWidget
    # ------------------------------------------------------------------

    def _setup_plot(self) -> None:
        self.setAntialiasing(False)
        self.showGrid(x=True, y=True, alpha=0.15)
        self.getPlotItem().setMenuEnabled(False)
        self.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.getViewBox().setMouseEnabled(x=True, y=True)
        # Désactiver le menu contextuel par défaut
        self.getViewBox().menu = None

        # Intercepter les clics via le ViewBox
        self.getViewBox().scene().sigMouseClicked.connect(self._on_scene_click)
        self.getViewBox().scene().sigMouseMoved.connect(self._on_scene_move)

    # ------------------------------------------------------------------
    # API publique — labels axes
    # ------------------------------------------------------------------

    def set_axis_labels(self, x_label: str, y_label: str) -> None:
        """Définit les labels PnS (marqueurs) à afficher sur les axes."""
        self._x_label = x_label
        self._y_label = y_label

    # ------------------------------------------------------------------
    # API publique — chargement données
    # ------------------------------------------------------------------

    def set_data_2d(
        self,
        df: pd.DataFrame,
        x_channel: str,
        y_channel: str,
        density_coloring: bool = False,
        gated_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Affiche un scatter 2D depuis un DataFrame compensé+transformé.

        Garanties PyQtGraph :
          - arrays float32 1D obligatoirement via .flatten() (évite broadcast (N,1))
          - NaN et Inf filtrés avant rendu (artefacts invisibles sinon)
          - labels axes = PnS (marqueur) si set_axis_labels() appelé avant
          - toutes les ROI/gates dessinées sont supprimées à chaque changement d'axes
            (fix "syndrome 0 events" : coordonnées perdent leur sens sur nouveaux axes)
          - si df contient 'Population_Color', coloration par population enfant
        """
        if x_channel not in df.columns or y_channel not in df.columns:
            _logger.error(
                "set_data_2d: canaux absents x='%s' y='%s'. Disponibles: %s",
                x_channel,
                y_channel,
                list(df.columns)[:10],
            )
            return

        # FIX BUG AXES : nettoyer TOUTES les ROI quand axes/données changent.
        # Les objets ROI PyQtGraph stockent des coords dans l'espace de l'ancien plot ;
        # les laisser en place avec de nouvelles dimensions produit 0 events ou crash.
        axes_changed = x_channel != self._x_channel or y_channel != self._y_channel
        if axes_changed:
            self.clear_gate_overlays()
            self._cancel_current_drawing()

        self._x_channel = x_channel
        self._y_channel = y_channel
        self.clear_data()

        # --- Extraction 1D float32 + nettoyage NaN/Inf ---
        xd = df[x_channel].to_numpy(dtype=np.float64).flatten()
        yd = df[y_channel].to_numpy(dtype=np.float64).flatten()

        # Extraire Population_Color AVANT tout filtrage pour conserver l'alignement
        pop_color_raw: Optional[np.ndarray] = None
        if "Population_Color" in df.columns:
            pop_color_raw = df["Population_Color"].to_numpy(dtype=object).flatten()

        if xd.shape[0] != yd.shape[0]:
            _logger.error("set_data_2d shape mismatch: x=%d y=%d", xd.shape[0], yd.shape[0])
            return

        # Masque valide : exclut NaN, Inf et valeurs aberrantes (>1e9 = artefact transform)
        valid = np.isfinite(xd) & np.isfinite(yd) & (np.abs(xd) < 1e9) & (np.abs(yd) < 1e9)
        if not valid.any():
            _logger.warning("set_data_2d: aucune donnée valide après nettoyage NaN/Inf")
            return

        if not valid.all():
            n_invalid = int((~valid).sum())
            _logger.debug("set_data_2d: %d points invalides filtrés (NaN/Inf/aberrant)", n_invalid)
            xd = xd[valid]
            yd = yd[valid]
            if gated_mask is not None:
                gated_mask = np.asarray(gated_mask, dtype=bool).flatten()[valid]
            if pop_color_raw is not None and pop_color_raw.shape[0] == valid.shape[0]:
                pop_color_raw = pop_color_raw[valid]

        # Conversion finale float32 pour PyQtGraph
        xd = xd.astype(np.float32)
        yd = yd.astype(np.float32)

        if gated_mask is not None:
            gated_mask = np.asarray(gated_mask, dtype=bool).flatten()
            if gated_mask.shape[0] != xd.shape[0]:
                _logger.warning(
                    "set_data_2d: gated_mask shape %d != data %d, ignoré",
                    gated_mask.shape[0],
                    xd.shape[0],
                )
                gated_mask = None

        # --- Sous-échantillonnage adaptatif ---
        n = len(xd)
        if n > MAX_SCATTER_PTS:
            rng = np.random.default_rng(42)
            ss_idx = rng.choice(n, MAX_SCATTER_PTS, replace=False)
            xd = xd[ss_idx]
            yd = yd[ss_idx]
            if gated_mask is not None:
                gated_mask = gated_mask[ss_idx]
            if pop_color_raw is not None and pop_color_raw.shape[0] == n:
                pop_color_raw = pop_color_raw[ss_idx]

        self._xdata = xd
        self._ydata = yd

        # --- Scatter principal ---
        # Priorité de coloration : Population_Color > density > couleur unie
        if pop_color_raw is not None:
            # Aligner en cas de mismatch résiduel
            n_pts = xd.shape[0]
            if pop_color_raw.shape[0] != n_pts:
                _logger.debug(
                    "set_data_2d: Population_Color shape %d != pts %d — tronqué",
                    pop_color_raw.shape[0],
                    n_pts,
                )
                pop_color_raw = (
                    pop_color_raw[:n_pts]
                    if pop_color_raw.shape[0] > n_pts
                    else np.concatenate(
                        [pop_color_raw, np.full(n_pts - pop_color_raw.shape[0], "#506070")]
                    )
                )
            colors = [pg.mkColor(str(c)) for c in pop_color_raw]
        elif density_coloring:
            colors = self._compute_density_colors(xd, yd)
        else:
            colors = [pg.mkColor(_POINT_COLOR)] * len(xd)

        self._scatter = pg.ScatterPlotItem(
            x=xd,
            y=yd,
            size=2,
            pen=None,
            brush=colors,
        )
        self.addItem(self._scatter)

        # --- Overlay événements gated ---
        if gated_mask is not None and gated_mask.any():
            xg = xd[gated_mask]
            yg = yd[gated_mask]
            gated_scatter = pg.ScatterPlotItem(
                x=xg,
                y=yg,
                size=3,
                pen=None,
                brush=pg.mkColor(_GATED_COLOR),
            )
            self.addItem(gated_scatter)

        # --- Labels axes : PnS si disponible, sinon Pnn ---
        self.getPlotItem().setLabel("bottom", self._x_label or x_channel)
        self.getPlotItem().setLabel("left", self._y_label or y_channel)
        self.autoRange()
        _logger.debug("set_data_2d: %d pts tracés (%s vs %s)", len(xd), x_channel, y_channel)

    def set_data_1d(
        self,
        df: pd.DataFrame,
        channel: str,
        n_bins: int = 256,
    ) -> None:
        """
        Affiche un histogramme 1D depuis un DataFrame.

        Args:
            df:      DataFrame (N_cells × N_channels).
            channel: Colonne à histogrammiser.
            n_bins:  Nombre de bins.
        """
        if channel not in df.columns:
            _logger.error(
                "set_data_1d: canal absent '%s'. Disponibles: %s", channel, list(df.columns)[:10]
            )
            return

        self._x_channel = channel
        self._y_channel = ""
        self.clear_data()

        xd = df[channel].to_numpy(dtype=np.float64).flatten()
        xd = xd[np.isfinite(xd) & (np.abs(xd) < 1e9)]
        xd = xd.astype(np.float32)
        self._xdata = xd
        self._ydata = np.array([], dtype=np.float32)

        counts, edges = np.histogram(xd, bins=n_bins)
        width = edges[1] - edges[0]
        centers = 0.5 * (edges[:-1] + edges[1:])

        self._hist_bars = pg.BarGraphItem(
            x=centers,
            height=counts,
            width=width * 0.9,
            brush=pg.mkColor(_HIST_COLOR),
            pen=pg.mkPen(None),
        )
        self.addItem(self._hist_bars)
        self.getPlotItem().setLabel("bottom", self._x_label or channel)
        self.getPlotItem().setLabel("left", "Count")
        self.autoRange()

    def set_data_overlay(
        self,
        datasets: List[Tuple[pd.DataFrame, str, str, str]],
        x_channel: str,
        y_channel: Optional[str] = None,
    ) -> None:
        """
        Superpose plusieurs populations sur le même canvas.

        Args:
            datasets: Liste de (df, label, color_hex, population_name).
            x_channel: Canal X commun.
            y_channel: Canal Y (None → histogramme 1D).
        """
        self.clear_data()
        self._x_channel = x_channel
        self._y_channel = y_channel or ""

        for df, label, color_hex, _pop in datasets:
            if x_channel not in df.columns:
                continue
            xd = df[x_channel].to_numpy(dtype=np.float32)

            if y_channel and y_channel in df.columns:
                yd = df[y_channel].to_numpy(dtype=np.float32)
                n = len(xd)
                if n > MAX_SCATTER_PTS // len(datasets):
                    rng = np.random.default_rng(42)
                    idx = rng.choice(n, MAX_SCATTER_PTS // len(datasets), replace=False)
                    xd, yd = xd[idx], yd[idx]
                color = pg.mkColor(color_hex)
                color.setAlpha(120)
                scatter = pg.ScatterPlotItem(x=xd, y=yd, size=2, pen=None, brush=color)
                scatter.setToolTip(label)
                self.addItem(scatter)
            else:
                xd = xd[np.isfinite(xd)]
                counts, edges = np.histogram(xd, bins=200)
                width = edges[1] - edges[0]
                centers = 0.5 * (edges[:-1] + edges[1:])
                color = pg.mkColor(color_hex)
                color.setAlpha(140)
                bars = pg.BarGraphItem(
                    x=centers,
                    height=counts,
                    width=width * 0.9,
                    brush=color,
                    pen=pg.mkPen(None),
                )
                self.addItem(bars)

        if y_channel:
            self.setLabel("bottom", x_channel)
            self.setLabel("left", y_channel)
        else:
            self.setLabel("bottom", x_channel)
            self.setLabel("left", "Count")
        self.autoRange()

    def set_data_density(
        self,
        histogram: np.ndarray,
        x_edges: np.ndarray,
        y_edges: np.ndarray,
    ) -> None:
        """
        Affiche un density plot depuis un histogram2d pré-calculé.
        Utilisé par PlotWidgetPanel._render_2d() en mode DENSITY ou HYBRID.
        L'ImageItem est positionné via setRect pour couvrir exactement la plage de données.
        """
        self.clear_data()

        # Normalisation log1p pour révéler les populations rares
        h_norm = np.log1p(histogram.astype(np.float32))
        h_max = float(h_norm.max())
        if h_max > 0:
            h_norm = h_norm / h_max  # [0, 1]

        # LUT turbo (256 niveaux)
        indices = np.linspace(0, 1, 256, dtype=np.float32)
        r_lut = np.clip((255 * (1.5 * indices - 0.5)), 0, 255).astype(np.uint8)
        g_lut = np.clip((255 * (1.5 - np.abs(3.0 * indices - 1.5))), 0, 255).astype(np.uint8)
        b_lut = np.clip((255 * (1.0 - 2.0 * indices)), 0, 255).astype(np.uint8)
        a_lut = np.clip((80 + 175 * indices), 0, 255).astype(np.uint8)

        lut = np.stack([r_lut, g_lut, b_lut, a_lut], axis=1)  # (256, 4)

        # ImageItem : l'axe 0 = X, axe 1 = Y (transpose pour pyqtgraph)
        item = pg.ImageItem(image=h_norm.T)
        item.setLookupTable(lut)

        x_min = float(x_edges[0])
        x_max = float(x_edges[-1])
        y_min = float(y_edges[0])
        y_max = float(y_edges[-1])
        x_width = x_max - x_min
        y_height = y_max - y_min

        from PyQt5.QtCore import QRectF
        item.setRect(QRectF(x_min, y_min, x_width, y_height))

        self._density_image = item
        self.addItem(item)
        self.autoRange()
        _logger.debug(
            "set_data_density: histogram %s, x=[%.3f,%.3f], y=[%.3f,%.3f]",
            histogram.shape,
            x_min,
            x_max,
            y_min,
            y_max,
        )

    def set_data_backgate(
        self,
        parent_df: pd.DataFrame,
        x_channel: str,
        y_channel: str,
        child_memberships: dict,
        colors: dict,
    ) -> None:
        """
        Affiche parent en gris semi-transparent, puis chaque sous-population colorée par-dessus.
        Utilise pg.ScatterPlotItem distinct par sous-population.
        Sous-échantillonnage MAX_SCATTER_PTS total réparti proportionnellement.

        Args:
            parent_df:        DataFrame de la population parente.
            x_channel:        Canal X.
            y_channel:        Canal Y.
            child_memberships: {gate_id: np.ndarray[bool]} — masque dans parent_df.
            colors:           {gate_id: str hex} — couleur de chaque population.
        """
        self.clear_data()
        self._x_channel = x_channel
        self._y_channel = y_channel

        if x_channel not in parent_df.columns or y_channel not in parent_df.columns:
            _logger.warning(
                "set_data_backgate: canaux absents x='%s' y='%s'", x_channel, y_channel
            )
            return

        xd_all = parent_df[x_channel].to_numpy(dtype=np.float32).flatten()
        yd_all = parent_df[y_channel].to_numpy(dtype=np.float32).flatten()

        valid = np.isfinite(xd_all) & np.isfinite(yd_all)
        xd_all = xd_all[valid]
        yd_all = yd_all[valid]

        n_total = len(xd_all)
        n_parent_show = min(MAX_SCATTER_PTS, n_total)

        # Sous-échantillonnage parent
        if n_total > n_parent_show:
            rng = np.random.default_rng(42)
            idx_parent = rng.choice(n_total, n_parent_show, replace=False)
        else:
            idx_parent = np.arange(n_total)

        xd_parent = xd_all[idx_parent]
        yd_parent = yd_all[idx_parent]

        # Population parente en gris semi-transparent
        parent_scatter = pg.ScatterPlotItem(
            x=xd_parent,
            y=yd_parent,
            size=2,
            pen=None,
            brush=pg.mkColor(80, 80, 100, 60),
        )
        self.addItem(parent_scatter)
        self._backgate_items.append(parent_scatter)

        # Chaque sous-population colorée
        n_children = len(child_memberships)
        if n_children > 0:
            max_per_child = max(1, MAX_SCATTER_PTS // n_children)
        else:
            max_per_child = MAX_SCATTER_PTS

        for gate_id, mask in child_memberships.items():
            try:
                mask_arr = np.asarray(mask, dtype=bool).flatten()
                if mask_arr.shape[0] != n_total:
                    _logger.debug(
                        "set_data_backgate: masque '%s' shape %d != parent %d",
                        gate_id,
                        mask_arr.shape[0],
                        n_total,
                    )
                    continue

                xd_child = xd_all[mask_arr]
                yd_child = yd_all[mask_arr]
                n_child = len(xd_child)

                if n_child == 0:
                    continue

                if n_child > max_per_child:
                    rng = np.random.default_rng(42)
                    idx_c = rng.choice(n_child, max_per_child, replace=False)
                    xd_child = xd_child[idx_c]
                    yd_child = yd_child[idx_c]

                color_hex = colors.get(gate_id, "#5BAAFF")
                qcolor = pg.mkColor(color_hex)
                qcolor.setAlpha(180)

                child_scatter = pg.ScatterPlotItem(
                    x=xd_child,
                    y=yd_child,
                    size=3,
                    pen=None,
                    brush=qcolor,
                )
                child_scatter.setToolTip(gate_id)
                self.addItem(child_scatter)
                self._backgate_items.append(child_scatter)
            except Exception as exc:
                _logger.debug("set_data_backgate child '%s' ignoré : %s", gate_id, exc)

        self.getPlotItem().setLabel("bottom", self._x_label or x_channel)
        self.getPlotItem().setLabel("left", self._y_label or y_channel)
        self.autoRange()
        _logger.debug(
            "set_data_backgate: %d événements parents, %d sous-populations",
            n_total,
            n_children,
        )

    # ------------------------------------------------------------------
    # API publique — modes interactifs
    # ------------------------------------------------------------------

    def set_draw_mode(self, mode: DrawMode, gate_name: str = "Gate") -> None:
        """
        Active un mode de dessin interactif.

        Args:
            mode:      DrawMode.POLYGON | RECTANGLE | QUADRANT | NAVIGATE.
            gate_name: Nom qui sera utilisé lors de l'émission du signal.
        """
        self._cancel_current_drawing()
        self._mode = mode
        self._next_gate_name = gate_name

        if mode == DrawMode.NAVIGATE:
            self.getViewBox().setMouseMode(pg.ViewBox.RectMode)
            self.setCursor(Qt.ArrowCursor)
        else:
            self.getViewBox().setMouseMode(pg.ViewBox.PanMode)
            self.setCursor(Qt.CrossCursor)

        _logger.debug("Mode dessin : %s", mode.name)

    def cancel_drawing(self) -> None:
        """Annule le dessin en cours et revient en mode navigation."""
        self._cancel_current_drawing()
        self.disable_1d_gate_drawing()
        self.set_draw_mode(DrawMode.NAVIGATE)

    # ------------------------------------------------------------------
    # CORRECTIF C5 : Gate 1D sur histogramme via LinearRegionItem
    # ------------------------------------------------------------------

    def enable_1d_gate_drawing(self, gate_name: str = "Gate1D") -> None:
        """
        Active un LinearRegionItem pour dessiner une gate 1D sur histogramme.

        À utiliser UNIQUEMENT lorsque y_channel est vide (vue histogramme).
        Si un y_channel est actif, lève un warning et ne fait rien.

        Lorsque l'utilisateur relâche la région, émet rectangleGateCompleted
        avec y_min=y_max=None encodés comme 0.0 (convention gate 1D).
        Le workspace détecte cette convention pour créer une RectangleGate
        à une dimension via fk.Dimension avec range_min/range_max.

        Args:
            gate_name: Nom de la gate qui sera transmis dans le signal.
        """
        if self._y_channel:
            _logger.warning(
                "enable_1d_gate_drawing() ignoré : y_channel='%s' actif. "
                "Utilisez set_draw_mode(DrawMode.RECTANGLE) pour une gate 2D.",
                self._y_channel,
            )
            return

        if not self._x_channel:
            _logger.warning("enable_1d_gate_drawing() : x_channel non défini.")
            return

        self.disable_1d_gate_drawing()
        self._region_1d_gate_name = gate_name

        # Estimer la plage de données pour positionner la région par défaut
        if len(self._xdata) > 0:
            x_lo = float(np.nanpercentile(self._xdata, 25))
            x_hi = float(np.nanpercentile(self._xdata, 75))
        else:
            x_lo, x_hi = 0.0, 1000.0

        pen = pg.mkPen(color=_GATE_DRAWING_COLOR, width=1.5)
        brush = pg.mkBrush(color=(255, 200, 50, 35))

        self._region_1d = pg.LinearRegionItem(
            values=(x_lo, x_hi),
            orientation="vertical",
            brush=brush,
            pen=pen,
            movable=True,
        )
        self._region_1d_signal_name = None
        if hasattr(self._region_1d, "sigRegionChangeFinished"):
            self._region_1d.sigRegionChangeFinished.connect(self._on_1d_region_finished)
            self._region_1d_signal_name = "sigRegionChangeFinished"
        elif hasattr(self._region_1d, "sigRegionChanged"):
            self._region_1d.sigRegionChanged.connect(self._on_1d_region_finished)
            self._region_1d_signal_name = "sigRegionChanged"
        self.addItem(self._region_1d)
        _logger.debug("LinearRegionItem 1D activé pour gate '%s'", gate_name)

    def disable_1d_gate_drawing(self) -> None:
        """Retire le LinearRegionItem 1D sans émettre de signal."""
        if self._region_1d is not None:
            try:
                if self._region_1d_signal_name == "sigRegionChangeFinished":
                    self._region_1d.sigRegionChangeFinished.disconnect(self._on_1d_region_finished)
                elif self._region_1d_signal_name == "sigRegionChanged":
                    self._region_1d.sigRegionChanged.disconnect(self._on_1d_region_finished)
            except TypeError:
                pass
            self.removeItem(self._region_1d)
            self._region_1d = None
            self._region_1d_signal_name = None

    def _on_1d_region_finished(self) -> None:
        """
        Slot : l'utilisateur a relâché la région 1D.

        Émet rectangleGateCompleted avec y_min=y_max=0.0 (convention 1D).
        Le workspace/engine interprète y_min==y_max==0.0 comme une gate 1D
        et appelle create_rectangle_gate_from_bounds() avec y_min=y_max=None.
        """
        if self._region_1d is None:
            return

        left, right = self._region_1d.getRegion()
        x_min = float(min(left, right))
        x_max = float(max(left, right))
        gate_name = self._region_1d_gate_name
        x_ch = self._x_channel

        _logger.info(
            "Gate 1D émise : '%s' canal=%s [%.4f, %.4f]",
            gate_name,
            x_ch,
            x_min,
            x_max,
        )

        # Convention 1D : y_channel vide, y_min=y_max=0.0
        # Le workspace doit tester : if not y_channel → gate 1D
        self.rectangleGateCompleted.emit(gate_name, x_ch, "", x_min, x_max, 0.0, 0.0)
        self.disable_1d_gate_drawing()

    # ------------------------------------------------------------------
    # API publique — affichage gates existantes
    # ------------------------------------------------------------------

    def add_gate_overlay(
        self,
        gate_id: str,
        vertices: List[Tuple[float, float]],
        color_hex: str = "#64DC78",
        label: str = "",
        draggable: bool = True,
    ) -> None:
        """
        Ajoute une gate polygonale/rectangulaire en overlay visuel draggable.

        Utilise pg.PolyLineROI (draggable) pour permettre le déplacement et
        la modification des gates directement sur le canvas.
        Émet gateModified(gate_id, x_ch, y_ch, new_vertices) quand déplacée.
        """
        self.remove_gate_overlay(gate_id)

        if not vertices:
            return

        color = pg.mkColor(color_hex)
        pen = pg.mkPen(color=color, width=1.5)
        fill_color = pg.mkColor(color_hex)
        fill_color.setAlpha(30)

        if draggable and self._y_channel:
            # ROI polygon draggable — fermeture automatique (closed=True)
            roi = None
            try:
                roi = pg.PolyLineROI(
                    positions=list(vertices),
                    closed=True,
                    pen=pen,
                    handlePen=pg.mkPen(color=color, width=1),
                    movable=True,
                )
            except TypeError:
                # Compat pyqtgraph anciens: handlePen/kwargs partiels non supportés.
                roi = pg.PolyLineROI(
                    positions=list(vertices),
                    closed=True,
                    pen=pen,
                    movable=True,
                )

            # Compat pyqtgraph: certaines versions n'exposent pas setBrush sur PolyLineROI.
            if hasattr(roi, "setBrush"):
                try:
                    roi.setBrush(pg.mkBrush(fill_color))
                except Exception:
                    pass

            def _make_handler(gid):
                def _on_roi_changed(r):
                    try:
                        new_verts: List[Tuple[float, float]] = []
                        r_pos = r.pos()
                        for h in r.getHandles():
                            h_item = h
                            if isinstance(h, dict):
                                h_item = h.get("item", h)
                            if not hasattr(h_item, "pos"):
                                continue
                            p = h_item.pos()
                            new_verts.append((float(p.x() + r_pos.x()), float(p.y() + r_pos.y())))
                        if not new_verts:
                            return
                        self.gateModified.emit(gid, self._x_channel, self._y_channel, new_verts)
                    except Exception:
                        pass

                return _on_roi_changed

            handler = _make_handler(gate_id)
            if hasattr(roi, "sigRegionChangeFinished"):
                roi.sigRegionChangeFinished.connect(handler)
            elif hasattr(roi, "sigRegionChanged"):
                # Fallback versions anciennes.
                roi.sigRegionChanged.connect(handler)
            self.addItem(roi)
            self._gate_rois[gate_id] = roi
            self._gate_overlays[gate_id] = roi  # same ref pour clear_gate_overlays
        else:
            # Mode histo 1D ou non-draggable : PlotDataItem statique
            xs = [v[0] for v in vertices] + [vertices[0][0]]
            ys = [v[1] for v in vertices] + [vertices[0][1]]
            item = pg.PlotDataItem(x=xs, y=ys, pen=pen)
            self.addItem(item)
            self._gate_overlays[gate_id] = item

        if label:
            cx = float(np.mean([v[0] for v in vertices]))
            cy = float(np.mean([v[1] for v in vertices]))
            text = pg.TextItem(label, color=color, anchor=(0.5, 0.5))
            text.setPos(cx, cy)
            self.addItem(text)
            self._gate_labels[gate_id] = text

    def remove_gate_overlay(self, gate_id: str) -> None:
        """Retire l'overlay visuel d'une gate (ROI ou PlotDataItem)."""
        self._gate_rois.pop(gate_id, None)
        label_item = self._gate_labels.pop(gate_id, None)
        if label_item is not None:
            try:
                self.removeItem(label_item)
            except Exception:
                pass
        item = self._gate_overlays.pop(gate_id, None)
        if item is not None:
            try:
                self.removeItem(item)
            except Exception:
                pass

    def clear_gate_overlays(self) -> None:
        """Retire tous les overlays de gates."""
        for item in list(self._gate_overlays.values()):
            try:
                self.removeItem(item)
            except Exception:
                pass
        for label_item in list(self._gate_labels.values()):
            try:
                self.removeItem(label_item)
            except Exception:
                pass
        self._gate_overlays.clear()
        self._gate_rois.clear()
        self._gate_labels.clear()

    def reload_gate_overlays_from_engine(
        self,
        engine: "PrismaFlowEngine",  # type: ignore[name-defined]
        sample_id: Optional[str] = None,
    ) -> None:
        """
        Reconstruit tous les overlays visuels depuis la session FlowKit.

        Parcourt les gates de type polygon/rectangle dans la stratégie
        et les réaffiche sur le canvas courant (si les canaux matchent).
        """
        self.clear_gate_overlays()
        try:
            gate_ids = engine.get_gate_ids()
        except Exception:
            return

        for gid in gate_ids:
            try:
                paths = engine.find_gate_paths(gid)
                gate_obj = engine.get_gate(gid, gate_path=paths[0] if paths else None)
                gate_type = type(gate_obj).__name__

                if "Polygon" in gate_type and hasattr(gate_obj, "vertices"):
                    verts = [(v[0], v[1]) for v in gate_obj.vertices]
                    if len(verts) >= 2:
                        self.add_gate_overlay(gid, verts, label=gid)
                elif "Rectangle" in gate_type:
                    # Reconstruire les 4 coins depuis les dimensions
                    dims = gate_obj.dimensions
                    if len(dims) >= 2:
                        x_min = dims[0].min or float("-inf")
                        x_max = dims[0].max or float("inf")
                        y_min = dims[1].min or float("-inf")
                        y_max = dims[1].max or float("inf")
                        verts = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                        self.add_gate_overlay(gid, verts, label=gid)
            except Exception as exc:
                _logger.debug("Overlay ignoré pour %s : %s", gid, exc)

    # ------------------------------------------------------------------
    # API publique — nettoyage
    # ------------------------------------------------------------------

    def clear_data(self) -> None:
        """Efface les données affichées (scatter/histo/density/backgate) mais garde les overlays."""
        if self._scatter is not None:
            self.removeItem(self._scatter)
            self._scatter = None
        if self._hist_bars is not None:
            self.removeItem(self._hist_bars)
            self._hist_bars = None
        if self._density_image is not None:
            try:
                self.removeItem(self._density_image)
            except Exception:
                pass
            self._density_image = None
        for item in self._backgate_items:
            try:
                self.removeItem(item)
            except Exception:
                pass
        self._backgate_items = []
        self._cancel_current_drawing()

    def clear_all(self) -> None:
        """Efface données + overlays + état dessin."""
        self.clear_data()
        self.clear_gate_overlays()
        self._xdata = np.array([])
        self._ydata = np.array([])

    # ------------------------------------------------------------------
    # Gestion événements souris
    # ------------------------------------------------------------------

    def _on_scene_click(self, event) -> None:
        """Intercepte les clics sur la scène PyQtGraph."""
        if self._mode == DrawMode.NAVIGATE:
            return

        if not event.isAccepted():
            vb = self.getViewBox()
            pos = vb.mapSceneToView(event.scenePos())
            x, y = float(pos.x()), float(pos.y())

            if self._mode == DrawMode.POLYGON:
                self._handle_polygon_click(x, y, double=event.double())
            elif self._mode == DrawMode.RECTANGLE:
                self._handle_rectangle_click(x, y)
            elif self._mode == DrawMode.QUADRANT:
                self._handle_quadrant_click(x, y)

            event.accept()

    def _on_scene_move(self, pos) -> None:
        """Mise à jour du preview lors du déplacement souris."""
        if self._mode == DrawMode.POLYGON and self._poly_vertices:
            vb = self.getViewBox()
            data_pos = vb.mapSceneToView(pos)
            self._update_polygon_preview(float(data_pos.x()), float(data_pos.y()))

    # ------------------------------------------------------------------
    # Mode Polygon
    # ------------------------------------------------------------------

    def _handle_polygon_click(self, x: float, y: float, double: bool) -> None:
        if double:
            if len(self._poly_vertices) >= 3:
                self._finalize_polygon()
            return

        self._poly_vertices.append((x, y))
        self._update_polygon_preview(x, y)

    def _update_polygon_preview(self, cursor_x: float, cursor_y: float) -> None:
        if not self._poly_vertices:
            return

        xs = [v[0] for v in self._poly_vertices] + [cursor_x]
        ys = [v[1] for v in self._poly_vertices] + [cursor_y]

        if self._poly_preview is not None:
            self.removeItem(self._poly_preview)

        pen = pg.mkPen(color=_GATE_DRAWING_COLOR, width=1.5, style=Qt.DashLine)
        self._poly_preview = pg.PlotDataItem(x=xs, y=ys, pen=pen)
        self.addItem(self._poly_preview)

    def _finalize_polygon(self) -> None:
        verts = list(self._poly_vertices)
        name = self._next_gate_name
        x_ch = self._x_channel
        y_ch = self._y_channel

        if not y_ch:
            _logger.warning("Mode histogramme actif : impossible de créer une PolygonGate 2D.")
            self.cancel_drawing()
            return

        self._cancel_current_drawing()
        self.set_draw_mode(DrawMode.NAVIGATE)
        _logger.info("PolygonGate émise : %s (%d sommets)", name, len(verts))
        self.polygonGateCompleted.emit(name, x_ch, y_ch, verts)

    # ------------------------------------------------------------------
    # Mode Rectangle
    # ------------------------------------------------------------------

    def _handle_rectangle_click(self, x: float, y: float) -> None:
        if not self._y_channel:
            _logger.warning("Mode histogramme actif : impossible de créer une RectangleGate 2D.")
            self.cancel_drawing()
            return

        if self._rect_start is None:
            self._rect_start = (x, y)
            _logger.debug("Rectangle : premier coin enregistré (%.3f, %.3f)", x, y)
        else:
            x0, y0 = self._rect_start
            x_min, x_max = (min(x0, x), max(x0, x))
            y_min, y_max = (min(y0, y), max(y0, y))
            self._rect_start = None
            self._cleanup_rect_preview()
            self.set_draw_mode(DrawMode.NAVIGATE)
            _logger.info(
                "RectangleGate émise : %s x=[%.3f,%.3f] y=[%.3f,%.3f]",
                self._next_gate_name,
                x_min,
                x_max,
                y_min,
                y_max,
            )
            self.rectangleGateCompleted.emit(
                self._next_gate_name,
                self._x_channel,
                self._y_channel,
                x_min,
                x_max,
                y_min,
                y_max,
            )

    def _cleanup_rect_preview(self) -> None:
        if self._rect_preview is not None:
            self.removeItem(self._rect_preview)
            self._rect_preview = None

    # ------------------------------------------------------------------
    # Mode Quadrant
    # ------------------------------------------------------------------

    def _handle_quadrant_click(self, x: float, y: float) -> None:
        if not self._y_channel:
            _logger.warning("Mode histogramme actif : impossible de créer une QuadrantGate 2D.")
            self.cancel_drawing()
            return

        self._cleanup_quad_lines()

        # Affichage des deux lignes croisées
        pen_x = pg.mkPen(color=_GATE_DRAWING_COLOR, width=1.2, style=Qt.DashLine)
        pen_y = pg.mkPen(color=_GATE_DRAWING_COLOR, width=1.2, style=Qt.DashLine)
        line_x = pg.InfiniteLine(pos=x, angle=90, pen=pen_x)
        line_y = pg.InfiniteLine(pos=y, angle=0, pen=pen_y)
        self.addItem(line_x)
        self.addItem(line_y)
        self._quad_lines = [line_x, line_y]

        name = self._next_gate_name
        x_ch = self._x_channel
        y_ch = self._y_channel
        self.set_draw_mode(DrawMode.NAVIGATE)
        _logger.info("QuadrantGate émise : %s x_thr=%.3f y_thr=%.3f", name, x, y)
        self.quadrantGateCompleted.emit(name, x_ch, y_ch, x, y)

    def _cleanup_quad_lines(self) -> None:
        for line in self._quad_lines:
            self.removeItem(line)
        self._quad_lines = []

    # ------------------------------------------------------------------
    # Annulation en cours
    # ------------------------------------------------------------------

    def _cancel_current_drawing(self) -> None:
        self._poly_vertices = []
        if self._poly_preview is not None:
            self.removeItem(self._poly_preview)
            self._poly_preview = None
        self._rect_start = None
        self._cleanup_rect_preview()
        # Ne pas nettoyer _region_1d ici : géré par disable_1d_gate_drawing()
        # Conserver les lignes quadrant (elles font partie de l'overlay final)

    # ------------------------------------------------------------------
    # Touche Escape = annulation
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.cancel_drawing()
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Calcul densité locale (proxy KDE via histogramme 2D)
    # ------------------------------------------------------------------

    def _compute_density_colors(self, xd: np.ndarray, yd: np.ndarray) -> List:
        """
        Estime la densité locale par histogramme 2D (numpy.histogram2d, <100 ms sur 150k pts).

        Pipeline :
          1. Grille 2D → densité par bin
          2. Normalisation log1p (compresse les pics, révèle les populations rares)
          3. Colormap turbo vectorisée (pas de boucle Python)
          4. Alpha adaptatif : points denses plus opaques
        """
        try:
            h, xe, ye = np.histogram2d(xd, yd, bins=DENSITY_BINS)

            xi = np.clip(np.searchsorted(xe, xd) - 1, 0, DENSITY_BINS - 1)
            yi = np.clip(np.searchsorted(ye, yd) - 1, 0, DENSITY_BINS - 1)
            densities = h[xi, yi].astype(np.float32)

            # Normalisation log1p : étire les faibles densités, comprime les pics
            densities = np.log1p(densities)
            d_min, d_max = float(densities.min()), float(densities.max())
            if d_max <= d_min:
                return [pg.mkColor(_POINT_COLOR)] * len(xd)

            norm = ((densities - d_min) / (d_max - d_min)).astype(np.float32)  # [0, 1]

            # Turbo colormap vectorisée (coefficients approximatifs)
            r = np.clip((255 * (1.5 * norm - 0.5)), 0, 255).astype(np.uint8)
            g = np.clip((255 * (1.5 - np.abs(3.0 * norm - 1.5))), 0, 255).astype(np.uint8)
            b = np.clip((255 * (1.0 - 2.0 * norm)), 0, 255).astype(np.uint8)
            # Alpha : 80 (creux) → 200 (pic) pour révéler la structure
            a = np.clip((80 + 120 * norm), 0, 255).astype(np.uint8)

            return [
                pg.mkColor(QColor(int(r[i]), int(g[i]), int(b[i]), int(a[i])))
                for i in range(len(norm))
            ]

        except Exception as exc:
            _logger.debug("Density coloring échoué : %s", exc)
            return [pg.mkColor(_POINT_COLOR)] * len(xd)
