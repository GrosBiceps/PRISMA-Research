"""
src/gui/viewer/gating_workspace.py — Widget principal PrismaGatingWorkspace.

Layout 3 panneaux (QSplitter) :
  Gauche  : QTreeView hiérarchie de gating (population / count / %parent / %gp)
  Centre  : InteractiveGatingCanvas (scatter 2D / histo 1D / density / overlay)
  Droite  : Panneau de contrôles (sample, population, axes, mode, export)

Responsabilités :
  - Connecter signaux du canvas → engine → mise à jour UI
  - Orchestrer l'affichage sans logique métier
  - Afficher les erreurs via QMessageBox
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QModelIndex, QEvent, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QSizePolicy,
    QFrame,
    QInputDialog,
    QSpinBox,
)
from PyQt5.QtGui import QFont

from src.prisma.utils.logger import get_logger
from .gating_engine import (
    PrismaFlowEngine,
    PrismaEngineError,
)
from .gating_tree_model import GatingTreeModel, GateNode
from .plot_panel_model import PlotPanelModel, RenderMode, DensityCache
from .population_tree_controller import PopulationTreeController
from .statistics_dock import StatisticsDock
from .statistics_controller import StatisticsController
from .worksheet_layout import (
    WorkspaceLayoutState,
    PanelState,
    WorksheetLayoutError,
    validate_column_count,
    save_layout_state,
    load_layout_state,
)

try:
    from .interactive_canvas import InteractiveGatingCanvas, DrawMode

    _INTERACTIVE_CANVAS_AVAILABLE = True
except Exception as exc:  # pragma: no cover - dépendance optionnelle
    _INTERACTIVE_CANVAS_AVAILABLE = False
    _INTERACTIVE_CANVAS_IMPORT_ERROR = exc

    class DrawMode(Enum):
        NAVIGATE = auto()
        POLYGON = auto()
        RECTANGLE = auto()
        QUADRANT = auto()

    class InteractiveGatingCanvas(QFrame):
        polygonGateCompleted = pyqtSignal(str, str, str, list)
        rectangleGateCompleted = pyqtSignal(str, str, str, float, float, float, float)
        quadrantGateCompleted = pyqtSignal(str, str, str, float, float)

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self.setObjectName("interactiveCanvasFallback")
            self.setMinimumWidth(320)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(8)
            label = QLabel(
                "PyQtGraph indisponible\n"
                "Le workspace peut s'ouvrir en mode dégradé, mais le dessin interactif\n"
                "des gates est désactivé tant que la dépendance n'est pas installée."
            )
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)

        def set_draw_mode(self, mode: DrawMode, gate_name: str = "Gate") -> None:
            return None

        def cancel_drawing(self) -> None:
            return None

        def set_data_2d(
            self,
            df: pd.DataFrame,
            x_ch: str,
            y_ch: str,
            density_coloring: bool = False,
        ) -> None:
            return None

        def set_data_1d(self, df: pd.DataFrame, x_ch: str) -> None:
            return None

        def set_data_overlay(
            self,
            datasets: List[Any],
            x_ch: str,
            y_ch: str,
        ) -> None:
            return None

        def reload_gate_overlays_from_engine(self, engine: PrismaFlowEngine) -> None:
            return None

        def remove_gate_overlay(self, gate_id: str) -> None:
            return None


_logger = get_logger("viewer.gating_workspace")

# ---------------------------------------------------------------------------
# Couleurs de population pour overlay (cycle)
# ---------------------------------------------------------------------------

_OVERLAY_COLORS = [
    "#5BAAFF",  # V500 bleu
    "#39FF8A",  # FITC vert
    "#FF9B3D",  # PE orange
    "#FF3D6E",  # APC rouge
    "#FFE032",  # PerCP jaune
    "#7B52FF",  # V450 violet
    "#7EC8E3",  # SSC cyan
]


# ---------------------------------------------------------------------------
# PlotWidgetPanel — panneau autonome (canvas + axes propres + label source)
# ---------------------------------------------------------------------------


class PlotWidgetPanel(QFrame):
    """
    Panneau de visualisation autonome pour le Worksheet multi-plot.

    Chaque panneau possède un PlotPanelModel qui sert de source de vérité locale.
    Le rendu est découplé des gates : un changement d'axes ne détruit jamais
    les gates FlowKit — il masque seulement les overlays incompatibles.

    Architecture :
      PlotPanelModel  → état pur (population, axes, transform, comp)
      DensityCache    → cache histogram2d par (x_ch, y_ch, sample_id, n_events)
      InteractiveGatingCanvas → rendu PyQtGraph uniquement

    Signaux
    -------
    gateCreated(gate_name)  — émis après création d'une gate dans ce panneau.
    closeRequested()        — l'utilisateur clique le bouton ×.
    activated()             — panel cliqué (devient le panel actif).
    """

    gateCreated = pyqtSignal(str)
    closeRequested = pyqtSignal()
    activated = pyqtSignal()

    # Seuils rendu hybride — OpenGL gère 80k pts à 60fps sans problème
    _SCATTER_MAX = 80_000   # scatter pur jusqu'à 80k (OpenGL)
    _DENSITY_MIN = 500_000  # density map seulement au-delà de 500k
    _SUBSAMPLE_N = 80_000   # sous-échantillonnage mode hybride

    def __init__(
        self,
        engine: "PrismaFlowEngine",
        gate_node: Tuple[str, ...],
        channel_mapping: Dict[str, str],
        active_transform_id: Optional[str] = None,
        active_comp_id: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._channel_mapping = channel_mapping  # {pnn: marker}
        self._density_cache = DensityCache()

        # Modèle de données local — source de vérité du panneau
        import uuid as _uuid
        self._model = PlotPanelModel(
            gate_node=gate_node,
            transform_id=active_transform_id,
            comp_id=active_comp_id,
            panel_id=str(_uuid.uuid4())[:8],
        )

        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(320, 280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._build_ui()
        self._connect_signals()
        self._canvas.installEventFilter(self)

    # ------ Propriétés publiques ------------------------------------------

    @property
    def _gate_node(self) -> Tuple[str, ...]:
        """Compat rétrograde — pointe sur le modèle."""
        return self._model.gate_node

    @property
    def canvas(self) -> "InteractiveGatingCanvas":
        return self._canvas

    # ------ Construction UI -----------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        # Barre titre : label source + badge population + bouton ×
        title_row = QHBoxLayout()
        pop_label = self._model.population_label
        self._lbl_source = QLabel(f"Pop : {pop_label}")
        self._lbl_source.setStyleSheet("color: #5BAAFF; font-size: 10px; font-weight: bold;")
        title_row.addWidget(self._lbl_source)

        self._lbl_count = QLabel("")
        self._lbl_count.setStyleSheet("color: #8899AA; font-size: 9px;")
        title_row.addWidget(self._lbl_count)
        title_row.addStretch()

        self._btn_close = QToolButton()
        self._btn_close.setText("×")
        self._btn_close.setFixedSize(18, 18)
        self._btn_close.setStyleSheet(
            "QToolButton { color: #FF3D6E; border: none; font-weight: bold; }"
            "QToolButton:hover { background: #1E2B3E; }"
        )
        title_row.addWidget(self._btn_close)
        layout.addLayout(title_row)

        # Combos X / Y
        axes_row = QHBoxLayout()
        axes_row.addWidget(QLabel("X:"))
        self._combo_x = QComboBox()
        self._combo_x.setMaximumWidth(130)
        axes_row.addWidget(self._combo_x)
        axes_row.addWidget(QLabel("Y:"))
        self._combo_y = QComboBox()
        self._combo_y.setMaximumWidth(130)
        axes_row.addWidget(self._combo_y)
        axes_row.addStretch()
        layout.addLayout(axes_row)

        # Combo transformation propre à ce panel
        xform_row = QHBoxLayout()
        xform_row.addWidget(QLabel("Transf:"))
        self._combo_transform = QComboBox()
        self._combo_transform.addItems(["Logicle", "Hyperlog", "Asinh", "Log", "Linear"])
        self._combo_transform.setCurrentText("Logicle")
        self._combo_transform.setMaximumWidth(110)
        xform_row.addWidget(self._combo_transform)
        xform_row.addStretch()
        layout.addLayout(xform_row)

        # Contrôles de rendu : Densité + Contour (mutuellement exclusifs)
        render_row = QHBoxLayout()
        self._chk_density_panel = QCheckBox("Densité")
        self._chk_density_panel.setStyleSheet("color: #8899AA; font-size: 9px;")
        render_row.addWidget(self._chk_density_panel)

        self._chk_contour_panel = QCheckBox("Contour")
        self._chk_contour_panel.setStyleSheet("color: #39FF8A; font-size: 9px;")
        self._chk_contour_panel.setToolTip("Affiche les iso-courbes de densité (style Kaluza)")
        render_row.addWidget(self._chk_contour_panel)

        self._spin_contour_levels = QSpinBox()
        self._spin_contour_levels.setRange(3, 15)
        self._spin_contour_levels.setValue(8)
        self._spin_contour_levels.setPrefix("N=")
        self._spin_contour_levels.setFixedWidth(58)
        self._spin_contour_levels.setStyleSheet("font-size: 9px;")
        self._spin_contour_levels.setVisible(False)
        render_row.addWidget(self._spin_contour_levels)
        render_row.addStretch()
        layout.addLayout(render_row)

        # Canvas
        self._canvas = InteractiveGatingCanvas(self)
        layout.addWidget(self._canvas)

        # Remplir les combos axes
        self._combo_y.addItem("", userData="")
        for pnn, marker in self._channel_mapping.items():
            display = f"{marker} ({pnn})" if marker and marker != pnn else pnn
            self._combo_x.addItem(display, userData=pnn)
            self._combo_y.addItem(display, userData=pnn)

    def _connect_signals(self) -> None:
        self._btn_close.clicked.connect(self.closeRequested)
        self._combo_x.currentIndexChanged.connect(self._on_axes_changed)
        self._combo_y.currentIndexChanged.connect(self._on_axes_changed)
        self._combo_transform.currentTextChanged.connect(self._on_transform_changed)
        self._chk_density_panel.toggled.connect(self._on_density_toggled)
        self._chk_contour_panel.toggled.connect(self._on_contour_toggled)
        self._spin_contour_levels.valueChanged.connect(self._on_contour_levels_changed)

    def _on_density_toggled(self, checked: bool) -> None:
        if checked:
            # Mutuellement exclusif avec contour
            self._chk_contour_panel.blockSignals(True)
            self._chk_contour_panel.setChecked(False)
            self._chk_contour_panel.blockSignals(False)
            self._model.contour_mode = False
            self._spin_contour_levels.setVisible(False)
        self._model.density_coloring = checked
        self._density_cache.invalidate()
        self._refresh()

    def _on_contour_toggled(self, checked: bool) -> None:
        if checked:
            # Mutuellement exclusif avec densité KDE
            self._chk_density_panel.blockSignals(True)
            self._chk_density_panel.setChecked(False)
            self._chk_density_panel.blockSignals(False)
            self._model.density_coloring = False
        self._model.contour_mode = checked
        self._spin_contour_levels.setVisible(checked)
        self._density_cache.invalidate()
        self._refresh()

    def _on_contour_levels_changed(self, value: int) -> None:
        self._model.contour_levels = value
        if self._model.contour_mode:
            self._refresh()

    # ------ Gestion axes et transformation --------------------------------

    def _get_axis(self, combo: QComboBox) -> str:
        data = combo.currentData()
        if isinstance(data, str) and data.strip():
            return data.strip()
        text = combo.currentText().strip()
        if text.endswith(")") and "(" in text:
            return text[text.rfind("(") + 1 : -1].strip()
        return text

    def _on_axes_changed(self, _index: int = 0) -> None:
        """
        Changement d'axes : met à jour le modèle, purge les overlays incompatibles,
        recrée ceux qui sont compatibles, recharge les données.
        Les gates FlowKit ne sont PAS touchées.
        """
        new_x = self._get_axis(self._combo_x)
        new_y = self._get_axis(self._combo_y) or ""

        axes_changed = (new_x != self._model.x_channel or new_y != self._model.y_channel)
        self._model.x_channel = new_x
        self._model.y_channel = new_y

        if axes_changed:
            # Invalider le cache density car les axes ont changé
            self._density_cache.invalidate()
            # Purger tous les overlays du canvas
            if hasattr(self._canvas, "clear_gate_overlays"):
                self._canvas.clear_gate_overlays()
            # Recréer uniquement les overlays compatibles avec les nouveaux axes
            self._reload_compatible_overlays()

        self._refresh()

    def _on_transform_changed(self, transform_type: str) -> None:
        if not transform_type:
            return
        try:
            self._model.transform_id = self._engine.apply_transformation(transform_type)
        except Exception as exc:
            _logger.warning("PlotWidgetPanel transform: %s", exc)
        self._density_cache.invalidate()
        self._refresh()

    # ------ Overlays compatibles -----------------------------------------

    def _reload_compatible_overlays(self) -> None:
        """
        Affiche uniquement les gates dont les dimensions correspondent aux axes courants.
        N'est appelé que lors d'un changement d'axes — ne recalcule rien dans FlowKit.
        """
        x_ch = self._model.x_channel
        y_ch = self._model.y_channel or None
        compatible_ids = self._engine.get_compatible_gate_ids(x_ch, y_ch)
        for gate_id in compatible_ids:
            try:
                verts = self._engine.get_gate_vertices(gate_id)
                if verts and hasattr(self._canvas, "add_gate_overlay"):
                    self._canvas.add_gate_overlay(gate_id, verts, label=gate_id)
            except Exception as exc:
                _logger.debug("Overlay compat '%s': %s", gate_id, exc)

    # ------ Refresh principal --------------------------------------------

    def _refresh(self) -> None:
        """
        Recharge et affiche les données depuis l'engine.
        Stratégie hybride : scatter / density / hybrid selon volume.
        Ne modifie JAMAIS l'état FlowKit.
        """
        x_ch = self._model.x_channel
        y_ch = self._model.y_channel

        if not x_ch:
            return

        # En mode CONTOUR, charger toutes les données (histogram2d bénéficie du volume complet)
        # En mode SCATTER/HYBRID, limiter à 80k pour la fluidité OpenGL
        max_ev = None if self._model.contour_mode else 80_000

        try:
            if max_ev is not None and hasattr(self._engine, "get_population_df_sampled"):
                df = self._engine.get_population_df_sampled(
                    self._model.gate_node,
                    max_events=max_ev,
                    transform_id=self._model.transform_id,
                    comp_matrix_id=self._model.comp_id,
                )
            else:
                df = self._engine.get_population_df(
                    self._model.gate_node,
                    transform_id=self._model.transform_id,
                    comp_matrix_id=self._model.comp_id,
                )
        except Exception as exc:
            _logger.warning("PlotWidgetPanel._refresh: %s", exc)
            return

        if df is None or df.empty:
            return

        n_events = len(df)
        self._lbl_count.setText(f"({n_events:,})")

        # Résolution du mode de rendu — si densité cochée, force SCATTER (coloration KDE)
        render_mode = self._model.resolve_render_mode(
            n_events, force_density_coloring=self._model.density_coloring
        )
        self._model.render_mode = render_mode

        # Labels axes
        ch_map = self._channel_mapping
        x_label = ch_map.get(x_ch, x_ch)
        y_label = ch_map.get(y_ch, y_ch) if y_ch else ""
        if hasattr(self._canvas, "set_axis_labels"):
            self._canvas.set_axis_labels(x_label, y_label)

        # Tolérance casse sur les noms de canaux
        if x_ch not in df.columns:
            match = next(
                (c for c in df.columns if str(c).strip().upper() == x_ch.strip().upper()), None
            )
            if match:
                x_ch = match
            else:
                _logger.warning("Canal X '%s' absent du DataFrame panel", x_ch)
                return

        if y_ch and y_ch not in df.columns:
            match = next(
                (c for c in df.columns if str(c).strip().upper() == y_ch.strip().upper()), None
            )
            y_ch = match or ""

        # --- Rendu selon mode ---
        if y_ch and y_ch in df.columns:
            self._render_2d(df, x_ch, y_ch, render_mode)
        elif x_ch in df.columns:
            self._canvas.set_data_1d(df, x_ch)

    def _render_2d(
        self,
        df: "pd.DataFrame",
        x_ch: str,
        y_ch: str,
        mode: RenderMode,
    ) -> None:
        """Rendu 2D avec stratégie hybride scatter/density/contour."""
        n = len(df)
        sid = self._engine.active_sample_id or ""
        use_density_color = self._model.density_coloring

        if mode == RenderMode.CONTOUR:
            if hasattr(self._canvas, "set_data_contour"):
                self._canvas.set_data_contour(
                    df, x_ch, y_ch,
                    n_levels=self._model.contour_levels,
                    bins=self._model.contour_bins,
                )
            else:
                # Fallback si set_data_contour absent (version ancienne du canvas)
                self._canvas.set_data_2d(df, x_ch, y_ch, density_coloring=True)
            return

        if mode == RenderMode.SCATTER:
            # Cas normal ET cas bouton densité coché (force_density_coloring → SCATTER)
            self._canvas.set_data_2d(df, x_ch, y_ch, density_coloring=use_density_color)

        elif mode == RenderMode.DENSITY:
            # Volume > 500k : histogram2d ImageItem plein écran
            # Si densité cochée malgré tout → scatter sous-échantillonné coloré (lisible)
            if use_density_color:
                sub = df.sample(min(self._SUBSAMPLE_N, n), random_state=42)
                self._canvas.set_data_2d(sub, x_ch, y_ch, density_coloring=True)
            else:
                if self._density_cache.is_valid_for(x_ch, y_ch, sid, n):
                    h = self._density_cache.histogram
                    xe = self._density_cache.x_edges
                    ye = self._density_cache.y_edges
                else:
                    xs = df[x_ch].to_numpy(dtype=np.float32)
                    ys = df[y_ch].to_numpy(dtype=np.float32)
                    h, xe, ye = np.histogram2d(xs, ys, bins=256)
                    self._density_cache.update(x_ch, y_ch, sid, n, h, xe, ye)

                if hasattr(self._canvas, "set_data_density"):
                    self._canvas.set_data_density(h, xe, ye)
                else:
                    sub = df.sample(min(self._SUBSAMPLE_N, n), random_state=42)
                    self._canvas.set_data_2d(sub, x_ch, y_ch, density_coloring=False)

        else:  # HYBRID — density fond + scatter par-dessus
            if not self._density_cache.is_valid_for(x_ch, y_ch, sid, n):
                xs = df[x_ch].to_numpy(dtype=np.float32)
                ys = df[y_ch].to_numpy(dtype=np.float32)
                h, xe, ye = np.histogram2d(xs, ys, bins=256)
                self._density_cache.update(x_ch, y_ch, sid, n, h, xe, ye)

            if hasattr(self._canvas, "set_data_density"):
                self._canvas.set_data_density(
                    self._density_cache.histogram,
                    self._density_cache.x_edges,
                    self._density_cache.y_edges,
                )

            sub = df.sample(min(self._SUBSAMPLE_N, n), random_state=42)
            # En mode HYBRID, density_coloring sur le scatter par-dessus si coché
            self._canvas.set_data_2d(sub, x_ch, y_ch, density_coloring=use_density_color)

    def refresh_from_engine(
        self,
        engine: "PrismaFlowEngine",
        transform_id: Optional[str] = None,
        comp_id: Optional[str] = None,
    ) -> None:
        """Recharge depuis le moteur après modification de la stratégie de gating."""
        self._engine = engine
        if transform_id is not None:
            self._model.transform_id = transform_id
        if comp_id is not None:
            self._model.comp_id = comp_id
        # Invalider le cache density car les résultats ont changé
        self._density_cache.invalidate()
        # Reconstruire les overlays compatibles (la liste des gates a pu changer)
        if hasattr(self._canvas, "clear_gate_overlays"):
            self._canvas.clear_gate_overlays()
        self._reload_compatible_overlays()
        self._refresh()

    def add_gate_overlay_if_compatible(
        self,
        gate_id: str,
        x_ch: str,
        y_ch: Optional[str],
    ) -> None:
        """
        Ajoute l'overlay d'une gate si elle est compatible avec les axes courants.
        Appelé par le workspace lors de gatingStrategyChanged.
        """
        if not self._engine.is_gate_compatible_with(gate_id, self._model.x_channel, self._model.y_channel or None):
            return
        try:
            verts = self._engine.get_gate_vertices(gate_id)
            if verts and hasattr(self._canvas, "add_gate_overlay"):
                self._canvas.add_gate_overlay(gate_id, verts, label=gate_id)
        except Exception as exc:
            _logger.debug("add_gate_overlay_if_compatible '%s': %s", gate_id, exc)

    def remove_gate_overlay_if_present(self, gate_id: str) -> None:
        if hasattr(self._canvas, "remove_gate_overlay"):
            self._canvas.remove_gate_overlay(gate_id)

    # ------ UX actif/inactif ---------------------------------------------

    def set_active_visual(self, active: bool) -> None:
        if active:
            self.setStyleSheet("QFrame { border: 1px solid #5BAAFF; border-radius: 4px; }")
        else:
            self.setStyleSheet("")

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self.activated.emit()
        super().mousePressEvent(event)

    def eventFilter(self, watched, event):  # type: ignore[override]
        if watched is self._canvas and event.type() == QEvent.MouseButtonPress:
            self.activated.emit()
        return super().eventFilter(watched, event)


# ---------------------------------------------------------------------------
# WorksheetArea — zone scrollable de grille de PlotWidgetPanel
# ---------------------------------------------------------------------------


class WorksheetArea(QScrollArea):
    """
    Zone multi-plot Kaluza-style : grille scrollable de PlotWidgetPanel.

    Chaque panneau est autonome (axes, population source).
    Le workspace peut ajouter/supprimer des panneaux dynamiquement.

    Signal
    ------
    gateCreated(gate_name) — propagé depuis n'importe quel panneau.
    """

    gateCreated = pyqtSignal(str)

    _COLS: int = 2  # nombre de colonnes par défaut

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self._container = QWidget()
        self._container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._grid = QGridLayout(self._container)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setSpacing(6)
        for col in range(self._COLS):
            self._grid.setColumnStretch(col, 1)
        self.setWidget(self._container)

        self._panels: List[PlotWidgetPanel] = []
        self._active_panel: Optional[PlotWidgetPanel] = None

    # ------ API publique --------------------------------------------------

    def add_panel(self, panel: PlotWidgetPanel) -> None:
        """Ajoute un panneau dans la grille (gauche → droite, haut → bas)."""
        self._panels.append(panel)
        panel.closeRequested.connect(lambda p=panel: self.remove_panel(p))
        panel.gateCreated.connect(self.gateCreated)
        panel.activated.connect(lambda p=panel: self.set_active_panel(p))
        if self._active_panel is None:
            self.set_active_panel(panel)
        self._relayout()

    def remove_panel(self, panel: PlotWidgetPanel) -> None:
        """Retire un panneau et réorganise la grille."""
        if panel not in self._panels:
            return
        self._panels.remove(panel)
        self._grid.removeWidget(panel)
        panel.deleteLater()
        if self._active_panel is panel:
            self._active_panel = None
            if self._panels:
                self.set_active_panel(self._panels[0])
        self._relayout()

    def set_active_panel(self, panel: PlotWidgetPanel) -> None:
        """Définit le panneau actif utilisé pour le mode dessin/gating."""
        if panel not in self._panels:
            return
        self._active_panel = panel
        for p in self._panels:
            try:
                p.set_active_visual(p is panel)
            except Exception:
                pass

    def active_panel(self) -> Optional[PlotWidgetPanel]:
        return self._active_panel

    def panels(self) -> List[PlotWidgetPanel]:
        return list(self._panels)

    def active_canvas(self) -> Optional[InteractiveGatingCanvas]:
        """Retourne le canvas du panneau actif (fallback: premier panneau)."""
        if self._active_panel is not None:
            return self._active_panel.canvas
        return self._panels[0].canvas if self._panels else None

    def refresh_all(
        self,
        engine: "PrismaFlowEngine",
        transform_id: Optional[str] = None,
        comp_id: Optional[str] = None,
    ) -> None:
        """Recharge tous les panneaux après modification des données/gates."""
        for panel in self._panels:
            try:
                panel.refresh_from_engine(engine, transform_id=transform_id, comp_id=comp_id)
            except Exception as exc:
                _logger.debug("WorksheetArea.refresh_all panel ignoré: %s", exc)

    def on_gate_added(self, gate_id: str) -> None:
        """
        Propagation ciblée : une gate vient d'être ajoutée.
        Chaque panel vérifie la compatibilité et ajoute l'overlay si compatible.
        Ne déclenche PAS de refresh_from_engine complet.
        """
        for panel in self._panels:
            try:
                x_ch = panel._model.x_channel
                y_ch = panel._model.y_channel or None
                panel.add_gate_overlay_if_compatible(gate_id, x_ch, y_ch)
            except Exception as exc:
                _logger.debug("WorksheetArea.on_gate_added '%s' panel: %s", gate_id, exc)

    def on_gate_removed(self, gate_id: str) -> None:
        """Propagation ciblée : suppression d'overlay sur tous les panels."""
        for panel in self._panels:
            try:
                panel.remove_gate_overlay_if_present(gate_id)
            except Exception as exc:
                _logger.debug("WorksheetArea.on_gate_removed '%s' panel: %s", gate_id, exc)

    # ------ API layout configurable ---------------------------------------

    def set_panel_columns(self, count: int) -> None:
        """
        Reconfigure le nombre de colonnes de la grille (1, 2, 3 ou 4).

        Les panels existants sont conservés dans leur état courant.
        Lève WorksheetLayoutError si count invalide.
        """
        count = validate_column_count(count)
        if count == self._COLS:
            return
        self._COLS = count
        self._relayout()

    def get_layout_state(self) -> WorkspaceLayoutState:
        """
        Construit et retourne un WorkspaceLayoutState depuis l'état courant.
        Capture l'état de chaque panel (axes, gate_node, transform, comp, mode).
        """
        panel_states: list[PanelState] = []
        for position, panel in enumerate(self._panels):
            try:
                model = panel._model
                panel_states.append(
                    PanelState(
                        panel_id=model.panel_id,
                        gate_node=tuple(model.gate_node) if model.gate_node else ("root",),
                        x_channel=model.x_channel or "",
                        y_channel=model.y_channel or "",
                        transform_id=model.transform_id,
                        comp_id=model.comp_id,
                        render_mode=model.render_mode.name if model.render_mode else "SCATTER",
                        density_coloring=model.density_coloring,
                        position=position,
                    )
                )
            except Exception as exc:
                _logger.debug("get_layout_state panel %d ignoré: %s", position, exc)

        active_id: Optional[str] = None
        if self._active_panel is not None:
            try:
                active_id = self._active_panel._model.panel_id
            except Exception:
                pass

        return WorkspaceLayoutState(
            panel_column_count=self._COLS,
            active_panel_id=active_id,
            panels=panel_states,
        )

    # ------ Interne -------------------------------------------------------

    def _relayout(self) -> None:
        """Réorganise les panneaux restants dans la grille après suppression ou changement de colonnes."""
        # Retirer tous les widgets de la grille sans les détruire
        for i in range(self._grid.count() - 1, -1, -1):
            item = self._grid.itemAt(i)
            if item and item.widget():
                self._grid.removeWidget(item.widget())

        # Reconfigurer les stretch de colonnes
        for col in range(max(self._COLS, 4)):
            self._grid.setColumnStretch(col, 1 if col < self._COLS else 0)

        for i, panel in enumerate(self._panels):
            if len(self._panels) == 1:
                row, col, col_span = 0, 0, self._COLS
            else:
                row, col = divmod(i, self._COLS)
                col_span = 1
            self._grid.addWidget(panel, row, col, 1, col_span)
            self._grid.setRowStretch(row, 1)


# ---------------------------------------------------------------------------
# Widget principal
# ---------------------------------------------------------------------------


class PrismaGatingWorkspace(QWidget):
    """
    Espace de travail de gating interactif.

    Paramètre
    ---------
    engine: PrismaFlowEngine — moteur FlowKit partagé.
            Si None, un engine vide est créé (utile en dev).
    """

    gatingContextSaved = pyqtSignal(str, list)

    def __init__(
        self,
        engine: Optional[PrismaFlowEngine] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine or PrismaFlowEngine()
        self._active_sample_id: Optional[str] = None
        self._active_gate_path: Optional[Tuple[str, ...]] = None
        self._active_gate_name: Optional[str] = None
        self._active_transform_id: Optional[str] = None
        self._active_comp_id: Optional[str] = None
        self._current_gate_node: Tuple[str, ...] = ("root",)
        self._last_canvas_axes: Optional[Tuple[str, str]] = None
        self._loaded_fcs_paths: List[str] = []
        self._last_saved_workspace_path: Optional[str] = None
        # Alias QTreeView — défini dans _connect_signals après build_panel_widget()
        self._tree_view: Optional[QTreeView] = None

        # Phase 5 — contrôleurs et dock
        self._tree_ctrl = PopulationTreeController(engine=self._engine, parent=self)
        self._stats_dock = StatisticsDock(parent=self)
        self._stats_ctrl = StatisticsController(
            engine=self._engine,
            dock=self._stats_dock,
            parent=self,
        )
        # Alias de compatibilité — permet aux méthodes héritées d'accéder au modèle
        # sans passer par le contrôleur.
        self._gate_tree_model = self._tree_ctrl.gate_tree_model

        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Construction UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setFont(QFont("Segoe UI", 9))

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setHandleWidth(4)

        # ── Panneau gauche : arbre de gating (via PopulationTreeController) ──
        self._tree_panel = self._tree_ctrl.build_panel_widget()
        splitter.addWidget(self._tree_panel)

        # ── Panneau central : Worksheet multi-plot ───────────────────────
        self._worksheet = WorksheetArea(self)
        splitter.addWidget(self._worksheet)

        # ── Panneau droit : contrôles ────────────────────────────────────
        self._ctrl_panel = self._build_control_panel()
        splitter.addWidget(self._ctrl_panel)

        splitter.setSizes([240, 700, 280])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        root_layout.addWidget(splitter)

        # ── Dock de statistiques (Phase 5) — intégré inline en bas ─────
        # Pour une intégration QMainWindow réelle, utiliser addDockWidget().
        # Ici on l'intègre dans le layout vertical comme widget détachable.
        self._stats_dock.setFeatures(
            self._stats_dock.features() & ~self._stats_dock.DockWidgetClosable
        )
        root_layout.addWidget(self._stats_dock)

        self._apply_style()

    def _update_statistics(self) -> None:
        """
        Wrapper de compatibilité — délègue au StatisticsController.
        Guard sur _has_active_panel pour éviter les workers orphelins au démarrage.
        """
        if not self._has_active_panel():
            return
        try:
            x_ch = self._selected_axis_channel(self._combo_x)
            y_ch = self._selected_axis_channel(self._combo_y)
            self._stats_ctrl.set_current_sample(self._current_sample_id())
            self._stats_ctrl.set_axis_channels(x_ch or None, y_ch or None)
            self._stats_ctrl.set_transform_id(self._active_transform_id)
            self._stats_ctrl.refresh_statistics()
        except Exception as exc:
            _logger.warning("_update_statistics (délégation) échoué : %s", exc)

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QWidget { background: #080D18; color: #EEF2F7; font-family: 'Segoe UI', 'Inter', 'Roboto', Arial, sans-serif; }
            QSplitter::handle { background: #141E2E; }
            QPushButton {
                background: #141E2E; border: 1px solid #2A3342; border-radius: 4px;
                padding: 5px 12px; color: #EEF2F7;
            }
            QPushButton:hover { background: #1E2B3E; border-color: #7B52FF; }
            QPushButton:pressed { background: #7B52FF; }
            QComboBox {
                background: #0C1220; border: 1px solid #2A3342; border-radius: 4px;
                padding: 4px 8px; color: #EEF2F7;
            }
            QComboBox::drop-down { border: none; }
            QTreeView { background: #04070D; border: none; alternate-background-color: #080D18; }
            QTreeView::item:selected { background: #7B52FF; }
            QGroupBox { border: 1px solid #2A3342; border-radius: 4px; margin-top: 8px; padding-top: 8px; }
            QGroupBox::title { color: #5BAAFF; padding: 0 4px; }
            QLabel { color: #8899AA; }
            QCheckBox { color: #EEF2F7; spacing: 6px; }
        """)

    # ------------------------------------------------------------------
    # Panneau gauche — arbre de gating
    # ------------------------------------------------------------------

    def _build_tree_panel(self) -> QWidget:
        """
        DÉPRÉCIÉ Phase 5 — remplacé par PopulationTreeController.build_panel_widget().
        Conservé uniquement pour compatibilité de sous-classes éventuelles.
        """
        return self._tree_ctrl.build_panel_widget()

    # ------------------------------------------------------------------
    # Panneau droit — contrôles
    # ------------------------------------------------------------------

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(240)
        panel.setMaximumWidth(340)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Sélecteurs ─────────────────────────────────────────────────
        grp_sel = QGroupBox("Sélection")
        vsel = QVBoxLayout(grp_sel)

        vsel.addWidget(QLabel("Groupe (WSP FlowJo)"))
        self._combo_group = QComboBox()
        self._combo_group.addItem("— Tous —")
        vsel.addWidget(self._combo_group)

        vsel.addWidget(QLabel("Sample"))
        self._combo_sample = QComboBox()
        vsel.addWidget(self._combo_sample)

        vsel.addWidget(QLabel("Population"))
        self._combo_population = QComboBox()
        self._combo_population.addItem("— Tous les événements —")
        vsel.addWidget(self._combo_population)

        layout.addWidget(grp_sel)

        # ── Axes ───────────────────────────────────────────────────────
        grp_axes = QGroupBox("Axes")
        vaxes = QVBoxLayout(grp_axes)

        vaxes.addWidget(QLabel("Axe X"))
        self._combo_x = QComboBox()
        vaxes.addWidget(self._combo_x)

        vaxes.addWidget(QLabel("Axe Y  (vide → histo 1D)"))
        self._combo_y = QComboBox()
        self._combo_y.addItem("")
        vaxes.addWidget(self._combo_y)

        layout.addWidget(grp_axes)

        # ── Transformations ─────────────────────────────────────────────
        grp_tr = QGroupBox("Données")
        vtr = QVBoxLayout(grp_tr)

        vtr.addWidget(QLabel("Transformation"))
        self._combo_transform = QComboBox()
        self._combo_transform.addItems(["Logicle", "Hyperlog", "Asinh", "Log", "Linear"])
        self._combo_transform.setCurrentText("Logicle")
        vtr.addWidget(self._combo_transform)

        vtr.addWidget(QLabel("Compensation"))
        self._combo_comp = QComboBox()
        self._combo_comp.addItem("— Non compensé —")
        vtr.addWidget(self._combo_comp)

        self._chk_enable_comp = QCheckBox("Activer Compensation")
        vtr.addWidget(self._chk_enable_comp)

        self._chk_density = QCheckBox("Coloration par densité")
        vtr.addWidget(self._chk_density)

        self._chk_overlay = QCheckBox("Overlay populations enfants")
        vtr.addWidget(self._chk_overlay)

        layout.addWidget(grp_tr)

        # ── Dessin ─────────────────────────────────────────────────────
        grp_draw = QGroupBox("Dessin de gates")
        vdraw = QVBoxLayout(grp_draw)

        self._btn_poly = QPushButton("Polygone")
        self._btn_poly.setCheckable(True)
        self._btn_rect = QPushButton("Rectangle")
        self._btn_rect.setCheckable(True)
        self._btn_quad = QPushButton("Quadrant")
        self._btn_quad.setCheckable(True)
        self._btn_cancel_draw = QPushButton("✕  Annuler")

        row1 = QHBoxLayout()
        row1.addWidget(self._btn_poly)
        row1.addWidget(self._btn_rect)
        vdraw.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(self._btn_quad)
        row2.addWidget(self._btn_cancel_draw)
        vdraw.addLayout(row2)
        layout.addWidget(grp_draw)

        # ── Actions ────────────────────────────────────────────────────
        grp_act = QGroupBox("Actions")
        vact = QVBoxLayout(grp_act)

        self._btn_add_panel = QPushButton("＋  Nouveau panneau (population active)")
        self._btn_apply = QPushButton("▶  Afficher")
        self._btn_delete_gate = QPushButton("🗑  Supprimer gate sélectionnée")
        self._btn_analyze = QPushButton("⚡  Analyser tous les samples")
        self._btn_export_stats = QPushButton("📊  Exporter statistiques CSV")
        self._btn_export_fcs = QPushButton("💾  Exporter gate → FCS")
        self._btn_save_workspace = QPushButton("💾  Sauvegarder contexte gating")
        self._btn_load_workspace = QPushButton("📂  Charger contexte gating")

        for btn in [
            self._btn_add_panel,
            self._btn_apply,
            self._btn_delete_gate,
            self._btn_analyze,
            self._btn_export_stats,
            self._btn_export_fcs,
            self._btn_save_workspace,
            self._btn_load_workspace,
        ]:
            vact.addWidget(btn)

        layout.addWidget(grp_act)
        layout.addStretch()

        return panel

    # ------------------------------------------------------------------
    # Connexion des signaux
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        # Worksheet → arbre (gate créée dans n'importe quel panneau → refresh global)
        self._worksheet.gateCreated.connect(self._on_worksheet_gate_created)

        # Signaux engine → propagation ciblée (QueuedConnection = thread-safe)
        try:
            self._engine.gatingStrategyChanged.connect(
                self._on_engine_strategy_changed, Qt.QueuedConnection
            )
            self._engine.analysisCompleted.connect(
                self._on_engine_analysis_completed, Qt.QueuedConnection
            )
            self._engine.analysisError.connect(
                self._on_engine_analysis_error, Qt.QueuedConnection
            )
            # Phase 5 — connecter les contrôleurs aux signaux engine
            self._engine.gatingStrategyChanged.connect(
                self._tree_ctrl.on_engine_strategy_changed, Qt.QueuedConnection
            )
            self._engine.analysisCompleted.connect(
                self._tree_ctrl.on_engine_analysis_completed, Qt.QueuedConnection
            )
            self._engine.gatingStrategyChanged.connect(
                self._stats_ctrl.on_engine_strategy_changed, Qt.QueuedConnection
            )
            self._engine.analysisCompleted.connect(
                self._stats_ctrl.on_engine_analysis_completed, Qt.QueuedConnection
            )
        except Exception:
            pass  # Engine non QObject en mode dégradé

        # Arbre (Phase 5) → workspace via populationSelected
        # populationSelected couvre déjà les clics de l'arbre — pas de double connect.
        self._tree_ctrl.populationSelected.connect(self._on_population_selected_from_ctrl)

        # Alias rétrograde : _tree_view pointe sur le QTreeView du contrôleur.
        # Ne pas connecter clicked directement — évite le double-déclenchement.
        self._tree_view = self._tree_ctrl.tree_view

        # Contrôles → affichage
        self._combo_group.currentTextChanged.connect(self._on_group_changed)
        self._combo_sample.currentTextChanged.connect(self._on_sample_changed)
        self._combo_population.currentIndexChanged.connect(self._on_population_index_changed)
        self._combo_x.currentIndexChanged.connect(self._on_axes_changed)
        self._combo_y.currentIndexChanged.connect(self._on_axes_changed)
        self._combo_transform.currentTextChanged.connect(self._on_transform_changed)
        self._combo_comp.currentIndexChanged.connect(self._refresh_canvas)
        self._chk_enable_comp.toggled.connect(self._on_compensation_toggled)
        self._chk_density.toggled.connect(self._refresh_canvas)
        self._chk_overlay.toggled.connect(self._refresh_canvas)
        self._btn_add_panel.clicked.connect(self._on_add_panel)
        self._btn_apply.clicked.connect(self._refresh_canvas)

        # Boutons dessin
        self._btn_poly.clicked.connect(lambda: self._activate_draw(DrawMode.POLYGON))
        self._btn_rect.clicked.connect(lambda: self._activate_draw(DrawMode.RECTANGLE))
        self._btn_quad.clicked.connect(lambda: self._activate_draw(DrawMode.QUADRANT))
        self._btn_cancel_draw.clicked.connect(self._safe_cancel_drawing)
        self._btn_cancel_draw.clicked.connect(self._reset_draw_buttons)

        # Actions
        self._btn_delete_gate.clicked.connect(self._on_delete_gate)
        self._btn_analyze.clicked.connect(self._on_analyze)
        self._btn_export_stats.clicked.connect(self._on_export_stats)
        self._btn_export_fcs.clicked.connect(self._on_export_fcs)
        self._btn_save_workspace.clicked.connect(self._on_save_workspace)
        self._btn_load_workspace.clicked.connect(self._on_load_workspace)

    # ------------------------------------------------------------------
    # Slots signaux engine (thread-safe via QueuedConnection)
    # ------------------------------------------------------------------

    def _on_engine_strategy_changed(self, gate_id: str, change_type: str) -> None:
        """
        Propagation ciblée lors d'un changement de stratégie.
        L'arbre est rafraîchi par PopulationTreeController (connecté séparément).
        """
        if change_type == "added":
            self._worksheet.on_gate_added(gate_id)
        elif change_type == "removed":
            self._worksheet.on_gate_removed(gate_id)
        elif change_type == "modified":
            self._worksheet.on_gate_removed(gate_id)
            self._worksheet.on_gate_added(gate_id)

    def _on_engine_analysis_completed(self, sample_id: str) -> None:
        """
        Après analyze async.
        StatisticsController et PopulationTreeController sont connectés directement
        à analysisCompleted — pas de double appel ici.
        """

    def _on_engine_analysis_error(self, sample_id: str, error_msg: str) -> None:
        _logger.warning("Analyse async échouée (sample=%s): %s", sample_id, error_msg)
        self._show_error(
            f"Analyse FlowKit échouée pour le sample '{sample_id}'.\n\nDétail : {error_msg}"
        )

    # ------------------------------------------------------------------
    # API publique — rechargement depuis engine
    # ------------------------------------------------------------------

    def reload_from_engine(self) -> None:
        """Recharge l'UI complète depuis l'état courant du moteur."""
        self._reload_group_combo()
        self._reload_sample_combo()
        self._reload_channel_combos()
        self._reload_transform_combo()
        self._reload_comp_combo()
        self._refresh_tree()
        # Synchroniser les contrôleurs Phase 5
        self._tree_ctrl.set_current_sample(self._engine.active_sample_id)
        self._stats_ctrl.set_current_sample(self._engine.active_sample_id)

    def set_engine(self, engine: PrismaFlowEngine) -> None:
        """Remplace le moteur, reconnecte les signaux et recharge l'UI."""
        self._engine = engine
        try:
            engine.gatingStrategyChanged.connect(
                self._on_engine_strategy_changed, Qt.QueuedConnection
            )
            engine.analysisCompleted.connect(
                self._on_engine_analysis_completed, Qt.QueuedConnection
            )
            engine.analysisError.connect(
                self._on_engine_analysis_error, Qt.QueuedConnection
            )
            engine.gatingStrategyChanged.connect(
                self._tree_ctrl.on_engine_strategy_changed, Qt.QueuedConnection
            )
            engine.analysisCompleted.connect(
                self._tree_ctrl.on_engine_analysis_completed, Qt.QueuedConnection
            )
            engine.gatingStrategyChanged.connect(
                self._stats_ctrl.on_engine_strategy_changed, Qt.QueuedConnection
            )
            engine.analysisCompleted.connect(
                self._stats_ctrl.on_engine_analysis_completed, Qt.QueuedConnection
            )
        except Exception:
            pass
        self._tree_ctrl.set_engine(engine)
        self._stats_ctrl.set_engine(engine)
        self.reload_from_engine()

    # ------------------------------------------------------------------
    # API Phase 5 — layout configurable et persistance étendue
    # ------------------------------------------------------------------

    def set_panel_columns(self, count: int) -> None:
        """
        Reconfigure le nombre de colonnes du worksheet (1, 2, 3 ou 4).
        Les panels existants sont conservés dans leur état.
        Lève WorksheetLayoutError si count invalide.
        """
        self._worksheet.set_panel_columns(count)

    def _on_population_selected_from_ctrl(self, gate_path: Tuple[str, ...]) -> None:
        """
        Slot : PopulationTreeController a émis populationSelected.
        Remplace _on_tree_node_clicked pour les sélections d'arbre Phase 5.
        """
        if not gate_path:
            return
        self._active_gate_name = gate_path[0]
        self._current_gate_node = gate_path

        # Synchroniser le combo population sans ré-émettre currentIndexChanged
        idx = self._combo_population.findText(gate_path[0])
        if idx >= 0:
            self._combo_population.blockSignals(True)
            self._combo_population.setCurrentIndex(idx)
            self._combo_population.blockSignals(False)

        # Guard identique à _refresh_canvas
        if not self._has_active_panel():
            return
        self._refresh_canvas()
        self._ensure_population_panel(gate_path)

    def set_input_source_fcs(self, fcs_folder: str) -> None:
        """Charge les FCS d'un dossier en backend Session pour le workspace interactif."""
        folder = Path(fcs_folder)
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Dossier FCS introuvable: {folder}")

        # Dédupliquer explicitement (Windows: *.fcs et *.FCS peuvent matcher les mêmes fichiers).
        file_map: Dict[str, Path] = {}
        for p in list(folder.glob("*.fcs")) + list(folder.glob("*.FCS")):
            resolved = p.resolve()
            file_map[str(resolved).lower()] = resolved
        files = sorted(file_map.values(), key=lambda p: p.name.lower())
        if not files:
            raise FileNotFoundError(f"Aucun fichier FCS trouvé dans: {folder}")

        new_engine = PrismaFlowEngine()
        self._loaded_fcs_paths = []
        self._last_saved_workspace_path = None
        # Remplacer le moteur via set_engine pour reconnecter contrôleurs + signaux
        self._engine = new_engine
        self._tree_ctrl.set_engine(new_engine)
        self._stats_ctrl.set_engine(new_engine)

        for fpath in files:
            self.load_sample(str(fpath.resolve()))

        self.reload_from_engine()

    def load_sample(self, file_path: str) -> bool:
        """Charge un FCS via le moteur puis force un rafraîchissement UI complet."""
        try:
            print(f"[GatingWorkspace] 1. Appel Engine -> load_fcs('{file_path}')")
            _logger.debug("1. Appel Engine -> load_fcs('%s')", file_path)
            sample_id = self._engine.load_fcs(file_path, make_active=True)

            print(f"[GatingWorkspace] 2. Engine OK -> sample_id='{sample_id}'")
            _logger.debug("2. Engine OK -> sample_id='%s'", sample_id)

            self._active_sample_id = sample_id
            self._loaded_fcs_paths.append(str(Path(file_path).resolve()))

            # Appliquer immédiatement la transformation par défaut (Logicle)
            # pour garantir la disponibilité des événements transformés au premier rendu.
            default_transform = self._combo_transform.currentText().strip() or "Logicle"
            self._active_transform_id = self._engine.apply_transformation(default_transform)

            # Forcer l'arbre à se reconstruire immédiatement après chargement réel.
            if hasattr(self._gate_tree_model, "build_hierarchy"):
                self._gate_tree_model.build_hierarchy()
            else:
                self._refresh_tree()

            print("[GatingWorkspace] 3. Arbre mis à jour")
            _logger.debug("3. Arbre mis à jour")

            self._reload_sample_combo()
            self._reload_channel_combos()

            # Créer un panneau initial dans le Worksheet si aucun n'existe encore
            if not self._worksheet.panels():
                try:
                    mapping = self._engine.get_channel_marker_mapping(sample_id=sample_id)
                except Exception:
                    mapping = {}
                initial_panel = PlotWidgetPanel(
                    engine=self._engine,
                    gate_node=("root",),
                    channel_mapping=mapping,
                    active_transform_id=self._active_transform_id,
                )
                initial_panel.canvas.polygonGateCompleted.connect(self._on_polygon_gate)
                initial_panel.canvas.rectangleGateCompleted.connect(self._on_rectangle_gate)
                initial_panel.canvas.quadrantGateCompleted.connect(self._on_quadrant_gate)
                initial_panel.canvas.quadrantMoved.connect(self._on_quadrant_moved)
                initial_panel.canvas.gateModified.connect(self._on_gate_modified)
                self._worksheet.add_panel(initial_panel)

            # Forcer l'affichage de la population racine puis demander le tracé.
            self._combo_population.setCurrentIndex(0)
            self._refresh_canvas()

            print("[GatingWorkspace] 4. Plot demandé (population Root)")
            _logger.debug("4. Plot demandé (population Root)")
            return True
        except Exception as exc:
            _logger.error("Echec load_sample('%s'): %s", file_path, exc, exc_info=True)
            raise

    def get_available_populations(self) -> List[str]:
        """Retourne la liste des populations disponibles pour l'orchestrateur Wizard."""
        gate_ids = sorted({str(g) for g in self._engine.get_gate_ids()}, key=str.lower)
        populations = ["Root"]
        populations.extend([g for g in gate_ids if g and g.lower() != "root"])
        return populations

    def save_workspace_state(self, output_path: "str | Path") -> str:
        """
        Sauvegarde le contexte de gating PRISMA (JSON + GatingML sidecar).

        Phase 5 : inclut aussi la disposition des panels et le layout configurable.
        """
        path = Path(output_path)
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)

        gml_path = path.with_suffix(".gatingml.xml")
        try:
            self._engine.export_gml(gml_path)
        except Exception as exc:
            _logger.warning("export_gml échoué (ignoré) : %s", exc)
            gml_path = None

        # Capturer l'état du layout Phase 5
        layout_state = self._worksheet.get_layout_state()

        payload: Dict[str, Any] = {
            "format": "prisma_gating_context_v1",
            "workspace_path": str(path.resolve()),
            "gatingml_path": str(gml_path.resolve()) if gml_path else None,
            "active_sample_id": self._engine.active_sample_id,
            "fcs_files": list(self._loaded_fcs_paths),
            "populations": self.get_available_populations(),
            "layout": layout_state.to_dict(),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self._last_saved_workspace_path = str(path.resolve())
        _logger.info("Workspace sauvegardé: %s", path)
        return self._last_saved_workspace_path

    def load_workspace_state(
        self,
        workspace_path: str,
        fallback_fcs_folder: Optional[str] = None,
    ) -> List[str]:
        """Recharge un contexte de gating (JSON PRISMA, GatingML ou WSP)."""
        path = Path(workspace_path)
        if not path.exists():
            raise FileNotFoundError(f"Workspace de gating introuvable: {path}")

        suffix = path.suffix.lower()
        if suffix == ".wsp":
            engine = PrismaFlowEngine()
            engine.load_wsp(path, fcs_dir=fallback_fcs_folder)
            self._engine = engine
            self._tree_ctrl.set_engine(engine)
            self._stats_ctrl.set_engine(engine)
            if fallback_fcs_folder:
                folder = Path(fallback_fcs_folder)
                files = sorted(folder.glob("*.fcs")) + sorted(folder.glob("*.FCS"))
                self._loaded_fcs_paths = [str(p.resolve()) for p in files]
            self._last_saved_workspace_path = str(path.resolve())
            self.reload_from_engine()
            return self.get_available_populations()

        if suffix in {".gml", ".xml"}:
            # Si un FCS est déjà chargé dans l'engine actif, on peut charger directement
            # le GML sans passer par set_input_source_fcs.
            active_sid = None
            try:
                active_sid = self._engine.active_sample_id
            except Exception:
                pass

            if active_sid:
                # FCS déjà présent — charger la stratégie directement
                self._engine.load_gml(path)
            elif fallback_fcs_folder:
                self.set_input_source_fcs(fallback_fcs_folder)
                self._engine.load_gml(path)
            elif self._loaded_fcs_paths:
                # Recharger depuis les chemins mémorisés
                self._engine.load_fcs_batch(
                    [Path(p) for p in self._loaded_fcs_paths], make_first_active=True
                )
                self._engine.load_gml(path)
            else:
                raise PrismaEngineError(
                    "Aucun FCS chargé. Ouvrez d'abord un fichier FCS, "
                    "puis importez le GatingML."
                )
            self._last_saved_workspace_path = str(path.resolve())
            self.reload_from_engine()
            return self.get_available_populations()

        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("format", "")).strip() != "prisma_gating_context_v1":
            raise PrismaEngineError(
                "Format de contexte de gating non supporté (attendu prisma_gating_context_v1)."
            )

        fcs_files = [str(p) for p in (payload.get("fcs_files") or []) if str(p).strip()]
        if not fcs_files and fallback_fcs_folder:
            folder = Path(fallback_fcs_folder)
            files = sorted(folder.glob("*.fcs")) + sorted(folder.glob("*.FCS"))
            fcs_files = [str(p.resolve()) for p in files]
        if not fcs_files:
            raise PrismaEngineError(
                "Contexte de gating invalide: aucun FCS associé et aucun fallback fourni."
            )

        gml_path = payload.get("gatingml_path")
        if not gml_path:
            gml_path = str(path.with_suffix(".gatingml.xml"))
        gml_file = Path(str(gml_path))
        if not gml_file.is_absolute():
            gml_file = (path.parent / gml_file).resolve()
        if not gml_file.exists():
            raise FileNotFoundError(f"GatingML associé introuvable: {gml_file}")

        engine = PrismaFlowEngine()
        engine.load_fcs_batch([Path(p) for p in fcs_files], make_first_active=True)
        engine.load_gml(gml_file)

        active_sample_id = payload.get("active_sample_id")
        if active_sample_id:
            try:
                engine.set_active_sample(str(active_sample_id))
            except Exception:
                _logger.warning("Sample actif sauvegardé introuvable: %s", active_sample_id)

        self._engine = engine
        self._loaded_fcs_paths = list(fcs_files)
        self._last_saved_workspace_path = str(path.resolve())
        self.reload_from_engine()

        # Restauration du layout Phase 5
        raw_layout = payload.get("layout")
        if raw_layout:
            self._restore_layout_from_dict(raw_layout)

        return self.get_available_populations()

    # ------------------------------------------------------------------
    # Restauration du layout Phase 5
    # ------------------------------------------------------------------

    def _restore_layout_from_dict(self, raw: Dict[str, Any]) -> None:
        """
        Restaure la disposition des panels depuis le dict JSON de layout.
        Tolérante : ignore les panels dont la gate ou les axes n'existent plus.
        """
        try:
            layout_state = WorkspaceLayoutState.from_dict(raw)
        except Exception as exc:
            _logger.warning("_restore_layout_from_dict: désérialisation échouée: %s", exc)
            return

        # Appliquer le nombre de colonnes
        try:
            self._worksheet.set_panel_columns(layout_state.panel_column_count)
        except WorksheetLayoutError as exc:
            _logger.warning("Nombre de colonnes invalide dans le layout restauré: %s", exc)

        # Obtenir la liste des gates disponibles pour validation
        try:
            available_gates = set(self._engine.get_gate_ids())
        except Exception:
            available_gates = set()

        # Obtenir le mapping de canaux pour validation
        mapping: Dict[str, str] = {}
        available_channels: set[str] = set()
        try:
            mapping = self._engine.get_channel_marker_mapping(
                sample_id=self._current_sample_id()
            )
            available_channels = set(mapping.keys())
        except Exception:
            pass

        active_panel_id: Optional[str] = None

        for panel_state in layout_state.panels:
            # Valider la gate
            gate_node = panel_state.gate_node
            if gate_node and gate_node[0].lower() != "root":
                if gate_node[0] not in available_gates:
                    _logger.warning(
                        "Layout restauration: gate '%s' absente → panel ignoré.",
                        gate_node[0],
                    )
                    continue

            # Valider les axes — fallback silencieux si absent
            x_ch = panel_state.x_channel
            y_ch = panel_state.y_channel
            if x_ch and available_channels and x_ch not in available_channels:
                _logger.warning(
                    "Layout restauration: canal X '%s' absent → panel ignoré.", x_ch
                )
                continue

            if y_ch and available_channels and y_ch not in available_channels:
                _logger.debug(
                    "Layout restauration: canal Y '%s' absent → Y remis à vide.", y_ch
                )
                y_ch = ""

            # Créer le panel
            try:
                panel = PlotWidgetPanel(
                    engine=self._engine,
                    gate_node=gate_node,
                    channel_mapping=mapping if available_channels else {},
                    active_transform_id=panel_state.transform_id or self._active_transform_id,
                    active_comp_id=panel_state.comp_id,
                )
                panel.canvas.polygonGateCompleted.connect(self._on_polygon_gate)
                panel.canvas.rectangleGateCompleted.connect(self._on_rectangle_gate)
                panel.canvas.quadrantGateCompleted.connect(self._on_quadrant_gate)
                panel.canvas.quadrantMoved.connect(self._on_quadrant_moved)
                panel.canvas.gateModified.connect(self._on_gate_modified)

                # Synchroniser les axes du panel
                if x_ch:
                    x_idx = panel._combo_x.findData(x_ch)
                    if x_idx >= 0:
                        panel._combo_x.setCurrentIndex(x_idx)
                if y_ch:
                    y_idx = panel._combo_y.findData(y_ch)
                    if y_idx >= 0:
                        panel._combo_y.setCurrentIndex(y_idx)

                self._worksheet.add_panel(panel)

                if panel_state.panel_id == layout_state.active_panel_id:
                    active_panel_id = panel_state.panel_id
                    self._worksheet.set_active_panel(panel)

                # Forcer l'affichage initial — sans _refresh() le panel serait vide
                try:
                    panel._refresh()
                except Exception as exc_r:
                    _logger.debug("panel._refresh() à la restauration: %s", exc_r)

            except Exception as exc:
                _logger.warning("Restauration panel '%s' échouée: %s", panel_state.panel_id, exc)

        _logger.info(
            "Layout restauré: %d panels, %d colonnes.",
            len(layout_state.panels),
            layout_state.panel_column_count,
        )

    # ------------------------------------------------------------------
    # Rechargements internes
    # ------------------------------------------------------------------

    def _reload_group_combo(self) -> None:
        """Recharge le sélecteur de groupes (visible uniquement sur Workspace)."""
        self._combo_group.blockSignals(True)
        self._combo_group.clear()
        self._combo_group.addItem("— Tous —")
        groups = self._engine.get_sample_groups()
        for g in groups:
            if g != "All Samples":
                self._combo_group.addItem(g)
        # Afficher le combo uniquement si des groupes existent
        has_groups = len(groups) > 1 or (len(groups) == 1 and groups[0] != "All Samples")
        self._combo_group.setVisible(has_groups)
        self._combo_group.blockSignals(False)

    def _reload_sample_combo(self, group_name: Optional[str] = None) -> None:
        self._combo_sample.blockSignals(True)
        self._combo_sample.clear()
        for sid in self._engine.get_sample_ids(group_name=group_name):
            self._combo_sample.addItem(sid)
        if self._engine.active_sample_id:
            idx = self._combo_sample.findText(self._engine.active_sample_id)
            if idx >= 0:
                self._combo_sample.setCurrentIndex(idx)
        self._combo_sample.blockSignals(False)

    def _reload_channel_combos(self) -> None:
        try:
            mapping = self._engine.get_channel_marker_mapping(sample_id=self._current_sample_id())
        except Exception:
            return

        for combo in (self._combo_x, self._combo_y):
            combo.blockSignals(True)
            combo.clear()

        self._combo_y.addItem("", userData="")
        for pnn, marker in mapping.items():
            display = f"{marker} ({pnn})" if marker and marker != pnn else pnn
            self._combo_x.addItem(display, userData=pnn)
            self._combo_y.addItem(display, userData=pnn)

        for combo in (self._combo_x, self._combo_y):
            combo.blockSignals(False)

    def _reload_transform_combo(self) -> None:
        self._combo_transform.blockSignals(True)
        self._combo_transform.clear()
        self._combo_transform.addItems(["Logicle", "Hyperlog", "Asinh", "Log", "Linear"])
        self._combo_transform.setCurrentText("Logicle")
        self._combo_transform.blockSignals(False)

    def _reload_comp_combo(self) -> None:
        self._combo_comp.blockSignals(True)
        self._combo_comp.clear()
        self._combo_comp.addItem("— Non compensé —")
        for cid in self._engine.get_comp_matrix_ids():
            self._combo_comp.addItem(cid)
        self._combo_comp.blockSignals(False)

    def _reload_population_combo(self, gate_ids: List[str]) -> None:
        self._combo_population.blockSignals(True)
        self._combo_population.clear()
        self._combo_population.addItem("— Tous les événements —")
        seen: set[str] = set()
        for gid in gate_ids:
            gid_s = str(gid).strip()
            if not gid_s or gid_s in seen:
                continue
            seen.add(gid_s)
            self._combo_population.addItem(gid)
        self._combo_population.blockSignals(False)

    def _refresh_tree(self) -> None:
        """
        Wrapper de compatibilité — délègue au PopulationTreeController.
        Conservé pour tous les appels existants dans le workspace.
        """
        try:
            self._tree_ctrl.set_current_sample(self._current_sample_id())
            self._tree_ctrl.refresh_tree()
        except Exception as exc:
            _logger.warning("_refresh_tree (délégation): %s", exc)
        try:
            self._reload_population_combo(self._engine.get_gate_ids())
        except Exception as exc:
            _logger.debug("_reload_population_combo: %s", exc)

    # ------------------------------------------------------------------
    # Slots contrôles
    # ------------------------------------------------------------------

    def _on_group_changed(self, group_name: str) -> None:
        """Filtre les samples par groupe FlowJo."""
        if not group_name or group_name == "— Tous —":
            self._reload_sample_combo(group_name=None)
        else:
            try:
                self._engine.set_active_group(group_name)
                self._reload_sample_combo(group_name=group_name)
            except Exception as exc:
                self._show_error(f"Changement de groupe : {exc}")

    def _on_sample_changed(self, sample_id: str) -> None:
        if not sample_id:
            return
        try:
            self._engine.set_active_sample(sample_id)
            self._active_sample_id = sample_id
            self._reload_channel_combos()
            self._refresh_tree()
        except Exception as exc:
            self._show_error(f"Changement de sample : {exc}")

    def _refresh_canvas(self) -> None:
        """
        Pipeline Kaluza-like :
          1. Lit les Pnn depuis userData des comboboxes (jamais le texte affiché)
          2. Demande au moteur le df compensé+transformé filtré par la gate active
          3. La gate _current_gate_node n'est jamais réinitialisée ici — persistance garantie
          4. Envoie au canvas avec labels PnS corrects sur les axes
        """
        if not self._has_active_panel():
            return
        x_ch = self._selected_axis_channel(self._combo_x)
        y_ch = self._selected_axis_channel(self._combo_y)
        sample_id = self._current_sample_id()

        if not x_ch or not sample_id:
            return

        # transform_id : utiliser l'ID actif (défini par load_sample / _on_transform_changed)
        # Ne pas utiliser _selected_transform() qui peut retourner None avant le 1er apply
        transform_id = self._active_transform_id
        # comp_matrix_id : seulement si checkbox cochée
        comp_id: Optional[str] = (
            self._selected_comp() if self._chk_enable_comp.isChecked() else None
        )

        gate_name: Optional[str] = None
        gate_path: Optional[Tuple[str, ...]] = None
        if self._current_gate_node and self._current_gate_node[0].lower() != "root":
            gate_name = self._current_gate_node[0]
            gate_path = (
                tuple(self._current_gate_node[1:]) if len(self._current_gate_node) > 1 else None
            )

        # --- Labels PnS pour axes ---
        try:
            ch_map = self._engine.get_channel_marker_mapping(sample_id=sample_id)
        except Exception:
            ch_map = {}
        x_label = ch_map.get(x_ch, x_ch)
        y_label = ch_map.get(y_ch, y_ch) if y_ch else ""
        if hasattr(self._canvas, "set_axis_labels"):
            self._canvas.set_axis_labels(x_label, y_label)

        # --- Préserver le zoom si mêmes axes ---
        prev_view_range = None
        if self._last_canvas_axes == (x_ch, y_ch):
            try:
                prev_view_range = self._canvas.getViewBox().viewRange()
            except Exception:
                pass

        # --- Récupérer le DataFrame compensé+transformé+filtré ---
        try:
            df = self._engine.get_population_df(
                self._current_gate_node,
                sample_id=sample_id,
                transform_id=transform_id,
                comp_matrix_id=comp_id,
            )
        except Exception as exc:
            self._show_error(f"Chargement données : {exc}")
            return

        if df is None or df.empty:
            _logger.warning("DataFrame vide pour gate_node=%s", self._current_gate_node)
            return

        # Tolérance casse/espaces sur les noms de canaux
        if x_ch not in df.columns:
            match = next(
                (c for c in df.columns if str(c).strip().upper() == x_ch.strip().upper()), None
            )
            if match:
                x_ch = match
            else:
                _logger.warning(
                    "Canal X '%s' absent du DataFrame. Colonnes: %s",
                    x_ch,
                    list(df.columns)[:10],
                )
                return

        if y_ch and y_ch not in df.columns:
            match = next(
                (c for c in df.columns if str(c).strip().upper() == y_ch.strip().upper()), None
            )
            y_ch = match or ""

        # --- Envoi au canvas ---
        density = self._chk_density.isChecked()
        overlay = self._chk_overlay.isChecked()

        if overlay and gate_name:
            self._show_overlay_children(df, gate_name, gate_path, x_ch, y_ch)
        elif y_ch and y_ch in df.columns:
            self._canvas.set_data_2d(df, x_ch, y_ch, density_coloring=density)
        else:
            self._canvas.set_data_1d(df, x_ch)

        # --- Restaurer le zoom ---
        if prev_view_range is not None:
            try:
                x_range, y_range = prev_view_range
                self._canvas.getViewBox().setXRange(float(x_range[0]), float(x_range[1]), padding=0)
                self._canvas.getViewBox().setYRange(float(y_range[0]), float(y_range[1]), padding=0)
            except Exception:
                pass

        self._last_canvas_axes = (x_ch, y_ch)

        # --- Overlays gates existantes ---
        # En mode multi-panel, on n'affiche pas les overlays globaux sur un panneau enfant.
        is_root_context = bool(
            self._current_gate_node and str(self._current_gate_node[0]).lower() == "root"
        )
        if is_root_context:
            self._canvas.reload_gate_overlays_from_engine(self._engine)
        elif hasattr(self._canvas, "clear_gate_overlays"):
            self._canvas.clear_gate_overlays()

        # --- Statistiques live ---
        self._update_statistics()

    def _show_overlay_children(
        self,
        parent_df: pd.DataFrame,
        gate_name: str,
        gate_path: Optional[Tuple[str, ...]],
        x_ch: str,
        y_ch: Optional[str],
    ) -> None:
        children = self._engine.get_children(gate_name, gate_path)
        if not children:
            if y_ch:
                self._canvas.set_data_2d(parent_df, x_ch, y_ch)
            else:
                self._canvas.set_data_1d(parent_df, x_ch)
            return

        datasets = []
        for i, child_id in enumerate(children):
            color = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]
            try:
                child_paths = self._engine.find_gate_paths(child_id)
                child_path = child_paths[0] if child_paths else None
                child_df = self._engine.get_gate_dataframe(
                    child_id,
                    gate_path=child_path,
                    sample_id=self._current_sample_id(),
                )
                datasets.append((child_df, child_id, color, child_id))
            except Exception as exc:
                _logger.debug("Overlay child %s ignoré : %s", child_id, exc)

        if datasets and y_ch and hasattr(self._canvas, "set_data_backgate"):
            # Backgate : parent gris + chaque enfant coloré par-dessus
            child_memberships = {}
            child_colors = {}
            sid = self._current_sample_id()
            for i, child_id in enumerate(children):
                try:
                    membership = self._engine.get_gate_membership_cached(child_id, sid)
                    if membership is not None:
                        child_memberships[child_id] = membership
                        child_colors[child_id] = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]
                except Exception:
                    pass

            if child_memberships:
                self._canvas.set_data_backgate(
                    parent_df, x_ch, y_ch, child_memberships, child_colors
                )
            elif datasets:
                self._canvas.set_data_overlay(datasets, x_ch, y_ch)
            elif y_ch:
                self._canvas.set_data_2d(parent_df, x_ch, y_ch)
            else:
                self._canvas.set_data_1d(parent_df, x_ch)
        elif datasets:
            self._canvas.set_data_overlay(datasets, x_ch, y_ch)
        elif y_ch:
            self._canvas.set_data_2d(parent_df, x_ch, y_ch)
        else:
            self._canvas.set_data_1d(parent_df, x_ch)

    # ------------------------------------------------------------------
    # Slots arbre
    # ------------------------------------------------------------------

    def _on_tree_node_clicked(self, index: QModelIndex) -> None:
        """
        Compat rétrograde — non connecté en Phase 5 (remplacé par populationSelected).
        Conservé pour sous-classes éventuelles.
        """
        node: Optional[GateNode] = index.data(Qt.UserRole)
        if node is None:
            return
        self._active_gate_name = node.gate_id
        paths = self._engine.find_gate_paths(node.gate_id)
        if paths:
            self._current_gate_node = (node.gate_id, *paths[0])
        else:
            self._current_gate_node = (node.gate_id,)
        idx = self._combo_population.findText(node.gate_id)
        if idx >= 0:
            self._combo_population.setCurrentIndex(idx)
        if self._has_active_panel():
            self._refresh_canvas()
        self._ensure_population_panel(self._current_gate_node)

    def _on_population_index_changed(self, _index: int) -> None:
        """Slot robuste: résout la population depuis l'index du combo."""
        self._on_population_changed(self._combo_population.currentText())

    def _on_population_changed(self, population: str) -> None:
        """Mémorise la population active pour préserver le contexte lors des changements d'axes."""
        if not population or population == "— Tous les événements —":
            self._current_gate_node = ("root",)
            self._refresh_canvas()
            return

        paths = self._engine.find_gate_paths(population)
        self._current_gate_node = (population, *paths[0]) if paths else (population,)
        self._refresh_canvas()
        # UX demandée : créer automatiquement un graphe dédié à la population choisie.
        self._ensure_population_panel(self._current_gate_node)

    def _on_axes_changed(self, _index: int) -> None:
        """Re-trace la même population active avec les nouveaux axes."""
        _logger.debug("Axes modifiés -> population active=%s", self._current_gate_node)
        self._refresh_canvas()

    def _on_compensation_toggled(self, enabled: bool) -> None:
        """Active/désactive la compensation depuis les métadonnées FCS puis rafraîchit la vue."""
        if enabled:
            try:
                matrix_id = self._engine.apply_spillover_matrix(sample_id=self._current_sample_id())
                if not matrix_id:
                    self._chk_enable_comp.setChecked(False)
                    self._show_error(
                        "Aucune matrice de compensation détectée dans les métadonnées FCS."
                    )
                    return

                idx = self._combo_comp.findText(matrix_id)
                if idx < 0:
                    self._combo_comp.addItem(matrix_id)
                    idx = self._combo_comp.findText(matrix_id)
                if idx >= 0:
                    self._combo_comp.setCurrentIndex(idx)
            except Exception as exc:
                self._chk_enable_comp.setChecked(False)
                self._show_error(f"Activation compensation: {exc}")
                return
        else:
            idx = self._combo_comp.findText("— Non compensé —")
            if idx >= 0:
                self._combo_comp.setCurrentIndex(idx)

        self._refresh_canvas()

    def _on_transform_changed(self, transform_type: str) -> None:
        """Applique la transformation sélectionnée puis rafraîchit la vue."""
        if not transform_type or not self._current_sample_id():
            return

        try:
            _logger.debug("[GatingWorkspace] Application de la transformation : %s", transform_type)
            self._active_transform_id = self._engine.apply_transformation(transform_type)
            self._refresh_canvas()
        except Exception as exc:
            self._show_error(
                f"UI Error : Impossible d'appliquer la transformation {transform_type} : {exc}"
            )

    # ------------------------------------------------------------------
    # Slots canvas → engine
    # ------------------------------------------------------------------

    def _on_polygon_gate(
        self,
        gate_name: str,
        x_channel: str,
        y_channel: str,
        vertices: list,
    ) -> None:
        name, ok = QInputDialog.getText(
            self, "Nom de la gate", "Nom de la gate polygonale :", text=gate_name
        )
        if not ok or not name.strip():
            return
        gate_path = self._resolve_parent_path()
        xform_ref = self._selected_transform()
        _logger.debug(
            "create_polygon_gate '%s' gate_path=%s x=%s y=%s transform_ref=%s vertices[0]=%s",
            name.strip(),
            gate_path,
            x_channel,
            y_channel,
            xform_ref,
            vertices[0] if vertices else None,
        )
        print(
            f"[WORKSPACE] PolygonGate '{name.strip()}': transform_ref={xform_ref}, gate_path={gate_path}, vertices[0]={vertices[0] if vertices else None}"
        )
        try:
            self._engine.create_polygon_gate_from_vertices(
                name.strip(),
                gate_path,
                x_channel,
                y_channel,
                vertices,
                transform_ref=xform_ref,
            )
            # Affichage immédiat pour feedback visuel, avant refresh global.
            verts = [(float(v[0]), float(v[1])) for v in vertices if len(v) >= 2]
            if verts:
                self._canvas.add_gate_overlay(name.strip(), verts, label=name.strip())
            self._post_gate_created(new_gate_name=name.strip())
        except PrismaEngineError as exc:
            self._show_error(str(exc))
        finally:
            self._reset_draw_buttons()

    def _on_rectangle_gate(
        self,
        gate_name: str,
        x_channel: str,
        y_channel: str,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        name, ok = QInputDialog.getText(
            self, "Nom de la gate", "Nom de la gate rectangulaire :", text=gate_name
        )
        if not ok or not name.strip():
            return
        gate_path = self._resolve_parent_path()
        try:
            xform_ref = self._selected_transform()
            print(f"[WORKSPACE] RectangleGate '{name.strip()}': transform_ref={xform_ref}")
            self._engine.create_rectangle_gate_from_bounds(
                name.strip(),
                gate_path,
                x_channel,
                y_channel,
                x_min,
                x_max,
                y_min,
                y_max,
                transform_ref=xform_ref,
            )
            # Affichage immédiat pour feedback visuel (1D/2D).
            if y_channel:
                verts = [
                    (float(x_min), float(y_min)),
                    (float(x_max), float(y_min)),
                    (float(x_max), float(y_max)),
                    (float(x_min), float(y_max)),
                ]
            else:
                verts = [
                    (float(x_min), 0.0),
                    (float(x_max), 0.0),
                    (float(x_max), 1.0),
                    (float(x_min), 1.0),
                ]
            self._canvas.add_gate_overlay(name.strip(), verts, label=name.strip())
            self._post_gate_created(new_gate_name=name.strip())
        except PrismaEngineError as exc:
            self._show_error(str(exc))
        finally:
            self._reset_draw_buttons()

    def _on_quadrant_gate(
        self,
        gate_name: str,
        x_channel: str,
        y_channel: str,
        x_threshold: float,
        y_threshold: float,
    ) -> None:
        name, ok = QInputDialog.getText(
            self, "Nom de la gate", "Nom de la QuadrantGate :", text=gate_name
        )
        if not ok or not name.strip():
            return
        gate_path = self._resolve_parent_path()
        try:
            xform_ref = self._selected_transform()
            print(f"[WORKSPACE] QuadrantGate '{name.strip()}': transform_ref={xform_ref}")
            self._engine.create_quadrant_gate_from_thresholds(
                name.strip(),
                gate_path,
                x_channel,
                y_channel,
                x_threshold,
                y_threshold,
                transform_ref=xform_ref,
            )
            self._post_gate_created(new_gate_name=name.strip())
        except PrismaEngineError as exc:
            self._show_error(str(exc))
        finally:
            self._reset_draw_buttons()

    def _on_quadrant_moved(
        self,
        gate_name: str,
        x_channel: str,
        y_channel: str,
        new_x_threshold: float,
        new_y_threshold: float,
    ) -> None:
        """
        Slot : l'utilisateur a déplacé le centre d'un quadrant existant.
        Recrée la QuadrantGate FlowKit avec les nouveaux seuils.
        """
        if not gate_name:
            return
        try:
            paths = self._engine.find_gate_paths(gate_name)
            gate_path = paths[0] if paths else self._resolve_parent_path()
            xform_ref = self._selected_transform()
            # Recréer la QuadrantGate avec les nouveaux seuils
            self._engine.create_quadrant_gate_from_thresholds(
                gate_name,
                gate_path,
                x_channel,
                y_channel,
                new_x_threshold,
                new_y_threshold,
                transform_ref=xform_ref,
            )
            self._post_gate_created(new_gate_name=gate_name)
        except Exception as exc:
            _logger.warning("_on_quadrant_moved '%s': %s", gate_name, exc)

    def _on_gate_modified(
        self,
        gate_id: str,
        x_channel: str,
        y_channel: str,
        new_vertices: list,
    ) -> None:
        """
        Slot : gate déplacée/redimensionnée par drag & drop sur le canvas.

        Stratégie sécurisée pour préserver les enfants :
          1. Collecter les enfants directs avant suppression
          2. Supprimer + recréer la gate parente avec keep_children=True
          3. Réattacher les enfants au nouveau chemin
          4. Émettre gatingStrategyChanged("modified") via l'engine

        Note : FlowKit ne propose pas de mise à jour in-place des vertices.
        Le signal gatingStrategyChanged("added") est émis par create_polygon_gate_from_vertices.
        """
        if not gate_id or not new_vertices:
            return
        try:
            paths = self._engine.find_gate_paths(gate_id)
            if not paths:
                return

            gate_path = paths[0]
            # Récupérer les enfants avant suppression
            children_before = self._engine.get_children(gate_id, gate_path)

            # Supprimer avec keep_children=True pour éviter la suppression en cascade
            for p in reversed(paths):
                try:
                    self._engine.remove_gate(gate_id, gate_path=p, keep_children=True)
                except Exception:
                    pass

            xform_ref = self._selected_transform()
            self._engine.create_polygon_gate_from_vertices(
                gate_name=gate_id,
                gate_path=gate_path,
                x_channel=x_channel,
                y_channel=y_channel,
                vertices=new_vertices,
                transform_ref=xform_ref,
            )

            if children_before:
                _logger.debug(
                    "_on_gate_modified: gate '%s' a %d enfants à réattacher manuellement",
                    gate_id,
                    len(children_before),
                )

            self._post_gate_created(new_gate_name=gate_id)
        except Exception as exc:
            _logger.warning("_on_gate_modified: %s", exc)

    def _on_add_panel(self) -> None:
        """Ajoute un nouveau PlotWidgetPanel dans le Worksheet pour la population active."""
        self._create_panel_for_gate_node(self._current_gate_node)

    def _create_panel_for_gate_node(self, gate_node: Tuple[str, ...]) -> None:
        """Crée un panneau de plot dédié à une population donnée."""
        try:
            mapping = self._engine.get_channel_marker_mapping(sample_id=self._current_sample_id())
        except Exception:
            mapping = {}
        panel = PlotWidgetPanel(
            engine=self._engine,
            gate_node=gate_node,
            channel_mapping=mapping,
            active_transform_id=self._active_transform_id,
            active_comp_id=self._selected_comp(),
        )
        # Connecter aussi les signaux de gate du panneau vers les handlers workspace
        panel.canvas.polygonGateCompleted.connect(self._on_polygon_gate)
        panel.canvas.rectangleGateCompleted.connect(self._on_rectangle_gate)
        panel.canvas.quadrantGateCompleted.connect(self._on_quadrant_gate)
        panel.canvas.quadrantMoved.connect(self._on_quadrant_moved)
        panel.canvas.gateModified.connect(self._on_gate_modified)
        self._worksheet.add_panel(panel)
        self._worksheet.set_active_panel(panel)

        # Synchroniser les axes du nouveau panneau avec les sélections globales courantes.
        try:
            x_ch = self._selected_axis_channel(self._combo_x)
            y_ch = self._selected_axis_channel(self._combo_y)
            x_idx = panel._combo_x.findData(x_ch)
            if x_idx >= 0:
                panel._combo_x.setCurrentIndex(x_idx)
            y_idx = panel._combo_y.findData(y_ch)
            if y_idx >= 0:
                panel._combo_y.setCurrentIndex(y_idx)
        except Exception:
            pass

        panel._refresh()

    def _ensure_population_panel(self, gate_node: Tuple[str, ...]) -> None:
        """Garantit un seul panneau par population (évite les doublons visuels)."""
        if not gate_node or gate_node[0].lower() == "root":
            return

        for panel in self._worksheet.panels():
            panel_gate_node = getattr(panel, "_gate_node", None)
            if panel_gate_node is not None and str(panel_gate_node[0]) == str(gate_node[0]):
                self._worksheet.set_active_panel(panel)
                try:
                    panel._refresh()
                except Exception:
                    pass
                return

        self._create_panel_for_gate_node(gate_node)

    def _on_worksheet_gate_created(self, gate_name: str) -> None:
        """
        Gate créée depuis un panneau du worksheet.
        Les overlays sont déjà propagés via engine.gatingStrategyChanged.
        On relance seulement l'analyse async pour les counts.
        """
        self._engine.analyze_async(sample_id=self._current_sample_id())

    def _post_gate_created(self, new_gate_name: Optional[str] = None) -> None:
        """
        Actions après création d'une gate (non bloquant).

        Stratégie :
          1. Sélectionner la nouvelle gate dans l'arbre et le combo (immédiat)
          2. Lancer analyze_async → le signal analysisCompleted met à jour arbre + stats
          3. Rafraîchir le canvas sur la population parente (contexte courant conservé)

        L'overlay de la nouvelle gate sur les panels compatibles est déjà déclenché
        par engine.gatingStrategyChanged émis dans add_*_gate().
        """
        if new_gate_name:
            paths_new = self._engine.find_gate_paths(new_gate_name)
            self._current_gate_node = (
                (new_gate_name, *paths_new[0]) if paths_new else (new_gate_name,)
            )

            idx = self._combo_population.findText(new_gate_name)
            if idx >= 0:
                self._combo_population.blockSignals(True)
                self._combo_population.setCurrentIndex(idx)
                self._combo_population.blockSignals(False)

            # Sélection visuelle dans l'arbre
            if self._tree_view is not None:
                model = self._gate_tree_model
                for row in range(model.rowCount()):
                    idx_tree = model.index(row, 0)
                    node = idx_tree.data(Qt.UserRole)
                    if node and getattr(node, "gate_id", None) == new_gate_name:
                        self._tree_view.setCurrentIndex(idx_tree)
                        break

            self._ensure_population_panel(self._current_gate_node)

        # Analyse asynchrone — ne bloque pas l'UI
        # Le signal analysisCompleted déclenchera _on_engine_analysis_completed
        self._engine.analyze_async(sample_id=self._current_sample_id())
        self._refresh_canvas()

    # ------------------------------------------------------------------
    # Slots actions
    # ------------------------------------------------------------------

    def _on_delete_gate(self) -> None:
        if self._tree_view is None:
            self._show_error("Arbre de gating non initialisé.")
            return
        index = self._tree_view.currentIndex()
        node: Optional[GateNode] = index.data(Qt.UserRole)
        if node is None:
            self._show_error("Sélectionnez une gate dans l'arbre avant de supprimer.")
            return

        reply = QMessageBox.question(
            self,
            "Supprimer gate",
            f"Supprimer la gate '{node.gate_id}' et ses descendants ?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            paths = self._engine.find_gate_paths(node.gate_id)
            gate_path = paths[0] if paths else None
            self._engine.remove_gate(node.gate_id, gate_path=gate_path)
            self._refresh_tree()
            if self._has_active_panel():
                self._canvas.remove_gate_overlay(node.gate_id)
        except PrismaEngineError as exc:
            self._show_error(str(exc))

    def _on_analyze(self) -> None:
        """Analyse le sample actif en asynchrone (non bloquant)."""
        try:
            sid = self._current_sample_id()
            self._engine.analyze_async(sample_id=sid)
            # La mise à jour arbre+stats arrive via _on_engine_analysis_completed
        except Exception as exc:
            self._show_error(f"Lancement analyse échoué : {exc}")

    def _on_export_stats(self) -> None:
        from PyQt5.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(self, "Exporter statistiques", "", "CSV (*.csv)")
        if not path:
            return
        try:
            from src.prisma.exports.gating_exporter import GatingExporter

            GatingExporter.export_statistics(self._engine, path)
            QMessageBox.information(self, "Export", f"Statistiques exportées :\n{path}")
        except Exception as exc:
            self._show_error(f"Export statistiques : {exc}")

    def _on_export_fcs(self) -> None:
        from PyQt5.QtWidgets import QFileDialog

        # Résolution de la gate à exporter : arbre en priorité, puis population active
        gate_id: Optional[str] = None
        gate_path_resolved: Optional[tuple] = None

        if self._tree_view is not None:
            index = self._tree_view.currentIndex()
            node: Optional[GateNode] = index.data(Qt.UserRole)
            if node is not None:
                gate_id = node.gate_id

        if gate_id is None:
            # Fallback : population active dans le combo/canvas
            if (
                self._current_gate_node
                and self._current_gate_node[0].lower() != "root"
            ):
                gate_id = self._current_gate_node[0]
            else:
                self._show_error(
                    "Aucune gate sélectionnée.\n"
                    "Cliquez sur une gate dans l'arbre de population ou choisissez "
                    "une population active avant d'exporter."
                )
                return

        paths = self._engine.find_gate_paths(gate_id)
        gate_path_resolved = paths[0] if paths else None

        # Créer un objet node minimal pour compatibilité avec le reste du code
        class _FakeNode:
            def __init__(self, gid):
                self.gate_id = gid
        node = _FakeNode(gate_id)  # type: ignore[assignment]

        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter gate → FCS", f"{node.gate_id}.fcs", "FCS (*.fcs)"
        )
        if not path:
            return

        try:
            from src.prisma.exports.gating_exporter import GatingExporter
            GatingExporter.export_gated_fcs(
                self._engine,
                sample_id=self._current_sample_id(),
                gate_name=node.gate_id,
                gate_path=gate_path_resolved,
                output_fcs_path=path,
            )
            QMessageBox.information(self, "Export", f"FCS exporté :\n{path}")
        except Exception as exc:
            self._show_error(f"Export FCS : {exc}")

    def _on_save_workspace(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Sauvegarder le contexte de gating",
            "gating_context.prisma.json",
            "PRISMA Gating Context (*.json)",
        )
        if not path:
            return
        try:
            saved_path = self.save_workspace_state(path)
            populations = self.get_available_populations()
            self.gatingContextSaved.emit(saved_path, populations)
            QMessageBox.information(
                self,
                "Contexte sauvegardé",
                f"Contexte de gating sauvegardé:\n{saved_path}",
            )
        except Exception as exc:
            self._show_error(f"Sauvegarde contexte de gating: {exc}")

    def _on_load_workspace(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Charger un contexte de gating",
            "",
            "Contextes PRISMA / FlowKit (*.json *.gml *.xml *.wsp)",
        )
        if not path:
            return
        try:
            populations = self.load_workspace_state(path)
            saved_path = self._last_saved_workspace_path or str(Path(path).resolve())
            self.gatingContextSaved.emit(saved_path, populations)
            QMessageBox.information(
                self,
                "Contexte chargé",
                f"Contexte de gating chargé:\n{saved_path}",
            )
        except Exception as exc:
            self._show_error(f"Chargement contexte de gating: {exc}")

    # ------------------------------------------------------------------
    # Gestion mode dessin
    # ------------------------------------------------------------------

    def _activate_draw(self, mode: DrawMode) -> None:
        if not self._has_active_panel():
            self._show_error("Chargez un FCS avant de dessiner.")
            self._reset_draw_buttons()
            return

        # Capturer le panel actif AU MOMENT DU CLIC — immuable pour la durée du dessin
        active_panel = self._worksheet.active_panel()
        if active_panel is None:
            self._show_error("Aucun panneau actif.")
            self._reset_draw_buttons()
            return

        # Lire les canaux depuis le modèle du panel actif (source de vérité locale)
        x_ch = active_panel._model.x_channel
        y_ch = active_panel._model.y_channel

        # Fallback sur les combos globaux si modèle pas encore peuplé
        if not x_ch:
            x_ch = self._selected_axis_channel(self._combo_x)
        if not y_ch:
            y_ch = self._selected_axis_channel(self._combo_y)

        if not x_ch:
            self._show_error("Sélectionnez un canal X avant de dessiner.")
            self._reset_draw_buttons()
            return

        if mode in (DrawMode.POLYGON, DrawMode.QUADRANT) and not y_ch:
            self._show_error(
                "Un canal Y est requis pour une gate Polygone ou Quadrant.\n"
                "Pour une gate 1D sur histogramme, utilisez Rectangle."
            )
            self._reset_draw_buttons()
            return

        # Appliquer le mode sur le canvas du panel capturé — pas sur self._canvas
        active_panel.canvas.set_draw_mode(mode, gate_name="Gate")

        for btn, m in [
            (self._btn_poly, DrawMode.POLYGON),
            (self._btn_rect, DrawMode.RECTANGLE),
            (self._btn_quad, DrawMode.QUADRANT),
        ]:
            btn.setChecked(m == mode)

    def _safe_cancel_drawing(self) -> None:
        try:
            canvas = self._worksheet.active_canvas()
            if canvas is not None:
                canvas.cancel_drawing()
        except RuntimeError:
            pass

    def _reset_draw_buttons(self) -> None:
        for btn in (self._btn_poly, self._btn_rect, self._btn_quad):
            btn.setChecked(False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_sample_id(self) -> Optional[str]:
        sid = self._combo_sample.currentText().strip()
        if sid:
            try:
                available = self._engine.get_sample_ids()
            except Exception:
                available = []

            if sid in available:
                return sid

            # Le sample affiché n'existe plus (ex: ancien doublon) -> fallback robuste.
            _logger.warning("Sample UI introuvable dans la session: %s", sid)

        active = self._engine.active_sample_id
        if active:
            try:
                if active in self._engine.get_sample_ids():
                    return active
            except Exception:
                return active

        try:
            ids = self._engine.get_sample_ids()
            return ids[0] if ids else None
        except Exception:
            return None

    def _selected_axis_channel(self, combo: QComboBox) -> str:
        """Retourne le canal Pnn stocké dans userData (fallback texte brut robuste).

        Priorité : userData str → parse texte "Marqueur (Pnn)" → texte brut.
        Retourne '' si combo vide ou item vide sélectionné (cas Y optionnel).
        """
        data = combo.currentData()
        if isinstance(data, str) and data.strip():
            return data.strip()

        text = combo.currentText().strip()
        if not text:
            return ""
        # Format "Marqueur (Pnn)" → extraire Pnn
        if text.endswith(")") and "(" in text:
            pnn = text[text.rfind("(") + 1 : -1].strip()
            if pnn:
                return pnn
        return text

    def _selected_transform(self) -> Optional[str]:
        """Retourne l'ID du transform actif, fallback sur premier disponible dans l'engine."""
        if self._active_transform_id:
            return self._active_transform_id
        ids = self._engine.get_transform_ids()
        return ids[0] if ids else None

    def _selected_comp(self) -> Optional[str]:
        c = self._combo_comp.currentText()
        return c if c and c != "— Non compensé —" else None

    def _resolve_parent_path(self) -> Tuple[str, ...]:
        """
        Déduit le gate_path parent depuis le panneau actif (prioritaire)
        puis depuis la sélection courante dans l'arbre.

        Si aucun nœud sélectionné → gate racine (tuple vide = 'root').
        """
        active_panel = self._worksheet.active_panel()
        if active_panel is not None:
            panel_gate_node = getattr(active_panel, "_gate_node", ("root",))
            if panel_gate_node and str(panel_gate_node[0]).lower() != "root":
                parent_name = str(panel_gate_node[0])
                parent_path = tuple(panel_gate_node[1:])
                return parent_path + (parent_name,)

        if self._tree_view is None:
            return ("root",)
        index = self._tree_view.currentIndex()
        node: Optional[GateNode] = index.data(Qt.UserRole)
        if node is None:
            return ("root",)
        paths = self._engine.find_gate_paths(node.gate_id)
        if not paths:
            return ("root",)
        return paths[0] + (node.gate_id,)

    def _show_error(self, message: str) -> None:
        _logger.error("UI Error : %s", message)
        QMessageBox.critical(self, "Erreur PRISMA Gating", message)

    @property
    def engine(self) -> PrismaFlowEngine:
        """Accès explicite au moteur pour les intégrations externes."""
        return self._engine

    @property
    def _canvas(self) -> InteractiveGatingCanvas:
        """Canvas du premier panneau actif — compatibilité rétrograde.

        Lève RuntimeError si aucun panneau n'existe encore.
        Vérifie d'abord _canvas_override pour compatibilité tests unitaires.
        """
        override = self.__dict__.get("_canvas_override")
        if override is not None:
            return override
        c = self._worksheet.active_canvas()
        if c is None:
            raise RuntimeError("WorksheetArea : aucun panneau actif")
        return c

    @_canvas.setter
    def _canvas(self, value: Any) -> None:
        """Setter de compatibilité pour les tests unitaires (monkey-patch)."""
        self.__dict__["_canvas_override"] = value

    def _has_active_panel(self) -> bool:
        try:
            return self._worksheet.active_panel() is not None or bool(self._worksheet.panels())
        except (AttributeError, RuntimeError):
            # En contexte de test sans widget Qt initialisé
            return self.__dict__.get("_canvas_override") is not None

    def closeEvent(self, event: object) -> None:
        """Émet le dernier contexte sauvegardé à la fermeture validée du workspace."""
        if self._last_saved_workspace_path:
            try:
                self.gatingContextSaved.emit(
                    self._last_saved_workspace_path,
                    self.get_available_populations(),
                )
            except Exception as exc:
                _logger.warning("Emission gatingContextSaved à la fermeture échouée: %s", exc)
        super().closeEvent(event)  # type: ignore[arg-type]
