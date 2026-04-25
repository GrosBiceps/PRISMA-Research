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

import pandas as pd
from PyQt5.QtCore import Qt, QModelIndex, pyqtSignal
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
)
from PyQt5.QtGui import QFont

from src.utils.logger import get_logger
from .gating_engine import (
    PrismaFlowEngine,
    PrismaEngineError,
    GateHierarchyNode,
)
from .gating_tree_model import GatingTreeModel, GateNode

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

    Contient :
      - un InteractiveGatingCanvas propre
      - ses propres QComboBox X et Y
      - un QLabel indiquant la population source
      - un bouton de fermeture

    Signaux
    -------
    gateCreated(gate_name)  — émis après création d'une gate dans ce panneau.
    closeRequested()        — l'utilisateur clique le bouton ×.
    """

    gateCreated = pyqtSignal(str)
    closeRequested = pyqtSignal()

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
        self._gate_node = gate_node
        self._active_transform_id = active_transform_id
        self._active_comp_id = active_comp_id
        self._channel_mapping = channel_mapping  # {pnn: marker}

        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(320, 280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._build_ui()
        self._connect_signals()

    # ------ Construction UI -----------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        # Barre titre : label source + bouton ×
        title_row = QHBoxLayout()
        pop_label = str(self._gate_node[0]) if self._gate_node else "Root"
        self._lbl_source = QLabel(f"Population : {pop_label}")
        self._lbl_source.setStyleSheet("color: #5BAAFF; font-size: 10px; font-weight: bold;")
        title_row.addWidget(self._lbl_source)
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
        xform_row.addWidget(QLabel("Transform:"))
        self._combo_transform = QComboBox()
        self._combo_transform.addItems(["Logicle", "Hyperlog", "Asinh", "Log", "Linear"])
        self._combo_transform.setCurrentText("Logicle")
        self._combo_transform.setMaximumWidth(110)
        xform_row.addWidget(self._combo_transform)
        xform_row.addStretch()
        layout.addLayout(xform_row)

        # Canvas
        if _INTERACTIVE_CANVAS_AVAILABLE:
            self._canvas = InteractiveGatingCanvas(self)
        else:
            self._canvas = InteractiveGatingCanvas(self)  # fallback stub
        layout.addWidget(self._canvas)

        # Remplir les combos
        self._combo_y.addItem("", userData="")
        for pnn, marker in self._channel_mapping.items():
            display = f"{marker} ({pnn})" if marker and marker != pnn else pnn
            self._combo_x.addItem(display, userData=pnn)
            self._combo_y.addItem(display, userData=pnn)

    def _connect_signals(self) -> None:
        self._btn_close.clicked.connect(self.closeRequested)
        self._combo_x.currentIndexChanged.connect(self._refresh)
        self._combo_y.currentIndexChanged.connect(self._refresh)
        self._combo_transform.currentTextChanged.connect(self._on_transform_changed)
        self._canvas.polygonGateCompleted.connect(self._on_gate_signal)
        self._canvas.rectangleGateCompleted.connect(self._on_gate_signal_rect)
        self._canvas.quadrantGateCompleted.connect(self._on_gate_signal_quad)

    # ------ Transformation locale ----------------------------------------

    def _on_transform_changed(self, transform_type: str) -> None:
        if not transform_type:
            return
        try:
            self._active_transform_id = self._engine.apply_transformation(transform_type)
        except Exception as exc:
            _logger.warning("PlotWidgetPanel transform: %s", exc)
        self._refresh()

    # ------ Refresh -------------------------------------------------------

    def _get_axis(self, combo: QComboBox) -> str:
        data = combo.currentData()
        if isinstance(data, str) and data.strip():
            return data.strip()
        text = combo.currentText().strip()
        if text.endswith(")") and "(" in text:
            return text[text.rfind("(") + 1 : -1].strip()
        return text

    def _refresh(self) -> None:
        x_ch = self._get_axis(self._combo_x)
        y_ch = self._get_axis(self._combo_y)
        if not x_ch:
            return
        try:
            df = self._engine.get_population_df(
                self._gate_node,
                transform_id=self._active_transform_id,
                comp_matrix_id=self._active_comp_id,
            )
        except Exception as exc:
            _logger.warning("PlotWidgetPanel._refresh: %s", exc)
            return
        if df is None or df.empty:
            return

        ch_map = self._channel_mapping
        x_label = ch_map.get(x_ch, x_ch)
        y_label = ch_map.get(y_ch, y_ch) if y_ch else ""
        if hasattr(self._canvas, "set_axis_labels"):
            self._canvas.set_axis_labels(x_label, y_label)

        if y_ch and y_ch in df.columns:
            self._canvas.set_data_2d(df, x_ch, y_ch)
        elif x_ch in df.columns:
            self._canvas.set_data_1d(df, x_ch)

    def refresh_from_engine(
        self,
        engine: "PrismaFlowEngine",
        transform_id: Optional[str] = None,
        comp_id: Optional[str] = None,
    ) -> None:
        """Recharge les données depuis le moteur mis à jour (ex: après nouvelle gate)."""
        self._engine = engine
        if transform_id is not None:
            self._active_transform_id = transform_id
        if comp_id is not None:
            self._active_comp_id = comp_id
        self._refresh()
        if hasattr(self._canvas, "reload_gate_overlays_from_engine"):
            self._canvas.reload_gate_overlays_from_engine(engine)

    # ------ Slots gates ---------------------------------------------------

    def _on_gate_signal(self, gate_name: str, *_args) -> None:
        self.gateCreated.emit(gate_name)

    def _on_gate_signal_rect(self, gate_name: str, *_args) -> None:
        self.gateCreated.emit(gate_name)

    def _on_gate_signal_quad(self, gate_name: str, *_args) -> None:
        self.gateCreated.emit(gate_name)

    # ------ Propriété canvas public (compatibilité workspace) -------------

    @property
    def canvas(self) -> InteractiveGatingCanvas:
        return self._canvas


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
        self._grid = QGridLayout(self._container)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setSpacing(6)
        self.setWidget(self._container)

        self._panels: List[PlotWidgetPanel] = []

    # ------ API publique --------------------------------------------------

    def add_panel(self, panel: PlotWidgetPanel) -> None:
        """Ajoute un panneau dans la grille (gauche → droite, haut → bas)."""
        idx = len(self._panels)
        row, col = divmod(idx, self._COLS)
        self._grid.addWidget(panel, row, col)
        self._panels.append(panel)
        panel.closeRequested.connect(lambda p=panel: self.remove_panel(p))
        panel.gateCreated.connect(self.gateCreated)

    def remove_panel(self, panel: PlotWidgetPanel) -> None:
        """Retire un panneau et réorganise la grille."""
        if panel not in self._panels:
            return
        self._panels.remove(panel)
        self._grid.removeWidget(panel)
        panel.deleteLater()
        self._relayout()

    def panels(self) -> List[PlotWidgetPanel]:
        return list(self._panels)

    def active_canvas(self) -> Optional[InteractiveGatingCanvas]:
        """Retourne le canvas du premier panneau (compatibilité avec l'API workspace)."""
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

    # ------ Interne -------------------------------------------------------

    def _relayout(self) -> None:
        """Réorganise les panneaux restants dans la grille après suppression."""
        for i, panel in enumerate(self._panels):
            row, col = divmod(i, self._COLS)
            self._grid.addWidget(panel, row, col)


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

        # ── Panneau gauche : arbre de gating ────────────────────────────
        self._tree_panel = self._build_tree_panel()
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

        # ── Tableau statistiques live (Kaluza-style) ────────────────────
        self._stats_table = self._build_stats_table()
        root_layout.addWidget(self._stats_table)

        self._apply_style()

    def _build_stats_table(self) -> QTableWidget:
        cols = ["Population", "Count", "% Parent", "% Total", "MFI X", "MFI Y"]
        table = QTableWidget(0, len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.setMaximumHeight(160)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, len(cols)):
            table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        table.setStyleSheet(
            "QTableWidget { background: #04070D; color: #EEF2F7; border: none; font-size: 10px; }"
            "QHeaderView::section { background: #0C1220; color: #5BAAFF; border: none; padding: 2px 6px; }"
            "QTableWidget::item:alternate { background: #080D18; }"
        )
        return table

    def _update_statistics(self) -> None:
        """Remplit le QTableWidget avec les stats FlowKit de toutes les gates du sample actif."""
        try:
            sample_id = self._current_sample_id()
            if not sample_id:
                return

            x_ch = self._selected_axis_channel(self._combo_x)
            y_ch = self._selected_axis_channel(self._combo_y)

            gate_ids = self._engine.get_gate_ids()
            if not gate_ids:
                self._stats_table.setRowCount(0)
                return

            # Tenter get_analysis_report (FlowKit ≥ 1.0)
            report_df: Optional[pd.DataFrame] = None
            try:
                report_df = self._engine.session.get_analysis_report(sample_id=sample_id)
            except Exception:
                pass

            rows = []
            for gid in gate_ids:
                count = 0
                pct_parent = 0.0
                pct_total = 0.0
                mfi_x = ""
                mfi_y = ""

                try:
                    if report_df is not None and not report_df.empty:
                        row = report_df[report_df["gate_name"] == gid]
                        if not row.empty:
                            count = int(row["count"].iloc[0]) if "count" in row.columns else 0
                            pct_parent = float(row["percent_of_parent"].iloc[0]) if "percent_of_parent" in row.columns else 0.0
                            pct_total = float(row["percent_of_total"].iloc[0]) if "percent_of_total" in row.columns else 0.0
                    else:
                        # Fallback manuel : DataFrame de la population
                        paths = self._engine.find_gate_paths(gid)
                        gate_path = paths[0] if paths else None
                        gdf = self._engine.get_gate_dataframe(gid, gate_path=gate_path, sample_id=sample_id)
                        if gdf is not None:
                            count = len(gdf)
                            try:
                                total_df = self._engine.get_population_df(("root",), sample_id=sample_id)
                                total = len(total_df) if total_df is not None else 0
                                pct_total = round(100.0 * count / total, 2) if total > 0 else 0.0
                            except Exception:
                                pass
                except Exception as exc:
                    _logger.debug("Stats gate '%s' ignorée : %s", gid, exc)

                try:
                    paths = self._engine.find_gate_paths(gid)
                    gate_path = paths[0] if paths else None
                    gdf = self._engine.get_gate_dataframe(gid, gate_path=gate_path, sample_id=sample_id,
                                                          transform_id=self._active_transform_id)
                    if gdf is not None and not gdf.empty:
                        if x_ch and x_ch in gdf.columns:
                            mfi_x = f"{float(gdf[x_ch].median()):.1f}"
                        if y_ch and y_ch in gdf.columns:
                            mfi_y = f"{float(gdf[y_ch].median()):.1f}"
                        if count == 0:
                            count = len(gdf)
                except Exception:
                    pass

                rows.append((gid, count, f"{pct_parent:.1f}", f"{pct_total:.1f}", mfi_x, mfi_y))

            self._stats_table.setRowCount(len(rows))
            for r, (pop, cnt, pp, pt, mx, my) in enumerate(rows):
                for c, val in enumerate([pop, str(cnt), pp, pt, mx, my]):
                    item = QTableWidgetItem(str(val))
                    item.setTextAlignment(Qt.AlignCenter)
                    self._stats_table.setItem(r, c, item)

        except Exception as exc:
            _logger.warning("_update_statistics échoué : %s", exc)

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
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        lbl = QLabel("Hiérarchie de gating")
        lbl.setStyleSheet("color: #5BAAFF; font-weight: bold; font-size: 12px;")
        layout.addWidget(lbl)

        self._gate_tree_model = GatingTreeModel()
        self._tree_view = QTreeView()
        self._tree_view.setModel(self._gate_tree_model)
        self._tree_view.setAlternatingRowColors(True)
        self._tree_view.setExpandsOnDoubleClick(True)
        self._tree_view.setUniformRowHeights(True)
        self._tree_view.header().setStretchLastSection(False)
        self._tree_view.header().resizeSection(0, 140)
        self._tree_view.header().resizeSection(1, 60)
        self._tree_view.header().resizeSection(2, 55)
        self._tree_view.header().resizeSection(3, 60)
        layout.addWidget(self._tree_view)

        btn_refresh = QPushButton("↺  Actualiser hiérarchie")
        btn_refresh.clicked.connect(self._refresh_tree)
        layout.addWidget(btn_refresh)

        panel.setMinimumWidth(200)
        panel.setMaximumWidth(320)
        return panel

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

        # Arbre → affichage
        self._tree_view.clicked.connect(self._on_tree_node_clicked)

        # Contrôles → affichage
        self._combo_group.currentTextChanged.connect(self._on_group_changed)
        self._combo_sample.currentTextChanged.connect(self._on_sample_changed)
        self._combo_population.currentTextChanged.connect(self._on_population_changed)
        self._combo_x.currentIndexChanged.connect(self._on_axes_changed)
        self._combo_y.currentIndexChanged.connect(self._on_axes_changed)
        self._combo_transform.currentTextChanged.connect(self._on_transform_changed)
        self._combo_comp.currentIndexChanged.connect(self._refresh_canvas)
        self._chk_enable_comp.toggled.connect(self._on_compensation_toggled)
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

    def set_engine(self, engine: PrismaFlowEngine) -> None:
        """Remplace le moteur et recharge l'UI."""
        self._engine = engine
        self.reload_from_engine()

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

        self._engine = PrismaFlowEngine()
        self._loaded_fcs_paths = []
        self._last_saved_workspace_path = None

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

    def save_workspace_state(self, output_path: str) -> str:
        """
        Sauvegarde un contexte de gating PRISMA réutilisable par le Wizard/Executor.

        Le fichier JSON référence un GatingML sidecar exporté depuis la Session.
        """
        path = Path(output_path)
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)

        gml_path = path.with_suffix(".gatingml.xml")
        self._engine.export_gml(gml_path)

        payload = {
            "format": "prisma_gating_context_v1",
            "workspace_path": str(path.resolve()),
            "gatingml_path": str(gml_path.resolve()),
            "active_sample_id": self._engine.active_sample_id,
            "fcs_files": list(self._loaded_fcs_paths),
            "populations": self.get_available_populations(),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self._last_saved_workspace_path = str(path.resolve())
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
            if fallback_fcs_folder:
                folder = Path(fallback_fcs_folder)
                files = sorted(folder.glob("*.fcs")) + sorted(folder.glob("*.FCS"))
                self._loaded_fcs_paths = [str(p.resolve()) for p in files]
            self._last_saved_workspace_path = str(path.resolve())
            self.reload_from_engine()
            return self.get_available_populations()

        if suffix in {".gml", ".xml"}:
            if not fallback_fcs_folder:
                raise PrismaEngineError("Un dossier FCS actif est requis pour charger un GatingML.")
            self.set_input_source_fcs(fallback_fcs_folder)
            self._engine.load_gml(path)
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
        return self.get_available_populations()

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
        for gid in gate_ids:
            self._combo_population.addItem(gid)
        self._combo_population.blockSignals(False)

    def _refresh_tree(self) -> None:
        """Recharge l'arbre de gating depuis le moteur."""
        self._gate_tree_model.clear()
        try:
            sample_id = self._current_sample_id()
            roots = self._engine.build_hierarchy(sample_id=sample_id)
            self._populate_tree(roots, parent_id=None)
        except Exception as exc:
            _logger.warning("Refresh arbre : %s", exc)

        self._tree_view.expandAll()
        self._reload_population_combo(self._engine.get_gate_ids())

    def _populate_tree(
        self,
        nodes: List[GateHierarchyNode],
        parent_id: Optional[str],
    ) -> None:
        for node in nodes:
            gate_node = GateNode(
                gate_id=node.gate_name,
                name=node.gate_name,
                parent_id=parent_id,
                gate_type=node.gate_type,
            )
            gate_node.mask = None  # comptes déjà dans node.count
            # Hack : stocker le count dans un attribut custom
            gate_node._fk_count = node.count  # type: ignore[attr-defined]
            self._gate_tree_model.add_gate(gate_node)
            if node.children:
                self._populate_tree(node.children, parent_id=node.gate_name)

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
        comp_id: Optional[str] = self._selected_comp() if self._chk_enable_comp.isChecked() else None

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
                    x_ch, list(df.columns)[:10],
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
        self._canvas.reload_gate_overlays_from_engine(self._engine)

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

        if datasets:
            self._canvas.set_data_overlay(datasets, x_ch, y_ch)
        elif y_ch:
            self._canvas.set_data_2d(parent_df, x_ch, y_ch)
        else:
            self._canvas.set_data_1d(parent_df, x_ch)

    # ------------------------------------------------------------------
    # Slots arbre
    # ------------------------------------------------------------------

    def _on_tree_node_clicked(self, index: QModelIndex) -> None:
        node: Optional[GateNode] = index.data(Qt.UserRole)
        if node is None:
            return
        self._active_gate_name = node.gate_id
        paths = self._engine.find_gate_paths(node.gate_id)
        if paths:
            self._current_gate_node = (node.gate_id, *paths[0])
        else:
            self._current_gate_node = (node.gate_id,)
        # Mettre à jour le combo population
        idx = self._combo_population.findText(node.gate_id)
        if idx >= 0:
            self._combo_population.setCurrentIndex(idx)
        self._refresh_canvas()

    def _on_population_changed(self, population: str) -> None:
        """Mémorise la population active pour préserver le contexte lors des changements d'axes."""
        if not population or population == "— Tous les événements —":
            self._current_gate_node = ("root",)
            self._refresh_canvas()
            return

        paths = self._engine.find_gate_paths(population)
        self._current_gate_node = (population, *paths[0]) if paths else (population,)
        self._refresh_canvas()

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
            name.strip(), gate_path, x_channel, y_channel, xform_ref,
            vertices[0] if vertices else None,
        )
        print(f"[WORKSPACE] PolygonGate '{name.strip()}': transform_ref={xform_ref}, gate_path={gate_path}, vertices[0]={vertices[0] if vertices else None}")
        try:
            self._engine.create_polygon_gate_from_vertices(
                name.strip(),
                gate_path,
                x_channel,
                y_channel,
                vertices,
                transform_ref=xform_ref,
            )
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

    def _on_gate_modified(
        self,
        gate_id: str,
        x_channel: str,
        y_channel: str,
        new_vertices: list,
    ) -> None:
        """
        Slot : gate déplacée/redimensionnée par drag & drop sur le canvas.
        Stratégie : supprimer l'ancienne gate et recréer avec les nouvelles coordonnées.
        """
        if not gate_id or not new_vertices:
            return
        try:
            paths = self._engine.find_gate_paths(gate_id)
            gate_path = paths[0] if paths else None
            self._engine.remove_gate(gate_id, gate_path=gate_path)
            xform_ref = self._selected_transform()
            self._engine.create_polygon_gate_from_vertices(
                gate_name=gate_id,
                gate_path=gate_path,
                x_channel=x_channel,
                y_channel=y_channel,
                vertices=new_vertices,
                transform_ref=xform_ref,
            )
            self._post_gate_created(new_gate_name=gate_id)
        except Exception as exc:
            _logger.warning("_on_gate_modified: %s", exc)

    def _on_add_panel(self) -> None:
        """Ajoute un nouveau PlotWidgetPanel dans le Worksheet pour la population active."""
        try:
            mapping = self._engine.get_channel_marker_mapping(
                sample_id=self._current_sample_id()
            )
        except Exception:
            mapping = {}
        panel = PlotWidgetPanel(
            engine=self._engine,
            gate_node=self._current_gate_node,
            channel_mapping=mapping,
            active_transform_id=self._active_transform_id,
            active_comp_id=self._selected_comp(),
        )
        # Connecter aussi les signaux de gate du panneau vers les handlers workspace
        panel.canvas.polygonGateCompleted.connect(self._on_polygon_gate)
        panel.canvas.rectangleGateCompleted.connect(self._on_rectangle_gate)
        panel.canvas.quadrantGateCompleted.connect(self._on_quadrant_gate)
        panel.canvas.gateModified.connect(self._on_gate_modified)
        self._worksheet.add_panel(panel)
        panel._refresh()

    def _on_worksheet_gate_created(self, gate_name: str) -> None:
        """Slot : gate créée dans n'importe quel panneau → rebuild arbre + refresh tous panneaux."""
        self._refresh_tree()
        self._worksheet.refresh_all(
            self._engine,
            transform_id=self._active_transform_id,
            comp_id=self._selected_comp() if self._chk_enable_comp.isChecked() else None,
        )

    def _post_gate_created(self, new_gate_name: Optional[str] = None) -> None:
        """
        Actions après création d'une gate :
          1. Rebuild arbre
          2. Auto-sélectionner la nouvelle gate si nom fourni
          3. Refresh canvas sur la gate parente (contexte courant conservé)
        """
        self._refresh_tree()
        self._update_statistics()
        # Sélectionner la gate nouvellement créée dans l'arbre et dans le combo population
        if new_gate_name:
            idx = self._combo_population.findText(new_gate_name)
            if idx >= 0:
                self._combo_population.blockSignals(True)
                self._combo_population.setCurrentIndex(idx)
                self._combo_population.blockSignals(False)
            # Trouver le nœud dans l'arbre et le sélectionner visuellement
            model = self._gate_tree_model
            for row in range(model.rowCount()):
                idx_tree = model.index(row, 0)
                node = idx_tree.data(Qt.UserRole)
                if node and getattr(node, 'gate_id', None) == new_gate_name:
                    self._tree_view.setCurrentIndex(idx_tree)
                    break
        self._refresh_canvas()

    # ------------------------------------------------------------------
    # Slots actions
    # ------------------------------------------------------------------

    def _on_delete_gate(self) -> None:
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
        """Analyse le sample actif, ou tout le groupe actif si Workspace."""
        try:
            group = self._engine.active_group
            if group:
                self._engine.analyze_group(group, use_mp=False)
            else:
                self._engine.analyze(use_mp=False)
            self._refresh_tree()
            self._refresh_canvas()
        except Exception as exc:
            self._show_error(f"Analyse échouée : {exc}")

    def _on_export_stats(self) -> None:
        from PyQt5.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(self, "Exporter statistiques", "", "CSV (*.csv)")
        if not path:
            return
        try:
            from src.exports.gating_exporter import GatingExporter

            GatingExporter.export_statistics(self._engine, path)
            QMessageBox.information(self, "Export", f"Statistiques exportées :\n{path}")
        except Exception as exc:
            self._show_error(f"Export statistiques : {exc}")

    def _on_export_fcs(self) -> None:
        from PyQt5.QtWidgets import QFileDialog

        index = self._tree_view.currentIndex()
        node: Optional[GateNode] = index.data(Qt.UserRole)
        if node is None:
            self._show_error("Sélectionnez une gate dans l'arbre pour l'export FCS.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter gate → FCS", f"{node.gate_id}.fcs", "FCS (*.fcs)"
        )
        if not path:
            return

        try:
            paths = self._engine.find_gate_paths(node.gate_id)
            gate_path = paths[0] if paths else None
            from src.exports.gating_exporter import GatingExporter

            GatingExporter.export_gated_fcs(
                self._engine,
                sample_id=self._current_sample_id(),
                gate_name=node.gate_id,
                gate_path=gate_path,
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

        # Lire les canaux depuis le canvas actif (pas les combos globaux)
        # pour supporter les panels multi-graphiques avec axes indépendants
        canvas = self._canvas
        x_ch = getattr(canvas, "_x_channel", "").strip()
        y_ch = getattr(canvas, "_y_channel", "").strip()

        # Fallback sur les combos globaux si canvas pas encore peuplé
        if not x_ch:
            x_ch = self._combo_x.currentText().strip()
        if not y_ch:
            y_ch = self._combo_y.currentText().strip()

        if not x_ch:
            self._show_error("Sélectionnez un canal X avant de dessiner.")
            self._reset_draw_buttons()
            return

        # Mode 2D nécessite Y — mais POLYGON/RECTANGLE/QUADRANT sur histo 1D
        # sont simplement ignorés (le canvas gère déjà le cas y_channel vide)
        if mode in (DrawMode.POLYGON, DrawMode.QUADRANT) and not y_ch:
            self._show_error(
                "Un canal Y est requis pour une gate Polygone ou Quadrant.\n"
                "Pour une gate 1D sur histogramme, utilisez Rectangle."
            )
            self._reset_draw_buttons()
            return

        self._canvas.set_draw_mode(mode, gate_name="Gate")
        # Visuellement, désactiver les deux autres boutons
        for btn, m in [
            (self._btn_poly, DrawMode.POLYGON),
            (self._btn_rect, DrawMode.RECTANGLE),
            (self._btn_quad, DrawMode.QUADRANT),
        ]:
            btn.setChecked(m == mode)

    def _safe_cancel_drawing(self) -> None:
        try:
            self._canvas.cancel_drawing()
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
        Déduit le gate_path parent depuis la sélection courante dans l'arbre.

        Si aucun nœud sélectionné → gate racine (tuple vide = 'root').
        """
        index = self._tree_view.currentIndex()
        node: Optional[GateNode] = index.data(Qt.UserRole)
        if node is None:
            return ("root",)
        paths = self._engine.find_gate_paths(node.gate_id)
        if not paths:
            return ("root",)
        # Le chemin parent = path du nœud sélectionné + son propre nom
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

        Lève RuntimeError si aucun panneau n'existe encore
        (avant le premier load_sample).
        """
        c = self._worksheet.active_canvas()
        if c is None:
            raise RuntimeError("WorksheetArea : aucun panneau actif")
        return c

    def _has_active_panel(self) -> bool:
        return bool(self._worksheet.panels())

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
