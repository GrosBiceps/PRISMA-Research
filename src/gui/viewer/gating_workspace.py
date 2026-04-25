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

import logging
from typing import List, Optional, Tuple

import pandas as pd
from PyQt5.QtCore import Qt, QModelIndex
from PyQt5.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeView,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QSizePolicy,
    QFrame,
    QInputDialog,
)

from src.utils.logger import get_logger
from .gating_engine import (
    PrismaFlowEngine,
    PrismaEngineError,
    GateHierarchyNode,
)
from .interactive_canvas import InteractiveGatingCanvas, DrawMode
from .gating_tree_model import GatingTreeModel, GateNode

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

        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Construction UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setHandleWidth(4)
        root_layout.addWidget(splitter)

        # ── Panneau gauche : arbre de gating ────────────────────────────
        self._tree_panel = self._build_tree_panel()
        splitter.addWidget(self._tree_panel)

        # ── Panneau central : canvas ─────────────────────────────────────
        self._canvas = InteractiveGatingCanvas(self)
        splitter.addWidget(self._canvas)

        # ── Panneau droit : contrôles ────────────────────────────────────
        self._ctrl_panel = self._build_control_panel()
        splitter.addWidget(self._ctrl_panel)

        splitter.setSizes([240, 700, 280])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        self._apply_style()

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QWidget { background: #080D18; color: #EEF2F7; font-family: 'Outfit', 'Segoe UI', sans-serif; }
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
        self._combo_transform.addItem("— Brut —")
        vtr.addWidget(self._combo_transform)

        vtr.addWidget(QLabel("Compensation"))
        self._combo_comp = QComboBox()
        self._combo_comp.addItem("— Non compensé —")
        vtr.addWidget(self._combo_comp)

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

        self._btn_apply = QPushButton("▶  Afficher")
        self._btn_delete_gate = QPushButton("🗑  Supprimer gate sélectionnée")
        self._btn_analyze = QPushButton("⚡  Analyser tous les samples")
        self._btn_export_stats = QPushButton("📊  Exporter statistiques CSV")
        self._btn_export_fcs = QPushButton("💾  Exporter gate → FCS")

        for btn in [
            self._btn_apply,
            self._btn_delete_gate,
            self._btn_analyze,
            self._btn_export_stats,
            self._btn_export_fcs,
        ]:
            vact.addWidget(btn)

        layout.addWidget(grp_act)
        layout.addStretch()

        return panel

    # ------------------------------------------------------------------
    # Connexion des signaux
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        # Canvas → engine
        self._canvas.polygonGateCompleted.connect(self._on_polygon_gate)
        self._canvas.rectangleGateCompleted.connect(self._on_rectangle_gate)
        self._canvas.quadrantGateCompleted.connect(self._on_quadrant_gate)

        # Arbre → affichage
        self._tree_view.clicked.connect(self._on_tree_node_clicked)

        # Contrôles → affichage
        self._combo_sample.currentTextChanged.connect(self._on_sample_changed)
        self._btn_apply.clicked.connect(self._refresh_canvas)

        # Boutons dessin
        self._btn_poly.clicked.connect(lambda: self._activate_draw(DrawMode.POLYGON))
        self._btn_rect.clicked.connect(lambda: self._activate_draw(DrawMode.RECTANGLE))
        self._btn_quad.clicked.connect(lambda: self._activate_draw(DrawMode.QUADRANT))
        self._btn_cancel_draw.clicked.connect(self._canvas.cancel_drawing)
        self._btn_cancel_draw.clicked.connect(self._reset_draw_buttons)

        # Actions
        self._btn_delete_gate.clicked.connect(self._on_delete_gate)
        self._btn_analyze.clicked.connect(self._on_analyze)
        self._btn_export_stats.clicked.connect(self._on_export_stats)
        self._btn_export_fcs.clicked.connect(self._on_export_fcs)

    # ------------------------------------------------------------------
    # API publique — rechargement depuis engine
    # ------------------------------------------------------------------

    def reload_from_engine(self) -> None:
        """Recharge l'UI complète depuis l'état courant du moteur."""
        self._reload_sample_combo()
        self._reload_channel_combos()
        self._reload_transform_combo()
        self._reload_comp_combo()
        self._refresh_tree()

    def set_engine(self, engine: PrismaFlowEngine) -> None:
        """Remplace le moteur et recharge l'UI."""
        self._engine = engine
        self.reload_from_engine()

    # ------------------------------------------------------------------
    # Rechargements internes
    # ------------------------------------------------------------------

    def _reload_sample_combo(self) -> None:
        self._combo_sample.blockSignals(True)
        self._combo_sample.clear()
        for sid in self._engine.get_sample_ids():
            self._combo_sample.addItem(sid)
        if self._engine.active_sample_id:
            idx = self._combo_sample.findText(self._engine.active_sample_id)
            if idx >= 0:
                self._combo_sample.setCurrentIndex(idx)
        self._combo_sample.blockSignals(False)

    def _reload_channel_combos(self) -> None:
        try:
            channels = self._engine.get_sample_channels()
        except Exception:
            return

        for combo in (self._combo_x, self._combo_y):
            combo.blockSignals(True)
            combo.clear()

        self._combo_y.addItem("")
        for ch in channels:
            self._combo_x.addItem(ch)
            self._combo_y.addItem(ch)

        for combo in (self._combo_x, self._combo_y):
            combo.blockSignals(False)

    def _reload_transform_combo(self) -> None:
        self._combo_transform.blockSignals(True)
        self._combo_transform.clear()
        self._combo_transform.addItem("— Brut —")
        for tid in self._engine.get_transform_ids():
            self._combo_transform.addItem(tid)
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
        """Récupère les données depuis le moteur et met à jour le canvas."""
        x_ch = self._combo_x.currentText().strip()
        y_ch = self._combo_y.currentText().strip()
        transform_id = self._selected_transform()
        comp_id = self._selected_comp()
        pop_sel = self._combo_population.currentText()

        gate_name = None
        gate_path = None
        if pop_sel and pop_sel != "— Tous les événements —":
            gate_name = pop_sel
            paths = self._engine.find_gate_paths(pop_sel)
            gate_path = paths[0] if paths else None

        if not x_ch:
            return

        try:
            df = self._engine.get_raw_dataframe(
                sample_id=self._current_sample_id(),
                gate_name=gate_name,
                gate_path=gate_path,
                transform_id=transform_id,
                comp_matrix_id=comp_id,
            )
        except Exception as exc:
            self._show_error(f"Chargement données : {exc}")
            return

        density = self._chk_density.isChecked()
        overlay = self._chk_overlay.isChecked()

        if overlay and gate_name:
            self._show_overlay_children(df, gate_name, gate_path, x_ch, y_ch)
        elif y_ch:
            self._canvas.set_data_2d(df, x_ch, y_ch, density_coloring=density)
        else:
            self._canvas.set_data_1d(df, x_ch)

        # Recharger les overlays de gates existantes
        self._canvas.reload_gate_overlays_from_engine(self._engine)

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
        # Mettre à jour le combo population
        idx = self._combo_population.findText(node.gate_id)
        if idx >= 0:
            self._combo_population.setCurrentIndex(idx)
        self._refresh_canvas()

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
        try:
            self._engine.create_polygon_gate_from_vertices(
                name.strip(),
                gate_path,
                x_channel,
                y_channel,
                vertices,
                transform_ref=self._selected_transform(),
            )
            self._post_gate_created()
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
            self._engine.create_rectangle_gate_from_bounds(
                name.strip(),
                gate_path,
                x_channel,
                y_channel,
                x_min, x_max, y_min, y_max,
                transform_ref=self._selected_transform(),
            )
            self._post_gate_created()
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
            self._engine.create_quadrant_gate_from_thresholds(
                name.strip(),
                gate_path,
                x_channel,
                y_channel,
                x_threshold,
                y_threshold,
                transform_ref=self._selected_transform(),
            )
            self._post_gate_created()
        except PrismaEngineError as exc:
            self._show_error(str(exc))
        finally:
            self._reset_draw_buttons()

    def _post_gate_created(self) -> None:
        """Actions communes après création d'une gate."""
        self._refresh_tree()
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
            self._canvas.remove_gate_overlay(node.gate_id)
        except PrismaEngineError as exc:
            self._show_error(str(exc))

    def _on_analyze(self) -> None:
        try:
            self._engine.analyze()
            self._refresh_tree()
            self._refresh_canvas()
        except Exception as exc:
            self._show_error(f"Analyse échouée : {exc}")

    def _on_export_stats(self) -> None:
        from PyQt5.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter statistiques", "", "CSV (*.csv)"
        )
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

    # ------------------------------------------------------------------
    # Gestion mode dessin
    # ------------------------------------------------------------------

    def _activate_draw(self, mode: DrawMode) -> None:
        x_ch = self._combo_x.currentText().strip()
        y_ch = self._combo_y.currentText().strip()

        if not x_ch:
            self._show_error("Sélectionnez un canal X avant de dessiner.")
            self._reset_draw_buttons()
            return

        if mode in (DrawMode.POLYGON, DrawMode.RECTANGLE, DrawMode.QUADRANT) and not y_ch:
            self._show_error(
                "Un canal Y est requis pour dessiner une gate 2D.\n"
                "Pour une gate 1D, utilisez une RectangleGate avec Y vide."
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

    def _reset_draw_buttons(self) -> None:
        for btn in (self._btn_poly, self._btn_rect, self._btn_quad):
            btn.setChecked(False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_sample_id(self) -> Optional[str]:
        sid = self._combo_sample.currentText()
        return sid if sid else self._engine.active_sample_id

    def _selected_transform(self) -> Optional[str]:
        t = self._combo_transform.currentText()
        return t if t and t != "— Brut —" else None

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
