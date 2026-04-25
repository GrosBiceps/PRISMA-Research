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
from typing import Any, List, Optional, Tuple

import pandas as pd
from PyQt5.QtCore import Qt, QModelIndex, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
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
        self._btn_save_workspace = QPushButton("💾  Sauvegarder contexte gating")
        self._btn_load_workspace = QPushButton("📂  Charger contexte gating")

        for btn in [
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
        # Canvas → engine
        self._canvas.polygonGateCompleted.connect(self._on_polygon_gate)
        self._canvas.rectangleGateCompleted.connect(self._on_rectangle_gate)
        self._canvas.quadrantGateCompleted.connect(self._on_quadrant_gate)

        # Arbre → affichage
        self._tree_view.clicked.connect(self._on_tree_node_clicked)

        # Contrôles → affichage
        self._combo_group.currentTextChanged.connect(self._on_group_changed)
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

        files = sorted(folder.glob("*.fcs")) + sorted(folder.glob("*.FCS"))
        if not files:
            raise FileNotFoundError(f"Aucun fichier FCS trouvé dans: {folder}")

        engine = PrismaFlowEngine()
        engine.load_fcs_batch(files, make_first_active=True)
        self._engine = engine
        self._loaded_fcs_paths = [str(p.resolve()) for p in files]
        self._last_saved_workspace_path = None
        self.reload_from_engine()

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
                x_min,
                x_max,
                y_min,
                y_max,
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
