"""
src/gui/viewer/population_tree_controller.py — Contrôleur de l'arbre de populations (Phase 5).

Extrait la logique inline de _refresh_tree() / _populate_tree() / _dedupe_hierarchy_nodes()
hors de PrismaGatingWorkspace.

Responsabilités :
  - Posséder et piloter le QTreeView + GatingTreeModel
  - Observer les signaux engine (gatingStrategyChanged, analysisCompleted)
  - Reconstruire le modèle arborescent
  - Gérer la sélection de population
  - Émettre populationSelected(gate_path) vers le workspace
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from PyQt5.QtCore import QModelIndex, QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QLabel,
    QPushButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from .gating_engine import GateHierarchyNode, PrismaFlowEngine
from .gating_tree_model import GateNode, GatingTreeModel

_logger = logging.getLogger("viewer.population_tree_controller")


class PopulationTreeController(QObject):
    """
    Contrôleur pour l'arbre de hiérarchie de gating.

    Instanciation typique dans PrismaGatingWorkspace :

        self._tree_ctrl = PopulationTreeController(
            engine=self._engine,
            parent=self,
        )
        self._tree_panel = self._tree_ctrl.build_panel_widget()
        self._tree_ctrl.populationSelected.connect(self._on_population_selected)
        self._engine.gatingStrategyChanged.connect(
            self._tree_ctrl.on_engine_strategy_changed, Qt.QueuedConnection
        )
        self._engine.analysisCompleted.connect(
            self._tree_ctrl.on_engine_analysis_completed, Qt.QueuedConnection
        )

    Signaux
    -------
    populationSelected(gate_path)
        Émis à chaque clic sur un nœud de l'arbre.
        gate_path est un tuple : (gate_name, *path_elements)
        Pour la racine : ("root",)
    """

    populationSelected = pyqtSignal(tuple)

    def __init__(
        self,
        engine: PrismaFlowEngine,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._gate_tree_model = GatingTreeModel()
        self._tree_view: Optional[QTreeView] = None
        self._current_sample_id: Optional[str] = None

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def build_panel_widget(self) -> QWidget:
        """
        Construit et retourne le widget panneau gauche contenant l'arbre.
        Peut être appelé une seule fois — crée le QTreeView interne.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        lbl = QLabel("Hiérarchie de gating")
        lbl.setStyleSheet("color: #5BAAFF; font-weight: bold; font-size: 12px;")
        layout.addWidget(lbl)

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
        self._tree_view.clicked.connect(self._on_node_clicked)
        layout.addWidget(self._tree_view)

        btn_refresh = QPushButton("↺  Actualiser hiérarchie")
        btn_refresh.clicked.connect(self.refresh_tree)
        layout.addWidget(btn_refresh)

        panel.setMinimumWidth(200)
        panel.setMaximumWidth(320)
        return panel

    def set_engine(self, engine: PrismaFlowEngine) -> None:
        """Remplace le moteur et rafraîchit l'arbre."""
        self._engine = engine
        self.refresh_tree()

    def set_current_sample(self, sample_id: Optional[str]) -> None:
        """Change le sample actif et rafraîchit l'arbre."""
        self._current_sample_id = sample_id
        self.refresh_tree()

    def refresh_tree(self) -> None:
        """
        Recharge l'arbre de gating depuis le moteur.
        Idempotent — peut être appelé à tout moment.
        """
        self._gate_tree_model.clear()
        try:
            sid = self._current_sample_id or self._engine.active_sample_id
            roots = self._engine.build_hierarchy(sample_id=sid)
            deduped = self._dedupe_hierarchy_nodes(roots)
            self._populate_tree(deduped, parent_id=None)
        except Exception as exc:
            _logger.warning("PopulationTreeController.refresh_tree: %s", exc)

        if self._tree_view is not None:
            self._tree_view.expandAll()

    def current_gate_path(self) -> Optional[Tuple[str, ...]]:
        """
        Retourne le gate_path du nœud actuellement sélectionné dans l'arbre.
        Retourne None si aucune sélection ou arbre vide.
        """
        if self._tree_view is None:
            return None
        index = self._tree_view.currentIndex()
        if not index.isValid():
            return None
        node: Optional[GateNode] = index.data(Qt.UserRole)
        if node is None:
            return None
        return self._node_to_gate_path(node)

    @property
    def tree_view(self) -> Optional[QTreeView]:
        """Accès direct au QTreeView (pour intégration dans splitter)."""
        return self._tree_view

    @property
    def gate_tree_model(self) -> GatingTreeModel:
        return self._gate_tree_model

    # ------------------------------------------------------------------
    # Slots engine (à connecter avec Qt.QueuedConnection)
    # ------------------------------------------------------------------

    def on_engine_strategy_changed(self, gate_id: str, change_type: str) -> None:
        """Slot : stratégie de gating modifiée — rafraîchit l'arbre."""
        _logger.debug(
            "PopulationTreeController: gatingStrategyChanged gate_id=%s type=%s",
            gate_id,
            change_type,
        )
        self.refresh_tree()

    def on_engine_analysis_completed(self, sample_id: str) -> None:
        """Slot : analyse async terminée — rafraîchit l'arbre avec les nouveaux comptes."""
        _logger.debug(
            "PopulationTreeController: analysisCompleted sample_id=%s", sample_id
        )
        self.refresh_tree()

    # ------------------------------------------------------------------
    # Slot interne — clic sur nœud de l'arbre
    # ------------------------------------------------------------------

    def _on_node_clicked(self, index: QModelIndex) -> None:
        node: Optional[GateNode] = index.data(Qt.UserRole)
        if node is None:
            return
        gate_path = self._node_to_gate_path(node)
        _logger.debug("PopulationTreeController: nœud cliqué → %s", gate_path)
        self.populationSelected.emit(gate_path)

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    def _node_to_gate_path(self, node: GateNode) -> Tuple[str, ...]:
        """Convertit un GateNode en gate_path compatible workspace."""
        try:
            paths = self._engine.find_gate_paths(node.gate_id)
            if paths:
                return (node.gate_id, *paths[0])
        except Exception as exc:
            _logger.debug("find_gate_paths(%s): %s", node.gate_id, exc)
        return (node.gate_id,)

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
                flow_count=int(node.count),
                flow_percent=float(node.pct_parent),
            )
            gate_node.mask = None
            self._gate_tree_model.add_gate(gate_node)
            if node.children:
                self._populate_tree(node.children, parent_id=node.gate_name)

    def _dedupe_hierarchy_nodes(
        self,
        nodes: List[GateHierarchyNode],
    ) -> List[GateHierarchyNode]:
        seen: set[Tuple[str, Tuple[str, ...]]] = set()

        def _walk(items: List[GateHierarchyNode]) -> List[GateHierarchyNode]:
            result: List[GateHierarchyNode] = []
            for item in items:
                key = (str(item.gate_name), tuple(item.gate_path))
                if key in seen:
                    continue
                seen.add(key)
                item.children = _walk(item.children)
                result.append(item)
            return result

        return _walk(nodes)
