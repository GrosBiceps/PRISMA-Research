"""
src/gui/viewer/gating_tree_model.py — Modèle QAbstractItemModel pour l'arbre de gating.

Usage minimal :
    model = GatingTreeModel()
    tree_view.setModel(model)

    model.add_gate(GateNode(gate_id="g1", name="Lymphocytes", mask=mask_lym))
    child = GateNode(gate_id="g2", name="CD4+", mask=mask_cd4, parent_id="g1")
    model.add_gate(child)

Architecture :
  GateNode          — nœud de l'arbre (données)
  GatingTreeModel   — QAbstractItemModel (interface Qt)
  GatingTreeDelegate— QStyledItemDelegate (badge de population %)

Chaque nœud stocke :
  gate_id       str      — identifiant unique
  name          str      — nom affiché
  parent_id     str|None — clé du nœud parent (None = racine)
  mask          ndarray  — masque booléen de l'ensemble source
  gate_type     str      — 'polygon' | 'rectangle' | 'quadrant' | 'root'
  source_pop    str      — identifiant de la population source ('' = all events)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from PyQt5.QtCore import (
    QAbstractItemModel,
    QModelIndex,
    Qt,
)
from PyQt5.QtGui import QBrush, QColor, QFont
from PyQt5.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem


# ---------------------------------------------------------------------------
# Nœud de l'arbre
# ---------------------------------------------------------------------------

@dataclass
class GateNode:
    """
    Nœud de la hiérarchie de gating.

    Attributes:
        gate_id     : Identifiant unique (UUID ou chaîne libre).
        name        : Nom de la gate (affiché dans le tree).
        mask        : Masque booléen des événements inclus (longueur N_events).
                      None si le nœud racine synthétique.
        parent_id   : gate_id du nœud parent ; None pour les nœuds de premier niveau.
        gate_type   : 'polygon' | 'rectangle' | 'quadrant' | 'root'.
        source_pop  : gate_id de la population source utilisée pour calculer ce masque.
                      Chaîne vide = tous les événements du Sample.
        children    : Liste des nœuds enfants (gérée par GatingTreeModel).
    """

    gate_id: str
    name: str
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    parent_id: Optional[str] = None
    gate_type: str = "polygon"
    source_pop: str = ""
    children: List["GateNode"] = field(default_factory=list, repr=False)

    # Référence au parent (non sérialisée, injectée par le modèle)
    _parent_node: Optional["GateNode"] = field(default=None, init=False, repr=False)

    @property
    def n_events(self) -> int:
        """Nombre d'événements positifs (sum du masque)."""
        if self.mask is None:
            return 0
        return int(np.sum(self.mask))

    @property
    def n_total(self) -> int:
        """Taille totale du masque (N_events du Sample)."""
        if self.mask is None:
            return 0
        return len(self.mask)

    @property
    def percent(self) -> float:
        """Pourcentage d'événements positifs dans la population source."""
        if self.mask is None or len(self.mask) == 0:
            return 0.0
        if self._parent_node is not None and self._parent_node.mask is not None:
            n_parent = int(np.sum(self._parent_node.mask))
        else:
            n_parent = len(self.mask)
        if n_parent == 0:
            return 0.0
        return 100.0 * self.n_events / n_parent

    def row(self) -> int:
        """Position de ce nœud parmi les enfants de son parent."""
        if self._parent_node is not None:
            return self._parent_node.children.index(self)
        return 0

    def child(self, row: int) -> Optional["GateNode"]:
        if 0 <= row < len(self.children):
            return self.children[row]
        return None

    def child_count(self) -> int:
        return len(self.children)


# ---------------------------------------------------------------------------
# Modèle Qt
# ---------------------------------------------------------------------------

_COL_NAME = 0
_COL_TYPE = 1
_COL_N = 2
_COL_PCT = 3
_NCOLS = 4

_HEADERS = ["Gate", "Type", "N", "%"]

_TYPE_COLORS: Dict[str, QColor] = {
    "polygon":   QColor(100, 220, 120),
    "rectangle": QColor(100, 180, 255),
    "quadrant":  QColor(255, 180,  80),
    "root":      QColor(180, 180, 180),
}


class GatingTreeModel(QAbstractItemModel):
    """
    Modèle pour QTreeView représentant la hiérarchie des gates.

    Colonnes :
      0 — Nom de la gate
      1 — Type ('polygon', 'rectangle', …)
      2 — N événements positifs
      3 — % par rapport à la population parente

    API publique :
        add_gate(node)          — ajoute un GateNode (parent résolu via parent_id)
        remove_gate(gate_id)    — retire un nœud et ses enfants
        get_node(gate_id)       — retourne le GateNode ou None
        update_mask(gate_id, mask) — met à jour le masque et notifie Qt
        clear()                 — efface tout

    Nœud racine synthétique :
        _root est un GateNode invisible ('root') portant les nœuds de premier niveau.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._root = GateNode(
            gate_id="__root__",
            name="root",
            gate_type="root",
        )
        self._registry: Dict[str, GateNode] = {"__root__": self._root}

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def add_gate(self, node: GateNode) -> None:
        """
        Ajoute un GateNode à l'arbre.

        Si node.parent_id est None ou introuvable, le nœud est rattaché à la racine.
        """
        parent_node = self._registry.get(node.parent_id or "", self._root)
        if parent_node is None:
            parent_node = self._root

        parent_index = self._index_of(parent_node)
        insert_row = parent_node.child_count()

        self.beginInsertRows(parent_index, insert_row, insert_row)
        node._parent_node = parent_node
        parent_node.children.append(node)
        self._registry[node.gate_id] = node
        self.endInsertRows()

    def remove_gate(self, gate_id: str) -> None:
        """Retire un nœud et tous ses descendants."""
        node = self._registry.get(gate_id)
        if node is None or node is self._root:
            return
        self._remove_recursive(node)

    def get_node(self, gate_id: str) -> Optional[GateNode]:
        return self._registry.get(gate_id)

    def update_mask(self, gate_id: str, mask: np.ndarray) -> None:
        """Met à jour le masque d'un nœud et notifie la vue."""
        node = self._registry.get(gate_id)
        if node is None:
            return
        node.mask = mask
        idx_n = self.createIndex(node.row(), _COL_N, node)
        idx_p = self.createIndex(node.row(), _COL_PCT, node)
        self.dataChanged.emit(idx_n, idx_p)

    def clear(self) -> None:
        """Efface tous les nœuds sauf la racine synthétique."""
        self.beginResetModel()
        self._root.children.clear()
        self._registry = {"__root__": self._root}
        self.endResetModel()

    def all_gate_ids(self) -> List[str]:
        return [k for k in self._registry if k != "__root__"]

    # ------------------------------------------------------------------
    # QAbstractItemModel — implémentation obligatoire
    # ------------------------------------------------------------------

    def index(
        self, row: int, column: int, parent: QModelIndex = QModelIndex()
    ) -> QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        parent_node: GateNode = (
            parent.internalPointer() if parent.isValid() else self._root
        )
        child = parent_node.child(row)
        if child is not None:
            return self.createIndex(row, column, child)
        return QModelIndex()

    def parent(self, index: QModelIndex = QModelIndex()) -> QModelIndex:
        if not index.isValid():
            return QModelIndex()
        node: GateNode = index.internalPointer()
        parent_node = node._parent_node
        if parent_node is None or parent_node is self._root:
            return QModelIndex()
        return self.createIndex(parent_node.row(), 0, parent_node)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        parent_node: GateNode = (
            parent.internalPointer() if parent.isValid() else self._root
        )
        return parent_node.child_count()

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return _NCOLS

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ) -> Any:
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return _HEADERS[section] if section < len(_HEADERS) else None
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        node: GateNode = index.internalPointer()
        col = index.column()

        if role == Qt.DisplayRole:
            if col == _COL_NAME:
                return node.name
            if col == _COL_TYPE:
                return node.gate_type
            if col == _COL_N:
                return f"{node.n_events:,}"
            if col == _COL_PCT:
                return f"{node.percent:.1f} %"

        if role == Qt.ForegroundRole:
            color = _TYPE_COLORS.get(node.gate_type, QColor(200, 200, 200))
            if col == _COL_NAME:
                return QBrush(color)
            return QBrush(QColor(180, 180, 180))

        if role == Qt.FontRole and col == _COL_NAME:
            font = QFont()
            font.setBold(True)
            return font

        if role == Qt.ToolTipRole:
            return (
                f"Gate: {node.name}\n"
                f"ID: {node.gate_id}\n"
                f"Type: {node.gate_type}\n"
                f"N: {node.n_events:,} / {node.n_total:,}\n"
                f"%: {node.percent:.2f}"
            )

        if role == Qt.UserRole:
            return node

        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    def _index_of(self, node: GateNode) -> QModelIndex:
        """Retourne le QModelIndex du nœud (colonne 0)."""
        if node is self._root:
            return QModelIndex()
        return self.createIndex(node.row(), 0, node)

    def _remove_recursive(self, node: GateNode) -> None:
        """Retire un nœud et ses enfants de l'arbre et du registre."""
        for child in list(node.children):
            self._remove_recursive(child)

        parent_node = node._parent_node or self._root
        parent_index = self._index_of(parent_node)
        row = node.row()

        self.beginRemoveRows(parent_index, row, row)
        parent_node.children.remove(node)
        self._registry.pop(node.gate_id, None)
        self.endRemoveRows()


# ---------------------------------------------------------------------------
# Délégué optionnel — badge de population %
# ---------------------------------------------------------------------------

class GatingTreeDelegate(QStyledItemDelegate):
    """
    Délégué pour la colonne % : affiche un badge coloré proportionnel.

    Optionnel — peut être appliqué via :
        tree_view.setItemDelegateForColumn(3, GatingTreeDelegate(tree_view))
    """

    _BAR_COLOR = QColor(100, 220, 120, 120)
    _BAR_MAX_W = 60

    def paint(self, painter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        if index.column() != _COL_PCT:
            super().paint(painter, option, index)
            return

        node: Optional[GateNode] = index.data(Qt.UserRole)
        if node is None:
            super().paint(painter, option, index)
            return

        super().paint(painter, option, index)

        pct = min(node.percent, 100.0)
        bar_w = int(pct / 100.0 * self._BAR_MAX_W)
        rect = option.rect
        bar_rect = rect.adjusted(2, rect.height() - 4, -2, -1)
        bar_rect.setWidth(bar_w)

        painter.save()
        painter.setRenderHint(painter.Antialiasing, False)
        painter.fillRect(bar_rect, self._BAR_COLOR)
        painter.restore()
