"""
src/gui/viewer/gating_items.py — Objets graphiques de gating pour QGraphicsScene.

Rôle strict : représentation visuelle locale d'une gate dans un canvas PyQtGraph.

Ces objets NE calculent PAS de membership. La vérité biologique vit exclusivement
dans FlowKit via PrismaFlowEngine. PolygonGate ici ne sert qu'à :
  - stocker les vertices pour la sérialisation de session
  - afficher le contour graphique dans QGraphicsScene

Hiérarchie :
  BaseGate     — conteneur de vertices et identité visuelle
  PolygonGate  — gate polygonale (QGraphicsPolygonItem)
"""

from __future__ import annotations

import uuid
from typing import Callable, List, Optional, Tuple

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QBrush, QColor, QPen, QPolygonF
from PyQt5.QtWidgets import (
    QGraphicsItem,
    QGraphicsPolygonItem,
    QGraphicsTextItem,
)


# ---------------------------------------------------------------------------
# Interface commune
# ---------------------------------------------------------------------------

class BaseGate:
    """
    Conteneur de vertices et identité visuelle d'une gate graphique.

    Ne calcule PAS de membership — cette responsabilité appartient à FlowKit
    via PrismaFlowEngine.get_gate_membership_cached().
    """

    def __init__(self, gate_id: str, name: str) -> None:
        self.gate_id: str = gate_id
        self.name: str = name

    def data_vertices(
        self,
        pixel_to_data_fn: Callable[[float, float], Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Retourne les sommets en coordonnées données via la fonction de transformation."""
        raise NotImplementedError(f"{type(self).__name__} doit implémenter data_vertices()")


# ---------------------------------------------------------------------------
# Gate polygonale libre
# ---------------------------------------------------------------------------

_GATE_FILL_ALPHA = 30        # opacité du remplissage (0-255)
_GATE_BORDER_WIDTH = 1.8     # épaisseur contour en px logiques
_GATE_COLOR = QColor(100, 220, 120)  # vert cytomètre par défaut


class PolygonGate(QGraphicsPolygonItem, BaseGate):
    """
    Gate polygonale interactive.

    Hérite de :
      - QGraphicsPolygonItem  (rendu, sélection, édition Qt)
      - BaseGate              (interface commune : gate_id, nom, masque)

    Paramètres
    ----------
    gate_id      : Identifiant unique (UUID).
    name         : Nom affiché (ex. 'Lymphocytes').
    scene_points : Sommets en coordonnées scène (QPointF).
    editable     : Si True, la gate est déplaçable et sélectionnable.
    color        : Couleur de la gate.
    """

    def __init__(
        self,
        gate_id: str,
        name: str,
        scene_points: List[QPointF],
        editable: bool = True,
        color: QColor = _GATE_COLOR,
    ) -> None:
        polygon = QPolygonF(scene_points)
        QGraphicsPolygonItem.__init__(self, polygon)
        BaseGate.__init__(self, gate_id=gate_id, name=name)

        self._color = color
        self._editable = editable
        self._label: Optional[QGraphicsTextItem] = None

        self._apply_style()
        if editable:
            self.setFlags(
                QGraphicsItem.ItemIsSelectable
                | QGraphicsItem.ItemIsMovable
                | QGraphicsItem.ItemSendsGeometryChanges
            )
        if name and name != "(drawing…)":
            self._create_label()

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------

    def _apply_style(self) -> None:
        fill = QColor(self._color)
        fill.setAlpha(_GATE_FILL_ALPHA)
        self.setBrush(QBrush(fill))
        pen = QPen(self._color, _GATE_BORDER_WIDTH)
        pen.setCosmetic(True)   # épaisseur constante indépendante du zoom
        self.setPen(pen)

    def _create_label(self) -> None:
        """Crée un QGraphicsTextItem centré sur le barycentre du polygone."""
        pts = self.polygon()
        if pts.isEmpty():
            return
        cx = sum(p.x() for p in pts) / len(pts)
        cy = sum(p.y() for p in pts) / len(pts)

        self._label = QGraphicsTextItem(self.name, self)
        self._label.setDefaultTextColor(self._color)
        br = self._label.boundingRect()
        self._label.setPos(cx - br.width() / 2, cy - br.height() / 2)
        self._label.setZValue(self.zValue() + 1)

    # ------------------------------------------------------------------
    # Mise à jour du nom
    # ------------------------------------------------------------------

    def set_name(self, name: str) -> None:
        self.name = name
        if self._label is not None:
            self._label.setPlainText(name)

    def set_data_verts(self, verts: List[Tuple[float, float]]) -> None:
        """Enregistre les sommets en coordonnées données (pour sérialisation de session)."""
        self._data_verts: List[Tuple[float, float]] = verts

    def data_vertices(
        self,
        pixel_to_data_fn,
    ) -> List[Tuple[float, float]]:
        pts = self.polygon()
        return [pixel_to_data_fn(p.x(), p.y()) for p in pts]

    # ------------------------------------------------------------------
    # Sérialisation minimale (pour sauvegarde/chargement de session)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "gate_id": self.gate_id,
            "name": self.name,
            "type": "polygon",
            "data_verts": getattr(self, "_data_verts", []),
        }

    @classmethod
    def from_dict(
        cls,
        d: dict,
        pixel_from_data_fn,
        color: QColor = _GATE_COLOR,
    ) -> "PolygonGate":
        """
        Reconstruit une PolygonGate depuis un dict (chargement session).

        Args:
            d: Dict produit par to_dict().
            pixel_from_data_fn: Fonction (xd, yd) → QPointF scène.
            color: Couleur de la gate.
        """
        verts = d["data_verts"]
        scene_pts = [pixel_from_data_fn(xd, yd) for xd, yd in verts]
        gate = cls(
            gate_id=d["gate_id"],
            name=d["name"],
            scene_points=scene_pts,
            editable=True,
            color=color,
        )
        gate.set_data_verts(verts)
        return gate
