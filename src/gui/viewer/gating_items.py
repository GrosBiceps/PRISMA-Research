"""
src/gui/viewer/gating_items.py — Objets graphiques de gating pour QGraphicsScene.

Hiérarchie :
  BaseGate           — interface commune (gate_id, nom, masque, sélection)
    PolygonGate      — gate polygonale libre (QGraphicsPolygonItem)

Extensible vers :
  RectangleGate      — hérite BaseGate, remplace QGraphicsPolygonItem par QGraphicsRectItem
  QuadrantGate       — quatre régions définies par un point croisé
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QBrush, QColor, QPen, QPolygonF
from PyQt5.QtWidgets import (
    QGraphicsItem,
    QGraphicsPolygonItem,
    QGraphicsTextItem,
)

# ---------------------------------------------------------------------------
# Inclusion vectorisée point-dans-polygone
# ---------------------------------------------------------------------------

def _points_in_polygon(
    xs: np.ndarray,
    ys: np.ndarray,
    poly_x: np.ndarray,
    poly_y: np.ndarray,
) -> np.ndarray:
    """
    Retourne un masque bool : True si (xs[i], ys[i]) est dans le polygone.

    Stratégie :
      1. shapely.vectorized.contains   (le plus rapide, ~O(N) via GEOS)
      2. matplotlib.path.Path.contains_points (fallback vectorisé, ~O(N·V))
      3. Ray-casting Python pur         (fallback de dernier recours)
    """
    verts_closed = list(zip(poly_x.tolist(), poly_y.tolist()))

    # --- tentative shapely ---
    try:
        from shapely.geometry import Point, Polygon as ShapelyPolygon
        from shapely.vectorized import contains  # shapely >= 2.0

        poly = ShapelyPolygon(verts_closed)
        return contains(poly, xs, ys)
    except ImportError:
        pass
    except Exception:
        pass

    # --- tentative shapely < 2.0 (vectorized non disponible) ---
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        poly = ShapelyPolygon(verts_closed)
        return np.array([poly.contains(Point(x, y)) for x, y in zip(xs, ys)], dtype=bool)
    except ImportError:
        pass

    # --- fallback matplotlib.path (toujours disponible dans ce projet) ---
    try:
        from matplotlib.path import Path as MplPath

        verts = np.column_stack([poly_x, poly_y])
        # Fermeture implicite : matplotlib exige le premier point répété
        verts_closed_arr = np.vstack([verts, verts[0]])
        path = MplPath(verts_closed_arr)
        pts = np.column_stack([xs, ys])
        return path.contains_points(pts)
    except Exception:
        pass

    # --- fallback ray-casting Python pur ---
    mask = np.zeros(len(xs), dtype=bool)
    n_verts = len(poly_x)
    for i, (px, py) in enumerate(zip(xs, ys)):
        inside = False
        j = n_verts - 1
        for k in range(n_verts):
            xi, yi = poly_x[k], poly_y[k]
            xj, yj = poly_x[j], poly_y[j]
            if ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi
            ):
                inside = not inside
            j = k
        mask[i] = inside
    return mask


# ---------------------------------------------------------------------------
# Interface commune
# ---------------------------------------------------------------------------

class BaseGate(ABC):
    """
    Interface minimale partagée par tous les types de gate.

    Chaque sous-classe est aussi un QGraphicsItem (héritage multiple PyQt5).
    """

    def __init__(self, gate_id: str, name: str) -> None:
        self.gate_id: str = gate_id
        self.name: str = name

    @abstractmethod
    def compute_mask(
        self,
        events: pd.DataFrame,
        x_channel: str,
        y_channel: str,
    ) -> np.ndarray:
        """
        Calcule le masque booléen des événements inclus dans cette gate.

        Args:
            events: DataFrame (N_events × N_channels).
            x_channel: Colonne utilisée comme axe X.
            y_channel: Colonne utilisée comme axe Y.

        Returns:
            ndarray bool de longueur N_events.
        """

    @abstractmethod
    def data_vertices(
        self,
        pixel_to_data_fn: "Callable[[float, float], Tuple[float, float]]",
    ) -> List[Tuple[float, float]]:
        """
        Retourne les sommets de la gate en coordonnées données.

        Args:
            pixel_to_data_fn: Fonction (px, py) → (xd, yd).

        Returns:
            Liste de tuples (xd, yd).
        """


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

    # ------------------------------------------------------------------
    # BaseGate — compute_mask
    # ------------------------------------------------------------------

    def compute_mask(
        self,
        events: pd.DataFrame,
        x_channel: str,
        y_channel: str,
    ) -> np.ndarray:
        """
        Calcule le masque booléen des événements inclus dans ce polygone.

        Les sommets sont en coordonnées scène ; le canvas fournit
        une transformée affine via FCSCanvas._data_to_pixel / _pixel_to_data.
        Ici, la gate stocke ses sommets en coordonnées *données* (attribut
        data_verts), mis à jour au moment de la création par FCSCanvas.

        Si data_verts n'est pas défini (gate preview ou ancienne gate scène),
        retourne un masque tout-False.
        """
        if not hasattr(self, "_data_verts") or self._data_verts is None:
            return np.zeros(len(events), dtype=bool)

        xs = events[x_channel].to_numpy(dtype=np.float64)
        ys = events[y_channel].to_numpy(dtype=np.float64)
        poly_x = np.array([v[0] for v in self._data_verts], dtype=np.float64)
        poly_y = np.array([v[1] for v in self._data_verts], dtype=np.float64)

        return _points_in_polygon(xs, ys, poly_x, poly_y)

    def set_data_verts(self, verts: List[Tuple[float, float]]) -> None:
        """Enregistre les sommets en coordonnées données (appelé par FCSCanvas)."""
        self._data_verts: List[Tuple[float, float]] = verts

    # ------------------------------------------------------------------
    # BaseGate — data_vertices
    # ------------------------------------------------------------------

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
