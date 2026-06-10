"""
src/gui/viewer/fcs_canvas.py — Canvas haute performance pour visualisation FCS 2D.

Architecture :
  - QGraphicsScene  : porte les points (QGraphicsPixmapItem) et les gates
  - QGraphicsView   : zoom/pan natif Qt + intercepte dessin de gate
  - ScatterLayer    : rasterise les points via numpy→QImage (évite N QGraphicsItem)

Optimisations retenues :
  1. Rasterisation des points dans un QImage (numpy vectorisé) : O(N) pixels,
     pas O(N) Qt items. Supporte 500k+ événements sans ralentissement.
  2. Sous-échantillonnage adaptatif : si N > MAX_RENDER_PTS, tirage aléatoire
     reproductible avant rendu (préserve la densité visuelle).
  3. Coordonnées axes → pixels calculées en numpy (pas de boucle Python).
  4. Les gates restent des QGraphicsItem séparés (couche overlay).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from .gating_items import PolygonGate

if TYPE_CHECKING:
    from prisma.core.models_legacy.sample import Sample

# ---------------------------------------------------------------------------
# Constantes de rendu
# ---------------------------------------------------------------------------

MAX_RENDER_PTS: int = 200_000   # Au-delà → sous-échantillonnage
POINT_RADIUS: int = 1           # px dans l'image rasterisée
CANVAS_W: int = 800             # Largeur scène en pixels logiques
CANVAS_H: int = 800             # Hauteur scène en pixels logiques
MARGIN: float = 0.05            # Marge relative autour des données (5 %)

DEFAULT_POINT_COLOR: QColor = QColor(80, 160, 255, 140)   # Bleu semi-transparent
GATE_DRAWING_COLOR: QColor = QColor(255, 200, 50, 200)    # Jaune dessin en cours


# ---------------------------------------------------------------------------
# Couche de rasterisation des points
# ---------------------------------------------------------------------------

class _ScatterLayer(QGraphicsPixmapItem):
    """
    QGraphicsPixmapItem qui contient le rendu rasterisé de tous les événements.

    Remplace N QGraphicsItem individuels par un unique pixmap, ce qui permet
    d'afficher des centaines de milliers de points sans perte de fluidité.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setZValue(0)

    def render_points(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        color: QColor = DEFAULT_POINT_COLOR,
        w: int = CANVAS_W,
        h: int = CANVAS_H,
    ) -> None:
        """
        Rasterise xs/ys (coordonnées pixel, float) dans un QImage ARGB32.

        Les coordonnées sont supposées déjà normalisées dans [0, w] × [0, h].
        """
        img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        img.fill(Qt.transparent)

        if len(xs) == 0:
            self.setPixmap(QPixmap.fromImage(img))
            return

        # Conversion sécurisée en entiers, clamp dans les bornes
        px = np.clip(xs.astype(np.int32), 0, w - 1)
        py = np.clip(ys.astype(np.int32), 0, h - 1)

        painter = QPainter(img)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)

        # Vectorisation : construire les rectangles en numpy puis appeler setPixel
        # Pour des performances maximales avec PyQt5 (pas de setPixelColor en batch),
        # on dessine des points 1×1 via painter.drawPoint en boucle compressée.
        # Alternative : remplir un buffer numpy ARGB et utiliser fromData.
        argb_val = (
            (color.alpha() << 24)
            | (color.red() << 16)
            | (color.green() << 8)
            | color.blue()
        )
        # Construction du buffer numpy pour transfert direct → QImage
        buf = np.zeros((h, w), dtype=np.uint32)
        np.add.at(buf, (py, px), argb_val)
        # Clamp : une cellule ne peut pas dépasser la valeur max ARGB
        buf = np.clip(buf, 0, 0xFFFFFFFF).astype(np.uint32)

        painter.end()

        # Transfert numpy → QImage via fromData
        raw_bytes = buf.tobytes()
        img2 = QImage(raw_bytes, w, h, w * 4, QImage.Format_ARGB32)
        img2 = img2.copy()  # Détache du buffer numpy (ownership)
        self.setPixmap(QPixmap.fromImage(img2))


# ---------------------------------------------------------------------------
# Scène principale
# ---------------------------------------------------------------------------

class FCSScene(QGraphicsScene):
    """
    QGraphicsScene portant la couche de points rasterisée et les gates overlay.
    Émet gate_drawing_finished quand l'utilisateur ferme un polygone.
    """

    gate_drawing_finished = pyqtSignal(list)   # list[QPointF] scène

    def __init__(self, parent=None) -> None:
        super().__init__(0, 0, CANVAS_W, CANVAS_H, parent)
        self._scatter = _ScatterLayer()
        self.addItem(self._scatter)

        self._drawing: bool = False
        self._draw_pts: List[QPointF] = []
        self._preview_gate: Optional[PolygonGate] = None

    # ------------------------------------------------------------------
    # API rendu points
    # ------------------------------------------------------------------

    def set_scatter(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        color: QColor = DEFAULT_POINT_COLOR,
    ) -> None:
        """Met à jour la couche rasterisée avec de nouvelles coordonnées pixel."""
        self._scatter.render_points(xs, ys, color=color)

    # ------------------------------------------------------------------
    # Gestion du dessin de gate
    # ------------------------------------------------------------------

    def start_gate_drawing(self) -> None:
        """Active le mode dessin de porte."""
        self._drawing = True
        self._draw_pts = []
        if self._preview_gate is not None:
            self.removeItem(self._preview_gate)
            self._preview_gate = None

    def cancel_gate_drawing(self) -> None:
        """Annule le dessin en cours."""
        self._drawing = False
        self._draw_pts = []
        if self._preview_gate is not None:
            self.removeItem(self._preview_gate)
            self._preview_gate = None

    def is_drawing(self) -> bool:
        return self._drawing

    def _add_point(self, scene_pos: QPointF) -> None:
        self._draw_pts.append(scene_pos)
        self._update_preview()

    def _close_polygon(self) -> None:
        """Ferme le polygone et émet gate_drawing_finished."""
        if len(self._draw_pts) < 3:
            return
        pts = list(self._draw_pts)
        self._drawing = False
        self._draw_pts = []
        if self._preview_gate is not None:
            self.removeItem(self._preview_gate)
            self._preview_gate = None
        self.gate_drawing_finished.emit(pts)

    def _update_preview(self) -> None:
        """Redessine le polygone de prévisualisation."""
        if self._preview_gate is not None:
            self.removeItem(self._preview_gate)
        if len(self._draw_pts) >= 2:
            self._preview_gate = PolygonGate(
                gate_id="__preview__",
                name="(drawing…)",
                scene_points=list(self._draw_pts),
                editable=False,
            )
            pen = QPen(GATE_DRAWING_COLOR, 1.5, Qt.DashLine)
            self._preview_gate.setPen(pen)
            self._preview_gate.setZValue(10)
            self.addItem(self._preview_gate)
        else:
            self._preview_gate = None

    # ------------------------------------------------------------------
    # Événements souris (délégués depuis la view)
    # ------------------------------------------------------------------

    def handle_scene_press(self, scene_pos: QPointF, button: Qt.MouseButton) -> None:
        if not self._drawing:
            return
        if button == Qt.LeftButton:
            self._add_point(scene_pos)
        elif button == Qt.RightButton:
            self._close_polygon()

    def handle_scene_dbl_click(self, scene_pos: QPointF) -> None:
        if self._drawing:
            self._close_polygon()


# ---------------------------------------------------------------------------
# Vue principale (zoom/pan + interception dessin)
# ---------------------------------------------------------------------------

class FCSView(QGraphicsView):
    """
    QGraphicsView avec :
      - zoom molette (facteur 1.15 par cran)
      - pan clic-milieu ou espace+drag
      - interception click gauche/droit pour dessin de gate
    """

    def __init__(self, scene: FCSScene, parent: Optional[QWidget] = None) -> None:
        super().__init__(scene, parent)
        self._scene: FCSScene = scene
        self._panning: bool = False
        self._pan_start: Optional[QPointF] = None

        self.setRenderHint(QPainter.Antialiasing, False)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setBackgroundBrush(QColor("#04070D"))

    # ------------------------------------------------------------------
    # Zoom molette
    # ------------------------------------------------------------------

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    # ------------------------------------------------------------------
    # Pan : bouton milieu ou espace + clic gauche
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if self._scene.is_drawing():
            sp = self.mapToScene(event.pos())
            self._scene.handle_scene_press(sp, event.button())
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if self._scene.is_drawing():
            sp = self.mapToScene(event.pos())
            self._scene.handle_scene_dbl_click(sp)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


# ---------------------------------------------------------------------------
# Widget principal — FCSCanvas
# ---------------------------------------------------------------------------

class FCSCanvas(QWidget):
    """
    Widget de visualisation FCS haute performance.

    Signals:
        gate_created(gate_id, name, scene_pts, data_pts):
            Émis quand l'utilisateur ferme un polygone.
            data_pts = liste (x_data, y_data) des sommets en coordonnées données.

    API publique :
        load_sample(sample)         — charge un Sample, conserve les axes
        set_axes(x_ch, y_ch)        — sélectionne les canaux X/Y et redessine
        start_gate_drawing(name)    — active mode dessin (Escape annule)
        add_gate(gate)              — ajoute une PolygonGate pré-construite
        remove_gate(gate_id)        — retire une gate de la scène
        recompute_masks()           — recalcule tous les masques dans le Sample
        clear()                     — efface tout
    """

    gate_created = pyqtSignal(str, str, list, list)
    # (gate_id, name, scene_pts: list[QPointF], data_pts: list[tuple[float,float]])

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._sample: Optional["Sample"] = None
        self._x_channel: str = ""
        self._y_channel: str = ""
        self._next_gate_name: str = "Gate"
        self._gates: Dict[str, PolygonGate] = {}

        self._scene = FCSScene(self)
        self._view = FCSView(self._scene, self)

        from PyQt5.QtWidgets import QVBoxLayout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        self._scene.gate_drawing_finished.connect(self._on_gate_drawn)

        # Transformée données → pixels (affine, mise à jour par _update_transform)
        self._data_min: np.ndarray = np.array([0.0, 0.0])
        self._data_max: np.ndarray = np.array([1.0, 1.0])

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def load_sample(self, sample: "Sample") -> None:
        """Charge un Sample et redessine si les axes sont déjà définis."""
        self._sample = sample
        if self._x_channel and self._y_channel:
            self._refresh()

    def set_axes(self, x_channel: str, y_channel: str) -> None:
        """Sélectionne les canaux X/Y et redessine."""
        self._x_channel = x_channel
        self._y_channel = y_channel
        if self._sample is not None:
            self._refresh()

    def start_gate_drawing(self, name: str = "Gate") -> None:
        """Active le mode dessin de porte. Escape annule."""
        self._next_gate_name = name
        self._scene.start_gate_drawing()
        self._view.setCursor(Qt.CrossCursor)

    def add_gate(self, gate: "PolygonGate") -> None:
        """Ajoute une PolygonGate pré-construite à la scène."""
        self._gates[gate.gate_id] = gate
        if gate.scene() is None:
            self._scene.addItem(gate)
        gate.setZValue(5)

    def remove_gate(self, gate_id: str) -> None:
        """Retire une gate de la scène et du registre interne."""
        gate = self._gates.pop(gate_id, None)
        if gate is not None and gate.scene() is not None:
            self._scene.removeItem(gate)

    def recompute_masks(self) -> None:
        """Recalcule tous les masques de gate dans le Sample courant."""
        if self._sample is None:
            return
        for gate_id, gate in self._gates.items():
            if gate_id == "__preview__":
                continue
            mask = gate.compute_mask(
                self._sample.events,
                self._x_channel,
                self._y_channel,
            )
            self._sample.set_mask(gate.name, mask)

    def clear(self) -> None:
        """Efface sample, axes et gates."""
        self._sample = None
        self._x_channel = ""
        self._y_channel = ""
        for gate in list(self._gates.values()):
            if gate.scene() is not None:
                self._scene.removeItem(gate)
        self._gates.clear()
        self._scene.set_scatter(np.array([]), np.array([]))

    # ------------------------------------------------------------------
    # Rafraîchissement interne
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        """Recalcule les coordonnées pixel et redessine le scatter."""
        if self._sample is None:
            return
        try:
            data = self._sample.get_data([self._x_channel, self._y_channel])
        except KeyError:
            return

        xd = data[self._x_channel].to_numpy(dtype=np.float32)
        yd = data[self._y_channel].to_numpy(dtype=np.float32)

        # Sous-échantillonnage adaptatif
        n = len(xd)
        if n > MAX_RENDER_PTS:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(n, MAX_RENDER_PTS, replace=False)
            xd, yd = xd[idx], yd[idx]

        self._update_transform(xd, yd)
        px, py = self._data_to_pixel(xd, yd)
        self._scene.set_scatter(px, py)

    def _update_transform(self, xd: np.ndarray, yd: np.ndarray) -> None:
        if len(xd) == 0:
            return
        xmin, xmax = float(np.nanmin(xd)), float(np.nanmax(xd))
        ymin, ymax = float(np.nanmin(yd)), float(np.nanmax(yd))
        # Marge 5 %
        xrng = max(xmax - xmin, 1e-9)
        yrng = max(ymax - ymin, 1e-9)
        self._data_min = np.array([xmin - MARGIN * xrng, ymin - MARGIN * yrng])
        self._data_max = np.array([xmax + MARGIN * xrng, ymax + MARGIN * yrng])

    def _data_to_pixel(
        self, xd: np.ndarray, yd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convertit coordonnées données → pixels scène (vectorisé numpy)."""
        xspan = self._data_max[0] - self._data_min[0]
        yspan = self._data_max[1] - self._data_min[1]
        px = (xd - self._data_min[0]) / xspan * CANVAS_W
        # Y inversé : origine Qt en haut
        py = (1.0 - (yd - self._data_min[1]) / yspan) * CANVAS_H
        return px, py

    def _pixel_to_data(self, px: float, py: float) -> Tuple[float, float]:
        """Inverse : pixel scène → coordonnées données."""
        xspan = self._data_max[0] - self._data_min[0]
        yspan = self._data_max[1] - self._data_min[1]
        xd = px / CANVAS_W * xspan + self._data_min[0]
        yd = (1.0 - py / CANVAS_H) * yspan + self._data_min[1]
        return xd, yd

    # ------------------------------------------------------------------
    # Slot : gate fermée par l'utilisateur
    # ------------------------------------------------------------------

    def _on_gate_drawn(self, scene_pts: List[QPointF]) -> None:
        """Crée une PolygonGate depuis les points scène émis par FCSScene."""
        self._view.setCursor(Qt.ArrowCursor)
        gate_id = str(uuid.uuid4())
        name = self._next_gate_name

        gate = PolygonGate(
            gate_id=gate_id,
            name=name,
            scene_points=scene_pts,
            editable=True,
        )

        # Sommets en coordonnées données — requis par compute_mask
        data_pts = [
            self._pixel_to_data(pt.x(), pt.y()) for pt in scene_pts
        ]
        gate.set_data_verts(data_pts)

        self.add_gate(gate)

        if self._sample is not None:
            mask = gate.compute_mask(
                self._sample.events,
                self._x_channel,
                self._y_channel,
            )
            self._sample.set_mask(name, mask)

        self.gate_created.emit(gate_id, name, scene_pts, data_pts)

    # ------------------------------------------------------------------
    # Touches clavier
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self._scene.cancel_gate_drawing()
            self._view.setCursor(Qt.ArrowCursor)
        super().keyPressEvent(event)
