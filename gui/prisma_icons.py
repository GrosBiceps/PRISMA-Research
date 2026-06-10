"""PRISMA iconography pack — v2.0 Brand Identity System.

All icons are reproduced faithfully from the PRISMA HTML brand guide:
  - 36×36 viewport, thin stroke 1–1.5 px
  - Scientific / cytometric vocabulary
  - Channel palette: V450=#7B52FF, FITC=#39FF8A, PE=#FF9B3D, APC=#FF3D6E, V500=#5BAAFF
  - Stroke-based, never filled shapes — selective color accents only
  - RoundCap + RoundJoin on all strokes
"""

from __future__ import annotations

from typing import Callable, Dict

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF

_ICON_CACHE: Dict[str, QIcon] = {}

# ── PRISMA channel palette ────────────────────────────────────────────────────
_V450  = QColor("#7B52FF")   # Brand / Lymphocytes
_FITC  = QColor("#39FF8A")   # Accent / Monocytes
_PE    = QColor("#FF9B3D")   # Warm / Granulocytes
_APC   = QColor("#FF3D6E")   # Danger / MRD
_V500  = QColor("#5BAAFF")   # Info / T-cells
_PERCP = QColor("#FFE032")   # Warn
_SSC   = QColor("#7EC8E3")   # Steel / debris
_PAPER = QColor("#EEF2F7")   # Default stroke
_DIM   = QColor(255, 255, 255, 28)


def _c(hex_color: str) -> QColor:
    return QColor(hex_color)


def _pen(color: QColor, w: float = 1.0, cap=Qt.RoundCap, join=Qt.RoundJoin) -> QPen:
    pen = QPen(color)
    pen.setWidthF(w)
    pen.setCapStyle(cap)
    pen.setJoinStyle(join)
    return pen


def _pen_dim(alpha: int = 28, w: float = 0.8) -> QPen:
    c = QColor(255, 255, 255, alpha)
    return _pen(c, w)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dot(p: QPainter, color: QColor, x: float, y: float, r: float, alpha: int = 255) -> None:
    c = QColor(color)
    c.setAlpha(alpha)
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(c))
    p.drawEllipse(QPointF(x, y), r, r)


def _axes(p: QPainter, s: int) -> None:
    p.setPen(_pen(_DIM, 0.8))
    p.drawLine(QPointF(s * 0.12, s * 0.88), QPointF(s * 0.88, s * 0.88))  # X
    p.drawLine(QPointF(s * 0.12, s * 0.88), QPointF(s * 0.12, s * 0.12))  # Y


# ══════════════════════════════════════════════════════════════════════════════
# ICONS — reproduced from prisma-branding-v2.html §04 Iconography
# ══════════════════════════════════════════════════════════════════════════════

def _draw_dot_plot(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    """Scatter plot with three color-coded populations + MRD outlier."""
    _axes(p, s)

    # Lymphocyte cluster — V450
    for cx, cy, r, a in (
        (0.32, 0.64, 2.1, 204), (0.40, 0.54, 1.9, 153),
        (0.27, 0.70, 1.5, 127), (0.44, 0.59, 1.3, 102),
    ):
        _dot(p, _V450, s * cx, s * cy, r, a)

    # Monocyte cluster — FITC
    for cx, cy, r, a in (
        (0.52, 0.40, 2.5, 204), (0.60, 0.32, 1.9, 153),
        (0.48, 0.47, 1.5, 127), (0.63, 0.38, 1.2, 127),
    ):
        _dot(p, _FITC, s * cx, s * cy, r, a)

    # Granulocyte cluster — PE
    for cx, cy, r, a in (
        (0.72, 0.72, 2.5, 224), (0.79, 0.63, 2.0, 178),
        (0.67, 0.77, 1.5, 140),
    ):
        _dot(p, _PE, s * cx, s * cy, r, a)

    # MRD rare population — APC (isolated, small)
    for cx, cy, r, a in (
        (0.42, 0.22, 1.9, 242), (0.48, 0.16, 1.3, 204),
        (0.37, 0.26, 1.1, 178),
    ):
        _dot(p, _APC, s * cx, s * cy, r, a)

    # MRD callout line
    p.setPen(_pen(_APC, 0.7))
    p.drawLine(QPointF(s * 0.46, s * 0.18), QPointF(s * 0.56, s * 0.12))


def _draw_som_grid(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    """3×3 SOM grid with active node highlights per channel."""
    m = s * 0.10
    cell = (s - 2 * m) / 3.0

    # Grid lines
    p.setPen(_pen(QColor(255, 255, 255, 25), 0.8))
    for i in range(4):
        x = m + i * cell
        y = m + i * cell
        p.drawLine(QPointF(x, m), QPointF(x, m + 3 * cell))
        p.drawLine(QPointF(m, y), QPointF(m + 3 * cell, y))

    # Active cells — filled with channel tint
    def _cell_fill(r: int, c: int, col: QColor, alpha: int = 90) -> None:
        fc = QColor(col)
        fc.setAlpha(alpha)
        x = m + c * cell + 0.5
        y = m + r * cell + 0.5
        p.fillRect(QRectF(x, y, cell - 1, cell - 1), fc)

    _cell_fill(1, 0, _V450, 89)
    _cell_fill(0, 1, _FITC, 114)
    _cell_fill(2, 2, _PE,   89)

    # Node dots
    def _node(r: int, c: int, col: QColor) -> None:
        cx = m + (c + 0.5) * cell
        cy = m + (r + 0.5) * cell
        _dot(p, col, cx, cy, 1.8)

    _node(1, 0, _V450)
    _node(0, 1, _FITC)
    _node(2, 2, _PE)


def _draw_mrd_kinetics(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    """Kinetics line chart with MRD detection point (APC dot + dashed drop)."""
    _axes(p, s)

    pts = [
        QPointF(s * 0.18, s * 0.72),
        QPointF(s * 0.32, s * 0.64),
        QPointF(s * 0.44, s * 0.65),
        QPointF(s * 0.58, s * 0.48),
        QPointF(s * 0.70, s * 0.50),
        QPointF(s * 0.82, s * 0.30),
        QPointF(s * 0.88, s * 0.32),
    ]

    p.setPen(_pen(_FITC, 1.4))
    for i in range(len(pts) - 1):
        p.drawLine(pts[i], pts[i + 1])

    # MRD detect point — APC
    end = pts[-1]
    _dot(p, _APC, end.x(), end.y(), 3.0)

    # Dashed drop line
    pen_dash = _pen(_APC, 0.7)
    pen_dash.setStyle(Qt.DashLine)
    p.setPen(pen_dash)
    p.drawLine(end, QPointF(end.x(), s * 0.88))


def _draw_cell_node(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    """Cell-node network: outer ring, dashed orbit, spokes, satellite nodes."""
    cx, cy = s * 0.5, s * 0.5

    # Outer ring — dim
    p.setPen(_pen(QColor(255, 255, 255, 30), 1.0))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(cx, cy), s * 0.40, s * 0.40)

    # Dashed orbit — FITC
    dash_pen = _pen(_FITC, 1.0)
    dash_pen.setStyle(Qt.DashLine)
    dash_pen.setDashPattern([2.5, 2.0])
    p.setPen(dash_pen)
    p.drawEllipse(QPointF(cx, cy), s * 0.22, s * 0.22)

    # Spokes
    satellites = [
        (0.50, 0.10), (0.82, 0.68), (0.18, 0.68),
    ]
    p.setPen(_pen(QColor(_FITC.red(), _FITC.green(), _FITC.blue(), 64), 0.8))
    for sx, sy in satellites:
        p.drawLine(QPointF(cx, cy), QPointF(s * sx, s * sy))

    # Satellite dots
    for sx, sy in satellites:
        _dot(p, _FITC, s * sx, s * sy, 1.8, 127)

    # Centre dot — FITC solid
    _dot(p, _FITC, cx, cy, 2.5)


def _draw_metacluster(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    """Hierarchical dendrogram: root → two branches → four leaves."""
    root  = QPointF(s * 0.50, s * 0.18)
    left  = QPointF(s * 0.28, s * 0.50)
    right = QPointF(s * 0.72, s * 0.50)
    ll    = QPointF(s * 0.18, s * 0.80)
    lr    = QPointF(s * 0.38, s * 0.80)
    rl    = QPointF(s * 0.62, s * 0.80)
    rr    = QPointF(s * 0.82, s * 0.80)

    # Root circle
    p.setPen(_pen(QColor(255, 255, 255, 64), 1.0))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(root, 2.5, 2.5)

    # Branch connectors
    p.setPen(_pen(QColor(255, 255, 255, 38), 0.8))
    for a, b in ((root, left), (root, right)):
        p.drawLine(a, b)

    # Left sub-branch — V450
    p.setPen(_pen(QColor(_V450.red(), _V450.green(), _V450.blue(), 77), 0.8))
    for a, b in ((left, ll), (left, lr)):
        p.drawLine(a, b)
    p.setPen(_pen(_V450, 1.0))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(left, 2.2, 2.2)

    # Right sub-branch — FITC
    p.setPen(_pen(QColor(_FITC.red(), _FITC.green(), _FITC.blue(), 77), 0.8))
    for a, b in ((right, rl), (right, rr)):
        p.drawLine(a, b)
    p.setPen(_pen(_FITC, 1.0))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(right, 2.2, 2.2)

    # Leaf dots
    for pt in (ll, lr):
        _dot(p, _V450, pt.x(), pt.y(), 1.8, 153)
    for pt in (rl, rr):
        _dot(p, _FITC, pt.x(), pt.y(), 1.8, 153)


def _draw_gate_strategy(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    """Gating: three concentric dashed ellipses + MRD core."""
    cx, cy = s * 0.50, s * 0.52

    def _ellipse_dash(rx: float, ry: float, col: QColor, alpha: int, w: float) -> None:
        c = QColor(col); c.setAlpha(alpha)
        pen = _pen(c, w); pen.setStyle(Qt.DashLine); pen.setDashPattern([3.0, 2.0])
        p.setPen(pen); p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), rx, ry)

    _ellipse_dash(s * 0.38, s * 0.30, _V500, 77, 1.1)   # outer — V500
    _ellipse_dash(s * 0.24, s * 0.18, _FITC, 97, 1.1)   # middle — FITC
    # Inner solid — APC
    p.setPen(_pen(_APC, 1.1)); p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(cx, cy), s * 0.13, s * 0.10)
    _dot(p, _APC, cx, cy, 2.2)


def _draw_fcs_file(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    """FCS file icon: document with folded corner, data lines, .FCS badge."""
    # Document body
    p.setPen(_pen(QColor(255, 255, 255, 46), 1.0))
    p.setBrush(Qt.NoBrush)
    # Body rect (leave space for fold)
    pts = [
        QPointF(s * 0.22, s * 0.10),
        QPointF(s * 0.64, s * 0.10),
        QPointF(s * 0.78, s * 0.24),
        QPointF(s * 0.78, s * 0.88),
        QPointF(s * 0.22, s * 0.88),
    ]
    poly = QPolygonF(pts)
    p.drawPolygon(poly)
    # Fold crease
    p.drawLine(QPointF(s * 0.64, s * 0.10), QPointF(s * 0.64, s * 0.24))
    p.drawLine(QPointF(s * 0.64, s * 0.24), QPointF(s * 0.78, s * 0.24))

    # Data lines
    p.setPen(_pen(_FITC, 1.0))
    p.drawLine(QPointF(s * 0.32, s * 0.44), QPointF(s * 0.68, s * 0.44))
    p.setPen(_pen(QColor(255, 255, 255, 51), 0.9))
    p.drawLine(QPointF(s * 0.32, s * 0.54), QPointF(s * 0.56, s * 0.54))
    p.drawLine(QPointF(s * 0.32, s * 0.62), QPointF(s * 0.48, s * 0.62))

    # .FCS badge (small filled rect)
    badge_c = QColor(_FITC); badge_c.setAlpha(38)
    p.fillRect(QRectF(s * 0.14, s * 0.68, s * 0.40, s * 0.22), badge_c)
    p.setPen(_pen(_FITC, 0.8))
    p.drawRect(QRectF(s * 0.14, s * 0.68, s * 0.40, s * 0.22))


def _draw_umap_embed(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    """UMAP: three rotated population ellipses with channel colors."""
    from PyQt5.QtGui import QTransform

    def _rotated_ellipse(cx: float, cy: float, rx: float, ry: float,
                         angle: float, col: QColor, alpha: int) -> None:
        c = QColor(col); c.setAlpha(alpha)
        pen = _pen(c, 1.1); p.setPen(pen); p.setBrush(Qt.NoBrush)
        tr = QTransform()
        tr.translate(cx, cy)
        tr.rotate(angle)
        tr.translate(-cx, -cy)
        p.setTransform(tr)
        p.drawEllipse(QPointF(cx, cy), rx, ry)
        p.resetTransform()

    _rotated_ellipse(s * 0.34, s * 0.60, s * 0.16, s * 0.11, -20, _V450, 89)
    _rotated_ellipse(s * 0.60, s * 0.40, s * 0.14, s * 0.10,  15, _FITC, 89)
    _rotated_ellipse(s * 0.70, s * 0.70, s * 0.11, s * 0.08, -10, _PE,   89)

    _dot(p, _V450, s * 0.34, s * 0.60, 1.8, 204)
    _dot(p, _FITC, s * 0.60, s * 0.40, 1.8, 204)
    _dot(p, _PE,   s * 0.70, s * 0.70, 1.8, 204)


def _draw_heatmap(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    """3×3 heatmap — cytometric channel color scale."""
    m = s * 0.08
    cell = (s - 2 * m) / 3.0

    CELLS = [
        # (row, col, color, alpha)
        (0, 0, _V450, 204), (0, 1, _V450, 114), (0, 2, _V500, 77),
        (1, 0, _FITC, 127), (1, 1, _PERCP, 140), (1, 2, _PE,  166),
        (2, 0, _FITC, 51),  (2, 1, _PE,   77),   (2, 2, _APC, 191),
    ]
    for row, col, col_c, alpha in CELLS:
        x = m + col * cell
        y = m + row * cell
        c = QColor(col_c); c.setAlpha(alpha)
        p.fillRect(QRectF(x + 0.5, y + 0.5, cell - 1, cell - 1), c)

    # Grid overlay
    p.setPen(_pen(QColor(255, 255, 255, 15), 0.6))
    for i in range(4):
        v = m + i * cell
        p.drawLine(QPointF(m, v), QPointF(m + 3 * cell, v))
        p.drawLine(QPointF(v, m), QPointF(v, m + 3 * cell))


def _draw_mst_tree(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    """Minimum spanning tree — radial hub with 5 colored satellite nodes."""
    hub = QPointF(s * 0.50, s * 0.50)
    satellites = [
        (0.22, 0.26, _V450),
        (0.78, 0.26, _FITC),
        (0.22, 0.74, _PE),
        (0.78, 0.74, _APC),
        (0.50, 0.14, _V500),
    ]

    # Connector lines
    p.setPen(_pen(QColor(255, 255, 255, 38), 1.0))
    for sx, sy, _ in satellites:
        p.drawLine(hub, QPointF(s * sx, s * sy))

    # Hub — white semi-transparent
    _dot(p, QColor(255, 255, 255, 140), hub.x(), hub.y(), 2.2)

    # Satellite dots — channel colors
    for sx, sy, col in satellites:
        _dot(p, col, s * sx, s * sy, 1.8, 178)


def _draw_batch_cohort(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    """Stacked FCS files — three offset rectangles with a bottom line."""
    offsets = [(0.04, 0.12), (0.14, 0.22), (0.26, 0.32)]
    alphas  = [25, 30, 77]
    strokes = [25, 30, 77]

    for i, (ox, oy) in enumerate(offsets):
        col = QColor(_FITC if i == 2 else _PAPER)
        col.setAlpha(strokes[i])
        p.setPen(_pen(col, 1.0))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(s * ox, s * oy, s * 0.55, s * 0.38))

    # Data line on front card
    p.setPen(_pen(_FITC, 1.0))
    p.drawLine(QPointF(s * 0.34, s * 0.46), QPointF(s * 0.72, s * 0.46))
    p.setPen(_pen(QColor(255, 255, 255, 46), 0.9))
    p.drawLine(QPointF(s * 0.34, s * 0.54), QPointF(s * 0.60, s * 0.54))

    # Bottom rule
    p.setPen(_pen(_FITC, 1.0))
    p.drawLine(QPointF(s * 0.10, s * 0.82), QPointF(s * 0.90, s * 0.82))


def _draw_export_fig(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    """Export: rectangle frame + upward arrow + baseline."""
    p.setPen(_pen(QColor(255, 255, 255, 30), 1.0))
    p.setBrush(Qt.NoBrush)
    p.drawRect(QRectF(s * 0.16, s * 0.14, s * 0.68, s * 0.62))

    # Arrow shaft — V450
    p.setPen(_pen(_V450, 1.4))
    p.drawLine(QPointF(s * 0.50, s * 0.78), QPointF(s * 0.50, s * 0.34))
    # Arrow head
    p.drawLine(QPointF(s * 0.36, s * 0.48), QPointF(s * 0.50, s * 0.34))
    p.drawLine(QPointF(s * 0.64, s * 0.48), QPointF(s * 0.50, s * 0.34))

    # Baseline
    p.setPen(_pen(QColor(255, 255, 255, 51), 1.0))
    p.drawLine(QPointF(s * 0.28, s * 0.78), QPointF(s * 0.72, s * 0.78))


# ── Utility icons (UI chrome) ─────────────────────────────────────────────────

def _draw_check_circle(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    p.setPen(_pen(_FITC, 1.4))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.34, s * 0.34)
    p.drawLine(QPointF(s * 0.34, s * 0.52), QPointF(s * 0.46, s * 0.64))
    p.drawLine(QPointF(s * 0.46, s * 0.64), QPointF(s * 0.70, s * 0.38))


def _draw_alert_triangle(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    p.setPen(_pen(accent, 1.4))
    p.setBrush(Qt.NoBrush)
    pts = [QPointF(s * 0.50, s * 0.16),
           QPointF(s * 0.82, s * 0.80),
           QPointF(s * 0.18, s * 0.80)]
    p.drawPolygon(QPolygonF(pts))
    p.drawLine(QPointF(s * 0.50, s * 0.36), QPointF(s * 0.50, s * 0.58))
    _dot(p, accent, s * 0.50, s * 0.69, 1.4)


def _draw_arrow_right(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    p.setPen(_pen(color, 1.6))
    p.drawLine(QPointF(s * 0.18, s * 0.50), QPointF(s * 0.78, s * 0.50))
    p.drawLine(QPointF(s * 0.62, s * 0.34), QPointF(s * 0.78, s * 0.50))
    p.drawLine(QPointF(s * 0.62, s * 0.66), QPointF(s * 0.78, s * 0.50))


def _draw_arrow_left(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    p.setPen(_pen(color, 1.6))
    p.drawLine(QPointF(s * 0.22, s * 0.50), QPointF(s * 0.82, s * 0.50))
    p.drawLine(QPointF(s * 0.38, s * 0.34), QPointF(s * 0.22, s * 0.50))
    p.drawLine(QPointF(s * 0.38, s * 0.66), QPointF(s * 0.22, s * 0.50))


def _draw_play(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    c = QColor(color); c.setAlpha(230)
    p.setPen(_pen(c, 1.2))
    p.setBrush(QBrush(c))
    pts = [QPointF(s * 0.34, s * 0.24),
           QPointF(s * 0.76, s * 0.50),
           QPointF(s * 0.34, s * 0.76)]
    p.drawPolygon(QPolygonF(pts))


def _draw_stop_circle(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    p.setPen(_pen(color, 1.4))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.34, s * 0.34)
    p.fillRect(QRectF(s * 0.38, s * 0.38, s * 0.24, s * 0.24), color)


def _draw_copy(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    p.setPen(_pen(color, 1.2))
    p.setBrush(Qt.NoBrush)
    p.drawRect(QRectF(s * 0.20, s * 0.28, s * 0.42, s * 0.46))
    p.drawRect(QRectF(s * 0.36, s * 0.16, s * 0.42, s * 0.46))


def _draw_folder_open(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    p.setPen(_pen(color, 1.3))
    p.setBrush(Qt.NoBrush)
    # Folder back
    p.drawLine(QPointF(s * 0.10, s * 0.36), QPointF(s * 0.34, s * 0.36))
    p.drawLine(QPointF(s * 0.34, s * 0.36), QPointF(s * 0.42, s * 0.24))
    p.drawLine(QPointF(s * 0.42, s * 0.24), QPointF(s * 0.88, s * 0.24))
    # Folder body
    p.drawLine(QPointF(s * 0.10, s * 0.40), QPointF(s * 0.90, s * 0.40))
    p.drawLine(QPointF(s * 0.16, s * 0.76), QPointF(s * 0.84, s * 0.76))
    p.drawLine(QPointF(s * 0.10, s * 0.40), QPointF(s * 0.16, s * 0.76))
    p.drawLine(QPointF(s * 0.90, s * 0.40), QPointF(s * 0.84, s * 0.76))
    # Accent line
    p.setPen(_pen(accent, 1.2))
    p.drawLine(QPointF(s * 0.24, s * 0.58), QPointF(s * 0.76, s * 0.58))


def _draw_external_link(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    p.setPen(_pen(color, 1.2))
    p.setBrush(Qt.NoBrush)
    p.drawRect(QRectF(s * 0.18, s * 0.30, s * 0.48, s * 0.50))
    p.setPen(_pen(accent, 1.5))
    p.drawLine(QPointF(s * 0.44, s * 0.24), QPointF(s * 0.82, s * 0.24))
    p.drawLine(QPointF(s * 0.82, s * 0.24), QPointF(s * 0.82, s * 0.62))
    p.drawLine(QPointF(s * 0.50, s * 0.56), QPointF(s * 0.82, s * 0.24))
    p.drawLine(QPointF(s * 0.66, s * 0.24), QPointF(s * 0.82, s * 0.24))
    p.drawLine(QPointF(s * 0.82, s * 0.24), QPointF(s * 0.82, s * 0.40))


def _draw_search(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    p.setPen(_pen(color, 1.4))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(s * 0.42, s * 0.42), s * 0.20, s * 0.20)
    p.drawLine(QPointF(s * 0.57, s * 0.57), QPointF(s * 0.82, s * 0.82))


def _draw_plus(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    p.setPen(_pen(color, 1.6))
    p.drawLine(QPointF(s * 0.50, s * 0.22), QPointF(s * 0.50, s * 0.78))
    p.drawLine(QPointF(s * 0.22, s * 0.50), QPointF(s * 0.78, s * 0.50))


def _draw_trash(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    p.setPen(_pen(color, 1.2))
    p.setBrush(Qt.NoBrush)
    p.drawRect(QRectF(s * 0.28, s * 0.32, s * 0.44, s * 0.48))
    p.drawLine(QPointF(s * 0.22, s * 0.32), QPointF(s * 0.78, s * 0.32))
    p.drawLine(QPointF(s * 0.38, s * 0.24), QPointF(s * 0.62, s * 0.24))
    for x in (0.40, 0.50, 0.60):
        p.drawLine(QPointF(s * x, s * 0.42), QPointF(s * x, s * 0.70))


def _draw_eraser(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    p.setPen(_pen(color, 1.3))
    p.setBrush(Qt.NoBrush)
    p.drawLine(QPointF(s * 0.20, s * 0.68), QPointF(s * 0.46, s * 0.36))
    p.drawLine(QPointF(s * 0.46, s * 0.36), QPointF(s * 0.74, s * 0.64))
    p.drawLine(QPointF(s * 0.74, s * 0.64), QPointF(s * 0.48, s * 0.86))
    p.drawLine(QPointF(s * 0.48, s * 0.86), QPointF(s * 0.20, s * 0.68))
    p.setPen(_pen(accent, 1.2))
    p.drawLine(QPointF(s * 0.28, s * 0.76), QPointF(s * 0.64, s * 0.76))


def _draw_check(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    p.setPen(_pen(color, 1.6))
    p.drawLine(QPointF(s * 0.24, s * 0.54), QPointF(s * 0.42, s * 0.72))
    p.drawLine(QPointF(s * 0.42, s * 0.72), QPointF(s * 0.78, s * 0.30))


def _draw_sync(p: QPainter, color: QColor, accent: QColor, s: int) -> None:
    p.setPen(_pen(color, 1.3))
    p.setBrush(Qt.NoBrush)
    p.drawArc(QRectF(s * 0.18, s * 0.18, s * 0.44, s * 0.44), 40 * 16, 250 * 16)
    p.drawArc(QRectF(s * 0.38, s * 0.38, s * 0.44, s * 0.44), 220 * 16, 250 * 16)
    p.setPen(_pen(accent, 1.4))
    p.drawLine(QPointF(s * 0.60, s * 0.18), QPointF(s * 0.72, s * 0.16))
    p.drawLine(QPointF(s * 0.60, s * 0.18), QPointF(s * 0.66, s * 0.28))
    p.drawLine(QPointF(s * 0.40, s * 0.82), QPointF(s * 0.28, s * 0.84))
    p.drawLine(QPointF(s * 0.40, s * 0.82), QPointF(s * 0.34, s * 0.72))


# ── Registry ──────────────────────────────────────────────────────────────────
_DRAWERS: Dict[str, Callable[[QPainter, QColor, QColor, int], None]] = {
    # Scientific / cytometric
    "dot-plot":       _draw_dot_plot,
    "som-grid":       _draw_som_grid,
    "mrd-kinetics":   _draw_mrd_kinetics,
    "cell-node":      _draw_cell_node,
    "metacluster":    _draw_metacluster,
    "gate-strategy":  _draw_gate_strategy,
    "fcs-file":       _draw_fcs_file,
    "umap-embed":     _draw_umap_embed,
    "heatmap":        _draw_heatmap,
    "mst-tree":       _draw_mst_tree,
    "batch-cohort":   _draw_batch_cohort,
    "export-fig":     _draw_export_fig,
    # UI chrome
    "check-circle":   _draw_check_circle,
    "alert-triangle": _draw_alert_triangle,
    "arrow-right":    _draw_arrow_right,
    "arrow-left":     _draw_arrow_left,
    "play":           _draw_play,
    "stop-circle":    _draw_stop_circle,
    "copy":           _draw_copy,
    "folder-open":    _draw_folder_open,
    "external-link":  _draw_external_link,
    "search":         _draw_search,
    "plus":           _draw_plus,
    "trash":          _draw_trash,
    "eraser":         _draw_eraser,
    "check":          _draw_check,
    "sync":           _draw_sync,
}


def get_prisma_icon(name: str, size: int = 18, color: str = "#EEF2F7") -> QIcon | None:
    """Return a PRISMA icon by name, rendered at *size* px square.

    Args:
        name:  Key from _DRAWERS (e.g. ``"dot-plot"``, ``"mrd-kinetics"``).
        size:  Square canvas size in device pixels.
        color: Hex string for the primary/stroke color.
    """
    drawer = _DRAWERS.get(name)
    if drawer is None:
        return None

    key = f"{name}:{size}:{color}"
    cached = _ICON_CACHE.get(key)
    if cached is not None:
        return cached

    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

    base   = _c(color)
    accent = _FITC  # PRISMA default accent — always FITC green
    drawer(painter, base, accent, size)

    painter.end()
    icon = QIcon(pix)
    _ICON_CACHE[key] = icon
    return icon
