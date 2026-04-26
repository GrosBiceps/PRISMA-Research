# -*- coding: utf-8 -*-
"""
mrd_gauge.py — Widget gauge MRD par méthode.

Affiche pour une méthode donnée :
  - Nom de la méthode
    - Pourcentage MRD (curseur visuel + valeur numérique)
  - Badge POSITIF / NÉGATIF / INDÉTECTABLE
  - Nombre de nœuds MRD et cellules impliquées
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QFrame,
)
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QFont, QColor, QPainter, QPen


# Couleurs PRISMA v2
_RED = "#FF3D6E"
_YELLOW = "#FFE032"
_GREEN = "#39FF8A"
_BLUE = "#5BAAFF"
_SUBTEXT = "rgba(238,242,247,0.55)"
_SURFACE0 = "#101825"
_SURFACE1 = "#141E2E"
_BASE = "#0C1220"
_MANTLE = "#080D18"
_TEXT = "#EEF2F7"


class SemiCircleGauge(QWidget):
    """Jauge semi-circulaire compacte pour afficher une valeur entre 0 et 1000."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._value = 0
        self._color = QColor(_BLUE)
        self.setMinimumHeight(78)

    def set_value(self, value: int) -> None:
        self._value = max(0, min(1000, int(value)))
        self.update()

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = float(self.width())
        h = float(self.height())

        margin = 10.0
        cx = w / 2.0
        cy = h - 4.0
        radius = max(12.0, min((w - 2.0 * margin) / 2.0, h - 12.0))
        arc_rect = QRectF(cx - radius, cy - radius, 2.0 * radius, 2.0 * radius)

        bg_pen = QPen(QColor(24, 26, 42, 230), 9.0, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(bg_pen)
        painter.drawArc(arc_rect, 180 * 16, -180 * 16)

        ratio = self._value / 1000.0
        val_pen = QPen(self._color, 9.0, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(val_pen)
        painter.drawArc(arc_rect, 180 * 16, int(-180 * 16 * ratio))

        angle = math.pi * (1.0 - ratio)
        needle_inner = QPointF(
            cx + (radius - 18.0) * math.cos(angle), cy - (radius - 18.0) * math.sin(angle)
        )
        needle_outer = QPointF(
            cx + (radius - 4.0) * math.cos(angle), cy - (radius - 4.0) * math.sin(angle)
        )
        needle_pen = QPen(self._color, 3.0, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(needle_pen)
        painter.drawLine(needle_inner, needle_outer)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(205, 214, 244, 230))
        painter.drawEllipse(QPointF(cx, cy), 3.2, 3.2)


class MRDGauge(QWidget):
    """
    Widget compact affichant le résultat MRD pour une méthode.

    Usage :
        gauge = MRDGauge(method_name="ELN 2025")
        gauge.update_data({
            "pct": 0.42,
            "n_cells": 1234,
            "n_nodes": 3,
            "positive": True,
            "positivity_threshold": 0.1,  # optionnel
        })
    """

    def __init__(self, method_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.method_name = method_name
        self._build_ui()
        self._set_empty()

    def _build_ui(self) -> None:
        self.setObjectName("mrdGaugeCard")
        self._apply_card_style(status="waiting")

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(10)

        # ── Titre méthode ──
        self.lbl_method = QLabel(self.method_name)
        self.lbl_method.setFont(QFont("Consolas", 9, QFont.Bold))
        self.lbl_method.setStyleSheet(
            "color: #EEF2F7; background: transparent;"
            "letter-spacing: 0.12em; text-transform: uppercase;"
        )
        self.lbl_method.setAlignment(Qt.AlignCenter)
        root.addWidget(self.lbl_method)

        # ── Badge statut ──
        self.lbl_badge = QLabel("EN ATTENTE")
        self.lbl_badge.setAlignment(Qt.AlignCenter)
        self.lbl_badge.setFont(QFont("Consolas", 8, QFont.Bold))
        self.lbl_badge.setFixedHeight(22)
        self.lbl_badge.setStyleSheet(
            f"color: {_SUBTEXT}; background: rgba(20,30,46,0.8); "
            f"border: 1px solid rgba(255,255,255,0.08);"
            f"border-radius: 0px; padding: 0 10px; letter-spacing: 0.08em;"
        )
        root.addWidget(self.lbl_badge)

        # ── Jauge semi-circulaire ──
        self.arc = SemiCircleGauge()
        root.addWidget(self.arc)

        # ── Valeur MRD ──
        self.lbl_pct = QLabel("—")
        self.lbl_pct.setAlignment(Qt.AlignCenter)
        self.lbl_pct.setFont(QFont("Segoe UI", 26, QFont.Bold))
        self.lbl_pct.setStyleSheet(
            f"color: {_TEXT}; background: transparent; letter-spacing: -0.02em;"
        )
        root.addWidget(self.lbl_pct)

        # ── Curseur visuel (non interactif) ──
        self.cursor = QSlider(Qt.Horizontal)
        self.cursor.setRange(0, 1000)
        self.cursor.setValue(0)
        self.cursor.setFixedHeight(14)
        self.cursor.setFocusPolicy(Qt.NoFocus)
        self.cursor.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.cursor.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 5px;
                background: rgba(4, 7, 13, 0.92);
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: {_BLUE};
                border-radius: 2px;
            }}
            QSlider::add-page:horizontal {{
                background: rgba(4, 7, 13, 0.92);
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {_BLUE};
                border: 1px solid rgba(238, 242, 247, 0.35);
                width: 9px;
                margin: -5px 0;
                border-radius: 4px;
            }}
        """)
        root.addWidget(self.cursor)

        # ── Infos secondaires ──
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: rgba(137,180,250,0.08); max-height: 1px;")
        root.addWidget(sep)

        info_row = QHBoxLayout()
        info_row.setSpacing(12)

        self.lbl_nodes = QLabel("— nœuds")
        self.lbl_nodes.setStyleSheet(
            "color: #EEF2F7; background: transparent; font-size: 10px;"
            "font-weight: 600; font-family: 'Consolas', 'Cascadia Code', monospace;"
        )
        self.lbl_nodes.setAlignment(Qt.AlignCenter)

        self.lbl_cells = QLabel("— cellules")
        self.lbl_cells.setStyleSheet(
            "color: #EEF2F7; background: transparent; font-size: 10px;"
            "font-weight: 600; font-family: 'Consolas', 'Cascadia Code', monospace;"
        )
        self.lbl_cells.setAlignment(Qt.AlignCenter)

        info_row.addWidget(self.lbl_nodes)
        info_row.addWidget(self.lbl_cells)
        root.addLayout(info_row)

    def _apply_card_style(self, status: str = "waiting") -> None:
        if status == "positive":
            border_color = "rgba(255, 61, 110, 0.42)"
            bg_top = "rgba(34, 17, 26, 0.92)"
            bg_bot = "rgba(24, 12, 20, 0.92)"
            top_border = "rgba(255, 61, 110, 0.66)"
        elif status == "negative":
            border_color = "rgba(57, 255, 138, 0.35)"
            bg_top = "rgba(14, 36, 25, 0.90)"
            bg_bot = "rgba(10, 27, 19, 0.90)"
            top_border = "rgba(57, 255, 138, 0.58)"
        elif status == "low":
            border_color = "rgba(255, 224, 50, 0.35)"
            bg_top = "rgba(37, 32, 12, 0.90)"
            bg_bot = "rgba(28, 24, 9, 0.90)"
            top_border = "rgba(255, 224, 50, 0.58)"
        else:  # waiting
            border_color = "rgba(255, 255, 255, 0.08)"
            bg_top = "rgba(16, 24, 37, 0.90)"
            bg_bot = "rgba(12, 18, 32, 0.90)"
            top_border = "rgba(123, 82, 255, 0.45)"
        self.setStyleSheet(f"""
            QWidget#mrdGaugeCard {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {bg_top}, stop:1 {bg_bot});
                border-radius: 0px;
                border: 1px solid {border_color};
                border-top: 1px solid {top_border};
            }}
        """)

    # ------------------------------------------------------------------

    def update_data(self, data: Dict[str, Any]) -> None:
        """Met à jour l'affichage avec les données MRD de la méthode."""
        pct = data.get("pct", 0.0)
        n_nodes = data.get("n_nodes", 0)
        n_cells = data.get("n_cells", 0)
        positive = data.get("positive", False)
        low_level = data.get("low_level", False)
        threshold = data.get("positivity_threshold", None)

        # Valeur numérique
        self.lbl_pct.setText(f"{pct:.4f} %")

        # Barre (max représente 5%, clampé)
        bar_val = min(int(pct * 200), 1000)  # 5% → 1000
        self.arc.set_value(bar_val)
        self.cursor.setValue(bar_val)

        # Badge et couleurs
        if positive or n_nodes > 0:
            badge_text = "POSITIF"
            badge_color = _RED
            badge_bg = "rgba(255, 61, 110, 0.18)"
            badge_border = "rgba(255, 61, 110, 0.42)"
            bar_color = f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {_RED}, stop:1 #FF7294)"
            pct_color = "#FF89A8"
            card_status = "positive"
        elif low_level:
            badge_text = "INDÉTECTABLE"
            badge_color = _YELLOW
            badge_bg = "rgba(255, 224, 50, 0.15)"
            badge_border = "rgba(255, 224, 50, 0.35)"
            bar_color = f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {_YELLOW}, stop:1 #FFB84A)"
            pct_color = _YELLOW
            card_status = "low"
        else:
            badge_text = "NÉGATIF"
            badge_color = _GREEN
            badge_bg = "rgba(57, 255, 138, 0.16)"
            badge_border = "rgba(57, 255, 138, 0.38)"
            bar_color = f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {_GREEN}, stop:1 #78FFB2)"
            pct_color = "#85FFC0"
            card_status = "negative"

        self._apply_card_style(card_status)
        self.lbl_pct.setStyleSheet(
            f"color: {pct_color}; background: transparent; letter-spacing: -0.02em;"
        )
        self.lbl_badge.setText(badge_text)
        self.lbl_badge.setStyleSheet(
            f"color: {badge_color}; background: {badge_bg}; "
            f"border: 1px solid {badge_border}; "
            f"border-radius: 0px; padding: 0 10px; font-weight: bold; letter-spacing: 0.1em;"
        )
        self.arc.set_color(pct_color)
        self.cursor.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 5px;
                background: rgba(4, 7, 13, 0.92);
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: {bar_color};
                border-radius: 2px;
            }}
            QSlider::add-page:horizontal {{
                background: rgba(4, 7, 13, 0.92);
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {pct_color};
                border: 1px solid rgba(238, 242, 247, 0.45);
                width: 9px;
                margin: -5px 0;
                border-radius: 4px;
            }}
        """)

        # Infos secondaires
        info_color = "rgba(238,242,247,0.50)"
        seuil_txt = f" (seuil : {threshold}%)" if threshold else ""
        self.lbl_nodes.setText(f"{n_nodes} nœud{'s' if n_nodes != 1 else ''} MRD{seuil_txt}")
        self.lbl_nodes.setStyleSheet(
            f"color: {info_color}; background: transparent; font-size: 10px; font-weight: 600;"
            "font-family: 'Consolas', 'Cascadia Code', monospace;"
        )
        self.lbl_cells.setText(f"{n_cells:,} cellule{'s' if n_cells != 1 else ''}")
        self.lbl_cells.setStyleSheet(
            f"color: {info_color}; background: transparent; font-size: 10px; font-weight: 600;"
            "font-family: 'Consolas', 'Cascadia Code', monospace;"
        )

    def _set_empty(self) -> None:
        """Affichage vide avant le premier résultat."""
        self.lbl_pct.setText("—")
        self.lbl_badge.setText("EN ATTENTE")
        self.lbl_badge.setStyleSheet(
            f"color: {_SUBTEXT}; background: {_SURFACE1}; border: 1px solid rgba(255,255,255,0.08);"
            f"border-radius: 0px; padding: 0 8px;"
        )
        self.arc.set_value(0)
        self.arc.set_color(_BLUE)
        self.cursor.setValue(0)
        self.lbl_nodes.setText("— nœuds")
        self.lbl_cells.setText("— cellules")


