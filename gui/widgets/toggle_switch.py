# -*- coding: utf-8 -*-
"""
toggle_switch.py — Toggle Switch iOS/Android style pour PyQt5.

Remplace les QCheckBox standard par un widget moderne paint-based.
Compatible avec le thème "Deep Medical Clarity".
"""

from __future__ import annotations

from PyQt5.QtCore import QEasingCurve, QPropertyAnimation, Qt, pyqtProperty, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget


class ToggleSwitch(QWidget):
    """
    Toggle Switch style iOS — dessiné via QPainter.

    Signaux :
        toggled(bool) — émis lors d'un changement d'état
    """

    toggled = pyqtSignal(bool)

    # Couleurs du thème Deep Medical Clarity
    _COLOR_ON = QColor("#00A3FF")        # Bleu technologie
    _COLOR_ON_HOVER = QColor("#33B8FF")
    _COLOR_OFF = QColor("#313244")       # Surface0 Catppuccin
    _COLOR_OFF_HOVER = QColor("#45475a")
    _COLOR_KNOB = QColor("#e8eeff")
    _COLOR_BORDER_ON = QColor("#0085D6")
    _COLOR_BORDER_OFF = QColor("#45475a")

    def __init__(
        self,
        label: str = "",
        checked: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._checked = checked
        self._hovered = False
        self._label_text = label

        # Animation de position du knob (0.0 → 1.0)
        self._knob_pos: float = 1.0 if checked else 0.0

        self._anim = QPropertyAnimation(self, b"knob_pos", self)
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

        # Layout : switch + label optionnel
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Placeholder pour le switch (peint dans paintEvent)
        self._switch_placeholder = QWidget(self)
        self._switch_placeholder.setFixedSize(44, 24)
        layout.addWidget(self._switch_placeholder)

        if label:
            self._lbl = QLabel(label, self)
            self._lbl.setStyleSheet(
                "color: #b8c2e8; font-size: 10pt; background: transparent;"
            )
            layout.addWidget(self._lbl)

        layout.addStretch()

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setFixedHeight(28)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_StyledBackground, False)

    # ── Propriété animable ────────────────────────────────────────────

    @pyqtProperty(float)
    def knob_pos(self) -> float:
        return self._knob_pos

    @knob_pos.setter
    def knob_pos(self, value: float) -> None:
        self._knob_pos = value
        self.update()

    # ── API publique ──────────────────────────────────────────────────

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, state: bool) -> None:
        if state == self._checked:
            return
        self._checked = state
        self._animate()

    def toggle(self) -> None:
        self.setChecked(not self._checked)

    # ── Événements ───────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.toggle()
            self.toggled.emit(self._checked)

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()

    # ── Animation ────────────────────────────────────────────────────

    def _animate(self) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._knob_pos)
        self._anim.setEndValue(1.0 if self._checked else 0.0)
        self._anim.start()

    # ── Rendu ────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        sw = self._switch_placeholder
        x, y = sw.x(), sw.y()
        w, h = sw.width(), sw.height()
        r = h // 2  # rayon des coins arrondis

        # ── Fond du track ──
        if self._checked:
            track_color = self._COLOR_ON_HOVER if self._hovered else self._COLOR_ON
            border_color = self._COLOR_BORDER_ON
        else:
            track_color = self._COLOR_OFF_HOVER if self._hovered else self._COLOR_OFF
            border_color = self._COLOR_BORDER_OFF

        painter.setPen(QPen(border_color, 1.5))
        painter.setBrush(QBrush(track_color))
        painter.drawRoundedRect(x, y + (h - 24) // 2, w, 24, r, r)

        # ── Knob (cercle blanc) ──
        knob_diam = 18
        margin = 3
        travel = w - 2 * margin - knob_diam
        knob_x = x + margin + int(self._knob_pos * travel)
        knob_y = y + (h - knob_diam) // 2 + 1

        # Ombre douce du knob
        shadow_pen = QPen(QColor(0, 0, 0, 60), 1)
        painter.setPen(shadow_pen)
        painter.setBrush(QBrush(QColor(200, 215, 255, 30)))
        painter.drawEllipse(knob_x + 1, knob_y + 2, knob_diam, knob_diam)

        # Knob principal
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self._COLOR_KNOB))
        painter.drawEllipse(knob_x, knob_y, knob_diam, knob_diam)

        painter.end()
