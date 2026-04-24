# -*- coding: utf-8 -*-
"""
shared_forms.py — Composants de formulaire réutilisables (PRISMA v2.0).

Fournit:
  - labeled_spin       : QLabel + QSpinBox dans une grille
  - labeled_dspin      : QLabel + QDoubleSpinBox dans une grille
  - labeled_combo      : QLabel + DarkComboBox dans une grille
  - labeled_lineedit   : QLabel + QLineEdit (pleine largeur)
  - section_label      : QLabel séparateur de sous-section
  - make_toggle        : ToggleSwitch préconfiguré
  - FormGrid           : QWidget enveloppant un QGridLayout avec helpers add_*
"""

from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QWidget,
)

# Import interne PRISMA
try:
    from gui.widgets.toggle_switch import ToggleSwitch
    from gui.main_window import DarkComboBox
except ImportError:
    # Fallback relatif (exécution directe / tests)
    import importlib, sys, os
    _pkg = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _pkg not in sys.path:
        sys.path.insert(0, _pkg)
    from gui.widgets.toggle_switch import ToggleSwitch  # type: ignore

    class DarkComboBox(QComboBox):  # type: ignore
        """Fallback minimal si main_window non importable."""
        def showPopup(self):
            from PyQt5.QtCore import Qt
            popup = self.view().window()
            popup.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
            super().showPopup()


# ─────────────────────────────────────────────────────────────────────────────
# Factories simples
# ─────────────────────────────────────────────────────────────────────────────

def labeled_spin(
    grid: QGridLayout,
    row: int,
    label: str,
    vmin: int,
    vmax: int,
    default: int,
    tooltip: str = "",
    suffix: str = "",
) -> QSpinBox:
    """Ajoute label+spinbox sur la ligne row de grid. Retourne le spinbox."""
    lbl = QLabel(label)
    lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    spin = QSpinBox()
    spin.setRange(vmin, vmax)
    spin.setValue(default)
    if suffix:
        spin.setSuffix(suffix)
    if tooltip:
        spin.setToolTip(tooltip)
        lbl.setToolTip(tooltip)
    grid.addWidget(lbl, row, 0)
    grid.addWidget(spin, row, 1)
    return spin


def labeled_dspin(
    grid: QGridLayout,
    row: int,
    label: str,
    vmin: float,
    vmax: float,
    default: float,
    step: float = 0.1,
    decimals: int = 2,
    tooltip: str = "",
    suffix: str = "",
) -> QDoubleSpinBox:
    """Ajoute label+doublespinbox sur la ligne row de grid. Retourne le widget."""
    lbl = QLabel(label)
    lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    spin = QDoubleSpinBox()
    spin.setRange(vmin, vmax)
    spin.setValue(default)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    if suffix:
        spin.setSuffix(suffix)
    if tooltip:
        spin.setToolTip(tooltip)
        lbl.setToolTip(tooltip)
    grid.addWidget(lbl, row, 0)
    grid.addWidget(spin, row, 1)
    return spin


def labeled_combo(
    grid: QGridLayout,
    row: int,
    label: str,
    items: List[str],
    default: str = "",
    tooltip: str = "",
) -> DarkComboBox:
    """Ajoute label+combobox sur la ligne row de grid. Retourne le widget."""
    lbl = QLabel(label)
    lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    combo = DarkComboBox()
    combo.addItems(items)
    if default:
        idx = combo.findText(default)
        if idx >= 0:
            combo.setCurrentIndex(idx)
    if tooltip:
        combo.setToolTip(tooltip)
        lbl.setToolTip(tooltip)
    grid.addWidget(lbl, row, 0)
    grid.addWidget(combo, row, 1)
    return combo


def labeled_lineedit(
    grid: QGridLayout,
    row: int,
    label: str,
    placeholder: str = "",
    tooltip: str = "",
    col_span: int = 2,
) -> QLineEdit:
    """Ajoute label sur row, lineedit sur row+1 (pleine largeur). Retourne le widget."""
    lbl = QLabel(label)
    grid.addWidget(lbl, row, 0, 1, col_span)
    edit = QLineEdit()
    if placeholder:
        edit.setPlaceholderText(placeholder)
    if tooltip:
        edit.setToolTip(tooltip)
    grid.addWidget(edit, row + 1, 0, 1, col_span)
    return edit


def section_label(grid: QGridLayout, row: int, text: str, col_span: int = 2) -> QLabel:
    """Insère un séparateur de sous-section dans la grille."""
    lbl = QLabel(text)
    lbl.setObjectName("subtitleLabel")
    grid.addWidget(lbl, row, 0, 1, col_span)
    return lbl


def make_toggle(label: str, checked: bool = False, tooltip: str = "") -> ToggleSwitch:
    """Crée un ToggleSwitch avec label et état initial."""
    t = ToggleSwitch(label, checked=checked)
    if tooltip:
        t.setToolTip(tooltip)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# FormGrid — wrapper pratique
# ─────────────────────────────────────────────────────────────────────────────

class FormGrid(QWidget):
    """
    Widget wrappant un QGridLayout (2 colonnes) avec méthodes utilitaires.

    Exemple:
        fg = FormGrid()
        fg.add_spin("Grille X :", 3, 50, 10, "xdim_spin")
        fg.add_toggle("Activer UMAP", False, "umap_toggle")
        layout.addWidget(fg)
    """

    def __init__(self, parent: Optional[QWidget] = None, spacing: int = 8) -> None:
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setSpacing(spacing)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._row = 0
        self._named: dict[str, QWidget] = {}

    # ── Accès aux widgets nommés ──────────────────────────────────────────────

    def __getitem__(self, name: str) -> QWidget:
        return self._named[name]

    def get(self, name: str, default: Optional[QWidget] = None) -> Optional[QWidget]:
        return self._named.get(name, default)

    # ── Méthodes d'ajout ──────────────────────────────────────────────────────

    def add_spin(
        self,
        label: str,
        vmin: int,
        vmax: int,
        default: int,
        name: str = "",
        tooltip: str = "",
        suffix: str = "",
    ) -> QSpinBox:
        w = labeled_spin(self._grid, self._row, label, vmin, vmax, default, tooltip, suffix)
        self._row += 1
        if name:
            self._named[name] = w
        return w

    def add_dspin(
        self,
        label: str,
        vmin: float,
        vmax: float,
        default: float,
        name: str = "",
        step: float = 0.1,
        decimals: int = 2,
        tooltip: str = "",
        suffix: str = "",
    ) -> QDoubleSpinBox:
        w = labeled_dspin(self._grid, self._row, label, vmin, vmax, default, step, decimals, tooltip, suffix)
        self._row += 1
        if name:
            self._named[name] = w
        return w

    def add_combo(
        self,
        label: str,
        items: List[str],
        default: str = "",
        name: str = "",
        tooltip: str = "",
    ) -> DarkComboBox:
        w = labeled_combo(self._grid, self._row, label, items, default, tooltip)
        self._row += 1
        if name:
            self._named[name] = w
        return w

    def add_lineedit(
        self,
        label: str,
        placeholder: str = "",
        name: str = "",
        tooltip: str = "",
    ) -> QLineEdit:
        w = labeled_lineedit(self._grid, self._row, label, placeholder, tooltip)
        self._row += 2  # label + lineedit occupent 2 lignes
        if name:
            self._named[name] = w
        return w

    def add_toggle(
        self,
        label: str,
        checked: bool = False,
        name: str = "",
        tooltip: str = "",
        col_span: int = 2,
    ) -> ToggleSwitch:
        w = make_toggle(label, checked, tooltip)
        self._grid.addWidget(w, self._row, 0, 1, col_span)
        self._row += 1
        if name:
            self._named[name] = w
        return w

    def add_section(self, text: str) -> QLabel:
        lbl = section_label(self._grid, self._row, text)
        self._row += 1
        return lbl

    def add_widget(
        self,
        widget: QWidget,
        name: str = "",
        col: int = 0,
        col_span: int = 2,
    ) -> QWidget:
        self._grid.addWidget(widget, self._row, col, 1, col_span)
        self._row += 1
        if name:
            self._named[name] = widget
        return widget

    def grid(self) -> QGridLayout:
        return self._grid
