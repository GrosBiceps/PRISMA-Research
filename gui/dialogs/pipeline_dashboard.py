# -*- coding: utf-8 -*-
"""
pipeline_dashboard.py â€” Vue dashboard 3-colonnes "PRISMA v2".

FenÃªtre modale (QDialog) affichant cÃ´te Ã  cÃ´te :
  - Colonne 1 : Data Import   (Drag & Drop FCS + tableau)
  - Colonne 2 : Settings      (SettingsCard + ToggleSwitch)
  - Colonne 3 : Execution Log (ProgressBar + LogConsole)

DÃ©clenchÃ©e par le bouton â›¶ "Expand" depuis la fenÃªtre principale.
Compatible avec le thÃ¨me PRISMA v2 (PyQt5).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

from PyQt5.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QProgressBar,
    QSizePolicy,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QFileDialog,
    QGridLayout,
    QMessageBox,
    QApplication,
    QSplitter,
    QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import Qt, QSize, QUrl, QMimeData, QTimer
from PyQt5.QtGui import QFont, QColor, QDragEnterEvent, QDropEvent, QIcon

from gui.widgets.toggle_switch import ToggleSwitch
from gui.widgets.settings_card import SettingsCard
from gui.widgets.log_console import LogConsole

try:
    import qtawesome as qta

    _QTA = True
except ImportError:
    _QTA = False


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Palette & helpers
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

_C = {
    "bg": "#04070D",
    "surface": "#0C1220",
    "surface2": "#101825",
    "border": "rgba(255,255,255,0.055)",
    "border2": "rgba(255,255,255,0.10)",
    "text": "#EEF2F7",
    "subtext": "rgba(238,242,247,0.45)",
    "blue": "#5BAAFF",
    "green": "#39FF8A",
    "red": "#FF3D6E",
    "orange": "#FF9B3D",
    "mauve": "#7B52FF",
    "dim": "rgba(238,242,247,0.26)",
}


def _icon(name: str, color: str = _C["text"], size: int = 16):
    if _QTA:
        try:
            return qta.icon(name, color=color)
        except Exception:
            pass
    return QIcon()


def _btn(
    text: str, obj_name: str, icon_name: str = "", icon_color: str = _C["text"]
) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName(obj_name)
    if icon_name:
        ico = _icon(icon_name, icon_color)
        if ico:
            b.setIcon(ico)
            b.setIconSize(QSize(14, 14))
    return b


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# QSS complet â€” PRISMA v2
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

_QSS = """
/* â”€â”€ Base â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
QDialog {
    background: #121418;
    color: #EEF2F7;
    font-family: 'Segoe UI', 'Inter', 'Roboto', Arial, sans-serif;
    font-size: 10pt;
}
QWidget {
    background: transparent;
    color: #EEF2F7;
    font-family: 'Segoe UI', 'Inter', 'Roboto', Arial, sans-serif;
    font-size: 10pt;
}

/* â”€â”€ Panneaux â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
QFrame#panel {
    background: #1a1d22;
    border: 1px solid #2C313C;
    border-radius: 14px;
}
QFrame#panelHeader {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 rgba(30,33,42,1), stop:1 rgba(22,25,32,1));
    border-bottom: 1px solid #2C313C;
    border-top-left-radius: 14px;
    border-top-right-radius: 14px;
}

/* â”€â”€ Titres â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
QLabel#panelTitle {
    font-size: 13pt;
    font-weight: 700;
    color: #c8d8fd;
    letter-spacing: -0.02em;
}
QLabel#panelSub {
    font-size: 9pt;
    color: #45475a;
    font-weight: 400;
}
QLabel#sectionLabel {
    font-size: 8pt;
    font-weight: 700;
    color: #5a5c78;
    letter-spacing: 0.09em;
}

/* â”€â”€ Drop Zone â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
QFrame#dropZone {
    background: rgba(0, 163, 255, 0.03);
    border: 2px dashed #2C313C;
    border-radius: 10px;
    min-height: 100px;
}
QFrame#dropZone:hover {
    background: rgba(0, 163, 255, 0.06);
    border-color: rgba(0, 163, 255, 0.35);
}
QFrame#dropZone[active="true"] {
    background: rgba(0, 163, 255, 0.10);
    border: 2px dashed #00A3FF;
}

/* â”€â”€ Table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
QTableWidget {
    background: #13151a;
    border: 1px solid #2C313C;
    border-radius: 10px;
    gridline-color: rgba(44, 49, 60, 0.5);
    font-size: 9pt;
    outline: none;
    selection-background-color: rgba(0, 163, 255, 0.14);
    selection-color: #dde6ff;
    alternate-background-color: #1a1d22;
}
QTableWidget::item {
    padding: 7px 12px;
    color: #b8c2e8;
    border-bottom: 1px solid rgba(44, 49, 60, 0.4);
}
QTableWidget::item:selected {
    background: rgba(0, 163, 255, 0.16);
    color: #e0eaff;
}
QTableWidget::item:hover {
    background: rgba(0, 163, 255, 0.07);
}
QHeaderView {
    background: transparent;
}
QHeaderView::section {
    background: #1a1d22;
    color: #45475a;
    padding: 8px 12px;
    border: none;
    border-right: 1px solid rgba(44, 49, 60, 0.5);
    border-bottom: 1px solid #2C313C;
    font-weight: 700;
    font-size: 8pt;
    letter-spacing: 0.07em;
}
QHeaderView::section:first { border-top-left-radius: 10px; }
QHeaderView::section:last  { border-top-right-radius: 10px; border-right: none; }

/* â”€â”€ Inputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
    background: #1E2227;
    border: 1px solid #2C313C;
    border-radius: 7px;
    padding: 7px 10px;
    color: #EEF2F7;
    font-size: 10pt;
    min-height: 14px;
    selection-background-color: rgba(0, 163, 255, 0.28);
}
QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover, QLineEdit:hover {
    border-color: rgba(0, 163, 255, 0.35);
    background: #22262d;
}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
    border: 1px solid rgba(0, 163, 255, 0.65);
    background: #1a1e25;
}
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QLineEdit:disabled {
    color: #EEF2F7;
    border-color: rgba(255,255,255,0.06);
    background: rgba(20,24,32,0.65);
}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background: rgba(0, 163, 255, 0.08);
    border: none;
    border-radius: 3px;
    width: 14px;
    margin: 2px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: rgba(0, 163, 255, 0.22);
}
QComboBox::drop-down {
    border: none;
    width: 24px;
    background: rgba(0, 163, 255, 0.08);
    border-top-right-radius: 7px;
    border-bottom-right-radius: 7px;
}
QComboBox QAbstractItemView {
    background: #1E2227;
    border: 1px solid #2C313C;
    selection-background-color: rgba(0, 163, 255, 0.20);
    selection-color: #EEF2F7;
    padding: 2px;
    outline: none;
}
QComboBox QAbstractItemView::item {
    padding: 7px 12px;
    border-radius: 4px;
    color: #EEF2F7;
}
QComboBox QAbstractItemView::item:hover {
    background: rgba(0, 163, 255, 0.14);
}

/* â”€â”€ ScrollArea â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 4px 2px;
}
QScrollBar::handle:vertical {
    background: rgba(0, 163, 255, 0.18);
    border-radius: 3px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: rgba(0, 163, 255, 0.38);
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { height: 6px; }
QScrollBar::handle:horizontal {
    background: rgba(0, 163, 255, 0.18);
    border-radius: 3px;
}

/* â”€â”€ ProgressBar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
QProgressBar#pipelineProgress {
    border: none;
    border-radius: 10px;
    background: rgba(10, 10, 20, 0.9);
    text-align: center;
    color: #e0eaff;
    font-weight: 700;
    font-size: 9pt;
    min-height: 20px;
    max-height: 20px;
}
QProgressBar#pipelineProgress::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0   #005faa,
        stop:0.3 #00A3FF,
        stop:0.7 #00c8ff,
        stop:1   #00A3FF);
    border-radius: 10px;
}

/* â”€â”€ Buttons â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
QPushButton {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 rgba(60,65,82,0.8), stop:1 rgba(46,51,66,0.8));
    border: 1px solid rgba(0, 163, 255, 0.18);
    border-radius: 8px;
    padding: 8px 18px;
    color: #c5cef0;
    font-weight: 600;
    font-size: 10pt;
    min-height: 14px;
}
QPushButton:hover {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 rgba(75,80,100,0.9), stop:1 rgba(58,63,80,0.9));
    border-color: rgba(0, 163, 255, 0.50);
    color: #dde6ff;
}
QPushButton:focus {
    border-color: rgba(0,163,255,0.70);
    background: rgba(75,80,100,0.92);
    color: #e8f1ff;
    outline: none;
}
QPushButton:pressed {
    background: rgba(30,34,46,0.95);
    border-color: rgba(0, 163, 255, 0.60);
    padding-top: 9px;
    padding-bottom: 7px;
}
QPushButton:disabled {
    background: rgba(28,32,42,0.5);
    color: #3a3e4a;
    border-color: rgba(44,49,60,0.2);
}

/* Primaire â€” bleu */
QPushButton#primaryBtn {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,

QCheckBox::indicator:focus {
    border-color: rgba(0,163,255,0.75);
    background: rgba(0,163,255,0.14);
}
        stop:0 #22bbff, stop:1 #007fd4);
    border: 1px solid rgba(0, 163, 255, 0.55);
    border-bottom: 1px solid rgba(0, 100, 180, 0.6);
    color: #e8f6ff;
    font-weight: 700;
    padding: 10px 26px;
}
QPushButton#primaryBtn:hover {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #40ccff, stop:1 #009ae6);
    border-color: rgba(0, 200, 255, 0.70);
}
QPushButton#primaryBtn:pressed {
    background: #006db8;
    padding-top: 11px;
    padding-bottom: 9px;
}

/* SuccÃ¨s â€” vert */
QPushButton#keepBtn {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 rgba(46, 204, 113, 0.90), stop:1 rgba(30, 156, 82, 0.95));
    border: 1px solid rgba(46, 204, 113, 0.55);
    border-bottom: 1px solid rgba(18, 100, 48, 0.5);
    color: #091a0f;
    font-weight: 700;
    padding: 10px 28px;
    min-height: 36px;
}
QPushButton#keepBtn:hover {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 rgba(60, 222, 125, 1.0), stop:1 rgba(46, 204, 113, 1.0));
    border-color: rgba(60, 222, 125, 0.70);
}

/* Danger â€” rouge */
QPushButton#discardBtn {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 rgba(231, 76, 60, 0.88), stop:1 rgba(185, 50, 38, 0.92));
    border: 1px solid rgba(231, 76, 60, 0.55);
    border-bottom: 1px solid rgba(130, 30, 20, 0.5);
    color: #160302;
    font-weight: 700;
    padding: 10px 28px;
    min-height: 36px;
}
QPushButton#discardBtn:hover {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 rgba(255, 90, 70, 1.0), stop:1 rgba(231, 76, 60, 1.0));
    border-color: rgba(255, 90, 70, 0.70);
}

/* Ghost */
QPushButton#ghostBtn {
    background: transparent;
    border: 1px solid rgba(0, 163, 255, 0.25);
    color: #00A3FF;
    font-weight: 600;
    padding: 7px 16px;
}
QPushButton#ghostBtn:hover {
    background: rgba(0, 163, 255, 0.08);
    border-color: rgba(0, 163, 255, 0.55);
    color: #33c0ff;
}

/* IcÃ´ne seule */
QPushButton#iconBtn {
    background: rgba(40, 45, 58, 0.6);
    border: 1px solid rgba(0, 163, 255, 0.14);
    border-radius: 8px;
    padding: 6px;
    min-width: 32px; max-width: 36px;
    min-height: 32px; max-height: 36px;
}
QPushButton#iconBtn:hover {
    background: rgba(0, 163, 255, 0.14);
    border-color: rgba(0, 163, 255, 0.40);
}

/* Close button */
QPushButton#closeBtn {
    background: rgba(231, 76, 60, 0.12);
    border: 1px solid rgba(231, 76, 60, 0.22);
    border-radius: 8px;
    padding: 6px 14px;
    color: #E74C3C;
    font-weight: 700;
    font-size: 9pt;
}
QPushButton#closeBtn:hover {
    background: rgba(231, 76, 60, 0.24);
    border-color: rgba(231, 76, 60, 0.50);
}

/* â”€â”€ Checkbox dans la table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
QCheckBox {
    spacing: 8px;
    color: #b8c2e8;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid rgba(0, 163, 255, 0.28);
    background: #1E2227;
}
QCheckBox::indicator:checked {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #22bbff, stop:1 #007fd4);
    border-color: #00A3FF;
}
QCheckBox::indicator:hover { border-color: rgba(0, 163, 255, 0.55); }

/* â”€â”€ Splitter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
QSplitter::handle {
    background: rgba(44, 49, 60, 0.6);
    width: 3px;
}
QSplitter::handle:hover {
    background: rgba(0, 163, 255, 0.30);
}

/* â”€â”€ SettingsCard body â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
QFrame#settingsCard {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 rgba(28,32,42,0.95), stop:1 rgba(22,26,36,0.90));
    border: 1px solid rgba(44, 49, 60, 0.8);
    border-top: 1px solid rgba(0, 163, 255, 0.22);
    border-radius: 12px;
}

/* â”€â”€ Status badges â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
QLabel#badgeOk {
    background: rgba(46, 204, 113, 0.14);
    color: #2ECC71;
    border: 1px solid rgba(46, 204, 113, 0.28);
    border-radius: 5px;
    padding: 2px 8px;
    font-size: 8pt;
    font-weight: 700;
}
QLabel#badgePatho {
    background: rgba(231, 76, 60, 0.14);
    color: #E74C3C;
    border: 1px solid rgba(231, 76, 60, 0.28);
    border-radius: 5px;
    padding: 2px 8px;
    font-size: 8pt;
    font-weight: 700;
}
QLabel#badgeWarn {
    background: rgba(243, 156, 18, 0.14);
    color: #F39C12;
    border: 1px solid rgba(243, 156, 18, 0.28);
    border-radius: 5px;
    padding: 2px 8px;
    font-size: 8pt;
    font-weight: 700;
}

/* â”€â”€ Tooltip â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
QToolTip {
    background: #1E2227;
    color: #c5cef0;
    border: 1px solid rgba(0, 163, 255, 0.22);
    border-radius: 7px;
    padding: 7px 12px;
    font-size: 9pt;
}

/* â”€â”€ Separator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
QFrame[frameShape="4"], QFrame[frameShape="5"] {
    color: rgba(44, 49, 60, 0.7);
    border: none;
    max-height: 1px;
    min-height: 1px;
}
"""


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Drop Zone Frame
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class DropZoneFrame(QFrame):
    """
    Zone de glisser-dÃ©poser peinte via QSS.
    Accepte les dossiers et fichiers .fcs.
    """

    def __init__(self, on_drop_callback, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._callback = on_drop_callback
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(96)

        vbox = QVBoxLayout(self)
        vbox.setAlignment(Qt.AlignCenter)
        vbox.setSpacing(8)

        # IcÃ´ne
        self._ico_lbl = QLabel(self)
        ico = _icon("fa5s.cloud-upload-alt", _C["blue"])
        if ico:
            self._ico_lbl.setPixmap(ico.pixmap(QSize(32, 32)))
        self._ico_lbl.setAlignment(Qt.AlignCenter)
        self._ico_lbl.setStyleSheet("background: transparent;")
        vbox.addWidget(self._ico_lbl)

        # Texte principal
        self._main_lbl = QLabel("Glissez-dÃ©posez vos fichiers FCS ici")
        self._main_lbl.setAlignment(Qt.AlignCenter)
        self._main_lbl.setStyleSheet(
            "color: #5a5c78; font-size: 10pt; font-weight: 600; background: transparent;"
        )
        vbox.addWidget(self._main_lbl)

        # Texte secondaire
        sub = QLabel("ou cliquez sur  Parcourir")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color: #3a3e4a; font-size: 9pt; background: transparent;")
        vbox.addWidget(sub)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("active", True)
            self.style().unpolish(self)
            self.style().polish(self)

    def dragLeaveEvent(self, event) -> None:
        self.setProperty("active", False)
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, event: QDropEvent) -> None:
        self.setProperty("active", False)
        self.style().unpolish(self)
        self.style().polish(self)
        paths = []
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if Path(p).suffix.lower() == ".fcs" or Path(p).is_dir():
                paths.append(p)
        if paths:
            self._callback(paths)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Colonne 1 â€” Data Import
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class DataImportPanel(QFrame):
    """Panneau d'importation FCS avec Drag & Drop et tableau de prÃ©visualisation."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self._files: List[
            dict
        ] = []  # {"path": str, "label": str, "cells": int, "markers": int, "condition": str}
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # â”€â”€ En-tÃªte â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        header = self._make_header(
            "Importation des DonnÃ©es",
            "Glissez-dÃ©posez ou parcourez vos fichiers .FCS",
            "fa5s.folder-open",
            _C["blue"],
        )
        root.addWidget(header)

        # â”€â”€ Corps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        body = QWidget()
        body.setStyleSheet("background: transparent;")
        vbox = QVBoxLayout(body)
        vbox.setContentsMargins(16, 16, 16, 16)
        vbox.setSpacing(12)

        # Drop zone
        self._drop = DropZoneFrame(self._on_drop)
        vbox.addWidget(self._drop)

        # Boutons d'action
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_browse_healthy = _btn("  NBM / Saine", "ghostBtn", "fa5s.folder", _C["green"])
        self.btn_browse_healthy.clicked.connect(lambda: self._browse("NBM/Saine"))
        btn_row.addWidget(self.btn_browse_healthy)

        self.btn_browse_patho = _btn("  Pathologique", "ghostBtn", "fa5s.folder", _C["red"])
        self.btn_browse_patho.clicked.connect(lambda: self._browse("Pathologique"))
        btn_row.addWidget(self.btn_browse_patho)

        btn_row.addStretch()

        self.btn_scan = _btn("  Scanner", "primaryBtn", "fa5s.search", "#e8f6ff")
        self.btn_scan.clicked.connect(self._scan_files)
        btn_row.addWidget(self.btn_scan)

        self.btn_clear = _btn("  Vider", "ghostBtn", "fa5s.trash-alt", _C["red"])
        self.btn_clear.clicked.connect(self._clear_table)
        btn_row.addWidget(self.btn_clear)

        vbox.addLayout(btn_row)

        # SÃ©parateur + titre tableau
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        vbox.addWidget(sep)

        hdr2 = QHBoxLayout()
        lbl_tbl = QLabel("APERÃ‡U DES FICHIERS FCS DÃ‰TECTÃ‰S")
        lbl_tbl.setObjectName("sectionLabel")
        hdr2.addWidget(lbl_tbl)
        hdr2.addStretch()
        self.lbl_count = QLabel("0 fichier(s)")
        self.lbl_count.setStyleSheet("color: #3a3e4a; font-size: 8.5pt; background: transparent;")
        hdr2.addWidget(self.lbl_count)
        vbox.addLayout(hdr2)

        # Tableau FCS
        self._table = self._build_table()
        vbox.addWidget(self._table, 1)

        # RÃ©sumÃ©
        self.lbl_summary = QLabel("SÃ©lectionnez des dossiers FCS pour commencer.")
        self.lbl_summary.setStyleSheet(
            "color: #3a3e4a; font-size: 9pt; padding: 4px 0; background: transparent;"
        )
        vbox.addWidget(self.lbl_summary)

        root.addWidget(body, 1)

    # â”€â”€ Construction du tableau â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _build_table(self) -> QTableWidget:
        t = QTableWidget()
        t.setObjectName("fcsTable")
        t.setColumnCount(5)
        t.setHorizontalHeaderLabels(["âœ“", "Fichier FCS", "Condition", "Cellules", "Marqueurs"])
        t.setAlternatingRowColors(True)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setShowGrid(False)
        t.verticalHeader().setVisible(False)

        h = t.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Fixed)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        t.setColumnWidth(0, 36)
        return t

    # â”€â”€ Slots â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _on_drop(self, paths: List[str]) -> None:
        """Traite les fichiers/dossiers dÃ©posÃ©s."""
        fcs_files = []
        for p in paths:
            pp = Path(p)
            if pp.is_dir():
                fcs_files.extend(pp.rglob("*.fcs"))
            elif pp.suffix.lower() == ".fcs":
                fcs_files.append(pp)
        self._add_files(fcs_files, condition="â€”")

    def _browse(self, condition: str) -> None:
        folder = QFileDialog.getExistingDirectory(self, f"SÃ©lectionner le dossier {condition}")
        if folder:
            files = list(Path(folder).rglob("*.fcs"))
            self._add_files(files, condition=condition)

    def _add_files(self, files: List[Path], condition: str) -> None:
        for fp in files:
            entry = {
                "path": str(fp),
                "label": fp.name,
                "condition": condition,
                "cells": "â€”",
                "markers": "â€”",
            }
            # Ã‰viter les doublons
            if not any(e["path"] == entry["path"] for e in self._files):
                self._files.append(entry)
        self._refresh_table()

    def _scan_files(self) -> None:
        """Tente de lire les mÃ©tadonnÃ©es FCS (cellules, marqueurs)."""
        try:
            from io.fcs_reader import read_fcs_metadata

            for entry in self._files:
                try:
                    meta = read_fcs_metadata(entry["path"])
                    entry["cells"] = str(meta.get("n_events", "â€”"))
                    entry["markers"] = str(meta.get("n_channels", "â€”"))
                except Exception:
                    pass
        except ImportError:
            pass
        self._refresh_table()

    def _clear_table(self) -> None:
        self._files.clear()
        self._table.setRowCount(0)
        self.lbl_count.setText("0 fichier(s)")
        self.lbl_summary.setText("Tableau vidÃ©.")

    def _refresh_table(self) -> None:
        self._table.setRowCount(0)
        for entry in self._files:
            row = self._table.rowCount()
            self._table.insertRow(row)

            # Checkbox
            chk = QCheckBox()
            chk.setChecked(True)
            chk.setStyleSheet("margin-left: 8px; background: transparent;")
            self._table.setCellWidget(row, 0, chk)

            # Fichier
            name_item = QTableWidgetItem(entry["label"])
            name_item.setToolTip(entry["path"])
            self._table.setItem(row, 1, name_item)

            # Condition
            cond_item = QTableWidgetItem(entry["condition"])
            color = (
                _C["green"]
                if "saine" in entry["condition"].lower() or "nbm" in entry["condition"].lower()
                else (_C["red"] if "patho" in entry["condition"].lower() else _C["text"])
            )
            cond_item.setForeground(QColor(color))
            self._table.setItem(row, 2, cond_item)

            # Cellules & Marqueurs
            self._table.setItem(row, 3, QTableWidgetItem(str(entry["cells"])))
            self._table.setItem(row, 4, QTableWidgetItem(str(entry["markers"])))

            # Hauteur de ligne
            self._table.setRowHeight(row, 36)

        n = len(self._files)
        self.lbl_count.setText(f"{n} fichier(s)")
        self.lbl_summary.setText(
            f"{n} fichier(s) FCS chargÃ©(s). Cliquez sur Scanner pour lire les mÃ©tadonnÃ©es."
            if n > 0
            else "SÃ©lectionnez des dossiers FCS pour commencer."
        )

    # â”€â”€ Helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _make_header(self, title: str, sub: str, icon_name: str, accent: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName("panelHeader")
        hbox = QHBoxLayout(frame)
        hbox.setContentsMargins(18, 14, 18, 14)
        hbox.setSpacing(10)

        if _QTA:
            try:
                ico_lbl = QLabel()
                ico = qta.icon(icon_name, color=accent)
                ico_lbl.setPixmap(ico.pixmap(QSize(20, 20)))
                ico_lbl.setStyleSheet("background: transparent;")
                hbox.addWidget(ico_lbl)
            except Exception:
                pass

        txt_col = QVBoxLayout()
        txt_col.setSpacing(2)
        t_lbl = QLabel(title)
        t_lbl.setObjectName("panelTitle")
        txt_col.addWidget(t_lbl)
        s_lbl = QLabel(sub)
        s_lbl.setObjectName("panelSub")
        txt_col.addWidget(s_lbl)
        hbox.addLayout(txt_col)
        hbox.addStretch()

        return frame

    # â”€â”€ Accesseurs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def get_selected_files(self) -> List[str]:
        """Renvoie les chemins des fichiers dont la case est cochÃ©e."""
        selected = []
        for row in range(self._table.rowCount()):
            chk = self._table.cellWidget(row, 0)
            if isinstance(chk, QCheckBox) and chk.isChecked():
                if row < len(self._files):
                    selected.append(self._files[row]["path"])
        return selected


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Colonne 2 â€” Settings
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class SettingsPanel(QFrame):
    """Panneau de paramÃ©trage avec SettingsCard et ToggleSwitch."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # En-tÃªte
        header = self._make_header(
            "ParamÃ¨tres du Pipeline",
            "Configurez FlowSOM, gating et export",
            "fa5s.sliders-h",
            _C["mauve"],
        )
        root.addWidget(header)

        # ScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(14, 14, 14, 14)
        vbox.setSpacing(12)

        # â”€â”€ Carte 1 : CatÃ©gories autorisÃ©es â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        card1 = SettingsCard(
            "CatÃ©gories AutorisÃ©es",
            icon_name="fa5s.tags",
            subtitle="Populations incluses dans l'analyse",
            accent=_C["blue"],
        )
        grid1 = QGridLayout()
        grid1.setSpacing(6)
        self.tgl_cd3 = ToggleSwitch("CD3+ T lymphocytes", checked=True)
        self.tgl_cd19 = ToggleSwitch("CD19+ B lymphocytes", checked=True)
        self.tgl_cd56 = ToggleSwitch("CD56+ NK cells", checked=True)
        self.tgl_cd34 = ToggleSwitch("CD34+ Blastes", checked=False)
        self.tgl_nk = ToggleSwitch("CD16+ Monocytes", checked=True)
        self.tgl_nkp = ToggleSwitch("CD45 dim (MRD)", checked=True)
        for i, w in enumerate(
            [self.tgl_cd3, self.tgl_cd19, self.tgl_cd56, self.tgl_cd34, self.tgl_nk, self.tgl_nkp]
        ):
            grid1.addWidget(w, i // 2, i % 2)
        card1.body_layout.addLayout(grid1)
        vbox.addWidget(card1)

        # â”€â”€ Carte 2 : Normalisation dÂ² â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        card2 = SettingsCard(
            "Normalisation dÂ²",
            icon_name="fa5s.chart-line",
            subtitle="ParamÃ¨tres de transformation",
            accent=_C["mauve"],
        )
        g2 = QGridLayout()
        g2.setSpacing(8)

        g2.addWidget(QLabel("Transformation :"), 0, 0)
        self.combo_transform = QComboBox()
        self.combo_transform.addItems(["logicle", "arcsinh", "log10", "none"])
        g2.addWidget(self.combo_transform, 0, 1)

        g2.addWidget(QLabel("Cofacteur :"), 1, 0)
        self.spin_cofactor = QDoubleSpinBox()
        self.spin_cofactor.setRange(1.0, 500.0)
        self.spin_cofactor.setValue(5.0)
        self.spin_cofactor.setDecimals(1)
        g2.addWidget(self.spin_cofactor, 1, 1)

        g2.addWidget(QLabel("Normalisation :"), 2, 0)
        self.combo_norm = QComboBox()
        self.combo_norm.addItems(["zscore", "minmax", "none"])
        g2.addWidget(self.combo_norm, 2, 1)

        self.tgl_d2 = ToggleSwitch("Activer la normalisation dÂ²", checked=True)
        g2.addWidget(self.tgl_d2, 3, 0, 1, 2)
        card2.body_layout.addLayout(g2)
        vbox.addWidget(card2)

        # â”€â”€ Carte 3 : Condition Patho / Sain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        card3 = SettingsCard(
            "Condition Patho / Sain",
            icon_name="fa5s.microscope",
            subtitle="Seuils de classification MRD",
            accent=_C["green"],
        )
        g3 = QGridLayout()
        g3.setSpacing(8)

        g3.addWidget(QLabel("MÃ©thode MRD :"), 0, 0)
        self.combo_mrd = QComboBox()
        self.combo_mrd.addItems(["FlowSOM-diff", "FLO", "ELN 2022", "JF Cytometry"])
        g3.addWidget(self.combo_mrd, 0, 1)

        g3.addWidget(QLabel("Seuil MRD (%) :"), 1, 0)
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.001, 100.0)
        self.spin_threshold.setValue(0.01)
        self.spin_threshold.setDecimals(3)
        self.spin_threshold.setSuffix(" %")
        g3.addWidget(self.spin_threshold, 1, 1)

        g3.addWidget(QLabel("Metaclusters :"), 2, 0)
        self.spin_meta = QSpinBox()
        self.spin_meta.setRange(2, 50)
        self.spin_meta.setValue(8)
        g3.addWidget(self.spin_meta, 2, 1)

        self.tgl_eln = ToggleSwitch("Porte biologique ELN 2022", checked=True)
        self.tgl_compare = ToggleSwitch("Mode comparaison Sain vs Patho", checked=True)
        g3.addWidget(self.tgl_eln, 3, 0, 1, 2)
        g3.addWidget(self.tgl_compare, 4, 0, 1, 2)
        card3.body_layout.addLayout(g3)
        vbox.addWidget(card3)

        # â”€â”€ Carte 4 : Options avancÃ©es â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        card4 = SettingsCard(
            "Options AvancÃ©es",
            icon_name="fa5s.cogs",
            accent=_C["orange"],
        )
        g4 = QGridLayout()
        g4.setSpacing(8)

        g4.addWidget(QLabel("Grille SOM X :"), 0, 0)
        self.spin_xdim = QSpinBox()
        self.spin_xdim.setRange(3, 50)
        self.spin_xdim.setValue(10)
        g4.addWidget(self.spin_xdim, 0, 1)

        g4.addWidget(QLabel("Grille SOM Y :"), 1, 0)
        self.spin_ydim = QSpinBox()
        self.spin_ydim.setRange(3, 50)
        self.spin_ydim.setValue(10)
        g4.addWidget(self.spin_ydim, 1, 1)

        g4.addWidget(QLabel("Seed :"), 2, 0)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 99999)
        self.spin_seed.setValue(42)
        g4.addWidget(self.spin_seed, 2, 1)

        self.tgl_umap = ToggleSwitch("Calculer UMAP", checked=False)
        self.tgl_gpu = ToggleSwitch("GPU (CUDA)", checked=True)
        self.tgl_batch = ToggleSwitch("Mode Batch", checked=False)
        g4.addWidget(self.tgl_umap, 3, 0, 1, 2)
        g4.addWidget(self.tgl_gpu, 4, 0, 1, 2)
        g4.addWidget(self.tgl_batch, 5, 0, 1, 2)
        card4.body_layout.addLayout(g4)
        vbox.addWidget(card4)

        vbox.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    def _make_header(self, title, sub, icon_name, accent) -> QFrame:
        frame = QFrame()
        frame.setObjectName("panelHeader")
        hbox = QHBoxLayout(frame)
        hbox.setContentsMargins(18, 14, 18, 14)
        hbox.setSpacing(10)
        if _QTA:
            try:
                ico_lbl = QLabel()
                ico_lbl.setPixmap(qta.icon(icon_name, color=accent).pixmap(QSize(20, 20)))
                ico_lbl.setStyleSheet("background: transparent;")
                hbox.addWidget(ico_lbl)
            except Exception:
                pass
        txt = QVBoxLayout()
        txt.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("panelTitle")
        txt.addWidget(t)
        s = QLabel(sub)
        s.setObjectName("panelSub")
        txt.addWidget(s)
        hbox.addLayout(txt)
        hbox.addStretch()
        return frame

    # â”€â”€ Accesseurs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def get_params(self) -> dict:
        return {
            "transform": self.combo_transform.currentText(),
            "cofactor": self.spin_cofactor.value(),
            "normalize": self.combo_norm.currentText(),
            "mrd_method": self.combo_mrd.currentText(),
            "mrd_threshold": self.spin_threshold.value(),
            "metaclusters": self.spin_meta.value(),
            "xdim": self.spin_xdim.value(),
            "ydim": self.spin_ydim.value(),
            "seed": self.spin_seed.value(),
            "umap": self.tgl_umap.isChecked(),
            "gpu": self.tgl_gpu.isChecked(),
            "batch": self.tgl_batch.isChecked(),
            "eln_gate": self.tgl_eln.isChecked(),
            "compare": self.tgl_compare.isChecked(),
            "d2_norm": self.tgl_d2.isChecked(),
            "categories": {
                "CD3": self.tgl_cd3.isChecked(),
                "CD19": self.tgl_cd19.isChecked(),
                "CD56": self.tgl_cd56.isChecked(),
                "CD34": self.tgl_cd34.isChecked(),
                "CD16": self.tgl_nk.isChecked(),
                "CD45dim": self.tgl_nkp.isChecked(),
            },
        }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Colonne 3 â€” Execution Log
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class ExecutionPanel(QFrame):
    """Panneau d'exÃ©cution avec ProgressBar stylisÃ©e et LogConsole."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # En-tÃªte
        header = self._make_header()
        root.addWidget(header)

        body = QWidget()
        body.setStyleSheet("background: transparent;")
        vbox = QVBoxLayout(body)
        vbox.setContentsMargins(16, 16, 16, 16)
        vbox.setSpacing(12)

        # â”€â”€ Indicateur d'Ã©tat â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        status_row = QHBoxLayout()
        status_row.setSpacing(10)

        self._status_ico = QLabel()
        self._set_status_icon("idle")
        self._status_ico.setStyleSheet("background: transparent;")
        status_row.addWidget(self._status_ico)

        self.lbl_step = QLabel("En attente du lancementâ€¦")
        self.lbl_step.setObjectName("pipelineStepLabel")
        self.lbl_step.setStyleSheet(
            "color: #3a3e4a; font-size: 10pt; font-weight: 500; background: transparent;"
        )
        status_row.addWidget(self.lbl_step)
        status_row.addStretch()

        self.lbl_pct = QLabel("0 %")
        self.lbl_pct.setStyleSheet(
            "color: #00A3FF; font-size: 12pt; font-weight: 700; background: transparent;"
        )
        status_row.addWidget(self.lbl_pct)
        vbox.addLayout(status_row)

        # â”€â”€ Barre de progression â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.progress = QProgressBar()
        self.progress.setObjectName("pipelineProgress")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)  # texte gÃ©rÃ© par lbl_pct
        self.progress.setMinimumHeight(20)
        self.progress.setMaximumHeight(20)
        vbox.addWidget(self.progress)

        # â”€â”€ Mini-Ã©tapes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        steps_row = QHBoxLayout()
        steps_row.setSpacing(4)
        self._step_labels: List[QLabel] = []
        _step_names = ["Import", "Transform", "FlowSOM", "Metaclust.", "MRD", "Export"]
        for name in _step_names:
            lbl = QLabel(name)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                "color: #2a2d38; font-size: 7.5pt; font-weight: 600; "
                "background: #1a1d22; border: 1px solid #2C313C; "
                "border-radius: 4px; padding: 3px 6px;"
            )
            steps_row.addWidget(lbl)
            self._step_labels.append(lbl)
        vbox.addLayout(steps_row)

        # â”€â”€ Titre console â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        vbox.addWidget(sep)

        console_hdr = QHBoxLayout()
        lbl_console = QLabel("CONSOLE")
        lbl_console.setObjectName("sectionLabel")
        console_hdr.addWidget(lbl_console)
        console_hdr.addStretch()

        btn_copy = _btn("", "iconBtn", "fa5s.copy", _C["subtext"])
        btn_copy.setToolTip("Copier les logs")
        btn_copy.setFixedSize(32, 32)
        console_hdr.addWidget(btn_copy)

        btn_clr = _btn("", "iconBtn", "fa5s.trash-alt", _C["red"])
        btn_clr.setToolTip("Effacer la console")
        btn_clr.setFixedSize(32, 32)
        console_hdr.addWidget(btn_clr)
        vbox.addLayout(console_hdr)

        # â”€â”€ Console â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.console = LogConsole()
        vbox.addWidget(self.console, 1)

        btn_copy.clicked.connect(
            lambda: QApplication.clipboard().setText(self.console.toPlainText())
        )
        btn_clr.clicked.connect(self.console.clear)

        # â”€â”€ Boutons d'action bas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        vbox.addWidget(sep2)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self.btn_run = _btn("  Lancer", "primaryBtn", "fa5s.play", "#e8f6ff")
        self.btn_run.setMinimumHeight(38)
        action_row.addWidget(self.btn_run)

        self.btn_stop = _btn("  ArrÃªter", "discardBtn", "fa5s.stop-circle", "#160302")
        self.btn_stop.setMinimumHeight(38)
        self.btn_stop.setEnabled(False)
        action_row.addWidget(self.btn_stop)

        action_row.addStretch()

        self.btn_export_mrd = _btn("  Export MRD", "ghostBtn", "fa5s.file-export", _C["blue"])
        self.btn_export_mrd.setEnabled(False)
        action_row.addWidget(self.btn_export_mrd)
        self.btn_export_mrd.clicked.connect(self._confirm_export_mrd)

        vbox.addLayout(action_row)

        root.addWidget(body, 1)

    def _make_header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("panelHeader")
        hbox = QHBoxLayout(frame)
        hbox.setContentsMargins(18, 14, 18, 14)
        hbox.setSpacing(10)
        if _QTA:
            try:
                ico_lbl = QLabel()
                ico_lbl.setPixmap(
                    qta.icon("fa5s.terminal", color=_C["green"]).pixmap(QSize(20, 20))
                )
                ico_lbl.setStyleSheet("background: transparent;")
                hbox.addWidget(ico_lbl)
            except Exception:
                pass
        txt = QVBoxLayout()
        txt.setSpacing(2)
        t = QLabel("Console d'ExÃ©cution")
        t.setObjectName("panelTitle")
        txt.addWidget(t)
        s = QLabel("Suivi en temps rÃ©el du pipeline FlowSOM")
        s.setObjectName("panelSub")
        txt.addWidget(s)
        hbox.addLayout(txt)
        hbox.addStretch()
        return frame

    # â”€â”€ API publique â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def set_progress(self, value: int, label: str = "") -> None:
        self.progress.setValue(value)
        self.lbl_pct.setText(f"{value} %")
        if label:
            self.lbl_step.setText(label)
        # Couleur dynamique du texte %
        if value >= 100:
            self.lbl_pct.setStyleSheet(
                "color: #2ECC71; font-size: 12pt; font-weight: 700; background: transparent;"
            )
        elif value > 0:
            self.lbl_pct.setStyleSheet(
                "color: #00A3FF; font-size: 12pt; font-weight: 700; background: transparent;"
            )

    def set_step_active(self, step_idx: int) -> None:
        """Met en Ã©vidence une mini-Ã©tape (0-5)."""
        for i, lbl in enumerate(self._step_labels):
            if i < step_idx:
                lbl.setStyleSheet(
                    "color: #2ECC71; font-size: 7.5pt; font-weight: 600;"
                    "background: rgba(46,204,113,0.12); border: 1px solid rgba(46,204,113,0.30);"
                    "border-radius: 4px; padding: 3px 6px;"
                )
            elif i == step_idx:
                lbl.setStyleSheet(
                    "color: #00A3FF; font-size: 7.5pt; font-weight: 700;"
                    "background: rgba(0,163,255,0.14); border: 1px solid rgba(0,163,255,0.40);"
                    "border-radius: 4px; padding: 3px 6px;"
                )
            else:
                lbl.setStyleSheet(
                    "color: #2a2d38; font-size: 7.5pt; font-weight: 600;"
                    "background: #1a1d22; border: 1px solid #2C313C;"
                    "border-radius: 4px; padding: 3px 6px;"
                )

    def _set_status_icon(self, state: str) -> None:
        if not _QTA:
            return
        icons = {
            "idle": ("fa5s.circle", _C["dim"]),
            "running": ("fa5s.spinner", _C["blue"]),
            "done": ("fa5s.check-circle", _C["green"]),
            "error": ("fa5s.exclamation-circle", _C["red"]),
        }
        name, color = icons.get(state, icons["idle"])
        try:
            ico = qta.icon(name, color=color)
            self._status_ico.setPixmap(ico.pixmap(QSize(18, 18)))
        except Exception:
            pass

    def set_running(self, running: bool) -> None:
        self.btn_run.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self._set_status_icon("running" if running else "idle")

    def set_done(self, success: bool) -> None:
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_export_mrd.setEnabled(success)
        self._set_status_icon("done" if success else "error")

    # â”€â”€ Export MRD avec confirmation Ã  2 facteurs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _confirm_export_mrd(self) -> None:
        """
        Confirmation visuelle Ã  deux facteurs avant export des nÅ“uds critiques.
        Ã‰tape 1 : Dialog d'avertissement.
        Ã‰tape 2 : Saisie du code de confirmation.
        """
        # â”€â”€ Ã‰tape 1 : Warning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        dlg1 = QMessageBox(self)
        dlg1.setWindowTitle("Confirmation Export MRD")
        dlg1.setIcon(QMessageBox.Warning)
        dlg1.setText("<b style='font-size:13pt;'>Validation des nÅ“uds critiques requise</b>")
        dlg1.setInformativeText(
            "Vous Ãªtes sur le point d'exporter les donnÃ©es MRD.<br><br>"
            "âš  <b>Les nÅ“uds marquÃ©s Ã‰CARTER</b> seront exclus du rapport final.<br>"
            "Avez-vous bien vÃ©rifiÃ© la classification de <b>tous les nÅ“uds critiques</b> ?<br><br>"
            "Cette action gÃ©nÃ¨re un rapport destinÃ© Ã  un usage clinique."
        )
        dlg1.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        dlg1.button(QMessageBox.Yes).setText("  Oui, j'ai vÃ©rifiÃ©")
        dlg1.button(QMessageBox.Cancel).setText("  Annuler")
        dlg1.setStyleSheet("""
            QMessageBox { background: #0C1220; color: #EEF2F7; }
            QMessageBox QLabel { color: #EEF2F7; font-size: 10pt; }
            QMessageBox QPushButton {
                background: rgba(91,170,255,0.14);
                border: 1px solid rgba(91,170,255,0.34);
                border-radius: 0px;
                padding: 8px 18px;
                color: #5BAAFF;
                font-weight: 600;
                min-width: 140px;
            }
            QMessageBox QPushButton:hover {
                background: rgba(91,170,255,0.24);
                border-color: rgba(91,170,255,0.60);
            }
        """)

        if dlg1.exec_() != QMessageBox.Yes:
            return

        # â”€â”€ Ã‰tape 2 : Saisie du code de confirmation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        from PyQt5.QtWidgets import QInputDialog

        code, ok = QInputDialog.getText(
            self,
            "Confirmation Ã  deux facteurs",
            "Saisissez EXPORT pour confirmer l'exportation finale :",
        )
        if ok and code.strip().upper() == "EXPORT":
            self.console.append_log("[SUCCESS] Export MRD validÃ©. GÃ©nÃ©ration du rapport en coursâ€¦")
            self.set_progress(100, "Export terminÃ© avec succÃ¨s")
            self.set_step_active(5)
        elif ok:
            self.console.append_log("[WARNING] Code de confirmation incorrect. Export annulÃ©.")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Dashboard principal â€” 3 colonnes
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class PipelineDashboard(QDialog):
    """
    FenÃªtre modale affichant les 3 panneaux cÃ´te Ã  cÃ´te.
    DÃ©clenchÃ©e par un bouton â›¶ "Expand" dans la fenÃªtre principale.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("FlowSOM — Dashboard Pipeline")
        self.setMinimumSize(1100, 720)
        self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        from gui.screen_utils import fit_dialog_to_screen
        fit_dialog_to_screen(self, ratio=0.92, min_w=1400, min_h=860)
        self.setStyleSheet(_QSS)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # â”€â”€ Barre de titre personnalisÃ©e â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        title_bar = self._build_titlebar()
        root.addWidget(title_bar)

        # â”€â”€ Avertissement usage non-clinique â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        banner = QLabel(
            "âš   Usage de recherche uniquement â€” Non destinÃ© Ã  un usage clinique direct "
            "sans validation par un professionnel de santÃ© qualifiÃ©."
        )
        banner.setAlignment(Qt.AlignCenter)
        banner.setStyleSheet(
            "background: rgba(255, 155, 61, 0.12);"
            "color: #FF9B3D;"
            "border-bottom: 1px solid rgba(255, 155, 61, 0.28);"
            "font-size: 8.8pt;"
            "font-weight: 600;"
            "font-family: 'Consolas', 'Cascadia Code', monospace;"
            "padding: 6px 20px;"
        )
        root.addWidget(banner)

        # â”€â”€ Splitter 3 colonnes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet(_QSS)

        self.panel_import = DataImportPanel()
        self.panel_settings = SettingsPanel()
        self.panel_exec = ExecutionPanel()

        splitter.addWidget(self.panel_import)
        splitter.addWidget(self.panel_settings)
        splitter.addWidget(self.panel_exec)

        # Ratios : 38 / 32 / 30 %
        total = 1600
        splitter.setSizes([int(total * 0.38), int(total * 0.32), int(total * 0.30)])

        # Wrapper avec marges
        wrapper = QWidget()
        wrapper.setStyleSheet("background: #04070D;")
        wl = QHBoxLayout(wrapper)
        wl.setContentsMargins(16, 16, 16, 16)
        wl.setSpacing(0)
        wl.addWidget(splitter)

        root.addWidget(wrapper, 1)

        # â”€â”€ Barre de statut bas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        status_bar = self._build_statusbar()
        root.addWidget(status_bar)

        # â”€â”€ Connexions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.panel_exec.btn_run.clicked.connect(self._run_pipeline)
        self.panel_exec.btn_stop.clicked.connect(self._stop_pipeline)

        # DÃ©mo : log initial
        QTimer.singleShot(300, self._demo_logs)

    # â”€â”€ Title bar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _build_titlebar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 #080D18, stop:1 #04070D);"
            "border-bottom: 1px solid rgba(255,255,255,0.055);"
        )
        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(20, 10, 16, 10)
        hbox.setSpacing(12)

        # Logo + titre
        logo_lbl = QLabel()
        if _QTA:
            try:
                logo_lbl.setPixmap(qta.icon("fa5s.dna", color=_C["blue"]).pixmap(QSize(24, 24)))
                logo_lbl.setStyleSheet("background: transparent;")
            except Exception:
                pass
        hbox.addWidget(logo_lbl)

        title_lbl = QLabel("FlowSOM  â€“  Pipeline Dashboard")
        title_lbl.setStyleSheet(
            "font-size: 14pt; font-weight: 700; color: #EEF2F7;"
            "letter-spacing: -0.02em; background: transparent;"
        )
        hbox.addWidget(title_lbl)

        sub_lbl = QLabel("MRD Analyzer Pro")
        sub_lbl.setStyleSheet(
            "font-size: 9pt; color: #EEF2F7; "
            "font-family: 'Consolas', 'Cascadia Code', monospace;"
            "background: transparent; margin-left: 6px;"
        )
        hbox.addWidget(sub_lbl)

        hbox.addStretch()

        # Bouton fermer
        btn_close = _btn("  Fermer", "closeBtn", "fa5s.times", _C["red"])
        btn_close.clicked.connect(self.close)
        hbox.addWidget(btn_close)

        return bar

    def _build_statusbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(32)
        bar.setStyleSheet("background: #080D18;border-top: 1px solid rgba(255,255,255,0.055);")
        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(16, 0, 16, 0)
        hbox.setSpacing(16)

        self._lbl_status = QLabel("PrÃªt.")
        self._lbl_status.setStyleSheet(
            "color: #EEF2F7; font-size: 8.8pt; "
            "font-family: 'Consolas', 'Cascadia Code', monospace; background: transparent;"
        )
        hbox.addWidget(self._lbl_status)

        hbox.addStretch()

        lbl_version = QLabel("FlowSOM-Gui  v3.0  Â·  Magne Florian")
        lbl_version.setStyleSheet(
            "color: #EEF2F7; font-size: 8pt; "
            "font-family: 'Consolas', 'Cascadia Code', monospace; background: transparent;"
        )
        hbox.addWidget(lbl_version)

        return bar

    # â”€â”€ Pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _run_pipeline(self) -> None:
        files = self.panel_import.get_selected_files()
        if not files:
            self.panel_exec.console.append_log(
                "[WARNING] Aucun fichier FCS sÃ©lectionnÃ©. Importez d'abord des donnÃ©es."
            )
            return

        params = self.panel_settings.get_params()
        self.panel_exec.set_running(True)
        self._lbl_status.setText("Pipeline en cours d'exÃ©cutionâ€¦")
        self._lbl_status.setStyleSheet("color: #5BAAFF; font-size: 9pt; background: transparent;")
        self.panel_exec.console.append_log(
            f"[INFO] DÃ©marrage du pipeline â€” {len(files)} fichier(s) Â· "
            f"mÃ©thode : {params['mrd_method']}"
        )
        # DÃ©lÃ©guer au worker de la fenÃªtre parente si disponible
        if self.parent() and hasattr(self.parent(), "_run_pipeline"):
            self.parent()._run_pipeline()

    def _stop_pipeline(self) -> None:
        self.panel_exec.set_running(False)
        self.panel_exec.console.append_log("[WARNING] Pipeline interrompu par l'utilisateur.")
        self._lbl_status.setText("Pipeline arrÃªtÃ©.")
        self._lbl_status.setStyleSheet("color: #FF9B3D; font-size: 9pt; background: transparent;")

    # â”€â”€ DÃ©mo logs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _demo_logs(self) -> None:
        """Affiche quelques lignes d'exemple dans la console au dÃ©marrage."""
        self.panel_exec.console.append_log(
            "[INFO] 2026-04-15 09:00:00 â€” FlowSOM Dashboard initialisÃ©."
        )
        self.panel_exec.console.append_log("[INFO] Design system 'PRISMA v2' chargÃ©.")
        self.panel_exec.console.append_log(
            "[INFO] Importez vos fichiers FCS et configurez le pipeline pour commencer."
        )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Bouton d'expansion â€” Ã  intÃ©grer dans la fenÃªtre principale
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


def make_expand_button(parent_window) -> QPushButton:
    """
    CrÃ©e un bouton â›¶ "Expand Dashboard" Ã  placer dans la fenÃªtre principale.
    Au clic, ouvre PipelineDashboard en tant que fenÃªtre modale.

    Usage dans main_window.py :
        from gui.dialogs.pipeline_dashboard import make_expand_button
        btn = make_expand_button(self)
        some_layout.addWidget(btn)
    """
    btn = QPushButton()
    btn.setObjectName("iconBtn")
    btn.setToolTip("Ouvrir le Dashboard Pipeline (vue 3 colonnes)")
    btn.setFixedSize(36, 36)
    if _QTA:
        try:
            btn.setIcon(qta.icon("fa5s.expand-alt", color=_C["blue"]))
            btn.setIconSize(QSize(16, 16))
        except Exception:
            btn.setText("â›¶")

    def _open():
        dlg = PipelineDashboard(parent_window)
        dlg.exec_()

    btn.clicked.connect(_open)
    return btn

