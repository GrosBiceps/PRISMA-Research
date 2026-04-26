# -*- coding: utf-8 -*-
"""
wizard_main.py — PRISMA Research Wizard RUO v3.0

Wizard interactif 6 étapes. Direction artistique identique à main_window.py:
  - STYLESHEET + COLORS de gui/styles.py
  - SVG logo PRISMA dans la sidebar
  - QPushButton#stepBtn / stepBtnActive / stepBtnDone
  - get_prisma_icon + qtawesome
  - DropZoneLabel drag-drop réel
  - DarkComboBox sans fond blanc popup

Toutes les fonctionnalités RUO sans aucune logique MRD/clinique.
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QByteArray, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontDatabase, QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    from PyQt5.QtSvg import QSvgWidget

    _SVG_AVAILABLE = True
except ImportError:
    _SVG_AVAILABLE = False

log = logging.getLogger("prisma.wizard")

# ── Design system ──────────────────────────────────────────────────────────────
from gui.styles import STYLESHEET, COLORS

# ── Icônes PRISMA ──────────────────────────────────────────────────────────────
from gui.prisma_icons import get_prisma_icon

try:
    import qtawesome as qta

    _QTA = True
except ImportError:
    _QTA = False


def _icon(name: str, color: str = "#EEF2F7", size: int = 16) -> Optional[QIcon]:
    """Retourne un QIcon PRISMA, sinon qtawesome, sinon None."""
    if name.startswith("prisma."):
        custom = get_prisma_icon(name.split(".", 1)[1], size=size, color=color)
        if custom is not None:
            return custom
    if _QTA:
        try:
            return qta.icon(name, color=color)
        except Exception:
            pass
    return None


def _asset_path(filename: str) -> Path:
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        base = Path(__file__).parent.parent
    return base / "assets" / filename


def _register_embedded_fonts() -> None:
    for rel in ["fonts/Outfit[wght].ttf"]:
        p = _asset_path(rel)
        if p.exists():
            QFontDatabase.addApplicationFont(str(p))


# ── DarkComboBox ───────────────────────────────────────────────────────────────
class DarkComboBox(QComboBox):
    def showPopup(self) -> None:
        popup = self.view().window()
        popup.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        super().showPopup()


# ══════════════════════════════════════════════════════════════════════════════
# DropZoneLabel — drag-drop FCS réel (repris de main_window.py)
# ══════════════════════════════════════════════════════════════════════════════


class DropZoneLabel(QLabel):
    files_dropped = pyqtSignal(list)  # liste de chemins

    def __init__(self, placeholder: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(placeholder, parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(88)
        self.setWordWrap(True)
        self._paths: List[str] = []

    def dragEnterEvent(self, event: Any) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("dragOver", True)
            self.style().unpolish(self)
            self.style().polish(self)

    def dragLeaveEvent(self, event: Any) -> None:
        self.setProperty("dragOver", False)
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, event: Any) -> None:
        self.setProperty("dragOver", False)
        self.style().unpolish(self)
        self.style().polish(self)
        paths = []
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if Path(p).suffix.lower() in (".fcs",):
                paths.append(p)
            elif Path(p).is_dir():
                paths += [str(f) for f in Path(p).glob("*.fcs")]
        if paths:
            self._set_paths(paths)

    def _set_paths(self, paths: List[str]) -> None:
        self._paths = paths
        self.setText(f"  {len(paths)} fichier(s) FCS chargé(s)")
        self.setObjectName("dropZoneOk")
        self.style().unpolish(self)
        self.style().polish(self)
        self.files_dropped.emit(paths)

    @property
    def paths(self) -> List[str]:
        return self._paths


# ══════════════════════════════════════════════════════════════════════════════
# SVG Logo (identique à main_window.py)
# ══════════════════════════════════════════════════════════════════════════════

_LOGO_SVG = b"""<svg width="196" height="56" viewBox="0 0 196 56" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="lg1s" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="white" stop-opacity=".5"/>
      <stop offset="100%" stop-color="white" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <polygon points="26,4 48,50 4,50" fill="rgba(255,255,255,.025)" stroke="rgba(255,255,255,.18)" stroke-width="1.1"/>
  <circle cx="26" cy="27" r="2" fill="white" opacity=".35"/>
  <circle cx="26" cy="27" r="5.5" fill="url(#lg1s)"/>
  <line x1="26" y1="27" x2="56" y2="10" stroke="#7B52FF" stroke-width="1" opacity=".7"/>
  <line x1="26" y1="27" x2="56" y2="19" stroke="#5BAAFF" stroke-width=".9" opacity=".55"/>
  <line x1="26" y1="27" x2="56" y2="27" stroke="#39FF8A" stroke-width="1.3" opacity=".8"/>
  <line x1="26" y1="27" x2="56" y2="36" stroke="#FF9B3D" stroke-width=".9" opacity=".6"/>
  <line x1="26" y1="27" x2="56" y2="45" stroke="#FF3D6E" stroke-width="1" opacity=".7"/>
  <circle cx="58" cy="9" r="2.5" fill="#7B52FF"/><circle cx="65" cy="6" r="1.8" fill="#7B52FF" opacity=".55"/>
  <circle cx="61" cy="15" r="1.5" fill="#7B52FF" opacity=".4"/>
  <circle cx="58" cy="18" r="2.2" fill="#5BAAFF" opacity=".8"/><circle cx="65" cy="14" r="1.5" fill="#5BAAFF" opacity=".45"/>
  <circle cx="59" cy="26" r="3" fill="#39FF8A"/><circle cx="67" cy="23" r="1.8" fill="#39FF8A" opacity=".6"/>
  <circle cx="65" cy="30" r="1.5" fill="#39FF8A" opacity=".5"/>
  <circle cx="58" cy="36" r="2.5" fill="#FF9B3D" opacity=".88"/><circle cx="66" cy="32" r="1.5" fill="#FF9B3D" opacity=".5"/>
  <circle cx="64" cy="40" r="1.8" fill="#FF9B3D" opacity=".45"/>
  <circle cx="58" cy="44" r="2.2" fill="#FF3D6E"/><circle cx="65" cy="41" r="1.2" fill="#FF3D6E" opacity=".5"/>
  <text x="76" y="34" font-family="Segoe UI, sans-serif" font-weight="900" font-size="26" fill="white" letter-spacing="-1.5">PRISM</text>
  <text x="162" y="34" font-family="Segoe UI, sans-serif" font-weight="900" font-size="26" fill="none" stroke="#39FF8A" stroke-width="1.5" letter-spacing="-1.5">A</text>
</svg>"""


# ══════════════════════════════════════════════════════════════════════════════
# Définition des 6 étapes
# ══════════════════════════════════════════════════════════════════════════════

_STEPS = [
    ("1", "Input & QC", "Import FCS · PeacoQC · FlowAI"),
    ("2", "Gating", "Auto-gating · Manuel · Spectral"),
    ("3", "Dim. Reduction", "viSNE · UMAP · PHATE"),
    ("4", "HD Clustering", "FlowSOM · PhenoGraph · HDBSCAN · PARC · SPADE"),
    ("5", "Biomarker & Adv.", "CITRUS · Trajectoire · Harmony"),
    ("6", "Export", "FCS · CSV · PDF · HTML"),
]

_STEP_ICONS = [
    "prisma.fcs-file",
    "prisma.gate",
    "prisma.scatter",
    "prisma.som-grid",
    "prisma.heatmap",
    "prisma.export",
]

# ══════════════════════════════════════════════════════════════════════════════
# WizardSidebar — reprend exactement StepSidebar de main_window.py
# ══════════════════════════════════════════════════════════════════════════════


class WizardSidebar(QWidget):
    STATE_PENDING = 0
    STATE_ACTIVE = 1
    STATE_DONE = 2
    STATE_ERROR = 3

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("stepSidebar")
        self.setFixedWidth(220)
        self._buttons: List[QPushButton] = []
        self._states: List[int] = [self.STATE_PENDING] * len(_STEPS)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header SVG logo ───────────────────────────────────────────────────
        header = QWidget()
        header.setObjectName("sidebarHeader")
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(12, 16, 12, 14)
        h_layout.setSpacing(4)

        if _SVG_AVAILABLE:
            logo = QSvgWidget()
            logo.load(QByteArray(_LOGO_SVG))
            logo.setFixedSize(196, 56)
            logo.setStyleSheet("background: transparent;")
            h_layout.addWidget(logo, alignment=Qt.AlignLeft)
        else:
            lbl = QLabel("PRISMA")
            lbl.setObjectName("brandTitle")
            h_layout.addWidget(lbl)

        sub = QLabel("RESEARCH · RUO · ANALYSIS SUITE")
        sub.setObjectName("brandSubtitle")
        h_layout.addWidget(sub)

        sep = QFrame()
        sep.setObjectName("sidebarSeparator")
        sep.setFrameShape(QFrame.HLine)
        h_layout.addWidget(sep)

        root.addWidget(header)
        root.addSpacing(8)

        # ── Boutons d'étapes ──────────────────────────────────────────────────
        for i in range(len(_STEPS)):
            btn = QPushButton()
            btn.setObjectName("stepBtn")
            btn.setCheckable(False)
            btn.setFlat(True)
            self._set_button_content(btn, i)
            btn.clicked.connect(lambda checked, idx=i: self._on_click(idx))
            self._buttons.append(btn)
            root.addWidget(btn)

        root.addStretch()

        ver = QLabel("V3.0-RUO · PRISMA CYTOMETRY")
        ver.setObjectName("versionLabel")
        ver.setAlignment(Qt.AlignLeft)
        root.addWidget(ver)

    def _set_button_content(self, btn: QPushButton, idx: int) -> None:
        num, title, sub = _STEPS[idx]
        state = self._states[idx]

        ic_name = _STEP_ICONS[idx]
        if state == self.STATE_DONE:
            ic_color = COLORS["accent"]
            ic_name = "prisma.check-circle"
        elif state == self.STATE_ERROR:
            ic_color = COLORS["danger"]
            ic_name = "prisma.alert-triangle"
        elif state == self.STATE_ACTIVE:
            ic_color = COLORS["brand"]
        else:
            ic_color = "#2A3342"

        ico = _icon(ic_name, ic_color, 24)
        if ico:
            btn.setIcon(ico)
            btn.setIconSize(QSize(24, 24))

        btn.setText(f"  {title}\n  {sub}")
        btn.setFont(QFont("Segoe UI", 9))

        if state == self.STATE_ACTIVE:
            btn.setObjectName("stepBtnActive")
        elif state == self.STATE_DONE:
            btn.setObjectName("stepBtnDone")
        else:
            btn.setObjectName("stepBtn")

        btn.style().unpolish(btn)
        btn.style().polish(btn)

    def set_active(self, idx: int) -> None:
        for i, s in enumerate(self._states):
            if s == self.STATE_ACTIVE:
                self._states[i] = self.STATE_PENDING
        self._states[idx] = self.STATE_ACTIVE
        self._refresh()

    def set_done(self, idx: int) -> None:
        self._states[idx] = self.STATE_DONE
        self._refresh()

    def set_error(self, idx: int) -> None:
        self._states[idx] = self.STATE_ERROR
        self._refresh()

    def _refresh(self) -> None:
        for i, btn in enumerate(self._buttons):
            self._set_button_content(btn, i)

    def _on_click(self, idx: int) -> None:
        p = self.parent()
        while p:
            if isinstance(p, PrismaWizard):
                p._navigate(idx)
                return
            p = p.parent() if hasattr(p, "parent") else None


# ══════════════════════════════════════════════════════════════════════════════
# FcsCanvasPanel — panneau droit escamotable
# ══════════════════════════════════════════════════════════════════════════════


class FcsCanvasPanel(QWidget):
    """Panneau droit escamotable: Scatter / Population Tree / Sunburst."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("rightPanel")
        self.setMinimumWidth(260)
        self.setMaximumWidth(480)
        self._build()

    def _build(self) -> None:
        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        # Header
        header = QWidget()
        header.setObjectName("titleBar")
        header.setFixedHeight(44)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(14, 0, 8, 0)

        title = QLabel("PRÉVISUALISATION")
        title.setObjectName("brandSubtitle")
        hl.addWidget(title)
        hl.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.setObjectName("ghostBtn")
        close_btn.clicked.connect(lambda: self.setVisible(False))
        hl.addWidget(close_btn)
        vl.addWidget(header)

        # Tabs
        self._tabs = QTabWidget()
        self._tabs.setObjectName("paramTabs")

        for title_tab, msg in [
            ("Scatter", "Importer des fichiers FCS\npuis lancer une analyse\npour visualiser."),
            ("Tree", "Arbres CITRUS / SPADE\naprès analyse clustering."),
            ("Sunburst", "Diagramme sunburst\ndes populations identifiées."),
        ]:
            ph = QWidget()
            vl_ph = QVBoxLayout(ph)
            vl_ph.setAlignment(Qt.AlignCenter)
            lbl = QLabel(f"◎\n\n{title_tab}\n\n{msg}")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setObjectName("subtitleLabel")
            lbl.setWordWrap(True)
            vl_ph.addWidget(lbl)
            self._tabs.addTab(ph, title_tab)

        vl.addWidget(self._tabs, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Pages de base
# ══════════════════════════════════════════════════════════════════════════════


def _scroll_wrap(w: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sa.setWidget(w)
    return sa


class _BaseStepPage(QWidget):
    """
    Layout deux colonnes:
      ┌──────────────┬────────────────────────────────────────┐
      │ Sélecteur    │ QTabWidget dynamique (paramètres)       │
      │ (toggles)    │                                         │
      └──────────────┴────────────────────────────────────────┘
    En-tête plat PRISMA au-dessus.
    """

    def __init__(self, step_idx: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._step_idx = step_idx
        self._tab_map: Dict[str, int] = {}
        self._build_base()

    def _build_base(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── En-tête ────────────────────────────────────────────────────────────
        header = QWidget()
        header.setObjectName("titleBar")
        header.setFixedHeight(56)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(24, 0, 24, 0)
        hl.setSpacing(12)

        num, title, sub = _STEPS[self._step_idx]

        ico_name = _STEP_ICONS[self._step_idx]
        ico = _icon(ico_name, COLORS["brand"], 20)
        if ico:
            ico_lbl = QLabel()
            ico_lbl.setPixmap(ico.pixmap(20, 20))
            hl.addWidget(ico_lbl)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("stepTitle")
        hl.addWidget(title_lbl)

        sep_v = QFrame()
        sep_v.setFrameShape(QFrame.VLine)
        sep_v.setObjectName("sidebarSeparator")
        sep_v.setFixedHeight(20)
        hl.addWidget(sep_v)

        sub_lbl = QLabel(sub)
        sub_lbl.setObjectName("brandSubtitle")
        hl.addWidget(sub_lbl)

        hl.addStretch()
        root.addWidget(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setObjectName("sidebarSeparator")
        root.addWidget(sep)

        # ── Corps ──────────────────────────────────────────────────────────────
        body = QWidget()
        body.setObjectName("stepContent")
        body_hl = QHBoxLayout(body)
        body_hl.setContentsMargins(0, 0, 0, 0)
        body_hl.setSpacing(0)

        # Panneau gauche: sélecteur d'algorithmes
        self._selector_panel = self._build_selector()
        body_hl.addWidget(self._selector_panel)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setObjectName("sidebarSeparator")
        body_hl.addWidget(sep2)

        # Panneau central: paramètres dynamiques
        self._params_widget = self._build_params()
        body_hl.addWidget(self._params_widget, 1)

        root.addWidget(body, 1)

    # ── Overrides dans les sous-classes ───────────────────────────────────────

    def _build_selector(self) -> QWidget:
        w = QWidget()
        w.setObjectName("leftPanel")
        w.setFixedWidth(196)
        return w

    def _build_params(self) -> QWidget:
        w = QWidget()
        w.setObjectName("stepContent")
        return w

    def _enabled_toggle_keys(self) -> List[str]:
        toggles = getattr(self, "_toggles", {})
        if not isinstance(toggles, dict):
            return []
        enabled: List[str] = []
        for key, widget in toggles.items():
            if key.startswith("__"):  # clés internes (__gpu__, …) exclues
                continue
            try:
                if bool(widget.isChecked()):
                    enabled.append(str(key))
            except Exception:
                continue
        return enabled

    def _active_method_key(self, fallback: Optional[str] = None) -> Optional[str]:
        enabled = self._enabled_toggle_keys()
        if not enabled:
            return fallback

        tab_map = getattr(self, "_tab_map", {})
        tab_widget = getattr(self, "_tab_widget", None)
        if tab_widget is not None and isinstance(tab_map, dict):
            try:
                current_idx = int(tab_widget.currentIndex())
                for key in enabled:
                    if key in tab_map and int(tab_map[key]) == current_idx:
                        return key
            except Exception:
                pass
        return enabled[0]

    @staticmethod
    def _collect_form_values(form: QWidget) -> Dict[str, Any]:
        from PyQt5.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox

        values: Dict[str, Any] = {}
        for name, value in vars(form).items():
            key = str(name)
            for prefix in ("spin_", "dspin_", "combo_", "edit_", "toggle_", "chk_"):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    break

            if isinstance(value, QSpinBox):
                values[key] = int(value.value())
            elif isinstance(value, QDoubleSpinBox):
                values[key] = float(value.value())
            elif isinstance(value, QComboBox):
                values[key] = str(value.currentText())
            elif isinstance(value, QLineEdit):
                values[key] = str(value.text()).strip()
            elif isinstance(value, QCheckBox):
                values[key] = bool(value.isChecked())
            elif hasattr(value, "isChecked"):
                try:
                    values[key] = bool(value.isChecked())
                except Exception:
                    continue
        return values

    def _params_for_method(self, method_key: str) -> Dict[str, Any]:
        tab_map = getattr(self, "_tab_map", {})
        tab_widget = getattr(self, "_tab_widget", None)
        if tab_widget is None or method_key not in tab_map:
            return {}

        idx = int(tab_map[method_key])
        page = tab_widget.widget(idx)
        if page is None:
            return {}
        try:
            form = page.widget() if hasattr(page, "widget") else page
        except Exception:
            form = page
        if not isinstance(form, QWidget):
            return {}
        return self._collect_form_values(form)

    def sync_to_config(self, cfg: Any) -> None:
        """Surcharge optionnelle dans les pages spécialisées."""
        return


# ══════════════════════════════════════════════════════════════════════════════
# Helper: panneau sélecteur standardisé
# ══════════════════════════════════════════════════════════════════════════════


def _make_selector_panel(section_title: str, toggles: List[tuple]) -> tuple:
    """
    Crée un panneau sélecteur gauche PRISMA.
    toggles: liste de (key, label, checked_default)
    Retourne (QWidget panel, dict {key: ToggleSwitch})
    """
    from gui.widgets.toggle_switch import ToggleSwitch

    w = QWidget()
    w.setObjectName("leftPanel")
    w.setFixedWidth(196)
    vl = QVBoxLayout(w)
    vl.setContentsMargins(14, 16, 14, 16)
    vl.setSpacing(6)

    hdr = QLabel(section_title.upper())
    hdr.setObjectName("brandSubtitle")
    vl.addWidget(hdr)

    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setObjectName("sidebarSeparator")
    vl.addWidget(sep)
    vl.addSpacing(6)

    widgets: Dict[str, Any] = {}
    for key, label, checked in toggles:
        t = ToggleSwitch(label, checked=checked)
        vl.addWidget(t)
        widgets[key] = t

    vl.addStretch()

    # ── Bloc GPU toggle (commun à toutes les pages algorithmes) ───────────────
    sep_gpu = QFrame()
    sep_gpu.setFrameShape(QFrame.HLine)
    sep_gpu.setObjectName("sidebarSeparator")
    vl.addWidget(sep_gpu)
    vl.addSpacing(4)

    gpu_toggle = ToggleSwitch("Accélération GPU", checked=True)
    gpu_toggle.setToolTip(
        "Active NVIDIA RAPIDS (cuML/cuGraph) si disponible.\n"
        "Désactiver force l'implémentation CPU de référence\n"
        "pour une reproductibilité exacte avec les publications."
    )
    vl.addWidget(gpu_toggle)

    note = QLabel("ℹ CPU = implémentation\nde référence (exacte)\n GPU = portage optimisé")
    note.setObjectName("brandSubtitle")
    note.setWordWrap(True)
    note.setStyleSheet("font-size: 8pt; color: rgba(238,242,247,0.55); padding: 2px 0px;")
    vl.addWidget(note)
    vl.addSpacing(6)

    widgets["__gpu__"] = gpu_toggle
    return w, widgets


def _make_params_tabs(forms: List[tuple]) -> tuple:
    """
    Crée un QTabWidget avec les formulaires fournis.
    forms: liste de (key, label, QWidget_instance)
    Retourne (QWidget container, QTabWidget, dict {key: tab_index})
    """
    w = QWidget()
    w.setObjectName("stepContent")
    vl = QVBoxLayout(w)
    vl.setContentsMargins(0, 0, 0, 0)
    vl.setSpacing(0)

    tabs = QTabWidget()
    tabs.setObjectName("paramTabs")

    tab_map: Dict[str, int] = {}
    for idx, (key, label, form) in enumerate(forms):
        tabs.addTab(_scroll_wrap(form), label)
        tab_map[key] = idx

    vl.addWidget(tabs, 1)
    return w, tabs, tab_map


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Input & QC
# ══════════════════════════════════════════════════════════════════════════════


class _FolderDropZone(DropZoneLabel):
    """DropZone acceptant un dossier (pas des fichiers individuels)."""

    folder_selected = pyqtSignal(str)

    def __init__(self, placeholder: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(placeholder, parent)
        self._folder: str = ""

    def dropEvent(self, event: Any) -> None:
        self.setProperty("dragOver", False)
        self.style().unpolish(self)
        self.style().polish(self)
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if Path(p).is_dir():
                self._set_folder(p)
                return
            # Si un .fcs est déposé, prendre le dossier parent
            if Path(p).suffix.lower() == ".fcs":
                self._set_folder(str(Path(p).parent))
                return

    def _set_folder(self, folder: str) -> None:
        self._folder = folder
        name = Path(folder).name
        self.setText(f"  {name}\n  {folder}")
        self.setObjectName("dropZoneOk")
        self.style().unpolish(self)
        self.style().polish(self)
        self.folder_selected.emit(folder)

    def set_folder(self, folder: str) -> None:
        if folder:
            self._set_folder(folder)

    @property
    def folder(self) -> str:
        return self._folder


class InputQCPage(_BaseStepPage):
    fcs_folder_changed = pyqtSignal(str)
    output_folder_changed = pyqtSignal(str)
    open_gating_workspace_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self._gating_workspace_path: Optional[str] = None
        self._available_populations: List[str] = ["Root"]
        super().__init__(0, parent)

    def _build_selector(self) -> QWidget:
        panel, self._toggles = _make_selector_panel(
            "Méthodes QC",
            [
                ("peacoqc", "PeacoQC", True),
                ("flowai", "FlowAI", False),
            ],
        )
        self._toggles["peacoqc"].toggled.connect(lambda v: self._on_toggle("peacoqc", v))
        self._toggles["flowai"].toggled.connect(lambda v: self._on_toggle("flowai", v))
        return panel

    def _build_params(self) -> QWidget:
        from gui.widgets.parameter_tabs import PeacoQCForm, FlowAIForm

        outer = QWidget()
        outer.setObjectName("stepContent")
        vl = QVBoxLayout(outer)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        import_widget = self._build_import_card()
        import_widget.setFixedHeight(240)
        vl.addWidget(import_widget)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setObjectName("sidebarSeparator")
        vl.addWidget(sep)

        _, self._tab_widget, self._tab_map = _make_params_tabs(
            [
                ("peacoqc", "PeacoQC", PeacoQCForm()),
                ("flowai", "FlowAI", FlowAIForm()),
            ]
        )
        self._tab_widget.setTabVisible(self._tab_map["flowai"], False)
        vl.addWidget(self._tab_widget.parent(), 1)

        return outer

    def _build_import_card(self) -> QWidget:
        card = QWidget()
        card.setObjectName("leftPanel")
        vl = QVBoxLayout(card)
        vl.setContentsMargins(20, 14, 20, 14)
        vl.setSpacing(10)

        # ── Dossier FCS d'entrée ──────────────────────────────────────────────
        row_in = QHBoxLayout()
        row_in.setSpacing(8)

        lbl_in = QLabel("DOSSIER FCS")
        lbl_in.setObjectName("brandSubtitle")
        lbl_in.setFixedWidth(110)
        row_in.addWidget(lbl_in)

        self._drop_fcs = _FolderDropZone("  ↓  Dossier .fcs d'entrée  (glisser ou parcourir)")
        self._drop_fcs.folder_selected.connect(self.fcs_folder_changed)
        row_in.addWidget(self._drop_fcs, 1)

        btn_in = QPushButton("Parcourir…")
        btn_in.setFixedWidth(90)
        btn_in.clicked.connect(self._browse_fcs)
        row_in.addWidget(btn_in)

        vl.addLayout(row_in)

        # ── Dossier de sortie ─────────────────────────────────────────────────
        row_out = QHBoxLayout()
        row_out.setSpacing(8)

        lbl_out = QLabel("DOSSIER SORTIE")
        lbl_out.setObjectName("brandSubtitle")
        lbl_out.setFixedWidth(110)
        row_out.addWidget(lbl_out)

        self._drop_output = _FolderDropZone("  ↓  Dossier de sortie  (glisser ou parcourir)")
        self._drop_output.folder_selected.connect(self.output_folder_changed)
        row_out.addWidget(self._drop_output, 1)

        btn_out = QPushButton("Parcourir…")
        btn_out.setFixedWidth(90)
        btn_out.clicked.connect(self._browse_output)
        row_out.addWidget(btn_out)

        vl.addLayout(row_out)

        # ── Contexte de données (gating workspace) ──────────────────────────
        row_gate = QHBoxLayout()
        row_gate.setSpacing(8)

        lbl_gate = QLabel("CONTEXTE GATING")
        lbl_gate.setObjectName("brandSubtitle")
        lbl_gate.setFixedWidth(110)
        row_gate.addWidget(lbl_gate)

        gate_box = QWidget()
        gate_vl = QVBoxLayout(gate_box)
        gate_vl.setContentsMargins(0, 0, 0, 0)
        gate_vl.setSpacing(6)

        self._btn_open_gating_workspace = QPushButton("🎯 Ouvrir l'Espace de Gating")
        self._btn_open_gating_workspace.setObjectName("ghostBtn")
        self._btn_open_gating_workspace.clicked.connect(
            lambda: self.open_gating_workspace_requested.emit()
        )
        gate_vl.addWidget(self._btn_open_gating_workspace)

        gate_hl = QHBoxLayout()
        gate_hl.setContentsMargins(0, 0, 0, 0)
        gate_hl.setSpacing(8)

        lbl_target = QLabel("Population cible pour l'analyse")
        lbl_target.setObjectName("brandSubtitle")
        lbl_target.setFixedWidth(190)
        gate_hl.addWidget(lbl_target)

        self._combo_target_population = DarkComboBox()
        self._combo_target_population.addItem("Toutes les cellules (Root)", "Root")
        self._combo_target_population.setEnabled(False)
        gate_hl.addWidget(self._combo_target_population, 1)
        gate_vl.addLayout(gate_hl)

        self._lbl_gating_status = QLabel("Aucun workspace de gating actif")
        self._lbl_gating_status.setObjectName("brandSubtitle")
        gate_vl.addWidget(self._lbl_gating_status)

        row_gate.addWidget(gate_box, 1)
        vl.addLayout(row_gate)

        return card

    def _browse_fcs(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Dossier FCS d'entrée")
        if folder:
            self._drop_fcs.set_folder(folder)

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Dossier de sortie")
        if folder:
            self._drop_output.set_folder(folder)

    @property
    def fcs_folder(self) -> str:
        return self._drop_fcs.folder

    @property
    def output_folder(self) -> str:
        return self._drop_output.folder

    @property
    def target_population(self) -> str:
        if self._combo_target_population.count() == 0:
            return "Root"
        data = self._combo_target_population.currentData()
        if data is None:
            return "Root"
        target = str(data).strip()
        return target or "Root"

    @property
    def gating_workspace_path(self) -> Optional[str]:
        return self._gating_workspace_path

    def set_gating_context(
        self,
        workspace_path: str,
        populations: List[str],
        preferred_population: Optional[str] = None,
    ) -> None:
        """Met à jour le contexte de gating affiché dans la page d'entrée."""
        cleaned: List[str] = ["Root"]
        for pop in populations:
            p = str(pop).strip()
            if p and p not in cleaned:
                cleaned.append(p)

        previous = preferred_population or self.target_population
        self._gating_workspace_path = str(workspace_path)
        self._available_populations = cleaned

        self._combo_target_population.blockSignals(True)
        self._combo_target_population.clear()
        self._combo_target_population.addItem("Toutes les cellules (Root)", "Root")
        for pop in cleaned:
            if pop == "Root":
                continue
            self._combo_target_population.addItem(pop, pop)

        target_to_select = previous if previous in cleaned else "Root"
        idx = self._combo_target_population.findData(target_to_select)
        if idx < 0:
            idx = 0
        self._combo_target_population.setCurrentIndex(idx)
        self._combo_target_population.setEnabled(True)
        self._combo_target_population.blockSignals(False)

        self._lbl_gating_status.setText(
            f"Workspace actif: {Path(self._gating_workspace_path).name}"
        )

    def clear_gating_context(self) -> None:
        """Réinitialise le contexte de gating (fallback robuste vers Root)."""
        self._gating_workspace_path = None
        self._available_populations = ["Root"]

        self._combo_target_population.blockSignals(True)
        self._combo_target_population.clear()
        self._combo_target_population.addItem("Toutes les cellules (Root)", "Root")
        self._combo_target_population.setCurrentIndex(0)
        self._combo_target_population.setEnabled(False)
        self._combo_target_population.blockSignals(False)

        self._lbl_gating_status.setText("Aucun workspace de gating actif")

    def _on_toggle(self, key: str, visible: bool) -> None:
        idx = self._tab_map.get(key)
        if idx is not None:
            self._tab_widget.setTabVisible(idx, visible)

    def sync_to_config(self, cfg: Any) -> None:
        enabled = self._enabled_toggle_keys()
        selected = self._active_method_key(fallback="peacoqc") or "peacoqc"

        cfg.qc_methods_enabled = enabled or [selected]
        cfg.qc_method = selected

        wizard_cfg = cfg._extra.setdefault("wizard", {})
        wizard_cfg["qc_params"] = {
            key: self._params_for_method(key) for key in cfg.qc_methods_enabled
        }

        if self.gating_workspace_path:
            cfg.gating_workspace_path = self.gating_workspace_path
            cfg.target_population = self.target_population or "Root"
        else:
            cfg.gating_workspace_path = None
            cfg.target_population = "Root"


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Gating Strategy
# ══════════════════════════════════════════════════════════════════════════════


class GatingPage(_BaseStepPage):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(1, parent)

    def _build_selector(self) -> QWidget:
        panel, self._toggles = _make_selector_panel(
            "Mode Gating",
            [
                ("auto", "Auto-gating", True),
                ("manual", "Manuel / Hiérarchie", False),
                ("pregate", "Pré-gating spectral", False),
            ],
        )
        for key in self._toggles:
            self._toggles[key].toggled.connect(lambda v, k=key: self._on_toggle(k, v))
        return panel

    def _build_params(self) -> QWidget:
        from gui.widgets.parameter_tabs import GatingStrategyForm

        w, self._tab_widget, self._tab_map = _make_params_tabs(
            [
                ("auto", "Stratégie", GatingStrategyForm()),
                ("manual", "Manuel", GatingStrategyForm()),
                ("pregate", "Spectral", GatingStrategyForm()),
            ]
        )
        self._tab_widget.setTabVisible(self._tab_map["manual"], False)
        self._tab_widget.setTabVisible(self._tab_map["pregate"], False)
        return w

    def _on_toggle(self, key: str, visible: bool) -> None:
        idx = self._tab_map.get(key)
        if idx is not None:
            self._tab_widget.setTabVisible(idx, visible)

    def sync_to_config(self, cfg: Any) -> None:
        enabled = self._enabled_toggle_keys()
        selected = self._active_method_key(fallback="auto") or "auto"

        cfg.pregate.apply = bool(enabled)
        cfg.pregate.mode = "manual" if selected == "manual" else "auto"

        wizard_cfg = cfg._extra.setdefault("wizard", {})
        wizard_cfg["gating_method"] = selected
        wizard_cfg["gating_methods_enabled"] = enabled
        wizard_cfg["gating_params"] = {key: self._params_for_method(key) for key in enabled}


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Dimensionality Reduction
# ══════════════════════════════════════════════════════════════════════════════


class DimRedPage(_BaseStepPage):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(2, parent)

    def _build_selector(self) -> QWidget:
        panel, self._toggles = _make_selector_panel(
            "Algorithmes",
            [
                ("visne", "viSNE  [GPU+/Réf:CPU]", True),
                ("umap", "UMAP   [GPU+/Réf:CPU]", True),
                ("phate", "PHATE  [Réf:CPU]", False),
            ],
        )
        for key in self._toggles:
            if key == "__gpu__":
                continue
            self._toggles[key].toggled.connect(lambda v, k=key: self._on_toggle(k, v))
        return panel

    def _build_params(self) -> QWidget:
        from gui.widgets.parameter_tabs import ViSNEForm, UMAPForm, PHATEForm

        w, self._tab_widget, self._tab_map = _make_params_tabs(
            [
                ("visne", "viSNE", ViSNEForm()),
                ("umap", "UMAP", UMAPForm()),
                ("phate", "PHATE", PHATEForm()),
            ]
        )
        self._tab_widget.setTabVisible(self._tab_map["phate"], False)
        return w

    def _on_toggle(self, key: str, visible: bool) -> None:
        idx = self._tab_map.get(key)
        if idx is not None:
            self._tab_widget.setTabVisible(idx, visible)

    def sync_to_config(self, cfg: Any) -> None:
        from prisma.core.gpu_context import GPUContext

        enabled = self._enabled_toggle_keys()
        selected = self._active_method_key(fallback="umap") or "umap"

        cfg.dimred_methods_enabled = enabled or [selected]
        cfg.dimred_method = selected

        gpu_toggle = self._toggles.get("__gpu__")
        use_gpu = gpu_toggle.isChecked() if gpu_toggle is not None else True
        cfg.gpu.enabled = use_gpu
        GPUContext.set(use_gpu)

        wizard_cfg = cfg._extra.setdefault("wizard", {})
        wizard_cfg["dimred_params"] = {
            key: self._params_for_method(key) for key in cfg.dimred_methods_enabled
        }


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — HD Clustering
# ══════════════════════════════════════════════════════════════════════════════


class ClusteringPage(_BaseStepPage):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(3, parent)

    def _build_selector(self) -> QWidget:
        panel, self._toggles = _make_selector_panel(
            "Algorithmes",
            [
                ("flowsom", "FlowSOM      [Réf:CPU]", True),
                ("phenograph", "PhenoGraph   [GPU+/Réf:CPU]", False),
                ("hdbscan", "HDBSCAN      [GPU+/Réf:CPU]", False),
                ("parc", "PARC         [Réf:CPU]", False),
                ("spade", "SPADE        [Réf:CPU]", False),
            ],
        )
        for key in self._toggles:
            if key == "__gpu__":
                continue
            self._toggles[key].toggled.connect(lambda v, k=key: self._on_toggle(k, v))
        return panel

    def _build_params(self) -> QWidget:
        from gui.widgets.parameter_tabs import (
            FlowSOMForm,
            PhenoGraphForm,
            HDBSCANForm,
            PARCForm,
            SPADEForm,
        )

        w, self._tab_widget, self._tab_map = _make_params_tabs(
            [
                ("flowsom", "FlowSOM", FlowSOMForm()),
                ("phenograph", "PhenoGraph", PhenoGraphForm()),
                ("hdbscan", "HDBSCAN", HDBSCANForm()),
                ("parc", "PARC", PARCForm()),
                ("spade", "SPADE", SPADEForm()),
            ]
        )
        for key in ["phenograph", "hdbscan", "parc", "spade"]:
            self._tab_widget.setTabVisible(self._tab_map[key], False)
        return w

    def _on_toggle(self, key: str, visible: bool) -> None:
        idx = self._tab_map.get(key)
        if idx is not None:
            self._tab_widget.setTabVisible(idx, visible)

    def sync_to_config(self, cfg: Any) -> None:
        from prisma.core.gpu_context import GPUContext

        enabled = self._enabled_toggle_keys()
        selected = self._active_method_key(fallback="flowsom") or "flowsom"

        cfg.clustering_methods_enabled = enabled or [selected]
        cfg.clustering_method = selected

        gpu_toggle = self._toggles.get("__gpu__")
        use_gpu = gpu_toggle.isChecked() if gpu_toggle is not None else True
        cfg.gpu.enabled = use_gpu
        GPUContext.set(use_gpu)

        wizard_cfg = cfg._extra.setdefault("wizard", {})
        wizard_cfg["clustering_params"] = {
            key: self._params_for_method(key) for key in cfg.clustering_methods_enabled
        }


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Biomarker & Advanced
# ══════════════════════════════════════════════════════════════════════════════


class AdvancedPage(_BaseStepPage):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(4, parent)

    def _build_selector(self) -> QWidget:
        from gui.widgets.toggle_switch import ToggleSwitch

        w = QWidget()
        w.setObjectName("leftPanel")
        w.setFixedWidth(196)
        vl = QVBoxLayout(w)
        vl.setContentsMargins(14, 16, 14, 16)
        vl.setSpacing(6)

        hdr = QLabel("ANALYSES AVANCÉES")
        hdr.setObjectName("brandSubtitle")
        vl.addWidget(hdr)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setObjectName("sidebarSeparator")
        vl.addWidget(sep)
        vl.addSpacing(6)

        self._toggles = {}
        for key, label, checked in [
            ("citrus", "CITRUS", False),
            ("wanderlust", "Wanderlust", False),
            ("wishbone", "Wishbone", False),
            ("harmony", "Harmony (Batch)", False),
        ]:
            t = ToggleSwitch(label, checked=checked)
            t.toggled.connect(lambda v, k=key: self._on_toggle(k, v))
            vl.addWidget(t)
            self._toggles[key] = t

        vl.addStretch()

        # Note verrouillage Harmony
        note = QLabel("⚠  Harmony: correction\nSSC uniquement (RUO)")
        note.setObjectName("brandSubtitle")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {COLORS['warn']}; padding-top: 8px;")
        vl.addWidget(note)

        return w

    def _build_params(self) -> QWidget:
        from gui.widgets.parameter_tabs import CITRUSForm, WanderlustForm, WishboneForm, HarmonyForm

        # Placeholder quand rien de sélectionné
        ph = QWidget()
        ph_vl = QVBoxLayout(ph)
        ph_vl.setAlignment(Qt.AlignCenter)
        lbl = QLabel("Activer une méthode dans\nle panneau gauche\npour configurer ses paramètres.")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setObjectName("subtitleLabel")
        ph_vl.addWidget(lbl)

        w, self._tab_widget, self._tab_map = _make_params_tabs(
            [
                ("placeholder", "—", ph),
                ("citrus", "CITRUS", CITRUSForm()),
                ("wanderlust", "Wanderlust", WanderlustForm()),
                ("wishbone", "Wishbone", WishboneForm()),
                ("harmony", "Harmony", HarmonyForm()),
            ]
        )
        for key in ["citrus", "wanderlust", "wishbone", "harmony"]:
            self._tab_widget.setTabVisible(self._tab_map[key], False)
        return w

    def _on_toggle(self, key: str, visible: bool) -> None:
        idx = self._tab_map.get(key)
        if idx is not None:
            self._tab_widget.setTabVisible(idx, visible)
        # Placeholder: masqué si au moins une méthode visible
        any_active = any(
            self._tab_widget.isTabVisible(self._tab_map[k])
            for k in ["citrus", "wanderlust", "wishbone", "harmony"]
        )
        ph_idx = self._tab_map.get("placeholder")
        if ph_idx is not None:
            self._tab_widget.setTabVisible(ph_idx, not any_active)

    def sync_to_config(self, cfg: Any) -> None:
        enabled = self._enabled_toggle_keys()
        cfg.data_integration.enabled = "harmony" in enabled

        wizard_cfg = cfg._extra.setdefault("wizard", {})
        wizard_cfg["advanced_methods_enabled"] = enabled
        wizard_cfg["advanced_params"] = {key: self._params_for_method(key) for key in enabled}


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Export & Reporting
# ══════════════════════════════════════════════════════════════════════════════


class ExportPage(_BaseStepPage):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(5, parent)

    def _build_selector(self) -> QWidget:
        panel, self._toggles = _make_selector_panel(
            "Formats",
            [
                ("fcs", "FCS annoté", True),
                ("csv", "CSV / Excel", True),
                ("pdf", "Rapport PDF", False),
                ("html", "Rapport HTML", False),
            ],
        )
        return panel

    def _build_params(self) -> QWidget:
        from gui.widgets.parameter_tabs import ExportForm

        w, self._tab_widget, self._tab_map = _make_params_tabs(
            [
                ("export", "Configuration", ExportForm()),
            ]
        )
        return w

    def sync_to_config(self, cfg: Any) -> None:
        enabled = self._enabled_toggle_keys()
        cfg.export_mode.export_csv = "csv" in enabled

        wizard_cfg = cfg._extra.setdefault("wizard", {})
        wizard_cfg["export_methods_enabled"] = enabled
        wizard_cfg["export_params"] = self._params_for_method("export")


# ══════════════════════════════════════════════════════════════════════════════
# PrismaWizard — Fenêtre principale
# ══════════════════════════════════════════════════════════════════════════════


class PrismaWizard(QMainWindow):
    """Fenêtre principale PRISMA Research — Wizard RUO 6 étapes."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PRISMA Research — RUO v3.0")
        self.setMinimumSize(1100, 720)
        # Taille initiale adaptée à l'écran — showMaximized() dans main()

        # Icône fenêtre
        ico_path = _asset_path("prisma_logo.ico")
        if ico_path.exists():
            self.setWindowIcon(QIcon(str(ico_path)))

        # Fonts + stylesheet PRISMA identique à main_window.py
        _register_embedded_fonts()
        self.setStyleSheet(STYLESHEET + _WIZARD_EXTRA_QSS)

        self._current_step = 0
        self._canvas_visible = True
        self._gating_win: Optional[QMainWindow] = None
        self._gating_workspace_widget: Optional[Any] = None

        self._build_ui()
        self._navigate(0)

    # ── Construction UI ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        from gui.widgets.log_console import LogConsole

        central = QWidget()
        central.setObjectName("wizardShell")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar
        self._sidebar = WizardSidebar(self)
        root.addWidget(self._sidebar)

        # Zone principale (header + splitter[stack+log] + nav)
        main_area = QWidget()
        main_area.setObjectName("stepContent")
        main_vl = QVBoxLayout(main_area)
        main_vl.setContentsMargins(0, 0, 0, 0)
        main_vl.setSpacing(0)

        # Action bar (boutons haut)
        main_vl.addWidget(self._build_action_bar())

        # Splitter vertical : pages en haut, log en bas
        splitter = QSplitter(Qt.Vertical)
        splitter.setObjectName("logSplitter")
        splitter.setHandleWidth(4)

        # Pages
        self._stack = QStackedWidget()
        self._stack.setObjectName("stepContent")
        self._input_page = InputQCPage()
        self._pages: List[_BaseStepPage] = [
            self._input_page,
            GatingPage(),
            DimRedPage(),
            ClusteringPage(),
            AdvancedPage(),
            ExportPage(),
        ]
        for page in self._pages:
            self._stack.addWidget(page)
        splitter.addWidget(self._stack)

        # Log console (identique à main_window)
        log_container = QWidget()
        log_container.setObjectName("logContainer")
        log_vl = QVBoxLayout(log_container)
        log_vl.setContentsMargins(0, 0, 0, 0)
        log_vl.setSpacing(0)

        log_header = QWidget()
        log_header.setObjectName("titleBar")
        log_header.setFixedHeight(32)
        lh_hl = QHBoxLayout(log_header)
        lh_hl.setContentsMargins(16, 0, 16, 0)
        lh_hl.setSpacing(8)
        log_title = QLabel("CONSOLE DE LOGS")
        log_title.setObjectName("brandSubtitle")
        lh_hl.addWidget(log_title)
        lh_hl.addStretch()
        clear_btn = QPushButton("Effacer")
        clear_btn.setObjectName("ghostBtn")
        clear_btn.setFixedHeight(24)
        clear_btn.clicked.connect(lambda: self.log_output.clear_logs())
        lh_hl.addWidget(clear_btn)
        log_vl.addWidget(log_header)

        sep_log = QFrame()
        sep_log.setFrameShape(QFrame.HLine)
        sep_log.setObjectName("sidebarSeparator")
        log_vl.addWidget(sep_log)

        self.log_output = LogConsole()
        self.log_output.setObjectName("logConsole")
        self.log_output.setMinimumHeight(120)
        log_vl.addWidget(self.log_output, 1)
        splitter.addWidget(log_container)

        splitter.setSizes([560, 200])
        main_vl.addWidget(splitter, 1)

        # Nav bar (bas)
        main_vl.addWidget(self._build_nav_bar())

        root.addWidget(main_area, 1)

        # Canvas panel droit
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setObjectName("sidebarSeparator")
        root.addWidget(sep)

        self._canvas = FcsCanvasPanel(self)
        root.addWidget(self._canvas)

        # Status bar
        sb = QStatusBar()
        sb.setObjectName("statusBar")
        self.setStatusBar(sb)
        self._status_lbl = QLabel("PRISMA Research · RUO · Prêt")
        sb.addWidget(self._status_lbl)

        self._worker = None

        # Wiring du contexte de données (gating) en mode non linéaire.
        self._input_page.open_gating_workspace_requested.connect(self._open_gating_workspace)
        self._input_page.fcs_folder_changed.connect(self._on_input_fcs_folder_changed)

    def _build_action_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("actionBar")
        bar.setFixedHeight(44)
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(16, 0, 16, 0)
        hl.setSpacing(8)

        hl.addStretch()

        self._canvas_btn = QPushButton("◧  Prévisualisation")
        self._canvas_btn.setObjectName("ghostBtn")
        self._canvas_btn.setCheckable(True)
        self._canvas_btn.setChecked(True)
        self._canvas_btn.clicked.connect(self._toggle_canvas)
        hl.addWidget(self._canvas_btn)

        self._run_btn = QPushButton("▶  Exécuter l'Analyse")
        self._run_btn.setObjectName("primaryBtn")
        self._run_btn.clicked.connect(self._on_run)
        hl.addWidget(self._run_btn)

        return bar

    def _build_nav_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("navBar")
        bar.setFixedHeight(52)
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(16, 0, 16, 0)
        hl.setSpacing(12)

        self._prev_btn = QPushButton("← Précédent")
        self._prev_btn.clicked.connect(self._go_prev)
        hl.addWidget(self._prev_btn)

        hl.addStretch()

        self._step_lbl = QLabel()
        self._step_lbl.setObjectName("brandSubtitle")
        self._step_lbl.setAlignment(Qt.AlignCenter)
        hl.addWidget(self._step_lbl)

        hl.addStretch()

        self._next_btn = QPushButton("Suivant →")
        self._next_btn.setObjectName("primaryBtn")
        self._next_btn.clicked.connect(self._go_next)
        hl.addWidget(self._next_btn)

        return bar

    # ── Navigation ────────────────────────────────────────────────────────────

    def _navigate(self, index: int) -> None:
        n = len(_STEPS)
        if not (0 <= index < n):
            return

        # Marquer l'étape précédente comme DONE
        if self._current_step != index:
            self._sidebar.set_done(self._current_step)

        self._current_step = index
        self._stack.setCurrentIndex(index)
        self._sidebar.set_active(index)

        self._prev_btn.setEnabled(index > 0)
        self._next_btn.setEnabled(index < n - 1)
        self._next_btn.setText("Terminer" if index == n - 1 else "Suivant →")

        num, title, sub = _STEPS[index]
        self._step_lbl.setText(f"Étape {index + 1} / {n}  ·  {title}")
        self._status_lbl.setText(f"Étape {index + 1} / {n} — {title} — {sub}")

    def _go_prev(self) -> None:
        self._navigate(self._current_step - 1)

    def _go_next(self) -> None:
        if self._current_step < len(_STEPS) - 1:
            self._navigate(self._current_step + 1)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _toggle_canvas(self, checked: bool) -> None:
        self._canvas.setVisible(checked)

    def _sync_pages_to_config(self, cfg: Any) -> None:
        """
        Synchronise les paramètres des formulaires vers la config.

        Cette méthode doit être appelée avant le lancement du pipeline.
        Elle boucle sur self._pages et appelle la méthode sync_to_config(cfg)
        de chaque page si elle existe.

        Pour l'instant, les pages ne possèdent pas encore cette interface.
        À implémenter dans les sous-classes de _BaseStepPage.

        Args:
            cfg: PipelineConfig à remplir avec les paramètres des pages.
        """
        for page in self._pages:
            if hasattr(page, "sync_to_config"):
                try:
                    page.sync_to_config(cfg)
                except Exception as exc:
                    log.warning("[wizard] Erreur sync page %s: %s", type(page).__name__, exc)

    def _on_input_fcs_folder_changed(self, folder: str) -> None:
        """Synchronise le workspace de gating avec un nouveau dossier FCS sélectionné."""
        if not folder:
            return

        if self._input_page.gating_workspace_path:
            self.log_output.append_log(
                "[WARNING] Dossier FCS modifié: le contexte de gating courant est invalidé et la cible repasse à Root."
            )
            self._input_page.clear_gating_context()

    def _open_gating_workspace(self) -> None:
        """Ouvre l'espace de gating et le connecte au Wizard comme contexte persistant."""
        fcs_folder = self._input_page.fcs_folder
        if not fcs_folder:
            QMessageBox.warning(
                self,
                "Contexte de données requis",
                "Sélectionnez d'abord un dossier FCS avant d'ouvrir l'espace de gating.",
            )
            return

        if self._gating_win is None or self._gating_workspace_widget is None:
            try:
                from src.prisma.gui.viewer.gating_workspace import PrismaGatingWorkspace

                self._gating_workspace_widget = PrismaGatingWorkspace()
                self._gating_workspace_widget.gatingContextSaved.connect(
                    self._on_gating_context_saved
                )

                self._gating_win = QMainWindow(self)
                self._gating_win.setWindowTitle("PRISMA Gating Workspace")
                self._gating_win.setMinimumSize(1100, 720)
                self._gating_win.setCentralWidget(self._gating_workspace_widget)
                from gui.screen_utils import maximize_window
                maximize_window(self._gating_win)
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Erreur ouverture workspace",
                    f"Impossible d'ouvrir PrismaGatingWorkspace:\n{exc}",
                )
                return

        try:
            assert self._gating_workspace_widget is not None
            if self._input_page.gating_workspace_path:
                self._gating_workspace_widget.load_workspace_state(
                    self._input_page.gating_workspace_path,
                    fallback_fcs_folder=fcs_folder,
                )
            else:
                self._gating_workspace_widget.set_input_source_fcs(fcs_folder)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Contexte de gating",
                (
                    "Le workspace de gating a été ouvert, mais le chargement initial a échoué.\n"
                    f"Détail: {exc}\n\n"
                    "Le Wizard restera sur Root tant que le contexte n'est pas sauvegardé."
                ),
            )

        assert self._gating_win is not None
        self._gating_win.show()
        self._gating_win.raise_()
        self._gating_win.activateWindow()

    def _on_gating_context_saved(self, workspace_path: str, populations: list) -> None:
        """Réception du contrat Workspace -> Wizard (path + populations disponibles)."""
        previous = self._input_page.target_population
        cleaned = [str(p).strip() for p in populations if str(p).strip()]
        if "Root" not in cleaned:
            cleaned.insert(0, "Root")

        self._input_page.set_gating_context(
            workspace_path=workspace_path,
            populations=cleaned,
            preferred_population=previous,
        )
        self.log_output.append_log(
            "[INFO] Contexte de gating synchronisé: %s | populations=%d"
            % (workspace_path, len(cleaned))
        )

    def _on_run(self) -> None:
        from gui.workers import PipelineWorker

        input_page: InputQCPage = self._pages[0]  # type: ignore[assignment]
        fcs_folder = input_page.fcs_folder
        output_folder = input_page.output_folder

        if not fcs_folder:
            self.log_output.append_log("[WARNING] Aucun dossier FCS sélectionné.")
            return
        if not output_folder:
            self.log_output.append_log("[WARNING] Aucun dossier de sortie sélectionné.")
            return

        if self._worker is not None and self._worker.isRunning():
            self.log_output.append_log("[WARNING] Analyse déjà en cours.")
            return

        # Construire la config et synchroniser les pages
        try:
            from config.pipeline_config import PipelineConfig

            cfg = PipelineConfig()

            # ✅ CORRECTION 1 : Mapper correctement vers cfg.paths (pas cfg.input_dir)
            cfg.paths.patho_folder = fcs_folder
            cfg.paths.output_dir = output_folder

            # ✅ CORRECTION 2 : Synchroniser les paramètres depuis les pages formulaires
            # Chaque page devrait implémenter sync_to_config(cfg) pour remplir ses paramètres
            self._sync_pages_to_config(cfg)
            self.log_output.append_log(
                "[INFO] Config RUO: QC=%s | DimRed=%s | Clustering=%s | Target=%s"
                % (cfg.qc_method, cfg.dimred_method, cfg.clustering_method, cfg.target_population)
            )

        except Exception as exc:
            self.log_output.append_log(f"[ERROR] Config: {exc}")
            return

        self._status_lbl.setText("Analyse en cours…")
        self._run_btn.setEnabled(False)

        self._worker = PipelineWorker(cfg, parent=self)
        self._worker.log_message.connect(self._on_log_message)
        self._worker.finished.connect(self._on_pipeline_finished)
        self._worker.error.connect(self._on_pipeline_error)
        self._worker.start()
        log.debug("[wizard] PipelineWorker started — calling start_drain(parent=self)")
        self._worker._log_capture.start_drain(parent=self)
        log.debug("[wizard] start_drain() done")

    def _on_log_message(self, msg: str) -> None:
        self.log_output.append(msg)

    def _stop_worker_drain(self) -> None:
        """Arrête proprement le timer de drainage depuis le thread principal."""
        if self._worker is not None and hasattr(self._worker, "_log_capture"):
            log.debug("[wizard] _stop_worker_drain — calling stop_drain()")
            try:
                self._worker._log_capture.stop_drain()
                log.debug("[wizard] stop_drain() OK")
            except Exception as exc:
                log.warning("[wizard] stop_drain() raised: %s", exc)

    def _on_pipeline_finished(self, result: Any) -> None:
        log.debug("[wizard] _on_pipeline_finished result=%s", result)
        self._stop_worker_drain()
        self._run_btn.setEnabled(True)
        self._status_lbl.setText("Analyse terminée.")
        self.log_output.append_log("[SUCCESS] Pipeline terminé.")

    def _on_pipeline_error(self, msg: str) -> None:
        log.debug("[wizard] _on_pipeline_error msg=%s", msg)
        self._stop_worker_drain()
        self._run_btn.setEnabled(True)
        self._status_lbl.setText("Erreur pipeline.")
        self.log_output.append_log(f"[ERROR] {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# QSS complémentaire — uniquement ce qui manque dans styles.py
# ══════════════════════════════════════════════════════════════════════════════

_WIZARD_EXTRA_QSS = """
/* ghostBtn — variante outline */
QPushButton#ghostBtn {
    background: transparent;
    border: 1px solid rgba(255,255,255,0.10);
    color: rgba(238,242,247,0.70);
    padding: 7px 16px;
    font-size: 9pt;
    letter-spacing: 0.08em;
}
QPushButton#ghostBtn:hover {
    background: rgba(255,255,255,0.05);
    border-color: rgba(255,255,255,0.20);
    color: #EEF2F7;
}
QPushButton#ghostBtn:checked {
    background: rgba(123,82,255,0.12);
    border-color: rgba(123,82,255,0.50);
    color: #7B52FF;
}

/* paramTabs — onglets dynamiques PRISMA */
QTabWidget#paramTabs::pane {
    border: none;
    background: #0C1220;
}
QTabWidget#paramTabs QTabBar::tab {
    background: #080D18;
    color: #4A5568;
    padding: 8px 18px;
    border: none;
    border-bottom: 2px solid transparent;
    font-family: 'Consolas', monospace;
    font-size: 9pt;
    letter-spacing: 0.10em;
    text-transform: uppercase;
}
QTabWidget#paramTabs QTabBar::tab:selected {
    color: #EEF2F7;
    border-bottom: 2px solid #7B52FF;
    background: #101825;
}
QTabWidget#paramTabs QTabBar::tab:hover {
    color: #A0AABC;
    background: rgba(123,82,255,0.05);
}

/* subtitleLabel */
QLabel#subtitleLabel {
    color: #4A5568;
    font-family: 'Consolas', 'Cascadia Code', monospace;
    font-size: 9pt;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════


def _install_global_exception_hook() -> None:
    """Affiche toute exception non gérée dans stderr et une QMessageBox critique."""
    previous_hook = sys.excepthook

    def _qt_excepthook(exc_type, exc_value, exc_tb) -> None:
        traceback.print_exception(exc_type, exc_value, exc_tb)
        log.exception("Exception non gérée dans l'UI", exc_info=(exc_type, exc_value, exc_tb))

        details = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        try:
            app = QApplication.instance()
            parent = app.activeWindow() if app is not None else None
            QMessageBox.critical(
                parent,
                "Erreur inattendue",
                f"{exc_type.__name__}: {exc_value}\n\nVoir le terminal pour le traceback complet.",
            )
        except Exception:
            pass

        if callable(previous_hook):
            try:
                previous_hook(exc_type, exc_value, exc_tb)
            except Exception:
                pass

    sys.excepthook = _qt_excepthook


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("PRISMA Research")
    app.setOrganizationName("PRISMA Cytometry")
    # IMPORTANT: installer le hook global avant d'entrer dans la boucle Qt.
    _install_global_exception_hook()
    win = PrismaWizard()
    from gui.screen_utils import maximize_window
    maximize_window(win)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
