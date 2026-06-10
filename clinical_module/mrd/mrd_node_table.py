# -*- coding: utf-8 -*-
"""
mrd_node_table.py — Grille de Validation par Cartes pour les nœuds SOM MRD positifs.

Architecture : QScrollArea + QGridLayout de MRDNodeCard
  - Chaque nœud suspect est affiché sous forme d'une carte avec :
      • En-tête : Node ID + % Cellules patho
      • Centre  : Graphique Radar (spider plot matplotlib) ou placeholder
      • Bas     : Boutons radio "GARDER (MRD)" / "ÉCARTER (Bruit)"
  - Chaque carte émet decisionChanged(node_id, is_included)
  - MRDCurationGrid accumule les décisions et calcule en temps réel :
      total_mrd_cells  = Σ n_patho des nœuds marqués GARDER
      final_ratio      = (total_mrd_cells / total_viable_cells) * 100
  - get_human_curated_results() retourne la liste des nœuds validés
    pour l'export PDF/CSV.

Compat : PyQt5, Matplotlib Qt5Agg (déjà importé dans home_tab.py)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSize,
    QSortFilterProxyModel,
    Qt,
    QVariant,
    pyqtSignal,
)
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# ── Palette PRISMA v2 ─────────────────────────────────────────────────
_C = {
    "surface0": "#101825",
    "surface1": "#141E2E",
    "base": "#0C1220",
    "mantle": "#080D18",
    "crust": "#04070D",
    "text": "#EEF2F7",
    "subtext": "rgba(238,242,247,0.55)",
    "red": "#FF3D6E",
    "green": "#39FF8A",
    "yellow": "#FFE032",
    "blue": "#5BAAFF",
    "lavender": "#7B52FF",
    "mauve": "#7B52FF",
    "overlay0": "rgba(238,242,247,0.36)",
    "pink": "#FF3D6E",
    "teal": "#5BAAFF",
}

# Colonnes (gardés pour rétrocompat du modèle, non utilisés dans la grille)
_COL_NODE = 0
_COL_METHODS = 1
_COL_PCT_SAIN = 2
_COL_PCT_PATHO = 3
_COL_N_PATHO = 4
_COL_TOTAL = 5

_HEADERS = [
    "Nœud SOM",
    "Méthodes",
    "% Sain",
    "% Patho",
    "Cellules patho",
    "Total nœud",
]

_METHOD_FLAG: Dict[str, str] = {
    "JF": "is_mrd_jf",
    "Flo": "is_mrd_flo",
    "ELN": "is_mrd_eln",
}

# Marqueurs techniques a exclure du radar (coherence avec HomeTab)
_TECHNICAL_MARKERS = {
    "fsc-a",
    "fsc-h",
    "fsc-w",
    "ssc-a",
    "ssc-h",
    "ssc-w",
    "time",
    "event_",
    "event",
    "width",
    "height",
    "area",
    "fsc",
    "ssc",
}

# ═══════════════════════════════════════════════════════════════════════════════
# MRDNodeCard — Carte de validation pour un nœud MRD
# ═══════════════════════════════════════════════════════════════════════════════


class MRDNodeCard(QFrame):
    """
    Carte QFrame représentant un nœud SOM MRD suspect.

    Signaux :
        decisionChanged(node_id: int, is_included: bool)
            Émis quand l'expert change son avis GARDER ↔ ÉCARTER.
    """

    decisionChanged = pyqtSignal(int, bool)

    def __init__(
        self,
        node: Dict[str, Any],
        mfi_data: Any = None,  # DataFrame MFI (index = node_id)
        marker_cols: List[str] = None,
        initial_included: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._node = node
        self._node_id: int = int(node.get("node_id", 0))
        self._is_included: bool = bool(initial_included)
        self._mfi_data = mfi_data
        self._marker_cols: List[str] = marker_cols or []

        self._build_ui()

    # ── Construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setObjectName("mrdNodeCard")
        self._apply_card_style(included=self._is_included)

        # Preferred horizontalement : la carte s'étire dans sa cellule de grille
        # sans rétrécir en dessous de son contenu.
        # MinimumExpanding verticalement : la carte prend la hauteur de son
        # contenu et n'est jamais écrasée à 0 par le layout parent.
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)

        # Plancher absolu : garantit qu'une carte est toujours lisible même
        # quand la fenêtre est très petite.
        self.setMinimumSize(200, 300)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # ── En-tête ──────────────────────────────────────────────────────
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)

        node_id_lbl = QLabel(f"Nœud  {self._node_id}")
        node_id_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        node_id_lbl.setStyleSheet(f"color: {_C['mauve']}; background: transparent;")
        header_row.addWidget(node_id_lbl)

        header_row.addStretch()

        pct_patho = self._node.get("pct_patho", 0.0)
        pct_lbl = QLabel(f"{pct_patho:.1f} %")
        pct_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        pct_lbl.setStyleSheet(f"color: {_C['red']}; background: transparent;")
        header_row.addWidget(pct_lbl)

        root.addLayout(header_row)

        # Méthodes actives
        methods = [m for m, key in _METHOD_FLAG.items() if self._node.get(key)]
        methods_lbl = QLabel("  ·  ".join(methods) if methods else "—")
        methods_lbl.setFont(QFont("Consolas", 8))
        methods_lbl.setAlignment(Qt.AlignCenter)
        methods_lbl.setStyleSheet(
            f"color: {_C['lavender']}; background: rgba(123,82,255,0.12); "
            f"border: 1px solid rgba(123,82,255,0.24); border-radius: 0px; padding: 2px 6px;"
        )
        root.addWidget(methods_lbl)

        # Séparateur
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: rgba(255,255,255,0.055); max-height:1px;")
        root.addWidget(sep)

        # ── Zone radar ───────────────────────────────────────────────────
        self._radar_widget = self._build_radar()
        root.addWidget(self._radar_widget, 1)

        # ── Infos secondaires ────────────────────────────────────────────
        n_patho = self._node.get("n_patho", 0)
        n_cells = self._node.get("n_cells", 0)
        info_lbl = QLabel(f"{n_patho:,} / {n_cells:,} cellules")
        info_lbl.setFont(QFont("Consolas", 8))
        info_lbl.setAlignment(Qt.AlignCenter)
        info_lbl.setStyleSheet(f"color: {_C['subtext']}; background: transparent;")
        root.addWidget(info_lbl)

        # ── Boutons de décision ──────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(6)

        self._btn_keep = QPushButton("✓  GARDER")
        self._btn_keep.setCheckable(True)
        self._btn_keep.setChecked(self._is_included)
        self._btn_keep.setFixedHeight(30)
        self._btn_keep.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._btn_discard = QPushButton("✗  ÉCARTER")
        self._btn_discard.setCheckable(True)
        self._btn_discard.setChecked(not self._is_included)
        self._btn_discard.setFixedHeight(30)
        self._btn_discard.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Groupe exclusif
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        self._btn_group.addButton(self._btn_keep, 0)
        self._btn_group.addButton(self._btn_discard, 1)

        self._apply_btn_styles(included=self._is_included)

        self._btn_keep.clicked.connect(self._on_keep)
        self._btn_discard.clicked.connect(self._on_discard)

        btn_row.addWidget(self._btn_keep)
        btn_row.addWidget(self._btn_discard)
        root.addLayout(btn_row)

    def _build_radar(self) -> QWidget:
        """
        Construit le graphique Radar (spider) pour ce nœud.
        Si les données MFI sont absentes, retourne un placeholder.
        """
        try:
            markers = self._filter_clinical_markers(self._marker_cols)
            if (
                self._mfi_data is not None
                and markers
                and len(markers) >= 3
                and self._node_id in self._mfi_data.index
            ):
                return self._build_radar_matplotlib()
        except Exception:
            pass
        return self._build_radar_placeholder()

    def _filter_clinical_markers(self, markers: List[str]) -> List[str]:
        """Retourne uniquement les marqueurs cliniques (exclut FSC/SSC/Time...)."""
        result: List[str] = []
        for marker in markers:
            m_low = marker.lower().strip()
            if any(m_low == t or m_low.startswith(t) for t in _TECHNICAL_MARKERS):
                continue
            result.append(marker)
        return result

    def _build_radar_matplotlib(self) -> QWidget:
        """Radar réel via matplotlib."""
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        markers = self._filter_clinical_markers(self._marker_cols)
        row = self._mfi_data.loc[self._node_id, markers]
        mfi_row = row.values.astype(float)

        # Normalisation par noeud (identique aux bons radars HomeTab/HTML)
        v_min, v_max = float(mfi_row.min()), float(mfi_row.max())
        norm_values = (mfi_row - v_min) / (v_max - v_min + 1e-10)

        # Labels courts pour eviter le chevauchement dans les petites cartes
        short_labels = []
        for marker in markers:
            parts = marker.strip().split()
            candidate = parts[-1] if len(parts) > 1 else parts[0]
            short_labels.append(candidate[:10])

        N = len(markers)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]
        norm_values = list(norm_values) + [norm_values[0]]

        # Choisir la couleur selon le type de nœud (identique à ExpertNodeCard)
        is_jf = bool(self._node.get("is_mrd_jf", False))
        is_flo = bool(self._node.get("is_mrd_flo", False))
        is_eln = bool(self._node.get("is_mrd_eln", False))
        is_algo = is_jf or is_flo or is_eln
        is_manual = bool(self._node.get("is_mrd_manual", False))

        if is_algo:
            radar_color = "#c084fc"  # Violet — nœud MRD algorithmique
        elif is_manual:
            radar_color = "#39FF8A"  # Vert — ajout manuel
        else:
            radar_color = "#5BAAFF"  # Bleu — nœud standard

        fig = Figure(figsize=(2.1, 2.1), facecolor="#080D18")
        ax = fig.add_subplot(111, polar=True)
        ax.set_facecolor("#0C1220")

        ax.plot(
            angles,
            norm_values,
            color=radar_color,
            linewidth=1.8,
            alpha=1.0,
            marker="o",
            markersize=2.5,
            markerfacecolor=radar_color,
            markeredgewidth=0,
        )
        ax.fill(angles, norm_values, color=radar_color, alpha=0.30)

        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.33, 0.66, 1.0])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            short_labels,
            fontsize=5,
            color="#B0B8C8",
            fontfamily=["Segoe UI", "Arial", "Arial", "sans-serif"],
        )
        ax.set_yticklabels([])
        ax.yaxis.grid(True, color=(1, 1, 1, 0.18), linewidth=0.6, linestyle=":")
        ax.xaxis.grid(True, color=(1, 1, 1, 0.12), linewidth=0.5)
        ax.spines["polar"].set_color((1, 1, 1, 0.25))
        ax.tick_params(pad=3)
        fig.tight_layout(pad=0.3)

        canvas = FigureCanvas(fig)
        # setMinimumSize au lieu de setFixedSize : le canvas peut grandir si la
        # carte s'étire, mais ne descend jamais en dessous de 180×180 px.
        # Expanding+Expanding : le layout parent (stretch=1) peut lui allouer
        # de l'espace supplémentaire sans le bloquer.
        canvas.setMinimumSize(QSize(180, 180))
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return canvas

    def _build_radar_placeholder(self) -> QWidget:
        """Placeholder quand pas de données MFI."""
        ph = QLabel("Radar\nn/d")
        ph.setAlignment(Qt.AlignCenter)
        # setMinimumHeight au lieu de setFixedHeight : le placeholder se défend
        # contre la compression mais n'empêche pas la carte de grandir.
        ph.setMinimumHeight(120)
        ph.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        ph.setStyleSheet(
            f"color: {_C['overlay0']}; background: rgba(20,30,46,0.8); "
            f"border: 1px solid rgba(255,255,255,0.06); border-radius: 0px;"
            f"font-size: 9pt; font-style: italic;"
        )
        return ph

    # ── Styles ───────────────────────────────────────────────────────────────

    def _apply_card_style(self, included: bool) -> None:
        if included:
            border = "rgba(57,255,138,0.40)"
            bg_top = "rgba(14, 36, 25, 0.92)"
            bg_bot = "rgba(10, 27, 19, 0.92)"
            top_acc = "rgba(57,255,138,0.62)"
        else:
            border = "rgba(255,255,255,0.10)"
            bg_top = "rgba(16, 24, 37, 0.90)"
            bg_bot = "rgba(12, 18, 32, 0.90)"
            top_acc = "rgba(123,82,255,0.45)"
        self.setStyleSheet(f"""
            QFrame#mrdNodeCard {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {bg_top}, stop:1 {bg_bot});
                border: 1px solid {border};
                border-top: 1px solid {top_acc};
                border-radius: 0px;
            }}
        """)

    def _apply_btn_styles(self, included: bool) -> None:
        keep_active = """
            QPushButton {
                background: rgba(57,255,138,0.18);
                color: #39FF8A;
                border: 1px solid rgba(57,255,138,0.48);
                border-radius: 0px;
                font-weight: 700;
                font-size: 8pt;
            }
            QPushButton:hover { background: rgba(57,255,138,0.28); }
            QPushButton:pressed { background: rgba(57,255,138,0.34); }
            QPushButton:focus {
                background: rgba(57,255,138,0.30);
                border-color: rgba(57,255,138,0.70);
                outline: none;
            }
        """
        keep_inactive = """
            QPushButton {
                background: rgba(20,30,46,0.80);
                color: #EEF2F7;
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 0px;
                font-weight: 600;
                font-size: 8pt;
            }
            QPushButton:hover { background: rgba(20,30,46,0.95); color: #EEF2F7; }
            QPushButton:pressed { background: rgba(20,30,46,1.0); }
            QPushButton:focus {
                border-color: rgba(123,82,255,0.60);
                background: rgba(20,30,46,0.95);
                outline: none;
            }
            QPushButton:disabled {
                color: #EEF2F7;
                border-color: rgba(255,255,255,0.05);
                background: rgba(20,30,46,0.55);
            }
        """
        discard_active = """
            QPushButton {
                background: rgba(255,61,110,0.18);
                color: #FF3D6E;
                border: 1px solid rgba(255,61,110,0.45);
                border-radius: 0px;
                font-weight: 700;
                font-size: 8pt;
            }
            QPushButton:hover { background: rgba(255,61,110,0.28); }
            QPushButton:pressed { background: rgba(255,61,110,0.34); }
            QPushButton:focus {
                background: rgba(255,61,110,0.30);
                border-color: rgba(255,61,110,0.68);
                outline: none;
            }
        """
        discard_inactive = """
            QPushButton {
                background: rgba(49,50,68,0.5);
                color: #45475a;
                border: 1px solid rgba(69,71,90,0.4);
                border-radius: 7px;
                font-weight: 600;
                font-size: 8pt;
            }
            QPushButton:hover { background: rgba(69,71,90,0.6); color: #a6adc8; }
        """
        if included:
            self._btn_keep.setStyleSheet(keep_active)
            self._btn_discard.setStyleSheet(discard_inactive)
        else:
            self._btn_keep.setStyleSheet(keep_inactive)
            self._btn_discard.setStyleSheet(discard_active)

    # ── Slots ────────────────────────────────────────────────────────────────

    def _on_keep(self) -> None:
        if self._is_included:
            return
        self._is_included = True
        self._apply_card_style(included=True)
        self._apply_btn_styles(included=True)
        self.decisionChanged.emit(self._node_id, True)

    def _on_discard(self) -> None:
        if not self._is_included:
            return
        self._is_included = False
        self._apply_card_style(included=False)
        self._apply_btn_styles(included=False)
        self.decisionChanged.emit(self._node_id, False)

    # ── Accès public ─────────────────────────────────────────────────────────

    @property
    def node_id(self) -> int:
        return self._node_id

    @property
    def is_included(self) -> bool:
        return self._is_included

    @property
    def node_data(self) -> Dict[str, Any]:
        return self._node


# ═══════════════════════════════════════════════════════════════════════════════
# MRDNodeTable — Grille de validation (alias public conservé pour rétrocompat)
# ═══════════════════════════════════════════════════════════════════════════════


class MRDNodeTable(QWidget):
    """
    Widget composite : en-tête + filtre méthode + grille de MRDNodeCard.

    Signaux :
        curated_ratio_changed(method: str, ratio: float, n_mrd_cells: int)
            Émis après chaque décision experte, pour mettre à jour les jauges.
        manually_added_nodes_changed(nodes: list)
            Émis quand l'utilisateur valide des ajouts manuels via ExpertFocusDialog.

    Interface publique :
        load_nodes(nodes, available_methods, mfi_data, marker_cols,
                   total_viable_cells)
        clear()
        get_human_curated_results() → List[Dict]
    """

    curated_ratio_changed = pyqtSignal(str, float, int)
    manually_added_nodes_changed = pyqtSignal(list)
    expert_focus_curation_applied = pyqtSignal(dict)
    verification_commit_requested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._nodes: List[Dict[str, Any]] = []
        self._cards: List[MRDNodeCard] = []
        # État global des décisions expertes par node_id (indépendant du filtre affiché)
        self._included_by_id: Dict[int, bool] = {}
        self._total_viable_cells: int = 1
        self._current_method_filter: str = ""
        self._mfi_data: Any = None
        self._marker_cols: List[str] = []
        # Tous les nœuds patients (y compris hors-MRD), pour ExpertFocusDialog
        self._all_patient_nodes: List[Dict[str, Any]] = []
        # node_id des nœuds ajoutés manuellement
        self._manually_added_ids: set = set()

        # Expanding/Expanding : le widget réclame tout l'espace disponible
        # dans les deux axes auprès de son layout parent.
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Filet de sécurité absolu : interdit à Qt de comprimer la zone
        # en dessous de 400 px quelle que soit la pression du layout parent.
        self.setMinimumHeight(400)

        self._build_ui()

    # ── Construction ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # ── En-tête ──────────────────────────────────────────────────────
        header_widget = QWidget()
        header_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 rgba(16,24,37,0.95), stop:1 rgba(12,18,32,0.95));
                border-radius: 0px;
                border: 1px solid rgba(255,255,255,0.055);
                border-bottom: none;
            }
        """)
        hdr = QHBoxLayout(header_widget)
        hdr.setContentsMargins(14, 10, 14, 10)
        hdr.setSpacing(10)

        lbl_title = QLabel("VALIDATION NŒUDS MRD — DÉCISION EXPERTE")
        lbl_title.setStyleSheet(
            "color: #7B52FF; font-size: 8.8pt; font-weight: 700; "
            "letter-spacing: 0.14em; background: transparent;"
            "font-family: 'Consolas', 'Cascadia Code', monospace;"
        )
        hdr.addWidget(lbl_title)
        hdr.addStretch()

        # Badge ratio validé
        self._lbl_ratio = QLabel("")
        self._lbl_ratio.setStyleSheet(
            "color: #39FF8A; background: rgba(57,255,138,0.10); "
            "border: 1px solid rgba(57,255,138,0.30); border-radius: 0px; "
            "padding: 2px 10px; font-size: 8.8pt; font-weight: 700;"
            "font-family: 'Consolas', 'Cascadia Code', monospace;"
        )
        self._lbl_ratio.hide()
        hdr.addWidget(self._lbl_ratio)

        lbl_filter = QLabel("Filtre :")
        lbl_filter.setStyleSheet(
            f"color: {_C['subtext']}; font-size: 9pt; background: transparent;"
        )
        hdr.addWidget(lbl_filter)

        self.combo_filter = QComboBox()
        self.combo_filter.setMinimumWidth(140)
        self.combo_filter.setMaximumWidth(180)
        self.combo_filter.currentTextChanged.connect(self._on_filter_changed)
        self.combo_filter.setStyleSheet("""
            QComboBox {
                background: rgba(16,24,37,0.95);
                border: 1px solid rgba(123,82,255,0.35);
                border-radius: 0px;
                color: #EEF2F7;
                padding: 6px 10px;
            }
            QComboBox:hover { border-color: rgba(123,82,255,0.55); }
            QComboBox QAbstractItemView {
                background: #0C1220;
                color: #EEF2F7;
                selection-background-color: rgba(123,82,255,0.20);
            }
        """)
        hdr.addWidget(self.combo_filter)

        # ── Séparateur vertical ───────────────────────────────────────────
        sep_v = QFrame()
        sep_v.setFrameShape(QFrame.VLine)
        sep_v.setStyleSheet("color: rgba(255,255,255,0.055); max-width: 1px;")
        sep_v.setFixedHeight(24)
        hdr.addWidget(sep_v)

        # ── Bouton Expert Focus View ──────────────────────────────────────
        self._btn_expert_focus = QPushButton("⊕  Expert Focus View")
        self._btn_expert_focus.setFixedHeight(30)
        self._btn_expert_focus.setToolTip(
            "Ouvrir la vue détaillée de TOUS les nœuds patients.\n"
            "Permet d'ajouter manuellement des nœuds non retenus par JF / Flo / ELN."
        )
        self._btn_expert_focus.setStyleSheet("""
            QPushButton {
                background: rgba(91,170,255,0.12);
                color: #5BAAFF;
                border: 1px solid rgba(91,170,255,0.34);
                border-radius: 0px;
                font-weight: 700;
                font-size: 8pt;
                padding: 0 12px;
            }
            QPushButton:hover {
                background: rgba(91,170,255,0.22);
                border-color: rgba(91,170,255,0.60);
                color: #B8D9FF;
            }
            QPushButton:pressed {
                background: rgba(91,170,255,0.30);
            }
        """)
        self._btn_expert_focus.clicked.connect(self._on_open_expert_focus)
        hdr.addWidget(self._btn_expert_focus)

        # ── Bouton Validation explicite (écriture HTML/FCS) ─────────────────
        self._btn_commit_verification = QPushButton("✓  Valider la vérification")
        self._btn_commit_verification.setFixedHeight(30)
        self._btn_commit_verification.setToolTip(
            "Valide explicitement la curation experte et demande l'écriture immédiate\n"
            "du rapport HTML et du FCS pathologique Is_MRD."
        )
        self._btn_commit_verification.setStyleSheet("""
            QPushButton {
                background: rgba(57,255,138,0.14);
                color: #39FF8A;
                border: 1px solid rgba(57,255,138,0.34);
                border-radius: 0px;
                font-weight: 800;
                font-size: 8pt;
                padding: 0 12px;
            }
            QPushButton:hover {
                background: rgba(57,255,138,0.24);
                border-color: rgba(57,255,138,0.60);
                color: #D5FFE8;
            }
            QPushButton:pressed {
                background: rgba(57,255,138,0.32);
            }
        """)
        self._btn_commit_verification.clicked.connect(self._on_commit_verification)
        hdr.addWidget(self._btn_commit_verification)

        # Badge ajouts manuels (caché tant que 0)
        self._lbl_manual_badge = QLabel("")
        self._lbl_manual_badge.setStyleSheet(
            "color: #5BAAFF; background: rgba(91,170,255,0.14); "
            "border: 1px solid rgba(91,170,255,0.38); border-radius: 0px; "
            "padding: 2px 8px; font-size: 8pt; font-weight: 700;"
        )
        self._lbl_manual_badge.hide()
        hdr.addWidget(self._lbl_manual_badge)

        root.addWidget(header_widget)

        # ── ScrollArea ───────────────────────────────────────────────────
        self._scroll = QScrollArea()
        # widgetResizable=True est OBLIGATOIRE : sans lui, le widget interne
        # garde sa taille hint initiale (souvent 0×0) et la scrollbar ne se
        # déclenche jamais.
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Expanding/Expanding : la ScrollArea se déploie dans les deux axes
        # pour occuper tout l'espace que son layout parent lui alloue.
        self._scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._scroll.setStyleSheet(f"""
            QScrollArea {{
                background: rgba(12,18,32,0.96);
                border: 1px solid rgba(255,255,255,0.055);
                border-top: none;
                border-radius: 0px;
            }}
            QScrollBar:vertical {{
                background: {_C["surface0"]};
                width: 6px;
                border-radius: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: rgba(123,82,255,0.45);
                border-radius: 0px;
            }}
        """)

        # ── Conteneur interne de la grille ───────────────────────────────
        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet("background: transparent;")
        # Expanding/Preferred : le widget se cale sur la largeur du viewport
        # (Expanding) et prend exactement la hauteur de son contenu (Preferred).
        # Avec setWidgetResizable(True), Qt ne comprimera pas ce widget
        # en dessous de son minimumSizeHint().
        self._grid_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._grid = QGridLayout(self._grid_widget)
        self._grid.setContentsMargins(12, 12, 12, 12)
        self._grid.setSpacing(10)
        # AlignTop sur le layout de la grille : sans cela, Qt répartit l'espace
        # vertical restant entre les lignes existantes, ce qui étire les cartes
        # de façon absurde quand il y en a peu (ex. 2 cartes → chacune prend
        # la moitié de la hauteur de la fenêtre).
        self._grid.setAlignment(Qt.AlignTop)

        self._scroll.setWidget(self._grid_widget)

        # stretch=1 : la ScrollArea absorbe tout l'espace vertical restant dans
        # MRDNodeTable, ce qui laisse l'en-tête et le compteur à leur taille naturelle.
        root.addWidget(self._scroll, 1)

        # ── Label état vide ──────────────────────────────────────────────
        self._lbl_empty = QLabel("Aucun nœud MRD positif détecté pour cette sélection")
        self._lbl_empty.setAlignment(Qt.AlignCenter)
        self._lbl_empty.setStyleSheet(
            f"color: {_C['overlay0']}; font-style: italic; padding: 24px; font-size: 10pt;"
        )
        self._lbl_empty.hide()
        root.addWidget(self._lbl_empty)

        # ── Compteur ─────────────────────────────────────────────────────
        self._lbl_count = QLabel("")
        self._lbl_count.setAlignment(Qt.AlignRight)
        self._lbl_count.setStyleSheet(f"color: {_C['overlay0']}; font-size: 8pt; padding: 2px 4px;")
        root.addWidget(self._lbl_count)

    # ── Interface publique ───────────────────────────────────────────────────

    def load_nodes(
        self,
        nodes: List[Dict[str, Any]],
        available_methods: List[str],
        mfi_data: Any = None,
        marker_cols: Optional[List[str]] = None,
        total_viable_cells: int = 1,
    ) -> None:
        """
        Charge les nœuds MRD et crée une carte par nœud.

        Paramètres :
            nodes               Liste de dicts nœuds (node_id, pct_patho, n_patho…)
            available_methods   Méthodes disponibles pour le filtre
            mfi_data            DataFrame MFI (index = node_id) pour les radars
            marker_cols         Liste de colonnes marqueurs pour le radar
            total_viable_cells  Dénominateur pour le calcul du ratio validé
        """
        self._nodes = nodes
        self._mfi_data = mfi_data
        self._marker_cols = marker_cols or []
        self._total_viable_cells = max(total_viable_cells, 1)
        self._included_by_id = {int(n.get("node_id", 0)): True for n in self._nodes}

        # Filtre
        self.combo_filter.blockSignals(True)
        self.combo_filter.clear()
        self.combo_filter.addItem("Toutes les méthodes")
        for m in available_methods:
            self.combo_filter.addItem(m)
        self.combo_filter.setCurrentIndex(0)
        self.combo_filter.blockSignals(False)

        self._current_method_filter = ""
        self._rebuild_grid()

    def clear(self) -> None:
        self._nodes = []
        self._cards = []
        self._included_by_id = {}
        self._all_patient_nodes = []
        self._manually_added_ids = set()
        self.combo_filter.blockSignals(True)
        self.combo_filter.clear()
        self.combo_filter.blockSignals(False)
        self._current_method_filter = ""
        self._clear_grid()
        self._lbl_ratio.hide()
        self._lbl_count.setText("")
        self._lbl_empty.hide()
        self._lbl_manual_badge.hide()
        self._scroll.show()

    def set_all_patient_nodes(self, all_nodes: List[Dict[str, Any]]) -> None:
        """
        Fournit la liste complète de tous les nœuds patients (y compris hors-MRD)
        pour alimenter ExpertFocusDialog.

        À appeler depuis HomeTab après la fin du pipeline, avant ou après load_nodes().
        Si non appelé, ExpertFocusDialog utilisera les nœuds MRD positifs seulement.
        """
        self._all_patient_nodes = all_nodes

    def get_human_curated_results(self) -> List[Dict[str, Any]]:
        """
        Retourne la liste des nœuds validés par l'expert (GARDER).
        Utilisé par l'export PDF/CSV.
        """
        return [n for n in self._nodes if self._included_by_id.get(int(n.get("node_id", 0)), True)]

    def get_included_node_ids(self) -> set[int]:
        """Retourne les node_id actuellement inclus (GARDER)."""
        return {
            int(n.get("node_id", 0))
            for n in self._nodes
            if self._included_by_id.get(int(n.get("node_id", 0)), True)
        }

    def apply_included_node_ids(self, included_ids: set[int], emit_ratio: bool = True) -> None:
        """
        Applique un état externe KEEP/DISCARD sur les nœuds suivis par la grille.

        Args:
            included_ids: Ensemble des node_id à marquer GARDER.
            emit_ratio: Si True, émet curated_ratio_changed après application.
        """
        if not self._nodes:
            return

        norm_ids = {int(nid) for nid in included_ids}
        for node in self._nodes:
            nid = int(node.get("node_id", 0))
            self._included_by_id[nid] = nid in norm_ids

        self._rebuild_grid()
        self._refresh_ratio_badge()

        if emit_ratio:
            total_mrd_cells = sum(
                int(n.get("n_patho", 0))
                for n in self._nodes
                if self._included_by_id.get(int(n.get("node_id", 0)), True)
            )
            final_ratio = (
                total_mrd_cells / self._total_viable_cells * 100.0
                if self._total_viable_cells > 0
                else 0.0
            )
            self.curated_ratio_changed.emit("Curated", final_ratio, total_mrd_cells)

    # ── Construction de la grille ────────────────────────────────────────────

    def _clear_grid(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards.clear()

    # Largeur minimale d'une carte (px) utilisée pour calculer le nombre de
    # colonnes à partir de la largeur réelle du conteneur.
    _CARD_MIN_W: int = 220

    def _rebuild_grid(self) -> None:
        """Reconstruit toutes les cartes selon le filtre actif."""
        self._clear_grid()

        filtered = self._filter_nodes(self._nodes)

        if not filtered:
            self._scroll.hide()
            self._lbl_empty.show()
            self._lbl_count.setText("")
            self._lbl_ratio.hide()
            return

        self._lbl_empty.hide()
        self._scroll.show()

        # ── Nombre de colonnes calculé depuis la largeur réelle ──────────
        # On interroge la largeur du viewport de la ScrollArea (plus fiable
        # que self.width() qui peut valoir 0 si le widget n'est pas encore
        # affiché). On se rabat sur 3 si aucune mesure n'est disponible.
        vp_w = self._scroll.viewport().width() if self._scroll.viewport() else 0
        if vp_w < self._CARD_MIN_W:
            # Fenêtre pas encore rendue : utilise une valeur par défaut raisonnable
            vp_w = max(self._scroll.width(), 700)
        cols = max(1, min(6, vp_w // self._CARD_MIN_W))
        # Ne pas dépasser le nombre de cartes
        cols = min(cols, len(filtered))

        for idx, node in enumerate(filtered):
            node_id = int(node.get("node_id", 0))
            card = MRDNodeCard(
                node=node,
                mfi_data=self._mfi_data,
                marker_cols=self._marker_cols,
                initial_included=self._included_by_id.get(node_id, True),
                parent=None,
            )
            card.decisionChanged.connect(self._on_decision_changed)
            row, col = divmod(idx, cols)
            # Pas d'AlignTop ici : Qt.AlignTop force la hauteur à sizeHint()
            # et écrase les cartes. Sans flag d'alignement, le QGridLayout
            # distribue l'espace correctement et MinimumExpanding fait le reste.
            self._grid.addWidget(card, row, col)
            self._cards.append(card)

        # Toutes les colonnes s'étirent de façon égale dans la largeur disponible.
        # Les lignes ne reçoivent aucun stretch (valeur par défaut = 0) : leur
        # hauteur est donc dictée uniquement par le contenu des cartes.
        # L'alignement AlignTop sur le layout (défini dans _build_ui) empêche Qt
        # de dilater les lignes pour remplir l'espace vide sous les cartes.
        for c in range(cols):
            self._grid.setColumnStretch(c, 1)

        total = len(self._nodes)
        visible = len(filtered)
        if visible == total:
            self._lbl_count.setText(f"{visible} nœud(s) MRD")
        else:
            self._lbl_count.setText(f"{visible}/{total} nœud(s) affiché(s)")

        self._refresh_ratio_badge()

    def _filter_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._current_method_filter:
            return nodes
        flag_key = _METHOD_FLAG.get(self._current_method_filter, "")
        if not flag_key:
            return nodes
        return [n for n in nodes if n.get(flag_key)]

    # ── Calcul dynamique ─────────────────────────────────────────────────────

    def _on_decision_changed(self, node_id: int, is_included: bool) -> None:
        """
        Recalcule total_mrd_cells et final_ratio après chaque décision.
        Émet curated_ratio_changed pour mettre à jour les jauges.
        """
        self._included_by_id[int(node_id)] = bool(is_included)
        self._refresh_ratio_badge()

        total_mrd_cells = sum(
            int(n.get("n_patho", 0))
            for n in self._nodes
            if self._included_by_id.get(int(n.get("node_id", 0)), True)
        )
        final_ratio = (
            total_mrd_cells / self._total_viable_cells * 100.0
            if self._total_viable_cells > 0
            else 0.0
        )
        # Signal → home_tab intercepte pour mettre à jour les gauges
        self.curated_ratio_changed.emit("Curated", final_ratio, total_mrd_cells)

    def _refresh_ratio_badge(self) -> None:
        """Met à jour le badge de ratio validé dans l'en-tête."""
        curated_nodes = self.get_human_curated_results()
        total_mrd_cells = sum(int(n.get("n_patho", 0)) for n in curated_nodes)
        if self._total_viable_cells > 0 and self._nodes:
            ratio = total_mrd_cells / self._total_viable_cells * 100.0
            n_kept = len(curated_nodes)
            self._lbl_ratio.setText(f"Validé : {ratio:.4f} %  ({n_kept}/{len(self._nodes)} nœuds)")
            # Couleur selon seuil 0.01 %
            if ratio > 0.01:
                self._lbl_ratio.setStyleSheet(
                    "color: #FF3D6E; background: rgba(255,61,110,0.12); "
                    "border: 1px solid rgba(255,61,110,0.40); border-radius: 0px; "
                    "padding: 2px 10px; font-size: 8.8pt; font-weight: 700;"
                    "font-family: 'Consolas', 'Cascadia Code', monospace;"
                )
            else:
                self._lbl_ratio.setStyleSheet(
                    "color: #39FF8A; background: rgba(57,255,138,0.10); "
                    "border: 1px solid rgba(57,255,138,0.32); border-radius: 0px; "
                    "padding: 2px 10px; font-size: 8.8pt; font-weight: 700;"
                    "font-family: 'Consolas', 'Cascadia Code', monospace;"
                )
            self._lbl_ratio.show()
        else:
            self._lbl_ratio.hide()

    # ── Redimensionnement ────────────────────────────────────────────────────

    def resizeEvent(self, event: Any) -> None:  # type: ignore[override]
        """
        Recalcule le nombre de colonnes quand le widget change de largeur.

        Appelé automatiquement par Qt lors de chaque redimensionnement.
        On ne reconstruit que si le nombre de colonnes optimal change,
        pour éviter de détruire/recréer les cartes inutilement.
        """
        super().resizeEvent(event)
        if not self._cards:
            return
        vp_w = self._scroll.viewport().width() if self._scroll.viewport() else 0
        if vp_w < self._CARD_MIN_W:
            vp_w = max(self.width(), 700)
        new_cols = max(1, min(6, vp_w // self._CARD_MIN_W))
        new_cols = min(new_cols, len(self._cards))
        # Nombre de colonnes actuel = nombre de colonnes non-vides dans la grille
        current_cols = self._grid.columnCount()
        if new_cols != current_cols:
            self._rebuild_grid()

    # ── Slots internes ───────────────────────────────────────────────────────

    def _on_filter_changed(self, text: str) -> None:
        if text in ("", "Toutes les méthodes"):
            self._current_method_filter = ""
        else:
            self._current_method_filter = text.split()[0]
        self._rebuild_grid()

    def _on_open_expert_focus(self) -> None:
        """Ouvre ExpertFocusDialog en passant tous les nœuds patients + données radar."""
        from gui.dialogs.expert_focus_dialog import ExpertFocusDialog

        # Fallback : si set_all_patient_nodes() n'a pas été appelé, on utilise
        # les nœuds MRD déjà chargés (fonctionnement dégradé mais non bloquant).
        source_nodes = self._all_patient_nodes if self._all_patient_nodes else self._nodes
        if not source_nodes:
            return

        dialog = ExpertFocusDialog(
            all_nodes=source_nodes,
            already_added=set(self._manually_added_ids),
            mfi_data=self._mfi_data,
            marker_cols=self._marker_cols,
            parent=self,
        )
        dialog.curation_applied.connect(self._on_expert_focus_curation_applied)
        dialog.nodes_manually_added.connect(self._on_manual_nodes_added)
        dialog.exec_()

    def _on_commit_verification(self) -> None:
        """Demande explicite de validation finale (HTML/FCS) depuis l'UI."""
        self.verification_commit_requested.emit(self.combo_filter.currentText())

    def _on_expert_focus_curation_applied(self, payload: Dict[str, Any]) -> None:
        """Synchronise l'état KEEP/DISCARD global depuis la popup Expert Focus."""
        included_ids = {
            int(nid)
            for nid in payload.get("included_ids", [])
            if isinstance(nid, (int, float, str)) and str(nid).strip() != ""
        }
        if not self._nodes:
            return

        # Appliquer la décision sur tous les nœuds actuellement suivis en MRD table.
        for node in self._nodes:
            nid = int(node.get("node_id", 0))
            self._included_by_id[nid] = nid in included_ids

        self._refresh_ratio_badge()
        total_mrd_cells = sum(
            int(n.get("n_patho", 0))
            for n in self._nodes
            if self._included_by_id.get(int(n.get("node_id", 0)), True)
        )
        final_ratio = (
            total_mrd_cells / self._total_viable_cells * 100.0
            if self._total_viable_cells > 0
            else 0.0
        )
        self.curated_ratio_changed.emit("Curated", final_ratio, total_mrd_cells)
        self.expert_focus_curation_applied.emit(
            {
                "included_ids": sorted(included_ids),
                "manual_ids": [
                    int(nid)
                    for nid in payload.get("manual_ids", [])
                    if isinstance(nid, (int, float, str)) and str(nid).strip() != ""
                ],
            }
        )
        self._rebuild_grid()

    def _on_manual_nodes_added(self, new_nodes: List[Dict[str, Any]]) -> None:
        """
        Reçoit les nœuds validés manuellement depuis ExpertFocusDialog.

        Stratégie :
          1. Marque ces nœuds avec is_mrd_manual=True pour les distinguer.
          2. Fusionne avec self._nodes (évite les doublons par node_id).
          3. Reconstruit la grille.
          4. Émet manually_added_nodes_changed pour que HomeTab puisse
             persister ou afficher l'information.
        """
        selected_manual_ids = {int(n.get("node_id", 0)) for n in new_nodes}
        source_nodes = self._all_patient_nodes if self._all_patient_nodes else self._nodes
        source_by_id = {int(n.get("node_id", 0)): n for n in source_nodes}

        # Supprimer les anciens nœuds manuels non retenus (désélection explicite)
        previous_manual_ids = set(self._manually_added_ids)
        manual_removed_ids = previous_manual_ids - selected_manual_ids
        if manual_removed_ids:
            self._nodes = [
                n
                for n in self._nodes
                if not (
                    int(n.get("node_id", 0)) in manual_removed_ids
                    and n.get("is_mrd_manual", False)
                    and not (n.get("is_mrd_jf") or n.get("is_mrd_flo") or n.get("is_mrd_eln"))
                )
            ]
            for rid in manual_removed_ids:
                self._included_by_id.pop(rid, None)

        existing_ids = {int(n.get("node_id", 0)) for n in self._nodes}

        for nid in selected_manual_ids:
            if nid in existing_ids:
                # Nœud déjà présent → marquer manuel + forcer état inclus
                for n in self._nodes:
                    if int(n.get("node_id", 0)) == nid:
                        n["is_mrd_manual"] = True
                        break
            else:
                base = dict(source_by_id.get(nid, {"node_id": nid}))
                base["is_mrd_manual"] = True
                self._nodes.append(base)
                existing_ids.add(nid)
            self._included_by_id[nid] = True

        self._manually_added_ids = selected_manual_ids

        # Mise à jour du badge
        total_manual = len(self._manually_added_ids)
        if total_manual > 0:
            self._lbl_manual_badge.setText(f"+{total_manual} manuel(s)")
            self._lbl_manual_badge.show()
        else:
            self._lbl_manual_badge.hide()

        self._rebuild_grid()
        self._refresh_ratio_badge()

        total_mrd_cells = sum(
            int(n.get("n_patho", 0))
            for n in self._nodes
            if self._included_by_id.get(int(n.get("node_id", 0)), True)
        )
        final_ratio = (
            total_mrd_cells / self._total_viable_cells * 100.0
            if self._total_viable_cells > 0
            else 0.0
        )
        self.curated_ratio_changed.emit("Curated", final_ratio, total_mrd_cells)

        self.manually_added_nodes_changed.emit([n for n in self._nodes if n.get("is_mrd_manual")])


# ═══════════════════════════════════════════════════════════════════════════════
# Rétrocompat : modèle et proxy conservés (non utilisés par la grille mais
# potentiellement importés ailleurs dans le projet)
# ═══════════════════════════════════════════════════════════════════════════════


class MRDNodeTableModel(QAbstractTableModel):
    """Modèle de données (conservé pour rétrocompat, non utilisé par MRDNodeTable)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._nodes: List[Dict[str, Any]] = []
        self._font_bold = QFont("Segoe UI", 9, QFont.Bold)
        self._font_mono = QFont("Cascadia Code", 9)

    def load(self, nodes: List[Dict[str, Any]]) -> None:
        self.beginResetModel()
        self._nodes = nodes
        self.endResetModel()

    def node_at(self, row: int) -> Optional[Dict[str, Any]]:
        if 0 <= row < len(self._nodes):
            return self._nodes[row]
        return None

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._nodes)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(_HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return _HEADERS[section]
        return QVariant()

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid() or index.row() >= len(self._nodes):
            return QVariant()
        return QVariant()


class MRDMethodFilterProxy(QSortFilterProxyModel):
    """Proxy filtre méthode (conservé pour rétrocompat)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._method: str = ""

    def set_method(self, method: str) -> None:
        self._method = method
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        if not self._method:
            return True
        model = self.sourceModel()
        if not isinstance(model, MRDNodeTableModel):
            return True
        node = model.node_at(source_row)
        if node is None:
            return False
        flag_key = _METHOD_FLAG.get(self._method, "")
        return bool(node.get(flag_key, False)) if flag_key else True
