# -*- coding: utf-8 -*-
"""
expert_focus_dialog.py — Vue "Expert Focus View" plein écran pour la validation experte.

Ouvre une QDialog plein écran depuis la zone "Validation nœuds MRD" de HomeTab.

Contenu :
  - Grille de cartes identiques à MRDNodeCard pour TOUS les nœuds patients.
  - Chaque carte affiche le radar chart + badges méthodes (JF/Flo/ELN) + boutons GARDER/ÉCARTER.
  - Filtre par méthode + recherche + filtre "Non sélectionnés algo".
  - Signal nodes_manually_added(List[Dict]) émis à la validation.

Architecture : QDialog (plein écran) → QVBoxLayout
  ├── En-tête (titre + légende méthodes + stats)
  ├── Barre d'outils (filtre méthode + recherche)
  ├── QScrollArea → QGridLayout de ExpertNodeCard
  └── Pied de page (ratio calculé + [Annuler] [Valider])
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set

from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QPushButton,
    QScrollArea,
    QGridLayout,
    QWidget,
    QSizePolicy,
    QButtonGroup,
    QComboBox,
    QLineEdit,
)

# ── Palette PRISMA v2 ─────────────────────────────────────────────────
_C = {
    "base": "#0C1220",
    "mantle": "#080D18",
    "crust": "#04070D",
    "surface0": "#101825",
    "surface1": "#141E2E",
    "surface2": "#2A3342",
    "overlay0": "rgba(238,242,247,0.35)",
    "text": "#EEF2F7",
    "subtext": "rgba(238,242,247,0.55)",
    "blue": "#5BAAFF",
    "lavender": "#7B52FF",
    "mauve": "#7B52FF",
    "green": "#39FF8A",
    "red": "#FF3D6E",
    "yellow": "#FFE032",
    "teal": "#5BAAFF",
    "peach": "#FF9B3D",
    "pink": "#FF3D6E",
}

_METHOD_FLAG: Dict[str, str] = {
    "JF": "is_mrd_jf",
    "Flo": "is_mrd_flo",
    "ELN": "is_mrd_eln",
}

_METHOD_COLOR: Dict[str, str] = {
    "JF": _C["blue"],
    "Flo": _C["teal"],
    "ELN": _C["peach"],
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


def _outfit(size: int, weight: int = QFont.Normal) -> QFont:
    """Police UI prioritaire: Outfit, avec fallback système géré par Qt."""
    return QFont("Segoe UI", size, weight)


# ═══════════════════════════════════════════════════════════════════════════════
# ExpertNodeCard — Carte plein format pour la vue Expert Focus
# ═══════════════════════════════════════════════════════════════════════════════


class ExpertNodeCard(QFrame):
    """
    Carte identique à MRDNodeCard mais avec :
      - Badges méthodes colorés (JF / Flo / ELN) indiquant quelles algos ont retenu le nœud.
      - Badge "MANUEL" si déjà ajouté manuellement.
      - Boutons GARDER / ÉCARTER (même logique que MRDNodeCard).
      - Carte grisée si aucune méthode ne l'a retenu et pas ajouté manuellement.

    Signaux :
        decisionChanged(node_id: int, is_included: bool)
    """

    decisionChanged = pyqtSignal(int, bool)

    def __init__(
        self,
        node: Dict[str, Any],
        mfi_data: Any = None,
        marker_cols: Optional[List[str]] = None,
        initial_included: bool = False,
        radar_cache: Optional[Dict] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._node = node
        self._node_id: int = int(node.get("node_id", 0))
        self._mfi_data = mfi_data
        self._marker_cols: List[str] = marker_cols or []
        self._radar_cache = radar_cache  # dict partagé {node_id: pixmap_data}

        self._is_jf = bool(node.get("is_mrd_jf", False))
        self._is_flo = bool(node.get("is_mrd_flo", False))
        self._is_eln = bool(node.get("is_mrd_eln", False))
        self._is_algo = self._is_jf or self._is_flo or self._is_eln
        self._is_manual = bool(node.get("is_mrd_manual", False))

        # État initial : GARDER si algo ou déjà ajouté manuellement, ÉCARTER sinon
        self._is_included: bool = initial_included or self._is_algo or self._is_manual

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setMinimumSize(200, 310)
        self._build_ui()

    # ── Construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setObjectName("expertNodeCard")
        self._apply_card_style()

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(5)

        # ── En-tête : ID + % patho ────────────────────────────────────────
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)

        node_id_lbl = QLabel(f"Nœud  {self._node_id}")
        node_id_lbl.setFont(_outfit(10, QFont.Bold))
        node_id_lbl.setStyleSheet(f"color: {_C['mauve']}; background: transparent;")
        header_row.addWidget(node_id_lbl)
        header_row.addStretch()

        pct_patho = self._node.get("pct_patho", 0.0)
        pct_lbl = QLabel(f"{pct_patho:.1f} %")
        pct_lbl.setFont(_outfit(10, QFont.Bold))
        pct_col = (
            _C["red"] if pct_patho > 1.0 else (_C["yellow"] if pct_patho > 0.01 else _C["subtext"])
        )
        pct_lbl.setStyleSheet(f"color: {pct_col}; background: transparent;")
        header_row.addWidget(pct_lbl)
        root.addLayout(header_row)

        # ── Badges méthodes ───────────────────────────────────────────────
        badges_row = QHBoxLayout()
        badges_row.setContentsMargins(0, 0, 0, 0)
        badges_row.setSpacing(4)

        for method, flag in _METHOD_FLAG.items():
            active = bool(self._node.get(flag, False))
            color = _METHOD_COLOR[method]
            if active:
                badge = QLabel(f"✓ {method}")
                badge.setStyleSheet(
                    f"color: {color}; background: rgba(0,0,0,0.25); "
                    f"border: 1px solid {color}; border-radius: 4px; "
                    f"padding: 1px 5px; font-size: 7pt; font-weight: 700;"
                )
            else:
                badge = QLabel(f"— {method}")
                badge.setStyleSheet(
                    f"color: {_C['surface2']}; background: transparent; "
                    f"border: 1px solid {_C['surface1']}; border-radius: 4px; "
                    f"padding: 1px 5px; font-size: 7pt;"
                )
            badges_row.addWidget(badge)

        # Badge MANUEL
        if self._is_manual:
            m_badge = QLabel("● MANUEL")
            m_badge.setStyleSheet(
                f"color: {_C['teal']}; background: rgba(148,226,213,0.14); "
                f"border: 1px solid rgba(148,226,213,0.45); border-radius: 4px; "
                f"padding: 1px 5px; font-size: 7pt; font-weight: 700;"
            )
            badges_row.addWidget(m_badge)

        badges_row.addStretch()
        root.addLayout(badges_row)

        # Séparateur
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: rgba(137,180,250,0.10); max-height:1px;")
        root.addWidget(sep)

        # ── Zone radar (placeholder léger, rendu différé) ────────────────
        self._radar_widget = self._build_radar_placeholder()
        self._radar_loaded = False
        root.addWidget(self._radar_widget, 1)

        # ── Infos cellules ───────────────────────────────────────────────
        n_patho = self._node.get("n_patho", 0)
        n_cells = self._node.get("n_cells", 0)
        info_lbl = QLabel(f"{n_patho:,} / {n_cells:,} cellules")
        info_lbl.setFont(_outfit(8, QFont.Light))
        info_lbl.setAlignment(Qt.AlignCenter)
        info_lbl.setStyleSheet(f"color: {_C['subtext']}; background: transparent;")
        root.addWidget(info_lbl)

        # ── Boutons GARDER / ÉCARTER ──────────────────────────────────────
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

        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        self._btn_group.addButton(self._btn_keep, 0)
        self._btn_group.addButton(self._btn_discard, 1)

        self._apply_btn_styles()

        self._btn_keep.clicked.connect(self._on_keep)
        self._btn_discard.clicked.connect(self._on_discard)

        btn_row.addWidget(self._btn_keep)
        btn_row.addWidget(self._btn_discard)
        root.addLayout(btn_row)

    def load_radar_deferred(self) -> None:
        """Remplace le placeholder par le vrai radar matplotlib (appelé en différé)."""
        if self._radar_loaded:
            return
        self._radar_loaded = True
        try:
            if self._mfi_data is None:
                return
            # On crée toujours un nouveau widget : les FigureCanvas Qt ne peuvent pas
            # être partagés entre layouts (parent unique), et deleteLater() peut laisser
            # un parent() == None sur un widget déjà détruit.
            new_radar = self._build_radar_matplotlib()
        except Exception as _e:
            import traceback as _tb

            print(f"[ExpertFocusView] Radar erreur nœud {self._node_id}: {_e}\n{_tb.format_exc()}")
            return
        layout = self.layout()
        if layout is None:
            return
        # Trouver et remplacer le placeholder
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item and item.widget() is self._radar_widget:
                old = layout.takeAt(i)
                if old and old.widget():
                    old.widget().deleteLater()
                layout.insertWidget(i, new_radar, 1)
                self._radar_widget = new_radar
                break

    def _build_radar(self) -> QWidget:
        try:
            if self._mfi_data is not None:
                return self._build_radar_matplotlib()
        except Exception:
            pass
        return self._build_radar_placeholder()

    def _resolve_mfi_index_key(self) -> Optional[Any]:
        """
        Résout une clé d'index DataFrame pour ce nœud, même si le format
        diffère (int/str, offset 0-based/1-based, labels type C12/MC12/Node 12).
        """
        if self._mfi_data is None or not hasattr(self._mfi_data, "index"):
            return None

        idx = self._mfi_data.index

        # 1) Correspondance directe (plus fiable)
        if self._node_id in idx:
            return self._node_id

        # 2) Variantes courantes int/str + offset +/- 1
        candidates: List[Any] = [
            str(self._node_id),
            self._node_id - 1,
            self._node_id + 1,
            str(self._node_id - 1),
            str(self._node_id + 1),
        ]
        for cand in candidates:
            if cand in idx:
                return cand

        # 3) Labels composites: C12, MC12, Node 12, etc.
        for key in idx:
            text = str(key).strip()
            found = re.findall(r"\d+", text)
            if not found:
                continue
            val = int(found[-1])
            if val == self._node_id or val == self._node_id - 1 or val == self._node_id + 1:
                return key

        return None

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
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        import numpy as np

        # Marqueurs réellement présents dans la matrice MFI
        mfi_cols = [str(c) for c in list(getattr(self._mfi_data, "columns", []))]
        markers = [m for m in self._filter_clinical_markers(self._marker_cols) if m in mfi_cols]

        # Fallback : si marker_cols est incomplet, on reprend les colonnes MFI cliniques.
        if len(markers) < 3:
            markers = self._filter_clinical_markers(mfi_cols)

        if len(markers) < 3:
            raise ValueError("Pas assez de marqueurs cliniques pour le radar")

        idx_key = self._resolve_mfi_index_key()
        if idx_key is None:
            # Nœud SOM vide (aucune cellule dans ce nœud) — radar plat à zéro
            mfi_row = np.zeros(len(markers), dtype=float)
            _empty_node = True
        else:
            row = self._mfi_data.loc[idx_key, markers]
            if hasattr(row, "ndim") and row.ndim > 1:
                row = row.iloc[0]
            mfi_row = np.asarray(row.values, dtype=float)
            _empty_node = False

        if mfi_row.size < 3:
            raise ValueError("Profil MFI insuffisant pour radar")

        # Normalisation par noeud (identique aux bons radars HomeTab/HTML)
        v_min, v_max = float(mfi_row.min()), float(mfi_row.max())
        norm_values = (mfi_row - v_min) / (v_max - v_min + 1e-10)

        short_labels = []
        for marker in markers:
            parts = marker.strip().split()
            candidate = parts[-1] if len(parts) > 1 else parts[0]
            short_labels.append(candidate[:10])

        N = len(markers)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]
        norm_values = list(norm_values) + [norm_values[0]]

        # Couleur PRISMA : gris si nœud vide, violet si algo, vert si manuel, dim sinon
        if _empty_node:
            radar_color = (0.45, 0.45, 0.55, 0.40)  # gris dim — nœud sans cellules
        elif self._is_algo:
            radar_color = "#7B52FF"  # brand — V450
        elif self._is_manual:
            radar_color = "#39FF8A"  # accent — FITC
        else:
            # Matplotlib n'accepte pas les chaînes CSS "rgba(...)".
            radar_color = (123 / 255.0, 82 / 255.0, 1.0, 0.30)  # brand dim

        fig = Figure(figsize=(2.1, 2.1), facecolor="#080D18")
        ax = fig.add_subplot(111, polar=True)
        ax.set_facecolor("#0C1220")

        line_alpha = 0.45 if _empty_node else 1.0
        fill_alpha = 0.12 if _empty_node else 0.30
        ax.plot(angles, norm_values, color=radar_color, linewidth=1.8,
                alpha=line_alpha, marker="o", markersize=2.5,
                markerfacecolor=radar_color, markeredgewidth=0)
        ax.fill(angles, norm_values, color=radar_color, alpha=fill_alpha)

        if _empty_node:
            ax.text(
                0,
                0,
                "vide",
                ha="center",
                va="center",
                fontsize=7,
                color=(0.6, 0.6, 0.7, 0.7),
                transform=ax.transData,
            )

        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.33, 0.66, 1.0])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            short_labels,
            fontsize=5,
            color="#EEF2F7",
            fontfamily=["Segoe UI", "Arial", "Arial", "sans-serif"],
        )
        ax.set_yticklabels([])
        # Axes/spokes plus lisibles en fond sombre
        ax.yaxis.grid(True, color=(1, 1, 1, 0.20), linewidth=0.7, linestyle=":")
        ax.xaxis.grid(True, color=(1, 1, 1, 0.18), linewidth=0.6, linestyle="-")
        ax.spines["polar"].set_color((1, 1, 1, 0.40))
        ax.spines["polar"].set_linewidth(0.9)
        ax.tick_params(axis="x", pad=3, colors="#EEF2F7")
        fig.tight_layout(pad=0.3)

        canvas = FigureCanvas(fig)
        canvas.setMinimumSize(QSize(180, 180))
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return canvas

    def _build_radar_placeholder(self) -> QWidget:
        ph = QLabel("Radar\nn/d")
        ph.setAlignment(Qt.AlignCenter)
        ph.setMinimumHeight(120)
        ph.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        ph.setStyleSheet(
            f"color: {_C['overlay0']}; background: rgba(49,50,68,0.4); "
            f"border-radius: 8px; font-size: 9pt; font-style: italic;"
        )
        return ph

    # ── Styles ───────────────────────────────────────────────────────────────

    def _apply_card_style(self) -> None:
        if self._is_included:
            if self._is_algo:
                border = "rgba(166, 227, 161, 0.55)"
                bg_top = "rgba(22, 42, 28, 0.90)"
                bg_bot = "rgba(16, 32, 20, 0.90)"
                top_acc = "rgba(166, 227, 161, 0.75)"
            else:
                # Manuel sélectionné : accent teal
                border = "rgba(148, 226, 213, 0.55)"
                bg_top = "rgba(18, 38, 36, 0.90)"
                bg_bot = "rgba(12, 28, 28, 0.90)"
                top_acc = "rgba(148, 226, 213, 0.75)"
        else:
            border = "rgba(69, 71, 90, 0.55)"
            bg_top = "rgba(26, 27, 40, 0.85)"
            bg_bot = "rgba(20, 21, 32, 0.85)"
            top_acc = "rgba(99, 101, 126, 0.5)"
        self.setStyleSheet(f"""
            QFrame#expertNodeCard {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {bg_top}, stop:1 {bg_bot});
                border: 1px solid {border};
                border-top: 2px solid {top_acc};
                border-radius: 10px;
            }}
        """)

    def _apply_btn_styles(self) -> None:
        keep_active = f"""
            QPushButton {{
                background: rgba(166, 227, 161, 0.22);
                color: {_C["green"]};
                border: 1px solid rgba(166,227,161,0.55);
                border-radius: 7px; font-weight: 700; font-size: 8pt;
            }}
            QPushButton:hover {{ background: rgba(166,227,161,0.32); }}
            QPushButton:pressed {{ background: rgba(166,227,161,0.40); }}
            QPushButton:focus {{
                border-color: rgba(166,227,161,0.78);
                background: rgba(166,227,161,0.34);
                outline: none;
            }}
        """
        keep_inactive = f"""
            QPushButton {{
                background: rgba(49,50,68,0.5); color: {_C["surface1"]};
                border: 1px solid rgba(69,71,90,0.4);
                border-radius: 7px; font-weight: 600; font-size: 8pt;
            }}
            QPushButton:hover {{ background: rgba(69,71,90,0.6); color: {_C["subtext"]}; }}
            QPushButton:pressed {{ background: rgba(69,71,90,0.72); }}
            QPushButton:focus {{
                border-color: rgba(123,82,255,0.65);
                background: rgba(69,71,90,0.66);
                outline: none;
            }}
            QPushButton:disabled {{
                color: #EEF2F7;
                border-color: rgba(255,255,255,0.05);
                background: rgba(49,50,68,0.35);
            }}
        """
        discard_active = f"""
            QPushButton {{
                background: rgba(243,139,168,0.20);
                color: {_C["red"]};
                border: 1px solid rgba(243,139,168,0.50);
                border-radius: 7px; font-weight: 700; font-size: 8pt;
            }}
            QPushButton:hover {{ background: rgba(243,139,168,0.30); }}
            QPushButton:pressed {{ background: rgba(243,139,168,0.38); }}
            QPushButton:focus {{
                border-color: rgba(243,139,168,0.72);
                background: rgba(243,139,168,0.34);
                outline: none;
            }}
        """
        discard_inactive = f"""
            QPushButton {{
                background: rgba(49,50,68,0.5); color: {_C["surface1"]};
                border: 1px solid rgba(69,71,90,0.4);
                border-radius: 7px; font-weight: 600; font-size: 8pt;
            }}
            QPushButton:hover {{ background: rgba(69,71,90,0.6); color: {_C["subtext"]}; }}
        """
        if self._is_included:
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
        self._apply_card_style()
        self._apply_btn_styles()
        self.decisionChanged.emit(self._node_id, True)

    def _on_discard(self) -> None:
        if not self._is_included:
            return
        self._is_included = False
        self._apply_card_style()
        self._apply_btn_styles()
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

    @property
    def is_algo_selected(self) -> bool:
        return self._is_algo


# ═══════════════════════════════════════════════════════════════════════════════
# ExpertFocusDialog — Dialog plein écran
# ═══════════════════════════════════════════════════════════════════════════════


class ExpertFocusDialog(QDialog):
    """
    Dialog plein écran "Expert Focus View".

    Affiche TOUS les nœuds patients sous forme de grille de cartes (comme MRDNodeCard),
    avec badges méthodes et radar charts. Permet d'ajouter manuellement des nœuds
    non retenus par JF / Flo / ELN.

    Paramètres :
        all_nodes       Liste complète des nœuds patients.
        already_added   Set des node_id déjà ajoutés manuellement.
        mfi_data        DataFrame MFI pour les radars (optionnel).
        marker_cols     Colonnes marqueurs pour les radars (optionnel).
        parent          Widget parent.

    Signal :
        nodes_manually_added(List[Dict])
            Émis à la validation. Contient les nœuds dont is_included=True
            et qui ne sont PAS dans la sélection algorithmique (nouveaux ajouts).
    """

    nodes_manually_added = pyqtSignal(list)
    # Payload: {"included_ids": List[int], "manual_ids": List[int]}
    curation_applied = pyqtSignal(dict)

    # Largeur minimale d'une carte pour le calcul des colonnes
    _CARD_MIN_W: int = 220

    def __init__(
        self,
        all_nodes: List[Dict[str, Any]],
        already_added: Optional[Set[int]] = None,
        mfi_data: Any = None,
        marker_cols: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._all_nodes = all_nodes
        self._already_added: Set[int] = already_added or set()
        self._mfi_data = mfi_data
        self._marker_cols: List[str] = marker_cols or []

        # Plein écran adapté à l'écran
        from gui.screen_utils import fit_dialog_to_screen
        fit_dialog_to_screen(self, ratio=0.92, min_w=1200, min_h=800)

        # Pré-calculer le statut initial de chaque nœud
        # included = algo OU déjà ajouté manuellement
        self._initial_included: Dict[int, bool] = {}
        for node in all_nodes:
            nid = int(node.get("node_id", 0))
            is_algo = bool(
                node.get("is_mrd_jf") or node.get("is_mrd_flo") or node.get("is_mrd_eln")
            )
            self._initial_included[nid] = is_algo or (nid in self._already_added)

        self._active_filter_idx: int = 0  # index du combo filtre
        self._search_text: str = ""
        self._cards: List[ExpertNodeCard] = []
        # Cache des widgets radar déjà rendus : {node_id: QWidget}
        self._radar_cache: Dict[int, Any] = {}

        self._build_ui()
        self._populate_grid()
        self._refresh_stats()

    # ──────────────────────────────────────────────────────────────────────────
    # Construction UI
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setWindowTitle("Expert Focus View — Validation experte de tous les nœuds MRD")
        self.setModal(True)

        # Plein écran dès l'ouverture
        self.setWindowState(Qt.WindowMaximized)
        self.setMinimumSize(900, 600)

        self.setStyleSheet(f"""
            QDialog {{
                background: {_C["base"]};
                color: {_C["text"]};
            }}
            QWidget {{ background: transparent; color: {_C["text"]}; }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        root.addWidget(self._build_header())

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: rgba(137,180,250,0.14); max-height:1px;")
        root.addWidget(sep)

        root.addWidget(self._build_toolbar())

        # ScrollArea principale
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._scroll.setStyleSheet(f"""
            QScrollArea {{
                background: rgba(24,24,37,0.96);
                border: 1px solid rgba(137,180,250,0.14);
                border-radius: 8px;
            }}
            QScrollBar:vertical {{
                background: {_C["surface0"]};
                width: 6px; border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: {_C["overlay0"]}; border-radius: 3px;
            }}
        """)

        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet(f"background: {_C['base']};")
        self._grid_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._grid = QGridLayout(self._grid_widget)
        self._grid.setContentsMargins(14, 14, 14, 14)
        self._grid.setSpacing(12)
        self._grid.setAlignment(Qt.AlignTop)

        self._scroll.setWidget(self._grid_widget)
        root.addWidget(self._scroll, 1)

        # Label état vide
        self._lbl_empty = QLabel("Aucun nœud à afficher pour ce filtre.")
        self._lbl_empty.setAlignment(Qt.AlignCenter)
        self._lbl_empty.setStyleSheet(
            f"color: {_C['overlay0']}; font-style: italic; font-size: 11pt; padding: 32px;"
        )
        self._lbl_empty.hide()
        root.addWidget(self._lbl_empty)

        root.addWidget(self._build_footer())

    def _build_header(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("efvHeader")
        widget.setStyleSheet(f"""
            QWidget#efvHeader {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 rgba(36,38,60,0.90), stop:1 rgba(22,24,40,0.85));
                border-radius: 10px;
                border: 1px solid rgba(137,180,250,0.15);
            }}
        """)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(14)

        # Titre
        title_col = QVBoxLayout()
        title_col.setSpacing(4)

        badge = QLabel("EXPERT FOCUS VIEW")
        badge.setStyleSheet(
            f"background: rgba(203,166,247,0.18); color: {_C['mauve']}; "
            f"border: 1px solid rgba(203,166,247,0.35); border-radius: 6px; "
            f"padding: 3px 10px; font-size: 8pt; font-weight: 700; "
            f"font-family: 'Segoe UI', 'Segoe UI', sans-serif; letter-spacing: 0.1em;"
        )
        title_col.addWidget(badge)

        title = QLabel("Validation Experte — Tous les Nœuds Patients")
        title.setFont(_outfit(15, QFont.Bold))
        title.setStyleSheet(f"color: {_C['text']};")
        title_col.addWidget(title)

        sub = QLabel(
            "Tous les nœuds SOM sont affichés. Les nœuds retenus par un algorithme ont leurs boutons pré-réglés "
            "sur GARDER. Réglez les nœuds non sélectionnés que vous souhaitez ajouter, puis validez."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet(
            f"color: {_C['subtext']}; font-size: 9pt; font-weight: 300; "
            f"font-family: 'Segoe UI', 'Segoe UI', sans-serif;"
        )
        title_col.addWidget(sub)

        layout.addLayout(title_col, 1)

        # Légende méthodes
        legend = QVBoxLayout()
        legend.setSpacing(5)
        legend.setAlignment(Qt.AlignTop | Qt.AlignRight)

        legend_title = QLabel("Légende méthodes")
        legend_title.setStyleSheet(
            f"color: {_C['subtext']}; font-size: 8pt; font-weight: 600; "
            f"font-family: 'Segoe UI', 'Segoe UI', sans-serif;"
        )
        legend.addWidget(legend_title, alignment=Qt.AlignRight)

        for method, color, desc in [
            ("JF", _C["blue"], "Jean Feuillard"),
            ("Flo", _C["teal"], "Florian"),
            ("ELN", _C["peach"], "EuropeanLeukemiaNet"),
        ]:
            row_w = QWidget()
            rl = QHBoxLayout(row_w)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(5)
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 10pt;")
            rl.addWidget(dot)
            lbl = QLabel(f"<b>{method}</b>  {desc}")
            lbl.setStyleSheet(
                f"color: {_C['subtext']}; font-size: 8pt; "
                f"font-family: 'Segoe UI', 'Segoe UI', sans-serif;"
            )
            rl.addWidget(lbl)
            legend.addWidget(row_w)

        layout.addLayout(legend)
        return widget

    def _build_toolbar(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Filtre méthode
        filter_lbl = QLabel("Afficher :")
        filter_lbl.setStyleSheet(
            f"color: {_C['subtext']}; font-size: 9pt; font-weight: 400; "
            f"font-family: 'Segoe UI', 'Segoe UI', sans-serif;"
        )
        layout.addWidget(filter_lbl)

        self._filter_combo = QComboBox()
        self._filter_combo.setFixedHeight(34)
        self._filter_combo.setMinimumWidth(200)
        for label in [
            "Tous les nœuds",
            "Sélectionnés par JF",
            "Sélectionnés par Flo",
            "Sélectionnés par ELN",
            "Non sélectionnés (aucun algo)",
            "Ajoutés manuellement",
        ]:
            self._filter_combo.addItem(label)
        self._filter_combo.setStyleSheet(f"""
            QComboBox {{
                background: rgba(49,50,68,0.95);
                border: 1px solid rgba(203,166,247,0.30);
                border-radius: 8px; color: {_C["text"]};
                padding: 0 10px; font-size: 9pt;
                font-family: 'Segoe UI', 'Segoe UI', sans-serif;
            }}
            QComboBox:hover {{ border-color: rgba(203,166,247,0.55); }}
            QComboBox QAbstractItemView {{
                background: {_C["surface0"]}; color: {_C["text"]};
                selection-background-color: rgba(203,166,247,0.20);
                border: 1px solid rgba(203,166,247,0.20);
            }}
        """)
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        layout.addWidget(self._filter_combo)

        layout.addSpacing(8)

        # Recherche par ID
        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("  Rechercher par n° nœud…")
        self._search_box.setFixedHeight(34)
        self._search_box.setMinimumWidth(180)
        self._search_box.setMaximumWidth(240)
        self._search_box.setStyleSheet(f"""
            QLineEdit {{
                background: rgba(49,50,68,0.9);
                border: 1px solid rgba(137,180,250,0.25);
                border-radius: 8px; color: {_C["text"]};
                padding: 0 10px; font-size: 9pt;
                font-family: 'Segoe UI', 'Segoe UI', sans-serif;
            }}
            QLineEdit:focus {{ border-color: rgba(137,180,250,0.60); }}
        """)
        self._search_box.textChanged.connect(self._on_search_changed)
        layout.addWidget(self._search_box)

        layout.addStretch()

        # Compteur
        self._lbl_count = QLabel("")
        self._lbl_count.setStyleSheet(
            f"color: {_C['overlay0']}; font-size: 9pt; "
            f"font-family: 'Segoe UI', 'Segoe UI', sans-serif;"
        )
        layout.addWidget(self._lbl_count)

        return widget

    def _build_footer(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("efvFooter")
        widget.setStyleSheet(f"""
            QWidget#efvFooter {{
                background: rgba(24,24,37,0.85);
                border-radius: 8px;
                border: 1px solid rgba(137,180,250,0.12);
            }}
        """)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(14)

        self._lbl_stats = QLabel("")
        self._lbl_stats.setStyleSheet(
            f"color: {_C['subtext']}; font-size: 9pt; font-weight: 300; "
            f"font-family: 'Segoe UI', 'Segoe UI', sans-serif;"
        )
        layout.addWidget(self._lbl_stats)

        self._lbl_manual_badge = QLabel("")
        self._lbl_manual_badge.setStyleSheet(
            f"background: rgba(148,226,213,0.14); color: {_C['teal']}; "
            f"border: 1px solid rgba(148,226,213,0.35); border-radius: 6px; "
            f"padding: 2px 10px; font-size: 9pt; font-weight: 700; "
            f"font-family: 'Segoe UI', 'Segoe UI', sans-serif;"
        )
        self._lbl_manual_badge.hide()
        layout.addWidget(self._lbl_manual_badge)

        layout.addStretch()

        btn_cancel = QPushButton("Annuler")
        btn_cancel.setFixedHeight(36)
        btn_cancel.setMinimumWidth(100)
        btn_cancel.setStyleSheet(self._btn_secondary_style())
        btn_cancel.clicked.connect(self.reject)
        layout.addWidget(btn_cancel)

        self._btn_validate = QPushButton("  Valider la sélection manuelle")
        self._btn_validate.setFixedHeight(36)
        self._btn_validate.setMinimumWidth(200)
        self._btn_validate.setStyleSheet(self._btn_primary_style())
        self._btn_validate.clicked.connect(self._on_validate)
        layout.addWidget(self._btn_validate)

        return widget

    # ──────────────────────────────────────────────────────────────────────────
    # Grille de cartes
    # ──────────────────────────────────────────────────────────────────────────

    def _filtered_nodes(self) -> List[Dict[str, Any]]:
        """Retourne les nœuds correspondant aux filtres actifs."""
        result = self._all_nodes

        # Filtre textuel sur l'ID
        if self._search_text:
            try:
                nid_search = int(self._search_text)
                result = [n for n in result if int(n.get("node_id", -1)) == nid_search]
            except ValueError:
                result = []
            return result

        idx = self._active_filter_idx
        if idx == 1:  # JF
            result = [n for n in result if n.get("is_mrd_jf")]
        elif idx == 2:  # Flo
            result = [n for n in result if n.get("is_mrd_flo")]
        elif idx == 3:  # ELN
            result = [n for n in result if n.get("is_mrd_eln")]
        elif idx == 4:  # Non sélectionnés — AUCUNE méthode algo ne les a retenus
            result = [
                n
                for n in result
                if not n.get("is_mrd_jf") and not n.get("is_mrd_flo") and not n.get("is_mrd_eln")
            ]
        elif idx == 5:  # Ajoutés manuellement
            result = [
                n
                for n in result
                if not n.get("is_mrd_jf")
                and not n.get("is_mrd_flo")
                and not n.get("is_mrd_eln")
                and self._initial_included.get(int(n.get("node_id", 0)), False)
            ]

        return result

    def _clear_grid(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards.clear()

    def _populate_grid(self) -> None:
        """Reconstruit la grille selon les filtres actifs."""
        # Sauvegarder les états de décision courants avant de détruire les cartes
        current_states: Dict[int, bool] = {c.node_id: c.is_included for c in self._cards}

        self._clear_grid()
        nodes = self._filtered_nodes()

        if not nodes:
            self._scroll.hide()
            self._lbl_empty.show()
            self._lbl_count.setText("0 nœud")
            return

        self._lbl_empty.hide()
        self._scroll.show()

        # Nombre de colonnes depuis la largeur du viewport
        vp_w = self._scroll.viewport().width() if self._scroll.viewport() else 0
        if vp_w < self._CARD_MIN_W:
            vp_w = max(self.width() - 60, 900)
        cols = max(1, min(6, vp_w // self._CARD_MIN_W))
        cols = min(cols, len(nodes))

        for idx, node in enumerate(nodes):
            nid = int(node.get("node_id", 0))

            # L'état de décision persiste si la carte existait déjà
            if nid in current_states:
                initial = current_states[nid]
            else:
                initial = self._initial_included.get(nid, False)

            card = ExpertNodeCard(
                node=node,
                mfi_data=self._mfi_data,
                marker_cols=self._marker_cols,
                initial_included=initial,
                radar_cache=self._radar_cache,
                parent=None,
            )
            card.decisionChanged.connect(self._on_card_decision_changed)
            row_i, col_i = divmod(idx, cols)
            self._grid.addWidget(card, row_i, col_i)
            self._cards.append(card)

        for c in range(cols):
            self._grid.setColumnStretch(c, 1)

        total = len(self._all_nodes)
        shown = len(nodes)
        if shown == total:
            self._lbl_count.setText(f"{total} nœud(s)")
        else:
            self._lbl_count.setText(f"{shown} / {total} affiché(s)")

        self._refresh_stats()

        # Annuler tout batch précédent en cours, puis démarrer le nouveau
        self._radar_batch_gen = iter(list(self._cards))
        self._radar_batch_id = getattr(self, "_radar_batch_id", 0) + 1
        _bid = self._radar_batch_id
        QTimer.singleShot(0, lambda: self._pump_radar_batch(_bid))

    def _pump_radar_batch(self, batch_id: int, chunk: int = 6) -> None:
        """Charge `chunk` radars puis cède la main à l'event loop, jusqu'à épuisement."""
        if batch_id != self._radar_batch_id:
            return  # Ce batch est périmé (un nouveau _populate_grid a été appelé)
        gen = getattr(self, "_radar_batch_gen", None)
        if gen is None:
            return
        loaded = 0
        while loaded < chunk:
            try:
                card = next(gen)
            except StopIteration:
                return
            try:
                card.load_radar_deferred()
            except Exception:
                pass
            loaded += 1
        QTimer.singleShot(15, lambda: self._pump_radar_batch(batch_id, chunk))

    # ──────────────────────────────────────────────────────────────────────────
    # Slots
    # ──────────────────────────────────────────────────────────────────────────

    def _schedule_radar_batch(self, cards: List, batch_size: int = 8, delay_ms: int = 0) -> None:
        """Obsolète — conservé pour compatibilité, délègue au nouveau pump."""
        pass

        QTimer.singleShot(delay_ms, _load_batch)

    def _on_filter_changed(self, idx: int) -> None:
        self._active_filter_idx = idx
        self._populate_grid()

    def _on_search_changed(self, text: str) -> None:
        self._search_text = text.strip()
        self._populate_grid()

    def _on_card_decision_changed(self, node_id: int, is_included: bool) -> None:
        # Mettre à jour l'état initial pour persistance lors des refiltrages
        self._initial_included[node_id] = is_included
        self._refresh_stats()

    def _on_validate(self) -> None:
        """
        Collecte les nœuds validés manuellement (GARDER + non-algo)
        en réconciliant les cartes visibles et les états persistés.
        """
        # Pour les cartes actuellement visibles : état direct
        for c in self._cards:
            self._initial_included[c.node_id] = c.is_included

        included_ids = [
            int(node.get("node_id", 0))
            for node in self._all_nodes
            if self._initial_included.get(int(node.get("node_id", 0)), False)
        ]

        # Construire la liste finale : tous les nœuds non-algo marqués comme inclus
        manually_added = [
            node
            for node in self._all_nodes
            if not (node.get("is_mrd_jf") or node.get("is_mrd_flo") or node.get("is_mrd_eln"))
            and self._initial_included.get(int(node.get("node_id", 0)), False)
        ]
        self.curation_applied.emit(
            {
                "included_ids": included_ids,
                "manual_ids": [int(n.get("node_id", 0)) for n in manually_added],
            }
        )
        self.nodes_manually_added.emit(manually_added)
        self.accept()

    # ──────────────────────────────────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────────────────────────────────

    def _refresh_stats(self) -> None:
        total = len(self._all_nodes)
        algo_count = sum(
            1
            for n in self._all_nodes
            if n.get("is_mrd_jf") or n.get("is_mrd_flo") or n.get("is_mrd_eln")
        )
        # Compter les ajouts manuels : non-algo ET marqués included (état persisté)
        manual_new = sum(
            1
            for n in self._all_nodes
            if not (n.get("is_mrd_jf") or n.get("is_mrd_flo") or n.get("is_mrd_eln"))
            and self._initial_included.get(int(n.get("node_id", 0)), False)
        )

        self._lbl_stats.setText(
            f"Total : <b>{total}</b> nœuds   ·   "
            f"Algo (JF/Flo/ELN) : <b>{algo_count}</b>   ·   "
            f"Ajouts manuels : <b>{manual_new}</b>"
        )
        if manual_new > 0:
            self._lbl_manual_badge.setText(f"+ {manual_new} ajout(s) manuel(s)")
            self._lbl_manual_badge.show()
        else:
            self._lbl_manual_badge.hide()

    # ──────────────────────────────────────────────────────────────────────────
    # Redimensionnement
    # ──────────────────────────────────────────────────────────────────────────

    def resizeEvent(self, event: Any) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if not self._cards:
            return
        vp_w = self._scroll.viewport().width() if self._scroll.viewport() else 0
        if vp_w < self._CARD_MIN_W:
            vp_w = max(self.width() - 60, 900)
        new_cols = max(1, min(6, vp_w // self._CARD_MIN_W))
        new_cols = min(new_cols, len(self._cards))
        current_cols = self._grid.columnCount()
        if new_cols != current_cols:
            # Debounce : annule le timer précédent si resize rapide (ex: passage plein écran)
            if hasattr(self, "_resize_timer") and self._resize_timer is not None:
                self._resize_timer.stop()
            from PyQt5.QtCore import QTimer as _QTimer

            self._resize_timer = _QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._populate_grid)
            self._resize_timer.start(150)

    # ──────────────────────────────────────────────────────────────────────────
    # Styles helper
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _btn_primary_style() -> str:
        return f"""
            QPushButton {{
                background: rgba(148,226,213,0.20); color: {_C["teal"]};
                border: 1px solid rgba(148,226,213,0.50);
                border-radius: 8px; font-weight: 700; font-size: 9pt;
                padding: 0 16px;
            }}
            QPushButton:hover {{
                background: rgba(148,226,213,0.32);
                border-color: rgba(148,226,213,0.75);
            }}
            QPushButton:pressed {{
                background: rgba(148,226,213,0.40);
            }}
            QPushButton:focus {{
                border-color: rgba(148,226,213,0.90);
                background: rgba(148,226,213,0.34);
                outline: none;
            }}
            QPushButton:disabled {{
                color: #EEF2F7;
                border-color: rgba(255,255,255,0.06);
                background: rgba(49,50,68,0.45);
            }}
        """

    @staticmethod
    def _btn_secondary_style() -> str:
        return f"""
            QPushButton {{
                background: rgba(49,50,68,0.70); color: {_C["subtext"]};
                border: 1px solid rgba(69,71,90,0.50);
                border-radius: 8px; font-size: 9pt; padding: 0 14px;
            }}
            QPushButton:hover {{
                background: rgba(69,71,90,0.80); color: {_C["text"]};
                border-color: rgba(99,101,126,0.70);
            }}
            QPushButton:pressed {{
                background: rgba(69,71,90,0.90);
            }}
            QPushButton:focus {{
                border-color: rgba(123,82,255,0.65);
                background: rgba(69,71,90,0.86);
                outline: none;
            }}
            QPushButton:disabled {{
                color: #EEF2F7;
                border-color: rgba(255,255,255,0.05);
                background: rgba(49,50,68,0.40);
            }}
        """

    # ──────────────────────────────────────────────────────────────────────────
    # Accès public
    # ──────────────────────────────────────────────────────────────────────────

    def get_manually_added_nodes(self) -> List[Dict[str, Any]]:
        """Retourne les nœuds non-algo marqués GARDER (utilisable après exec_())."""
        return [
            node
            for node in self._all_nodes
            if not (node.get("is_mrd_jf") or node.get("is_mrd_flo") or node.get("is_mrd_eln"))
            and self._initial_included.get(int(node.get("node_id", 0)), False)
        ]
