# -*- coding: utf-8 -*-
"""
home_tab.py â€” Onglet Accueil MRD (rÃ©sultats principaux).

Affiche aprÃ¨s analyse :
  - Informations patient / fichier
  - Gauges MRD par mÃ©thode (selon le choix utilisateur)
  - Tableau des nÅ“uds SOM MRD positifs avec filtre par mÃ©thode
  - RÃ©sumÃ© clinique textuel

Avant analyse : Ã©cran d'attente avec Ã©tat des donnÃ©es chargÃ©es.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QGridLayout,
    QStackedWidget,
    QPushButton,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib

matplotlib.use("Qt5Agg")

from src.legacy_clinical.mrd_gauge import MRDGauge
from src.legacy_clinical.mrd_node_table import MRDNodeTable
from src.legacy_clinical.mrd_adapter import adapt_mrd_result, adapt_all_nodes

# â”€â”€ PRISMA v2.0 palette (replaces Catppuccin Mocha) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_SURFACE0 = "#0C1220"  # --surface
_SURFACE1 = "#101825"  # --raised
_BASE = "#080D18"  # --deep
_MANTLE = "#04070D"  # --void
_TEXT = "#EEF2F7"  # --paper
_SUBTEXT = "rgba(238,242,247,0.55)"
_BLUE = "#5BAAFF"  # ch-v500 / info
_GREEN = "#39FF8A"  # ch-fitc / accent
_RED = "#FF3D6E"  # ch-apc  / danger
_YELLOW = "#FFE032"  # ch-percp / warn
_LAVENDER = "#7B52FF"  # ch-v450 / brand

# Font used in all matplotlib figures
_FONT_FAMILY = "Segoe UI"
_FONT_FALLBACK = ["Segoe UI", "Arial", "Arial", "Helvetica", "sans-serif"]


class HomeTab(QWidget):
    """
    Onglet Accueil â€” rÃ©sumÃ© MRD post-analyse.

    Interfaces publiques :
      load_result(result, method_used)  â†’ affiche les rÃ©sultats
      show_waiting()                    â†’ revient Ã  l'Ã©cran d'attente

    Signaux :
      curation_changed()  â†’ Ã©mis aprÃ¨s toute modification de validation experte
                            (ajout/suppression de nÅ“ud). MainWindow l'Ã©coute pour
                            patcher le HTML et mettre Ã  jour son Ã©tat interne.
    """

    curation_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._gauges: List[MRDGauge] = []
        # Ã‰tat du toggle dÃ©nominateur MRD
        # False = toutes cellules patho  |  True = cellules patho CD45+ seulement
        self._denom_cd45_only: bool = False
        self._raw_mrd_result: Any = None  # MRDResult brut stockÃ© pour recalcul
        self._current_method: str = "all"
        self._build_ui()

    # ------------------------------------------------------------------
    # Construction UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Stack : page 0 = attente, page 1 = rÃ©sultats
        from PyQt5.QtWidgets import QStackedWidget

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_waiting_page())
        self._stack.addWidget(self._build_results_page())
        root.addWidget(self._stack)

        self._stack.setCurrentIndex(0)

    def _build_waiting_page(self) -> QWidget:
        """Ã‰cran placeholder avant analyse."""
        page = QWidget()
        page.setObjectName("waitingPage")
        page.setStyleSheet("""
            QWidget#waitingPage {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #080D18, stop:1 #04070D);
            }
        """)
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(24, 24, 24, 32)
        layout.setSpacing(20)

        # Conteneur principal
        container = QWidget()
        container.setObjectName("waitingContainer")
        container.setMinimumWidth(820)
        container.setMaximumWidth(980)
        container.setStyleSheet("""
            QWidget#waitingContainer {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(16, 24, 37, 0.90), stop:1 rgba(12, 18, 32, 0.90));
                border-radius: 0px;
                border: 1px solid rgba(255, 255, 255, 0.055);
                border-top: 1px solid rgba(123, 82, 255, 0.35);
            }
        """)
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(36, 32, 36, 28)
        c_layout.setSpacing(20)

        badge = QLabel("PRÃŠT POUR L'ANALYSE")
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(
            "background: rgba(123, 82, 255, 0.14); color: #EEF2F7; "
            "border: 1px solid rgba(123, 82, 255, 0.35); "
            "border-radius: 0px; padding: 6px 14px; "
            "font-family: 'Consolas', 'Cascadia Code', monospace; "
            "font-size: 8.5pt; font-weight: 600; letter-spacing: 0.16em;"
        )
        c_layout.addWidget(badge, alignment=Qt.AlignHCenter)

        title = QLabel("FlowSOM MRD Analyzer")
        title.setFont(QFont("Segoe UI", 27, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #EEF2F7; background: transparent;")
        c_layout.addWidget(title)

        sub = QLabel(
            "Configurez vos dossiers FCS et paramÃ¨tres, puis lancez le pipeline.\n"
            "L'accueil affichera automatiquement les rÃ©sultats MRD dÃ¨s la fin du calcul."
        )
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(
            "color: #EEF2F7; background: transparent;"
            "font-size: 11pt; font-family: 'Segoe UI', 'Segoe UI', sans-serif;"
        )
        sub.setWordWrap(True)
        c_layout.addWidget(sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: rgba(255,255,255,0.055); max-height: 1px;")
        c_layout.addWidget(sep)

        body = QWidget()
        body.setObjectName("waitingBody")
        body.setStyleSheet("QWidget#waitingBody { background: transparent; }")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(18)

        left = QWidget()
        left.setObjectName("waitingLeftCard")
        left.setStyleSheet(
            "QWidget#waitingLeftCard {"
            "background: rgba(16, 24, 37, 0.88);"
            "border-radius: 0px;"
            "border: 1px solid rgba(255, 255, 255, 0.055);"
            "border-top: 1px solid rgba(57, 255, 138, 0.30);"
            "}"
        )
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(20, 18, 20, 18)
        left_v.setSpacing(12)

        left_title = QLabel("Ã‰tapes avant lancement")
        left_title.setStyleSheet(
            "color: #EEF2F7; font-size: 9pt; font-weight: 700; background: transparent;"
            "font-family: 'Consolas', 'Cascadia Code', monospace; letter-spacing: 0.12em;"
        )
        left_v.addWidget(left_title)

        def _step_line(num: str, text: str) -> QLabel:
            lbl = QLabel(f"{num}. {text}")
            lbl.setWordWrap(True)
            lbl.setStyleSheet(
                "color: #EEF2F7; font-size: 9pt; background: transparent;"
                "font-family: 'Consolas', 'Cascadia Code', monospace;"
            )
            return lbl

        left_v.addWidget(_step_line("1", "Importer les dossiers FCS (sain et pathologique)."))
        left_v.addWidget(_step_line("2", "VÃ©rifier les paramÃ¨tres SOM, MRD et prÃ©-gating."))
        left_v.addWidget(_step_line("3", "Cliquer sur Lancer le Pipeline dans l'Ã©tape ExÃ©cution."))

        tip = QLabel(
            "Conseil: si vous relancez une analyse, les checkpoints accÃ©lÃ¨rent les recalculs."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(
            "color: #EEF2F7; font-size: 8.5pt; "
            "font-family: 'Consolas', 'Cascadia Code', monospace;"
            "background: rgba(123, 82, 255, 0.08);"
            "border: 1px solid rgba(123, 82, 255, 0.22); border-radius: 0px; padding: 10px;"
        )
        left_v.addWidget(tip)
        left_v.addStretch()

        right = QWidget()
        right.setObjectName("waitingRightCard")
        right.setStyleSheet(
            "QWidget#waitingRightCard {"
            "background: rgba(16, 24, 37, 0.88);"
            "border-radius: 0px;"
            "border: 1px solid rgba(255, 255, 255, 0.055);"
            "border-top: 1px solid rgba(91, 170, 255, 0.30);"
            "}"
        )
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(20, 18, 20, 18)
        right_v.setSpacing(12)

        right_title = QLabel("Ce qui apparaÃ®tra ici aprÃ¨s calcul")
        right_title.setStyleSheet(
            "color: #EEF2F7; font-size: 9pt; font-weight: 700; background: transparent;"
            "font-family: 'Consolas', 'Cascadia Code', monospace; letter-spacing: 0.12em;"
        )
        right_v.addWidget(right_title)

        features = [
            "RÃ©sultat MRD par mÃ©thode (gauges JF / Flo / ELN)",
            "Conclusion clinique synthÃ©tique positive/nÃ©gative",
            "Tableau des nÅ“uds SOM MRD positifs avec filtres",
            "Profils radar d'expression pour les nÅ“uds dÃ©tectÃ©s",
        ]
        for feat in features:
            lbl = QLabel(f"â€¢ {feat}")
            lbl.setWordWrap(True)
            lbl.setStyleSheet(
                "color: #EEF2F7; font-size: 9pt; background: transparent;"
                "font-family: 'Consolas', 'Cascadia Code', monospace;"
            )
            right_v.addWidget(lbl)

        right_v.addStretch()

        body_layout.addWidget(left, 1)
        body_layout.addWidget(right, 1)
        c_layout.addWidget(body)

        footer = QLabel("Statut actuel: en attente d'une exÃ©cution du pipeline")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet(
            "color: #EEF2F7; background: transparent;"
            "font-size: 8.8pt; font-weight: 500;"
            "font-family: 'Consolas', 'Cascadia Code', monospace; letter-spacing: 0.06em;"
        )
        c_layout.addWidget(footer)

        layout.addWidget(container, alignment=Qt.AlignCenter)
        return page

    def _build_results_page(self) -> QWidget:
        """Page principale des rÃ©sultats MRD."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea {{ background: {_BASE}; border: none; }}")

        content = QWidget()
        content.setStyleSheet(f"background: {_BASE};")
        self._results_layout = QVBoxLayout(content)
        self._results_layout.setContentsMargins(20, 20, 20, 20)
        self._results_layout.setSpacing(16)

        # â”€â”€ Infos patient â”€â”€
        self._patient_card = self._build_patient_card()
        self._results_layout.addWidget(self._patient_card)

        # â”€â”€ Bande de dÃ©cision clinique (prioritÃ© de lecture) â”€â”€
        self._summary_card = self._build_summary_card()
        self._results_layout.addWidget(self._summary_card, 0)

        # â”€â”€ Barre de contrÃ´le dÃ©nominateur MRD â”€â”€
        self._denom_bar = self._build_denom_bar()
        self._results_layout.addWidget(self._denom_bar)

        # â”€â”€ Gauges MRD â”€â”€
        self._gauge_container = QWidget()
        self._gauge_container.setStyleSheet("background: transparent;")
        # Fixed verticalement : les gauges ont une hauteur naturelle et ne
        # doivent JAMAIS voler de l'espace Ã  la grille de validation en dessous.
        self._gauge_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._gauge_row = QHBoxLayout(self._gauge_container)
        self._gauge_row.setContentsMargins(0, 0, 0, 0)
        self._gauge_row.setSpacing(12)
        # stretch=0 : prend uniquement sa hauteur naturelle (sizeHint)
        self._results_layout.addWidget(self._gauge_container, 0)

        # â”€â”€ Tableau nÅ“uds MRD (grille de validation) â”€â”€
        self._node_table = MRDNodeTable()
        self._node_table.combo_filter.currentIndexChanged.connect(self._on_node_filter_changed)
        self._node_table.curated_ratio_changed.connect(self._on_curated_ratio_changed)
        self._node_table.manually_added_nodes_changed.connect(self._on_manually_added_nodes_changed)
        # Expanding/Expanding : la grille de validation rÃ©clame tout l'espace
        # vertical restant. setMinimumHeight est le filet de sÃ©curitÃ© absolu
        # (dÃ©jÃ  dÃ©fini dans MRDNodeTable.__init__, rÃ©pÃ©tÃ© ici par cohÃ©rence).
        self._node_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._node_table.setMinimumHeight(400)
        # stretch=1 : reÃ§oit tout l'espace vertical non attribuÃ© par les
        # widgets Ã  stretch=0 au-dessus (patient_card, denom_bar, gauges, summary).
        self._results_layout.addWidget(self._node_table, 1)

        # â”€â”€ Mini Spider Plots des nÅ“uds MRD â”€â”€
        self._spider_section = QWidget()
        self._spider_section.setStyleSheet("background: transparent;")
        spider_v = QVBoxLayout(self._spider_section)
        spider_v.setContentsMargins(0, 0, 0, 0)
        spider_v.setSpacing(6)

        self._spider_lbl = QLabel("Profils d'Expression â€” Clusters MRD (Radar)")
        self._spider_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self._spider_lbl.setStyleSheet(f"color: {_LAVENDER};")
        spider_v.addWidget(self._spider_lbl)

        # ScrollArea horizontal pour les spider plots (8 plots Ã— 244px = dÃ©borde sinon)
        spider_scroll = QScrollArea()
        spider_scroll.setWidgetResizable(True)
        spider_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        spider_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        spider_scroll.setFixedHeight(290)
        spider_scroll.setStyleSheet(f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:horizontal {{ background: {_SURFACE1}; height: 6px; border-radius: 3px; }}
            QScrollBar::handle:horizontal {{ background: {_SUBTEXT}; border-radius: 3px; }}
        """)

        self._spider_grid_widget = QWidget()
        self._spider_grid_widget.setStyleSheet("background: transparent;")
        self._spider_grid = QHBoxLayout(self._spider_grid_widget)
        self._spider_grid.setContentsMargins(4, 4, 4, 4)
        self._spider_grid.setSpacing(10)
        spider_scroll.setWidget(self._spider_grid_widget)
        spider_v.addWidget(spider_scroll)

        # stretch=0 : la section spider est masquÃ©e par dÃ©faut et ne prend de
        # place que quand elle est visible ; elle ne doit pas voler d'espace.
        self._results_layout.addWidget(self._spider_section, 0)
        self._spider_section.hide()

        # Pas de addStretch() ici : le stretch=1 sur _node_table absorbe dÃ©jÃ 
        # tout l'espace rÃ©siduel. Un stretch supplÃ©mentaire comprimerait la grille.

        # DonnÃ©es MFI stockÃ©es pour les spider plots
        self._mfi_data: Any = None
        self._marker_cols: List[str] = []
        self._mrd_nodes_all: List[Dict] = []

        scroll.setWidget(content)
        return scroll

    def _build_denom_bar(self) -> QWidget:
        """Barre de contrÃ´le explicite du dÃ©nominateur MRD actif."""
        bar = QWidget()
        bar.setObjectName("denomBar")
        bar.setStyleSheet(f"""
            QWidget#denomBar {{
                background: rgba(12, 18, 32, 0.92);
                border-radius: 0px;
                border: 1px solid rgba(255,255,255,0.055);
                border-top: 1px solid rgba(91,170,255,0.35);
            }}
        """)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(12)

        icon_lbl = QLabel("Ã·")
        icon_lbl.setFont(QFont("Segoe UI", 14, QFont.Bold))
        icon_lbl.setStyleSheet("color: #5BAAFF; background: transparent;")
        layout.addWidget(icon_lbl)

        lbl = QLabel("MODE DÃ‰NOMINATEUR")
        lbl.setFont(QFont("Consolas", 8, QFont.Bold))
        lbl.setStyleSheet(
            "color: #EEF2F7; background: transparent; letter-spacing: 0.12em;"
        )
        layout.addWidget(lbl)

        self.lbl_denom_active = QLabel("TOTAL PATHO")
        self.lbl_denom_active.setAlignment(Qt.AlignCenter)
        self.lbl_denom_active.setFixedHeight(24)
        self.lbl_denom_active.setStyleSheet(
            "color: #EEF2F7; background: rgba(91,170,255,0.14); "
            "border: 1px solid rgba(91,170,255,0.34); border-radius: 0px; "
            "padding: 0 10px; font-size: 10px; font-weight: 800; letter-spacing: 0.06em;"
        )
        layout.addWidget(self.lbl_denom_active)

        layout.addSpacing(6)

        self.lbl_denom_status = QLabel("Toutes cellules pathologiques")
        self.lbl_denom_status.setFont(QFont("Segoe UI", 10, QFont.DemiBold))
        self.lbl_denom_status.setStyleSheet(
            "color: #EEF2F7; background: transparent;"
        )
        layout.addWidget(self.lbl_denom_status)

        self.lbl_denom_count = QLabel("")
        self.lbl_denom_count.setStyleSheet(
            "color: #EEF2F7; background: transparent; font-size: 9px; "
            "font-weight: 600; font-family: 'Consolas', 'Cascadia Code', monospace;"
        )
        layout.addWidget(self.lbl_denom_count)

        layout.addStretch()

        self.btn_toggle_denom = QPushButton("  Activer mode CD45+")
        self.btn_toggle_denom.setEnabled(False)
        self.btn_toggle_denom.setFixedHeight(30)
        self.btn_toggle_denom.setStyleSheet(f"""
            QPushButton {{
                background: rgba(91,170,255,0.16);
                color: #EEF2F7;
                border: 1px solid rgba(91,170,255,0.42);
                border-radius: 0px;
                padding: 0 14px;
                font-size: 10px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background: rgba(91,170,255,0.28);
                border-color: rgba(91,170,255,0.60);
            }}
            QPushButton:pressed {{
                background: rgba(91,170,255,0.34);
                border-color: rgba(91,170,255,0.70);
            }}
            QPushButton:focus {{
                background: rgba(91,170,255,0.24);
                border-color: rgba(91,170,255,0.70);
                outline: none;
            }}
            QPushButton:disabled {{
                background: rgba(20,30,46,0.55);
                color: #EEF2F7;
                border-color: rgba(255,255,255,0.10);
            }}
            QPushButton:pressed {{
                background: rgba(91,170,255,0.34);
                border-color: rgba(91,170,255,0.70);
            }}
            QPushButton:focus {{
                background: rgba(91,170,255,0.24);
                border-color: rgba(91,170,255,0.70);
                outline: none;
            }}
        """)
        self.btn_toggle_denom.clicked.connect(self._toggle_mrd_denominator)
        layout.addWidget(self.btn_toggle_denom)

        return bar

    def _build_patient_card(self) -> QWidget:
        """Carte infos patient/fichier."""
        card = QWidget()
        card.setObjectName("patientCard")
        card.setStyleSheet(f"""
            QWidget#patientCard {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(16,24,37,0.94), stop:1 rgba(12,18,32,0.94));
                border-radius: 0px;
                border: 1px solid rgba(255,255,255,0.055);
                border-top: 1px solid rgba(91,170,255,0.30);
            }}
        """)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(32)

        def _info_block(label: str, attr: str) -> QWidget:
            block = QWidget()
            block.setStyleSheet("background: transparent;")
            v = QVBoxLayout(block)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(3)
            lbl = QLabel(label.upper())
            lbl.setStyleSheet(
                f"color: #EEF2F7; font-size: 8px; background: transparent; "
                f"font-weight: 700; letter-spacing: 0.1em;"
            )
            val = QLabel("â€”")
            val.setFont(QFont("Segoe UI", 11, QFont.Bold))
            val.setStyleSheet(f"color: {_TEXT}; background: transparent;")
            val.setWordWrap(True)
            setattr(self, attr, val)
            v.addWidget(lbl)
            v.addWidget(val)
            return block

        layout.addWidget(_info_block("Ã‰chantillon", "lbl_patient_stem"))

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("color: rgba(255,255,255,0.055);")
        layout.addWidget(sep1)

        layout.addWidget(_info_block("Date", "lbl_patient_date"))

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("color: rgba(255,255,255,0.055);")
        layout.addWidget(sep2)

        layout.addWidget(_info_block("Cellules totales", "lbl_patient_cells"))

        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setStyleSheet("color: rgba(255,255,255,0.055);")
        layout.addWidget(sep3)

        layout.addWidget(_info_block("Cellules pathologiques", "lbl_patient_patho"))
        layout.addStretch()

        self.lbl_run_time = QLabel("")
        self.lbl_run_time.setStyleSheet(
            "color: #EEF2F7; font-size: 8.5px; background: transparent;"
            "font-weight: 600; font-family: 'Consolas', 'Cascadia Code', monospace;"
        )
        self.lbl_run_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.lbl_run_time)

        return card

    def _build_summary_card(self) -> QWidget:
        """Bande de dÃ©cision clinique prioritaire."""
        card = QWidget()
        card.setObjectName("summaryCard")
        card.setStyleSheet(f"""
            QWidget#summaryCard {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(20, 18, 36, 0.92), stop:1 rgba(12, 18, 32, 0.92));
                border-radius: 0px;
                border: 1px solid rgba(255,255,255,0.055);
                border-left: 2px solid rgba(123,82,255,0.60);
            }}
        """)
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 14, 20, 14)
        v.setSpacing(5)

        lbl = QLabel("DÃ‰CISION CLINIQUE")
        lbl.setFont(QFont("Consolas", 8, QFont.Bold))
        lbl.setStyleSheet(
            "color: #EEF2F7; background: transparent; letter-spacing: 0.12em;"
        )
        v.addWidget(lbl)

        self.lbl_clinical = QLabel("â€”")
        self.lbl_clinical.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.lbl_clinical.setWordWrap(True)
        self.lbl_clinical.setStyleSheet(f"color: {_TEXT}; background: transparent;")
        v.addWidget(self.lbl_clinical)

        self.lbl_decision_ref = QLabel("")
        self.lbl_decision_ref.setWordWrap(True)
        self.lbl_decision_ref.setStyleSheet(
            "color: #EEF2F7; background: transparent; font-size: 10.5px; font-weight: 600;"
        )
        v.addWidget(self.lbl_decision_ref)

        self.lbl_decision_denom = QLabel("")
        self.lbl_decision_denom.setWordWrap(True)
        self.lbl_decision_denom.setStyleSheet(
            "color: #EEF2F7; background: transparent; font-size: 10px;"
        )
        v.addWidget(self.lbl_decision_denom)

        self.lbl_clinical_detail = QLabel("")
        self.lbl_clinical_detail.setWordWrap(True)
        self.lbl_clinical_detail.setStyleSheet(
            "color: #EEF2F7; background: transparent; font-size: 10.5px;"
        )
        v.addWidget(self.lbl_clinical_detail)

        return card

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def show_waiting(self) -> None:
        """Revient Ã  l'Ã©cran d'attente (ex: avant un nouveau run)."""
        self._stack.setCurrentIndex(0)

    def load_result(self, result: Any, method_used: str = "all") -> None:
        """
        Met Ã  jour l'affichage avec un nouveau PipelineResult.

        Args:
            result: PipelineResult post-pipeline.
            method_used: mÃ©thode MRD choisie ("all", "jf", "flo", "eln").
        """
        # Stocker le rÃ©sultat brut pour le toggle dÃ©nominateur
        self._raw_mrd_result = getattr(result, "mrd_result", None)
        self._current_method = method_used
        self._denom_cd45_only = False  # RÃ©initialiser Ã  chaque nouveau rÃ©sultat
        self._last_patient_stem = getattr(result, "patho_stem", "") or ""

        # Configurer le bouton toggle selon la disponibilitÃ© du dÃ©nominateur CD45+
        self._refresh_denom_bar()

        data = adapt_mrd_result(result, method_used)

        if not data["has_data"]:
            self.show_waiting()
            return

        # Stocker les donnÃ©es MFI pour les spider plots
        # PrioritÃ© : node_mfi_matrix (mÃ©diane par nÅ“ud SOM, mÃªme source que le radar HTML)
        # Fallback : recalcul depuis result.data (moyenne, moins prÃ©cis)
        self._mfi_data = None
        self._marker_cols = []
        try:
            import numpy as np

            node_mfi = getattr(result, "node_mfi_matrix", None)
            if node_mfi is not None and not node_mfi.empty:
                self._mfi_data = node_mfi
                self._marker_cols = list(node_mfi.columns)
            else:
                df = result.data
                if df is not None and "FlowSOM_cluster" in df.columns:
                    _meta_cols = {
                        "FlowSOM_cluster",
                        "FlowSOM_metacluster",
                        "condition",
                        "file_origin",
                        "xGrid",
                        "yGrid",
                        "xNodes",
                        "yNodes",
                        "size",
                        "Condition_Num",
                        "Condition",
                        "Timepoint",
                        "Timepoint_Num",
                    }
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    self._marker_cols = [c for c in numeric_cols if c not in _meta_cols]
                    if self._marker_cols:
                        self._mfi_data = df.groupby("FlowSOM_cluster")[self._marker_cols].mean()
        except Exception:
            pass

        self._update_patient_info(data["patient_info"])
        self._update_gauges(data["gauges"])
        self._last_gauges_data: List[Dict] = data["gauges"]  # exposÃ© pour export dashboard
        self._update_summary(data["gauges"], data["patient_info"])
        self._update_node_table(data["nodes"], data["gauges"])

        self._stack.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # Mise Ã  jour des sous-widgets
    # ------------------------------------------------------------------

    def _update_patient_info(self, info: Dict) -> None:
        self.lbl_patient_stem.setText(info.get("stem", "â€”") or "â€”")
        self.lbl_patient_date.setText(info.get("date", "â€”") or "â€”")
        n_cells = info.get("n_cells", 0)
        n_patho = info.get("n_cells_patho", 0)
        self.lbl_patient_cells.setText(f"{n_cells:,}" if n_cells else "â€”")
        self.lbl_patient_patho.setText(f"{n_patho:,}" if n_patho else "â€”")
        ts = info.get("timestamp", "")
        self.lbl_run_time.setText(f"Analyse : {ts}" if ts else "")

    def _update_gauges(self, gauges: List[Dict]) -> None:
        """RecrÃ©e les gauges selon les mÃ©thodes Ã  afficher."""
        # Vider les gauges existantes
        while self._gauge_row.count():
            item = self._gauge_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._gauges.clear()

        for g_data in gauges:
            gauge = MRDGauge(method_name=g_data["method"])
            gauge.update_data(g_data)
            gauge.setMinimumWidth(220)
            gauge.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._gauge_row.addWidget(gauge)
            self._gauges.append(gauge)

    def _get_active_denom_text(self) -> str:
        """Retourne une description lisible du dÃ©nominateur actif."""
        mrd = self._raw_mrd_result
        if mrd is None:
            return "Total pathologique"

        n_pre = getattr(mrd, "n_patho_pre_cd45", 0)
        total_patho = n_pre if n_pre > 0 else getattr(mrd, "total_cells_patho", 0)
        n_cd45pos = getattr(mrd, "n_patho_cd45pos", 0)

        if self._denom_cd45_only:
            return f"CD45+ pathologique ({n_cd45pos:,} cellules)"
        return f"Total pathologique ({total_patho:,} cellules)"

    def _update_summary(self, gauges: List[Dict], info: Dict) -> None:
        """GÃ©nÃ¨re le texte de conclusion clinique.

        La conclusion est basÃ©e sur JF et Flo uniquement (pas ELN).
        ELN est affichÃ© en gauge mais ne dicte pas la conclusion clinique finale.
        """
        if not gauges:
            self.lbl_clinical.setText("Aucune donnÃ©e MRD disponible")
            self.lbl_decision_ref.setText("")
            self.lbl_decision_denom.setText("")
            self.lbl_clinical_detail.setText("")
            return

        # Exclure ELN de la conclusion principale
        non_eln_gauges = [g for g in gauges if g["method"] not in ("ELN 2025", "ELN")]
        ref_gauges = non_eln_gauges if non_eln_gauges else gauges

        positives = [g for g in ref_gauges if g.get("positive") or g.get("low_level")]

        if not positives:
            self.lbl_clinical.setText("MRD NÃ©gative")
            self.lbl_clinical.setStyleSheet(
                f"color: {_GREEN}; background: transparent; font-size: 16px;"
            )
            self.lbl_decision_ref.setText(
                "MÃ©thodes de rÃ©fÃ©rence concordantes: JF/FLO non dÃ©tectÃ©es"
            )
            self.lbl_decision_denom.setText(f"DÃ©nominateur actif: {self._get_active_denom_text()}")
            details = [f"{g['method']} : {g['pct']:.4f} %" for g in gauges]
            self.lbl_clinical_detail.setText("  Â·  ".join(details))
        else:
            # La mÃ©thode de rÃ©fÃ©rence prioritaire est JF, puis Flo
            priority = ["JF", "Flo"]
            ref = None
            for p in priority:
                ref = next((g for g in positives if g["method"] == p), None)
                if ref:
                    break
            if ref is None:
                ref = max(positives, key=lambda g: g.get("pct", 0))

            self.lbl_clinical.setText(f"MRD Positive â€” {ref['pct']:.4f} % ({ref['method']})")
            self.lbl_clinical.setStyleSheet(
                f"color: {_RED}; background: transparent; font-size: 16px;"
            )
            self.lbl_decision_ref.setText(
                f"MÃ©thode de rÃ©fÃ©rence retenue: {ref['method']} ({ref['pct']:.4f} %)"
            )
            self.lbl_decision_denom.setText(f"DÃ©nominateur actif: {self._get_active_denom_text()}")
            details = []
            for g in gauges:
                status = "POSITIF" if (g.get("positive") or g.get("low_level")) else "nÃ©gatif"
                details.append(f"{g['method']} : {g['pct']:.4f} % ({status})")
            self.lbl_clinical_detail.setText("  Â·  ".join(details))

    def _update_node_table(self, nodes: List[Dict], gauges: List[Dict]) -> None:
        """Charge la grille de validation des nÅ“uds MRD."""
        available_methods = [g["method"] for g in gauges]
        self._mrd_nodes_all = nodes

        # DÃ©nominateur : total cellules viables (patho) pour le ratio validÃ©
        mrd = self._raw_mrd_result
        n_pre = getattr(mrd, "n_patho_pre_cd45", 0) if mrd else 0
        total_viable = n_pre if n_pre > 0 else (getattr(mrd, "total_cells_patho", 0) if mrd else 0)

        # Fournir TOUS les nÅ“uds SOM (y compris non-MRD) Ã  ExpertFocusDialog
        all_patient_nodes = adapt_all_nodes(self._raw_mrd_result)
        self._node_table.set_all_patient_nodes(all_patient_nodes if all_patient_nodes else nodes)

        self._node_table.load_nodes(
            nodes,
            available_methods,
            mfi_data=self._mfi_data,
            marker_cols=self._marker_cols,
            total_viable_cells=max(total_viable, 1),
        )
        # Afficher les spider plots pour la sÃ©lection initiale
        self._refresh_spider_plots(nodes, method_label="")

    def _on_curated_ratio_changed(self, method: str, ratio: float, n_mrd_cells: int) -> None:
        """
        Slot connectÃ© Ã  MRDNodeTable.curated_ratio_changed.

        Met Ã  jour (ou crÃ©e) une jauge "ValidÃ©" reflÃ©tant le ratio
        recalculÃ© aprÃ¨s validation experte des nÅ“uds.
        Toutes les gauges existantes JF/Flo/ELN sont conservÃ©es.
        """
        # Chercher une jauge validÃ©e dÃ©jÃ  prÃ©sente.
        # NOTE: la carte est crÃ©Ã©e avec method_name="ValidÃ©".
        # Ancien bug: recherche sur "Curated" uniquement => doublons Ã  chaque clic.
        curated_candidates: List[MRDGauge] = [
            g for g in self._gauges if g.method_name in ("ValidÃ©", "Curated")
        ]

        curated_gauge: Optional[MRDGauge] = curated_candidates[0] if curated_candidates else None

        # DÃ©fensif: si des doublons existent dÃ©jÃ , on conserve la premiÃ¨re jauge
        # et on supprime les autres de l'UI.
        if len(curated_candidates) > 1:
            for dup in curated_candidates[1:]:
                self._gauge_row.removeWidget(dup)
                if dup in self._gauges:
                    self._gauges.remove(dup)
                dup.deleteLater()

        if curated_gauge is None:
            curated_gauge = MRDGauge(method_name="ValidÃ©")
            curated_gauge.setMinimumWidth(220)
            curated_gauge.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._gauge_row.addWidget(curated_gauge)
            self._gauges.append(curated_gauge)

        curated_gauge.update_data(
            {
                "pct": ratio,
                "n_cells": n_mrd_cells,
                "n_nodes": sum(1 for c in self._node_table._cards if c.is_included),
                "positive": ratio > 0.01,
                "low_level": 0 < ratio <= 0.01,
                "positivity_threshold": 0.01,
            }
        )
        # Notifie MainWindow â†’ patch HTML + inject curation
        self.curation_changed.emit()

    def _on_manually_added_nodes_changed(self, manual_nodes: list) -> None:
        """
        Slot connectÃ© Ã  MRDNodeTable.manually_added_nodes_changed.

        RafraÃ®chit les spider plots avec l'Ã©tat courant complet (inclus tous
        les nÅ“uds retenus, pas seulement les manuels) et Ã©met curation_changed
        pour que MainWindow mette Ã  jour l'HTML + son _result interne.
        """
        all_curated = self._node_table.get_human_curated_results()
        # RafraÃ®chir les spider plots qu'il y ait des nÅ“uds ou non
        self._refresh_spider_plots(
            all_curated,
            method_label="Expert" if manual_nodes else "",
        )
        # Signale Ã  MainWindow qu'une curation a eu lieu â†’ patch HTML
        self.curation_changed.emit()

    # ------------------------------------------------------------------
    # Toggle dÃ©nominateur MRD
    # ------------------------------------------------------------------

    def _refresh_denom_bar(self) -> None:
        """Met Ã  jour l'Ã©tat de la barre de contrÃ´le dÃ©nominateur."""
        mrd = self._raw_mrd_result
        if mrd is None:
            self.btn_toggle_denom.setEnabled(False)
            self.lbl_denom_active.setText("TOTAL PATHO")
            self.lbl_denom_status.setText("Toutes cellules pathologiques")
            self.lbl_denom_count.setText("")
            return

        # n_patho_pre_cd45 = total patho avant gate CD45 (CD45+ + CD45-)
        # Si non disponible (gate inactive), fallback sur total_cells_patho.
        n_pre = getattr(mrd, "n_patho_pre_cd45", 0)
        total_patho = n_pre if n_pre > 0 else getattr(mrd, "total_cells_patho", 0)
        n_cd45pos = getattr(mrd, "n_patho_cd45pos", 0)

        # Le toggle est disponible uniquement si les deux valeurs sont distinctes
        cd45_available = n_cd45pos > 0 and n_cd45pos != total_patho

        if self._denom_cd45_only:
            self.lbl_denom_active.setText("CD45+ ACTIF")
            self.lbl_denom_active.setStyleSheet(
                "color: #d6fff0; background: rgba(137,220,235,0.20); "
                "border: 1px solid rgba(137,220,235,0.45); border-radius: 6px; "
                "padding: 0 10px; font-size: 10px; font-weight: 800; letter-spacing: 0.06em;"
            )
            self.lbl_denom_status.setText("Cellules pathologiques CD45+")
            self.lbl_denom_status.setStyleSheet(
                "color: #c7f6ff; background: transparent; font-size: 10px; font-weight: bold;"
            )
            self.lbl_denom_count.setText(f"({n_cd45pos:,} cellules)" if n_cd45pos > 0 else "")
            btn_label = "  Revenir en mode total patho"
        else:
            self.lbl_denom_active.setText("TOTAL PATHO")
            self.lbl_denom_active.setStyleSheet(
                "color: #dbe7ff; background: rgba(137,180,250,0.20); "
                "border: 1px solid rgba(137,180,250,0.42); border-radius: 6px; "
                "padding: 0 10px; font-size: 10px; font-weight: 800; letter-spacing: 0.06em;"
            )
            self.lbl_denom_status.setText("Toutes cellules pathologiques")
            self.lbl_denom_status.setStyleSheet(
                "color: #d3def9; background: transparent; font-size: 10px; font-weight: bold;"
            )
            self.lbl_denom_count.setText(f"({total_patho:,} cellules)" if total_patho > 0 else "")
            btn_label = "  Activer mode CD45+"

        self.btn_toggle_denom.setText(btn_label)
        self.btn_toggle_denom.setEnabled(cd45_available)

    def _toggle_mrd_denominator(self) -> None:
        """Bascule entre le dÃ©nominateur total patho et CD45+ patho."""
        self._denom_cd45_only = not self._denom_cd45_only
        self._refresh_denom_bar()

        mrd = self._raw_mrd_result
        if mrd is None:
            return

        # Recalculer les pourcentages MRD avec le nouveau dÃ©nominateur
        new_gauges = self._recompute_gauges_with_denom(mrd, self._current_method)
        self._update_gauges(new_gauges)
        self._update_summary(new_gauges, {"stem": self._last_patient_stem})

        # Conserver la curation experte en cours (Expert Focus / cartes MRD)
        # lors du changement de dÃ©nominateur : on recalcule le ratio validÃ©
        # sur les nÅ“uds actuellement gardÃ©s, avec le dÃ©nominateur actif.
        active_denom = self._get_active_patho_denominator(mrd)
        curated_nodes = (
            self._node_table.get_human_curated_results() if hasattr(self, "_node_table") else []
        )
        curated_cells = int(sum(int(n.get("n_patho", 0)) for n in curated_nodes))
        curated_ratio = (curated_cells / max(active_denom, 1) * 100.0) if active_denom > 0 else 0.0

        # Mettre Ã  jour le dÃ©nominateur interne de la table pour que les
        # recalculs suivants (clic GARDER/Ã‰CARTER) restent cohÃ©rents.
        if hasattr(self, "_node_table"):
            self._node_table._total_viable_cells = max(active_denom, 1)
            self._node_table._refresh_ratio_badge()

        self._on_curated_ratio_changed("Curated", curated_ratio, curated_cells)

        # Mettre Ã  jour le compteur "Cellules pathologiques" dans la carte patient
        n_cd45pos = getattr(mrd, "n_patho_cd45pos", 0)
        n_pre = getattr(mrd, "n_patho_pre_cd45", 0)
        total_patho = n_pre if n_pre > 0 else getattr(mrd, "total_cells_patho", 0)
        displayed = n_cd45pos if self._denom_cd45_only else total_patho
        if hasattr(self, "lbl_patient_patho") and displayed > 0:
            suffix = " (CD45+)" if self._denom_cd45_only else ""
            self.lbl_patient_patho.setText(f"{displayed:,}{suffix}")

    def _get_active_patho_denominator(self, mrd: Any) -> int:
        """Retourne le dÃ©nominateur pathologique actif selon le mode courant."""
        n_pre = getattr(mrd, "n_patho_pre_cd45", 0)
        total_patho = n_pre if n_pre > 0 else getattr(mrd, "total_cells_patho", 0)
        n_cd45pos = getattr(mrd, "n_patho_cd45pos", 0)

        if self._denom_cd45_only and n_cd45pos > 0:
            return int(n_cd45pos)
        if total_patho > 0:
            return int(total_patho)
        return 1

    def _recompute_gauges_with_denom(self, mrd: Any, method_used: str) -> List[Dict]:
        """
        Recalcule les gauges MRD avec le dÃ©nominateur choisi.

        Si _denom_cd45_only=True  â†’ dÃ©nominateur = mrd.n_patho_cd45pos (CD45+ patho)
        Sinon                     â†’ dÃ©nominateur = mrd.n_patho_pre_cd45 (total patho avant gate)
        """
        n_pre = getattr(mrd, "n_patho_pre_cd45", 0)
        total_patho = n_pre if n_pre > 0 else getattr(mrd, "total_cells_patho", 0)
        n_cd45pos = getattr(mrd, "n_patho_cd45pos", 0)

        denom = n_cd45pos if (self._denom_cd45_only and n_cd45pos > 0) else total_patho
        if denom <= 0:
            denom = max(total_patho, 1)

        def _pct(n_cells: int) -> float:
            return round(n_cells / denom * 100.0, 4) if denom > 0 else 0.0

        gauges: List[Dict] = []
        _show_jf = method_used in ("all", "jf")
        _show_flo = method_used in ("all", "flo")
        _show_eln = method_used == "eln"

        if _show_jf:
            n = getattr(mrd, "mrd_cells_jf", 0)
            p = _pct(n)
            gauges.append(
                {
                    "method": "JF",
                    "pct": p,
                    "n_cells": n,
                    "n_nodes": getattr(mrd, "n_nodes_mrd_jf", 0),
                    "positive": p > 0,
                    "positivity_threshold": None,
                }
            )

        if _show_flo:
            n = getattr(mrd, "mrd_cells_flo", 0)
            p = _pct(n)
            gauges.append(
                {
                    "method": "Flo",
                    "pct": p,
                    "n_cells": n,
                    "n_nodes": getattr(mrd, "n_nodes_mrd_flo", 0),
                    "positive": p > 0,
                    "positivity_threshold": None,
                }
            )

        if _show_eln:
            n = getattr(mrd, "mrd_cells_eln", 0)
            p = _pct(n)
            cfg = getattr(mrd, "config_snapshot", {})
            eln_cfg = cfg.get("eln_standards", {}) if isinstance(cfg, dict) else {}
            threshold = eln_cfg.get("clinical_positivity_pct", 0.1)
            gauges.append(
                {
                    "method": "ELN 2025",
                    "pct": p,
                    "n_cells": n,
                    "n_nodes": getattr(mrd, "n_nodes_mrd_eln", 0),
                    "positive": p >= threshold,
                    "low_level": (n > 0) and (p < threshold),
                    "positivity_threshold": threshold,
                }
            )

        return gauges

    def _on_node_filter_changed(self) -> None:
        """RafraÃ®chit les spider plots quand le filtre de mÃ©thode change."""
        selected = self._node_table.combo_filter.currentText()
        if selected == "Toutes les mÃ©thodes":
            filtered = self._mrd_nodes_all
            method_label = ""
        elif selected == "JF":
            filtered = [n for n in self._mrd_nodes_all if n["is_mrd_jf"]]
            method_label = "JF"
        elif selected == "Flo":
            filtered = [n for n in self._mrd_nodes_all if n["is_mrd_flo"]]
            method_label = "Flo"
        elif selected in ("ELN 2025", "ELN"):
            filtered = [n for n in self._mrd_nodes_all if n["is_mrd_eln"]]
            method_label = "ELN"
        else:
            filtered = self._mrd_nodes_all
            method_label = ""
        self._refresh_spider_plots(filtered, method_label=method_label)

    # Marqueurs techniques Ã  exclure du spider plot (pas cliniquement pertinents)
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

    def _filter_clinical_markers(self, markers: List[str]) -> List[str]:
        """Retourne uniquement les marqueurs cliniques (exclut FSC, SSC, Time, etc.)."""
        result = []
        for m in markers:
            m_low = m.lower().strip()
            if any(m_low == t or m_low.startswith(t) for t in self._TECHNICAL_MARKERS):
                continue
            result.append(m)
        return result

    # Palette PRISMA v2.0 â€” cytometric channels, visuellement distincts sur fond sombre
    _NODE_COLORS = [
        "#7B52FF",  # V450 Â· brand
        "#39FF8A",  # FITC Â· accent
        "#5BAAFF",  # V500 Â· info
        "#FF9B3D",  # PE   Â· warm
        "#FF3D6E",  # APC  Â· danger
        "#FFE032",  # PerCPÂ· warn
        "#7EC8E3",  # SSC  Â· steel
        "#E8F4FF",  # FSC  Â· light
    ]

    def _refresh_spider_plots(self, nodes: List[Dict], *, method_label: str = "") -> None:
        """
        GÃ©nÃ¨re les mini spider plots pour les nÅ“uds MRD filtrÃ©s (max 8).

        Normalisation par nÅ“ud (identique Ã  plot_cluster_radar_mrd) :
        chaque profil est mis Ã  l'Ã©chelle sur son propre min/max pour que
        la forme du radar reflÃ¨te fidÃ¨lement le profil d'expression relatif,
        exactement comme dans le rapport HTML MÃ©thode JF/Flo.
        """
        # Vider l'ancienne grille
        while self._spider_grid.count():
            item = self._spider_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not nodes or self._mfi_data is None or not self._marker_cols:
            self._spider_section.hide()
            return

        import numpy as np

        # Exclure les marqueurs techniques
        markers = self._filter_clinical_markers(self._marker_cols)
        n_markers = len(markers)
        if n_markers < 3:
            self._spider_section.hide()
            return

        # Mettre Ã  jour le titre de section avec mÃ©thode + nombre de nÅ“uds
        n_nodes = len(nodes)
        if method_label:
            title_text = (
                f"Profils d'Expression â€” Clusters MRD MÃ©thode {method_label} (Radar)"
                f"  Â·  {n_nodes} nÅ“ud(s)"
            )
        else:
            title_text = f"Profils d'Expression â€” Clusters MRD (Radar)  Â·  {n_nodes} nÅ“ud(s)"
        self._spider_lbl.setText(title_text)

        angles = np.linspace(0, 2 * np.pi, n_markers, endpoint=False).tolist()
        angles_closed = angles + angles[:1]

        max_plots = min(n_nodes, 8)
        display_nodes = nodes[:max_plots]

        # Labels marqueurs courts
        short_labels = []
        for m in markers:
            parts = m.strip().split()
            candidate = parts[-1] if len(parts) > 1 else parts[0]
            short_labels.append(candidate[:10])

        for idx, node in enumerate(display_nodes):
            node_id = node["node_id"]
            if node_id not in self._mfi_data.index:
                continue

            mfi_row = self._mfi_data.loc[node_id, markers].values.astype(float)

            # â”€â”€ Normalisation par nÅ“ud (identique Ã  plot_cluster_radar_mrd) â”€â”€
            v_min, v_max = float(mfi_row.min()), float(mfi_row.max())
            norm_vals = (mfi_row - v_min) / (v_max - v_min + 1e-10)
            vals = list(norm_vals) + [norm_vals[0]]

            pct_p = node["pct_patho"]
            node_color = self._NODE_COLORS[idx % len(self._NODE_COLORS)]

            # Widget conteneur (radar + label sous)
            container = QWidget()
            container.setStyleSheet(f"background: {_SURFACE0}; border-radius: 0px;")
            c_layout = QVBoxLayout(container)
            c_layout.setContentsMargins(6, 6, 6, 6)
            c_layout.setSpacing(3)

            # Figure matplotlib radar
            fig = Figure(figsize=(2.8, 2.8), dpi=88)
            fig.patch.set_facecolor(_SURFACE0)
            ax = fig.add_subplot(111, polar=True)
            ax.set_facecolor(_BASE)

            # TracÃ© avec couleur PRISMA distincte par nÅ“ud
            ax.plot(angles_closed, vals, color=node_color, linewidth=2.2, zorder=3)
            ax.fill(angles_closed, vals, color=node_color, alpha=0.18, zorder=2)

            # Grille radiale discrÃ¨te
            ax.set_ylim(0, 1.05)
            ax.set_yticks([0.33, 0.66, 1.0])
            ax.set_yticklabels([])
            ax.yaxis.grid(True, color=(1, 1, 1, 0.12), linewidth=0.5, linestyle=":")
            ax.spines["polar"].set_color("rgba(255,255,255,0.10)" if False else (1, 1, 1, 0.10))
            ax.spines["polar"].set_linewidth(0.6)

            ax.set_xticks(angles)
            ax.set_xticklabels(
                short_labels,
                fontsize=7,
                color=_TEXT,
                fontweight="600",
                fontfamily=_FONT_FALLBACK,
            )
            ax.tick_params(axis="x", pad=9)

            # Marges pour laisser de la place aux labels
            fig.subplots_adjust(top=0.88, bottom=0.10, left=0.10, right=0.90)

            canvas = FigureCanvas(fig)
            canvas.setMinimumSize(246, 246)
            canvas.setMaximumSize(280, 280)
            c_layout.addWidget(canvas)

            # Label sous le radar : nÅ“ud + % patho + mÃ©thode
            label_parts = [f"NÅ“ud {node_id}"]
            if method_label:
                label_parts.append(f"[{method_label}]")
            label_parts.append(f"{pct_p:.0f}% patho")
            lbl_node = QLabel("  â€”  ".join(label_parts))
            lbl_node.setAlignment(Qt.AlignCenter)
            lbl_node.setStyleSheet(
                f"color: {node_color}; font-size: 10px; font-weight: bold; background: transparent;"
            )
            c_layout.addWidget(lbl_node)

            container.setFixedWidth(290)
            self._spider_grid.addWidget(container)

        self._spider_grid.addStretch()
        self._spider_section.show()

