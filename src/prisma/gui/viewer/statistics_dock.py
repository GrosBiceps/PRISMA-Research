"""
src/gui/viewer/statistics_dock.py — Dock de statistiques détachable (Phase 5).

QDockWidget affichant un tableau population/count/%parent/%total/MFI.
Aucune logique métier : les données arrivent via set_dataframe().
États gérés : vide, chargement, données disponibles, erreur.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Optional

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDockWidget,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_logger = logging.getLogger("viewer.statistics_dock")

# Colonnes minimales attendues dans le DataFrame alimentant le dock.
# Les colonnes absentes sont simplement laissées vides.
_EXPECTED_COLUMNS = ["Population", "Count", "% Parent", "% Total", "MFI X", "MFI Y"]


class _DockState(Enum):
    EMPTY = auto()
    LOADING = auto()
    READY = auto()
    ERROR = auto()


class StatisticsDock(QDockWidget):
    """
    Dock détachable de statistiques de population.

    Utilisation typique (depuis StatisticsController) :

        dock.set_loading()
        ...calcul...
        dock.set_dataframe(df)   # ou dock.set_error("msg")

    Le dock peut être intégré dans un QMainWindow via addDockWidget()
    ou utilisé comme widget flottant autonome.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Statistiques de population", parent)
        self.setObjectName("StatisticsDock")
        self.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea | Qt.RightDockWidgetArea)

        self._state = _DockState.EMPTY
        self._build_ui()

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def set_loading(self) -> None:
        """Passe en état 'chargement' : affiche un indicateur textuel."""
        self._state = _DockState.LOADING
        self._lbl_status.setText("Calcul des statistiques en cours…")
        self._stack.setCurrentWidget(self._status_page)
        _logger.debug("StatisticsDock → LOADING")

    def set_error(self, message: str) -> None:
        """Passe en état 'erreur' : affiche le message."""
        self._state = _DockState.ERROR
        self._lbl_status.setText(f"Erreur : {message}")
        self._stack.setCurrentWidget(self._status_page)
        _logger.warning("StatisticsDock → ERROR : %s", message)

    def clear(self) -> None:
        """Remet le dock à l'état vide."""
        self._state = _DockState.EMPTY
        self._table.setRowCount(0)
        self._lbl_status.setText("Aucune donnée.")
        self._stack.setCurrentWidget(self._status_page)
        _logger.debug("StatisticsDock → EMPTY")

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """
        Alimente le tableau avec un DataFrame.

        Colonnes reconnues (insensibles à la casse) :
          Population, Count, % Parent, % Total, MFI X, MFI Y

        Les colonnes absentes du DataFrame sont laissées vides.
        Une DataFrame vide déclenche clear().
        """
        if df is None or df.empty:
            self.clear()
            return

        # Normaliser les noms de colonnes du df en correspondance avec _EXPECTED_COLUMNS
        col_map: dict[str, str] = {}
        for expected in _EXPECTED_COLUMNS:
            for actual in df.columns:
                if str(actual).strip().lower() == expected.strip().lower():
                    col_map[expected] = str(actual)
                    break

        self._table.setRowCount(len(df))
        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, col_name in enumerate(_EXPECTED_COLUMNS):
                actual_col = col_map.get(col_name)
                if actual_col is not None and actual_col in row.index:
                    val = str(row[actual_col])
                else:
                    val = ""
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self._table.setItem(row_idx, col_idx, item)

        self._state = _DockState.READY
        self._stack.setCurrentWidget(self._table_page)
        _logger.debug("StatisticsDock → READY (%d lignes)", len(df))

    # ------------------------------------------------------------------
    # Construction UI interne
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root_layout = QVBoxLayout(container)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._stack = QStackedWidget()

        # Page statut (loading / empty / error)
        self._status_page = QWidget()
        sp_layout = QVBoxLayout(self._status_page)
        sp_layout.setAlignment(Qt.AlignCenter)
        self._lbl_status = QLabel("Aucune donnée.")
        self._lbl_status.setAlignment(Qt.AlignCenter)
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setStyleSheet("color: #8899AA; font-size: 11px;")
        sp_layout.addWidget(self._lbl_status)

        # Page tableau
        self._table_page = QWidget()
        tp_layout = QVBoxLayout(self._table_page)
        tp_layout.setContentsMargins(0, 0, 0, 0)
        tp_layout.setSpacing(0)
        self._table = self._build_table()
        tp_layout.addWidget(self._table)

        self._stack.addWidget(self._status_page)
        self._stack.addWidget(self._table_page)
        self._stack.setCurrentWidget(self._status_page)

        root_layout.addWidget(self._stack)
        self.setWidget(container)
        self._apply_style()

    def _build_table(self) -> QTableWidget:
        table = QTableWidget(0, len(_EXPECTED_COLUMNS))
        table.setHorizontalHeaderLabels(_EXPECTED_COLUMNS)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, len(_EXPECTED_COLUMNS)):
            table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        return table

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QDockWidget {
                background: #04070D;
                color: #EEF2F7;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QDockWidget::title {
                background: #0C1220;
                color: #5BAAFF;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 11px;
            }
            QTableWidget {
                background: #04070D;
                color: #EEF2F7;
                border: none;
                font-size: 10px;
                gridline-color: #141E2E;
            }
            QHeaderView::section {
                background: #0C1220;
                color: #5BAAFF;
                border: none;
                padding: 2px 6px;
                font-size: 10px;
            }
            QTableWidget::item:alternate { background: #080D18; }
            QTableWidget::item:selected  { background: #7B52FF; color: #FFFFFF; }
        """)
