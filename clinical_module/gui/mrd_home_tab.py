# -*- coding: utf-8 -*-
"""
home_tab.py — Résumé de recherche pour PRISMA Research.

L'onglet d'accueil expose un résumé générique des résultats du pipeline
sans composants MRD dédiés. Il sert de page de synthèse pour les tâches de
clustering, réduction dimensionnelle et analyse de population.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

_BASE = "#080D18"
_MANTLE = "#04070D"
_SURFACE0 = "#0C1220"
_SURFACE1 = "#101825"
_TEXT = "#EEF2F7"
_SUBTEXT = "rgba(238,242,247,0.55)"
_BLUE = "#5BAAFF"
_GREEN = "#39FF8A"
_LAVENDER = "#7B52FF"


class HomeTab(QWidget):
    """Résumé générique des résultats de recherche."""

    curation_changed = pyqtSignal()
    expert_focus_curation_applied = pyqtSignal(dict)
    verification_commit_requested = pyqtSignal(str)
    open_html_requested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._result: Any = None
        self._current_method: str = "all"
        self._last_gauges_data: List[Dict[str, Any]] = []
        self._validation_status: str = "Aucune validation"
        self._html_bar_visible: bool = True
        self._curation_emit_timer = QTimer(self)
        self._curation_emit_timer.setSingleShot(True)
        self._curation_emit_timer.timeout.connect(self.curation_changed.emit)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_waiting_page())
        self._stack.addWidget(self._build_results_page())
        self._stack.setCurrentIndex(0)
        root.addWidget(self._stack)

    def _build_waiting_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("waitingPage")
        page.setStyleSheet(
            "QWidget#waitingPage {"
            f"background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {_BASE}, stop:1 {_MANTLE});"
            "}"
        )
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(24, 24, 24, 32)
        layout.setSpacing(20)

        container = QWidget()
        container.setMinimumWidth(820)
        container.setMaximumWidth(980)
        container.setStyleSheet(
            "QWidget {"
            f"background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {_SURFACE1}, stop:1 {_SURFACE0});"
            "border: 1px solid rgba(255, 255, 255, 0.055);"
            "border-top: 1px solid rgba(123, 82, 255, 0.35);"
            "}"
        )
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(36, 32, 36, 28)
        c_layout.setSpacing(18)

        badge = QLabel("PRISMA RESEARCH")
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(
            "background: rgba(123, 82, 255, 0.14); color: #EEF2F7; "
            "border: 1px solid rgba(123, 82, 255, 0.35); "
            "padding: 6px 14px; font-family: 'Consolas', 'Cascadia Code', monospace; "
            "font-size: 8.5pt; font-weight: 600; letter-spacing: 0.16em;"
        )
        c_layout.addWidget(badge, alignment=Qt.AlignHCenter)

        title = QLabel("Analyse de populations et clustering")
        title.setFont(QFont("Segoe UI", 27, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {_TEXT}; background: transparent;")
        c_layout.addWidget(title)

        sub = QLabel(
            "Chargez vos échantillons, lancez la réduction dimensionnelle et le clustering, "
            "puis inspectez les populations détectées dans l'onglet résultats."
        )
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(
            "color: #EEF2F7; background: transparent;"
            "font-size: 11pt; font-family: 'Segoe UI', sans-serif;"
        )
        sub.setWordWrap(True)
        c_layout.addWidget(sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: rgba(255,255,255,0.055); max-height: 1px;")
        c_layout.addWidget(sep)

        row = QHBoxLayout()
        row.setSpacing(18)
        for title_text, body_text, accent in (
            ("Étape 1", "Importer les FCS et vérifier les métadonnées.", _GREEN),
            ("Étape 2", "Lancer la réduction dimensionnelle et le clustering.", _BLUE),
            ("Étape 3", "Explorer les populations et exporter les résultats.", _LAVENDER),
        ):
            row.addWidget(self._make_step_card(title_text, body_text, accent))

        c_layout.addLayout(row)
        c_layout.addStretch()
        layout.addWidget(container)
        return page

    def _make_step_card(self, title: str, text: str, accent: str) -> QWidget:
        card = QWidget()
        card.setMinimumHeight(150)
        card.setStyleSheet(
            "QWidget {"
            f"background: rgba(16, 24, 37, 0.88); border: 1px solid rgba(255,255,255,0.055);"
            f"border-top: 1px solid {accent};"
            "}"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(
            "color: #EEF2F7; font-size: 9pt; font-weight: 700; background: transparent;"
            "font-family: 'Consolas', 'Cascadia Code', monospace; letter-spacing: 0.12em;"
        )
        layout.addWidget(lbl_title)

        lbl_body = QLabel(text)
        lbl_body.setWordWrap(True)
        lbl_body.setStyleSheet(f"color: {_TEXT}; background: transparent; font-size: 10pt;")
        layout.addWidget(lbl_body)
        layout.addStretch()
        return card

    def _build_results_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = QWidget()
        self._header.setObjectName("resultsHeader")
        self._header.setStyleSheet(
            "QWidget#resultsHeader {"
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {_SURFACE0}, stop:1 {_SURFACE1});"
            "border-bottom: 1px solid rgba(255,255,255,0.05);"
            "}"
        )
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(20, 16, 20, 16)
        header_layout.setSpacing(12)

        left = QVBoxLayout()
        self.lbl_title = QLabel("Résumé de l'analyse")
        self.lbl_title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.lbl_title.setStyleSheet(f"color: {_TEXT}; background: transparent;")
        self.lbl_status = QLabel("En attente de résultats")
        self.lbl_status.setStyleSheet(f"color: {_SUBTEXT}; background: transparent;")
        left.addWidget(self.lbl_title)
        left.addWidget(self.lbl_status)
        header_layout.addLayout(left, 1)

        self.btn_open_report = QPushButton("Ouvrir le rapport")
        self.btn_open_report.setObjectName("primaryBtn")
        self.btn_open_report.clicked.connect(lambda: self.open_html_requested.emit("main"))
        header_layout.addWidget(self.btn_open_report)

        layout.addWidget(self._header)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(20, 18, 20, 18)
        body_layout.setSpacing(12)

        cards = QWidget()
        cards_layout = QHBoxLayout(cards)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(12)

        self.card_cells = self._metric_card("Cellules")
        self.card_clusters = self._metric_card("Métaclusters")
        self.card_elapsed = self._metric_card("Durée")
        self.card_outputs = self._metric_card("Exports")

        for card in (self.card_cells, self.card_clusters, self.card_elapsed, self.card_outputs):
            cards_layout.addWidget(card)

        body_layout.addWidget(cards)

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(320)
        self.details.setStyleSheet(
            "QTextEdit {"
            f"background: {_MANTLE}; color: {_TEXT}; border: 1px solid rgba(255,255,255,0.05);"
            "font-family: 'Consolas', 'Cascadia Code', monospace; font-size: 9.5pt;"
            "}"
        )
        body_layout.addWidget(self.details, 1)

        self.validation_bar = QFrame()
        self.validation_bar.setVisible(True)
        self.validation_bar.setStyleSheet(
            "QFrame {"
            "background: rgba(123, 82, 255, 0.10);"
            "border: 1px solid rgba(123, 82, 255, 0.22);"
            "}"
        )
        validation_layout = QHBoxLayout(self.validation_bar)
        validation_layout.setContentsMargins(14, 10, 14, 10)
        self.lbl_validation = QLabel("Aucune validation")
        self.lbl_validation.setStyleSheet(f"color: {_TEXT}; background: transparent;")
        validation_layout.addWidget(self.lbl_validation)
        validation_layout.addStretch()
        body_layout.addWidget(self.validation_bar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        return page

    def _metric_card(self, title: str) -> QWidget:
        card = QFrame()
        card.setObjectName("metricCard")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        card.setMinimumHeight(92)
        card.setStyleSheet(
            "QFrame#metricCard {"
            f"background: {_SURFACE1}; border: 1px solid rgba(255,255,255,0.055);"
            "border-top: 1px solid rgba(91, 170, 255, 0.28);"
            "}"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(
            "color: rgba(238,242,247,0.65); background: transparent;"
            "font-size: 8.5pt; letter-spacing: 0.08em; text-transform: uppercase;"
        )
        lbl_value = QLabel("—")
        lbl_value.setStyleSheet(
            f"color: {_TEXT}; background: transparent; font-size: 18pt; font-weight: 700;"
        )
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_value)
        card._value_label = lbl_value  # type: ignore[attr-defined]
        return card

    @staticmethod
    def _set_card_value(card: QWidget, value: str) -> None:
        label = getattr(card, "_value_label", None)
        if label is not None:
            label.setText(value)

    def show_waiting(self) -> None:
        self._stack.setCurrentIndex(0)
        self.lbl_title.setText("Résumé de l'analyse")
        self.lbl_status.setText("En attente de résultats")
        self.details.clear()
        self._set_card_value(self.card_cells, "—")
        self._set_card_value(self.card_clusters, "—")
        self._set_card_value(self.card_elapsed, "—")
        self._set_card_value(self.card_outputs, "—")

    def load_result(self, result: Any, method_used: str = "all") -> None:
        self._current_method = method_used
        self._result = result
        self._last_gauges_data = []

        if result is None or not getattr(result, "success", False):
            self.show_waiting()
            self.lbl_status.setText("Aucun résultat disponible")
            return

        self._stack.setCurrentIndex(1)
        n_cells = int(getattr(result, "n_cells", 0) or 0)
        n_metaclusters = int(getattr(result, "n_metaclusters", 0) or 0)
        elapsed = float(getattr(result, "elapsed_seconds", 0.0) or 0.0)
        output_files = getattr(result, "output_files", {}) or {}
        warnings = list(getattr(result, "warnings", []) or [])

        self.lbl_title.setText("Résumé de l'analyse")
        self.lbl_status.setText(f"Mode {method_used} · {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        self._set_card_value(self.card_cells, f"{n_cells:,}")
        self._set_card_value(self.card_clusters, f"{n_metaclusters}")
        self._set_card_value(self.card_elapsed, f"{elapsed:.1f}s")
        self._set_card_value(self.card_outputs, f"{len(output_files)}")

        lines: List[str] = []
        summary_text = getattr(result, "summary", None)
        if callable(summary_text):
            try:
                lines.append(summary_text())
            except Exception:
                pass

        if not lines:
            lines.extend(
                [
                    "Résumé de pipeline",
                    f"Cellules: {n_cells:,}",
                    f"Métaclusters: {n_metaclusters}",
                    f"Durée: {elapsed:.1f}s",
                ]
            )

        if warnings:
            lines.append("")
            lines.append("Avertissements:")
            lines.extend(f"- {warn}" for warn in warnings)

        if output_files:
            lines.append("")
            lines.append("Fichiers produits:")
            for key, path in output_files.items():
                lines.append(f"- {key}: {path}")

        self.details.setPlainText("\n".join(lines))

    def show_eln_html_bar(self, visible: bool) -> None:
        self._html_bar_visible = bool(visible)
        self.validation_bar.setVisible(visible)

    def set_validation_status(self, method_label: str, filter_label: str) -> None:
        self._validation_status = f"{method_label} · {filter_label}"
        self.lbl_validation.setText(self._validation_status)
