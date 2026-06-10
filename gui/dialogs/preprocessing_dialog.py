# -*- coding: utf-8 -*-
"""
preprocessing_dialog.py — Popup de configuration du pré-traitement par colonne.

Lit les colonnes d'un fichier FCS et permet de choisir, pour chaque canal,
une transformation (logicle / arcsinh / log10 / none) avec ses paramètres
(cofacteur arcsinh, paramètres logicle T/M/W/A globaux).

Sortie : dict de specs par colonne, prêt pour
``DataTransformer.apply_per_column_transforms`` :
    {nom_colonne: {"method", "cofactor", "T", "M", "W", "A"}}

Inspiré du moteur de transformation de la pipeline MRD de référence
(logicle T=2^18/M=4.5/W=0.5/A=0 ; arcsinh cofacteur 5).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_logger = logging.getLogger("prisma.preprocessing_dialog")

# Canaux scatter/techniques → défaut "none" (pas de transformation fluorescence).
_SCATTER_PATTERNS = ["FSC", "SSC", "TIME", "EVENT", "WIDTH", "HEIGHT"]
_METHODS = ["none", "logicle", "arcsinh", "log10"]
_DEFAULT_FLUO_METHOD = "logicle"


def read_fcs_channels(fcs_path: Path | str) -> List[str]:
    """Lit la liste des canaux (PnN, fallback PnS) d'un FCS sans charger les events.

    Returns:
        Liste ordonnée des noms de canaux, ou [] en cas d'échec.
    """
    try:
        import flowio

        fd = flowio.FlowData(str(fcs_path), only_text=True)
        names: List[str] = []
        for i in range(1, fd.channel_count + 1):
            ch = fd.channels.get(str(i), {})
            pnn = ch.get("PnN") or ch.get("PnS") or f"ch{i}"
            names.append(str(pnn))
        return names
    except Exception as exc:  # noqa: BLE001
        _logger.warning("Lecture canaux FCS échouée (%s) : %s", fcs_path, exc)
        return []


def _is_scatter(name: str) -> bool:
    return any(re.search(p, name, re.IGNORECASE) for p in _SCATTER_PATTERNS)


class PreprocessingDialog(QDialog):
    """Dialog de configuration des transformations par colonne."""

    def __init__(
        self,
        fcs_path: Optional[Path | str] = None,
        existing_specs: Optional[Dict[str, dict]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pré-traitement des canaux (transformation par colonne)")
        self.setMinimumSize(720, 560)
        self.setObjectName("preprocessingDialog")

        self._fcs_path = Path(fcs_path) if fcs_path else None
        self._existing = existing_specs or {}
        self._channels: List[str] = []
        self._row_method: Dict[str, QComboBox] = {}
        self._row_cofactor: Dict[str, QDoubleSpinBox] = {}

        self._build_ui()
        if self._fcs_path is not None:
            self._load_channels(self._fcs_path)

    # ── Construction UI ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("Configuration du pré-traitement")
        title.setStyleSheet("font-size: 15px; font-weight: 700; color: #e2e8f0;")
        root.addWidget(title)

        hint = QLabel(
            "Choisir la transformation par canal. Logicle (biexponentiel) "
            "recommandé pour les fluorochromes ; arcsinh avec cofacteur pour "
            "spectral ; FSC/SSC/Time laissés bruts (none) par défaut."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #94a3b8; font-size: 11px;")
        root.addWidget(hint)

        # ── Paramètres logicle globaux ────────────────────────────────────────
        logicle_box = QFrame()
        logicle_box.setObjectName("paramCard")
        gl = QGridLayout(logicle_box)
        gl.setContentsMargins(12, 10, 12, 10)
        gl.setHorizontalSpacing(14)
        gl.addWidget(QLabel("Paramètres logicle globaux :"), 0, 0, 1, 8)

        self._spin_T = self._mk_spin(1.0, 1e7, 262144.0, 0)
        self._spin_M = self._mk_spin(0.5, 10.0, 4.5, 2)
        self._spin_W = self._mk_spin(0.0, 5.0, 0.5, 2)
        self._spin_A = self._mk_spin(-2.0, 5.0, 0.0, 2)
        for col, (lbl, sp) in enumerate(
            [("T", self._spin_T), ("M", self._spin_M), ("W", self._spin_W), ("A", self._spin_A)]
        ):
            gl.addWidget(QLabel(lbl), 1, col * 2)
            gl.addWidget(sp, 1, col * 2 + 1)
        root.addWidget(logicle_box)

        # ── Actions groupées ──────────────────────────────────────────────────
        bulk = QHBoxLayout()
        bulk.setSpacing(8)
        bulk.addWidget(QLabel("Appliquer à tous les fluorochromes :"))
        for m in ("logicle", "arcsinh", "log10", "none"):
            b = QPushButton(m)
            b.setFixedWidth(80)
            b.clicked.connect(lambda _=False, mm=m: self._apply_bulk_fluo(mm))
            bulk.addWidget(b)
        bulk.addStretch()
        root.addLayout(bulk)

        # ── Table des canaux ──────────────────────────────────────────────────
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Canal", "Transformation", "Cofacteur (arcsinh)"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self._table.setColumnWidth(1, 150)
        self._table.setColumnWidth(2, 160)
        self._table.verticalHeader().setVisible(False)
        root.addWidget(self._table, 1)

        # ── Boutons OK/Annuler ────────────────────────────────────────────────
        btns = QHBoxLayout()
        btns.addStretch()
        cancel = QPushButton("Annuler")
        cancel.clicked.connect(self.reject)
        btns.addWidget(cancel)
        ok = QPushButton("Valider")
        ok.setObjectName("primaryBtn")
        ok.clicked.connect(self.accept)
        btns.addWidget(ok)
        root.addLayout(btns)

    def _mk_spin(self, lo: float, hi: float, val: float, dec: int) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(dec)
        sp.setValue(val)
        sp.setFixedWidth(110)
        return sp

    # ── Chargement des canaux ─────────────────────────────────────────────────

    def _load_channels(self, fcs_path: Path) -> None:
        self._channels = read_fcs_channels(fcs_path)
        self._table.setRowCount(len(self._channels))
        self._row_method.clear()
        self._row_cofactor.clear()

        for r, name in enumerate(self._channels):
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(r, 0, item)

            combo = QComboBox()
            combo.addItems(_METHODS)
            # Défaut : logicle si fluo, none si scatter — sauf spec existante.
            prev = self._existing.get(name, {}).get("method")
            default = prev or ("none" if _is_scatter(name) else _DEFAULT_FLUO_METHOD)
            combo.setCurrentText(default if default in _METHODS else "none")
            self._table.setCellWidget(r, 1, combo)
            self._row_method[name] = combo

            cof = QDoubleSpinBox()
            cof.setRange(0.1, 100000.0)
            cof.setDecimals(1)
            cof.setValue(float(self._existing.get(name, {}).get("cofactor", 5.0)))
            self._table.setCellWidget(r, 2, cof)
            self._row_cofactor[name] = cof

    def set_fcs_path(self, fcs_path: Path | str) -> None:
        """Recharge la liste des canaux depuis un nouveau FCS."""
        self._fcs_path = Path(fcs_path)
        self._load_channels(self._fcs_path)

    def _apply_bulk_fluo(self, method: str) -> None:
        """Applique `method` à tous les canaux NON scatter."""
        for name, combo in self._row_method.items():
            if not _is_scatter(name):
                combo.setCurrentText(method)

    # ── Récupération des specs ────────────────────────────────────────────────

    def get_specs(self) -> Dict[str, dict]:
        """Retourne les specs par colonne (canaux avec method != none uniquement)."""
        T = float(self._spin_T.value())
        M = float(self._spin_M.value())
        W = float(self._spin_W.value())
        A = float(self._spin_A.value())
        specs: Dict[str, dict] = {}
        for name in self._channels:
            method = self._row_method[name].currentText()
            if method == "none":
                continue
            spec = {"method": method}
            if method == "arcsinh":
                spec["cofactor"] = float(self._row_cofactor[name].value())
            elif method == "logicle":
                spec.update({"T": T, "M": M, "W": W, "A": A})
            specs[name] = spec
        return specs
