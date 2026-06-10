# -*- coding: utf-8 -*-
"""
log_console.py — Terminal de logs avec coloration syntaxique.

LogConsole est un QPlainTextEdit enrichi avec :
  - Police monospace (Cascadia Code / Fira Code / Consolas)
  - Fond #0a0a14 (terminal sombre)
  - Coloration syntaxique : [INFO] vert, [WARNING] orange, [ERROR] rouge,
    [SUCCESS] cyan, timestamps en gris
  - Méthode append_log(msg) qui parse et colore automatiquement

Design System "PRISMA v2" :
    - [INFO]    → #39FF8A (FITC / accent)
    - [WARNING] → #FF9B3D (PE / warning)
    - [ERROR]   → #FF3D6E (APC / danger)
    - [SUCCESS] → #5BAAFF (V500 / info)
    - Timestamp → rgba(238,242,247,0.28)
"""

from __future__ import annotations

import re
from typing import Optional

from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QTextCharFormat,
    QTextCursor,
)
from PyQt5.QtWidgets import QPlainTextEdit, QWidget

# ── Patterns de coloration ────────────────────────────────────────────

_LOG_RULES: list[tuple[re.Pattern, str]] = [
    # Timestamps ISO : 2024-01-15 12:34:56
    (re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"), "#647088"),
    # Horodatage court : [12:34:56]
    (re.compile(r"\[\d{2}:\d{2}:\d{2}\]"), "#647088"),
    # Niveaux de log
    (re.compile(r"\[INFO\]|\[info\]"), "#39FF8A"),
    (re.compile(r"\bINFO\b"), "#39FF8A"),
    (re.compile(r"\[SUCCESS\]|\[success\]"), "#5BAAFF"),
    (re.compile(r"\bSUCCESS\b"), "#5BAAFF"),
    (re.compile(r"\[WARNING\]|\[warning\]|\[WARN\]|\[warn\]"), "#FF9B3D"),
    (re.compile(r"\bWARNING\b|\bWARN\b"), "#FF9B3D"),
    (re.compile(r"\[ERROR\]|\[error\]|\[ERR\]|\[err\]"), "#FF3D6E"),
    (re.compile(r"\bERROR\b|\bERR\b|\bCRITICAL\b"), "#FF3D6E"),
    (re.compile(r"\[DEBUG\]|\[debug\]"), "#8A95AD"),
    (re.compile(r"\bDEBUG\b"), "#8A95AD"),
    # Marqueurs pipeline / gating importants
    (re.compile(r"\bG[1-4]_[a-zA-Z0-9_]+\b|\bGate\s+[1-4]\b"), "#5BAAFF"),
    (re.compile(r"\b(MRD|KDE|RANSAC|GMM|UMAP|FlowSOM)\b", re.IGNORECASE), "#7B52FF"),
    (
        re.compile(
            r"\b(Marqueurs\s+retenus\s+dans\s+le\s+panel\s+commun|Marqueurs\s+pour\s+FlowSOM)\b",
            re.IGNORECASE,
        ),
        "#FFE032",
    ),
    (re.compile(r"\bD[ée]s[ée]quilibre\s+Ma[îi]tris[ée]\s+activ[ée]\b", re.IGNORECASE), "#FF9B3D"),
    (re.compile(r"\bfallback\b|\bfallbacks\b|\béchou[ée]\b", re.IGNORECASE), "#FF9B3D"),
    (re.compile(r"\bPipeline\b|\bÉtape\b", re.IGNORECASE), "#7B52FF"),
    # Valeurs numériques importantes (pourcentages, MRD)
    (re.compile(r"\b\d+\.?\d*\s*%"), "#7B52FF"),
    (re.compile(r"\bratio\s*=\s*\d+(?:\.\d+)?×\b", re.IGNORECASE), "#FF3D6E"),
    (re.compile(r"\bseed\s*=\s*\d+\b", re.IGNORECASE), "#5BAAFF"),
    (re.compile(r"\([^)]*\)"), "#AAB6CB"),
    (re.compile(r"\b\d[\d,]*\s*/\s*\d[\d,]*\b"), "#FFE032"),
    # Chemins de fichiers
    (re.compile(r"[A-Za-z]:\\[^\s]+|/[^\s]+\.[a-z]+"), "#FFE032"),
    # Flèches / transitions de comptage
    (re.compile(r"→|->|=>|\|"), "#8491AA"),
    # Mots-clés état
    (re.compile(r"\b(terminé|sauvegardé|chargé|exporté|conservées?)\b", re.IGNORECASE), "#39FF8A"),
    (re.compile(r"\b(absent|ignoré|déséquilibre|warning|erreur)\b", re.IGNORECASE), "#FF9B3D"),
    # Étapes pipelines (ex: "Étape 1/5" ou "Step 1 of 5")
    (re.compile(r"[ÉEé]tape\s+\d+\s*/\s*\d+|Step\s+\d+\s+of\s+\d+"), "#5BAAFF"),
]

_MAJOR_LINE_PATTERNS: list[tuple[re.Pattern, str, int]] = [
    # Blocs de séparation visuels
    (re.compile(r"^\s*[=]{20,}\s*$"), "#5BAAFF", 63),
    (re.compile(r"^\s*[═]{10,}\s*$"), "#5BAAFF", 63),
    # Démarrage / fin du pipeline
    (re.compile(r"\bPIPELINE\b.*\bD[ÉE]MARRAGE\b", re.IGNORECASE), "#7B52FF", 75),
    (re.compile(r"\bD[ÉE]MARRAGE\s+DU\s+PIPELINE\b", re.IGNORECASE), "#7B52FF", 75),
    (re.compile(r"\bPIPELINE\b.*\bTERMIN[ÉE]\b", re.IGNORECASE), "#39FF8A", 75),
    # Bloc d'analyse important
    (re.compile(r"\bANALYSE\s+DES\s+CLUSTERS\s+EXCLUSIFS\b", re.IGNORECASE), "#FFE032", 75),
    # Étapes du pipeline: colorer toute la ligne, pas seulement le token "Étape"
    (re.compile(r"\b[ÉEé]tape\s+\d+\b.*", re.IGNORECASE), "#5BAAFF", 75),
    (re.compile(r"\bStep\s+\d+\b.*", re.IGNORECASE), "#5BAAFF", 75),
    # KPI finaux de synthèse
    (re.compile(r"\bCellules\s*:", re.IGNORECASE), "#5BAAFF", 75),
    (re.compile(r"\bMarqueurs\s*:", re.IGNORECASE), "#7B52FF", 75),
    (re.compile(r"\bM[ée]taclusters\s*:", re.IGNORECASE), "#FF3D6E", 75),
    # Lignes stratégiques de sélection des marqueurs / balance
    (
        re.compile(r"\bMarqueurs\s+retenus\s+dans\s+le\s+panel\s+commun\b", re.IGNORECASE),
        "#FFE032",
        75,
    ),
    (re.compile(r"\bMarqueurs\s+pour\s+FlowSOM\b", re.IGNORECASE), "#7B52FF", 75),
    (
        re.compile(r"\bD[ée]s[ée]quilibre\s+Ma[îi]tris[ée]\s+activ[ée]\b", re.IGNORECASE),
        "#FF9B3D",
        75,
    ),
]


class LogConsole(QPlainTextEdit):
    """
    Terminal de logs avec coloration syntaxique intégrée.

    Usage :
        console = LogConsole()
        console.append_log("[INFO] Pipeline démarré.")
        console.append_log("[ERROR] Fichier introuvable : data.fcs")
    """

    # Nombre max de blocs (lignes) pour éviter une croissance infinie
    MAX_BLOCKS = 5000

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.setObjectName("logConsole")
        self.setReadOnly(True)
        self.setMaximumBlockCount(self.MAX_BLOCKS)
        self.setPlaceholderText("Les logs du pipeline apparaîtront ici…")

        # Police monospace
        font = QFont()
        font.setFamilies(["Cascadia Code", "Fira Code", "Consolas", "Courier New"])
        font.setPointSize(9)
        font.setStyleHint(QFont.Monospace)
        self.setFont(font)

        # Style visuel principal défini dans gui/styles.py

        # Format de texte par défaut (texte brut)
        self._default_format = QTextCharFormat()
        self._default_format.setForeground(QBrush(QColor("#EEF2F7")))

        # Cache de QTextCharFormat par couleur hex — évite des centaines
        # d'allocations Python par tick lors de logs denses (50 msg × N spans).
        self._fmt_cache: dict[str, QTextCharFormat] = {}

        # Nettoyage visuel des préfixes doublés: "[hh:mm:ss] INFO [hh:mm:ss] INFO ..."
        self._dup_prefix_re = re.compile(
            r"^(\[\d{2}:\d{2}:\d{2}\]\s+(?:INFO|WARNING|WARN|ERROR|ERR|DEBUG|SUCCESS)\s+)"
            r"\1+"
        )
        self._ansi_re = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

    # ── API publique ──────────────────────────────────────────────────

    def append_log(self, message: str) -> None:
        """
        Ajoute un message de log (ligne unique ou bloc multiligne) avec
        coloration syntaxique automatique.

        Pour les blocs volumineux, le repaint est suspendu temporairement
        afin d'éviter un redessin à chaque ligne.
        """
        if message is None:
            return

        text = str(message).replace("\r\n", "\n").replace("\r", "\n")
        lines = text.split("\n")

        if not lines:
            return

        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)

        self.setUpdatesEnabled(False)
        try:
            for idx, raw_line in enumerate(lines):
                # Saut de ligne entre lignes, et avant la première si document non vide
                if not self.document().isEmpty() or idx > 0:
                    cursor.insertText("\n", self._default_format)

                line = self._normalize_line(raw_line)
                self._insert_colored(cursor, line)

            # Auto-scroll vers le bas
            self.setTextCursor(cursor)
            self.ensureCursorVisible()
        finally:
            self.setUpdatesEnabled(True)
            self.viewport().update()

    def append(self, text: str) -> None:
        """
        Surcharge de QPlainTextEdit.appendPlainText pour compatibilité avec
        l'API QTextEdit.append() utilisée dans main_window.py.
        Redirige vers append_log() pour appliquer la coloration syntaxique.
        """
        # Retirer les balises HTML simples si le texte vient d'un insertHtml
        clean = re.sub(r"<[^>]+>", "", text)
        self.append_log(clean)

    def clear_logs(self) -> None:
        """Efface tous les logs."""
        self.clear()

    # ── Coloration syntaxique ─────────────────────────────────────────

    def _get_fmt(self, color: str) -> QTextCharFormat:
        """Retourne un QTextCharFormat mis en cache par couleur hex."""
        fmt = self._fmt_cache.get(color)
        if fmt is None:
            fmt = QTextCharFormat()
            fmt.setForeground(QBrush(QColor(color)))
            fmt.setFont(self.font())
            self._fmt_cache[color] = fmt
        return fmt

    def _insert_colored(self, cursor: QTextCursor, line: str) -> None:
        """
        Insère une ligne de log en colorant les tokens correspondant aux règles.
        Algorithme : on calcule tous les spans qui matchent, on les trie,
        on insère segment par segment.
        """
        if not line:
            cursor.insertText("", self._default_format)
            return

        # Collecter tous les spans (start, end, color)
        spans: list[tuple[int, int, str]] = []
        for pattern, color in _LOG_RULES:
            for m in pattern.finditer(line):
                spans.append((m.start(), m.end(), color))

        if not spans:
            # Ligne sans token spécial : couleur de base selon niveau
            fmt = self._format_for_line(line)
            cursor.insertText(line, fmt)
            return

        # Trier et dédupliquer (priorité au premier match en cas de chevauchement)
        # A position égale, donner priorité au motif le plus long
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        merged: list[tuple[int, int, str]] = []
        last_end = 0
        for start, end, color in spans:
            if start >= last_end:
                merged.append((start, end, color))
                last_end = end

        # Couleur de base de ligne (niveau log + sections majeures)
        base_fmt = self._format_for_line(line)
        major_fmt = self._major_line_format(line)
        if major_fmt is not None:
            base_fmt = major_fmt

        pos = 0
        for start, end, color in merged:
            # Segment avant le match
            if pos < start:
                cursor.insertText(line[pos:start], base_fmt)
            # Segment coloré — format mis en cache pour éviter les allocations répétées
            cursor.insertText(line[start:end], self._get_fmt(color))
            pos = end

        # Reste après le dernier match
        if pos < len(line):
            cursor.insertText(line[pos:], base_fmt)

    def _format_for_line(self, line: str) -> QTextCharFormat:
        """Détermine la couleur de base d'une ligne selon son niveau de log."""
        fmt = QTextCharFormat()
        fmt.setFont(self.font())
        line_up = line.upper()
        if (
            "[ERROR]" in line_up
            or "[ERR]" in line_up
            or " ERROR " in f" {line_up} "
            or " ERR " in f" {line_up} "
            or " CRITICAL " in f" {line_up} "
        ):
            fmt.setForeground(QBrush(QColor("#FF3D6E")))
        elif (
            "[WARNING]" in line_up
            or "[WARN]" in line_up
            or " WARNING " in f" {line_up} "
            or " WARN " in f" {line_up} "
            or " FALLBACK" in line_up
        ):
            fmt.setForeground(QBrush(QColor("#FF9B3D")))
        elif "[SUCCESS]" in line_up or " SUCCESS " in f" {line_up} ":
            fmt.setForeground(QBrush(QColor("#5BAAFF")))
        elif "[DEBUG]" in line_up or " DEBUG " in f" {line_up} ":
            fmt.setForeground(QBrush(QColor("#8A95AD")))
        elif " INFO " in f" {line_up} " or "[INFO]" in line_up:
            fmt.setForeground(QBrush(QColor("#39FF8A")))
        else:
            fmt.setForeground(QBrush(QColor("#EEF2F7")))
        return fmt

    def _major_line_format(self, line: str) -> Optional[QTextCharFormat]:
        """Retourne un format renforcé pour les lignes structurantes du pipeline."""
        for pattern, color, weight in _MAJOR_LINE_PATTERNS:
            if pattern.search(line):
                fmt = QTextCharFormat()
                fmt.setFont(self.font())
                fmt.setForeground(QBrush(QColor(color)))
                fmt.setFontWeight(weight)
                return fmt
        return None

    def _normalize_line(self, line: str) -> str:
        """Nettoie une ligne brute pour éviter le bruit visuel dans la console."""
        if not line:
            return ""
        cleaned = self._ansi_re.sub("", line)
        # Collapse des préfixes log dupliqués en tête de ligne.
        while True:
            collapsed = self._dup_prefix_re.sub(r"\1", cleaned)
            if collapsed == cleaned:
                break
            cleaned = collapsed
        return cleaned
