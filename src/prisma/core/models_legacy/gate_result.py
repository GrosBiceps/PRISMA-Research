"""
gate_result.py — Structure de données pour les résultats de gating.

GateResult est le type de retour standard de chaque opération de gating.
Il est conçu pour être sérialisable en JSON (sans le masque numpy) afin
de permettre un audit complet des décisions de gating.

État global legacy
------------------
``gating_reports`` et ``gating_log_entries`` sont des listes module-level
conservées pour la rétrocompatibilité avec le pipeline legacy. Tout nouveau
code doit utiliser ``GatingContext`` (contexte de run isolé, thread-safe).

ARCH-5 TODO : lorsque pipeline_executor_legacy sera migré vers PRISMA,
supprimer les deux listes globales et ``log_gating_event``.
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

import numpy as np

_logger = logging.getLogger("models.gate_result")


@dataclass
class GateResult:
    """
    Résultat structuré d'une opération de gating.

    Attributes:
        mask: Masque booléen numpy (True = cellule conservée).
        n_kept: Nombre de cellules conservées.
        n_total: Nombre total de cellules avant ce gate.
        method: Méthode utilisée ('auto_gmm_debris', 'ransac_singlets', …).
        gate_name: Identifiant du gate (ex: 'G1_debris', 'G3_cd45').
        details: Paramètres et métriques de la méthode.
        warnings: Messages d'avertissement levés pendant le gate.
    """

    mask: np.ndarray
    n_kept: int
    n_total: int
    method: str
    gate_name: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Propriétés calculées
    # ------------------------------------------------------------------

    @property
    def pct_kept(self) -> float:
        """Pourcentage de cellules conservées (0–100)."""
        return (self.n_kept / max(self.n_total, 1)) * 100

    @property
    def n_excluded(self) -> int:
        """Nombre de cellules exclues par ce gate."""
        return self.n_total - self.n_kept

    @property
    def pct_excluded(self) -> float:
        """Pourcentage de cellules exclues (0–100)."""
        return 100.0 - self.pct_kept

    @property
    def is_good_quality(self) -> bool:
        """
        Qualité basique du gate : True si > 20% des cellules ont été conservées.
        Un gate qui élimine > 80% des cellules mérite une vérification manuelle.
        """
        return self.pct_kept > 20.0

    # ------------------------------------------------------------------
    # Sérialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Sérialisation JSON-safe (sans le masque numpy).

        Returns:
            Dictionnaire sérialisable.
        """
        return {
            "gate_name": self.gate_name,
            "method": self.method,
            "n_kept": self.n_kept,
            "n_total": self.n_total,
            "n_excluded": self.n_excluded,
            "pct_kept": round(self.pct_kept, 2),
            "pct_excluded": round(self.pct_excluded, 2),
            "is_good_quality": self.is_good_quality,
            "details": self.details,
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        """Sérialise en JSON formaté."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def __repr__(self) -> str:
        return (
            f"GateResult({self.gate_name!r}, method={self.method!r}, "
            f"kept={self.n_kept}/{self.n_total} ({self.pct_kept:.1f}%))"
        )


# =============================================================================
# GatingContext — contexte de run isolé, thread-safe (nouvelle API)
# =============================================================================

class GatingContext:
    """
    Conteneur de run pour les résultats et logs de gating.

    Remplace les listes globales ``gating_reports`` / ``gating_log_entries``
    par un objet à durée de vie explicite, passé par injection de dépendance.
    Thread-safe : chaque run batch instancie son propre contexte.

    Usage::

        ctx = GatingContext()
        ctx.add_report(gate_result)
        ctx.log_event("G1_debris", "auto_gmm", "success", details={...})
        reports = ctx.reports          # lecture immutable
        entries = ctx.log_entries      # lecture immutable

    Ou via gestionnaire de contexte::

        with GatingContext.for_run() as ctx:
            ...
    """

    def __init__(self) -> None:
        self._reports: List[GateResult] = []
        self._log_entries: List[Dict[str, Any]] = []
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Écriture
    # ------------------------------------------------------------------

    def add_report(self, result: GateResult) -> None:
        with self._lock:
            self._reports.append(result)

    def log_event(
        self,
        gate_name: str,
        method: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        warning_msg: Optional[str] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "gate_name": gate_name,
            "method": method,
            "status": status,
            "details": details or {},
        }
        if warning_msg:
            entry["warning"] = warning_msg
            _logger.warning("%s: %s", gate_name, warning_msg)
        with self._lock:
            self._log_entries.append(entry)

    # ------------------------------------------------------------------
    # Lecture (copies défensives)
    # ------------------------------------------------------------------

    @property
    def reports(self) -> List[GateResult]:
        with self._lock:
            return list(self._reports)

    @property
    def log_entries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._log_entries)

    def clear(self) -> None:
        with self._lock:
            self._reports.clear()
            self._log_entries.clear()

    # ------------------------------------------------------------------
    # Gestionnaire de contexte
    # ------------------------------------------------------------------

    @classmethod
    @contextmanager
    def for_run(cls) -> Generator["GatingContext", None, None]:
        """Crée un contexte de gating pour un run, nettoyé à la sortie."""
        ctx = cls()
        try:
            yield ctx
        finally:
            ctx.clear()


# =============================================================================
# État global legacy — rétrocompatibilité uniquement
# =============================================================================
# ARCH-5 TODO : supprimer ces deux listes une fois pipeline_executor_legacy
# migré vers PRISMA. Tout nouveau code doit utiliser GatingContext.

gating_reports: List[GateResult] = []
gating_log_entries: List[Dict[str, Any]] = []


def log_gating_event(
    gate_name: str,
    method: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
    warning_msg: Optional[str] = None,
) -> None:
    """
    Log structuré d'un événement de gating (JSON exportable).

    .. deprecated::
        Utiliser ``GatingContext.log_event()`` dans tout nouveau code.
        Cette fonction écrit dans la liste globale ``gating_log_entries``
        (non thread-safe — legacy uniquement).
    """
    entry: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "gate_name": gate_name,
        "method": method,
        "status": status,
        "details": details or {},
    }
    if warning_msg:
        entry["warning"] = warning_msg
        _logger.warning("%s: %s", gate_name, warning_msg)
    gating_log_entries.append(entry)
