"""
logger.py — Logging structuré des événements de gating.

Fournit un enregistrement JSON de chaque étape de gating pour traçabilité
clinique (ELN recommande un audit trail complet de chaque décision de gating).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_stream_handler() -> logging.Handler:
    """
    Retourne un StreamHandler toujours valide, y compris en mode
    PyInstaller --noconsole (sys.stderr/sys.stdout peuvent être None).
    En mode frozen console=False les deux streams sont None/devnull — on
    retourne un NullHandler pour ne pas polluer le root logger avec un
    StreamHandler mort qui bloquerait la propagation vers _QtLogHandler.
    """
    stream = sys.stderr if sys.stderr is not None else sys.stdout
    if stream is None:
        return logging.NullHandler()
    # En mode frozen, os.devnull ouvert en écriture → NullHandler aussi
    try:
        name = getattr(stream, "name", "")
        if name in (os.devnull, "/dev/null", "nul"):
            return logging.NullHandler()
    except Exception:
        pass
    return logging.StreamHandler(stream=stream)


# Configurer un logger Python standard pour les messages console/fichier,
# de manière robuste en environnement --noconsole.
# force=True garantit que basicConfig s'applique même si une lib tierce
# a déjà appelé basicConfig avant (ex: flowsom, scanpy au premier import).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[_safe_stream_handler()],
    force=True,
)

# Silencer les loggers DEBUG verbeux des bibliothèques tierces
for _noisy_logger in ("numba", "numba.core", "pynndescent", "umap"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str = "flowsom_pipeline") -> logging.Logger:
    """Retourne un logger nommé pour le module appelant."""
    return logging.getLogger(name)


@dataclass
class GatingEvent:
    """
    Enregistrement d'un événement de gating individuel.

    Attributes:
        file: Nom du fichier FCS concerné.
        gate_name: Nom de la gate (ex: 'G1_viable', 'G2_singlets').
        n_before: Nombre de cellules avant gating.
        n_after: Nombre de cellules après gating.
        pct_kept: Pourcentage conservé.
        timestamp: Horodatage Unix.
        warnings: Avertissements cliniques éventuels.
        extra: Données supplémentaires (seuils, méthode, etc.).
    """

    file: str
    gate_name: str
    n_before: int
    n_after: int
    pct_kept: float
    timestamp: float = field(default_factory=time.time)
    warnings: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "gate_name": self.gate_name,
            "n_before": self.n_before,
            "n_after": self.n_after,
            "n_excluded": self.n_before - self.n_after,
            "pct_kept": round(self.pct_kept, 4),
            "timestamp": self.timestamp,
            "warnings": self.warnings,
            **self.extra,
        }


class GatingLogger:
    """
    Collecteur d'événements de gating avec export JSON.

    Usage:
        logger = GatingLogger()
        logger.log("patient.fcs", "G1_viable", 500000, 420000)
        logger.save("gating_log.json")
    """

    def __init__(self) -> None:
        self._events: List[GatingEvent] = []
        self._logger = get_logger("gating")

    def log(
        self,
        file: str,
        gate_name: str,
        n_before: int,
        n_after: int,
        warnings: Optional[List[str]] = None,
        **extra: Any,
    ) -> GatingEvent:
        """
        Enregistre un événement de gating.

        Args:
            file: Nom du fichier FCS.
            gate_name: Nom de la gate.
            n_before: Cellules avant la gate.
            n_after: Cellules après la gate.
            warnings: Avertissements cliniques.
            **extra: Données supplémentaires (seuils, méthode...).

        Returns:
            L'événement créé.
        """
        pct = (n_after / n_before * 100) if n_before > 0 else 0.0
        event = GatingEvent(
            file=file,
            gate_name=gate_name,
            n_before=n_before,
            n_after=n_after,
            pct_kept=pct,
            warnings=warnings or [],
            extra=extra,
        )
        self._events.append(event)
        self._logger.info(
            "%s | %s: %d → %d cellules (%.1f%% conservées)",
            file,
            gate_name,
            n_before,
            n_after,
            pct,
        )
        if warnings:
            for w in warnings:
                self._logger.warning("%s | %s: %s", file, gate_name, w)
        return event

    @property
    def events(self) -> List[GatingEvent]:
        return list(self._events)

    def events_for_file(self, file: str) -> List[GatingEvent]:
        """Retourne les événements pour un fichier donné."""
        return [e for e in self._events if e.file == file]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": "1.0",
            "n_events": len(self._events),
            "events": [e.to_dict() for e in self._events],
        }

    def save(self, output_path: Path | str) -> None:
        """
        Sauvegarde le log de gating en JSON.

        Args:
            output_path: Chemin du fichier JSON de sortie.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        self._logger.info("Log de gating sauvegardé: %s", output_path)

    @classmethod
    def load(cls, path: Path | str) -> "GatingLogger":
        """
        Charge un log de gating depuis un fichier JSON.

        Args:
            path: Chemin vers le fichier JSON.

        Returns:
            Instance GatingLogger reconstruite.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        instance = cls()
        for event_data in data.get("events", []):
            extra = {
                k: v
                for k, v in event_data.items()
                if k
                not in {
                    "file",
                    "gate_name",
                    "n_before",
                    "n_after",
                    "n_excluded",
                    "pct_kept",
                    "timestamp",
                    "warnings",
                }
            }
            event = GatingEvent(
                file=event_data["file"],
                gate_name=event_data["gate_name"],
                n_before=event_data["n_before"],
                n_after=event_data["n_after"],
                pct_kept=event_data["pct_kept"],
                timestamp=event_data.get("timestamp", 0.0),
                warnings=event_data.get("warnings", []),
                extra=extra,
            )
            instance._events.append(event)
        return instance

    def summary(self) -> str:
        """Retourne un résumé lisible du log de gating."""
        lines = [f"=== GATING AUDIT LOG ({len(self._events)} événements) ==="]
        files_seen = []
        for e in self._events:
            if e.file not in files_seen:
                files_seen.append(e.file)
                lines.append(f"\nFichier: {e.file}")
            lines.append(
                f"  {e.gate_name:<20} {e.n_before:>8} → {e.n_after:>8} ({e.pct_kept:5.1f}%)"
            )
            for w in e.warnings:
                lines.append(f"    ⚠️  {w}")
        return "\n".join(lines)
