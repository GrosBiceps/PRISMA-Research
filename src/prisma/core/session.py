"""
session.py — SessionManager : état global de l'application PRISMA Research.

Singleton thread-safe centralisant l'état courant (expérience active, runs,
samples chargés). La GUI ne touche les données que via ce manager et les
signals Qt.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

from prisma.core.models import Experiment, RunMetadata, Sample

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Singleton thread-safe gérant l'état courant de l'application.

    Usage:
        session = SessionManager.instance()
        session.set_experiment(exp)
    """

    _instance: Optional["SessionManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "SessionManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    @classmethod
    def instance(cls) -> "SessionManager":
        return cls()

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._experiment: Optional[Experiment] = None
        self._active_sample: Optional[Sample] = None
        self._run_history: List[RunMetadata] = []
        self._output_root: Path = Path("outputs")
        self._callbacks: Dict[str, list] = {}

        logger.info("SessionManager initialized")

    # ------------------------------------------------------------------
    # Experiment
    # ------------------------------------------------------------------

    @property
    def experiment(self) -> Optional[Experiment]:
        return self._experiment

    def set_experiment(self, exp: Experiment) -> None:
        self._experiment = exp
        logger.info("Active experiment set: %s (%d samples)", exp.name, exp.n_samples)
        self._emit("experiment_changed", exp)

    def clear_experiment(self) -> None:
        self._experiment = None
        self._emit("experiment_changed", None)

    # ------------------------------------------------------------------
    # Sample actif
    # ------------------------------------------------------------------

    @property
    def active_sample(self) -> Optional[Sample]:
        return self._active_sample

    def set_active_sample(self, sample: Optional[Sample]) -> None:
        self._active_sample = sample
        name = sample.name if sample else "None"
        logger.debug("Active sample: %s", name)
        self._emit("sample_changed", sample)

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def register_run(self, meta: RunMetadata) -> None:
        self._run_history.append(meta)
        logger.info("Run registered: %s [%s]", meta.run_id, meta.pipeline_name)

    @property
    def run_history(self) -> List[RunMetadata]:
        return list(self._run_history)

    def last_run(self) -> Optional[RunMetadata]:
        return self._run_history[-1] if self._run_history else None

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @property
    def output_root(self) -> Path:
        return self._output_root

    def set_output_root(self, path: Path) -> None:
        self._output_root = path
        path.mkdir(parents=True, exist_ok=True)

    def run_output_dir(self, run_id: str) -> Path:
        d = self._output_root / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Callbacks légers (sans Qt — utilisés en mode headless)
    # ------------------------------------------------------------------

    def on(self, event: str, callback) -> None:
        self._callbacks.setdefault(event, []).append(callback)

    def _emit(self, event: str, payload) -> None:
        for cb in self._callbacks.get(event, []):
            try:
                cb(payload)
            except Exception:
                logger.exception("Callback error on event '%s'", event)
