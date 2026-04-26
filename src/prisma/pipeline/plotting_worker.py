"""plotting_worker.py — Worker producer-consumer pour le plotting asynchrone.

Le producteur (pipeline principal) enfile des tâches de visualisation
(type de graphe + données + chemin de sortie). Le consommateur (thread)
dépile et exécute les fonctions de plotting en arrière-plan.
"""

from __future__ import annotations

import importlib
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger

_logger = get_logger("pipeline.plotting_worker")


@dataclass
class PlottingTask:
    """Tâche de plotting asynchrone."""

    name: str
    key: str
    target: str  # "mpl" | "plotly"
    module: str
    function: str
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)


class PlottingWorker:
    """Consommateur de tâches de plotting basé sur un thread standard."""

    def __init__(self, max_queue_size: int = 32) -> None:
        self._queue: "queue.Queue[Optional[PlottingTask]]" = queue.Queue(
            maxsize=max_queue_size
        )
        self._thread = threading.Thread(
            target=self._run,
            name="flowsom-plotting-worker",
            daemon=True,
        )
        self._started = False
        self._lock = threading.Lock()
        self._results: Dict[str, Dict[str, Any]] = {"mpl": {}, "plotly": {}}
        self._errors: List[str] = []

    def start(self) -> None:
        if self._started:
            return
        self._thread.start()
        self._started = True
        _logger.info("PlottingWorker démarré.")

    def submit(self, task: PlottingTask) -> None:
        if not self._started:
            self.start()
        self._queue.put(task)

    def close_and_wait(self) -> None:
        if not self._started:
            return
        self._queue.put(None)
        self._thread.join()
        self._started = False
        _logger.info("PlottingWorker arrêté.")

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                "mpl": dict(self._results.get("mpl", {})),
                "plotly": dict(self._results.get("plotly", {})),
            }

    def get_errors(self) -> List[str]:
        with self._lock:
            return list(self._errors)

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return

                result = self._execute(task)
                if result is not None:
                    with self._lock:
                        target_dict = self._results.setdefault(task.target, {})
                        target_dict[task.key] = result
                _logger.info("%s terminé.", task.name)
            except Exception as exc:
                msg = f"{task.name} échoué (non bloquant): {exc}"
                _logger.warning(msg)
                with self._lock:
                    self._errors.append(msg)
            finally:
                self._queue.task_done()

    @staticmethod
    def _execute(task: PlottingTask) -> Any:
        module = importlib.import_module(task.module)
        fn = getattr(module, task.function)
        return fn(*task.args, **task.kwargs)
