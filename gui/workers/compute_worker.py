"""
compute_worker.py — QRunnable non-bloquant pour tous les calculs lourds.

Déporte 100% des calculs hors du Main Thread UI :
    - Lecture HDF5 / chargement FCS
    - Rasterisation Datashader
    - UMAP / t-SNE / FlowSOM
    - Gating (masques booléens)

Usage :
    worker = ComputeWorker(fn, arg1, arg2, kwarg=val)
    worker.signals.finished.connect(self._on_result)
    worker.signals.error.connect(self._on_error)
    worker.signals.progress.connect(self.progress_bar.setValue)
    QThreadPool.globalInstance().start(worker)
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot


class WorkerSignals(QObject):
    """Signaux thread-safe pour communiquer avec le Main Thread."""
    finished = pyqtSignal(object)   # résultat (any)
    error    = pyqtSignal(str)      # traceback complet
    progress = pyqtSignal(int)      # 0-100
    status   = pyqtSignal(str)      # message texte libre


class ComputeWorker(QRunnable):
    """
    Worker générique QRunnable.

    Le callable peut accepter un kwarg optionnel `progress_callback`
    (Callable[[int], None]) pour émettre des signaux de progression.
    """

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

        # Injecte le callback de progression si le callable le supporte
        import inspect
        sig = inspect.signature(fn)
        if "progress_callback" in sig.parameters:
            self.kwargs["progress_callback"] = self.signals.progress.emit

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Helpers pratiques
# ---------------------------------------------------------------------------

def run_async(
    fn: Callable[..., Any],
    *args: Any,
    on_done: Callable[[Any], None] | None = None,
    on_error: Callable[[str], None] | None = None,
    on_progress: Callable[[int], None] | None = None,
    pool: QThreadPool | None = None,
    **kwargs: Any,
) -> ComputeWorker:
    """
    Lance fn(*args, **kwargs) dans le thread pool global.

    Returns:
        Worker déjà soumis au thread pool.
    """
    worker = ComputeWorker(fn, *args, **kwargs)
    if on_done:
        worker.signals.finished.connect(on_done)
    if on_error:
        worker.signals.error.connect(on_error)
    if on_progress:
        worker.signals.progress.connect(on_progress)

    target_pool = pool or QThreadPool.globalInstance()
    target_pool.start(worker)
    return worker
