# -*- coding: utf-8 -*-
"""Package workers — QThread pipeline + QRunnable générique.

Réexporte les workers historiques (_pipeline.py) et le worker générique
(compute_worker.py) pour conserver l'API `from gui.workers import X`.
"""

from ._pipeline import (
    BatchWorker,
    FcsLoaderWorker,
    LogCapture,
    PipelineWorker,
    SpiderPlotWorker,
)
from .compute_worker import ComputeWorker, WorkerSignals, run_async

__all__ = [
    "PipelineWorker",
    "BatchWorker",
    "SpiderPlotWorker",
    "FcsLoaderWorker",
    "LogCapture",
    "ComputeWorker",
    "WorkerSignals",
    "run_async",
]
