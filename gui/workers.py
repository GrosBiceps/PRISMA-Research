# -*- coding: utf-8 -*-
"""
workers.py — QThread workers pour FlowSomAnalyzerPro.

Exécute le pipeline FlowSOM dans un thread séparé pour garder l'UI fluide.
Capture les logs via un QueueHandler thread-safe et les transmet en temps réel
sans détournement de sys.stdout ni scraping des messages de log.
"""

from __future__ import annotations

import logging
import logging.handlers
import queue
import traceback
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from config.pipeline_config import PipelineConfig

import threading

from PyQt5.QtCore import QCoreApplication, QThread, QTimer, pyqtSignal

_dbg = logging.getLogger("prisma.workers.debug")


class _QtLogHandler(logging.handlers.QueueHandler):
    """
    Handler thread-safe qui place les LogRecord dans une queue interne.
    Un QTimer dans le thread principal draine la queue et émet les signaux Qt.

    Avantages par rapport à l'approche directe :
    - Aucun appel Qt depuis un thread secondaire (évite segfaults).
    - Aucun blocage sur l'Event Loop Qt en cas de burst de messages.
    - QueueHandler est conçu exactement pour cet usage (stdlib Python).
    """

    def __init__(self) -> None:
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        super().__init__(self._queue)
        self.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(message)s",
                datefmt="%H:%M:%S",
            )
        )


class LogCapture:
    """
    Capture de logs thread-safe pour l'UI PyQt5.

    Installe un QueueHandler sur le logger root depuis le thread secondaire,
    puis draine la queue via un QTimer dans le thread principal (sans appel
    Qt cross-thread).

    Usage :
        capture = LogCapture(log_signal)
        capture.install()          # depuis le thread worker (avant pipeline)
        capture.start_drain()      # depuis le thread principal (après démarrage worker)
        ...
        capture.stop_drain()       # depuis le thread principal (après fin worker)
        capture.uninstall()        # depuis le thread principal (cleanup)
    """

    # Loggers tiers verbeux à silencer — produisent des milliers de messages
    # DEBUG lors de la compilation JIT (Numba) ou du calcul KNN (pynndescent),
    # ce qui sature la SimpleQueue et freeze l'UI.
    _NOISY_LOGGERS = ("numba", "numba.core", "pynndescent", "umap")

    def __init__(self, log_signal: pyqtSignal) -> None:
        self._log_signal = log_signal
        self._handler = _QtLogHandler()
        self._handler.setLevel(logging.INFO)
        self._timer: Optional[QTimer] = None
        self._root_logger = logging.getLogger()

    # ── Gestion du handler ─────────────────────────────────────────────────

    def install(self) -> None:
        """Ajoute le handler au logger root. Appelé depuis le thread worker."""
        _dbg.debug(
            "[LogCapture.install] thread=%s tid=%s",
            threading.current_thread().name,
            threading.get_ident(),
        )
        # Forcer INFO sur le root — en mode frozen certaines libs imposent WARNING
        # au démarrage, ce qui silencerait tous les messages INFO du pipeline.
        self._root_logger.setLevel(logging.INFO)
        for name in self._NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

        # Retirer tous les StreamHandlers qui écrivent sur stdout/stderr
        # (installés par logging.basicConfig ou les libs tierces) pour éviter
        # les exceptions silencieuses sur streams None en mode frozen console=False.
        import sys as _sys

        dead_streams = {None, getattr(_sys, "__stdout__", None), getattr(_sys, "__stderr__", None)}
        for h in list(self._root_logger.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.handlers.QueueHandler
            ):
                stream = getattr(h, "stream", None)
                if stream is None or stream in dead_streams:
                    self._root_logger.removeHandler(h)

        self._root_logger.addHandler(self._handler)

    def uninstall(self) -> None:
        """Retire le handler. Appelé depuis le thread principal après la fin."""
        _dbg.debug(
            "[LogCapture.uninstall] thread=%s tid=%s timer_active=%s",
            threading.current_thread().name,
            threading.get_ident(),
            self._timer is not None and self._timer.isActive() if self._timer else False,
        )
        self._root_logger.removeHandler(self._handler)
        self._drain_once()  # Vide les derniers messages

    # ── Drainage de la queue vers l'UI ────────────────────────────────────

    def start_drain(self, parent=None) -> None:
        """
        Démarre un QTimer dans le thread principal qui draine la queue toutes
        les 100 ms. Doit être appelé depuis le thread principal.

        parent — QObject optionnel propriétaire du timer (évite les fuites
                 mémoire et ancre le timer à la boucle d'événements Qt même
                 en mode frozen console=False).
        """
        _dbg.debug(
            "[LogCapture.start_drain] caller_thread=%s tid=%s parent=%r",
            threading.current_thread().name,
            threading.get_ident(),
            type(parent).__name__ if parent else None,
        )
        if self._timer is not None:
            _dbg.warning("[LogCapture.start_drain] timer already running — stopping old timer first")
            self._timer.stop()
            self._timer = None
        self._timer = QTimer(parent)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._drain_once)
        self._timer.start()
        _dbg.debug("[LogCapture.start_drain] timer started id=%d", id(self._timer))

    def stop_drain(self) -> None:
        """Arrête le timer de drainage. Doit être appelé depuis le thread principal."""
        _dbg.debug(
            "[LogCapture.stop_drain] caller_thread=%s tid=%s timer=%s active=%s",
            threading.current_thread().name,
            threading.get_ident(),
            id(self._timer) if self._timer else None,
            self._timer.isActive() if self._timer else False,
        )
        if self._timer is not None:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None
        self._drain_once()  # Dernier vidage

    def _drain_once(self) -> None:
        """
        Draine au plus 50 messages par tick pour ne pas bloquer l'UI.

        Si la queue contient plus de 200 messages (burst post-GIL après compilation
        Numba), on vide en plusieurs batches différés via QTimer.singleShot pour
        éviter de bloquer l'event loop Qt le temps du traitement complet.
        """
        q = self._handler._queue
        batch: List[str] = []
        for _ in range(50):
            try:
                record = q.get_nowait()
                msg = self._handler.format(record)
                batch.append(msg)
            except Exception:
                break

        if batch:
            self._log_signal.emit("\n".join(batch))


class PipelineWorker(QThread):
    """
    Thread dédié à l'exécution du pipeline FlowSOM complet.

    La progression est émise via un progress_callback propre transmis à
    FlowSOMPipeline.execute(), sans scraping des messages de log.

    Signaux :
        log_message  — chaque ligne de log (str)
        progress     — pourcentage estimé 0–100 (int)
        gating_done  — résultats du pre-gating prêts pour validation UI
                       (dict avec clés : n_kept, n_total, pct_kept, fallbacks)
        finished     — résultat PipelineResult (ou None si échec)
        error        — message d'erreur (str)
    """

    log_message = pyqtSignal(str)
    progress = pyqtSignal(int)
    gating_done = pyqtSignal(dict)
    prescreening_done = pyqtSignal(dict)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        config: "PipelineConfig",
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._log_capture = LogCapture(self.log_message)

    # ------------------------------------------------------------------
    @staticmethod
    def _warmup_numba() -> None:
        """
        Déclenche la compilation JIT Numba/pynndescent sur un micro-dataset avant
        le vrai pipeline.

        Sans ce warmup, Numba compile ses kernels LLVM lors du premier appel
        FlowSOM réel (20–90 s sur CPU) tout en tenant le GIL Python, ce qui
        gèle l'UI Qt et marque le processus "Pas de réponse" dans le Gestionnaire
        des tâches. Ce warmup s'exécute dans le QThread, hors thread principal.
        """
        try:
            import numpy as _np
            import warnings as _w

            # Micro-SOM : 500 cellules × 4 marqueurs — assez pour déclencher le JIT,
            # trop petit pour durer plus de 2–3 s.
            _rng = _np.random.default_rng(0)
            _X_tiny = _rng.random((500, 4), dtype=_np.float32)

            with _w.catch_warnings():
                _w.simplefilter("ignore")
                import anndata as _ad
                import flowsom as _fs

                _adata = _ad.AnnData(_X_tiny)
                _fs.FlowSOM(
                    _adata,
                    cols_to_use=list(range(4)),
                    xdim=3,
                    ydim=3,
                    n_clusters=2,
                    rlen=5,
                    seed=0,
                )
        except Exception:
            pass  # Non bloquant — si le warmup échoue, le vrai pipeline prend le relais

    def run(self) -> None:
        """Point d'entrée du thread — exécute FlowSOMPipeline.execute()."""
        # Installe le handler de log (thread-safe via QueueHandler)
        self._log_capture.install()

        # Logger racine local pour passer par la queue (jamais d'emit direct cross-thread)
        _log = logging.getLogger("worker")

        try:
            from pipeline.pipeline_executor import (
                FlowSOMPipeline,
                PipelineStep,
            )

            _log.info("═══ Démarrage du pipeline FlowSOM ═══")

            # Préchauffage Numba/pynndescent JIT avant le vrai pipeline.
            # Évite le freeze "Pas de réponse" dû à la compilation LLVM au premier appel.
            _log.info("[Init] Compilation JIT Numba (première exécution)…")
            self._warmup_numba()
            _log.info("[Init] JIT prêt.")

            pipeline = FlowSOMPipeline(self._config)
            _gating_emitted = False

            def _on_progress(step: PipelineStep, pct: int) -> None:
                """Callback appelé par le pipeline à chaque étape majeure."""
                nonlocal _gating_emitted
                self.progress.emit(pct)
                # Émettre le résumé de gating une seule fois dès que GATING_DONE
                if not _gating_emitted and pct >= PipelineStep.GATING_DONE:
                    _gating_emitted = True
                    try:
                        from models.gate_result import gating_reports

                        # Aligner le résumé UI sur la logique Sankey: chaîne COMBINED
                        # (G1 -> G2 -> G3 -> G4), sans sommer les gates entre elles.
                        events = pipeline._gating_logger.events
                        gate_map = {e.gate_name: e for e in events if e.file == "COMBINED"}

                        gate_order = ("G4_cd34", "G3_cd45", "G2_singlets", "G1_debris")
                        terminal_event = next(
                            (gate_map[g] for g in gate_order if g in gate_map), None
                        )
                        first_event = gate_map.get("G1_debris", terminal_event)

                        if terminal_event is not None and first_event is not None:
                            n_kept = int(terminal_event.n_after)
                            n_total = int(first_event.n_before)
                            n_gates = len(gate_map)
                        elif gating_reports:
                            # Fallback défensif en cas d'événements COMBINED absents
                            n_kept = int(gating_reports[-1].n_kept)
                            n_total = int(gating_reports[0].n_total)
                            n_gates = len(gating_reports)
                        else:
                            n_kept = 0
                            n_total = 0
                            n_gates = 0

                        fallbacks = [
                            g.gate_name
                            for g in gating_reports
                            if g.warnings or (g.method and "fallback" in g.method.lower())
                        ]
                        self.gating_done.emit(
                            {
                                "n_kept": n_kept,
                                "n_total": n_total,
                                "pct_kept": round(n_kept / max(n_total, 1) * 100, 1),
                                "n_gates": n_gates,
                                "fallbacks": fallbacks,
                            }
                        )
                    except Exception:
                        pass  # Non bloquant

            result = pipeline.execute(progress_callback=_on_progress)

            self.progress.emit(100)

            # Émettre le résultat du pré-screening si disponible
            _ps = getattr(result, "prescreening_result", None) if result else None
            if _ps is not None:
                try:
                    self.prescreening_done.emit(
                        {
                            "n_cd34_pos": int(_ps.n_cd34_pos),
                            "n_cd34_neg": int(_ps.n_cd34_neg),
                            "n_cd45dim": int(_ps.n_cd45dim),
                            "ratio_pct": float(_ps.ratio_pct),
                            "gmm_ratio_pct": float(_ps.gmm_ratio_pct),
                            "kde_ratio_pct": float(_ps.kde_ratio_pct),
                            "alert_level": str(_ps.alert_level),
                            "alert_message": str(_ps.alert_message),
                            "method_used": str(_ps.method_used),
                            "laip_tracking_recommended": bool(_ps.laip_tracking_recommended),
                            "interpretation_warning": str(_ps.interpretation_warning),
                        }
                    )
                except Exception:
                    pass  # Non bloquant

            if result is not None and result.success:
                _log.info(
                    "═══ Pipeline terminé — %s cellules, %s métaclusters ═══",
                    f"{result.n_cells:,}",
                    result.n_metaclusters,
                )
            else:
                _log.info("═══ Pipeline terminé avec des avertissements ═══")

            self.finished.emit(result)

        except Exception as exc:
            tb = traceback.format_exc()
            _log.error("[ERREUR] %s", exc)
            _log.error("%s", tb)
            # Flush queue avant d'émettre les signaux de fin
            self._log_capture._drain_once()
            self.error.emit(str(exc))
            self.finished.emit(None)

        finally:
            self._log_capture.uninstall()


class BatchWorker(QThread):
    """
    Thread dédié au traitement par lots (BatchPipeline).

    Signaux :
        log_message     — chaque ligne de log (str)
        progress        — pourcentage estimé 0–100 (int)
        file_started    — (index: int, total: int, filename: str)
        file_finished   — (stem: str, success: bool)
        finished        — dict {"results": [...], "excel": str|None}
        error           — message d'erreur (str)
    """

    log_message = pyqtSignal(str)
    progress = pyqtSignal(int)
    file_started = pyqtSignal(int, int, str)
    file_finished = pyqtSignal(str, bool)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        config: "PipelineConfig",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._log_capture = LogCapture(self.log_message)

    def run(self) -> None:
        """Point d'entrée du thread batch."""
        from pipeline.batch_pipeline import BatchPipeline

        self._log_capture.install()

        _log = logging.getLogger("batch_worker")

        try:
            _log.info("═══ Démarrage du mode Batch ═══")
            self.progress.emit(2)

            batch = BatchPipeline(self._config)
            summary = batch.execute(progress_callback=self._on_file_progress)

            self.progress.emit(100)
            n_ok = sum(1 for _, r in summary.get("results", []) if r is not None and r.success)
            n_total = len(summary.get("results", []))
            _log.info("═══ Batch terminé — %d/%d fichier(s) réussis ═══", n_ok, n_total)
            self.finished.emit(summary)

        except Exception as exc:
            tb = traceback.format_exc()
            _log.error("[ERREUR BATCH] %s", exc)
            _log.error("%s", tb)
            self._log_capture._drain_once()
            self.error.emit(str(exc))
            self.finished.emit(None)

        finally:
            self._log_capture.uninstall()

    def _on_file_progress(self, current: int, total: int, filename: str) -> None:
        """Appelé par BatchPipeline entre chaque fichier."""
        self.file_started.emit(current, total, filename)
        if total > 0:
            pct = int(current / total * 95)
            self.progress.emit(max(2, pct))


class SpiderPlotWorker(QThread):
    """
    Thread pour générer un Spider Plot (radar) matplotlib pour un nœud SOM donné.

    Signaux :
        figure_ready — matplotlib.figure.Figure prête à afficher
        error        — message d'erreur (str)
    """

    figure_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        mfi_row: Any,  # ndarray ou Series — profil MFI moyen total du cluster
        marker_names: list,  # subset des marqueurs à afficher
        cluster_label: str,  # libellé pour le titre (ex. "Cluster 42")
        patho_row: Any = None,  # profil MFI pathologique du cluster (Series ou None)
        nbm_row: Any = None,  # profil MFI NBM pour ce cluster (Series ou None)
        nbm_mean_row: Any = None,  # profil MFI moyen global NBM (Series ou None)
        canvas_width: int = 800,  # largeur cible du widget radar (px)
        canvas_height: int = 400,  # hauteur cible du widget radar (px)
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self._mfi_row = mfi_row
        self._marker_names = marker_names
        self._cluster_label = cluster_label
        self._patho_row = patho_row
        self._nbm_row = nbm_row
        self._nbm_mean_row = nbm_mean_row
        self._canvas_width = max(1, int(canvas_width))
        self._canvas_height = max(1, int(canvas_height))

    def run(self) -> None:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd

            markers = self._marker_names
            if not markers:
                self.error.emit("Aucun marqueur sélectionné pour le Spider Plot.")
                return

            def _extract(row) -> Optional[np.ndarray]:
                if row is None:
                    return None
                if isinstance(row, pd.Series):
                    return np.array([row.get(m, 0.0) for m in markers], dtype=float)
                return np.asarray(row, dtype=float)[: len(markers)]

            total_vals = _extract(self._mfi_row)
            patho_vals = _extract(self._patho_row)
            nbm_vals = _extract(self._nbm_row)
            nbm_mean = _extract(self._nbm_mean_row)

            if patho_vals is None:
                patho_vals = total_vals

            # Normalisation min-max sur l'union de toutes les données disponibles
            all_vals = [total_vals]
            if patho_vals is not None:
                all_vals.append(patho_vals)
            if nbm_vals is not None:
                all_vals.append(nbm_vals)
            if nbm_mean is not None:
                all_vals.append(nbm_mean)
            combined = np.concatenate(all_vals)
            vmin, vmax = combined.min(), combined.max()
            rng = vmax - vmin if vmax != vmin else 1.0

            def _norm(v: np.ndarray) -> np.ndarray:
                return (v - vmin) / rng

            n = len(markers)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()

            def _plot_series(
                ax,
                values: np.ndarray,
                color: str,
                label: str,
                lw: float = 2.0,
                alpha_line: float = 1.0,
                alpha_fill: float = 0.28,
                linestyle: str = "-",
            ) -> None:
                v = _norm(values)
                v_plot = np.concatenate([v, [v[0]]])
                a_plot = angles + angles[:1]
                ax.plot(
                    a_plot,
                    v_plot,
                    color=color,
                    linewidth=lw,
                    alpha=alpha_line,
                    label=label,
                    zorder=3,
                    linestyle=linestyle,
                    marker="o",
                    markersize=4,
                    markerfacecolor=color,
                    markeredgewidth=0,
                )
                if alpha_fill > 0:
                    ax.fill(a_plot, v_plot, color=color, alpha=alpha_fill, zorder=2)

            from matplotlib.figure import Figure as _Figure

            fig = _Figure(figsize=(6, 6), facecolor="#1e1e2e")
            ax = fig.add_subplot(111, projection="polar")
            ax.set_facecolor("#1e1e2e")

            compact = (self._canvas_width < 760) or (self._canvas_height < 320)
            marker_font = 7 if compact else 8
            y_font = 6 if compact else 7
            title_font = 9 if compact else 11
            title_pad = 10 if compact else 20
            tick_pad = 3 if compact else 6

            # Violet (#c084fc) — Moyenne totale du cluster
            _plot_series(
                ax,
                total_vals,
                color="#c084fc",
                label="Cluster total (moyenne)",
                lw=2.2,
                alpha_line=1.0,
                alpha_fill=0.32,
            )

            # Orange (#FF9B3D) — Profil pathologique du cluster
            if patho_vals is not None:
                _plot_series(
                    ax,
                    patho_vals,
                    color="#FF9B3D",
                    label="Patho (cluster)",
                    lw=2.0,
                    alpha_line=0.98,
                    alpha_fill=0.24,
                )

            # Bleu (#5BAAFF) — NBM du même cluster
            if nbm_vals is not None:
                _plot_series(
                    ax,
                    nbm_vals,
                    color="#5BAAFF",
                    label="NBM (cluster)",
                    lw=1.8,
                    alpha_line=0.95,
                    alpha_fill=0.22,
                )

            # Vert (#39FF8A) — NBM moyen global de référence
            if nbm_mean is not None:
                _plot_series(
                    ax,
                    nbm_mean,
                    color="#39FF8A",
                    label="NBM moyen (réf.)",
                    lw=1.6,
                    alpha_line=0.95,
                    alpha_fill=0.0,
                    linestyle="--",
                )

            ax.set_xticks(angles)
            ax.set_xticklabels(markers, size=marker_font, color="#cdd6f4")
            ax.tick_params(axis="y", colors="#9399b2", labelsize=y_font)
            ax.tick_params(axis="x", pad=tick_pad, colors="#cdd6f4")
            ax.spines["polar"].set_color("#585b70")
            ax.yaxis.grid(True, color="#3a3a52", linewidth=0.7, linestyle=":")
            ax.xaxis.grid(True, color="#45475a", linewidth=0.6)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], size=6, color="#6c7086")
            fig.suptitle(
                self._cluster_label,
                color="#cdd6f4",
                fontsize=title_font,
                fontweight="bold",
                y=0.98,
            )

            # Légende toujours affichée, avec description explicite des séries
            legend_handles = []
            from matplotlib.lines import Line2D

            label_total = "Cluster total"
            label_patho = "Patho"
            label_nbm = "NBM cluster"
            label_nbm_mean = "NBM moyen (réf.)"

            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#c084fc",
                    linewidth=2.2,
                    marker="o",
                    markersize=4,
                    label=label_total,
                )
            )
            if patho_vals is not None:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color="#FF9B3D",
                        linewidth=2.0,
                        marker="o",
                        markersize=4,
                        label=label_patho,
                    )
                )
            if nbm_vals is not None:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color="#5BAAFF",
                        linewidth=1.8,
                        marker="o",
                        markersize=4,
                        label=label_nbm,
                    )
                )
            if nbm_mean is not None:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color="#39FF8A",
                        linewidth=1.5,
                        marker="o",
                        markersize=4,
                        linestyle="--",
                        label=label_nbm_mean,
                    )
                )

            # Layout responsive : réserve une colonne de légende à gauche,
            # puis recentre visuellement le radar pour éviter l'effet "décalé à droite".
            if compact:
                legend_col = min(0.28, max(0.19, 170.0 / float(self._canvas_width)))
                ax_left = legend_col + 0.02
                ax_right = 0.90
                # top réduit à 0.88 pour laisser 10 % au suptitle
                ax.set_position([ax_left, 0.10, max(0.46, ax_right - ax_left), 0.72])
                lg_font = 6.8
                lg_anchor = (0.02, 0.50)
                lg_loc = "center left"
                lg_handle = 1.3
                lg_pad = 0.58
            else:
                legend_col = min(0.32, max(0.22, 210.0 / float(self._canvas_width)))
                ax_left = legend_col + 0.02
                ax_right = 0.92
                # top réduit à 0.88 pour laisser 10 % au suptitle
                ax.set_position([ax_left, 0.08, max(0.48, ax_right - ax_left), 0.76])
                lg_font = 7.4
                lg_anchor = (0.02, 0.50)
                lg_loc = "center left"
                lg_handle = 1.45
                lg_pad = 0.78

            fig.legend(
                handles=legend_handles,
                loc=lg_loc,
                bbox_to_anchor=lg_anchor,
                fontsize=lg_font,
                framealpha=0.78,
                facecolor="#1a1a2e",
                edgecolor="#45475a",
                labelcolor="#cdd6f4",
                handlelength=lg_handle,
                borderpad=lg_pad,
            )
            self.figure_ready.emit(fig)
        except Exception as exc:
            self.error.emit(str(exc))


class FcsLoaderWorker(QThread):
    """
    Charge un fichier FCS hors du thread principal pour éviter tout gel de l'UI.

    Tente 4 backends dans l'ordre :
      1. flowsom.io.read_FCS
      2. flowio + anndata
      3. fcsparser
      4. lecture binaire brute (_read_fcs_binary du main_window)

    Signaux :
        loaded  — AnnData résultant (objet Python)
        error   — message d'erreur (str) si tous les backends échouent
        log     — messages de progression (str)
    """

    loaded = pyqtSignal(object)  # AnnData
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, file_path: str, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self._file_path = file_path

    def run(self) -> None:
        import numpy as np

        fp = self._file_path
        adata = None
        last_error: Optional[Exception] = None

        # ── Backend 1 : flowsom ───────────────────────────────────────────
        try:
            import flowsom as fs

            adata = fs.io.read_FCS(fp)
            self.log.emit("Chargé avec flowsom")
        except Exception as e1:
            self.log.emit(f"flowsom échoué : {str(e1)[:60]}")
            last_error = e1

        # ── Backend 2 : flowio + anndata ─────────────────────────────────
        if adata is None:
            try:
                import flowio
                import anndata as ad

                fcs_data = flowio.FlowData(fp)
                events = np.reshape(fcs_data.events, (-1, fcs_data.channel_count))
                n_ev, n_ch = events.shape
                if n_ev > 0 and n_ch > 0:
                    ch_names: List[str] = []
                    for i in range(1, n_ch + 1):
                        name = None
                        for key in (f"$P{i}N", f"P{i}N", f"$P{i}S", f"P{i}S"):
                            if key in fcs_data.text:
                                name = fcs_data.text[key]
                                break
                        ch_names.append(str(name) if name else f"Channel_{i}")
                    adata = ad.AnnData(events.astype(np.float32))
                    adata.var_names = ch_names
                    self.log.emit("Chargé avec flowio")
                else:
                    raise ValueError(f"Fichier vide : {n_ev} events, {n_ch} canaux")
            except ImportError:
                self.log.emit("flowio non installé")
            except Exception as e2:
                self.log.emit(f"flowio échoué : {str(e2)[:60]}")
                last_error = e2

        # ── Backend 3 : fcsparser ─────────────────────────────────────────
        if adata is None:
            try:
                import fcsparser
                import anndata as ad

                for naming in ("$PnS", "$PnN"):
                    try:
                        meta, data = fcsparser.parse(
                            fp,
                            meta_data_only=False,
                            reformat_meta=False,
                            channel_naming=naming,
                        )
                        adata = ad.AnnData(data.values.astype(np.float32))
                        adata.var_names = list(data.columns)
                        self.log.emit(f"Chargé avec fcsparser ({naming})")
                        break
                    except Exception:
                        pass
            except ImportError:
                self.log.emit("fcsparser non installé")
            except Exception as e3:
                self.log.emit(f"fcsparser échoué : {str(e3)[:60]}")
                last_error = e3

        # ── Backend 4 : lecture binaire brute ────────────────────────────
        if adata is None:
            try:
                adata = self._read_fcs_binary(fp)
                self.log.emit("Chargé avec lecture binaire directe")
            except Exception as e4:
                self.log.emit(f"Binaire échoué : {str(e4)[:60]}")
                last_error = e4

        if adata is None:
            self.error.emit(f"Impossible de charger le FCS. Dernière erreur : {last_error}")
            return

        self.loaded.emit(adata)

    def _read_fcs_binary(self, file_path: str):
        """Lecture binaire FCS brute — fallback ultime."""
        import struct
        import numpy as np
        import anndata as ad

        with open(file_path, "rb") as f:
            header = f.read(58)
            if len(header) < 58:
                raise ValueError("Fichier FCS trop court")
            version = header[:6].decode("ascii", errors="replace").strip()
            if not version.startswith("FCS"):
                raise ValueError(f"Version FCS non reconnue : {version}")
            text_start = int(header[10:18].strip())
            text_end = int(header[18:26].strip())
            data_start = int(header[26:34].strip())
            data_end = int(header[34:42].strip())
            f.seek(text_start)
            text_raw = f.read(text_end - text_start + 1).decode("latin-1", errors="replace")
        if not text_raw:
            raise ValueError("Segment TEXT vide")
        delim = text_raw[0]
        parts = text_raw.split(delim)
        kv: dict = {}
        for i in range(1, len(parts) - 1, 2):
            kv[parts[i].strip().upper()] = parts[i + 1].strip() if i + 1 < len(parts) else ""
        n_params = int(kv.get("$PAR", 0))
        n_events = int(kv.get("$TOT", 0))
        data_type = kv.get("$DATATYPE", "F").upper()
        byte_order = kv.get("$BYTEORD", "1,2,3,4")
        endian = "<" if byte_order.startswith("1") else ">"
        fmt_char = {"F": "f", "D": "d", "I": "i"}.get(data_type, "f")
        dtype = np.dtype(f"{endian}{fmt_char}")
        with open(file_path, "rb") as f:
            f.seek(data_start)
            n_bytes = (
                (data_end - data_start + 1)
                if data_end > data_start
                else (n_events * n_params * dtype.itemsize)
            )
            raw = f.read(n_bytes)
        arr = np.frombuffer(raw, dtype=dtype).reshape(n_events, n_params).astype(np.float32)
        channel_names = []
        for i in range(1, n_params + 1):
            name = kv.get(f"$P{i}N") or kv.get(f"$P{i}S") or f"Channel_{i}"
            channel_names.append(name)
        adata = ad.AnnData(arr)
        adata.var_names = channel_names
        return adata
