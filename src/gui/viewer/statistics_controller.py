"""
src/gui/viewer/statistics_controller.py — Contrôleur statistiques dédié (Phase 5).

Remplace _update_statistics() inline dans PrismaGatingWorkspace.

Responsabilités :
  - Écouter l'engine (analysisCompleted, gatingStrategyChanged)
  - Récupérer les statistiques depuis engine.session.get_analysis_report()
  - Alimenter StatisticsDock
  - Gérer les états stale / loading / ready / error
  - Refresh asynchrone via QThreadPool (ne bloque pas l'UI)
  - Cache minimal de la dernière table
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from PyQt5.QtCore import QObject, QRunnable, QThreadPool, Qt, pyqtSignal, pyqtSlot

from .gating_engine import PrismaFlowEngine
from .statistics_dock import StatisticsDock

_logger = logging.getLogger("viewer.statistics_controller")


# ---------------------------------------------------------------------------
# Worker asynchrone — collecte les stats hors du thread UI
# ---------------------------------------------------------------------------


class _StatsWorkerSignals(QObject):
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)


class _StatsWorker(QRunnable):
    """
    Collecte les statistiques dans un thread pool.
    N'accède qu'aux méthodes thread-safe de l'engine (lecture seule).
    """

    def __init__(
        self,
        engine: PrismaFlowEngine,
        sample_id: str,
        x_channel: Optional[str],
        y_channel: Optional[str],
        transform_id: Optional[str],
    ) -> None:
        super().__init__()
        self._engine = engine
        self._sample_id = sample_id
        self._x_channel = x_channel
        self._y_channel = y_channel
        self._transform_id = transform_id
        self.signals = _StatsWorkerSignals()
        self.setAutoDelete(True)

    @pyqtSlot()
    def run(self) -> None:
        try:
            df = _collect_statistics(
                self._engine,
                self._sample_id,
                self._x_channel,
                self._y_channel,
                self._transform_id,
            )
            self.signals.finished.emit(df)
        except Exception as exc:
            _logger.warning("_StatsWorker.run error: %s", exc)
            self.signals.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Collecte des stats (fonction pure — appelée depuis thread worker)
# ---------------------------------------------------------------------------


def _collect_statistics(
    engine: PrismaFlowEngine,
    sample_id: str,
    x_channel: Optional[str],
    y_channel: Optional[str],
    transform_id: Optional[str],
) -> pd.DataFrame:
    """
    Construit un DataFrame de statistiques avec les colonnes :
    Population | Count | % Parent | % Total | MFI X | MFI Y

    Utilise get_analysis_report() si disponible, sinon calcul manuel.
    """
    gate_ids = engine.get_gate_ids()
    if not gate_ids:
        return pd.DataFrame(columns=["Population", "Count", "% Parent", "% Total", "MFI X", "MFI Y"])

    # Tenter get_analysis_report (FlowKit >= 1.0)
    report_df: Optional[pd.DataFrame] = None
    try:
        report_df = engine.session.get_analysis_report(sample_id=sample_id)
    except Exception:
        pass

    rows = []
    for gid in gate_ids:
        count = 0
        pct_parent = 0.0
        pct_total = 0.0
        mfi_x = ""
        mfi_y = ""

        try:
            if report_df is not None and not report_df.empty:
                row = report_df[report_df["gate_name"] == gid]
                if not row.empty:
                    count = int(row["count"].iloc[0]) if "count" in row.columns else 0
                    pct_parent = (
                        float(row["percent_of_parent"].iloc[0])
                        if "percent_of_parent" in row.columns
                        else 0.0
                    )
                    pct_total = (
                        float(row["percent_of_total"].iloc[0])
                        if "percent_of_total" in row.columns
                        else 0.0
                    )
            else:
                paths = engine.find_gate_paths(gid)
                gate_path = paths[0] if paths else None
                gdf = engine.get_gate_dataframe(gid, gate_path=gate_path, sample_id=sample_id)
                if gdf is not None:
                    count = len(gdf)
                    try:
                        total_df = engine.get_population_df(("root",), sample_id=sample_id)
                        total = len(total_df) if total_df is not None else 0
                        pct_total = round(100.0 * count / total, 2) if total > 0 else 0.0
                    except Exception:
                        pass
        except Exception as exc:
            _logger.debug("Stats gate '%s' ignorée : %s", gid, exc)

        # MFI optionnelle
        try:
            paths = engine.find_gate_paths(gid)
            gate_path = paths[0] if paths else None
            gdf = engine.get_gate_dataframe(
                gid,
                gate_path=gate_path,
                sample_id=sample_id,
                transform_id=transform_id,
            )
            if gdf is not None and not gdf.empty:
                if x_channel and x_channel in gdf.columns:
                    mfi_x = f"{float(gdf[x_channel].median()):.1f}"
                if y_channel and y_channel in gdf.columns:
                    mfi_y = f"{float(gdf[y_channel].median()):.1f}"
                if count == 0:
                    count = len(gdf)
        except Exception:
            pass

        rows.append(
            {
                "Population": gid,
                "Count": str(count),
                "% Parent": f"{pct_parent:.1f}",
                "% Total": f"{pct_total:.1f}",
                "MFI X": mfi_x,
                "MFI Y": mfi_y,
            }
        )

    return pd.DataFrame(rows, columns=["Population", "Count", "% Parent", "% Total", "MFI X", "MFI Y"])


# ---------------------------------------------------------------------------
# StatisticsController
# ---------------------------------------------------------------------------


class StatisticsController(QObject):
    """
    Contrôleur qui orchestre la collecte et l'affichage des statistiques.

    Instanciation typique dans PrismaGatingWorkspace :

        self._stats_ctrl = StatisticsController(
            engine=self._engine,
            dock=self._stats_dock,
            parent=self,
        )
        self._engine.analysisCompleted.connect(
            self._stats_ctrl.on_engine_analysis_completed, Qt.QueuedConnection
        )
        self._engine.gatingStrategyChanged.connect(
            self._stats_ctrl.on_engine_strategy_changed, Qt.QueuedConnection
        )
    """

    def __init__(
        self,
        engine: PrismaFlowEngine,
        dock: StatisticsDock,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._dock = dock
        self._sample_id: Optional[str] = None
        self._x_channel: Optional[str] = None
        self._y_channel: Optional[str] = None
        self._transform_id: Optional[str] = None
        self._last_df: Optional[pd.DataFrame] = None
        self._refresh_pending: bool = False

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def set_engine(self, engine: PrismaFlowEngine) -> None:
        """Remplace le moteur actif."""
        self._engine = engine

    def set_current_sample(self, sample_id: Optional[str]) -> None:
        """Définit le sample actif et invalide le cache."""
        if sample_id != self._sample_id:
            self._sample_id = sample_id
            self._last_df = None

    def set_axis_channels(
        self,
        x_channel: Optional[str],
        y_channel: Optional[str],
    ) -> None:
        """Met à jour les canaux utilisés pour le calcul MFI."""
        self._x_channel = x_channel
        self._y_channel = y_channel

    def set_transform_id(self, transform_id: Optional[str]) -> None:
        """Met à jour l'identifiant de transformation utilisé pour MFI."""
        self._transform_id = transform_id

    def refresh_statistics(self) -> None:
        """
        Lance un refresh asynchrone des statistiques.
        Si aucun sample actif → clear du dock sans erreur.
        """
        sample_id = self._sample_id or (
            self._engine.active_sample_id if self._engine else None
        )

        if not sample_id:
            self._dock.clear()
            return

        if self._engine is None:
            self._dock.set_error("Engine non initialisé.")
            return

        self._dock.set_loading()
        self._refresh_pending = True

        worker = _StatsWorker(
            engine=self._engine,
            sample_id=sample_id,
            x_channel=self._x_channel,
            y_channel=self._y_channel,
            transform_id=self._transform_id,
        )
        worker.signals.finished.connect(self._on_worker_finished, Qt.QueuedConnection)
        worker.signals.error.connect(self._on_worker_error, Qt.QueuedConnection)
        QThreadPool.globalInstance().start(worker)

    # ------------------------------------------------------------------
    # Slots engine (connecter avec Qt.QueuedConnection)
    # ------------------------------------------------------------------

    def on_engine_analysis_completed(self, sample_id: str) -> None:
        """Slot : analyse async terminée → refresh stats."""
        _logger.debug("StatisticsController: analysisCompleted sample=%s", sample_id)
        self.set_current_sample(sample_id)
        self.refresh_statistics()

    def on_engine_strategy_changed(self, gate_id: str, change_type: str) -> None:
        """
        Slot : stratégie de gating changée → marquer stale.
        Ne recharge pas immédiatement (l'analyse async suit généralement).
        """
        _logger.debug(
            "StatisticsController: gatingStrategyChanged gate=%s type=%s",
            gate_id,
            change_type,
        )
        self._last_df = None

    # ------------------------------------------------------------------
    # Slots worker internes
    # ------------------------------------------------------------------

    @pyqtSlot(pd.DataFrame)
    def _on_worker_finished(self, df: pd.DataFrame) -> None:
        self._refresh_pending = False
        self._last_df = df
        self._dock.set_dataframe(df)

    @pyqtSlot(str)
    def _on_worker_error(self, message: str) -> None:
        self._refresh_pending = False
        _logger.warning("StatisticsController worker error: %s", message)
        self._dock.set_error(message)
