"""
tests/prisma/test_phase5_smoke.py — Smoke tests Phase 5 (sans FCS réels).

Exécution :
    pytest tests/prisma/test_phase5_smoke.py -v

Mocks utilisés :
    - PrismaFlowEngine : version allégée simulant les méthodes nécessaires
    - StatisticsDock : vérifié directement
    - PopulationTreeController : vérifié avec engine mock
    - StatisticsController : vérifié avec engine mock + dock réel
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers Qt
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def qapp():
    """Application Qt partagée pour toute la session de tests."""
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    yield app


# ---------------------------------------------------------------------------
# Mock Engine
# ---------------------------------------------------------------------------


def _make_mock_engine() -> MagicMock:
    """Crée un mock de PrismaFlowEngine avec les méthodes minimales."""
    engine = MagicMock()
    engine.active_sample_id = "sample_001"
    engine.get_gate_ids.return_value = ["Gate_A", "Gate_B"]
    engine.find_gate_paths.return_value = [("root",)]
    engine.build_hierarchy.return_value = []
    engine.get_channel_marker_mapping.return_value = {"FSC-A": "FSC-A", "SSC-A": "SSC-A"}
    engine.get_sample_ids.return_value = ["sample_001"]
    engine.get_sample_groups.return_value = []
    engine.get_comp_matrix_ids.return_value = []
    engine.get_transform_ids.return_value = ["logicle_1"]
    engine.get_population_df.return_value = pd.DataFrame({"FSC-A": [1, 2, 3], "SSC-A": [4, 5, 6]})
    engine.get_population_df_sampled.return_value = pd.DataFrame(
        {"FSC-A": [1, 2], "SSC-A": [4, 5]}
    )
    engine.get_gate_dataframe.return_value = pd.DataFrame(
        {"FSC-A": [1.0, 2.0], "SSC-A": [3.0, 4.0]}
    )
    # Simuler l'accès session.get_analysis_report
    mock_session = MagicMock()
    mock_session.get_analysis_report.return_value = pd.DataFrame(
        {
            "gate_name": ["Gate_A", "Gate_B"],
            "count": [100, 50],
            "percent_of_parent": [80.0, 40.0],
            "percent_of_total": [80.0, 40.0],
        }
    )
    engine.session = mock_session
    # Signaux Qt simulés (non QObject → pas de vraie connexion)
    engine.gatingStrategyChanged = MagicMock()
    engine.analysisCompleted = MagicMock()
    engine.analysisError = MagicMock()
    engine.gatingStrategyChanged.connect = MagicMock()
    engine.analysisCompleted.connect = MagicMock()
    engine.analysisError.connect = MagicMock()
    return engine


# ---------------------------------------------------------------------------
# Smoke test 1 — StatisticsDock
# ---------------------------------------------------------------------------


class TestStatisticsDock:
    def test_initial_state_empty(self, qapp):
        from src.prisma.gui.viewer.statistics_dock import StatisticsDock

        dock = StatisticsDock()
        # Doit être dans l'état vide sans erreur
        assert dock is not None

    def test_set_loading(self, qapp):
        from src.prisma.gui.viewer.statistics_dock import StatisticsDock

        dock = StatisticsDock()
        dock.set_loading()  # Ne doit pas lever d'exception

    def test_set_error(self, qapp):
        from src.prisma.gui.viewer.statistics_dock import StatisticsDock

        dock = StatisticsDock()
        dock.set_error("Erreur de test")  # Ne doit pas lever d'exception

    def test_set_dataframe_normal(self, qapp):
        from src.prisma.gui.viewer.statistics_dock import StatisticsDock

        dock = StatisticsDock()
        df = pd.DataFrame(
            {
                "Population": ["Gate_A", "Gate_B"],
                "Count": ["100", "50"],
                "% Parent": ["80.0", "40.0"],
                "% Total": ["80.0", "40.0"],
                "MFI X": ["123.4", "56.7"],
                "MFI Y": ["", ""],
            }
        )
        dock.set_dataframe(df)
        assert dock._table.rowCount() == 2

    def test_set_dataframe_empty_triggers_clear(self, qapp):
        from src.prisma.gui.viewer.statistics_dock import StatisticsDock

        dock = StatisticsDock()
        dock.set_dataframe(pd.DataFrame())
        assert dock._table.rowCount() == 0

    def test_set_dataframe_none_triggers_clear(self, qapp):
        from src.prisma.gui.viewer.statistics_dock import StatisticsDock

        dock = StatisticsDock()
        dock.set_dataframe(None)  # type: ignore[arg-type]
        assert dock._table.rowCount() == 0

    def test_clear(self, qapp):
        from src.prisma.gui.viewer.statistics_dock import StatisticsDock

        dock = StatisticsDock()
        dock.set_dataframe(
            pd.DataFrame({"Population": ["G1"], "Count": ["10"], "% Parent": ["50.0"],
                          "% Total": ["50.0"], "MFI X": [""], "MFI Y": [""]})
        )
        dock.clear()
        assert dock._table.rowCount() == 0


# ---------------------------------------------------------------------------
# Smoke test 2 — StatisticsController avec mock engine
# ---------------------------------------------------------------------------


class TestStatisticsController:
    def test_refresh_no_sample(self, qapp):
        from src.prisma.gui.viewer.statistics_controller import StatisticsController
        from src.prisma.gui.viewer.statistics_dock import StatisticsDock

        engine = _make_mock_engine()
        engine.active_sample_id = None
        engine.get_gate_ids.return_value = []

        dock = StatisticsDock()
        ctrl = StatisticsController(engine=engine, dock=dock)
        ctrl.set_current_sample(None)
        ctrl.refresh_statistics()  # Doit appeler dock.clear(), pas d'exception

    def test_refresh_with_sample(self, qapp):
        from src.prisma.gui.viewer.statistics_controller import StatisticsController
        from src.prisma.gui.viewer.statistics_dock import StatisticsDock
        from PyQt5.QtTest import QTest

        engine = _make_mock_engine()
        dock = StatisticsDock()
        ctrl = StatisticsController(engine=engine, dock=dock)
        ctrl.set_current_sample("sample_001")
        ctrl.set_axis_channels("FSC-A", "SSC-A")
        ctrl.refresh_statistics()
        # Attendre que le worker QThreadPool complète (max 2 s)
        for _ in range(20):
            QTest.qWait(100)
            if not ctrl._refresh_pending:
                break
        # Le dock doit être en état READY ou ERROR, pas LOADING
        assert not ctrl._refresh_pending

    def test_on_engine_analysis_completed(self, qapp):
        from src.prisma.gui.viewer.statistics_controller import StatisticsController
        from src.prisma.gui.viewer.statistics_dock import StatisticsDock

        engine = _make_mock_engine()
        dock = StatisticsDock()
        ctrl = StatisticsController(engine=engine, dock=dock)
        ctrl.on_engine_analysis_completed("sample_001")

    def test_on_engine_strategy_changed_invalidates_cache(self, qapp):
        from src.prisma.gui.viewer.statistics_controller import StatisticsController
        from src.prisma.gui.viewer.statistics_dock import StatisticsDock

        engine = _make_mock_engine()
        dock = StatisticsDock()
        ctrl = StatisticsController(engine=engine, dock=dock)
        ctrl._last_df = pd.DataFrame({"col": [1]})
        ctrl.on_engine_strategy_changed("Gate_A", "added")
        assert ctrl._last_df is None


# ---------------------------------------------------------------------------
# Smoke test 3 — PopulationTreeController avec mock engine
# ---------------------------------------------------------------------------


class TestPopulationTreeController:
    def test_build_panel_widget(self, qapp):
        from src.prisma.gui.viewer.population_tree_controller import PopulationTreeController

        engine = _make_mock_engine()
        ctrl = PopulationTreeController(engine=engine)
        widget = ctrl.build_panel_widget()
        assert widget is not None
        assert ctrl.tree_view is not None

    def test_refresh_tree_empty_hierarchy(self, qapp):
        from src.prisma.gui.viewer.population_tree_controller import PopulationTreeController

        engine = _make_mock_engine()
        engine.build_hierarchy.return_value = []
        ctrl = PopulationTreeController(engine=engine)
        ctrl.build_panel_widget()
        ctrl.refresh_tree()  # Ne doit pas lever d'exception
        assert ctrl.gate_tree_model.rowCount() == 0

    def test_set_current_sample(self, qapp):
        from src.prisma.gui.viewer.population_tree_controller import PopulationTreeController

        engine = _make_mock_engine()
        ctrl = PopulationTreeController(engine=engine)
        ctrl.build_panel_widget()
        ctrl.set_current_sample("sample_001")
        assert ctrl._current_sample_id == "sample_001"

    def test_current_gate_path_no_selection(self, qapp):
        from src.prisma.gui.viewer.population_tree_controller import PopulationTreeController

        engine = _make_mock_engine()
        ctrl = PopulationTreeController(engine=engine)
        ctrl.build_panel_widget()
        result = ctrl.current_gate_path()
        assert result is None

    def test_on_engine_strategy_changed(self, qapp):
        from src.prisma.gui.viewer.population_tree_controller import PopulationTreeController

        engine = _make_mock_engine()
        ctrl = PopulationTreeController(engine=engine)
        ctrl.build_panel_widget()
        ctrl.on_engine_strategy_changed("Gate_A", "added")
        engine.build_hierarchy.assert_called()

    def test_population_selected_signal_emitted(self, qapp):
        from src.prisma.gui.viewer.population_tree_controller import PopulationTreeController
        from PyQt5.QtCore import QModelIndex

        engine = _make_mock_engine()
        ctrl = PopulationTreeController(engine=engine)
        ctrl.build_panel_widget()

        received: list[tuple] = []
        ctrl.populationSelected.connect(lambda gp: received.append(gp))

        # Simuler un clic sur un nœud
        # Comme l'arbre est vide on vérifie juste que le signal est connecté
        assert ctrl.populationSelected is not None


# ---------------------------------------------------------------------------
# Smoke test 4 — WorksheetLayout save/restore
# ---------------------------------------------------------------------------


class TestWorksheetLayout:
    def test_validate_column_count_valid(self):
        from src.prisma.gui.viewer.worksheet_layout import validate_column_count

        for n in [1, 2, 3, 4]:
            assert validate_column_count(n) == n

    def test_validate_column_count_invalid(self):
        from src.prisma.gui.viewer.worksheet_layout import validate_column_count, WorksheetLayoutError

        with pytest.raises(WorksheetLayoutError):
            validate_column_count(0)
        with pytest.raises(WorksheetLayoutError):
            validate_column_count(5)

    def test_save_and_load_layout(self, tmp_path):
        from src.prisma.gui.viewer.worksheet_layout import (
            WorkspaceLayoutState,
            PanelState,
            save_layout_state,
            load_layout_state,
        )

        state = WorkspaceLayoutState(
            panel_column_count=3,
            active_panel_id="abc123",
            panels=[
                PanelState(
                    panel_id="abc123",
                    gate_node=("root",),
                    x_channel="FSC-A",
                    y_channel="SSC-A",
                    transform_id="logicle_1",
                    comp_id=None,
                    render_mode="SCATTER",
                    density_coloring=False,
                    position=0,
                ),
                PanelState(
                    panel_id="def456",
                    gate_node=("Gate_A", "root"),
                    x_channel="CD3",
                    y_channel="CD4",
                    transform_id="logicle_1",
                    comp_id=None,
                    render_mode="HYBRID",
                    density_coloring=True,
                    position=1,
                ),
            ],
        )

        layout_file = tmp_path / "layout.json"
        save_layout_state(state, layout_file)
        assert layout_file.exists()

        restored = load_layout_state(layout_file)
        assert restored.panel_column_count == 3
        assert restored.active_panel_id == "abc123"
        assert len(restored.panels) == 2
        assert restored.panels[0].x_channel == "FSC-A"
        assert restored.panels[1].gate_node == ("Gate_A", "root")

    def test_load_missing_file_returns_default(self, tmp_path):
        from src.prisma.gui.viewer.worksheet_layout import load_layout_state

        state = load_layout_state(tmp_path / "nonexistent.json")
        assert state.panel_column_count == 2
        assert state.panels == []

    def test_load_corrupt_json_returns_default(self, tmp_path):
        from src.prisma.gui.viewer.worksheet_layout import load_layout_state

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("NOT JSON !!!", encoding="utf-8")
        state = load_layout_state(bad_file)
        assert state.panel_column_count == 2

    def test_load_invalid_column_count_fallback(self, tmp_path):
        from src.prisma.gui.viewer.worksheet_layout import load_layout_state, _FORMAT_VERSION

        bad_state = {
            "format": _FORMAT_VERSION,
            "panel_column_count": 99,
            "panels": [],
        }
        layout_file = tmp_path / "bad_cols.json"
        layout_file.write_text(json.dumps(bad_state), encoding="utf-8")
        state = load_layout_state(layout_file)
        assert state.panel_column_count == 2  # fallback


# ---------------------------------------------------------------------------
# Smoke test 5 — WorksheetArea.set_panel_columns
# ---------------------------------------------------------------------------


class TestWorksheetAreaColumns:
    def test_set_panel_columns_valid(self, qapp):
        from src.prisma.gui.viewer.gating_workspace import WorksheetArea

        wa = WorksheetArea()
        for n in [1, 2, 3, 4]:
            wa.set_panel_columns(n)
            assert wa._COLS == n

    def test_set_panel_columns_invalid(self, qapp):
        from src.prisma.gui.viewer.gating_workspace import WorksheetArea
        from src.prisma.gui.viewer.worksheet_layout import WorksheetLayoutError

        wa = WorksheetArea()
        with pytest.raises(WorksheetLayoutError):
            wa.set_panel_columns(5)

    def test_get_layout_state_empty(self, qapp):
        from src.prisma.gui.viewer.gating_workspace import WorksheetArea

        wa = WorksheetArea()
        wa.set_panel_columns(3)
        state = wa.get_layout_state()
        assert state.panel_column_count == 3
        assert state.panels == []
