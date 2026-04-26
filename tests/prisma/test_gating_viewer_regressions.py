from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.prisma.gui.viewer import gating_engine as ge_mod
from src.prisma.gui.viewer import gating_workspace as gw_mod
from src.prisma.gui.viewer import interactive_canvas as ic_mod


class _DummyCanvas:
    def __init__(self) -> None:
        self.last_2d: tuple[pd.DataFrame, str, str, bool] | None = None
        self.reloaded_with = None

    def set_data_2d(self, df: pd.DataFrame, x_ch: str, y_ch: str, density_coloring: bool = False):
        self.last_2d = (df, x_ch, y_ch, density_coloring)

    def set_data_1d(self, df: pd.DataFrame, x_ch: str):
        self.last_2d = (df, x_ch, "", False)

    def reload_gate_overlays_from_engine(self, engine):
        self.reloaded_with = engine


class _DummyCombo:
    def __init__(self, text: str = "", data: str | None = None) -> None:
        self._text = text
        self._data = data

    def currentText(self) -> str:
        return self._text

    def currentData(self):
        return self._data


class _DummyCheck:
    def __init__(self, checked: bool = False) -> None:
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked


class _FakeColumn:
    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def to_numpy(self, dtype=np.float32):
        return self._values.astype(dtype)


class _Fake2DLikeDataFrame:
    """Minimal stub that returns (N, 1) arrays for channel extraction."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.columns = ["X", "Y"]
        self._x = _FakeColumn(x.reshape(-1, 1))
        self._y = _FakeColumn(y.reshape(-1, 1))

    def __getitem__(self, key: str):
        if key == "X":
            return self._x
        if key == "Y":
            return self._y
        raise KeyError(key)


class _ScatterCapture:
    last_x = None
    last_y = None

    def __init__(self, x, y, size, pen, brush) -> None:
        _ScatterCapture.last_x = x
        _ScatterCapture.last_y = y


class _DummyEngine:
    def __init__(self) -> None:
        self.called_gate_node = None

    def get_population_df(
        self,
        gate_node,
        sample_id=None,
        transform_id=None,
        comp_matrix_id=None,
    ) -> pd.DataFrame:
        self.called_gate_node = gate_node
        return pd.DataFrame(
            {
                "FL1-A": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "FL2-A": np.array([4.0, 5.0, 6.0], dtype=np.float32),
            }
        )

    def get_children(self, gate_name, gate_path=None):
        return []


def test_set_data_2d_accepts_n_by_1_arrays(monkeypatch) -> None:
    """Non-régression: un channel retourné en (N,1) doit être aplati en (N,)."""
    monkeypatch.setattr(ic_mod.pg, "ScatterPlotItem", _ScatterCapture)
    monkeypatch.setattr(ic_mod.pg, "mkColor", lambda c: c)

    # Instance sans initialiser PlotWidget Qt (test purement logique).
    canvas = ic_mod.InteractiveGatingCanvas.__new__(ic_mod.InteractiveGatingCanvas)
    canvas._x_channel = ""
    canvas._y_channel = ""
    canvas._xdata = np.array([])
    canvas._ydata = np.array([])
    canvas._scatter = None
    canvas.clear_data = lambda: None
    canvas.addItem = lambda _item: None
    canvas.setLabel = lambda *_args, **_kwargs: None
    canvas.autoRange = lambda: None
    canvas._compute_density_colors = lambda x, y: [None] * len(x)

    x = np.arange(10, dtype=np.float32)
    y = np.arange(10, dtype=np.float32) * 2.0
    df_like = _Fake2DLikeDataFrame(x, y)

    ic_mod.InteractiveGatingCanvas.set_data_2d(canvas, df_like, "X", "Y", density_coloring=False)

    assert _ScatterCapture.last_x is not None
    assert _ScatterCapture.last_y is not None
    assert _ScatterCapture.last_x.ndim == 1
    assert _ScatterCapture.last_y.ndim == 1
    assert _ScatterCapture.last_x.shape == (10,)
    assert _ScatterCapture.last_y.shape == (10,)


def test_refresh_canvas_keeps_active_population_on_axis_change() -> None:
    """Non-régression: le refresh axes doit rester sur la gate active (pas Root)."""
    ws = gw_mod.PrismaGatingWorkspace.__new__(gw_mod.PrismaGatingWorkspace)
    ws._engine = _DummyEngine()
    ws._canvas = _DummyCanvas()

    ws._combo_x = _DummyCombo(text="CD45 (FL1-A)", data="FL1-A")
    ws._combo_y = _DummyCombo(text="CD34 (FL2-A)", data="FL2-A")
    ws._combo_transform = _DummyCombo(text="— Brut —", data=None)
    ws._combo_comp = _DummyCombo(text="— Non compensé —", data=None)

    ws._chk_density = _DummyCheck(False)
    ws._chk_overlay = _DummyCheck(False)

    ws._current_gate_node = ("Lymphocytes", "root")
    ws._show_error = lambda _msg: (_ for _ in ()).throw(AssertionError("Unexpected UI error"))
    ws._show_overlay_children = lambda *_args, **_kwargs: None

    # Utilise les helpers réels de la classe.
    ws._current_sample_id = lambda: "sample_1"
    ws._selected_transform = lambda: None
    ws._selected_comp = lambda: None

    gw_mod.PrismaGatingWorkspace._on_axes_changed(ws, 0)

    assert ws._engine.called_gate_node == ("Lymphocytes", "root")
    assert ws._canvas.last_2d is not None
    assert ws._canvas.last_2d[1] == "FL1-A"
    assert ws._canvas.last_2d[2] == "FL2-A"


def test_apply_spillover_matrix_adds_comp_matrix(monkeypatch) -> None:
    """Non-régression: un SPILL metadata doit créer une comp_matrix en session."""

    class _FakeSession:
        def __init__(self) -> None:
            self.added = []

        def add_comp_matrix(self, matrix_id, matrix) -> None:
            self.added.append((matrix_id, matrix))

    class _FakeSample:
        def get_metadata(self):
            return {"$SPILL": "spill-raw-content"}

    class _FakeBackend:
        def get_sample(self, _sid):
            return _FakeSample()

    class _FakeFk:
        class Matrix:
            def __init__(self, raw):
                self.raw = raw

    fake_session = _FakeSession()

    engine = ge_mod.PrismaFlowEngine.__new__(ge_mod.PrismaFlowEngine)
    engine.log = SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None
    )
    engine._backend = _FakeBackend()
    engine._comp_matrix_ids = []
    engine._active_sample_id = "sample_a"

    engine._is_workspace = lambda: False
    engine._as_session = lambda: fake_session
    engine._require_active_sample = lambda: "sample_a"

    monkeypatch.setattr(ge_mod, "fk", _FakeFk)

    matrix_id = ge_mod.PrismaFlowEngine.apply_spillover_matrix(engine)

    assert matrix_id == "spill_sample_a"
    assert engine._comp_matrix_ids == ["spill_sample_a"]
    assert len(fake_session.added) == 1
    assert fake_session.added[0][0] == "spill_sample_a"
    assert isinstance(fake_session.added[0][1], _FakeFk.Matrix)
    assert fake_session.added[0][1].raw == "spill-raw-content"


def test_normalize_dataframe_channels_prefers_pnn_tuple_first_element() -> None:
    """Non-régression: les colonnes tuple FlowKit doivent être normalisées en Pnn."""
    engine = ge_mod.PrismaFlowEngine.__new__(ge_mod.PrismaFlowEngine)

    df = pd.DataFrame(
        {
            ("FL1-A", "CD45"): np.array([1.0, 2.0], dtype=np.float32),
            ("FL2-A", "CD34"): np.array([3.0, 4.0], dtype=np.float32),
        }
    )

    normalized = ge_mod.PrismaFlowEngine._normalize_dataframe_channels(engine, df)

    assert list(normalized.columns) == ["FL1-A", "FL2-A"]
