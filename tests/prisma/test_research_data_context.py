from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from config.pipeline_config import PipelineConfig
from prisma.pipeline.research_executor import ResearchPipelineExecutor


class _DummyAdata:
    def __init__(self, name: str, df: pd.DataFrame) -> None:
        self._df = df
        self.obs = pd.DataFrame({"file_origin": [name] * len(df)})

    def to_df(self) -> pd.DataFrame:
        return self._df.copy()


def test_pipeline_config_gating_defaults_and_helpers() -> None:
    cfg = PipelineConfig()
    assert cfg.gating_workspace_path is None
    assert cfg.target_population == "Root"
    assert cfg.uses_root_population() is True
    assert cfg.has_gating_context() is False


def test_load_analysis_dataframe_requires_workspace_for_non_root() -> None:
    cfg = PipelineConfig()
    cfg.target_population = "Lymphocytes"
    cfg.gating_workspace_path = None

    executor = ResearchPipelineExecutor()
    with pytest.raises(ValueError, match="gating_workspace_path"):
        executor._load_analysis_dataframe(cfg, files=[])


def test_load_analysis_dataframe_full_file_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = PipelineConfig()

    adata = _DummyAdata("sample_a.fcs", pd.DataFrame({"CD3": [1.0, 2.0], "CD19": [3.0, 4.0]}))

    import prisma.pipeline.research_executor as re_mod

    monkeypatch.setattr(re_mod, "load_fcs_files", lambda files, condition="RUO": [adata])

    executor = ResearchPipelineExecutor()
    result = executor._load_analysis_dataframe(cfg, files=[tmp_path / "sample_a.fcs"])

    assert result.data_context_mode == "full_file"
    assert result.target_population == "Root"
    assert result.input_events == 2
    assert result.selected_events == 2
    assert "__sample__" in result.dataframe.columns
    assert "CD3" in result.marker_columns


def test_load_gated_population_dataframe_from_json_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fcs_path = tmp_path / "sample_a.fcs"
    gml_path = tmp_path / "ctx.gatingml.xml"
    ctx_path = tmp_path / "ctx.json"

    fcs_path.write_text("fake-fcs", encoding="utf-8")
    gml_path.write_text("fake-gml", encoding="utf-8")

    ctx_path.write_text(
        json.dumps(
            {
                "format": "prisma_gating_context_v1",
                "fcs_files": [str(fcs_path)],
                "gatingml_path": str(gml_path),
            }
        ),
        encoding="utf-8",
    )

    fake_module = types.ModuleType("src.gui.viewer.gating_engine")

    class PrismaEngineError(Exception):
        pass

    class PrismaFlowEngine:
        def load_fcs_batch(self, files: list[Path], make_first_active: bool = True) -> None:
            self._files = files

        def load_gml(self, path: Path) -> None:
            self._gml = path

        def analyze(self, use_mp: bool = False) -> None:
            return None

        def get_sample_ids(self) -> list[str]:
            return ["sample_a"]

        def find_gate_paths(self, gate_name: str) -> list[tuple[str, ...]]:
            return [("root",)]

        def get_raw_dataframe(self, sample_id: str | None = None) -> pd.DataFrame:
            return pd.DataFrame({"CD3": [1.0, 2.0, 3.0], "CD19": [1.0, 2.0, 3.0]})

        def get_gate_dataframe(
            self,
            gate_name: str,
            gate_path: tuple[str, ...] | None = None,
            sample_id: str | None = None,
        ) -> pd.DataFrame:
            return pd.DataFrame({("CD3", "unused"): [1.0, 2.0], ("CD19", "unused"): [3.0, 4.0]})

    fake_module.PrismaFlowEngine = PrismaFlowEngine
    fake_module.PrismaEngineError = PrismaEngineError
    monkeypatch.setitem(sys.modules, "src.gui.viewer.gating_engine", fake_module)

    executor = ResearchPipelineExecutor()
    result = executor._load_gated_population_dataframe(
        files=[],
        workspace_path=str(ctx_path),
        target_population="Lymphocytes",
    )

    assert result.data_context_mode == "gated_population"
    assert result.target_population == "Lymphocytes"
    assert result.input_events == 3
    assert result.selected_events == 2
    assert set(result.marker_columns) == {"CD3", "CD19"}
    assert result.gating_workspace_path == str(ctx_path.resolve())
