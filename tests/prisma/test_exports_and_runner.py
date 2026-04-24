"""Tests de non-régression: modèles PRISMA, runner et exports analytiques."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from exports.export_manager import ExportManager
from exports.heatmaps import build_cohort_export_tables
from exports.run_serializer import serialize_full_run
from prisma.core.models import Experiment, RunMetadata, Sample
from prisma.core.registry import StrategyRegistry
from prisma.pipeline.base import PipelineContext, PipelineRunner


def _synthetic_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "cluster": [0, 0, 1, 0, 1, 1],
            "CD34": [1.0, 1.2, 2.4, 0.9, 2.0, 2.2],
            "CD38": [3.0, 3.1, 1.2, 3.3, 1.0, 1.1],
        }
    )


def test_sample_and_experiment_minimal_contract() -> None:
    data = pd.DataFrame({"CD34": [1.0, 2.0], "CD38": [3.0, 4.0]})
    sample = Sample(
        name="sample_a.fcs",
        path="/tmp/sample_a.fcs",
        condition="patient",
        data=data,
        n_cells_raw=2,
    )
    experiment = Experiment(name="mvp")
    experiment.add_sample(sample)

    assert sample.n_cells == 2
    assert sample.markers == ["CD34", "CD38"]
    assert experiment.n_samples == 1
    assert experiment.conditions == ["patient"]


def test_strategy_registry_has_core_entries() -> None:
    import prisma.strategies  # noqa: F401

    dim_reduc = StrategyRegistry.list_dimreduc()
    cluster = StrategyRegistry.list_clustering()

    assert "umap" in dim_reduc
    assert "spectral" in dim_reduc
    assert "flowsom" in cluster


def test_pipeline_runner_failure_marks_metadata(tmp_path: Path) -> None:
    class FailingStep:
        name = "failing_step"

        def validate(self, ctx: PipelineContext) -> None:
            return None

        def run(self, ctx: PipelineContext) -> PipelineContext:
            raise ValueError("synthetic failure")

    sample = Sample(
        name="x.fcs",
        path="/tmp/x.fcs",
        condition="healthy",
        data=pd.DataFrame({"CD3": [1.0]}),
        n_cells_raw=1,
    )
    exp = Experiment(name="pipeline")
    exp.add_sample(sample)

    meta = RunMetadata.start("runner_test", seed=42, parameters={"mode": "test"})
    ctx = PipelineContext(
        experiment=exp,
        run_meta=meta,
        output_dir=tmp_path,
        params={},
        seed=42,
    )

    runner = PipelineRunner([FailingStep()])
    try:
        runner.run(ctx)
        assert False, "PipelineRunner devait lever RuntimeError"
    except RuntimeError:
        pass

    assert ctx.run_meta.status == "failed"
    assert "synthetic failure" in (ctx.run_meta.error or "")


def test_export_manager_and_heatmap_outputs(tmp_path: Path) -> None:
    events = _synthetic_events()
    tables = build_cohort_export_tables(events)

    manager = ExportManager(tmp_path)
    artifacts = manager.export_tables(tables, export_parquet=True)
    heatmap_path = manager.export_heatmap_png(tables["cluster_marker_mfi"], "heatmap_mfi")
    meta_path = manager.export_json({"run": "ok"}, "metadata")

    assert "cluster_marker_mfi" in artifacts
    assert Path(artifacts["cluster_marker_mfi"]["csv"]).exists()
    assert heatmap_path.exists()
    assert meta_path.exists()


def test_full_run_serializer(tmp_path: Path) -> None:
    meta = RunMetadata.start("exports_pipeline", seed=7, parameters={"alpha": 0.1})
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    result = serialize_full_run(
        output_dir=tmp_path,
        run_metadata=meta,
        config={"pipeline": {"name": "mvp"}},
        run_context={"cohort": "AML", "n_samples": 2},
        modules_enabled=["spectral", "flowsom", "exports"],
        seeds={"global": 7, "umap": 42},
        artifacts=[tables_dir / "abundance.csv"],
    )

    bundle = result["bundle"]
    config = result["config"]

    assert bundle.exists()
    assert config.exists()

    bundle_data = json.loads(bundle.read_text(encoding="utf-8"))
    assert "run_metadata" in bundle_data
    assert "software_versions" in bundle_data
    assert "modules_enabled" in bundle_data
