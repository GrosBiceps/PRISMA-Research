"""
smoke_test.py — Tests minimaux PRISMA Research.

Vérifie que les modules fondateurs s'importent et fonctionnent
sans aucune dépendance optionnelle (flowsom, cuml, etc.).

Run: pytest tests/prisma/smoke_test.py -x -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_sample_creation():
    from prisma.core.models import Sample

    df = pd.DataFrame({"CD3": [1.0, 2.0], "CD19": [3.0, 4.0]})
    s = Sample(name="test.fcs", path="/tmp/test.fcs", condition="healthy", data=df, n_cells_raw=2)
    assert s.n_cells == 2
    assert s.markers == ["CD3", "CD19"]
    assert s.matrix.shape == (2, 2)


def test_sample_filter():
    from prisma.core.models import Sample

    df = pd.DataFrame({"CD3": [1.0, 2.0, 3.0]})
    s = Sample(name="t.fcs", path="/tmp/t.fcs", condition="x", data=df, n_cells_raw=3)
    mask = np.array([True, False, True])
    s2 = s.filter(mask)
    assert s2.n_cells == 2


def test_sample_downsample():
    from prisma.core.models import Sample

    df = pd.DataFrame({"CD3": np.random.rand(100)})
    s = Sample(name="t.fcs", path="/tmp/t.fcs", condition="x", data=df, n_cells_raw=100)
    s2 = s.downsample(10, seed=42)
    assert s2.n_cells == 10


def test_experiment():
    from prisma.core.models import Experiment, Sample

    exp = Experiment(name="Test Experiment")
    df = pd.DataFrame({"CD3": [1.0]})
    s = Sample(name="a.fcs", path="/tmp/a.fcs", condition="healthy", data=df)
    exp.add_sample(s)
    assert exp.n_samples == 1
    assert "healthy" in exp.conditions


def test_run_metadata(tmp_path):
    from prisma.core.models import RunMetadata

    meta = RunMetadata.start("test_pipeline", seed=42, parameters={"k": 1})
    assert meta.status == "running"
    meta.finish(status="success")
    assert meta.status == "success"
    saved = meta.save(tmp_path)
    assert saved.exists()
    loaded = RunMetadata.load(saved)
    assert loaded.run_id == meta.run_id


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def test_session_singleton():
    from prisma.core.session import SessionManager

    s1 = SessionManager.instance()
    s2 = SessionManager.instance()
    assert s1 is s2


def test_session_experiment():
    from prisma.core.models import Experiment
    from prisma.core.session import SessionManager

    session = SessionManager.instance()
    exp = Experiment(name="Smoke")
    session.set_experiment(exp)
    assert session.experiment is exp


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_strategies():
    import prisma.strategies  # déclenche auto-registration
    from prisma.core.registry import StrategyRegistry

    names = StrategyRegistry.list_dimreduc()
    assert "umap" in names
    assert "tsne" in names
    assert "phate" in names
    assert "spectral" in names
    assert "flowsom" in StrategyRegistry.list_clustering()


def test_phate_strategy_fallback_embedding():
    from prisma.strategies.phate_strategy import PHATEStrategy, PHATEParams

    data = np.random.rand(24, 6).astype(np.float32)
    strategy = PHATEStrategy()
    embedding = strategy.fit_transform(data, PHATEParams(n_components=2, knn=5, decay=40))

    assert embedding.shape == (24, 2)


def test_harmony_alignment_indices_are_ssc_only():
    from analysis.batch import _ssc_alignment_indices

    feature_names = ["FSC-A", "SSC-A", "CD34", "SSC-H", "CD45", "FSC-H"]
    assert _ssc_alignment_indices(feature_names) == [1, 3]


# ---------------------------------------------------------------------------
# Strategies — spectral (pas de dépendance GPU)
# ---------------------------------------------------------------------------


def test_spectral_unmixing():
    from prisma.strategies.spectral_strategy import SpectralUnmixingStrategy, SpectralParams

    n_cells, n_det, n_fluoro = 50, 8, 4
    A = np.random.rand(n_det, n_fluoro)
    data = np.random.rand(n_cells, n_det)
    params = SpectralParams(reference_matrix=A)
    strat = SpectralUnmixingStrategy()
    out = strat.fit_transform(data, params)
    assert out.shape == (n_cells, n_fluoro)
    assert (out >= 0).all()


# ---------------------------------------------------------------------------
# Pipeline base
# ---------------------------------------------------------------------------


def test_pipeline_runner(tmp_path):
    from prisma.core.models import Experiment, RunMetadata, Sample
    from prisma.pipeline.base import PipelineContext, PipelineRunner

    class NoopStep:
        name = "noop"

        def validate(self, ctx):
            pass

        def run(self, ctx):
            return ctx

    df = pd.DataFrame({"CD3": [1.0]})
    exp = Experiment(name="Smoke")
    exp.add_sample(Sample(name="x.fcs", path="/tmp/x.fcs", condition="c", data=df))

    meta = RunMetadata.start("smoke", seed=42, parameters={})
    ctx = PipelineContext(
        experiment=exp,
        run_meta=meta,
        output_dir=tmp_path,
        params={},
        seed=42,
    )
    runner = PipelineRunner([NoopStep()])
    ctx_out = runner.run(ctx)
    assert ctx_out.run_meta.status == "success"


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


def test_seed():
    from prisma.utils.seed import set_global_seed

    set_global_seed(0)
    a = np.random.rand(5)
    set_global_seed(0)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_logging_setup():
    from prisma.utils.logging_config import setup_logging

    setup_logging(level=20)  # INFO — ne doit pas lever
