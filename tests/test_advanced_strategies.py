"""
tests/test_advanced_strategies.py — Tests unitaires pour stratégies cohorte avancées.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.advanced_strategies import (
    AdvancedCohortExecutor,
    HarmonyStrategy,
    HarmonyStrategyParams,
    PHATEStrategy,
    PHATEStrategyParams,
    PhenoGraphStrategy,
    PhenoGraphStrategyParams,
    build_advanced_executor_from_config,
    run_advanced_cohort_from_config,
)
from config.pipeline_config import PipelineConfig
from models.experiment import Experiment
from models.sample import Sample


def _make_sample(sample_id: str, shift: float, seed: int) -> Sample:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(600, 6)).astype(np.float32)
    x[:, :2] += shift
    df = pd.DataFrame(x, columns=[f"C{i}" for i in range(6)])
    s = Sample(sample_id=sample_id, events=df)
    mask = np.ones(s.n_events, dtype=bool)
    mask[:30] = False
    s.set_mask("qc_pass", mask)
    return s


def _experiment() -> Experiment:
    exp = Experiment("cohort_test")
    s1 = _make_sample("S1", shift=0.0, seed=1)
    s2 = _make_sample("S2", shift=1.5, seed=2)
    exp.add_sample(s1)
    exp.add_sample(s2)
    exp.set_cohort_metadata(pd.DataFrame({"batch": ["A", "B"]}, index=[s1.sample_id, s2.sample_id]))
    return exp


def test_harmony_strategy_mapping_back():
    exp = _experiment()
    strat = HarmonyStrategy()
    exp = strat.run_experiment(
        exp,
        params=HarmonyStrategyParams(
            channels=[f"C{i}" for i in range(6)],
            source_masks=["qc_pass"],
            batch_key="batch",
            store_corrected_channels=True,
            corrected_channel_prefix="h_",
        ),
    )

    for s in exp.samples:
        assert "h_C0" in s.virtual_channels
        arr = s.virtual_channels["h_C0"]
        assert arr.shape == (s.n_events,)
        assert np.isnan(arr[~s.get_mask("qc_pass")]).all()

    out = getattr(exp, "analysis_results", {})
    assert "harmony" in out
    assert "corrected_matrix" in out["harmony"]


def test_executor_chain_phate_and_phenograph():
    exp = _experiment()
    executor = (
        AdvancedCohortExecutor()
        .register(HarmonyStrategy())
        .register(PHATEStrategy())
        .register(PhenoGraphStrategy())
    )

    exp = executor.run(
        exp,
        harmony_params=HarmonyStrategyParams(
            channels=[f"C{i}" for i in range(6)],
            source_masks=["qc_pass"],
            batch_key="batch",
        ),
        phate_params=PHATEStrategyParams(
            channels=[f"C{i}" for i in range(6)],
            source_masks=["qc_pass"],
            batch_key="batch",
            embedding_name="phate_2d",
            n_components=2,
        ),
        phenograph_params=PhenoGraphStrategyParams(
            channels=[f"C{i}" for i in range(6)],
            source_masks=["qc_pass"],
            batch_key="batch",
            clustering_name="phg",
            k=20,
        ),
    )

    for s in exp.samples:
        assert "phate_2d" in s.embeddings
        assert s.embeddings["phate_2d"].shape == (s.n_events, 2)
        assert "phg" in s.cluster_assignments
        assert s.cluster_assignments["phg"].shape == (s.n_events,)

    out = getattr(exp, "analysis_results", {})
    assert "phate" in out
    assert "phenograph" in out


def test_config_driven_advanced_execution():
    exp = _experiment()
    cfg = PipelineConfig()
    cfg._extra["advanced_cohort_analysis"] = {
        "enabled": True,
        "channels": [f"C{i}" for i in range(6)],
        "source_masks": ["qc_pass"],
        "batch_key": "batch",
        "harmony": {
            "enabled": True,
            "store_corrected_channels": True,
            "corrected_channel_prefix": "hc_",
        },
        "phate": {
            "enabled": True,
            "embedding_name": "phate_cfg",
            "n_components": 2,
        },
        "phenograph": {
            "enabled": True,
            "clustering_name": "phg_cfg",
            "k": 20,
            "min_rare_size": 20,
        },
    }

    executor, kwargs = build_advanced_executor_from_config(cfg)
    assert executor.strategy_names == ["harmony", "phate", "phenograph"]
    assert "harmony_params" in kwargs
    assert "phate_params" in kwargs
    assert "phenograph_params" in kwargs

    exp = run_advanced_cohort_from_config(exp, cfg)
    for s in exp.samples:
        assert "hc_C0" in s.virtual_channels
        assert "phate_cfg" in s.embeddings
        assert "phg_cfg" in s.cluster_assignments
