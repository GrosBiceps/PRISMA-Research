"""
smoke_advanced_cohort.py — Démo synthétique des stratégies cohorte avancées.

Chaîne exécutée:
1) Harmony (alignement batch)
2) PHATE (trajectoires)
3) PhenoGraph/fallback (rares populations)

Run:
    cd "PRISMA Research"
    python tests/smoke_advanced_cohort.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Stubs legacy requis car analysis/__init__.py importe des modules legacy.
import tests.conftest as _stubs  # noqa: F401

from analysis.advanced_strategies import (  # noqa: E402
    AdvancedCohortExecutor,
    HarmonyStrategy,
    HarmonyStrategyParams,
    PHATEStrategy,
    PHATEStrategyParams,
    PhenoGraphStrategy,
    PhenoGraphStrategyParams,
)
from models.experiment import Experiment  # noqa: E402
from models.sample import Sample  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s — %(message)s")
logger = logging.getLogger("smoke_advanced_cohort")


def _make_sample(
    sample_id: str,
    batch: str,
    n_events: int,
    n_channels: int,
    seed: int,
) -> Sample:
    rng = np.random.default_rng(seed)
    channels = [f"M{i}" for i in range(n_channels)]

    # Population majoritaire
    major = rng.normal(0.0, 1.0, size=(int(n_events * 0.92), n_channels))

    # Population intermédiaire
    inter = rng.normal(2.5, 0.8, size=(int(n_events * 0.06), n_channels))

    # Population rare (objectif de détection)
    rare = rng.normal(5.0, 0.35, size=(n_events - len(major) - len(inter), n_channels))

    x = np.vstack([major, inter, rare]).astype(np.float32)

    # Effet batch simulé sur un sous-ensemble de canaux
    if batch == "B":
        x[:, :4] += 1.5

    df = pd.DataFrame(x, columns=channels)
    sample = Sample(sample_id=sample_id, events=df)

    # Masque QC simulé
    qc = np.ones(sample.n_events, dtype=bool)
    bad_idx = rng.choice(sample.n_events, size=int(0.05 * sample.n_events), replace=False)
    qc[bad_idx] = False
    sample.set_mask("qc_pass", qc)

    return sample


def build_fake_experiment() -> Experiment:
    exp = Experiment(name="synthetic_cohort")

    s1 = _make_sample("S1", "A", n_events=1500, n_channels=10, seed=1)
    s2 = _make_sample("S2", "A", n_events=1300, n_channels=10, seed=2)
    s3 = _make_sample("S3", "B", n_events=1400, n_channels=10, seed=3)

    exp.add_sample(s1)
    exp.add_sample(s2)
    exp.add_sample(s3)

    exp.set_cohort_metadata(
        pd.DataFrame(
            {
                "batch": ["A", "A", "B"],
                "condition": ["ctrl", "ctrl", "patient"],
            },
            index=[s1.sample_id, s2.sample_id, s3.sample_id],
        )
    )
    return exp


def main() -> None:
    exp = build_fake_experiment()

    executor = (
        AdvancedCohortExecutor()
        .register(HarmonyStrategy())
        .register(PHATEStrategy())
        .register(PhenoGraphStrategy())
    )

    exp = executor.run(
        exp,
        harmony_params=HarmonyStrategyParams(
            channels=[f"M{i}" for i in range(10)],
            source_masks=["qc_pass"],
            batch_key="batch",
            store_corrected_channels=True,
            corrected_channel_prefix="harmony_",
        ),
        phate_params=PHATEStrategyParams(
            channels=[f"M{i}" for i in range(10)],
            source_masks=["qc_pass"],
            batch_key="batch",
            embedding_name="phate_2d",
            n_components=2,
            use_harmony_if_available=True,
        ),
        phenograph_params=PhenoGraphStrategyParams(
            channels=[f"M{i}" for i in range(10)],
            source_masks=["qc_pass"],
            batch_key="batch",
            clustering_name="phenograph",
            k=30,
            min_rare_size=40,
        ),
    )

    # Vérifications mapping retour par sample
    for sample in exp.samples:
        assert "phate_2d" in sample.embeddings
        assert "phenograph" in sample.cluster_assignments
        assert "harmony_M0" in sample.virtual_channels

        emb = sample.embeddings["phate_2d"]
        labels = sample.cluster_assignments["phenograph"]

        assert emb.shape == (sample.n_events, 2)
        assert labels.shape == (sample.n_events,)

        qc = sample.get_mask("qc_pass")
        assert np.all(labels[~qc] == -1)
        assert np.all(np.isnan(emb[~qc, 0]))

    out = getattr(exp, "analysis_results", {})
    assert "harmony" in out
    assert "phate" in out
    assert "phenograph" in out

    logger.info("=" * 64)
    logger.info("SMOKE ADVANCED COHORT PASSED")
    logger.info("Strategies: %s", executor.strategy_names)
    logger.info("Harmony backend: %s", out["harmony"]["backend"])
    logger.info("PHATE backend: %s", out["phate"]["backend"])
    logger.info("PhenoGraph backend: %s", out["phenograph"]["backend"])
    logger.info("Rare clusters: %s", out["phenograph"]["rare_cluster_ids"])
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
