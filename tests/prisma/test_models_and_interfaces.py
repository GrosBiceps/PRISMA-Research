"""
tests/prisma/test_models_and_interfaces.py

Tests unitaires + exemple d'utilisation pour :
  - src/models/sample.py       → Sample
  - src/models/experiment.py   → Experiment
  - src/core/interfaces.py     → INodeProcessor, IAnalysisStrategy, IExportStrategy

Exécution : pytest tests/prisma/test_models_and_interfaces.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Résolution du PYTHONPATH si lancé directement
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from prisma.core.models_legacy.sample import Sample, SampleValidationError, ChannelMeta
from prisma.core.models_legacy.experiment import Experiment, ExperimentError

# interfaces.py importe ses modèles via le package prisma — import direct suffit
from prisma.core.interfaces import (
    INodeProcessor,
    IAnalysisStrategy,
    IExportStrategy,
)


# ===========================================================================
# Fixtures
# ===========================================================================

def _make_sample(n: int = 100, n_channels: int = 8, sid: str | None = None) -> Sample:
    """Crée un Sample synthétique avec n événements et n_channels canaux."""
    channels = [f"CD{i+3}" for i in range(n_channels)]
    data = pd.DataFrame(
        np.random.default_rng(0).standard_normal((n, n_channels)),
        columns=channels,
    )
    return Sample(sample_id=sid, events=data)


# ===========================================================================
# Tests Sample
# ===========================================================================

class TestSampleBasics:
    def test_n_events(self):
        s = _make_sample(50)
        assert s.n_events == 50

    def test_channels(self):
        s = _make_sample(10, 4)
        assert s.channels == ["CD3", "CD4", "CD5", "CD6"]

    def test_get_data_all(self):
        s = _make_sample(30)
        df = s.get_data()
        assert df.shape == (30, 8)

    def test_get_data_subset(self):
        s = _make_sample(30)
        df = s.get_data(["CD3", "CD4"])
        assert list(df.columns) == ["CD3", "CD4"]

    def test_get_data_missing_channel_raises(self):
        s = _make_sample(10)
        with pytest.raises(KeyError, match="GHOST"):
            s.get_data(["GHOST"])


class TestSampleMasks:
    def test_set_and_get_mask(self):
        s = _make_sample(100)
        mask = np.zeros(100, dtype=bool)
        mask[:50] = True
        s.set_mask("lymphocytes", mask)
        assert np.array_equal(s.get_mask("lymphocytes"), mask)

    def test_set_mask_wrong_length_raises(self):
        s = _make_sample(100)
        with pytest.raises(SampleValidationError, match="longueur"):
            s.set_mask("bad", np.ones(99, dtype=bool))

    def test_set_mask_wrong_dtype_raises(self):
        s = _make_sample(10)
        with pytest.raises(SampleValidationError, match="bool"):
            s.set_mask("bad", np.ones(10, dtype=int))

    def test_get_missing_mask_raises(self):
        s = _make_sample(10)
        with pytest.raises(KeyError):
            s.get_mask("nonexistent")

    def test_get_pre_gated_no_masks(self):
        s = _make_sample(50)
        df = s.get_pre_gated_data()
        assert len(df) == 50

    def test_get_pre_gated_with_mask(self):
        s = _make_sample(100)
        m = np.array([i % 2 == 0 for i in range(100)])
        s.set_mask("even", m)
        df = s.get_pre_gated_data()
        assert len(df) == 50

    def test_get_pre_gated_multiple_masks_and(self):
        s = _make_sample(100)
        m1 = np.array([i < 60 for i in range(100)])   # 60 True
        m2 = np.array([i % 2 == 0 for i in range(100)])  # 50 True
        s.set_mask("m1", m1)
        s.set_mask("m2", m2)
        # AND: i<60 AND i%2==0 → 30 events
        df = s.get_pre_gated_data()
        assert len(df) == 30


class TestSampleVirtualChannels:
    def test_add_and_access(self):
        s = _make_sample(50)
        data = np.arange(50, dtype=float)
        s.add_virtual_channel("ratio_CD4_CD8", data)
        assert np.array_equal(s.virtual_channels["ratio_CD4_CD8"], data)

    def test_wrong_length_raises(self):
        s = _make_sample(50)
        with pytest.raises(SampleValidationError):
            s.add_virtual_channel("bad", np.ones(49))


class TestSampleEmbeddings:
    def test_add_embedding_2d(self):
        s = _make_sample(100)
        emb = np.random.default_rng(1).standard_normal((100, 2))
        s.add_embedding("umap_2d", emb)
        assert s.embeddings["umap_2d"].shape == (100, 2)

    def test_add_embedding_1d_reshaped(self):
        s = _make_sample(50)
        emb = np.arange(50, dtype=float)
        s.add_embedding("pc1", emb)
        assert s.embeddings["pc1"].shape == (50, 1)

    def test_wrong_n_events_raises(self):
        s = _make_sample(100)
        with pytest.raises(SampleValidationError):
            s.add_embedding("bad", np.zeros((99, 2)))


class TestSampleClustering:
    def test_add_labels_int_cast(self):
        s = _make_sample(60)
        lbls = np.zeros(60, dtype=float)
        s.add_cluster_labels("flowsom_meta20", lbls)
        assert s.cluster_assignments["flowsom_meta20"].dtype == int

    def test_wrong_length_raises(self):
        s = _make_sample(60)
        with pytest.raises(SampleValidationError):
            s.add_cluster_labels("bad", np.ones(59))


class TestSampleSubset:
    def test_subset_events(self):
        s = _make_sample(100)
        mask = np.zeros(100, dtype=bool)
        mask[10:40] = True
        s.set_mask("gate_A", mask)
        sub = s.subset("gate_A")
        assert sub.n_events == 30

    def test_subset_propagates_virtual_channels(self):
        s = _make_sample(100)
        vc = np.arange(100, dtype=float)
        s.add_virtual_channel("ratio", vc)
        mask = np.zeros(100, dtype=bool)
        mask[:25] = True
        s.set_mask("g", mask)
        sub = s.subset("g")
        assert len(sub.virtual_channels["ratio"]) == 25
        assert np.array_equal(sub.virtual_channels["ratio"], vc[:25])

    def test_subset_propagates_embeddings(self):
        s = _make_sample(100)
        emb = np.ones((100, 2))
        s.add_embedding("umap", emb)
        mask = np.ones(100, dtype=bool)
        mask[50:] = False
        s.set_mask("g", mask)
        sub = s.subset("g")
        assert sub.embeddings["umap"].shape == (50, 2)

    def test_subset_clears_masks(self):
        s = _make_sample(100)
        mask = np.ones(100, dtype=bool)
        s.set_mask("g", mask)
        sub = s.subset("g")
        assert sub.masks == {}


class TestSampleValidation:
    def test_validate_clean_sample(self):
        s = _make_sample(50)
        s.validate()  # no exception

    def test_validate_bad_mask_length(self):
        s = _make_sample(50)
        s.masks["bad"] = np.ones(51, dtype=bool)
        with pytest.raises(SampleValidationError):
            s.validate()

    def test_validate_bad_mask_dtype(self):
        s = _make_sample(50)
        s.masks["bad"] = np.ones(50, dtype=int)
        with pytest.raises(SampleValidationError):
            s.validate()

    def test_validate_bad_compensation_shape(self):
        s = _make_sample(50)
        s.compensation_matrix = np.ones((3, 4))
        with pytest.raises(SampleValidationError):
            s.validate()


# ===========================================================================
# Tests Experiment
# ===========================================================================

class TestExperiment:
    def test_add_and_get_sample(self):
        exp = Experiment("AML_cohort")
        s = _make_sample(80, sid="S001")
        exp.add_sample(s)
        retrieved = exp.get_sample("S001")
        assert retrieved is s

    def test_remove_sample(self):
        exp = Experiment("test")
        s = _make_sample(10, sid="X1")
        exp.add_sample(s)
        removed = exp.remove_sample("X1")
        assert removed is s
        assert exp.n_samples == 0

    def test_duplicate_add_raises(self):
        exp = Experiment("test")
        s = _make_sample(10, sid="DUP")
        exp.add_sample(s)
        with pytest.raises(ExperimentError, match="déjà"):
            exp.add_sample(s)

    def test_list_channels_union(self):
        exp = Experiment("test")
        channels_a = ["CD3", "CD4", "CD19"]
        channels_b = ["CD3", "CD8", "CD45"]
        s1 = Sample(sample_id="A", events=pd.DataFrame(np.zeros((5, 3)), columns=channels_a))
        s2 = Sample(sample_id="B", events=pd.DataFrame(np.zeros((5, 3)), columns=channels_b))
        exp.add_sample(s1)
        exp.add_sample(s2)
        assert sorted(exp.list_channels()) == ["CD19", "CD3", "CD4", "CD45", "CD8"]

    def test_common_channels_intersection(self):
        exp = Experiment("test")
        channels_a = ["CD3", "CD4", "CD19"]
        channels_b = ["CD3", "CD19", "CD45"]
        s1 = Sample(sample_id="A", events=pd.DataFrame(np.zeros((5, 3)), columns=channels_a))
        s2 = Sample(sample_id="B", events=pd.DataFrame(np.zeros((5, 3)), columns=channels_b))
        exp.add_sample(s1)
        exp.add_sample(s2)
        assert exp.common_channels() == ["CD19", "CD3"]

    def test_concatenate_pre_gated(self):
        exp = Experiment("test")
        channels = ["CD3", "CD4"]
        s1 = Sample(sample_id="S1", events=pd.DataFrame(np.ones((40, 2)), columns=channels))
        s2 = Sample(sample_id="S2", events=pd.DataFrame(np.zeros((60, 2)), columns=channels))
        exp.add_sample(s1)
        exp.add_sample(s2)
        concat = exp.concatenate_pre_gated(add_batch_columns=True)
        assert len(concat) == 100
        assert "sample_id" in concat.columns

    def test_concatenate_pre_gated_respects_masks(self):
        exp = Experiment("test")
        channels = ["CD3", "CD4"]
        s = Sample(sample_id="S1", events=pd.DataFrame(np.ones((100, 2)), columns=channels))
        mask = np.zeros(100, dtype=bool)
        mask[:30] = True
        s.set_mask("lympho", mask)
        exp.add_sample(s)
        concat = exp.concatenate_pre_gated(add_batch_columns=False)
        assert len(concat) == 30

    def test_summarize(self):
        exp = Experiment("test")
        exp.add_sample(_make_sample(50, sid="A"))
        exp.add_sample(_make_sample(80, sid="B"))
        summary = exp.summarize()
        assert len(summary) == 2
        assert set(summary.columns) >= {"sample_id", "n_events", "n_channels"}

    def test_cohort_metadata_alignment(self):
        exp = Experiment("test")
        s1 = _make_sample(10, sid="P1")
        s2 = _make_sample(10, sid="P2")
        exp.add_sample(s1)
        exp.add_sample(s2)
        meta = pd.DataFrame(
            {"age": [45, 62], "diagnosis": ["AML", "MDS"]},
            index=["P1", "P2"],
        )
        exp.set_cohort_metadata(meta)
        info = exp.get_cohort_info("P1")
        assert info["diagnosis"] == "AML"

    def test_set_cohort_metadata_missing_ids_raises(self):
        exp = Experiment("test")
        exp.add_sample(_make_sample(10, sid="P1"))
        meta = pd.DataFrame({"age": [30]}, index=["OTHER"])
        with pytest.raises(ExperimentError, match="P1"):
            exp.set_cohort_metadata(meta)


# ===========================================================================
# Tests interfaces (smoke tests via implémentations minimales)
# ===========================================================================

class _MockProcessor(INodeProcessor):
    """Processor de test qui ajoute un masque 'all_events' au Sample."""

    @property
    def name(self) -> str:
        return "mock_processor"

    def validate_input(self, target):
        if not isinstance(target, Sample):
            raise TypeError("Attend un Sample.")

    def process(self, target: Sample) -> Sample:
        mask = np.ones(target.n_events, dtype=bool)
        target.set_mask("all_events", mask)
        return target


class _MockAnalysis(IAnalysisStrategy):
    """Stratégie analytique nulle (retourne données normalisées)."""

    _fitted = False

    @property
    def name(self) -> str:
        return "mock_zscore"

    def fit(self, data: np.ndarray, **kwargs) -> "_MockAnalysis":
        self._mean = data.mean(axis=0)
        self._std = data.std(axis=0) + 1e-8
        self._fitted = True
        return self

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return (data - self._mean) / self._std

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def get_params(self) -> dict:
        return {"name": self.name}


class _MockExport(IExportStrategy):
    @property
    def name(self) -> str:
        return "mock_csv"

    def export(self, source, output_path, **kwargs):
        return Path(output_path)

    def supported_types(self):
        return [Sample, pd.DataFrame]


class TestInterfaces:
    def test_node_processor_run(self):
        s = _make_sample(50)
        proc = _MockProcessor()
        result = proc.run(s)
        assert "all_events" in result.masks

    def test_node_processor_validate_wrong_type_raises(self):
        proc = _MockProcessor()
        with pytest.raises(TypeError):
            proc.run("not_a_sample")

    def test_analysis_strategy_fit_transform(self):
        strategy = _MockAnalysis()
        data = np.random.default_rng(2).standard_normal((200, 6))
        result = strategy.fit_transform(data)
        assert result.shape == (200, 6)
        assert strategy.is_fitted

    def test_export_strategy_can_export(self):
        exp = _MockExport()
        assert exp.can_export(_make_sample(10))
        assert exp.can_export(pd.DataFrame())
        assert not exp.can_export(np.zeros((5, 5)))


# ===========================================================================
# Exemple d'utilisation (exécutable seul)
# ===========================================================================

def usage_example() -> None:
    """
    Exemple complet : création Sample → gating → embedding → clustering
    → constitution Experiment → concatenation batch.
    """
    print("=" * 60)
    print("EXEMPLE D'UTILISATION — Sample + Experiment + Interfaces")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # 1. Création d'un Sample
    channels = ["FSC-A", "SSC-A", "CD45", "CD3", "CD4", "CD8", "CD19", "CD56"]
    events = pd.DataFrame(rng.standard_normal((5_000, 8)), columns=channels)

    s = Sample(
        sample_id="PATIENT_001",
        events=events,
        per_sample_metadata={"patient": "P001", "diagnosis": "AML", "date": "2024-03-15"},
    )
    print(s)

    # 2. Gating : garder les lymphocytes (CD45 > 0)
    lympho_mask = s.events["CD45"].values > 0
    s.set_mask("lymphocytes", lympho_mask)
    print(f"  Après gate 'lymphocytes': {lympho_mask.sum():,} / {s.n_events:,} événements")

    # 3. Canal virtuel : ratio CD4/CD8
    cd4 = s.events["CD4"].values
    cd8 = s.events["CD8"].values
    ratio = cd4 / (np.abs(cd8) + 1e-6)
    s.add_virtual_channel("CD4_CD8_ratio", ratio)

    # 4. Embedding (PCA simulée sur tous les événements — dim N_events × 2)
    all_data = s.get_data(channels[2:]).values  # N_events × 6
    cov = np.cov(all_data, rowvar=False)
    _, vecs = np.linalg.eigh(cov)
    emb_2d = all_data @ vecs[:, -2:]            # N_events × 2
    s.add_embedding("pca_2d", emb_2d)
    print(f"  Embedding 'pca_2d': {s.embeddings['pca_2d'].shape}")

    # 5. Clustering (labels aléatoires)
    labels = rng.integers(0, 20, size=s.n_events)
    s.add_cluster_labels("flowsom_meta20", labels)
    print(f"  Clusters uniques: {np.unique(labels).size}")

    # 6. Validation
    s.validate()
    print("  Validation OK")

    # 7. Subset
    sub = s.subset("lymphocytes")
    print(f"  Subset 'lymphocytes': {sub}")

    # 8. Experiment
    print()
    s2 = Sample(
        sample_id="PATIENT_002",
        events=pd.DataFrame(rng.standard_normal((3_000, 8)), columns=channels),
        per_sample_metadata={"patient": "P002", "diagnosis": "MDS"},
    )

    exp = Experiment(name="AML_MDS_Cohort", description="Cohorte pilote 2024")
    exp.add_sample(s)
    exp.add_sample(s2)

    # Ajout métadonnées de cohorte
    meta = pd.DataFrame(
        {"age": [54, 67], "treatment": ["HDAC_inh", "azacitidine"]},
        index=["PATIENT_001", "PATIENT_002"],
    )
    exp.set_cohort_metadata(meta)

    print(exp)
    print(exp.summarize().to_string(index=False))
    print(f"  Canaux communs: {exp.common_channels()}")

    # 9. Concatenation batch
    batch = exp.concatenate_pre_gated(channels=["CD3", "CD4", "CD8"], add_batch_columns=True)
    print(f"  Batch pre-gaté: {batch.shape} — colonnes: {list(batch.columns)}")

    # 10. INodeProcessor en action
    proc = _MockProcessor()
    proc.run(s2)
    print(f"  Processor 'mock_processor' applique -> masks de S2: {list(s2.masks)}")

    print()
    print("OK — exemple terminé sans erreur.")


if __name__ == "__main__":
    usage_example()
