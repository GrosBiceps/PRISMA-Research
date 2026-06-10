"""
tests/test_analysis_strategies.py — Smoke tests pour le moteur d'analyse.

Vérifie sur données synthétiques :
- downsampling
- UMAPStrategy (CPU)
- TSNEStrategy (sklearn fallback)
- FlowSOMStrategy proxy
- ResearchPipelineExecutor (chaîne complète)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prisma.core.models_legacy.sample import Sample
from clinical_module.legacy_analysis.flowsom_like_strategy import (
    FlowSOMlikeStrategy,
    FlowSOMlikeParams,
    find_optimal_metaclusters,
    _phase1_silhouette_on_codebook as phase1_silhouette_on_codebook,
    _phase3_composite_selection as phase3_composite_selection,
)
from prisma.analysis.downsampling import (
    random_downsample,
    random_downsample_df,
    expand_to_full,
)
from clinical_module.legacy_analysis.strategies import (
    FlowSOMStrategy,
    FlowSOMStrategyParams,
    ResearchPipelineExecutor,
    TSNEStrategy,
    TSNEStrategyParams,
    UMAPStrategy,
    UMAPStrategyParams,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_CELLS = 500
N_CHANNELS = 8
CHANNELS = [f"CD{i}" for i in range(N_CHANNELS)]


def make_sample(n_cells: int = N_CELLS, seed: int = 0) -> Sample:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_cells, N_CHANNELS)).astype(np.float32)
    df = pd.DataFrame(data, columns=CHANNELS)
    return Sample(events=df)


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------

class TestDownsampling:
    def test_no_downsampling_below_threshold(self):
        data = np.random.rand(200, 5).astype(np.float32)
        result = random_downsample(data, max_events=500, seed=42)
        assert not result.was_downsampled
        assert result.data.shape == (200, 5)
        assert len(result.indices) == 200

    def test_downsampling_above_threshold(self):
        data = np.random.rand(1000, 5).astype(np.float32)
        result = random_downsample(data, max_events=300, seed=42)
        assert result.was_downsampled
        assert result.data.shape == (300, 5)
        assert len(result.indices) == 300
        assert result.n_original == 1000

    def test_reproducibility(self):
        data = np.random.rand(1000, 5).astype(np.float32)
        r1 = random_downsample(data, max_events=300, seed=7)
        r2 = random_downsample(data, max_events=300, seed=7)
        np.testing.assert_array_equal(r1.indices, r2.indices)

    def test_expand_1d(self):
        data = np.random.rand(1000, 3).astype(np.float32)
        result = random_downsample(data, max_events=200, seed=0)
        values = np.arange(200, dtype=np.float32)
        full = expand_to_full(values, result, fill_value=np.nan)
        assert full.shape == (1000,)
        assert np.sum(np.isnan(full)) == 800
        np.testing.assert_array_equal(full[result.indices], values)

    def test_expand_2d(self):
        data = np.random.rand(1000, 3).astype(np.float32)
        result = random_downsample(data, max_events=200, seed=0)
        values = np.ones((200, 2), dtype=np.float32)
        full = expand_to_full(values, result, fill_value=np.nan)
        assert full.shape == (1000, 2)

    def test_downsample_df(self):
        df = pd.DataFrame(np.random.rand(600, 4), columns=list("ABCD"))
        result = random_downsample_df(df, max_events=100, seed=1)
        assert result.data.shape == (100, 4)
        assert isinstance(result.data, np.ndarray)


# ---------------------------------------------------------------------------
# UMAPStrategy
# ---------------------------------------------------------------------------

class TestUMAPStrategy:
    def test_basic_run(self):
        sample = make_sample()
        strategy = UMAPStrategy()
        params = UMAPStrategyParams(n_neighbors=10, max_events=10_000)
        result = strategy.run(sample, params=params)
        assert "umap_2d" in result.embeddings
        emb = result.embeddings["umap_2d"]
        assert emb.shape == (N_CELLS, 2)

    def test_channel_selection(self):
        sample = make_sample()
        strategy = UMAPStrategy()
        channels = CHANNELS[:4]
        params = UMAPStrategyParams(channels=channels, n_neighbors=5)
        result = strategy.run(sample, params=params)
        assert "umap_2d" in result.embeddings

    def test_custom_embedding_name(self):
        sample = make_sample()
        strategy = UMAPStrategy()
        params = UMAPStrategyParams(embedding_name="umap_custom", n_neighbors=5)
        result = strategy.run(sample, params=params)
        assert "umap_custom" in result.embeddings

    def test_downsampling_produces_nan_rows(self):
        sample = make_sample(n_cells=300)
        strategy = UMAPStrategy()
        params = UMAPStrategyParams(max_events=100, n_neighbors=10)
        result = strategy.run(sample, params=params)
        emb = result.embeddings["umap_2d"]
        assert emb.shape[0] == 300
        nan_mask = np.isnan(emb[:, 0])
        assert nan_mask.sum() == 200

    def test_empty_sample_raises(self):
        sample = Sample(events=pd.DataFrame(columns=CHANNELS))
        strategy = UMAPStrategy()
        params = UMAPStrategyParams()
        with pytest.raises(ValueError, match="0 événements"):
            strategy.run(sample, params=params)

    def test_with_gating_mask(self):
        sample = make_sample(n_cells=200)
        mask = np.zeros(200, dtype=bool)
        mask[:100] = True
        sample.set_mask("live", mask)
        strategy = UMAPStrategy()
        params = UMAPStrategyParams(n_neighbors=5)
        result = strategy.run(sample, params=params)
        assert "umap_2d" in result.embeddings
        emb = result.embeddings["umap_2d"]
        # Embedding expandé à N_events=200, NaN pour les cellules exclues (100..199)
        assert emb.shape == (200, 2)
        assert np.all(np.isnan(emb[100:, 0]))
        assert np.all(~np.isnan(emb[:100, 0]))


# ---------------------------------------------------------------------------
# TSNEStrategy
# ---------------------------------------------------------------------------

class TestTSNEStrategy:
    def test_basic_run(self):
        sample = make_sample()
        strategy = TSNEStrategy()
        params = TSNEStrategyParams(n_iter=250, perplexity=20.0)
        result = strategy.run(sample, params=params)
        assert "tsne_2d" in result.embeddings
        emb = result.embeddings["tsne_2d"]
        assert emb.shape == (N_CELLS, 2)

    def test_channel_selection(self):
        sample = make_sample()
        strategy = TSNEStrategy()
        params = TSNEStrategyParams(channels=CHANNELS[:5], n_iter=250, perplexity=10.0)
        result = strategy.run(sample, params=params)
        assert "tsne_2d" in result.embeddings

    def test_downsampling(self):
        sample = make_sample(n_cells=300)
        strategy = TSNEStrategy()
        params = TSNEStrategyParams(max_events=100, n_iter=250, perplexity=10.0)
        result = strategy.run(sample, params=params)
        emb = result.embeddings["tsne_2d"]
        assert emb.shape[0] == 300


# ---------------------------------------------------------------------------
# FlowSOMStrategy proxy
# ---------------------------------------------------------------------------

class TestFlowSOMStrategy:
    def test_basic_run(self):
        sample = make_sample()
        strategy = FlowSOMStrategy()
        params = FlowSOMStrategyParams(xdim=5, ydim=5, n_metaclusters=5)
        result = strategy.run(sample, params=params)
        assert "flowsom_nodes" in result.cluster_assignments
        assert "flowsom_meta" in result.cluster_assignments
        nodes = result.cluster_assignments["flowsom_nodes"]
        meta = result.cluster_assignments["flowsom_meta"]
        assert nodes.shape == (N_CELLS,)
        assert meta.shape == (N_CELLS,)
        assert np.issubdtype(nodes.dtype, np.integer)
        assert meta.min() >= 0
        assert meta.max() < 5

    def test_custom_labels(self):
        sample = make_sample()
        strategy = FlowSOMStrategy()
        params = FlowSOMStrategyParams(
            xdim=4, ydim=4, n_metaclusters=3,
            som_label="som_custom", meta_label="meta_custom",
        )
        result = strategy.run(sample, params=params)
        assert "som_custom" in result.cluster_assignments
        assert "meta_custom" in result.cluster_assignments

    def test_compute_centers_stores_mfi(self):
        sample = make_sample()
        strategy = FlowSOMStrategy()
        params = FlowSOMStrategyParams(xdim=4, ydim=4, n_metaclusters=3, compute_centers=True)
        result = strategy.run(sample, params=params)
        assert "flowsom_mfi" in result.results
        mfi = result.results["flowsom_mfi"]
        assert isinstance(mfi, dict)
        assert len(mfi) == 3

    def test_backend_stored(self):
        sample = make_sample()
        strategy = FlowSOMStrategy()
        params = FlowSOMStrategyParams(xdim=4, ydim=4, n_metaclusters=3)
        result = strategy.run(sample, params=params)
        assert "flowsom_backend" in result.results
        assert result.results["flowsom_backend"] in ("saeyslab", "proxy")

    def test_counts_stored(self):
        sample = make_sample()
        strategy = FlowSOMStrategy()
        params = FlowSOMStrategyParams(xdim=4, ydim=4, n_metaclusters=3)
        result = strategy.run(sample, params=params)
        assert "flowsom_counts" in result.results
        counts = result.results["flowsom_counts"]
        assert isinstance(counts, dict)
        assert sum(counts.values()) == N_CELLS

    def test_codes_stored(self):
        sample = make_sample()
        strategy = FlowSOMStrategy()
        params = FlowSOMStrategyParams(xdim=4, ydim=4, n_metaclusters=3, store_codes=True)
        result = strategy.run(sample, params=params)
        assert "flowsom_codes" in result.results
        codes = result.results["flowsom_codes"]
        assert isinstance(codes, list)
        assert len(codes) == 16  # 4x4 nodes

    def test_downsampling_assigns_minus_one(self):
        sample = make_sample(n_cells=300)
        strategy = FlowSOMStrategy()
        params = FlowSOMStrategyParams(xdim=3, ydim=3, n_metaclusters=3, max_events=100)
        result = strategy.run(sample, params=params)
        nodes = result.cluster_assignments["flowsom_nodes"]
        assert nodes.shape == (300,)
        assert (nodes == -1).sum() == 200

    def test_strategy_name(self):
        assert FlowSOMStrategy().name == "flowsom"


# ---------------------------------------------------------------------------
# ResearchPipelineExecutor
# ---------------------------------------------------------------------------

class TestResearchPipelineExecutor:
    def test_empty_executor(self):
        executor = ResearchPipelineExecutor()
        samples = [make_sample()]
        result = executor.run(samples)
        assert len(result) == 1

    def test_umap_only(self):
        executor = ResearchPipelineExecutor()
        executor.register(UMAPStrategy())
        samples = [make_sample(), make_sample(seed=1)]
        result = executor.run(
            samples,
            umap_params=UMAPStrategyParams(n_neighbors=10),
        )
        for s in result:
            assert "umap_2d" in s.embeddings

    def test_full_chain(self):
        executor = (
            ResearchPipelineExecutor()
            .register(UMAPStrategy())
            .register(TSNEStrategy())
            .register(FlowSOMStrategy())
        )
        sample = make_sample()
        results = executor.run(
            [sample],
            umap_params=UMAPStrategyParams(n_neighbors=10),
            tsne_params=TSNEStrategyParams(n_iter=250, perplexity=20.0),
            flowsom_params=FlowSOMStrategyParams(xdim=4, ydim=4, n_metaclusters=3),
        )
        s = results[0]
        assert "umap_2d" in s.embeddings
        assert "tsne_2d" in s.embeddings
        assert "flowsom_nodes" in s.cluster_assignments
        assert "flowsom_meta" in s.cluster_assignments

    def test_strategy_names(self):
        executor = (
            ResearchPipelineExecutor()
            .register(UMAPStrategy())
            .register(FlowSOMStrategy())
        )
        assert executor.strategy_names == ["umap", "flowsom"]

    def test_multiple_samples_independent(self):
        executor = ResearchPipelineExecutor()
        executor.register(UMAPStrategy())
        samples = [make_sample(seed=i) for i in range(3)]
        results = executor.run(
            samples,
            umap_params=UMAPStrategyParams(n_neighbors=10),
        )
        assert len(results) == 3
        for s in results:
            assert "umap_2d" in s.embeddings


# ---------------------------------------------------------------------------
# FlowSOMlikeStrategy
# ---------------------------------------------------------------------------

class TestFlowSOMlikeStrategy:
    def test_basic_run(self):
        sample = make_sample()
        strategy = FlowSOMlikeStrategy()
        params = FlowSOMlikeParams(xdim=4, ydim=4, n_metaclusters=4)
        result = strategy.run(sample, params=params)
        assert "flowsom_like_nodes" in result.cluster_assignments
        assert "flowsom_like_meta" in result.cluster_assignments
        nodes = result.cluster_assignments["flowsom_like_nodes"]
        meta = result.cluster_assignments["flowsom_like_meta"]
        assert nodes.shape == (N_CELLS,)
        assert meta.shape == (N_CELLS,)
        assert np.issubdtype(nodes.dtype, np.integer)
        assert meta.min() >= 0
        assert meta.max() < 4

    def test_custom_labels(self):
        sample = make_sample()
        strategy = FlowSOMlikeStrategy()
        params = FlowSOMlikeParams(
            xdim=3, ydim=3, n_metaclusters=3,
            som_label="my_nodes", meta_label="my_meta",
        )
        result = strategy.run(sample, params=params)
        assert "my_nodes" in result.cluster_assignments
        assert "my_meta" in result.cluster_assignments

    def test_compute_centers(self):
        sample = make_sample()
        strategy = FlowSOMlikeStrategy()
        params = FlowSOMlikeParams(xdim=4, ydim=4, n_metaclusters=3, compute_centers=True)
        result = strategy.run(sample, params=params)
        assert "flowsom_like_mfi" in result.results
        mfi = result.results["flowsom_like_mfi"]
        assert isinstance(mfi, dict)
        assert len(mfi) == 3

    def test_counts_stored(self):
        sample = make_sample()
        strategy = FlowSOMlikeStrategy()
        params = FlowSOMlikeParams(xdim=4, ydim=4, n_metaclusters=4)
        result = strategy.run(sample, params=params)
        assert "flowsom_like_counts" in result.results
        counts = result.results["flowsom_like_counts"]
        assert sum(counts.values()) == N_CELLS

    def test_codes_stored(self):
        sample = make_sample()
        strategy = FlowSOMlikeStrategy()
        params = FlowSOMlikeParams(xdim=3, ydim=3, n_metaclusters=3, store_codes=True)
        result = strategy.run(sample, params=params)
        assert "flowsom_like_codes" in result.results
        codes = result.results["flowsom_like_codes"]
        assert len(codes) == 9  # 3×3 nœuds

    def test_kmeans_metacluster(self):
        sample = make_sample()
        strategy = FlowSOMlikeStrategy()
        params = FlowSOMlikeParams(
            xdim=4, ydim=4, n_metaclusters=3, metacluster_method="kmeans"
        )
        result = strategy.run(sample, params=params)
        assert "flowsom_like_meta" in result.cluster_assignments

    def test_downsampling_assigns_minus_one(self):
        sample = make_sample(n_cells=300)
        strategy = FlowSOMlikeStrategy()
        params = FlowSOMlikeParams(xdim=3, ydim=3, n_metaclusters=3, max_events=100)
        result = strategy.run(sample, params=params)
        nodes = result.cluster_assignments["flowsom_like_nodes"]
        assert nodes.shape == (300,)
        assert (nodes == -1).sum() == 200

    def test_auto_metaclustering(self):
        sample = make_sample(n_cells=600)
        strategy = FlowSOMlikeStrategy()
        params = FlowSOMlikeParams(
            xdim=4, ydim=4,
            auto_metaclusters=True,
            min_clusters=3, max_clusters=6,
            n_bootstrap=3,
            sample_size_bootstrap=300,
            min_stability_threshold=0.1,
            seed=42,
        )
        result = strategy.run(sample, params=params)
        assert "flowsom_like_meta" in result.cluster_assignments
        assert "flowsom_like_auto_k" in result.results
        k = result.results["flowsom_like_auto_k"]
        assert 3 <= k <= 6
        assert "flowsom_like_composite_scores" in result.results

    def test_strategy_name(self):
        assert FlowSOMlikeStrategy().name == "flowsom_like"

    def test_with_gating_mask(self):
        sample = make_sample(n_cells=200)
        mask = np.zeros(200, dtype=bool)
        mask[:100] = True
        sample.set_mask("live", mask)
        strategy = FlowSOMlikeStrategy()
        params = FlowSOMlikeParams(xdim=3, ydim=3, n_metaclusters=3)
        result = strategy.run(sample, params=params)
        meta = result.cluster_assignments["flowsom_like_meta"]
        assert meta.shape == (200,)
        assert np.all(meta[100:] == -1)
        assert np.all(meta[:100] >= 0)


class TestAutoMetaclustering:
    def test_phase1_silhouette(self):
        rng = np.random.default_rng(0)
        codebook = rng.standard_normal((16, 6)).astype(np.float32)
        scores = phase1_silhouette_on_codebook(codebook, range(3, 7), seed=42)
        assert set(scores.keys()) == {3, 4, 5, 6}
        assert all(isinstance(v, float) for v in scores.values())

    def test_phase3_composite(self):
        sil = {3: 0.4, 4: 0.5, 5: 0.3}
        stab = {3: 0.9, 4: 0.8, 5: 0.6}
        best_k, composite = phase3_composite_selection(
            sil, stab, min_stability_threshold=0.7,
            weight_stability=0.65, weight_silhouette=0.35
        )
        assert best_k in (3, 4)
        assert set(composite.keys()).issubset({3, 4, 5})

    def test_phase3_fallback_no_threshold(self):
        sil = {3: 0.4, 4: 0.5}
        stab = {3: 0.2, 4: 0.3}
        # Seuil 0.9 → aucun candidat → fallback au plus stable
        best_k, _ = phase3_composite_selection(
            sil, stab, min_stability_threshold=0.9
        )
        assert best_k == 4  # stab[4]=0.3 > stab[3]=0.2

    def test_find_optimal_metaclusters(self):
        rng = np.random.default_rng(7)
        data = rng.standard_normal((500, 6)).astype(np.float32)
        codebook = rng.standard_normal((16, 6)).astype(np.float32)
        best_k, composite = find_optimal_metaclusters(
            X=data,
            codebook=codebook,
            min_clusters=3,
            max_clusters=6,
            n_bootstrap=2,
            sample_size_bootstrap=200,
            min_stability_threshold=0.0,
            seed=42,
        )
        assert 3 <= best_k <= 6
        assert isinstance(composite, dict)
