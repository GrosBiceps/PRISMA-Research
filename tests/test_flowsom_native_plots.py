# -*- coding: utf-8 -*-
"""
tests/test_flowsom_native_plots.py — Visualisations natives FlowSOM + API new_data/subset.

Vérifie sur données synthétiques (3 blobs séparables) :
  - FlowSOMClusterer.fsom_model exposé (backend CPU natif)
  - map_new_data() et subset_by_metacluster() fonctionnels
  - plot_stars_native (MST + grille), plot_marker, plot_numbers
  - plot_new_data_stars, plot_subset_stars
  - generate_all_native_plots produit des fichiers
  - colonnes FlowSOM complètes dans le DataFrame de sortie + export FCS
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# matplotlib non interactif pour les tests headless
import matplotlib

matplotlib.use("Agg")

from prisma.core.clustering import FlowSOMClusterer
from prisma.visualization import flowsom_native_plots as fnp


@pytest.fixture(scope="module")
def fitted_clusterer():
    """FlowSOMClusterer entraîné sur 3 blobs synthétiques séparables (CPU)."""
    rng = np.random.default_rng(42)
    X = np.vstack(
        [rng.normal(m, 0.3, (600, 7)) for m in (0.0, 3.0, 6.0)]
    ).astype(np.float32)
    markers = [f"CD{i}" for i in range(7)]
    clust = FlowSOMClusterer(
        xdim=7, ydim=7, n_metaclusters=6, rlen="auto", seed=42, use_gpu=False
    )
    clust.fit(X, marker_names=markers)
    return clust, X, markers


class TestFlowSOMNativeAPI:
    def test_fsom_model_exposed(self, fitted_clusterer):
        clust, _, _ = fitted_clusterer
        assert clust.fsom_model is not None
        assert hasattr(clust.fsom_model, "get_cluster_data")

    def test_marker_names_retained(self, fitted_clusterer):
        clust, _, markers = fitted_clusterer
        assert clust.marker_names_ == markers

    def test_grid_dims_synced_after_fit(self, fitted_clusterer):
        clust, _, _ = fitted_clusterer
        # 7×7 demandé → conservé (≤49 nœuds, pas de réduction)
        assert clust.n_nodes == clust.xdim * clust.ydim
        assert clust.get_grid_coords().shape[0] == clust.n_nodes

    def test_map_new_data(self, fitted_clusterer):
        clust, X, _ = fitted_clusterer
        fsom_new = clust.map_new_data(X[:200])
        assert fsom_new is not None
        assert fsom_new.get_cell_data().shape[0] == 200

    def test_subset_by_metacluster(self, fitted_clusterer):
        clust, _, _ = fitted_clusterer
        # Métacluster réellement présent
        mc = int(np.unique(clust.metacluster_assignments_)[0])
        sub = clust.subset_by_metacluster(mc)
        assert sub is not None


class TestFlowSOMNativePlots:
    def test_plot_stars_mst(self, fitted_clusterer, tmp_path):
        clust, _, _ = fitted_clusterer
        p = fnp.plot_stars_native(clust, tmp_path / "stars_mst.png", view="MST")
        assert p is not None and p.exists()

    def test_plot_stars_grid_jpg(self, fitted_clusterer, tmp_path):
        clust, _, _ = fitted_clusterer
        p = fnp.plot_stars_native(
            clust, tmp_path / "stars_grid.jpg", view="grid",
            equal_node_size=True, equal_background_size=True,
        )
        assert p is not None and p.exists()

    def test_plot_marker(self, fitted_clusterer, tmp_path):
        clust, _, markers = fitted_clusterer
        p = fnp.plot_marker_native(clust, markers[0], tmp_path / "marker.svg")
        assert p is not None and p.exists()

    def test_plot_numbers(self, fitted_clusterer, tmp_path):
        clust, _, _ = fitted_clusterer
        p = fnp.plot_numbers_native(clust, tmp_path / "numbers.png", level="clusters")
        assert p is not None and p.exists()

    def test_plot_new_data_stars(self, fitted_clusterer, tmp_path):
        clust, X, _ = fitted_clusterer
        p = fnp.plot_new_data_stars(clust, X[:200], tmp_path / "new.png")
        assert p is not None and p.exists()

    def test_plot_subset_stars(self, fitted_clusterer, tmp_path):
        clust, _, _ = fitted_clusterer
        mc = int(np.unique(clust.metacluster_assignments_)[0])
        p = fnp.plot_subset_stars(clust, mc, tmp_path / "subset.png")
        assert p is not None and p.exists()

    def test_generate_all(self, fitted_clusterer, tmp_path):
        clust, _, markers = fitted_clusterer
        produced = fnp.generate_all_native_plots(
            clust, tmp_path / "all", marker_names=markers[:3], fmt="png"
        )
        # Au moins star charts + numéros + cartes marqueurs
        assert "stars_mst" in produced
        assert "stars_grid" in produced
        assert "numbers_clusters" in produced
        assert any(k.startswith("marker_") for k in produced)
        for path in produced.values():
            from pathlib import Path as _P

            assert _P(path).exists()


class TestFlowSOMOutputColumns:
    def test_build_output_dataframe_has_flowsom_columns(self, fitted_clusterer):
        from prisma.pipeline.research_executor import ResearchPipelineExecutor

        clust, X, markers = fitted_clusterer
        df = pd.DataFrame(X, columns=markers)
        ex = ResearchPipelineExecutor()
        out = ex._build_output_dataframe(
            df, markers, {}, {"flowsom": clust.metacluster_assignments_},
            flowsom_clusterer=clust,
        )
        for col in (
            "FlowSOM_cluster", "FlowSOM_metacluster",
            "xGrid", "yGrid", "xNodes", "yNodes", "size",
        ):
            assert col in out.columns, f"colonne FlowSOM manquante: {col}"
        # 1-based + cohérence n_nodes
        assert out["FlowSOM_cluster"].min() >= 1.0
        assert out["FlowSOM_cluster"].max() <= float(clust.n_nodes)
        # Toutes numériques → exportables en FCS
        assert out.select_dtypes("number").shape[1] == out.shape[1]


class TestAutoClustering:
    """Auto-sélection du nombre de métaclusters (3 phases) + graphique métriques."""

    def test_find_optimal_with_scores_separable(self):
        from prisma.core.metaclustering import find_optimal_clusters_with_scores

        rng = np.random.default_rng(42)
        # 4 blobs nets → k optimal attendu proche de 4
        X = np.vstack(
            [rng.normal(m, 0.25, (400, 7)) for m in (0.0, 3.0, 6.0, 9.0)]
        ).astype(np.float32)
        res = find_optimal_clusters_with_scores(
            X, min_clusters=3, max_clusters=8, n_bootstrap=3,
            sample_size_bootstrap=1000, xdim=7, ydim=7, seed=42, verbose=False,
        )
        assert "best_k" in res
        assert 3 <= res["best_k"] <= 8
        assert list(res["results_df"].columns) == ["k", "silhouette", "composite_score"]
        assert len(res["results_df"]) == 6  # k ∈ [3, 8]
        assert all(
            "mean_ari" in v and "std_ari" in v
            for v in res["stability_results"].values()
        )

    def test_auto_cluster_metrics_plot(self, tmp_path):
        from prisma.core.metaclustering import find_optimal_clusters_with_scores
        from prisma.visualization.flowsom_plots import plot_optimization_results

        rng = np.random.default_rng(7)
        X = np.vstack(
            [rng.normal(m, 0.25, (400, 6)) for m in (0.0, 4.0, 8.0)]
        ).astype(np.float32)
        res = find_optimal_clusters_with_scores(
            X, min_clusters=2, max_clusters=6, n_bootstrap=3,
            sample_size_bootstrap=800, xdim=6, ydim=6, seed=7, verbose=False,
        )
        out = tmp_path / "opt.png"
        fig = plot_optimization_results(
            res["results_df"], res["best_k"], res["stability_results"],
            output_path=out,
        )
        assert fig is not None
        assert out.exists()


class TestPerColumnPreprocessing:
    """Transformations par colonne (popup pré-traitement)."""

    def test_apply_per_column_transforms_mixed(self):
        from prisma.core.transformers import DataTransformer

        X = np.array(
            [[100.0, 200.0, 5000.0], [50000.0, 30000.0, 1000.0]], dtype=np.float32
        )
        vn = ["CD34", "SSC-A", "FSC-A"]
        specs = {
            "CD34": {"method": "logicle", "T": 262144, "M": 4.5, "W": 0.5, "A": 0.0},
            "SSC-A": {"method": "arcsinh", "cofactor": 150},
        }
        out = DataTransformer.apply_per_column_transforms(X, vn, specs)
        assert out.shape == X.shape
        # CD34 + SSC-A transformés, FSC-A (sans spec) inchangé
        assert not np.allclose(out[:, 0], X[:, 0])
        assert not np.allclose(out[:, 1], X[:, 1])
        assert np.allclose(out[:, 2], X[:, 2])

    def test_unknown_column_ignored(self):
        from prisma.core.transformers import DataTransformer

        X = np.array([[1.0, 2.0]], dtype=np.float32)
        out = DataTransformer.apply_per_column_transforms(
            X, ["A", "B"], {"GHOST": {"method": "logicle"}}
        )
        assert np.allclose(out, X)  # aucune colonne touchée

    def test_executor_uses_per_column_specs(self):
        from config.pipeline_config import PipelineConfig
        from prisma.pipeline.research_executor import ResearchPipelineExecutor

        cfg = PipelineConfig()
        cfg.normalize.method = "none"  # isoler l'effet transform
        cfg.transform.per_column_specs = {"CD34": {"method": "logicle"}}
        X = np.array([[100.0, 5000.0], [50000.0, 1000.0]], dtype=np.float32)
        ex = ResearchPipelineExecutor()
        out = ex._apply_basic_preprocessing(X, cfg, ["CD34", "FSC-A"])
        assert not np.allclose(out[:, 0], X[:, 0])   # CD34 transformé
        assert np.allclose(out[:, 1], X[:, 1])       # FSC-A inchangé
