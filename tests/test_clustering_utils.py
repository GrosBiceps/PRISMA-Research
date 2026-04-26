"""
tests/test_clustering_utils.py — Tests unitaires pour les utilitaires de clustering FlowSOM.

Teste les fonctions compute_optimal_rlen() et compute_optimal_grid() avec des
entrées synthétiques connues, garantissant la stabilité numérique de ces calculs.
"""

import pytest
from flowsom_pipeline_pro.src.prisma.core.clustering import compute_optimal_rlen, compute_optimal_grid


# ── compute_optimal_rlen ──────────────────────────────────────────────────────

class TestComputeOptimalRlen:
    """Tests pour la formule rlen ∝ √N × 0.1, borné [10, 100]."""

    def test_explicit_int_is_returned_unchanged(self):
        assert compute_optimal_rlen(100_000, rlen_setting=25) == 25

    def test_explicit_zero_is_returned_unchanged(self):
        # Le paramètre est int, retourné tel quel même si biologiquement absurde
        assert compute_optimal_rlen(100_000, rlen_setting=0) == 0

    def test_auto_small_dataset_returns_minimum(self):
        # √10_000 × 0.1 = 10 → borne basse
        result = compute_optimal_rlen(10_000, rlen_setting="auto")
        assert result == 10

    def test_auto_medium_dataset(self):
        # √100_000 × 0.1 ≈ 31.6 → 31
        result = compute_optimal_rlen(100_000, rlen_setting="auto")
        assert result == 31

    def test_auto_large_dataset_returns_maximum(self):
        # √1_000_000 × 0.1 = 100 → borne haute
        result = compute_optimal_rlen(1_000_000, rlen_setting="auto")
        assert result == 100

    def test_auto_very_large_dataset_capped_at_100(self):
        # √100_000_000 × 0.1 = 1000 → plafonné à 100
        result = compute_optimal_rlen(100_000_000, rlen_setting="auto")
        assert result == 100

    def test_auto_tiny_dataset_floored_at_10(self):
        result = compute_optimal_rlen(1, rlen_setting="auto")
        assert result == 10

    def test_return_type_is_int(self):
        result = compute_optimal_rlen(50_000, rlen_setting="auto")
        assert isinstance(result, int)


# ── compute_optimal_grid ──────────────────────────────────────────────────────

class TestComputeOptimalGrid:
    """Tests pour l'ajustement automatique de la grille SOM selon la taille du dataset."""

    def test_large_dataset_keeps_configured_grid(self):
        # > 50k cellules → grille configurée conservée
        assert compute_optimal_grid(100_000, xdim=10, ydim=10) == (10, 10)

    def test_large_dataset_large_grid_kept(self):
        assert compute_optimal_grid(500_000, xdim=15, ydim=15) == (15, 15)

    def test_small_dataset_with_big_grid_reduced_to_7x7(self):
        # < 50k cellules et grille > 7×7 → réduit à 7×7
        assert compute_optimal_grid(20_000, xdim=10, ydim=10) == (7, 7)

    def test_small_dataset_already_7x7_unchanged(self):
        # < 50k cellules mais grille déjà 7×7 (49 noeuds) → inchangée
        assert compute_optimal_grid(20_000, xdim=7, ydim=7) == (7, 7)

    def test_small_dataset_smaller_than_7x7_unchanged(self):
        # < 50k cellules et grille 5×5 (25 noeuds < 49) → inchangée
        assert compute_optimal_grid(20_000, xdim=5, ydim=5) == (5, 5)

    def test_boundary_exactly_50k_uses_configured(self):
        # Exactement 50k → seuil non franchi, grille conservée
        assert compute_optimal_grid(50_000, xdim=10, ydim=10) == (10, 10)

    def test_return_type_is_tuple_of_ints(self):
        xd, yd = compute_optimal_grid(100_000, xdim=10, ydim=10)
        assert isinstance(xd, int) and isinstance(yd, int)
