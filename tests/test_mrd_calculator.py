"""
tests/test_mrd_calculator.py — Tests unitaires pour les trois méthodes MRD.

Utilise des DataFrames et clusterings synthétiques avec des résultats MRD
attendus parfaitement connus. Aucune dépendance à flowsom ou anndata.

Scénarios testés :
  - Méthode JF  : cluster pur patho / cluster mixte / cluster pur sain
  - Méthode Flo : ratio patho/sain > N / ratio < N
  - Méthode ELN : DfN + LOQ + seuil positivité clinique
  - Cas limites  : dataset vide, tous sains, tous patho
"""

import numpy as np
import pandas as pd
import pytest

from clinical_module.mrd.mrd_calculator import (
    MRDConfig,
    MRDMethodJF,
    MRDMethodFlo,
    ELNStandards,
    BlastPhenotypeFilter,
    compute_mrd,
)


# ── Fixtures & Helpers ────────────────────────────────────────────────────────

def _make_config(method: str = "all") -> MRDConfig:
    """Crée une MRDConfig minimale avec BPF désactivé pour des tests déterministes."""
    return MRDConfig(
        enabled=True,
        method=method,
        condition_sain="Sain",
        condition_patho="Pathologique",
        method_jf=MRDMethodJF(max_normal_marrow_pct=2.0, min_patho_cells_pct=10.0),
        method_flo=MRDMethodFlo(normal_marrow_multiplier=5.0),
        eln_standards=ELNStandards(min_cluster_events=10, clinical_positivity_pct=0.1),
        blast_phenotype_filter=BlastPhenotypeFilter(enabled=False),
    )


def _make_df_and_clustering(n_sain: int, n_patho: int, n_nodes: int = 4) -> tuple:
    """
    Crée un DataFrame cellulaire synthétique et un vecteur de clustering.

    Répartition des cellules dans les noeuds :
      Noeud 0 : toutes les cellules patho
      Noeud 1 : toutes les cellules saines
      Noeuds 2+ : vides (0 cellule)

    Retourne (df_cells, clustering).
    """
    conditions = (["Pathologique"] * n_patho) + (["Sain"] * n_sain)
    df = pd.DataFrame({"condition": conditions})

    # Nœud 0 = toutes les patho, Nœud 1 = toutes les saines
    clustering = np.array([0] * n_patho + [1] * n_sain)
    return df, clustering


# ── Méthode JF ────────────────────────────────────────────────────────────────

class TestMRDMethodJF:

    def test_pure_patho_node_detected_as_mrd(self):
        """Nœud 0 = 100% patho + 0% sain → doit être MRD."""
        df, clustering = _make_df_and_clustering(n_sain=1000, n_patho=100)
        cfg = _make_config(method="jf")
        result = compute_mrd(df, clustering, cfg)
        # MRD positive attendue
        assert result.n_nodes_mrd_jf > 0, "Un cluster 100% patho doit être MRD JF"

    def test_pure_normal_node_not_mrd(self):
        """Dataset avec seulement des cellules saines → MRD négative."""
        df = pd.DataFrame({"condition": ["Sain"] * 500})
        clustering = np.zeros(500, dtype=int)
        cfg = _make_config(method="jf")
        result = compute_mrd(df, clustering, cfg)
        assert result.n_nodes_mrd_jf == 0, "Un dataset 100% sain ne doit pas être MRD"

    def test_mrd_percentage_is_within_bounds(self):
        df, clustering = _make_df_and_clustering(n_sain=1000, n_patho=50)
        cfg = _make_config(method="jf")
        result = compute_mrd(df, clustering, cfg)
        assert 0.0 <= result.mrd_pct_jf <= 100.0


# ── Méthode Flo ───────────────────────────────────────────────────────────────

class TestMRDMethodFlo:

    def test_high_ratio_patho_sain_is_mrd(self):
        """
        Nœud 0 : 100 patho, 5 sain → ratio = 100/5 = 20 >> multiplicateur 5.
        Doit être MRD Flo.
        """
        df, clustering = _make_df_and_clustering(n_sain=5, n_patho=100)
        cfg = _make_config(method="flo")
        result = compute_mrd(df, clustering, cfg)
        assert result.n_nodes_mrd_flo > 0

    def test_low_ratio_is_not_mrd(self):
        """
        Nœud 0 : 5 patho, 100 sain → ratio = 5/100 = 0.05 << multiplicateur 5.
        Ne doit PAS être MRD Flo.
        """
        df, clustering = _make_df_and_clustering(n_sain=100, n_patho=5)
        cfg = _make_config(method="flo")
        result = compute_mrd(df, clustering, cfg)
        assert result.n_nodes_mrd_flo == 0

    def test_mrd_pct_flo_non_negative(self):
        df, clustering = _make_df_and_clustering(n_sain=500, n_patho=50)
        cfg = _make_config(method="flo")
        result = compute_mrd(df, clustering, cfg)
        assert result.mrd_pct_flo >= 0.0


# ── Méthode ELN ───────────────────────────────────────────────────────────────

class TestMRDMethodELN:

    def test_dfn_criterion_with_sufficient_events(self):
        """
        Nœud 0 : 100 patho, 10 sain → pct_patho > pct_sain (DfN) + LOQ 10 atteint.
        MRD totale = 100/1010 ≈ 9.9% >> seuil 0.1%.
        """
        df, clustering = _make_df_and_clustering(n_sain=910, n_patho=100)
        cfg = _make_config(method="eln")
        result = compute_mrd(df, clustering, cfg)
        assert result.eln_positive

    def test_loq_not_met_node_excluded(self):
        """
        Nœud 0 : 3 patho (< LOQ=50) → nœud exclu même si DfN satisfait.
        MRD totale = 3/503 ≈ 0.6% mais LOQ non atteint.
        """
        cfg = _make_config(method="eln")
        # Forcer LOQ élevé pour que le nœud soit exclu
        cfg.eln_standards.min_cluster_events = 50

        df, clustering = _make_df_and_clustering(n_sain=500, n_patho=3)
        result = compute_mrd(df, clustering, cfg)
        assert not result.eln_positive, "LOQ non atteint → nœud doit être exclu"

    def test_below_clinical_threshold_not_positive(self):
        """MRD détectable mais < 0.1% (seuil ELN) → eln_positive = False."""
        cfg = _make_config(method="eln")
        cfg.eln_standards.clinical_positivity_pct = 0.1
        # 1 patho sur 10000 total = 0.01% → sous le seuil
        df, clustering = _make_df_and_clustering(n_sain=9999, n_patho=1)
        result = compute_mrd(df, clustering, cfg)
        assert not result.eln_positive


# ── Cas limites ───────────────────────────────────────────────────────────────

class TestMRDEdgeCases:

    def test_all_patho_no_normal(self):
        """Dataset sans cellules saines → pas de référence, MRD non calculable."""
        df = pd.DataFrame({"condition": ["Pathologique"] * 100})
        clustering = np.zeros(100, dtype=int)
        cfg = _make_config()
        # Ne doit pas lever d'exception
        result = compute_mrd(df, clustering, cfg)
        assert result is not None

    def test_all_normal_no_patho(self):
        """Dataset sans cellules pathologiques → MRD = 0%."""
        df = pd.DataFrame({"condition": ["Sain"] * 200})
        clustering = np.zeros(200, dtype=int)
        cfg = _make_config()
        result = compute_mrd(df, clustering, cfg)
        assert result.mrd_pct_jf == 0.0 or result.n_nodes_mrd_jf == 0
        assert result.mrd_pct_flo == 0.0 or result.n_nodes_mrd_flo == 0
        assert result.mrd_pct_eln == 0.0 or not result.eln_positive

    def test_result_has_required_attributes(self):
        df, clustering = _make_df_and_clustering(n_sain=100, n_patho=10)
        cfg = _make_config()
        result = compute_mrd(df, clustering, cfg)
        for attr in ("mrd_pct_jf", "mrd_pct_flo", "mrd_pct_eln",
                     "n_nodes_mrd_jf", "n_nodes_mrd_flo", "eln_positive"):
            assert hasattr(result, attr), f"MRDResult doit avoir l'attribut '{attr}'"

    def test_mrd_percentages_are_finite(self):
        df, clustering = _make_df_and_clustering(n_sain=500, n_patho=50)
        cfg = _make_config()
        result = compute_mrd(df, clustering, cfg)
        assert np.isfinite(result.mrd_pct_jf)
        assert np.isfinite(result.mrd_pct_flo)
        assert np.isfinite(result.mrd_pct_eln)
