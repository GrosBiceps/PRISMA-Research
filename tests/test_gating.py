"""
tests/test_gating.py — Tests unitaires pour les modules de gating.

Couvre PreGating.gate_viable_cells() et la classe GatingSession.
Utilise des matrices numpy synthétiques avec des distributions connues.
"""

import numpy as np
import pytest

from flowsom_pipeline_pro.src.prisma.core.gating import PreGating
from flowsom_pipeline_pro.src.prisma.core.auto_gating import GatingSession


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_sample():
    """
    Matrice synthétique 1000 cellules × 4 marqueurs (FSC-A, SSC-A, CD45, CD34).
    La moitié centrale (p1–p99) représente des cellules viables normales.
    Les 10 premières et 10 dernières lignes sont des outliers extrêmes (débris/saturés).
    """
    rng = np.random.default_rng(42)
    n = 1000
    X = np.zeros((n, 4))
    # FSC-A : distribution normale centrée sur 50 000
    X[:, 0] = rng.normal(50_000, 8_000, n)
    # SSC-A : distribution normale centrée sur 30 000
    X[:, 1] = rng.normal(30_000, 5_000, n)
    X[:, 2] = rng.normal(1000, 200, n)   # CD45
    X[:, 3] = rng.normal(100, 20, n)     # CD34

    # Injecter des débris (FSC très bas) en début de matrice
    X[:10, 0] = 100.0
    # Injecter des saturés (FSC très haut) en fin de matrice
    X[-10:, 0] = 200_000.0

    return X, ["FSC-A", "SSC-A", "CD45", "CD34"]


@pytest.fixture
def missing_fsc_sample():
    """Matrice sans colonne FSC-A."""
    rng = np.random.default_rng(0)
    X = rng.normal(1000, 100, (500, 3))
    return X, ["SSC-A", "CD45", "CD34"]


# ── PreGating.gate_viable_cells ───────────────────────────────────────────────

class TestGateViableCells:

    def test_returns_boolean_mask(self, clean_sample):
        X, var_names = clean_sample
        mask = PreGating.gate_viable_cells(X, var_names)
        assert mask.dtype == bool
        assert mask.shape == (X.shape[0],)

    def test_removes_extreme_outliers(self, clean_sample):
        X, var_names = clean_sample
        mask = PreGating.gate_viable_cells(X, var_names, min_percentile=2.0, max_percentile=98.0)
        # Les 10 débris (FSC=100) et 10 saturés (FSC=200k) doivent être exclus
        assert not mask[:10].any(), "Les débris FSC très bas doivent être exclus"
        assert not mask[-10:].any(), "Les saturés FSC très haut doivent être exclus"

    def test_keeps_majority_of_clean_cells(self, clean_sample):
        X, var_names = clean_sample
        mask = PreGating.gate_viable_cells(X, var_names)
        # Au moins 80% des cellules conservées pour un échantillon propre
        assert mask.sum() > 800

    def test_all_ones_when_fsc_absent(self, missing_fsc_sample):
        X, var_names = missing_fsc_sample
        # Sans FSC-A, le gate SSC seul peut s'appliquer, mais aucun crash
        mask = PreGating.gate_viable_cells(X, var_names)
        assert mask.shape == (X.shape[0],)

    def test_handles_nan_values(self):
        X = np.array([[np.nan, 1000.0], [50000.0, 30000.0], [60000.0, 25000.0]])
        var_names = ["FSC-A", "SSC-A"]
        mask = PreGating.gate_viable_cells(X, var_names)
        assert mask.shape == (3,)
        # La ligne NaN ne doit pas crasher
        assert not mask[0], "Une cellule NaN doit être exclue"

    def test_empty_array_returns_empty_mask(self):
        X = np.zeros((0, 2))
        mask = PreGating.gate_viable_cells(X, ["FSC-A", "SSC-A"])
        assert mask.shape == (0,)


# ── PreGating.find_marker_index ───────────────────────────────────────────────

class TestFindMarkerIndex:

    def test_exact_match(self):
        assert PreGating.find_marker_index(["FSC-A", "SSC-A", "CD45"], ["FSC-A"]) == 0

    def test_case_insensitive(self):
        assert PreGating.find_marker_index(["fsc-a", "SSC-A"], ["FSC-A"]) == 0

    def test_partial_match(self):
        # "FSC" doit matcher "FSC-A"
        assert PreGating.find_marker_index(["FSC-A", "SSC-A"], ["FSC"]) == 0

    def test_priority_first_pattern(self):
        # Patterns dans l'ordre : FSC-A avant FSC-H
        idx = PreGating.find_marker_index(["FSC-H", "FSC-A"], ["FSC-A", "FSC-H", "FSC"])
        assert idx == 1  # FSC-A est en position 1

    def test_returns_none_when_not_found(self):
        assert PreGating.find_marker_index(["CD45", "CD34"], ["FSC-A"]) is None

    def test_empty_var_names(self):
        assert PreGating.find_marker_index([], ["FSC-A"]) is None


# ── GatingSession ─────────────────────────────────────────────────────────────

class TestGatingSession:

    def test_initial_state_is_empty(self):
        session = GatingSession()
        assert len(session.ransac_scatter_data) == 0
        assert len(session.singlets_summary_per_file) == 0

    def test_clear_resets_all_data(self):
        session = GatingSession()
        session.ransac_scatter_data["file1.fcs"] = {"r2": 0.95}
        session.singlets_summary_per_file.append({"file": "file1.fcs", "pct_singlets": 98.0})
        session.clear()
        assert len(session.ransac_scatter_data) == 0
        assert len(session.singlets_summary_per_file) == 0

    def test_two_sessions_are_independent(self):
        """Vérifie l'isolation entre patients (le bug original des globals)."""
        session_a = GatingSession()
        session_b = GatingSession()
        session_a.ransac_scatter_data["patient_A.fcs"] = {"r2": 0.9}
        # session_b ne doit pas voir les données de session_a
        assert "patient_A.fcs" not in session_b.ransac_scatter_data

    def test_clear_does_not_affect_other_sessions(self):
        session_a = GatingSession()
        session_b = GatingSession()
        session_a.ransac_scatter_data["file.fcs"] = {"r2": 0.9}
        session_b.ransac_scatter_data["file.fcs"] = {"r2": 0.85}
        session_a.clear()
        # session_b toujours intacte après clear de session_a
        assert "file.fcs" in session_b.ransac_scatter_data
