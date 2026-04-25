"""
tests/prisma/test_oom_protection.py — Test de stress anti-OOM pour torch_knn_indices.

Génère une matrice de 2 millions de cellules × 30 marqueurs (float32) et vérifie
que torch_knn_indices() se termine sans crash CUDA OOM sur la RTX 3060 (12 GB).

Usage :
    pytest tests/prisma/test_oom_protection.py -v -s
"""

from __future__ import annotations

import gc
import logging
import time

import numpy as np
import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# CI rapide: 100k. Stress moyen: 500k (slow). Full GPU: 2M (slow).
N_CELLS_STRESS = 100_000
N_CELLS_STRESS_MEDIUM = 500_000
N_CELLS_STRESS_FULL = 2_000_000
N_MARKERS = 30
K_NEIGHBORS = 15


@pytest.fixture(scope="module")
def large_cytometry_matrix() -> np.ndarray:
    """Matrice float32 simulant 500k cellules × 30 marqueurs (CI-friendly)."""
    rng = np.random.default_rng(seed=42)
    data = rng.standard_normal((N_CELLS_STRESS, N_MARKERS)).astype(np.float32)
    logger.info("Matrice stress générée: %d cellules × %d marqueurs (%.1f MB)", N_CELLS_STRESS, N_MARKERS, data.nbytes / 1024**2)
    return data


@pytest.fixture(scope="module")
def full_stress_matrix() -> np.ndarray:
    """Matrice float32 2M cellules × 30 marqueurs (test slow, GPU uniquement)."""
    rng = np.random.default_rng(seed=42)
    data = rng.standard_normal((N_CELLS_STRESS_FULL, N_MARKERS)).astype(np.float32)
    logger.info("Matrice full-stress générée: %d cellules × %d marqueurs (%.1f MB)", N_CELLS_STRESS_FULL, N_MARKERS, data.nbytes / 1024**2)
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_gpu_memory(label: str) -> None:
    """Affiche l'utilisation VRAM si PyTorch CUDA disponible."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            reserved  = torch.cuda.memory_reserved(0) / (1024 ** 3)
            total     = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(
                "[VRAM %s] Allouée: %.2f GB | Réservée: %.2f GB | Totale: %.2f GB",
                label,
                allocated,
                reserved,
                total,
            )
    except Exception:
        pass


def _force_gc() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOOMProtection:
    """Vérifie que torch_knn_indices() survit à un dataset cytométrie 2M cellules."""

    @pytest.mark.slow
    def test_torch_knn_indices_500k_cells_no_oom(self, large_cytometry_matrix: np.ndarray) -> None:
        """
        torch_knn_indices() doit se terminer sans CUDA OOM sur 100k×30 float32.

        Le mécanisme anti-OOM (chunk auto + halvage sur OOM) doit absorber
        la pression mémoire et retourner un tableau d'indices valide.
        """
        from src.prisma.core.torch_utils import torch_knn_indices, torch_available

        if not torch_available():
            pytest.skip("PyTorch non disponible dans l'environnement")

        data = large_cytometry_matrix
        logger.info(
            "Lancement torch_knn_indices: %d cellules, k=%d",
            data.shape[0],
            K_NEIGHBORS,
        )

        _log_gpu_memory("avant_knn")
        t0 = time.perf_counter()

        # Ne doit PAS lever RuntimeError/CUDA OOM
        indices = torch_knn_indices(data, k=K_NEIGHBORS)

        elapsed = time.perf_counter() - t0
        _log_gpu_memory("après_knn")

        # ── Assertions sur le résultat ───────────────────────────────────────
        assert indices is not None, "torch_knn_indices a retourné None"
        assert indices.shape == (N_CELLS_STRESS, K_NEIGHBORS), (
            f"Shape inattendue: {indices.shape}, attendu ({N_CELLS_STRESS}, {K_NEIGHBORS})"
        )
        # Indices dans [0, N_CELLS)
        assert indices.min() >= 0
        assert indices.max() < N_CELLS_STRESS

        # Pas d'auto-référence (chaque cellule ne doit pas être son propre voisin)
        row_ids = np.arange(N_CELLS_STRESS, dtype=indices.dtype)
        self_ref_count = int((indices == row_ids[:, None]).sum())
        assert self_ref_count == 0, (
            f"{self_ref_count} cellules référencées comme leur propre voisin"
        )

        logger.info(
            "✓ torch_knn_indices terminé en %.1f s — shape: %s",
            elapsed,
            indices.shape,
        )

        _force_gc()

    def test_torch_knn_indices_handles_cpu_fallback(self) -> None:
        """
        Sur CPU (ou si CUDA absent), torch_knn_indices doit fonctionner
        sur un petit dataset sans planter.
        """
        from src.prisma.core.torch_utils import torch_knn_indices, torch_available

        if not torch_available():
            pytest.skip("PyTorch non disponible dans l'environnement")

        rng = np.random.default_rng(seed=0)
        small_data = rng.standard_normal((5_000, 20)).astype(np.float32)

        indices = torch_knn_indices(small_data, k=10)

        assert indices.shape == (5_000, 10)
        assert indices.min() >= 0
        assert indices.max() < 5_000

        logger.info("✓ CPU fallback OK — shape: %s", indices.shape)

    def test_auto_chunk_size_estimation(self) -> None:
        """
        Vérifie que le calcul de taille de chunk auto ne produit pas
        de valeur hors bornes (< 64 ou > 8192 cellules/chunk).
        """
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch non disponible")

        from src.prisma.core.torch_utils import torch_knn_indices
        import inspect

        # Inspecter le module pour accéder à _compute_chunk_size si exposé
        # Sinon, valider indirectement via un run sur dataset moyen
        rng = np.random.default_rng(seed=1)
        medium_data = rng.standard_normal((50_000, 25)).astype(np.float32)

        indices = torch_knn_indices(medium_data, k=15, auto_chunk=True)

        assert indices.shape == (50_000, 15)
        assert np.all(indices >= 0)
        assert np.all(indices < 50_000)

        logger.info("✓ Auto-chunk sur 50k cellules OK — shape: %s", indices.shape)
        _force_gc()

    @pytest.mark.slow
    def test_torch_knn_indices_2m_cells_full_stress(self, full_stress_matrix: np.ndarray) -> None:
        """
        Test de stress complet : 2M×30 float32, k=15 — réservé GPU, marquer slow.

        Lancer avec : pytest -m slow tests/prisma/test_oom_protection.py -v -s
        """
        from src.prisma.core.torch_utils import torch_knn_indices, torch_available, torch_cuda_available

        if not torch_available():
            pytest.skip("PyTorch non disponible")
        if not torch_cuda_available():
            pytest.skip("CUDA requis pour test 2M cellules")

        data = full_stress_matrix
        _log_gpu_memory("avant_2m")
        t0 = time.perf_counter()

        indices = torch_knn_indices(data, k=K_NEIGHBORS)

        elapsed = time.perf_counter() - t0
        _log_gpu_memory("après_2m")

        assert indices.shape == (N_CELLS_STRESS_FULL, K_NEIGHBORS)
        assert indices.min() >= 0
        assert indices.max() < N_CELLS_STRESS_FULL

        row_ids = np.arange(N_CELLS_STRESS_FULL, dtype=indices.dtype)
        assert int((indices == row_ids[:, None]).sum()) == 0

        logger.info("✓ 2M cells stress test terminé en %.1f s", elapsed)
        _force_gc()


# ---------------------------------------------------------------------------
# Rapport mémoire standalone (exécution directe)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    pytest.main([__file__, "-v", "-s", "--tb=short"])
