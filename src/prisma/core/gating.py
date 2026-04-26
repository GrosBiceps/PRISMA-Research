"""
gating.py — Pre-gating séquentiel basé sur percentiles fixes (mode manuel).

PreGating implémente 4 gates hiérarchiques :
  G1: Débris      (SSC-A vs FSC-A, exclusion polygonale)
  G2: Doublets    (FSC-H vs FSC-A, ratio ou polygone)
  G3: Leucocytes  (CD45+, percentile bas)
  G4: Blastes     (CD34+ bright, percentile haut + SSC low optionnel)

Toutes les méthodes sont statiques et retournent un masque booléen numpy.
Pour le gating adaptatif par GMM, voir auto_gating.py.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


class PreGating:
    """
    Pre-gating automatique pour la sélection des populations d'intérêt.
    Basé sur FSC/SSC pour exclure les débris et les doublets.
    """

    @staticmethod
    def find_marker_index(var_names: List[str], patterns: List[str]) -> Optional[int]:
        """Trouve l'index d'un marqueur parmi les patterns donnés."""
        var_upper = [v.upper() for v in var_names]
        for pattern in patterns:
            for i, name in enumerate(var_upper):
                if pattern.upper() in name:
                    return i
        return None

    @staticmethod
    def gate_viable_cells(
        X: np.ndarray,
        var_names: List[str],
        min_percentile: float = 2.0,
        max_percentile: float = 98.0,
    ) -> np.ndarray:
        """
        Gate les cellules viables basé sur FSC/SSC.

        Args:
            X: Matrice des données (n_cells, n_markers)
            var_names: Liste des noms de marqueurs
            min_percentile: Percentile minimum (exclusion débris)
            max_percentile: Percentile maximum (exclusion doublets)

        Returns:
            Masque booléen des cellules viables
        """
        n_cells = X.shape[0]
        mask = np.ones(n_cells, dtype=bool)

        # Trouver FSC (priorité à FSC-A)
        fsc_idx = PreGating.find_marker_index(var_names, ["FSC-A", "FSC-H", "FSC"])
        if fsc_idx is not None:
            fsc_vals = X[:, fsc_idx].astype(np.float64)
            fsc_vals = np.where(np.isfinite(fsc_vals), fsc_vals, np.nan)
            low = np.nanpercentile(fsc_vals, min_percentile)
            high = np.nanpercentile(fsc_vals, max_percentile)
            mask &= np.isfinite(fsc_vals) & (fsc_vals >= low) & (fsc_vals <= high)

        # Trouver SSC (priorité à SSC-A)
        ssc_idx = PreGating.find_marker_index(var_names, ["SSC-A", "SSC-H", "SSC"])
        if ssc_idx is not None:
            ssc_vals = X[:, ssc_idx].astype(np.float64)
            ssc_vals = np.where(np.isfinite(ssc_vals), ssc_vals, np.nan)
            low = np.nanpercentile(ssc_vals, min_percentile)
            high = np.nanpercentile(ssc_vals, max_percentile)
            mask &= np.isfinite(ssc_vals) & (ssc_vals >= low) & (ssc_vals <= high)

        return mask

    @staticmethod
    def gate_singlets(
        X: np.ndarray,
        var_names: List[str],
        ratio_min: float = 0.6,
        ratio_max: float = 1.5,
    ) -> np.ndarray:
        """
        Gate les singlets basé sur le ratio FSC-A/FSC-H.
        Les doublets ont typiquement un ratio > 1.3-1.5.

        Args:
            X: Matrice des données
            var_names: Liste des noms de marqueurs
            ratio_min: Ratio minimum acceptable
            ratio_max: Ratio maximum acceptable

        Returns:
            Masque booléen des singlets
        """
        n_cells = X.shape[0]

        fsc_a_idx = PreGating.find_marker_index(var_names, ["FSC-A"])
        fsc_h_idx = PreGating.find_marker_index(var_names, ["FSC-H"])

        if fsc_a_idx is None or fsc_h_idx is None:
            print("[!] FSC-A ou FSC-H non trouvé, pas de gating singlets")
            return np.ones(n_cells, dtype=bool)

        fsc_a = X[:, fsc_a_idx].astype(np.float64)
        fsc_h = X[:, fsc_h_idx].astype(np.float64)

        # Valeurs minimum pour éviter division par zéro
        min_val = 100
        valid_h = fsc_h > min_val

        ratio = np.full(n_cells, np.nan)
        ratio[valid_h] = fsc_a[valid_h] / fsc_h[valid_h]

        mask = np.isfinite(ratio) & (ratio >= ratio_min) & (ratio <= ratio_max)

        return mask

    @staticmethod
    def gate_cd45_positive(
        X: np.ndarray, var_names: List[str], threshold_percentile: float = 10
    ) -> np.ndarray:
        """
        Gate les cellules CD45+ (leucocytes).

        Returns:
            Masque booléen des cellules CD45+
        """
        n_cells = X.shape[0]

        cd45_idx = PreGating.find_marker_index(
            var_names, ["CD45", "CD45-PECY5", "CD45-PC5"]
        )
        if cd45_idx is None:
            print("[!] CD45 non trouvé, pas de gating CD45+")
            return np.ones(n_cells, dtype=bool)

        cd45_vals = X[:, cd45_idx].astype(np.float64)
        cd45_vals = np.where(np.isfinite(cd45_vals), cd45_vals, np.nan)

        threshold = np.nanpercentile(cd45_vals, threshold_percentile)

        return np.where(np.isnan(cd45_vals), False, cd45_vals > threshold)

    @staticmethod
    def gate_cd34_blasts(
        X: np.ndarray,
        var_names: List[str],
        threshold_percentile: float = 85,
        use_ssc_filter: bool = True,
        ssc_max_percentile: float = 70,
    ) -> np.ndarray:
        """
        Gate les blastes CD34+ (cellules souches/progénitrices).

        Les blastes sont typiquement:
        - CD34 bright (haute expression)
        - SSC low (faible granularité)

        Args:
            X: Matrice des données (n_cells, n_markers)
            var_names: Liste des noms de marqueurs
            threshold_percentile: Percentile pour définir le seuil CD34+ (ex: 85 = top 15%)
            use_ssc_filter: Appliquer aussi un filtre SSC pour enrichir en blastes
            ssc_max_percentile: Percentile max de SSC pour blastes (faible granularité)

        Returns:
            Masque booléen des blastes CD34+
        """
        n_cells = X.shape[0]

        # Chercher CD34 avec différents nommages possibles
        cd34_idx = PreGating.find_marker_index(
            var_names, ["CD34", "CD34-PE", "CD34-APC", "CD34-PECY7"]
        )
        if cd34_idx is None:
            print("[!] CD34 non trouvé, pas de gating blastes")
            return np.ones(n_cells, dtype=bool)

        cd34_vals = X[:, cd34_idx].astype(np.float64)
        cd34_vals = np.where(np.isfinite(cd34_vals), cd34_vals, np.nan)

        # Seuil CD34+ (prendre les cellules avec haute expression)
        threshold_cd34 = np.nanpercentile(cd34_vals, threshold_percentile)
        mask_cd34 = np.where(np.isnan(cd34_vals), False, cd34_vals >= threshold_cd34)

        # Optionnel: filtrer aussi par SSC low (blastes = faible granularité)
        if use_ssc_filter:
            ssc_idx = PreGating.find_marker_index(var_names, ["SSC-A", "SSC-H", "SSC"])
            if ssc_idx is not None:
                ssc_vals = X[:, ssc_idx].astype(np.float64)
                ssc_vals = np.where(np.isfinite(ssc_vals), ssc_vals, np.nan)
                threshold_ssc = np.nanpercentile(ssc_vals, ssc_max_percentile)
                mask_ssc = np.where(
                    np.isnan(ssc_vals), False, ssc_vals <= threshold_ssc
                )
                return mask_cd34 & mask_ssc

        return mask_cd34

    @staticmethod
    def gate_debris_polygon(
        X: np.ndarray,
        var_names: List[str],
        fsc_min: float = None,
        fsc_max: float = None,
        ssc_min: float = None,
        ssc_max: float = None,
        auto_percentiles: bool = True,
        min_pct: float = 1.0,
        max_pct: float = 99.0,
    ) -> np.ndarray:
        """
        Gate rectangulaire/polygonal pour exclure les débris sur FSC-A vs SSC-A.

        Args:
            X: Matrice des données
            var_names: Liste des noms de marqueurs
            fsc_min/fsc_max: Seuils FSC manuels (si auto_percentiles=False)
            ssc_min/ssc_max: Seuils SSC manuels (si auto_percentiles=False)
            auto_percentiles: Calculer automatiquement les seuils via percentiles
            min_pct/max_pct: Percentiles pour auto-calcul

        Returns:
            Masque booléen des cellules (non-débris)
        """
        n_cells = X.shape[0]

        fsc_idx = PreGating.find_marker_index(var_names, ["FSC-A"])
        ssc_idx = PreGating.find_marker_index(var_names, ["SSC-A"])

        if fsc_idx is None or ssc_idx is None:
            print("[!] FSC-A ou SSC-A non trouvé pour gate débris")
            return np.ones(n_cells, dtype=bool)

        fsc_vals = X[:, fsc_idx].astype(np.float64)
        ssc_vals = X[:, ssc_idx].astype(np.float64)

        # Calculer les seuils automatiquement si demandé
        if auto_percentiles:
            fsc_min = np.nanpercentile(fsc_vals, min_pct)
            fsc_max = np.nanpercentile(fsc_vals, max_pct)
            ssc_min = np.nanpercentile(ssc_vals, min_pct)
            ssc_max = np.nanpercentile(ssc_vals, max_pct)

        # Appliquer le gate rectangulaire
        mask = (
            np.isfinite(fsc_vals)
            & np.isfinite(ssc_vals)
            & (fsc_vals >= fsc_min)
            & (fsc_vals <= fsc_max)
            & (ssc_vals >= ssc_min)
            & (ssc_vals <= ssc_max)
        )

        return mask
