"""
transformers.py — Transformations cytométriques et normalisation.

Toutes les méthodes sont statiques et sans état : elles transforment
un ndarray et le retournent. La sélection de la méthode à appliquer
est déléguée à PreprocessingService.

Méthodes disponibles (monolithe exact + apply() dispatcher):
    - arcsinh_transform  : Standard pour cytométrie en flux (cofacteur 5)
    - arcsinh_inverse    : Inverse de l'arcsinh
    - logicle_transform  : Biexponentielle précise (FlowKit si disponible)
    - log_transform      : Log base 10 sur valeurs positives
    - zscore_normalize   : Centrage-réduction (moyenne=0, std=1)
    - min_max_normalize  : Mise à l'échelle [0, 1]
    - apply()            : Dispatcher (architecture)
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional

import numpy as np

_logger = logging.getLogger("prisma.transformers")

# FlowKit pour la transformation Logicle précise (comportement dégradé si absent).
# On attrape Exception (pas seulement ImportError) car flowutils.logicle_c
# peut lever AttributeError si compilé contre NumPy 1.x et exécuté sous NumPy 2.x.
try:
    import flowkit as fk

    FLOWKIT_AVAILABLE = True
except Exception:
    FLOWKIT_AVAILABLE = False
    fk = None


class DataTransformer:
    """
    Transformations de données de cytométrie (Logicle, Arcsinh, etc.).
    Classe statique réutilisable sans dépendance à l'UI.
    """

    @staticmethod
    def arcsinh_transform(data: np.ndarray, cofactor: float = 5.0) -> np.ndarray:
        """
        Transformation Arcsinh (inverse hyperbolic sine).

        Args en entrée:
            data: Matrice de données (n_cells, n_markers)
            cofactor: Facteur de division (5 pour flow cytometry)

        Returns:
            Données transformées
        """
        return np.arcsinh(data / cofactor)

    @staticmethod
    def arcsinh_inverse(data: np.ndarray, cofactor: float = 5.0) -> np.ndarray:
        """Inverse de la transformation Arcsinh."""
        return np.sinh(data) * cofactor

    @staticmethod
    def logicle_transform(
        data: np.ndarray,
        T: float = 262144.0,
        M: float = 4.5,
        W: float = 0.5,
        A: float = 0.0,
    ) -> np.ndarray:
        """
        Transformation Logicle (biexponentielle).

        Args en entrée:
            data: Matrice de données
            T: Maximum de l'échelle linéaire (262144 = 2^18)
            M: Décades de largeur
            W: Linéarisation près de zéro
            A: Décades additionnelles (négatifs)

        Returns:
            Données transformées
        """
        if FLOWKIT_AVAILABLE:
            # Utiliser FlowKit si disponible (plus précis) avec une fonction prédéfinie
            try:
                xform = fk.transforms.LogicleTransform(T=T, M=M, W=W, A=A)
                return xform.apply(data)
            except Exception as exc:
                # Repli logicle → arcsinh : différence clinique non négligeable,
                # on trace explicitement pour audit (jamais de repli silencieux).
                _logger.warning(
                    "LogicleTransform FlowKit échoué (%s: %s) — repli arcsinh approximé",
                    type(exc).__name__, exc,
                )

        # Approximation si FlowKit absent: Arcsinh modifié.
        # NB : le paramètre W (largeur linéaire) n'intervient pas dans cette
        # approximation arcsinh — c'est précisément pourquoi le repli est tracé.
        return np.arcsinh(data / (T / (10**M))) * (M / np.log(10))

    @staticmethod
    def log_transform(
        data: np.ndarray, base: float = 10.0, min_val: float = 1.0
    ) -> np.ndarray:
        """Transformation logarithmique standard."""
        data_clipped = np.maximum(data, min_val)
        return np.log(data_clipped) / np.log(base)

    @staticmethod
    def zscore_normalize(data: np.ndarray) -> np.ndarray:
        """Normalisation Z-score (moyenne=0, std=1)."""
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        std[std == 0] = 1  # Éviter division par zéro
        return (data - mean) / std

    @staticmethod
    def min_max_normalize(data: np.ndarray) -> np.ndarray:
        """Normalisation Min-Max [0, 1]."""
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        return (data - min_val) / range_val
    @staticmethod
    def apply(
        data,
        method: str,
        cofactor: float = 5.0,
        var_names=None,
        apply_to_scatter: bool = False,
    ):
        """
        Applique la transformation spécifiée, avec gestion optionnelle
        des canaux scatter (FSC/SSC/Time).

        Args:
            data: Matrice (n_cells, n_markers).
            method: 'arcsinh' | 'logicle' | 'log10' | 'log' | 'none'.
            cofactor: Cofacteur pour arcsinh.
            var_names: Noms des colonnes.
            apply_to_scatter: Si True, transforme TOUS les canaux.

        Returns:
            Données transformées (même shape que data).
        """
        import re
        import numpy as np
        try:
            from config.constants import SCATTER_PATTERNS
        except Exception:
            SCATTER_PATTERNS = ["FSC", "SSC", "TIME", "EVENT"]

        method_lc = method.lower()
        if method_lc == "none":
            return data.copy() if hasattr(data, "copy") else np.array(data)

        if var_names is not None and not apply_to_scatter:
            fluorescence_idx = [
                i
                for i, name in enumerate(var_names)
                if not any(re.search(p, name, re.IGNORECASE) for p in SCATTER_PATTERNS)
            ]
        else:
            fluorescence_idx = list(range(data.shape[1]))

        result = data.copy() if hasattr(data, "copy") else np.array(data, dtype=float)
        if not fluorescence_idx:
            return result
        sub = result[:, fluorescence_idx]

        if method_lc == "arcsinh":
            transformed = DataTransformer.arcsinh_transform(sub, cofactor=cofactor)
        elif method_lc == "logicle":
            transformed = DataTransformer.logicle_transform(sub)
        elif method_lc in ("log10", "log"):
            transformed = DataTransformer.log_transform(sub)
        else:
            raise ValueError(
                f"Méthode de transformation inconnue: '{method}'. "
                "Valeurs acceptées: 'arcsinh', 'logicle', 'log10', 'none'."
            )
        result[:, fluorescence_idx] = transformed
        return result

    @staticmethod
    def apply_per_column_transforms(
        data: np.ndarray,
        var_names: List[str],
        specs: dict,
    ) -> np.ndarray:
        """Applique une transformation **par colonne** selon des specs explicites.

        Permet de mélanger logicle/arcsinh/log/none sur des canaux différents
        (ex. logicle sur les fluo, none sur FSC/SSC), avec cofacteur/paramètres
        logicle propres à chaque colonne.

        Args:
            data: Matrice (n_cells, n_markers).
            var_names: Noms des colonnes (ordre = colonnes de data).
            specs: Dict {nom_colonne: {"method": str, "cofactor": float,
                   "T": float, "M": float, "W": float, "A": float}}.
                   Les colonnes absentes du dict sont laissées inchangées.

        Returns:
            Matrice transformée (même shape).
        """
        result = np.array(data, dtype=np.float32, copy=True)
        name_to_idx = {str(n): i for i, n in enumerate(var_names)}

        for col, spec in (specs or {}).items():
            idx = name_to_idx.get(str(col))
            if idx is None:
                _logger.debug("apply_per_column_transforms: colonne '%s' absente — ignorée", col)
                continue
            method = str(spec.get("method", "none")).lower()
            if method == "none":
                continue
            sub = result[:, idx : idx + 1]
            try:
                if method == "arcsinh":
                    cof = float(spec.get("cofactor", 5.0) or 5.0)
                    out = DataTransformer.arcsinh_transform(sub, cofactor=cof)
                elif method == "logicle":
                    out = DataTransformer.logicle_transform(
                        sub,
                        T=float(spec.get("T", 262144.0)),
                        M=float(spec.get("M", 4.5)),
                        W=float(spec.get("W", 0.5)),
                        A=float(spec.get("A", 0.0)),
                    )
                elif method in ("log10", "log"):
                    out = DataTransformer.log_transform(sub)
                else:
                    _logger.warning("Méthode '%s' inconnue pour '%s' — colonne inchangée", method, col)
                    continue
                result[:, idx : idx + 1] = np.asarray(out, dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("Transformation '%s' échouée sur '%s' : %s", method, col, exc)
        return result

