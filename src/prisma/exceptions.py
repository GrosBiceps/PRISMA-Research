# -*- coding: utf-8 -*-
"""
exceptions.py — Hiérarchie d'exceptions PRISMA Research.

Toutes les exceptions remontent jusqu'à PRISMAError. La GUI et le CLI
capturent PRISMAError pour afficher un message explicite sans crash silencieux.

Hiérarchie :
    PRISMAError
    ├── DataError              — données d'entrée invalides ou corrompues
    │   ├── TooFewCellsError   — échantillon rejeté : trop peu de cellules
    │   ├── NaNDataError       — NaN détectés, imputation impossible ou refusée
    │   └── MarkerMismatchError — marqueurs attendus absents du fichier
    ├── GatingError            — échec d'une étape de gating
    │   └── GatingRejectedError — gate a éliminé trop de cellules
    ├── ProcessingError        — erreur dans la chaîne de preprocessing
    ├── ClusteringError        — erreur dans le pipeline FlowSOM/métaclustering
    ├── MRDError               — erreur dans le calcul MRD
    ├── ConfigError            — configuration invalide (ancien PanelConfigError)
    │   └── PanelConfigError   — panel non sélectionné ou fichier absent
    └── PipelineStageError     — étape de pipeline échouée (ancien nom conservé)

Les exceptions cliniquement critiques (ClinicalMathError, PanelConfigError,
PipelineStageError) sont maintenues pour la rétrocompatibilité.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Racine
# ---------------------------------------------------------------------------

class PRISMAError(Exception):
    """Exception racine de PRISMA Research. Toutes les exceptions en héritent."""


# ---------------------------------------------------------------------------
# DataError — données d'entrée
# ---------------------------------------------------------------------------

class DataError(PRISMAError):
    """Données d'entrée invalides, manquantes ou corrompues."""


class TooFewCellsError(DataError):
    """
    Échantillon rejeté car il contient trop peu de cellules.

    Attributes:
        sample_name: Nom du fichier FCS concerné.
        n_cells:     Nombre de cellules trouvées.
        min_cells:   Seuil minimum requis.
    """

    def __init__(
        self,
        sample_name: str,
        n_cells: int,
        min_cells: int,
        stage: str = "QC",
    ) -> None:
        self.sample_name = sample_name
        self.n_cells = n_cells
        self.min_cells = min_cells
        self.stage = stage
        super().__init__(
            f"[{stage}] {sample_name!r}: {n_cells} cellules < minimum requis "
            f"({min_cells}). Échantillon rejeté."
        )


class NaNDataError(DataError):
    """
    Valeurs NaN détectées dans la matrice de données.

    Attributes:
        sample_name: Nom du fichier FCS concerné.
        n_nan:       Nombre de valeurs NaN détectées.
        action:      Action prise ('imputed_zero', 'rejected', …).
    """

    def __init__(
        self,
        sample_name: str,
        n_nan: int,
        action: str = "imputed_zero",
    ) -> None:
        self.sample_name = sample_name
        self.n_nan = n_nan
        self.action = action
        super().__init__(
            f"{sample_name!r}: {n_nan} valeur(s) NaN détectée(s) — "
            f"action: {action}."
        )


class MarkerMismatchError(DataError):
    """
    Marqueurs attendus absents du fichier FCS.

    Attributes:
        sample_name: Nom du fichier FCS.
        missing:     Marqueurs introuvables.
        available:   Marqueurs disponibles dans le fichier.
    """

    def __init__(
        self,
        sample_name: str,
        missing: list[str],
        available: list[str] | None = None,
    ) -> None:
        self.sample_name = sample_name
        self.missing = missing
        self.available = available or []
        super().__init__(
            f"{sample_name!r}: marqueurs manquants {missing}. "
            f"Disponibles: {self.available or '(non fournis)'}."
        )


# ---------------------------------------------------------------------------
# GatingError — gating
# ---------------------------------------------------------------------------

class GatingError(PRISMAError):
    """Erreur dans une étape de gating."""

    def __init__(self, gate_name: str, message: str) -> None:
        self.gate_name = gate_name
        super().__init__(f"[Gate:{gate_name}] {message}")


class GatingRejectedError(GatingError):
    """
    Gate a éliminé trop de cellules — résultat en dessous du seuil minimal.

    Attributes:
        gate_name:  Nom du gate (ex: 'G1_debris', 'G3_cd45').
        n_kept:     Cellules conservées.
        n_total:    Cellules avant ce gate.
        min_cells:  Seuil minimum requis après gate.
    """

    def __init__(
        self,
        gate_name: str,
        n_kept: int,
        n_total: int,
        min_cells: int,
    ) -> None:
        self.n_kept = n_kept
        self.n_total = n_total
        self.min_cells = min_cells
        pct = (n_kept / max(n_total, 1)) * 100
        super().__init__(
            gate_name,
            f"{n_kept}/{n_total} cellules conservées ({pct:.1f}%) "
            f"< minimum requis ({min_cells}).",
        )


# ---------------------------------------------------------------------------
# ProcessingError — prétraitement
# ---------------------------------------------------------------------------

class ProcessingError(PRISMAError):
    """Erreur dans la chaîne de prétraitement (transformation, normalisation, …)."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


# ---------------------------------------------------------------------------
# ClusteringError — FlowSOM / métaclustering
# ---------------------------------------------------------------------------

class ClusteringError(PRISMAError):
    """Erreur dans le pipeline de clustering FlowSOM ou métaclustering."""

    def __init__(self, message: str, details: str | None = None) -> None:
        self.details = details
        full = message + (f" — {details}" if details else "")
        super().__init__(full)


# ---------------------------------------------------------------------------
# MRDError — calcul MRD
# ---------------------------------------------------------------------------

class MRDError(PRISMAError):
    """Erreur dans le calcul MRD (données insuffisantes, méthode invalide, …)."""

    def __init__(self, method: str, message: str) -> None:
        self.method = method
        super().__init__(f"[MRD:{method}] {message}")


# ---------------------------------------------------------------------------
# ConfigError — configuration
# ---------------------------------------------------------------------------

class ConfigError(PRISMAError):
    """Configuration invalide ou manquante."""


class PanelConfigError(ConfigError):
    """
    Aucun panel sélectionné ou fichier de configuration du panel absent.

    Rétrocompatible avec l'ancien PanelConfigError (même interface).
    """

    def __init__(self, message: str, panel_path: str | None = None) -> None:
        self.message = message
        self.panel_path = panel_path
        full_msg = message
        if panel_path:
            full_msg += f" (chemin : {panel_path})"
        super().__init__(full_msg)


# ---------------------------------------------------------------------------
# PipelineStageError — compatibilité rétrograde
# ---------------------------------------------------------------------------

class PipelineStageError(PRISMAError):
    """
    Erreur dans une étape nommée du pipeline.

    Rétrocompatible avec l'ancien PipelineStageError.
    """

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        self.message = message
        super().__init__(f"[{stage}] {message}")


# ---------------------------------------------------------------------------
# ClinicalMathError — compatibilité rétrograde
# ---------------------------------------------------------------------------

class ClinicalMathError(PRISMAError):
    """
    Erreur mathématique à impact clinique potentiel.

    Levée quand une opération critique (inversion de matrice, Mahalanobis)
    ne peut pas être effectuée de manière fiable.

    Rétrocompatible avec l'ancien ClinicalMathError.

    Attributes:
        message:          Description lisible.
        condition_number: Conditionnement de la matrice (si applicable).
        details:          Informations techniques supplémentaires.
    """

    def __init__(
        self,
        message: str,
        condition_number: float | None = None,
        details: str | None = None,
    ) -> None:
        self.message = message
        self.condition_number = condition_number
        self.details = details
        full_msg = message
        if condition_number is not None:
            full_msg += f" (cond={condition_number:.2e})"
        if details:
            full_msg += f" — {details}"
        super().__init__(full_msg)
