"""
core/interfaces.py — Interfaces abstraites du pipeline PRISMA Research.

Trois contrats ABC garantissent des signatures stables pour :
  - INodeProcessor  : traitement unitaire sur Sample ou Experiment
  - IAnalysisStrategy : algorithme analytique (clustering, embedding, scoring…)
  - IExportStrategy   : sérialisation des résultats vers un format de sortie

Ces interfaces sont conçues pour extension : toute classe concrète
hérite de l'ABC correspondant et implémente les méthodes abstraites.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Imports des modèles canoniques — relatifs dans le package, absolus en fallback
try:
    from ..models.sample import Sample
    from ..models.experiment import Experiment
except ImportError:
    from models.sample import Sample  # type: ignore[no-redef]
    from models.experiment import Experiment  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# Alias de type pour la cible d'un processor
ProcessorTarget = Union[Sample, Experiment]


# ---------------------------------------------------------------------------
# INodeProcessor
# ---------------------------------------------------------------------------

class INodeProcessor(ABC):
    """
    Interface abstraite pour un nœud de traitement du pipeline.

    Un INodeProcessor reçoit un Sample ou un Experiment en entrée, applique
    une transformation (gating, normalisation, compensation…) et retourne
    l'objet modifié (ou un nouvel objet si immutabilité requise).

    Contrainte d'implémentation :
      - process() NE DOIT PAS muter silencieusement les dimensions (N_events).
        Toute réduction du nombre d'événements doit passer par Sample.set_mask()
        ou Sample.subset(), jamais par remplacement direct de Sample.events.
      - validate_input() doit être appelée avant process() par le runner.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifiant lisible du nœud (ex. 'arcsinh_transform')."""

    @abstractmethod
    def validate_input(self, target: ProcessorTarget) -> None:
        """
        Vérifie que target est exploitable par ce processor.

        Args:
            target: Sample ou Experiment à valider.

        Raises:
            ValueError: Si target ne satisfait pas les prérequis.
            TypeError: Si target n'est pas du bon type.
        """

    @abstractmethod
    def process(self, target: ProcessorTarget) -> ProcessorTarget:
        """
        Applique la transformation et retourne la cible mise à jour.

        Args:
            target: Sample ou Experiment à transformer.

        Returns:
            Sample ou Experiment modifié (peut être le même objet ou un nouveau).

        Raises:
            RuntimeError: En cas d'échec du traitement.
        """

    def run(self, target: ProcessorTarget) -> ProcessorTarget:
        """
        Point d'entrée standard : valide puis traite.

        Wrapper non-surchargeable garantissant l'ordre validate → process.

        Args:
            target: Cible à traiter.

        Returns:
            Résultat de process().
        """
        logger.debug("[%s] validate_input…", self.name)
        self.validate_input(target)
        logger.debug("[%s] process…", self.name)
        result = self.process(target)
        logger.debug("[%s] done.", self.name)
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# IAnalysisStrategy
# ---------------------------------------------------------------------------

class IAnalysisStrategy(ABC):
    """
    Interface abstraite pour les algorithmes analytiques PRISMA.

    Couvre : clustering, réduction dimensionnelle, détection de population,
    scoring MRD, unmixing spectral, etc.

    Distinction avec INodeProcessor :
      - INodeProcessor opère sur Sample/Experiment et retourne le même type.
      - IAnalysisStrategy prend une matrice numpy brute et retourne un résultat
        analytique (embedding, labels, scores…). Le caller est responsable
        d'injecter le résultat dans le Sample via add_embedding(),
        add_cluster_labels(), etc.

    Design :
      - fit() et transform() sont séparés pour permettre la réutilisation du
        modèle ajusté sur de nouveaux données (ex. : inférence batch).
      - fit_transform() est fourni comme raccourci par défaut.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifiant de la stratégie (ex. 'umap', 'flowsom', 'mrd_scorer')."""

    @abstractmethod
    def fit(self, data: np.ndarray, **kwargs: Any) -> "IAnalysisStrategy":
        """
        Entraîne la stratégie sur les données fournies.

        Args:
            data: Matrice (N_events × N_features).
            **kwargs: Hyperparamètres spécifiques à la stratégie.

        Returns:
            self (pour chaînage).
        """

    @abstractmethod
    def transform(self, data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Applique la stratégie ajustée à de nouvelles données.

        Args:
            data: Matrice (N_events × N_features).
            **kwargs: Options supplémentaires.

        Returns:
            Résultat (N_events × N_out) ou (N_events,) selon la stratégie.
        """

    def fit_transform(self, data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Raccourci fit() suivi de transform() sur les mêmes données.

        Peut être surchargé pour une implémentation plus efficace.

        Args:
            data: Matrice (N_events × N_features).
            **kwargs: Hyperparamètres.

        Returns:
            Résultat de transform().
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)

    @property
    def is_fitted(self) -> bool:
        """Indique si fit() a été appelé. Surchargeable."""
        return False

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Retourne les hyperparamètres courants sous forme de dict sérialisable.

        Returns:
            Dict param_name → valeur.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# IExportStrategy
# ---------------------------------------------------------------------------

class IExportStrategy(ABC):
    """
    Interface abstraite pour la sérialisation des résultats.

    Exemples d'implémentations concrètes :
      - FCSExportStrategy   → écriture FCS (Kaluza-compatible)
      - CSVExportStrategy   → export DataFrame
      - JSONExportStrategy  → rapport JSON
      - PDFReportStrategy   → rapport PDF clinique
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifiant de la stratégie d'export (ex. 'fcs', 'csv', 'pdf')."""

    @abstractmethod
    def export(
        self,
        source: Union[Sample, Experiment, pd.DataFrame, np.ndarray],
        output_path: Path,
        **kwargs: Any,
    ) -> Path:
        """
        Sérialise source vers output_path.

        Args:
            source: Données à exporter.
            output_path: Chemin de destination (fichier ou dossier selon stratégie).
            **kwargs: Options spécifiques au format (compression, encoding…).

        Returns:
            Chemin effectif du fichier produit.

        Raises:
            IOError: En cas d'échec d'écriture.
            TypeError: Si source n'est pas supporté par cette stratégie.
        """

    @abstractmethod
    def supported_types(self) -> List[type]:
        """
        Retourne les types acceptés par export().

        Returns:
            Liste de types (ex. [Sample, pd.DataFrame]).
        """

    def can_export(self, source: Any) -> bool:
        """Vérifie si source est d'un type supporté."""
        return isinstance(source, tuple(self.supported_types()))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
