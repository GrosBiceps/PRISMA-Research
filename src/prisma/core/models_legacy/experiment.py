"""
models/experiment.py — Conteneur de cohorte multi-sample.

Experiment regroupe plusieurs Sample liés à une même étude clinique ou
expérimentale. Il gère l'alignement entre identifiants de samples et
métadonnées de cohorte, et expose des méthodes de concatenation préparées
pour les analyses batch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

import pandas as pd

from .sample import Sample


class ExperimentError(ValueError):
    """Erreur de cohérence dans un Experiment."""


class Experiment:
    """
    Conteneur de cohorte : regroupe N Samples avec leurs métadonnées cliniques.

    Attributes:
        experiment_id: UUID unique de l'expérience.
        name: Nom court de l'expérience.
        samples: Liste ordonnée de Sample.
        cohort_metadata: DataFrame (N_samples × M_variables) indexé sur sample_id.
        description: Description libre.
        tags: Labels libres (panel, protocole, institution…).
    """

    def __init__(
        self,
        name: str,
        experiment_id: Optional[str] = None,
        cohort_metadata: Optional[pd.DataFrame] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> None:
        self.experiment_id: str = experiment_id or str(uuid4())
        self.name: str = name
        self.description: str = description
        self.tags: List[str] = tags or []
        self._samples: Dict[str, Sample] = {}  # sample_id → Sample
        self.cohort_metadata: pd.DataFrame = (
            cohort_metadata if cohort_metadata is not None else pd.DataFrame()
        )

    # ------------------------------------------------------------------
    # Gestion des samples
    # ------------------------------------------------------------------

    @property
    def samples(self) -> List[Sample]:
        """Samples dans l'ordre d'insertion."""
        return list(self._samples.values())

    @property
    def sample_ids(self) -> List[str]:
        return list(self._samples.keys())

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    def add_sample(self, sample: Sample) -> None:
        """
        Ajoute un Sample à l'expérience.

        Si cohort_metadata existe déjà et contient sample_id comme index,
        la ligne correspondante est conservée. Si l'index ne contient pas
        ce sample_id, une ligne vide est insérée pour maintenir l'alignement.

        Args:
            sample: Sample à ajouter.

        Raises:
            ExperimentError: Si sample_id déjà présent.
        """
        if sample.sample_id in self._samples:
            raise ExperimentError(
                f"Sample '{sample.sample_id}' déjà dans l'expérience '{self.name}'."
            )
        self._samples[sample.sample_id] = sample

        if not self.cohort_metadata.empty and sample.sample_id not in self.cohort_metadata.index:
            empty_row = pd.DataFrame(
                [{col: None for col in self.cohort_metadata.columns}],
                index=[sample.sample_id],
            )
            self.cohort_metadata = pd.concat([self.cohort_metadata, empty_row])

    def remove_sample(self, sample_id: str) -> Sample:
        """
        Retire et retourne un Sample par son identifiant.

        Args:
            sample_id: Identifiant du Sample à retirer.

        Returns:
            Le Sample retiré.

        Raises:
            KeyError: Si sample_id introuvable.
        """
        if sample_id not in self._samples:
            raise KeyError(f"sample_id '{sample_id}' introuvable dans '{self.name}'.")
        sample = self._samples.pop(sample_id)
        if not self.cohort_metadata.empty and sample_id in self.cohort_metadata.index:
            self.cohort_metadata = self.cohort_metadata.drop(index=sample_id)
        return sample

    def get_sample(self, sample_id: str) -> Sample:
        """
        Retourne un Sample par son identifiant.

        Args:
            sample_id: UUID du Sample.

        Returns:
            Sample correspondant.

        Raises:
            KeyError: Si absent.
        """
        if sample_id not in self._samples:
            raise KeyError(f"sample_id '{sample_id}' introuvable. Disponibles: {self.sample_ids}")
        return self._samples[sample_id]

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def list_channels(self) -> List[str]:
        """
        Retourne l'union des canaux primaires sur tous les samples.

        Returns:
            Liste triée de noms de canaux.
        """
        all_channels: set[str] = set()
        for s in self.samples:
            all_channels.update(s.channels)
        return sorted(all_channels)

    def common_channels(self) -> List[str]:
        """
        Retourne l'intersection des canaux présents dans tous les samples.

        Returns:
            Liste triée des canaux communs.
        """
        if not self._samples:
            return []
        sets = [set(s.channels) for s in self.samples]
        common = sets[0].intersection(*sets[1:])
        return sorted(common)

    # ------------------------------------------------------------------
    # Concatenation
    # ------------------------------------------------------------------

    def concatenate_pre_gated(
        self,
        channels: Optional[List[str]] = None,
        sample_ids: Optional[Sequence[str]] = None,
        add_batch_columns: bool = True,
    ) -> pd.DataFrame:
        """
        Concatène les données pré-gatées de plusieurs (ou tous) samples.

        Applique tous les masques de chaque Sample (AND logique via
        get_pre_gated_data) avant concatenation.

        Args:
            channels: Canaux à inclure ; None = canaux communs à tous.
            sample_ids: Subset de samples à inclure ; None = tous.
            add_batch_columns: Si True, ajoute 'sample_id' et les colonnes
                               de cohort_metadata à chaque ligne.

        Returns:
            DataFrame concatené (N_total_events × M_channels [+ batch cols]).

        Raises:
            KeyError: Si un sample_id demandé est absent.
            ExperimentError: Si aucun sample disponible.
        """
        if not self._samples:
            raise ExperimentError("Aucun sample dans l'expérience.")

        target_ids = list(sample_ids) if sample_ids is not None else self.sample_ids
        for sid in target_ids:
            if sid not in self._samples:
                raise KeyError(f"sample_id '{sid}' introuvable.")

        if channels is None:
            channels = self.common_channels()

        parts: List[pd.DataFrame] = []
        for sid in target_ids:
            s = self._samples[sid]
            df = s.get_pre_gated_data(channels if channels else None).copy()
            if add_batch_columns:
                df["sample_id"] = sid
                if not self.cohort_metadata.empty and sid in self.cohort_metadata.index:
                    for col in self.cohort_metadata.columns:
                        df[col] = self.cohort_metadata.loc[sid, col]
            parts.append(df)

        return pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------

    def summarize(self) -> pd.DataFrame:
        """
        Retourne un DataFrame récapitulatif par sample.

        Colonnes : sample_id, n_events, n_channels, masks, embeddings, clusters.

        Returns:
            DataFrame (N_samples × 6).
        """
        rows: List[Dict[str, Any]] = []
        for s in self.samples:
            rows.append({
                "sample_id": s.sample_id,
                "n_events": s.n_events,
                "n_channels": len(s.channels),
                "masks": list(s.masks.keys()),
                "embeddings": list(s.embeddings.keys()),
                "clusters": list(s.cluster_assignments.keys()),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Cohort metadata helpers
    # ------------------------------------------------------------------

    def set_cohort_metadata(self, df: pd.DataFrame) -> None:
        """
        Affecte un nouveau DataFrame de métadonnées de cohorte.

        L'index du DataFrame doit être les sample_ids.

        Args:
            df: DataFrame indexé sur sample_id.

        Raises:
            ExperimentError: Si des sample_ids enregistrés sont absents de l'index.
        """
        missing = set(self.sample_ids) - set(df.index)
        if missing:
            raise ExperimentError(
                f"cohort_metadata manque les sample_ids suivants: {sorted(missing)}"
            )
        self.cohort_metadata = df

    def get_cohort_info(self, sample_id: str) -> Dict[str, Any]:
        """
        Retourne les métadonnées de cohorte d'un sample donné.

        Args:
            sample_id: Identifiant du sample.

        Returns:
            Dict colonne → valeur.

        Raises:
            KeyError: Si sample_id absent de cohort_metadata.
        """
        if self.cohort_metadata.empty or sample_id not in self.cohort_metadata.index:
            raise KeyError(f"Pas de métadonnées de cohorte pour '{sample_id}'.")
        return self.cohort_metadata.loc[sample_id].to_dict()

    # ------------------------------------------------------------------
    # Représentation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Experiment(name={self.name!r}, "
            f"id={self.experiment_id[:8]}…, "
            f"samples={self.n_samples}, "
            f"channels={len(self.list_channels())})"
        )
