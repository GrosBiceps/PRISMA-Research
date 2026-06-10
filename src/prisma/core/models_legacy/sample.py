"""
models/sample.py — Modèle de données central orienté analyse RUO.

Sample encapsule un fichier FCS avec :
  - événements bruts + canaux compensés
  - masques de gating nommés (bool, longueur N)
  - canaux virtuels (dérivés ou unmixed)
  - embeddings dimensionnels (UMAP, t-SNE…)
  - labels de clustering
  - métadonnées par-sample (lot, opérateur, date…)

Toutes les structures indexées sur les événements ont longueur N stricte.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd


class SampleValidationError(ValueError):
    """Violation de cohérence dimensionnelle dans un Sample."""


@dataclass
class ChannelMeta:
    """Métadonnées d'un canal FCS ($PnN, $PnS, plage de détection…)."""

    name: str
    label: str = ""
    detector: str = ""
    pne_range: Optional[tuple[float, float]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class Sample:
    """
    Représente un échantillon cytométrique complet, exploitable dans tout le pipeline.

    Attributes:
        sample_id: Identifiant unique (UUID4 par défaut).
        events: DataFrame (N_events × N_channels) — données compensées/transformées.
        channel_metadata: Métadonnées par canal (dict nom→ChannelMeta ou DataFrame).
        compensation_matrix: Matrice de compensation (N_channels × N_channels) ou None.
        masks: Masques de gating booléens nommés, chacun de longueur N_events.
        virtual_channels: Canaux dérivés (unmixing, ratios…), longueur N_events.
        embeddings: Embeddings dimensionnels, dict nom→ndarray (N_events × n_dims).
        cluster_assignments: Labels de cluster, dict nom→ndarray (N_events,).
        per_sample_metadata: Métadonnées libres (lot, opérateur, date, panel…).
        artifacts: Chemins des artefacts produits (plots, exports…).
        results: Résultats analytiques libres (MRD %, population counts…).
    """

    def __init__(
        self,
        sample_id: Optional[str] = None,
        events: Optional[pd.DataFrame] = None,
        channel_metadata: Optional[Dict[str, Any] | pd.DataFrame] = None,
        compensation_matrix: Optional[np.ndarray] = None,
        per_sample_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.sample_id: str = sample_id or str(uuid4())
        self.events: pd.DataFrame = events if events is not None else pd.DataFrame()
        self.channel_metadata: Dict[str, Any] | pd.DataFrame = (
            channel_metadata if channel_metadata is not None else {}
        )
        self.compensation_matrix: Optional[np.ndarray] = compensation_matrix
        self.masks: Dict[str, np.ndarray] = {}
        self.virtual_channels: Dict[str, np.ndarray] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.cluster_assignments: Dict[str, np.ndarray] = {}
        self.per_sample_metadata: Dict[str, Any] = per_sample_metadata or {}
        self.artifacts: List[str] = []
        self.results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Propriétés de base
    # ------------------------------------------------------------------

    @property
    def n_events(self) -> int:
        """Nombre d'événements (cellules) dans le Sample."""
        return len(self.events)

    @property
    def channels(self) -> List[str]:
        """Noms des canaux primaires (colonnes de events)."""
        return list(self.events.columns)

    # ------------------------------------------------------------------
    # Accès aux données
    # ------------------------------------------------------------------

    def get_data(self, channels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Retourne le DataFrame d'événements, optionnellement restreint à certains canaux.

        Args:
            channels: Liste de noms de canaux à sélectionner ; None = tous.

        Returns:
            Sous-DataFrame (vue ou copie selon pandas).

        Raises:
            KeyError: Si un canal demandé est absent.
        """
        if channels is None:
            return self.events
        missing = set(channels) - set(self.events.columns)
        if missing:
            raise KeyError(f"Canaux absents de events: {sorted(missing)}")
        return self.events[channels]

    def get_mask(self, mask_name: str) -> np.ndarray:
        """
        Retourne le masque booléen nommé.

        Args:
            mask_name: Clé du masque.

        Returns:
            ndarray bool de longueur N_events.

        Raises:
            KeyError: Si le masque n'existe pas.
        """
        if mask_name not in self.masks:
            raise KeyError(f"Masque '{mask_name}' introuvable. Disponibles: {list(self.masks)}")
        return self.masks[mask_name]

    def set_mask(self, mask_name: str, mask: np.ndarray) -> None:
        """
        Enregistre un masque booléen de gating.

        Args:
            mask_name: Nom du masque.
            mask: Array bool de longueur N_events.

        Raises:
            SampleValidationError: Si longueur ou dtype incorrect.
        """
        self._check_length(mask, f"masque '{mask_name}'")
        if mask.dtype != bool:
            raise SampleValidationError(
                f"Masque '{mask_name}' doit être dtype=bool, reçu {mask.dtype}."
            )
        self.masks[mask_name] = mask

    def get_pre_gated_data(self, channels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Retourne les événements après application de tous les masques actifs (AND logique).

        Si aucun masque n'existe, retourne get_data(channels) sans filtrage.

        Args:
            channels: Canaux à sélectionner ; None = tous.

        Returns:
            DataFrame filtré, index remis à zéro.
        """
        data = self.get_data(channels)
        if not self.masks:
            return data.reset_index(drop=True)
        combined = np.ones(self.n_events, dtype=bool)
        for m in self.masks.values():
            combined &= m
        return data[combined].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Canaux virtuels
    # ------------------------------------------------------------------

    def add_virtual_channel(self, name: str, data: np.ndarray) -> None:
        """
        Ajoute un canal dérivé (ratio, unmixing, fluorescence corrigée…).

        Args:
            name: Nom du canal virtuel.
            data: Array float de longueur N_events.

        Raises:
            SampleValidationError: Si longueur incompatible.
        """
        self._check_length(data, f"canal virtuel '{name}'")
        self.virtual_channels[name] = data

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def add_embedding(
        self,
        name: str,
        embedding: np.ndarray,
        dimensions: Optional[List[str]] = None,
    ) -> None:
        """
        Enregistre un embedding dimensionnel (UMAP, t-SNE, PHATE…).

        Args:
            name: Identifiant de l'embedding (ex. 'umap_2d').
            embedding: Array (N_events × n_dims).
            dimensions: Noms des axes (ex. ['UMAP1', 'UMAP2']). Ignoré si None.

        Raises:
            SampleValidationError: Si N_events ne correspond pas.
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(-1, 1)
        if embedding.shape[0] != self.n_events:
            raise SampleValidationError(
                f"Embedding '{name}': {embedding.shape[0]} lignes ≠ {self.n_events} événements."
            )
        self.embeddings[name] = embedding

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def add_cluster_labels(self, name: str, labels: np.ndarray) -> None:
        """
        Enregistre un vecteur de labels de cluster.

        Args:
            name: Identifiant de la partition (ex. 'flowsom_meta20').
            labels: Array int de longueur N_events.

        Raises:
            SampleValidationError: Si longueur incompatible.
        """
        self._check_length(labels, f"labels '{name}'")
        self.cluster_assignments[name] = labels.astype(int)

    # ------------------------------------------------------------------
    # Sous-échantillonnage / subset
    # ------------------------------------------------------------------

    def subset(self, mask_name: str) -> "Sample":
        """
        Retourne un nouveau Sample filtré par le masque nommé.

        Tous les attributs indexés (virtual_channels, embeddings,
        cluster_assignments) sont réindexés sur les événements retenus.
        Les masks ne sont PAS propagés (le subset est déjà filtré).

        Args:
            mask_name: Nom d'un masque enregistré via set_mask().

        Returns:
            Nouveau Sample avec N' événements (N' = sum(mask)).
        """
        mask = self.get_mask(mask_name)
        sub = Sample(
            sample_id=self.sample_id,
            events=self.events[mask].reset_index(drop=True),
            channel_metadata=self.channel_metadata,
            compensation_matrix=self.compensation_matrix,
            per_sample_metadata=dict(self.per_sample_metadata),
        )
        for vname, vdata in self.virtual_channels.items():
            sub.virtual_channels[vname] = vdata[mask]
        for ename, emb in self.embeddings.items():
            sub.embeddings[ename] = emb[mask]
        for cname, lbls in self.cluster_assignments.items():
            sub.cluster_assignments[cname] = lbls[mask]
        sub.artifacts = list(self.artifacts)
        sub.results = dict(self.results)
        return sub

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Vérifie la cohérence dimensionnelle complète du Sample.

        Contrôles :
          - masks : bool, longueur N_events
          - virtual_channels : longueur N_events
          - embeddings : première dimension == N_events
          - cluster_assignments : longueur N_events
          - compensation_matrix : carré, taille == n_channels si présente

        Raises:
            SampleValidationError: Dès la première incohérence détectée.
        """
        N = self.n_events

        for mname, m in self.masks.items():
            if m.dtype != bool:
                raise SampleValidationError(f"Masque '{mname}': dtype doit être bool, reçu {m.dtype}.")
            if len(m) != N:
                raise SampleValidationError(f"Masque '{mname}': {len(m)} ≠ {N} événements.")

        for vname, vdata in self.virtual_channels.items():
            if len(vdata) != N:
                raise SampleValidationError(f"Canal virtuel '{vname}': {len(vdata)} ≠ {N}.")

        for ename, emb in self.embeddings.items():
            if emb.shape[0] != N:
                raise SampleValidationError(f"Embedding '{ename}': {emb.shape[0]} ≠ {N}.")

        for cname, lbls in self.cluster_assignments.items():
            if len(lbls) != N:
                raise SampleValidationError(f"Labels '{cname}': {len(lbls)} ≠ {N}.")

        if self.compensation_matrix is not None:
            mat = self.compensation_matrix
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                raise SampleValidationError(
                    f"compensation_matrix doit être carrée, reçu shape {mat.shape}."
                )

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    def _check_length(self, arr: np.ndarray, label: str) -> None:
        if len(arr) != self.n_events:
            raise SampleValidationError(
                f"{label}: longueur {len(arr)} incompatible avec {self.n_events} événements."
            )

    # ------------------------------------------------------------------
    # Représentation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        masks_str = ", ".join(self.masks) or "—"
        emb_str = ", ".join(self.embeddings) or "—"
        clus_str = ", ".join(self.cluster_assignments) or "—"
        return (
            f"Sample(id={self.sample_id[:8]}…, "
            f"events={self.n_events:,}, "
            f"channels={len(self.channels)}, "
            f"masks=[{masks_str}], "
            f"embeddings=[{emb_str}], "
            f"clusters=[{clus_str}])"
        )


# ===========================================================================
# FlowSample — modèle legacy (conservé pour compatibilité pipeline existant)
# ===========================================================================

from dataclasses import dataclass
from dataclasses import field as _field


@dataclass
class FlowSample:
    """
    Représente un échantillon FCS après chargement (modèle legacy).

    Attributes:
        name: Nom court du fichier (sans chemin).
        path: Chemin complet vers le fichier FCS d'origine.
        condition: Label de condition ('Sain', 'Pathologique', …).
        data: DataFrame pandas (cellules × marqueurs).
        metadata: Métadonnées FCS brutes ($PnN, $DATE, …).
        n_cells_raw: Nombre de cellules avant tout gating.
    """

    name: str
    path: str
    condition: str
    data: pd.DataFrame
    metadata: Dict[str, Any] = _field(default_factory=dict)
    n_cells_raw: int = 0
    raw_data: Optional[pd.DataFrame] = _field(default=None, repr=False)

    @classmethod
    def from_anndata(cls, adata: Any, path: str, condition: str) -> "FlowSample":
        """Construit un FlowSample depuis un objet AnnData (flowsom.io.read_FCS)."""
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        df = pd.DataFrame(X, columns=list(adata.var_names))
        meta = dict(adata.uns) if hasattr(adata, "uns") else {}
        return cls(
            name=Path(path).name,
            path=path,
            condition=condition,
            data=df,
            metadata=meta,
            n_cells_raw=len(df),
        )

    @property
    def n_cells(self) -> int:
        return len(self.data)

    @property
    def markers(self) -> List[str]:
        return list(self.data.columns)

    @property
    def var_names(self) -> List[str]:
        return list(self.data.columns)

    @property
    def matrix(self) -> np.ndarray:
        return self.data.values

    def get_marker_index(self, name: str) -> Optional[int]:
        upper = [m.upper() for m in self.markers]
        try:
            return upper.index(name.upper())
        except ValueError:
            return None

    def filter(self, mask: np.ndarray) -> "FlowSample":
        return FlowSample(
            name=self.name,
            path=self.path,
            condition=self.condition,
            data=self.data[mask].reset_index(drop=True),
            metadata=self.metadata,
            n_cells_raw=self.n_cells_raw,
        )

    def downsample(self, n: int, seed: int = 42) -> "FlowSample":
        if n >= self.n_cells:
            return self
        return FlowSample(
            name=self.name,
            path=self.path,
            condition=self.condition,
            data=self.data.sample(n=n, random_state=seed).reset_index(drop=True),
            metadata=self.metadata,
            n_cells_raw=self.n_cells_raw,
        )

    def summary(self) -> str:
        return (
            f"FlowSample({self.name!r}, condition={self.condition!r}, "
            f"cells={self.n_cells:,}/{self.n_cells_raw:,}, "
            f"markers={len(self.markers)})"
        )

    def __repr__(self) -> str:
        return self.summary()


# ===========================================================================
# FlowCytometrySample — utilitaire de chargement FCS (legacy)
# ===========================================================================

try:
    import flowsom as _fs
    _FS_AVAILABLE = True
except ImportError:
    _FS_AVAILABLE = False


class FlowCytometrySample:
    """Utilitaire de chargement d'un fichier FCS en DataFrame pandas (legacy)."""

    @classmethod
    def from_fcs(cls, fcs_file: str, verbose: bool = True) -> pd.DataFrame:
        adata = _fs.io.read_FCS(fcs_file)
        df = pd.DataFrame(adata.X, columns=list(adata.var_names))
        if verbose:
            print(f"    [OK] {Path(fcs_file).name}: {len(df):,} cellules")
        return df
