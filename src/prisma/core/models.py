"""
models.py — Modèles de données fondateurs de PRISMA Research.

Toutes les structures de données traversant le pipeline sont définies ici.
Sérialisables JSON. Aucune dépendance interne à PRISMA.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """
    Représente un fichier FCS chargé et pré-traité.

    Attributes:
        name: Nom court (sans chemin).
        path: Chemin absolu du FCS d'origine.
        condition: Label de condition ('healthy', 'patient', …).
        data: DataFrame (cellules × marqueurs), peut être remplacé après gating.
        metadata: Métadonnées FCS brutes ($PnN, $DATE, …).
        n_cells_raw: Nombre de cellules avant tout filtrage.
        raw_data: Valeurs pré-transformation (optionnel, pour export FCS Kaluza).
        sample_id: Identifiant unique auto-généré.
    """

    name: str
    path: str
    condition: str
    data: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    n_cells_raw: int = 0
    raw_data: Optional[pd.DataFrame] = field(default=None, repr=False)
    sample_id: str = field(default_factory=lambda: str(uuid4()))

    @property
    def n_cells(self) -> int:
        return len(self.data)

    @property
    def markers(self) -> List[str]:
        return list(self.data.columns)

    @property
    def matrix(self) -> np.ndarray:
        return self.data.values

    def filter(self, mask: np.ndarray) -> "Sample":
        """Retourne nouveau Sample filtré par masque booléen."""
        return Sample(
            name=self.name,
            path=self.path,
            condition=self.condition,
            data=self.data[mask].reset_index(drop=True),
            metadata=self.metadata,
            n_cells_raw=self.n_cells_raw,
            sample_id=self.sample_id,
        )

    def downsample(self, n: int, seed: int = 42) -> "Sample":
        """Sous-échantillonne à n cellules."""
        if n >= self.n_cells:
            return self
        return Sample(
            name=self.name,
            path=self.path,
            condition=self.condition,
            data=self.data.sample(n=n, random_state=seed).reset_index(drop=True),
            metadata=self.metadata,
            n_cells_raw=self.n_cells_raw,
            sample_id=self.sample_id,
        )

    def summary(self) -> str:
        return (
            f"Sample({self.name!r}, condition={self.condition!r}, "
            f"cells={self.n_cells:,}/{self.n_cells_raw:,}, "
            f"markers={len(self.markers)})"
        )

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    """
    Conteneur d'un groupe de Samples liés à une analyse cohérente.

    Attributes:
        name: Nom de l'expérience.
        samples: Liste de Samples chargés.
        description: Description libre.
        experiment_id: UUID auto-généré.
        created_at: Timestamp ISO de création.
        tags: Labels libres pour filtrage.
    """

    name: str
    samples: List[Sample] = field(default_factory=list)
    description: str = ""
    experiment_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    tags: List[str] = field(default_factory=list)

    def add_sample(self, sample: Sample) -> None:
        self.samples.append(sample)

    def get_by_condition(self, condition: str) -> List[Sample]:
        return [s for s in self.samples if s.condition == condition]

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def conditions(self) -> List[str]:
        return sorted(set(s.condition for s in self.samples))

    def __repr__(self) -> str:
        return (
            f"Experiment({self.name!r}, samples={self.n_samples}, "
            f"conditions={self.conditions})"
        )


# ---------------------------------------------------------------------------
# GatingResult
# ---------------------------------------------------------------------------

@dataclass
class GatingResult:
    """
    Résultat d'une étape de gating sur un Sample.

    Attributes:
        sample_id: Référence au Sample source.
        gate_name: Nom du gate appliqué.
        mask: Masque booléen de sélection (n_cells_raw,).
        n_selected: Nombre de cellules retenues.
        frequency: Fréquence sélectionnée (0–1).
        method: Méthode utilisée ('manual', 'auto', 'pregating').
        parameters: Paramètres du gate (seuils, polygones, …).
    """

    sample_id: str
    gate_name: str
    mask: np.ndarray
    n_selected: int
    frequency: float
    method: str = "manual"
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "gate_name": self.gate_name,
            "n_selected": self.n_selected,
            "frequency": self.frequency,
            "method": self.method,
            "parameters": self.parameters,
        }


# ---------------------------------------------------------------------------
# RunMetadata
# ---------------------------------------------------------------------------

@dataclass
class RunMetadata:
    """
    Métadonnées complètes d'un run de pipeline — garantit la reproductibilité.

    Attributes:
        run_id: UUID du run.
        pipeline_name: Nom du pipeline exécuté.
        started_at: Timestamp ISO de démarrage.
        finished_at: Timestamp ISO de fin (None si en cours).
        seed: Graine globale utilisée.
        parameters: Paramètres du run sérialisés.
        lib_versions: Versions des bibliothèques clés.
        hostname: Machine d'exécution.
        status: 'running' | 'success' | 'failed'.
        error: Message d'erreur si status='failed'.
        artifacts: Chemins des artefacts produits.
    """

    run_id: str = field(default_factory=lambda: str(uuid4()))
    pipeline_name: str = "unknown"
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: Optional[str] = None
    seed: int = 42
    parameters: Dict[str, Any] = field(default_factory=dict)
    lib_versions: Dict[str, str] = field(default_factory=dict)
    hostname: str = field(default_factory=platform.node)
    status: str = "running"
    error: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)

    @classmethod
    def start(cls, pipeline_name: str, seed: int, parameters: Dict[str, Any]) -> "RunMetadata":
        """Crée un RunMetadata au lancement d'un run."""
        versions = _collect_lib_versions()
        return cls(
            pipeline_name=pipeline_name,
            seed=seed,
            parameters=parameters,
            lib_versions=versions,
        )

    def finish(self, status: str = "success", error: Optional[str] = None) -> None:
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.status = status
        self.error = error

    def save(self, output_dir: Path) -> Path:
        """Sérialise en JSON dans output_dir/metadata.json."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "metadata.json"
        data = {k: v for k, v in asdict(self).items()}
        # numpy/np types ne sont pas JSON-sérialisables nativement
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "RunMetadata":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_lib_versions() -> Dict[str, str]:
    """Collecte les versions des bibliothèques clés pour la reproductibilité."""
    versions: Dict[str, str] = {}
    libs = [
        "numpy", "pandas", "scipy", "sklearn", "umap", "openTSNE",
        "harmonypy", "phate", "anndata", "PySide6",
    ]
    for lib in libs:
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not_installed"
    return versions
