"""
pipeline_result.py — Résultat complet d'une exécution du pipeline.

PipelineResult agrège toutes les sorties du pipeline (données, clusters,
métriques, exports) en un seul objet inspectable et sérialisable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ClusteringMetrics:
    """Métriques de qualité du clustering FlowSOM."""

    n_nodes: int = 0
    n_metaclusters: int = 0
    silhouette_score: Optional[float] = None
    stability_score: Optional[float] = None
    optimal_k_found: bool = False
    n_cells_per_cluster: Dict[int, int] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """
    Résultat complet d'une exécution du pipeline FlowSOM.

    Attributes:
        data: DataFrame final avec assignations FlowSOM.
        metacluster_stats: Statistiques par métacluster (MFI, n_cells, …).
        mfi_matrix: Matrice MFI (marqueurs × metaclusters).
        gating_report: Rapport JSON des étapes de gating.
        clustering_metrics: Métriques de qualité du clustering.
        output_files: Chemins des fichiers produits.
        config_snapshot: Snapshot de la configuration utilisée.
        timestamp: Horodatage de l'exécution.
        elapsed_seconds: Durée totale d'exécution.
        warnings: Avertissements émis pendant l'exécution.
    """

    data: Optional[pd.DataFrame] = None
    metacluster_stats: Optional[pd.DataFrame] = None
    mfi_matrix: Optional[pd.DataFrame] = None
    gating_report: List[Dict[str, Any]] = field(default_factory=list)
    clustering_metrics: ClusteringMetrics = field(default_factory=ClusteringMetrics)
    output_files: Dict[str, str] = field(default_factory=dict)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    elapsed_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    population_mapping: Optional[Any] = None  # PopulationMappingResult (§10)
    mrd_result: Optional[Any] = None          # MRDResult (§8) — pour synthèse batch
    patho_stem: Optional[str] = None          # Stem du fichier patho traité (batch)
    patho_date: Optional[str] = None         # Date du prélèvement extraite du FCS (format YYYY-MM-DD)
    node_mfi_matrix: Optional[pd.DataFrame] = None  # MFI médiane par nœud SOM (index=node_id int, cols=markers)
    raw_data: Optional[pd.DataFrame] = None          # Intensités brutes pré-transformation, alignées sur data (même index)

    # ── Curation humaine (optionnelle) ─────────────────────────────────
    # Renseigné par HomeTab._inject_human_curation() avant tout export.
    # Si None : l'export utilise les valeurs algorithmiques brutes.
    curated_mrd_percent: Optional[float] = None   # MRD re-calculée après curation experte (%)
    curated_mrd_cells:   Optional[int]   = None   # Nombre de cellules MRD après curation
    curated_nodes:       Optional[List[Dict[str, Any]]] = None  # Nœuds validés par l'expert

    # ── Citrus (analyse stratifiante cohorte) ───────────────────────────
    citrus_result: Optional[Any] = None  # CitrusResult — None si non activé

    # ── Pré-screening CD34+/CD45dim ─────────────────────────────────────
    # Résultat du pré-screening heuristique (toujours calculé, indépendamment des options).
    prescreening_result: Optional[Any] = None  # PrescreeningResult

    # ------------------------------------------------------------------
    # Accès
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """Nombre total de cellules dans le résultat final."""
        return len(self.data) if self.data is not None else 0

    @property
    def n_metaclusters(self) -> int:
        """Nombre de métaclusters produits."""
        if self.data is not None and "FlowSOM_metacluster" in self.data.columns:
            return int(self.data["FlowSOM_metacluster"].nunique())
        return self.clustering_metrics.n_metaclusters

    @property
    def success(self) -> bool:
        """True si le pipeline s'est terminé avec des données valides."""
        return self.data is not None and self.n_cells > 0

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """
        Résumé textuel du résultat pour affichage console.

        Returns:
            Chaîne multi-lignes.
        """
        lines = [
            "=" * 60,
            "RÉSUMÉ PIPELINE FlowSOM",
            "=" * 60,
            f"  Statut:          {'OK' if self.success else 'ÉCHEC'}",
            f"  Timestamp:       {self.timestamp}",
            f"  Durée:           {self.elapsed_seconds:.1f}s",
            f"  Cellules:        {self.n_cells:,}",
            f"  Métaclusters:    {self.n_metaclusters}",
        ]

        if self.clustering_metrics.silhouette_score is not None:
            lines.append(
                f"  Silhouette:      {self.clustering_metrics.silhouette_score:.3f}"
            )
        if self.clustering_metrics.stability_score is not None:
            lines.append(
                f"  Stabilité:       {self.clustering_metrics.stability_score:.3f}"
            )

        if self.gating_report:
            lines.append(f"\n  Gates appliqués: {len(self.gating_report)}")
            for gate in self.gating_report:
                pct = gate.get("pct_kept", 0)
                lines.append(
                    f"    [{gate.get('gate_name', '?')}] "
                    f"{gate.get('n_after', gate.get('n_kept', 0)):,}/{gate.get('n_before', gate.get('n_total', 0)):,} "
                    f"({pct:.1f}%) — {gate.get('method', gate.get('mode', 'auto'))}"
                )

        if self.output_files:
            lines.append(f"\n  Fichiers exportés:")
            for key, path in self.output_files.items():
                lines.append(f"    {key}: {path}")

        if self.warnings:
            lines.append(f"\n  Avertissements ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    [!] {w}")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    def export_csv(self, output_path: str | Path) -> str:
        """
        Exporte les données principales en CSV.

        Args:
            output_path: Chemin de destination.

        Returns:
            Chemin absolu du fichier créé.

        Raises:
            ValueError: Si aucune donnée disponible.
        """
        if self.data is None:
            raise ValueError("Aucune donnée à exporter (data=None).")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(path, index=False)
        self.output_files["csv"] = str(path)
        return str(path)

    def export_metadata(self, output_path: str | Path) -> str:
        """
        Exporte les métadonnées du run en JSON.

        Args:
            output_path: Chemin de destination.

        Returns:
            Chemin absolu du fichier créé.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "timestamp": self.timestamp,
            "elapsed_seconds": self.elapsed_seconds,
            "n_cells": self.n_cells,
            "n_metaclusters": self.n_metaclusters,
            "success": self.success,
            "clustering_metrics": {
                "n_nodes": self.clustering_metrics.n_nodes,
                "n_metaclusters": self.clustering_metrics.n_metaclusters,
                "silhouette_score": self.clustering_metrics.silhouette_score,
                "stability_score": self.clustering_metrics.stability_score,
            },
            "gating_report": self.gating_report,
            "output_files": self.output_files,
            "config": self.config_snapshot,
            "warnings": self.warnings,
        }
        # Curation humaine — ajoutée uniquement si renseignée
        if self.curated_mrd_percent is not None:
            meta["human_curation"] = {
                "curated_mrd_percent": self.curated_mrd_percent,
                "curated_mrd_cells":   self.curated_mrd_cells,
                "n_curated_nodes":     len(self.curated_nodes) if self.curated_nodes else 0,
                "curated_node_ids": [
                    n.get("node_id") for n in (self.curated_nodes or [])
                ],
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        self.output_files["metadata_json"] = str(path)
        return str(path)

    # ------------------------------------------------------------------
    # Constructeur alternatif — échec
    # ------------------------------------------------------------------

    @classmethod
    def failure(cls, error: str, config: Any = None) -> "PipelineResult":
        """
        Construit un PipelineResult représentant un échec.

        Args:
            error: Message d'erreur.
            config: PipelineConfig ayant conduit à l'échec (optionnel).

        Returns:
            PipelineResult avec data=None et warnings contenant l'erreur.
        """
        config_snapshot: Dict[str, Any] = {}
        if config is not None:
            try:
                config_snapshot = config.to_dict()
            except Exception:
                config_snapshot = {}
        return cls(
            data=None,
            config_snapshot=config_snapshot,
            warnings=[f"ÉCHEC: {error}"],
        )

    def __repr__(self) -> str:
        return (
            f"PipelineResult(success={self.success}, cells={self.n_cells:,}, "
            f"metaclusters={self.n_metaclusters}, elapsed={self.elapsed_seconds:.1f}s)"
        )
