"""
prisma/exports/cluster_splitter.py — Export FCS par cluster de sous-populations.

Après une analyse UMAP + PhenoGraph/FlowSOM, permet d'exporter chaque cluster
en fichier FCS indépendant pour analyse downstream ou validation manuelle.

Usage typique :
    exporter = ClusterFCSExporter(
        source_df=annotated_df,
        cluster_column="PhenoGraph_Cluster",
        output_dir=Path("results/exports/clusters"),
        sample_name="Patient_001",
    )
    report = exporter.export_all()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from prisma.io.fcs_writer import export_to_fcs

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rapport d'export
# ---------------------------------------------------------------------------


@dataclass
class ClusterExportReport:
    """Résumé de l'export FCS par cluster."""

    sample_name: str
    cluster_column: str
    total_cells: int
    n_clusters: int
    exported: Dict[str, Path] = field(default_factory=dict)
    failed: Dict[str, str] = field(default_factory=dict)

    @property
    def success_count(self) -> int:
        return len(self.exported)

    @property
    def failure_count(self) -> int:
        return len(self.failed)

    def summary(self) -> str:
        lines = [
            f"Export FCS par cluster — {self.sample_name}",
            f"  Colonne cluster : {self.cluster_column}",
            f"  Cellules totales: {self.total_cells:,}",
            f"  Clusters        : {self.n_clusters}",
            f"  Exportés        : {self.success_count} ✓  |  Échecs: {self.failure_count} ✗",
        ]
        for label, path in sorted(self.exported.items()):
            lines.append(f"    ✓ Cluster {label:<6} → {path.name}")
        for label, err in sorted(self.failed.items()):
            lines.append(f"    ✗ Cluster {label:<6} : {err}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exporter principal
# ---------------------------------------------------------------------------


class ClusterFCSExporter:
    """
    Divise un DataFrame annoté en un fichier FCS par cluster unique.

    Chaque fichier produit est nommé :
        {sample_name}_Cluster_{label}.fcs

    Seules les colonnes numériques sont écrites dans les fichiers FCS.
    La colonne cluster est incluse dans l'export pour traçabilité.

    Args:
        source_df:        DataFrame annoté (n_cells × n_features + colonne cluster).
        cluster_column:   Nom de la colonne contenant les labels de cluster
                          (ex: "PhenoGraph_Cluster", "FlowSOM_metacluster").
        output_dir:       Dossier de destination pour les fichiers FCS générés.
        sample_name:      Préfixe utilisé pour nommer les fichiers de sortie.
                          Si None, utilise "Sample".
        skip_noise:       Si True, ignore le cluster -1 (bruit HDBSCAN/PARC).
        min_cells:        Nombre minimal de cellules pour exporter un cluster.
                          Les clusters sous ce seuil sont ignorés avec un warning.
        compat_chn_names: Nettoyer les noms de canaux pour compatibilité Kaluza/FlowJo.
    """

    def __init__(
        self,
        source_df: pd.DataFrame,
        cluster_column: str,
        output_dir: Union[Path, str],
        sample_name: Optional[str] = None,
        *,
        skip_noise: bool = True,
        min_cells: int = 10,
        compat_chn_names: bool = True,
    ) -> None:
        if cluster_column not in source_df.columns:
            raise ValueError(
                f"Colonne cluster '{cluster_column}' introuvable dans le DataFrame. "
                f"Colonnes disponibles : {list(source_df.columns)}"
            )

        self._df = source_df
        self._cluster_col = cluster_column
        self._output_dir = Path(output_dir)
        self._sample_name = sample_name or "Sample"
        self._skip_noise = skip_noise
        self._min_cells = min_cells
        self._compat_chn_names = compat_chn_names

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def export_all(self) -> ClusterExportReport:
        """
        Exporte chaque cluster unique en fichier FCS distinct.

        Returns:
            ClusterExportReport avec le détail des fichiers produits et des erreurs.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        cluster_labels = self._df[self._cluster_col]
        unique_labels = sorted(cluster_labels.unique())

        report = ClusterExportReport(
            sample_name=self._sample_name,
            cluster_column=self._cluster_col,
            total_cells=len(self._df),
            n_clusters=len(unique_labels),
        )

        _logger.info(
            "ClusterFCSExporter: %d clusters détectés dans '%s' (%d cellules totales)",
            len(unique_labels),
            self._cluster_col,
            len(self._df),
        )

        for label in unique_labels:
            label_str = str(label)

            # Ignorer le cluster bruit -1 (HDBSCAN/PARC)
            if self._skip_noise and label in (-1, "-1", -1.0):
                _logger.debug("Cluster bruit ignoré: %s", label_str)
                continue

            cluster_df = self._df[cluster_labels == label].copy()
            n_cells = len(cluster_df)

            if n_cells < self._min_cells:
                msg = f"Trop peu de cellules ({n_cells} < {self._min_cells}) — ignoré"
                _logger.warning("Cluster %s: %s", label_str, msg)
                report.failed[label_str] = msg
                continue

            out_path = self._output_dir / self._build_filename(label)

            success = export_to_fcs(
                cluster_df,
                out_path,
                compat_chn_names=self._compat_chn_names,
            )

            if success:
                report.exported[label_str] = out_path
                _logger.debug(
                    "Cluster %s exporté: %d cellules → %s",
                    label_str,
                    n_cells,
                    out_path.name,
                )
            else:
                msg = f"export_to_fcs a retourné False pour {out_path.name}"
                _logger.error("Cluster %s: %s", label_str, msg)
                report.failed[label_str] = msg

        _logger.info(report.summary())
        return report

    def export_cluster(self, label: Union[int, float, str]) -> Optional[Path]:
        """
        Exporte un seul cluster en fichier FCS.

        Args:
            label: Valeur du cluster à exporter.

        Returns:
            Chemin du fichier produit, ou None si échec.
        """
        cluster_df = self._df[self._df[self._cluster_col] == label].copy()

        if cluster_df.empty:
            _logger.warning("Cluster '%s' introuvable ou vide", label)
            return None

        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / self._build_filename(label)

        success = export_to_fcs(cluster_df, out_path, compat_chn_names=self._compat_chn_names)
        return out_path if success else None

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    def _build_filename(self, label: Union[int, float, str]) -> str:
        """Construit le nom de fichier sanitisé pour un cluster donné."""
        safe_sample = _sanitize_filename(self._sample_name)
        # Convertir float → int si entier (ex: 1.0 → 1)
        if isinstance(label, float) and label.is_integer():
            label = int(label)
        safe_label = _sanitize_filename(str(label))
        return f"{safe_sample}_Cluster_{safe_label}.fcs"


# ---------------------------------------------------------------------------
# Utilitaire
# ---------------------------------------------------------------------------


def _sanitize_filename(name: str) -> str:
    """Remplace les caractères interdits dans un nom de fichier."""
    forbidden = r'\/:*?"<>|'
    for ch in forbidden:
        name = name.replace(ch, "_")
    return name.strip()


# ---------------------------------------------------------------------------
# Fonction helper de haut niveau
# ---------------------------------------------------------------------------


def export_clusters_to_fcs(
    df: pd.DataFrame,
    cluster_column: str,
    output_dir: Union[Path, str],
    sample_name: Optional[str] = None,
    *,
    skip_noise: bool = True,
    min_cells: int = 10,
) -> ClusterExportReport:
    """
    Fonction convenience pour exporter tous les clusters d'un DataFrame en FCS.

    Args:
        df:             DataFrame annoté avec colonne de cluster.
        cluster_column: Nom de la colonne contenant les labels de cluster.
        output_dir:     Dossier de destination.
        sample_name:    Préfixe des fichiers de sortie.
        skip_noise:     Si True, ignore le cluster -1.
        min_cells:      Seuil minimal de cellules par cluster.

    Returns:
        ClusterExportReport avec le détail de l'export.
    """
    exporter = ClusterFCSExporter(
        source_df=df,
        cluster_column=cluster_column,
        output_dir=output_dir,
        sample_name=sample_name,
        skip_noise=skip_noise,
        min_cells=min_cells,
    )
    return exporter.export_all()
