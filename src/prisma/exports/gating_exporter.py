"""
src/exports/gating_exporter.py — Export analytique du module PrismaGatingViewer.

Deux fonctions principales (méthodes de classe) :
  GatingExporter.export_statistics(engine, output_csv_path)
      → CSV : sample_id / gate_name / gate_path / parent / count / %parent / %gp / MFI / CV

  GatingExporter.export_gated_fcs(engine, sample_id, gate_name, gate_path, output_fcs_path)
      → FCS filtré : événements appartenant à la gate

Stratégie export FCS :
  1. flowkit.Sample.export() si sample disponible dans la session
  2. Fallback : src.prisma.io.fcs_writer.export_to_fcs() sur le DataFrame filtré
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.prisma.utils.logger import get_logger

_logger = get_logger("exports.gating_exporter")


class GatingExporterError(Exception):
    """Erreur métier export de gating."""


class GatingExporter:
    """
    Exporteur analytique pour le module PrismaGatingViewer.

    Toutes les méthodes sont des classmethods ou staticmethods :
    aucune instance n'est requise.
    """

    # ------------------------------------------------------------------
    # CORRECTIF C4 : Normalisation robuste du rapport FlowKit
    # ------------------------------------------------------------------

    # Colonnes canoniques internes → candidats FlowKit par ordre de priorité
    _REPORT_COL_MAP: Dict[str, List[str]] = {
        "gate_name":       ["gate_name", "gate_id", "name"],
        "count":           ["count", "event_count", "n_events", "events"],
        "pct_parent":      ["relative_percent", "percent_parent", "pct_parent", "percent"],
        "pct_total":       ["absolute_percent", "percent_total", "pct_total"],
        "sample_id":       ["sample_id", "sample", "file"],
        "gate_path":       ["gate_path", "path"],
        "parent":          ["parent", "quadrant_parent", "parent_gate"],
    }

    @classmethod
    def standardize_report(cls, report_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise le DataFrame brut de FlowKit vers des colonnes canoniques.

        Robuste aux renommages de colonnes entre versions de FlowKit :
          count / event_count / n_events
          relative_percent / percent / pct_parent
          gate_name / gate_id / name
          etc.

        Args:
            report_df: DataFrame brut de Session.get_analysis_report().

        Returns:
            DataFrame avec colonnes canoniques garanties. Les colonnes non
            trouvées sont ajoutées avec des valeurs None.
        """
        if report_df is None or report_df.empty:
            return pd.DataFrame(columns=list(cls._REPORT_COL_MAP.keys()))

        existing = set(report_df.columns)
        mapping: Dict[str, str] = {}  # canonical → actual column name

        for canonical, candidates in cls._REPORT_COL_MAP.items():
            for candidate in candidates:
                if candidate in existing:
                    mapping[canonical] = candidate
                    break

        result = report_df.copy()

        # Renommer les colonnes trouvées vers leurs noms canoniques
        rename_map = {v: k for k, v in mapping.items() if v != k}
        if rename_map:
            result = result.rename(columns=rename_map)

        # Ajouter les colonnes canoniques manquantes avec None
        for canonical in cls._REPORT_COL_MAP:
            if canonical not in result.columns:
                result[canonical] = None
                _logger.debug(
                    "Colonne rapport '%s' absente dans FlowKit — remplie avec None", canonical
                )

        return result

    # ------------------------------------------------------------------
    # Export statistiques
    # ------------------------------------------------------------------

    @classmethod
    def export_statistics(
        cls,
        engine: "PrismaFlowEngine",  # type: ignore[name-defined]
        output_csv_path: Union[str, Path],
    ) -> pd.DataFrame:
        """
        Exporte les statistiques de gating pour tous les samples et toutes les gates.

        Colonnes produites :
          sample_id, gate_name, gate_path, parent_gate,
          count, pct_parent, pct_grandparent,
          {channel}_mfi, {channel}_cv  pour chaque canal numérique.

        Args:
            engine:           Instance de PrismaFlowEngine après analyze().
            output_csv_path:  Chemin du fichier CSV de sortie.

        Returns:
            DataFrame des statistiques (copie).

        Raises:
            GatingExporterError: si aucun sample ou gate n'est disponible.
        """
        output_csv_path = Path(output_csv_path)
        sample_ids = engine.get_sample_ids()
        gate_pairs = engine.get_gate_id_path_pairs()  # List[Tuple[name, path]]

        if not sample_ids:
            raise GatingExporterError("Aucun sample dans la session.")
        if not gate_pairs:
            raise GatingExporterError("Aucune gate dans la stratégie.")

        rows: List[Dict[str, Any]] = []

        for sample_id in sample_ids:
            try:
                engine.set_active_sample(sample_id)
            except Exception:
                _logger.warning("Sample inaccessible : %s", sample_id)
                continue

            # Nombre total d'événements du sample (population racine)
            try:
                df_all = engine.get_raw_dataframe(sample_id=sample_id)
                n_total = len(df_all)
            except Exception as exc:
                _logger.warning("Données brutes inaccessibles pour %s : %s", sample_id, exc)
                continue

            for gate_id, gate_path in gate_pairs:
                parent_name = gate_path[-1] if gate_path else ""
                grandparent_name = gate_path[-2] if len(gate_path) >= 2 else ""

                try:
                    df_gate = engine.get_gate_dataframe(
                        gate_id,
                        gate_path=gate_path,
                        sample_id=sample_id,
                    )
                except Exception as exc:
                    _logger.debug("Gate %s ignorée pour %s : %s", gate_id, sample_id, exc)
                    continue

                count = len(df_gate)

                # % parent
                try:
                    if parent_name and parent_name != "root":
                        parent_paths = engine.find_gate_paths(parent_name)
                        parent_path = parent_paths[0] if parent_paths else None
                        df_parent = engine.get_gate_dataframe(
                            parent_name, gate_path=parent_path, sample_id=sample_id
                        )
                        n_parent = len(df_parent)
                    else:
                        n_parent = n_total
                except Exception:
                    n_parent = n_total

                pct_parent = 100.0 * count / n_parent if n_parent > 0 else 0.0

                # % grand-parent
                try:
                    if grandparent_name and grandparent_name != "root":
                        gp_paths = engine.find_gate_paths(grandparent_name)
                        gp_path = gp_paths[0] if gp_paths else None
                        df_gp = engine.get_gate_dataframe(
                            grandparent_name, gate_path=gp_path, sample_id=sample_id
                        )
                        n_gp = len(df_gp)
                    else:
                        n_gp = n_total
                except Exception:
                    n_gp = n_total

                pct_gp = 100.0 * count / n_gp if n_gp > 0 else 0.0

                row: Dict[str, Any] = {
                    "sample_id": sample_id,
                    "gate_name": gate_id,
                    "gate_path": " > ".join(gate_path),
                    "parent_gate": parent_name,
                    "count": count,
                    "pct_parent": round(pct_parent, 4),
                    "pct_grandparent": round(pct_gp, 4),
                }

                # MFI et CV par canal
                if count > 0:
                    numeric_cols = df_gate.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        vals = df_gate[col].dropna()
                        if len(vals) > 0:
                            mfi = float(vals.median())
                            cv = (float(vals.std()) / float(vals.mean()) * 100.0
                                  if float(vals.mean()) != 0 else 0.0)
                            row[f"{col}_mfi"] = round(mfi, 4)
                            row[f"{col}_cv"] = round(cv, 4)

                rows.append(row)

        if not rows:
            raise GatingExporterError(
                "Aucune statistique produite. Avez-vous lancé analyze() ?"
            )

        df_stats = pd.DataFrame(rows)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_stats.to_csv(output_csv_path, index=False, encoding="utf-8")
        _logger.info(
            "Statistiques exportées : %d lignes → %s",
            len(df_stats),
            output_csv_path,
        )
        return df_stats

    # ------------------------------------------------------------------
    # Export FCS filtré par gate
    # ------------------------------------------------------------------

    @classmethod
    def export_gated_fcs(
        cls,
        engine: "PrismaFlowEngine",  # type: ignore[name-defined]
        sample_id: Optional[str],
        gate_name: str,
        gate_path: Optional[Tuple[str, ...]],
        output_fcs_path: Union[str, Path],
        transform_id: Optional[str] = None,
        comp_matrix_id: Optional[str] = None,
    ) -> Path:
        """
        Exporte les événements d'une gate en fichier FCS.

        Stratégie d'écriture (ordre de priorité) :
          1. flowkit.Sample.export() via masque membership — préserve les métadonnées FCS.
          2. Fallback : src.prisma.io.fcs_writer.export_to_fcs() sur le DataFrame filtré.

        Args:
            engine:          Instance de PrismaFlowEngine.
            sample_id:       ID du sample (None → sample actif).
            gate_name:       Nom de la gate.
            gate_path:       Chemin complet des ancêtres.
            output_fcs_path: Chemin du fichier FCS de sortie.
            transform_id:    Transformation à appliquer sur les données exportées.
            comp_matrix_id:  Compensation à appliquer.

        Returns:
            Path du fichier produit.

        Raises:
            GatingExporterError: si la gate est vide ou l'écriture échoue.
        """
        output_fcs_path = Path(output_fcs_path)
        output_fcs_path.parent.mkdir(parents=True, exist_ok=True)

        sid = sample_id or engine.active_sample_id
        if not sid:
            raise GatingExporterError("Aucun sample actif dans la session.")

        _logger.info(
            "Export FCS gate '%s' @ %s — sample=%s → %s",
            gate_name, gate_path, sid, output_fcs_path.name,
        )

        # --- Tentative 1 : flowkit.Sample.export() avec masque -----------
        exported = cls._try_export_via_flowkit(
            engine, sid, gate_name, gate_path, output_fcs_path
        )
        if exported:
            _logger.info("Export FCS via FlowKit réussi : %s", output_fcs_path)
            return output_fcs_path

        # --- Tentative 2 : fallback DataFrame → fcs_writer ---------------
        cls._export_via_dataframe(
            engine, sid, gate_name, gate_path, output_fcs_path,
            transform_id=transform_id, comp_matrix_id=comp_matrix_id,
        )
        _logger.info("Export FCS via fallback DataFrame réussi : %s", output_fcs_path)
        return output_fcs_path

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    @classmethod
    def _try_export_via_flowkit(
        cls,
        engine: Any,
        sample_id: str,
        gate_name: str,
        gate_path: Optional[Tuple[str, ...]],
        output_fcs_path: Path,
    ) -> bool:
        """
        Tente d'utiliser flowkit.Sample.export() avec set_flagged_events().

        flowkit.Sample.export(filename, source='xform', exclude_flagged=False, ...)
        Stratégie : flaguer les événements HORS de la gate, exporter les non-flaggés.

        Returns:
            True si réussi, False sinon.
        """
        try:
            session = engine.session
            sample = session.get_sample(sample_id)

            # Masque membership
            membership = engine.get_gate_membership(
                gate_name, gate_path=gate_path, sample_id=sample_id
            )
            if not membership.any():
                raise GatingExporterError(
                    f"Aucun événement dans la gate '{gate_name}' pour le sample '{sample_id}'."
                )

            # Flaguer les événements hors gate (à exclure de l'export)
            outside_mask = ~membership
            outside_indices = np.where(outside_mask)[0]
            sample.set_flagged_events(outside_indices)

            # Exporter uniquement les non-flaggés
            sample.export(
                str(output_fcs_path.name),
                source="xform",
                exclude_flagged=True,
                directory=str(output_fcs_path.parent),
            )

            # Réinitialiser les flags pour ne pas polluer les analyses suivantes
            sample.set_flagged_events(np.array([], dtype=int))
            return True

        except GatingExporterError:
            raise
        except Exception as exc:
            _logger.debug("Export FlowKit échoué, fallback DataFrame : %s", exc)
            return False

    @classmethod
    def _export_via_dataframe(
        cls,
        engine: Any,
        sample_id: str,
        gate_name: str,
        gate_path: Optional[Tuple[str, ...]],
        output_fcs_path: Path,
        transform_id: Optional[str],
        comp_matrix_id: Optional[str],
    ) -> None:
        """
        Exporte via src.prisma.io.fcs_writer.export_to_fcs() sur DataFrame filtré.

        Raises:
            GatingExporterError: si le DataFrame est vide ou l'écriture échoue.
        """
        try:
            df = engine.get_gate_dataframe(
                gate_name,
                gate_path=gate_path,
                sample_id=sample_id,
                transform_id=transform_id,
                matrix_id=comp_matrix_id,
            )
        except Exception as exc:
            raise GatingExporterError(
                f"Impossible d'extraire les événements de la gate '{gate_name}' : {exc}"
            ) from exc

        if df.empty:
            raise GatingExporterError(
                f"DataFrame vide pour la gate '{gate_name}' — export annulé."
            )

        try:
            from src.prisma.io.fcs_writer import export_to_fcs
            # FlowKit retourne les colonnes sous forme de tuples (pnn, pns)
            # fcs_writer attend des chaînes — normaliser
            if df.columns.dtype == object and isinstance(df.columns[0], tuple):
                df = df.copy()
                df.columns = [col[0] if isinstance(col, tuple) else str(col) for col in df.columns]
            success = export_to_fcs(df, output_fcs_path, compat_chn_names=True)
            if not success:
                raise GatingExporterError(
                    f"export_to_fcs() a retourné False pour {output_fcs_path.name}."
                )
        except ImportError as exc:
            raise GatingExporterError(
                "fcs_writer indisponible. Installez fcswrite : pip install fcswrite"
            ) from exc
