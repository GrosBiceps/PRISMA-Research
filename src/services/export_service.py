"""
export_service.py — Orchestration complète des exports de résultats.

Gère tous les exports en cascade:
  1. FCS (Kaluza/FlowJo compatible)
  2. CSV complet + par fichier + statistiques + MFI
  3. Métadonnées JSON (traçabilité complète)
  4. Log de gating JSON
  5. Graphiques (gating + FlowSOM)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.pipeline_config import PipelineConfig
from src.io.fcs_writer import export_to_fcs_kaluza
from src.io.csv_exporter import (
    export_cells_csv,
    export_statistics_csv,
    export_mfi_matrix_csv,
    export_per_file_csv,
    compute_cluster_statistics,
)
from src.io.cluster_distribution_exporter import (
    export_cluster_distribution,
)
from src.io.json_exporter import (
    export_analysis_metadata,
    build_analysis_metadata,
    export_gating_log,
)
from src.utils.logger import GatingLogger, get_logger

_logger = get_logger("services.export")


class ExportService:
    """
    Orchestrateur des exports de résultats du pipeline FlowSOM.

    Usage:
        exporter = ExportService(config, output_base_dir)
        exporter.export_all(df_cells, mfi_matrix, metaclustering, ...)
    """

    def __init__(
        self,
        config: PipelineConfig,
        output_dir: Path | str,
        timestamp: Optional[str] = None,
        name_stem: Optional[str] = None,
        patho_name: Optional[str] = None,
        patho_date: Optional[str] = None,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        # Stem utilisé pour nommer les rapports HTML/PDF (Feature 1 : nommage dynamique)
        self.name_stem = name_stem or f"flowsom_report_{self.timestamp}"
        # Informations sur la moelle pathologique (pour l'encadré des rapports)
        self.patho_name = patho_name
        self.patho_date = patho_date

        # Créer la structure de dossiers de sortie
        self._dirs = {
            "fcs": self.output_dir / "fcs",
            "csv": self.output_dir / "csv",
            "plots": self.output_dir / "plots",
            "other": self.output_dir / "other",
        }
        for d in self._dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def export_all(
        self,
        df_cells: pd.DataFrame,
        mfi_matrix: pd.DataFrame,
        metaclustering: np.ndarray,
        selected_markers: List[str],
        input_files: List[str],
        gating_logger: Optional[GatingLogger] = None,
        df_fcs: Optional[pd.DataFrame] = None,
        clustering: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """
        Exporte tous les fichiers de résultats en une seule passe.

        Args:
            df_cells: DataFrame cellulaire (marqueurs FlowSOM + cluster labels).
            df_fcs: DataFrame complet pour l'export FCS Kaluza — contient TOUS
                les marqueurs (données brutes pré-transformation) + FlowSOM_cluster,
                FlowSOM_metacluster, xGrid, yGrid, xNodes, yNodes, size,
                Condition_Num. Si None, replie sur df_cells.
            mfi_matrix: MFI par métacluster [n_clusters × n_markers].
            metaclustering: Assignation cellule → métacluster.
            selected_markers: Marqueurs utilisés pour le clustering.
            input_files: Chemins des fichiers FCS sources.
            gating_logger: Logger de gating (pour export JSON).

        Returns:
            Dict {type_export: chemin_fichier}.
        """
        ts = self.timestamp
        paths: Dict[str, str] = {}

        # Mode compact : on saute FCS complet, CSV et JSON de métadonnées/gating.
        # Seuls le rapport HTML/PDF, le MRD JSON et le FCS pathologique Is_MRD sont produits.
        _compact = (
            getattr(getattr(self.config, "export_mode", None), "mode", "standard")
            == "compact"
        )

        _do_csv = getattr(
            getattr(self.config, "export_mode", None), "export_csv", True
        )

        if _compact:
            _logger.info(
                "Mode compact activé — FCS complet, CSV et JSON métadonnées ignorés."
            )
        else:
            # ── 1. FCS ────────────────────────────────────────────────────────
            fcs_source = df_fcs if df_fcs is not None else df_cells
            fcs_path = self._dirs["fcs"] / f"flowsom_results_{ts}.fcs"
            ok = export_to_fcs_kaluza(fcs_source, fcs_path)
            if ok:
                paths["fcs_complete"] = str(fcs_path)
                _logger.info("FCS exporté: %s", fcs_path.name)

            if not _do_csv:
                _logger.info(
                    "Export CSV désactivé (export_csv=false) — CSV complet, stats, MFI et per_file ignorés."
                )
            else:
                # ── 2. CSV complet ─────────────────────────────────────────────────
                csv_path = self._dirs["csv"] / f"flowsom_complete_{ts}.csv"
                ok = export_cells_csv(df_cells, csv_path)
                if ok:
                    paths["csv_complete"] = str(csv_path)

                # ── 3. CSV par fichier FCS source ──────────────────────────────────
                _do_per_file_csv = getattr(
                    getattr(self.config, "export_mode", None), "export_per_file_csv", True
                )
                if _do_per_file_csv:
                    export_per_file_csv(
                        df_cells, self._dirs["csv"] / "per_file", timestamp=ts
                    )
                    paths["csv_per_file"] = str(self._dirs["csv"] / "per_file")
                else:
                    _logger.info("Export CSV per_file désactivé (export_per_file_csv=false).")

                # ── 4. Statistiques par cluster ────────────────────────────────────
                stats_df = compute_cluster_statistics(
                    df_cells,
                    marker_columns=[m for m in selected_markers if m in df_cells.columns],
                    cluster_column="FlowSOM_metacluster",
                )
                if self.patho_name is not None:
                    stats_df.insert(0, "Fichier_Patho", self.patho_name)
                if self.patho_date is not None:
                    stats_df.insert(
                        1 if self.patho_name is not None else 0, "Date_Patho", self.patho_date
                    )
                stats_path = self._dirs["csv"] / f"flowsom_statistics_{ts}.csv"
                ok = export_statistics_csv(stats_df, stats_path)
                if ok:
                    paths["csv_statistics"] = str(stats_path)

                # ── 5. Matrice MFI ─────────────────────────────────────────────────
                mfi_path = self._dirs["csv"] / f"flowsom_mfi_{ts}.csv"
                ok = export_mfi_matrix_csv(mfi_matrix, mfi_path)
                if ok:
                    paths["csv_mfi"] = str(mfi_path)

            # ── 6. Log de gating ──────────────────────────────────────────────
            if gating_logger is not None:
                gating_path = self._dirs["other"] / f"gating_log_{ts}.json"
                ok = export_gating_log(
                    [e.to_dict() for e in gating_logger.events],
                    gating_path,
                )
                if ok:
                    paths["gating_log"] = str(gating_path)

            # ── 7. Métadonnées JSON ───────────────────────────────────────────
            metadata = build_analysis_metadata(
                input_files=input_files,
                config_dict=self.config.to_dict()
                if hasattr(self.config, "to_dict")
                else {},
                n_cells=len(df_cells),
                marker_names=list(df_cells.columns),
                used_markers=selected_markers,
                metaclustering=metaclustering,
                n_metaclusters=int(metaclustering.max()) + 1
                if len(metaclustering) > 0
                else 0,
                cell_data_obs=df_cells[["condition", "file_origin"]]
                if "condition" in df_cells.columns
                else None,
                export_paths=paths,
            )
            meta_path = self._dirs["other"] / f"flowsom_metadata_{ts}.json"
            ok = export_analysis_metadata(metadata, meta_path)
            if ok:
                paths["json_metadata"] = str(meta_path)

        _logger.info(
            "Exports terminés: %d fichiers dans %s",
            len(paths),
            self.output_dir,
        )
        return paths

    def export_cluster_distribution(
        self,
        clustering: np.ndarray,
        metaclustering: np.ndarray,
        condition_labels: np.ndarray,
    ) -> Dict[str, str]:
        """
        Exporte la distribution Sain/Patho par noeud SOM et par metacluster.

        Args:
            clustering:       Assignation cellule -> noeud SOM.
            metaclustering:   Assignation cellule -> metacluster.
            condition_labels: Label de condition par cellule.

        Returns:
            Dict {cle: chemin_fichier}.
        """
        export_cfg = getattr(self.config, "_extra", {}).get(
            "export_cluster_distribution", {}
        )
        return export_cluster_distribution(
            clustering=clustering,
            metaclustering=metaclustering,
            condition_labels=condition_labels,
            output_dir=self._dirs["csv"],
            timestamp=self.timestamp,
            export_cfg=export_cfg,
        )

    def export_gating_plots(
        self,
        X_raw: np.ndarray,
        var_names: List[str],
        masks: Dict[str, np.ndarray],
        sample_name: str = "sample",
    ) -> bool:
        """
        Génère et sauvegarde les graphiques de gating.

        Args:
            X_raw: Matrice brute pré-gating.
            var_names: Noms des marqueurs.
            masks: Dict {gate_name: bool_mask}.
            sample_name: Nom de l'échantillon.

        Returns:
            True si au moins un graphique généré.
        """
        from src.visualization.gating_plots import (
            generate_all_gating_plots,
        )

        results = generate_all_gating_plots(
            X_raw,
            var_names,
            masks,
            output_dir=self._dirs["plots"] / "gating",
            sample_name=sample_name,
        )
        return any(results.values())

    def export_flowsom_plots(
        self,
        mfi_matrix: pd.DataFrame,
        metaclustering: np.ndarray,
        n_metaclusters: int,
        condition_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Génère et sauvegarde les graphiques FlowSOM.

        Returns:
            Dict {fig_name: figure_object} des figures générées.
        """
        from src.visualization.flowsom_plots import (
            plot_mfi_heatmap,
            plot_metacluster_sizes,
        )

        ts = self.timestamp
        figures: Dict[str, Any] = {}

        fig1 = plot_mfi_heatmap(
            mfi_matrix,
            self._dirs["plots"] / f"mfi_heatmap_{ts}.png",
        )
        if fig1 is not None:
            figures["fig_heatmap"] = fig1

        fig2 = plot_metacluster_sizes(
            metaclustering,
            n_metaclusters,
            self._dirs["plots"] / f"metacluster_distribution_{ts}.png",
            condition_labels=condition_labels,
        )
        if fig2 is not None:
            figures["fig_comp"] = fig2

        return figures

    def export_sankey(
        self,
        gate_counts: Dict[str, int],
        *,
        filter_blasts: bool = False,
        per_file_counts: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> Dict[str, Any]:
        """
        Génère le diagramme Sankey global + mini-Sankey par fichier.

        Returns:
            Dict avec clés: chemins fichiers + 'fig_sankey' (go.Figure).
        """
        from src.visualization.gating_plots import (
            generate_sankey_diagram,
            generate_per_file_sankey,
        )

        paths: Dict[str, Any] = {}
        ts = self.timestamp

        sankey_dir = self._dirs["plots"] / "sankey"
        sankey_dir.mkdir(parents=True, exist_ok=True)

        fig_sankey = generate_sankey_diagram(
            gate_counts,
            sankey_dir / f"sankey_global_{ts}.html",
            filter_blasts=filter_blasts,
        )
        if fig_sankey is not None:
            paths["sankey_global"] = str(sankey_dir / f"sankey_global_{ts}.html")
            paths["fig_sankey"] = fig_sankey

        if per_file_counts:
            generate_per_file_sankey(
                per_file_counts,
                sankey_dir,
                filter_blasts=filter_blasts,
                timestamp=ts,
            )
            paths["sankey_per_file"] = str(sankey_dir)

        return paths

    def export_html_report(
        self,
        *,
        plotly_figures: Optional[Dict[str, Any]] = None,
        matplotlib_figures: Optional[Dict[str, Any]] = None,
        figure_labels: Optional[Dict[str, str]] = None,
        analysis_params: Optional[Dict[str, Any]] = None,
        summary_stats: Optional[Dict[str, Any]] = None,
        metacluster_table: Optional[List[Dict[str, Any]]] = None,
        markers: Optional[List[str]] = None,
        condition_data: Optional[List[Dict[str, Any]]] = None,
        files_data: Optional[List[Dict[str, Any]]] = None,
        export_paths: Optional[Dict[str, str]] = None,
        self_contained: bool = True,
        patho_info: Optional[Dict[str, str]] = None,
        ransac_summary: Optional[Dict[str, Any]] = None,
        pipeline_result: Optional[Any] = None,
        algo_gauges: Optional[List[Dict[str, Any]]] = None,
        prescreening_result: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Génère le rapport HTML complet self-contained.

        Args:
            plotly_figures: Dict {nom: go.Figure}.
            matplotlib_figures: Dict {nom: mpl.Figure}.
            figure_labels: Dict {nom: label_lisible}.
            analysis_params: Paramètres d'analyse.
            summary_stats: Statistiques résumé {n_cells, n_markers, etc.}.
            metacluster_table: Lignes du tableau métaclusters.
            markers: Liste des marqueurs.
            self_contained: Embarquer plotly.js dans le HTML.
            prescreening_result: PrescreeningResult (pré-screening CD34+/CD45dim).

        Returns:
            Chemin du fichier HTML si succès, None sinon.
        """
        from src.visualization.html_report import (
            generate_html_report,
        )

        html_path = self._dirs["other"] / f"{self.name_stem}.html"

        # Utiliser les infos patho de l'instance si non passées explicitement
        _patho_info = patho_info
        if _patho_info is None and self.patho_name:
            _patho_info = {
                "name": self.patho_name,
                "date": self.patho_date or "Date inconnue",
            }
        _viz_cfg_html = getattr(self.config, "visualization", None)
        _html_dpi_mpl = int(getattr(_viz_cfg_html, "pdf_dpi_mpl", 100))
        _curated_pct   = getattr(pipeline_result, "curated_mrd_percent", None) if pipeline_result else None
        _curated_cells = getattr(pipeline_result, "curated_mrd_cells",   None) if pipeline_result else None
        _curated_nodes = getattr(pipeline_result, "curated_nodes",        None) if pipeline_result else None

        ok = generate_html_report(
            html_path,
            plotly_figures=plotly_figures,
            matplotlib_figures=matplotlib_figures,
            figure_labels=figure_labels,
            analysis_params=analysis_params,
            summary_stats=summary_stats,
            metacluster_table=metacluster_table,
            markers=markers,
            condition_data=condition_data,
            files_data=files_data,
            export_paths=export_paths,
            self_contained=self_contained,
            patho_info=_patho_info,
            dpi_mpl=_html_dpi_mpl,
            ransac_summary=ransac_summary,
            curated_mrd_percent=_curated_pct,
            curated_mrd_cells=_curated_cells,
            curated_nodes=_curated_nodes,
            algo_gauges=algo_gauges,
            prescreening_result=prescreening_result,
        )

        return str(html_path) if ok else None

    def export_pdf_report(
        self,
        *,
        plotly_figures: Optional[Dict[str, Any]] = None,
        matplotlib_figures: Optional[Dict[str, Any]] = None,
        figure_labels: Optional[Dict[str, str]] = None,
        analysis_params: Optional[Dict[str, Any]] = None,
        summary_stats: Optional[Dict[str, Any]] = None,
        metacluster_table: Optional[List[Dict[str, Any]]] = None,
        markers: Optional[List[str]] = None,
        condition_data: Optional[List[Dict[str, Any]]] = None,
        files_data: Optional[List[Dict[str, Any]]] = None,
        export_paths: Optional[Dict[str, str]] = None,
        patho_info: Optional[Dict[str, str]] = None,
        ransac_summary: Optional[Dict[str, Any]] = None,
        pipeline_result: Optional[Any] = None,
        algo_gauges: Optional[List[Dict[str, Any]]] = None,
        prescreening_result: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Génère le rapport PDF A4 complet (même contenu que le HTML).

        Nécessite : reportlab >= 4.0 et kaleido >= 0.1 pour les figures Plotly.

        Returns:
            Chemin du fichier PDF si succès, None sinon.
        """
        from src.visualization.pdf_report import (
            generate_pdf_report,
        )

        pdf_path = self._dirs["other"] / f"{self.name_stem}.pdf"

        _patho_info = patho_info
        if _patho_info is None and self.patho_name:
            _patho_info = {
                "name": self.patho_name,
                "date": self.patho_date or "Date inconnue",
            }
        _viz_cfg = getattr(self.config, "visualization", None)
        _pdf_dpi_mpl = int(getattr(_viz_cfg, "pdf_dpi_mpl", 100))
        # Extraire les champs de curation depuis pipeline_result si fourni
        _curated_pct   = getattr(pipeline_result, "curated_mrd_percent", None) if pipeline_result else None
        _curated_cells = getattr(pipeline_result, "curated_mrd_cells",   None) if pipeline_result else None
        _curated_nodes = getattr(pipeline_result, "curated_nodes",        None) if pipeline_result else None

        return generate_pdf_report(
            pdf_path,
            plotly_figures=plotly_figures,
            matplotlib_figures=matplotlib_figures,
            figure_labels=figure_labels,
            analysis_params=analysis_params,
            summary_stats=summary_stats,
            metacluster_table=metacluster_table,
            markers=markers,
            condition_data=condition_data,
            files_data=files_data,
            export_paths=export_paths,
            timestamp=datetime.now().strftime("%d/%m/%Y %H:%M"),
            patho_info=_patho_info,
            dpi_mpl=_pdf_dpi_mpl,
            ransac_summary=ransac_summary,
            curated_mrd_percent=_curated_pct,
            curated_mrd_cells=_curated_cells,
            curated_nodes=_curated_nodes,
            algo_gauges=algo_gauges,
        )

    def export_mrd_dashboard_csv(
        self,
        pipeline_result: Any,
        *,
        gauges: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Exporte le fichier ``dashboard_metrics.csv`` avec une ligne par patient.

        Colonnes produites :
            Timestamp, Fichier_Patho, Date_Patho,
            MRD_JF (%), MRD_Flo (%), MRD_ELN (%),
            MRD_Algo_Best (%),          ← meilleur résultat algorithmique
            MRD_Curated (%),            ← résultat après curation experte (ou «» si absent)
            N_MRD_Cells_Algo,
            N_MRD_Cells_Curated,        ← cellules après curation (ou «» si absent)
            N_Curated_Nodes,            ← nœuds validés par l'expert (ou «» si absent)
            N_Cells_Total

        Args:
            pipeline_result:  Instance de PipelineResult contenant mrd_result
                              et les champs curated_*.
            gauges:           Liste de dicts ``{method, pct, n_cells}`` déjà
                              calculés par HomeTab._recompute_gauges_with_denom().
                              Si None, les colonnes MRD_JF/Flo/ELN restent vides.

        Returns:
            Chemin absolu du CSV écrit, ou None si erreur.
        """
        try:
            mrd = getattr(pipeline_result, "mrd_result", None)
            gauges = gauges or []

            # ── Valeurs algorithmiques brutes ──────────────────────────
            algo_by_method: Dict[str, float] = {
                g["method"]: float(g.get("pct", 0.0)) for g in gauges
            }
            algo_best = max(algo_by_method.values(), default=0.0)

            n_algo_cells = 0
            if mrd is not None:
                n_algo_cells = max(
                    getattr(mrd, "mrd_cells_jf",  0),
                    getattr(mrd, "mrd_cells_flo", 0),
                )

            n_total = getattr(pipeline_result, "n_cells", 0)

            # ── Valeurs curées (optionnelles) ──────────────────────────
            curated_pct   = pipeline_result.curated_mrd_percent  # float | None
            curated_cells = pipeline_result.curated_mrd_cells    # int   | None
            curated_nodes_list = pipeline_result.curated_nodes or []
            n_curated_nodes = len(curated_nodes_list) if curated_nodes_list else (
                "" if curated_pct is None else 0
            )

            row: Dict[str, Any] = {
                "Timestamp":            pipeline_result.timestamp,
                "Fichier_Patho":        self.patho_name or "",
                "Date_Patho":           self.patho_date or "",
                "MRD_JF (%)":           algo_by_method.get("JF",  ""),
                "MRD_Flo (%)":          algo_by_method.get("Flo", ""),
                "MRD_ELN (%)":          algo_by_method.get("ELN", ""),
                "MRD_Algo_Best (%)":    round(algo_best, 6) if algo_best else "",
                "MRD_Curated (%)":      round(curated_pct, 6) if curated_pct is not None else "",
                "N_MRD_Cells_Algo":     n_algo_cells or "",
                "N_MRD_Cells_Curated":  curated_cells if curated_cells is not None else "",
                "N_Curated_Nodes":      n_curated_nodes,
                "N_Cells_Total":        n_total,
            }

            csv_path = self._dirs["csv"] / f"dashboard_metrics_{self.timestamp}.csv"

            # Append si le fichier existe déjà (mode batch), sinon création
            mode = "a" if csv_path.exists() else "w"
            header = not csv_path.exists()
            pd.DataFrame([row]).to_csv(csv_path, mode=mode, header=header, index=False)

            pipeline_result.output_files["csv_dashboard"] = str(csv_path)
            _logger.info("Dashboard CSV exporté : %s", csv_path.name)
            return str(csv_path)

        except Exception as exc:
            _logger.error("export_mrd_dashboard_csv: %s", exc)
            return None

