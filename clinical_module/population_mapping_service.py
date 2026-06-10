"""
population_mapping_service.py — Orchestrateur du mapping populations (Section 10).

Ce service exécute l'intégralité de la Section 10 du pipeline FlowSOM :

Étapes :
  10.1  Chargement de la configuration de mapping.
  10.2  Extraction des centroïdes SOM depuis le FCS exporté.
  10.3  Chargement des CSV de référence MFI (avec cache Parquet transformé).
  10.3b Calcul des statistiques biologiques de référence (Welford + KNN).
  10.4  Mapping V3 (euclidien, fallback rapide).
  10.4b Mapping V5 M12 (cosine + prior log10³ + hard limit) [recommandé ELN 2022].
  10.4c Scoring blast des nœuds Unknown.
  10.4d Traçabilité FCS source pour les nœuds blast.
  10.5  Visualisation MST interactif par population.
  10.5b Visualisation grille SOM ScatterGL.
  10.6  Heatmap comparative CSV-ref vs MetaClusters.

Usage :
    service = PopulationMappingService(config)
    results = service.run_full_mapping(
        fcs_path=Path("output/final_with_clusters.fcs"),
        cluster_data=cluster_data,  # ClusteringResult
        cell_data=adata,            # AnnData
        output_dir=Path("output"),
        timestamp="20250101_120000",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from clinical_module.phenotyping.blast_detection import (
    build_blast_score_dataframe,
    build_blast_weights,
    compute_reference_normalization,
    trace_blast_cells_to_fcs_source,
)
from clinical_module.phenotyping.population_mapping import (
    build_population_color_map,
    compute_pop_stats_from_csv,
    extract_node_centroids_from_fcs,
    load_population_csv_transformed,
    map_nodes_to_metaclusters,
    map_populations_to_nodes_v3,
    map_populations_to_nodes_v5,
)
from config.pipeline_config import PopulationMappingConfig
from prisma.utils.logger import get_logger

_logger = get_logger("services.population_mapping")


class PopulationMappingResult:
    """Conteneur typé pour les résultats du mapping population."""

    __slots__ = (
        "mapping_v3_df",
        "mapping_v5_df",
        "df_ref_mfi",
        "node_mfi_df",
        "node_coords_df",
        "node_counts",
        "mc_per_node",
        "blast_candidates_df",
        "blast_source_df",
        "method_used",
        "color_map",
        "marker_cols_a",
        "figures_plotly",  # Dict[str, go.Figure] collecté lors des visualisations
    )

    def __init__(self) -> None:
        self.mapping_v3_df: Optional[pd.DataFrame] = None
        self.mapping_v5_df: Optional[pd.DataFrame] = None
        self.df_ref_mfi: Optional[pd.DataFrame] = None
        self.node_mfi_df: Optional[pd.DataFrame] = None
        self.node_coords_df: Optional[pd.DataFrame] = None
        self.node_counts: Optional[np.ndarray] = None
        self.mc_per_node: Optional[np.ndarray] = None
        self.blast_candidates_df: Optional[pd.DataFrame] = None
        self.blast_source_df: Optional[pd.DataFrame] = None
        self.method_used: str = "M12"
        self.color_map: Dict[str, str] = {}
        self.marker_cols_a: List[str] = []
        self.figures_plotly: Dict[str, Any] = {}  # clé = nom notebook (fig_mst, etc.)


class PopulationMappingService:
    """
    Orchestre l'ensemble de la Section 10 du pipeline FlowSOM.

    Injection de dépendance par configuration : tous les paramètres
    (chemins, méthodes, seuils ELN) sont lus depuis PopulationMappingConfig.
    """

    def __init__(self, config: PopulationMappingConfig) -> None:
        self.cfg = config

    # ─────────────────────────────────────────────────────────────────────────
    #  Point d'entrée principal
    # ─────────────────────────────────────────────────────────────────────────

    def run_full_mapping(
        self,
        fcs_path: Path,
        cluster_data: Any,
        cell_data: Any,
        output_dir: Path,
        timestamp: str = "",
    ) -> PopulationMappingResult:
        """
        Exécute toutes les étapes §10.1 → §10.6.

        Args:
            fcs_path: Chemin du FCS final exporté (avec colonnes FlowSOM_cluster, etc.).
            cluster_data: Résultat du clustering (ClusteringResult ou AnnData).
            cell_data: AnnData complet avec .obs contenant toutes les cellules.
            output_dir: Répertoire racine des sorties (figures, exports CSV).
            timestamp: Suffixe ISO pour les noms de fichiers.

        Returns:
            PopulationMappingResult avec tous les résultats intermédiaires.
        """
        result = PopulationMappingResult()

        if not self.cfg.enabled:
            _logger.info("[§10] Population mapping désactivé dans la config.")
            return result

        _logger.info("=" * 70)
        _logger.info("[§10] DÉBUT DU MAPPING DES POPULATIONS VIA MFI")
        _logger.info("=" * 70)

        # ── §10.2 — Extraction des centroïdes SOM ─────────────────────────────
        _logger.info("[§10.2] Extraction des centroïdes SOM depuis FCS...")
        try:
            (
                result.node_mfi_df,
                result.node_coords_df,
                result.node_counts,
                result.mc_per_node,
                _node_ids,
            ) = extract_node_centroids_from_fcs(
                fcs_path=fcs_path,
                transform_type=self.cfg.transform_method,
                cofactor=self.cfg.arcsinh_cofactor,
                apply_to_scatter=False,
            )
            result.marker_cols_a = list(result.node_mfi_df.columns)
            _logger.info(
                "[§10.2] OK: %d nœuds × %d marqueurs -A",
                len(result.node_mfi_df),
                len(result.marker_cols_a),
            )
        except Exception as exc:
            _logger.error("[§10.2] Échec extraction centroïdes: %s", exc)
            return result

        # ── §10.3 — Chargement des CSV de référence MFI ───────────────────────
        _logger.info("[§10.3] Chargement des CSV de référence MFI...")
        ref_mfi_dir = Path(self.cfg.ref_mfi_dir) if self.cfg.ref_mfi_dir else None
        if ref_mfi_dir is None or not ref_mfi_dir.exists():
            _logger.warning(
                "[§10.3] Répertoire ref_mfi_dir introuvable: %s", ref_mfi_dir
            )
            return result

        cache_dir = (
            Path(self.cfg.cache_dir)
            if self.cfg.cache_dir
            else (output_dir / "ref_mfi_parquet_cache")
        )

        pop_mfi_raw_ref: Dict[str, pd.Series] = {}
        csv_paths_by_pop: Dict[str, Path] = {}

        for csv_file in sorted(ref_mfi_dir.glob("*.csv")):
            pop_name = csv_file.stem
            try:
                series = load_population_csv_transformed(
                    csv_path=csv_file,
                    cache_dir=cache_dir,
                    area_cols_to_keep=result.marker_cols_a,
                    transform_type=self.cfg.transform_method,
                    cofactor=self.cfg.arcsinh_cofactor,
                    apply_to_scatter=False,
                )
                pop_mfi_raw_ref[pop_name] = series
                csv_paths_by_pop[pop_name] = csv_file
                _logger.info("  [OK] %s (%d valeurs)", pop_name, len(series))
            except Exception as exc:
                _logger.warning("  [!] %s: %s", csv_file.name, exc)

        if not pop_mfi_raw_ref:
            _logger.error("[§10.3] Aucune référence MFI chargée — mapping impossible.")
            return result

        # Construire le DataFrame de référence (n_pops × n_markers_A)
        df_ref_raw = pd.DataFrame(pop_mfi_raw_ref).T  # index = pop_name, cols = markers
        # Aligner strictement sur les colonnes -A du FCS
        cols_in_both = [c for c in result.marker_cols_a if c in df_ref_raw.columns]
        df_ref_raw = df_ref_raw[cols_in_both]
        result.df_ref_mfi = df_ref_raw
        _logger.info(
            "[§10.3] Référence constituée: %d populations × %d marqueurs",
            len(df_ref_raw),
            len(cols_in_both),
        )

        # ── §10.3b — Statistiques biologiques de référence ───────────────────
        _logger.info("[§10.3b] Calcul statistiques biologiques (Welford + KNN)...")
        pop_cell_counts: Dict[str, int] = {}
        pop_cov_matrices: Dict[str, np.ndarray] = {}
        pop_knn_samples: Dict[str, np.ndarray] = {}

        if self.cfg.compute_population_stats:
            for pop_name, csv_path in csv_paths_by_pop.items():
                try:
                    n, cov, knn = compute_pop_stats_from_csv(
                        csv_path=csv_path,
                        cols=cols_in_both,
                        transform_type=self.cfg.transform_method,
                        cofactor=self.cfg.arcsinh_cofactor,
                        knn_sample_size=self.cfg.knn_sample_size,
                    )
                    pop_cell_counts[pop_name] = n
                    pop_cov_matrices[pop_name] = cov
                    pop_knn_samples[pop_name] = knn
                    _logger.info(
                        "  [OK] %s: n=%d, knn_sample=%d", pop_name, n, len(knn)
                    )
                except Exception as exc:
                    _logger.debug("  [!] Stats %s: %s", pop_name, exc)

        # Aligner les node_mfi_df sur cols_in_both
        node_mfi_aligned = result.node_mfi_df[cols_in_both]

        # ── §10.4 — Mapping V3 (Euclidean — fallback rapide) ─────────────────
        _logger.info("[§10.4] Mapping V3 (euclidien)...")
        try:
            result.mapping_v3_df = map_populations_to_nodes_v3(
                node_mfi_raw=node_mfi_aligned,
                pop_mfi_ref=df_ref_raw,
                include_scatter=self.cfg.include_scatter,
                distance_percentile=self.cfg.distance_percentile,
                normalization_method=self.cfg.normalization_method,
            )
            _logger.info(
                "[§10.4] V3 terminé: %d nœuds, %d Unknown",
                len(result.mapping_v3_df),
                int((result.mapping_v3_df["assigned_pop"] == "Unknown").sum()),
            )
        except Exception as exc:
            _logger.error("[§10.4] V3 échoué: %s", exc)

        # ── §10.4b — Mapping V5 (méthode recommandée M12 ou benchmark) ───────
        method = self.cfg.mapping_method  # ex: "M12" ou "cosine_prior"
        run_benchmark = method.lower() == "benchmark"
        _logger.info(
            "[§10.4b] Mapping V5 [%s]%s...",
            method,
            " (benchmark M1–M12)" if run_benchmark else "",
        )
        try:
            node_sizes = result.node_counts if result.node_counts is not None else None
            result.mapping_v5_df = map_populations_to_nodes_v5(
                node_mfi_raw=node_mfi_aligned,
                pop_mfi_ref=df_ref_raw,
                node_sizes=node_sizes,
                cell_counts=pop_cell_counts if pop_cell_counts else None,
                pop_cov_matrices=pop_cov_matrices if pop_cov_matrices else None,
                pop_knn_samples=pop_knn_samples if pop_knn_samples else None,
                method=method,
                include_scatter=self.cfg.include_scatter,
                threshold_mode=self.cfg.unknown_threshold_mode,
                threshold_percentile=self.cfg.distance_percentile,
                normalization_method=self.cfg.normalization_method,
                hard_limit_factor=self.cfg.hard_limit_factor,
                prior_mode=self.cfg.prior_mode,
                transform_method="none",  # déjà transformé à §10.2
                data_already_transformed=True,
                run_benchmark=run_benchmark,
                knn_k=15,
                total_knn_points=15_000,
            )
            result.method_used = (
                str(result.mapping_v5_df["method"].iloc[0])
                if "method" in result.mapping_v5_df.columns
                else method
            )
            _logger.info(
                "[§10.4b] V5 [%s] terminé: %d nœuds, %d Unknown",
                result.method_used,
                len(result.mapping_v5_df),
                int((result.mapping_v5_df["assigned_pop"] == "Unknown").sum()),
            )

            # Export CSV du mapping
            self._export_mapping_csv(
                result.mapping_v5_df, output_dir, timestamp, result.method_used
            )

        except Exception as exc:
            _logger.error("[§10.4b] V5 échoué: %s", exc)
            result.mapping_v5_df = result.mapping_v3_df  # fallback V3

        # Enrichir avec les métaclusters
        best_mapping = (
            result.mapping_v5_df
            if result.mapping_v5_df is not None
            else result.mapping_v3_df
        )
        if best_mapping is not None and result.mc_per_node is not None:
            best_mapping = map_nodes_to_metaclusters(best_mapping, result.mc_per_node)
            if result.mapping_v5_df is not None:
                result.mapping_v5_df = best_mapping

        # ── §10.4c — Blast scoring ────────────────────────────────────────────
        if self.cfg.blast_enabled and best_mapping is not None:
            _logger.info("[§10.4c] Scoring blast des nœuds Unknown...")
            result.blast_candidates_df = self._run_blast_scoring(
                mapping_df=best_mapping,
                node_mfi_aligned=node_mfi_aligned,
                df_ref_raw=df_ref_raw,
                node_counts=result.node_counts,
                output_dir=output_dir,
                timestamp=timestamp,
            )

        # ── §10.4d — Traçabilité FCS source ───────────────────────────────────
        if (
            result.blast_candidates_df is not None
            and not result.blast_candidates_df.empty
            and cell_data is not None
        ):
            _logger.info("[§10.4d] Traçabilité FCS source...")
            try:
                result.blast_source_df = trace_blast_cells_to_fcs_source(
                    blast_candidates_df=result.blast_candidates_df,
                    cell_data=cell_data,
                    condition_col="Condition",
                )
            except Exception as exc:
                _logger.warning("[§10.4d] Traçabilité échouée: %s", exc)

        # ── §10.5–10.6 — Visualisations ───────────────────────────────────────
        self._run_visualizations(
            result=result,
            best_mapping=best_mapping,
            df_ref_raw=df_ref_raw,
            node_mfi_aligned=node_mfi_aligned,
            cell_data=cell_data,
            cluster_data=cluster_data,
            output_dir=output_dir,
            timestamp=timestamp,
        )

        _logger.info("[§10] MAPPING DES POPULATIONS TERMINÉ.")
        _logger.info("=" * 70)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    #  Sous-méthodes internes
    # ─────────────────────────────────────────────────────────────────────────

    def _run_blast_scoring(
        self,
        mapping_df: pd.DataFrame,
        node_mfi_aligned: pd.DataFrame,
        df_ref_raw: pd.DataFrame,
        node_counts: Optional[np.ndarray],
        output_dir: Path,
        timestamp: str,
    ) -> Optional[pd.DataFrame]:
        """§10.4c — Scoring ELN 2022 des nœuds Unknown."""
        try:
            unknown_mask = mapping_df["assigned_pop"] == "Unknown"
            if not unknown_mask.any():
                _logger.info("[§10.4c] Aucun nœud Unknown — blast scoring ignoré.")
                return None

            unknown_node_ids = mapping_df.loc[unknown_mask, "node_id"].values
            X_unknown = node_mfi_aligned.iloc[unknown_node_ids].values.astype(float)
            X_reference = df_ref_raw.values.astype(float)

            X_norm, _, _, _ = compute_reference_normalization(X_unknown, X_reference)

            marker_names = list(node_mfi_aligned.columns)
            weights = build_blast_weights(marker_names)
            cell_counts_per_node = (
                {
                    int(nid): int(node_counts[nid])
                    for nid in unknown_node_ids
                    if nid < len(node_counts)
                }
                if node_counts is not None
                else None
            )

            blast_df = build_blast_score_dataframe(
                node_ids=unknown_node_ids,
                X_norm=X_norm,
                marker_names=marker_names,
                cell_counts_per_node=cell_counts_per_node,
                weights=weights,
            )

            # Export CSV
            blast_csv = output_dir / f"blast_candidates_10.4c_{timestamp}.csv"
            blast_df.to_csv(blast_csv, index=False, sep=";", decimal=",")
            _logger.info("[§10.4c] Blast candidates: %s", blast_csv.name)

            return blast_df

        except Exception as exc:
            _logger.error("[§10.4c] Blast scoring échoué: %s", exc)
            return None

    def _run_visualizations(
        self,
        result: PopulationMappingResult,
        best_mapping: Optional[pd.DataFrame],
        df_ref_raw: pd.DataFrame,
        node_mfi_aligned: pd.DataFrame,
        cell_data: Any,
        cluster_data: Any,
        output_dir: Path,
        timestamp: str,
    ) -> None:
        """§10.4c–§10.6 — Génération de toutes les visualisations."""
        try:
            from prisma.visualization.population_viz import (
                plot_blast_fcs_source,
                plot_blast_heatmap,
                plot_blast_radar,
                plot_blast_scores_bar,
                plot_heatmap_comparative,
                plot_lympho_verification,
                plot_mrd_blast_radar,
                plot_mst_interactive,
                plot_som_grid_interactive,
            )
        except ImportError as exc:
            _logger.warning("Visualisations population non disponibles: %s", exc)
            return

        color_map = build_population_color_map(
            list(best_mapping["assigned_pop"].unique())
            if best_mapping is not None
            else [],
            custom_colors=self.cfg.population_colors,
        )
        result.color_map = color_map

        # ── §10.4c Blast charts ───────────────────────────────────────────────
        if (
            result.blast_candidates_df is not None
            and not result.blast_candidates_df.empty
        ):
            _logger.info("[§10.4c] Génération des charts blast...")
            for viz_fn, name, fig_key in [
                (plot_blast_heatmap, "heatmap", "fig_barplots"),
                (plot_blast_radar, "radar", "fig_radar"),
                (plot_blast_scores_bar, "bar", "fig_stars"),
            ]:
                try:
                    _fig = viz_fn(
                        blast_df=result.blast_candidates_df,
                        output_dir=output_dir,
                        timestamp=timestamp,
                    )
                    if _fig is not None:
                        result.figures_plotly[fig_key] = _fig
                except Exception as exc:
                    _logger.debug("[§10.4c] Chart '%s' échoué: %s", name, exc)

        # ── §10.4d Traçabilité FCS source ──────────────────────────────────────
        if result.blast_source_df is not None and not result.blast_source_df.empty:
            try:
                plot_blast_fcs_source(
                    source_df=result.blast_source_df,
                    output_dir=output_dir,
                    timestamp=timestamp,
                )
            except Exception as exc:
                _logger.debug("[§10.4d] Source FCS chart échoué: %s", exc)

        # ── §10.4e Radar MRD clinique (Porte 2) ───────────────────────────────
        if (
            result.blast_candidates_df is not None
            and not result.blast_candidates_df.empty
        ):
            try:
                # Récupérer les MRDClusterResult si disponibles
                _mrd_per_node = getattr(result, "mrd_result", None)
                _mrd_per_node_list = (
                    getattr(_mrd_per_node, "per_node", None)
                    if _mrd_per_node is not None
                    else None
                )
                _fig_mrd_radar = plot_mrd_blast_radar(
                    blast_df=result.blast_candidates_df,
                    mrd_per_node=_mrd_per_node_list,
                    output_dir=output_dir,
                    timestamp=timestamp,
                )
                if _fig_mrd_radar is not None:
                    result.figures_plotly["fig_mrd_blast_radar"] = _fig_mrd_radar
                    _logger.info("[§10.4e] Radar MRD clinique généré.")
            except Exception as exc:
                _logger.warning("[§10.4e] Radar MRD clinique échoué: %s", exc, exc_info=True)

        # ── §10.5 MST interactif ──────────────────────────────────────────────
        if best_mapping is not None and result.node_coords_df is not None:
            _logger.info("[§10.5] MST interactif par population...")
            try:
                mst_edges = self._extract_mst_edges(cluster_data)
                _fig_mst = plot_mst_interactive(
                    mapping_df=best_mapping,
                    node_coords_df=result.node_coords_df,
                    node_counts=result.node_counts
                    if result.node_counts is not None
                    else np.ones(len(best_mapping)),
                    mst_edges=mst_edges,
                    color_map=color_map,
                    mc_per_node=result.mc_per_node,
                    output_dir=output_dir,
                    timestamp=timestamp,
                )
                if _fig_mst is not None:
                    result.figures_plotly["fig_mst"] = _fig_mst
            except Exception as exc:
                _logger.warning("[§10.5] MST interactif échoué: %s", exc)

        # ── §10.5 Vérification lymphocytes (T/NK vs B) ───────────────────────
        if best_mapping is not None and result.node_mfi_df is not None:
            _logger.info("[§10.5] Vérification sous-classification lymphocytaire...")
            try:
                _fig_lympho = plot_lympho_verification(
                    mapping_df=best_mapping,
                    node_mfi_df=node_mfi_aligned,
                    node_counts=result.node_counts,
                    mc_per_node=result.mc_per_node,
                    map_name=result.method_used or "N/A",
                    output_dir=output_dir,
                    timestamp=timestamp,
                )
                if _fig_lympho is not None:
                    result.figures_plotly["fig_lympho_verification"] = _fig_lympho
            except Exception as exc:
                _logger.warning("[§10.5] Vérification lymphocytes échouée: %s", exc)

        # ── §10.5b Grille SOM ScatterGL ───────────────────────────────────────
        if best_mapping is not None and cell_data is not None:
            _logger.info("[§10.5b] Grille SOM ScatterGL...")
            try:
                _fig_grid = plot_som_grid_interactive(
                    cell_data=cell_data,
                    mapping_df=best_mapping,
                    color_map=color_map,
                    output_dir=output_dir,
                    timestamp=timestamp,
                )
                if _fig_grid is not None:
                    result.figures_plotly["fig_grid_mc"] = _fig_grid
            except Exception as exc:
                _logger.warning("[§10.5b] SOM Grid ScatterGL échoué: %s", exc)

        # ── §10.6 Heatmap comparative ─────────────────────────────────────────
        if result.node_mfi_df is not None and len(df_ref_raw) > 0:
            _logger.info("[§10.6] Heatmap comparative z-score...")
            try:
                _fig_hmap = plot_heatmap_comparative(
                    df_ref_mfi=df_ref_raw,
                    node_mfi_df=node_mfi_aligned,
                    mc_per_node=result.mc_per_node,
                    method_name=result.method_used,
                    output_dir=output_dir,
                    timestamp=timestamp,
                )
                if _fig_hmap is not None:
                    result.figures_plotly["fig_heatmap_clinical"] = _fig_hmap
            except Exception as exc:
                _logger.warning("[§10.6] Heatmap comparative échouée: %s", exc)

    def _extract_mst_edges(
        self,
        cluster_data: Any,
    ) -> Optional[List[Tuple[int, int]]]:
        """Extrait les arêtes MST depuis cluster_data.uns['mst'] ou équivalent."""
        try:
            # AnnData: cluster_data.uns['mst'] = dict d'adjacence ou liste de tuples
            mst_data = cluster_data.uns.get("mst")
            if mst_data is None:
                return None

            if isinstance(mst_data, list):
                return [(int(e[0]), int(e[1])) for e in mst_data]

            # Tentative igraph
            try:
                import igraph as ig  # type: ignore

                g = ig.Graph.TupleList(
                    [(str(k), str(v)) for k, vs in mst_data.items() for v in vs],
                    directed=False,
                )
                return [(int(e.source), int(e.target)) for e in g.es]
            except Exception:
                pass

            # Cas dict d'adjacence {i: [j, k, ...]}
            if isinstance(mst_data, dict):
                edges = []
                for src, targets in mst_data.items():
                    for tgt in targets if isinstance(targets, list) else [targets]:
                        if int(src) <= int(tgt):
                            edges.append((int(src), int(tgt)))
                return edges

        except Exception as exc:
            _logger.debug("MST edges non extraits: %s", exc)
        return None

    def _export_mapping_csv(
        self,
        mapping_df: pd.DataFrame,
        output_dir: Path,
        timestamp: str,
        method: str,
    ) -> None:
        """Exporte le DataFrame de mapping en CSV."""
        try:
            csv_path = output_dir / f"population_mapping_{method}_{timestamp}.csv"
            mapping_df.to_csv(csv_path, index=False, sep=";", decimal=",")
            _logger.info("[§10.4b] Export mapping: %s", csv_path.name)
        except Exception as exc:
            _logger.debug("Export mapping CSV échoué: %s", exc)
