"""Exécuteur RUO modulaire basé sur Strategy Pattern.

Ce module orchestre uniquement des traitements recherche (QC, réduction
dimensionnelle, clustering, exports techniques) et exclut toute logique MRD
ou clinique.
"""

from __future__ import annotations

import enum
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from prisma.core.registry import StrategyRegistry
from prisma.strategies.base import ClusterParams, DimReducParams

from src.prisma.io.fcs_reader import get_fcs_files, load_fcs_files
from src.prisma.io.fcs_writer import export_to_fcs_kaluza
from src.prisma.core.models_legacy.pipeline_result import ClusteringMetrics, PipelineResult

log = logging.getLogger("prisma.research.executor")


StepKind = Literal["dimreduc", "clustering"]


class ResearchRunStep(enum.IntEnum):
    START = 5
    LOADING = 20
    QC = 30
    DIMRED = 55
    CLUSTERING = 75
    EXPORT = 90
    DONE = 100


@dataclass(frozen=True)
class ResearchPipelineStep:
    """Décrit une étape du pipeline modulaire."""

    kind: StepKind
    strategy_name: str
    params: DimReducParams | ClusterParams
    output_name: str


@dataclass
class ResearchPipelineResult:
    """Résultats intermédiaires et final du pipeline."""

    outputs: Dict[str, np.ndarray] = field(default_factory=dict)
    final_output: np.ndarray | None = None


@dataclass(frozen=True)
class ResolvedInputData:
    """Contexte de données effectif résolu avant exécution des stratégies."""

    dataframe: pd.DataFrame
    marker_columns: List[str]
    data_context_mode: Literal["full_file", "gated_population"]
    input_events: int
    selected_events: int
    input_fcs_files: List[str]
    gating_workspace_path: Optional[str]
    target_population: str


class ResearchPipelineExecutor:
    """Exécute une séquence de stratégies enregistrées dans StrategyRegistry."""

    def __init__(self, registry: type[StrategyRegistry] = StrategyRegistry) -> None:
        self._registry = registry

    def fit_transform(
        self,
        data: np.ndarray,
        strategy_name: str,
        params: DimReducParams,
    ) -> np.ndarray:
        strategy = self._registry.create_dimreduc(strategy_name)
        return strategy.fit_transform(data, params)

    def fit_predict(
        self,
        data: np.ndarray,
        strategy_name: str,
        params: ClusterParams,
    ) -> np.ndarray:
        strategy = self._registry.create_clustering(strategy_name)
        return strategy.fit_predict(data, params)

    def run(
        self,
        data: np.ndarray,
        steps: Sequence[ResearchPipelineStep],
    ) -> ResearchPipelineResult:
        current = np.asarray(data)
        outputs: Dict[str, np.ndarray] = {}

        for step in steps:
            if step.kind == "dimreduc":
                current = self.fit_transform(
                    current,
                    step.strategy_name,
                    self._coerce_dimreduc_params(step.params),
                )
            elif step.kind == "clustering":
                current = self.fit_predict(
                    current,
                    step.strategy_name,
                    self._coerce_cluster_params(step.params),
                )
            else:
                raise ValueError(f"Unknown pipeline step kind: {step.kind!r}")

            outputs[step.output_name] = np.asarray(current)

        return ResearchPipelineResult(outputs=outputs, final_output=current)

    def execute(
        self,
        cfg: Any,
        progress_callback: Optional[Callable[[ResearchRunStep, int], None]] = None,
    ) -> PipelineResult:
        """Exécute le pipeline RUO complet piloté par la config du Wizard."""
        from prisma import strategies as _registered_strategies  # noqa: F401

        t0 = time.perf_counter()
        result = PipelineResult()

        def _progress(step: ResearchRunStep) -> None:
            if progress_callback is not None:
                progress_callback(step, int(step))

        _progress(ResearchRunStep.START)
        wizard_cfg = dict(getattr(cfg, "_extra", {}).get("wizard", {}) or {})

        files = self._resolve_input_files(cfg)

        _progress(ResearchRunStep.LOADING)
        resolved_input = self._load_analysis_dataframe(cfg, files)
        data_df = resolved_input.dataframe
        marker_columns = resolved_input.marker_columns
        log.info(
            "[RUO] Chargement OK: mode=%s | avant=%d | après=%d | marqueurs=%d",
            resolved_input.data_context_mode,
            resolved_input.input_events,
            resolved_input.selected_events,
            len(marker_columns),
        )

        data_matrix = data_df[marker_columns].to_numpy(dtype=np.float32, copy=True)
        data_matrix = self._apply_basic_preprocessing(data_matrix, cfg, marker_columns)

        _progress(ResearchRunStep.QC)
        qc_method = str(getattr(cfg, "qc_method", "peacoqc"))
        log.info("[RUO] QC sélectionné: %s", qc_method)

        _progress(ResearchRunStep.DIMRED)
        dimred_methods = self._get_methods(
            cfg,
            "dimred_methods_enabled",
            "dimred_method",
            default=["umap"],
        )
        dimred_outputs: Dict[str, np.ndarray] = {}
        for method in dimred_methods:
            strategy_name = self._map_dimred_name(method)
            params = self._build_dimred_params(strategy_name, cfg, wizard_cfg)
            log.info("[RUO] DimRed -> %s", strategy_name)
            try:
                from prisma.cache.embedding_cache import get_or_compute

                params_dict = {k: getattr(params, k) for k in vars(params)} \
                    if hasattr(params, "__dataclass_fields__") else dict(vars(params))

                # Bind explicite (défaut) : évite la capture tardive de la variable
                # de boucle si get_or_compute différait l'appel.
                def _run_dimred(data, _sn=strategy_name, _p=params, **kw):
                    return self.fit_transform(data, _sn, _p)

                embedding = get_or_compute(
                    data_matrix, params_dict, _run_dimred, tag=strategy_name
                )
                dimred_outputs[method] = np.asarray(embedding)
            except Exception as exc:
                log.warning("[RUO] DimRed '%s' ignoré: %s", strategy_name, exc)

        _progress(ResearchRunStep.CLUSTERING)
        clustering_methods = self._get_methods(
            cfg,
            "clustering_methods_enabled",
            "clustering_method",
            default=["flowsom"],
        )
        cluster_outputs: Dict[str, np.ndarray] = {}
        # Clusterer FlowSOM retenu pour l'enrichissement FCS (grille SOM + MST).
        # Aligne la sortie RUO sur la pipeline de référence : FlowSOM_cluster,
        # FlowSOM_metacluster, xGrid, yGrid, xNodes, yNodes, size.
        flowsom_clusterer: Optional[Any] = None
        # Résultat de l'auto-clustering (best_k + scores) pour le graphique.
        auto_cluster_result: Optional[Dict[str, Any]] = None
        for method in clustering_methods:
            strategy_name = self._map_clustering_name(method)
            params = self._build_cluster_params(strategy_name, cfg, wizard_cfg)
            log.info("[RUO] Clustering -> %s", strategy_name)
            try:
                # Cas FlowSOM : on utilise FlowSOMClusterer directement (comme la
                # pipeline de référence) pour disposer de node_assignments_,
                # metacluster_map_, grille SOM et layout MST — indispensables aux
                # colonnes FlowSOM_*/xGrid/yGrid/xNodes/yNodes du FCS exporté.
                if strategy_name == "flowsom":
                    # Auto-sélection du nombre de métaclusters (3 phases :
                    # silhouette codebook → stabilité bootstrap → score composite).
                    if bool(getattr(cfg.auto_clustering, "enabled", False)):
                        auto_cluster_result = self._auto_select_k(data_matrix, params, cfg)
                        if auto_cluster_result:
                            best_k = int(auto_cluster_result["best_k"])
                            log.info("[RUO] Auto-clustering : k optimal = %d", best_k)
                            params.n_metaclusters = best_k
                    fs_clust = self._fit_flowsom_clusterer(
                        data_matrix, params, cfg, marker_names=marker_columns
                    )
                    flowsom_clusterer = fs_clust
                    # Label "métacluster par cellule" (cohérent avec les autres méthodes)
                    cluster_outputs[method] = np.asarray(
                        fs_clust.metacluster_assignments_, dtype=np.int32
                    )
                    continue

                from prisma.cache.embedding_cache import get_or_compute

                params_dict = {k: getattr(params, k) for k in vars(params)} \
                    if hasattr(params, "__dataclass_fields__") else dict(vars(params))

                # Bind explicite (défaut) : même protection que _run_dimred.
                def _run_cluster(data, _sn=strategy_name, _p=params, **kw):
                    return self.fit_predict(data, _sn, _p)

                labels = get_or_compute(
                    data_matrix, params_dict, _run_cluster, tag=strategy_name
                )
                cluster_outputs[method] = np.asarray(labels)
            except Exception as exc:
                log.warning("[RUO] Clustering '%s' ignoré: %s", strategy_name, exc)

        if not cluster_outputs:
            raise RuntimeError(
                "Aucune stratégie de clustering exécutable n'est disponible pour la sélection courante."
            )

        _progress(ResearchRunStep.EXPORT)
        final_df = self._build_output_dataframe(
            data_df, marker_columns, dimred_outputs, cluster_outputs,
            flowsom_clusterer=flowsom_clusterer,
        )
        output_dir = Path(getattr(cfg.paths, "output_dir", "Results"))
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "ruo_results.csv"
        final_df.to_csv(csv_path, index=False)

        metadata_path = output_dir / "ruo_run_metadata.json"
        input_fcs_path_value: str | List[str] | None = None
        if len(resolved_input.input_fcs_files) == 1:
            input_fcs_path_value = resolved_input.input_fcs_files[0]
        elif resolved_input.input_fcs_files:
            input_fcs_path_value = list(resolved_input.input_fcs_files)

        metadata_payload = {
            "qc_method": qc_method,
            "dimred_methods": dimred_methods,
            "clustering_methods": clustering_methods,
            "n_cells": int(len(final_df)),
            "n_markers": int(len(marker_columns)),
            "input_fcs_path": input_fcs_path_value,
            "gating_workspace_path": resolved_input.gating_workspace_path,
            "target_population": resolved_input.target_population,
            "data_context_mode": resolved_input.data_context_mode,
            "events_before_context": int(resolved_input.input_events),
            "events_after_context": int(resolved_input.selected_events),
        }
        metadata_path.write_text(
            json.dumps(metadata_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        output_files: Dict[str, str] = {
            "csv": str(csv_path),
            "metadata_json": str(metadata_path),
        }

        export_cfg = wizard_cfg.get("export_params", {})
        if bool(export_cfg.get("annotate_fcs", True)):
            fcs_path = output_dir / "ruo_results_annotated.fcs"
            if export_to_fcs_kaluza(final_df, fcs_path):
                output_files["fcs"] = str(fcs_path)

        # Visualisations natives FlowSOM (star charts MST/grille, marqueurs,
        # numéros, synthèse) — uniquement si un clusterer FlowSOM est disponible.
        if flowsom_clusterer is not None and bool(export_cfg.get("flowsom_plots", True)):
            plots = self._generate_flowsom_native_plots(
                flowsom_clusterer, output_dir, marker_columns, export_cfg
            )
            output_files.update(plots)

        # Graphique des métriques d'auto-clustering (silhouette / stabilité /
        # composite) avec marquage du k optimal retenu.
        if auto_cluster_result is not None:
            opt_path = self._plot_auto_cluster_metrics(
                auto_cluster_result, output_dir, cfg, export_cfg
            )
            if opt_path:
                output_files["auto_clustering_metrics"] = opt_path

        # n_nodes : dimensions réelles du SOM entraîné (peut différer de la config
        # après compute_optimal_grid). Fallback config si pas de clusterer.
        if flowsom_clusterer is not None:
            n_nodes_eff = int(flowsom_clusterer.n_nodes)
        else:
            n_nodes_eff = int(
                getattr(cfg.flowsom, "xdim", 0) * getattr(cfg.flowsom, "ydim", 0)
            )
        cluster_metrics = ClusteringMetrics(
            n_nodes=n_nodes_eff,
            n_metaclusters=int(
                max((np.unique(v).size for v in cluster_outputs.values()), default=0)
            ),
        )

        result.data = final_df
        result.output_files = output_files
        result.clustering_metrics = cluster_metrics
        result.config_snapshot = cfg.to_dict() if hasattr(cfg, "to_dict") else {}
        result.elapsed_seconds = float(time.perf_counter() - t0)
        result.gating_report = [
            {
                "mode": resolved_input.data_context_mode,
                "target_population": resolved_input.target_population,
                "workspace_path": resolved_input.gating_workspace_path,
                "n_before": int(resolved_input.input_events),
                "n_after": int(resolved_input.selected_events),
            }
        ]

        _progress(ResearchRunStep.DONE)
        log.info(
            "[RUO] Pipeline terminé: %d cellules, %d méthodes dimred, %d méthodes clustering",
            result.n_cells,
            len(dimred_methods),
            len(clustering_methods),
        )
        return result

    @staticmethod
    def _resolve_input_files(cfg: Any) -> List[Path]:
        folder = Path(getattr(cfg.paths, "patho_folder", ""))
        if not folder.exists():
            return []
        return get_fcs_files(folder)

    @staticmethod
    def _normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise les colonnes FlowKit (tuples) vers des noms de marqueurs stables."""
        normalized = df.copy()
        normalized.columns = [
            col[0] if isinstance(col, tuple) else str(col) for col in normalized.columns
        ]
        return normalized

    def _load_analysis_dataframe(self, cfg: Any, files: List[Path]) -> ResolvedInputData:
        """Résout le contexte de données (fichier complet ou population gated)."""
        target_population = (
            str(getattr(cfg, "target_population", "Root") or "Root").strip() or "Root"
        )
        gating_workspace_path = getattr(cfg, "gating_workspace_path", None)

        use_gated_context = target_population.lower() != "root"
        if use_gated_context:
            if not gating_workspace_path:
                raise ValueError(
                    "target_population != 'Root' mais aucun gating_workspace_path n'est configuré."
                )
            log.info(
                "[RUO] Extraction d'une population ciblée depuis le workspace de gating: %s",
                target_population,
            )
            return self._load_gated_population_dataframe(
                files=files,
                workspace_path=str(gating_workspace_path),
                target_population=target_population,
            )

        if not files:
            raise ValueError("Aucun fichier FCS trouvé dans le dossier d'entrée.")

        log.info("[RUO] Aucun contexte de gating spécifique, analyse du fichier complet")
        adatas = load_fcs_files(files, condition="RUO")
        if not adatas:
            raise ValueError("Chargement FCS vide: aucun fichier exploitable.")

        data_df = self._normalize_dataframe_columns(self._stack_adatas(adatas))
        marker_columns = [c for c in data_df.columns if c != "__sample__"]
        selected_events = int(len(data_df))
        input_files = [str(p.resolve()) for p in files]
        return ResolvedInputData(
            dataframe=data_df,
            marker_columns=marker_columns,
            data_context_mode="full_file",
            input_events=selected_events,
            selected_events=selected_events,
            input_fcs_files=input_files,
            gating_workspace_path=None,
            target_population="Root",
        )

    def _load_gated_population_dataframe(
        self,
        files: List[Path],
        workspace_path: str,
        target_population: str,
    ) -> ResolvedInputData:
        """Charge dynamiquement une population depuis un contexte de gating persisté."""
        workspace = Path(workspace_path)
        if not workspace.exists():
            raise FileNotFoundError(f"Workspace de gating introuvable: {workspace}")

        try:
            from src.prisma.gui.viewer.gating_engine import PrismaEngineError, PrismaFlowEngine
        except Exception as exc:
            raise RuntimeError(
                "FlowKit/PrismaFlowEngine indisponible: impossible de résoudre le contexte de gating."
            ) from exc

        runtime_files: List[Path] = list(files)
        engine = PrismaFlowEngine()
        suffix = workspace.suffix.lower()

        try:
            if suffix == ".wsp":
                if not runtime_files:
                    raise ValueError(
                        "Aucun fichier FCS d'entrée disponible pour valider le workspace WSP."
                    )
                engine.load_wsp(workspace, fcs_dir=str(runtime_files[0].parent))
            elif suffix in {".gml", ".xml"}:
                if not runtime_files:
                    raise ValueError(
                        "Aucun fichier FCS d'entrée disponible pour appliquer le GatingML."
                    )
                engine.load_fcs_batch(runtime_files, make_first_active=True)
                engine.load_gml(workspace)
            elif suffix == ".json":
                payload = json.loads(workspace.read_text(encoding="utf-8"))
                runtime_files = [Path(str(p)) for p in (payload.get("fcs_files") or []) if str(p)]
                if not runtime_files:
                    runtime_files = list(files)
                if not runtime_files:
                    raise ValueError("Contexte de gating JSON invalide: aucun fichier FCS associé.")

                missing_files = [str(p) for p in runtime_files if not p.exists()]
                if missing_files:
                    raise FileNotFoundError(
                        "Fichiers FCS absents dans le contexte de gating: "
                        + ", ".join(missing_files)
                    )

                gml_path_raw = payload.get("gatingml_path")
                gml_path = (
                    Path(str(gml_path_raw))
                    if gml_path_raw
                    else workspace.with_suffix(".gatingml.xml")
                )
                if not gml_path.is_absolute():
                    gml_path = (workspace.parent / gml_path).resolve()
                if not gml_path.exists():
                    raise FileNotFoundError(f"GatingML associé introuvable: {gml_path}")

                engine.load_fcs_batch(runtime_files, make_first_active=True)
                engine.load_gml(gml_path)
            else:
                raise ValueError(
                    "Format de workspace non supporté. Utilisez .wsp, .gml/.xml ou .json PRISMA."
                )

            engine.analyze(use_mp=False)
            sample_ids = engine.get_sample_ids()
            if not sample_ids:
                raise ValueError("Aucun sample disponible après chargement du contexte de gating.")

            gate_paths = engine.find_gate_paths(target_population)
            if not gate_paths:
                raise ValueError(f"Population cible absente dans le workspace: {target_population}")
            if len(gate_paths) > 1:
                raise ValueError(
                    "Population cible ambiguë: plusieurs chemins trouvés pour "
                    f"'{target_population}' ({gate_paths})."
                )
            gate_path = gate_paths[0]

            selected_frames: List[pd.DataFrame] = []
            input_events = 0
            selected_events = 0

            for sample_id in sample_ids:
                raw_df = engine.get_raw_dataframe(sample_id=sample_id)
                input_events += int(len(raw_df))

                gated_df = engine.get_gate_dataframe(
                    gate_name=target_population,
                    gate_path=gate_path,
                    sample_id=sample_id,
                )
                gated_df = self._normalize_dataframe_columns(gated_df)
                if gated_df.empty:
                    continue

                gated_df = gated_df.copy()
                gated_df["__sample__"] = str(sample_id)
                selected_frames.append(gated_df)
                selected_events += int(len(gated_df))

            if selected_events <= 0 or not selected_frames:
                raise ValueError(
                    f"Extraction vide: la population '{target_population}' ne contient aucun événement."
                )

            data_df = pd.concat(selected_frames, axis=0, ignore_index=True)
            marker_columns = [c for c in data_df.columns if c != "__sample__"]
            return ResolvedInputData(
                dataframe=data_df,
                marker_columns=marker_columns,
                data_context_mode="gated_population",
                input_events=input_events,
                selected_events=selected_events,
                input_fcs_files=[str(p.resolve()) for p in runtime_files],
                gating_workspace_path=str(workspace.resolve()),
                target_population=target_population,
            )
        except PrismaEngineError as exc:
            raise RuntimeError(
                f"Erreur FlowKit lors de la résolution du contexte de gating: {exc}"
            ) from exc

    @staticmethod
    def _stack_adatas(adatas: List[Any]) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for adata in adatas:
            df = adata.to_df()
            sample_name = str(adata.obs.get("file_origin", "unknown").iloc[0])
            df = df.copy()
            df["__sample__"] = sample_name
            frames.append(df)
        return pd.concat(frames, axis=0, ignore_index=True)

    @staticmethod
    def _apply_basic_preprocessing(
        matrix: np.ndarray, cfg: Any, var_names: Optional[List[str]] = None
    ) -> np.ndarray:
        x = np.asarray(matrix, dtype=np.float32)

        # Priorité 1 : transformations par colonne (popup pré-traitement).
        per_col = dict(getattr(cfg.transform, "per_column_specs", {}) or {})
        if per_col and var_names is not None:
            from prisma.core.transformers import DataTransformer

            x = DataTransformer.apply_per_column_transforms(x, list(var_names), per_col)
            log.info("[RUO] Pré-traitement par colonne appliqué (%d colonnes spécifiées)", len(per_col))
        else:
            # Priorité 2 : transformation globale (method commun à tous les canaux fluo).
            method = str(getattr(cfg.transform, "method", "none")).lower()
            if method in ("arcsinh", "logicle", "log10", "log") and method != "none":
                from prisma.core.transformers import DataTransformer

                x = DataTransformer.apply(
                    x,
                    method=method,
                    cofactor=float(getattr(cfg.transform, "cofactor", 5.0) or 5.0),
                    var_names=list(var_names) if var_names is not None else None,
                    apply_to_scatter=bool(getattr(cfg.transform, "apply_to_scatter", False)),
                ).astype(np.float32)

        norm = str(getattr(cfg.normalize, "method", "none")).lower()
        if norm == "zscore":
            mu = np.nanmean(x, axis=0, keepdims=True)
            sigma = np.nanstd(x, axis=0, keepdims=True)
            x = (x - mu) / np.where(sigma <= 1e-8, 1.0, sigma)
        elif norm == "minmax":
            mn = np.nanmin(x, axis=0, keepdims=True)
            mx = np.nanmax(x, axis=0, keepdims=True)
            x = (x - mn) / np.where((mx - mn) <= 1e-8, 1.0, (mx - mn))
        return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=0.0).astype(np.float32)

    @staticmethod
    def _get_methods(
        cfg: Any,
        list_attr: str,
        scalar_attr: str,
        default: List[str],
    ) -> List[str]:
        values = list(getattr(cfg, list_attr, []) or [])
        if not values:
            values = [str(getattr(cfg, scalar_attr, default[0]))]
        return [str(v).lower().strip() for v in values if str(v).strip()]

    @staticmethod
    def _map_dimred_name(name: str) -> str:
        normalized = name.lower().strip()
        return "tsne" if normalized == "visne" else normalized

    @staticmethod
    def _map_clustering_name(name: str) -> str:
        return name.lower().strip()

    @staticmethod
    def _build_dimred_params(
        strategy_name: str, cfg: Any, wizard_cfg: Dict[str, Any]
    ) -> DimReducParams:
        params_map = dict(wizard_cfg.get("dimred_params", {}) or {})
        raw = dict(params_map.get("visne" if strategy_name == "tsne" else strategy_name, {}) or {})

        seed_default = int(getattr(cfg.flowsom, "seed", 42))
        n_components = int(raw.get("n_components", 2) or 2)

        if strategy_name == "umap":
            from prisma.strategies.umap_strategy import UMAPParams

            return UMAPParams(
                n_components=n_components,
                seed=int(raw.get("seed", seed_default)),
                n_neighbors=int(raw.get("n_neighbors", 15)),
                min_dist=float(raw.get("min_dist", 0.1)),
                metric=str(raw.get("metric", "euclidean")),
                extra={"max_events": int(raw.get("max_events", 0))},
            )

        if strategy_name == "tsne":
            from prisma.strategies.tsne_strategy import TSNEParams

            return TSNEParams(
                n_components=2,
                seed=int(raw.get("seed", seed_default)),
                perplexity=float(raw.get("perplexity", 30.0)),
                n_iter=int(raw.get("iterations", 1000)),
                learning_rate=float(raw.get("learning_rate", 200.0)),
                max_events=int(raw.get("max_events", 50_000)),
            )

        if strategy_name == "phate":
            from prisma.strategies.phate_strategy import PHATEParams

            # PHATEForm expose spin_k (knn) et spin_t (decay=diffusion steps)
            return PHATEParams(
                n_components=int(raw.get("n_components", n_components)),
                seed=seed_default,
                knn=int(raw.get("k", 10)),
                decay=int(raw.get("a", 40)),  # spin_a → alpha/decay
            )

        return DimReducParams(n_components=n_components, seed=seed_default, extra=raw)

    @staticmethod
    def _build_cluster_params(
        strategy_name: str, cfg: Any, wizard_cfg: Dict[str, Any]
    ) -> ClusterParams:
        params_map = dict(wizard_cfg.get("clustering_params", {}) or {})
        raw = dict(params_map.get(strategy_name, {}) or {})
        seed_default = int(getattr(cfg.flowsom, "seed", 42))

        if strategy_name == "flowsom":
            from prisma.strategies.flowsom_strategy import FlowSOMParams

            # rlen peut être "auto" dans la config YAML — convertir proprement
            rlen_raw = raw.get("rlen", getattr(cfg.flowsom, "rlen", 10))
            try:
                rlen_val = int(rlen_raw)
            except (TypeError, ValueError):
                rlen_val = 10  # "auto" → fallback 10

            return FlowSOMParams(
                seed=int(raw.get("seed", seed_default)),
                xdim=int(raw.get("xdim", getattr(cfg.flowsom, "xdim", 10))),
                ydim=int(raw.get("ydim", getattr(cfg.flowsom, "ydim", 10))),
                rlen=rlen_val,
                n_metaclusters=int(raw.get("k", getattr(cfg.flowsom, "n_metaclusters", 20))),
            )

        if strategy_name == "flowsom_like":
            from prisma.strategies.flowsom_like_strategy import FlowSOMlikeParams

            return FlowSOMlikeParams(
                seed=seed_default,
                xdim=int(raw.get("xdim", getattr(cfg.flowsom, "xdim", 10))),
                ydim=int(raw.get("ydim", getattr(cfg.flowsom, "ydim", 10))),
                n_metaclusters=int(raw.get("k", getattr(cfg.flowsom, "n_metaclusters", 20))),
                auto_metaclusters=bool(raw.get("auto_k", False)),
            )

        if strategy_name == "phenograph":
            return ClusterParams(
                n_clusters=None,
                seed=seed_default,
                extra={
                    "k": int(raw.get("k", 30)),
                    "metric": str(raw.get("metric", "euclidean")),
                    "clustering_algo": str(raw.get("clustering", "louvain")),
                    "seed": seed_default,
                },
            )

        if strategy_name == "hdbscan":
            return ClusterParams(
                n_clusters=None,
                seed=seed_default,
                extra={
                    "min_cluster_size": int(raw.get("min_cluster_size", 50)),
                    "min_samples": int(raw.get("min_samples", 10)),
                    "cluster_selection_epsilon": float(raw.get("cluster_eps", 0.0)),
                    "metric": str(raw.get("metric", "euclidean")),
                    "allow_noise": bool(raw.get("allow_noise", True)),
                },
            )

        if strategy_name == "parc":
            return ClusterParams(
                n_clusters=None,
                seed=seed_default,
                extra={
                    "dist_std_local": float(raw.get("dist_std", 2.0)),
                    "jac_std_global": float(raw.get("jac_std", 0.15)),
                    "l2_std_factor": float(raw.get("jac_std", 0.15)),
                    "knn": int(raw.get("knn", 30)),
                    "n_iter_leiden": int(raw.get("n_iter", 5)),
                },
            )

        if strategy_name == "spade":
            return ClusterParams(
                n_clusters=int(raw.get("k", 200)),
                seed=seed_default,
                extra={
                    "density_factor": float(raw.get("density_factor", 0.1)),
                    "max_cells": int(raw.get("max_cells", 500_000)),
                    "layout": str(raw.get("layout", "kruskal")),
                },
            )

        return ClusterParams(
            n_clusters=int(raw.get("k", getattr(cfg.flowsom, "n_metaclusters", 20))),
            seed=seed_default,
            extra=raw,
        )

    def _build_output_dataframe(
        self,
        data_df: pd.DataFrame,
        marker_columns: List[str],
        dimred_outputs: Dict[str, np.ndarray],
        cluster_outputs: Dict[str, np.ndarray],
        flowsom_clusterer: Optional[Any] = None,
    ) -> pd.DataFrame:
        out = data_df.copy()
        for method, emb in dimred_outputs.items():
            arr = np.asarray(emb)
            if arr.ndim == 1:
                out[f"{method}_1"] = arr
                continue
            for i in range(arr.shape[1]):
                out[f"{method}_{i + 1}"] = arr[:, i]

        for method, labels in cluster_outputs.items():
            out[f"cluster_{method}"] = np.asarray(labels, dtype=np.int32)

        # Enrichissement FlowSOM : reproduit les colonnes de la pipeline de
        # référence (FlowSOM_cluster, FlowSOM_metacluster 1-based + xGrid/yGrid
        # grille SOM et xNodes/yNodes MST, avec jitter circulaire + size).
        if flowsom_clusterer is not None:
            self._add_flowsom_columns(out, flowsom_clusterer)

        # Le FCS export requiert des colonnes numériques ; __sample__ reste pour CSV/traçabilité.
        for col in marker_columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    @staticmethod
    def _fit_flowsom_clusterer(
        data_matrix: np.ndarray,
        params: Any,
        cfg: Any,
        marker_names: Optional[List[str]] = None,
    ) -> Any:
        """Entraîne un FlowSOMClusterer (grille SOM + MST) et l'expose pour l'export.

        Utilise le clusterer de bas niveau (et non la stratégie générique) afin
        d'accéder à node_assignments_, metacluster_map_, get_grid_coords() et
        get_layout_coords() — nécessaires aux colonnes FCS FlowSOM_*/xGrid/yGrid/
        xNodes/yNodes, à l'identique de la pipeline de référence.

        marker_names est transmis au fit : requis pour les visualisations natives
        par marqueur (fs.pl.plot_marker) et pour new_data()/subset().
        """
        from prisma.core.clustering import FlowSOMClusterer

        use_gpu = bool(getattr(cfg.flowsom, "use_gpu", True))
        clusterer = FlowSOMClusterer(
            xdim=int(getattr(params, "xdim", 10)),
            ydim=int(getattr(params, "ydim", 10)),
            n_metaclusters=int(getattr(params, "n_metaclusters", 20)),
            rlen=getattr(params, "rlen", "auto"),
            seed=int(getattr(params, "seed", 42)),
            use_gpu=use_gpu,
        )
        clusterer.fit(
            np.asarray(data_matrix, dtype=np.float32),
            marker_names=marker_names,
        )
        return clusterer

    @staticmethod
    def _auto_select_k(
        data_matrix: np.ndarray, params: Any, cfg: Any
    ) -> Optional[Dict[str, Any]]:
        """Auto-sélection du nombre de métaclusters (3 phases) + scores.

        Délègue à ``find_optimal_clusters_with_scores`` (silhouette codebook →
        stabilité bootstrap → score composite). Les paramètres proviennent de
        ``cfg.auto_clustering``. Retourne le dict complet (best_k + métriques)
        ou None en cas d'échec (le pipeline retombe sur le k configuré).
        """
        try:
            from prisma.core.metaclustering import find_optimal_clusters_with_scores
        except Exception as exc:  # noqa: BLE001
            log.warning("[RUO] Auto-clustering indisponible : %s", exc)
            return None

        ac = cfg.auto_clustering
        try:
            return find_optimal_clusters_with_scores(
                np.asarray(data_matrix, dtype=np.float32),
                min_clusters=int(getattr(ac, "min_clusters", 5)),
                max_clusters=int(getattr(ac, "max_clusters", 20)),
                n_bootstrap=int(getattr(ac, "n_bootstrap", 10)),
                sample_size_bootstrap=int(getattr(ac, "sample_size_bootstrap", 20000)),
                min_stability_threshold=float(getattr(ac, "min_stability_threshold", 0.75)),
                weight_stability=float(getattr(ac, "weight_stability", 0.65)),
                weight_silhouette=float(getattr(ac, "weight_silhouette", 0.35)),
                xdim=int(getattr(params, "xdim", 10)),
                ydim=int(getattr(params, "ydim", 10)),
                seed=int(getattr(params, "seed", 42)),
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[RUO] Auto-clustering échoué : %s", exc)
            return None

    @staticmethod
    def _plot_auto_cluster_metrics(
        auto_result: Dict[str, Any],
        output_dir: Path,
        cfg: Any,
        export_cfg: Dict[str, Any],
    ) -> Optional[str]:
        """Trace les métriques d'auto-clustering (silhouette/stabilité/composite).

        3 panneaux avec marquage du k optimal, via ``plot_optimization_results``.
        """
        try:
            from prisma.visualization.flowsom_plots import plot_optimization_results
        except Exception as exc:  # noqa: BLE001
            log.warning("[RUO] Graphique auto-clustering indisponible : %s", exc)
            return None

        fmt = str(export_cfg.get("flowsom_plot_format", "png")).lstrip(".")
        ac = cfg.auto_clustering
        out_path = output_dir / "flowsom_plots" / f"auto_clustering_metrics.{fmt}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig = plot_optimization_results(
                results_df=auto_result["results_df"],
                best_k=int(auto_result["best_k"]),
                stability_results=auto_result.get("stability_results"),
                w_stability=float(getattr(ac, "weight_stability", 0.65)),
                w_silhouette=float(getattr(ac, "weight_silhouette", 0.35)),
                min_stability_threshold=float(getattr(ac, "min_stability_threshold", 0.75)),
                output_path=out_path,
            )
            if fig is not None and out_path.exists():
                log.info("[RUO] Graphique métriques auto-clustering : %s", out_path)
                return str(out_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("[RUO] Échec graphique auto-clustering : %s", exc)
        return None

    @staticmethod
    def _circular_jitter(
        n_points: int,
        cluster_ids: np.ndarray,
        node_sizes: np.ndarray,
        max_radius: float = 0.45,
        min_radius: float = 0.1,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Jitter circulaire style FlowSOM R (rayon fonction de la taille du node).

        sqrt(u) garantit une distribution uniforme dans le disque. Identique à
        la pipeline de référence pour des sorties superposables.
        """
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0, 2 * np.pi, n_points)
        u = rng.uniform(0, 1, n_points)
        max_size = node_sizes.max() if node_sizes.max() > 0 else 1.0
        radii = min_radius + (max_radius - min_radius) * np.sqrt(
            node_sizes[cluster_ids.astype(int)] / max_size
        )
        r = np.sqrt(u) * radii
        return (r * np.cos(theta)).astype(np.float32), (r * np.sin(theta)).astype(np.float32)

    def _add_flowsom_columns(self, out: pd.DataFrame, clusterer: Any) -> None:
        """Ajoute les colonnes FlowSOM (cluster/metacluster/grille/MST) au DataFrame.

        Reproduit `_build_fcs_dataframe` de la pipeline de référence :
          - FlowSOM_cluster      : node SOM 1-based (Kaluza ≥ 1)
          - FlowSOM_metacluster  : métacluster 1-based
          - xGrid, yGrid         : grille SOM + jitter circulaire, min = 1
          - xNodes, yNodes       : layout MST + jitter circulaire, min = 1
          - size                 : nombre de cellules du node
        """
        node_assign = getattr(clusterer, "node_assignments_", None)
        if node_assign is None:
            log.warning("[RUO] FlowSOM sans node_assignments_ — colonnes FCS topologiques ignorées")
            return

        cl_int = np.asarray(node_assign, dtype=int)
        n_cells = cl_int.shape[0]

        # Métacluster par cellule : via metacluster_map_ (node→meta) si dispo,
        # sinon metacluster_assignments_ (cell→meta) directement.
        mc_map = getattr(clusterer, "metacluster_map_", None)
        if mc_map is not None:
            mc_per_cell = np.asarray(mc_map, dtype=int)[cl_int]
        else:
            mc_per_cell = np.asarray(
                getattr(clusterer, "metacluster_assignments_"), dtype=int
            )

        node_sizes = clusterer.get_node_sizes()

        # — Grille SOM + jitter (min = 1, comme le monolithe de référence) —
        grid_coords = clusterer.get_grid_coords()  # (n_nodes, 2)
        xg = grid_coords[cl_int, 0].astype(np.float32)
        yg = grid_coords[cl_int, 1].astype(np.float32)
        jx, jy = self._circular_jitter(n_cells, cl_int, node_sizes, 0.45, 0.1)
        xg = (xg + jx); yg = (yg + jy)
        xGrid = xg - xg.min() + 1.0
        yGrid = yg - yg.min() + 1.0

        # — Layout MST + jitter (échelle relative à la grille) —
        layout = clusterer.get_layout_coords()  # (n_nodes, 2)
        x_ptp = float(layout[:, 0].max() - layout[:, 0].min()) or 1.0
        y_ptp = float(layout[:, 1].max() - layout[:, 1].min()) or 1.0
        mst_scale = min(x_ptp, y_ptp) / (clusterer.xdim * 2)
        xn = layout[cl_int, 0].astype(np.float32)
        yn = layout[cl_int, 1].astype(np.float32)
        mjx, mjy = self._circular_jitter(
            n_cells, cl_int, node_sizes, mst_scale * 0.8, mst_scale * 0.2
        )
        xn = (xn + mjx); yn = (yn + mjy)
        xNodes = xn - xn.min() + 1.0
        yNodes = yn - yn.min() + 1.0

        out["FlowSOM_cluster"] = (cl_int + 1).astype(np.float32)
        out["FlowSOM_metacluster"] = (mc_per_cell + 1).astype(np.float32)
        out["xGrid"] = xGrid.astype(np.float32)
        out["yGrid"] = yGrid.astype(np.float32)
        out["xNodes"] = xNodes.astype(np.float32)
        out["yNodes"] = yNodes.astype(np.float32)
        out["size"] = node_sizes[cl_int].astype(np.float32)
        log.info(
            "[RUO] Colonnes FlowSOM ajoutées — xGrid[%.2f-%.2f] yGrid[%.2f-%.2f] "
            "xNodes[%.2f-%.2f] yNodes[%.2f-%.2f]",
            xGrid.min(), xGrid.max(), yGrid.min(), yGrid.max(),
            xNodes.min(), xNodes.max(), yNodes.min(), yNodes.max(),
        )

    @staticmethod
    def _generate_flowsom_native_plots(
        clusterer: Any,
        output_dir: Path,
        marker_columns: List[str],
        export_cfg: Dict[str, Any],
    ) -> Dict[str, str]:
        """Génère les visualisations natives FlowSOM (fs.pl.*) dans un sous-dossier.

        Produit star charts (MST + grille), cartes par marqueur, numéros de
        nœuds/métaclusters et planche de synthèse — au format configuré.
        Sans effet (et sans erreur) si le backend n'est pas le FlowSOM natif.
        """
        plots: Dict[str, str] = {}
        try:
            from prisma.visualization.flowsom_native_plots import generate_all_native_plots
        except Exception as exc:  # noqa: BLE001
            log.warning("[RUO] Module de visualisation FlowSOM indisponible : %s", exc)
            return plots

        fmt = str(export_cfg.get("flowsom_plot_format", "png")).lstrip(".")
        dpi = int(export_cfg.get("flowsom_plot_dpi", 150))
        # Limiter les cartes par marqueur pour ne pas saturer (configurable).
        max_marker_plots = int(export_cfg.get("flowsom_max_marker_plots", 12))
        markers_subset = marker_columns[:max_marker_plots] if max_marker_plots > 0 else None

        plots_dir = output_dir / "flowsom_plots"
        try:
            produced = generate_all_native_plots(
                clusterer,
                plots_dir,
                marker_names=markers_subset,
                fmt=fmt,
                dpi=dpi,
            )
            # Préfixer les clés pour le manifeste output_files.
            for key, path in produced.items():
                plots[f"flowsom_plot_{key}"] = path
            log.info("[RUO] %d visualisations FlowSOM natives générées.", len(produced))
        except Exception as exc:  # noqa: BLE001
            log.warning("[RUO] Génération des visualisations FlowSOM échouée : %s", exc)
        return plots

    @staticmethod
    def _coerce_dimreduc_params(params: DimReducParams | ClusterParams) -> DimReducParams:
        if isinstance(params, DimReducParams):
            return params
        return DimReducParams(seed=params.seed, n_jobs=params.n_jobs, extra=dict(params.extra))

    @staticmethod
    def _coerce_cluster_params(params: DimReducParams | ClusterParams) -> ClusterParams:
        if isinstance(params, ClusterParams):
            return params
        return ClusterParams(seed=params.seed, n_jobs=params.n_jobs, extra=dict(params.extra))
