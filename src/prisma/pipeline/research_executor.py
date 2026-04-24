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

from src.io.fcs_reader import get_fcs_files, load_fcs_files
from src.io.fcs_writer import export_to_fcs_kaluza
from src.models.pipeline_result import ClusteringMetrics, PipelineResult

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
        if not files:
            raise ValueError("Aucun fichier FCS trouvé dans le dossier d'entrée.")

        _progress(ResearchRunStep.LOADING)
        adatas = load_fcs_files(files, condition="RUO")
        if not adatas:
            raise ValueError("Chargement FCS vide: aucun fichier exploitable.")

        data_df = self._stack_adatas(adatas)
        marker_columns = [c for c in data_df.columns if c != "__sample__"]
        log.info(
            "[RUO] Chargement OK: %d cellules, %d marqueurs", len(data_df), len(marker_columns)
        )

        data_matrix = data_df[marker_columns].to_numpy(dtype=np.float32, copy=True)
        data_matrix = self._apply_basic_preprocessing(data_matrix, cfg)

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
                embedding = self.fit_transform(data_matrix, strategy_name, params)
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
        for method in clustering_methods:
            strategy_name = self._map_clustering_name(method)
            params = self._build_cluster_params(strategy_name, cfg, wizard_cfg)
            log.info("[RUO] Clustering -> %s", strategy_name)
            try:
                labels = self.fit_predict(data_matrix, strategy_name, params)
                cluster_outputs[method] = np.asarray(labels)
            except Exception as exc:
                log.warning("[RUO] Clustering '%s' ignoré: %s", strategy_name, exc)

        if not cluster_outputs:
            raise RuntimeError(
                "Aucune stratégie de clustering exécutable n'est disponible pour la sélection courante."
            )

        _progress(ResearchRunStep.EXPORT)
        final_df = self._build_output_dataframe(
            data_df, marker_columns, dimred_outputs, cluster_outputs
        )
        output_dir = Path(getattr(cfg.paths, "output_dir", "Results"))
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "ruo_results.csv"
        final_df.to_csv(csv_path, index=False)

        metadata_path = output_dir / "ruo_run_metadata.json"
        metadata_payload = {
            "qc_method": qc_method,
            "dimred_methods": dimred_methods,
            "clustering_methods": clustering_methods,
            "n_cells": int(len(final_df)),
            "n_markers": int(len(marker_columns)),
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

        cluster_metrics = ClusteringMetrics(
            n_nodes=int(getattr(cfg.flowsom, "xdim", 0) * getattr(cfg.flowsom, "ydim", 0)),
            n_metaclusters=int(
                max((np.unique(v).size for v in cluster_outputs.values()), default=0)
            ),
        )

        result.data = final_df
        result.output_files = output_files
        result.clustering_metrics = cluster_metrics
        result.config_snapshot = cfg.to_dict() if hasattr(cfg, "to_dict") else {}
        result.elapsed_seconds = float(time.perf_counter() - t0)

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
    def _apply_basic_preprocessing(matrix: np.ndarray, cfg: Any) -> np.ndarray:
        x = np.asarray(matrix, dtype=np.float32)
        method = str(getattr(cfg.transform, "method", "none")).lower()
        if method == "arcsinh":
            cofactor = float(getattr(cfg.transform, "cofactor", 5.0) or 5.0)
            x = np.arcsinh(x / max(cofactor, 1e-6)).astype(np.float32)
        elif method == "log10":
            x = np.log10(np.clip(x, a_min=1e-6, a_max=None)).astype(np.float32)

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

    @staticmethod
    def _build_output_dataframe(
        data_df: pd.DataFrame,
        marker_columns: List[str],
        dimred_outputs: Dict[str, np.ndarray],
        cluster_outputs: Dict[str, np.ndarray],
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

        # Le FCS export requiert des colonnes numériques ; __sample__ reste pour CSV/traçabilité.
        for col in marker_columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

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
