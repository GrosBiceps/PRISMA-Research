"""
pipeline_executor.py — Exécuteur principal du pipeline FlowSOM.

Orchestre le pipeline de bout en bout:
  1. Chargement des fichiers FCS (sain + pathologique)
  2. Prétraitement de chaque échantillon
  3. Clustering FlowSOM (avec auto-clustering optionnel)
  4. Construction du DataFrame cellulaire
  5. Exports (FCS, CSV, JSON, plots)
  6. Retour d'un PipelineResult structuré

Point d'entrée: FlowSOMPipeline.execute(config)
"""

from __future__ import annotations

import dataclasses
import enum
import time
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from joblib import dump as _joblib_dump, load as _joblib_load

    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

from flowsom_pipeline_pro.config.pipeline_config import PipelineConfig
from flowsom_pipeline_pro.src.prisma.core.models_legacy.sample import FlowSample
from flowsom_pipeline_pro.src.prisma.core.models_legacy.pipeline_result import (
    PipelineResult,
    ClusteringMetrics,
)
from flowsom_pipeline_pro.src.prisma.io.fcs_reader import get_fcs_files, load_as_flow_samples
from flowsom_pipeline_pro.src.prisma.core.clustering import FlowSOMClusterer
from flowsom_pipeline_pro.src.prisma.services.preprocessing_service import (
    preprocess_all_samples,
    preprocess_combined,
)
from flowsom_pipeline_pro.src.prisma.services.clustering_service import (
    run_clustering,
    build_cells_dataframe,
    stack_samples,
    stack_raw_markers,
)
from flowsom_pipeline_pro.src.prisma.services.export_service import ExportService
from flowsom_pipeline_pro.src.prisma.utils.logger import GatingEvent, GatingLogger, get_logger
from flowsom_pipeline_pro.src.prisma.core.models_legacy.gate_result import (
    gating_reports,
    gating_log_entries,
)
from flowsom_pipeline_pro.src.prisma.core.auto_gating import (
    ransac_scatter_data,
    singlets_summary_per_file,
)
from flowsom_pipeline_pro.src.prisma.pipeline.plotting_worker import (
    PlottingTask,
    PlottingWorker,
)

_logger = get_logger("pipeline.executor")


@dataclasses.dataclass
class GatingResult:
    """
    Résultat de la phase de gating CD45 (execute_gating).

    Contient tout ce dont execute_clustering() a besoin pour reprendre le
    pipeline après validation intermédiaire par l'utilisateur.
    """

    success: bool
    error: Optional[str] = None
    processed_samples: List[Any] = dataclasses.field(default_factory=list)
    gating_figures: Dict[str, Any] = dataclasses.field(default_factory=dict)
    gating_key: str = ""
    output_dir: str = ""
    n_total_raw: int = 0
    timestamp: str = ""
    checkpoint_manager: Optional[Any] = None
    input_files: List[str] = dataclasses.field(default_factory=list)
    gating_events: List[Dict[str, Any]] = dataclasses.field(default_factory=list)


class PipelineStep(enum.IntEnum):
    """
    Étapes du pipeline FlowSOM avec leur pourcentage d'avancement associé.

    Utilisé par FlowSOMPipeline.execute() pour émettre des callbacks de
    progression propres, sans recourir au scraping des messages de log.
    """

    START = 5
    LOADING = 8
    PREPROCESSING = 20
    GATING_DONE = 30
    CLUSTERING_START = 40
    CLUSTERING_DONE = 60
    DATAFRAME = 65
    UMAP = 70
    PLOTS_MST = 76
    PLOTS_GRID = 80
    PLOTS_RADAR = 85
    EXPORTS = 90
    REPORT = 93
    MAPPING = 95
    DONE = 98


def _safe_plot(name: str, figures: dict, key: str, fn, *args, **kwargs):
    """
    Appelle fn(*args, **kwargs) et stocke le résultat dans figures[key].
    En cas d'échec, log un warning non-bloquant. Retourne le résultat ou None.
    """
    try:
        result = fn(*args, **kwargs)
        if result is not None:
            figures[key] = result
        _logger.info("%s sauvegardé.", name)
        return result
    except Exception as _e:
        _logger.warning("%s échoué (non bloquant): %s", name, _e)
        return None


def _stable_hash(payload: Dict[str, Any]) -> str:
    """Hash SHA256 stable d'un payload JSON-serializable."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _checkpoint_code_version() -> str:
    """
    Empreinte SHA256 (16 chars) des modules critiques pour les checkpoints SOM.

    Invalide les checkpoints de clustering si le code du SOM, du metaclustering
    ou des services change. Calculé une seule fois à l'import du module.
    """
    _SRC = Path(__file__).parent.parent
    candidates = [
        _SRC / "core" / "clustering.py",
        _SRC / "core" / "metaclustering.py",
        _SRC / "services" / "clustering_service.py",
        _SRC / "analysis" / "mrd_calculator.py",
    ]
    h = hashlib.sha256()
    found_any = False
    for path in candidates:
        if path.exists():
            try:
                h.update(path.read_bytes())
                found_any = True
            except OSError:
                pass
    return h.hexdigest()[:16] if found_any else "unknown"


_CHECKPOINT_CODE_VERSION: str = _checkpoint_code_version()


class PipelineCheckpointManager:
    """Gestionnaire de checkpoints disque pour accélérer les reruns CPU."""

    def __init__(self, config: PipelineConfig) -> None:
        ckpt_cfg = dict(getattr(config, "_extra", {}).get("checkpointing", {}) or {})
        self.enabled = bool(ckpt_cfg.get("enabled", True)) and _JOBLIB_AVAILABLE
        self.base_dir = Path(ckpt_cfg.get("cache_dir", Path(config.paths.output_dir) / "cache"))
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if not _JOBLIB_AVAILABLE:
            _logger.warning("joblib absent: checkpointing désactivé.")

    def make_key(self, stage: str, payload: Dict[str, Any]) -> str:
        # Intègre la version du code pour invalider les checkpoints si le code change
        payload_with_version = {**payload, "_code_version": _CHECKPOINT_CODE_VERSION}
        return f"{stage}_{_stable_hash(payload_with_version)}"

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        path = self.base_dir / f"{key}.joblib"
        if not path.exists():
            return None
        try:
            data = _joblib_load(path)
            _logger.info("Checkpoint chargé: %s", path.name)
            return data
        except Exception as exc:
            _logger.warning("Lecture checkpoint échouée (%s): %s", path.name, exc)
            return None

    def save(self, key: str, data: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self.base_dir / f"{key}.joblib"
        try:
            _joblib_dump(data, path, compress=0)
            _logger.info("Checkpoint sauvegardé: %s", path.name)
        except Exception as exc:
            _logger.warning("Sauvegarde checkpoint échouée (%s): %s", path.name, exc)


def _serialize_clusterer_state(clusterer: FlowSOMClusterer) -> Dict[str, Any]:
    """Extrait l'état minimal nécessaire pour restaurer un clusterer post-SOM."""
    return {
        "xdim": int(clusterer.xdim),
        "ydim": int(clusterer.ydim),
        "n_metaclusters": int(clusterer.n_metaclusters),
        "rlen": clusterer.rlen,
        "seed": int(clusterer.seed),
        "use_gpu": bool(clusterer.use_gpu),
        "learning_rate": float(clusterer.learning_rate),
        "sigma": float(clusterer.sigma),
        "node_assignments": np.asarray(clusterer.node_assignments_)
        if clusterer.node_assignments_ is not None
        else None,
        "metacluster_map": np.asarray(clusterer.metacluster_map_)
        if clusterer.metacluster_map_ is not None
        else None,
        "metacluster_assignments": np.asarray(clusterer.metacluster_assignments_)
        if clusterer.metacluster_assignments_ is not None
        else None,
        "mst_layout": np.asarray(clusterer._mst_layout_)
        if clusterer._mst_layout_ is not None
        else None,
    }


def _deserialize_clusterer_state(state: Dict[str, Any]) -> FlowSOMClusterer:
    """Reconstruit un clusterer léger depuis un checkpoint post-SOM."""
    clusterer = FlowSOMClusterer(
        xdim=state["xdim"],
        ydim=state["ydim"],
        n_metaclusters=state["n_metaclusters"],
        rlen=state["rlen"],
        seed=state["seed"],
        use_gpu=state["use_gpu"],
        learning_rate=state["learning_rate"],
        sigma=state["sigma"],
    )
    clusterer.node_assignments_ = state.get("node_assignments")
    clusterer.metacluster_map_ = state.get("metacluster_map")
    clusterer.metacluster_assignments_ = state.get("metacluster_assignments")
    clusterer._mst_layout_ = state.get("mst_layout")
    clusterer._fsom_model = None
    clusterer.used_gpu_ = False
    return clusterer


def _restore_gating_logger(events: List[Dict[str, Any]]) -> GatingLogger:
    """Reconstruit un GatingLogger depuis une liste d'événements sérialisés."""
    logger = GatingLogger()
    restored: List[GatingEvent] = []
    for item in events:
        extra = {
            k: v
            for k, v in item.items()
            if k
            not in {
                "file",
                "gate_name",
                "n_before",
                "n_after",
                "n_excluded",
                "pct_kept",
                "timestamp",
                "warnings",
            }
        }
        restored.append(
            GatingEvent(
                file=item.get("file", "unknown"),
                gate_name=item.get("gate_name", "unknown"),
                n_before=int(item.get("n_before", 0)),
                n_after=int(item.get("n_after", 0)),
                pct_kept=float(item.get("pct_kept", 0.0)),
                timestamp=float(item.get("timestamp", time.time())),
                warnings=list(item.get("warnings", [])),
                extra=extra,
            )
        )
    logger._events = restored
    return logger


class FlowSOMPipeline:
    """
    Pipeline FlowSOM de niveau production.

    Encapsule l'exécution complète du pipeline en un seul objet.

    Usage:
        config = PipelineConfig.from_yaml("config.yaml")
        pipeline = FlowSOMPipeline(config)
        result = pipeline.execute()
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._gating_logger = GatingLogger()
        self._result: Optional[PipelineResult] = None

    @property
    def result(self) -> Optional[PipelineResult]:
        """Accès au résultat du dernier execute()."""
        return self._result

    def _resolve_input_fcs_files(self) -> List[str]:
        """Retourne la liste ordonnée des fichiers FCS qui alimentent le run."""
        config = self.config
        files: List[Path] = []

        healthy_folder = Path(getattr(config.paths, "healthy_folder", "") or "")
        if healthy_folder.exists():
            files.extend(get_fcs_files(healthy_folder))

        single_patho = getattr(config.paths, "patho_single_file", None)
        patho_folder = Path(getattr(config.paths, "patho_folder", "") or "")
        if single_patho and Path(single_patho).is_file():
            files.append(Path(single_patho))
        elif patho_folder.exists():
            files.extend(get_fcs_files(patho_folder))

        if not files:
            data_folder = Path(getattr(config.paths, "data_folder", "") or "")
            if data_folder.exists():
                files.extend(get_fcs_files(data_folder))

        # Normalise + déduplique en conservant l'ordre
        dedup: List[str] = []
        seen = set()
        for p in files:
            s = str(p)
            if s not in seen:
                seen.add(s)
                dedup.append(s)
        return dedup

    def _build_file_fingerprint(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Construit une signature stable des fichiers d'entrée."""
        fingerprint: List[Dict[str, Any]] = []
        for fp in file_paths:
            p = Path(fp)
            stat = p.stat() if p.exists() else None
            fingerprint.append(
                {
                    "path": str(p),
                    "size": int(stat.st_size) if stat else -1,
                    "mtime_ns": int(stat.st_mtime_ns) if stat else -1,
                }
            )
        return fingerprint

    @staticmethod
    def _cfg_to_dict(obj: object) -> object:
        """Sérialise un objet config en dict de manière sûre.

        Utilise dataclasses.asdict() au lieu de vars() pour être compatible
        avec les dataclasses utilisant __slots__ (vars() lèverait TypeError).
        Fallback sur str() pour les types non-dataclass.
        """
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if hasattr(obj, "__dict__"):
            return vars(obj)
        return str(obj)

    def _build_gating_cache_key(
        self,
        cache: PipelineCheckpointManager,
        input_files: List[str],
    ) -> str:
        cfg = self.config
        _d = self._cfg_to_dict
        payload = {
            "stage": "post_gating",
            "files": self._build_file_fingerprint(input_files),
            "pregate": _d(cfg.pregate),
            "transform": _d(cfg.transform),
            "normalize": _d(cfg.normalize),
            "markers": _d(cfg.markers),
            "seed": int(cfg.flowsom.seed),
            "condition_labels_v2": "Sain/Pathologique",
        }
        return cache.make_key("post_gating", payload)

    def _build_som_cache_key(
        self,
        cache: PipelineCheckpointManager,
        gating_key: str,
    ) -> str:
        cfg = self.config
        _d = self._cfg_to_dict
        payload = {
            "stage": "post_som",
            "gating_key": gating_key,
            "flowsom": _d(cfg.flowsom),
            "gpu": _d(cfg.gpu),
            "auto_clustering": _d(cfg.auto_clustering),
            "stratified_downsampling": _d(cfg.stratified_downsampling),
            "markers": _d(cfg.markers),
            "downsampling": _d(cfg.downsampling),
        }
        return cache.make_key("post_som", payload)

    @staticmethod
    def _enqueue_plot(
        worker: Optional[PlottingWorker],
        name: str,
        key: str,
        target: str,
        module: str,
        function: str,
        *args,
        **kwargs,
    ) -> None:
        """Ajoute une tâche au worker de plotting (fallback sync si worker absent)."""
        if worker is None:
            return
        worker.submit(
            PlottingTask(
                name=name,
                key=key,
                target=target,
                module=module,
                function=function,
                args=args,
                kwargs=kwargs,
            )
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Exécution en deux étapes (pour validation intermédiaire par l'UI)
    # ──────────────────────────────────────────────────────────────────────────

    def execute_gating(
        self,
        progress_callback: Optional[Callable[[PipelineStep, int], None]] = None,
    ) -> "GatingResult":
        """
        Étape 1/2 : Chargement + Prétraitement + Gating CD45.

        Exécute uniquement la phase de gating sans lancer le clustering FlowSOM.
        Permet à l'UI d'afficher les plots de validation (CD45, scatter) et
        d'attendre la confirmation de l'utilisateur avant de continuer.

        Args:
            progress_callback: Callback de progression (step, percentage).

        Returns:
            GatingResult contenant les échantillons prétraités, les plots de gating
            et les métadonnées nécessaires pour execute_clustering().

        Raises:
            PipelineResult.failure sera encapsulé dans GatingResult.success=False
            si une étape bloque.
        """

        def _report(step: PipelineStep) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(step, int(step))
                except Exception:
                    pass

        config = self.config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        gating_reports.clear()
        gating_log_entries.clear()
        ransac_scatter_data.clear()
        singlets_summary_per_file.clear()

        _logger.info("=" * 60)
        _logger.info("PIPELINE — PHASE GATING UNIQUEMENT")
        _logger.info("=" * 60)
        _report(PipelineStep.START)

        checkpoint_manager = PipelineCheckpointManager(config)
        input_files = self._resolve_input_fcs_files()
        if not input_files:
            return GatingResult(
                success=False,
                error="Aucun fichier FCS détecté",
                timestamp=timestamp,
            )

        viz_cfg = getattr(config, "visualization", None)
        viz_save = getattr(viz_cfg, "save_plots", True)

        _single_patho = getattr(config.paths, "patho_single_file", None)
        if not _single_patho:
            _patho_folder_path = Path(getattr(config.paths, "patho_folder", "") or "")
            if _patho_folder_path.exists():
                _patho_files_found = get_fcs_files(_patho_folder_path)
                if len(_patho_files_found) == 1:
                    _single_patho = str(_patho_files_found[0])

        _base_output = Path(config.paths.output_dir)
        _patho_stem = Path(_single_patho).stem if _single_patho else None
        output_dir = _base_output / f"résultats_{_patho_stem}" if _patho_stem else _base_output
        output_dir.mkdir(parents=True, exist_ok=True)

        _logger.info("Étape 2: Prétraitement (gating combiné + transformation)...")
        _report(PipelineStep.PREPROCESSING)

        gating_key = self._build_gating_cache_key(checkpoint_manager, input_files)
        gating_ckpt = checkpoint_manager.load(gating_key)
        samples: List[FlowSample] = []
        n_total_for_sankey = 0

        if gating_ckpt is not None:
            processed_samples = gating_ckpt.get("processed_samples", [])
            gating_figures = {}
            self._gating_logger = _restore_gating_logger(gating_ckpt.get("gating_events", []))
            n_total_for_sankey = int(gating_ckpt.get("n_total_raw", 0))
            _logger.info(
                "  Checkpoint post-gating utilisé: %d échantillon(s)", len(processed_samples)
            )
        else:
            _logger.info("Étape 1: Chargement des fichiers FCS...")
            _report(PipelineStep.LOADING)
            samples = self._load_all_samples()
            if not samples:
                return GatingResult(
                    success=False, error="Aucun échantillon chargé", timestamp=timestamp
                )

            n_total_for_sankey = int(sum(s.matrix.shape[0] for s in samples))
            processed_samples, gating_figures, gating_checkpoint = preprocess_combined(
                samples,
                config,
                gating_logger=self._gating_logger,
                gating_plot_dir=output_dir / "plots" if viz_save else None,
            )
            checkpoint_manager.save(
                gating_key,
                {
                    "processed_samples": processed_samples,
                    "gating_payload": gating_checkpoint,
                    "gating_events": [e.to_dict() for e in self._gating_logger.events],
                    "n_total_raw": n_total_for_sankey,
                },
            )

        if not processed_samples:
            return GatingResult(
                success=False,
                error="Aucun échantillon valide après prétraitement",
                timestamp=timestamp,
            )

        _report(PipelineStep.GATING_DONE)
        _logger.info(
            "  Gating terminé : %d/%d échantillon(s) valides",
            len(processed_samples),
            max(len(samples), len(input_files)),
        )

        return GatingResult(
            success=True,
            processed_samples=processed_samples,
            gating_figures=gating_figures if not gating_ckpt else {},
            gating_key=gating_key,
            output_dir=str(output_dir),
            n_total_raw=n_total_for_sankey,
            timestamp=timestamp,
            checkpoint_manager=checkpoint_manager,
            input_files=input_files,
            gating_events=[e.to_dict() for e in self._gating_logger.events],
        )

    def execute_clustering(
        self,
        gating_result: "GatingResult",
        progress_callback: Optional[Callable[[PipelineStep, int], None]] = None,
    ) -> PipelineResult:
        """
        Étape 2/2 : Clustering FlowSOM + MRD + Exports.

        Reçoit les échantillons prétraités depuis execute_gating() et exécute
        la suite du pipeline (clustering, MRD, visualisations, exports).

        Args:
            gating_result: Résultat de execute_gating().
            progress_callback: Callback de progression (step, percentage).

        Returns:
            PipelineResult complet.

        Raises:
            ValueError si gating_result.success est False.
        """
        if not gating_result.success:
            raise ValueError(
                f"execute_clustering() appelé avec un GatingResult en échec : {gating_result.error}"
            )
        # Injecte les données de gating dans l'état interne et délègue à execute()
        # en court-circuitant la phase de gating via le checkpoint déjà sauvegardé.
        return self.execute(progress_callback=progress_callback)

    def execute(
        self,
        progress_callback: Optional[Callable[[PipelineStep, int], None]] = None,
    ) -> PipelineResult:
        """
        Exécute le pipeline complet.

        Args:
            progress_callback: Callable optionnel appelé à chaque étape majeure.
                Signature : ``callback(step: PipelineStep, percentage: int)``.
                Permet à l'UI d'afficher une progression propre sans scraper les logs.
                Exemple : ``lambda step, pct: progress_signal.emit(pct)``

        Returns:
            PipelineResult avec tous les résultats et les chemins d'export.
        """

        def _report(step: PipelineStep) -> None:
            """Émet le callback de progression de manière non-bloquante."""
            if progress_callback is not None:
                try:
                    progress_callback(step, int(step))
                except Exception:
                    pass  # Ne jamais bloquer le pipeline sur un callback UI défaillant

        start_time = time.time()
        config = self.config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Réinitialiser les registres globaux pour éviter la contamination entre runs
        gating_reports.clear()
        gating_log_entries.clear()
        ransac_scatter_data.clear()
        singlets_summary_per_file.clear()

        _logger.info("=" * 60)
        _logger.info("PRISMA — DÉMARRAGE")
        _logger.info("=" * 60)
        _report(PipelineStep.START)

        # ── Monitoring de performance (optionnel) ─────────────────────────────
        _monitor = None
        _perf_cfg = getattr(config, "performance_monitoring", None)
        if _perf_cfg and getattr(_perf_cfg, "enabled", False):
            try:
                from flowsom_pipeline_pro.src.prisma.monitoring.performance_monitor import (
                    PerformanceMonitor,
                )

                _monitor = PerformanceMonitor(
                    interval=getattr(_perf_cfg, "interval_seconds", 1.0),
                    include_gpu=getattr(_perf_cfg, "include_gpu", True),
                )
                _monitor.start()
                _logger.info(
                    "Performance monitor actif (interval=%.1fs)",
                    _perf_cfg.interval_seconds,
                )
            except Exception as _me:
                _logger.debug("Performance monitor non disponible: %s", _me)
                _monitor = None

        # ── Warm-up Matplotlib (charge le cache de polices une seule fois) ──────
        # Sans ce warm-up, chaque plt.figure() dans les plots de gating
        # déclenche un scan complet du système de polices → 30k logs findfont.
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt_warmup
            import matplotlib.font_manager as _fm

            # Forcer le chargement du FontManager en mémoire
            _ = _fm.fontManager.ttflist
            # Créer et fermer une figure vide pour initialiser le renderer
            _fig_tmp = _plt_warmup.figure()
            _plt_warmup.close(_fig_tmp)
            del _plt_warmup, _fig_tmp, _fm
            _logger.info("Matplotlib warm-up: cache polices chargé.")
        except Exception as _mpl_e:
            _logger.debug("Matplotlib warm-up ignoré: %s", _mpl_e)

        # ── Warm-up kaleido (démarre Chromium une seule fois, avant les exports) ──
        try:
            from flowsom_pipeline_pro.src.prisma.utils.kaleido_scope import (
                ensure_kaleido_scope,
                warm_up_kaleido,
            )

            ensure_kaleido_scope()
            warm_up_kaleido()
        except Exception:
            pass

        try:
            plot_worker: Optional[PlottingWorker] = None
            checkpoint_manager = PipelineCheckpointManager(config)
            input_files = self._resolve_input_fcs_files()
            if not input_files:
                return PipelineResult.failure(
                    error="Aucun fichier FCS détecté",
                    config=config,
                )

            samples: List[FlowSample] = []
            n_total_for_sankey = 0

            viz_cfg = getattr(config, "visualization", None)
            viz_save = getattr(viz_cfg, "save_plots", True)
            umap_enabled = getattr(viz_cfg, "umap_enabled", True)

            gating_key = self._build_gating_cache_key(checkpoint_manager, input_files)
            gating_ckpt = checkpoint_manager.load(gating_key)

            # ── Nommage dynamique (Features 1 & 3) ────────────────────────────
            # Si un fichier patho unique est spécifié (mode batch ou run simple),
            # on utilise son stem pour nommer les rapports et structurer le dossier.
            _single_patho = getattr(config.paths, "patho_single_file", None)

            # Mode unitaire : patho_single_file non défini → déduire depuis patho_folder
            if not _single_patho:
                _patho_folder_path = Path(getattr(config.paths, "patho_folder", "") or "")
                if _patho_folder_path.exists():
                    _patho_files_found = get_fcs_files(_patho_folder_path)
                    if len(_patho_files_found) == 1:
                        _single_patho = str(_patho_files_found[0])

            _patho_stem = Path(_single_patho).stem if _single_patho else None

            # Extraire la date du fichier FCS pathologique
            from flowsom_pipeline_pro.src.prisma.io.csv_exporter import (
                extract_date_from_filename,
            )

            _patho_date: Optional[str] = None
            if _single_patho:
                _dt = extract_date_from_filename(_single_patho)
                if _dt is not None:
                    _patho_date = _dt.strftime("%Y-%m-%d")

            # Définir output_dir tôt (utilisé pour les plots de gating)
            _base_output = Path(config.paths.output_dir)
            if _patho_stem:
                # Inclure la date dans le nom du dossier si disponible
                if _patho_date:
                    output_dir = _base_output / f"résultats_{_patho_stem}_{_patho_date}"
                else:
                    output_dir = _base_output / f"résultats_{_patho_stem}"
            else:
                output_dir = _base_output
            output_dir.mkdir(parents=True, exist_ok=True)

            # ── Étape 2: Prétraitement ───────────────────────────────────────────
            _logger.info("Étape 2: Prétraitement (gating combiné + transformation)...")
            _report(PipelineStep.PREPROCESSING)
            if _monitor:
                _monitor.mark_phase("Prétraitement / Gating")
            # Utilise l'approche combinée (fidèle à flowsom_pipeline.py) :
            # gating sur les données brutes concaténées, sous-échantillonnage APRÈS.
            if gating_ckpt is not None:
                processed_samples = gating_ckpt.get("processed_samples", [])
                gating_figures = {}
                self._gating_logger = _restore_gating_logger(gating_ckpt.get("gating_events", []))
                n_total_for_sankey = int(gating_ckpt.get("n_total_raw", 0))
                _logger.info(
                    "  Checkpoint post-gating utilisé: %d échantillon(s)",
                    len(processed_samples),
                )
            else:
                _logger.info("Étape 1: Chargement des fichiers FCS...")
                _report(PipelineStep.LOADING)
                if _monitor:
                    _monitor.mark_phase("Chargement FCS")
                samples = self._load_all_samples()

                if not samples:
                    return PipelineResult.failure(
                        error="Aucun échantillon chargé",
                        config=config,
                    )

                n_total_for_sankey = int(sum(s.matrix.shape[0] for s in samples))
                _logger.info("  %d échantillon(s) chargé(s)", len(samples))

                processed_samples, gating_figures, gating_checkpoint = preprocess_combined(
                    samples,
                    config,
                    gating_logger=self._gating_logger,
                    gating_plot_dir=output_dir / "plots" if viz_save else None,
                )

                checkpoint_manager.save(
                    gating_key,
                    {
                        "processed_samples": processed_samples,
                        "gating_payload": gating_checkpoint,
                        "gating_events": [e.to_dict() for e in self._gating_logger.events],
                        "n_total_raw": n_total_for_sankey,
                    },
                )

            if not processed_samples:
                return PipelineResult.failure(
                    error="Aucun échantillon valide après prétraitement",
                    config=config,
                )

            _logger.info(
                "  %d/%d échantillon(s) après prétraitement",
                len(processed_samples),
                max(len(samples), len(input_files)),
            )

            # ── Étape 3: Clustering FlowSOM ───────────────────────────────────
            _logger.info("Étape 3: Clustering FlowSOM...")
            _report(PipelineStep.CLUSTERING_START)
            if _monitor:
                _monitor.mark_phase("Clustering SOM")
            som_key = self._build_som_cache_key(checkpoint_manager, gating_key)
            som_ckpt = checkpoint_manager.load(som_key)
            if som_ckpt is not None:
                metaclustering = som_ckpt["metaclustering"]
                clustering = som_ckpt["clustering"]
                clusterer = _deserialize_clusterer_state(som_ckpt["clusterer_state"])
                selected_markers = som_ckpt["selected_markers"]
                X_stacked = som_ckpt["X_stacked"]
                obs = som_ckpt["obs"]
                samples_used = som_ckpt["samples_used"]
                _logger.info("  Checkpoint post-SOM utilisé.")
            else:
                (
                    metaclustering,
                    clustering,
                    clusterer,
                    selected_markers,
                    X_stacked,
                    obs,
                    samples_used,
                ) = run_clustering(processed_samples, config)
                checkpoint_manager.save(
                    som_key,
                    {
                        "metaclustering": metaclustering,
                        "clustering": clustering,
                        "clusterer_state": _serialize_clusterer_state(clusterer),
                        "selected_markers": selected_markers,
                        "X_stacked": X_stacked,
                        "obs": obs,
                        "samples_used": samples_used,
                    },
                )

            n_meta = (
                int(metaclustering.max()) + 1
                if metaclustering is not None and len(metaclustering) > 0
                else 0
            )
            _report(PipelineStep.CLUSTERING_DONE)
            _logger.info(
                "  FlowSOM terminé: %d cellules, %d marqueurs, %d métaclusters",
                X_stacked.shape[0],
                len(selected_markers),
                n_meta,
            )

            # ── Étape 4: Construction du DataFrame cellulaire ─────────────────
            _logger.info("Étape 4: Construction du DataFrame cellulaire...")
            _report(PipelineStep.DATAFRAME)
            if _monitor:
                _monitor.mark_phase("Construction DataFrame")
            df_cells = build_cells_dataframe(
                X_stacked, selected_markers, obs, metaclustering, clustering
            )

            # ── Données brutes alignées sur df_cells ──────────────────────────
            # Construit un DataFrame de même longueur que df_cells avec les
            # intensités linéaires pré-transformation, pour la visualisation GUI.
            _raw_data_df = None
            try:
                _X_raw, _raw_var_names, _ = stack_raw_markers(samples_used)
                if _X_raw.shape[0] == len(df_cells):
                    import pandas as _pd_raw
                    _raw_data_df = _pd_raw.DataFrame(
                        _X_raw, columns=_raw_var_names, index=df_cells.index
                    )
                else:
                    _logger.warning(
                        "raw_data_df: taille incohérente (%d vs %d) — ignoré",
                        _X_raw.shape[0], len(df_cells),
                    )
            except Exception as _e_raw:
                _logger.debug("raw_data_df: non disponible (%s)", _e_raw)

            # ── Étape 4c: Pré-screening CD34+/CD45dim (TOUJOURS exécuté) ─────
            # Calcul heuristique sur les données transformées post-gating.
            # Indépendant des paramètres cd34 de l'utilisateur.
            _prescreening_result = None
            try:
                from flowsom_pipeline_pro.src.prisma.analysis.prescreening import (
                    compute_cd34_prescreening,
                )

                _density_method_ps = getattr(config.pregate, "cd34_cd45dim_density_method", "KDE")
                _logger.info(
                    "Étape 4c: Pré-screening CD34+/CD45dim (méthode=%s)...",
                    _density_method_ps,
                )
                # Filtrer uniquement les cellules pathologiques
                # (obs contient la colonne "condition" alignée sur X_stacked)
                _patho_label = "Pathologique"
                if "condition" in obs.columns:
                    _patho_mask_ps = obs["condition"].values == _patho_label
                    _X_patho_ps = X_stacked[_patho_mask_ps]
                    _logger.info(
                        "   Pré-screening sur %d cellules patho uniquement",
                        int(_patho_mask_ps.sum()),
                    )
                else:
                    _X_patho_ps = X_stacked
                    _logger.warning(
                        "   Colonne 'condition' absente — pré-screening sur toutes les cellules"
                    )
                _prescreening_result = compute_cd34_prescreening(
                    _X_patho_ps,
                    list(selected_markers),
                    density_method=_density_method_ps,
                )
                if _prescreening_result is not None:
                    _logger.info(
                        "   Pré-screening: CD34+=%d, CD45dim=%d, ratio=%.1f%% [%s]",
                        _prescreening_result.n_cd34_pos,
                        _prescreening_result.n_cd45dim,
                        _prescreening_result.ratio_pct,
                        _prescreening_result.alert_level,
                    )
                    if _prescreening_result.alert_level != "none":
                        _logger.warning(
                            "   [ALERTE Pré-screening] %s",
                            _prescreening_result.alert_message,
                        )
                    # Générer le plot KDE CD34
                    if viz_save:
                        try:
                            from flowsom_pipeline_pro.src.prisma.visualization.gating_plots import (
                                plot_cd34_kde_prescreening as _plot_cd34_ps,
                            )

                            _fig_cd34_ps = _plot_cd34_ps(
                                _prescreening_result,
                                output_path=output_dir
                                / "plots"
                                / "gating"
                                / f"prescreening_cd34_kde_{timestamp}.png",
                            )
                            if _fig_cd34_ps is not None:
                                gating_figures["fig_prescreening_cd34"] = _fig_cd34_ps
                        except Exception as _plot_ps_exc:
                            _logger.warning(
                                "Plot pré-screening CD34 KDE échoué (non bloquant): %s",
                                _plot_ps_exc,
                            )
            except Exception as _ps_exc:
                _logger.warning("Pré-screening CD34/CD45dim échoué (non bloquant): %s", _ps_exc)

            # ── Étape 4b: Construction du DataFrame FCS complet ───────────────
            # Identique au monolithe flowsom_pipeline.py :
            # - Toutes les colonnes (données brutes pré-transformation)
            # - FlowSOM_metacluster, FlowSOM_cluster (+1 pour Kaluza ≥ 1)
            # - xGrid, yGrid, xNodes, yNodes avec jitter circulaire (style R)
            # - size (nb cellules par node), Condition_Num
            _logger.info("  Construction du DataFrame FCS complet (style Kaluza)...")
            df_fcs = self._build_fcs_dataframe(
                samples_used,
                metaclustering,
                clustering,
                clusterer,
            )

            # Matrice MFI — convertie en DataFrame (marqueurs × metaclusters)
            mfi_raw = clusterer.get_mfi_matrix(X_stacked, selected_markers)
            unique_mc = np.unique(metaclustering)
            mfi_matrix = pd.DataFrame(
                mfi_raw,
                index=[f"MC{i}" for i in unique_mc],
                columns=selected_markers[: mfi_raw.shape[1]],
            )

            # MFI médiane par nœud SOM — même logique que plot_cluster_radar_mrd
            # Utilisée pour les spider plots de l'accueil (cohérence avec le radar HTML)
            # Vectorisé : tri par node id + np.split évite la boucle Python O(n_nodes).
            _node_ids = np.unique(clustering)
            _sort_idx = np.argsort(clustering, kind="stable")
            _sorted_X = X_stacked[_sort_idx]
            _sorted_labels = clustering[_sort_idx]
            _split_points = np.searchsorted(_sorted_labels, _node_ids[1:])
            _node_chunks = np.split(_sorted_X, _split_points)
            _node_medians = np.array(
                [np.median(chunk, axis=0) for chunk in _node_chunks], dtype=float
            )
            node_mfi_matrix = pd.DataFrame(_node_medians, index=_node_ids, columns=selected_markers)

            # ── Étape 5: Metrics de clustering ────────────────────────────────
            if _monitor:
                _monitor.mark_phase("Métriques + Visualisations")
            metrics = self._compute_metrics(X_stacked, metaclustering)
            # ── Accumulation des figures pour le rapport HTML ─────────────────
            # Clés calquées sur les noms du notebook de référence
            _mpl_figures: Dict[str, object] = dict(gating_figures)  # fig_overview, fig_gate_*
            _plotly_figures: Dict[str, object] = {}

            # ── Diagnostic Harmony (si disponible) ────────────────────────────
            # _HARMONY_DIAG est rempli par _build_harmony_diag() dans
            # clustering_service.run_clustering() — non bloquant, silencieux.
            import flowsom_pipeline_pro.src.prisma.services.clustering_service as _cs_mod

            _hd = _cs_mod._HARMONY_DIAG
            if _hd.get("plotly") is not None:
                _plotly_figures["fig_harmony_diag"] = _hd["plotly"]
            if _hd.get("mpl") is not None:
                _mpl_figures["fig_harmony_diag"] = _hd["mpl"]

            plot_cfg = dict(getattr(config, "_extra", {}).get("plotting_worker", {}) or {})
            plot_async_enabled = bool(viz_save)
            plot_worker: Optional[PlottingWorker] = None
            if plot_async_enabled:
                plot_worker = PlottingWorker(max_queue_size=int(plot_cfg.get("queue_size", 48)))
                plot_worker.start()

            # ── Étape 5b: UMAP (si save_plots ET umap_enabled) ───────────────────
            if viz_save and umap_enabled:
                try:
                    from umap import UMAP

                    _logger.info("Calcul UMAP...")
                    _report(PipelineStep.UMAP)
                    fig_cfg = getattr(viz_cfg, "figures", {}) or {}
                    umap_cfg = fig_cfg.get("umap", {})
                    umap_sample_size = min(umap_cfg.get("n_sample", 100_000), X_stacked.shape[0])
                    rng_umap = np.random.default_rng(config.flowsom.seed)
                    idx_umap = rng_umap.choice(X_stacked.shape[0], umap_sample_size, replace=False)
                    umap_coords = UMAP(
                        n_components=2, random_state=config.flowsom.seed, n_jobs=-1
                    ).fit_transform(X_stacked[idx_umap])
                    self._enqueue_plot(
                        plot_worker,
                        "UMAP",
                        "fig_umap",
                        "mpl",
                        "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                        "plot_umap",
                        umap_coords,
                        metaclustering[idx_umap],
                        output_dir / "plots" / f"umap_{timestamp}.png",
                        n_metaclusters=int(metaclustering.max()) + 1,
                        seed=config.flowsom.seed,
                    )
                except Exception as _e:
                    _logger.warning("UMAP échoué (non bloquant): %s", _e)

            # ── MST + SOM Plotly/Matplotlib ───────────────────────────────────
            if viz_save:
                self._enqueue_plot(
                    plot_worker,
                    "MST statique",
                    "fig_mst_static",
                    "mpl",
                    "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                    "plot_mst_static",
                    clusterer,
                    mfi_matrix,
                    metaclustering,
                    output_dir / "plots" / f"mst_static_{timestamp}.png",
                )
                self._enqueue_plot(
                    plot_worker,
                    "MST Plotly",
                    "fig_mst",
                    "plotly",
                    "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                    "plot_mst_plotly",
                    clusterer,
                    mfi_matrix,
                    metaclustering,
                    output_dir / "plots" / f"mst_interactive_{timestamp}.html",
                )
                self._enqueue_plot(
                    plot_worker,
                    "SOM Grid Plotly",
                    "fig_grid_mc",
                    "plotly",
                    "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                    "plot_som_grid_plotly",
                    clustering,
                    metaclustering,
                    clusterer,
                    output_dir / "plots" / f"som_grid_{timestamp}.html",
                    seed=config.flowsom.seed,
                )

                self._enqueue_plot(
                    plot_worker,
                    "Heatmap MFI",
                    "fig_heatmap",
                    "mpl",
                    "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                    "plot_mfi_heatmap",
                    mfi_matrix,
                    output_dir / "plots" / f"mfi_heatmap_{timestamp}.png",
                )
                _cond_labels = (
                    df_cells["condition"].values if "condition" in df_cells.columns else None
                )
                self._enqueue_plot(
                    plot_worker,
                    "Distribution métaclusters",
                    "fig_comp",
                    "mpl",
                    "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                    "plot_metacluster_sizes",
                    metaclustering,
                    int(metaclustering.max()) + 1,
                    output_dir / "plots" / f"metacluster_distribution_{timestamp}.png",
                    condition_labels=_cond_labels,
                )

            # ── Star Chart FlowSOM (§13) ──────────────────────────────────────
            # fs.pl.plot_stars requiert un objet fs.FlowSOM natif (CPU uniquement).
            # Pour le backend GPU on utilise plot_star_chart_custom (matplotlib pur).
            if viz_save:
                try:
                    _fsom_native = getattr(clusterer, "_fsom_model", None)
                    _used_gpu = getattr(clusterer, "used_gpu_", False)

                    if (
                        _fsom_native is not None
                        and not _used_gpu
                        and hasattr(_fsom_native, "get_cluster_data")
                    ):
                        self._enqueue_plot(
                            plot_worker,
                            "Star Chart CPU",
                            "fig_star_chart",
                            "mpl",
                            "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                            "plot_star_chart",
                            _fsom_native,
                            output_dir / "plots" / f"flowsom_star_chart_{timestamp}.png",
                        )
                    else:
                        _star_marker_names = (
                            list(processed_samples[0].markers)
                            if processed_samples and hasattr(processed_samples[0], "markers")
                            else None
                        )
                        self._enqueue_plot(
                            plot_worker,
                            "Star Chart GPU custom",
                            "fig_star_chart",
                            "mpl",
                            "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                            "plot_star_chart_custom",
                            clusterer,
                            output_dir / "plots" / f"flowsom_star_chart_{timestamp}.png",
                            marker_names=_star_marker_names,
                            title=f"FlowSOM Star Chart — GPU ({clusterer.xdim}×{clusterer.ydim})",
                        )
                except Exception as _e:
                    _logger.warning("Star Chart échoué (non bloquant): %s", _e)

            # ── Grille SOM statique PNG (§14) ────────────────────────────────
            if viz_save:
                try:
                    _cond_labels_grid = (
                        df_cells["condition"].values if "condition" in df_cells.columns else None
                    )
                    # plot_som_grid_static attend un metaclustering par NODE
                    # (n_nodes,), PAS par cellule (n_cells,).
                    _mc_per_node = getattr(clusterer, "metacluster_map_", None)
                    if _mc_per_node is None:
                        _mc_per_node = np.array(
                            [
                                int(np.bincount(metaclustering[clustering == i]).argmax())
                                if (clustering == i).any()
                                else 0
                                for i in range(clusterer.n_nodes)
                            ],
                            dtype=int,
                        )
                    _viz_cfg = getattr(config, "visualization", None)
                    _som_max_cells = int(getattr(_viz_cfg, "som_grid_max_display_cells", 100_000))
                    _som_dpi = int(getattr(_viz_cfg, "som_grid_dpi", 100))
                    self._enqueue_plot(
                        plot_worker,
                        "SOM grid statique",
                        "fig_som_grid_static",
                        "mpl",
                        "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                        "plot_som_grid_static",
                        clustering,
                        _mc_per_node,
                        clusterer.get_grid_coords(),
                        _cond_labels_grid,
                        clusterer.xdim,
                        clusterer.ydim,
                        output_dir / "plots" / f"flowsom_som_grid_{timestamp}.png",
                        seed=config.flowsom.seed,
                        dpi=_som_dpi,
                        max_display_cells=_som_max_cells,
                    )
                except Exception as _e:
                    _logger.warning("Grille SOM statique échouée (non bloquant): %s", _e)

            # ── Radar + bar charts + vue combinée ────────────────────────────
            if viz_save:
                _cond_labels = (
                    df_cells["condition"].values if "condition" in df_cells.columns else None
                )

                self._enqueue_plot(
                    plot_worker,
                    "Radar métaclusters",
                    "fig_radar",
                    "plotly",
                    "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                    "plot_metacluster_radar",
                    mfi_matrix.values,
                    list(mfi_matrix.columns),
                    metaclustering,
                    output_dir / "plots" / f"metacluster_radar_{timestamp}.html",
                    n_metaclusters=n_meta,
                )

                self._enqueue_plot(
                    plot_worker,
                    "Radar clusters SOM",
                    "fig_cluster_radar",
                    "plotly",
                    "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                    "plot_cluster_radar",
                    X_stacked,
                    clustering,
                    list(mfi_matrix.columns),
                    output_dir / "plots" / f"cluster_radar_{timestamp}.html",
                )

                # Clusters exclusifs (log uniquement, pas de figure)
                try:
                    if _cond_labels is not None:
                        from flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots import (
                            compute_exclusive_clusters,
                        )

                        _excl = compute_exclusive_clusters(metaclustering, _cond_labels, n_meta)
                        for _line in _excl.get("summary_lines", []):
                            _logger.info("%s", _line)
                    else:
                        _logger.info(
                            "Clusters exclusifs ignorés (condition_labels non disponible)."
                        )
                except Exception as _e:
                    _logger.warning("Clusters exclusifs échoués (non bloquant): %s", _e)

                # Bar charts par métacluster
                _viz_cfg2 = getattr(config, "visualization", None)
                _export_jpg = getattr(_viz_cfg2, "export_jpg_barcharts", False)

                if _cond_labels is not None:
                    self._enqueue_plot(
                        plot_worker,
                        "Bar chart % patho par cluster",
                        "fig_patho_pct",
                        "plotly",
                        "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                        "plot_patho_pct_per_cluster",
                        metaclustering,
                        _cond_labels,
                        output_html=output_dir
                        / "plots"
                        / f"patho_pct_per_cluster_{timestamp}.html",
                        output_jpg=output_dir / "plots" / f"patho_pct_per_cluster_{timestamp}.jpg"
                        if _export_jpg
                        else None,
                    )
                else:
                    _logger.info("Bar chart %% patho ignoré (condition_labels non disponible).")

                self._enqueue_plot(
                    plot_worker,
                    "Bar chart % cellules par cluster",
                    "fig_cells_pct",
                    "plotly",
                    "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                    "plot_cells_pct_per_cluster",
                    metaclustering,
                    output_html=output_dir / "plots" / f"cells_pct_per_cluster_{timestamp}.html",
                    output_jpg=output_dir / "plots" / f"cells_pct_per_cluster_{timestamp}.jpg"
                    if _export_jpg
                    else None,
                    condition_labels=_cond_labels,
                )

                # Bar charts par nœud SOM
                if _cond_labels is not None:
                    self._enqueue_plot(
                        plot_worker,
                        "Bar chart % patho par nœud SOM",
                        "fig_patho_pct_som",
                        "plotly",
                        "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                        "plot_patho_pct_per_som_node",
                        clustering,
                        _cond_labels,
                        output_html=output_dir
                        / "plots"
                        / f"patho_pct_per_som_node_{timestamp}.html",
                        output_jpg=output_dir / "plots" / f"patho_pct_per_som_node_{timestamp}.jpg"
                        if _export_jpg
                        else None,
                    )
                else:
                    _logger.info("Bar chart %% patho SOM ignoré (condition_labels non disponible).")

                self._enqueue_plot(
                    plot_worker,
                    "Bar chart % cellules par nœud SOM",
                    "fig_cells_pct_som",
                    "plotly",
                    "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                    "plot_cells_pct_per_som_node",
                    clustering,
                    output_html=output_dir / "plots" / f"cells_pct_per_som_node_{timestamp}.html",
                    output_jpg=output_dir / "plots" / f"cells_pct_per_som_node_{timestamp}.jpg"
                    if _export_jpg
                    else None,
                    condition_labels=_cond_labels,
                )

                # Vue combinée
                if _cond_labels is not None:
                    self._enqueue_plot(
                        plot_worker,
                        "Vue combinée nœuds SOM",
                        "fig_som_combined",
                        "plotly",
                        "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                        "plot_combined_som_node_html",
                        clustering,
                        _cond_labels,
                        output_html=output_dir / "plots" / f"som_node_combined_{timestamp}.html",
                        X=X_stacked,
                        used_markers=list(mfi_matrix.columns),
                    )
                else:
                    _logger.info("Vue combinée SOM ignorée (condition_labels non disponible).")

            # ── Étape 6: Exports ───────────────────────────────────────────────────
            _logger.info("Étape 6: Exports...")
            _report(PipelineStep.EXPORTS)
            if _monitor:
                _monitor.mark_phase("Exports FCS / CSV / JSON")
            # name_stem pour les rapports : "rapport_<stem>" si fichier unique, sinon défaut
            _report_name_stem = (
                f"rapport_{_patho_stem}" if _patho_stem else f"flowsom_report_{timestamp}"
            )
            exporter = ExportService(
                config,
                output_dir,
                timestamp=timestamp,
                name_stem=_report_name_stem,
                patho_name=_patho_stem,
                patho_date=_patho_date,
            )

            export_paths = exporter.export_all(
                df_cells=df_cells,
                df_fcs=df_fcs,
                mfi_matrix=mfi_matrix,
                metaclustering=metaclustering,
                selected_markers=selected_markers,
                input_files=input_files,
                gating_logger=self._gating_logger,
                clustering=clustering,
            )

            # ── Étape 6b: Distribution Sain/Patho par cluster SOM ────────────
            _cond_dist = df_cells["condition"].values if "condition" in df_cells.columns else None
            _compact_mode = (
                getattr(getattr(config, "export_mode", None), "mode", "standard") == "compact"
            )
            if _cond_dist is not None and not _compact_mode:
                try:
                    dist_paths = exporter.export_cluster_distribution(
                        clustering=clustering,
                        metaclustering=metaclustering,
                        condition_labels=_cond_dist,
                    )
                    export_paths.update(dist_paths)
                    _logger.info(
                        "Distribution Sain/Patho exportee: %d fichier(s)",
                        len(dist_paths),
                    )
                except Exception as _e:
                    _logger.warning("Export distribution clusters echoue (non bloquant): %s", _e)

            # ── Sankey gating ─────────────────────────────────────────
            if viz_cfg is not None and viz_save:
                try:
                    events = self._gating_logger.events
                    gate_map = {e.gate_name: e for e in events if e.file == "COMBINED"}
                    n_total = (
                        int(n_total_for_sankey)
                        if n_total_for_sankey > 0
                        else int(metaclustering.shape[0])
                    )
                    gate_counts = {
                        "n_total": n_total,
                        "n_g1_pass": gate_map["G1_debris"].n_after
                        if "G1_debris" in gate_map
                        else n_total,
                        "n_g2_pass": gate_map["G2_singlets"].n_after
                        if "G2_singlets" in gate_map
                        else n_total,
                        "n_g3_pass": gate_map["G3_cd45"].n_after
                        if "G3_cd45" in gate_map
                        else n_total,
                        "n_final": int(metaclustering.shape[0]),
                    }
                    sankey_result = exporter.export_sankey(
                        gate_counts,
                        filter_blasts=getattr(config.pregate, "cd34", False),
                    )
                    if (
                        isinstance(sankey_result, dict)
                        and sankey_result.get("fig_sankey") is not None
                    ):
                        _plotly_figures["fig_sankey"] = sankey_result["fig_sankey"]
                    _logger.info("Sankey exporté.")
                except Exception as _e:
                    _logger.warning("Sankey échoué (non bloquant): %s", _e)

            # ── Étape 7: Mapping des populations (Section 10) ─────────────────
            if _monitor:
                _monitor.mark_phase("Population Mapping")
            population_mapping_result = None
            pop_map_cfg = getattr(config, "population_mapping", None)
            if pop_map_cfg is not None and getattr(pop_map_cfg, "enabled", False):
                _logger.info("Étape 7: Mapping populations via MFI (Section 10)...")
                _report(PipelineStep.REPORT)
                try:
                    from flowsom_pipeline_pro.src.services.population_mapping_service import (
                        PopulationMappingService,
                    )

                    # Trouver le FCS exporté à l'étape 6
                    fcs_exported = None
                    for key in (
                        "fcs_complete",
                        "fcs_with_clusters",
                        "fcs_export",
                        "fcs",
                    ):
                        p = export_paths.get(key)
                        if p and Path(p).exists():
                            fcs_exported = Path(p)
                            break

                    if fcs_exported is None:
                        # Fallback : chercher le FCS le plus récent dans output_dir (récursif)
                        fcs_candidates = sorted(
                            output_dir.glob("**/*.fcs"),
                            key=lambda f: f.stat().st_mtime,
                            reverse=True,
                        )
                        if fcs_candidates:
                            fcs_exported = fcs_candidates[0]

                    if fcs_exported is not None:
                        _logger.info("  FCS source: %s", fcs_exported.name)
                        pm_service = PopulationMappingService(pop_map_cfg)
                        population_mapping_result = pm_service.run_full_mapping(
                            fcs_path=fcs_exported,
                            cluster_data=clusterer,
                            cell_data=None,  # AnnData non exposé ici — passer si disponible
                            output_dir=output_dir,
                            timestamp=timestamp,
                        )
                        _logger.info("Étape 7: Mapping populations OK.")
                    else:
                        _logger.warning("Étape 7: FCS exporté introuvable — mapping ignoré.")

                except Exception as exc:
                    _logger.error("Étape 7 échouée (non bloquant): %s", exc)

            # Récupérer les figures Plotly de la cartographie de populations
            if population_mapping_result is not None:
                pop_figs = getattr(population_mapping_result, "figures_plotly", {})
                if pop_figs:
                    _plotly_figures.update({k: v for k, v in pop_figs.items() if v is not None})

            # ── Étape 8: Calcul MRD résiduelle ────────────────────────────────
            if _monitor:
                _monitor.mark_phase("MRD")
            mrd_result = None
            try:
                from flowsom_pipeline_pro.src.analysis.mrd_calculator import (
                    load_mrd_config,
                    compute_mrd,
                )

                _mrd_config_path = getattr(config, "_extra", {}).get("mrd_config_path") or None
                mrd_cfg = load_mrd_config(_mrd_config_path)

                if mrd_cfg.enabled and "condition" in df_cells.columns:
                    _logger.info("Étape 8: Calcul MRD résiduelle (nœuds SOM)...")

                    # ── Masque CD45 pour la MRD ───────────────────────────────
                    # Construit systématiquement pour permettre le toggle UI
                    # (blastes / total patho ↔ blastes / CD45+ patho), quelle
                    # que soit la valeur de cd45_autogating_mode.
                    #
                    # Cas 1 — gate CD45 active (pregate.cd45=True) :
                    #   df_cells ne contient que des cellules CD45+ → mask all True
                    # Cas 2 — gate CD45 inactive :
                    #   reconstruction depuis la colonne CD45 de df_cells
                    _cd45_autogating_mode = getattr(config.pregate, "cd45_autogating_mode", "none")
                    _pregate_cd45_active = getattr(config.pregate, "cd45", True)
                    _cd45_mask_mrd: Optional[np.ndarray] = None

                    if _pregate_cd45_active:
                        # Gate CD45 active : df_cells ne contient que des CD45+.
                        # Récupération du nombre exact de cellules patho CD45+
                        # depuis l'event COMBINED G3_cd45 (extras n_patho_post_cd45),
                        # stocké lors du gating asymétrique mode_blastes_vs_normal.
                        _combined_g3_event = next(
                            (
                                e
                                for e in self._gating_logger.events
                                if e.gate_name == "G3_cd45" and e.file == "COMBINED"
                            ),
                            None,
                        )
                        _n_patho_post_cd45 = (
                            _combined_g3_event.extra.get("n_patho_post_cd45")
                            if _combined_g3_event is not None
                            else None
                        )
                        _n_patho_pre_cd45 = (
                            _combined_g3_event.extra.get("n_patho_pre_cd45")
                            if _combined_g3_event is not None
                            else None
                        )
                        if _n_patho_post_cd45 is not None and _n_patho_post_cd45 > 0:
                            # On passe un masque all-True à compute_mrd (df_cells
                            # est déjà CD45+), et on surcharge n_patho_cd45pos
                            # et n_patho_pre_cd45 après coup avec les valeurs exactes.
                            _cd45_mask_mrd = np.ones(len(df_cells), dtype=bool)
                            _n_patho_cd45_override: Optional[int] = int(_n_patho_post_cd45)
                            _n_patho_pre_override: Optional[int] = (
                                int(_n_patho_pre_cd45) if _n_patho_pre_cd45 is not None else None
                            )
                            _logger.info(
                                "n_patho_cd45pos (COMBINED G3 extras): %d CD45+ / %d total patho",
                                _n_patho_cd45_override,
                                _n_patho_pre_override or 0,
                            )
                        else:
                            _cd45_mask_mrd = np.ones(len(df_cells), dtype=bool)
                            _n_patho_cd45_override = None
                            _n_patho_pre_override = None
                            _logger.warning(
                                "n_patho_post_cd45 absent de l'event COMBINED G3_cd45 "
                                "— relancer sans checkpoint pour régénérer les extras."
                            )
                    else:
                        # Gate CD45 inactive → recalcul depuis la colonne CD45
                        _n_patho_cd45_override = None
                        _n_patho_pre_override = None
                        _cd45_col = next(
                            (
                                c
                                for c in df_cells.columns
                                if c.upper() in ("CD45", "CD45-PECY5", "CD45-PC5")
                            ),
                            None,
                        )
                        if _cd45_col is not None:
                            _pct = getattr(config.pregate, "cd45_threshold_percentile", 5.0)
                            _cd45_vals = df_cells[_cd45_col].values
                            _thr = np.nanpercentile(_cd45_vals, _pct)
                            _cd45_mask_mrd = _cd45_vals >= _thr
                            _logger.info(
                                "Masque CD45 MRD (seuil %.1f%%): %d/%d cellules CD45+",
                                _pct,
                                int(_cd45_mask_mrd.sum()),
                                len(_cd45_mask_mrd),
                            )
                        else:
                            _logger.warning(
                                "Colonne CD45 absente dans df_cells — "
                                "n_patho_cd45pos non disponible pour le toggle UI."
                            )

                    # ── Données pour la porte biologique hybride ──────────────
                    # Trois modes par ordre de précision décroissante :
                    #   Mode 1 : X_norm (z-scores pré-calculés par population_mapping)
                    #   Mode 2 : node_medians + nbm_center + nbm_scale
                    #            → z-scoring à la volée contre les stats NBM
                    #            → calculé ici depuis df_cells (cellules "Sain")
                    #   Mode 3 : node_medians seul, z-scoring intra-dataset (dégradé)
                    _node_medians_arr: Optional[np.ndarray] = None
                    _x_norm_arr: Optional[np.ndarray] = None
                    _marker_names_list: Optional[list] = None
                    _nbm_center: Optional[np.ndarray] = None
                    _nbm_scale: Optional[np.ndarray] = None
                    _nbm_inv_cov: Optional[np.ndarray] = None

                    if mrd_cfg.blast_phenotype_filter.enabled:
                        try:
                            from flowsom_pipeline_pro.src.analysis.blast_detection import (
                                compute_reference_stats,
                            )

                            # Priorité 1 : X_norm depuis population_mapping
                            _pm_result = population_mapping_result
                            if (
                                _pm_result is not None
                                and hasattr(_pm_result, "node_medians_norm")
                                and _pm_result.node_medians_norm is not None
                            ):
                                _x_norm_arr = np.asarray(_pm_result.node_medians_norm, dtype=float)
                                _marker_names_list = list(
                                    getattr(
                                        _pm_result,
                                        "marker_names",
                                        node_mfi_matrix.columns,
                                    )
                                )
                                _logger.info(
                                    "Porte biologique Mode 1 : X_norm depuis "
                                    "population_mapping (%d noeuds, %d marqueurs)",
                                    _x_norm_arr.shape[0],
                                    _x_norm_arr.shape[1],
                                )
                            else:
                                # Mode 2 : z-scoring contre les cellules NBM (Sain)
                                # présentes dans df_cells (déjà transformées arcsinh)
                                _marker_names_list = list(node_mfi_matrix.columns)
                                _node_medians_arr = node_mfi_matrix.values

                                _sain_mask = (
                                    df_cells["condition"].values == mrd_cfg.condition_sain
                                    if "condition" in df_cells.columns
                                    else None
                                )
                                if _sain_mask is not None and _sain_mask.sum() >= 100:
                                    # Extraire les valeurs des marqueurs pour les
                                    # cellules saines (NBM) déjà transformées
                                    _cols_present = [
                                        c for c in _marker_names_list if c in df_cells.columns
                                    ]
                                    if _cols_present:
                                        _X_sain = df_cells.loc[
                                            _sain_mask, _cols_present
                                        ].values.astype(float)
                                        _t_stats0 = time.perf_counter()
                                        _c, _s, _ic = compute_reference_stats(
                                            _X_sain,
                                            robust=True,
                                            max_samples_for_stats=mrd_cfg.blast_phenotype_filter.nbm_stats_max_cells,
                                            max_samples_for_covariance=mrd_cfg.blast_phenotype_filter.nbm_cov_max_cells,
                                            max_samples_for_mincovdet=mrd_cfg.blast_phenotype_filter.nbm_mincovdet_max_cells,
                                        )
                                        _t_stats_s = time.perf_counter() - _t_stats0
                                        # Aligner sur marker_names_list complet
                                        _nbm_center = np.zeros(len(_marker_names_list))
                                        _nbm_scale = np.ones(len(_marker_names_list))
                                        for _i, _m in enumerate(_marker_names_list):
                                            if _m in _cols_present:
                                                _j = _cols_present.index(_m)
                                                _nbm_center[_i] = _c[_j]
                                                _nbm_scale[_i] = _s[_j]
                                        # inv_cov est déjà dans l'espace des cols_present
                                        # — on la stocke directement (même espace que node_medians)
                                        _nbm_inv_cov = _ic
                                        _logger.info(
                                            "Porte biologique Mode 2 : z-scoring vs "
                                            "NBM interne (%d cellules Sain, %d marqueurs) "
                                            "[stats en %.1fs]",
                                            int(_sain_mask.sum()),
                                            len(_cols_present),
                                            _t_stats_s,
                                        )
                                    else:
                                        _logger.warning(
                                            "Porte biologique Mode 3 (dégradé) : "
                                            "aucun marqueur commun entre node_mfi_matrix "
                                            "et df_cells — z-scoring intra-dataset."
                                        )
                                else:
                                    _logger.warning(
                                        "Porte biologique Mode 3 (dégradé) : "
                                        "pas assez de cellules Sain (%d) pour "
                                        "calculer les stats NBM — z-scoring intra-dataset.",
                                        int(_sain_mask.sum()) if _sain_mask is not None else 0,
                                    )
                        except Exception as _e:
                            _logger.warning(
                                "Impossible de preparer les donnees pour le "
                                "filtre phenotypique hybride : %s",
                                _e,
                            )

                    mrd_result = compute_mrd(
                        df_cells,
                        clustering,
                        mrd_cfg,
                        cd45_autogating_mode=_cd45_autogating_mode,
                        cd45_mask=_cd45_mask_mrd,
                        X_norm=_x_norm_arr,
                        node_medians=_node_medians_arr,
                        marker_names=_marker_names_list,
                        nbm_center=_nbm_center,
                        nbm_scale=_nbm_scale,
                        nbm_inv_cov=_nbm_inv_cov,
                        # Audit trail : chemins des fichiers YAML pour le hash SHA256
                        mrd_config_path=getattr(mrd_cfg, "_config_file_path", None),
                        panel_config_path=getattr(config, "panel_config_path", None),
                    )

                    # Override n_patho_cd45pos et n_patho_pre_cd45 avec les valeurs
                    # exactes du gating logger (gating combiné asymétrique).
                    if (
                        mrd_result is not None
                        and _n_patho_cd45_override is not None
                        and _n_patho_cd45_override > 0
                    ):
                        mrd_result.n_patho_cd45pos = _n_patho_cd45_override
                    if (
                        mrd_result is not None
                        and _n_patho_pre_override is not None
                        and _n_patho_pre_override > 0
                    ):
                        mrd_result.n_patho_pre_cd45 = _n_patho_pre_override

                    # ── Regénération QC Gate 3 KDE CD45 avec ratio MRD ────────
                    # Le plot KDE CD45 initial est généré avant le calcul MRD
                    # dans preprocessing_service. On le régénère ici pour
                    # afficher le ratio MRD corrigé (blastes / CD45+ patho).
                    if viz_save and mrd_result is not None:
                        try:
                            from flowsom_pipeline_pro.src.prisma.visualization.gating_plots import (
                                plot_cd45_kde_qc as _plot_cd45_kde_mrd,
                            )
                            from flowsom_pipeline_pro.src.prisma.core.gating import (
                                PreGating as _PG_mrd,
                            )

                            _cd45_col_name = next(
                                (
                                    c
                                    for c in df_cells.columns
                                    if c.upper() in ("CD45", "CD45-PECY5", "CD45-PC5")
                                ),
                                None,
                            )
                            if _cd45_col_name is not None:
                                _cd45_data_mrd = df_cells[_cd45_col_name].values.astype(np.float64)
                                # Masque CD45 : toutes les cellules de df_cells
                                # sont CD45+ si gate CD45 était active
                                _mask_cd45_mrd_plot = (
                                    _cd45_mask_mrd
                                    if _cd45_mask_mrd is not None
                                    else np.ones(len(df_cells), dtype=bool)
                                )
                                _cond_mrd = (
                                    df_cells["condition"].values
                                    if "condition" in df_cells.columns
                                    else None
                                )
                                _fig_kde_cd45_mrd, _, _, _ = _plot_cd45_kde_mrd(
                                    cd45_data=_cd45_data_mrd,
                                    mask_cd45=_mask_cd45_mrd_plot,
                                    output_path=output_dir
                                    / "plots"
                                    / "gating"
                                    / f"combined_07_kde_cd45_mrd_{timestamp}.png",
                                    conditions=_cond_mrd,
                                    condition_patho=mrd_cfg.condition_patho,
                                    kde_seuil_relatif=getattr(
                                        config.pregate, "kde_cd45_seuil_relatif", 0.05
                                    ),
                                    kde_finesse=getattr(config.pregate, "kde_cd45_finesse", 0.6),
                                    kde_sigma_smooth=getattr(
                                        config.pregate, "kde_cd45_sigma_smooth", 10
                                    ),
                                    kde_n_grid=getattr(config.pregate, "kde_cd45_n_grid", 1000),
                                    mrd_result=mrd_result,
                                )
                                if _fig_kde_cd45_mrd is not None:
                                    # Remplace le plot KDE CD45 initial par la
                                    # version enrichie avec le ratio MRD
                                    gating_figures["fig_kde_cd45"] = _fig_kde_cd45_mrd
                                    _logger.info(
                                        "QC Gate 3 KDE CD45 regénéré avec ratio MRD corrigé"
                                    )
                        except Exception as _kde_mrd_e:
                            _logger.warning(
                                "Regénération QC KDE CD45 + MRD échouée (non bloquant): %s",
                                _kde_mrd_e,
                            )

                    # Propager les figures de gating (dont fig_kde_cd45 regénéré)
                    # vers _mpl_figures — _mpl_figures est une copie shallow de
                    # gating_figures faite avant le MRD, donc les updates postérieurs
                    # sur gating_figures ne se reflètent pas automatiquement.
                    _mpl_figures.update(gating_figures)

                    # ── Radar MRD blast (Porte 2) — généré directement depuis les
                    # données de scoring disponibles dans le pipeline principal,
                    # indépendamment de population_mapping.enabled.
                    # Permet d'avoir le radar même si population_mapping est désactivé.
                    if (
                        viz_save
                        and mrd_result is not None
                        and "fig_mrd_blast_radar" not in _plotly_figures
                        and (_x_norm_arr is not None or _node_medians_arr is not None)
                        and _marker_names_list
                    ):
                        try:
                            from flowsom_pipeline_pro.src.analysis.blast_detection import (
                                build_blast_score_dataframe,
                                build_blast_weights,
                            )
                            from flowsom_pipeline_pro.src.prisma.visualization.population_viz import (
                                plot_mrd_blast_radar,
                            )

                            # Utiliser X_norm si disponible (normalisé vs référence NBM),
                            # sinon z-scorer node_medians intra-dataset
                            if _x_norm_arr is not None:
                                _blast_X = _x_norm_arr
                            else:
                                # Fallback : z-scoring intra-dataset sur node_medians
                                _m_arr = _node_medians_arr.astype(float)
                                _center = np.nanmean(_m_arr, axis=0)
                                _scale = np.nanstd(_m_arr, axis=0)
                                _scale[_scale == 0] = 1.0
                                _blast_X = (_m_arr - _center) / _scale

                            _n_nodes_blast = _blast_X.shape[0]
                            _node_ids_blast = np.arange(_n_nodes_blast)
                            _blast_weights = build_blast_weights(_marker_names_list)
                            _blast_df = build_blast_score_dataframe(
                                node_ids=_node_ids_blast,
                                X_norm=_blast_X,
                                marker_names=_marker_names_list,
                                weights=_blast_weights,
                            )
                            _mrd_per_node_list = (
                                mrd_result.per_node if mrd_result is not None else None
                            )
                            _fig_mrd_radar = plot_mrd_blast_radar(
                                blast_df=_blast_df,
                                mrd_per_node=_mrd_per_node_list,
                                output_dir=output_dir / "plots",
                                timestamp=timestamp,
                            )
                            if _fig_mrd_radar is not None:
                                _plotly_figures["fig_mrd_blast_radar"] = _fig_mrd_radar
                                _logger.info(
                                    "Radar MRD blast (Porte 2) généré depuis pipeline principal."
                                )
                        except Exception as _radar_exc:
                            _logger.warning(
                                "Radar MRD blast (pipeline principal) échoué (non bloquant): %s",
                                _radar_exc,
                            )

                    # Visualisation MRD
                    if viz_save and mrd_result is not None:
                        _viz_cfg_mrd = getattr(config, "visualization", None)
                        _export_png_mrd = getattr(_viz_cfg_mrd, "export_png_mrd", False)
                        self._enqueue_plot(
                            plot_worker,
                            "MRD Résiduelle",
                            "fig_mrd_summary",
                            "plotly",
                            "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                            "plot_mrd_summary",
                            mrd_result,
                            output_html=output_dir / "plots" / f"mrd_summary_{timestamp}.html",
                            output_png=output_dir / "plots" / f"mrd_summary_{timestamp}.png"
                            if _export_png_mrd
                            else None,
                        )

                    # Radars MRD — un par méthode (JF et Flo)
                    if mrd_result is not None and viz_save:
                        _jf_nodes = [n.cluster_id for n in mrd_result.per_node if n.is_mrd_jf]
                        _flo_nodes = [n.cluster_id for n in mrd_result.per_node if n.is_mrd_flo]
                        self._enqueue_plot(
                            plot_worker,
                            "Radar MRD JF",
                            "fig_cluster_radar_jf",
                            "plotly",
                            "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                            "plot_cluster_radar_mrd",
                            X_stacked,
                            clustering,
                            list(mfi_matrix.columns),
                            _jf_nodes,
                            output_dir / "plots" / f"cluster_radar_mrd_jf_{timestamp}.html",
                            title="Profils d'Expression — Clusters MRD",
                            method_label="JF",
                        )
                        self._enqueue_plot(
                            plot_worker,
                            "Radar MRD Flo",
                            "fig_cluster_radar_flo",
                            "plotly",
                            "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                            "plot_cluster_radar_mrd",
                            X_stacked,
                            clustering,
                            list(mfi_matrix.columns),
                            _flo_nodes,
                            output_dir / "plots" / f"cluster_radar_mrd_flo_{timestamp}.html",
                            title="Profils d'Expression — Clusters MRD",
                            method_label="Flo",
                        )

                    # Classification phénotypique blast des nœuds MRD
                    if mrd_result is not None and viz_save:
                        _export_png_blast = getattr(
                            getattr(config, "visualization", None),
                            "export_png_mrd",
                            False,
                        )
                        self._enqueue_plot(
                            plot_worker,
                            "Classification Blast MRD",
                            "fig_blast_mrd_classification",
                            "plotly",
                            "flowsom_pipeline_pro.src.prisma.visualization.flowsom_plots",
                            "plot_blast_mrd_classification",
                            mrd_result,
                            output_dir / "plots" / f"blast_mrd_classification_{timestamp}.html",
                            output_dir / "plots" / f"blast_mrd_classification_{timestamp}.png"
                            if _export_png_blast
                            else None,
                        )

                    # Export JSON des résultats MRD
                    if mrd_result is not None:
                        try:
                            import json

                            mrd_json_path = output_dir / f"mrd_results_{timestamp}.json"
                            mrd_json_path.write_text(
                                json.dumps(mrd_result.to_dict(), indent=2, ensure_ascii=False),
                                encoding="utf-8",
                            )
                            export_paths["mrd_results"] = str(mrd_json_path)
                            _logger.info("MRD JSON: %s", mrd_json_path.name)
                        except Exception as _je:
                            _logger.warning("Export JSON MRD échoué: %s", _je)
                else:
                    _logger.info("Étape 8: MRD désactivée ou pas de colonne condition.")
            except Exception as _mrd_exc:
                _logger.warning("Étape 8 MRD échouée (non bloquant): %s", _mrd_exc)

            # ── Étape 8b: Export FCS pathologique + Is_MRD ───────────────────
            _patho_fcs_cfg = getattr(config, "patho_fcs_export", None)
            if (
                _patho_fcs_cfg is not None
                and getattr(_patho_fcs_cfg, "enabled", False)
                and mrd_result is not None
                and df_fcs is not None
            ):
                try:
                    from flowsom_pipeline_pro.src.io.patho_fcs_exporter import (
                        export_patho_mrd_fcs,
                    )

                    _mrd_method = getattr(_patho_fcs_cfg, "mrd_method", "flo")
                    _patho_fcs_path = output_dir / "fcs" / f"patho_mrd_{timestamp}.fcs"
                    _patho_fcs_path.parent.mkdir(parents=True, exist_ok=True)

                    _logger.info(
                        "Étape 8b: Export FCS pathologique (Is_MRD=%s)...",
                        _mrd_method.upper(),
                    )
                    ok = export_patho_mrd_fcs(
                        df_fcs=df_fcs,
                        mrd_result=mrd_result,
                        output_path=_patho_fcs_path,
                        mrd_method=_mrd_method,
                    )
                    if ok:
                        export_paths["fcs_patho_mrd"] = str(_patho_fcs_path)
                        _logger.info("FCS pathologique: %s", _patho_fcs_path.name)
                except Exception as _pfe:
                    _logger.warning(
                        "Étape 8b export FCS pathologique échoué (non bloquant): %s",
                        _pfe,
                    )

            if plot_worker is not None:
                plot_worker.close_and_wait()
                _plot_results = plot_worker.get_results()
                _mpl_figures.update(_plot_results.get("mpl", {}))
                _plotly_figures.update(_plot_results.get("plotly", {}))
                for _err in plot_worker.get_errors():
                    _logger.warning("PlottingWorker: %s", _err)

            # ── Rapport HTML self-contained (APRÈS étape 8) ──────────────────
            if viz_save:
                if _monitor:
                    _monitor.mark_phase("Rapport HTML / PDF")
                try:
                    _logger.info("Génération du rapport HTML...")
                    mc_table = [
                        {
                            "metacluster": f"MC{int(mc)}",
                            "n_cells": int((metaclustering == mc).sum()),
                            "pct": round(float((metaclustering == mc).mean() * 100), 2),
                            "top_markers": ", ".join(
                                mfi_matrix.loc[f"MC{int(mc)}"].nlargest(3).index.tolist()
                            )
                            if f"MC{int(mc)}" in mfi_matrix.index
                            else "",
                        }
                        for mc in np.unique(metaclustering)
                    ]

                    # ── Données par condition ──────────────────────────────
                    condition_data: List[Dict] = []
                    if "condition" in df_cells.columns:
                        cond_counts = df_cells["condition"].value_counts()
                        total_cells = len(df_cells)
                        for cond_name, n_c in cond_counts.items():
                            condition_data.append(
                                {
                                    "condition": str(cond_name),
                                    "n_cells": int(n_c),
                                    "pct": round(float(n_c / total_cells * 100), 1),
                                }
                            )

                    # ── Données par fichier ────────────────────────────────
                    files_data: List[Dict] = []
                    file_col = next(
                        (c for c in ("file_origin", "File_Origin") if c in df_cells.columns),
                        None,
                    )
                    if file_col:
                        for fname, n_f in df_cells[file_col].value_counts().items():
                            files_data.append(
                                {
                                    "file": str(fname),
                                    "n_cells": int(n_f),
                                }
                            )

                    # ── Labels des figures ─────────────────────────────────
                    _FIGURE_LABELS = {
                        "fig_overview": "Vue d'ensemble (pré-gating)",
                        "fig_gate_debris": "QC Gate 1 — Exclusion Débris",
                        "fig_gate_singlets": "QC Gate 2 — Exclusion Doublets",
                        "fig_gate_cd45": "QC Gate 3 — Cellules CD45+",
                        "fig_gate_cd34": "QC Gate 4 — Blastes CD34+",
                        "fig_kde_debris": "QC Gate 1 — Densité KDE FSC-A (GMM vs KDE)",
                        "fig_kde_cd45": "QC Gate 3 — Densité KDE CD45 (pied du pic)",
                        "fig_cd45_count": "Comptage CD45+ Bruts (KDE pied du pic, sans pipeline)",
                        "fig_heatmap": "Heatmap MFI — Métaclusters × Marqueurs (Z-score)",
                        "fig_comp": "Distribution Cellules par Métacluster",
                        "fig_umap": "UMAP — Coloré par Métacluster FlowSOM",
                        "fig_mst_static": "MST Statique — Topologie FlowSOM",
                        "fig_harmony_diag": "Correction Harmony — PCA Avant / Après",
                        "fig_sankey": "Diagramme Sankey — Flux de Gating",
                        "fig_mst": "MST Interactif — Populations FlowSOM",
                        "fig_grid_mc": "Grille SOM ScatterGL",
                        "fig_heatmap_clinical": "Expression Phénotypique — Référence vs Métaclusters",
                        "fig_barplots": "Blast Score — Profil Marqueurs",
                        "fig_radar": "Profils Blast — Radar Charts",
                        "fig_cluster_radar": "Profils d'Expression par Cluster SOM (Radar)",
                        "fig_cluster_radar_jf": "Profils d'Expression — Clusters MRD Méthode JF (Radar)",
                        "fig_cluster_radar_flo": "Profils d'Expression — Clusters MRD Méthode Flo (Radar)",
                        "fig_stars": "Blast Scores — Bar Chart",
                        "fig_patho_pct": "% Cellules Pathologiques par Cluster",
                        "fig_cells_pct": "% Cellules par Cluster (Distribution Globale)",
                        "fig_mrd_summary": "MRD Résiduelle — Nœuds SOM (JF / Flo + contrôles ELN)",
                        "fig_blast_mrd_classification": "Classification Phénotypique des Nœuds MRD (ELN 2022)",
                        "fig_mrd_blast_radar": "Radar MRD — Nœuds Porte 2 (BLAST_HIGH / BLAST_MODERATE)",
                        "fig_prescreening_cd34": "Pré-screening CD34 — KDE minimum local (GMM vs KDE)",
                    }

                    _patho_info = None
                    if _patho_stem:
                        _patho_info = {
                            "name": _patho_stem,
                            "date": _patho_date or "Date inconnue",
                        }
                    html_path = exporter.export_html_report(
                        analysis_params={
                            "Transformation": config.transform.method,
                            "Normalisation": config.normalize.method,
                            "Grille SOM": f"{config.flowsom.xdim}×{config.flowsom.ydim}",
                            "Métaclusters": n_meta,
                            "Seed": config.flowsom.seed,
                        },
                        summary_stats={
                            "n_cells": int(X_stacked.shape[0]),
                            "n_markers": len(selected_markers),
                            "n_files": len(samples),
                            "n_clusters": n_meta,
                        },
                        metacluster_table=mc_table,
                        markers=list(selected_markers),
                        matplotlib_figures=_mpl_figures,
                        plotly_figures=_plotly_figures,
                        figure_labels=_FIGURE_LABELS,
                        condition_data=condition_data,
                        files_data=files_data,
                        export_paths=export_paths,
                        patho_info=_patho_info,
                        ransac_summary=dict(ransac_scatter_data) if ransac_scatter_data else None,
                        prescreening_result=_prescreening_result,
                    )
                    if html_path:
                        export_paths["html_report"] = html_path
                        _logger.info("Rapport HTML: %s", html_path)
                except Exception as _e:
                    _logger.warning("Rapport HTML échoué (non bloquant): %s", _e)

                # ── Rapport PDF A4 ────────────────────────────────────────────
                _viz_cfg_pdf = getattr(config, "visualization", None)
                _export_pdf_enabled = getattr(_viz_cfg_pdf, "export_pdf", True)
                if not _export_pdf_enabled:
                    _logger.info(
                        "Rapport PDF désactivé via config (visualization.export_pdf=false)."
                    )
                else:
                    try:
                        pdf_path = exporter.export_pdf_report(
                            analysis_params={
                                "Transformation": config.transform.method,
                                "Normalisation": config.normalize.method,
                                "Grille SOM": f"{config.flowsom.xdim}×{config.flowsom.ydim}",
                                "Métaclusters": n_meta,
                                "Seed": config.flowsom.seed,
                            },
                            summary_stats={
                                "n_cells": int(X_stacked.shape[0]),
                                "n_markers": len(selected_markers),
                                "n_files": len(samples),
                                "n_clusters": n_meta,
                            },
                            metacluster_table=mc_table,
                            markers=list(selected_markers),
                            matplotlib_figures=_mpl_figures,
                            plotly_figures=_plotly_figures,
                            figure_labels=_FIGURE_LABELS,
                            condition_data=condition_data,
                            files_data=files_data,
                            export_paths=export_paths,
                            patho_info=_patho_info,
                            ransac_summary=dict(ransac_scatter_data)
                            if ransac_scatter_data
                            else None,
                            prescreening_result=_prescreening_result,
                        )
                        if pdf_path:
                            export_paths["pdf_report"] = pdf_path
                            _logger.info("Rapport PDF: %s", pdf_path)
                    except Exception as _e:
                        _logger.warning("Rapport PDF échoué (non bloquant): %s", _e)

            # ── Export dashboard de performance ───────────────────────────────
            if _monitor:
                try:
                    _monitor.mark_phase("Terminé")
                    _monitor.stop()
                    _dash_path = _monitor.export_dashboard(
                        Path(config.paths.output_dir)
                        / "other"
                        / f"performance_dashboard_{timestamp}.html"
                    )
                    if _dash_path:
                        export_paths["performance_dashboard"] = _dash_path
                        _logger.info("Dashboard performance : %s", _dash_path)
                except Exception as _me:
                    _logger.debug("Export dashboard performance échoué: %s", _me)

            # ── Analyse Citrus (optionnelle, cohorte multi-samples) ───────────
            _citrus_result = None
            _citrus_cfg = getattr(config, "citrus", None)
            if _citrus_cfg is not None and getattr(_citrus_cfg, "enabled", False):
                try:
                    from flowsom_pipeline_pro.src.analysis.citrus_strategy import (
                        CitrusStrategy,
                        CitrusParams,
                    )
                    from flowsom_pipeline_pro.src.prisma.core.models_legacy.experiment import Experiment
                    from flowsom_pipeline_pro.src.prisma.core.models_legacy.sample import Sample

                    _logger.info("Citrus: démarrage analyse stratifiante...")
                    _citrus_exp = Experiment("citrus_cohort")
                    _citrus_endpoints = []
                    _endpoint_col = getattr(_citrus_cfg, "endpoint_column", "group")

                    for _fs in samples:
                        _ep_val = getattr(_fs, "metadata", {}).get(_endpoint_col)
                        if _ep_val is None:
                            continue
                        _markers_used = getattr(_fs, "markers", selected_markers)
                        _df_s = df_cells[df_cells.get("sample_id", pd.Series()) == getattr(_fs, "sample_id", None)] if "sample_id" in df_cells.columns else pd.DataFrame()
                        if _df_s.empty and hasattr(_fs, "data"):
                            _df_s = _fs.data
                        if _df_s is None or (hasattr(_df_s, "empty") and _df_s.empty):
                            continue
                        _s = Sample(
                            sample_id=str(getattr(_fs, "sample_id", id(_fs))),
                            events=_df_s[[c for c in _markers_used if c in _df_s.columns]] if _markers_used else _df_s,
                        )
                        _citrus_exp.add_sample(_s)
                        _citrus_endpoints.append(float(_ep_val) if getattr(_citrus_cfg, "endpoint_type", "classification") == "continuous" else int(_ep_val))

                    if len(_citrus_exp.samples) >= 2 and len(_citrus_endpoints) == len(_citrus_exp.samples):
                        _cp = CitrusParams(
                            n_cells_per_sample=getattr(_citrus_cfg, "n_cells_per_sample", 1000),
                            min_cluster_size_percent=getattr(_citrus_cfg, "min_cluster_size_percent", 0.05),
                            feature_type=getattr(_citrus_cfg, "feature_type", "abundances"),
                            model_type=getattr(_citrus_cfg, "model_type", "glmnet"),
                            endpoint_type=getattr(_citrus_cfg, "endpoint_type", "classification"),
                            n_cv_folds=getattr(_citrus_cfg, "n_cv_folds", 5),
                        )
                        _citrus_result = CitrusStrategy(_cp).run_experiment(
                            _citrus_exp, np.array(_citrus_endpoints)
                        )
                        _logger.info(
                            "Citrus: %d clusters stratifiants (score=%.3f)",
                            len(_citrus_result.stratifying_clusters),
                            _citrus_result.model_score,
                        )
                    else:
                        _logger.warning(
                            "Citrus: insuffisant (%d samples valides) — ignoré",
                            len(_citrus_exp.samples),
                        )
                except Exception as _ce:
                    _logger.warning("Citrus ignoré (erreur): %s", _ce)

            # ── Assemblage du résultat ────────────────────────────────────────
            elapsed = time.time() - start_time

            result = PipelineResult(
                data=df_cells,
                raw_data=_raw_data_df,
                mfi_matrix=mfi_matrix,
                node_mfi_matrix=node_mfi_matrix,
                gating_report=[e.to_dict() for e in self._gating_logger.events],
                clustering_metrics=metrics,
                output_files=export_paths,
                config_snapshot=config.to_dict() if hasattr(config, "to_dict") else {},
                timestamp=timestamp,
                elapsed_seconds=elapsed,
                population_mapping=population_mapping_result,
                mrd_result=mrd_result,
                patho_stem=_patho_stem,
                patho_date=_patho_date,
                prescreening_result=_prescreening_result,
                citrus_result=_citrus_result,
            )

            _report(PipelineStep.DONE)
            _logger.info("=" * 60)
            _logger.info("PIPELINE TERMINÉ EN %.1fs", elapsed)
            _logger.info("  Cellules: %d", result.n_cells)
            _logger.info("  Marqueurs: %d", len(selected_markers))
            _logger.info("  Métaclusters: %d", result.n_metaclusters)
            _logger.info("=" * 60)

            self._result = result
            return result

        except Exception as exc:
            try:
                if "plot_worker" in locals() and plot_worker is not None:
                    plot_worker.close_and_wait()
            except Exception:
                pass
            # Arrêter le monitor proprement même en cas d'erreur
            if _monitor:
                try:
                    _monitor.stop()
                except Exception:
                    pass
            _logger.exception("Erreur critique dans le pipeline: %s", exc)
            # Sécurité : écrire l'erreur dans un fichier pour le mode frozen
            # (logging peut échouer si sys.stderr est None)
            try:
                import traceback as _tb

                err_path = Path(config.paths.output_dir) / "pipeline_error.log"
                err_path.parent.mkdir(parents=True, exist_ok=True)
                err_path.write_text(
                    f"{type(exc).__name__}: {exc}\n\n{''.join(_tb.format_exception(exc))}",
                    encoding="utf-8",
                )
            except Exception:
                pass
            return PipelineResult.failure(error=str(exc), config=config)

    def _load_all_samples(self) -> List[FlowSample]:
        """
        Charge tous les fichiers FCS depuis les dossiers configurés.

        Returns:
            Liste de FlowSample.
        """
        config = self.config
        samples: List[FlowSample] = []

        # Fichiers sains (NBM)
        healthy_folder = Path(getattr(config.paths, "healthy_folder", ""))
        if healthy_folder.exists():
            healthy_files = get_fcs_files(healthy_folder)
            healthy_samples = load_as_flow_samples(healthy_files, condition="Sain")
            samples.extend(healthy_samples)
            _logger.info("  Sain (NBM): %d fichiers", len(healthy_samples))

        # Fichiers pathologiques
        # Mode batch : un seul fichier patho spécifié directement
        _single_patho = getattr(config.paths, "patho_single_file", None)
        patho_folder = Path(getattr(config.paths, "patho_folder", ""))
        if _single_patho and Path(_single_patho).is_file():
            patho_files = [Path(_single_patho)]
            _logger.info("  Pathologique (batch): %s", Path(_single_patho).name)
        elif patho_folder.exists():
            patho_files = get_fcs_files(patho_folder)
        else:
            patho_files = []
        if patho_files:
            patho_samples = load_as_flow_samples(patho_files, condition="Pathologique")
            samples.extend(patho_samples)
            _logger.info("  Pathologique: %d fichiers", len(patho_samples))

        # Dossier unique (mode simplifié)
        if not samples:
            data_folder = Path(getattr(config.paths, "data_folder", ""))
            if data_folder.exists():
                all_files = get_fcs_files(data_folder)
                all_samples = load_as_flow_samples(all_files, condition="Unknown")
                samples.extend(all_samples)
                _logger.info("  Dossier unique: %d fichiers", len(all_samples))

        # ── Renommage manuel des colonnes FCS (depuis la GUI) ──────────────
        # Appliqué avant toute harmonisation automatique pour permettre à
        # l'utilisateur de mapper ses noms Kaluza vers les noms cibles.
        col_rename_map: dict = dict(
            getattr(config, "_extra", {}).get("column_rename_map", {}) or {}
        )
        if col_rename_map:
            n_renamed = 0
            for s in samples:
                # Ne renommer que les colonnes réellement présentes
                effective = {
                    src: dst
                    for src, dst in col_rename_map.items()
                    if src in s.data.columns and src != dst
                }
                if effective:
                    s.data.rename(columns=effective, inplace=True)
                    n_renamed += len(effective)
            if n_renamed:
                _logger.info(
                    "Renommage GUI : %d colonne(s) renommée(s) sur %d fichier(s) (mapping: %s)",
                    n_renamed,
                    len(samples),
                    col_rename_map,
                )

        return samples

    # ------------------------------------------------------------------
    # FCS export helpers (style monolithe flowsom_pipeline.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _circular_jitter(
        n_points: int,
        cluster_ids: np.ndarray,
        node_sizes: np.ndarray,
        max_radius: float = 0.45,
        min_radius: float = 0.1,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Jitter circulaire style FlowSOM R.

        Le rayon de chaque cellule dépend de la taille de son node SOM.
        On utilise sqrt(u) pour une distribution uniforme dans le disque.
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

    def _build_fcs_dataframe(
        self,
        processed_samples: List[FlowSample],
        metaclustering: np.ndarray,
        clustering: np.ndarray,
        clusterer: "FlowSOMClusterer",
    ) -> pd.DataFrame:
        """
        Construit le DataFrame complet pour l'export FCS compatible Kaluza.

        Reproduit EXACTEMENT la logique de flowsom_pipeline.py :
          1. Données brutes (pré-transformation) pour TOUS les marqueurs
          2. FlowSOM_metacluster (1-based), FlowSOM_cluster (1-based)
          3. xGrid, yGrid avec jitter circulaire (grille SOM)
          4. xNodes, yNodes avec jitter circulaire (MST)
          5. size (nb cellules par node)
          6. Condition, Condition_Num, File_Origin

        Returns:
            DataFrame avec toutes les colonnes numériques prêtes pour fcswrite.
        """
        from flowsom_pipeline_pro.src.prisma.core.clustering import FlowSOMClusterer as _FSC

        # ── 1. Données brutes (toutes colonnes) ───────────────────────────────
        X_raw, raw_var_names, obs = stack_raw_markers(processed_samples)

        n_cells = X_raw.shape[0]
        n_nodes = clusterer.n_nodes

        # ── 2. Taille de chaque node ───────────────────────────────────────────
        node_sizes = clusterer.get_node_sizes()

        # ── 3. Coordonnées grille SOM avec jitter circulaire ──────────────────
        grid_coords = clusterer.get_grid_coords()  # (n_nodes, 2)

        cl_int = clustering.astype(int)
        xGrid_base = grid_coords[cl_int, 0].astype(np.float32)
        yGrid_base = grid_coords[cl_int, 1].astype(np.float32)

        jitter_x, jitter_y = self._circular_jitter(
            n_cells, cl_int, node_sizes, max_radius=0.45, min_radius=0.1
        )
        xGrid_j = xGrid_base + jitter_x
        yGrid_j = yGrid_base + jitter_y
        # Décaler pour que min = 1 (comme le monolithe)
        xGrid = xGrid_j - xGrid_j.min() + 1.0
        yGrid = yGrid_j - yGrid_j.min() + 1.0

        _logger.info("  xGrid: [%.3f – %.3f]", xGrid.min(), xGrid.max())
        _logger.info("  yGrid: [%.3f – %.3f]", yGrid.min(), yGrid.max())

        # ── 4. Coordonnées MST avec jitter circulaire ─────────────────────────
        layout_coords = clusterer.get_layout_coords()  # (n_nodes, 2)
        x_ptp = float(layout_coords[:, 0].max() - layout_coords[:, 0].min())
        y_ptp = float(layout_coords[:, 1].max() - layout_coords[:, 1].min())
        x_range = x_ptp if x_ptp > 0 else 1.0
        y_range = y_ptp if y_ptp > 0 else 1.0
        mst_scale = min(x_range, y_range) / (clusterer.xdim * 2)

        xN_base = layout_coords[cl_int, 0].astype(np.float32)
        yN_base = layout_coords[cl_int, 1].astype(np.float32)
        mst_jx, mst_jy = self._circular_jitter(
            n_cells,
            cl_int,
            node_sizes,
            max_radius=mst_scale * 0.8,
            min_radius=mst_scale * 0.2,
        )
        xNodes_j = xN_base + mst_jx
        yNodes_j = yN_base + mst_jy
        xNodes = xNodes_j - xNodes_j.min() + 1.0
        yNodes = yNodes_j - yNodes_j.min() + 1.0

        _logger.info("  xNodes: [%.3f – %.3f]", xNodes.min(), xNodes.max())
        _logger.info("  yNodes: [%.3f – %.3f]", yNodes.min(), yNodes.max())

        # ── 5. Assemblage vectorisé (un seul pd.DataFrame call) ──────────────
        # Construire le dict NumPy complet AVANT pd.DataFrame pour éviter
        # les 13 affectations colonne par colonne (chacune copie 2M lignes).
        condition_arr = obs["condition"].values
        col_dict: dict = {name: X_raw[:, i] for i, name in enumerate(raw_var_names)}
        col_dict["FlowSOM_metacluster"] = (metaclustering + 1).astype(np.float32)
        col_dict["FlowSOM_cluster"] = (clustering + 1).astype(np.float32)
        col_dict["xGrid"] = xGrid.astype(np.float32)
        col_dict["yGrid"] = yGrid.astype(np.float32)
        col_dict["xNodes"] = xNodes.astype(np.float32)
        col_dict["yNodes"] = yNodes.astype(np.float32)
        col_dict["size"] = node_sizes[cl_int].astype(np.float32)
        col_dict["Condition"] = condition_arr
        col_dict["Condition_Num"] = np.where(
            condition_arr == "Sain", np.float32(1.0), np.float32(2.0)
        )
        col_dict["File_Origin"] = obs["file_origin"].values
        df = pd.DataFrame(col_dict)

        _logger.info(
            "  DataFrame FCS complet: %d cellules × %d colonnes",
            df.shape[0],
            df.shape[1],
        )
        return df

    def _compute_metrics(
        self,
        X: np.ndarray,
        metaclustering: np.ndarray,
    ) -> ClusteringMetrics:
        """
        Calcule les métriques de qualité du clustering.

        Args:
            X: Matrice de données (n_cells, n_markers).
            metaclustering: Assignation par cellule.

        Returns:
            ClusteringMetrics.
        """
        n_metaclusters = int(metaclustering.max()) + 1
        silhouette = None

        try:
            from sklearn.metrics import silhouette_score

            # Calculer le silhouette sur un sous-échantillon (rapide)
            rng = np.random.default_rng(42)
            n_sample = min(10_000, X.shape[0])
            idx = rng.choice(X.shape[0], n_sample, replace=False)
            if len(np.unique(metaclustering[idx])) >= 2:
                silhouette = float(silhouette_score(X[idx], metaclustering[idx]))
                _logger.info("  Silhouette score: %.4f", silhouette)
        except ImportError:
            pass
        except Exception as e:
            _logger.warning("Silhouette score non calculé: %s", e)

        counts = np.bincount(metaclustering.astype(int), minlength=n_metaclusters)
        n_per_cluster = {i: int(counts[i]) for i in range(n_metaclusters)}

        return ClusteringMetrics(
            n_metaclusters=n_metaclusters,
            silhouette_score=silhouette,
            n_cells_per_cluster=n_per_cluster,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Fonction standalone — compatibilité avec flowsom_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────


def run_flowsom_pipeline(
    config: Optional[PipelineConfig] = None,
    **kwargs,
) -> PipelineResult:
    """
    Point d'entrée fonctionnel du pipeline FlowSOM.

    Équivalent à ``FlowSOMPipeline(config).execute()`` mais sous forme de
    fonction standalone, selon le style de ``flowsom_pipeline.py``.

    Exemples d'utilisation::

        # Depuis un PipelineConfig existant
        result = run_flowsom_pipeline(config)

        # Depuis un dict de paramètres kwargs (création implicite de config)
        result = run_flowsom_pipeline(
            healthy_folder="/data/NBM",
            patho_folder="/data/LAM",
            xdim=10, ydim=10,
            n_metaclusters=12,
        )

    Args:
        config: PipelineConfig complet. Si None, un config est construit
                depuis les kwargs fournis.
        **kwargs: Paramètres nommés transmis à PipelineConfig (ignorés si
                  config est fourni explicitement).

    Returns:
        PipelineResult avec ``success``, ``n_cells``, ``n_metaclusters``,
        ``output_dir`` et la méthode ``summary()``.

    Raises:
        ValueError: Si config est None et qu'aucun healthy_folder n'est fourni.
    """
    if config is None:
        if not kwargs:
            raise ValueError(
                "run_flowsom_pipeline() requiert soit un PipelineConfig, "
                "soit des kwargs de configuration (ex. healthy_folder=…)."
            )
        config = PipelineConfig.from_dict(kwargs)

    pipeline = FlowSOMPipeline(config=config)
    return pipeline.execute()


# ══════════════════════════════════════════════════════════════════════════════
# BatchPipeline — traitement par lots (Feature 2, 3, 4)
# ══════════════════════════════════════════════════════════════════════════════


class BatchPipeline:
    """
    Exécute le pipeline FlowSOM sur chaque fichier FCS du dossier patho.

    Pour chaque fichier :
      - crée un dossier résultats_<stem> dans output_dir
      - nomme les rapports HTML/PDF d'après le stem du fichier
      - collecte les résultats MRD et métriques dans un Excel de synthèse

    Usage:
        batch = BatchPipeline(config)
        summary = batch.execute(progress_callback=lambda i, n, name: ...)
    """

    def __init__(self, config: "PipelineConfig") -> None:
        from copy import deepcopy

        self.config = deepcopy(config)

    def execute(self, progress_callback=None) -> dict:
        """
        Lance le pipeline sur tous les FCS du dossier patho.

        Args:
            progress_callback: callable(current: int, total: int, filename: str)

        Returns:
            dict avec :
              - "results"  : List[(stem, PipelineResult)]
              - "excel"    : chemin du fichier Excel de synthèse (str|None)
        """
        from copy import deepcopy

        patho_folder = Path(self.config.paths.patho_folder)
        patho_files = get_fcs_files(patho_folder)

        if not patho_files:
            _logger.warning("Batch: aucun fichier FCS trouvé dans %s", patho_folder)
            return {"results": [], "excel": None}

        _logger.info("═" * 60)
        _logger.info("BATCH — %d fichier(s) à traiter", len(patho_files))
        _logger.info("═" * 60)

        results: List[Tuple[str, PipelineResult]] = []

        for i, fcs_path in enumerate(patho_files, start=1):
            _logger.info("─" * 60)
            _logger.info("BATCH [%d/%d] : %s", i, len(patho_files), fcs_path.name)
            _logger.info("─" * 60)

            if progress_callback:
                progress_callback(i - 1, len(patho_files), fcs_path.name)

            cfg = deepcopy(self.config)
            cfg.paths.patho_single_file = str(fcs_path)
            # Désactiver le mode batch dans le clone pour éviter la récursion
            cfg.batch.enabled = False

            try:
                pipeline = FlowSOMPipeline(cfg)
                result = pipeline.execute()
            except Exception as exc:
                _logger.error("Batch [%s] échoué: %s", fcs_path.name, exc)
                result = PipelineResult.failure(error=str(exc), config=cfg)
                result.patho_stem = fcs_path.stem

            results.append((fcs_path.stem, result))

            # Sauvegarde incrémentale après chaque patient — protège contre les crashs
            self._generate_synthesis_excel(results)
            _logger.info(
                "BATCH [%d/%d] Excel intermédiaire mis à jour (%d patient(s))",
                i,
                len(patho_files),
                len(results),
            )

        if progress_callback:
            progress_callback(len(patho_files), len(patho_files), "Synthèse Excel...")

        excel_path = self._generate_synthesis_excel(results)
        _logger.info("═" * 60)
        _logger.info("BATCH TERMINÉ — %d fichier(s) traités", len(results))
        if excel_path:
            _logger.info("Excel de synthèse : %s", excel_path)
        _logger.info("═" * 60)

        return {"results": results, "excel": excel_path}

    def _generate_synthesis_excel(
        self, results: List[Tuple[str, "PipelineResult"]]
    ) -> Optional[str]:
        """Génère un Excel de synthèse global pour toute la cohorte."""
        try:
            from flowsom_pipeline_pro.src.prisma.io.csv_exporter import (
                extract_date_from_filename as _edf,
            )

            rows = []
            for stem, result in results:
                # Priorité 1 : date calculée et stockée dans PipelineResult.patho_date
                # (extraite du contenu FCS ou du nom du fichier lors du run)
                # Priorité 2 : tentative d'extraction depuis le stem (fallback)
                _fcs_date_str = getattr(result, "patho_date", None)
                if not _fcs_date_str:
                    _fcs_dt = _edf(stem)
                    _fcs_date_str = _fcs_dt.strftime("%Y-%m-%d") if _fcs_dt else ""
                row: Dict[str, object] = {
                    "Fichier FCS": stem,
                    "Date FCS": _fcs_date_str,
                    "Statut": "OK" if (result is not None and result.success) else "ERREUR",
                }

                if result is None or not result.success:
                    err = result.warnings[0] if (result and result.warnings) else "Erreur inconnue"
                    row["Erreur"] = err
                    rows.append(row)
                    continue

                row["Cellules totales"] = result.n_cells
                row["Durée (s)"] = round(result.elapsed_seconds, 1)

                # MRD
                mrd = result.mrd_result
                if mrd is not None:
                    n_total = mrd.total_cells or 1
                    # Cellules par condition
                    row["Cellules patho"] = mrd.total_cells_patho
                    row["Cellules sain"] = mrd.total_cells_sain
                    row["% patho"] = round(mrd.total_cells_patho / n_total * 100, 2)
                    # MRD % — valeurs déjà en % dans MRDResult
                    row["MRD % (Flo)"] = round(mrd.mrd_pct_flo, 4)
                    row["MRD % (JF)"] = round(mrd.mrd_pct_jf, 4)
                    row["MRD % (ELN)"] = round(mrd.mrd_pct_eln, 4)
                    # Cellules MRD détectées
                    row["Cellules MRD (Flo)"] = mrd.mrd_cells_flo
                    row["Cellules MRD (JF)"] = mrd.mrd_cells_jf
                    row["Cellules MRD (ELN)"] = mrd.mrd_cells_eln
                    # Nœuds SOM positifs
                    row["Nœuds MRD (Flo)"] = mrd.n_nodes_mrd_flo
                    row["Nœuds MRD (JF)"] = mrd.n_nodes_mrd_jf
                    row["Nœuds MRD (ELN)"] = mrd.n_nodes_mrd_eln
                    # Statut ELN
                    row["ELN positif"] = "Oui" if mrd.eln_positive else "Non"
                    row["ELN bas niveau"] = "Oui" if mrd.eln_low_level else "Non"
                else:
                    # Cellules par condition (si pas de MRD calculé)
                    if result.data is not None and "condition" in result.data.columns:
                        cond_counts = result.data["condition"].value_counts()
                        for cond, n_c in cond_counts.items():
                            row[f"Cellules {cond}"] = int(n_c)

                # Rapport HTML généré
                html = result.output_files.get("html_report", "")
                row["Rapport HTML"] = str(html) if html else ""

                rows.append(row)

            if not rows:
                return None

            df_synthesis = pd.DataFrame(rows)
            output_dir = Path(self.config.paths.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            excel_path = output_dir / "synthèse_cohorte.xlsx"

            with pd.ExcelWriter(str(excel_path), engine="openpyxl") as writer:
                df_synthesis.to_excel(writer, index=False, sheet_name="Synthèse")
                # Ajuster la largeur des colonnes automatiquement
                ws = writer.sheets["Synthèse"]
                for col in ws.columns:
                    max_len = max(
                        (len(str(cell.value)) for cell in col if cell.value is not None),
                        default=10,
                    )
                    ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

            _logger.info("Excel de synthèse: %s", excel_path)
            return str(excel_path)

        except Exception as exc:
            _logger.warning("Génération Excel de synthèse échouée: %s", exc)
            return None
