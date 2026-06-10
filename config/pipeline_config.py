"""
pipeline_config.py — Dataclass centralisée de configuration du pipeline.

Chargement en cascade :
  1. Valeurs par défaut (constants.py)
  2. default_config.yaml (fourni avec le package)
  3. Fichier YAML utilisateur (chemin explicite ou auto-détecté)
  4. Arguments CLI (priorité maximale)

Usage:
    config = PipelineConfig.from_yaml("my_config.yaml")
    config = PipelineConfig.from_args(args)   # args = argparse.Namespace
    config = PipelineConfig()                  # valeurs par défaut
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any

from .constants import (
    DEFAULT_XDIM,
    DEFAULT_YDIM,
    DEFAULT_N_METACLUSTERS,
    DEFAULT_SEED,
    DEFAULT_RLEN,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SIGMA,
    DEFAULT_N_ITERATIONS,
    DEFAULT_TRANSFORM_METHOD,
    DEFAULT_ARCSINH_COFACTOR,
    DEFAULT_NORMALIZE_METHOD,
    DEFAULT_MAX_CELLS_PER_FILE,
    DEFAULT_MAX_CELLS_TOTAL,
    DEFAULT_MIN_CLUSTERS,
    DEFAULT_MAX_CLUSTERS,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_SAMPLE_SIZE_BOOTSTRAP,
    DEFAULT_MIN_STABILITY_THRESHOLD,
    DEFAULT_W_STABILITY,
    DEFAULT_W_SILHOUETTE,
    DEFAULT_DEBRIS_MIN_PCT,
    DEFAULT_DEBRIS_MAX_PCT,
    DEFAULT_SINGLETS_RATIO_MIN,
    DEFAULT_SINGLETS_RATIO_MAX,
    DEFAULT_CD45_THRESHOLD_PCT,
    DEFAULT_CD34_THRESHOLD_PCT,
    DEFAULT_CD34_SSC_MAX_PCT,
    DEFAULT_PLOT_FORMAT,
    DEFAULT_DPI,
    GMM_MAX_SAMPLES,
    RANSAC_R2_THRESHOLD,
    RANSAC_MAD_FACTOR,
)


@dataclass
class PathsConfig:
    healthy_folder: str = "Data/Moelle normale"
    patho_folder: str = "Data/Patho"
    output_dir: str = "Results"
    # Mode batch : fichier patho unique pour ce run (défini par BatchPipeline)
    patho_single_file: Optional[str] = None


@dataclass
class AnalysisConfig:
    compare_mode: bool = True


@dataclass
class PregateConfig:
    apply: bool = True
    mode: str = "auto"  # "auto" | "manual"
    mode_blastes_vs_normal: bool = True
    viable: bool = True
    singlets: bool = True
    cd45: bool = True
    cd34: bool = False
    # Paramètres avancés
    debris_min_percentile: float = DEFAULT_DEBRIS_MIN_PCT
    debris_max_percentile: float = DEFAULT_DEBRIS_MAX_PCT
    doublets_ratio_min: float = DEFAULT_SINGLETS_RATIO_MIN
    doublets_ratio_max: float = DEFAULT_SINGLETS_RATIO_MAX
    cd45_threshold_percentile: float = DEFAULT_CD45_THRESHOLD_PCT
    cd34_threshold_percentile: float = DEFAULT_CD34_THRESHOLD_PCT
    cd34_use_ssc_filter: bool = True
    cd34_ssc_max_percentile: float = DEFAULT_CD34_SSC_MAX_PCT
    # Paramètres GMM (sous-échantillonnage avant fit)
    gmm_max_samples: int = GMM_MAX_SAMPLES
    # Paramètres RANSAC singlets
    ransac_r2_threshold: float = RANSAC_R2_THRESHOLD
    ransac_mad_factor: float = RANSAC_MAD_FACTOR
    # Méthode d'estimation de densité : "GMM" | "KDE"
    density_method: str = "GMM"
    # Paramètres avancés GMM
    gmm_covariance_type: str = "full"  # "full" | "tied" | "diag" | "spherical"
    gmm_n_components_debris: int = 3  # Nombre de composantes pour le gating débris
    gmm_export_plot: bool = True  # Exporter le graphique des densités GMM
    # Paramètres KDE 1D pour le gating CD45 (méthode pied du pic)
    kde_cd45_seuil_relatif: float = 0.05  # Fraction du max densité pour le pied du pic
    kde_cd45_finesse: float = 0.6  # Facteur bandwidth Silverman
    kde_cd45_sigma_smooth: int = 10  # Lissage gaussien sur la courbe KDE (sigma)
    kde_cd45_n_grid: int = 1000  # Résolution de la grille KDE
    kde_cd45_max_samples: int = 10000  # Sous-échantillonnage max pour KDE CD45
    # Mode d'autogating CD45 pour le calcul MRD :
    #   "none"     → pas de gate CD45 (dénominateur = toutes cellules patho)
    #   "cd45"     → gate CD45+ standard (dénominateur = cellules patho CD45+)
    #   "cd45_dim" → gate CD45 incluant les blastes CD45-dim (dénominateur = cellules patho CD45+)
    cd45_autogating_mode: str = "none"  # "none" | "cd45" | "cd45_dim"
    # ── Pré-screening CD34+/CD45dim ────────────────────────────────────────────
    # Méthode de référence pour le gating CD34 dans le pré-screening heuristique.
    # "KDE" (défaut) : Kernel Density Estimation 1D — non-paramétrique, robuste.
    # "GMM"          : Gaussian Mixture Model 2 composantes.
    # Les deux méthodes sont toujours calculées et affichées dans le rapport HTML.
    cd34_cd45dim_density_method: str = "KDE"  # "KDE" | "GMM"


@dataclass
class FlowSOMConfig:
    xdim: int = DEFAULT_XDIM
    ydim: int = DEFAULT_YDIM
    rlen: Any = DEFAULT_RLEN  # "auto" ou entier
    n_metaclusters: int = DEFAULT_N_METACLUSTERS
    learning_rate: float = DEFAULT_LEARNING_RATE
    sigma: float = DEFAULT_SIGMA
    n_iterations: int = DEFAULT_N_ITERATIONS
    seed: int = DEFAULT_SEED


@dataclass
class AutoClusteringConfig:
    enabled: bool = False
    min_clusters: int = DEFAULT_MIN_CLUSTERS
    max_clusters: int = DEFAULT_MAX_CLUSTERS
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    sample_size_bootstrap: int = DEFAULT_SAMPLE_SIZE_BOOTSTRAP
    min_stability_threshold: float = DEFAULT_MIN_STABILITY_THRESHOLD
    weight_stability: float = DEFAULT_W_STABILITY
    weight_silhouette: float = DEFAULT_W_SILHOUETTE


@dataclass
class TransformConfig:
    method: str = DEFAULT_TRANSFORM_METHOD
    cofactor: float = DEFAULT_ARCSINH_COFACTOR
    apply_to_scatter: bool = False
    # Paramètres logicle globaux (utilisés si method == "logicle")
    logicle_T: float = 262144.0
    logicle_M: float = 4.5
    logicle_W: float = 0.5
    logicle_A: float = 0.0
    # Transformations par colonne (priorité sur method global si non vide).
    # Format : {nom_colonne: {"method","cofactor","T","M","W","A"}}
    per_column_specs: dict = field(default_factory=dict)


@dataclass
class NormalizeConfig:
    method: str = DEFAULT_NORMALIZE_METHOD


@dataclass
class MarkersConfig:
    exclude_scatter: bool = True
    exclude_additional: List[str] = field(default_factory=list)
    # Supprime les marqueurs -H (Height) quand le doublon -A (Area) existe.
    # Recommandé : réduit la colinéarité et accélère le SOM.
    keep_area_only: bool = True


@dataclass
class DownsamplingConfig:
    enabled: bool = True
    max_cells_per_file: int = DEFAULT_MAX_CELLS_PER_FILE
    max_cells_total: int = DEFAULT_MAX_CELLS_TOTAL


@dataclass
class HarmonyParamsConfig:
    sigma: float = 0.05
    nclust: Optional[int] = 30  # None = auto (N/30, très lent sur grands datasets)
    block_size: float = 0.20  # Fraction de cellules par bloc (défaut harmonypy: 0.05 = 20 blocs)
    max_iter: int = 10
    max_iter_kmeans: int = 10  # Itérations K-means internes (défaut harmonypy: 20)
    verbose: bool = False
    # Liste des marqueurs à aligner avec Harmony.
    # Liste vide = aligner tous les marqueurs du clustering (comportement historique).
    markers_to_align: List[str] = field(default_factory=list)


@dataclass
class DataIntegrationConfig:
    """Configuration de l'intégration de données (correction d'effet batch via Harmony)."""

    enabled: bool = True
    method: str = "harmony"  # "harmony" — extensible à "scanorama", "combat", etc.
    harmony_params: HarmonyParamsConfig = field(default_factory=HarmonyParamsConfig)


@dataclass
class StratifiedDownsamplingConfig:
    """
    Déséquilibre Maîtrisé — rééquilibrage du pool d'entraînement FlowSOM.

    Résout le problème d'invisibilité des clusters rares (blastes <1%) en
    forçant un rapport sain/patho contrôlé avant le SOM.

    Attributs:
        balance_conditions: Active le rééquilibrage. Si False, aucun effet.
        imbalance_ratio: Rapport cible n_sain / n_patho.
            1.0 = équilibre parfait (50/50)
            2.0 = 2 sains pour 1 blaste
        nbm_ids: Liste de FlowSample.name des fichiers NBM à utiliser comme
            source de cellules saines. Si vide, tous les fichiers sains sont
            utilisés (réparti équitablement).
        seed: Graine aléatoire pour la reproductibilité.
        allow_oversampling: Si True, le pool NBM est rééchantillonné avec
            remplacement pour atteindre la cible quand les cellules disponibles
            sont insuffisantes. Si False (défaut), le tirage est limité au
            nombre de cellules disponibles et un WARNING est émis.
    """

    balance_conditions: bool = False
    imbalance_ratio: float = 2.0
    nbm_ids: List[str] = field(default_factory=list)
    seed: int = 42
    allow_oversampling: bool = False


@dataclass
class VisualizationConfig:
    save_plots: bool = True
    umap_enabled: bool = True  # Calculer UMAP après clustering
    plot_format: str = DEFAULT_PLOT_FORMAT
    dpi: int = DEFAULT_DPI
    figures: dict = field(default_factory=dict)  # Paramètres par figure


@dataclass
class GPUConfig:
    enabled: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class PopulationMappingConfig:
    """Configuration Section 10 — Mapping des populations via MFI de référence."""

    enabled: bool = True
    ref_mfi_dir: str = "Data/Ref MFI"
    cache_dir: str = "output/ref_mfi_parquet_cache"
    # Distance et mapping
    distance_percentile: int = 60
    include_scatter: bool = True
    normalization_method: str = "range"  # range | zscore
    mapping_method: str = "cosine_prior"  # M12 recommandé ELN 2022
    unknown_threshold_mode: str = "auto_otsu"  # auto_otsu | percentile | mean_std
    hard_limit_factor: float = 5.0
    prior_mode: str = "log10_cubed"
    # Transformation des CSV de référence
    transform_method: str = "arcsinh"  # arcsinh | logicle | none
    arcsinh_cofactor: float = 5.0
    apply_to_scatter: bool = False
    # Stats biologiques (Mahalanobis / KNN)
    compute_population_stats: bool = True
    knn_sample_size: int = 2000
    knn_k: int = 15
    cov_reg_alpha: float = 1e-4
    total_knn_points: int = 15000
    # Blast detection
    blast_enabled: bool = True
    blast_suspect_categories: List[str] = field(
        default_factory=lambda: ["BLAST_HIGH", "BLAST_MODERATE"]
    )
    # Visualisation interactive (Plotly)
    viz_interactive: bool = True
    viz_max_points: int = 50_000
    # Couleurs population
    population_colors: dict = field(
        default_factory=lambda: {
            "Granulo": "#e26f1a",
            "Granulocytes": "#e26f1a",
            "Hématogone 34+": "#9467bd",
            "Hematogones19+": "#2ca02c",
            "Ly T_NK": "#17becf",
            "Lymphos B": "#1f77b4",
            "Lymphos": "#aec7e8",
            "Plasmo": "#d62728",
            "Unknown": "#7f7f7f",
        }
    )


@dataclass
class PerformanceMonitoringConfig:
    """Configuration du monitoring de performance système."""

    enabled: bool = False  # true = activer la collecte pendant la pipeline
    interval_seconds: float = 1.0  # intervalle de collecte en secondes
    include_gpu: bool = True  # true = collecter les métriques GPU (nécessite gputil)


@dataclass
class PathoFcsExportConfig:
    """Configuration de l'export FCS restreint à la moelle pathologique + Is_MRD."""

    enabled: bool = False  # true = générer le FCS pathologique avec Is_MRD
    mrd_method: str = "flo"  # méthode Is_MRD : "jf" ou "flo"


@dataclass
class CitrusConfig:
    """Configuration de l'analyse Citrus (identification clusters stratifiants)."""

    enabled: bool = False
    endpoint_column: str = "group"  # Colonne de cohort_metadata portant l'endpoint
    endpoint_type: str = "classification"  # "classification" | "continuous"
    feature_type: str = "abundances"  # "abundances" | "medians" | "both"
    model_type: str = "glmnet"  # "glmnet" | "sam"
    n_cells_per_sample: int = 1000
    min_cluster_size_percent: float = 0.05
    n_cv_folds: int = 5


@dataclass
class BatchConfig:
    """Configuration du mode traitement par lots (batch)."""

    enabled: bool = False  # true = traiter tous les FCS du dossier patho un par un


@dataclass
class ExportModeConfig:
    """
    Mode d'export des résultats.

    - "standard" : tous les fichiers (FCS complet, CSV, JSON métadonnées, plots, rapports, MRD).
    - "compact"  : uniquement les sorties essentielles —
                   rapport PDF, rapport HTML, MRD JSON et FCS pathologique avec Is_MRD.
                   Les CSV, FCS complet, TXT et JSON de métadonnées sont ignorés.
                   Les figures sont toujours générées car nécessaires aux rapports.
    """

    mode: str = "standard"  # "standard" | "compact"
    export_csv: bool = (
        True  # false = ne pas exporter les CSV dans le dossier csv (complet, stats, MFI)
    )
    export_per_file_csv: bool = True  # true = exporter un CSV par fichier FCS source


@dataclass
class PipelineConfig:
    """Configuration complète du pipeline FlowSOM Pro."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    pregate: PregateConfig = field(default_factory=PregateConfig)
    flowsom: FlowSOMConfig = field(default_factory=FlowSOMConfig)
    auto_clustering: AutoClusteringConfig = field(default_factory=AutoClusteringConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    markers: MarkersConfig = field(default_factory=MarkersConfig)
    downsampling: DownsamplingConfig = field(default_factory=DownsamplingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    # Sections YAML libres — accessibles via config.extra("section_name")
    _extra: dict = field(default_factory=dict, repr=False)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    population_mapping: PopulationMappingConfig = field(default_factory=PopulationMappingConfig)
    performance_monitoring: PerformanceMonitoringConfig = field(
        default_factory=PerformanceMonitoringConfig
    )
    patho_fcs_export: PathoFcsExportConfig = field(default_factory=PathoFcsExportConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    export_mode: ExportModeConfig = field(default_factory=ExportModeConfig)
    citrus: CitrusConfig = field(default_factory=CitrusConfig)
    stratified_downsampling: StratifiedDownsamplingConfig = field(
        default_factory=StratifiedDownsamplingConfig
    )
    data_integration: DataIntegrationConfig = field(default_factory=DataIntegrationConfig)
    # Sélection dynamique RUO (Wizard v3)
    qc_method: str = "peacoqc"  # peacoqc | flowai
    dimred_method: str = "umap"  # umap | visne | phate
    clustering_method: str = "flowsom"  # flowsom | phenograph | parc | hdbscan
    qc_methods_enabled: List[str] = field(default_factory=lambda: ["peacoqc"])
    dimred_methods_enabled: List[str] = field(default_factory=lambda: ["umap"])
    clustering_methods_enabled: List[str] = field(default_factory=lambda: ["flowsom"])
    # Contexte de données RUO (optionnel): workspace de gating + population cible.
    gating_workspace_path: str | None = None
    target_population: str = "Root"

    # ------------------------------------------------------------------
    # Constructeurs alternatifs
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "PipelineConfig":
        """
        Charge la configuration depuis un fichier YAML.

        Args:
            yaml_path: Chemin vers le fichier YAML.

        Returns:
            PipelineConfig initialisé à partir du YAML.

        Raises:
            FileNotFoundError: Si le fichier YAML n'existe pas.
            ImportError: Si PyYAML n'est pas installé.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML requis: pip install pyyaml") from exc

        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier de configuration non trouvé: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict) -> "PipelineConfig":
        """Construit un PipelineConfig depuis un dictionnaire YAML brut."""
        cfg = cls()

        # Conserver les sections présentes dans le YAML mais non structurées
        # (ex: export_cluster_distribution, mrd, lsc_markers...)
        _structured_keys = {
            "paths",
            "analysis",
            "pregate",
            "pregate_advanced",
            "flowsom",
            "auto_clustering",
            "transform",
            "normalize",
            "markers",
            "downsampling",
            "visualization",
            "gpu",
            "logging",
            "population_mapping",
            "performance_monitoring",
            "patho_fcs_export",
            "batch",
            "export_mode",
            "stratified_downsampling",
            "data_integration",
            "qc_method",
            "dimred_method",
            "clustering_method",
            "qc_methods_enabled",
            "dimred_methods_enabled",
            "clustering_methods_enabled",
            "gating_workspace_path",
            "target_population",
            "pipeline_version",
        }
        cfg._extra = {k: v for k, v in raw.items() if k not in _structured_keys}

        p = raw.get("paths", {})
        if p.get("healthy_folder"):
            cfg.paths.healthy_folder = p["healthy_folder"]
        if p.get("patho_folder"):
            cfg.paths.patho_folder = p["patho_folder"]
        if p.get("output_dir"):
            cfg.paths.output_dir = p["output_dir"]

        an = raw.get("analysis", {})
        if "compare_mode" in an:
            cfg.analysis.compare_mode = bool(an["compare_mode"])

        pg = raw.get("pregate", {})
        for attr in (
            "apply",
            "mode",
            "mode_blastes_vs_normal",
            "viable",
            "singlets",
            "cd45",
            "cd34",
        ):
            if attr in pg:
                setattr(cfg.pregate, attr, pg[attr])

        pg_adv = raw.get("pregate_advanced", {})
        mapping_pregate = {
            "debris_min_percentile": "debris_min_percentile",
            "debris_max_percentile": "debris_max_percentile",
            "doublets_ratio_min": "doublets_ratio_min",
            "doublets_ratio_max": "doublets_ratio_max",
            "cd45_threshold_percentile": "cd45_threshold_percentile",
            "cd34_threshold_percentile": "cd34_threshold_percentile",
            "cd34_use_ssc_filter": "cd34_use_ssc_filter",
            "cd34_ssc_max_percentile": "cd34_ssc_max_percentile",
            "gmm_max_samples": "gmm_max_samples",
            "ransac_r2_threshold": "ransac_r2_threshold",
            "ransac_mad_factor": "ransac_mad_factor",
            "density_method": "density_method",
            "gmm_covariance_type": "gmm_covariance_type",
            "gmm_n_components_debris": "gmm_n_components_debris",
            "gmm_export_plot": "gmm_export_plot",
            "kde_cd45_seuil_relatif": "kde_cd45_seuil_relatif",
            "kde_cd45_finesse": "kde_cd45_finesse",
            "kde_cd45_sigma_smooth": "kde_cd45_sigma_smooth",
            "kde_cd45_n_grid": "kde_cd45_n_grid",
            "kde_cd45_max_samples": "kde_cd45_max_samples",
            "cd45_autogating_mode": "cd45_autogating_mode",
        }
        for yaml_key, attr in mapping_pregate.items():
            if yaml_key in pg_adv:
                setattr(cfg.pregate, attr, pg_adv[yaml_key])

        fs = raw.get("flowsom", {})
        for attr in (
            "xdim",
            "ydim",
            "rlen",
            "n_metaclusters",
            "learning_rate",
            "sigma",
            "n_iterations",
            "seed",
        ):
            if attr in fs:
                setattr(cfg.flowsom, attr, fs[attr])

        ac = raw.get("auto_clustering", {})
        for attr in (
            "enabled",
            "min_clusters",
            "max_clusters",
            "n_bootstrap",
            "sample_size_bootstrap",
            "min_stability_threshold",
            "weight_stability",
            "weight_silhouette",
        ):
            if attr in ac:
                setattr(cfg.auto_clustering, attr, ac[attr])

        tr = raw.get("transform", {})
        for attr in (
            "method", "cofactor", "apply_to_scatter",
            "logicle_T", "logicle_M", "logicle_W", "logicle_A",
            "per_column_specs",
        ):
            if attr in tr:
                setattr(cfg.transform, attr, tr[attr])

        no = raw.get("normalize", {})
        if "method" in no:
            cfg.normalize.method = no["method"]

        mk = raw.get("markers", {})
        if "exclude_scatter" in mk:
            cfg.markers.exclude_scatter = bool(mk["exclude_scatter"])
        if "exclude_additional" in mk:
            cfg.markers.exclude_additional = list(mk["exclude_additional"] or [])
        if "keep_area_only" in mk:
            cfg.markers.keep_area_only = bool(mk["keep_area_only"])

        ds = raw.get("downsampling", {})
        for attr in ("enabled", "max_cells_per_file", "max_cells_total"):
            if attr in ds:
                setattr(cfg.downsampling, attr, ds[attr])

        viz = raw.get("visualization", {})
        for attr in ("save_plots", "umap_enabled", "plot_format", "dpi"):
            if attr in viz:
                setattr(cfg.visualization, attr, viz[attr])

        gp = raw.get("gpu", {})
        if "enabled" in gp:
            cfg.gpu.enabled = bool(gp["enabled"])

        lg = raw.get("logging", {})
        if "level" in lg:
            cfg.logging.level = lg["level"]

        pm = raw.get("population_mapping", {})
        if pm:
            for attr in (
                "enabled",
                "ref_mfi_dir",
                "cache_dir",
                "distance_percentile",
                "include_scatter",
                "normalization_method",
                "mapping_method",
                "unknown_threshold_mode",
                "hard_limit_factor",
                "prior_mode",
                "transform_method",
                "arcsinh_cofactor",
                "apply_to_scatter",
                "compute_population_stats",
                "knn_sample_size",
                "knn_k",
                "cov_reg_alpha",
                "total_knn_points",
                "blast_enabled",
                "viz_interactive",
                "viz_max_points",
            ):
                if attr in pm:
                    setattr(cfg.population_mapping, attr, pm[attr])
            if "blast_suspect_categories" in pm:
                cfg.population_mapping.blast_suspect_categories = list(
                    pm["blast_suspect_categories"]
                )
            if "population_colors" in pm:
                cfg.population_mapping.population_colors = dict(pm["population_colors"])

        pm_mon = raw.get("performance_monitoring", {})
        if pm_mon:
            for attr in ("enabled", "interval_seconds", "include_gpu"):
                if attr in pm_mon:
                    setattr(cfg.performance_monitoring, attr, pm_mon[attr])

        pfe = raw.get("patho_fcs_export", {})
        if pfe:
            for attr in ("enabled", "mrd_method"):
                if attr in pfe:
                    setattr(cfg.patho_fcs_export, attr, pfe[attr])

        bt = raw.get("batch", {})
        if bt:
            if "enabled" in bt:
                cfg.batch.enabled = bool(bt["enabled"])

        em = raw.get("export_mode", {})
        if em:
            if "mode" in em:
                cfg.export_mode.mode = str(em["mode"])
            if "export_csv" in em:
                cfg.export_mode.export_csv = bool(em["export_csv"])
            if "export_per_file_csv" in em:
                cfg.export_mode.export_per_file_csv = bool(em["export_per_file_csv"])

        sd = raw.get("stratified_downsampling", {})
        if sd:
            if "balance_conditions" in sd:
                cfg.stratified_downsampling.balance_conditions = bool(sd["balance_conditions"])
            if "imbalance_ratio" in sd:
                cfg.stratified_downsampling.imbalance_ratio = float(sd["imbalance_ratio"])
            if "nbm_ids" in sd:
                cfg.stratified_downsampling.nbm_ids = list(sd["nbm_ids"] or [])
            if "seed" in sd:
                cfg.stratified_downsampling.seed = int(sd["seed"])
            if "allow_oversampling" in sd:
                cfg.stratified_downsampling.allow_oversampling = bool(sd["allow_oversampling"])

        di = raw.get("data_integration", {})
        if di:
            if "enabled" in di:
                cfg.data_integration.enabled = bool(di["enabled"])
            if "method" in di:
                cfg.data_integration.method = str(di["method"])
            hp = di.get("harmony_params", {})
            if hp:
                if "sigma" in hp:
                    cfg.data_integration.harmony_params.sigma = float(hp["sigma"])
                if "nclust" in hp:
                    v = hp["nclust"]
                    cfg.data_integration.harmony_params.nclust = int(v) if v is not None else None
                if "block_size" in hp:
                    cfg.data_integration.harmony_params.block_size = float(hp["block_size"])
                if "max_iter" in hp:
                    cfg.data_integration.harmony_params.max_iter = int(hp["max_iter"])
                if "max_iter_kmeans" in hp:
                    cfg.data_integration.harmony_params.max_iter_kmeans = int(hp["max_iter_kmeans"])
                if "verbose" in hp:
                    cfg.data_integration.harmony_params.verbose = bool(hp["verbose"])
                if "markers_to_align" in hp:
                    cfg.data_integration.harmony_params.markers_to_align = [
                        str(m).strip() for m in (hp["markers_to_align"] or []) if str(m).strip()
                    ]

        if "qc_method" in raw:
            cfg.qc_method = str(raw["qc_method"])
        if "dimred_method" in raw:
            cfg.dimred_method = str(raw["dimred_method"])
        if "clustering_method" in raw:
            cfg.clustering_method = str(raw["clustering_method"])

        if "qc_methods_enabled" in raw:
            cfg.qc_methods_enabled = [str(v).lower() for v in (raw["qc_methods_enabled"] or [])]
        if "dimred_methods_enabled" in raw:
            cfg.dimred_methods_enabled = [
                str(v).lower() for v in (raw["dimred_methods_enabled"] or [])
            ]
        if "clustering_methods_enabled" in raw:
            cfg.clustering_methods_enabled = [
                str(v).lower() for v in (raw["clustering_methods_enabled"] or [])
            ]

        if "gating_workspace_path" in raw:
            raw_path = raw.get("gating_workspace_path")
            cfg.gating_workspace_path = None if raw_path in (None, "") else str(raw_path)

        if "target_population" in raw:
            cfg.target_population = str(raw.get("target_population") or "Root")

        # Validation
        cfg._validate()
        return cfg

    @classmethod
    def from_args(cls, args: Any, yaml_path: Optional[str] = None) -> "PipelineConfig":
        """
        Construit la configuration depuis des arguments CLI (argparse.Namespace).
        Si yaml_path est fourni, charge le YAML d'abord, puis surcharge avec args.

        Args:
            args: Namespace argparse.
            yaml_path: Chemin optionnel vers un fichier YAML.

        Returns:
            PipelineConfig.
        """
        cfg = cls.from_yaml(yaml_path) if yaml_path else cls()

        # Surcharger avec les args CLI (seulement si explicitement définis)
        if getattr(args, "healthy_folder", None):
            cfg.paths.healthy_folder = args.healthy_folder
        if getattr(args, "patho_folder", None):
            cfg.paths.patho_folder = args.patho_folder
        if getattr(args, "output", None):
            cfg.paths.output_dir = args.output

        if getattr(args, "compare_mode", None) is not None:
            cfg.analysis.compare_mode = args.compare_mode

        for attr_args, attr_cfg in (
            ("xdim", "xdim"),
            ("ydim", "ydim"),
            ("n_metaclusters", "n_metaclusters"),
            ("learning_rate", "learning_rate"),
            ("sigma", "sigma"),
            ("n_iterations", "n_iterations"),
            ("seed", "seed"),
        ):
            val = getattr(args, attr_args, None)
            if val is not None:
                setattr(cfg.flowsom, attr_cfg, val)

        if getattr(args, "transform", None):
            cfg.transform.method = args.transform
        if getattr(args, "cofactor", None) is not None:
            cfg.transform.cofactor = args.cofactor
        if getattr(args, "normalize", None):
            cfg.normalize.method = args.normalize

        if getattr(args, "downsample", None) is not None:
            cfg.downsampling.enabled = args.downsample
        if getattr(args, "max_cells_per_file", None) is not None:
            cfg.downsampling.max_cells_per_file = args.max_cells_per_file
        if getattr(args, "max_cells_total", None) is not None:
            cfg.downsampling.max_cells_total = args.max_cells_total

        if getattr(args, "save_plots", None) is not None:
            cfg.visualization.save_plots = args.save_plots
        if getattr(args, "plot_format", None):
            cfg.visualization.plot_format = args.plot_format
        if getattr(args, "dpi", None) is not None:
            cfg.visualization.dpi = args.dpi

        if getattr(args, "use_gpu", None) is not None:
            cfg.gpu.enabled = args.use_gpu

        cfg._validate()
        return cfg

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Validation des paramètres critiques et vérification de types YAML."""
        # ── Validation de types critiques (erreurs YAML silencieuses) ──────────
        int_fields = {
            "flowsom.xdim": self.flowsom.xdim,
            "flowsom.ydim": self.flowsom.ydim,
            "flowsom.n_metaclusters": self.flowsom.n_metaclusters,
            "flowsom.seed": self.flowsom.seed,
        }
        for field_name, value in int_fields.items():
            if not isinstance(value, int):
                raise TypeError(
                    f"Configuration: {field_name} doit être un entier, "
                    f"obtenu {type(value).__name__} ({value!r}). "
                    "Vérifiez votre fichier YAML (pas de guillemets autour des entiers)."
                )
        if not isinstance(self.transform.cofactor, (int, float)):
            raise TypeError(
                f"Configuration: transform.cofactor doit être un nombre, "
                f"obtenu {type(self.transform.cofactor).__name__} ({self.transform.cofactor!r})."
            )

        if self.pregate.mode not in ("auto", "manual"):
            warnings.warn(
                f"pregate.mode '{self.pregate.mode}' inconnu. Valeurs acceptées: 'auto', 'manual'. "
                "Basculement sur 'auto'."
            )
            self.pregate.mode = "auto"

        if self.transform.method not in ("arcsinh", "logicle", "log10", "none"):
            warnings.warn(
                f"transform.method '{self.transform.method}' inconnu. Basculement sur 'logicle'."
            )
            self.transform.method = "logicle"

        if self.normalize.method not in ("zscore", "minmax", "none"):
            warnings.warn(
                f"normalize.method '{self.normalize.method}' inconnu. Basculement sur 'zscore'."
            )
            self.normalize.method = "zscore"

        if self.flowsom.xdim < 2 or self.flowsom.ydim < 2:
            raise ValueError("flowsom.xdim et flowsom.ydim doivent être >= 2")

        if self.flowsom.n_metaclusters < 2:
            raise ValueError("flowsom.n_metaclusters doit être >= 2")

        if self.pregate.mode_blastes_vs_normal and not self.analysis.compare_mode:
            warnings.warn(
                "pregate.mode_blastes_vs_normal nécessite analysis.compare_mode=True. "
                "Désactivation du mode asymétrique."
            )
            self.pregate.mode_blastes_vs_normal = False

        if not isinstance(self.data_integration.harmony_params.markers_to_align, list):
            raise TypeError(
                "Configuration: data_integration.harmony_params.markers_to_align "
                "doit être une liste de chaînes."
            )

        allowed_qc = {"peacoqc", "flowai"}
        allowed_dimred = {"umap", "visne", "tsne", "phate"}
        allowed_clustering = {"flowsom", "phenograph", "parc", "hdbscan", "spade", "flowsom_like"}

        self.qc_method = str(self.qc_method).lower().strip()
        self.dimred_method = str(self.dimred_method).lower().strip()
        self.clustering_method = str(self.clustering_method).lower().strip()

        if self.qc_method not in allowed_qc:
            warnings.warn(f"qc_method '{self.qc_method}' inconnu. Basculement sur 'peacoqc'.")
            self.qc_method = "peacoqc"

        if self.dimred_method not in allowed_dimred:
            warnings.warn(f"dimred_method '{self.dimred_method}' inconnu. Basculement sur 'umap'.")
            self.dimred_method = "umap"

        if self.clustering_method not in allowed_clustering:
            warnings.warn(
                f"clustering_method '{self.clustering_method}' inconnu. Basculement sur 'flowsom'."
            )
            self.clustering_method = "flowsom"

        self.qc_methods_enabled = [
            str(v).lower().strip() for v in self.qc_methods_enabled if str(v).strip()
        ]
        self.dimred_methods_enabled = [
            str(v).lower().strip() for v in self.dimred_methods_enabled if str(v).strip()
        ]
        self.clustering_methods_enabled = [
            str(v).lower().strip() for v in self.clustering_methods_enabled if str(v).strip()
        ]

        self.qc_methods_enabled = [v for v in self.qc_methods_enabled if v in allowed_qc]
        self.dimred_methods_enabled = [
            v for v in self.dimred_methods_enabled if v in allowed_dimred
        ]
        self.clustering_methods_enabled = [
            v for v in self.clustering_methods_enabled if v in allowed_clustering
        ]

        if not self.qc_methods_enabled:
            self.qc_methods_enabled = [self.qc_method]
        if self.qc_method not in self.qc_methods_enabled:
            self.qc_methods_enabled.insert(0, self.qc_method)

        if not self.dimred_methods_enabled:
            self.dimred_methods_enabled = [self.dimred_method]
        if self.dimred_method not in self.dimred_methods_enabled:
            self.dimred_methods_enabled.insert(0, self.dimred_method)

        if not self.clustering_methods_enabled:
            self.clustering_methods_enabled = [self.clustering_method]
        if self.clustering_method not in self.clustering_methods_enabled:
            self.clustering_methods_enabled.insert(0, self.clustering_method)

        if self.gating_workspace_path is not None and not isinstance(
            self.gating_workspace_path, str
        ):
            raise TypeError("Configuration: gating_workspace_path doit être une chaîne ou null.")

        if isinstance(self.gating_workspace_path, str):
            normalized_path = self.gating_workspace_path.strip()
            self.gating_workspace_path = normalized_path or None

        self.target_population = str(self.target_population or "").strip()
        if not self.target_population:
            warnings.warn("target_population vide détecté. Basculement sur 'Root'.")
            self.target_population = "Root"

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Sérialise la configuration en dictionnaire (pour JSON/YAML export)."""
        from dataclasses import asdict

        return asdict(self)

    def has_gating_context(self) -> bool:
        """True si un workspace de gating est configuré."""
        return bool(self.gating_workspace_path)

    def uses_root_population(self) -> bool:
        """True si l'analyse cible la population racine (fichier complet)."""
        return self.target_population.strip().lower() == "root"

    def __repr__(self) -> str:
        return (
            f"PipelineConfig("
            f"mode={self.pregate.mode}, "
            f"transform={self.transform.method}, "
            f"grid={self.flowsom.xdim}x{self.flowsom.ydim}, "
            f"metaclusters={self.flowsom.n_metaclusters}, "
            f"compare_mode={self.analysis.compare_mode}, "
            f"qc={self.qc_method}, "
            f"dimred={self.dimred_method}, "
            f"clustering={self.clustering_method})"
        )
