"""
src/analysis/__init__.py — Exports publics de la couche analyse.
"""

from .population_mapping import (
    map_populations_to_nodes_v3,
    map_populations_to_nodes_v5,
    map_nodes_to_metaclusters,
    get_population_summary,
    normalize_matrix,
    build_population_color_map,
    _apply_bayesian_prior,
    compute_unknown_threshold,
    assign_with_auto_threshold,
    filter_area_columns,
    normalize_col_name,
    build_direct_mapping_a_only,
    load_population_csv_transformed,
    compute_pop_stats_from_csv,
    extract_node_centroids_from_fcs,
    apply_cyto_transform_matrix,
    arcsinh_transform,
    robust_scale,
    POPULATION_COLORS,
)
from .blast_detection import (
    build_blast_weights,
    score_nodes_for_blasts,
    categorize_blast_score,
    build_blast_score_dataframe,
    compute_reference_normalization,
    compute_reference_stats,
    score_nodes_mahalanobis,
    score_nodes_hybrid,
    build_weight_vector,
    BLAST_HIGH_THRESHOLD,
    BLAST_MODERATE_THRESHOLD,
)
from .statistics import (
    mann_whitney_u,
    kolmogorov_smirnov,
    compute_fold_change,
    assess_mrd_status,
    compare_conditions_per_cluster,
)
from .advanced_strategies import (
    AdvancedCohortExecutor,
    HarmonyStrategy,
    HarmonyStrategyParams,
    PHATEStrategy,
    PHATEStrategyParams,
    PhenoGraphStrategy,
    PhenoGraphStrategyParams,
    build_advanced_executor_from_config,
    run_advanced_cohort_from_config,
)

__all__ = [
    # Population mapping
    "map_populations_to_nodes_v3",
    "map_populations_to_nodes_v5",
    "map_nodes_to_metaclusters",
    "get_population_summary",
    "normalize_matrix",
    "build_population_color_map",
    "_apply_bayesian_prior",
    "compute_unknown_threshold",
    "assign_with_auto_threshold",
    # Blast detection
    "build_blast_weights",
    "score_nodes_for_blasts",
    "categorize_blast_score",
    "build_blast_score_dataframe",
    "compute_reference_normalization",
    "compute_reference_stats",
    "score_nodes_mahalanobis",
    "score_nodes_hybrid",
    "build_weight_vector",
    "BLAST_HIGH_THRESHOLD",
    "BLAST_MODERATE_THRESHOLD",
    # Statistics
    "mann_whitney_u",
    "kolmogorov_smirnov",
    "compute_fold_change",
    "assess_mrd_status",
    "compare_conditions_per_cluster",
    # Advanced cohort strategies
    "AdvancedCohortExecutor",
    "HarmonyStrategy",
    "HarmonyStrategyParams",
    "PHATEStrategy",
    "PHATEStrategyParams",
    "PhenoGraphStrategy",
    "PhenoGraphStrategyParams",
    "build_advanced_executor_from_config",
    "run_advanced_cohort_from_config",
]
