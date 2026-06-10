"""
src/visualization/__init__.py — Exports publics de la couche visualisation.
"""

from .flowsom_native_plots import (
    generate_all_native_plots,
    plot_2d_scatters_native,
    plot_flowsommary_native,
    plot_marker_native,
    plot_new_data_stars,
    plot_numbers_native,
    plot_stars_native,
    plot_subset_stars,
    plot_variable_native,
)
from .flowsom_plots import (
    circular_jitter,
    circular_jitter_viz,
    plot_metacluster_sizes,
    plot_mfi_heatmap,
    plot_mst_plotly,
    plot_mst_static,
    plot_optimization_results,
    plot_som_grid_plotly,
    plot_umap,
)
from .gating_plots import (
    generate_all_gating_plots,
    generate_interactive_gating_dashboard,
    generate_per_file_sankey,
    generate_sankey_diagram,
    plot_cd34_gate,
    plot_cd45_gate,
    plot_debris_gate,
    plot_gmm_vs_kde_qc,
    plot_overview,
    plot_singlets_gate,
)
from .html_report import fig_to_base64, generate_html_report, plotly_to_html_div
from .plot_helpers import (
    add_gate_rectangle,
    apply_dark_style,
    format_axis,
    plot_density,
    plot_gating,
    save_figure,
)
from .population_viz import (
    get_mean_profile,
    plot_blast_fcs_source,
    plot_blast_heatmap,
    plot_blast_radar,
    plot_blast_scores_bar,
    plot_heatmap_comparative,
    plot_mrd_blast_radar,
    zscore_df,
)

__all__ = [
    # Helpers bas-niveau
    "format_axis",
    "apply_dark_style",
    "plot_density",
    "plot_gating",
    "add_gate_rectangle",
    "save_figure",
    # Gating
    "plot_overview",
    "plot_debris_gate",
    "plot_singlets_gate",
    "plot_cd45_gate",
    "plot_cd34_gate",
    "generate_all_gating_plots",
    "generate_sankey_diagram",
    "generate_per_file_sankey",
    "plot_gmm_vs_kde_qc",
    "generate_interactive_gating_dashboard",
    # FlowSOM
    "plot_mfi_heatmap",
    "plot_metacluster_sizes",
    "plot_umap",
    "circular_jitter",
    "circular_jitter_viz",
    "plot_optimization_results",
    "plot_mst_static",
    "plot_mst_plotly",
    "plot_som_grid_plotly",
    # FlowSOM natif (fs.pl.*)
    "plot_stars_native",
    "plot_marker_native",
    "plot_numbers_native",
    "plot_variable_native",
    "plot_2d_scatters_native",
    "plot_flowsommary_native",
    "plot_new_data_stars",
    "plot_subset_stars",
    "generate_all_native_plots",
    # HTML Report
    "generate_html_report",
    "fig_to_base64",
    "plotly_to_html_div",
    # Population viz
    "plot_blast_heatmap",
    "plot_blast_radar",
    "plot_blast_scores_bar",
    "plot_blast_fcs_source",
    "plot_mrd_blast_radar",
    "plot_heatmap_comparative",
    "get_mean_profile",
    "zscore_df",
]
