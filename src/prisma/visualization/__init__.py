"""
src/visualization/__init__.py — Exports publics de la couche visualisation.
"""

from .plot_helpers import (
    format_axis,
    apply_dark_style,
    plot_density,
    plot_gating,
    add_gate_rectangle,
    save_figure,
)
from .gating_plots import (
    plot_overview,
    plot_debris_gate,
    plot_singlets_gate,
    plot_cd45_gate,
    plot_cd34_gate,
    generate_all_gating_plots,
    generate_sankey_diagram,
    generate_per_file_sankey,
    plot_gmm_vs_kde_qc,
    generate_interactive_gating_dashboard,
)
from .flowsom_plots import (
    plot_mfi_heatmap,
    plot_metacluster_sizes,
    plot_umap,
    circular_jitter,
    circular_jitter_viz,
    plot_optimization_results,
    plot_mst_static,
    plot_mst_plotly,
    plot_som_grid_plotly,
)
from .html_report import generate_html_report, fig_to_base64, plotly_to_html_div
from .population_viz import (
    plot_blast_heatmap,
    plot_blast_radar,
    plot_blast_scores_bar,
    plot_blast_fcs_source,
    plot_mrd_blast_radar,
    plot_heatmap_comparative,
    get_mean_profile,
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
