"""
html_report.py — Génération d'un rapport HTML self-contained avec Plotly + Matplotlib.

Assemble toutes les figures interactives (Plotly) et statiques (Matplotlib)
en un seul fichier HTML autonome, sans dépendance CDN ou fichier externe.

Le rapport inclut :
- En-tête avec résumé de l'analyse
- Table des matières navigable
- Paramètres de l'analyse
- Statistiques par métacluster
- Figures Plotly interactives (Sankey, heatmaps, spider plots, etc.)
- Figures Matplotlib en base64 inline (gating, RANSAC, etc.)
"""

from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

_logger = get_logger("visualization.html_report")

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    import plotly.offline

    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

# Cache module-level du bundle plotly.js (~3.5 MB) :
# get_plotlyjs() lit et compresse le fichier à chaque appel.
# En le mémorisant dès le premier appel on évite le re-chargement complet
# pour chaque rapport généré dans la même session.
_PLOTLYJS_CACHE: "Optional[str]" = None


def _get_plotlyjs_cached() -> str:
    """Retourne le bundle plotly.js en le chargeant au plus une fois par session."""
    global _PLOTLYJS_CACHE
    if _PLOTLYJS_CACHE is None:
        _PLOTLYJS_CACHE = plotly.offline.get_plotlyjs()
    return _PLOTLYJS_CACHE


try:
    import matplotlib.figure

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


def _fig_to_base64(fig_mpl: Any, dpi: int = 100) -> str:
    """Convertit une figure matplotlib en string base64 PNG."""
    text_color = "#e2e8f0"
    spine_color = "#45475a"

    # Harmonise le contraste pour le rapport HTML (axes/titres toujours lisibles).
    for ax in getattr(fig_mpl, "axes", []):
        try:
            ax.title.set_color(text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.tick_params(colors=text_color)
            for spine in ax.spines.values():
                spine.set_color(spine_color)
        except Exception:
            continue

    buf = BytesIO()
    fig_face = (
        fig_mpl.get_facecolor() if hasattr(fig_mpl, "get_facecolor") else "#1e1e2e"
    )
    # Si figure transparente, forcer un fond sombre pour préserver le contraste
    if isinstance(fig_face, tuple) and len(fig_face) == 4 and fig_face[3] == 0:
        fig_face = "#1e1e2e"
    fig_mpl.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig_face,
        edgecolor="none",
        transparent=False,
    )
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _plotly_to_html_div(fig_plotly: Any, fig_id: str = "") -> str:
    """Convertit une figure Plotly en div HTML (sans plotly.js embarqué)."""
    fig_height = fig_plotly.layout.height or 500
    fig_width_val = fig_plotly.layout.width
    default_w = f"{fig_width_val}px" if fig_width_val else "100%"
    return pio.to_html(
        fig_plotly,
        full_html=False,
        include_plotlyjs=False,
        div_id=fig_id if fig_id else None,
        default_height=f"{fig_height}px",
        default_width=default_w,
        config={"responsive": True},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CSS Template (réplique exacte du monolithique)
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
:root {
    --primary: #667eea;
    --primary-dark: #764ba2;
    --bg: #f8f9fa;
    --card-bg: #ffffff;
    --text: #2d3748;
    --text-light: #718096;
    --border: #e2e8f0;
    --success: #48bb78;
    --warning: #ed8936;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
}
.header {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color: white; padding: 40px 0; text-align: center; margin-bottom: 30px;
}
.header h1 { font-size: 2.2em; margin-bottom: 8px; font-weight: 700; }
.header .subtitle { font-size: 1.1em; opacity: 0.9; }
.container { max-width: 1400px; margin: 0 auto; padding: 0 20px; }
.section {
    background: var(--card-bg); border-radius: 12px; padding: 30px;
    margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border: 1px solid var(--border);
}
.section h2 {
    font-size: 1.4em; color: var(--primary); margin-bottom: 20px;
    padding-bottom: 10px; border-bottom: 2px solid var(--border);
}
.grid-3 {
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;
}
.stat-card {
    background: linear-gradient(135deg, #f6f8ff, #f0f4ff);
    border-radius: 10px; padding: 20px; text-align: center;
    border: 1px solid #dde4f0;
}
.stat-card .value { font-size: 2em; font-weight: 700; color: var(--primary); }
.stat-card .label { font-size: 0.9em; color: var(--text-light); margin-top: 4px; }
table { width: 100%; border-collapse: collapse; margin-top: 15px; }
th {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color: white; padding: 12px 16px; text-align: left; font-weight: 600;
}
td { padding: 10px 16px; border-bottom: 1px solid var(--border); }
tr:nth-child(even) { background: #f7fafc; }
tr:hover { background: #edf2f7; }
.marker-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea22, #764ba222);
    color: var(--primary-dark); padding: 4px 12px; border-radius: 20px;
    font-size: 0.85em; margin: 3px; border: 1px solid #667eea44;
    font-weight: 500;
}
.plotly-container {
    width: 100%; overflow-x: auto; display: flex; justify-content: center;
}
.plotly-container > div { min-width: 0; }
.param-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 12px;
}
.param-item {
    background: #f7fafc; padding: 10px 15px; border-radius: 8px;
    border-left: 3px solid var(--primary);
}
.param-item .param-label {
    font-size: 0.8em; color: var(--text-light); text-transform: uppercase;
    letter-spacing: 0.5px;
}
.param-item .param-value { font-size: 1.1em; font-weight: 600; color: var(--text); }
.toc {
    background: #f0f4ff; border-radius: 10px; padding: 20px 30px;
    margin-bottom: 24px;
}
.toc h3 { margin-bottom: 10px; color: var(--primary-dark); }
.toc ul { list-style: none; columns: 2; }
.toc li { padding: 4px 0; }
.toc a { color: var(--primary); text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.footer {
    text-align: center; padding: 30px; color: var(--text-light);
    font-size: 0.9em;
}
@media (max-width: 768px) {
    .grid-3 { grid-template-columns: 1fr; }
    .toc ul { columns: 1; }
}
/* ── Bandeau MRD Validée par l'Expert ── */
.mrd-curated-banner {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 2px solid #86efac;
    border-left: 6px solid #16a34a;
    border-radius: 10px;
    padding: 18px 22px 14px 22px;
    margin-bottom: 18px;
    box-shadow: 0 2px 10px rgba(22,163,74,0.13);
}
.mrd-curated-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.mrd-curated-icon {
    font-size: 1.5em;
    color: #16a34a;
    font-weight: 900;
}
.mrd-curated-label {
    font-size: 0.75em;
    font-weight: 800;
    letter-spacing: 0.12em;
    color: #166534;
    text-transform: uppercase;
}
.mrd-curated-value {
    font-size: 2.4em;
    font-weight: 900;
    color: #15803d;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 4px;
}
.mrd-curated-sub {
    font-size: 0.9em;
    color: #4b5563;
    margin-bottom: 12px;
}
.mrd-algo-trace {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #bbf7d0;
}
.mrd-algo-title {
    font-size: 0.72em;
    font-weight: 700;
    color: #6b7280;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.mrd-algo-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82em;
}
.mrd-algo-table th {
    background: #dcfce7;
    color: #166534;
    font-weight: 700;
    padding: 4px 10px;
    border: 1px solid #bbf7d0;
    text-align: left;
}
.mrd-algo-table td {
    padding: 3px 10px;
    border: 1px solid #d1fae5;
    color: #374151;
}
.mrd-algo-table tr:nth-child(even) td { background: #f0fdf4; }
.mrd-algo-pos { color: #b91c1c; font-weight: 700; }
.mrd-algo-neg { color: #166534; }

.patho-banner {
    background: linear-gradient(135deg, #fff3cd, #ffe082);
    border: 2px solid #f59e0b;
    border-left: 6px solid #d97706;
    border-radius: 10px;
    padding: 18px 24px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 2px 8px rgba(245,158,11,0.2);
}
.patho-banner .patho-icon {
    font-size: 2em;
    flex-shrink: 0;
}
.patho-banner .patho-title {
    font-size: 0.75em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #92400e;
    margin-bottom: 4px;
}
.patho-banner .patho-name {
    font-size: 1.15em;
    font-weight: 700;
    color: #78350f;
    word-break: break-all;
}
.patho-banner .patho-date {
    font-size: 1em;
    color: #b45309;
    margin-top: 4px;
}
"""


def generate_html_report(
    output_path: Path | str,
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
    dpi_mpl: int = 100,
    ransac_summary: Optional[Dict[str, Any]] = None,
    # ── Curation humaine (optionnelle) ──────────────────────────────────
    curated_mrd_percent: Optional[float] = None,
    curated_mrd_cells:   Optional[int]   = None,
    curated_nodes:       Optional[List[Dict[str, Any]]] = None,
    algo_gauges:         Optional[List[Dict[str, Any]]] = None,
    # ── Pré-screening CD34+/CD45dim ──────────────────────────────────────
    prescreening_result: Optional[Any] = None,
) -> bool:
    """
    Génère un rapport HTML complet avec toutes les visualisations.

    Le rapport est self-contained : plotly.js est embarqué directement dans le
    fichier HTML pour un fonctionnement hors-ligne.

    Args:
        output_path: Chemin du fichier HTML de sortie.
        plotly_figures: Dict {nom: go.Figure} — figures Plotly interactives.
        matplotlib_figures: Dict {nom: mpl.Figure} — figures matplotlib.
        figure_labels: Dict {nom: "Titre lisible"} pour les légendes.
        analysis_params: Dict des paramètres d'analyse (transformation, grille, etc.).
        summary_stats: Dict avec n_cells, n_markers, n_files, n_clusters.
        metacluster_table: Liste de dicts [{metacluster, n_cells, pct, top_markers}].
        markers: Liste des marqueurs utilisés pour le clustering.
        self_contained: Si True, embarque plotly.js (~3.5 MB). Sinon, CDN.

    Returns:
        True si succès.
    """
    if not _PLOTLY_AVAILABLE:
        _logger.error("plotly requis pour le rapport HTML")
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plotly_figures = plotly_figures or {}
    matplotlib_figures = matplotlib_figures or {}
    figure_labels = figure_labels or {}
    analysis_params = analysis_params or {}
    summary_stats = summary_stats or {}
    metacluster_table = metacluster_table or []
    markers = markers or []
    condition_data = condition_data or []
    files_data = files_data or []
    export_paths = export_paths or {}

    now_str = datetime.now().strftime("%d/%m/%Y à %H:%M")
    n_cells = summary_stats.get("n_cells", 0)
    n_markers = summary_stats.get("n_markers", len(markers))
    n_files = summary_stats.get("n_files", 0)
    n_clusters = summary_stats.get("n_clusters", 0)

    # ── Bandeau MRD Validée par l'Expert ─────────────────────────────────
    _curated_banner = "<!-- MRD_CURATED_BANNER_START --><!-- MRD_CURATED_BANNER_END -->"
    if curated_mrd_percent is not None:
        _c_pct_str = f"{curated_mrd_percent:.4f} %"
        _c_cells_str = f"{curated_mrd_cells:,} cellules" if curated_mrd_cells else ""
        _c_nodes = curated_nodes or []
        _c_nodes_str = f"{len(_c_nodes)} nœud(s) validé(s)" if _c_nodes else ""
        _c_sub = "  ·  ".join(p for p in [_c_cells_str, _c_nodes_str] if p)

        # Tableau de traçabilité algorithmique
        _algo_rows_html = ""
        for g in (algo_gauges or []):
            _status = "POSITIF" if (g.get("positive") or g.get("low_level")) else "négatif"
            _status_cls = "mrd-algo-pos" if g.get("positive") else "mrd-algo-neg"
            _algo_rows_html += (
                f"<tr>"
                f"<td>{g.get('method','?')}</td>"
                f"<td>{g.get('pct', 0.0):.4f} %</td>"
                f"<td>{g.get('n_cells', 0):,}</td>"
                f'<td class="{_status_cls}">{_status}</td>'
                f"</tr>"
            )
        _algo_table_html = ""
        if _algo_rows_html:
            _algo_table_html = f"""
            <div class="mrd-algo-trace">
                <div class="mrd-algo-title">Valeurs algorithmiques brutes — traçabilité</div>
                <table class="mrd-algo-table">
                    <thead><tr>
                        <th>Méthode</th><th>MRD Algo (%)</th>
                        <th>Cellules</th><th>Statut</th>
                    </tr></thead>
                    <tbody>{_algo_rows_html}</tbody>
                </table>
            </div>"""

        _curated_banner = f"""<!-- MRD_CURATED_BANNER_START -->
<div class="mrd-curated-banner">
  <div class="mrd-curated-header">
    <span class="mrd-curated-icon">&#10003;</span>
    <span class="mrd-curated-label">MRD VALIDÉE PAR L'EXPERT</span>
  </div>
  <div class="mrd-curated-value">{_c_pct_str}</div>
  <div class="mrd-curated-sub">{_c_sub}</div>
  {_algo_table_html}
</div>
<!-- MRD_CURATED_BANNER_END -->"""

    # ── Encadré moelle pathologique ───────────────────────────────────────
    _patho_banner = ""
    if patho_info:
        _pname = patho_info.get("name", "")
        _pdate = patho_info.get("date", "Date inconnue")
        _patho_banner = (
            f'<div class="patho-banner">'
            f'  <div class="patho-icon">&#9888;</div>'
            f"  <div>"
            f'    <div class="patho-title">Moelle pathologique analysée</div>'
            f'    <div class="patho-name">{_pname}</div>'
            f'    <div class="patho-date">Date du prélèvement : <strong>{_pdate}</strong></div>'
            f"  </div>"
            f"</div>"
        )

    # ── Script Plotly.js ──────────────────────────────────────────────────
    if self_contained:
        plotly_js = _get_plotlyjs_cached()
        plotly_script = f'<script type="text/javascript">{plotly_js}</script>'
    else:
        plotly_script = (
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
        )

    # ── Sections de paramètres ────────────────────────────────────────────
    param_items = ""
    for key, val in analysis_params.items():
        param_items += (
            f'<div class="param-item">'
            f'<div class="param-label">{key}</div>'
            f'<div class="param-value">{val}</div>'
            f"</div>\n"
        )

    # ── Statistiques KPI ──────────────────────────────────────────────────
    stats_cards = f"""
    <div class="grid-3">
        <div class="stat-card">
            <div class="value">{n_cells:,}</div>
            <div class="label">Cellules totales</div>
        </div>
        <div class="stat-card">
            <div class="value">{n_markers}</div>
            <div class="label">Marqueurs (clustering)</div>
        </div>
        <div class="stat-card">
            <div class="value">{n_files}</div>
            <div class="label">Fichiers analysés</div>
        </div>
    </div>
    """

    # ── Tableau métaclusters ──────────────────────────────────────────────
    mc_rows = ""
    for row in metacluster_table:
        mc_rows += (
            f"<tr>"
            f'<td style="font-weight:bold; text-align:center;">{row.get("metacluster", "")}</td>'
            f'<td style="text-align:right;">{row.get("n_cells", 0):,}</td>'
            f'<td style="text-align:right;">{row.get("pct", 0):.1f}%</td>'
            f"<td>{row.get('top_markers', 'N/A')}</td>"
            f"</tr>\n"
        )

    # ── Badges marqueurs ─────────────────────────────────────────────────
    markers_html = "\n".join(f'<span class="marker-badge">{m}</span>' for m in markers)

    # ── Données par condition ────────────────────────────────────────────
    cond_rows = ""
    for row in condition_data:
        cond_rows += (
            f"<tr>"
            f'<td style="font-weight:bold;">{row.get("condition", "")}</td>'
            f'<td style="text-align:right;">{row.get("n_cells", 0):,}</td>'
            f'<td style="text-align:right;">{row.get("pct", 0):.1f}%</td>'
            f"</tr>\n"
        )

    # ── Données par fichier source ────────────────────────────────────────
    files_rows = ""
    for row in files_data:
        files_rows += (
            f"<tr>"
            f"<td>{row.get('file', '')}</td>"
            f'<td style="text-align:right;">{row.get("n_cells", 0):,}</td>'
            f"</tr>\n"
        )

    # ── Table exports ─────────────────────────────────────────────────────
    _EXPORT_LABELS = {
        "csv_complete": "CSV complet",
        "fcs_complete": "FCS (Kaluza compatible)",
        "csv_statistics": "Statistiques par cluster",
        "csv_mfi": "Matrice MFI",
        "json_metadata": "Métadonnées JSON",
        "gating_log": "Log de gating JSON",
        "html_report": "Rapport HTML",
        "sankey_global": "Sankey global (HTML interactif)",
        "mrd_results": "Résultats MRD (JSON)",
    }
    export_rows = ""
    for key, path_val in export_paths.items():
        if not isinstance(path_val, str) or not path_val.endswith(
            (".csv", ".fcs", ".json", ".html", ".png")
        ):
            continue
        label = _EXPORT_LABELS.get(key, key)
        export_rows += (
            f"<tr><td>{label}</td>"
            f"<td style='font-family:monospace; font-size:0.85em;'>{path_val}</td></tr>\n"
        )

    # ── Sections Plotly ──────────────────────────────────────────────────
    plotly_sections = ""
    for fig_name, fig_obj in plotly_figures.items():
        label = figure_labels.get(fig_name, fig_name)
        try:
            div_html = _plotly_to_html_div(fig_obj, fig_id=fig_name)
            plotly_sections += (
                f'<div class="section">\n'
                f"  <h2>{label}</h2>\n"
                f'  <div class="plotly-container">{div_html}</div>\n'
                f"</div>\n"
            )
        except Exception as exc:
            _logger.warning("Erreur conversion Plotly %s: %s", fig_name, exc)

    # ── Sections Matplotlib ──────────────────────────────────────────────
    mpl_sections = ""
    if _MPL_AVAILABLE:
        for fig_name, fig_obj in matplotlib_figures.items():
            label = figure_labels.get(fig_name, fig_name)
            try:
                b64 = _fig_to_base64(fig_obj, dpi=dpi_mpl)
                mpl_sections += (
                    f'<div class="section">\n'
                    f"  <h2>{label}</h2>\n"
                    f'  <div style="text-align:center; background:#1e1e2e; padding:12px; border-radius:8px;">\n'
                    f'    <img src="data:image/png;base64,{b64}" '
                    f'style="max-width:100%; border-radius:6px; '
                    f'box-shadow:0 2px 8px rgba(0,0,0,0.1);" />\n'
                    f"  </div>\n"
                    f"</div>\n"
                )
            except Exception as exc:
                _logger.warning("Erreur conversion Matplotlib %s: %s", fig_name, exc)

    # ── Assemblage HTML ──────────────────────────────────────────────────
    _cond_section = ""
    if cond_rows:
        _cond_section = f"""
    <h3 style="margin-top:25px; margin-bottom:10px;">Par condition</h3>
    <table>
        <tr><th>Condition</th><th>Cellules</th><th>Pourcentage</th></tr>
        {cond_rows}
    </table>"""

    _files_section = ""
    if files_rows:
        _files_section = f"""
    <h3 style="margin-top:25px; margin-bottom:10px;">Par fichier source</h3>
    <table>
        <tr><th>Fichier</th><th>Cellules</th></tr>
        {files_rows}
    </table>"""

    _exports_section = ""
    if export_rows:
        _exports_section = f"""
<div class="section" id="exports">
    <h2>7. Fichiers Exportés</h2>
    <table>
        <tr><th>Type</th><th>Fichier</th></tr>
        {export_rows}
    </table>
</div>"""

    _toc_exports = (
        '\n        <li><a href="#exports">7. Fichiers exportés</a></li>'
        if export_rows
        else ""
    )

    # ── Section résumé RANSAC ─────────────────────────────────────────────────
    _ransac_section = ""
    _toc_ransac = ""
    if ransac_summary:
        ransac_rows = ""
        for fname, rdata in ransac_summary.items():
            r2_val = rdata.get("r2", float("nan"))
            slope_val = rdata.get("slope", float("nan"))
            intercept_val = rdata.get("intercept", float("nan"))
            pct_val = rdata.get("pct_singlets", rdata.get("pct", None))
            r2_str = (
                f"{r2_val:.4f}"
                if isinstance(r2_val, float) and r2_val == r2_val
                else "N/A"
            )
            slope_str = (
                f"{slope_val:.4f}"
                if isinstance(slope_val, float) and slope_val == slope_val
                else "N/A"
            )
            intercept_str = (
                f"{intercept_val:.4f}"
                if isinstance(intercept_val, float) and intercept_val == intercept_val
                else "N/A"
            )
            pct_str = f"{pct_val:.1f}%" if pct_val is not None else "N/A"
            ransac_rows += (
                f"<tr>"
                f"<td style='font-size:0.85em; font-family:monospace;'>{fname}</td>"
                f'<td style="text-align:right;">{slope_str}</td>'
                f'<td style="text-align:right;">{intercept_str}</td>'
                f'<td style="text-align:right; font-weight:bold; color:var(--primary);">{r2_str}</td>'
                f'<td style="text-align:right;">{pct_str}</td>'
                f"</tr>\n"
            )
        _ransac_section = f"""
<div class="section" id="ransac">
    <h2>8. Résumé des Modèles RANSAC (Singlets)</h2>
    <p style="color:var(--text-light); margin-bottom:15px;">
        Régression linéaire robuste FSC-H → FSC-A par fichier.
        R² mesure la qualité de la corrélation (objectif &gt; 0.85).
    </p>
    <table>
        <tr>
            <th>Fichier</th>
            <th>Pente (slope)</th>
            <th>Intercept</th>
            <th>R² (corrélation)</th>
            <th>% Singlets</th>
        </tr>
        {ransac_rows}
    </table>
</div>"""
        _toc_ransac = (
            '\n        <li><a href="#ransac">8. Résumé RANSAC (Singlets)</a></li>'
        )

    # ── Section Pré-screening CD34+/CD45dim ──────────────────────────────────
    _prescreening_section = ""
    _toc_prescreening = ""
    if prescreening_result is not None:
        ps = prescreening_result
        _ps_ratio = getattr(ps, "ratio_pct", 0.0)
        _ps_gmm = getattr(ps, "gmm_ratio_pct", 0.0)
        _ps_kde = getattr(ps, "kde_ratio_pct", 0.0)
        _ps_n_pos = getattr(ps, "n_cd34_pos", 0)
        _ps_n_neg = getattr(ps, "n_cd34_neg", 0)
        _ps_n_dim = getattr(ps, "n_cd45dim", 0)
        _ps_method = getattr(ps, "method_used", "KDE")
        _ps_level = getattr(ps, "alert_level", "none")
        _ps_interp = getattr(ps, "interpretation_warning", "")
        _ps_laip = getattr(ps, "laip_tracking_recommended", False)
        _ps_cd45_low = getattr(ps, "cd45dim_threshold_low", 0.0)
        _ps_cd45_high = getattr(ps, "cd45dim_threshold_high", 0.0)
        _ps_cd34_thr = getattr(ps, "cd34_threshold", 0.0)
        _ps_warnings = getattr(ps, "warnings", [])

        # Couleur d'alerte
        if _ps_level == "high":
            _ps_badge_color = "#FF3D6E"
            _ps_badge_text = "ALERTE ÉLEVÉE"
            _ps_border = "border-left: 4px solid #FF3D6E;"
        elif _ps_level == "moderate":
            _ps_badge_color = "#F59E0B"
            _ps_badge_text = "MODÉRÉ"
            _ps_border = "border-left: 4px solid #F59E0B;"
        else:
            _ps_badge_color = "#86efac"
            _ps_badge_text = "NORMAL"
            _ps_border = "border-left: 4px solid #86efac;"

        _ps_ref_row = (
            f"<tr style='font-weight:bold;'>"
            f"<td>{_ps_method} (référence)</td>"
            f"<td style='text-align:right;'>{_ps_n_pos:,}</td>"
            f"<td style='text-align:right;'>{_ps_n_neg:,}</td>"
            f"<td style='text-align:right; color:{_ps_badge_color};'><b>{_ps_ratio:.1f}%</b></td>"
            f"</tr>"
        )
        _ps_gmm_label = f"GMM{'&nbsp;✓' if _ps_method == 'GMM' else ''}"
        _ps_kde_label = f"KDE{'&nbsp;✓' if _ps_method == 'KDE' else ''}"
        _ps_gmm_n = int(_ps_n_dim * _ps_gmm / 100) if _ps_n_dim else 0
        _ps_kde_n = int(_ps_n_dim * _ps_kde / 100) if _ps_n_dim else 0

        _ps_comparison_rows = (
            f"<tr><td>{_ps_gmm_label}</td>"
            f"<td style='text-align:right;'>{_ps_gmm_n:,}</td>"
            f"<td style='text-align:right;'>{_ps_n_dim - _ps_gmm_n:,}</td>"
            f"<td style='text-align:right;'>{_ps_gmm:.1f}%</td></tr>"
            f"<tr><td>{_ps_kde_label}</td>"
            f"<td style='text-align:right;'>{_ps_kde_n:,}</td>"
            f"<td style='text-align:right;'>{_ps_n_dim - _ps_kde_n:,}</td>"
            f"<td style='text-align:right;'>{_ps_kde:.1f}%</td></tr>"
        )

        _ps_warns_html = ""
        for w in _ps_warnings:
            _ps_warns_html += f"<p style='color:#F59E0B; font-size:0.85em;'>⚠ {w}</p>"

        _ps_laip_html = ""
        if _ps_laip:
            _ps_laip_html = """
            <div style="margin-top:16px; padding:12px 16px;
                        background:rgba(245,158,11,0.1); border-radius:8px;
                        border-left:4px solid #F59E0B;">
                <b style="color:#F59E0B;">→ LAIP Tracking classique recommandé</b><br>
                <span style="color:#e2e8f0;">Rapport CD34+/CD45dim élevé — attention pour l'interprétation de la MRD.
                Vérifier la morphologie et contextualiser avec les données cliniques.</span>
            </div>"""

        _prescreening_section = f"""
<div class="section" id="prescreening" style="{_ps_border} padding-left:20px;">
    <h2>9. Pré-screening CD34+ / CD45dim</h2>
    <p style="color:var(--text-light); margin-bottom:15px;">
        Calcul heuristique systématique du rapport CD34+/CD45dim sur les données post-gating.
        Indépendant des paramètres de gating CD34 sélectionnés. Méthode de référence : <b>{_ps_method}</b>.
        Deux méthodes sont comparées : GMM (2 composantes) et KDE 1D (vallée entre pics).
    </p>

    <div style="display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px;">
        <div class="stat-card">
            <div class="stat-value" style="color:{_ps_badge_color};">{_ps_ratio:.1f}%</div>
            <div class="stat-label">Ratio CD34+/CD45dim ({_ps_method})</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{_ps_n_pos:,}</div>
            <div class="stat-label">CD34+ dans CD45dim</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{_ps_n_neg:,}</div>
            <div class="stat-label">CD34− dans CD45dim</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{_ps_n_dim:,}</div>
            <div class="stat-label">Total CD45dim</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:{_ps_badge_color};">{_ps_badge_text}</div>
            <div class="stat-label">Niveau d'alerte</div>
        </div>
    </div>

    <h3 style="margin-top:20px; margin-bottom:10px;">Comparaison GMM vs KDE</h3>
    <table>
        <tr>
            <th>Méthode</th>
            <th>CD34+ dans CD45dim</th>
            <th>CD34− dans CD45dim</th>
            <th>Ratio (%)</th>
        </tr>
        {_ps_comparison_rows}
    </table>

    <h3 style="margin-top:20px; margin-bottom:8px;">Paramètres de gate</h3>
    <table>
        <tr><th>Paramètre</th><th>Valeur</th></tr>
        <tr><td>CD45dim (seuil bas)</td><td>{_ps_cd45_low:.0f}</td></tr>
        <tr><td>CD45dim (seuil haut)</td><td>{_ps_cd45_high:.0f}</td></tr>
        <tr><td>CD34 seuil ({_ps_method})</td><td>{_ps_cd34_thr:.0f}</td></tr>
    </table>

    {_ps_warns_html}

    <div style="margin-top:16px; padding:12px 16px;
                background:rgba(134,239,172,0.07); border-radius:8px;">
        <i style="color:{_ps_badge_color};">{_ps_interp}</i>
    </div>
    {_ps_laip_html}
</div>"""
        _toc_prescreening = (
            '\n        <li><a href="#prescreening">9. Pré-screening CD34+/CD45dim</a></li>'
        )

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlowSOM Analysis Report — {now_str}</title>
    {plotly_script}
    <style>{_CSS}</style>
</head>
<body>

<div class="header">
    <div class="container">
        <h1>FlowSOM Analysis Report</h1>
        <div class="subtitle">
            Analyse générée le {now_str} —
            {n_cells:,} cellules · {n_markers} marqueurs · {n_clusters} métaclusters
        </div>
    </div>
</div>

<div class="container">

{_curated_banner}
{_patho_banner}
<div class="toc">
    <h3>Table des matières</h3>
    <ul>
        <li><a href="#params">1. Paramètres de l'analyse</a></li>
        <li><a href="#data">2. Résumé des données</a></li>
        <li><a href="#markers">3. Marqueurs utilisés</a></li>
        <li><a href="#metaclusters">4. Métaclusters</a></li>
        <li><a href="#plotly-viz">5. Visualisations interactives</a></li>
        <li><a href="#static-viz">6. Visualisations statiques</a></li>{_toc_exports}{_toc_ransac}{_toc_prescreening}
    </ul>
</div>

<div class="section" id="params">
    <h2>1. Paramètres de l'Analyse</h2>
    <div class="param-grid">
        {param_items}
    </div>
</div>

<div class="section" id="data">
    <h2>2. Résumé des Données</h2>
    {stats_cards}
    {_cond_section}
    {_files_section}
</div>

<div class="section" id="markers">
    <h2>3. Marqueurs Utilisés pour le Clustering</h2>
    <p style="margin-bottom:15px; color:var(--text-light);">
        {n_markers} marqueurs sélectionnés (scatter et Time exclus)
    </p>
    {markers_html}
</div>

<div class="section" id="metaclusters">
    <h2>4. Résumé des Métaclusters</h2>
    <table>
        <tr>
            <th>Métacluster</th>
            <th>Cellules</th>
            <th>% Total</th>
            <th>Top 3 Marqueurs</th>
        </tr>
        {mc_rows}
    </table>
</div>

<div id="plotly-viz">
    <div class="section">
        <h2>5. Visualisations Interactives (Plotly)</h2>
        <p style="color:var(--text-light); margin-bottom:10px;">
            {len(plotly_figures)} figures interactives — zoom, pan, hover
        </p>
    </div>
    {plotly_sections}
</div>

<div id="static-viz">
    <div class="section">
        <h2>6. Visualisations Statiques (Matplotlib)</h2>
        <p style="color:var(--text-light); margin-bottom:10px;">
            {len(matplotlib_figures)} figures haute résolution
        </p>
    </div>
    {mpl_sections}
</div>

{_exports_section}

{_ransac_section}

{_prescreening_section}

</div>

<div class="footer">
    <p>FlowSOM Analysis Pipeline Pro — Rapport généré le {now_str}</p>
    <p>{n_cells:,} cellules · {n_markers} marqueurs · {n_clusters} métaclusters</p>
</div>

</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    html_size_mb = output_path.stat().st_size / (1024 * 1024)
    _logger.info(
        "Rapport HTML exporté: %s (%.1f MB, %d Plotly + %d Matplotlib)",
        output_path.name,
        html_size_mb,
        len(plotly_figures),
        len(matplotlib_figures),
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Alias publics — compatibilité avec flowsom_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────


def fig_to_base64(fig_mpl: Any) -> str:
    """
    Alias public de ``_fig_to_base64``.

    Convertit une figure matplotlib en chaîne base64 PNG embarquable
    directement dans un bloc HTML ``<img src="data:image/png;base64,...">``.

    Args:
        fig_mpl: Figure matplotlib.

    Returns:
        Chaîne base64 encodée (str).
    """
    return _fig_to_base64(fig_mpl)


def plotly_to_html_div(fig_plotly: Any, fig_id: str = "") -> str:
    """
    Alias public de ``_plotly_to_html_div``.

    Convertit une figure Plotly en div HTML auto-contenu (sans CDN externe),
    prêt à être inséré dans un rapport HTML.

    Args:
        fig_plotly: Figure Plotly.
        fig_id: Identifiant CSS optionnel du div englobant.

    Returns:
        Chaîne HTML contenant le div Plotly.
    """
    return _plotly_to_html_div(fig_plotly, fig_id)


_BANNER_START = "<!-- MRD_CURATED_BANNER_START -->"
_BANNER_END   = "<!-- MRD_CURATED_BANNER_END -->"


def patch_curated_banner_in_html(
    html_path: Any,
    curated_mrd_percent: float,
    curated_mrd_cells: Optional[int] = None,
    curated_nodes: Optional[List[Dict[str, Any]]] = None,
    algo_gauges: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """
    Remplace le bandeau MRD Validée par l'Expert dans un rapport HTML existant
    sans régénérer l'intégralité du fichier.

    Cherche les marqueurs ``<!-- MRD_CURATED_BANNER_START -->`` /
    ``<!-- MRD_CURATED_BANNER_END -->`` déjà présents dans le HTML et remplace
    tout ce qui se trouve entre eux (inclus) par le nouveau bandeau.

    Args:
        html_path:            Chemin du fichier HTML à patcher (str ou Path).
        curated_mrd_percent:  Pourcentage MRD curé.
        curated_mrd_cells:    Nombre de cellules MRD curées.
        curated_nodes:        Liste des nœuds validés par l'expert.
        algo_gauges:          Gauges algorithmiques brutes pour la traçabilité.

    Returns:
        True si le patch a été appliqué avec succès, False sinon.
    """
    from pathlib import Path as _Path

    path = _Path(html_path)
    if not path.exists():
        return False

    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return False

    # ── Construction du nouveau bandeau ──────────────────────────────────
    _c_pct_str   = f"{curated_mrd_percent:.4f} %"
    _c_cells_str = f"{curated_mrd_cells:,} cellules" if curated_mrd_cells else ""
    _c_nodes     = curated_nodes or []
    _n_kept      = len(_c_nodes)
    _n_discarded = ""  # calculé si on a l'info
    _c_nodes_str = f"{_n_kept} nœud(s) validé(s)" if _c_nodes else "Tous les nœuds écartés"
    _c_sub       = "  ·  ".join(p for p in [_c_cells_str, _c_nodes_str] if p)

    _algo_rows_html = ""
    for g in (algo_gauges or []):
        _status     = "POSITIF" if (g.get("positive") or g.get("low_level")) else "négatif"
        _status_cls = "mrd-algo-pos" if g.get("positive") else "mrd-algo-neg"
        _algo_rows_html += (
            f"<tr>"
            f"<td>{g.get('method', '?')}</td>"
            f"<td>{g.get('pct', 0.0):.4f} %</td>"
            f"<td>{g.get('n_cells', 0):,}</td>"
            f'<td class="{_status_cls}">{_status}</td>'
            f"</tr>"
        )
    _algo_table_html = ""
    if _algo_rows_html:
        _algo_table_html = f"""
            <div class="mrd-algo-trace">
                <div class="mrd-algo-title">Valeurs algorithmiques brutes — traçabilité</div>
                <table class="mrd-algo-table">
                    <thead><tr>
                        <th>Méthode</th><th>MRD Algo (%)</th>
                        <th>Cellules</th><th>Statut</th>
                    </tr></thead>
                    <tbody>{_algo_rows_html}</tbody>
                </table>
            </div>"""

    new_banner = (
        f"{_BANNER_START}\n"
        f'<div class="mrd-curated-banner">\n'
        f'  <div class="mrd-curated-header">\n'
        f'    <span class="mrd-curated-icon">&#10003;</span>\n'
        f'    <span class="mrd-curated-label">MRD VALIDÉE PAR L\'EXPERT</span>\n'
        f'  </div>\n'
        f'  <div class="mrd-curated-value">{_c_pct_str}</div>\n'
        f'  <div class="mrd-curated-sub">{_c_sub}</div>\n'
        f'  {_algo_table_html}\n'
        f'</div>\n'
        f"{_BANNER_END}"
    )

    # ── Remplacement ou insertion ─────────────────────────────────────────
    # Cas 1 : les marqueurs sont présents (rapport généré avec cette version)
    if _BANNER_START in content and _BANNER_END in content:
        idx_start = content.index(_BANNER_START)
        idx_end   = content.index(_BANNER_END) + len(_BANNER_END)
        patched   = content[:idx_start] + new_banner + content[idx_end:]

    # Cas 2 : rapport généré avant l'ajout des marqueurs → insertion
    # juste avant la balise <div class="toc"> ou, en dernier recours, après <div class="container">
    else:
        _INSERTION_ANCHORS = [
            '<div class="toc">',
            '<div class="toc" ',
            '<div class="container">',
        ]
        inserted = False
        for anchor in _INSERTION_ANCHORS:
            idx = content.find(anchor)
            if idx != -1:
                patched = content[:idx] + new_banner + "\n" + content[idx:]
                inserted = True
                break
        if not inserted:
            return False

    try:
        path.write_text(patched, encoding="utf-8")
        return True
    except Exception:
        return False
