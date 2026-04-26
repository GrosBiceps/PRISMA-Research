"""
mrd_radar.py — Radar Chart interactif MRD pour le rapport clinique.

Visualise les nœuds MRD acceptés (approche hybride FlowSOM + scoring ELN 2022)
sous forme d'un Spider Plot Plotly avec :
  - Légende interactive à gauche (cases à cocher Plotly)
  - Radar géant aéré à droite
  - Référence NBM (Z = 0) en ligne pointillée
  - Z-scores normalisés, axe de -3.5 à +4.5

Usage :
    from src.prisma.visualization.mrd_radar import (
        plot_mrd_blast_radar_final,
    )
    fig = plot_mrd_blast_radar_final(
        mrd_result=mrd_result,
        node_zscores=zscores_dict,   # {node_id: {marker: zscore}}
        marker_names=selected_markers,
    )
    fig.write_html("mrd_radar_report.html")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.prisma.utils.logger import get_logger

_logger = get_logger("visualization.mrd_radar")

try:
    import plotly.graph_objects as go

    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    _logger.warning("plotly absent — mrd_radar non disponible (pip install plotly)")


# ─────────────────────────────────────────────────────────────────────────────
#  Constantes de design
# ─────────────────────────────────────────────────────────────────────────────

# Marqueurs radar canoniques (ordre ELN 2022 / Ogata — sens horaire)
_DEFAULT_RADAR_MARKERS = [
    "CD34",
    "CD117",
    "HLA-DR",
    "CD33",
    "CD13",
    "CD7",
    "CD56",
    "CD19",
    "CD45",
    "SSC",
]

# Limites de l'axe radial : Z = 0 au milieu
_RMIN = -3.5
_RMAX = 4.5

# Palettes : nœuds MRD (bordeaux → orange → violet selon rang)
_NODE_COLORS = [
    "#c0392b",  # bordeaux vif
    "#e67e22",  # orange
    "#8e44ad",  # violet
    "#16a085",  # teal
    "#2980b9",  # bleu
    "#d35400",  # orange foncé
    "#27ae60",  # vert
    "#c0392b",  # répétition si > 7 nœuds
    "#7f8c8d",
    "#2c3e50",
]

# Référence NBM
_NBM_COLOR = "rgba(100, 120, 180, 0.85)"
_NBM_FILL  = "rgba(100, 120, 180, 0.08)"


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers internes
# ─────────────────────────────────────────────────────────────────────────────


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convertit '#rrggbb' → 'rgba(r, g, b, alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _build_nbm_trace(radar_markers: List[str]) -> "go.Scatterpolar":
    """Trace NBM — Z = 0 sur tous les axes (ligne pointillée bleue)."""
    categories = radar_markers + [radar_markers[0]]  # fermer le polygone
    values = [0.0] * len(categories)

    return go.Scatterpolar(
        r=values,
        theta=categories,
        mode="lines",
        name="NBM — Norme Saine (Z=0)",
        line=dict(color=_NBM_COLOR, width=3, dash="dot"),
        fill="toself",
        fillcolor=_NBM_FILL,
        hovertemplate=(
            "<b>NBM — Référence saine</b><br>"
            "Marqueur : %{theta}<br>"
            "Z-score : %{r:.2f}<br>"
            "<i>Z = 0 = médiane normale</i>"
            "<extra></extra>"
        ),
        legendrank=1,
    )


def _build_node_trace(
    node_id: int,
    zscores: Dict[str, float],
    radar_markers: List[str],
    blast_score: Optional[float],
    blast_category: Optional[str],
    pct_patho: float,
    color_hex: str,
    rank: int,
) -> "go.Scatterpolar":
    """Construit le trace Scatterpolar pour un nœud MRD."""
    categories = radar_markers + [radar_markers[0]]
    values = []
    for m in radar_markers:
        z = zscores.get(m, 0.0)
        # Clipper pour rester dans [rmin, rmax] sans distordre visuellement
        values.append(float(np.clip(z, _RMIN, _RMAX)))
    values.append(values[0])  # fermer

    # Libellé légende
    score_str = f"{blast_score:.1f}/10" if blast_score is not None else "N/A"
    cat_str   = blast_category or "N/A"
    purity_str = f"{pct_patho:.1f}% Patho"

    legend_label = (
        f"Nœud #{node_id} — {score_str} {cat_str} — {purity_str}"
    )

    # Tooltip par point
    hover_lines = []
    for m, v in zip(categories, values):
        raw_z = zscores.get(m, 0.0)
        direction = "↑ sur-exprimé" if raw_z > 0.5 else ("↓ sous-exprimé" if raw_z < -0.5 else "≈ normal")
        hover_lines.append(f"{m}: Z={raw_z:.2f} ({direction})")
    hovertext = "<br>".join(hover_lines)

    return go.Scatterpolar(
        r=values,
        theta=categories,
        mode="lines+markers",
        name=legend_label,
        line=dict(color=color_hex, width=2.5),
        marker=dict(size=6, color=color_hex),
        fill="toself",
        fillcolor=_hex_to_rgba(color_hex, 0.15),
        customdata=[[blast_score, blast_category, pct_patho]] * len(values),
        hovertemplate=(
            f"<b>Nœud #{node_id}</b><br>"
            f"Score ELN : {score_str} — {cat_str}<br>"
            f"Pureté : {purity_str}<br>"
            "───────────────────<br>"
            f"{hovertext}"
            "<extra></extra>"
        ),
        legendrank=rank + 2,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Fonction principale
# ─────────────────────────────────────────────────────────────────────────────


def plot_mrd_blast_radar_final(
    mrd_result: Any,
    node_zscores: Dict[int, Dict[str, float]],
    marker_names: Optional[List[str]] = None,
    radar_markers: Optional[List[str]] = None,
    method: str = "hybrid",
    top_n: int = 10,
    title: str = "Profil MRD — Z-scores ELN 2022 / Ogata",
    output_dir: Optional[Path] = None,
    timestamp: str = "",
) -> Optional["go.Figure"]:
    """
    Génère un Radar Chart Plotly clinique pour les nœuds MRD acceptés.

    Layout : légende interactive à gauche + Radar Chart géant à droite.

    Args:
        mrd_result:     Instance ``MRDResult`` issue de ``compute_mrd()``.
                        Doit posséder l'attribut ``per_node`` (List[MRDClusterResult]).
        node_zscores:   Dict {node_id: {marker_name: zscore}}.
                        Produit par ``compute_reference_normalization()`` ou
                        ``blast_detection.score_nodes_for_blasts()``.
        marker_names:   Liste complète des marqueurs du panel (pour la validation).
        radar_markers:  Liste ordonnée des marqueurs affichés sur le radar.
                        Par défaut : _DEFAULT_RADAR_MARKERS (ELN 2022 / Ogata).
                        CD34 est TOUJOURS inclus.
        method:         Méthode MRD utilisée pour filtrer les nœuds acceptés.
                        "hybrid" | "jf" | "flo" | "eln" | "all".
        top_n:          Nombre maximum de nœuds MRD affichés (triés par blast_score).
        title:          Titre du graphique.
        output_dir:     Si fourni, sauvegarde le HTML dans ce dossier.
        timestamp:      Suffixe optionnel pour le nom de fichier.

    Returns:
        go.Figure Plotly ou None si plotly absent.
    """
    if not _PLOTLY_AVAILABLE:
        _logger.error("plotly requis pour plot_mrd_blast_radar_final")
        return None

    # ── 1. Résoudre les marqueurs radar ──────────────────────────────────────
    if radar_markers is None:
        radar_markers = list(_DEFAULT_RADAR_MARKERS)
    # Garantir la présence de CD34 (vital dans le panel)
    if "CD34" not in radar_markers:
        radar_markers.insert(0, "CD34")
        _logger.info("CD34 ajouté en tête des marqueurs radar (marqueur vital).")

    # ── 2. Filtrer les nœuds MRD acceptés ───────────────────────────────────
    per_node = getattr(mrd_result, "per_node", [])
    if not per_node:
        _logger.warning("mrd_result.per_node est vide — graphique vide généré.")

    def _is_accepted(node: Any) -> bool:
        if method == "hybrid":
            # Approche hybride : porte topologique ET porte biologique
            topo = node.is_mrd_jf or node.is_mrd_flo or node.is_mrd_eln
            bio  = node.blast_category in {"BLAST_HIGH", "BLAST_MODERATE"}
            return topo and bio
        elif method == "jf":
            return node.is_mrd_jf
        elif method == "flo":
            return node.is_mrd_flo
        elif method == "eln":
            return node.is_mrd_eln
        else:  # "all"
            return node.is_mrd_jf or node.is_mrd_flo or node.is_mrd_eln

    accepted = [n for n in per_node if _is_accepted(n)]

    # Trier par blast_score décroissant (None → score = 0)
    accepted.sort(key=lambda n: n.blast_score if n.blast_score is not None else 0.0, reverse=True)
    accepted = accepted[:top_n]

    _logger.info(
        "Radar MRD : %d/%d nœuds acceptés (méthode='%s', top_n=%d)",
        len(accepted), len(per_node), method, top_n,
    )

    # ── 3. Construction des traces ────────────────────────────────────────────
    traces: List["go.Scatterpolar"] = []

    # Trace NBM (référence Z = 0)
    traces.append(_build_nbm_trace(radar_markers))

    # Traces nœuds MRD
    for rank, node in enumerate(accepted):
        nid = node.cluster_id
        zscores = node_zscores.get(nid, {})

        if not zscores:
            _logger.warning(
                "Nœud #%d accepté mais absent de node_zscores — tracé avec Z=0.", nid
            )

        color = _NODE_COLORS[rank % len(_NODE_COLORS)]
        trace = _build_node_trace(
            node_id=nid,
            zscores=zscores,
            radar_markers=radar_markers,
            blast_score=node.blast_score,
            blast_category=node.blast_category,
            pct_patho=node.pct_patho,
            color_hex=color,
            rank=rank,
        )
        traces.append(trace)

    # ── 4. Layout ─────────────────────────────────────────────────────────────
    n_legend_items = len(accepted) + 1  # +1 pour NBM
    # Hauteur dynamique : au moins 600, +30px par nœud au-delà de 5
    fig_height = max(650, 600 + max(0, n_legend_items - 5) * 30)

    fig = go.Figure(data=traces)

    fig.update_layout(
        # ── Titre ──
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=18, color="#2d3748"),
        ),
        # ── Radar polaire ──
        polar=dict(
            domain=dict(x=[0.28, 1.0], y=[0.0, 1.0]),  # radar à droite, laisse 28% pour légende
            bgcolor="rgba(248, 249, 255, 0.95)",
            radialaxis=dict(
                range=[_RMIN, _RMAX],
                tickvals=[-3, -2, -1, 0, 1, 2, 3, 4],
                ticktext=["-3", "-2", "-1", "<b>0</b>", "+1", "+2", "+3", "+4"],
                tickfont=dict(size=11, color="#555"),
                gridcolor="rgba(180, 180, 200, 0.4)",
                linecolor="rgba(180, 180, 200, 0.6)",
                showline=True,
                zeroline=True,
                zerolinecolor="rgba(100, 120, 180, 0.5)",
                zerolinewidth=2,
                # Annotations sur les anneaux
                layer="below traces",
            ),
            angularaxis=dict(
                tickfont=dict(size=13, color="#2d3748", family="'Segoe UI', sans-serif"),
                linecolor="rgba(180, 180, 200, 0.5)",
                gridcolor="rgba(180, 180, 200, 0.3)",
                rotation=90,   # CD34 en haut (12h)
                direction="clockwise",
            ),
        ),
        # ── Légende interactive à gauche ──
        legend=dict(
            x=0.0,
            y=1.0,
            xanchor="left",
            yanchor="top",
            orientation="v",
            bgcolor="rgba(255, 255, 255, 0.92)",
            bordercolor="rgba(200, 200, 220, 0.8)",
            borderwidth=1,
            font=dict(size=12, color="#2d3748"),
            itemsizing="constant",
            itemwidth=40,
            traceorder="normal",
            # Espace entre items
            entrywidth=0,
            entrywidthmode="pixels",
        ),
        # ── Fond & marges ──
        paper_bgcolor="rgba(248, 249, 252, 1)",
        plot_bgcolor="rgba(248, 249, 252, 1)",
        height=fig_height,
        margin=dict(l=20, r=30, t=70, b=40),
        # ── Annotations contextuelles ──
        annotations=[
            dict(
                x=0.0, y=-0.02,
                xref="paper", yref="paper",
                text=(
                    "<i>Axe radial : Z-score par rapport à la médiane NBM "
                    f"(n={getattr(mrd_result, 'total_cells_sain', '?')} cellules saines) "
                    f"— {len(accepted)} nœud(s) MRD affiché(s) / "
                    f"{len(per_node)} nœuds analysés</i>"
                ),
                showarrow=False,
                font=dict(size=10, color="#718096"),
                align="left",
                xanchor="left",
            ),
            # Indicateur Z=0
            dict(
                x=0.63, y=0.5,
                xref="paper", yref="paper",
                text="Z=0",
                showarrow=False,
                font=dict(size=9, color="rgba(100,120,180,0.7)"),
                align="center",
            ),
        ],
        # Hover unifié
        hovermode="closest",
    )

    # ── 5. Sauvegarde optionnelle ─────────────────────────────────────────────
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{timestamp}" if timestamp else ""
        out_path = output_dir / f"mrd_radar_final{suffix}.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        _logger.info("Radar MRD sauvegardé : %s", out_path)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Exemple d'utilisation (module run direct)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ── Simulation d'un MRDResult minimal ─────────────────────────────────────
    from dataclasses import dataclass, field as dc_field
    from typing import List as L

    @dataclass
    class _FakeNode:
        cluster_id: int
        pct_patho: float
        is_mrd_jf: bool = True
        is_mrd_flo: bool = True
        is_mrd_eln: bool = True
        blast_score: Optional[float] = None
        blast_category: Optional[str] = None

    @dataclass
    class _FakeMRDResult:
        per_node: L[_FakeNode] = dc_field(default_factory=list)
        total_cells_sain: int = 50000

    fake_nodes = [
        _FakeNode(45, pct_patho=92.3, blast_score=8.5, blast_category="BLAST_HIGH"),
        _FakeNode(12, pct_patho=85.1, blast_score=6.2, blast_category="BLAST_MODERATE"),
        _FakeNode(78, pct_patho=71.4, blast_score=4.0, blast_category="BLAST_MODERATE"),
        _FakeNode(33, pct_patho=60.0, blast_score=2.1, blast_category="BLAST_WEAK"),  # exclu hybrid
    ]
    fake_result = _FakeMRDResult(per_node=fake_nodes)

    # Z-scores simulés (signature blastique typique : CD34↑, CD117↑, CD45↓, SSC↓)
    node_zscores: Dict[int, Dict[str, float]] = {
        45: {"CD34": 3.2, "CD117": 2.8, "HLA-DR": 1.5, "CD33": 2.1, "CD13": -0.3,
             "CD7": 0.8, "CD56": 0.2, "CD19": -0.5, "CD45": -2.1, "SSC": -1.8},
        12: {"CD34": 2.1, "CD117": 1.9, "HLA-DR": 0.8, "CD33": 1.4, "CD13": -0.8,
             "CD7": 1.2, "CD56": -0.1, "CD19": 0.3, "CD45": -1.5, "SSC": -1.2},
        78: {"CD34": 1.5, "CD117": 2.2, "HLA-DR": -0.2, "CD33": 0.9, "CD13": -1.1,
             "CD7": 0.5, "CD56": 0.4, "CD19": -0.2, "CD45": -0.9, "SSC": -0.7},
        33: {"CD34": 0.3, "CD117": 0.5, "HLA-DR": -0.1, "CD33": 0.2, "CD13": 0.1,
             "CD7": 0.0, "CD56": 0.0, "CD19": 0.0, "CD45": -0.2, "SSC": -0.3},
    }

    fig = plot_mrd_blast_radar_final(
        mrd_result=fake_result,
        node_zscores=node_zscores,
        method="hybrid",
        title="Profil MRD — Z-scores ELN 2022 / Ogata (Démo)",
        output_dir=Path("."),
        timestamp="demo",
    )

    if fig:
        print("Figure générée. Ouverture dans le navigateur...")
        fig.show()
    else:
        print("Erreur : plotly non disponible.", file=sys.stderr)
        sys.exit(1)
