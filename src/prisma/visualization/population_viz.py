"""
population_viz.py — Visualisations interactives des populations cellulaires.

Fonctions de visualisation Plotly pour :
  §10.4c — Blast scoring (heatmap, radar, bar chart)
  §10.4d — Traçabilité FCS (stacked bar par source pathologique)
  §10.5  — MST interactif coloré par population
  §10.5b — Grille SOM ScatterGL (MC / Condition / Population)
  §10.6  — Heatmap comparative CSV-ref vs MetaClusters (z-score)

Toutes les fonctions retournent un `plotly.graph_objects.Figure` et sauvegardent
le HTML dans output_dir si spécifié.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from prisma.utils.logger import get_logger

_logger = get_logger("visualization.population_viz")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    _logger.warning(
        "plotly absent — visualisations population non disponibles (pip install plotly)"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  §10.4c — Blast scoring visualizations
# ─────────────────────────────────────────────────────────────────────────────


def plot_blast_heatmap(
    blast_df: pd.DataFrame,
    marker_cols: Optional[List[str]] = None,
    title: str = "Blast Score — Profil Marqueurs (ELN 2022)",
    output_dir: Optional[Path] = None,
    timestamp: str = "",
) -> Optional["go.Figure"]:
    """
    Heatmap divergente des scores normalisés par marqueur pour les nœuds blast.

    Axe X = marqueurs, axe Y = nœuds (triés par blast_score décroissant).
    Colorscale : bleu = en-dessous référence, rouge = au-dessus.

    Args:
        blast_df: DataFrame de build_blast_score_dataframe.
        marker_cols: Colonnes '_M8' à visualiser (auto-detected si None).
        title: Titre du graphe.
        output_dir: Répertoire de sortie HTML.
        timestamp: Suffixe de nom de fichier.

    Returns:
        Figure Plotly ou None si plotly absent.
    """
    if not _PLOTLY_AVAILABLE:
        return None

    # Détecter les colonnes de valeurs normalisées
    if marker_cols is None:
        marker_cols = [c for c in blast_df.columns if c.endswith("_M8")]

    if not marker_cols:
        _logger.warning("plot_blast_heatmap: aucune colonne marqueur trouvée.")
        return None

    df_sorted = blast_df.sort_values("blast_score", ascending=False)
    Z = df_sorted[marker_cols].values.astype(float)
    y_labels = [
        f"Node {int(r['node_id'])} | {r['blast_category']} | {r['blast_score']:.1f}"
        for _, r in df_sorted.iterrows()
    ]
    x_labels = [c.replace("_M8", "") for c in marker_cols]

    fig = go.Figure(
        data=go.Heatmap(
            z=Z,
            x=x_labels,
            y=y_labels,
            colorscale="RdBu_r",
            zmid=0.0,
            colorbar=dict(title="Valeur norm."),
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(tickangle=-45),
        height=max(400, len(df_sorted) * 30 + 150),
        template="plotly_dark",
        margin=dict(l=200, r=80, t=80, b=120),
    )

    if output_dir is not None:
        _save_html(fig, output_dir, f"blast_heatmap_{timestamp}.html")

    return fig


def plot_blast_radar(
    blast_df: pd.DataFrame,
    marker_cols: Optional[List[str]] = None,
    max_nodes: int = 12,
    title: str = "Profils Blast — Radar Charts",
    output_dir: Optional[Path] = None,
    timestamp: str = "",
    ref_profiles: Optional[Dict[str, "pd.Series"]] = None,
) -> Optional["go.Figure"]:
    """
    Radar (spider) charts qualité publication : un subplot par nœud candidat blast.

    ── Design clinique ──────────────────────────────────────────────────────────

    Les valeurs affichées sont des **z-scores** (déviations vs moelle normale NBM).
    L'axe radial est centré sur z = 0 (NBM) et s'étend dans les négatifs (CD45-dim,
    SSC-bas) et les positifs (CD34++, CD117++), ce qui permet de visualiser les deux
    directions d'anomalie sur la même toile.

    Chaque subplot contient :
      • Zone saine (z = 0) : polygone de référence NBM en pointillés verts translucides.
        Tout ce qui dépasse vers l'extérieur = surexpression vs NBM.
        Tout ce qui rentre vers l'intérieur = sous-expression vs NBM (ex: CD45-dim).
      • Profil du nœud : polygone coloré selon la catégorie blast (rouge = BLAST_HIGH,
        orange = BLAST_MODERATE, jaune = BLAST_WEAK, gris = NON_BLAST).

    ── Gestion des z-scores négatifs ───────────────────────────────────────────

    Plotly en mode polar impose r ≥ 0 par défaut. Pour afficher correctement les
    z-scores négatifs (ex: CD45 z = -2.0 pour un blaste CD45-dim), on applique un
    décalage : r_display = z_score + R_OFFSET, où R_OFFSET = |r_min| + marge.
    La ligne de référence NBM (z = 0) correspond alors à r_display = R_OFFSET.
    Les cercles de grille sont étiquetés avec les vraies valeurs z (-2, -1, 0, +1, +2).

    Args:
        blast_df: DataFrame de build_blast_score_dataframe (colonnes *_M8 = z-scores).
        marker_cols: Colonnes '_M8' à afficher (auto-détectées si None).
        max_nodes: Nombre maximum de nœuds à afficher (défaut 12).
        title: Titre global de la figure.
        output_dir: Répertoire de sortie HTML.
        timestamp: Suffixe du nom de fichier.
        ref_profiles: Dict optionnel {nom_population: pd.Series(valeurs_M8_normalisées)}
                      pour superposer des profils de référence sur chaque subplot.

    Returns:
        Figure Plotly interactive, ou None si données insuffisantes.
    """
    if not _PLOTLY_AVAILABLE:
        return None

    if marker_cols is None:
        marker_cols = [c for c in blast_df.columns if c.endswith("_M8")]

    if not marker_cols:
        return None

    # ── 1. Tri : BLAST_HIGH en premier, puis score décroissant ───────────────
    category_order = ["BLAST_HIGH", "BLAST_MODERATE", "BLAST_WEAK", "NON_BLAST_UNK"]
    df_sorted = (
        blast_df.sort_values(
            ["blast_category", "blast_score"],
            key=lambda col: (
                col.map({c: i for i, c in enumerate(category_order)})
                if col.name == "blast_category"
                else col
            ),
            ascending=[True, False],
        )
        .head(max_nodes)
        .reset_index(drop=True)
    )

    n_plots = min(len(df_sorted), max_nodes)
    n_cols = min(3, n_plots)   # 3 colonnes max pour garder chaque radar lisible
    n_rows = math.ceil(n_plots / n_cols)

    # ── 2. Calcul du décalage radial pour gérer les z-scores négatifs ────────
    # Plotly polar n'affiche pas r < 0. On décale toutes les valeurs de R_OFFSET
    # pour que les z-scores négatifs restent visibles (ex: CD45 z=-2 → r=1.0).
    all_z = df_sorted[marker_cols].values.astype(float)
    z_min = float(np.nanmin(all_z))
    z_max = float(np.nanmax(all_z))
    # R_OFFSET : assure que même le z-score le plus négatif donne r > 0
    R_OFFSET = max(3.5, abs(z_min) + 0.5)
    # Plage d'affichage : [0, R_OFFSET + z_max + marge]
    r_display_max = R_OFFSET + max(z_max + 0.5, 2.0)

    # Cercles de grille aux valeurs z clés (-2, -1, 0, +1, +2, +3)
    z_ticks = [z for z in [-3, -2, -1, 0, 1, 2, 3] if -R_OFFSET <= z <= z_max + 1]
    r_ticks = [R_OFFSET + z for z in z_ticks]
    tick_labels = [f"z={z:+d}" if z != 0 else "NBM (z=0)" for z in z_ticks]

    # ── 3. Labels d'axes angulaires (noms des marqueurs) ─────────────────────
    marker_labels = [c.replace("_M8", "") for c in marker_cols]
    # Polygone fermé : on répète le premier axe à la fin
    theta_closed = marker_labels + [marker_labels[0]]

    # ── 4. Polygone de référence NBM (z = 0 sur tous les axes) ───────────────
    # Dans l'espace décalé : r_nbm = R_OFFSET pour tous les marqueurs
    r_nbm = [R_OFFSET] * len(marker_labels) + [R_OFFSET]

    # ── 5. Couleurs par catégorie blast ───────────────────────────────────────
    # Remplissage (fillcolor) = couleur ligne + alpha 0.20 pour garder la lisibilité
    _COLORS = {
        "BLAST_HIGH":     {"line": "#c0392b", "fill": "rgba(192,57,43,0.20)"},
        "BLAST_MODERATE": {"line": "#e67e22", "fill": "rgba(230,126,34,0.20)"},
        "BLAST_WEAK":     {"line": "#f1c40f", "fill": "rgba(241,196,15,0.15)"},
        "NON_BLAST_UNK":  {"line": "#7f8c8d", "fill": "rgba(127,140,141,0.12)"},
    }
    _DEFAULT_COLOR = {"line": "#95a5a6", "fill": "rgba(149,165,166,0.12)"}

    # ── 6. Titres des subplots avec score et ID nœud ─────────────────────────
    subplot_titles = []
    for i in range(n_plots):
        row = df_sorted.loc[i]
        nid = int(row["node_id"])
        score = float(row["blast_score"])
        cat = str(row["blast_category"])
        subplot_titles.append(f"Nœud #{nid} — {score:.1f}/10 ({cat})")

    specs = [[{"type": "polar"} for _ in range(n_cols)] for _ in range(n_rows)]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        # Espacement généreux entre subplots pour éviter chevauchement des titres
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )

    # ── 7. Tracé de chaque nœud ───────────────────────────────────────────────
    for i, row_data in df_sorted.iterrows():
        r_idx = (i // n_cols) + 1
        c_idx = (i % n_cols) + 1
        cat = str(row_data["blast_category"])
        colors = _COLORS.get(cat, _DEFAULT_COLOR)

        # Valeurs z-scores décalées pour affichage polar (r = z + R_OFFSET)
        z_vals = list(row_data[marker_cols].values.astype(float))
        r_vals = [z + R_OFFSET for z in z_vals]
        r_closed = r_vals + [r_vals[0]]

        # ── Trace NBM (z=0) : zone saine de référence ─────────────────────
        # fill="toself" avec couleur verte très transparente = "zone saine"
        # showlegend uniquement sur le 1er subplot pour éviter la répétition
        fig.add_trace(
            go.Scatterpolar(
                r=r_nbm,
                theta=theta_closed,
                fill="toself",
                fillcolor="rgba(39,174,96,0.08)",   # vert très transparent = zone saine
                line=dict(color="rgba(39,174,96,0.60)", width=1.5, dash="dash"),
                name="NBM (z=0)",
                hoverinfo="skip",
                showlegend=(i == 0),
                legendgroup="nbm",
            ),
            row=r_idx,
            col=c_idx,
        )

        # ── Trace profil du nœud ───────────────────────────────────────────
        node_label = f"Nœud #{int(row_data['node_id'])}"
        hover_lines = [
            f"<b>{node_label}</b><br>"
            f"Score: {float(row_data['blast_score']):.1f}/10 — {cat}<br>"
            f"<br>"
        ] + [
            f"{marker_labels[j]}: z = {z_vals[j]:+.2f}"
            for j in range(len(marker_labels))
        ]
        hover_text = "<br>".join(hover_lines) + "<extra></extra>"

        fig.add_trace(
            go.Scatterpolar(
                r=r_closed,
                theta=theta_closed,
                fill="toself",
                fillcolor=colors["fill"],
                line=dict(color=colors["line"], width=2.0),
                mode="lines+markers",
                marker=dict(size=5, color=colors["line"]),
                name=node_label,
                hovertemplate=hover_text,
                showlegend=False,
            ),
            row=r_idx,
            col=c_idx,
        )

        # ── Superposition des profils de référence optionnels ──────────────
        if ref_profiles:
            _ref_colors = ["#2980b9", "#8e44ad", "#16a085", "#d35400"]
            for _ri, (_ref_name, _ref_series) in enumerate(ref_profiles.items()):
                _ref_vals = []
                for m in marker_labels:
                    _candidates = [k for k in _ref_series.index if m in k]
                    _z = float(_ref_series[_candidates[0]]) if _candidates else 0.0
                    _ref_vals.append(_z + R_OFFSET)
                _ref_closed = _ref_vals + [_ref_vals[0]]
                fig.add_trace(
                    go.Scatterpolar(
                        r=_ref_closed,
                        theta=theta_closed,
                        mode="lines",
                        name=_ref_name,
                        line=dict(
                            color=_ref_colors[_ri % len(_ref_colors)],
                            width=1.5,
                            dash="dot",
                        ),
                        opacity=0.80,
                        showlegend=(i == 0),
                        legendgroup=f"ref_{_ri}",
                    ),
                    row=r_idx,
                    col=c_idx,
                )

    # ── 8. Mise en page des axes polaires ─────────────────────────────────────
    # Appliqué à chaque subplot via update_polars (plotly accepte polar, polar2…)
    polar_layout_common = dict(
        bgcolor="rgba(15,20,30,0.85)",
        radialaxis=dict(
            # Axe radial masqué : on contrôle l'apparence via tickvals/ticktext
            visible=True,
            range=[0, r_display_max],
            # Cercles de grille uniquement aux valeurs z clés (-2, -1, 0, +1, +2)
            tickvals=r_ticks,
            ticktext=tick_labels,
            tickfont=dict(size=8, color="rgba(180,180,180,0.75)"),
            gridcolor="rgba(100,100,100,0.25)",
            gridwidth=1,
            showline=False,
        ),
        angularaxis=dict(
            # Labels des marqueurs (CD117, CD45…) : taille accrue + espacés du bord
            tickfont=dict(size=11, color="white", family="Arial"),
            ticklabelstep=1,
            gridcolor="rgba(100,100,100,0.30)",
            linecolor="rgba(120,120,120,0.40)",
            # `rotation` + `direction` pour commencer le premier marqueur en haut
            rotation=90,
            direction="clockwise",
        ),
    )

    # Applique le layout à tous les axes polaires générés par make_subplots
    polar_updates = {}
    for k in range(1, n_plots + 1):
        key = "polar" if k == 1 else f"polar{k}"
        polar_updates[key] = polar_layout_common

    # ── 9. Layout global ──────────────────────────────────────────────────────
    cell_size_px = 320   # hauteur de chaque ligne de subplots
    fig.update_layout(
        **polar_updates,
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16, color="white"),
        ),
        # Hauteur dynamique : laisse 80 px de marge par ligne pour les titres
        height=n_rows * cell_size_px + 80,
        paper_bgcolor="rgba(10,10,20,0.95)",
        template="plotly_dark",
        legend=dict(
            font=dict(size=10),
            bgcolor="rgba(20,20,30,0.80)",
            bordercolor="rgba(150,150,150,0.40)",
            borderwidth=1,
            x=1.01,
            y=1.0,
            xanchor="left",
        ),
        annotations=[
            # Note de bas de page sur l'interprétation des z-scores
            dict(
                text=(
                    "z-scores vs moelle normale (NBM) | "
                    "Extérieur du polygone vert = surexpression | "
                    "Intérieur = sous-expression (ex: CD45-dim, SSC-bas)"
                ),
                xref="paper", yref="paper",
                x=0.5, y=-0.03,
                showarrow=False,
                font=dict(size=9, color="rgba(160,160,160,0.70)"),
                align="center",
            )
        ],
    )

    # Ajuste les titres des subplots (annotation internes) pour qu'ils soient
    # suffisamment au-dessus du graphique et ne chevauchent pas la toile
    for ann in fig.layout.annotations:
        if ann.text and "Nœud" in ann.text:
            ann.update(
                font=dict(size=11, color="rgba(220,220,220,0.90)"),
                yshift=8,
            )

    if output_dir is not None:
        _save_html(fig, output_dir, f"blast_radar_{timestamp}.html")

    return fig


def plot_blast_scores_bar(
    blast_df: pd.DataFrame,
    title: str = "Scores Blast par Nœud (ELN 2022)",
    output_dir: Optional[Path] = None,
    timestamp: str = "",
) -> Optional["go.Figure"]:
    """
    Bar chart horizontal des scores blast, coloré par catégorie clinique.

    Lignes de référence : HIGH=6, MODERATE=3.

    Args:
        blast_df: DataFrame de build_blast_score_dataframe.
        title: Titre du graphe.
        output_dir: Répertoire de sortie.
        timestamp: Suffixe du fichier.

    Returns:
        Figure Plotly.
    """
    if not _PLOTLY_AVAILABLE:
        return None

    color_map = {
        "BLAST_HIGH": "#d62728",
        "BLAST_MODERATE": "#ff7f0e",
        "BLAST_WEAK": "#ffdd55",
        "NON_BLAST_UNK": "#7f7f7f",
    }

    df_s = blast_df.sort_values("blast_score", ascending=True)
    y_labels = [f"Node {int(r['node_id'])}" for _, r in df_s.iterrows()]
    colors = [
        color_map.get(str(r["blast_category"]), "#999") for _, r in df_s.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_s["blast_score"].values,
            y=y_labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}" for v in df_s["blast_score"].values],
            textposition="outside",
            name="Score Blast",
        )
    )

    # Lignes de seuil ELN
    fig.add_vline(
        x=6.0,
        line_dash="dash",
        line_color="red",
        annotation_text="BLAST_HIGH (6)",
        annotation_position="top right",
    )
    fig.add_vline(
        x=3.0,
        line_dash="dot",
        line_color="orange",
        annotation_text="BLAST_MODERATE (3)",
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title="Score /10", range=[0, 11]),
        height=max(400, len(df_s) * 25 + 150),
        template="plotly_dark",
        showlegend=False,
    )

    if output_dir is not None:
        _save_html(fig, output_dir, f"blast_scores_bar_{timestamp}.html")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  §10.4d — Traçabilité FCS source
# ─────────────────────────────────────────────────────────────────────────────


def plot_blast_fcs_source(
    source_df: pd.DataFrame,
    title: str = "Cellules Blast par Fichier Source FCS",
    output_dir: Optional[Path] = None,
    timestamp: str = "",
) -> Optional["go.Figure"]:
    """
    Stacked bar : pour chaque nœud blast, proportion Patho/Sain par FCS source.

    Barre rouge = Patho (diagnositc/FU positif), vert = Sain/NBM.
    ALERTE CLINIQUE en annotaion si >50% Patho.

    Args:
        source_df: Résultat de trace_blast_cells_to_fcs_source.
        title: Titre du graphe.
        output_dir: Répertoire de sortie.
        timestamp: Suffixe du fichier.

    Returns:
        Figure Plotly.
    """
    if not _PLOTLY_AVAILABLE or source_df.empty:
        return None

    df_s = source_df.sort_values("blast_score", ascending=False)
    x_labels = [
        f"Node {int(r['node_id'])} ({r['blast_category']})" for _, r in df_s.iterrows()
    ]
    n_patho = df_s["n_cells_patho"].fillna(0).values
    n_sain = df_s["n_cells_sain"].fillna(0).values
    n_other = np.maximum(0, df_s["n_cells_total"].values - n_patho - n_sain)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=n_patho,
            name="Pathologique",
            marker_color="#d62728",
            text=n_patho,
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=n_sain,
            name="Sain/NBM",
            marker_color="#2ca02c",
            text=n_sain,
            textposition="inside",
        )
    )
    if n_other.sum() > 0:
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=n_other,
                name="Autre",
                marker_color="#7f7f7f",
            )
        )

    # Annotations ALERTE CLINIQUE
    for i, row in enumerate(df_s.itertuples()):
        if getattr(row, "clinical_alert", False):
            fig.add_annotation(
                x=x_labels[i],
                y=row.n_cells_total,
                text="⚠ ALERTE",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                font=dict(color="red", size=11),
                yshift=10,
            )

    fig.update_layout(
        barmode="stack",
        title=dict(text=title, x=0.5),
        xaxis=dict(tickangle=-30),
        yaxis=dict(title="N cellules"),
        height=500,
        template="plotly_dark",
    )

    if output_dir is not None:
        _save_html(fig, output_dir, f"blast_source_fcs_{timestamp}.html")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  §10.4e — Radar MRD clinique (nœuds acceptés Porte 2)
# ─────────────────────────────────────────────────────────────────────────────


def plot_mrd_blast_radar(
    blast_df: pd.DataFrame,
    mrd_per_node: Optional[List[Any]] = None,
    marker_cols: Optional[List[str]] = None,
    max_nodes: int = 10,
    title: str = "Radar MRD — Nœuds acceptés Porte 2 (BLAST_HIGH / BLAST_MODERATE)",
    output_dir: Optional[Path] = None,
    timestamp: str = "",
) -> Optional["go.Figure"]:
    """
    Radar chart Plotly dédié aux nœuds SOM acceptés comme MRD par la Porte 2
    (filtre biologique BLAST_HIGH / BLAST_MODERATE).

    Contrairement à ``plot_blast_radar()`` qui génère des subplots par nœud,
    cette fonction produit **un seul radar** avec une trace par nœud, toutes
    activables/désactivables via la légende.  Un polygone de référence NBM
    (z = 0, moelle saine) est affiché en fond, ainsi qu'une zone de tolérance
    ±1 σ en gris translucide.

    Chaque trace indique dans son label :
      • L'ID du nœud (Nœud #45)
      • Le blast_score et la catégorie (Score: 7.5/10 – BLAST_HIGH)
      • Le pourcentage de pureté pathologique si disponible (95 % Patho)

    Args:
        blast_df: DataFrame produit par ``build_blast_score_dataframe()``.
                  Colonnes requises : ``node_id``, ``blast_score``,
                  ``blast_category``, plus les colonnes ``{marker}_M8``.
        mrd_per_node: Liste de ``MRDClusterResult`` (optionnel).  Utilisée pour
                      récupérer ``pct_patho`` par nœud.  Si ``None``, le
                      pourcentage n'est pas affiché dans la légende.
        marker_cols: Colonnes ``_M8`` à inclure (auto-détectées si ``None``).
        max_nodes: Nombre max de nœuds à afficher, triés par ``blast_score``
                   décroissant (défaut : 10).
        title: Titre du graphique.
        output_dir: Répertoire de sortie HTML (sous-dossier ``other/``).
        timestamp: Suffixe ajouté au nom de fichier.

    Returns:
        Figure Plotly interactive, ou ``None`` si Plotly est absent ou si
        aucun nœud MRD n'est éligible.
    """
    if not _PLOTLY_AVAILABLE:
        return None

    # ── 1. Filtrer les nœuds MRD acceptés (Porte 2) ──────────────────────────
    mrd_categories = {"BLAST_HIGH", "BLAST_MODERATE"}
    df_mrd = blast_df[blast_df["blast_category"].isin(mrd_categories)].copy()

    if df_mrd.empty:
        _logger.warning(
            "plot_mrd_blast_radar: aucun nœud BLAST_HIGH/MODERATE dans blast_df."
        )
        return None

    # ── 2. Détecter les colonnes de marqueurs ────────────────────────────────
    if marker_cols is None:
        marker_cols = [c for c in blast_df.columns if c.endswith("_M8")]

    if not marker_cols:
        _logger.warning("plot_mrd_blast_radar: aucune colonne marqueur _M8 trouvée.")
        return None

    # ── 3. Trier par blast_score décroissant, limiter à max_nodes ────────────
    df_mrd = df_mrd.sort_values("blast_score", ascending=False).head(max_nodes).reset_index(drop=True)

    # ── 4. Construire la table de pureté patho depuis mrd_per_node ───────────
    pct_patho_map: Dict[int, float] = {}
    if mrd_per_node is not None:
        for node_res in mrd_per_node:
            nid = getattr(node_res, "cluster_id", None)
            pct = getattr(node_res, "pct_patho", None)
            if nid is not None and pct is not None:
                pct_patho_map[int(nid)] = float(pct)

    # ── 5. Axes du radar ─────────────────────────────────────────────────────
    marker_labels = [c.replace("_M8", "") for c in marker_cols]
    # Fermer le polygone en répétant le premier axe
    theta_closed = marker_labels + [marker_labels[0]]

    fig = go.Figure()

    # ── 6. Zone de tolérance NBM ±1 σ (gris très translucide) ────────────────
    # Représente la variabilité intra-NBM : z ∈ [−1, +1] pour chaque axe.
    # On construit un polygone extérieur (+1) et intérieur (−1) combinés
    # via fillcolor pour créer la bande.
    ones_outer = [1.0] * len(marker_labels) + [1.0]
    ones_inner = [-1.0] * len(marker_labels) + [-1.0]

    fig.add_trace(
        go.Scatterpolar(
            r=ones_outer,
            theta=theta_closed,
            fill="toself",
            fillcolor="rgba(120,120,120,0.10)",
            line=dict(color="rgba(180,180,180,0.0)", width=0),
            name="NBM ±1σ (tolérance)",
            hoverinfo="skip",
            showlegend=True,
            legendgroup="reference",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=ones_inner,
            theta=theta_closed,
            fill="toself",
            fillcolor="rgba(30,30,30,1.0)",  # masque l'intérieur pour créer la bande
            line=dict(color="rgba(180,180,180,0.0)", width=0),
            name=None,
            hoverinfo="skip",
            showlegend=False,
            legendgroup="reference",
        )
    )

    # ── 7. Polygone de référence NBM (z = 0 sur tous les axes) ───────────────
    zeros = [0.0] * len(theta_closed)
    fig.add_trace(
        go.Scatterpolar(
            r=zeros,
            theta=theta_closed,
            mode="lines+markers",
            line=dict(color="rgba(100,200,100,0.9)", width=2, dash="dash"),
            marker=dict(size=5, color="rgba(100,200,100,0.9)"),
            fill=None,
            name="Référence NBM (z = 0)",
            hovertemplate="Référence NBM<br>%{theta}: z = 0<extra></extra>",
            showlegend=True,
            legendgroup="reference",
        )
    )

    # ── 8. Traces des nœuds MRD ───────────────────────────────────────────────
    color_map = {
        "BLAST_HIGH": "#ef4444",      # rouge vif
        "BLAST_MODERATE": "#f97316",  # orange
    }
    # Palette de nuances pour différencier plusieurs nœuds HIGH ou MODERATE
    _shades_high = [
        "#ef4444", "#dc2626", "#b91c1c", "#991b1b", "#7f1d1d",
        "#f87171", "#fca5a5", "#fee2e2", "#fecaca", "#fde8e8",
    ]
    _shades_mod = [
        "#f97316", "#ea580c", "#c2410c", "#9a3412", "#7c2d12",
        "#fb923c", "#fdba74", "#fed7aa", "#ffedd5", "#fff7ed",
    ]
    _high_idx = 0
    _mod_idx = 0

    for _, row in df_mrd.iterrows():
        node_id = int(row["node_id"])
        score = float(row["blast_score"])
        cat = str(row["blast_category"])
        pct = pct_patho_map.get(node_id)

        # Label de légende complet
        pct_str = f" | {pct:.0f}% Patho" if pct is not None else ""
        legend_label = f"Nœud #{node_id} | Score: {score:.1f}/10 – {cat}{pct_str}"

        # Couleur individualisée par nœud pour lisibilité
        if cat == "BLAST_HIGH":
            color = _shades_high[_high_idx % len(_shades_high)]
            _high_idx += 1
        else:
            color = _shades_mod[_mod_idx % len(_shades_mod)]
            _mod_idx += 1

        values = list(row[marker_cols].values.astype(float))
        values_closed = values + [values[0]]

        # Convertir la couleur hex en rgba pour le fillcolor (alpha 15%)
        # Plotly accepte rgba() mais pas les hex 8 caractères (#rrggbbaa)
        r_hex = int(color[1:3], 16)
        g_hex = int(color[3:5], 16)
        b_hex = int(color[5:7], 16)
        fill_rgba = f"rgba({r_hex},{g_hex},{b_hex},0.15)"

        pct_line = f"<br>Pureté patho: {pct:.0f}%" if pct is not None else ""
        hover_tpl = (
            f"<b>Nœud #{node_id}</b><br>"
            f"Score: {score:.1f}/10 \u2013 {cat}"
            f"{pct_line}"
            "<br>Marqueur: %{theta}<br>z-score: %{r}<extra></extra>"
        )

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=theta_closed,
                fill="toself",
                fillcolor=fill_rgba,
                line=dict(color=color, width=2.5),
                mode="lines+markers",
                marker=dict(size=6, color=color),
                name=legend_label,
                hovertemplate=hover_tpl,
                legendgroup=f"node_{node_id}",
                showlegend=True,
            )
        )

    # ── 9. Mise en page ───────────────────────────────────────────────────────
    # Calcul de la plage radiale : max absolu des z-scores + marge
    all_values = df_mrd[marker_cols].values.astype(float)
    r_max = max(2.0, float(np.nanmax(np.abs(all_values))) + 0.5)

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=18),
        ),
        polar=dict(
            bgcolor="rgba(20,20,30,0.9)",
            # Décale le radar vers la gauche pour laisser de la place à la légende
            domain=dict(x=[0.0, 0.62], y=[0.0, 1.0]),
            radialaxis=dict(
                visible=True,
                range=[-r_max, r_max],
                tickfont=dict(size=11, color="rgba(200,200,200,0.85)"),
                gridcolor="rgba(100,100,100,0.3)",
                tickvals=[-2, -1, 0, 1, 2],
                ticktext=["-2σ", "-1σ", "NBM", "+1σ", "+2σ"],
                showticklabels=True,
            ),
            angularaxis=dict(
                tickfont=dict(size=14, color="white"),
                gridcolor="rgba(100,100,100,0.4)",
                linecolor="rgba(150,150,150,0.5)",
            ),
        ),
        legend=dict(
            title=dict(text="Nœuds MRD acceptés", font=dict(size=13)),
            font=dict(size=11),
            bgcolor="rgba(20,20,30,0.85)",
            bordercolor="rgba(150,150,150,0.4)",
            borderwidth=1,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            x=0.65,
            y=0.98,
            xanchor="left",
            yanchor="top",
        ),
        template="plotly_dark",
        height=850,
        margin=dict(l=60, r=20, t=80, b=80),
        paper_bgcolor="rgba(10,10,20,0.95)",
        annotations=[
            dict(
                text=(
                    "Les valeurs sont des z-scores relatifs à la moelle saine de référence (NBM).<br>"
                    "z > +1 = sur-expression | z < −1 = sous-expression | "
                    "Zone grise = tolérance NBM ±1σ"
                ),
                xref="paper",
                yref="paper",
                x=0.31,
                y=-0.07,
                showarrow=False,
                font=dict(size=10, color="rgba(180,180,180,0.7)"),
                align="center",
            )
        ],
    )

    if output_dir is not None:
        _save_html(fig, output_dir, f"mrd_blast_radar_{timestamp}.html")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  §10.5 — MST interactif coloré par population
# ─────────────────────────────────────────────────────────────────────────────


def plot_mst_interactive(
    mapping_df: pd.DataFrame,
    node_coords_df: pd.DataFrame,
    node_counts: np.ndarray,
    mst_edges: Optional[List[Tuple[int, int]]] = None,
    color_map: Optional[Dict[str, str]] = None,
    mc_per_node: Optional[np.ndarray] = None,
    title: str = "MST FlowSOM — Populations cytométriques",
    output_dir: Optional[Path] = None,
    timestamp: str = "",
    dashboard_dir: Optional[Path] = None,
) -> Optional["go.Figure"]:
    """
    MST interactif Plotly : bulles colorées par population, taille ∝ n_cells.

    Chaque nœud est une bulle :
      - Couleur → population assignée
      - Taille  → √(n_cells) (rendu visuel équilibré)
      - Hover   → node_id | pop | mc | n_cells | dist | seuil

    Les arêtes MST sont tracées en gris transparent.

    Args:
        mapping_df: Résultat de map_populations_to_nodes_v5 (colonnes:
                    node_id, assigned_pop, best_dist, threshold, method).
        node_coords_df: DataFrame (n_nodes, [xNodes, yNodes, xGrid, yGrid]).
        node_counts: np.ndarray (n_nodes,) — cellules par nœud.
        mst_edges: Liste de (i, j) 0-indexés (None = pas d'arêtes MST).
        color_map: dict {pop_name: hex_color}.
        mc_per_node: Métacluster par nœud (optionnel pour le hover).
        title: Titre du graphe.
        output_dir: Répertoire global de sortie.
        timestamp: Suffixe du fichier.
        dashboard_dir: Répertoire dashboard additionnel (copie du HTML).

    Returns:
        Figure Plotly.
    """
    if not _PLOTLY_AVAILABLE:
        return None

    from clinical_module.phenotyping.population_mapping import (
        build_population_color_map,
    )

    pops = list(mapping_df["assigned_pop"].unique())
    if color_map is None:
        color_map = build_population_color_map(pops)

    # Choisir les coordonnées de layout
    x_col = "xNodes" if "xNodes" in node_coords_df.columns else "xGrid"
    y_col = "yNodes" if "yNodes" in node_coords_df.columns else "yGrid"

    n_nodes = len(mapping_df)
    x_pos = np.zeros(n_nodes)
    y_pos = np.zeros(n_nodes)
    for i, node_id in enumerate(mapping_df["node_id"].values):
        if node_id < len(node_coords_df):
            x_pos[i] = float(node_coords_df.loc[node_id, x_col])
            y_pos[i] = float(node_coords_df.loc[node_id, y_col])

    fig = go.Figure()

    # ── Arêtes MST ────────────────────────────────────────────────────────────
    if mst_edges:
        node_id_map = {
            int(row["node_id"]): idx
            for idx, (_, row) in enumerate(mapping_df.iterrows())
        }
        for e_from, e_to in mst_edges:
            i1 = node_id_map.get(e_from)
            i2 = node_id_map.get(e_to)
            if i1 is not None and i2 is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[x_pos[i1], x_pos[i2], None],
                        y=[y_pos[i1], y_pos[i2], None],
                        mode="lines",
                        line=dict(color="rgba(150,150,150,0.3)", width=0.8),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    # ── Nœuds par population ─────────────────────────────────────────────────
    for pop in sorted(pops):
        mask = mapping_df["assigned_pop"] == pop
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue

        node_ids = mapping_df.loc[mask, "node_id"].values
        counts = np.array(
            [node_counts[nid] if nid < len(node_counts) else 1 for nid in node_ids]
        )
        sizes = np.sqrt(np.maximum(counts, 1)) * 3.0
        sizes = np.clip(sizes, 5, 40)

        mc_labels = [
            f"MC{int(mc_per_node[nid])}"
            if mc_per_node is not None and nid < len(mc_per_node)
            else "N/A"
            for nid in node_ids
        ]
        dists = mapping_df.loc[mask, "best_dist"].values
        thresholds = mapping_df.loc[mask, "threshold"].values

        hover_texts = [
            f"Nœud {nid}<br>Population: {pop}<br>MC: {mc}<br>"
            f"N cellules: {cnt:.0f}<br>Dist: {d:.4f}<br>Seuil: {th:.4f}"
            for nid, mc, cnt, d, th in zip(
                node_ids, mc_labels, counts, dists, thresholds
            )
        ]

        fig.add_trace(
            go.Scatter(
                x=x_pos[idxs],
                y=y_pos[idxs],
                mode="markers",
                name=pop,
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    size=sizes,
                    color=color_map.get(pop, "#7f7f7f"),
                    line=dict(width=0.5, color="white"),
                    opacity=0.85,
                ),
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(showgrid=False, zeroline=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, title=""),
        template="plotly_dark",
        height=700,
        legend=dict(title="Population"),
        hovermode="closest",
    )

    fname = f"mst_population_v3_{timestamp}.html"
    if output_dir is not None:
        _save_html(fig, output_dir, fname)
    if dashboard_dir is not None:
        _save_html(fig, dashboard_dir, fname)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  §10.5b — Grille SOM ScatterGL interactive
# ─────────────────────────────────────────────────────────────────────────────


def plot_som_grid_interactive(
    cell_data: Any,
    mapping_df: pd.DataFrame,
    color_map: Optional[Dict[str, str]] = None,
    condition_colors: Optional[Dict[str, str]] = None,
    viz_max_points: int = 50_000,
    title: str = "Grille SOM FlowSOM — Vue Populations",
    output_dir: Optional[Path] = None,
    timestamp: str = "",
    dashboard_dir: Optional[Path] = None,
) -> Optional["go.Figure"]:
    """
    Trois ScatterGL côte à côte sur la grille SOM : Métacluster | Condition | Population.

    Utilise un jitter circulaire pour disperser les cellules à l'intérieur
    de chaque nœud (style FlowSOM-R).

    Args:
        cell_data: AnnData avec .obs (colonnes: FlowSOM_cluster, xGrid, yGrid,
                   FlowSOM_metacluster, Condition).
        mapping_df: Résultat du mapping population → nœud.
        color_map: dict {pop_name: hex_color}.
        condition_colors: dict {condition_name: hex_color}.
        viz_max_points: Maximum de cellules à afficher (sous-échantillonnage).
        title: Titre global.
        output_dir: Répertoire de sortie.
        timestamp: Suffixe du fichier.
        dashboard_dir: Répertoire dashboard additionnel.

    Returns:
        Figure Plotly avec 3 subplots.
    """
    if not _PLOTLY_AVAILABLE:
        return None

    # ── Préparer les données cellulaires ─────────────────────────────────────
    try:
        obs = cell_data.obs.copy()
    except AttributeError:
        if isinstance(cell_data, pd.DataFrame):
            obs = cell_data.copy()
        else:
            _logger.warning("cell_data inaccessible pour SOM grid.")
            return None

    # Sous-échantillonnage si nécessaire
    if len(obs) > viz_max_points:
        rng = np.random.default_rng(42)
        idx_sample = rng.choice(len(obs), viz_max_points, replace=False)
        obs = obs.iloc[idx_sample].reset_index(drop=True)

    # Colonnes nécessaires
    required = ["FlowSOM_cluster", "xGrid", "yGrid"]
    for col in required:
        if col not in obs.columns:
            _logger.warning(
                "Colonne '%s' absente de cell_data.obs — SOM grid incomplet.", col
            )

    # Jitter circulaire style FlowSOM-R
    n_cells = len(obs)
    rng = np.random.default_rng(42)
    angles = rng.uniform(0, 2 * np.pi, n_cells)
    radii = rng.uniform(0, 0.4, n_cells)
    jitter_x = radii * np.cos(angles)
    jitter_y = radii * np.sin(angles)

    if "xGrid" in obs.columns and "yGrid" in obs.columns:
        x_cells = obs["xGrid"].to_numpy(dtype=float) + jitter_x
        y_cells = obs["yGrid"].to_numpy(dtype=float) + jitter_y
    else:
        x_cells = jitter_x
        y_cells = jitter_y

    # Construire le map nœud → population assignée
    node_to_pop = {}
    if "node_id" in mapping_df.columns and "assigned_pop" in mapping_df.columns:
        node_to_pop = dict(zip(mapping_df["node_id"], mapping_df["assigned_pop"]))

    # Couleurs par population
    from clinical_module.phenotyping.population_mapping import (
        build_population_color_map,
    )

    all_pops = (
        list(mapping_df["assigned_pop"].unique())
        if "assigned_pop" in mapping_df.columns
        else []
    )
    if color_map is None:
        color_map = build_population_color_map(all_pops)

    # Populations par cellule
    if "FlowSOM_cluster" in obs.columns:
        cluster_arr = obs["FlowSOM_cluster"].to_numpy(dtype=np.int32) - 1  # 0-indexed
        pop_labels = np.array(
            [node_to_pop.get(int(nid), "Unknown") for nid in cluster_arr]
        )
    else:
        pop_labels = np.full(n_cells, "Unknown", dtype=object)

    pop_colors = np.array([color_map.get(p, "#7f7f7f") for p in pop_labels])

    # Métacluster par cellule
    if "FlowSOM_metacluster" in obs.columns:
        mc_arr = obs["FlowSOM_metacluster"].to_numpy(dtype=np.int32)
        n_mc = max(int(mc_arr.max()), 1) + 1
        mc_palette = (
            px.colors.qualitative.Plotly if _PLOTLY_AVAILABLE else ["#999"] * 50
        )
        mc_colors = np.array([mc_palette[int(m) % len(mc_palette)] for m in mc_arr])
        mc_labels = [f"MC{int(m)}" for m in mc_arr]
    else:
        mc_colors = np.full(n_cells, "#7f7f7f", dtype=object)
        mc_labels = ["N/A"] * n_cells

    # Condition par cellule
    if "Condition" in obs.columns:
        cond_arr = obs["Condition"].astype(str).values
    elif condition_colors and len(obs.columns) > 0:
        cond_arr = np.full(n_cells, "Unknown", dtype=object)
    else:
        cond_arr = np.full(n_cells, "Unknown", dtype=object)

    default_cond_colors = {
        "Sain": "#2ca02c",
        "Normal": "#2ca02c",
        "NBM": "#2ca02c",
        "Patho": "#d62728",
        "Diag": "#d62728",
        "Dx": "#d62728",
        "FU": "#ff7f0e",
        "Suivi": "#ff7f0e",
        "Unknown": "#7f7f7f",
    }
    if condition_colors:
        default_cond_colors.update(condition_colors)
    cond_colors = np.array(
        [
            next(
                (v for k, v in default_cond_colors.items() if k.upper() in c.upper()),
                "#7f7f7f",
            )
            for c in cond_arr
        ]
    )

    # ── Construction des 3 subplots ──────────────────────────────────────────
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Métacluster", "Condition", "Population cellulaire"],
        shared_yaxes=True,
    )

    common_kwargs = dict(
        x=x_cells,
        y=y_cells,
        mode="markers",
        marker=dict(size=2, opacity=0.6),
        showlegend=False,
        hoverinfo="skip",
    )

    # Subplot 1 : Métacluster
    fig.add_trace(
        go.Scattergl(
            **common_kwargs,
            marker=dict(size=2, color=mc_colors, opacity=0.6),
            text=mc_labels,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Subplot 2 : Condition
    fig.add_trace(
        go.Scattergl(
            **common_kwargs,
            marker=dict(size=2, color=cond_colors, opacity=0.6),
            text=cond_arr,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Subplot 3 : Population (légende)
    for pop in sorted(set(pop_labels)):
        mask_pop = pop_labels == pop
        fig.add_trace(
            go.Scattergl(
                x=x_cells[mask_pop],
                y=y_cells[mask_pop],
                mode="markers",
                name=pop,
                marker=dict(size=2, color=color_map.get(pop, "#7f7f7f"), opacity=0.6),
                showlegend=True,
                hovertemplate=f"{pop}<extra></extra>",
            ),
            row=1,
            col=3,
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=600,
        template="plotly_dark",
        legend=dict(title="Population", itemsizing="constant"),
    )
    for ax in ["xaxis", "xaxis2", "xaxis3", "yaxis"]:
        fig.update_layout(**{ax: dict(showgrid=False, zeroline=False)})

    fname = f"som_grid_populations_{timestamp}.html"
    if output_dir is not None:
        _save_html(fig, output_dir, fname)
    if dashboard_dir is not None:
        _save_html(fig, dashboard_dir, fname)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  §10.6 — Heatmap comparative CSV-ref vs MetaClusters (z-score)
# ─────────────────────────────────────────────────────────────────────────────


def plot_heatmap_comparative(
    df_ref_mfi: pd.DataFrame,
    node_mfi_df: pd.DataFrame,
    mc_per_node: Optional[np.ndarray] = None,
    method_name: str = "M12",
    title: str = "Heatmap Comparative — CSV Référence vs MetaClusters SOM",
    output_dir: Optional[Path] = None,
    timestamp: str = "",
    dashboard_dir: Optional[Path] = None,
) -> Optional["go.Figure"]:
    """
    Double heatmap z-score : populations de référence CSV (gauche) vs
    centroïdes des métaclusters SOM (droite).

    Permet de vérifier visuellement la cohérence entre le mapping calculé
    et les profils biologiques de référence.

    Args:
        df_ref_mfi: DataFrame [n_pops × n_markers] — MFI CSV de référence.
        node_mfi_df: DataFrame [n_nodes × n_markers] — centroïdes SOM.
        mc_per_node: Vecteur métacluster par nœud (groupement par MC).
        method_name: Méthode de mapping pour le titre.
        title: Titre global.
        output_dir: Répertoire de sortie.
        timestamp: Suffixe du fichier.
        dashboard_dir: Répertoire dashboard.

    Returns:
        Figure Plotly avec 2 subplots côte à côte.
    """
    if not _PLOTLY_AVAILABLE:
        return None

    # Aligner les colonnes communes
    cols_common = sorted(set(df_ref_mfi.columns) & set(node_mfi_df.columns))
    if not cols_common:
        _logger.warning("plot_heatmap_comparative: aucune colonne commune.")
        return None

    X_ref = df_ref_mfi[cols_common].values.astype(float)
    pop_labels = list(df_ref_mfi.index)

    # Calculer les centroïdes par métacluster si mc_per_node fourni
    if mc_per_node is not None and len(mc_per_node) == len(node_mfi_df):
        unique_mcs = sorted(set(mc_per_node.tolist()))
        mc_means: List[np.ndarray] = []
        mc_row_labels: List[str] = []
        for mc in unique_mcs:
            mask_mc = mc_per_node == mc
            if mask_mc.any():
                mc_means.append(
                    node_mfi_df.iloc[np.where(mask_mc)[0]][cols_common].mean().values
                )
                mc_row_labels.append(f"MC{int(mc)}")
        X_mc = (
            np.array(mc_means)
            if mc_means
            else node_mfi_df[cols_common].values.astype(float)
        )
        mc_labels = (
            mc_row_labels if mc_row_labels else [f"Node {i}" for i in range(len(X_mc))]
        )
    else:
        X_mc = node_mfi_df[cols_common].values.astype(float)
        mc_labels = [f"Node {i}" for i in range(len(X_mc))]

    # Z-score par colonne (marqueur)
    def _zscore_cols(X: np.ndarray) -> np.ndarray:
        std = X.std(axis=0)
        std[std == 0] = 1.0
        return (X - X.mean(axis=0)) / std

    Z_ref = _zscore_cols(X_ref)
    Z_mc = _zscore_cols(X_mc)

    zmin = min(Z_ref.min(), Z_mc.min(), -2.0)
    zmax = max(Z_ref.max(), Z_mc.max(), 2.0)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Populations de référence CSV (z-score)",
            f"Métaclusters SOM — méthode {method_name} (z-score)",
        ],
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Heatmap(
            z=Z_ref,
            x=cols_common,
            y=pop_labels,
            colorscale="RdBu_r",
            zmin=zmin,
            zmax=zmax,
            zmid=0.0,
            colorbar=dict(title="z-score", x=0.44, len=0.8),
            showscale=True,
            hoverongaps=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=Z_mc,
            x=cols_common,
            y=mc_labels,
            colorscale="RdBu_r",
            zmin=zmin,
            zmax=zmax,
            zmid=0.0,
            showscale=False,
            hoverongaps=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=max(500, max(len(pop_labels), len(mc_labels)) * 22 + 200),
        template="plotly_dark",
        margin=dict(l=180, r=60, t=80, b=150),
    )
    for ax in ["xaxis", "xaxis2"]:
        fig.update_layout(**{ax: dict(tickangle=-45)})

    fname = f"heatmap_{method_name}_{timestamp}.html"
    if output_dir is not None:
        _save_html(fig, output_dir, fname)
    if dashboard_dir is not None:
        _save_html(fig, dashboard_dir, fname)

    # Log de la table de correspondance population → MC dominant
    _log_population_mc_correspondence(
        df_ref_mfi, node_mfi_df, mc_per_node, cols_common, pop_labels
    )

    return fig


def _log_population_mc_correspondence(
    df_ref_mfi: pd.DataFrame,
    node_mfi_df: pd.DataFrame,
    mc_per_node: Optional[np.ndarray],
    cols_common: List[str],
    pop_labels: List[str],
) -> None:
    """Log la table : Population → MC dominant + Top 3 MC par cosine."""
    if mc_per_node is None:
        return

    from scipy.spatial.distance import cdist

    X_ref = df_ref_mfi[cols_common].values.astype(float)
    Z_ref = X_ref - X_ref.mean(axis=0)

    unique_mcs = sorted(set(mc_per_node.tolist()))
    mc_means = {
        mc: node_mfi_df.iloc[np.where(mc_per_node == mc)[0]][cols_common].mean().values
        for mc in unique_mcs
        if (mc_per_node == mc).any()
    }

    if not mc_means:
        return

    mc_keys = list(mc_means.keys())
    X_mc = np.array([mc_means[k] for k in mc_keys])
    Z_mc = X_mc - X_mc.mean(axis=0)

    D = cdist(Z_ref, Z_mc, metric="cosine")

    _logger.info("Table correspondance Population → Métacluster:")
    _logger.info("  %-30s | MC dominant | Top 3 MC", "Population")
    _logger.info("  " + "-" * 65)
    for i, pop in enumerate(pop_labels):
        sorted_mc_idx = np.argsort(D[i])
        dom_mc = mc_keys[sorted_mc_idx[0]]
        top3 = [f"MC{mc_keys[j]}" for j in sorted_mc_idx[:3]]
        _logger.info("  %-30s | MC%-8d | %s", pop, dom_mc, ", ".join(top3))


# ─────────────────────────────────────────────────────────────────────────────
#  Utilitaire de sauvegarde HTML
# ─────────────────────────────────────────────────────────────────────────────


def _save_html(
    fig: "go.Figure",
    output_dir: Path,
    filename: str,
    include_plotlyjs: str = "cdn",
) -> Path:
    """Sauvegarde un figure Plotly en HTML dans output_dir/other/."""
    dest_dir = Path(output_dir) / "other"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    fig.write_html(str(dest), include_plotlyjs=include_plotlyjs)
    _logger.info("Sauvegardé: %s", dest)
    return dest


# ─────────────────────────────────────────────────────────────────────────────
#  Utilitaires — profil MFI moyen et z-score DataFrame
# ─────────────────────────────────────────────────────────────────────────────


def get_mean_profile(
    node_ids: List[int],
    mfi_df: pd.DataFrame,
    node_count_raw: Optional[np.ndarray] = None,
) -> Optional[pd.Series]:
    """
    Profil MFI moyen pondéré par la taille des nœuds SOM spécifiés.

    Utilisé dans la sous-classification lymphocytaire (T_NK vs lymphocytes B)
    pour choisir le profil de référence le plus représentatif.

    Args:
        node_ids: Indices des nœuds à moyenner (0-indexés dans mfi_df).
        mfi_df: DataFrame [n_nodes × n_markers] avec les MFI des nœuds.
        node_count_raw: Taille brute de chaque nœud sous forme de vecteur
                        (n_nodes,). Si None, tous les nœuds sont pondérés
                        uniformément.

    Returns:
        pd.Series de shape (n_markers,) avec les MFI pondérées,
        ou None si aucun node_id n'est valide.
    """
    valid_ids = [i for i in node_ids if 0 <= i < len(mfi_df)]
    if not valid_ids:
        return None

    sub = mfi_df.iloc[valid_ids]

    if node_count_raw is not None:
        raw_weights = np.array([node_count_raw[i] for i in valid_ids], dtype=np.float64)
        weights = np.maximum(raw_weights, 1.0)
        weighted_mean = np.average(
            sub.values.astype(np.float64), axis=0, weights=weights
        )
    else:
        weighted_mean = sub.mean(axis=0).values.astype(np.float64)

    return pd.Series(weighted_mean, index=sub.columns)


def zscore_df(df: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    """
    Z-score par colonne d'un DataFrame.

    Standardise chaque marqueur (colonne) pour avoir μ=0 et σ=1.
    Les colonnes de variance nulle sont renvoyées à 0.

    Args:
        df: DataFrame numérique [n_rows × n_cols].
        ddof: Degrés de liberté pour l'écart-type (défaut 1 = σ empirique).

    Returns:
        DataFrame de même shape et index/colonnes avec les z-scores.
    """
    mu = df.mean(axis=0)
    sig = df.std(axis=0, ddof=ddof).replace(0.0, 1.0)
    return (df - mu) / sig


# ─────────────────────────────────────────────────────────────────────────────
#  §10.5 — Vérification biologique des nœuds Lymphocytes bruts
# ─────────────────────────────────────────────────────────────────────────────

# Jeux de labels pour la classification lymphocytaire
_LYMPHO_GENERIC: set = {"Lymphos", "Lymphocytes", "Lympho", "Lymphs"}
_LYTNK_LABELS: set = {"Ly T_NK", "Lymphos T", "Lymphos T/NK", "T_NK", "T NK"}
_LYMPHOB_LABELS: set = {"Lymphos B", "Lymphocytes B", "B cells", "B lymphocytes"}


def plot_lympho_verification(
    mapping_df: pd.DataFrame,
    node_mfi_df: pd.DataFrame,
    node_counts: Any,
    mc_per_node: Any,
    map_name: str,
    output_dir: Optional[Path],
    timestamp: str = "",
) -> Optional["go.Figure"]:
    """
    §10.5 — Vérification biologique des nœuds 'Lymphocytes bruts'.

    Les nœuds étiquetés comme lymphocytes génériques (ex: "Lymphos") devraient
    contenir à la fois des sous-populations T/NK et B.  Cette fonction utilise
    la distance cosinus pour classifier chaque nœud lympho générique comme
    T/NK-like ou B-like par rapport aux profils de référence extraits du mapping.

    Génère lympho_verification_{timestamp}.html avec une heatmap z-score MFI.

    Args:
        mapping_df: DataFrame du meilleur mapping population→nœud, avec colonnes
                    'node_id' et 'assigned_pop'.
        node_mfi_df: DataFrame MFI [n_nodes × n_markers] (centroïdes SOM).
        node_counts: Taille de chaque nœud (np.ndarray ou dict {node_id: n}).
        mc_per_node: Métacluster par nœud (np.ndarray ou dict {node_id: mc}).
        map_name: Nom de la méthode de mapping (pour le titre).
        output_dir: Répertoire de sauvegarde HTML (None = pas de sauvegarde).
        timestamp: Suffixe de nom de fichier.

    Returns:
        Figure Plotly (heatmap z-score) ou None si plotly absent / données insuffisantes.
    """
    if not _PLOTLY_AVAILABLE:
        return None

    from scipy.spatial.distance import cosine as cosine_dist

    # ── Normalisation de node_counts / mc_per_node en dict ───────────────────
    def _to_count_dict(source: Any, n_nodes: int) -> Dict[int, int]:
        if isinstance(source, dict):
            return {int(k): int(v) for k, v in source.items()}
        if source is not None:
            try:
                arr = np.asarray(source).ravel()
                return {i: int(arr[i]) for i in range(len(arr))}
            except Exception:
                pass
        return {i: 1 for i in range(n_nodes)}

    n_nodes = len(node_mfi_df)
    cnt_dict = _to_count_dict(node_counts, n_nodes)

    # ── Extraire les nœuds par catégorie depuis mapping_df ───────────────────
    if "node_id" not in mapping_df.columns or "assigned_pop" not in mapping_df.columns:
        _logger.warning(
            "plot_lympho_verification: colonnes node_id/assigned_pop manquantes."
        )
        return None

    def _nodes_for_labels(label_set: set) -> List[int]:
        mask = mapping_df["assigned_pop"].apply(
            lambda p: any(lbl.lower() in str(p).lower() for lbl in label_set)
        )
        return [int(nid) for nid in mapping_df.loc[mask, "node_id"].values]

    generic_nodes = _nodes_for_labels(_LYMPHO_GENERIC)
    tnk_ref_nodes = _nodes_for_labels(_LYTNK_LABELS)
    b_ref_nodes = _nodes_for_labels(_LYMPHOB_LABELS)

    # ── Cas : tout déjà sous-classifié ───────────────────────────────────────
    if not generic_nodes:
        _logger.info(
            "[lympho_verification] Tous les lymphocytes sont déjà sous-classifiés "
            "(T_NK / B) — aucune vérification nécessaire."
        )
        return None

    _logger.info(
        "[lympho_verification] %d nœud(s) lympho générique(s) à vérifier "
        "| %d réf T/NK | %d réf B",
        len(generic_nodes),
        len(tnk_ref_nodes),
        len(b_ref_nodes),
    )

    # ── Profils de référence pondérés ────────────────────────────────────────
    node_count_arr: Optional[np.ndarray] = None
    if node_counts is not None:
        try:
            node_count_arr = np.asarray(node_counts).ravel()
        except Exception:
            pass

    prof_tnk = get_mean_profile(tnk_ref_nodes, node_mfi_df, node_count_arr)
    prof_b = get_mean_profile(b_ref_nodes, node_mfi_df, node_count_arr)

    # ── Classification de chaque nœud générique ──────────────────────────────
    results: List[Dict] = []
    for nid in generic_nodes:
        if nid < 0 or nid >= n_nodes:
            continue
        profile_node = node_mfi_df.iloc[nid].values.astype(float)
        n_cells = cnt_dict.get(nid, 1)

        d_tnk: Optional[float] = None
        d_b: Optional[float] = None

        if prof_tnk is not None:
            ref_tnk = prof_tnk.values.astype(float)
            try:
                d_tnk = float(cosine_dist(profile_node, ref_tnk))
            except Exception:
                d_tnk = None

        if prof_b is not None:
            ref_b = prof_b.values.astype(float)
            try:
                d_b = float(cosine_dist(profile_node, ref_b))
            except Exception:
                d_b = None

        # Décision : argmin(d_T, d_B) avec seuil d'indétermination 0.05
        _INDETERMINATE_THRESHOLD = 0.05
        if d_tnk is None and d_b is None:
            decision = "Indéterminé"
        elif d_tnk is None:
            decision = "Lymphos B probable"
        elif d_b is None:
            decision = "Ly T_NK probable"
        else:
            diff = abs(d_tnk - d_b)
            if diff < _INDETERMINATE_THRESHOLD:
                decision = "Indéterminé"
            elif d_tnk < d_b:
                decision = "Ly T_NK probable"
            else:
                decision = "Lymphos B probable"

        results.append(
            {
                "node_id": nid,
                "n_cells": n_cells,
                "d_tnk": d_tnk,
                "d_b": d_b,
                "decision": decision,
            }
        )

    # ── Résumé et alertes cliniques ──────────────────────────────────────────
    n_tnk_like = sum(1 for r in results if r["decision"] == "Ly T_NK probable")
    n_b_like = sum(1 for r in results if r["decision"] == "Lymphos B probable")
    n_indet = sum(1 for r in results if r["decision"] == "Indéterminé")

    _logger.info(
        "[lympho_verification] Résultat — T/NK-like: %d | B-like: %d | Indéterminé: %d",
        n_tnk_like,
        n_b_like,
        n_indet,
    )

    # Alertes cliniques
    if n_tnk_like > 0 and n_b_like > 0:
        _logger.info(
            "[lympho_verification] ✅ ATTENDU : lymphocytes T et B co-localisés dans "
            "les nœuds génériques (SOM a correctement regroupé les deux lignées)."
        )
    elif n_b_like > 0 and n_tnk_like == 0:
        _logger.warning(
            "[lympho_verification] ⚠️  ALERTE : uniquement des nœuds B-like détectés "
            "parmi les lymphos génériques — vérifier la représentation CD3/CD7."
        )
    elif n_tnk_like > 0 and n_b_like == 0:
        _logger.warning(
            "[lympho_verification] ⚠️  ALERTE : uniquement des nœuds T/NK-like détectés "
            "parmi les lymphos génériques — vérifier CD19 et représentation B dans l'échantillon."
        )

    # ── Construction de la heatmap z-score ───────────────────────────────────
    marker_cols = list(node_mfi_df.columns)

    # Construire le DataFrame de toutes les lignes à afficher
    rows_data: List[np.ndarray] = []
    row_labels: List[str] = []

    # Lignes de référence (en premier)
    if prof_tnk is not None:
        rows_data.append(prof_tnk.values.astype(float))
        row_labels.append("◀ Ly T_NK (réf)")
    if prof_b is not None:
        rows_data.append(prof_b.values.astype(float))
        row_labels.append("◀ Lymphos B (réf)")

    # Lignes des nœuds génériques
    for r in sorted(results, key=lambda x: -x["n_cells"]):
        nid = r["node_id"]
        icon = (
            "🟣"
            if r["decision"] == "Ly T_NK probable"
            else ("🔵" if r["decision"] == "Lymphos B probable" else "⚪")
        )
        label = f"{icon} Node {nid} ({r['n_cells']} cells) → {r['decision']}"
        rows_data.append(node_mfi_df.iloc[nid].values.astype(float))
        row_labels.append(label)

    if not rows_data:
        return None

    mat = np.array(rows_data)  # shape (n_rows, n_markers)

    # Z-score par colonne sur l'ensemble des lignes (ref + génériques)
    col_mean = mat.mean(axis=0)
    col_std = mat.std(axis=0)
    col_std[col_std == 0] = 1.0
    Z = (mat - col_mean) / col_std

    # Annotations : valeurs z-score arrondies
    annotations = []
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            annotations.append(
                dict(
                    x=marker_cols[j],
                    y=row_labels[i],
                    text=f"{Z[i, j]:.1f}",
                    font=dict(size=9, color="black"),
                    showarrow=False,
                )
            )

    n_rows_fig = len(row_labels)
    height = max(400, n_rows_fig * 45 + 200)

    fig = go.Figure(
        data=go.Heatmap(
            z=Z,
            x=marker_cols,
            y=row_labels,
            colorscale="RdBu_r",
            zmid=0.0,
            colorbar=dict(title="z-score MFI"),
            hoverongaps=False,
        )
    )
    fig.update_layout(
        annotations=annotations,
        title=dict(
            text=f"Vérification Lymphocytes — {map_name} | {timestamp}",
            x=0.5,
        ),
        xaxis=dict(tickangle=-45),
        height=height,
        paper_bgcolor="#fafafa",
        plot_bgcolor="#fafafa",
        font=dict(color="#333333"),
        margin=dict(l=300, r=80, t=100, b=130),
    )

    if output_dir is not None:
        _save_html(fig, output_dir, f"lympho_verification_{timestamp}.html")

    return fig
