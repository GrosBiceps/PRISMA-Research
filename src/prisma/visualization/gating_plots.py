"""
gating_plots.py — Visualisations des étapes de pré-gating.

Génère les graphiques de QC du gating en utilisant les helpers bas-niveau
de plot_helpers.py. Un graphique par gate, sauvegardé en PNG.
Inclut la génération de diagrammes Sankey Plotly (global + par fichier).
Inclut plot_gmm_vs_kde_qc (QC débris Gate1 KDE+GMM) et
generate_interactive_gating_dashboard (dashboard Plotly complet).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    import plotly.graph_objects as go

    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

from prisma.core.gating import PreGating
from prisma.utils.logger import get_logger
from prisma.visualization.plot_helpers import (
    BG_COLOR,
    add_gate_rectangle,
    plot_density,
    plot_gating,
    save_figure,
)

_logger = get_logger("visualization.gating_plots")


def _find_idx(var_names: List[str], patterns: List[str]) -> Optional[int]:
    """Wrapper autour de PreGating.find_marker_index."""
    return PreGating.find_marker_index(var_names, patterns)


def plot_overview(
    X: np.ndarray,
    var_names: List[str],
    output_path: Path | str,
    n_sample: int = 60_000,
    seed: int = 42,
) -> bool:
    """
    Graphique 1: Vue d'ensemble FSC-A vs SSC-A (pré-gating).

    Args:
        X: Matrice brute (n_cells, n_markers).
        var_names: Noms des marqueurs.
        output_path: Chemin PNG de sortie.
        n_sample: Nombre de cellules à afficher.
        seed: Graine pour sous-échantillonnage.

    Returns:
        True si succès.
    """
    if not _MPL_AVAILABLE:
        _logger.warning("matplotlib manquant — plot_overview ignoré")
        return None

    fsc_a = _find_idx(var_names, ["FSC-A"])
    ssc_a = _find_idx(var_names, ["SSC-A", "SSC-H", "SSC"])

    if fsc_a is None or ssc_a is None:
        _logger.warning("FSC-A ou SSC-A non trouvé — plot_overview ignoré")
        return None

    rng = np.random.default_rng(seed)
    n = min(n_sample, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
    plot_density(
        ax,
        X[idx, fsc_a],
        X[idx, ssc_a],
        f"VUE D'ENSEMBLE\n{X.shape[0]:,} événements totaux",
        "FSC-A (Forward Scatter - Taille)",
        "SSC-A (Side Scatter - Granularité)",
    )
    save_figure(fig, output_path)
    _logger.info("Sauvegardé: %s", output_path)
    return fig


def plot_debris_gate(
    X: np.ndarray,
    var_names: List[str],
    mask: np.ndarray,
    output_path: Path | str,
    debris_min_pct: float = 2.0,
    debris_max_pct: float = 99.0,
    n_sample: int = 60_000,
    seed: int = 42,
) -> Optional[Any]:
    """
    Graphique 2: Gate débris (FSC-A vs SSC-A avec overlay).

    Args:
        X: Matrice brute (n_cells, n_markers).
        var_names: Noms des marqueurs.
        mask: Masque G1 — True = cellule viable.
        output_path: Chemin PNG de sortie.
        debris_min_pct: Percentile borne inférieure de la gate.
        debris_max_pct: Percentile borne supérieure de la gate.
        n_sample: Nombre de cellules à afficher.
        seed: Graine pour sous-échantillonnage.

    Returns:
        True si succès.
    """
    if not _MPL_AVAILABLE:
        return None

    fsc_a = _find_idx(var_names, ["FSC-A"])
    ssc_a = _find_idx(var_names, ["SSC-A", "SSC-H", "SSC"])

    if fsc_a is None or ssc_a is None:
        _logger.warning("FSC-A ou SSC-A non trouvé — plot_debris_gate ignoré")
        return None

    rng = np.random.default_rng(seed)
    n = min(n_sample, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
    plot_gating(
        ax,
        X[idx, fsc_a],
        X[idx, ssc_a],
        mask[idx],
        "GATE 1: Exclusion des Débris",
        "FSC-A (Taille)",
        "SSC-A (Granularité)",
        "Cellules viables",
        "Débris/Bruit",
    )

    # Rectangle de gate indicatif
    fsc_lo = np.nanpercentile(X[:, fsc_a], debris_min_pct)
    fsc_hi = np.nanpercentile(X[:, fsc_a], debris_max_pct)
    ssc_lo = np.nanpercentile(X[:, ssc_a], debris_min_pct)
    ssc_hi = np.nanpercentile(X[:, ssc_a], debris_max_pct)
    add_gate_rectangle(ax, fsc_lo, fsc_hi, ssc_lo, ssc_hi)

    save_figure(fig, output_path)
    _logger.info("Sauvegardé: %s", output_path)
    return fig


def plot_singlets_gate(
    X: np.ndarray,
    var_names: List[str],
    mask: np.ndarray,
    output_path: Path | str,
    n_sample: int = 60_000,
    seed: int = 42,
) -> Optional[Any]:
    """
    Graphique 3: Gate singlets (FSC-A vs FSC-H).

    Args:
        X: Matrice brute.
        var_names: Noms des marqueurs.
        mask: Masque G2 — True = singlet.
        output_path: Chemin PNG de sortie.
        n_sample: Nombre de cellules à afficher.
        seed: Graine.

    Returns:
        True si succès.
    """
    if not _MPL_AVAILABLE:
        return None

    fsc_a = _find_idx(var_names, ["FSC-A"])
    fsc_h = _find_idx(var_names, ["FSC-H"])

    if fsc_a is None or fsc_h is None:
        _logger.warning("FSC-A ou FSC-H non trouvé — plot_singlets_gate ignoré")
        return None

    rng = np.random.default_rng(seed)
    n = min(n_sample, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
    plot_gating(
        ax,
        X[idx, fsc_a],
        X[idx, fsc_h],
        mask[idx],
        "GATE 2: Exclusion des Doublets",
        "FSC-A (Area)",
        "FSC-H (Height)",
        "Singlets",
        "Doublets",
    )
    save_figure(fig, output_path)
    _logger.info("Sauvegardé: %s", output_path)
    return fig


def plot_cd45_gate(
    X: np.ndarray,
    var_names: List[str],
    mask: np.ndarray,
    output_path: Path | str,
    n_sample: int = 60_000,
    seed: int = 42,
) -> Optional[Any]:
    """
    Graphique 4: Gate CD45+ (CD45 vs SSC-A).

    Args:
        X: Matrice brute.
        var_names: Noms des marqueurs.
        mask: Masque G3 — True = CD45+.
        output_path: Chemin PNG de sortie.
        n_sample: Nombre de cellules à afficher.
        seed: Graine.

    Returns:
        True si succès.
    """
    if not _MPL_AVAILABLE:
        return None

    cd45_idx = _find_idx(var_names, ["CD45", "CD45-PECY5", "CD45-PC5"])
    ssc_a = _find_idx(var_names, ["SSC-A", "SSC-H", "SSC"])

    if cd45_idx is None or ssc_a is None:
        _logger.warning("CD45 ou SSC-A non trouvé — plot_cd45_gate ignoré")
        return None

    rng = np.random.default_rng(seed)
    n = min(n_sample, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
    plot_gating(
        ax,
        X[idx, cd45_idx],
        X[idx, ssc_a],
        mask[idx],
        "GATE 3: Cellules CD45+",
        "CD45",
        "SSC-A",
        "CD45+",
        "CD45-",
    )
    save_figure(fig, output_path)
    _logger.info("Sauvegardé: %s", output_path)
    return fig


def plot_cd34_gate(
    X: np.ndarray,
    var_names: List[str],
    mask: np.ndarray,
    output_path: Path | str,
    n_sample: int = 60_000,
    seed: int = 42,
) -> Optional[Any]:
    """
    Graphique 5: Gate CD34+ blastes (CD34 vs SSC-A).

    Args:
        X: Matrice brute.
        var_names: Noms des marqueurs.
        mask: Masque G4 — True = blast CD34+.
        output_path: Chemin PNG de sortie.
        n_sample: Nombre de cellules à afficher.
        seed: Graine.

    Returns:
        True si succès.
    """
    if not _MPL_AVAILABLE:
        return None

    cd34_idx = _find_idx(var_names, ["CD34", "CD34-PE", "CD34-APC"])
    ssc_a = _find_idx(var_names, ["SSC-A", "SSC-H", "SSC"])

    if cd34_idx is None or ssc_a is None:
        _logger.warning("CD34 ou SSC-A non trouvé — plot_cd34_gate ignoré")
        return None

    rng = np.random.default_rng(seed)
    n = min(n_sample, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
    plot_gating(
        ax,
        X[idx, cd34_idx],
        X[idx, ssc_a],
        mask[idx],
        "GATE 4: CD34+ Blastes (SSC bas)",
        "CD34",
        "SSC-A",
        "CD34+ blastes",
        "Autres",
    )
    save_figure(fig, output_path)
    _logger.info("Sauvegardé: %s", output_path)
    return fig


def generate_all_gating_plots(
    X_raw: np.ndarray,
    var_names: List[str],
    masks: Dict[str, np.ndarray],
    output_dir: Path | str,
    sample_name: str = "sample",
    n_sample: int = 60_000,
    seed: int = 42,
    conditions: Optional[np.ndarray] = None,
) -> Dict[str, bool]:
    """
    Génère tous les graphiques de gating pour un échantillon.

    Args:
        X_raw: Matrice brute (pré-transformation, pré-gating).
        var_names: Noms des marqueurs.
        conditions: Condition par cellule (ex. "Pathologique" / "Sain"), pré-gating.
        masks: Dict {gate_name: mask_array} — ex: {"G1": ..., "G2": ..., ...}.
        output_dir: Dossier de sortie.
        sample_name: Identifiant de l'échantillon pour les noms de fichiers.
        n_sample: Cellules max à afficher.
        seed: Graine.

    Returns:
        Dict {plot_name: figure_object_or_None}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = sample_name.replace(".fcs", "").replace(".FCS", "")

    results: Dict[str, Any] = {}

    results["01_overview"] = plot_overview(
        X_raw,
        var_names,
        output_dir / f"{name}_01_overview.png",
        n_sample=n_sample,
        seed=seed,
    )

    # Accept both short keys ("G1", "debris") and full keys ("G1_debris")
    def _get_mask(keys: List[str]) -> Optional[np.ndarray]:
        for k in keys:
            if k in masks:
                return masks[k]
        return None

    mask_g1 = _get_mask(["G1", "debris", "G1_debris"])
    if mask_g1 is not None:
        results["02_debris"] = plot_debris_gate(
            X_raw,
            var_names,
            mask_g1,
            output_dir / f"{name}_02_debris.png",
            n_sample=n_sample,
            seed=seed,
        )

    mask_g2 = _get_mask(["G2", "singlets", "G2_singlets"])
    if mask_g2 is not None:
        results["03_singlets"] = plot_singlets_gate(
            X_raw,
            var_names,
            mask_g2,
            output_dir / f"{name}_03_singlets.png",
            n_sample=n_sample,
            seed=seed,
        )

    mask_g3 = _get_mask(["G3", "cd45", "G3_cd45"])
    if mask_g3 is not None:
        results["04_cd45"] = plot_cd45_gate(
            X_raw,
            var_names,
            mask_g3,
            output_dir / f"{name}_04_cd45.png",
            n_sample=n_sample,
            seed=seed,
        )

    mask_g4 = _get_mask(["G4", "cd34", "G4_cd34"])
    if mask_g4 is not None:
        results["05_cd34"] = plot_cd34_gate(
            X_raw,
            var_names,
            mask_g4,
            output_dir / f"{name}_05_cd34.png",
            n_sample=n_sample,
            seed=seed,
        )

    # ── KDE QC plots ─────────────────────────────────────────────────────────
    # Gate 1 KDE : densité FSC-A avec seuil GMM + vallée KDE
    fsc_idx = _find_idx(var_names, ["FSC-A", "FSC-H"])
    if mask_g1 is not None and fsc_idx is not None:
        try:
            fig_kde_g1, _, _, _ = plot_gmm_vs_kde_qc(
                fsc_a_data=X_raw[:, fsc_idx],
                mask_debris=mask_g1,
                random_seed=seed,
                output_path=output_dir / f"{name}_06_kde_debris.png",
            )
            results["06_kde_debris"] = fig_kde_g1
        except Exception as _e:
            _logger.warning("KDE Gate 1 échoué (non bloquant): %s", _e)

    # Gate 3 KDE CD45 : généré séparément dans preprocessing_service.py
    # (après la transformation logicle/arcsinh) pour afficher en espace transformé.
    # Ne PAS générer ici (données encore en linéaire brut).

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Sankey Diagram — Flux des événements à travers le gating hiérarchique
# ─────────────────────────────────────────────────────────────────────────────


def _pct_of(value: int, parent: int) -> str:
    """Pourcentage relatif à la gate parente."""
    return f"{value / max(parent, 1) * 100:.1f}%"


def generate_sankey_diagram(
    gate_counts: Dict[str, int],
    output_path: Path | str,
    *,
    filter_blasts: bool = False,
    title: str = "Gating Hierarchy — Flux des Événements",
) -> Optional[Any]:
    """
    Génère un diagramme Sankey Plotly du flux de gating hiérarchique.

    Représente le parcours : Total → G1 (débris) → G2 (singlets) → G3 (CD45)
    → optionnellement G4 (CD34) → Population finale.

    Lien vert = conservé, gris/orange = exclu.

    Args:
        gate_counts: Dict avec les clés:
            - n_total: événements totaux
            - n_g1_pass: après gate débris
            - n_g2_pass: après gate singlets
            - n_g3_pass: après gate CD45
            - n_final: population finale (après G4 si filter_blasts)
        output_path: Chemin du fichier HTML de sortie.
        filter_blasts: Si True, ajoute la gate G4 (CD34+).
        title: Titre du diagramme.

    Returns:
        True si succès.
    """
    if not _PLOTLY_AVAILABLE:
        _logger.warning("plotly manquant — Sankey ignoré")
        return None

    n_total = gate_counts["n_total"]
    n_g1_pass = gate_counts["n_g1_pass"]
    n_g1_fail = n_total - n_g1_pass
    n_g2_pass = gate_counts["n_g2_pass"]
    n_g2_fail = n_g1_pass - n_g2_pass
    n_g3_pass = gate_counts["n_g3_pass"]
    n_g3_fail = n_g2_pass - n_g3_pass
    n_final = gate_counts["n_final"]
    n_g4_fail = n_g3_pass - n_final

    labels = [
        f"Événements<br>totaux<br>{n_total:,}",
        f"Gate 1<br>Viables<br>{n_g1_pass:,}<br>({_pct_of(n_g1_pass, n_total)} of total)",
        f"Débris<br>exclus<br>{n_g1_fail:,}<br>({_pct_of(n_g1_fail, n_total)})",
        f"Gate 2<br>Singlets<br>{n_g2_pass:,}<br>({_pct_of(n_g2_pass, n_g1_pass)} of G1)",
        f"Doublets<br>exclus<br>{n_g2_fail:,}<br>({_pct_of(n_g2_fail, n_g1_pass)})",
        f"Gate 3<br>CD45+<br>{n_g3_pass:,}<br>({_pct_of(n_g3_pass, n_g2_pass)} of G2)",
        f"CD45-<br>exclus<br>{n_g3_fail:,}<br>({_pct_of(n_g3_fail, n_g2_pass)})",
    ]

    src = [0, 0, 1, 1, 3, 3]
    tgt = [1, 2, 3, 4, 5, 6]
    vals = [n_g1_pass, n_g1_fail, n_g2_pass, n_g2_fail, n_g3_pass, n_g3_fail]
    link_colors = [
        "rgba(49,163,84,0.4)",
        "rgba(99,99,99,0.3)",
        "rgba(49,163,84,0.4)",
        "rgba(230,85,13,0.3)",
        "rgba(49,163,84,0.4)",
        "rgba(253,141,60,0.3)",
    ]

    node_colors = [
        "#4a90d9",
        "#31a354",
        "#636363",
        "#31a354",
        "#e6550d",
        "#31a354",
        "#fd8d3c",
    ]

    if filter_blasts:
        labels.append(
            f"Gate 4<br>CD34+<br>{n_final:,}<br>({_pct_of(n_final, n_g3_pass)} of G3)"
        )
        labels.append(
            f"Non-blastes<br>exclus<br>{n_g4_fail:,}<br>({_pct_of(n_g4_fail, n_g3_pass)})"
        )
        final_label = f"Population<br>finale<br>{n_final:,}<br>({_pct_of(n_final, n_total)} of total)"
        labels.append(final_label)
        src += [5, 5, 7]
        tgt += [7, 8, 9]
        vals += [n_final, max(n_g4_fail, 1), n_final]
        link_colors += [
            "rgba(49,163,84,0.4)",
            "rgba(253,174,107,0.3)",
            "rgba(49,163,84,0.6)",
        ]
        node_colors += ["#31a354", "#fdae6b", "#2ca02c"]
    else:
        final_label = f"Population<br>finale<br>{n_g3_pass:,}<br>({_pct_of(n_g3_pass, n_total)} of total)"
        labels.append(final_label)
        src.append(5)
        tgt.append(7)
        vals.append(n_g3_pass)
        link_colors.append("rgba(49,163,84,0.6)")
        node_colors.append("#2ca02c")

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color="#333", width=1),
                label=labels,
                color=node_colors[: len(labels)],
            ),
            link=dict(source=src, target=tgt, value=vals, color=link_colors),
        )
    )
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=18)),
        font=dict(size=13, color="#222"),
        paper_bgcolor="#fafafa",
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    _logger.info("Sankey global sauvegardé: %s", output_path.name)
    return fig


def generate_per_file_sankey(
    per_file_counts: Dict[str, Dict[str, int]],
    output_dir: Path | str,
    *,
    filter_blasts: bool = False,
    max_files: int = 6,
    timestamp: str = "",
) -> Dict[str, bool]:
    """
    Génère un mini-Sankey par fichier FCS (max 6 fichiers).

    Args:
        per_file_counts: Dict {nom_fichier: {n_total, n_g1_pass, n_g2_pass,
                         n_g3_pass, n_final}}.
        output_dir: Dossier de sortie.
        filter_blasts: Si True, ajoute la gate G4.
        max_files: Nombre max de mini-Sankey à générer.
        timestamp: Suffixe optionnel.

    Returns:
        Dict {nom_fichier: succès}.
    """
    if not _PLOTLY_AVAILABLE:
        _logger.warning("plotly manquant — mini-Sankey ignorés")
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, bool] = {}

    files_to_show = list(per_file_counts.keys())[:max_files]

    for f_name in files_to_show:
        counts = per_file_counts[f_name]
        n_total = counts["n_total"]
        if n_total < 10:
            results[f_name] = False
            continue

        n_g1 = counts["n_g1_pass"]
        n_g1_fail = n_total - n_g1
        n_g2 = counts["n_g2_pass"]
        n_g2_fail = n_g1 - n_g2
        n_g3 = counts["n_g3_pass"]
        n_g3_fail = n_g2 - n_g3
        n_final = counts["n_final"]
        n_g4_fail = n_g3 - n_final

        short_name = str(f_name) if len(str(f_name)) <= 30 else str(f_name)[:27] + "..."

        labels = [
            f"Total<br>{n_total:,}",
            f"G1<br>{n_g1:,}<br>({_pct_of(n_g1, n_total)})",
            f"Débris<br>{n_g1_fail:,}",
            f"G2<br>{n_g2:,}<br>({_pct_of(n_g2, n_g1)})",
            f"Doubl.<br>{n_g2_fail:,}",
            f"G3<br>{n_g3:,}<br>({_pct_of(n_g3, n_g2)})",
            f"CD45-<br>{n_g3_fail:,}",
        ]
        src = [0, 0, 1, 1, 3, 3]
        tgt = [1, 2, 3, 4, 5, 6]
        vals = [
            n_g1,
            max(n_g1_fail, 1),
            n_g2,
            max(n_g2_fail, 1),
            n_g3,
            max(n_g3_fail, 1),
        ]
        link_col = [
            "rgba(49,163,84,0.4)",
            "rgba(99,99,99,0.3)",
            "rgba(49,163,84,0.4)",
            "rgba(230,85,13,0.3)",
            "rgba(49,163,84,0.4)",
            "rgba(253,141,60,0.3)",
        ]
        node_col = [
            "#4a90d9",
            "#31a354",
            "#636363",
            "#31a354",
            "#e6550d",
            "#31a354",
            "#fd8d3c",
        ]

        if filter_blasts:
            labels += [
                f"G4<br>{n_final:,}<br>({_pct_of(n_final, n_g3)})",
                f"Excl.<br>{n_g4_fail:,}",
                f"Final<br>{n_final:,}",
            ]
            src += [5, 5, 7]
            tgt += [7, 8, 9]
            vals += [n_final, max(n_g4_fail, 1), n_final]
            link_col += [
                "rgba(49,163,84,0.4)",
                "rgba(253,174,107,0.3)",
                "rgba(49,163,84,0.6)",
            ]
            node_col += ["#31a354", "#fdae6b", "#2ca02c"]
        else:
            labels.append(f"Final<br>{n_g3:,}")
            src.append(5)
            tgt.append(7)
            vals.append(n_g3)
            link_col.append("rgba(49,163,84,0.6)")
            node_col.append("#2ca02c")

        fig = go.Figure(
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="#333", width=0.5),
                    label=labels,
                    color=node_col[: len(labels)],
                ),
                link=dict(source=src, target=tgt, value=vals, color=link_col),
            )
        )
        fig.update_layout(
            title=dict(text=f"<b>Gating — {short_name}</b>", font=dict(size=14)),
            font=dict(size=11, color="#222"),
            paper_bgcolor="#fafafa",
            height=300,
            margin=dict(l=10, r=10, t=45, b=10),
        )

        safe = f_name.replace(" ", "_").replace(".fcs", "").replace(".FCS", "")
        suffix = f"_{timestamp}" if timestamp else ""
        out_path = output_dir / f"sankey_{safe}{suffix}.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        results[f_name] = True
        _logger.info("Mini-Sankey sauvegardé: %s", out_path.name)

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  QC Gate 1 — GMM vs KDE (Exclusion Débris)
# ─────────────────────────────────────────────────────────────────────────────


def plot_gmm_vs_kde_qc(
    fsc_a_data: np.ndarray,
    mask_debris: np.ndarray,
    n_subsample: int = 10_000,
    n_grid: int = 1_000,
    valley_range: Tuple[float, float] = (-100_000.0, 600_000.0),
    random_seed: int = 42,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Any, Any, float, Optional[float]]:
    """
    Trace la densité KDE de FSC-A avec le seuil débris GMM et la vallée KDE.

    Interface simplifiée : mask_debris (bool) indique Gate1. Si tout True
    (tri cellulaire/propre), un GMM interne est utilisé pour simuler le seuil.

    Args:
        fsc_a_data: Vecteur FSC-A brut (n_events,).
        mask_debris: Masque booléen issu du Gate G1 (True = cellule conservée).
        n_subsample: Nombre de points pour estimer la KDE (≤10 000 recommandé).
        n_grid: Résolution de la grille d'évaluation KDE.
        valley_range: Intervalle de recherche de la vallée KDE [x_min, x_max].
        random_seed: Reproductibilité.
        ax: Axes matplotlib existants (optionnel).
        title: Titre du graphique (auto-généré si None).
        output_path: Chemin de sauvegarde PNG (optionnel).

    Returns:
        Tuple (fig, ax, gmm_threshold, kde_valley).
    """
    if not _MPL_AVAILABLE:
        _logger.warning("matplotlib requis pour plot_gmm_vs_kde_qc")
        return None, None, 0.0, None

    try:
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks
        from scipy.stats import gaussian_kde
    except ImportError:
        _logger.warning(
            "scipy requis pour plot_gmm_vs_kde_qc (gaussian_kde, find_peaks, gaussian_filter1d)"
        )
        return None, None, 0.0, None

    fsc_a_data = np.asarray(fsc_a_data, dtype=np.float64).ravel()
    mask_debris = np.asarray(mask_debris, dtype=bool).ravel()

    if len(fsc_a_data) != len(mask_debris):
        raise ValueError(
            f"fsc_a_data ({len(fsc_a_data)}) et mask_debris ({len(mask_debris)}) "
            "doivent avoir la même longueur."
        )

    n_total = len(fsc_a_data)
    n_exclu = int((~mask_debris).sum())
    tri_cellulaire = n_exclu == 0

    # ── 1. Seuil GMM ──────────────────────────────────────────────────────────
    if not tri_cellulaire:
        gmm_threshold = float(np.max(fsc_a_data[~mask_debris]))
        debris_label = "Débris exclus G1"
    else:
        _logger.debug("mask_debris intégral — simulation GMM interne (tri cellulaire)")
        try:
            from sklearn.mixture import GaussianMixture as _GMM
        except ImportError:
            _logger.warning("sklearn requis pour GMM fallback dans plot_gmm_vs_kde_qc")
            gmm_threshold = float(
                np.percentile(fsc_a_data[np.isfinite(fsc_a_data)], 20)
            )
            debris_label = "Seuil estimé (sklearn absent)"
            tri_cellulaire = False
        else:
            _rng = np.random.default_rng(random_seed)
            _valid = np.isfinite(fsc_a_data)
            _n_fit = min(50_000, int(_valid.sum()))
            _idx = _rng.choice(np.where(_valid)[0], size=_n_fit, replace=False)
            _gmm = _GMM(n_components=2, random_state=random_seed, n_init=5)
            _gmm.fit(fsc_a_data[_idx].reshape(-1, 1))
            _means = _gmm.means_.flatten()
            _all_labels = np.full(n_total, -1, dtype=int)
            _all_labels[_valid] = _gmm.predict(fsc_a_data[_valid].reshape(-1, 1))
            _low_cluster = int(np.argmin(_means))
            _low_vals = fsc_a_data[_all_labels == _low_cluster]
            gmm_threshold = (
                float(np.percentile(_low_vals, 95))
                if len(_low_vals) > 0
                else float(np.percentile(fsc_a_data[_valid], 20))
            )
            debris_label = f"Cluster bas FSC-A (simul., μ={_means[_low_cluster]:.0f})"
            mask_debris = _all_labels != _low_cluster
            if title is None:
                title = "QC Gate 1 — Simulation FSC-A bas [Tri cellulaire / propre]"

    if title is None:
        title = "QC Gate 1 — Exclusion débris (GMM vs KDE)"

    # ── 2. Sous-échantillonnage pour KDE ──────────────────────────────────────
    rng = np.random.default_rng(random_seed)
    if n_total > n_subsample:
        fsc_sub = fsc_a_data[rng.choice(n_total, size=n_subsample, replace=False)]
    else:
        fsc_sub = fsc_a_data.copy()

    # ── 3. KDE ────────────────────────────────────────────────────────────────
    finite_sub = fsc_sub[np.isfinite(fsc_sub)]
    if len(finite_sub) < 5:
        _logger.warning("plot_gmm_vs_kde_qc: trop peu de valeurs finies pour KDE.")
        return None, None, gmm_threshold, None

    data_min = float(np.nanmin(fsc_sub))
    data_max = float(np.nanpercentile(fsc_sub, 99.9))
    x_min = min(
        valley_range[0], data_min - max(150_000.0, (data_max - data_min) * 0.15)
    )
    n_grid = int(np.clip(n_grid, 300, 700))
    use_fast_density = len(finite_sub) > 5_000

    if use_fast_density:
        # Histogramme lissé: beaucoup plus rapide que gaussian_kde sur gros volumes.
        hist_counts, hist_edges = np.histogram(
            finite_sub,
            bins=max(120, n_grid // 2),
            range=(x_min, data_max),
            density=True,
        )
        x_grid = 0.5 * (hist_edges[:-1] + hist_edges[1:])
        density = gaussian_filter1d(hist_counts.astype(np.float64), sigma=2.0)
    else:
        kde = gaussian_kde(finite_sub, bw_method="scott")
        x_grid = np.linspace(x_min, data_max, n_grid)
        density = kde(x_grid)

    # ── 4. Détection de la vallée ─────────────────────────────────────────────
    vm = (x_grid >= valley_range[0]) & (x_grid <= valley_range[1])
    kde_valley: Optional[float] = None
    if vm.any():
        peaks_idx, _ = find_peaks(
            -density[vm], prominence=(density[vm].max() - density[vm].min()) * 0.005
        )
        if len(peaks_idx) > 0:
            kde_valley = float(x_grid[vm][peaks_idx[0]])
        else:
            fb = (x_grid >= max(x_min, gmm_threshold - 100_000)) & (
                x_grid <= min(data_max, gmm_threshold + 150_000)
            )
            if fb.any():
                kde_valley = float(x_grid[fb][np.argmin(density[fb])])

    # ── 5. Visualisation ──────────────────────────────────────────────────────
    BG = "#1e1e2f"
    PANEL = "#16213e"
    TXT = "#e2e8f0"
    GRID = "#2d2d4e"
    RED_F = "#ef4444"
    GRN_F = "#22c55e"
    RED_L = "#ff6b6b"
    BLU_L = "#60a5fa"

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    else:
        fig = ax.get_figure()
        fig.patch.set_facecolor(BG)

    ax.set_facecolor(PANEL)
    ax.fill_between(
        x_grid,
        density,
        where=(x_grid <= gmm_threshold),
        color=RED_F,
        alpha=0.35,
        label=debris_label,
        rasterized=True,
    )
    ax.fill_between(
        x_grid,
        density,
        where=(x_grid > gmm_threshold),
        color=GRN_F,
        alpha=0.30,
        label="Cellules conservées",
        rasterized=True,
    )
    density_label = "Densité FSC-A (hist lissé)" if use_fast_density else "KDE FSC-A"
    ax.plot(
        x_grid,
        density,
        color=TXT,
        linewidth=1.8,
        label=density_label,
        rasterized=True,
    )
    ax.axvline(
        gmm_threshold,
        color=RED_L,
        linestyle="--",
        linewidth=2.5,
        label=f"Seuil G1 : {gmm_threshold:,.0f}",
    )

    if kde_valley is not None:
        ax.axvline(
            kde_valley,
            color=BLU_L,
            linestyle="--",
            linewidth=2.0,
            label=f"Vallée KDE : {kde_valley:,.0f}",
        )
        delta = abs(gmm_threshold - kde_valley)
        ax.annotate(
            f"Δ = {delta:,.0f}",
            xy=((gmm_threshold + kde_valley) / 2, 0.88),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            color=TXT,
            fontsize=9,
        )

    n_debris = int((~mask_debris).sum())
    pct_deb = 100.0 * n_debris / n_total
    stats_text = (
        f"Total   : {n_total:,}\n"
        f"Exclus  : {n_debris:,} ({pct_deb:.1f}%)\n"
        f"Conserv.: {n_total - n_debris:,} ({100 - pct_deb:.1f}%)\n"
        f"KDE sur : {len(fsc_sub):,} pts"
    )
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=TXT,
        family="monospace",
        bbox=dict(facecolor=BG, edgecolor=GRID, alpha=0.85, pad=6),
    )

    ax.set_xlabel("FSC-A (unités linéaires)", color=TXT, fontsize=11)
    ax.set_ylabel("Densité KDE", color=TXT, fontsize=11)
    ax.set_title(title, color=TXT, fontsize=13, pad=12)
    ax.tick_params(colors=TXT, labelsize=9)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", " "))
    )
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xlim(x_min, data_max)
    ax.set_ylim(bottom=0, top=np.max(density) * 1.05)
    ax.legend(
        loc="upper left", fontsize=9, facecolor=BG, edgecolor=GRID, labelcolor=TXT
    )
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=100, bbox_inches="tight")
        _logger.info("QC GMM-KDE sauvegardé: %s", output_path)

    return fig, ax, gmm_threshold, kde_valley


# ─────────────────────────────────────────────────────────────────────────────
#  QC Gate 3 — KDE pied du pic (Auto-gating CD45)
# ─────────────────────────────────────────────────────────────────────────────


def plot_cd45_kde_qc(
    cd45_data: np.ndarray,
    mask_cd45: np.ndarray,
    n_subsample: int = 15_000,
    random_seed: int = 42,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    conditions: Optional[np.ndarray] = None,
    condition_patho: str = "Pathologique",
    kde_seuil_relatif: float = 0.05,
    kde_finesse: float = 0.6,
    kde_sigma_smooth: int = 10,
    kde_n_grid: int = 1000,
    # kept for backwards compat — ignored
    n_grid: int = 1_000,
    mrd_result: Optional[Any] = None,
) -> Tuple[Any, Any, float, Optional[float]]:
    """
    QC Gate 3 — Densité KDE CD45 avec seuil pied du pic.

    Visualise la densité KDE 1D sur CD45 (espace logicle/arcsinh) et le seuil
    calculé par la méthode pied du pic (même algorithme que auto_gate_cd45).
    Le seuil est recalculé localement pour la visualisation afin de montrer
    exactement la courbe KDE, le seuil relatif (5% du max), et la ligne de
    coupure.

    Args:
        cd45_data: Vecteur CD45 transformé (logicle/arcsinh).
        mask_cd45: Masque booléen Gate G3 (True = CD45+).
        n_subsample: Points pour estimer la KDE visuelle.
        random_seed: Reproductibilité.
        ax: Axes matplotlib existants (optionnel).
        title: Titre du graphique (auto-généré si None).
        output_path: Chemin de sauvegarde PNG (optionnel).
        conditions: Conditions par cellule (pour encadré patho).
        condition_patho: Label de la condition pathologique.
        kde_seuil_relatif: Fraction du max densité pour le pied du pic.
        kde_finesse: Facteur bandwidth Silverman.
        kde_sigma_smooth: Sigma du lissage gaussien.
        kde_n_grid: Résolution de la grille KDE.
        mrd_result: MRDResult optionnel — si fourni, affiche le ratio MRD
            (blastes / CD45+ patho) calculé par le pipeline dans l'encadré.

    Returns:
        Tuple (fig, ax, threshold, None).
    """
    if not _MPL_AVAILABLE:
        _logger.warning("matplotlib requis pour plot_cd45_kde_qc")
        return None, None, 0.0, None

    try:
        from scipy.ndimage import gaussian_filter1d
        from scipy.stats import gaussian_kde
    except ImportError:
        _logger.warning("scipy requis pour plot_cd45_kde_qc")
        return None, None, 0.0, None

    cd45_data = np.asarray(cd45_data, dtype=np.float64).ravel()
    mask_cd45 = np.asarray(mask_cd45, dtype=bool).ravel()

    if len(cd45_data) != len(mask_cd45):
        raise ValueError(
            f"cd45_data ({len(cd45_data)}) et mask_cd45 ({len(mask_cd45)}) "
            "doivent avoir la même longueur."
        )

    valid = np.isfinite(cd45_data)
    n_total = len(cd45_data)
    cd45_valid = cd45_data[valid]

    if len(cd45_valid) < 5:
        _logger.warning("plot_cd45_kde_qc: trop peu de valeurs finies.")
        return None, None, 0.0, None

    # ── 1. Sous-échantillonnage pour KDE visuelle ─────────────────────────────
    rng = np.random.default_rng(random_seed)
    n_sub = min(n_subsample, len(cd45_valid))
    sub = cd45_valid[rng.choice(len(cd45_valid), size=n_sub, replace=False)]

    # ── 2. Densité visuelle: KDE exact (petits jeux) ou histogramme lissé ─────
    kde_n_grid = int(np.clip(kde_n_grid, 250, 700))
    use_fast_density = len(sub) > 5_000
    if use_fast_density:
        hist_counts, hist_edges = np.histogram(
            sub,
            bins=max(100, kde_n_grid // 2),
            range=(float(cd45_valid.min()), float(cd45_valid.max())),
            density=True,
        )
        x_grid = 0.5 * (hist_edges[:-1] + hist_edges[1:])
        density = gaussian_filter1d(
            hist_counts.astype(np.float64), sigma=max(1.0, kde_sigma_smooth / 4)
        )
    else:
        n = len(sub)
        std = np.std(sub)
        iqr = np.percentile(sub, 75) - np.percentile(sub, 25)
        bw = 0.9 * min(std, iqr / 1.34) * n ** (-1 / 5) * kde_finesse
        if std < 1e-12:
            bw = 0.1
        kde = gaussian_kde(sub, bw_method=bw / (std if std > 1e-12 else 1.0))
        x_grid = np.linspace(cd45_valid.min(), cd45_valid.max(), kde_n_grid)
        density = gaussian_filter1d(kde(x_grid), sigma=kde_sigma_smooth)

    # ── 3. Seuil pied du pic ─────────────────────────────────────────────────
    seuil_abs = density.max() * kde_seuil_relatif
    idx_max = int(np.argmax(density))
    threshold = x_grid[0]
    method_tag = "pied non trouvé (borne gauche)"
    for i in range(idx_max, 0, -1):
        if density[i] < seuil_abs:
            threshold = float(x_grid[i])
            method_tag = "pied du pic"
            break

    # Si le masque venant du pipeline est trivial, on recalcule un masque
    # d'affichage depuis le seuil local pour une QC plus lisible.
    mask_cd45_vis = mask_cd45.copy()
    if bool(mask_cd45_vis.all()) or (not bool(mask_cd45_vis.any())):
        mask_cd45_vis = np.zeros_like(mask_cd45_vis, dtype=bool)
        mask_cd45_vis[valid] = cd45_data[valid] >= threshold

    if title is None:
        title = "QC Gate 3 — Auto-gating CD45 (KDE pied du pic)"

    # ── 4. Stats du masque réel (pipeline) ────────────────────────────────────
    n_cd45_neg = int((~mask_cd45_vis).sum())
    n_cd45_pos = n_total - n_cd45_neg
    pct_neg = 100.0 * n_cd45_neg / n_total
    pct_pos = 100.0 - pct_neg

    # ── 5. Visualisation ──────────────────────────────────────────────────────
    BG = "#1e1e2f"
    PANEL = "#16213e"
    TXT = "#e2e8f0"
    GRID = "#2d2d4e"
    RED_F = "#ef4444"
    GRN_F = "#22c55e"
    AMBER = "#f9e2af"

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=BG)
    else:
        fig = ax.get_figure()
        fig.patch.set_facecolor(BG)

    ax.set_facecolor(PANEL)

    # Histogramme en arrière-plan
    ax.hist(
        cd45_valid,
        bins=200,
        density=True,
        alpha=0.20,
        color="#89b4fa",
        edgecolor="none",
        label="_nolegend_",
        rasterized=True,
    )

    # Zones colorées CD45− / CD45+
    ax.fill_between(
        x_grid,
        density,
        where=(x_grid < threshold),
        color=RED_F,
        alpha=0.30,
        label=f"CD45− ({n_cd45_neg:,} | {pct_neg:.1f}%)",
        rasterized=True,
    )
    ax.fill_between(
        x_grid,
        density,
        where=(x_grid >= threshold),
        color=GRN_F,
        alpha=0.30,
        label=f"CD45+ ({n_cd45_pos:,} | {pct_pos:.1f}%)",
        rasterized=True,
    )

    # Courbe KDE
    density_label = "Densité CD45 (hist lissé)" if use_fast_density else "KDE CD45"
    ax.plot(
        x_grid,
        density,
        color=TXT,
        linewidth=2.0,
        label=density_label,
        zorder=3,
        rasterized=True,
    )

    # Ligne de seuil
    ax.axvline(
        threshold,
        color=AMBER,
        linestyle="--",
        linewidth=2.5,
        zorder=4,
        label=f"Seuil KDE ({method_tag}) : {threshold:.3f}",
    )

    # Ligne horizontale seuil_abs (5% du max)
    ax.axhline(
        seuil_abs,
        color=AMBER,
        linestyle=":",
        linewidth=1.0,
        alpha=0.5,
        label=f"Seuil relatif ({100 * kde_seuil_relatif:.0f}% du max)",
    )

    # Annotation fléchée sur le seuil
    y_ann = density.max() * 0.50
    ax.annotate(
        f"  {threshold:.3f}",
        xy=(threshold, y_ann),
        xytext=(threshold + (x_grid[-1] - x_grid[0]) * 0.05, y_ann),
        color=AMBER,
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5),
        zorder=5,
    )

    # Point du pic max
    ax.plot(
        x_grid[idx_max],
        density[idx_max],
        "o",
        color=TXT,
        markersize=6,
        zorder=5,
        label=f"Pic max : {x_grid[idx_max]:.3f}",
    )

    # ── Encadré stats ─────────────────────────────────────────────────────────
    stats_text = (
        f"Total     : {n_total:,}\n"
        f"CD45−     : {n_cd45_neg:,} ({pct_neg:.1f}%)\n"
        f"CD45+     : {n_cd45_pos:,} ({pct_pos:.1f}%)\n"
        f"Finesse   : {kde_finesse}\n"
        f"Seuil rel.: {kde_seuil_relatif}"
    )
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=TXT,
        family="monospace",
        bbox=dict(facecolor=BG, edgecolor=GRID, alpha=0.85, pad=6),
    )

    # ── Encadré pathologique ──────────────────────────────────────────────────
    if conditions is not None:
        cond_arr = np.asarray(conditions).ravel()
        if len(cond_arr) == len(mask_cd45_vis):
            patho_mask = cond_arr == condition_patho
            n_patho_total = int(patho_mask.sum())
            if n_patho_total > 0:
                n_patho_pos = int((patho_mask & mask_cd45_vis).sum())
                n_patho_neg = int((patho_mask & ~mask_cd45_vis).sum())
                p_pos = 100.0 * n_patho_pos / n_patho_total
                p_neg = 100.0 * n_patho_neg / n_patho_total
                patho_text = (
                    "─ Moelle pathologique ─\n"
                    f"CD45+  : {n_patho_pos:,} ({p_pos:.1f}%)\n"
                    f"CD45−  : {n_patho_neg:,} ({p_neg:.1f}%)\n"
                    f"Total  : {n_patho_total:,}"
                )

                # ── Ratio MRD corrigé (blastes / CD45+ patho) ─────────────
                if mrd_result is not None:
                    _denom_mode = getattr(mrd_result, "mrd_denominator_mode", "none")
                    _denom = getattr(mrd_result, "mrd_denominator", n_patho_total)
                    _denom_lbl = (
                        "CD45+ patho" if _denom_mode in ("cd45", "cd45_dim")
                        else "patho total"
                    )
                    _mrd_jf = getattr(mrd_result, "mrd_pct_jf", None)
                    _mrd_flo = getattr(mrd_result, "mrd_pct_flo", None)
                    _mrd_eln = getattr(mrd_result, "mrd_pct_eln", None)
                    _mrd_lines = [f"\n─── MRD (÷ {_denom_lbl}) ───"]
                    _mrd_lines.append(f"  Dénominateur: {_denom:,}")
                    if _mrd_jf is not None:
                        _mrd_lines.append(f"  JF : {_mrd_jf:.4f}%")
                    if _mrd_flo is not None:
                        _mrd_lines.append(f"  Flo: {_mrd_flo:.4f}%")
                    if _mrd_eln is not None:
                        _mrd_lines.append(f"  ELN: {_mrd_eln:.4f}%")
                    patho_text += "\n".join(_mrd_lines)

                ax.text(
                    0.98,
                    0.58,
                    patho_text,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8.5,
                    color=TXT,
                    family="monospace",
                    bbox=dict(
                        facecolor=BG,
                        edgecolor="#f38ba8",
                        linewidth=1.4,
                        alpha=0.90,
                        pad=6,
                    ),
                )

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlabel("CD45 (espace logicle / arcsinh)", color=TXT, fontsize=11)
    ax.set_ylabel("Densité KDE", color=TXT, fontsize=11)
    ax.set_title(title, color=TXT, fontsize=13, pad=12)
    ax.tick_params(colors=TXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.5, linestyle="--", alpha=0.4)
    x_span = max(float(x_grid[-1] - x_grid[0]), 1e-9)
    x_margin = 0.02 * x_span
    ax.set_xlim(float(x_grid[0] - x_margin), float(x_grid[-1] + x_margin))
    ax.set_ylim(bottom=0, top=density.max() * 1.08)
    ax.legend(
        loc="upper left",
        fontsize=8.5,
        facecolor=BG,
        edgecolor=GRID,
        labelcolor=TXT,
        framealpha=0.9,
    )

    if threshold <= x_grid[1]:
        ax.text(
            0.02,
            0.06,
            "Seuil au bord gauche: population CD45- quasi absente",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color=AMBER,
            bbox=dict(facecolor=BG, edgecolor=GRID, alpha=0.7, pad=4),
        )

    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            str(output_path),
            dpi=100,
            bbox_inches="tight",
            facecolor=BG,
        )
        _logger.info("QC CD45-KDE sauvegardé: %s", output_path)

    return fig, ax, threshold, None


# ─────────────────────────────────────────────────────────────────────────────
#  Pré-screening CD34 — KDE minimum local (2 pics)
# ─────────────────────────────────────────────────────────────────────────────


def plot_cd34_kde_prescreening(
    prescreening_result: Any,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> Optional[Any]:
    """
    QC Pré-screening — Densité KDE CD34 avec seuil minimum local (2 pics).

    Visualise la courbe KDE 1D sur CD34 (espace log), le seuil de séparation
    CD34−/CD34+ (minimum local entre les deux pics), et les zones colorées.
    Affiche aussi la comparaison GMM vs KDE en encadré.

    Args:
        prescreening_result: PrescreeningResult depuis compute_cd34_prescreening().
        output_path: Chemin de sauvegarde PNG (optionnel).
        title: Titre du graphique (auto-généré si None).

    Returns:
        Figure matplotlib ou None si données insuffisantes.
    """
    if not _MPL_AVAILABLE:
        _logger.warning("matplotlib requis pour plot_cd34_kde_prescreening")
        return None

    ps = prescreening_result
    x_grid = getattr(ps, "kde_x_grid", None)
    density = getattr(ps, "kde_density", None)

    if x_grid is None or density is None or len(x_grid) == 0:
        _logger.warning("plot_cd34_kde_prescreening: pas de données KDE disponibles")
        return None

    x_grid = np.asarray(x_grid, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)

    threshold_log = float(getattr(ps, "kde_threshold_log", 0.0))
    n_cd34_pos = int(getattr(ps, "kde_n_cd34_pos", 0))
    n_cd45dim = int(getattr(ps, "n_cd45dim", 0))
    n_cd34_neg = n_cd45dim - n_cd34_pos
    ratio_kde = float(getattr(ps, "kde_ratio_pct", 0.0))
    ratio_gmm = float(getattr(ps, "gmm_ratio_pct", 0.0))
    ratio_ref = float(getattr(ps, "ratio_pct", 0.0))
    method_used = str(getattr(ps, "method_used", "KDE"))
    alert_level = str(getattr(ps, "alert_level", "none"))
    n_gmm_pos = int(getattr(ps, "gmm_n_cd34_pos", 0))

    if title is None:
        title = "Pré-screening — KDE CD34 (minimum local entre 2 pics)"

    BG = "#1e1e2f"
    PANEL = "#16213e"
    TXT = "#e2e8f0"
    GRID = "#2d2d4e"
    RED_F = "#ef4444"
    GRN_F = "#22c55e"
    AMBER = "#f9e2af"
    ORANGE = "#fb923c"

    alert_color = {"high": RED_F, "moderate": ORANGE, "none": GRN_F}.get(alert_level, GRN_F)

    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=BG)
    ax.set_facecolor(PANEL)

    # Zones CD34− / CD34+
    ax.fill_between(
        x_grid, density,
        where=(x_grid < threshold_log),
        color=RED_F, alpha=0.28,
        label=f"CD34− ({n_cd34_neg:,} | {100 - ratio_kde:.1f}%)",
        rasterized=True,
    )
    ax.fill_between(
        x_grid, density,
        where=(x_grid >= threshold_log),
        color=GRN_F, alpha=0.28,
        label=f"CD34+ ({n_cd34_pos:,} | {ratio_kde:.1f}%)",
        rasterized=True,
    )

    # Courbe KDE
    ax.plot(x_grid, density, color=TXT, linewidth=2.2,
            label="KDE CD34 (Silverman + lissage gaussien)", zorder=3, rasterized=True)

    # Ligne de seuil (minimum entre les 2 pics)
    ax.axvline(
        threshold_log, color=AMBER, linestyle="--", linewidth=2.5, zorder=4,
        label=f"Seuil KDE (vallée entre 2 pics) : {threshold_log:.3f}",
    )

    # Annotation seuil
    y_ann = density.max() * 0.55
    ax.annotate(
        f"  {threshold_log:.3f}",
        xy=(threshold_log, y_ann),
        xytext=(threshold_log + (x_grid[-1] - x_grid[0]) * 0.05, y_ann),
        color=AMBER, fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5), zorder=5,
    )

    # Point du minimum local
    idx_thr = int(np.argmin(np.abs(x_grid - threshold_log)))
    ax.plot(
        x_grid[idx_thr], density[idx_thr], "v",
        color=AMBER, markersize=8, zorder=5,
        label="Minimum local (vallée)",
    )

    # Encadré stats
    stats_text = (
        f"CD45dim total : {n_cd45dim:,}\n"
        f"CD34+ KDE     : {n_cd34_pos:,} ({ratio_kde:.1f}%)\n"
        f"CD34+ GMM     : {n_gmm_pos:,} ({ratio_gmm:.1f}%)\n"
        f"Ref ({method_used})  : {ratio_ref:.1f}%"
    )
    ax.text(
        0.98, 0.97, stats_text,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8.5, color=TXT, family="monospace",
        bbox=dict(facecolor=BG, edgecolor=GRID, alpha=0.85, pad=6),
    )

    # Encadré alerte
    if alert_level != "none":
        interp = str(getattr(ps, "interpretation_warning", ""))
        ax.text(
            0.02, 0.97, interp,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color=alert_color, style="italic",
            bbox=dict(facecolor=BG, edgecolor=alert_color, alpha=0.85, pad=5),
        )

    ax.set_xlabel("CD34 (espace logicle / arcsinh)", color=TXT, fontsize=11)
    ax.set_ylabel("Densité KDE", color=TXT, fontsize=11)
    ax.set_title(title, color=TXT, fontsize=13, pad=12)
    ax.tick_params(colors=TXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.5, linestyle="--", alpha=0.4)
    x_span = max(float(x_grid[-1] - x_grid[0]), 1e-9)
    ax.set_xlim(float(x_grid[0]) - 0.02 * x_span, float(x_grid[-1]) + 0.02 * x_span)
    ax.set_ylim(bottom=0, top=density.max() * 1.10)
    ax.legend(
        loc="upper left" if alert_level == "none" else "center left",
        fontsize=8.5, facecolor=BG, edgecolor=GRID, labelcolor=TXT, framealpha=0.9,
    )

    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=100, bbox_inches="tight", facecolor=BG)
        _logger.info("Pré-screening CD34 KDE sauvegardé: %s", output_path)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Dashboard interactif Plotly — Gating complet
# ─────────────────────────────────────────────────────────────────────────────


def generate_interactive_gating_dashboard(
    X_raw: np.ndarray,
    var_names: List[str],
    masks: Dict[str, np.ndarray],
    conditions: np.ndarray,
    output_dir: Path,
    *,
    ransac_scatter_data: Optional[Dict[str, Any]] = None,
    singlets_summary_per_file: Optional[List[Dict[str, Any]]] = None,
    filter_blasts: bool = False,
    mode_blastes_vs_normal: bool = False,
    timestamp: str = "",
    seed: int = 42,
) -> Dict[str, bool]:
    """
    Génère un dashboard Plotly interactif complet du pre-gating.

    Sections générées :
      [1]  Sankey hiérarchique global
      [1b] Mini-Sankey par fichier (si ransac_scatter_data fourni)
      [1c] Scatter FSC-A vs FSC-H par fichier avec droite RANSAC
      [1d] Tableau % singlets par fichier (R², méthode)
      [2]  Density Scatter Plots interactifs (G1→G4) en Plotly ScatterGL
      [3]  Histogrammes 1D avec seuils GMM annotés
      [4]  Comparaison Patho vs Sain (si mode_blastes_vs_normal)
      [5]  Overview final coloré par étape d'exclusion + tables résumé

    Args:
        X_raw: Matrice brute pré-gating (n_cells, n_markers).
        var_names: Noms des marqueurs.
        masks: Dict {gate_name: mask_bool} — ex {"G1": ..., "G2": ..., ...}.
        conditions: Vecteur de conditions par cellule.
        output_dir: Dossier de sortie des HTML.
        ransac_scatter_data: Dict {filename: {fsc_h, fsc_a, slope, intercept, r2, method}}.
        singlets_summary_per_file: Liste de dicts {file, pct_singlets, r2, method}.
        filter_blasts: True si la gate G4 (CD34+) est active.
        mode_blastes_vs_normal: True pour activer la comparaison Patho/Sain.
        timestamp: Suffixe de fichier.
        seed: Graine aléatoire pour sous-échantillonnage.

    Returns:
        Dict {section_name: succès}.
    """
    if not _PLOTLY_AVAILABLE:
        _logger.warning("plotly manquant — dashboard interactif désactivé")
        return {}

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _get_mask(keys: List[str]) -> Optional[np.ndarray]:
        for k in keys:
            if k in masks:
                return masks[k]
        return None

    from prisma.core.gating import PreGating

    fsc_a_idx = PreGating.find_marker_index(var_names, ["FSC-A"])
    fsc_h_idx = PreGating.find_marker_index(var_names, ["FSC-H"])
    ssc_a_idx = PreGating.find_marker_index(var_names, ["SSC-A", "SSC-H", "SSC"])
    cd45_idx = PreGating.find_marker_index(
        var_names, ["CD45", "CD45-PECY5", "CD45-PC5"]
    )
    cd34_idx = PreGating.find_marker_index(var_names, ["CD34", "CD34-PE", "CD34-APC"])

    mask_g1 = _get_mask(["G1", "debris", "G1_debris"])
    mask_g2 = _get_mask(["G2", "singlets", "G2_singlets"])
    mask_g3 = _get_mask(["G3", "cd45", "G3_cd45"])
    mask_g4 = _get_mask(["G4", "cd34", "G4_cd34"])
    mask_final = _get_mask(["final", "mask_final"])

    if mask_g1 is None:
        mask_g1 = np.ones(len(conditions), dtype=bool)
    if mask_g2 is None:
        mask_g2 = mask_g1.copy()
    if mask_g3 is None:
        mask_g3 = mask_g2.copy()
    if mask_final is None:
        mask_final = (
            mask_g3.copy()
            if mask_g4 is None
            else (mask_g4 if mask_g4 is not None else mask_g3.copy())
        )

    n_before = len(conditions)
    n_final = int(mask_final.sum())

    # Sous-échantillonnage pour Plotly
    rng = np.random.default_rng(seed)
    n_pts = min(40_000, n_before)
    idx = rng.choice(n_before, n_pts, replace=False)

    _m_g1 = mask_g1[idx]
    _m_g12 = (mask_g1 & mask_g2)[idx]
    _m_g123 = (mask_g1 & mask_g2 & mask_g3)[idx]
    _m_final = mask_final[idx]
    _cond_sub = conditions[idx]

    def _gate_label_fn(i: int) -> str:
        """Retourne l'étiquette de gate pour l'événement i du sous-échantillon."""
        if not _m_g1[i]:
            return "Débris (exclu G1)"
        if not _m_g12[i]:
            return "Doublet (exclu G2)"
        if not _m_g123[i]:
            return "CD45- Patho (exclu G3)"
        if filter_blasts and not _m_final[i]:
            return "Non-blaste (exclu G4)"
        if str(_cond_sub[i]) == "Pathologique":
            return "CD45+ Patho conservés ✓"
        elif str(_cond_sub[i]) == "Sain":
            return "Conservés sains NBM ✓"
        return "Conservé ✓"

    _labels = np.array([_gate_label_fn(i) for i in range(n_pts)])

    _color_map = {
        "Débris (exclu G1)": "#636363",
        "Doublet (exclu G2)": "#e6550d",
        "CD45- Patho (exclu G3)": "#fd8d3c",
        "Non-blaste (exclu G4)": "#fdae6b",
        "CD45+ Patho conservés ✓": "#d62728",
        "Conservés sains NBM ✓": "#2ca02c",
        "Conservé ✓": "#31a354",
    }

    def _pct_of_local(value: int, parent: int) -> str:
        return f"{value / max(parent, 1) * 100:.1f}%"

    results: Dict[str, bool] = {}
    ts = f"_{timestamp}" if timestamp else ""

    # ── [1c] RANSAC scatter per file ──────────────────────────────────────────
    if ransac_scatter_data:
        try:
            _n_scatter = len(ransac_scatter_data)
            _n_cols_sc = min(3, _n_scatter)
            _n_rows_sc = int(np.ceil(_n_scatter / _n_cols_sc))
            fig_ransac = make_subplots(
                rows=_n_rows_sc,
                cols=_n_cols_sc,
                subplot_titles=[k[:30] for k in ransac_scatter_data.keys()],
                horizontal_spacing=0.06,
                vertical_spacing=0.10,
            )
            for _si, (_sf_name, _sf_data) in enumerate(ransac_scatter_data.items()):
                _row = _si // _n_cols_sc + 1
                _col = _si % _n_cols_sc + 1
                _r2_disp = (
                    f"R²={_sf_data['r2']:.3f}"
                    if _sf_data.get("r2") is not None
                    else "R²=N/A"
                )
                _method = _sf_data.get("method", "unknown")
                _color_pts = "#d62728" if _method == "ratio_fallback" else "#2ca02c"
                fig_ransac.add_trace(
                    go.Scattergl(
                        x=_sf_data["fsc_h"],
                        y=_sf_data["fsc_a"],
                        mode="markers",
                        marker=dict(size=2, color=_color_pts, opacity=0.3),
                        name=f"{_sf_name[:20]} ({_method})",
                        showlegend=False,
                        hovertemplate=f"FSC-H: %{{x:.0f}}<br>FSC-A: %{{y:.0f}}<br>{_r2_disp}<extra></extra>",
                    ),
                    row=_row,
                    col=_col,
                )
                _x_line = sorted(_sf_data["fsc_h"])
                _y_line = [
                    _sf_data["slope"] * x + _sf_data["intercept"] for x in _x_line
                ]
                fig_ransac.add_trace(
                    go.Scatter(
                        x=_x_line,
                        y=_y_line,
                        mode="lines",
                        line=dict(color="#ff7f0e", width=2, dash="dash"),
                        showlegend=False,
                    ),
                    row=_row,
                    col=_col,
                )
                fig_ransac.update_xaxes(title_text="FSC-H", row=_row, col=_col)
                fig_ransac.update_yaxes(title_text="FSC-A", row=_row, col=_col)
            fig_ransac.update_layout(
                title="<b>QC RANSAC — FSC-A vs FSC-H par fichier</b>",
                height=350 * _n_rows_sc,
                paper_bgcolor="#fafafa",
                plot_bgcolor="#f5f5f5",
            )
            out = output_dir / f"fig_ransac_scatter{ts}.html"
            fig_ransac.write_html(str(out), include_plotlyjs="cdn")
            results["1c_ransac"] = True
            _logger.info("RANSAC scatter: %s", out.name)
        except Exception as exc:
            _logger.warning("Échec RANSAC scatter: %s", exc)
            results["1c_ransac"] = False

    # ── [1d] Tableau singlets ─────────────────────────────────────────────────
    if singlets_summary_per_file:
        try:
            import pandas as pd

            _df_s = pd.DataFrame(singlets_summary_per_file)
            _cell_colors = []
            for col_name in _df_s.columns:
                col_colors = []
                for ri, row in _df_s.iterrows():
                    if (
                        col_name == "r2"
                        and row.get("r2") is not None
                        and row["r2"] < 0.85
                    ):
                        col_colors.append("#ffe0e0")
                    elif col_name == "method" and row.get("method") == "ratio_fallback":
                        col_colors.append("#fff3cd")
                    else:
                        col_colors.append("#f9f9f9" if int(ri) % 2 == 0 else "#fff")
                _cell_colors.append(col_colors)

            fig_sing = go.Figure(
                go.Table(
                    header=dict(
                        values=[f"<b>{c.upper()}</b>" for c in _df_s.columns],
                        fill_color="#4a90d9",
                        font=dict(color="white", size=12),
                        align="center",
                        height=35,
                    ),
                    cells=dict(
                        values=[_df_s[c] for c in _df_s.columns],
                        fill_color=_cell_colors,
                        font=dict(size=11),
                        align="center",
                        height=28,
                    ),
                )
            )
            fig_sing.update_layout(
                title="<b>QC Singlets — % par fichier (R² RANSAC)</b>",
                height=50 + 30 * (len(_df_s) + 1),
                width=900,
                margin=dict(l=20, r=20, t=50, b=10),
            )
            out = output_dir / f"fig_singlets_table{ts}.html"
            fig_sing.write_html(str(out), include_plotlyjs="cdn")
            results["1d_singlets_table"] = True
            _logger.info("Table singlets: %s", out.name)
        except Exception as exc:
            _logger.warning("Échec table singlets: %s", exc)
            results["1d_singlets_table"] = False

    # ── [2] Density Scatter Plots ─────────────────────────────────────────────
    try:
        _gate_plots: List[Dict[str, Any]] = []

        if fsc_a_idx is not None and ssc_a_idx is not None:
            _gate_plots.append(
                {
                    "title": "Gate 1 — Débris (SSC-A vs FSC-A)",
                    "x": X_raw[idx, fsc_a_idx],
                    "y": X_raw[idx, ssc_a_idx],
                    "mask": _m_g1,
                    "xlabel": "FSC-A (Taille)",
                    "ylabel": "SSC-A (Granu.)",
                    "label_in": "Cellules viables",
                    "label_out": "Débris",
                }
            )
        if fsc_a_idx is not None and fsc_h_idx is not None:
            _g1_ok = _m_g1
            _gate_plots.append(
                {
                    "title": "Gate 2 — Doublets (FSC-H vs FSC-A)",
                    "x": X_raw[idx, fsc_a_idx][_g1_ok],
                    "y": X_raw[idx, fsc_h_idx][_g1_ok],
                    "mask": (mask_g1 & mask_g2)[idx][_g1_ok],
                    "xlabel": "FSC-A (Area)",
                    "ylabel": "FSC-H (Height)",
                    "label_in": "Singlets",
                    "label_out": "Doublets",
                }
            )
        if cd45_idx is not None and ssc_a_idx is not None:
            _g12_ok = _m_g12
            if mode_blastes_vs_normal:
                _patho_g12 = _g12_ok & (_cond_sub == "Pathologique")
                _gate_plots.append(
                    {
                        "title": "Gate 3 — CD45+ PATHO seul",
                        "x": X_raw[idx, cd45_idx][_patho_g12],
                        "y": X_raw[idx, ssc_a_idx][_patho_g12],
                        "mask": (mask_g1 & mask_g2 & mask_g3)[idx][_patho_g12],
                        "xlabel": "CD45",
                        "ylabel": "SSC-A",
                        "label_in": "CD45+ Patho",
                        "label_out": "CD45− Patho",
                    }
                )
            else:
                _gate_plots.append(
                    {
                        "title": "Gate 3 — CD45+ Leucocytes",
                        "x": X_raw[idx, cd45_idx][_g12_ok],
                        "y": X_raw[idx, ssc_a_idx][_g12_ok],
                        "mask": (mask_g1 & mask_g2 & mask_g3)[idx][_g12_ok],
                        "xlabel": "CD45",
                        "ylabel": "SSC-A",
                        "label_in": "Leucocytes CD45+",
                        "label_out": "CD45−",
                    }
                )
        if filter_blasts and cd34_idx is not None and ssc_a_idx is not None:
            _g123_ok = _m_g123
            _gate_plots.append(
                {
                    "title": "Gate 4 — CD34+ Blastes",
                    "x": X_raw[idx, cd34_idx][_g123_ok],
                    "y": X_raw[idx, ssc_a_idx][_g123_ok],
                    "mask": mask_final[idx][_g123_ok],
                    "xlabel": "CD34",
                    "ylabel": "SSC-A",
                    "label_in": "Blastes CD34+",
                    "label_out": "Autres leucocytes",
                }
            )

        if _gate_plots:
            fig_gates = make_subplots(
                rows=1,
                cols=len(_gate_plots),
                subplot_titles=[g["title"] for g in _gate_plots],
                horizontal_spacing=0.06,
            )
            for col_i, gp in enumerate(_gate_plots, 1):
                _x, _y, _mk = gp["x"], gp["y"], gp["mask"]
                _valid = np.isfinite(_x) & np.isfinite(_y)
                _x, _y, _mk = _x[_valid], _y[_valid], _mk[_valid]
                fig_gates.add_trace(
                    go.Scattergl(
                        x=_x[~_mk],
                        y=_y[~_mk],
                        mode="markers",
                        marker=dict(size=2, color="#d62728", opacity=0.25),
                        name=gp["label_out"],
                        legendgroup=f"g{col_i}_out",
                        showlegend=(col_i == 1),
                        hovertemplate=f"{gp['xlabel']}: %{{x:.0f}}<br>{gp['ylabel']}: %{{y:.0f}}<br>{gp['label_out']}<extra></extra>",
                    ),
                    row=1,
                    col=col_i,
                )
                fig_gates.add_trace(
                    go.Scattergl(
                        x=_x[_mk],
                        y=_y[_mk],
                        mode="markers",
                        marker=dict(size=2, color="#2ca02c", opacity=0.4),
                        name=gp["label_in"],
                        legendgroup=f"g{col_i}_in",
                        showlegend=(col_i == 1),
                        hovertemplate=f"{gp['xlabel']}: %{{x:.0f}}<br>{gp['ylabel']}: %{{y:.0f}}<br>{gp['label_in']}<extra></extra>",
                    ),
                    row=1,
                    col=col_i,
                )
                fig_gates.update_xaxes(title_text=gp["xlabel"], row=1, col=col_i)
                fig_gates.update_yaxes(title_text=gp["ylabel"], row=1, col=col_i)
            fig_gates.update_layout(
                title="<b>Gating Séquentiel — Density Scatter (Plotly interactif)</b>",
                height=500,
                width=min(500 * len(_gate_plots), 2000),
                paper_bgcolor="#fafafa",
                plot_bgcolor="#f0f0f0",
                font=dict(size=11),
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5
                ),
                margin=dict(t=80, b=100),
            )
            out = output_dir / f"fig_gates{ts}.html"
            fig_gates.write_html(str(out), include_plotlyjs="cdn")
            results["2_density_scatter"] = True
            _logger.info("Density scatter: %s", out.name)
    except Exception as exc:
        _logger.warning("Échec density scatter: %s", exc)
        results["2_density_scatter"] = False

    # ── [3] Histogrammes 1D ───────────────────────────────────────────────────
    try:
        _hist_data: List[Tuple[str, np.ndarray, Optional[str]]] = []
        if fsc_a_idx is not None:
            _hist_data.append(("FSC-A", X_raw[:, fsc_a_idx], None))
        if cd45_idx is not None:
            _hist_data.append(("CD45", X_raw[:, cd45_idx], "cd45"))
        if filter_blasts and cd34_idx is not None:
            _hist_data.append(("CD34", X_raw[:, cd34_idx], "cd34"))

        if _hist_data:
            fig_hist = make_subplots(
                rows=1,
                cols=len(_hist_data),
                subplot_titles=[h[0] for h in _hist_data],
                horizontal_spacing=0.08,
            )
            for hi, (name, vals, marker_type) in enumerate(_hist_data, 1):
                _v = vals[np.isfinite(vals)]
                fig_hist.add_trace(
                    go.Histogram(
                        x=_v,
                        nbinsx=200,
                        name=f"{name} (tous)",
                        marker_color="rgba(100,100,100,0.4)",
                        showlegend=(hi == 1),
                        legendgroup="all",
                    ),
                    row=1,
                    col=hi,
                )
                _v_kept = vals[mask_final & np.isfinite(vals)]
                fig_hist.add_trace(
                    go.Histogram(
                        x=_v_kept,
                        nbinsx=200,
                        name=f"{name} (conservés)",
                        marker_color="rgba(44,160,44,0.6)",
                        showlegend=(hi == 1),
                        legendgroup="kept",
                    ),
                    row=1,
                    col=hi,
                )
                fig_hist.update_xaxes(title_text=name, row=1, col=hi)
                fig_hist.update_yaxes(title_text="N événements", row=1, col=hi)
            fig_hist.update_layout(
                title="<b>Distributions 1D — Avant / Après Gating</b>",
                barmode="overlay",
                height=400,
                paper_bgcolor="#fafafa",
                plot_bgcolor="#f5f5f5",
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5
                ),
                margin=dict(t=70, b=90),
            )
            out = output_dir / f"fig_hist{ts}.html"
            fig_hist.write_html(str(out), include_plotlyjs="cdn")
            results["3_histograms"] = True
            _logger.info("Histogrammes: %s", out.name)
    except Exception as exc:
        _logger.warning("Échec histogrammes: %s", exc)
        results["3_histograms"] = False

    # ── [4] Comparaison Patho vs Sain ─────────────────────────────────────────
    if mode_blastes_vs_normal and cd45_idx is not None and ssc_a_idx is not None:
        try:
            fig_comp = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=[
                    "Pathologique (CD45 strict)",
                    "Sain / NBM (pas de gate CD45)",
                ],
                horizontal_spacing=0.08,
            )
            for ci, (cond_label, color_kept, color_all) in enumerate(
                [
                    ("Pathologique", "#d62728", "#ffcccc"),
                    ("Sain", "#2ca02c", "#ccffcc"),
                ],
                1,
            ):
                _sel = conditions == cond_label
                _sel_final = _sel & mask_final
                _xc = X_raw[:, cd45_idx]
                _yc = X_raw[:, ssc_a_idx]
                _sub_idx = rng.choice(
                    min(len(_xc), 20_000), min(20_000, _sel.sum()), replace=False
                )
                _sel_idx = np.where(_sel)[0]
                if len(_sel_idx) > 20_000:
                    _sel_idx = _sel_idx[
                        rng.choice(len(_sel_idx), 20_000, replace=False)
                    ]
                fig_comp.add_trace(
                    go.Scattergl(
                        x=_xc[_sel_idx],
                        y=_yc[_sel_idx],
                        mode="markers",
                        marker=dict(size=2, color=color_all, opacity=0.15),
                        name=f"{cond_label} (tous)",
                        showlegend=True,
                    ),
                    row=1,
                    col=ci,
                )
                _final_idx = np.where(_sel_final)[0]
                if len(_final_idx) > 20_000:
                    _final_idx = _final_idx[
                        rng.choice(len(_final_idx), 20_000, replace=False)
                    ]
                fig_comp.add_trace(
                    go.Scattergl(
                        x=_xc[_final_idx],
                        y=_yc[_final_idx],
                        mode="markers",
                        marker=dict(size=2.5, color=color_kept, opacity=0.5),
                        name=f"{cond_label} (conservés)",
                        showlegend=True,
                    ),
                    row=1,
                    col=ci,
                )
                fig_comp.update_xaxes(title_text="CD45", row=1, col=ci)
                fig_comp.update_yaxes(title_text="SSC-A", row=1, col=ci)
            fig_comp.update_layout(
                title="<b>Gating Asymétrique — Patho vs Sain</b>",
                height=500,
                width=1100,
                paper_bgcolor="#fafafa",
                plot_bgcolor="#f0f0f0",
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5
                ),
                margin=dict(t=70, b=90),
            )
            out = output_dir / f"fig_comp{ts}.html"
            fig_comp.write_html(str(out), include_plotlyjs="cdn")
            results["4_comparison"] = True
            _logger.info("Comparaison Patho/Sain: %s", out.name)
        except Exception as exc:
            _logger.warning("Échec comparaison Patho/Sain: %s", exc)
            results["4_comparison"] = False

    # ── [5] Overview final coloré par étape ───────────────────────────────────
    if fsc_a_idx is not None and ssc_a_idx is not None:
        try:
            _xo = X_raw[idx, fsc_a_idx]
            _yo = X_raw[idx, ssc_a_idx]
            fig_ov = go.Figure()
            _order = [
                "Débris (exclu G1)",
                "Doublet (exclu G2)",
                "CD45- Patho (exclu G3)",
                "Non-blaste (exclu G4)",
                "CD45+ Patho conservés ✓",
                "Conservés sains NBM ✓",
                "Conservé ✓",
            ]
            for cat in _order:
                _sel_cat = _labels == cat
                if _sel_cat.sum() == 0:
                    continue
                _is_kept = cat.endswith("✓")
                fig_ov.add_trace(
                    go.Scattergl(
                        x=_xo[_sel_cat],
                        y=_yo[_sel_cat],
                        mode="markers",
                        marker=dict(
                            size=2.5,
                            color=_color_map.get(cat, "#999"),
                            opacity=0.55 if _is_kept else 0.25,
                        ),
                        name=f"{cat} ({_sel_cat.sum():,})",
                        hovertemplate=f"FSC-A: %{{x:.0f}}<br>SSC-A: %{{y:.0f}}<br>{cat}<extra></extra>",
                    )
                )
            fig_ov.update_layout(
                title="<b>Overview — Événements colorés par étape d'exclusion</b>",
                xaxis_title="FSC-A (Taille)",
                yaxis_title="SSC-A (Granularité)",
                height=600,
                width=900,
                paper_bgcolor="#fafafa",
                plot_bgcolor="#f0f0f0",
                legend=dict(
                    title="Catégorie",
                    font=dict(size=12),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#ccc",
                    borderwidth=1,
                ),
                margin=dict(t=70, b=50),
            )
            out = output_dir / f"fig_overview{ts}.html"
            fig_ov.write_html(str(out), include_plotlyjs="cdn")
            results["5_overview"] = True
            _logger.info("Overview: %s", out.name)
        except Exception as exc:
            _logger.warning("Échec overview: %s", exc)
            results["5_overview"] = False

    # ── [5b] Table résumé ─────────────────────────────────────────────────────
    try:
        import pandas as pd

        _stages = [
            "Initial",
            "Gate 1 (Débris)",
            "Gate 2 (Doublets)",
            "Gate 3 (CD45+ Patho)" if mode_blastes_vs_normal else "Gate 3 (CD45+)",
        ]
        _events = [
            n_before,
            int(mask_g1.sum()),
            int((mask_g1 & mask_g2).sum()),
            int((mask_g1 & mask_g2 & mask_g3).sum()),
        ]
        if filter_blasts:
            _stages.append("Gate 4 (CD34+)")
            _events.append(n_final)
        _stages.append("Population finale")
        _events.append(n_final)
        _retention = [round(e / max(n_before, 1) * 100, 1) for e in _events]

        _sum_df = pd.DataFrame(
            {"Étape": _stages, "Événements": _events, "Rétention (%)": _retention}
        )
        fig_tbl = go.Figure(
            go.Table(
                header=dict(
                    values=[f"<b>{c}</b>" for c in _sum_df.columns],
                    fill_color="#4a90d9",
                    font=dict(color="white", size=13),
                    align="center",
                    height=35,
                ),
                cells=dict(
                    values=[_sum_df[c] for c in _sum_df.columns],
                    fill_color=[["#f9f9f9", "#fff"] * (len(_sum_df) // 2 + 1)] * 3,
                    font=dict(size=12),
                    align="center",
                    height=30,
                ),
            )
        )
        fig_tbl.update_layout(
            title="<b>Résumé Pre-Gating — Statistiques par étape</b>",
            height=50 + 35 * (len(_sum_df) + 1),
            width=700,
            margin=dict(l=20, r=20, t=50, b=10),
        )
        out = output_dir / f"fig_table{ts}.html"
        fig_tbl.write_html(str(out), include_plotlyjs="cdn")
        results["5b_summary_table"] = True
        _logger.info("Table résumé: %s", out.name)
    except Exception as exc:
        _logger.warning("Échec table résumé: %s", exc)
        results["5b_summary_table"] = False

    _logger.info("Dashboard interactif généré (%d sections)", len(results))
    return results
