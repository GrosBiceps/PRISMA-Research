"""
plot_helpers.py — Helpers de visualisation cytométrique bas-niveau.

Fournit les fonctions de rendu de base pour tous les graphiques du pipeline:
  - format_axis: formatage des axes (K/M)
  - plot_density: scatter 2D avec densité (style FlowJo)
  - plot_gating: scatter 2D avec overlay gating (vert/rouge)
  - apply_dark_style: applique le thème sombre cohérent sur un axe matplotlib
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import logging as _logging
    import matplotlib

    matplotlib.use("Agg")  # Backend non-interactif par défaut (compatible serveur)

    # Court-circuite la recherche dynamique de polices : force DejaVu Sans
    # (toujours disponible dans matplotlib) pour éviter les 30 000 logs findfont.
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # Silence le logger DEBUG du font_manager dans chaque worker multiprocessing
    _logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)
    _logging.getLogger("matplotlib").setLevel(_logging.WARNING)

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter
    from matplotlib.patches import Rectangle
    from matplotlib.patches import Patch

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constantes de style (thème sombre cohérent avec le monolith)
# ---------------------------------------------------------------------------
BG_COLOR = "#1e1e2e"
SPINE_COLOR = "#45475a"
TEXT_COLOR = "white"
LEGEND_BG = "#313244"

DENSITY_CMAP_COLORS = [
    "#0d0d0d",
    "#1a1a2e",
    "#0077b6",
    "#00b4d8",
    "#90e0ef",
    "#f9e2af",
    "#ffffff",
]
COLOR_KEPT = "#a6e3a1"  # Vert pastel
COLOR_EXCLUDED = "#f38ba8"  # Rouge pastel

HEXBIN_CMAP_EXCLUDED = LinearSegmentedColormap.from_list(
    "hex_excluded", ["#2a111b", "#7b1e3b", COLOR_EXCLUDED]
)
HEXBIN_CMAP_KEPT = LinearSegmentedColormap.from_list(
    "hex_kept", ["#0f2118", "#2f7a4f", COLOR_KEPT]
)


def _require_matplotlib() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib requis pour la visualisation: pip install matplotlib"
        )


def format_axis(value: float, pos: int) -> str:
    """
    Formateur d'axes intelligent (K pour milliers, M pour millions).

    Utilisé avec matplotlib.ticker.FuncFormatter.

    Args:
        value: Valeur de l'axe.
        pos: Position (ignoré, requis par l'interface FuncFormatter).

    Returns:
        Chaîne formatée ("100K", "1.2M", "500").
    """
    if abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.0f}K"
    return f"{value:.0f}"


def apply_dark_style(ax: "plt.Axes") -> None:
    """
    Applique le thème sombre sur un axe matplotlib.

    Args:
        ax: Axe matplotlib à styliser.
    """
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)

    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(1.5)

    if ax.get_xlabel():
        ax.xaxis.label.set_color(TEXT_COLOR)
    if ax.get_ylabel():
        ax.yaxis.label.set_color(TEXT_COLOR)
    if ax.get_title():
        ax.title.set_color(TEXT_COLOR)

    ax.xaxis.set_major_formatter(FuncFormatter(format_axis))
    ax.yaxis.set_major_formatter(FuncFormatter(format_axis))


def plot_density(
    ax: "plt.Axes",
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    n_bins: int = 120,
) -> Optional[object]:
    """
    Scatter plot avec densité 2D (style FlowJo).

    Utilise un histogramme 2D avec colormap logarithmique pour rendre
    les populations rares visibles (contrairement à un scatter simple).

    Args:
        ax: Axe matplotlib cible.
        x: Valeurs axe X (n_cells,).
        y: Valeurs axe Y (n_cells,).
        title: Titre du graphique.
        xlabel: Label axe X.
        ylabel: Label axe Y.
        n_bins: Résolution de l'histogramme (défaut 120).

    Returns:
        Résultat de hist2d ou None si données insuffisantes.
    """
    _require_matplotlib()

    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < 100:
        ax.text(
            0.5,
            0.5,
            "Données insuffisantes",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
            color=TEXT_COLOR,
        )
        ax.set_facecolor(BG_COLOR)
        return None

    x_lo, x_hi = np.percentile(x, [0.5, 99.5])
    y_lo, y_hi = np.percentile(y, [0.5, 99.5])

    cmap = LinearSegmentedColormap.from_list("density", DENSITY_CMAP_COLORS)

    h = ax.hist2d(
        x,
        y,
        bins=n_bins,
        range=[[x_lo, x_hi], [y_lo, y_hi]],
        cmap=cmap,
        norm=plt.matplotlib.colors.LogNorm(vmin=1),
    )

    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold", color=TEXT_COLOR)
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold", color=TEXT_COLOR)
    ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=12)
    apply_dark_style(ax)

    cbar = plt.colorbar(h[3], ax=ax, shrink=0.85)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    cbar.set_label("Densité", color=TEXT_COLOR, fontsize=11)

    return h


def plot_gating(
    ax: "plt.Axes",
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    label_in: str = "Conservés",
    label_out: str = "Exclus",
    max_pts: int = 100_000,
) -> None:
    """
    Density plot avec overlay gating en hexbin (vert=conservés, rouge=exclus).

    Args:
        ax: Axe matplotlib cible.
        x: Valeurs axe X (n_cells,).
        y: Valeurs axe Y (n_cells,).
        mask: Masque booléen — True = cellule conservée (n_cells,).
        title: Titre du graphique.
        xlabel: Label axe X.
        ylabel: Label axe Y.
        label_in: Légende cellules conservées.
        label_out: Légende cellules exclues.
        max_pts: Nombre maximum de points à afficher (sous-échantillonnage).
    """
    _require_matplotlib()

    valid = np.isfinite(x) & np.isfinite(y)
    x, y, mask = x[valid], y[valid], mask[valid]

    if len(x) < 100:
        ax.text(
            0.5,
            0.5,
            "Données insuffisantes",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
            color=TEXT_COLOR,
        )
        ax.set_facecolor(BG_COLOR)
        return

    if len(x) > max_pts:
        idx = np.random.choice(len(x), max_pts, replace=False)
        x, y, mask = x[idx], y[idx], mask[idx]

    excluded = ~mask
    kept = mask

    if excluded.any():
        ax.hexbin(
            x[excluded],
            y[excluded],
            gridsize=70,
            mincnt=1,
            cmap=HEXBIN_CMAP_EXCLUDED,
            linewidths=0,
            alpha=0.75,
            rasterized=True,
            zorder=1,
        )
    if kept.any():
        ax.hexbin(
            x[kept],
            y[kept],
            gridsize=70,
            mincnt=1,
            cmap=HEXBIN_CMAP_KEPT,
            linewidths=0,
            alpha=0.80,
            rasterized=True,
            zorder=2,
        )

    n_tot = len(x)
    n_in = int(mask.sum())
    pct = n_in / n_tot * 100 if n_tot > 0 else 0.0

    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold", color=TEXT_COLOR)
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold", color=TEXT_COLOR)
    ax.set_title(
        f"{title}\n{n_in:,} / {n_tot:,} ({pct:.1f}%)",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=12,
    )
    apply_dark_style(ax)

    legend_handles = [
        Patch(facecolor=COLOR_KEPT, edgecolor="none", label=label_in),
        Patch(facecolor=COLOR_EXCLUDED, edgecolor="none", label=label_out),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=10,
        facecolor=LEGEND_BG,
        labelcolor=TEXT_COLOR,
        edgecolor=SPINE_COLOR,
    )

    x_lo, x_hi = np.percentile(x, [0.5, 99.5])
    y_lo, y_hi = np.percentile(y, [0.5, 99.5])
    margin_x = (x_hi - x_lo) * 0.05
    margin_y = (y_hi - y_lo) * 0.05
    ax.set_xlim(x_lo - margin_x, x_hi + margin_x)
    ax.set_ylim(y_lo - margin_y, y_hi + margin_y)


def add_gate_rectangle(
    ax: "plt.Axes",
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    label: str = "Zone de sélection",
    color: str = "#f9e2af",
    linewidth: float = 3.0,
) -> None:
    """
    Ajoute un rectangle de gate sur un axe matplotlib.

    Args:
        ax: Axe cible.
        x_lo: Borne X inférieure.
        x_hi: Borne X supérieure.
        y_lo: Borne Y inférieure.
        y_hi: Borne Y supérieure.
        label: Texte affiché au-dessus du rectangle.
        color: Couleur du rectangle.
        linewidth: Épaisseur du trait.
    """
    _require_matplotlib()

    rect = Rectangle(
        (x_lo, y_lo),
        x_hi - x_lo,
        y_hi - y_lo,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle="--",
    )
    ax.add_patch(rect)
    ax.text(
        x_lo + (x_hi - x_lo) / 2,
        y_hi,
        f" {label}",
        ha="center",
        va="bottom",
        fontsize=11,
        color=color,
        fontweight="bold",
    )


def save_figure(
    fig: "plt.Figure",
    output_path: str,
    dpi: int = 100,
    tight: bool = True,
) -> None:
    """
    Sauvegarde et ferme une figure matplotlib.

    Args:
        fig: Figure à sauvegarder.
        output_path: Chemin de sortie (PNG, PDF, SVG).
        dpi: Résolution en DPI (100 par défaut — réduit le pic I/O disque de ~30%).
        tight: Appliquer tight_layout avant sauvegarde.
    """
    _require_matplotlib()

    if tight:
        fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=dpi,
        facecolor=BG_COLOR,
        bbox_inches="tight",
    )
    plt.close(fig)
