"""Utilitaires analytiques pour tables cohortes et heatmaps cluster x marqueurs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib
import pandas as pd
import seaborn as sns

# Pourquoi: forcer un backend non interactif évite les plantages en CI/headless.
matplotlib.use("Agg")


def _resolve_marker_columns(
    df: pd.DataFrame,
    cluster_col: str,
    sample_col: str,
    marker_cols: Optional[Sequence[str]],
) -> list[str]:
    if cluster_col not in df.columns:
        raise KeyError(f"Colonne cluster absente: {cluster_col}")
    if sample_col not in df.columns:
        raise KeyError(f"Colonne sample absente: {sample_col}")

    if marker_cols is None:
        excluded = {cluster_col, sample_col}
        marker_candidates = [c for c in df.columns if c not in excluded]
        marker_cols = [c for c in marker_candidates if pd.api.types.is_numeric_dtype(df[c])]

    missing = [c for c in marker_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes marqueurs absentes: {missing}")
    if not marker_cols:
        raise ValueError("Aucune colonne marqueur exploitable.")
    return list(marker_cols)


def build_cluster_marker_matrix(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    sample_col: str = "sample_id",
    marker_cols: Optional[Sequence[str]] = None,
    agg: str = "median",
) -> pd.DataFrame:
    """Construit la matrice cluster x marqueurs (MFI/median par défaut)."""
    markers = _resolve_marker_columns(df, cluster_col, sample_col, marker_cols)

    if agg not in {"median", "mean"}:
        raise ValueError("agg doit valoir 'median' ou 'mean'.")

    grouped = df.groupby(cluster_col, observed=False)[markers]
    matrix = grouped.median() if agg == "median" else grouped.mean()
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    matrix.index.name = cluster_col
    return matrix


def compute_mfi_table(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    sample_col: str = "sample_id",
    marker_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Alias explicite métier: table MFI (médianes) par cluster."""
    return build_cluster_marker_matrix(
        df=df,
        cluster_col=cluster_col,
        sample_col=sample_col,
        marker_cols=marker_cols,
        agg="median",
    )


def compute_cluster_abundance(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    sample_col: str = "sample_id",
    normalize: bool = True,
) -> pd.DataFrame:
    """Calcule l'abondance cluster par sample (fréquence ou comptes)."""
    if cluster_col not in df.columns:
        raise KeyError(f"Colonne cluster absente: {cluster_col}")
    if sample_col not in df.columns:
        raise KeyError(f"Colonne sample absente: {sample_col}")

    normalize_mode: Any = "index" if normalize else False
    table = pd.crosstab(
        index=df[sample_col],
        columns=df[cluster_col],
        normalize=normalize_mode,
    )
    table = table.sort_index(axis=0).sort_index(axis=1)
    table.index.name = sample_col
    table.columns.name = cluster_col
    return table


def build_cohort_export_tables(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    sample_col: str = "sample_id",
    marker_cols: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Prépare les tables cohorte prêtes à exporter."""
    mfi = compute_mfi_table(
        df=df,
        cluster_col=cluster_col,
        sample_col=sample_col,
        marker_cols=marker_cols,
    )
    abundance_freq = compute_cluster_abundance(
        df=df,
        cluster_col=cluster_col,
        sample_col=sample_col,
        normalize=True,
    )
    abundance_counts = compute_cluster_abundance(
        df=df,
        cluster_col=cluster_col,
        sample_col=sample_col,
        normalize=False,
    )

    overall = (
        df.groupby(cluster_col, observed=False).size().rename("n_cells").to_frame().sort_index()
    )
    total_cells = int(overall["n_cells"].sum())
    overall["frequency"] = overall["n_cells"] / total_cells if total_cells else 0.0

    return {
        "cluster_marker_mfi": mfi,
        "abundance_by_sample_freq": abundance_freq,
        "abundance_by_sample_counts": abundance_counts,
        "cluster_abundance_overall": overall,
    }


def save_cluster_heatmap(
    matrix: pd.DataFrame,
    output_path: Path | str,
    *,
    cmap: str = "mako",
    metric: str = "euclidean",
    method: str = "average",
    row_cluster: bool = True,
    col_cluster: bool = True,
    z_score: Optional[int] = None,
    standard_scale: Optional[int] = None,
    dpi: int = 160,
) -> Path:
    """Sauvegarde une heatmap clusterisée (seaborn.clustermap) en PNG."""
    if matrix.empty:
        raise ValueError("Matrice vide: impossible de générer une heatmap.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    grid = sns.clustermap(
        matrix,
        cmap=cmap,
        metric=metric,
        method=method,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        z_score=z_score,
        standard_scale=standard_scale,
        linewidths=0.05,
        figsize=(10, 8),
    )
    grid.ax_heatmap.set_xlabel("Marqueurs")
    grid.ax_heatmap.set_ylabel("Clusters")
    grid.fig.suptitle("Heatmap cluster x marqueurs", y=1.02)
    grid.savefig(out, dpi=dpi, bbox_inches="tight")
    grid.fig.clear()
    return out
