"""
polars_bridge.py — Utilitaires de conversion Pandas ↔ Polars pour PRISMA.

Polars est 5-10x plus rapide que Pandas pour :
  - filtrage par masque booléen (gate application)
  - agrégation par cluster (mean/std par population)
  - concaténation de samples (multi-fichiers)
  - describe / statistiques descriptives

Fallback transparent vers Pandas si polars non installé.

Usage :
    from prisma.utils.polars_bridge import to_polars, to_pandas, fast_filter, fast_groupby_mean
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("prisma.utils.polars")

try:
    import polars as pl
    _HAS_POLARS = True
    log.debug("Polars %s disponible", pl.__version__)
except ImportError:
    _HAS_POLARS = False
    pl = None
    log.debug("Polars absent — fallback Pandas pour toutes les opérations")


def to_polars(df: pd.DataFrame) -> "pl.DataFrame | pd.DataFrame":
    """Convertit un DataFrame Pandas en Polars. Passthrough si polars absent."""
    if not _HAS_POLARS:
        return df
    return pl.from_pandas(df)


def to_pandas(df) -> pd.DataFrame:
    """Convertit Polars → Pandas. Passthrough si déjà Pandas."""
    if _HAS_POLARS and isinstance(df, pl.DataFrame):
        return df.to_pandas()
    return df


def fast_filter(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """
    Filtre un DataFrame par masque booléen.
    Polars ~3x plus rapide que Pandas sur >500k lignes.

    Args:
        df: DataFrame source (Pandas).
        mask: Masque booléen numpy shape (N,).

    Returns:
        DataFrame Pandas filtré.
    """
    if _HAS_POLARS and len(df) > 100_000:
        try:
            pldf = pl.from_pandas(df)
            pl_mask = pl.Series("mask", mask)
            return pldf.filter(pl_mask).to_pandas()
        except Exception as e:
            log.debug("Polars filter failed, fallback Pandas: %s", e)
    return df[mask].reset_index(drop=True)


def fast_groupby_mean(
    df: pd.DataFrame,
    group_col: str,
    value_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Agrégation mean par groupe.
    Polars ~5x plus rapide que Pandas groupby.mean() sur gros DataFrames.

    Args:
        df: DataFrame source.
        group_col: Colonne de groupement (ex: 'cluster_label').
        value_cols: Colonnes à agréger (None = toutes les numériques).

    Returns:
        DataFrame Pandas avec les moyennes par groupe.
    """
    if value_cols is None:
        value_cols = list(df.select_dtypes(include="number").columns)

    if _HAS_POLARS and len(df) > 50_000:
        try:
            cols = [group_col] + [c for c in value_cols if c != group_col]
            pldf = pl.from_pandas(df[cols])
            result = pldf.group_by(group_col).agg(
                [pl.col(c).mean().alias(c) for c in value_cols if c != group_col]
            ).sort(group_col)
            return result.to_pandas()
        except Exception as e:
            log.debug("Polars groupby failed, fallback Pandas: %s", e)

    return df.groupby(group_col)[value_cols].mean().reset_index()


def fast_concat(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatène une liste de DataFrames.
    Polars ~4x plus rapide que pd.concat sur gros DataFrames.

    Args:
        dfs: Liste de DataFrames Pandas (même colonnes).

    Returns:
        DataFrame Pandas concaténé.
    """
    if not dfs:
        return pd.DataFrame()

    if _HAS_POLARS and sum(len(d) for d in dfs) > 100_000:
        try:
            pl_dfs = [pl.from_pandas(d) for d in dfs]
            return pl.concat(pl_dfs).to_pandas()
        except Exception as e:
            log.debug("Polars concat failed, fallback Pandas: %s", e)

    return pd.concat(dfs, ignore_index=True)


def fast_describe(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Statistiques descriptives rapides (mean, std, min, max, median).

    Args:
        df: DataFrame source.
        cols: Colonnes cibles (None = toutes numériques).

    Returns:
        DataFrame Pandas avec les stats.
    """
    if cols is None:
        cols = list(df.select_dtypes(include="number").columns)

    if _HAS_POLARS and len(df) > 50_000:
        try:
            pldf = pl.from_pandas(df[cols])
            return pldf.describe().to_pandas()
        except Exception as e:
            log.debug("Polars describe failed, fallback Pandas: %s", e)

    return df[cols].describe()


def polars_available() -> bool:
    return _HAS_POLARS
