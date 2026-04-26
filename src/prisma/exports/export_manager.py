"""Gestionnaire d'exports analytiques (CSV, parquet, PNG, JSON)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from src.prisma.exports.heatmaps import save_cluster_heatmap

logger = logging.getLogger(__name__)


class ExportManager:
    """Centralise les exports du pipeline vers un répertoire de run."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_csv(self, df: pd.DataFrame, name: str, index: bool = True) -> Path:
        path = self.output_dir / f"{name}.csv"
        df.to_csv(path, index=index)
        return path

    def export_parquet(
        self,
        df: pd.DataFrame,
        name: str,
        index: bool = True,
        engine: Optional[str] = None,
    ) -> Optional[Path]:
        """Exporte en parquet si un moteur est disponible, sinon retourne None."""
        path = self.output_dir / f"{name}.parquet"
        try:
            df.to_parquet(path, index=index, engine=engine)
        except (ImportError, ValueError, ModuleNotFoundError) as exc:
            logger.warning("Parquet indisponible pour %s: %s", name, exc)
            return None
        return path

    def export_json(self, payload: Mapping[str, Any], name: str) -> Path:
        path = self.output_dir / f"{name}.json"
        path.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
        return path

    def export_heatmap_png(self, matrix: pd.DataFrame, name: str, **kwargs: Any) -> Path:
        path = self.output_dir / f"{name}.png"
        return save_cluster_heatmap(matrix=matrix, output_path=path, **kwargs)

    def export_tables(
        self,
        tables: Mapping[str, pd.DataFrame],
        *,
        export_parquet: bool = True,
    ) -> Dict[str, Dict[str, str]]:
        """Exporte un lot de tables en CSV + parquet optionnel."""
        artifacts: Dict[str, Dict[str, str]] = {}

        for table_name, df in tables.items():
            table_artifacts: Dict[str, str] = {}

            csv_path = self.export_csv(df, table_name)
            table_artifacts["csv"] = str(csv_path)

            if export_parquet:
                parquet_path = self.export_parquet(df, table_name)
                if parquet_path is not None:
                    table_artifacts["parquet"] = str(parquet_path)

            artifacts[table_name] = table_artifacts

        return artifacts
