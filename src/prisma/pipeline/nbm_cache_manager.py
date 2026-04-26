"""
nbm_cache_manager.py — Cache des données NBM pré-processées pour le mode batch.

Objectif : éviter de relire les FCS NBM, de refaire le gating et la
transformation (arcsinh, z-score…) à chaque itération de la boucle batch.

Le modèle SOM N'est PAS mis en cache. Il est ré-entraîné de zéro à chaque
itération sur [NBM subsamplé + Patho], ce qui permet à chaque grille de
s'adapter à la topologie de la moelle pathologique courante.

Niveaux de cache :
  • Un seul niveau : DataFrame NBM pré-processé → fichier Parquet (pyarrow).
    Fallback : fichier pickle via joblib si pyarrow est absent.

Invalidation automatique :
  Le nom du fichier cache intègre deux hashes courts :
    - hash des fichiers FCS source (chemin + taille + mtime)
    - hash des paramètres de preprocessing (gating, transform, normalize, markers)
  Si l'un d'eux change, le cache est ignoré et recalculé.

Usage dans BatchPipeline :
    manager = NBMCacheManager(config, cache_dir, nbm_file_paths)

    if not manager.has_cache():
        samples = preprocess_nbm(...)
        manager.save(samples, selected_markers)

    df_nbm, selected_markers = manager.load()   # < 1 s
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _PARQUET_AVAILABLE = True
except ImportError:
    _PARQUET_AVAILABLE = False

try:
    from joblib import dump as _jdump, load as _jload
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

from config.pipeline_config import PipelineConfig
from src.prisma.core.models_legacy.sample import FlowSample
from src.prisma.utils.logger import get_logger

_logger = get_logger("pipeline.nbm_cache")


# ---------------------------------------------------------------------------
# Helpers de hash
# ---------------------------------------------------------------------------

def _hash_files(file_paths: List[str]) -> str:
    """Empreinte SHA256 (16 chars) basée sur chemin + taille + mtime des FCS."""
    entries = []
    for fp in sorted(file_paths):
        p = Path(fp)
        stat = p.stat() if p.exists() else None
        entries.append({
            "path": str(p),
            "size": int(stat.st_size) if stat else -1,
            "mtime_ns": int(stat.st_mtime_ns) if stat else -1,
        })
    blob = json.dumps(entries, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _hash_code_version() -> str:
    """
    Empreinte SHA256 (16 chars) des fichiers Python critiques du pipeline.

    Invalide automatiquement le cache si le code de preprocessing ou de
    clustering change — évite de recharger un cache calculé avec une ancienne
    logique. Seuls les modules qui influencent le DataFrame NBM pré-processé
    sont hashés (gating, transformers, normalizers, clustering, fcs_reader).

    Fallback sur la chaîne "unknown" si les fichiers sont introuvables
    (ex : exécution post-PyInstaller depuis un .exe).
    """
    _HERE = Path(__file__).parent
    _SRC = _HERE.parent

    # Modules qui, s'ils changent, invalident le résultat du preprocessing NBM
    candidates = [
        _SRC / "core" / "gating.py",
        _SRC / "core" / "auto_gating.py",
        _SRC / "core" / "transformers.py",
        _SRC / "core" / "normalizers.py",
        _SRC / "io" / "fcs_reader.py",
        _SRC / "services" / "preprocessing_service.py",
    ]

    h = hashlib.sha256()
    found_any = False
    for path in candidates:
        if path.exists():
            try:
                h.update(path.read_bytes())
                found_any = True
            except OSError:
                pass

    return h.hexdigest()[:16] if found_any else "unknown"


# Calculé une seule fois au démarrage du processus — stable pour toute la session.
_CODE_VERSION_HASH: str = _hash_code_version()


def _hash_prep_config(config: PipelineConfig) -> str:
    """Empreinte SHA256 (16 chars) des paramètres de preprocessing + version du code.

    ARCH-3 FIX : utilise dataclasses.asdict() au lieu de vars() pour garantir :
      - compatibilité avec les dataclasses utilisant __slots__
      - sérialisation correcte des sous-objets dataclass imbriqués
      - comportement stable indépendant de l'implémentation interne Python

    Le hash intègre _CODE_VERSION_HASH pour invalider le cache si le code
    de preprocessing ou de clustering change entre deux runs.
    """
    def _safe_asdict(obj: object) -> object:
        """Convertit récursivement un dataclass en dict, fallback sur str."""
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        return str(obj)

    payload = {
        "pregate":      _safe_asdict(config.pregate),
        "transform":    _safe_asdict(config.transform),
        "normalize":    _safe_asdict(config.normalize),
        "markers":      _safe_asdict(config.markers),
        "downsampling": _safe_asdict(config.downsampling),
        # Invalide le cache si le code change
        "_code_version": _CODE_VERSION_HASH,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# NBMCacheManager
# ---------------------------------------------------------------------------

class NBMCacheManager:
    """
    Gère la sérialisation/désérialisation du DataFrame NBM pré-processé.

    Le DataFrame stocké contient :
      - Une colonne par marqueur (valeurs transformées + normalisées).
      - Une colonne '_sample_id' (nom du fichier FCS source).
    Les marqueurs sélectionnés pour le clustering sont stockés dans un
    fichier JSON meta adjacent.

    Args:
        config:          Configuration du pipeline.
        cache_dir:       Dossier de stockage (créé si absent).
        nbm_file_paths:  Chemins des FCS NBM (pour le hash d'invalidation).
    """

    _META_SUFFIX = "_meta.json"

    def __init__(
        self,
        config: PipelineConfig,
        cache_dir: Path,
        nbm_file_paths: List[str],
    ) -> None:
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        _fh = _hash_files(nbm_file_paths)
        _ph = _hash_prep_config(config)
        _stem = f"nbm_data_{_fh}_{_ph}"

        # Format préféré : Parquet ; fallback : joblib pickle
        if _PARQUET_AVAILABLE:
            self._data_path = self.cache_dir / f"{_stem}.parquet"
        elif _JOBLIB_AVAILABLE:
            self._data_path = self.cache_dir / f"{_stem}.pkl.joblib"
        else:
            self._data_path = self.cache_dir / f"{_stem}.pkl.joblib"
            _logger.error(
                "Ni pyarrow ni joblib ne sont installés. "
                "Le cache NBM sera non fonctionnel. "
                "Installez : pip install pyarrow"
            )

        self._meta_path = self.cache_dir / f"{_stem}{self._META_SUFFIX}"

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def has_cache(self) -> bool:
        """Retourne True si les deux fichiers cache (data + meta) existent."""
        return self._data_path.exists() and self._meta_path.exists()

    def save(
        self,
        samples: List[FlowSample],
        selected_markers: List[str],
    ) -> None:
        """
        Sérialise les FlowSamples NBM pré-processés sur disque.

        Le DataFrame écrit contient toutes les colonnes de marqueurs
        présentes dans les samples, plus une colonne '_sample_id'.
        selected_markers (sous-ensemble utilisé pour le SOM) est stocké
        dans le fichier meta JSON.

        Args:
            samples:          FlowSamples pré-processés (condition healthy).
            selected_markers: Marqueurs retenus pour le clustering.
        """
        if not samples:
            _logger.warning("NBMCacheManager.save: liste vide, rien à sauvegarder.")
            return

        _logger.info(
            "Sauvegarde cache DATA NBM: %d échantillons → %s",
            len(samples),
            self._data_path.name,
        )

        # Construire le DataFrame concaténé
        chunks: List[pd.DataFrame] = []
        for s in samples:
            df = s.data.copy()
            df["_sample_id"] = s.name
            chunks.append(df)

        df_all = pd.concat(chunks, ignore_index=True)

        # Supprimer les colonnes dupliquées (ex: FSC-Width, Time présents deux fois
        # dans certains fichiers FCS) pour éviter l'erreur pyarrow à la sauvegarde.
        # On garde toujours la PREMIÈRE occurrence (comportement pandas par défaut).
        if df_all.columns.duplicated().any():
            dup_col_names = df_all.columns[df_all.columns.duplicated(keep=False)].unique().tolist()
            # Vérifier si les colonnes dupliquées ont des valeurs identiques
            non_identical = []
            for col in dup_col_names:
                col_data = df_all.loc[:, df_all.columns == col]
                if col_data.shape[1] > 1:
                    first = col_data.iloc[:, 0]
                    for k in range(1, col_data.shape[1]):
                        if not first.equals(col_data.iloc[:, k]):
                            non_identical.append(col)
                            break
            if non_identical:
                _logger.error(
                    "Colonnes dupliquées avec valeurs DIFFÉRENTES détectées : %s. "
                    "La première occurrence est conservée — vérifiez les fichiers FCS source.",
                    non_identical,
                )
            else:
                _logger.warning(
                    "Colonnes dupliquées (valeurs identiques) supprimées avant sauvegarde cache: %s",
                    dup_col_names,
                )
            df_all = df_all.loc[:, ~df_all.columns.duplicated(keep="first")]

        n_cells = len(df_all)

        try:
            if _PARQUET_AVAILABLE and self._data_path.suffix == ".parquet":
                table = pa.Table.from_pandas(df_all, preserve_index=False)
                pq.write_table(table, self._data_path, compression="snappy")
            elif _JOBLIB_AVAILABLE:
                _jdump(df_all, self._data_path, compress=3)
            else:
                _logger.error("Impossible de sauvegarder le cache: aucun sérialiseur disponible.")
                return

            meta = {
                "selected_markers": selected_markers,
                "n_cells": int(n_cells),
                "n_samples": int(len(samples)),
            }
            with open(self._meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            _logger.info(
                "Cache DATA NBM sauvegardé: %d cellules, %d marqueurs sélectionnés.",
                n_cells, len(selected_markers),
            )

        except Exception as exc:
            _logger.error("Échec sauvegarde cache DATA NBM: %s", exc)
            # Nettoyer pour éviter un cache partiel/corrompu
            for p in (self._data_path, self._meta_path):
                if p.exists():
                    p.unlink(missing_ok=True)

    def load(self) -> Optional[Tuple[pd.DataFrame, List[str]]]:
        """
        Charge le DataFrame NBM pré-processé et les marqueurs sélectionnés.

        Returns:
            (df_nbm, selected_markers) si le cache est valide.
            None en cas d'erreur ou cache absent.

        Note:
            Le DataFrame retourné contient la colonne '_sample_id'.
            Pour extraire X_nbm : df_nbm[selected_markers].values
        """
        if not self.has_cache():
            return None

        _logger.info("Chargement cache DATA NBM depuis: %s", self._data_path.name)

        try:
            with open(self._meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            selected_markers: List[str] = meta["selected_markers"]

            if _PARQUET_AVAILABLE and self._data_path.suffix == ".parquet":
                table = pq.read_table(self._data_path)
                df_nbm = table.to_pandas()
            elif _JOBLIB_AVAILABLE:
                df_nbm = _jload(self._data_path)
            else:
                _logger.error("Impossible de charger le cache: aucun désérialiseur disponible.")
                return None

            # Restreindre le DataFrame aux colonnes utiles uniquement :
            # selected_markers (pour le SOM) + _sample_id (pour le groupby).
            # Cela évite que des colonnes résiduelles du preprocessing
            # (ex: marqueurs scatter, _condition, etc.) se retrouvent dans
            # les FlowSamples reconstruits et polluent le select_markers.
            keep_cols = [m for m in selected_markers if m in df_nbm.columns]
            if "_sample_id" in df_nbm.columns:
                keep_cols = keep_cols + ["_sample_id"]
            df_nbm = df_nbm[keep_cols]

            _logger.info(
                "Cache DATA NBM chargé: %d cellules, %d marqueurs sélectionnés.",
                len(df_nbm), len(selected_markers),
            )
            return df_nbm, selected_markers

        except Exception as exc:
            _logger.error("Échec chargement cache DATA NBM: %s", exc)
            self.invalidate()
            return None

    def invalidate(self) -> None:
        """Supprime les fichiers cache (données corrompues ou périmées)."""
        for p in (self._data_path, self._meta_path):
            if p.exists():
                p.unlink(missing_ok=True)
                _logger.info("Cache NBM invalidé: %s", p.name)

    def summary(self) -> str:
        """Résumé lisible de l'état du cache."""
        if self.has_cache():
            size_mb = self._data_path.stat().st_size / 1_048_576
            return (
                f"NBMCacheManager — OK "
                f"({self._data_path.name}, {size_mb:.1f} MB)"
            )
        return f"NBMCacheManager — absent ({self._data_path.name})"
