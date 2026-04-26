"""
fcs_reader.py — Chargement de fichiers FCS vers AnnData/FlowSample.

Fournit get_fcs_files() et load_fcs_files() pour la découverte
et le chargement de fichiers FCS via la librairie flowsom.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import anndata as ad

    _ANNDATA_AVAILABLE = True
except ImportError:
    _ANNDATA_AVAILABLE = False
    ad = None

try:
    import flowsom as fs

    _FLOWSOM_AVAILABLE = True
except Exception as _flowsom_err:
    _FLOWSOM_AVAILABLE = False
    warnings.warn(f"flowsom non disponible ({type(_flowsom_err).__name__}: {_flowsom_err})")

from src.prisma.core.models_legacy.sample import FlowSample
from src.prisma.utils.logger import get_logger

_logger = get_logger("io.fcs_reader")


def get_fcs_files(folder: Path | str) -> List[Path]:
    """
    Découverte de tous les fichiers FCS dans un dossier (non récursif).

    Gère les extensions .fcs et .FCS (case-insensitive sur Windows).

    Args:
        folder: Chemin vers le dossier contenant les fichiers FCS.

    Returns:
        Liste triée des chemins absolus des fichiers FCS trouvés.
    """
    folder = Path(folder)

    if not folder.exists():
        _logger.warning("Dossier non trouvé: %s", folder)
        return []

    # Dédupliquer (cas Windows: .fcs et .FCS)
    seen: set[str] = set()
    files: List[Path] = []

    for pattern in ("*.fcs", "*.FCS"):
        for f in folder.glob(pattern):
            lower_name = f.name.lower()
            if lower_name not in seen:
                seen.add(lower_name)
                files.append(f.resolve())

    files.sort(key=lambda p: p.name.lower())
    _logger.info("%d fichier(s) FCS trouvés dans %s", len(files), folder)
    return files


def load_fcs_file(
    fpath: Path | str,
    condition: str = "Unknown",
) -> Optional["ad.AnnData"]:
    """
    Charge un seul fichier FCS en AnnData.

    Args:
        fpath: Chemin vers le fichier FCS.
        condition: Label de condition à stocker dans obs["condition"].

    Returns:
        AnnData ou None si échec.
    """
    if not _FLOWSOM_AVAILABLE:
        raise ImportError("flowsom requis pour load_fcs_file: pip install flowsom")

    fpath = Path(fpath)
    try:
        adata = fs.io.read_FCS(str(fpath))
        adata.obs["condition"] = condition
        adata.obs["file_origin"] = fpath.name
        _logger.info("Chargé: %s — %d cellules", fpath.name, adata.n_obs)
        return adata
    except Exception as exc:
        _logger.error("Échec chargement %s: %s", fpath.name, exc)
        return None


def load_fcs_files(
    files: List[Path | str],
    condition: str = "Unknown",
) -> List["ad.AnnData"]:
    """
    Charge plusieurs fichiers FCS et retourne une liste d'AnnData.

    Les fichiers en échec sont ignorés (log d'erreur émis).

    Args:
        files: Liste des chemins de fichiers FCS.
        condition: Label de condition ("Sain", "Pathologique", etc.).

    Returns:
        Liste des AnnData chargés avec succès.
    """
    adatas: List["ad.AnnData"] = []

    for fpath in files:
        adata = load_fcs_file(fpath, condition=condition)
        if adata is not None:
            adatas.append(adata)

    _logger.info(
        "%d/%d fichier(s) FCS chargés pour condition '%s'",
        len(adatas),
        len(files),
        condition,
    )
    return adatas


def load_as_flow_samples(
    files: List[Path | str],
    condition: str = "Unknown",
) -> List[FlowSample]:
    """
    Charge plusieurs fichiers FCS directement en FlowSample.

    Wrapper pratique autour de load_fcs_files() + FlowSample.from_anndata().

    Args:
        files: Liste des chemins de fichiers FCS.
        condition: Label de condition.

    Returns:
        Liste des FlowSample créés.
    """
    adatas = load_fcs_files(files, condition=condition)
    samples: List[FlowSample] = []

    for adata in adatas:
        fpath = (
            Path(adata.obs["file_origin"].iloc[0])
            if "file_origin" in adata.obs.columns
            else Path("unknown.fcs")
        )
        sample = FlowSample.from_anndata(adata, path=fpath, condition=condition)
        samples.append(sample)

    return samples
