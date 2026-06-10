# -*- coding: utf-8 -*-
"""
flowsom_native_plots.py — Visualisations natives de la librairie FlowSOM (saeyslab).

Wrappe l'intégralité des fonctions ``fs.pl.*`` de la librairie officielle FlowSOM
(https://flowsom.readthedocs.io) et les sauvegarde en image (PNG/JPG/SVG/PDF).

Sorties reproduites depuis la documentation officielle :
  - plot_stars (vue MST + vue grille)        → étoiles de marqueurs par nœud
  - plot_marker                              → expression d'un marqueur sur la grille
  - plot_pies                                → camemberts de types cellulaires par nœud
  - plot_numbers / plot_labels               → numéros / étiquettes des nœuds
  - plot_variable                            → variable continue par nœud
  - plot_2D_scatters                         → nuages 2D par cluster/métacluster
  - FlowSOMmary                              → planche de synthèse complète

API additionnelle (workflow saeyslab) :
  - new_data : projection de nouvelles données sur un FlowSOM existant
  - subset   : sous-FlowSOM par métacluster

Toutes les fonctions sont défensives : si FlowSOM/matplotlib indisponible, ou si
le backend est GPU (pas d'objet ``fs.FlowSOM`` natif), elles loguent et renvoient
None sans planter le pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

_logger = logging.getLogger("prisma.viz.flowsom_native")

# Imports optionnels — dégradation gracieuse.
try:
    import matplotlib

    matplotlib.use("Agg")  # backend non interactif (sûr hors thread principal)
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except Exception:  # noqa: BLE001
    _MPL_AVAILABLE = False

try:
    import flowsom as fs  # noqa: F401

    _FLOWSOM_AVAILABLE = True
except Exception:  # noqa: BLE001
    _FLOWSOM_AVAILABLE = False


# Formats image supportés (déterminés par l'extension du chemin de sortie).
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".tif", ".tiff"}


def _resolve_fsom(obj: Any) -> Optional[Any]:
    """Accepte soit un FlowSOMClusterer, soit un objet fs.FlowSOM natif.

    Retourne l'objet fs.FlowSOM natif (avec get_cluster_data) ou None.
    """
    if obj is None:
        return None
    # FlowSOMClusterer : expose .fsom_model (objet natif si CPU)
    model = getattr(obj, "fsom_model", obj)
    if model is not None and hasattr(model, "get_cluster_data"):
        return model
    return None


def _save_figure(fig: Any, output_path: Path | str, dpi: int = 150) -> Optional[Path]:
    """Sauvegarde une figure matplotlib dans le format déduit de l'extension.

    JPG nécessite un fond opaque (pas d'alpha) — on force facecolor blanc.
    """
    if fig is None:
        return None
    output_path = Path(output_path)
    ext = output_path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        _logger.warning("Format '%s' non supporté — bascule en .png", ext)
        output_path = output_path.with_suffix(".png")
        ext = ".png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict = {"dpi": dpi, "bbox_inches": "tight"}
    if ext in (".jpg", ".jpeg"):
        # JPEG ne gère pas la transparence : fond blanc opaque.
        save_kwargs["facecolor"] = "white"
    try:
        fig.savefig(str(output_path), **save_kwargs)
        return output_path
    except Exception as exc:  # noqa: BLE001
        _logger.warning("Échec sauvegarde figure %s : %s", output_path, exc)
        return None
    finally:
        try:
            plt.close(fig)
        except Exception:  # noqa: BLE001
            pass


def _guard() -> bool:
    """Vérifie la disponibilité de matplotlib + flowsom."""
    if not _MPL_AVAILABLE:
        _logger.warning("matplotlib indisponible — visualisation FlowSOM ignorée.")
        return False
    if not _FLOWSOM_AVAILABLE:
        _logger.warning("flowsom indisponible — visualisation native ignorée.")
        return False
    return True


# ---------------------------------------------------------------------------
# plot_stars — Star Chart (étoiles de marqueurs)
# ---------------------------------------------------------------------------

def plot_stars_native(
    fsom: Any,
    output_path: Path | str,
    *,
    view: str = "MST",
    background_metaclusters: bool = True,
    equal_node_size: bool = False,
    equal_background_size: bool = False,
    markers: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    dpi: int = 150,
) -> Optional[Path]:
    """Star Chart FlowSOM via ``fs.pl.plot_stars``.

    Reproduit la documentation officielle ::

        fs.pl.plot_stars(
            fsom,
            background_values=fsom.get_cluster_data().obs.metaclustering,
            view="grid", equal_node_size=True, equal_background_size=True,
        )

    Args:
        fsom: FlowSOMClusterer ou objet fs.FlowSOM natif.
        output_path: Chemin image (extension détermine le format).
        view: "MST" (par défaut) ou "grid".
        background_metaclusters: Colorer le fond par métacluster.
        equal_node_size: Forcer une taille de nœud uniforme (vue grid).
        equal_background_size: Forcer une taille de fond uniforme.
        markers: Sous-ensemble de marqueurs à afficher (None = tous).
        title: Titre optionnel.
        dpi: Résolution.

    Returns:
        Chemin de l'image générée ou None.
    """
    if not _guard():
        return None
    model = _resolve_fsom(fsom)
    if model is None:
        _logger.warning("plot_stars_native : objet FlowSOM natif indisponible (backend GPU ?).")
        return None
    try:
        kwargs: dict = {"view": view}
        if background_metaclusters:
            kwargs["background_values"] = model.get_cluster_data().obs.metaclustering
        if equal_node_size:
            kwargs["equal_node_size"] = True
        if equal_background_size:
            kwargs["equal_background_size"] = True
        if markers is not None:
            kwargs["markers"] = list(markers)
        if title is not None:
            kwargs["title"] = title

        fig = fs.pl.plot_stars(model, **kwargs)
        return _save_figure(fig, output_path, dpi=dpi)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("plot_stars_native échoué (view=%s) : %s", view, exc)
        return None


# ---------------------------------------------------------------------------
# plot_marker — expression d'un marqueur sur la grille
# ---------------------------------------------------------------------------

def plot_marker_native(
    fsom: Any,
    marker: str,
    output_path: Path | str,
    *,
    dpi: int = 150,
) -> Optional[Path]:
    """Carte d'expression d'un marqueur via ``fs.pl.plot_marker``."""
    if not _guard():
        return None
    model = _resolve_fsom(fsom)
    if model is None:
        return None
    try:
        # fs.pl.plot_marker itère sur `marker` → exige une liste (une chaîne
        # serait parcourue caractère par caractère → "Marker ^C$ not found").
        fig = fs.pl.plot_marker(model, marker=[marker])
        fig = fig if fig is not None else plt.gcf()
        return _save_figure(fig, output_path, dpi=dpi)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("plot_marker_native('%s') échoué : %s", marker, exc)
        return None


# ---------------------------------------------------------------------------
# plot_numbers / plot_labels — annotations des nœuds
# ---------------------------------------------------------------------------

def plot_numbers_native(
    fsom: Any,
    output_path: Path | str,
    *,
    level: str = "clusters",
    dpi: int = 150,
) -> Optional[Path]:
    """Numéros des nœuds/métaclusters via ``fs.pl.plot_numbers``.

    Args:
        level: "clusters" (nœuds SOM) ou "metaclusters".
    """
    if not _guard():
        return None
    model = _resolve_fsom(fsom)
    if model is None:
        return None
    try:
        fig = fs.pl.plot_numbers(model, level=level)
        # plot_numbers dessine sur la figure courante et peut retourner None.
        fig = fig if fig is not None else plt.gcf()
        return _save_figure(fig, output_path, dpi=dpi)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("plot_numbers_native(level=%s) échoué : %s", level, exc)
        return None


# ---------------------------------------------------------------------------
# plot_variable — variable continue par nœud
# ---------------------------------------------------------------------------

def plot_variable_native(
    fsom: Any,
    variable: np.ndarray,
    output_path: Path | str,
    *,
    labels: Optional[Sequence[str]] = None,
    dpi: int = 150,
) -> Optional[Path]:
    """Visualise une variable continue (n_nodes,) via ``fs.pl.plot_variable``."""
    if not _guard():
        return None
    model = _resolve_fsom(fsom)
    if model is None:
        return None
    try:
        kwargs: dict = {"variable": np.asarray(variable)}
        if labels is not None:
            kwargs["labels"] = list(labels)
        fig = fs.pl.plot_variable(model, **kwargs)
        return _save_figure(fig, output_path, dpi=dpi)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("plot_variable_native échoué : %s", exc)
        return None


# ---------------------------------------------------------------------------
# plot_2D_scatters — nuages 2D par cluster/métacluster
# ---------------------------------------------------------------------------

def plot_2d_scatters_native(
    fsom: Any,
    channelpairs: Sequence[Tuple[str, str]],
    output_path: Path | str,
    *,
    clusters: Optional[Sequence[int]] = None,
    metaclusters: Optional[Sequence[int]] = None,
    dpi: int = 150,
) -> Optional[Path]:
    """Nuages 2D bicanaux via ``fs.pl.plot_2D_scatters``."""
    if not _guard():
        return None
    model = _resolve_fsom(fsom)
    if model is None:
        return None
    try:
        kwargs: dict = {"channelpairs": [list(p) for p in channelpairs]}
        if clusters is not None:
            kwargs["clusters"] = list(clusters)
        if metaclusters is not None:
            kwargs["metaclusters"] = list(metaclusters)
        fig = fs.pl.plot_2D_scatters(model, **kwargs)
        return _save_figure(fig, output_path, dpi=dpi)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("plot_2d_scatters_native échoué : %s", exc)
        return None


# ---------------------------------------------------------------------------
# FlowSOMmary — planche de synthèse complète
# ---------------------------------------------------------------------------

def plot_flowsommary_native(
    fsom: Any,
    output_path: Path | str,
    *,
    dpi: int = 150,
) -> Optional[Path]:
    """Planche récapitulative complète via ``fs.pl.FlowSOMmary``.

    FlowSOMmary génère et sauvegarde directement (paramètre ``plot_file``).
    """
    if not _guard():
        return None
    model = _resolve_fsom(fsom)
    if model is None:
        return None
    output_path = Path(output_path)
    if output_path.suffix.lower() not in SUPPORTED_FORMATS:
        output_path = output_path.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fs.pl.FlowSOMmary(model, plot_file=str(output_path))
        return output_path if output_path.exists() else None
    except Exception as exc:  # noqa: BLE001
        _logger.warning("plot_flowsommary_native échoué : %s", exc)
        return None


# ---------------------------------------------------------------------------
# new_data — projection de nouvelles données
# ---------------------------------------------------------------------------

def plot_new_data_stars(
    fsom: Any,
    X_new: np.ndarray,
    output_path: Path | str,
    *,
    view: str = "MST",
    dpi: int = 150,
) -> Optional[Path]:
    """Projette de nouvelles données sur le FlowSOM puis trace le Star Chart.

    Reproduit la documentation officielle ::

        fsom_new = fsom.new_data(ff_t[1:200, :])
        fs.pl.plot_stars(fsom_new,
            background_values=fsom_new.get_cluster_data().obs.metaclustering)

    Args:
        fsom: FlowSOMClusterer (entraîné) ou objet fs.FlowSOM natif.
        X_new: Nouvelles cellules (n_new, n_markers) — mêmes marqueurs que fit().
        output_path: Chemin image.
        view: "MST" ou "grid".

    Returns:
        Chemin image ou None.
    """
    if not _guard():
        return None
    model = _resolve_fsom(fsom)
    if model is None:
        return None
    try:
        import anndata as ad

        adata_new = ad.AnnData(np.asarray(X_new, dtype=np.float32))
        # Aligner var_names sur le modèle d'origine (sinon marqueurs introuvables).
        ref_names = getattr(fsom, "marker_names_", None)
        if ref_names is None:
            try:
                ref_names = list(model.get_cell_data().var_names)
            except Exception:  # noqa: BLE001
                ref_names = None
        if ref_names and len(ref_names) == adata_new.shape[1]:
            adata_new.var_names = ref_names
        fsom_new = model.new_data(adata_new)
        return plot_stars_native(fsom_new, output_path, view=view, dpi=dpi)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("plot_new_data_stars échoué : %s", exc)
        return None


# ---------------------------------------------------------------------------
# subset — sous-FlowSOM par métacluster
# ---------------------------------------------------------------------------

def plot_subset_stars(
    fsom: Any,
    metacluster_id: int,
    output_path: Path | str,
    *,
    view: str = "MST",
    dpi: int = 150,
) -> Optional[Path]:
    """Isole un métacluster et trace son Star Chart.

    Reproduit la documentation officielle ::

        fsom_subset = fsom.subset(
            fsom.get_cell_data().obs["metaclustering"] == 2)
        fs.pl.plot_stars(fsom_subset,
            background_values=fsom_subset.get_cluster_data().obs.metaclustering)

    Args:
        fsom: FlowSOMClusterer (entraîné) ou objet fs.FlowSOM natif.
        metacluster_id: Métacluster à isoler.
        output_path: Chemin image.
        view: "MST" ou "grid".

    Returns:
        Chemin image ou None.
    """
    if not _guard():
        return None
    model = _resolve_fsom(fsom)
    if model is None:
        return None
    try:
        mask = model.get_cell_data().obs["metaclustering"] == metacluster_id
        fsom_subset = model.subset(mask)
        return plot_stars_native(fsom_subset, output_path, view=view, dpi=dpi)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("plot_subset_stars(mc=%s) échoué : %s", metacluster_id, exc)
        return None


# ---------------------------------------------------------------------------
# Orchestration — génère toutes les sorties natives en un appel
# ---------------------------------------------------------------------------

def generate_all_native_plots(
    fsom: Any,
    output_dir: Path | str,
    *,
    marker_names: Optional[Sequence[str]] = None,
    fmt: str = "png",
    dpi: int = 150,
    prefix: str = "flowsom",
) -> dict:
    """Génère l'ensemble des visualisations natives FlowSOM dans un dossier.

    Args:
        fsom: FlowSOMClusterer entraîné ou objet fs.FlowSOM natif.
        output_dir: Dossier de sortie.
        marker_names: Marqueurs pour les cartes par marqueur (None = aucune carte marqueur).
        fmt: Extension image ("png", "jpg", "svg", "pdf").
        dpi: Résolution.
        prefix: Préfixe des noms de fichiers.

    Returns:
        Dict {nom_logique: chemin} des images réellement produites.
    """
    produced: dict = {}
    model = _resolve_fsom(fsom)
    if model is None:
        _logger.info("generate_all_native_plots : pas d'objet FlowSOM natif — sorties natives ignorées.")
        return produced

    out = Path(output_dir)
    ext = fmt if fmt.startswith(".") else f".{fmt}"

    # Star charts — MST + grille (avec et sans tailles égales)
    p = plot_stars_native(model, out / f"{prefix}_stars_MST{ext}", view="MST", dpi=dpi)
    if p:
        produced["stars_mst"] = str(p)
    p = plot_stars_native(
        model, out / f"{prefix}_stars_grid{ext}", view="grid",
        equal_node_size=True, equal_background_size=True, dpi=dpi,
    )
    if p:
        produced["stars_grid"] = str(p)

    # Numéros des nœuds + métaclusters
    p = plot_numbers_native(model, out / f"{prefix}_numbers_clusters{ext}", level="clusters", dpi=dpi)
    if p:
        produced["numbers_clusters"] = str(p)
    p = plot_numbers_native(model, out / f"{prefix}_numbers_metaclusters{ext}", level="metaclusters", dpi=dpi)
    if p:
        produced["numbers_metaclusters"] = str(p)

    # Cartes par marqueur (limitées aux marqueurs fournis)
    if marker_names:
        for mk in marker_names:
            safe = "".join(c if c.isalnum() else "_" for c in str(mk))
            p = plot_marker_native(model, mk, out / f"{prefix}_marker_{safe}{ext}", dpi=dpi)
            if p:
                produced[f"marker_{safe}"] = str(p)

    # Planche de synthèse (PDF par défaut, robuste)
    p = plot_flowsommary_native(model, out / f"{prefix}_summary.pdf", dpi=dpi)
    if p:
        produced["flowsommary"] = str(p)

    _logger.info("generate_all_native_plots : %d visualisations natives produites.", len(produced))
    return produced
