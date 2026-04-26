"""
preprocessing_service.py — Orchestration du prétraitement des échantillons FCS.

Chaîne de traitement complète pour un FlowSample:
  1. Contrôle QC (NaN, cellules minimales, scatter exclusion)
  2. Sous-échantillonnage (si configuré)
  3. Gating (auto ou manuel) selon PipelineConfig
  4. Transformation cytométrique (logicle, arcsinh)
  5. Normalisation (zscore, minmax)

Retourne une liste de FlowSample prêts pour le clustering FlowSOM.
"""

from __future__ import annotations

import warnings
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import anndata as ad

    _ANNDATA_AVAILABLE = True
except ImportError:
    _ANNDATA_AVAILABLE = False

from config.pipeline_config import PipelineConfig
from src.prisma.exceptions import TooFewCellsError, NaNDataError, GatingRejectedError, ProcessingError
from src.prisma.core.models_legacy.sample import FlowSample
from src.prisma.core.transformers import DataTransformer
from src.prisma.core.normalizers import DataNormalizer
from src.prisma.core.gating import PreGating
from src.prisma.core.auto_gating import AutoGating
from src.prisma.utils.logger import GatingLogger, get_logger
from src.prisma.utils.marker_harmonizer import harmonize_marker_names
from src.prisma.utils.validators import (
    check_nan,
    check_min_cells,
    check_no_fsc_ssc_in_analysis_markers,
)

_logger = get_logger("services.preprocessing")


def _canonical_marker_name(raw_name: str) -> str:
    """
    Extrait un nom canonique suffix-aware pour l'harmonisation inter-fichiers.

    Exemples:
      - "CD7+56 FITC-A" -> "CD7+56-A"
      - "CD7+56 FITC-H" -> "CD7+56-H"
      - "CD34 Cy55"     -> "CD34"
      - "FSC-A"         -> "FSC-A"
    """
    raw = (raw_name or "").strip()
    if not raw:
        return raw

    base = raw.split(" ")[0]
    m = re.search(r"-(A|H)\b", raw, re.IGNORECASE)
    if m and not (base.upper().endswith("-A") or base.upper().endswith("-H")):
        return f"{base}-{m.group(1).upper()}"
    return base


def preprocess_sample(
    sample: FlowSample,
    config: PipelineConfig,
    gating_logger: Optional[GatingLogger] = None,
) -> FlowSample:
    """
    Applique la chaîne complète de prétraitement à un FlowSample.

    Args:
        sample: Échantillon FCS chargé.
        config: Configuration du pipeline.
        gating_logger: Logger de gating (optionnel).

    Returns:
        FlowSample prétraité.

    Raises:
        TooFewCellsError:    Moins de 100 cellules avant ou après gating.
        NaNDataError:        NaN détectés (loggés et imputés à 0, exception
                             levée uniquement si toutes les valeurs sont NaN).
        GatingRejectedError: Un gate élimine trop de cellules.
        ProcessingError:     Toute autre erreur inattendue du pipeline.
    """
    if gating_logger is None:
        gating_logger = GatingLogger()

    X = sample.matrix
    var_names = sample.var_names
    file_name = sample.name
    n_original = X.shape[0]

    _logger.info("Prétraitement: %s (%d cellules)", file_name, n_original)

    # ── 1. Contrôle QC ────────────────────────────────────────────────────────
    n_nan = check_nan(X)
    if n_nan > 0:
        # Imputation à 0 tracée explicitement dans les logs et dans l'exception
        # pour audit RUO. On lève NaNDataError uniquement si toutes les valeurs
        # sont NaN (données inutilisables). Sinon on impute et on continue.
        if n_nan == X.size:
            raise NaNDataError(file_name, n_nan, action="rejected_all_nan")
        _logger.warning(
            "%s: %d NaN détectés sur %d valeurs (%.1f%%) — imputation à 0 [audit trail]",
            file_name, n_nan, X.size, n_nan / X.size * 100,
        )
        X = np.nan_to_num(X, nan=0.0)
        # Lever l'exception en mode avertissement (non bloquant) pour que les
        # couches supérieures puissent décider de rejeter ou d'accepter.
        raise NaNDataError(file_name, n_nan, action="imputed_zero")

    if not check_min_cells(X.shape[0], min_cells=100):
        raise TooFewCellsError(file_name, n_cells=X.shape[0], min_cells=100, stage="QC")

    # ── 2. Sous-échantillonnage initial ───────────────────────────────────────
    max_cells = config.downsampling.max_cells_per_file
    if max_cells and max_cells > 0 and X.shape[0] > max_cells:
        rng = np.random.default_rng(config.flowsom.seed)
        idx = rng.choice(X.shape[0], size=max_cells, replace=False)
        X = X[idx]
        _logger.info(
            "%s: sous-échantillonné à %d/%d cellules", file_name, max_cells, n_original
        )

    # ── 3. Gating ─────────────────────────────────────────────────────────────
    pregate_cfg = config.pregate
    condition = getattr(sample, "condition", "Unknown") or "Unknown"
    if pregate_cfg.apply:
        X, var_names, mask_combined = _apply_gating(
            X,
            var_names,
            pregate_cfg,
            gating_logger,
            file_name,
            condition=condition,
        )
        if X.shape[0] < 100:
            raise TooFewCellsError(file_name, n_cells=X.shape[0], min_cells=100, stage="post-gating")

    # ── 3b. Filtrage -A/-H (déduplication marqueurs Area vs Height) ───────────
    keep_area = getattr(config.markers, "keep_area_only", False)
    if keep_area:
        X, var_names = _filter_area_markers(X, var_names)
        _logger.info(
            "%s: filtrage -A uniquement → %d marqueurs", file_name, len(var_names)
        )

    # ── 4+5. Transformation cytométrique + Normalisation ──────────────────────
    X = _apply_transforms(X, var_names, config)

    # ── Reconstruire le FlowSample avec les données traitées ─────────────────
    import pandas as _pd

    processed = FlowSample(
        name=sample.name,
        path=sample.path,
        condition=sample.condition,
        data=_pd.DataFrame(X, columns=var_names),
        metadata={
            **sample.metadata,
            "preprocessed": True,
            "n_original_cells": n_original,
        },
        n_cells_raw=n_original,
    )

    _logger.info(
        "%s: %d → %d cellules après prétraitement",
        file_name,
        n_original,
        X.shape[0],
    )
    return processed


def _apply_transforms(
    X: np.ndarray,
    var_names: List[str],
    config: "PipelineConfig",
) -> np.ndarray:
    """Applique la transformation cytométrique et la normalisation sur X."""
    transform_cfg = config.transform
    if transform_cfg.method != "none":
        X = DataTransformer.apply(
            X,
            method=transform_cfg.method,
            cofactor=transform_cfg.cofactor,
            var_names=var_names,
            apply_to_scatter=transform_cfg.apply_to_scatter,
        )
    normalize_cfg = config.normalize
    if normalize_cfg.method != "none":
        X = DataNormalizer.apply(X, method=normalize_cfg.method)
    return X


def _filter_area_markers(
    X: np.ndarray,
    var_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Supprime les marqueurs -H (Height) UNIQUEMENT quand le doublon -A (Area) existe.

    Les instruments BD et Beckman Coulter génèrent systématiquement des paires
    Marqueur-A / Marqueur-H. Garder les deux introduit de la colinéarité dans
    le SOM. On ne conserve que -A + tout marqueur sans suffixe -A/-H
    (ex: FSC-Width, Time). Un marqueur -H sans -A correspondant est conservé.

    La correspondance est détectée par le préfixe avant le suffixe -A/-H.
    Exemple: « CD13 PE-A » et « CD13 PE-H » → seul « CD13 PE-A » est conservé.
             « FSC-H » sans « FSC-A » → « FSC-H » est conservé.

    Returns:
        Tuple (X_filtré, var_names_filtrés).
    """
    upper_names = [n.upper() for n in var_names]
    # Construire l'ensemble des préfixes des marqueurs -A existants
    area_prefixes: set = set()
    for name_u in upper_names:
        if name_u.endswith("-A"):
            area_prefixes.add(name_u[:-2])  # ex: "CD13 PE"

    cols_keep: List[int] = []
    for i, name_u in enumerate(upper_names):
        if name_u.endswith("-H"):
            prefix = name_u[:-2]
            if prefix in area_prefixes:
                # Le doublon -A existe → supprimer ce -H
                _logger.debug(
                    "Marqueur %s supprimé (doublon -A existant)", var_names[i]
                )
                continue
        cols_keep.append(i)

    if not cols_keep:
        # Aucun marqueur à conserver (cas pathologique) — retourner intact
        return X, list(var_names)
    if len(cols_keep) == len(var_names):
        # Aucun marqueur -H supprimé — retourner intact
        return X, list(var_names)

    n_removed = len(var_names) - len(cols_keep)
    _logger.info(
        "Filtrage -A/-H : %d marqueurs -H supprimés (doublon -A existant)",
        n_removed,
    )
    return X[:, cols_keep], [var_names[i] for i in cols_keep]


def _apply_gating(
    X: np.ndarray,
    var_names: List[str],
    pregate_cfg: object,
    gating_logger: GatingLogger,
    file_name: str,
    condition: str = "Unknown",
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Applique le gating (auto ou manuel) sur la matrice.

    Gating :
    - G1 (débris) et G2 (singlets) : appliqués à toutes les conditions.
    - G3 (CD45) : SYMÉTRIQUE — appliqué à toutes les conditions, y compris Sain/NBM.
      Le KDE calcule le seuil sur la distribution de l'échantillon en cours.
      Cela garantit que le pool NBM est strictement composé de leucocytes CD45+.
    - G4 (CD34) : asymétrique en mode_blastes_vs_normal — ignoré pour les sains
      afin de ne pas exclure les progéniteurs CD34+ normaux.

    Args:
        condition: "Sain", "Pathologique" ou autre. Case-insensitive.

    Returns:
        Tuple (X_gated, var_names, combined_mask).
    """
    mode = getattr(pregate_cfg, "mode", "auto")
    mode_blastes_vs_normal = getattr(pregate_cfg, "mode_blastes_vs_normal", False)
    n_before = X.shape[0]
    combined_mask = np.ones(n_before, dtype=bool)

    # is_sain est conservé pour le gating CD34 (asymétrique) uniquement.
    is_sain = mode_blastes_vs_normal and condition.lower() in (
        "sain",
        "normal",
        "healthy",
        "nbm",
        "moelle normale",
    )

    # Gate 1 – Débris
    if getattr(pregate_cfg, "viable", True):
        if mode == "auto":
            mask = AutoGating.auto_gate_debris(
                X,
                var_names,
                n_components=getattr(pregate_cfg, "gmm_n_components_debris", 3),
                gmm_max_samples=getattr(pregate_cfg, "gmm_max_samples", 50000),
                covariance_type=getattr(pregate_cfg, "gmm_covariance_type", "full"),
                density_method=getattr(pregate_cfg, "density_method", "GMM"),
            )
        else:
            mask = PreGating.gate_viable_cells(
                X,
                var_names,
                min_percentile=getattr(pregate_cfg, "debris_min_percentile", 2.0),
                max_percentile=getattr(pregate_cfg, "debris_max_percentile", 99.0),
            )
        gating_logger.log(
            file_name, "G1_debris", n_before, int(mask.sum()), method=mode
        )
        combined_mask &= mask

    # Gate 2 – Singlets
    if getattr(pregate_cfg, "singlets", True):
        if mode == "auto":
            mask = AutoGating.auto_gate_singlets(
                X,
                var_names,
                r2_threshold=getattr(pregate_cfg, "ransac_r2_threshold", 0.85),
                mad_factor=getattr(pregate_cfg, "ransac_mad_factor", 3.0),
            )
        else:
            mask = PreGating.gate_singlets(
                X,
                var_names,
                ratio_min=getattr(pregate_cfg, "singlet_ratio_min", 0.6),
                ratio_max=getattr(pregate_cfg, "singlet_ratio_max", 1.5),
            )
        gating_logger.log(
            file_name,
            "G2_singlets",
            int(combined_mask.sum()),
            int((combined_mask & mask).sum()),
            method=mode,
        )
        combined_mask &= mask

    # Gate 3 – CD45 (SYMÉTRIQUE : appliqué à toutes les conditions, y compris Sain/NBM)
    # Le pool NBM doit être strictement composé de leucocytes CD45+.
    # Le KDE calcule le seuil sur la distribution de l'échantillon en cours (sain ou patho).
    if getattr(pregate_cfg, "cd45", True):
        if mode == "auto":
            mask = AutoGating.auto_gate_cd45(
                X,
                var_names,
                kde_seuil_relatif=getattr(pregate_cfg, "kde_cd45_seuil_relatif", 0.05),
                kde_finesse=getattr(pregate_cfg, "kde_cd45_finesse", 0.6),
                kde_sigma_smooth=getattr(pregate_cfg, "kde_cd45_sigma_smooth", 10),
                kde_n_grid=getattr(pregate_cfg, "kde_cd45_n_grid", 1000),
                kde_max_samples=getattr(pregate_cfg, "kde_cd45_max_samples", 10000),
                threshold_percentile=getattr(
                    pregate_cfg, "cd45_threshold_percentile", 5.0
                ),
            )
        else:
            mask = PreGating.gate_cd45_positive(
                X,
                var_names,
                threshold_percentile=getattr(pregate_cfg, "cd45_min_percentile", 5.0),
            )
        _logger.info(
            "%s [%s]: G3_cd45 appliqué (gating symétrique — leucocytes CD45+ uniquement)",
            file_name,
            condition,
        )
        gating_logger.log(
            file_name,
            "G3_cd45",
            int(combined_mask.sum()),
            int((combined_mask & mask).sum()),
            method=mode,
        )
        combined_mask &= mask

    # Gate 4 – CD34+ blastes (optionnel, ASYMÉTRIQUE : ignoré si échantillon sain)
    if getattr(pregate_cfg, "cd34", False):
        if is_sain:
            _logger.info(
                "%s [%s]: G4_cd34 ignoré (gating asymétrique)",
                file_name,
                condition,
            )
            gating_logger.log(
                file_name,
                "G4_cd34",
                int(combined_mask.sum()),
                int(combined_mask.sum()),
                method="skip_asymmetric",
            )
        else:
            if mode == "auto":
                mask = AutoGating.auto_gate_cd34(X, var_names)
            else:
                mask = PreGating.gate_cd34_blasts(
                    X,
                    var_names,
                    threshold_percentile=getattr(
                        pregate_cfg, "cd34_min_percentile", 70.0
                    ),
                    ssc_max_percentile=getattr(pregate_cfg, "cd34_ssc_max", 40.0),
                )
            gating_logger.log(
                file_name,
                "G4_cd34",
                int(combined_mask.sum()),
                int((combined_mask & mask).sum()),
                method=mode,
            )
            combined_mask &= mask

    X_gated = X[combined_mask]

    _logger.info(
        "%s [%s]: gating %s → %d/%d cellules conservées (%.1f%%)",
        file_name,
        condition,
        mode,
        X_gated.shape[0],
        n_before,
        X_gated.shape[0] / max(n_before, 1) * 100,
    )

    return X_gated, var_names, combined_mask


def _safe_preprocess_sample(
    sample: FlowSample,
    config: PipelineConfig,
    gating_logger: Optional[GatingLogger] = None,
) -> Optional[FlowSample]:
    """
    Wrapper de preprocess_sample qui absorbe TooFewCellsError et NaNDataError
    (imputation uniquement) pour le mode batch tolérant.

    Les erreurs sont loggées comme avertissements — l'échantillon est ignoré
    sans bloquer le pipeline complet. ProcessingError et GatingError remontent.
    """
    from src.prisma.exceptions import TooFewCellsError, NaNDataError

    try:
        return preprocess_sample(sample, config, gating_logger)
    except NaNDataError as exc:
        if exc.action == "imputed_zero":
            # Imputation déjà appliquée dans preprocess_sample avant l'exception.
            # On relance avec X déjà corrigé : reconstruire le sample patchié.
            _logger.warning(
                "%s: NaN imputés à 0 (%d valeurs) — échantillon conservé [audit trail]",
                exc.sample_name, exc.n_nan,
            )
            # Reconstruire le FlowSample avec la matrice patchée
            import numpy as _np
            import pandas as _pd
            X_clean = _np.nan_to_num(sample.matrix, nan=0.0)
            patched = type(sample)(
                name=sample.name,
                path=sample.path,
                condition=sample.condition,
                data=_pd.DataFrame(X_clean, columns=sample.var_names),
                metadata={**sample.metadata, "nan_imputed": exc.n_nan},
                n_cells_raw=sample.n_cells_raw,
            )
            try:
                return preprocess_sample(patched, config, gating_logger)
            except TooFewCellsError as e2:
                _logger.warning("Rejeté après imputation NaN: %s", e2)
                return None
        # action == "rejected_all_nan" → ignorer
        _logger.warning("Rejeté (toutes valeurs NaN): %s", exc)
        return None
    except TooFewCellsError as exc:
        _logger.warning("Rejeté: %s", exc)
        return None


def preprocess_all_samples(
    samples: List[FlowSample],
    config: PipelineConfig,
    gating_logger: Optional[GatingLogger] = None,
) -> List[FlowSample]:
    """
    Prétraite une liste complète de FlowSample en parallèle (joblib).

    Chaque échantillon est traité dans un worker indépendant pour exploiter
    tous les cœurs disponibles. Le GatingLogger partagé est ignoré en mode
    parallèle (chaque worker instancie le sien) pour éviter les race-conditions.

    Les échantillons rejetés (TooFewCellsError, NaN complets) sont logués comme
    avertissements et ignorés. Les erreurs non récupérables (ProcessingError,
    GatingError) remontent au caller.

    Args:
        samples: Liste des échantillons à traiter.
        config: Configuration du pipeline.
        gating_logger: Logger de gating partagé (optionnel, ignoré si n_jobs > 1).

    Returns:
        Liste des échantillons prétraités, dans l'ordre d'origine.
    """
    try:
        from joblib import Parallel, delayed

        _use_parallel = True
    except ImportError:
        _use_parallel = False

    n_samples = len(samples)

    if _use_parallel and n_samples > 1:
        import os

        # Multiprocessing (loky) : chaque worker a sa propre instance Matplotlib/Pandas
        # → contourne le GIL et évite le lock du font_manager Matplotlib.
        # Bridé à 4 workers max pour limiter la consommation RAM (chaque process
        # copie les DataFrames via pickle).
        n_jobs = min(4, os.cpu_count() or 1)
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_safe_preprocess_sample)(s, config, GatingLogger()) for s in samples
        )
    else:
        if gating_logger is None:
            gating_logger = GatingLogger()
        results = [
            _safe_preprocess_sample(s, config, gating_logger=gating_logger) for s in samples
        ]

    processed = [r for r in results if r is not None]

    _logger.info(
        "Prétraitement terminé: %d/%d échantillons",
        len(processed),
        n_samples,
    )
    return processed


# =============================================================================
# PRÉTRAITEMENT COMBINÉ — Reproduit fidèlement flowsom_pipeline.py
# =============================================================================
# Dans le monolithe, le gating est appliqué sur la CONCATÉNATION de tous les
# fichiers avant toute autre opération. Cela est crucial car :
#  - Le KDE CD45 (pied du pic) est calculé séparément par condition (sain/patho) :
#    chaque sous-population obtient un seuil adapté à sa propre distribution.
#    Le gating CD45 est SYMÉTRIQUE : les cellules saines sont également filtrées
#    → le pool NBM contient uniquement des leucocytes CD45+, médiane correcte.
#  - Le RANSAC singlets est exécuté par fichier sur l'ensemble combiné.
#  - Il n'y a AUCUN sous-échantillonnage avant le gating.
# =============================================================================


def preprocess_combined(
    samples: List[FlowSample],
    config: "PipelineConfig",
    gating_logger: Optional[GatingLogger] = None,
    gating_plot_dir: "Optional[Path]" = None,
) -> Tuple[List[FlowSample], Dict[str, Any], Dict[str, Any]]:
    """
    Reproduit fidèlement flowsom_pipeline.py.

    Contrairement à ``preprocess_all_samples`` (qui traite chaque fichier
    indépendamment), cette fonction:

    1. Calcule l'intersection des marqueurs communs (équivalent join='inner').
    2. Empile toutes les cellules brutes en une seule matrice.
    3. Applique le gating sur les données **combinées** :
       - Gate débris (GMM 2D FSC-A/SSC-A) sur toutes les cellules.
       - Gate singlets (RANSAC) par fichier sur les données combinées.
       - Gate CD45 (si activé) : KDE pied du pic SYMÉTRIQUE — appliqué à toutes
         les conditions. En mode_blastes_vs_normal, le KDE est calculé séparément
         par sous-population (sain/patho) pour calibrer le seuil sur chaque
         distribution. Le pool NBM contiendra uniquement des leucocytes CD45+.
       - Gate CD34 (si activé) : asymétrique — ignoré pour les sains afin de
         conserver les progéniteurs CD34+ normaux.
    4. Filtre les marqueurs -H redondants (keep_area_only).
    5. Applique la transformation et la normalisation sur l'ensemble.
    6. Reconstruit la liste de :class:`FlowSample` par fichier d'origine.
       **AUCUN sous-échantillonnage avant FlowSOM** (identique au monolithe).
       Le sous-échantillonnage bootstrap (20 000 cellules) n'est utilisé que
       dans la phase 2 de l'auto-clustering.

    Args:
        samples: Échantillons bruts (non prétraités).
        config: Configuration du pipeline.
        gating_logger: Logger de gating partagé (optionnel).

    Returns:
        Tuple (
            liste FlowSample prétraités,
            dict figures matplotlib gating,
            dict de checkpoint gating (masques + métadonnées),
        ).
    """
    import pandas as _pd

    if gating_logger is None:
        gating_logger = GatingLogger()

    if not samples:
        return [], {}, {}

    # ── 1. Harmonisation des noms de marqueurs + intersection ─────────────────
    #
    # Problème : "$PnS" instable selon l'export — "CD34 Cy55" vs "CD34",
    # "CD13 BV421" vs "CD13". Sans harmonisation, l'intersection exclut ces
    # marqueurs alors qu'ils sont présents dans tous les fichiers.
    #
    # Solution :
    #   a) Extraire les noms canoniques (partie avant le premier espace) depuis
    #      l'union de tous les var_names pour construire la liste cible.
    #   b) Harmoniser chaque sample EN PLACE avant le calcul d'intersection.

    # a) Liste cible canonique = union des préfixes de tous les marqueurs
    all_raw: List[str] = []
    for s in samples:
        all_raw.extend(s.var_names)
    # Dédupliquer en préservant l'ordre d'apparition (boucle explicite, pas de
    # side-effect dans comprehension qui repose sur add() retournant None).
    seen_raw: set = set()
    all_raw_unique: List[str] = []
    for _m in all_raw:
        if _m not in seen_raw:
            seen_raw.add(_m)
            all_raw_unique.append(_m)

    # Extraire le préfixe canonique (suffix-aware pour -A/-H) de chaque nom brut
    canonical_set: dict = {}  # canonical → premier nom brut rencontré
    for raw in all_raw_unique:
        canon = _canonical_marker_name(raw)
        if canon not in canonical_set:
            canonical_set[canon] = raw
    target_markers: List[str] = list(canonical_set.keys())

    # b) Harmoniser chaque sample EN PLACE
    n_harmonized_total = 0
    # Cache par schéma de colonnes pour éviter de recalculer le regex mapping
    # sur des panels identiques (cas typique en batch NBM/patho).
    _rename_cache: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for s in samples:
        schema_key = tuple(s.var_names)
        if schema_key in _rename_cache:
            rename_map = _rename_cache[schema_key]
        else:
            rename_map = harmonize_marker_names(s.var_names, target_markers)
            _rename_cache[schema_key] = rename_map
        renames = {src: dst for src, dst in rename_map.items() if src != dst}
        if renames:
            # Assignment (pas inplace) pour ne pas muter un DataFrame potentiellement
            # partagé entre plusieurs références (ex: cache réutilisé).
            s.data = s.data.rename(columns=renames)
            n_harmonized_total += len(renames)
            _logger.debug(
                "%s : %d colonne(s) harmonisée(s) : %s",
                s.name,
                len(renames),
                renames,
            )

    if n_harmonized_total:
        _logger.info(
            "Harmonisation panel : %d renommage(s) effectué(s) sur %d fichiers.",
            n_harmonized_total,
            len(samples),
        )

    # Intersection sur les noms harmonisés (dédupliqué pour éviter les colonnes
    # en double héritées de fichiers FCS avec des marqueurs répétés, ex: FSC-Width, Time)
    _seen_cm: set = set()
    common_markers: List[str] = []
    for m in samples[0].var_names:
        if m not in _seen_cm:
            _seen_cm.add(m)
            common_markers.append(m)
    for s in samples[1:]:
        s_set = set(s.var_names)
        common_markers = [m for m in common_markers if m in s_set]

    if not common_markers:
        _logger.error("Aucun marqueur commun entre tous les fichiers !")
        return [], {}, {}

    _logger.info("Panel commun (intersection): %d marqueurs", len(common_markers))
    _logger.info(
        "Marqueurs retenus dans le panel commun : %s",
        common_markers,
    )

    missing_summary: dict = {}
    for s in samples:
        # Signaler uniquement les marqueurs non-techniques réellement absents
        missing = [m for m in s.var_names if m not in common_markers]
        if missing:
            missing_summary[s.name] = missing
    if missing_summary:
        for fname, miss in missing_summary.items():
            _logger.warning(
                "%s: %d marqueur(s) exclus du panel commun: %s",
                fname,
                len(miss),
                miss,
            )

    # ── 2. Empilage des données brutes avec tracking ───────────────────────────
    all_X: List[np.ndarray] = []
    all_conditions: List[str] = []
    all_file_origins: List[str] = []
    # Ordre canonique des samples pour reconstruire les FlowSample après gating
    sample_order: List[FlowSample] = []

    for s in samples:
        var_s = s.var_names
        X_s = s.matrix
        col_idx = [var_s.index(m) for m in common_markers]
        # copy=True : évite le partage de buffer mémoire avec sample.matrix
        # quand X_s est déjà en float32 (aliasing silencieux sinon).
        X_sel = X_s[:, col_idx].astype(np.float32, copy=True)
        n = X_sel.shape[0]
        all_X.append(X_sel)
        all_conditions.extend([s.condition] * n)
        all_file_origins.extend([s.name] * n)
        sample_order.append(s)
        _logger.debug("%s: %d cellules empilées", s.name, n)

    X_raw = np.vstack(all_X)
    conditions = np.array(all_conditions)
    file_origins = np.array(all_file_origins)
    n_total_raw = X_raw.shape[0]
    var_names: List[str] = list(common_markers)

    _logger.info(
        "Données combinées: %d cellules × %d marqueurs (%d fichiers)",
        n_total_raw,
        len(var_names),
        len(samples),
    )
    for s in sample_order:
        _logger.info(
            "  %s [%s]: %d cellules",
            s.name,
            s.condition,
            int((file_origins == s.name).sum()),
        )

    # ── QC NaN sur les données brutes combinées ───────────────────────────────
    n_nan = check_nan(X_raw)
    if n_nan > 0:
        _logger.warning("%d NaN détectés — imputation à 0", n_nan)
        X_raw = np.nan_to_num(X_raw, nan=0.0)

    # ── 3. Gating sur données combinées ───────────────────────────────────────
    pregate_cfg = config.pregate
    gate_masks: Dict[str, np.ndarray] = {}
    _combined_mask: Optional[np.ndarray] = None  # masque final gating (n_total_raw)
    X_for_plot = X_raw  # Référence brute avant gating (évite une copie massive)
    var_for_plot: List[str] = list(var_names)
    conditions_for_plot = conditions  # Référence avant gating (évite copie objet)
    if getattr(pregate_cfg, "apply", True):
        # Chemin pour le graphique GMM si demandé
        _gmm_plot_path = None
        if (
            gating_plot_dir is not None
            and getattr(pregate_cfg, "gmm_export_plot", False)
            and getattr(pregate_cfg, "density_method", "GMM").upper() == "GMM"
        ):
            from pathlib import Path as _Path

            _gmm_plot_path = str(
                _Path(gating_plot_dir) / "gating" / "gmm_debris_density.png"
            )
        X_raw, var_names, conditions, file_origins, gate_masks, _combined_mask = (
            _apply_gating_combined(
                X_raw,
                var_names,
                conditions,
                file_origins,
                pregate_cfg,
                gating_logger,
                gmm_plot_path=_gmm_plot_path,
            )
        )

    # ── Plots QC de gating (si plot_dir fourni) ───────────────────────────────
    gating_figures: Dict[str, Any] = {}
    if gating_plot_dir is not None and gate_masks:
        try:
            from src.prisma.visualization.gating_plots import (
                generate_all_gating_plots,
            )

            gating_plot_dir = Path(gating_plot_dir)
            _fig_results = generate_all_gating_plots(
                X_for_plot,
                var_for_plot,
                gate_masks,
                output_dir=gating_plot_dir / "gating",
                sample_name="combined",
                conditions=conditions_for_plot,
            )
            # Renommer les figures selon les labels notebook
            _fig_map = {
                "01_overview": "fig_overview",
                "02_debris": "fig_gate_debris",
                "03_singlets": "fig_gate_singlets",
                "04_cd45": "fig_gate_cd45",
                "05_cd34": "fig_gate_cd34",
                "06_kde_debris": "fig_kde_debris",
                "07_kde_cd45": "fig_kde_cd45",
            }
            for _key, _fig in _fig_results.items():
                if _fig is not None:
                    label = _fig_map.get(_key, f"fig_gate_{_key}")
                    gating_figures[label] = _fig
            _logger.info(
                "Plots QC gating générés: %d figures dans %s",
                len(gating_figures),
                gating_plot_dir / "gating",
            )
        except Exception as _e:
            _logger.warning("Plots QC gating échoués (non bloquant): %s", _e)

    n_after_gate = X_raw.shape[0]
    _logger.info(
        "Après gating combiné: %d/%d cellules (%.1f%%)",
        n_after_gate,
        n_total_raw,
        n_after_gate / max(n_total_raw, 1) * 100,
    )

    # ── 4. Filtrage -A/-H (keep_area_only) ───────────────────────────────────
    keep_area = getattr(config.markers, "keep_area_only", False)
    if keep_area:
        X_raw, var_names = _filter_area_markers(X_raw, var_names)
        _logger.info("Filtrage -A uniquement: %d marqueurs conservés", len(var_names))

    # ── Save pré-transformation — pour export FCS Kaluza (valeurs linéaires) ─
    # Les données ici sont après gating + filtre -A/-H, mais AVANT toute
    # transformation (logicle/arcsinh) et normalisation. C'est exactement ce
    # qu'attend Kaluza pour afficher les données dans l'espace d'intensité.
    X_pre_transform = (
        X_raw if X_raw.dtype == np.float32 else X_raw.astype(np.float32, copy=True)
    )
    var_names_pre_transform: List[str] = list(var_names)

    # ── 5+6. Transformation cytométrique + Normalisation ──────────────────────
    X_raw = _apply_transforms(X_raw, var_names, config)

    # Données transformées pour le plot KDE CD45 (espace logicle/arcsinh).
    # On utilise un alias séparé pour ne pas écraser X_for_plot pré-gating
    # (utilisé par generate_all_gating_plots avec gate_masks de taille n_total_raw).
    X_transformed_for_kde = X_raw

    # ── Plot KDE CD45 sur données transformées (logicle/arcsinh) ─────────────
    # Ce plot doit être généré APRÈS la transformation pour afficher le CD45
    # dans l'espace logicle (comme dans le notebook .ipynb), et non en linéaire.
    # EXPORT SYSTÉMATIQUE : le KDE CD45 est toujours exporté, même si CD45 n'est
    # pas sélectionné pour le tri cellulaire.
    #
    # Les masques dans gate_masks ont n_total_raw lignes. X_for_plot en a n_gated.
    # On projette G3_cd45[_combined_mask] pour obtenir le masque dans l'espace
    # post-gating sans recalculer le KDE (rapide, pas de double-calcul).
    if gating_plot_dir is not None:
        try:
            from src.prisma.visualization.gating_plots import (
                plot_cd45_kde_qc as _plot_cd45_kde,
            )
            from src.prisma.core.gating import PreGating as _PG

            _cd45_idx = _PG.find_marker_index(
                var_names, ["CD45", "CD45-PECY5", "CD45-PC5"]
            )
            if _cd45_idx is not None:
                # Utiliser la population qui ENTRE dans Gate 3 (après G1+G2), pas
                # la population finale post-gating, sinon les CD45- exclus par G3
                # disparaissent artificiellement du QC KDE.
                _raw_g3 = gate_masks.get(
                    "G3_cd45", gate_masks.get("G3", gate_masks.get("cd45"))
                )
                _raw_g1 = gate_masks.get("G1_debris", gate_masks.get("G1"))
                _raw_g2 = gate_masks.get("G2_singlets", gate_masks.get("G2"))

                if _raw_g3 is not None:
                    _pre_g3_mask = np.ones(len(X_for_plot), dtype=bool)
                    if _raw_g1 is not None:
                        _pre_g3_mask &= _raw_g1
                    if _raw_g2 is not None:
                        _pre_g3_mask &= _raw_g2

                    _is_patho_global = conditions_for_plot == "Pathologique"
                    if not bool(_is_patho_global.any()):
                        _patho_labels = {"pathologique", "patho", "aml", "disease"}
                        _unique_cond = np.unique(conditions_for_plot)
                        _patho_values = [
                            c for c in _unique_cond if str(c).lower() in _patho_labels
                        ]
                        _is_patho_global = np.isin(conditions_for_plot, _patho_values)

                    _plot_mask_global = _pre_g3_mask & _is_patho_global
                    _n_plot = int(_plot_mask_global.sum())

                    if _n_plot >= 5:
                        _cd45_name = var_for_plot[_cd45_idx]
                        _cd45_raw_col = X_for_plot[
                            _plot_mask_global, _cd45_idx
                        ].reshape(-1, 1)
                        _cd45_data_plot = _apply_transforms(
                            _cd45_raw_col,
                            [_cd45_name],
                            config,
                        ).ravel()
                        _mask_g3_plot = _raw_g3[_plot_mask_global]
                        _logger.info(
                            "Plot KDE CD45 : %d cellules pathologiques (pré-G3) utilisées",
                            _n_plot,
                        )
                    else:
                        # Fallback défensif: conserver l'ancien comportement
                        _cd45_data_plot = X_transformed_for_kde[:, _cd45_idx]
                        _mask_g3_plot = np.ones(
                            X_transformed_for_kde.shape[0], dtype=bool
                        )
                        _logger.info(
                            "Plot KDE CD45 : fallback post-gating (pré-G3 indisponible)"
                        )
                else:
                    # Gate CD45 désactivée ou masque indisponible
                    _cd45_data_plot = X_transformed_for_kde[:, _cd45_idx]
                    _mask_g3_plot = np.ones(X_transformed_for_kde.shape[0], dtype=bool)
                    _logger.info(
                        "Plot KDE CD45 : masque G3 indisponible — fallback post-gating"
                    )

                _fig_kde_cd45, _, _, _ = _plot_cd45_kde(
                    cd45_data=_cd45_data_plot,
                    mask_cd45=_mask_g3_plot,
                    output_path=Path(gating_plot_dir)
                    / "gating"
                    / "combined_07_kde_cd45.png",
                    conditions=None,
                    kde_seuil_relatif=getattr(
                        pregate_cfg, "kde_cd45_seuil_relatif", 0.05
                    ),
                    kde_finesse=getattr(pregate_cfg, "kde_cd45_finesse", 0.6),
                    kde_sigma_smooth=getattr(pregate_cfg, "kde_cd45_sigma_smooth", 10),
                    kde_n_grid=getattr(pregate_cfg, "kde_cd45_n_grid", 1000),
                )
                if _fig_kde_cd45 is not None:
                    gating_figures["fig_kde_cd45"] = _fig_kde_cd45
                    _logger.info("Plot KDE CD45 (logicle) généré avec succès")
        except Exception as _e:
            _logger.warning(
                "Plot KDE CD45 post-transform échoué (non bloquant): %s", _e
            )

    # ── 7. Split par fichier — AUCUN sous-échantillonnage avant FlowSOM ────────
    # Identique au monolithe flowsom_pipeline.py : FlowSOM s'entraîne sur
    # L'INTÉGRALITÉ des cellules gatées. Le SOM (grille xdim×ydim) fait lui-même
    # le résumé en prototypes de nodes — aucun sous-échantillonnage a priori.
    # Le sous-échantillonnage bootstrap (20 000 cellules) n'est utilisé QUE
    # dans la phase 2 de l'auto-clustering (stability bootstrap).
    processed: List[FlowSample] = []
    for orig_sample in sample_order:
        file_mask = file_origins == orig_sample.name
        X_file = X_raw[file_mask]
        n_file = X_file.shape[0]

        if n_file < 100:
            _logger.warning(
                "%s: %d cellules après gating — ignoré",
                orig_sample.name,
                n_file,
            )
            gating_logger.log(
                orig_sample.name,
                "final_count",
                n_file,
                n_file,
                warning=f"Ignoré: seulement {n_file} cellules après gating",
            )
            continue

        n_raw_orig = orig_sample.n_cells_raw or n_file

        processed.append(
            FlowSample(
                name=orig_sample.name,
                path=orig_sample.path,
                condition=orig_sample.condition,
                data=_pd.DataFrame(X_file, columns=var_names),
                metadata={
                    **orig_sample.metadata,
                    "preprocessed": True,
                    "n_original_cells": n_raw_orig,
                    "n_after_gating": n_file,
                },
                n_cells_raw=n_raw_orig,
                raw_data=_pd.DataFrame(
                    X_pre_transform[file_mask], columns=var_names_pre_transform
                ),
            )
        )
        _logger.info(
            "%s [%s]: %d → %d cellules après gating (toutes conservées pour FlowSOM)",
            orig_sample.name,
            orig_sample.condition,
            n_raw_orig,
            n_file,
        )

    # ── Comptage CD45+ brut (optionnel, debug uniquement) ────────────────────
    # Désactivé par défaut car l'objectif opérationnel est le CD45+ pathologique
    # utilisé dans le pipeline principal.
    export_cd45_raw_plot = bool(getattr(pregate_cfg, "export_cd45_raw_plot", False))
    if gating_plot_dir is not None and export_cd45_raw_plot:
        try:
            _cd45_fig = _count_cd45_raw(
                X_for_plot,
                var_for_plot,
                conditions_for_plot,
                output_dir=Path(gating_plot_dir) / "gating",
                kde_seuil_relatif=getattr(pregate_cfg, "kde_cd45_seuil_relatif", 0.05),
                kde_finesse=getattr(pregate_cfg, "kde_cd45_finesse", 0.6),
                kde_sigma_smooth=getattr(pregate_cfg, "kde_cd45_sigma_smooth", 10),
                kde_n_grid=getattr(pregate_cfg, "kde_cd45_n_grid", 1000),
            )
            if _cd45_fig is not None:
                gating_figures["fig_cd45_count"] = _cd45_fig
        except Exception as _e:
            _logger.warning("Comptage CD45+ brut échoué (non bloquant): %s", _e)

    _logger.info(
        "Prétraitement combiné terminé: %d/%d fichiers",
        len(processed),
        len(samples),
    )
    gating_checkpoint = {
        "gate_masks": gate_masks,
        "combined_mask": _combined_mask,
        "var_names_before_gating": var_for_plot,
        "n_total_raw": int(n_total_raw),
        "n_after_gating": int(n_after_gate),
    }
    return processed, gating_figures, gating_checkpoint


def _count_cd45_raw(
    X: np.ndarray,
    var_names: List[str],
    conditions: Optional[np.ndarray],
    output_dir: Optional[Path] = None,
    kde_seuil_relatif: float = 0.05,
    kde_finesse: float = 0.6,
    kde_sigma_smooth: int = 10,
    kde_n_grid: int = 1000,
) -> Optional[Any]:
    """
    Compte les événements CD45+ sur les données brutes (avant tout gating pipeline)
    via KDE 1D méthode pied du pic (identique au gating G3 auto_gate_cd45).

    Ce comptage est purement informatif — il n'influence pas le pipeline de gating.
    L'objectif est de visualiser la population CD45+ telle qu'elle apparaît dans
    les données brutes, sans biais lié aux gates précédentes.

    Args:
        X: Matrice brute (n_cells × n_markers).
        var_names: Noms des marqueurs.
        conditions: Conditions par cellule (optionnel, pour affichage).
        output_dir: Répertoire de sortie pour la figure PNG.
        kde_seuil_relatif: Fraction du max densité pour le pied du pic KDE.
        kde_finesse: Facteur bandwidth Silverman.
        kde_sigma_smooth: Sigma du lissage gaussien sur la courbe KDE.
        kde_n_grid: Résolution de la grille KDE.

    Returns:
        Figure matplotlib ou None.
    """
    from src.prisma.core.auto_gating import AutoGating

    cd45_idx = PreGating.find_marker_index(
        var_names, ["CD45", "CD45-PECY5", "CD45-PC5"]
    )
    if cd45_idx is None:
        _logger.info("_count_cd45_raw: marqueur CD45 absent — comptage ignoré")
        return None

    cd45 = X[:, cd45_idx].astype(np.float64)
    valid = np.isfinite(cd45)
    cd45_v = cd45[valid]
    n_total = len(cd45)

    if valid.sum() < 200:
        _logger.warning("_count_cd45_raw: pas assez de valeurs valides CD45")
        return None

    # ── KDE 1D pied du pic (même méthode que auto_gate_cd45) ──────────────────
    mask_pos = np.ones(n_total, dtype=bool)  # fallback = tout positif
    threshold = np.nan
    method_used = "none"

    try:
        # Sous-échantillonnage pour KDE (gaussian_kde est O(N) en mémoire — blocage >1M pts)
        _kde_max = 80_000
        if len(cd45_v) > _kde_max:
            _rng_kde = np.random.default_rng(42)
            cd45_v_kde = cd45_v[
                _rng_kde.choice(len(cd45_v), size=_kde_max, replace=False)
            ]
        else:
            cd45_v_kde = cd45_v
        threshold, method_used = AutoGating._kde1d_seuil_pied_pic(
            cd45_v_kde,
            seuil_relatif=kde_seuil_relatif,
            finesse=kde_finesse,
            sigma_smooth=kde_sigma_smooth,
            n_grid=kde_n_grid,
        )
        mask_pos = np.zeros(n_total, dtype=bool)
        mask_pos[valid] = cd45[valid] >= threshold
        method_used = f"KDE pied du pic ({method_used})"
    except Exception as _kde_err:
        _logger.warning("_count_cd45_raw KDE échoué: %s", _kde_err)
        return None

    n_pos = int(mask_pos.sum())
    pct_pos = n_pos / n_total * 100

    _logger.info(
        "[CD45-RAW] %s — CD45+: %d/%d (%.1f%%), seuil=%.0f",
        method_used,
        n_pos,
        n_total,
        pct_pos,
        threshold if np.isfinite(threshold) else -1,
    )
    # Plot brut explicitement désactivé pour éviter ~12s de coût I/O + rendu.
    _ = output_dir
    _ = conditions
    return None


def _apply_gating_combined(
    X: np.ndarray,
    var_names: List[str],
    conditions: np.ndarray,
    file_origins: np.ndarray,
    pregate_cfg,
    gating_logger: GatingLogger,
    gmm_plot_path: Optional[str] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    Gating sur les données combinées — reproduit exactement flowsom_pipeline.py.

    Chaque gate est calculé sur la TOTALITÉ de X (toutes conditions mélangées),
    puis les masques sont combinés avec la logique asymétrique
    sain/patho si ``mode_blastes_vs_normal=True``.

    Args:
        X: Matrice brute combinée (toutes cellules).
        var_names: Noms des marqueurs.
        conditions: Condition par cellule (ex: "Sain", "Pathologique").
        file_origins: Nom de fichier par cellule.
        pregate_cfg: Configuration du gating.
        gating_logger: Logger de gating.

    Returns:
        Tuple (X_filtré, var_names, conditions_filtré, file_origins_filtré).
    """
    mode = getattr(pregate_cfg, "mode", "auto")
    mode_blastes_vs_normal = getattr(pregate_cfg, "mode_blastes_vs_normal", False)
    n_before = X.shape[0]
    combined_mask = np.ones(n_before, dtype=bool)
    gate_masks: Dict[str, np.ndarray] = {}

    _sain_labels = {"sain", "normal", "healthy", "nbm", "moelle normale"}
    is_sain_vec = np.array([c.lower() in _sain_labels for c in conditions], dtype=bool)
    is_patho_vec = ~is_sain_vec

    # ── Gate 1 : Débris (GMM 2D FSC-A/SSC-A) ─────────────────────────────────
    # Calculé sur TOUTES les cellules combinées.
    if getattr(pregate_cfg, "viable", True):
        _logger.info("Gate 1 — Débris [%s] sur %d cellules combinées", mode, n_before)
        if mode == "auto":
            _do_gmm_plot = (
                getattr(pregate_cfg, "gmm_export_plot", False)
                and gmm_plot_path is not None
                and getattr(pregate_cfg, "density_method", "GMM").upper() == "GMM"
            )
            mask_debris = AutoGating.auto_gate_debris(
                X,
                var_names,
                n_components=getattr(pregate_cfg, "gmm_n_components_debris", 3),
                gmm_max_samples=getattr(pregate_cfg, "gmm_max_samples", 50000),
                covariance_type=getattr(pregate_cfg, "gmm_covariance_type", "full"),
                density_method=getattr(pregate_cfg, "density_method", "GMM"),
                export_plot=_do_gmm_plot,
                plot_output_path=gmm_plot_path,
            )
        else:
            mask_debris = PreGating.gate_viable_cells(
                X,
                var_names,
                min_percentile=getattr(pregate_cfg, "debris_min_percentile", 2.0),
                max_percentile=getattr(pregate_cfg, "debris_max_percentile", 99.0),
            )
        n_debris_kept = int(mask_debris.sum())
        _logger.info(
            "  G1_debris: %d/%d (%.1f%%)",
            n_debris_kept,
            n_before,
            n_debris_kept / max(n_before, 1) * 100,
        )
        gating_logger.log(
            "COMBINED",
            "G1_debris",
            n_before,
            n_debris_kept,
            method=mode,
        )
        combined_mask &= mask_debris
        gate_masks["G1_debris"] = mask_debris
    else:
        mask_debris = np.ones(n_before, dtype=bool)

    # ── Gate 2 : Singlets (RANSAC par fichier, sur données combinées) ─────────
    # Passage explicite de file_origins → RANSAC per_file comme dans le monolithe.
    if getattr(pregate_cfg, "singlets", True):
        n_after_g1 = int(combined_mask.sum())
        _logger.info(
            "Gate 2 — Singlets [%s] par fichier sur %d cellules", mode, n_after_g1
        )
        if mode == "auto":
            mask_singlets = AutoGating.auto_gate_singlets(
                X,
                var_names,
                file_origin=file_origins,
                per_file=True,
                r2_threshold=getattr(pregate_cfg, "ransac_r2_threshold", 0.85),
                mad_factor=getattr(pregate_cfg, "ransac_mad_factor", 3.0),
            )
        else:
            mask_singlets = PreGating.gate_singlets(
                X,
                var_names,
                ratio_min=getattr(pregate_cfg, "singlet_ratio_min", 0.6),
                ratio_max=getattr(pregate_cfg, "singlet_ratio_max", 1.5),
            )
        n_g1g2 = int((combined_mask & mask_singlets).sum())
        _logger.info(
            "  G2_singlets: %d/%d (%.1f%%)",
            n_g1g2,
            n_after_g1,
            n_g1g2 / max(n_after_g1, 1) * 100,
        )
        gating_logger.log(
            "COMBINED",
            "G2_singlets",
            n_after_g1,
            n_g1g2,
            method=mode,
        )
        combined_mask &= mask_singlets
        gate_masks["G2_singlets"] = mask_singlets
    else:
        mask_singlets = np.ones(n_before, dtype=bool)

    # ── Gate 3 : CD45 (SYMÉTRIQUE — appliqué à toutes les conditions) ───────────
    # Le gating CD45 s'applique à TOUTES les cellules (saines et pathologiques).
    # Pour les données combinées (mode_blastes_vs_normal), le KDE est calculé
    # séparément sur chaque sous-population afin que le seuil soit adapté à la
    # distribution propre de chaque condition (sain : pic CD45+ très dominant ;
    # patho : distribution potentiellement décalée vers CD45-dim).
    # Résultat : le pool NBM ne contient que des leucocytes CD45+, ce qui garantit
    # une médiane NBM correcte et des Z-scores CD45 valides.
    if getattr(pregate_cfg, "cd45", True):
        n_after_g1g2 = int(combined_mask.sum())
        _logger.info(
            "Gate 3 — CD45 [%s] SYMÉTRIQUE (KDE pied du pic sur chaque condition)",
            mode,
        )

        _kde_kwargs = dict(
            kde_seuil_relatif=getattr(pregate_cfg, "kde_cd45_seuil_relatif", 0.05),
            kde_finesse=getattr(pregate_cfg, "kde_cd45_finesse", 0.6),
            kde_sigma_smooth=getattr(pregate_cfg, "kde_cd45_sigma_smooth", 10),
            kde_n_grid=getattr(pregate_cfg, "kde_cd45_n_grid", 1000),
            kde_max_samples=getattr(pregate_cfg, "kde_cd45_max_samples", 10000),
            threshold_percentile=getattr(pregate_cfg, "cd45_threshold_percentile", 5.0),
        )
        _pct_kwargs = dict(
            threshold_percentile=getattr(pregate_cfg, "cd45_min_percentile", 5.0),
        )

        if mode_blastes_vs_normal and (is_sain_vec.any() and is_patho_vec.any()):
            # Calcul du KDE séparé par condition pour calibrer le seuil sur
            # la distribution propre à chaque sous-population.
            mask_cd45 = np.zeros(n_before, dtype=bool)

            X_sain = X[is_sain_vec]
            if mode == "auto":
                mask_cd45_sain = AutoGating.auto_gate_cd45(
                    X_sain, var_names, **_kde_kwargs
                )
            else:
                mask_cd45_sain = PreGating.gate_cd45_positive(
                    X_sain, var_names, **_pct_kwargs
                )
            mask_cd45[is_sain_vec] = mask_cd45_sain

            X_patho = X[is_patho_vec]
            if mode == "auto":
                mask_cd45_patho = AutoGating.auto_gate_cd45(
                    X_patho, var_names, **_kde_kwargs
                )
            else:
                mask_cd45_patho = PreGating.gate_cd45_positive(
                    X_patho, var_names, **_pct_kwargs
                )
            mask_cd45[is_patho_vec] = mask_cd45_patho

            n_sain_pre = int((combined_mask & is_sain_vec).sum())
            n_sain_post = int((combined_mask & mask_cd45 & is_sain_vec).sum())
            n_patho_pre = int((combined_mask & is_patho_vec).sum())
            n_patho_post = int((combined_mask & mask_cd45 & is_patho_vec).sum())
            _logger.info(
                "  Sain  CD45+ conservés: %d/%d (%.1f%%)",
                n_sain_post,
                n_sain_pre,
                n_sain_post / max(n_sain_pre, 1) * 100,
            )
            _logger.info(
                "  Patho CD45+ conservés: %d/%d (%.1f%%)",
                n_patho_post,
                n_patho_pre,
                n_patho_post / max(n_patho_pre, 1) * 100,
            )
            _g3_extras: dict = {
                "method": mode,
                "n_sain_pre_cd45": n_sain_pre,
                "n_sain_post_cd45": n_sain_post,
                "n_patho_pre_cd45": n_patho_pre,
                "n_patho_post_cd45": n_patho_post,
            }
        else:
            # Pas de distinction sain/patho : KDE unique sur toutes les cellules.
            if mode == "auto":
                mask_cd45 = AutoGating.auto_gate_cd45(X, var_names, **_kde_kwargs)
            else:
                mask_cd45 = PreGating.gate_cd45_positive(X, var_names, **_pct_kwargs)
            _g3_extras = {"method": mode}

        n_after_cd45 = int((combined_mask & mask_cd45).sum())
        _logger.info(
            "  G3_cd45: %d/%d (%.1f%%)",
            n_after_cd45,
            n_after_g1g2,
            n_after_cd45 / max(n_after_g1g2, 1) * 100,
        )
        gating_logger.log(
            "COMBINED",
            "G3_cd45",
            n_after_g1g2,
            n_after_cd45,
            **_g3_extras,
        )
        combined_mask &= mask_cd45
        gate_masks["G3_cd45"] = mask_cd45

    # ── Gate 4 : CD34+ blastes (asymétrique) ──────────────────────────────────
    if getattr(pregate_cfg, "cd34", False):
        n_after_g123 = int(combined_mask.sum())
        if mode_blastes_vs_normal:
            _logger.info("Gate 4 — CD34 [%s] ASYMÉTRIQUE (patho uniquement)", mode)
            if mode == "auto":
                mask_cd34_full = AutoGating.auto_gate_cd34(X, var_names)
            else:
                mask_cd34_full = PreGating.gate_cd34_blasts(
                    X,
                    var_names,
                    threshold_percentile=getattr(
                        pregate_cfg, "cd34_min_percentile", 70.0
                    ),
                    ssc_max_percentile=getattr(pregate_cfg, "cd34_ssc_max", 40.0),
                )
            mask_cd34 = np.ones(n_before, dtype=bool)
            mask_cd34[is_patho_vec] = mask_cd34_full[is_patho_vec]
        else:
            _logger.info("Gate 4 — CD34 [%s] sur toutes les cellules", mode)
            if mode == "auto":
                mask_cd34 = AutoGating.auto_gate_cd34(X, var_names)
            else:
                mask_cd34 = PreGating.gate_cd34_blasts(
                    X,
                    var_names,
                    threshold_percentile=getattr(
                        pregate_cfg, "cd34_min_percentile", 70.0
                    ),
                    ssc_max_percentile=getattr(pregate_cfg, "cd34_ssc_max", 40.0),
                )
        n_after_cd34 = int((combined_mask & mask_cd34).sum())
        _logger.info(
            "  G4_cd34: %d/%d (%.1f%%)",
            n_after_cd34,
            n_after_g123,
            n_after_cd34 / max(n_after_g123, 1) * 100,
        )
        gating_logger.log(
            "COMBINED",
            "G4_cd34",
            n_after_g123,
            n_after_cd34,
            method=mode,
        )
        combined_mask &= mask_cd34
        gate_masks["G4_cd34"] = mask_cd34

    # ── Résumé par condition ───────────────────────────────────────────────────
    n_final = int(combined_mask.sum())
    _logger.info(
        "Gating combiné: %d/%d cellules totales (%.1f%%)",
        n_final,
        n_before,
        n_final / max(n_before, 1) * 100,
    )
    if mode_blastes_vs_normal:
        n_sain_final = int((combined_mask & is_sain_vec).sum())
        n_patho_final = int((combined_mask & is_patho_vec).sum())
        _logger.info(
            "  Sain: %d cellules conservées | Patho: %d cellules conservées",
            n_sain_final,
            n_patho_final,
        )
        if n_patho_final > 0 and n_sain_final > 0:
            ratio = n_sain_final / n_patho_final
            if ratio > 10:
                _logger.warning(
                    "Déséquilibre conditions: %d sain vs %d patho (ratio %.1f×) "
                    "— considérer balance_conditions=true dans la config",
                    n_sain_final,
                    n_patho_final,
                    ratio,
                )

    # ── Application du masque final ───────────────────────────────────────────
    X_filtered = X[combined_mask]
    conditions_filtered = conditions[combined_mask]
    file_origins_filtered = file_origins[combined_mask]

    return (
        X_filtered,
        var_names,
        conditions_filtered,
        file_origins_filtered,
        gate_masks,
        combined_mask,
    )
