"""
analysis/batch.py — Utilitaires cohorte pour correction batch et remapping.

Objectifs:
- Construire une matrice concaténée (Experiment -> matrix) en conservant les
  index d'origine (sample_id + event_index).
- Appliquer Harmony (si disponible) avec fallback déterministe sans crash.
- Réinjecter proprement matrices/embeddings/labels vers chaque Sample.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from models.experiment import Experiment, ExperimentError

logger = logging.getLogger(__name__)

try:
    import harmonypy  # type: ignore

    _HARMONY_AVAILABLE = True
except (ImportError, OSError):
    _HARMONY_AVAILABLE = False


@dataclass
class CohortMatrix:
    """Conteneur de données concaténées avec mapping vers les samples."""

    X: np.ndarray
    channels: List[str]
    row_sample_ids: np.ndarray
    row_event_indices: np.ndarray
    batch_labels: np.ndarray


@dataclass
class HarmonyParams:
    """Paramètres MVP pour Harmony (harmonypy)."""

    channels: Optional[List[str]] = None
    source_masks: Optional[List[str]] = None
    batch_key: str = "batch"
    theta: float = 2.0
    lambda_: float = 1.0
    sigma: float = 0.1
    nclust: Optional[int] = None
    max_iter_harmony: int = 10
    max_iter_kmeans: int = 20
    random_state: int = 42


def _ensure_experiment_results(experiment: Experiment) -> Dict[str, Any]:
    """Ajoute un namespace de résultats cohorte si absent."""
    if not hasattr(experiment, "analysis_results"):
        setattr(experiment, "analysis_results", {})
    return getattr(experiment, "analysis_results")


def _combined_mask(sample: Any, source_masks: Optional[Sequence[str]]) -> np.ndarray:
    """Conjonction logique des masques demandés sur un sample."""
    if source_masks is None:
        source_masks = list(sample.masks.keys())

    if not source_masks:
        return np.ones(sample.n_events, dtype=bool)

    combined = np.ones(sample.n_events, dtype=bool)
    for name in source_masks:
        if name not in sample.masks:
            raise ValueError(f"Mask '{name}' not found in sample '{sample.sample_id}'.")
        combined &= sample.get_mask(name)
    return combined


def _resolve_batch_label(experiment: Experiment, sample: Any, batch_key: str) -> str:
    """Résout un label batch par sample avec fallback explicite."""
    sid = sample.sample_id
    if not experiment.cohort_metadata.empty and sid in experiment.cohort_metadata.index:
        if batch_key in experiment.cohort_metadata.columns:
            value = experiment.cohort_metadata.loc[sid, batch_key]
            return str(value)

    meta = getattr(sample, "per_sample_metadata", {}) or {}
    if batch_key in meta:
        return str(meta[batch_key])

    logger.warning(
        "[Batch] sample '%s': batch_key '%s' absent; fallback sample_id.",
        sid,
        batch_key,
    )
    return sid


def _marker_key(name: str) -> str:
    """Normalise un nom de canal pour les comparaisons techniques."""
    key = str(name).upper().strip().replace(" ", "")
    for suffix in ("-A", "-H", "-W", "_A", "_H", "_W"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    return key


def _ssc_alignment_indices(feature_names: Optional[Sequence[str]]) -> List[int]:
    """Retourne les indices des canaux SSC uniquement."""
    if feature_names is None:
        return []
    return [i for i, name in enumerate(feature_names) if _marker_key(name) == "SSC"]


def build_cohort_matrix(
    experiment: Experiment,
    channels: Optional[List[str]] = None,
    source_masks: Optional[List[str]] = None,
    batch_key: str = "batch",
) -> CohortMatrix:
    """
    Concatène les événements de tous les samples en préservant index d'origine.

    Notes:
    - Ne passe pas par get_pre_gated_data() pour éviter la perte d'index.
    - Utilise les index natifs (0..N-1) de chaque Sample pour mapping retour.
    """
    if experiment.n_samples == 0:
        raise ExperimentError("Experiment has no samples.")

    resolved_channels = channels or experiment.common_channels()
    if not resolved_channels:
        raise ValueError("No common channels available for cohort concatenation.")

    blocks: List[np.ndarray] = []
    row_sample_ids: List[str] = []
    row_event_indices: List[int] = []
    batch_labels: List[str] = []

    for sample in experiment.samples:
        sid = sample.sample_id
        missing = sorted(set(resolved_channels) - set(sample.channels))
        if missing:
            raise ValueError(f"Sample '{sid}' missing required channels: {missing}")

        mask = _combined_mask(sample, source_masks)
        event_idx = np.where(mask)[0]
        if event_idx.size == 0:
            logger.warning("[Batch] sample '%s': 0 events after masking; skipped.", sid)
            continue

        x = sample.events[resolved_channels].to_numpy(dtype=np.float32)[event_idx]
        blocks.append(x)

        batch_value = _resolve_batch_label(experiment, sample, batch_key)
        row_sample_ids.extend([sid] * len(event_idx))
        row_event_indices.extend(event_idx.tolist())
        batch_labels.extend([batch_value] * len(event_idx))

    if not blocks:
        raise ValueError("All samples are empty after masking.")

    X = np.vstack(blocks).astype(np.float32)
    return CohortMatrix(
        X=X,
        channels=resolved_channels,
        row_sample_ids=np.asarray(row_sample_ids, dtype=object),
        row_event_indices=np.asarray(row_event_indices, dtype=np.int64),
        batch_labels=np.asarray(batch_labels, dtype=object),
    )


def run_harmony(
    X: np.ndarray,
    batch_labels: np.ndarray,
    params: HarmonyParams,
    feature_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Lance Harmony avec fallback propre si dépendance absente ou échec runtime.

    Returns:
        (X_corrected, metadata)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape={X.shape}.")
    if len(batch_labels) != X.shape[0]:
        raise ValueError(f"batch_labels length ({len(batch_labels)}) != n_rows ({X.shape[0]}).")

    unique_batches = np.unique(batch_labels)
    if unique_batches.size < 2:
        logger.warning("[Harmony] <2 batches detected; returning uncorrected matrix.")
        return X.copy(), {
            "backend": "identity",
            "reason": "single_batch",
            "n_batches": int(unique_batches.size),
        }

    if not _HARMONY_AVAILABLE:
        logger.warning("[Harmony] harmonypy unavailable; returning uncorrected matrix.")
        return X.copy(), {
            "backend": "identity",
            "reason": "harmonypy_missing",
            "n_batches": int(unique_batches.size),
        }

    align_indices = _ssc_alignment_indices(feature_names)
    if not align_indices:
        logger.warning("[Harmony] aucun canal SSC détecté; retour de la matrice brute.")
        return X.copy(), {
            "backend": "identity",
            "reason": "ssc_missing",
            "n_batches": int(unique_batches.size),
        }

    if len(align_indices) == X.shape[1]:
        X_to_correct = X
    else:
        X_to_correct = X[:, align_indices]

    meta = pd.DataFrame({params.batch_key: batch_labels.astype(str)})

    try:
        ho = harmonypy.run_harmony(
            X_to_correct,
            meta,
            [params.batch_key],
            theta=params.theta,
            lamb=params.lambda_,
            sigma=params.sigma,
            nclust=params.nclust,
            max_iter_harmony=params.max_iter_harmony,
            max_iter_kmeans=params.max_iter_kmeans,
            random_state=params.random_state,
            verbose=False,
        )
        raw = np.asarray(ho.Z_corr)

        if raw.shape == X_to_correct.shape:
            X_corr = raw
        elif raw.T.shape == X_to_correct.shape:
            X_corr = raw.T
        else:
            raise ValueError(
                f"Harmony output shape {raw.shape} incompatible with input {X_to_correct.shape}."
            )

        if len(align_indices) == X.shape[1]:
            corrected = X_corr.astype(np.float32)
        else:
            corrected = X.copy().astype(np.float32)
            corrected[:, align_indices] = X_corr.astype(np.float32)

        return corrected, {
            "backend": "harmonypy",
            "n_batches": int(unique_batches.size),
            "shape": tuple(int(v) for v in corrected.shape),
            "aligned_channels": [feature_names[i] for i in align_indices] if feature_names else [],
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Harmony] failed (%s); returning uncorrected matrix.", exc)
        return X.copy(), {
            "backend": "identity",
            "reason": "harmonypy_error",
            "error": str(exc),
            "n_batches": int(unique_batches.size),
        }


def inject_corrected_channels(
    experiment: Experiment,
    cohort: CohortMatrix,
    corrected_matrix: np.ndarray,
    channel_prefix: str = "harmony_",
) -> None:
    """Réinjecte la matrice corrigée en canaux virtuels alignés par événement."""
    if corrected_matrix.shape != cohort.X.shape:
        raise ValueError(
            f"corrected_matrix shape mismatch: {corrected_matrix.shape} vs {cohort.X.shape}."
        )

    for sample in experiment.samples:
        sid = sample.sample_id
        rows = np.where(cohort.row_sample_ids == sid)[0]
        if rows.size == 0:
            continue

        event_idx = cohort.row_event_indices[rows]
        sample_block = corrected_matrix[rows]

        for j, ch in enumerate(cohort.channels):
            full = np.full(sample.n_events, np.nan, dtype=np.float32)
            full[event_idx] = sample_block[:, j]
            sample.add_virtual_channel(f"{channel_prefix}{ch}", full)


def inject_embedding(
    experiment: Experiment,
    cohort: CohortMatrix,
    embedding: np.ndarray,
    embedding_name: str,
) -> None:
    """Réinjecte un embedding cohorte au niveau Sample en gardant les index."""
    if embedding.ndim != 2:
        raise ValueError(f"embedding must be 2-D, got shape={embedding.shape}.")
    if embedding.shape[0] != cohort.X.shape[0]:
        raise ValueError(
            f"embedding rows ({embedding.shape[0]}) != cohort rows ({cohort.X.shape[0]})."
        )

    for sample in experiment.samples:
        sid = sample.sample_id
        rows = np.where(cohort.row_sample_ids == sid)[0]
        full = np.full((sample.n_events, embedding.shape[1]), np.nan, dtype=np.float32)
        if rows.size > 0:
            event_idx = cohort.row_event_indices[rows]
            full[event_idx] = embedding[rows]
        sample.add_embedding(embedding_name, full)


def inject_cluster_labels(
    experiment: Experiment,
    cohort: CohortMatrix,
    labels: np.ndarray,
    label_name: str,
) -> None:
    """Réinjecte des labels cohorte au niveau Sample (-1 hors sous-ensemble)."""
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape={labels.shape}.")
    if labels.shape[0] != cohort.X.shape[0]:
        raise ValueError(f"labels length ({labels.shape[0]}) != cohort rows ({cohort.X.shape[0]}).")

    for sample in experiment.samples:
        sid = sample.sample_id
        rows = np.where(cohort.row_sample_ids == sid)[0]
        full = np.full(sample.n_events, -1, dtype=np.int32)
        if rows.size > 0:
            event_idx = cohort.row_event_indices[rows]
            full[event_idx] = labels[rows].astype(np.int32)
        sample.add_cluster_labels(label_name, full)


def store_experiment_mapping(experiment: Experiment, key: str, payload: Dict[str, Any]) -> None:
    """Stocke un résultat cohorte standardisé dans experiment.analysis_results."""
    out = _ensure_experiment_results(experiment)
    out[key] = payload
