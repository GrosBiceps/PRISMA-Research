"""
analysis/spectral.py — Spectral unmixing strategy (MVP: NNLS per-event).

SpectralUnmixingStrategy:
  - Inputs: reference single-stain signature matrix + spectral data
  - Algorithm: scipy.optimize.nnls (event-by-event)
  - Optional: autofluorescence component
  - Output: unmixed channels injected via sample.add_virtual_channel()

Config dataclass: SpectralUnmixingParams
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.optimize import nnls

from models.sample import Sample
from analysis.strategies import BaseAnalysisStrategy
from analysis.downsampling import expand_to_full

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AutofluorescenceConfig:
    """Autofluorescence component configuration."""

    enabled: bool = False
    channel_name: str = "Autofluorescence"


@dataclass
class SpectralUnmixingParams:
    """Parameters for SpectralUnmixingStrategy.

    Args:
        reference_matrix: (n_detectors × n_fluorochromes) — each column is a
            normalized single-stain spectrum. If autofluorescence.enabled, the
            last column must be the AF spectrum (or it is appended from
            af_spectrum).
        fluorochrome_names: Names matching reference_matrix columns (excl. AF).
        channels: Detector channels to use. None = all sample channels.
        autofluorescence: AF config.
        af_spectrum: AF reference vector (n_detectors,), required if
            autofluorescence.enabled and AF column not already in
            reference_matrix.
        seed: Reproducibility seed for any stochastic steps.
        max_events: Downsampling cap.
        unmixed_prefix: Prefix for virtual channel names.
    """

    reference_matrix: Optional[np.ndarray] = None
    fluorochrome_names: Optional[List[str]] = None
    channels: Optional[List[str]] = None
    autofluorescence: AutofluorescenceConfig = field(default_factory=AutofluorescenceConfig)
    af_spectrum: Optional[np.ndarray] = None
    seed: int = 42
    max_events: int = 100_000
    unmixed_prefix: str = "unmixed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_reference(
    ref: np.ndarray,
    n_detectors: int,
    n_fluorochromes: int,
) -> None:
    """Strict shape + finite check on reference matrix."""
    if ref.ndim != 2:
        raise ValueError(f"reference_matrix must be 2-D, got {ref.ndim}-D.")
    if ref.shape[0] != n_detectors:
        raise ValueError(f"reference_matrix rows ({ref.shape[0]}) ≠ n_detectors ({n_detectors}).")
    if ref.shape[1] != n_fluorochromes:
        raise ValueError(
            f"reference_matrix cols ({ref.shape[1]}) ≠ n_fluorochromes ({n_fluorochromes})."
        )
    if not np.all(np.isfinite(ref)):
        raise ValueError("reference_matrix contains NaN/Inf values.")


def build_full_reference(
    ref: np.ndarray,
    fluorochrome_names: List[str],
    af_cfg: AutofluorescenceConfig,
    af_spectrum: Optional[np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    """Append AF column to reference if needed; return (full_ref, full_names)."""
    full_ref = ref.copy().astype(np.float64)
    full_names = list(fluorochrome_names)

    if af_cfg.enabled:
        if af_spectrum is not None:
            af_vec = np.array(af_spectrum, dtype=np.float64)
            if af_vec.shape[0] != ref.shape[0]:
                raise ValueError(
                    f"af_spectrum length ({af_vec.shape[0]}) ≠ n_detectors ({ref.shape[0]})."
                )
            # Normalize AF column (L2)
            norm = np.linalg.norm(af_vec)
            if norm > 0:
                af_vec /= norm
            full_ref = np.column_stack([full_ref, af_vec])
            full_names.append(af_cfg.channel_name)
            logger.debug("AF column appended: '%s'.", af_cfg.channel_name)
        else:
            if af_cfg.channel_name not in full_names:
                raise ValueError(
                    "autofluorescence.enabled=True requires af_spectrum or an "
                    f"existing AF component named '{af_cfg.channel_name}' "
                    "in fluorochrome_names/reference_matrix."
                )
            logger.info(
                "AF component '%s' already present in reference matrix.",
                af_cfg.channel_name,
            )

    return full_ref, full_names


def nnls_unmix(
    data: np.ndarray,
    ref: np.ndarray,
) -> np.ndarray:
    """
    NNLS unmixing: event-by-event.

    Args:
        data: (n_events × n_detectors)
        ref:  (n_detectors × n_components) — normalized reference spectra

    Returns:
        coefficients: (n_events × n_components)
    """
    n_events, n_detectors = data.shape
    n_components = ref.shape[1]

    if ref.shape[0] != n_detectors:
        raise ValueError(f"ref rows ({ref.shape[0]}) ≠ data cols ({n_detectors}).")

    coefficients = np.empty((n_events, n_components), dtype=np.float32)
    for i in range(n_events):
        coefficients[i], _ = nnls(ref, data[i])

    return coefficients


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class SpectralUnmixingStrategy(BaseAnalysisStrategy):
    """
    Spectral unmixing via NNLS (non-negative least squares), event-by-event.

    Stores each unmixed fluorochrome as a virtual channel in the Sample:
        sample.virtual_channels["{unmixed_prefix}_{fluorochrome_name}"]

    If autofluorescence is enabled, the AF component is stored as:
        sample.virtual_channels["{unmixed_prefix}_{af_cfg.channel_name}"]

    Extension point:
        Override _unmix() to swap NNLS for OLS/WLS/ridge when needed.
    """

    name: str = "spectral_unmixing"

    def run(self, sample: Sample, **kwargs: Any) -> Sample:
        p: SpectralUnmixingParams = kwargs.get("params", SpectralUnmixingParams())
        if isinstance(p, dict):
            p = SpectralUnmixingParams(**p)

        if p.reference_matrix is None:
            raise ValueError("[SpectralUnmixing] reference_matrix is required.")
        if p.fluorochrome_names is None:
            raise ValueError("[SpectralUnmixing] fluorochrome_names is required.")

        ref_raw = np.array(p.reference_matrix, dtype=np.float64)
        fluoro_names = list(p.fluorochrome_names)

        if len(fluoro_names) != ref_raw.shape[1]:
            raise ValueError(
                "[SpectralUnmixing] fluorochrome_names length "
                f"({len(fluoro_names)}) must match reference_matrix columns "
                f"({ref_raw.shape[1]})."
            )
        if len(set(fluoro_names)) != len(fluoro_names):
            raise ValueError("[SpectralUnmixing] fluorochrome_names must be unique.")

        # Prepare spectral data (may downsample)
        data, ds, used_channels, gate_idx = self._prepare_data(
            sample, p.channels, p.seed, p.max_events
        )
        n_detectors = data.shape[1]
        n_fluoro = ref_raw.shape[1]

        # Validate reference shape vs detectors
        _validate_reference(ref_raw, n_detectors, n_fluoro)

        # Build full reference (optionally with AF)
        full_ref, full_names = build_full_reference(
            ref_raw, fluoro_names, p.autofluorescence, p.af_spectrum
        )

        if len(set(full_names)) != len(full_names):
            raise ValueError("[SpectralUnmixing] component names are not unique after AF handling.")

        logger.info(
            "[SpectralUnmixing] '%s': %d events × %d detectors → %d components (%s).",
            sample.sample_id,
            data.shape[0],
            n_detectors,
            full_ref.shape[1],
            ", ".join(full_names),
        )

        # Run NNLS
        coefficients = self._unmix(data.astype(np.float64), full_ref)

        # Expand downsampling → gated events → full events
        if ds is not None and ds.was_downsampled:
            coefficients = expand_to_full(coefficients, ds, fill_value=np.nan)

        coefficients = self._expand_to_full_events(
            coefficients, gate_idx, sample.n_events, fill_value=np.nan
        )

        # Inject virtual channels
        for k, name in enumerate(full_names):
            vname = f"{p.unmixed_prefix}_{name}"
            sample.add_virtual_channel(vname, coefficients[:, k])

        logger.info(
            "[SpectralUnmixing] '%s': %d virtual channels added (prefix='%s').",
            sample.sample_id,
            len(full_names),
            p.unmixed_prefix,
        )

        sample.results["spectral_unmixing"] = {
            "fluorochrome_names": full_names,
            "n_detectors": n_detectors,
            "af_enabled": p.autofluorescence.enabled,
            "unmixed_prefix": p.unmixed_prefix,
        }

        return sample

    def _unmix(self, data: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """NNLS unmixing. Override for OLS/WLS/ridge."""
        return nnls_unmix(data, ref)
