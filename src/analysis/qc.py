"""
analysis/qc.py — Quality Control strategy for flow cytometry data.

QCStrategy:
  - Temporal anomaly detection (signal drift / instrument instability)
  - Robust dispersion outlier detection (MAD-based)
  - Produces a boolean QC mask injected into Sample.masks["qc_pass"]
  - PeacoQC wrapper when available; fallback = minimal reproducible QC

PeacoQC Python availability: tested at runtime; falls back gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from models.sample import Sample
from analysis.strategies import BaseAnalysisStrategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PeacoQC availability
# ---------------------------------------------------------------------------

try:
    import peacoqc  # type: ignore

    _PEACOQC_AVAILABLE = True
    logger.debug("PeacoQC available.")
except ImportError:
    _PEACOQC_AVAILABLE = False
    logger.debug("PeacoQC not available — using fallback QC.")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class QCParams:
    """Parameters for QCStrategy.

    Args:
        channels: Channels to monitor for stability. None = all sample channels.
        time_channel: Column name of the Time/acquisition-time channel.
            None = auto-detect ('Time', 'TIME', 'time').
        n_bins: Number of temporal bins for sliding-window analysis.
        mad_threshold: Events with |z_mad| > threshold in any channel are
            flagged as outliers (fallback QC).
        min_events_per_bin: Bins with fewer events are flagged as anomalous.
        min_pass_fraction: Raise warning if < this fraction of events pass QC.
        mask_name: Key under which the boolean QC mask is stored in Sample.masks.
        use_peacoqc: Attempt PeacoQC if available (set False to force fallback).
        max_events: Downsampling cap (QC is run on full data — cap only for
            heavy peacoqc path).
        seed: Reproducibility seed.
    """

    channels: Optional[List[str]] = None
    time_channel: Optional[str] = None
    n_bins: int = 200
    mad_threshold: float = 5.0
    min_events_per_bin: int = 10
    min_pass_fraction: float = 0.5
    mask_name: str = "qc_pass"
    use_peacoqc: bool = True
    max_events: int = 100_000
    seed: int = 42


# ---------------------------------------------------------------------------
# Helpers — Fallback QC
# ---------------------------------------------------------------------------


def _detect_time_channel(columns: List[str]) -> Optional[str]:
    """Return first column matching common time channel names."""
    for col in columns:
        if col.lower() in ("time", "time_channel", "acquisition_time"):
            return col
    return None


def _mad_z(x: np.ndarray) -> np.ndarray:
    """Median-absolute-deviation z-score (robust)."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x, dtype=np.float32)
    return np.abs(x - med) / (1.4826 * mad)


def _temporal_stability_mask(
    data: np.ndarray,
    time_values: np.ndarray,
    n_bins: int,
    min_events_per_bin: int,
    mad_threshold: float,
) -> np.ndarray:
    """
    Detect temporal instability by sliding-window median shift.

    For each bin, compute per-channel median; flag bin if any channel's
    bin-median deviates > mad_threshold MADs from the global median.

    Returns:
        Boolean mask (n_events,) — True = event passes temporal QC.
    """
    n_events = data.shape[0]
    pass_mask = np.ones(n_events, dtype=bool)

    if n_events < 2 * min_events_per_bin:
        logger.warning("Too few events (%d) for temporal QC — all pass.", n_events)
        return pass_mask

    # Sort by time
    sort_idx = np.argsort(time_values)
    sorted_data = data[sort_idx]

    bins = np.array_split(np.arange(n_events), n_bins)
    global_med = np.median(sorted_data, axis=0)  # (n_channels,)
    global_mad = np.median(np.abs(sorted_data - global_med), axis=0)
    global_mad = np.where(global_mad == 0, 1e-9, global_mad)

    flagged_sorted = np.zeros(n_events, dtype=bool)

    for bin_idx in bins:
        if len(bin_idx) < min_events_per_bin:
            flagged_sorted[bin_idx] = True
            continue
        bin_med = np.median(sorted_data[bin_idx], axis=0)
        deviation = np.abs(bin_med - global_med) / (1.4826 * global_mad)
        if np.any(deviation > mad_threshold):
            flagged_sorted[bin_idx] = True

    # Un-sort
    unsort_idx = np.argsort(sort_idx)
    pass_mask = ~flagged_sorted[unsort_idx]
    return pass_mask


def _dispersion_mask(
    data: np.ndarray,
    mad_threshold: float,
) -> np.ndarray:
    """
    Flag individual outlier events via per-channel MAD z-score.

    An event is flagged if it exceeds mad_threshold in ANY channel.

    Returns:
        Boolean mask (n_events,) — True = event passes dispersion QC.
    """
    n_events, n_channels = data.shape
    pass_mask = np.ones(n_events, dtype=bool)
    for c in range(n_channels):
        z = _mad_z(data[:, c])
        pass_mask &= z <= mad_threshold
    return pass_mask


def _fallback_qc(
    data: np.ndarray,
    columns: List[str],
    time_values: Optional[np.ndarray],
    p: QCParams,
) -> np.ndarray:
    """
    Minimal reproducible QC:
      1. Temporal stability (if time channel available)
      2. Robust MAD dispersion

    Returns boolean mask (n_events,) — True = pass.
    """
    n_events = data.shape[0]
    pass_mask = np.ones(n_events, dtype=bool)

    if time_values is not None:
        temporal_mask = _temporal_stability_mask(
            data, time_values, p.n_bins, p.min_events_per_bin, p.mad_threshold
        )
        n_flagged_temporal = (~temporal_mask).sum()
        logger.info(
            "[QC/temporal] %d / %d events flagged (%.1f%%).",
            n_flagged_temporal,
            n_events,
            100.0 * n_flagged_temporal / n_events,
        )
        pass_mask &= temporal_mask
    else:
        logger.warning("[QC] No time channel found — temporal QC skipped.")

    dispersion_mask = _dispersion_mask(data, p.mad_threshold)
    n_flagged_disp = (~dispersion_mask).sum()
    logger.info(
        "[QC/dispersion] %d / %d events flagged (%.1f%%).",
        n_flagged_disp,
        n_events,
        100.0 * n_flagged_disp / n_events,
    )
    pass_mask &= dispersion_mask

    return pass_mask


def _peacoqc_wrapper(
    df: pd.DataFrame,
    channels: List[str],
    time_col: Optional[str],
    p: QCParams,
) -> np.ndarray:
    """
    Attempt PeacoQC via Python binding.

    Falls back to _fallback_qc on any error.
    """
    try:
        data = df[channels].to_numpy(dtype=np.float64)
        time_values = df[time_col].to_numpy() if time_col and time_col in df.columns else None

        # PeacoQC API varies by version; attempt common signatures
        if hasattr(peacoqc, "PeacoQC"):
            result = peacoqc.PeacoQC(
                df[channels] if time_col is None else df[[time_col] + channels],
                channels=channels,
                time_channel=time_col,
                determine_good_cells=True,
            )
            if hasattr(result, "good_cells"):
                mask = np.array(result.good_cells, dtype=bool)
                logger.info("[QC/PeacoQC] %d / %d events pass.", mask.sum(), len(mask))
                return mask
        raise RuntimeError("PeacoQC result format unrecognized — falling back.")

    except Exception as exc:  # noqa: BLE001
        logger.warning("[QC] PeacoQC failed (%s) — using fallback QC.", exc)
        data = df[channels].to_numpy(dtype=np.float64)
        time_values = df[time_col].to_numpy() if time_col and time_col in df.columns else None
        return _fallback_qc(data, channels, time_values, p)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class QCStrategy(BaseAnalysisStrategy):
    """
    Flow cytometry QC strategy.

    Stores a boolean mask in sample.masks[p.mask_name]:
      True  = event passes QC
      False = event flagged as anomalous

    QC summary is stored in sample.results["qc"]:
      {
        "n_events": int,
        "n_pass": int,
        "n_fail": int,
        "pass_fraction": float,
        "method": str,
        "channels_monitored": List[str],
      }
    """

    name: str = "qc"

    def run(self, sample: Sample, **kwargs: Any) -> Sample:
        p: QCParams = kwargs.get("params", QCParams())
        if isinstance(p, dict):
            p = QCParams(**p)

        used_channels = p.channels or sample.channels
        missing = sorted(set(used_channels) - set(sample.events.columns))
        if missing:
            raise ValueError(f"[QC] Unknown channels requested: {missing}")

        all_cols = list(sample.events.columns)

        # Resolve time channel
        time_col = p.time_channel or _detect_time_channel(all_cols)

        # QC runs on full data (no gating — we want to flag instrument artifacts
        # before biological gating).
        df = sample.events[used_channels].copy()
        n_events = len(df)

        if n_events == 0:
            raise ValueError(f"[QC] Sample '{sample.sample_id}': 0 events — cannot run QC.")

        logger.info(
            "[QC] '%s': %d events, %d channels, time_col=%s, method=%s.",
            sample.sample_id,
            n_events,
            len(used_channels),
            time_col,
            "peacoqc" if (_PEACOQC_AVAILABLE and p.use_peacoqc) else "fallback",
        )

        # Run QC
        if _PEACOQC_AVAILABLE and p.use_peacoqc:
            qc_mask = _peacoqc_wrapper(sample.events, used_channels, time_col, p)
        else:
            data = df.to_numpy(dtype=np.float64)
            time_values = (
                sample.events[time_col].to_numpy()
                if time_col and time_col in sample.events.columns
                else None
            )
            qc_mask = _fallback_qc(data, used_channels, time_values, p)

        # Shape validation
        if qc_mask.shape[0] != n_events:
            raise RuntimeError(f"[QC] Mask length {qc_mask.shape[0]} ≠ n_events {n_events}.")

        qc_mask = qc_mask.astype(bool)
        n_pass = qc_mask.sum()
        pass_fraction = n_pass / n_events

        if pass_fraction < p.min_pass_fraction:
            logger.warning(
                "[QC] '%s': only %.1f%% events pass QC (threshold=%.1f%%). "
                "Check instrument stability or QC parameters.",
                sample.sample_id,
                100.0 * pass_fraction,
                100.0 * p.min_pass_fraction,
            )

        # Inject into sample
        sample.set_mask(p.mask_name, qc_mask)

        method_used = "peacoqc" if (_PEACOQC_AVAILABLE and p.use_peacoqc) else "fallback_mad"
        sample.results["qc"] = {
            "n_events": n_events,
            "n_pass": int(n_pass),
            "n_fail": int(n_events - n_pass),
            "pass_fraction": float(pass_fraction),
            "method": method_used,
            "channels_monitored": used_channels,
            "time_channel": time_col,
            "mad_threshold": p.mad_threshold,
        }

        logger.info(
            "[QC] '%s': %d / %d pass (%.1f%%) [%s].",
            sample.sample_id,
            n_pass,
            n_events,
            100.0 * pass_fraction,
            method_used,
        )

        return sample
