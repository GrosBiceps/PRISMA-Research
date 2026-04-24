"""
smoke_spectral_qc.py — Smoke test for SpectralUnmixingStrategy + QCStrategy.

Generates synthetic flow cytometry data and exercises the full pipeline:
  1. QC  →  qc_pass mask injected into Sample
  2. Spectral unmixing → virtual channels injected into Sample
  3. Downstream UMAP (optional)

Run:
    cd "PRISMA Research"
    python tests/smoke_spectral_qc.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---- path setup --------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Stubs legacy requis par analysis/__init__.py quand exécuté hors pytest.
import tests.conftest as _stubs  # noqa: F401

from models.sample import Sample
from analysis.qc import QCStrategy, QCParams
from analysis.spectral import (
    SpectralUnmixingStrategy,
    SpectralUnmixingParams,
    AutofluorescenceConfig,
)
from analysis.strategies import (
    PreGatingParams,
    PreGatingStrategy,
    ResearchPipelineExecutor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("smoke_test")

# =========================================================================
# 1. Synthetic data generation
# =========================================================================

RNG = np.random.default_rng(42)

N_EVENTS = 5_000
N_DETECTORS = 8  # simulated spectral detectors
N_FLUOROCHROMES = 4  # PE, APC, FITC, BV421
FLUOROCHROME_NAMES = ["PE", "APC", "FITC", "BV421"]


def make_reference_matrix(n_detectors: int, n_fluoro: int) -> np.ndarray:
    """
    Synthetic single-stain reference matrix (n_detectors × n_fluoro).

    Each column is a normalized unit-norm spectrum drawn from a random
    positive Gaussian blob covering a few detectors.
    """
    ref = np.zeros((n_detectors, n_fluoro))
    centers = np.linspace(0, n_detectors - 1, n_fluoro)
    for j, c in enumerate(centers):
        x = np.arange(n_detectors)
        col = np.exp(-0.5 * ((x - c) / 1.2) ** 2)
        col /= np.linalg.norm(col)
        ref[:, j] = col
    return ref.astype(np.float64)


def make_synthetic_events(
    ref: np.ndarray,
    n_events: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Generate N_EVENTS measured spectra as linear mixture + Poisson noise.

    Returns (DataFrame with detector + Time columns, true_abundances array).
    """
    n_detectors, n_fluoro = ref.shape

    # True abundances per event (positive)
    true_abundances = rng.exponential(scale=1_000.0, size=(n_events, n_fluoro))

    # Mixed signal
    mixed = true_abundances @ ref.T  # (n_events × n_detectors)

    # Add noise
    noise = rng.normal(0, 50.0, size=mixed.shape)
    mixed = np.clip(mixed + noise, 0, None).astype(np.float32)

    # Build DataFrame
    det_names = [f"Det_{i}" for i in range(n_detectors)]
    df = pd.DataFrame(mixed, columns=det_names)

    # Time channel (monotone + one instability burst in the middle)
    time = np.linspace(0, 1000.0, n_events)
    # Simulate a 10% drift in detectors 2-4 in events 2000-2500
    drift_mask = (np.arange(n_events) >= 2000) & (np.arange(n_events) < 2500)
    for d in ["Det_2", "Det_3", "Det_4"]:
        df.loc[drift_mask, d] *= 5.0  # ← instrument instability artifact
    df["Time"] = time

    return df, true_abundances


def make_autofluorescence_spectrum(n_detectors: int) -> np.ndarray:
    """Synthetic broad AF spectrum (mostly low channels)."""
    x = np.arange(n_detectors)
    af = np.exp(-0.5 * ((x - 1.0) / 2.0) ** 2)
    af /= np.linalg.norm(af)
    return af.astype(np.float64)


# =========================================================================
# 2. Build Sample
# =========================================================================

ref_matrix = make_reference_matrix(N_DETECTORS, N_FLUOROCHROMES)
events_df, true_abundances = make_synthetic_events(ref_matrix, N_EVENTS, RNG)
af_spectrum = make_autofluorescence_spectrum(N_DETECTORS)

det_channels = [f"Det_{i}" for i in range(N_DETECTORS)]

sample = Sample(
    sample_id="smoke_test_001",
    events=events_df,
)

logger.info("Sample created: %d events × %d channels.", sample.n_events, len(sample.channels))

# =========================================================================
# 3. QC Strategy
# =========================================================================

qc_params = QCParams(
    channels=det_channels,
    time_channel="Time",
    n_bins=50,
    mad_threshold=5.0,
    min_events_per_bin=20,
    mask_name="qc_pass",
    use_peacoqc=False,  # force fallback for portability
)

qc_strategy = QCStrategy()
sample = qc_strategy.run(sample, params=qc_params)

qc_result = sample.results["qc"]
logger.info(
    "QC done: %d / %d pass (%.1f%%) [method=%s]",
    qc_result["n_pass"],
    qc_result["n_events"],
    100.0 * qc_result["pass_fraction"],
    qc_result["method"],
)

# Validate: the drift block (events 2000-2500) should be mostly flagged
qc_mask = sample.get_mask("qc_pass")
drift_idx = np.arange(2000, 2500)
drift_fail_rate = (~qc_mask[drift_idx]).mean()
logger.info(
    "Drift block fail rate: %.1f%% (expected high — artifact injected).",
    100.0 * drift_fail_rate,
)

assert "qc_pass" in sample.masks, "QC mask not found in sample.masks"
assert sample.masks["qc_pass"].shape[0] == N_EVENTS, "QC mask wrong length"

# =========================================================================
# 4. Spectral Unmixing Strategy
# =========================================================================

spectral_params = SpectralUnmixingParams(
    reference_matrix=ref_matrix,
    fluorochrome_names=FLUOROCHROME_NAMES,
    channels=det_channels,
    autofluorescence=AutofluorescenceConfig(enabled=True, channel_name="AF"),
    af_spectrum=af_spectrum,
    unmixed_prefix="unmixed",
    max_events=100_000,
)

spectral_strategy = SpectralUnmixingStrategy()
sample = spectral_strategy.run(sample, params=spectral_params)

spectral_meta = sample.results["spectral_unmixing"]
logger.info(
    "Unmixing done: %d virtual channels (AF_enabled=%s).",
    len(spectral_meta["fluorochrome_names"]),
    spectral_meta["af_enabled"],
)

# Validate virtual channels exist
expected_channels = [f"unmixed_{f}" for f in FLUOROCHROME_NAMES] + ["unmixed_AF"]
for vch in expected_channels:
    assert vch in sample.virtual_channels, f"Missing virtual channel: {vch}"
    assert sample.virtual_channels[vch].shape[0] == N_EVENTS, f"{vch} wrong length"

# Quality check: correlation between unmixed PE and true PE abundance
unmixed_PE = sample.virtual_channels["unmixed_PE"]
valid_idx = ~np.isnan(unmixed_PE)
corr = np.corrcoef(unmixed_PE[valid_idx], true_abundances[valid_idx, 0])[0, 1]
logger.info("Pearson r(unmixed_PE, true_PE) = %.4f (expect > 0.9).", corr)
assert corr > 0.9, f"NNLS correlation too low: {corr:.4f}"

# =========================================================================
# 5. Pipeline integration (QC → Spectral → UMAP if available)
# =========================================================================

sample2 = Sample(
    sample_id="pipeline_test_001",
    events=events_df.copy(),
)

executor = ResearchPipelineExecutor()
executor.register(QCStrategy())
executor.register(SpectralUnmixingStrategy())
executor.register(PreGatingStrategy())

try:
    from analysis.strategies import UMAPStrategy, UMAPStrategyParams

    executor.register(UMAPStrategy())
    umap_params = UMAPStrategyParams(
        channels=det_channels,
        n_neighbors=15,
        min_dist=0.1,
        max_events=2_000,
        embedding_name="umap_2d",
    )
    run_umap = True
    logger.info("UMAPStrategy registered.")
except Exception:
    run_umap = False
    umap_params = None

kwargs: dict = {
    "qc_params": qc_params,
    "spectral_unmixing_params": spectral_params,
    "pre_gating_params": PreGatingParams(
        source_masks=["qc_pass"],
        output_mask_name="pre_gating",
    ),
}
if run_umap and umap_params is not None:
    kwargs["umap_params"] = umap_params

results = executor.run([sample2], **kwargs)
sample2 = results[0]

assert "qc_pass" in sample2.masks
assert "pre_gating" in sample2.masks
assert "unmixed_PE" in sample2.virtual_channels

if run_umap:
    assert "umap_2d" in sample2.embeddings
    logger.info("UMAP embedding shape: %s", sample2.embeddings["umap_2d"].shape)

# =========================================================================
# 6. Summary
# =========================================================================

logger.info("")
logger.info("=" * 60)
logger.info("SMOKE TEST PASSED")
logger.info("  QC mask:          %s events pass / %s", qc_result["n_pass"], N_EVENTS)
logger.info("  Virtual channels: %s", list(sample.virtual_channels.keys()))
logger.info("  Corr(PE):         %.4f", corr)
logger.info(
    "  Pipeline:         QC + Spectral + Pre-gating + %s",
    "UMAP" if run_umap else "no UMAP",
)
logger.info("=" * 60)
