# src/core — Scope

Low-level, stateless algorithms. No clinical thresholds hardcoded here — read from config.

## Contents
- `auto_gating.py` — adaptive 4-gate pipeline (GMM, KDE, RANSAC): viability → singlets → CD45 → CD34
- `gating.py` — manual percentile gating
- `clustering.py` — FlowSOM SOM + Numba JIT, GPU/CPU fallback
- `metaclustering.py` — optimal k via silhouette + stability
- `transformers.py` — Logicle, ArcSinh, Log10
- `normalizers.py` — Z-score, MinMax

## Rules
- All functions must be pure and deterministic (seed-fixed).
- No pandas/AnnData imports — work on numpy arrays only.
- No plotting, no logging, no config loading.
- Never loop over individual cells — vectorize.
- GPU path must have a CPU fallback with identical output.
- Logicle params (T, M, W, A) always passed explicitly, never defaulted silently.
