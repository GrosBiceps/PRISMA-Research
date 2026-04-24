# src/pipeline — Scope

Main executor: coordinates all services across the 7 pipeline phases.

## 7 phases (in order)
1. Loading — FCS discovery and FlowSample construction
2. Preprocessing — QC, gating, transform, normalize (via preprocessing_service)
3. Clustering — FlowSOM + metaclustering (via clustering_service)
4. MRD Scoring — 3 parallel methods (via analysis/mrd_calculator)
5. Visualization — SOM grids, radar plots, heatmaps (via visualization/)
6. Export — FCS, CSV, JSON, HTML, PDF (via export_service)
7. Population Mapping — optional LAIP reference alignment

## Rules
- The executor calls services only — no direct algorithm calls.
- Phase order is invariant; do not reorder without updating tests and docs.
- Each phase result is stored in `PipelineResult` (src/models/pipeline_result.py).
- Checkpoints must be saved between phases to allow resume on failure.
- Batch mode (`batch_pipeline.py`) reuses the same 7 phases per sample.
- NBM cache (`nbm_cache_manager.py`) is frozen per ELN — do not invalidate silently.
