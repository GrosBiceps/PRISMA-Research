# PRISMA — Persistent project memory

## Architecture decisions
- Analytical logic lives in `src/analysis/`, never in GUI or notebooks
- Plotting code lives in `src/visualization/`, never computes scientific results
- GUI layer (`gui/`) is orchestration-only — thin wrappers around services
- `notebooks/` are exploratory only, not the source of truth for any algorithm
- All parameters (MRD thresholds, gating, FlowSOM) live in `config/*.yaml`
- Harmony is applied to scatter only (FSC-A, SSC-A, CD45-A) — "Biological Locking"
- Three MRD methods run in parallel: JF (conservative), Flo (ratio-tolerant), ELN DfN (standard)

## Known constraints
- Never apply Harmony to lineage markers (CD34, CD117, HLA-DR) — fuses blasts with HSC
- FSC/SSC/Time channels must be excluded from FlowSOM input matrix
- Logicle transform must precede any FlowSOM or normalization step
- NaN in AnnData.X causes silent FlowSOM failures — validate before analysis
- FlowSOM is stochastic — always fix seed=42 for reproducibility
- Compensation matrix ($SPILL in FCS header) must be applied before analysis
- Downsampling must be stratified to preserve rare blast populations

## Testing policy
- Unit tests for pure transformations (core/, analysis/)
- Integration tests for end-to-end pipeline paths
- Regression tests for fixed clinical bugs (MRD thresholds, gating logic)
- GUI workers tested via signal mock, not by launching the full Qt app

## ELN 2022 standards (frozen)
- LOD 0.009%, LOQ 0.005% (50-event minimum per node)
- NBM reference: ≥15 pooled donors, SOM frozen per ELN
- Clinical positivity: MRD ≥ 0.1%
- Fold-change: patient node% / NBM node% ≥ 1.9×

## Key pivots (retrospective)
- Abandoned supervised ML: LAIP heterogeneity defies universal classification
- Adopted Harmony Partial: full Harmony erased tumor signal (biological erasure problem)
- Added human-in-the-loop: algorithm proposes nodes, clinician curates, audit trail generated
