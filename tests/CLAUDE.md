# tests/ — Scope

Unit and integration tests. Run with `pytest -q`.

## Structure
- `test_mrd_calculator.py` — 3 MRD method correctness (JF, Flo, ELN DfN)
- `test_clustering_utils.py` — FlowSOM + metaclustering
- `test_gating.py` — auto-gating pipeline (GMM/KDE/RANSAC)

## Rules
- Unit tests for pure transformations (core/, analysis/): fast, no FCS files needed.
- Integration tests for end-to-end pipeline paths: use fixture FCS files from `data_examples/`.
- Regression tests for fixed clinical bugs: include the ELN threshold that was violated.
- GUI workers tested via signal mocking — do not launch the full Qt app in tests.
- Never mock the actual scientific algorithms (FlowSOM, Logicle) — use real computation.
- Each test must be reproducible: fix seed=42, use deterministic fixtures.
- Test only the behavior modified by the current change.
- ELN invariants (LOD, LOQ, fold-change) must have dedicated regression tests.
