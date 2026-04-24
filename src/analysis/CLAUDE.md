# src/analysis — Scope

Scientific and clinical business logic. Highest correctness requirement in the codebase.

## Contents
- `mrd_calculator.py` — three MRD methods: JF (conservative), Flo (ratio-tolerant), ELN DfN (standard)
- `blast_detection.py` — phenotypic blast scoring (Ogata + Mahalanobis distance)
- `population_mapping.py` — LAIP reference alignment
- `prescreening.py` — CD34+/CD45dim pre-screening heuristic
- `statistics.py` — Mann-Whitney U, KS test, statistical utilities

## Rules
- No plotting or UI logic in this folder.
- Any threshold, heuristic, or default parameter must be documented in code (why it exists).
- Functions must be testable without notebooks or a running pipeline.
- Expose deterministic functions wherever possible (explicit seed, no global state).
- If behavior changes, describe the expected clinical impact in the PR/commit message.
- ELN 2022 invariants (LOD, LOQ, fold-change, event minimum) must not be changed without justification.
- New methods need a unit test before merging.

## Key invariants
- MRD positivity: node MRD% ≥ 0.1% AND events ≥ 50 AND fold-change ≥ 1.9×
- NBM node frequency cap: 1.1% (prevents normal HSC from flagging as MRD)
- Mahalanobis distance computed in transformed space (post-Logicle)
