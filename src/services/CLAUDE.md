# src/services — Scope

Orchestration layer. Coordinates calls between core, io, and analysis modules.

## Contents
- `preprocessing_service.py` — QC → Gating → Transform → Normalize pipeline
- `clustering_service.py` — marker selection → stacking → Harmony Partial → FlowSOM
- `export_service.py` — orchestrates all export formats
- `population_mapping_service.py` — LAIP reference alignment service

## Rules
- Services call into core/, analysis/, io/, and visualization/ — never implement logic directly.
- Harmony must only correct scatter channels (FSC-A, SSC-A, CD45-A); assert this in code.
- Services are stateless — receive inputs, return outputs, no global mutation.
- Preprocessing order is fixed: QC → 4 biological gates → Logicle transform → Z-score normalize.
- Do not add new preprocessing steps without updating `config/default_config.yaml`.
