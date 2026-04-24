# PRISMA Research — Claude working rules

## Mission
PRISMA Research est une plateforme RUO d'analyse de cytométrie en flux et spectrale.
Deux couches coexistent dans ce dépôt :
- **Couche legacy** (`src/` hors `src/prisma/`) : fork monolithique flowsom_pipeline_pro — modifier avec parcimonie.
- **Couche PRISMA** (`src/prisma/`) : nouvelle architecture modulaire — c'est ici que tout le nouveau code va.

## Stack
- Python 3.10+, PySide6, FlowSOM, AnnData, Harmony, scipy, umap-learn, openTSNE
- Tests: pytest — `pytest -x tests/prisma/smoke_test.py`
- Lint: ruff — `ruff check src/prisma/`
- Format: black — `black src/prisma/`
- Typecheck: mypy — `mypy src/prisma/`

## Repository map — couche PRISMA (src/prisma/)
- `src/prisma/core/` — modèles (Sample, Experiment, RunMetadata), SessionManager, StrategyRegistry
- `src/prisma/pipeline/` — PipelineStep (Protocol), PipelineRunner, QC, pré-gating, transforms, batch
- `src/prisma/strategies/` — UMAP, t-SNE, PHATE, FlowSOM, Spectral, Clustering (Protocol DimReducStrategy / ClusterStrategy)
- `src/prisma/analysis/` — stats populations, heatmaps, différentiel
- `src/prisma/exports/` — CSV, FCS, PNG, nommage explicite
- `src/prisma/gui/` — PySide6, découplé via SessionManager et Qt signals
- `src/prisma/io/` — wrapper FCS reader/writer
- `src/prisma/utils/` — logging structuré, seed, versioning

## Repository map — couche legacy (src/)
- `src/core/` — algorithmes bas niveau (gating, clustering, transforms, normalizers)
- `src/analysis/` — MRD scoring, blast detection, population mapping
- `src/services/` — orchestration (preprocessing, clustering, export)
- `src/pipeline/` — exécuteur 7-phases, batch mode
- `src/io/` — FCS/CSV/JSON read and write
- `src/visualization/` — plots, radar charts, HTML/PDF reports
- `src/models/` — FlowSample, PipelineResult, GateResult
- `src/utils/` — logging, validation, marker harmonization
- `gui/` — PyQt5 UI legacy

## Global rules
- Nouveau code → `src/prisma/` uniquement.
- Legacy (`src/` hors prisma) : changer surface minimale, garder APIs stables.
- Jamais de seuils cliniques hardcodés — tout dans `config/`.
- Fonctions pures, déterministes, vectorisées (jamais de for-loop sur FCS rows).
- Chaque run génère un RunMetadata sérialisé dans `outputs/<run_id>/metadata.json`.
- Fallbacks CPU explicites pour toute dépendance GPU.
- Imports absolus `from prisma.core.models import Sample`.
- Logging via `logging.getLogger(__name__)` — jamais print().

## ELN 2022 invariants (ne pas modifier sans justification scientifique)
- LOD: 0.009% (9e-5)
- LOQ: 0.005% (5e-5) — hard stop à 50 events/node
- NBM max frequency: 1.1%
- Fold-change threshold: 1.9× (patient vs NBM)
- Logicle params: T=2^18, M=4.5, W=0.5, A=0
- Harmony: corriger scatter uniquement (FSC-A, SSC-A, CD45-A)

## Response format
- Plan: 3–5 bullets avant tout code
- Arborescence diff en tête de réponse
- Code complet — jamais de `...` ni `# reste inchangé`
- Fichiers modifiés listés en fin de réponse
- Risques scientifiques ou cliniques si applicable

## Global rules
- Change the smallest possible surface.
- Do not refactor unrelated modules.
- Keep public APIs stable unless explicitly asked.
- Never hardcode clinical thresholds — all parameters live in `config/`.
- Prefer pure, deterministic functions with explicit parameters.
- Business logic stays outside GUI, notebooks, and export layers.
- Add or update only the tests relevant to the modified behavior.
- Any new clinical parameter must also appear in `config/default_config.yaml`.
- Vectorize over cells — never `for` loops on FCS rows (200k–1M+ cells).

## ELN 2022 invariants (do not change without scientific justification)
- LOD: 0.009% (9e-5) — detection limit
- LOQ: 0.005% (5e-5) — quantification limit; hard stop at 50 events/node
- NBM max frequency: 1.1%
- Fold-change threshold: 1.9× (patient vs NBM)
- Logicle params: T=2^18, M=4.5, W=0.5, A=0
- Harmony: correct scatter only (FSC-A, SSC-A, CD45-A) — never lineage markers

## Response format
- Plan: 3–5 bullets before any code
- Changes: minimal diff, commented in French (why, not what)
- Files changed: list at end
- Risks: scientific or clinical risks if any
