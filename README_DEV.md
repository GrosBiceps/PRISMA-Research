# README DEV — PRISMA Research MVP

## 1) Arborescence finale recommandée (MVP)

```text
PRISMA Research/
├── config/
│   ├── default_config.yaml
│   ├── mrd_config.yaml
│   └── panels/
├── src/
│   ├── prisma/
│   │   ├── core/
│   │   ├── pipeline/
│   │   ├── strategies/
│   │   ├── analysis/
│   │   ├── exports/                  # Bridge vers moteur d'exports
│   │   ├── io/
│   │   ├── gui/
│   │   └── utils/
│   ├── exports/                      # Nouveau moteur analytique + sérialisation
│   │   ├── export_manager.py
│   │   ├── heatmaps.py
│   │   └── run_serializer.py
│   ├── analysis/                     # Legacy (maintenance minimale)
│   ├── pipeline/                     # Legacy
│   └── ...
├── tests/
│   ├── prisma/
│   │   ├── smoke_test.py
│   │   ├── test_models_and_interfaces.py
│   │   └── test_exports_and_runner.py
│   └── ...
├── docs/
├── gui/
└── outputs/                          # Artefacts de run (runtime)
```

## 2) Dépendances Python recommandées

### Noyau MVP
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn
- pyyaml
- anndata

### Clustering / cytométrie
- flowsom
- flowio
- fcswrite
- pytometry
- umap-learn

### Exports et reproductibilité
- pyarrow (recommandé pour parquet)
- reportlab (PDF)
- plotly + kaleido (figures statiques)

### UI (si GUI desktop activée)
- PyQt5, PyQtWebEngine, qtawesome

## 3) Ordre conseillé d'intégration réel

1. Verrouiller `config/default_config.yaml` et la validation de paramètres.
2. Stabiliser `src/prisma/core` (Sample, Experiment, RunMetadata).
3. Stabiliser `src/prisma/strategies` + registre `StrategyRegistry`.
4. Valider le pipeline minimal `PipelineRunner` en mode headless.
5. Intégrer `src/exports/*` (tables, heatmaps, sérialisation).
6. Brancher l'export manager dans les jobs batch/GUI.
7. Ajouter les tests de non-régression sur parcours complet.
8. Packager (wheel + script CLI) et figer une release candidate.

## 4) Dettes techniques restantes

- Unifier les couches legacy et PRISMA pour éviter la double maintenance.
- Ajouter des schémas stricts (pydantic/jsonschema) pour config et run bundles.
- Introduire une convention unique de nommage des artefacts.
- Améliorer la couverture de tests sur les stratégies optionnelles (GPU/flowsom absent).
- Séparer clairement les tests RUO exploratoires des tests release gating.

## 5) Prochaines priorités produit

1. Export bundle signé (hash SHA256) pour audit trail.
2. Rapport HTML/PDF standardisé par run (QC + clusters + heatmaps).
3. Validation automatique ELN invariants depuis config.
4. Pipeline batch multi-cohortes avec reprise sur incident.
5. Dashboard QA pour suivi longitudinal des performances.

## Checklist validation MVP

- [ ] `pytest -x tests/prisma/smoke_test.py`
- [ ] `pytest -q tests/prisma/test_exports_and_runner.py`
- [ ] Exécution headless d'un run de bout en bout
- [ ] Génération de `run_bundle.json` + `config_snapshot.json`
- [ ] Export des tables CSV (+ parquet si pyarrow)
- [ ] Génération d'au moins une heatmap PNG
- [ ] Vérification manuelle des artefacts dans `outputs/<run_id>/`

## Plan de durcissement

### v0.2 (stabilisation technique)
- Validation stricte des entrées/sorties des modules d'exports.
- Contrats de compatibilité de schémas JSON versionnés.
- Tests d'intégration multi-OS (Windows/Linux).
- Gestion robuste des dépendances optionnelles (messages explicites).

### v0.3 (pré-industrialisation)
- Packaging reproductible (lockfile + build déterministe).
- Artefacts signés et vérification d'intégrité.
- Profiling performance sur cohortes volumineuses.
- Monitoring qualité run-to-run (drift des clusters/MFI).
