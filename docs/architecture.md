# PRISMA — Architecture détaillée

> Ce fichier est hors-contexte par défaut. Le charger manuellement quand on travaille sur l'architecture globale.

## Flux de données complet

```
FCS files (healthy + patho)
    └─▶ fcs_reader.py          [io/]         — discovery, loading, $SPILL check
    └─▶ preprocessing_service  [services/]
            ├─▶ auto_gating    [core/]       — viability → singlets → CD45 → CD34
            ├─▶ transformers   [core/]       — Logicle (T=2^18, M=4.5, W=0.5, A=0)
            └─▶ normalizers    [core/]       — Z-score
    └─▶ clustering_service     [services/]
            ├─▶ marker_harmonizer [utils/]   — channel name normalization
            ├─▶ class_balancer    [utils/]   — stratified downsampling
            ├─▶ Harmony Partial              — scatter only (FSC-A, SSC-A, CD45-A)
            └─▶ clustering     [core/]       — FlowSOM 10×10 SOM, metaclustering
    └─▶ mrd_calculator         [analysis/]
            ├─▶ Method JF      — conservative (node% normal < 0.1% AND patho > 10%)
            ├─▶ Method Flo     — ratio (patho/sain > 5.0×, topological shield)
            └─▶ Method ELN DfN — standard (patho% > sain% AND ≥50 events AND ≥0.1%)
    └─▶ blast_detection        [analysis/]   — Ogata score + Mahalanobis distance
    └─▶ Expert curation        [gui/]        — node selection, MRD recalculation, audit
    └─▶ export_service         [services/]
            ├─▶ fcs_writer     — FCS + Is_MRD column (Kaluza-compatible)
            ├─▶ csv_exporter   — cell data, MFI, cluster statistics
            ├─▶ json_exporter  — gating log, metadata snapshot
            ├─▶ html_report    — self-contained interactive Plotly
            └─▶ pdf_report     — clinical audit PDF (ReportLab)
```

## Structure AnnData (matrice principale)

```
adata.X     — transformed + normalized expression (cells × markers), float32
adata.obs   — cell metadata:
                condition, gate_status, FlowSOM_cluster, FlowSOM_metacluster,
                is_mrd, mrd_pct_jf, mrd_pct_flo, mrd_pct_eln, blast_score
adata.var   — marker metadata: marker_name, exclude_flag
adata.uns   — global: config_snapshot, gating_log, mrd_result, nbm_reference
adata.raw   — pre-transformation raw intensities (for FCS re-export)
```

## Populations cliniques clés

| Population | Marqueurs | Seuil |
|---|---|---|
| LSC | CD34+/CD38−/CD123+ | Score Ogata ≥ threshold |
| LAIP | Aberrant vs NBM | DfN Z-score ≥ 1.9× fold |
| NBM | ≥15 donors pooled | Fréquence max 1.1% |
| Blasts (BLAST_HIGH) | CD34++/CD117+/HLA-DR+/CD45dim | Mahalanobis + Ogata |

## Tubes ALFA standardisés

| Tube | Marqueurs | Usage |
|---|---|---|
| Tube 1 | LAIP panel | Diagnostic + suivi MRD |
| Tube 2 | LSC panel (CD45RA, CD90, TIM3, CLL-1, CD97, GPR56) | Suivi LSC |
| Tube 3 | Monocyte panel | Contrôle qualité |

## Harmony Partial — rationnel

La correction Harmony complète appliquée à tous les marqueurs fusionne les blastes
leucémiques avec les HSC en régénération (effet "biological erasure").
Solution retenue : corriger uniquement les scatter (FSC-A, SSC-A, CD45-A) pour
normaliser les variations inter-lots sans effacer la signature tumorale.
Ce choix est appelé "Biological Locking" dans le code.

## Human-in-the-loop — flux

1. Pipeline propose N nœuds FlowSOM comme MRD-positifs
2. Clinicien visualise le radar Z-score de chaque nœud vs NBM
3. Clinicien sélectionne KEEP (MRD) ou DISCARD (bruit normal)
4. MRD% recalculé en temps réel
5. Décision tracée dans GatingLogger → rapport PDF final

## Décisions architecturales majeures

1. **ML supervisé abandonné** : l'hétérogénéité des LAIP rend impossible la généralisation
2. **DfN retenu** : détection d'anomalie topologique (FlowSOM) + référence NBM frozen
3. **Harmony Partial** : compromis correction batch / préservation signal tumoral
4. **3 méthodes MRD parallèles** : consensus clinique, désaccord = flag pour revue experte
5. **PyQt5 desktop** : contrainte clinique (pas de cloud, données patient locales)
