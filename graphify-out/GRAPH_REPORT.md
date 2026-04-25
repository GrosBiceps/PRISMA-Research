# Graph Report - PRISMA Research  (2026-04-25)

## Corpus Check
- 184 files · ~280,593 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 3300 nodes · 10771 edges · 74 communities detected
- Extraction: 44% EXTRACTED · 56% INFERRED · 0% AMBIGUOUS · INFERRED: 6065 edges (avg confidence: 0.6)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]

## God Nodes (most connected - your core abstractions)
1. `PipelineConfig` - 230 edges
2. `Sample` - 216 edges
3. `Experiment` - 171 edges
4. `ToggleSwitch` - 140 edges
5. `FlowSomAnalyzerPro` - 139 edges
6. `PipelineResult` - 137 edges
7. `path()` - 128 edges
8. `LogConsole` - 124 edges
9. `ExportService` - 117 edges
10. `FlowSample` - 108 edges

## Surprising Connections (you probably didn't know these)
- `Point d'entrée principal de la CLI.      Charge la configuration, exécute le pip` --uses--> `PipelineConfig`  [INFERRED]
  cli\main.py → config\pipeline_config.py
- `Construit le PipelineConfig, puis applique les surcharges CLI.` --uses--> `PipelineConfig`  [INFERRED]
  cli\main.py → config\pipeline_config.py
- `Charge un fichier YAML et retourne un dict plat de clés argparse.      Toutes le` --uses--> `PipelineConfig`  [INFERRED]
  cli\main.py → config\pipeline_config.py
- `Résolution de priorité CLI → YAML → défaut.      Retourne la valeur CLI si non-N` --uses--> `PipelineConfig`  [INFERRED]
  cli\main.py → config\pipeline_config.py
- `§10.4c — Scoring ELN 2022 des nœuds Unknown.` --uses--> `PopulationMappingConfig`  [INFERRED]
  src\services\population_mapping_service.py → config\pipeline_config.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.01
Nodes (389): heatmap(), analyze_results.py Genere toutes les statistiques et heatmaps depuis results_gr, _marker_key(), _df_to_flow_samples(), compute_reference_stats(), Calcule les statistiques de la population de référence (moelle normale / NBM), trace_blast_cells_to_fcs_source(), _detect_patho_mask() (+381 more)

### Community 1 - "Community 1"
Cohesion: 0.01
Nodes (284): ABC, AdvancedCohortExecutor, _as_dict(), build_advanced_executor_from_config(), _build_dataclass_params(), CohortBaseParams, HarmonyStrategy, HarmonyStrategyParams (+276 more)

### Community 2 - "Community 2"
Cohesion: 0.01
Nodes (122): Connecte les signaux de chaque widget → save immédiat à chaque changement., _btn_primary_style(), _btn_secondary_style(), ExpertFocusDialog, ExpertNodeCard, _outfit(), Reconstruit la grille selon les filtres actifs., Charge `chunk` radars puis cède la main à l'event loop, jusqu'à épuisement. (+114 more)

### Community 3 - "Community 3"
Cohesion: 0.03
Nodes (269): AutoGating, Gating automatique adaptatif basé sur des modèles de mélange gaussien (GMM), BatchPipeline, _invalidate_som_checkpoint(), batch_pipeline.py — Orchestrateur du mode batch avec cache DATA NBM.  Objectif s, Lance le pipeline batch complet.          Args:             progress_callback: O, Retourne la liste triée des FCS dans le dossier NBM., Charge le cache NBM s'il existe, sinon le construit.          Returns: (+261 more)

### Community 4 - "Community 4"
Cohesion: 0.02
Nodes (174): ClusterParams, DimReducParams, Hyperparamètres communs aux stratégies de réduction dimensionnelle., Hyperparamètres communs aux stratégies de clustering., ClusterParams, DimReducParams, DownsampleResult, expand_to_full() (+166 more)

### Community 5 - "Community 5"
Cohesion: 0.05
Nodes (110): _BaseBinding, CheckBinding, ComboBinding, ConfigBinder, DoubleSpinBinding, _get_nested(), LineEditBinding, NClustBinding (+102 more)

### Community 6 - "Community 6"
Cohesion: 0.01
Nodes (113): constants.py — Constantes globales du pipeline FlowSOM.  Toutes les constantes m, FCSCanvas, FCSScene, FCSView, src/gui/viewer/fcs_canvas.py — Canvas haute performance pour visualisation FCS 2, QGraphicsScene portant la couche de points rasterisée et les gates overlay., Met à jour la couche rasterisée avec de nouvelles coordonnées pixel., Active le mode dessin de porte. (+105 more)

### Community 7 - "Community 7"
Cohesion: 0.02
Nodes (122): BaseFlowSOMEstimator, Base class for all FlowSOM estimators in FlowSOM., Initialize the FlowSOMEstimator object., Fit the model and predict the clusters., Predict the clusters., Set the number of clusters., BaseFlowSOMEstimator, compute_optimal_grid() (+114 more)

### Community 8 - "Community 8"
Cohesion: 0.05
Nodes (89): build_blast_score_dataframe(), build_blast_weights(), build_weight_vector(), _categorize_blast(), categorize_blast_score(), compute_reference_normalization(), load_panel_weights(), blast_detection.py — Scoring et classification phénotypique des nœuds SOM en bla (+81 more)

### Community 9 - "Community 9"
Cohesion: 0.03
Nodes (56): BaseClusterEstimator, Check fitted status and return a Boolean value., Base class for all cluster estimators in FlowSOM., BaseClusterEstimator, BaseEstimator, BatchFlowSOMEstimator, Initialize the FlowSOMEstimator object., A class that implements the FlowSOM model. (+48 more)

### Community 10 - "Community 10"
Cohesion: 0.04
Nodes (62): auto_gate_cd34(), auto_gate_cd34_cd45dim(), auto_gate_cd45(), auto_gate_debris(), _auto_gate_debris_kde(), auto_gate_singlets(), _export_gmm_density_plot(), GatingSession (+54 more)

### Community 11 - "Community 11"
Cohesion: 0.08
Nodes (47): _axes(), _c(), _dot(), _draw_alert_triangle(), _draw_arrow_left(), _draw_arrow_right(), _draw_batch_cohort(), _draw_cell_node() (+39 more)

### Community 12 - "Community 12"
Cohesion: 0.08
Nodes (32): _force_gc(), full_stress_matrix(), large_cytometry_matrix(), _log_gpu_memory(), tests/prisma/test_oom_protection.py — Test de stress anti-OOM pour torch_knn_ind, Sur CPU (ou si CUDA absent), torch_knn_indices doit fonctionner         sur un p, Vérifie que le calcul de taille de chunk auto ne produit pas         de valeur h, Matrice float32 simulant 500k cellules × 30 marqueurs (CI-friendly). (+24 more)

### Community 13 - "Community 13"
Cohesion: 0.11
Nodes (34): _build_styles(), _cb_landscape(), _cb_portrait(), _cover_page(), _data_table(), _draw_header_footer(), generate_pdf_report(), _kv_table() (+26 more)

### Community 14 - "Community 14"
Cohesion: 0.11
Nodes (27): ClusterStrategy, DimReducStrategy, Interface commune pour UMAP, t-SNE, PHATE, etc.      Chaque stratégie expose fit, Réduit data (n_cells × n_features) en embedding (n_cells × n_dims).          Arg, Interface commune pour PhenoGraph, Leiden, KMeans, FlowSOM.      Retourne un vec, Affecte chaque cellule à un cluster.          Args:             data: Matrice (n, Protocol, create_clustering() (+19 more)

### Community 15 - "Community 15"
Cohesion: 0.07
Nodes (25): get_logger(), logger.py — Logging structuré des événements de gating.  Fournit un enregistreme, Retourne un StreamHandler toujours valide, y compris en mode     PyInstaller --n, Retourne un logger nommé pour le module appelant., _safe_stream_handler(), apply_harmonization(), _extract_channel_suffix(), marker_harmonizer.py — Harmonisation des noms de marqueurs FCS inter-fichiers. (+17 more)

### Community 16 - "Community 16"
Cohesion: 0.11
Nodes (17): ExportManager, Gestionnaire d'exports analytiques (CSV, parquet, PNG, JSON)., Centralise les exports du pipeline vers un répertoire de run., Exporte en parquet si un moteur est disponible, sinon retourne None., Exporte un lot de tables en CSV + parquet optionnel., build_cluster_marker_matrix(), build_cohort_export_tables(), compute_cluster_abundance() (+9 more)

### Community 17 - "Community 17"
Cohesion: 0.23
Nodes (11): apply_along_axis_0(), nb_amax_axis_0(), nb_mean_axis_0(), nb_median_axis_0(), nb_std_axis_0(), Like calling func1d(arr, axis=0, out=out). Require arr to be 2d or bigger., Like calling np.mean(arr, axis=0)., Like calling np.std(arr, axis=0). (+3 more)

### Community 18 - "Community 18"
Cohesion: 0.21
Nodes (11): _fig_to_base64(), generate_html_report(), _get_plotlyjs_cached(), _plotly_to_html_div(), html_report.py — Génération d'un rapport HTML self-contained avec Plotly + Matpl, Génère un rapport HTML complet avec toutes les visualisations.      Le rapport e, Retourne le bundle plotly.js en le chargeant au plus une fois par session., Convertit une figure matplotlib en string base64 PNG. (+3 more)

### Community 19 - "Community 19"
Cohesion: 0.22
Nodes (8): _enable_windows_crisp_rendering(), _install_crash_guard(), _install_legacy_import_aliases(), launch_gui.py — Point d'entrée pour la compilation PyInstaller (GUI uniquement)., Assure la compatibilité des imports historiques `flowsom_pipeline_pro.*`     en, # NOTE: dossier parent (Perplexity/) volontairement exclu — évite de charger, Request per-monitor DPI awareness to avoid blurry bitmap-scaled UI on Windows., Installe sys.excepthook (thread principal) ET threading.excepthook (tous les

### Community 20 - "Community 20"
Cohesion: 0.25
Nodes (7): backend_report(), build_tsne(), build_umap(), analysis/backends.py — Factory CPU/GPU pour la réduction dimensionnelle.  Tente, Retourne un objet t-SNE (openTSNE ou sklearn).      Priorité : openTSNE (multith, Retourne un dict résumant les backends disponibles., Retourne un objet UMAP prêt à l'emploi (interface sklearn : fit_transform).

### Community 21 - "Community 21"
Cohesion: 0.47
Nodes (5): get_image(), Generate prisma_logo.ico (multi-resolution) from prisma_logo.svg. Run once from, Pure-Pillow fallback: draws the PRISMA prism icon programmatically     so we nev, render_fallback(), render_svg_cairosvg()

### Community 22 - "Community 22"
Cohesion: 0.4
Nodes (4): utils/logging_config.py — Configuration logging structuré PRISMA Research.  Appe, Configure le logging global.      Args:         level: Niveau de log (logging.DE, setup_logging(), test_logging_setup()

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): run_pipeline.py — Lanceur de développement (sans pip install).  Usage depuis n'i

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): setup.py — Packaging de flowsom_pipeline_pro.  Installation:     pip install -e

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Charge la configuration depuis un fichier YAML.          Args:             ya

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Construit un PipelineConfig depuis un dictionnaire YAML brut.

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Construit la configuration depuis des arguments CLI (argparse.Namespace).

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Convertit #RRGGBB â†’ 'R, G, B' pour usage dans rgba().

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): Nombre total de nodes dans la grille SOM.

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Trouve l'index d'un marqueur parmi les patterns donnés.

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Gate les cellules viables basé sur FSC/SSC.          Args:             X: Matric

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Gate les singlets basé sur le ratio FSC-A/FSC-H.         Les doublets ont typiqu

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Gate les cellules CD45+ (leucocytes).          Returns:             Masque boolé

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Gate les blastes CD34+ (cellules souches/progénitrices).          Les blastes so

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Gate rectangulaire/polygonal pour exclure les débris sur FSC-A vs SSC-A.

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Normalisation Z-score par marqueur (colonne).          Après normalisation: moye

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Normalisation Min-Max par marqueur vers [0, 1].          Les marqueurs constants

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Applique la normalisation spécifiée.          Args:             data: Matrice (n

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Transformation Arcsinh (inverse hyperbolic sine).          Args en entrée:

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Inverse de la transformation Arcsinh.

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Transformation Logicle (biexponentielle).          Args en entrée:             d

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Transformation logarithmique standard.

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Normalisation Z-score (moyenne=0, std=1).

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Normalisation Min-Max [0, 1].

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Applique la transformation spécifiée, avec gestion optionnelle         des canau

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Calcule le masque booléen des événements inclus dans cette gate.          Args:

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Retourne les sommets de la gate en coordonnées données.          Args:

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Reconstruit une PolygonGate depuis un dict (chargement session).          Args:

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Nombre d'événements positifs (sum du masque).

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Taille totale du masque (N_events du Sample).

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Pourcentage d'événements positifs dans la population source.

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Pourcentage de cellules conservées (0–100).

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Nombre de cellules exclues par ce gate.

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Pourcentage de cellules exclues (0–100).

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Qualité basique du gate : True si > 20% des cellules ont été conservées.

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): Nombre total de cellules dans le résultat final.

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): Nombre de métaclusters produits.

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): True si le pipeline s'est terminé avec des données valides.

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): Construit un PipelineResult représentant un échec.          Args:             er

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Nombre d'événements (cellules) dans le Sample.

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Noms des canaux primaires (colonnes de events).

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): Construit un FlowSample depuis un objet AnnData (flowsom.io.read_FCS).

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): Remplace l'alpha d'une couleur rgba(r,g,b,a) par new_alpha.

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (1): Définit la préférence GPU. Appelé par la GUI avant le lancement.

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (1): Retourne True si l'accélération GPU est autorisée.

### Community 71 - "Community 71"
Cohesion: 1.0
Nodes (1): Crée un RunMetadata au lancement d'un run.

### Community 74 - "Community 74"
Cohesion: 1.0
Nodes (1): Return the codes, shaped: (n_clusters, n_features).

### Community 75 - "Community 75"
Cohesion: 1.0
Nodes (1): Return the distances.

### Community 76 - "Community 76"
Cohesion: 1.0
Nodes (1): Return the cluster labels.

### Community 77 - "Community 77"
Cohesion: 1.0
Nodes (1): Return the metacluster labels.

### Community 82 - "Community 82"
Cohesion: 1.0
Nodes (1): Charge un log de gating depuis un fichier JSON.          Args:             path:

### Community 84 - "Community 84"
Cohesion: 1.0
Nodes (1): torch_knn_indices() doit se terminer sans CUDA OOM sur 100k×30 float32.

### Community 85 - "Community 85"
Cohesion: 1.0
Nodes (1): Test de stress complet : 2M×30 float32, k=15 — réservé GPU, marquer slow.

## Knowledge Gaps
- **535 isolated node(s):** `analyze_results.py Genere toutes les statistiques et heatmaps depuis results_gr`, `launch_gui.py — Point d'entrée pour la compilation PyInstaller (GUI uniquement).`, `Request per-monitor DPI awareness to avoid blurry bitmap-scaled UI on Windows.`, `Installe sys.excepthook (thread principal) ET threading.excepthook (tous les`, `Assure la compatibilité des imports historiques `flowsom_pipeline_pro.*`     en` (+530 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 23`** (2 nodes): `run_pipeline.py`, `run_pipeline.py — Lanceur de développement (sans pip install).  Usage depuis n'i`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (2 nodes): `setup.py`, `setup.py — Packaging de flowsom_pipeline_pro.  Installation:     pip install -e`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Charge la configuration depuis un fichier YAML.          Args:             ya`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Construit un PipelineConfig depuis un dictionnaire YAML brut.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Construit la configuration depuis des arguments CLI (argparse.Namespace).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Convertit #RRGGBB â†’ 'R, G, B' pour usage dans rgba().`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `Nombre total de nodes dans la grille SOM.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Trouve l'index d'un marqueur parmi les patterns donnés.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Gate les cellules viables basé sur FSC/SSC.          Args:             X: Matric`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Gate les singlets basé sur le ratio FSC-A/FSC-H.         Les doublets ont typiqu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Gate les cellules CD45+ (leucocytes).          Returns:             Masque boolé`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Gate les blastes CD34+ (cellules souches/progénitrices).          Les blastes so`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Gate rectangulaire/polygonal pour exclure les débris sur FSC-A vs SSC-A.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Normalisation Z-score par marqueur (colonne).          Après normalisation: moye`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Normalisation Min-Max par marqueur vers [0, 1].          Les marqueurs constants`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Applique la normalisation spécifiée.          Args:             data: Matrice (n`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Transformation Arcsinh (inverse hyperbolic sine).          Args en entrée:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Inverse de la transformation Arcsinh.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Transformation Logicle (biexponentielle).          Args en entrée:             d`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Transformation logarithmique standard.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Normalisation Z-score (moyenne=0, std=1).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Normalisation Min-Max [0, 1].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Applique la transformation spécifiée, avec gestion optionnelle         des canau`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Calcule le masque booléen des événements inclus dans cette gate.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Retourne les sommets de la gate en coordonnées données.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Reconstruit une PolygonGate depuis un dict (chargement session).          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Nombre d'événements positifs (sum du masque).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Taille totale du masque (N_events du Sample).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Pourcentage d'événements positifs dans la population source.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Pourcentage de cellules conservées (0–100).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Nombre de cellules exclues par ce gate.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Pourcentage de cellules exclues (0–100).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Qualité basique du gate : True si > 20% des cellules ont été conservées.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `Nombre total de cellules dans le résultat final.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `Nombre de métaclusters produits.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `True si le pipeline s'est terminé avec des données valides.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `Construit un PipelineResult représentant un échec.          Args:             er`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `Nombre d'événements (cellules) dans le Sample.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Noms des canaux primaires (colonnes de events).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `Construit un FlowSample depuis un objet AnnData (flowsom.io.read_FCS).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `Remplace l'alpha d'une couleur rgba(r,g,b,a) par new_alpha.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (1 nodes): `Définit la préférence GPU. Appelé par la GUI avant le lancement.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (1 nodes): `Retourne True si l'accélération GPU est autorisée.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 71`** (1 nodes): `Crée un RunMetadata au lancement d'un run.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 74`** (1 nodes): `Return the codes, shaped: (n_clusters, n_features).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 75`** (1 nodes): `Return the distances.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 76`** (1 nodes): `Return the cluster labels.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 77`** (1 nodes): `Return the metacluster labels.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 82`** (1 nodes): `Charge un log de gating depuis un fichier JSON.          Args:             path:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 84`** (1 nodes): `torch_knn_indices() doit se terminer sans CUDA OOM sur 100k×30 float32.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 85`** (1 nodes): `Test de stress complet : 2M×30 float32, k=15 — réservé GPU, marquer slow.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `path()` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 8`, `Community 13`, `Community 16`, `Community 18`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Why does `Sample` connect `Community 1` to `Community 3`, `Community 4`, `Community 6`?**
  _High betweenness centrality (0.083) - this node is a cross-community bridge._
- **Why does `PipelineConfig` connect `Community 3` to `Community 0`, `Community 1`, `Community 2`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Are the 225 inferred relationships involving `PipelineConfig` (e.g. with `PatientCache` and `optimize_optuna_mrd.py — Optimisation Bayesienne MRD avec Optuna ===============`) actually correct?**
  _`PipelineConfig` has 225 INFERRED edges - model-reasoned connections that need verification._
- **Are the 201 inferred relationships involving `Sample` (e.g. with `CitrusParams` and `CitrusCluster`) actually correct?**
  _`Sample` has 201 INFERRED edges - model-reasoned connections that need verification._
- **Are the 158 inferred relationships involving `Experiment` (e.g. with `IExperimentStrategy` and `CohortBaseParams`) actually correct?**
  _`Experiment` has 158 INFERRED edges - model-reasoned connections that need verification._
- **Are the 128 inferred relationships involving `ToggleSwitch` (e.g. with `DarkComboBox` and `MatplotlibCanvas`) actually correct?**
  _`ToggleSwitch` has 128 INFERRED edges - model-reasoned connections that need verification._