"""
mrd_calculator.py — Calcul de la MRD résiduelle post-FlowSOM.

═══════════════════════════════════════════════════════════════════════════════
  TROIS MÉTHODES DE DÉTECTION DES NŒUDS SOM MRD
═══════════════════════════════════════════════════════════════════════════════

  Méthode JF :
    Nœud MRD si % moelle normale GLOBAL < seuil ET % patho DANS LE NŒUD > seuil.
    Critère conservateur qui exige que le cluster soit quasi-exclusivement
    pathologique ET ne contienne qu'une infime fraction de la moelle normale.

  Méthode Flo :
    Nœud MRD si % patho > N × % sain dans le même nœud.
    Mesure le rapport de déséquilibre intra-cluster. Tolère des clusters mixtes
    à condition que le ratio patho/sain soit supérieur au multiplicateur N.

  Méthode ELN (European LeukemiaNet 2022 — DfN, « Different from Normal ») :
    Recommandations Schuurhuis et al., Blood 2018 ; Heuser et al., Leukemia 2022.
    1. Filtre LOQ (Limit Of Quantification) : >= min_cluster_events cellules
       dans le nœud — garantit la robustesse statistique du ratio.
    2. Critère DfN : % patho > % sain dans le nœud — l'enrichissement relatif
       est la signature topologique d'une population anormale.
    3. Positivité globale : MRD% >= clinical_positivity_pct (0.1% ELN standard)
       — seuil clinique de décision thérapeutique.

═══════════════════════════════════════════════════════════════════════════════
  APPROCHE HYBRIDE (optionnelle) — Entonnoir à Deux Portes
═══════════════════════════════════════════════════════════════════════════════

  Problème résolu : l'effet batch provoque l'isolation de cellules saines
  atypiques dans des nœuds SOM dédiés. Ces nœuds franchissent la porte
  mathématique (ratio patho/sain élevé) mais ne sont pas des blastes.

  Solution — double porte (ET logique) :
    1. Porte Topologique/Mathématique : critère de la méthode (JF / Flo / ELN).
    2. Porte Phénotypique/Biologique  : blast_category IN allowed_categories.
       Calculée via blast_detection.py (scoring ELN 2022 / score d'Ogata).
       Un nœud doit présenter une signature blastique (CD34/CD117 bright,
       CD45-dim, SSC-bas) pour être validé comme nœud MRD.

  Activé via blast_phenotype_filter.enabled dans config/mrd_config.yaml.

  Données requises pour la porte biologique :
    - X_norm         : médianes SOM normalisées dans l'espace de référence
                       (valeurs produites par compute_reference_normalization)
    - marker_names   : noms des marqueurs correspondants
    OU (si X_norm non disponible) :
    - node_medians   : médianes SOM brutes (normalisation intra-dataset appliquée
                       automatiquement — moins précise sans référence externe)

Les seuils sont paramétrables via config/mrd_config.yaml.

Usage:
    from clinical_module.mrd.mrd_calculator import (
        load_mrd_config, compute_mrd,
    )
    mrd_cfg = load_mrd_config()
    results = compute_mrd(df_cells, clustering, mrd_cfg)
    # Avec porte biologique :
    results = compute_mrd(
        df_cells, clustering, mrd_cfg,
        X_norm=node_medians_normalized,   # médianes dans l'espace de référence
        marker_names=selected_markers,
    )
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ARCH-1 : imports au niveau du module (évite les imports différés dans les fonctions)
from clinical_module.phenotyping.blast_detection import (
    build_blast_weights,
    categorize_blast_score,
    compute_reference_stats,
    score_nodes_for_blasts,
    score_nodes_hybrid,
    score_nodes_mahalanobis,
)
from prisma.exceptions import ClinicalMathError
from prisma.utils.logger import get_logger

_logger = get_logger("analysis.mrd_calculator")


def _hash_yaml_file(yaml_path: str | Path | None) -> Optional[str]:
    """
    Calcule le hash SHA256 du contenu d'un fichier YAML de configuration.

    Permet de tracer quelle version exacte des paramètres a été utilisée
    pour un calcul MRD donné (audit trail clinique).

    Returns:
        Hash SHA256 hexadécimal (64 caractères) ou None si le fichier est absent.
    """
    if yaml_path is None:
        return None
    path = Path(yaml_path)
    if not path.exists():
        return None
    try:
        content = path.read_bytes()
        return hashlib.sha256(content).hexdigest()
    except Exception as exc:
        _logger.debug("_hash_yaml_file : impossible de lire '%s' : %s", path, exc)
        return None

# ─────────────────────────────────────────────────────────────────────────────
#  Config dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MRDMethodJF:
    """
    Seuils pour la méthode JF (Jabbour-Faderl, adaptée cytométrie clinique).

    Logique : un nœud SOM est MRD si et seulement si les deux critères sont remplis
    dans cet ordre :

      1. (min_patho_cells_pct) Le cluster est suffisamment dominé par les cellules
         pathologiques : pct_patho_intra > min_patho_cells_pct.
         Filtre préalable — si le cluster est trop peu patho, on n'évalue pas le
         critère sain (évite les faux positifs sur clusters mixtes faiblement chargés).

      2. (max_normal_marrow_pct) La fraction de cellules saines CONTENUES dans ce
         cluster est infime par rapport au pool total de moelle normale :
         n_sain_cluster / total_sain < max_normal_marrow_pct.
         Exemple : 1 180 saines / 980 000 total = 0.12 % → rejeté si seuil = 0.10 %.

    Ce double critère garantit que seuls les clusters très chargés en blastes ET
    très peu représentés dans la moelle normale saine sont retenus comme MRD.
    """

    max_normal_marrow_pct: float = 2.0  # % max : n_sain_cluster / total_sain_global
    min_patho_cells_pct: float = 10.0   # % min de cellules patho DANS le cluster


@dataclass
class MRDMethodFlo:
    """
    Seuils pour la méthode Flo (ratio intra-cluster, corrigé du biais dataset).

    Logique : un nœud SOM est MRD si le % de cellules patho est supérieur à
    multiplicateur_effectif × % de cellules saines dans le même nœud.

    Le multiplicateur effectif est calculé à l'exécution :
        _flo_mult_eff = normal_marrow_multiplier × (total_sain / total_patho)

    Cette correction compense le déséquilibre de composition du dataset.
    Exemple avec normal_marrow_multiplier=2.0 et dataset 10× plus de sain :
        _flo_mult_eff = 2.0 × 10 = 20 → pct_patho > 20 × pct_sain_intra.
    Cela garantit qu'un cluster n'est MRD que s'il est enrichi en patho
    au-delà de ce qu'attendrait le hasard pur (proportion dataset).
    """

    normal_marrow_multiplier: float = 2.0  # ratio de base (ajusté par ratio dataset)


@dataclass
class ELNStandards:
    """
    Recommandations ELN 2022 pour la MRD par cytométrie en flux multiparamétrique.

    Référence : Schuurhuis G.J. et al. (2018) Blood 131(12):1275–1291.
               Heuser M. et al. (2022) Leukemia 36:5–22.

    min_cluster_events (LOQ — Limit Of Quantification) :
      Nombre minimum d'événements dans un nœud SOM pour que son ratio
      patho/sain soit statistiquement robuste. L'ELN recommande un minimum
      de 50 événements pour éviter les ratios artefactuels sur petits clusters.

    clinical_positivity_pct :
      Seuil de positivité clinique MRD : 0.1% des cellules totales.
      Valeur de référence ELN pour la décision thérapeutique (rechute précoce).
      En dessous : MRD détectable mais non positive (MRD low-level).
    """

    min_cluster_events: int = 50  # LOQ ELN 2022 : min d'événements/nœud
    clinical_positivity_pct: float = 0.1  # Seuil de positivité clinique (%)


@dataclass
class BlastPhenotypeFilter:
    """
    Porte biologique hybride — filtre phénotypique ELN 2022 / score d'Ogata
    avec distance de Mahalanobis et modulation par la pureté topologique.

    ── Choix du moteur de scoring (scoring_method) ─────────────────────────────

    "linear"   (historique) : score ELN 2022 / Ogata pur — somme pondérée de
                              Z-scores directionnels. Rapide, interprétable, mais
                              aveugle aux corrélations entre marqueurs. Échoue sur
                              les LAM matures CD34−/CD117− (P25/P57).
                              Comportement identique à l'implémentation pré-Stratégie A.

    "mahalanobis"           : distance de Mahalanobis seule, modulée par la
                              pureté topologique. Résout le paradoxe des LAM
                              matures mais sans connaissance clinique directionnelle.

    "hybrid"   (recommandé) : combinaison pondérée Mahalanobis + linéaire.
                              mahal_weight × D²_normalisé + linear_weight × Score_ELN.
                              Capture à la fois la distance géométrique (brisant le
                              paradoxe P25/P57 vs P105) et la direction clinique.

    "none"                  : bypass total de la Porte Biologique. Équivalent à
                              enabled=False mais sans avoir à modifier ce flag.
                              Utile pour les tests de la Porte 1 seule, ou pour
                              des cas cliniques où seule la topologie FlowSOM
                              fait foi (MRD massive évidente > 50%).

    ── Paramètres Mahalanobis ───────────────────────────────────────────────────

    d2_threshold_high, d2_threshold_moderate :
      Seuils de D² pour l'admission en "BLAST_HIGH" / "BLAST_MODERATE" dans le
      mode "mahalanobis" pur. En mode "hybrid", ces seuils ne sont pas utilisés
      directement — c'est le score hybride /10 qui est comparé à high_threshold.

      Valeurs de référence (χ² avec k marqueurs) :
        k=10 : χ²(10, 0.95)=18.3 → d2_high≈20, d2_moderate≈12
        k=8  : χ²(8,  0.95)=15.5 → d2_high≈16, d2_moderate≈10

    d2_normalization :
      Valeur D² normalisée à 10/10 dans le score hybride.
      Défaut : 20.0 (≈ χ²(10, 0.95) + marge de sécurité).

    purity_modulation_power :
      Exposant de la modulation par pureté topologique (Porte 1).
      0.5 = racine carrée (défaut recommandé).
      0.0 = pas de modulation (comportement pre-Mahalanobis).
      1.0 = linéaire (modulation plus agressive).

    mahal_weight, linear_weight :
      Poids des composantes dans le score hybride.
      Défaut : 0.65 / 0.35 (Mahalanobis dominant).
      Augmenter mahal_weight si cohorte à forte prévalence de LAM matures.
      Augmenter linear_weight si risque de faux positifs géométriques.

    mahal_boost_factor :
      Facteur multiplicatif appliqué au score linéaire quand D² > d2_normalization.
      Booste les blastes atypiques géométriquement très anormaux (ex: P25).
      Défaut : 1.5. Réduire si trop de FP sur les patients très inflammatoires.

    ── Seuil dynamique (purity_threshold_override) ─────────────────────────────

    Quand purity_threshold_override=True, le seuil de validation du score
    hybride devient dynamique en fonction de la pureté du nœud :

        seuil_effectif = high_threshold × (1 − (purity − purity_strict_above)
                         × relax_factor) si purity ≥ purity_strict_above

    Comportement :
      • Nœud très pur (≥ purity_strict_above, ex: 95%) : seuil abaissé jusqu'à
        high_threshold × (1 − relax_factor). Un phénotype "pas parfait" est accepté
        car la pureté topologique est une preuve en soi.
      • Nœud mixte (< purity_strict_above) : seuil maintenu à high_threshold.
        La biologie doit compenser l'ambiguïté topologique.

    Référence : blast_detection.py — score_nodes_mahalanobis(), score_nodes_hybrid(),
               compute_reference_stats() — basés sur ELN 2022 + score Ogata + Mahalanobis.
    """

    enabled: bool = False
    allowed_categories: List[str] = field(
        default_factory=lambda: ["BLAST_HIGH", "BLAST_MODERATE"]
    )
    apply_to_jf: bool = True
    apply_to_flo: bool = True
    apply_to_eln: bool = True
    # Seuils de score (configuration optimisée LAIP — Optuna-tuned)
    high_threshold: float = 2.60
    moderate_threshold: float = 2.20
    weak_threshold: float = 2.25
    # LAIP strategy : marker_weights surcharge les poids hardcodés dans build_blast_weights.
    # HLA-DR est le MARQUEUR STAR (4.88), CD33 quasi-nul (0.10), CD45 neutre (-0.05).
    marker_weights: Optional[Dict[str, float]] = field(
        default_factory=lambda: {
            "CD34": 1.31,
            "CD117": 0.87,
            "HLA-DR": 4.88,  # MARQUEUR STAR : détection basée sur HLA-DR
            "CD33": 0.10,
            "CD13": 2.14,
            "CD45": -0.05,   # Quasiment neutre
            "SSC": -2.87,    # Pénalité modérée pour les granuleux
        }
    )

    # ── Choix du moteur de scoring ────────────────────────────────────────────
    # "linear"       → score ELN 2022 / Ogata pur (historique, comportement inchangé)
    # "mahalanobis"  → distance D² seule (modulée par pureté topologique)
    # "hybrid"       → combinaison pondérée Mahalanobis + linéaire (recommandé)
    # "none"         → bypass total de la Porte Biologique (Porte 1 seule fait foi)
    scoring_method: str = "hybrid"  # "linear" | "mahalanobis" | "hybrid" | "none"
    # LAIP strategy : Mix Géométrie / Biologie — la géométrie domine légèrement (56%)
    mahal_weight: float = 0.56   # Géométrie (Mahalanobis) — détecteur d'anomalie
    linear_weight: float = 0.17  # Biologie classique (ELN 2022) — signature clinique
    d2_normalization: float = 109.28  # D² → 10/10 (tuned Optuna)
    d2_threshold_high: float = 19.03  # seuil D² BLAST_HIGH
    d2_threshold_moderate: float = 25.80  # seuil D² BLAST_MODERATE
    purity_modulation_power: float = 0.98  # modulation pureté quasi-linéaire
    # d2_min_normal_threshold supprimé — masquait les LAM matures (impasse géométrique)
    mahal_boost_factor: float = 1.03  # boost minimaliste si D² > d2_normalization

    # ── Seuil dynamique par pureté ────────────────────────────────────────────
    purity_threshold_override: bool = True  # activer le seuil dynamique
    purity_strict_above: float = 0.93   # pureté ≥ 93% → seuil assoupli
    # LAIP strategy : purity_relax_factor très bas (0.13) — confiance mesurée en FlowSOM.
    # Avec high_threshold=2.60, purity_strict_above=0.93, relax_factor=0.13 :
    #   purity=100% → seuil = 2.60 × (1 − 0.13×1.0) = 2.26 (assouplissement minimal)
    #   purity=96%  → seuil = 2.60 × (1 − 0.13×0.43) = 2.45 (quasi pas d'assouplissement)
    #   purity=93%  → seuil = 2.60 (nœud proche du seuil, pas d'assouplissement)
    # On exige de la preuve biologique même sur nœuds purs — stratégie très conservatrice.
    purity_relax_factor: float = 0.13   # assouplissement 13% max (on exige de la preuve)

    # ── Performance Mode 2 (stats NBM internes) ─────────────────────────────
    nbm_stats_max_cells: int = 250000   # max lignes pour mediane/IQR NBM
    nbm_cov_max_cells: int = 120000     # max lignes pour covariance NBM
    nbm_mincovdet_max_cells: int = 30000  # seuil max pour utiliser MinCovDet


@dataclass
class MRDConfig:
    """Configuration complète du calcul MRD."""

    enabled: bool = True
    method: str = "all"  # "jf", "flo", "eln", "all"
    method_jf: MRDMethodJF = field(default_factory=MRDMethodJF)
    method_flo: MRDMethodFlo = field(default_factory=MRDMethodFlo)
    eln_standards: ELNStandards = field(default_factory=ELNStandards)
    blast_phenotype_filter: BlastPhenotypeFilter = field(
        default_factory=BlastPhenotypeFilter
    )
    condition_sain: str = "Sain"
    condition_patho: str = "Pathologique"

    # ── Traçabilité — chemin du fichier YAML source (audit trail) ────────────
    # Rempli automatiquement par load_mrd_config() avec le chemin résolu.
    _config_file_path: Optional[str] = field(default=None, repr=False)


def load_mrd_config(config_path: Optional[Path | str] = None) -> MRDConfig:
    """
    Charge la configuration MRD depuis un fichier YAML.

    Si aucun chemin n'est donné, cherche config/mrd_config.yaml à côté du
    package flowsom_pipeline_pro.
    """
    cfg = MRDConfig()

    if config_path is None:
        candidates = [
            Path(__file__).parent.parent.parent / "config" / "mrd_config.yaml",
        ]
        for p in candidates:
            if p.exists():
                config_path = p
                break

    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            try:
                raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                params = raw.get("mrd_parameters", {})

                cfg.enabled = params.get("enabled", cfg.enabled)
                cfg.method = params.get("method", cfg.method)
                cfg.condition_sain = params.get("condition_sain", cfg.condition_sain)
                cfg.condition_patho = params.get("condition_patho", cfg.condition_patho)

                jf = params.get("method_jf", {})
                cfg.method_jf.max_normal_marrow_pct = jf.get(
                    "max_normal_marrow_pct", cfg.method_jf.max_normal_marrow_pct
                )
                cfg.method_jf.min_patho_cells_pct = jf.get(
                    "min_patho_cells_pct", cfg.method_jf.min_patho_cells_pct
                )

                flo = params.get("method_flo", {})
                cfg.method_flo.normal_marrow_multiplier = flo.get(
                    "normal_marrow_multiplier", cfg.method_flo.normal_marrow_multiplier
                )

                eln = params.get("eln_standards", {})
                cfg.eln_standards.min_cluster_events = eln.get(
                    "min_cluster_events", cfg.eln_standards.min_cluster_events
                )
                cfg.eln_standards.clinical_positivity_pct = eln.get(
                    "clinical_positivity_pct", cfg.eln_standards.clinical_positivity_pct
                )

                bpf = params.get("blast_phenotype_filter", {})
                cfg.blast_phenotype_filter.enabled = bpf.get(
                    "enabled", cfg.blast_phenotype_filter.enabled
                )
                cfg.blast_phenotype_filter.allowed_categories = bpf.get(
                    "allowed_categories", cfg.blast_phenotype_filter.allowed_categories
                )
                cfg.blast_phenotype_filter.apply_to_jf = bpf.get(
                    "apply_to_jf", cfg.blast_phenotype_filter.apply_to_jf
                )
                cfg.blast_phenotype_filter.apply_to_flo = bpf.get(
                    "apply_to_flo", cfg.blast_phenotype_filter.apply_to_flo
                )
                cfg.blast_phenotype_filter.apply_to_eln = bpf.get(
                    "apply_to_eln", cfg.blast_phenotype_filter.apply_to_eln
                )
                cfg.blast_phenotype_filter.high_threshold = float(
                    bpf.get("high_threshold", cfg.blast_phenotype_filter.high_threshold)
                )
                cfg.blast_phenotype_filter.moderate_threshold = float(
                    bpf.get(
                        "moderate_threshold",
                        cfg.blast_phenotype_filter.moderate_threshold,
                    )
                )
                cfg.blast_phenotype_filter.weak_threshold = float(
                    bpf.get("weak_threshold", cfg.blast_phenotype_filter.weak_threshold)
                )
                _raw_mw = bpf.get("marker_weights", None)
                if isinstance(_raw_mw, dict) and _raw_mw:
                    cfg.blast_phenotype_filter.marker_weights = {
                        str(k): float(v) for k, v in _raw_mw.items()
                    }

                # ── Choix du moteur de scoring ────────────────────────────
                # Accepte "scoring_method" (spec officielle) et "scoring_mode"
                # (nom utilisé en interne lors du dev) pour la rétrocompatibilité.
                _sm_value = bpf.get("scoring_method", None) or bpf.get(
                    "scoring_mode", cfg.blast_phenotype_filter.scoring_method
                )
                cfg.blast_phenotype_filter.scoring_method = _sm_value
                cfg.blast_phenotype_filter.mahal_weight = float(
                    bpf.get("mahal_weight", cfg.blast_phenotype_filter.mahal_weight)
                )
                cfg.blast_phenotype_filter.linear_weight = float(
                    bpf.get("linear_weight", cfg.blast_phenotype_filter.linear_weight)
                )
                cfg.blast_phenotype_filter.d2_normalization = float(
                    bpf.get(
                        "d2_normalization", cfg.blast_phenotype_filter.d2_normalization
                    )
                )
                cfg.blast_phenotype_filter.d2_threshold_high = float(
                    bpf.get(
                        "d2_threshold_high",
                        cfg.blast_phenotype_filter.d2_threshold_high,
                    )
                )
                cfg.blast_phenotype_filter.d2_threshold_moderate = float(
                    bpf.get(
                        "d2_threshold_moderate",
                        cfg.blast_phenotype_filter.d2_threshold_moderate,
                    )
                )
                cfg.blast_phenotype_filter.purity_modulation_power = float(
                    bpf.get(
                        "purity_modulation_power",
                        cfg.blast_phenotype_filter.purity_modulation_power,
                    )
                )
                cfg.blast_phenotype_filter.mahal_boost_factor = float(
                    bpf.get(
                        "mahal_boost_factor",
                        cfg.blast_phenotype_filter.mahal_boost_factor,
                    )
                )

                # ── Seuil dynamique par pureté ────────────────────────────
                cfg.blast_phenotype_filter.purity_threshold_override = bool(
                    bpf.get(
                        "purity_threshold_override",
                        cfg.blast_phenotype_filter.purity_threshold_override,
                    )
                )
                cfg.blast_phenotype_filter.purity_strict_above = float(
                    bpf.get(
                        "purity_strict_above",
                        cfg.blast_phenotype_filter.purity_strict_above,
                    )
                )
                cfg.blast_phenotype_filter.purity_relax_factor = float(
                    bpf.get(
                        "purity_relax_factor",
                        cfg.blast_phenotype_filter.purity_relax_factor,
                    )
                )
                cfg.blast_phenotype_filter.nbm_stats_max_cells = int(
                    bpf.get(
                        "nbm_stats_max_cells",
                        cfg.blast_phenotype_filter.nbm_stats_max_cells,
                    )
                )
                cfg.blast_phenotype_filter.nbm_cov_max_cells = int(
                    bpf.get(
                        "nbm_cov_max_cells",
                        cfg.blast_phenotype_filter.nbm_cov_max_cells,
                    )
                )
                cfg.blast_phenotype_filter.nbm_mincovdet_max_cells = int(
                    bpf.get(
                        "nbm_mincovdet_max_cells",
                        cfg.blast_phenotype_filter.nbm_mincovdet_max_cells,
                    )
                )

                _logger.info("MRD config chargée depuis %s", config_path.name)
            except Exception as e:
                _logger.warning(
                    "Erreur lecture mrd_config.yaml (%s) — valeurs par défaut.", e
                )

    # Traçabilité : stocker le chemin résolu pour l'audit trail dans config_snapshot
    if config_path is not None:
        cfg._config_file_path = str(Path(config_path).resolve())

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
#  Résultats
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MRDClusterResult:
    """Résultat MRD pour un nœud SOM individuel."""

    cluster_id: int  # ID du nœud SOM
    n_cells_total: int
    n_cells_sain: int
    n_cells_patho: int
    pct_sain: float  # % de cellules saines DANS ce nœud (par rapport au total du nœud)
    pct_patho: float  # % de cellules pathologiques DANS ce nœud
    pct_sain_global: (
        float  # % des cellules saines de ce nœud / total moelle normale (méthode JF)
    )
    is_mrd_jf: bool  # qualifié MRD par méthode JF
    is_mrd_flo: bool  # qualifié MRD par méthode Flo
    is_mrd_eln: bool  # qualifié MRD par méthode ELN
    # Porte biologique (None si filtre désactivé ou données absentes)
    blast_score: Optional[float] = None
    blast_category: Optional[str] = None
    mahal_d2: Optional[float] = None  # D² brut (debug / calibration)
    node_purity: Optional[float] = None  # fraction cellules patho (0–1)


@dataclass
class MRDResult:
    """Résultat global du calcul MRD."""

    method_used: str
    total_cells: int
    total_cells_patho: int
    total_cells_sain: int
    # Dénominateur effectif utilisé pour le calcul MRD%
    # = total_cells_patho si cd45_autogating_mode="none"
    # = cellules patho CD45+ si cd45_autogating_mode in ("cd45", "cd45_dim")
    mrd_denominator: int = 0
    mrd_denominator_mode: str = "none"  # "none" | "cd45" | "cd45_dim"
    # Nombre de cellules patho CD45+ — toujours calculé si cd45_mask fourni,
    # indépendamment du dénominateur effectif. Permet le toggle UI.
    n_patho_cd45pos: int = 0
    # Nombre de cellules patho AVANT la gate CD45 (CD45+ + CD45-).
    # = total_cells_patho si gate CD45 inactive.
    # Stocké pour le toggle UI : dénominateur "toutes cellules patho".
    n_patho_pre_cd45: int = 0

    # Méthode JF
    mrd_cells_jf: int = 0
    mrd_pct_jf: float = 0.0
    n_nodes_mrd_jf: int = 0

    # Méthode Flo
    mrd_cells_flo: int = 0
    mrd_pct_flo: float = 0.0
    n_nodes_mrd_flo: int = 0

    # Méthode ELN
    mrd_cells_eln: int = 0
    mrd_pct_eln: float = 0.0
    n_nodes_mrd_eln: int = 0
    eln_positive: bool = False  # MRD% >= seuil clinique ELN
    eln_low_level: bool = False  # MRD détectable mais < seuil clinique

    # Filtre phénotypique hybride — état pour la traçabilité
    blast_filter_active: bool = False  # True si le filtre a été appliqué
    blast_scoring_mode: str = "linear"  # "linear" | "mahalanobis" | "hybrid"

    # Détail par nœud SOM
    per_node: List[MRDClusterResult] = field(default_factory=list)

    # Config utilisée (traçabilité)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise le résultat pour export JSON / rapport HTML."""
        return {
            "method_used": self.method_used,
            "total_cells": self.total_cells,
            "total_cells_patho": self.total_cells_patho,
            "total_cells_sain": self.total_cells_sain,
            "mrd_denominator": self.mrd_denominator,
            "mrd_denominator_mode": self.mrd_denominator_mode,
            "n_patho_cd45pos": self.n_patho_cd45pos,
            "n_patho_pre_cd45": self.n_patho_pre_cd45,
            "blast_filter_active": self.blast_filter_active,
            "blast_scoring_mode": self.blast_scoring_mode,
            "mrd_jf": {
                "mrd_cells": self.mrd_cells_jf,
                "mrd_pct": round(self.mrd_pct_jf, 6),
                "n_nodes_mrd": self.n_nodes_mrd_jf,
            },
            "mrd_flo": {
                "mrd_cells": self.mrd_cells_flo,
                "mrd_pct": round(self.mrd_pct_flo, 6),
                "n_nodes_mrd": self.n_nodes_mrd_flo,
            },
            "mrd_eln": {
                "mrd_cells": self.mrd_cells_eln,
                "mrd_pct": round(self.mrd_pct_eln, 6),
                "n_nodes_mrd": self.n_nodes_mrd_eln,
                "eln_positive": self.eln_positive,
                "eln_low_level": self.eln_low_level,
            },
            "per_node": [
                {
                    "som_node_id": c.cluster_id,
                    "n_cells_total": c.n_cells_total,
                    "n_cells_sain": c.n_cells_sain,
                    "n_cells_patho": c.n_cells_patho,
                    "pct_sain": round(c.pct_sain, 4),
                    "pct_sain_global": round(c.pct_sain_global, 6),
                    "pct_patho": round(c.pct_patho, 4),
                    "is_mrd_jf": c.is_mrd_jf,
                    "is_mrd_flo": c.is_mrd_flo,
                    "is_mrd_eln": c.is_mrd_eln,
                    "blast_score": round(c.blast_score, 2)
                    if c.blast_score is not None
                    else None,
                    "blast_category": c.blast_category,
                    "mahal_d2": round(c.mahal_d2, 3)
                    if c.mahal_d2 is not None
                    else None,
                    "node_purity": round(c.node_purity, 4)
                    if c.node_purity is not None
                    else None,
                }
                for c in self.per_node
            ],
            "config": self.config_snapshot,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Calcul MRD
# ─────────────────────────────────────────────────────────────────────────────


def _build_node_blast_scores(
    marker_names: List[str],
    X_norm: Optional[np.ndarray] = None,
    node_medians: Optional[np.ndarray] = None,
    nbm_center: Optional[np.ndarray] = None,
    nbm_scale: Optional[np.ndarray] = None,
    nbm_inv_cov: Optional[np.ndarray] = None,
    patho_purity: Optional[np.ndarray] = None,
    scoring_method: str = "hybrid",
    mahal_weight: float = 0.65,
    linear_weight: float = 0.35,
    d2_normalization: float = 20.0,
    purity_modulation_power: float = 0.5,
    mahal_boost_factor: float = 1.5,
    custom_weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Routeur principal des moteurs de scoring blast.

    Calcule le blast_score /10 pour chaque nœud SOM et redirige vers le moteur
    choisi via scoring_method. C'est la seule fonction à appeler depuis compute_mrd.

    ── Moteurs disponibles (scoring_method) ─────────────────────────────────────

    "linear"       : Moteur historique — score ELN 2022 / Ogata pur.
                     Somme pondérée de Z-scores directionnels par rapport à la NBM.
                     Comportement identique à l'implémentation pre-Stratégie A.
                     Requiert : X_norm OU (node_medians + nbm_center + nbm_scale).

    "mahalanobis"  : Moteur Mahalanobis seul — distance D² modulée par purity.
                     Résout le paradoxe P25/P57 vs P105 sans connaissance directionnelle.
                     Requiert : node_medians (brutes) + nbm_inv_cov.

    "hybrid"       : Moteur hybride — combinaison pondérée Mahalanobis + linéaire.
                     mahal_weight × D²_normalisé + linear_weight × Score_linéaire.
                     Recommandé : capture la géométrie ET la connaissance clinique.
                     Requiert : node_medians + nbm_inv_cov + z-scores.

    "none"         : Bypass — retourne un tableau de scores nuls (10.0 partout)
                     afin que _passes_blast_gate() laisse passer tous les nœuds.
                     La Porte Biologique est entièrement désactivée.
                     Ne requiert aucune donnée supplémentaire.

    ── Fallback automatique ─────────────────────────────────────────────────────

    Si scoring_method est "mahalanobis" ou "hybrid" mais que nbm_inv_cov est None
    (non calculée, ex: trop peu de cellules NBM), le routeur rétrograde
    silencieusement sur "linear" avec un warning dans le log.

    ── Modes d'entrée pour la composante linéaire ───────────────────────────────

    Mode 1 — X_norm fourni (z-scores pré-calculés, prioritaire).
    Mode 2 — node_medians + nbm_center + nbm_scale (z-scoring à la volée).
    Mode 3 — BLOQUÉ : z-scoring intra-dataset instable à haute MRD (> 20%).

    Args:
        marker_names: Noms des marqueurs (colonnes de X_norm / node_medians).
        X_norm: Médianes z-scorées [n_nodes, n_markers] vs NBM (Mode 1).
        node_medians: Médianes brutes [n_nodes, n_markers] (arcsinh/5).
                      Requis pour Mahalanobis. Aussi utilisé comme Mode 2 si
                      nbm_center+nbm_scale fournis.
        nbm_center: Vecteur [n_markers] médiane NBM (centre du nuage).
        nbm_scale: Vecteur [n_markers] IQR/1.35 NBM (échelle robuste).
        nbm_inv_cov: Matrice [n_markers × n_markers] inverse de covariance NBM.
                     Requis pour "mahalanobis" et "hybrid".
        patho_purity: Tableau [n_nodes] ∈ [0, 1] — fraction de cellules patient
                      par nœud. Utilisé pour la modulation de la distance D².
        scoring_method: Moteur choisi : "linear" | "mahalanobis" | "hybrid" | "none".
        mahal_weight: Poids Mahalanobis dans le mode "hybrid" (défaut 0.65).
        linear_weight: Poids linéaire dans le mode "hybrid" (défaut 0.35).
        d2_normalization: Valeur D² normalisée à 10/10 (défaut 20.0 ≈ χ²(10,0.95)).
        purity_modulation_power: Exposant de modulation pureté (0.5 = racine carrée).
        custom_weights: Poids personnalisés pour le moteur linéaire (YAML marker_weights).

    Returns:
        Tuple (scores, d2_raw) :
          - scores  : np.ndarray [n_nodes] dans [0.0, 10.0].
          - d2_raw  : np.ndarray [n_nodes] distances D² brutes (None si moteur
                      linéaire pur ou mode "none").

    Raises:
        ValueError: Si les données requises sont absentes pour le moteur demandé
                    (ex: moteur "linear" sans z-scores ni stats NBM).
    """
    # ARCH-1 : imports déplacés au niveau du module — suppression des imports différés.

    # ── Mode "none" : bypass total, scores maximaux pour laisser tout passer ─
    if scoring_method == "none":
        _logger.debug(
            "_build_node_blast_scores : scoring_method='none' — "
            "Porte Biologique désactivée, tous les nœuds validés."
        )
        # Déterminer n_nodes à partir de la première entrée disponible
        _n_nodes = (
            len(X_norm)
            if X_norm is not None
            else len(node_medians)
            if node_medians is not None
            else 0
        )
        return np.full(_n_nodes, 10.0, dtype=float), None

    # ── Résoudre le z-score pour la composante linéaire ─────────────────────
    _X_zscore: Optional[np.ndarray] = None

    if X_norm is not None:
        _X_zscore = np.asarray(X_norm, dtype=float)

    elif node_medians is not None and nbm_center is not None and nbm_scale is not None:
        _raw = np.asarray(node_medians, dtype=float)
        _center = np.asarray(nbm_center, dtype=float)
        _scale = np.asarray(nbm_scale, dtype=float)
        _scale = np.where(_scale < 0.01, 0.01, _scale)
        _X_zscore = (_raw - _center) / _scale

    elif node_medians is not None and scoring_method == "linear":
        raise ValueError(
            "build_node_blast_scores (moteur 'linear') : stats NBM requises. "
            "Fournir nbm_center + nbm_scale, ou X_norm (z-scores pré-calculés). "
            "Le z-scoring intra-dataset est désactivé (instable à haute MRD)."
        )

    elif node_medians is None and scoring_method not in ("mahalanobis",):
        raise ValueError("build_node_blast_scores : X_norm ou node_medians requis.")

    # ── Résoudre node_medians_raw pour la composante Mahalanobis ────────────
    _raw_medians: Optional[np.ndarray] = (
        np.asarray(node_medians, dtype=float) if node_medians is not None else None
    )

    # ── Vérification stricte : inv_cov obligatoire pour Mahalanobis / hybrid ─
    # Plus de fallback silencieux sur le score linéaire : si l'appelant demande
    # Mahalanobis et que la matrice est absente, c'est une erreur clinique.
    _effective_method = scoring_method
    if scoring_method in ("mahalanobis", "hybrid") and nbm_inv_cov is None:
        raise ClinicalMathError(
            message=(
                f"Le moteur de scoring '{scoring_method}' requiert la matrice de covariance "
                "inverse NBM (nbm_inv_cov), mais celle-ci est absente. "
                "Vérifiez que compute_reference_stats() a été appelé avec suffisamment "
                "de cellules NBM de référence."
            ),
            details=f"scoring_method={scoring_method}, nbm_inv_cov=None",
        )

    # ── Routage vers le moteur ────────────────────────────────────────────────
    weights = build_blast_weights(marker_names, custom_weights=custom_weights)

    if _effective_method == "linear":
        if _X_zscore is None:
            raise ValueError(
                "build_node_blast_scores (moteur 'linear') : z-scores requis."
            )
        return score_nodes_for_blasts(_X_zscore, marker_names, weights), None

    elif _effective_method == "mahalanobis":
        if _raw_medians is None:
            raise ValueError(
                "build_node_blast_scores (moteur 'mahalanobis') : "
                "node_medians (brutes, arcsinh/5) requis."
            )
        _mu = (
            np.asarray(nbm_center, dtype=float)
            if nbm_center is not None
            else np.zeros(len(marker_names))
        )
        d2_raw, d2_mod = score_nodes_mahalanobis(
            _raw_medians,
            _mu,
            np.asarray(nbm_inv_cov, dtype=float),
            patho_purity=patho_purity,
            purity_modulation_power=purity_modulation_power,
        )
        scores = np.clip(d2_mod / max(d2_normalization, 1e-6) * 10.0, 0.0, 10.0)
        return scores, d2_raw

    else:
        # Moteur hybride
        if _raw_medians is None:
            raise ValueError(
                "build_node_blast_scores (moteur 'hybrid') : "
                "node_medians (brutes, arcsinh/5) requis."
            )
        if _X_zscore is None:
            raise ValueError(
                "build_node_blast_scores (moteur 'hybrid') : z-scores requis "
                "pour la composante linéaire (X_norm ou node_medians+stats_NBM)."
            )
        scores, d2_raw = score_nodes_hybrid(
            node_medians_raw=_raw_medians,
            node_medians_zscore=_X_zscore,
            nbm_center=np.asarray(nbm_center, dtype=float),
            nbm_inv_cov=np.asarray(nbm_inv_cov, dtype=float),
            marker_names=marker_names,
            patho_purity=patho_purity,
            weights=weights,
            mahal_weight=mahal_weight,
            linear_weight=linear_weight,
            d2_normalization=d2_normalization,
            purity_modulation_power=purity_modulation_power,
            mahal_boost_factor=mahal_boost_factor,
        )
        return scores, d2_raw


# Alias public du routeur (sans underscore) — exposé dans __init__ et
# utilisable directement dans les scripts d'analyse et notebooks.
build_node_blast_scores = _build_node_blast_scores


def compute_mrd(
    df_cells: pd.DataFrame,
    clustering: np.ndarray,
    mrd_cfg: MRDConfig,
    condition_column: str = "condition",
    cd45_autogating_mode: str = "none",
    cd45_mask: Optional[np.ndarray] = None,
    X_norm: Optional[np.ndarray] = None,
    node_medians: Optional[np.ndarray] = None,
    marker_names: Optional[List[str]] = None,
    nbm_center: Optional[np.ndarray] = None,
    nbm_scale: Optional[np.ndarray] = None,
    nbm_inv_cov: Optional[np.ndarray] = None,
    mrd_config_path: Optional[str | Path] = None,
    panel_config_path: Optional[str | Path] = None,
) -> MRDResult:
    """
    Calcule la MRD résiduelle selon les méthodes JF, Flo et/ou ELN.

    Opère au niveau des **nœuds SOM** (clustering), pas des métaclusters.

    ── Approche Hybride (si blast_phenotype_filter.enabled=True) ───────────────

    Un nœud est classé MRD seulement s'il franchit deux portes successives :

      Porte 1 — Topologique/Mathématique (critère de la méthode) :
        JF  : pct_sain_global < seuil ET pct_patho > seuil dans le nœud
        Flo : pct_patho > N × pct_sain dans le nœud
        ELN : n_cells >= LOQ ET pct_patho > pct_sain (critère DfN)

      Porte 2 — Phénotypique/Biologique (scoring ELN 2022 / Ogata) :
        blast_category IN allowed_categories (défaut: BLAST_HIGH, BLAST_MODERATE)
        Calculée vectoriellement sur toute la grille SOM AVANT la boucle,
        via score_nodes_for_blasts() de blast_detection.py.

    Données pour la porte biologique (par ordre de priorité) :
      X_norm        : médianes SOM normalisées dans l'espace de référence —
                      produit de compute_reference_normalization() (blast_detection.py).
                      Recommandé : la normalisation relative à la moelle normale est
                      biologiquement plus significative que l'intra-dataset.
      node_medians  : médianes SOM brutes — normalisation intra-dataset appliquée
                      automatiquement en fallback (moins précis, mais fonctionnel
                      sans population de référence externe).

    ── Dénominateur MRD ─────────────────────────────────────────────────────────

    Le pourcentage MRD (blastes / cellules totales) utilise différents
    dénominateurs selon le mode d'autogating CD45 :
      "none"     → dénominateur = toutes cellules patho (comportement historique)
      "cd45"     → dénominateur = cellules patho CD45+ uniquement
      "cd45_dim" → idem "cd45" (inclut blastes CD45-dim passés par la gate)

    Ce design permet de refléter la pratique clinique ELN 2022 qui mesure
    la MRD comme % des cellules CD45+ leucocytaires totales.

    Args:
        df_cells: DataFrame cellulaire avec colonne condition (shape: n_cells).
        clustering: Array (n_cells,) d'assignation nœud SOM (entiers 0-based).
                    Aligné ligne-à-ligne avec df_cells.
        mrd_cfg: Configuration MRD chargée via load_mrd_config().
        condition_column: Nom de la colonne condition dans df_cells.
                          Doit contenir mrd_cfg.condition_sain et condition_patho.
        cd45_autogating_mode: Mode de dénominateur CD45.
            "none"     → toutes cellules patho (comportement historique).
            "cd45"     → cellules patho CD45+ seulement.
            "cd45_dim" → idem (inclut blastes CD45-dim).
        cd45_mask: Masque booléen (n_cells,) — True = cellule CD45+.
            Nécessaire si cd45_autogating_mode != "none".
        X_norm: Matrice [n_nodes, n_markers] des z-scores des médianes SOM
            par rapport à la moelle normale (NBM). Prioritaire sur node_medians.
            Produit par compute_reference_normalization(node_medians, X_nbm).
            Les blastes ont z_CD34 ≈ +2, z_SSC ≈ −2 dans cet espace.
        node_medians: Matrice [n_nodes, n_markers] des médianes brutes par nœud
            (dans l'espace transformé, ex: arcsinh/5). Utilisé si X_norm=None.
            Combiné avec nbm_center+nbm_scale pour le z-scoring (Mode 2).
            Z-scoring intra-dataset en fallback si stats NBM absentes (Mode 3 dégradé).
        marker_names: Noms des marqueurs (colonnes de X_norm ou node_medians).
            Requis si X_norm ou node_medians est fourni.
        nbm_center: Vecteur [n_markers] des médianes NBM par marqueur (arcsinh/5).
            Calcule par compute_reference_stats() sur les données NBM.
        nbm_scale: Vecteur [n_markers] des IQR/1.35 NBM par marqueur.
            Combiné avec nbm_center pour le z-scoring du Mode 2.
        nbm_inv_cov: Matrice [n_markers, n_markers] — inverse de la covariance NBM.
            Calculée par compute_reference_stats() (3e valeur retournée).
            Requise pour scoring_method "mahalanobis" ou "hybrid" dans
            blast_phenotype_filter. Si None et mode hybride demandé, fallback
            automatique sur le score linéaire.

    Returns:
        MRDResult avec :
          - Totaux MRD par méthode (mrd_cells_*, mrd_pct_*, n_nodes_mrd_*)
          - Statut ELN (eln_positive, eln_low_level)
          - Détail par nœud SOM (per_node : List[MRDClusterResult])
            incluant blast_score et blast_category si porte biologique active
          - blast_filter_active : True si la porte biologique a été appliquée
    """
    _logger.info(
        "Calcul MRD (nœuds SOM) — méthode(s): %s | dénominateur CD45: %s",
        mrd_cfg.method,
        cd45_autogating_mode,
    )

    bpf = mrd_cfg.blast_phenotype_filter
    blast_filter_active = False

    condition = (
        df_cells[condition_column].values
        if condition_column in df_cells.columns
        else None
    )
    if condition is None:
        _logger.warning("Colonne '%s' absente — MRD impossible.", condition_column)
        return MRDResult(
            method_used=mrd_cfg.method,
            total_cells=len(df_cells),
            total_cells_patho=0,
            total_cells_sain=0,
        )

    unique_nodes = np.unique(clustering)
    n_total = len(clustering)
    is_patho = condition == mrd_cfg.condition_patho
    total_patho = int(is_patho.sum())
    total_sain = int((condition == mrd_cfg.condition_sain).sum())

    # ── Comptage cellules patho CD45+ (toujours calculé si masque disponible) ──
    n_patho_cd45pos = 0
    if cd45_mask is not None:
        cd45_arr = np.asarray(cd45_mask, dtype=bool)
        if len(cd45_arr) == len(condition):
            n_patho_cd45pos = int((is_patho & cd45_arr).sum())
        else:
            _logger.warning(
                "cd45_mask longueur %d ≠ n_cells %d — n_patho_cd45pos non calculé.",
                len(cd45_arr),
                len(condition),
            )

    # ── Dénominateur effectif pour le calcul MRD% ─────────────────────────────
    use_cd45_denom = cd45_autogating_mode in ("cd45", "cd45_dim")
    if use_cd45_denom and n_patho_cd45pos > 0:
        total_patho_cd45pos = n_patho_cd45pos
        _logger.info(
            "Dénominateur MRD CD45+ : %d cellules patho CD45+ (sur %d patho totales)",
            total_patho_cd45pos,
            total_patho,
        )
    else:
        total_patho_cd45pos = total_patho

    run_jf = mrd_cfg.method in ("jf", "both", "all")
    run_flo = mrd_cfg.method in ("flo", "both", "all")
    run_eln = mrd_cfg.method in ("eln", "all")

    # === Calcul dynamique des stats NBM (centre, échelle, inv_cov) ===
    # Si les stats NBM ne sont pas fournies explicitement, on les calcule à la
    # volée sur les cellules saines du pool d'analyse. Cela garantit que le
    # z-score de référence est ancré sur la moelle normale réelle, même quand
    # les blastes représentent 80-90 % du dataset.
    # On calcule aussi inv_cov ici si non fournie (requis pour Mahalanobis).
    if marker_names is not None and (nbm_center is None or nbm_scale is None):
        sain_mask = condition == mrd_cfg.condition_sain
        if sain_mask.sum() > 0:
            _marker_cols = [m for m in marker_names if m in df_cells.columns]
            if _marker_cols:
                sain_cells = df_cells.loc[sain_mask, _marker_cols].values
                # ARCH-1 : compute_reference_stats importé au niveau du module.
                nbm_center, nbm_scale, _inv_cov_auto = compute_reference_stats(
                    sain_cells,
                    robust=True,
                    max_samples_for_stats=bpf.nbm_stats_max_cells,
                    max_samples_for_covariance=bpf.nbm_cov_max_cells,
                    max_samples_for_mincovdet=bpf.nbm_mincovdet_max_cells,
                )
                # N'écraser nbm_inv_cov que s'il n'a pas été fourni explicitement
                if nbm_inv_cov is None:
                    nbm_inv_cov = _inv_cov_auto
                marker_names = _marker_cols
                _logger.info(
                    "Stats NBM calculées à la volée sur %d cellules saines (%d marqueurs) "
                    "— inv_cov %s.",
                    int(sain_mask.sum()),
                    len(_marker_cols),
                    "calculée"
                    if nbm_inv_cov is not None
                    else "indisponible (trop peu de points)",
                )

    # ── Pré-vectorisation O(N) : comptages par nœud via np.bincount ─────────
    # On calcule une seule fois les tableaux de taille (n_nodes_total,) indexés
    # directement par node_id — ce qui résout simultanément le goulot O(N×K)
    # et le bug de désynchronisation node_idx / node_id.
    _n_nodes_total = int(clustering.max()) + 1  # taille des tableaux bincount
    _is_patho_int = is_patho.astype(np.int32)
    _is_sain_int = (condition == mrd_cfg.condition_sain).astype(np.int32)

    _counts_total = np.bincount(clustering, minlength=_n_nodes_total)        # n_total par nœud
    _counts_patho = np.bincount(clustering, weights=_is_patho_int, minlength=_n_nodes_total).astype(np.int64)
    _counts_sain  = np.bincount(clustering, weights=_is_sain_int,  minlength=_n_nodes_total).astype(np.int64)

    # ── Porte Biologique : pré-calcul de la pureté par nœud ──────────────────
    # purity_per_node[node_id] = n_patho / n_in_node — indexé par node_id directement.
    _purity_per_node: Optional[np.ndarray] = None
    if bpf.enabled and bpf.purity_modulation_power > 0.0:
        _denom_counts = np.where(_counts_total > 0, _counts_total, 1)
        _purity_per_node = _counts_patho / _denom_counts  # shape (n_nodes_total,)

    # ── Porte Biologique : pré-calcul des blast scores par nœud ──────────────
    # node_blast_scores[i] = blast_score /10 du i-ème nœud dans unique_nodes.
    # node_blast_cats[i]   = blast_category correspondante (pour le mode linéaire).
    # node_d2_raw[i]       = D² brut pour la traçabilité / calibration.
    node_blast_scores: Optional[np.ndarray] = None
    node_blast_cats: Optional[List[str]] = None
    node_d2_raw: Optional[np.ndarray] = None

    _active_scoring_mode = bpf.scoring_method  # mode effectif (peut être rétrogradé)

    if bpf.scoring_method == "none":
        _logger.info(
            "Porte biologique : scoring_method='none' — Porte 2 bypassée, "
            "seule la Porte 1 (topologique) est active."
        )

    if bpf.enabled and bpf.scoring_method != "none":
        _has_xnorm = X_norm is not None and marker_names is not None
        _has_medians = node_medians is not None and marker_names is not None
        _has_inv_cov = nbm_inv_cov is not None

        if _has_xnorm or _has_medians:
            try:
                _has_nbm_stats = nbm_center is not None and nbm_scale is not None

                # Log du mode effectif
                _mode_label = (
                    "Mode 1 — X_norm (z-scores pré-calculés)"
                    if _has_xnorm
                    else "Mode 2 — node_medians + stats NBM"
                )
                _logger.info(
                    "Porte biologique — scoring_method='%s' | %s | inv_cov=%s",
                    bpf.scoring_method,
                    _mode_label,
                    "OK"
                    if _has_inv_cov
                    else "absent (fallback linéaire si hybrid/mahal)",
                )

                # CR-1 FIX : les scores sont calculés sur unique_nodes (dans l'ordre)
                # → on construit une table de correspondance node_id → rank pour une
                # indexation sûre, indépendante de la continuité des IDs de nœuds.
                _node_rank_map: Dict[int, int] = {
                    int(nid): rank for rank, nid in enumerate(unique_nodes)
                }

                node_blast_scores, node_d2_raw = _build_node_blast_scores(
                    marker_names=list(marker_names),
                    X_norm=np.asarray(X_norm, dtype=float) if _has_xnorm else None,
                    node_medians=np.asarray(node_medians, dtype=float)
                    if _has_medians
                    else None,
                    nbm_center=np.asarray(nbm_center, dtype=float)
                    if _has_nbm_stats
                    else None,
                    nbm_scale=np.asarray(nbm_scale, dtype=float)
                    if _has_nbm_stats
                    else None,
                    nbm_inv_cov=np.asarray(nbm_inv_cov, dtype=float)
                    if _has_inv_cov
                    else None,
                    patho_purity=_purity_per_node,
                    scoring_method=bpf.scoring_method,
                    mahal_weight=bpf.mahal_weight,
                    linear_weight=bpf.linear_weight,
                    d2_normalization=bpf.d2_normalization,
                    purity_modulation_power=bpf.purity_modulation_power,
                    mahal_boost_factor=bpf.mahal_boost_factor,
                    custom_weights=bpf.marker_weights,
                )

                # Le mode effectif est identique au mode demandé (plus de fallback silencieux).
                # ClinicalMathError est levée en amont si inv_cov est absent pour mahal/hybrid.

                node_blast_cats = [
                    categorize_blast_score(
                        float(s),
                        high_thresh=bpf.high_threshold,
                        mod_thresh=bpf.moderate_threshold,
                        weak_thresh=bpf.weak_threshold,
                    )
                    for s in node_blast_scores
                ]
                blast_filter_active = True
                _n_high = sum(1 for c in node_blast_cats if c == "BLAST_HIGH")
                _n_mod = sum(1 for c in node_blast_cats if c == "BLAST_MODERATE")
                _logger.info(
                    "Filtre phénotypique ACTIF — mode='%s' — %d nœuds : "
                    "%d BLAST_HIGH, %d BLAST_MODERATE (catégories acceptées : %s)",
                    _active_scoring_mode,
                    len(node_blast_scores),
                    _n_high,
                    _n_mod,
                    bpf.allowed_categories,
                )
            except ValueError:
                raise
            except Exception as exc:
                _logger.warning(
                    "Filtre phénotypique : erreur inattendue (%s) — porte biologique "
                    "désactivée pour cette analyse.",
                    exc,
                )
        else:
            _logger.warning(
                "blast_phenotype_filter.enabled=True mais ni X_norm ni node_medians "
                "n'est fourni avec marker_names — porte biologique désactivée."
            )

    # CR-3 FIX : logger explicitement l'état final de la porte biologique AVANT
    # la boucle, pour qu'un bypass silencieux soit immédiatement visible dans les logs.
    _logger.info(
        "État porte biologique (Porte 2) : %s",
        "ACTIVE — filtrage phénotypique appliqué" if blast_filter_active
        else "INACTIF — tous les nœuds topologiquement valides sont acceptés sans filtrage biologique",
    )

    # ── Porte Biologique : fonction évaluatrice (définie hors boucle) ────────
    # Sortie de la boucle for pour éviter la reconstruction à chaque itération.
    # Reçoit les valeurs du nœud courant en paramètres explicites.
    def _passes_blast_gate(
        apply_flag: bool,
        blast_score: Optional[float],
        blast_cat: Optional[str],
        node_purity: float,
    ) -> bool:
        """
        Routeur de la Porte Biologique — adapte sa logique au moteur actif.

        ── Comportement selon scoring_method ────────────────────────────────

        "none"  :  Bypass total. La porte est toujours ouverte.

        "linear":  Logique historique. Vérifie que blast_category figure dans
                   bpf.allowed_categories. Seuil fixe — pas de modulation pureté.

        "mahalanobis" / "hybrid" :
                   Seuil dynamique modulé par la pureté.
                   Purity ≥ purity_strict_above → seuil assoupli (confiance FlowSOM).
                   Nœud mixte → seuil nominal (la biologie compense).

        ── Seuil dynamique (mode mahalanobis/hybrid) ────────────────────────

        Exemple : purity_strict_above=0.85, relax_factor=0.90, high_threshold=4.0
          purity=100% → seuil = 4.0 × (1 - 0.90×1.0) = 0.40 (confiance aveugle)
          purity=90%  → seuil = 4.0 × (1 - 0.90×0.33) = 2.81
          purity=80%  → seuil = 4.0 (nœud mixte, seuil plein)

        Floor de sécurité : le seuil effectif ne descend jamais sous moderate_threshold.
        """
        if not blast_filter_active or not apply_flag:
            return True
        if bpf.scoring_method == "none":
            return True
        if blast_score is None or blast_cat is None:
            return True
        if _active_scoring_mode == "linear":
            return blast_cat in bpf.allowed_categories

        # Moteur mahalanobis / hybrid — seuil dynamique
        effective_threshold = bpf.high_threshold
        if bpf.purity_threshold_override and node_purity >= bpf.purity_strict_above:
            excess_purity = (node_purity - bpf.purity_strict_above) / max(
                1.0 - bpf.purity_strict_above, 1e-6
            )
            excess_purity = min(excess_purity, 1.0)
            relax = bpf.purity_relax_factor * excess_purity
            effective_threshold = max(
                bpf.high_threshold * (1.0 - relax),
                bpf.moderate_threshold,  # floor de sécurité
            )

        passes = blast_score >= effective_threshold
        if not passes and node_purity >= bpf.purity_strict_above:
            _logger.debug(
                "Nœud : score=%.2f < seuil_effectif=%.2f "
                "(pureté=%.1f%% → assouplissement actif) — REJETÉ.",
                blast_score,
                effective_threshold,
                node_purity * 100,
            )
        return passes

    per_node: List[MRDClusterResult] = []
    mrd_cells_jf = 0
    mrd_cells_flo = 0
    mrd_cells_eln = 0
    n_nodes_jf = 0
    n_nodes_flo = 0
    n_nodes_eln = 0

    # Dénominateurs globaux (protections contre division par zéro)
    _denom_sain   = total_sain  if total_sain  > 0 else 1
    _denom_patho_g = total_patho if total_patho > 0 else 1

    # ── Multiplicateur Flo adapté au ratio dataset sain/patho ────────────────
    # La méthode Flo compare pct_patho_intra vs pct_sain_intra (% dans le cluster).
    # Ces % sont biaisés par le déséquilibre global du dataset : si le dataset
    # contient 10× plus de cellules saines que patho, même un cluster MRD réel
    # aura un pct_patho_intra faible par construction.
    # Correction : on ajuste le multiplicateur par le ratio (total_sain/total_patho)
    # pour que le critère soit équitable quelle que soit la composition du dataset.
    # Exemple : ratio=10 (10× plus de sain), multiplier=2 → _flo_mult_eff = 2×10 = 20
    # → pct_patho > 20 × pct_sain_intra, ce qui correspond à "cluster dominé patho
    # au-delà de ce qu'on attendrait si les cellules étaient mélangées au hasard".
    _dataset_ratio_sain_patho = total_sain / max(total_patho, 1)
    # Floor à 1.0 : si sain ≪ patho (MRD très chargé ou dataset déséquilibré
    # dans l'autre sens), on ne descend pas sous le multiplicateur de base.
    _flo_mult_eff = mrd_cfg.method_flo.normal_marrow_multiplier * max(_dataset_ratio_sain_patho, 1.0)
    _logger.info(
        "Méthode Flo — ratio dataset sain/patho=%.2f → multiplicateur effectif=%.2f "
        "(base=%s × ratio=%.2f)",
        _dataset_ratio_sain_patho,
        _flo_mult_eff,
        mrd_cfg.method_flo.normal_marrow_multiplier,
        _dataset_ratio_sain_patho,
    )

    for node_id in unique_nodes:
        # ── Comptages vectorisés : lecture directe dans les tableaux bincount ──
        # node_id est l'index réel → pas de risque de désynchronisation
        n_in_node = int(_counts_total[node_id])
        if n_in_node == 0:
            continue

        n_sain  = int(_counts_sain[node_id])
        n_patho = int(_counts_patho[node_id])

        # pct_sain / pct_patho : % AU SEIN du cluster (Flo et ELN)
        pct_sain  = (n_sain  / n_in_node) * 100.0
        pct_patho = (n_patho / n_in_node) * 100.0

        # Pureté du nœud : fraction de cellules patient [0, 1]
        node_purity = n_patho / n_in_node

        # pct_sain_global : % des cellules saines / total moelle normale (méthode JF)
        pct_sain_global  = (n_sain  / _denom_sain)    * 100.0
        # pct_patho_global : % des cellules patho / total cellules patho
        pct_patho_global = (n_patho / _denom_patho_g) * 100.0

        # ── Porte Biologique pour ce nœud ─────────────────────────────────
        # CR-1 FIX : node_blast_scores est indexé par RANG dans unique_nodes
        # (pas par node_id). On utilise _node_rank_map pour la correspondance sûre.
        _blast_score: Optional[float] = None
        _blast_cat: Optional[str] = None
        _mahal_d2: Optional[float] = None

        if node_blast_scores is not None:
            _rank = _node_rank_map.get(int(node_id))
            if _rank is not None:
                _blast_score = float(node_blast_scores[_rank])
                _blast_cat = (
                    node_blast_cats[_rank] if node_blast_cats is not None else None
                )
                if node_d2_raw is not None:
                    _mahal_d2 = float(node_d2_raw[_rank])

        # ── Méthode JF ────────────────────────────────────────────────
        # Critère 1 (min_patho_cells_pct) : le cluster doit être dominé par les
        # cellules pathologiques (% intra-cluster). Si insuffisant, on stoppe.
        # Critère 2 (max_normal_marrow_pct) : parmi toutes les cellules saines du
        # pool de moelle normale (total_sain), la fraction contenue dans ce cluster
        # doit être inférieure au seuil — i.e. n_sain_cluster / total_sain < seuil.
        # Exemple : 1180 cellules saines / 980 000 total sain = 0.12 % → si seuil=0.10,
        # le cluster est REJETÉ (0.12 > 0.10).
        is_mrd_jf = False
        if run_jf:
            if pct_patho > mrd_cfg.method_jf.min_patho_cells_pct:
                if pct_sain_global < mrd_cfg.method_jf.max_normal_marrow_pct:
                    if _passes_blast_gate(bpf.apply_to_jf, _blast_score, _blast_cat, node_purity):
                        is_mrd_jf = True
                        mrd_cells_jf += n_patho
                        n_nodes_jf += 1

        # ── Méthode Flo (ratio intra-cluster corrigé du biais dataset) ───────
        # _flo_mult_eff = normal_marrow_multiplier × (total_sain / total_patho).
        # Cette correction compense le déséquilibre de composition du dataset :
        # si le dataset contient 10× plus de sain que de patho, un mélange aléatoire
        # donnerait pct_sain_intra ≈ 10× pct_patho_intra → on rehausse le seuil
        # d'autant pour ne valider que les clusters réellement dominés par le patho.
        # Cas limite : nœud 100% patho (pct_sain=0) → MRD si min LOQ respecté.
        is_mrd_flo = False
        if run_flo:
            _flo_criterion = False
            if pct_sain > 0:
                _flo_criterion = pct_patho > pct_sain * _flo_mult_eff
            else:
                # Nœud 100% pathologique : MRD si assez de cellules (robustesse LOQ)
                _flo_criterion = n_patho >= mrd_cfg.eln_standards.min_cluster_events
            if _flo_criterion:
                if _passes_blast_gate(bpf.apply_to_flo, _blast_score, _blast_cat, node_purity):
                    is_mrd_flo = True
                    mrd_cells_flo += n_patho
                    n_nodes_flo += 1

        # ── Méthode ELN (DfN — Different from Normal, enrichissement intra-cluster)
        # CR-2 FIX : le critère DfN ELN 2022 compare les proportions intra-cluster
        # (pct_patho vs pct_sain DANS le nœud), pas les % globaux.
        # Référence : Schuurhuis et al. Blood 2018 — "more patient than normal cells".
        is_mrd_eln = False
        if run_eln:
            if n_in_node >= mrd_cfg.eln_standards.min_cluster_events:
                if pct_patho > pct_sain:
                    if _passes_blast_gate(bpf.apply_to_eln, _blast_score, _blast_cat, node_purity):
                        is_mrd_eln = True
                        mrd_cells_eln += n_patho
                        n_nodes_eln += 1

        per_node.append(
            MRDClusterResult(
                cluster_id=int(node_id),
                n_cells_total=n_in_node,
                n_cells_sain=n_sain,
                n_cells_patho=n_patho,
                pct_sain=pct_sain,
                pct_patho=pct_patho,
                pct_sain_global=pct_sain_global,
                is_mrd_jf=is_mrd_jf,
                is_mrd_flo=is_mrd_flo,
                is_mrd_eln=is_mrd_eln,
                blast_score=_blast_score,
                blast_category=_blast_cat,
                mahal_d2=_mahal_d2,
                node_purity=node_purity,
            )
        )

    # MRD % = cellules MRD / dénominateur patho
    _denom_patho = total_patho_cd45pos if total_patho_cd45pos > 0 else 1
    _has_patho = total_patho_cd45pos > 0
    mrd_pct_jf = (mrd_cells_jf / _denom_patho * 100.0) if _has_patho else 0.0
    mrd_pct_flo = (mrd_cells_flo / _denom_patho * 100.0) if _has_patho else 0.0
    mrd_pct_eln = (mrd_cells_eln / _denom_patho * 100.0) if _has_patho else 0.0

    # Statut clinique ELN
    eln_positive = mrd_pct_eln >= mrd_cfg.eln_standards.clinical_positivity_pct
    eln_low_level = (mrd_cells_eln > 0) and (not eln_positive)

    result = MRDResult(
        method_used=mrd_cfg.method,
        total_cells=n_total,
        total_cells_patho=total_patho,
        total_cells_sain=total_sain,
        mrd_denominator=total_patho_cd45pos,
        mrd_denominator_mode=cd45_autogating_mode,
        n_patho_cd45pos=n_patho_cd45pos,
        n_patho_pre_cd45=total_patho,
        blast_filter_active=blast_filter_active,
        blast_scoring_mode=_active_scoring_mode,
        mrd_cells_jf=mrd_cells_jf,
        mrd_pct_jf=mrd_pct_jf,
        n_nodes_mrd_jf=n_nodes_jf,
        mrd_cells_flo=mrd_cells_flo,
        mrd_pct_flo=mrd_pct_flo,
        n_nodes_mrd_flo=n_nodes_flo,
        mrd_cells_eln=mrd_cells_eln,
        mrd_pct_eln=mrd_pct_eln,
        n_nodes_mrd_eln=n_nodes_eln,
        eln_positive=eln_positive,
        eln_low_level=eln_low_level,
        per_node=per_node,
        config_snapshot={
            # ── Méthodes de détection ───────────────────────────────────────────
            "method": mrd_cfg.method,
            "jf_max_normal_pct": mrd_cfg.method_jf.max_normal_marrow_pct,
            "jf_min_patho_pct": mrd_cfg.method_jf.min_patho_cells_pct,
            "flo_multiplier": mrd_cfg.method_flo.normal_marrow_multiplier,
            "eln_min_events": mrd_cfg.eln_standards.min_cluster_events,
            "eln_positivity_pct": mrd_cfg.eln_standards.clinical_positivity_pct,

            # ── Porte biologique (Porte 2) ─────────────────────────────────────
            "blast_filter_enabled": bpf.enabled,
            "blast_filter_categories": bpf.allowed_categories,

            # ── Méthode de scoring utilisée EFFECTIVEMENT ─────────────────────
            # Valeur reflétant le scoring réellement appliqué (pas la valeur demandée
            # dans la config — capture les cas où le scoring a été modifié).
            "scoring_method_effective": _active_scoring_mode,
            "scoring_method_requested": bpf.scoring_method,
            "mahal_weight": bpf.mahal_weight,
            "linear_weight": bpf.linear_weight,
            "d2_normalization": bpf.d2_normalization,

            # ── Variables de pureté utilisées ─────────────────────────────────
            # Capture les paramètres exacts de modulation par la pureté topologique.
            "purity_modulation_power": bpf.purity_modulation_power,
            "purity_strict_above": bpf.purity_strict_above,
            "purity_relax_factor": bpf.purity_relax_factor,
            "purity_threshold_override": bpf.purity_threshold_override,
            "high_threshold": bpf.high_threshold,
            "moderate_threshold": bpf.moderate_threshold,
            "weak_threshold": bpf.weak_threshold,

            # ── Traçabilité des fichiers de configuration YAML ────────────────
            # Hash SHA256 des fichiers YAML utilisés : garantit la reproductibilité
            # exacte du calcul et détecte les modifications non documentées.
            "mrd_config_yaml_hash": _hash_yaml_file(mrd_config_path),
            "panel_config_yaml_hash": _hash_yaml_file(panel_config_path),
            "mrd_config_yaml_path": str(mrd_config_path) if mrd_config_path else None,
            "panel_config_yaml_path": str(panel_config_path) if panel_config_path else None,
        },
    )

    _logger.info(
        "MRD JF : %d cellules patho dans %d nœuds SOM → MRD = %.4f%%%s",
        mrd_cells_jf,
        n_nodes_jf,
        mrd_pct_jf,
        " [+porte biologique]" if blast_filter_active and bpf.apply_to_jf else "",
    )
    _logger.info(
        "MRD Flo: %d cellules patho dans %d nœuds SOM → MRD = %.4f%%%s",
        mrd_cells_flo,
        n_nodes_flo,
        mrd_pct_flo,
        " [+porte biologique]" if blast_filter_active and bpf.apply_to_flo else "",
    )
    _logger.info(
        "MRD ELN: %d cellules patho dans %d nœuds SOM → MRD = %.4f%% — %s%s",
        mrd_cells_eln,
        n_nodes_eln,
        mrd_pct_eln,
        "POSITIVE" if eln_positive else ("LOW-LEVEL" if eln_low_level else "NEGATIVE"),
        " [+porte biologique]" if blast_filter_active and bpf.apply_to_eln else "",
    )

    return result
