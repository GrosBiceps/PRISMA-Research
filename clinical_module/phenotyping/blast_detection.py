"""
blast_detection.py — Scoring et classification phénotypique des nœuds SOM en blastes.

═══════════════════════════════════════════════════════════════════════════════
  FONDEMENTS BIOLOGIQUES ET CLINIQUES
═══════════════════════════════════════════════════════════════════════════════

Ce module implémente un scoring heuristique inspiré de deux référentiels
cliniques majeurs pour la détection des blastes LAM par cytométrie en flux :

  1. Score d'Ogata (Ogata et al., Blood 2006 ; Sandes et al., Cytometry B 2013)
     ─────────────────────────────────────────────────────────────────────────
     Conçu initialement pour le diagnostic des syndromes myélodysplasiques
     (SMD) et adapté à la LAM, ce score évalue la déviation phénotypique des
     blastes par rapport à une moelle normale de référence. Les critères
     biologiques retenus sont :

       • CD34 bright       → Marqueur de cellule souche hématopoïétique /
                              progéniteur immature. Surexprimé dans les LAM
                              à progéniteurs. Poids fort : +3.0

       • CD117 (c-Kit)     → Récepteur de la stem cell factor (SCF). Marqueur
                              clé des progéniteurs myéloïdes précoces (CFU-GM).
                              Complémentaire à CD34 dans les LAM CD34−. Poids : +2.5

       • CD45 dim          → Dans la moelle normale, les blastes sont CD45-
                              faibles (population "blasts gate" sur CD45/SSC).
                              Un signal CD45 très bas est discriminant pour les
                              blastes LAM, conformément au score d'Ogata
                              (critère morphologique CD45-dim). Poids : −2.0
                              (contribution quand valeur en-dessous du plancher
                              de la référence normale)

       • SSC bas           → Faible granularité cytoplasmique. Critère
                              morphologique classique des blastes (taille petite,
                              peu de granules). Aligné sur le quadrant
                              SSC-bas / CD45-dim du score d'Ogata. Poids : −1.0

  2. Recommandations ELN 2022 (Schuurhuis et al., Blood 2018 ; Heuser et al.,
     Leukemia 2022 ; ELN MRD Working Party 2022)
     ─────────────────────────────────────────────────────────────────────────
     L'ELN définit les LAIPs (Leukemia-Associated ImmunoPhénotypes) comme des
     combinaisons anormales de marqueurs permettant de traquer la MRD :

       • HLA-DR positif    → Blaste myéloïde mature (monocytaire ou granulocytaire).
                              Fortement exprimé sur les blastes LAM M4/M5. Poids : +1.5

       • CD33 variable     → Engagement myéloïde (antigène pan-myéloïde).
                              Présent à des niveaux anormaux sur les blastes.
                              Poids : +1.0

       • CD13 variable     → Engagement myéloïde (amino-peptidase N). Souvent
                              co-exprimé avec CD33. Poids : +0.5

       • CD19 / CD3 pos    → Marqueurs lymphoïdes B et T. Leur présence
                              suggère une population normale ou un LAIP
                              lympho-myéloïde rare. Considérés ici comme
                              freins biologiques anti-blaste. Poids : −1.5

═══════════════════════════════════════════════════════════════════════════════
  SCORE COMPOSITE ET SEUILS DE CATÉGORISATION
═══════════════════════════════════════════════════════════════════════════════

Le score composite /10 est calculé comme une somme pondérée de contributions
directionnelles (cf. score_nodes_for_blasts) :

  • Marqueurs positifs (CD34, CD117, HLA-DR, CD33, CD13) :
    Contribuent quand leur valeur normalisée dépasse le plafond de la
    référence (valeur > 1.0 dans l'espace normalisé), c'est-à-dire quand
    ils sont surexprimés par rapport à la moelle normale.

  • Marqueurs négatifs (CD45, SSC) :
    Contribuent quand leur valeur normalisée est sous le plancher de la
    référence (valeur < 0.0), c'est-à-dire quand ils sont sous-exprimés
    (CD45-dim, SSC-bas).

  • Marqueurs frein (CD19, CD3) :
    Contribuent négativement au score quand surexprimés (population non-blaste).

Seuils de catégorisation (heuristiques calibrables) :
──────────────────────────────────────────────────────
  BLAST_HIGH     ≥ 6.0 / 10   Équivalent à ≥2 anomalies majeures (LAIP fort).
                               Interprétation clinique : fort indice de blaste LAM.
                               Recommandation : confirmer par autre méthode.

  BLAST_MODERATE ≥ 3.0 / 10   Équivalent à 1 anomalie majeure + anomalies mineures.
                               Interprétation : phénotype atypique, surveillance requise.

  BLAST_WEAK     > 0.0 / 10   Signal léger. Population atypique mais non décisive.

  NON_BLAST_UNK  = 0.0 / 10   Aucune signature blastique décelable.

⚠  Ces seuils sont des HEURISTIQUES d'initialisation, non des valeurs ELN
   officiellement validées. Ils doivent être calibrés sur une cohorte locale
   via une analyse ROC (AUC blast vs non-blast) avant utilisation diagnostique.
   La procédure recommandée est :
     1. Annoter manuellement ≥50 nœuds (blast / non-blast) sur des cas connus.
     2. Calculer l'AUC du score composite.
     3. Choisir le seuil BLAST_HIGH maximisant sensibilité × spécificité (Youden J).
     4. Mettre à jour BLAST_HIGH_THRESHOLD et BLAST_MODERATE_THRESHOLD ci-dessous.

Références :
  - Ogata K. et al. (2006) Diagnostic utility of flow cytometry in low-grade
    myelodysplastic syndromes. Haematologica.
  - Sandes A.F. et al. (2013) Flow cytometric diagnosis of myelodysplastic
    syndromes using the Ogata score. Cytometry B Clin Cytom.
  - Schuurhuis G.J. et al. (2018) Minimal/measurable residual disease in AML:
    a consensus document from the ELN MRD Working Party. Blood.
  - Heuser M. et al. (2022) ELN MRD Working Party consensus on MRD in AML.
    Leukemia.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from prisma.exceptions import ClinicalMathError, PanelConfigError
from prisma.utils.logger import get_logger

# Imports optionnels — dégradation gracieuse si scipy/sklearn absents
try:
    from scipy.linalg import pinv as _scipy_pinv

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

try:
    from sklearn.covariance import MinCovDet as _MinCovDet

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False

_logger = get_logger("analysis.blast_detection")

# ─────────────────────────────────────────────────────────────────────────────
#  Seuils de catégorisation (score /10)
#
#  BLAST_HIGH_THRESHOLD     = 6.0 → ≥2 anomalies phénotypiques majeures (LAIP fort)
#  BLAST_MODERATE_THRESHOLD = 2.0 → blastes matures CD34- (CD45-dim/HLA-DR), anciennement 3.0
#  BLAST_WEAK_THRESHOLD     = 0.0 → signal positif minimal
#
#  ⚠  Valeurs heuristiques — calibrer via ROC sur cohorte locale avant usage clinique.
# ─────────────────────────────────────────────────────────────────────────────
BLAST_HIGH_THRESHOLD = 6.0
BLAST_MODERATE_THRESHOLD = 2.0
BLAST_WEAK_THRESHOLD = 0.0


def load_panel_weights(panel_path: str | Path) -> Dict[str, float]:
    """
    Charge les poids de marqueurs depuis un fichier YAML de configuration de panel.

    Le fichier doit contenir une clé ``marker_weights`` mappant les patterns de
    marqueurs à leurs poids (float). Les patterns sont matchés par substring
    insensible à la casse sur le nom complet du marqueur.

    Args:
        panel_path: Chemin vers le fichier YAML du panel (ex: config/panels/aml_panel.yaml).

    Returns:
        Dict {pattern: poids_float} — utilisé comme ``custom_weights`` dans
        ``build_blast_weights``.

    Raises:
        PanelConfigError: Si le fichier est absent, illisible ou ne contient
            pas la clé ``marker_weights``.
    """
    path = Path(panel_path)
    if not path.exists():
        raise PanelConfigError(
            message=(
                f"Fichier de configuration du panel introuvable : '{path}'. "
                "Vérifiez le chemin dans la configuration ou fournissez "
                "un panel valide (ex: config/panels/aml_panel.yaml)."
            ),
            panel_path=str(path),
        )

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except Exception as exc:
        raise PanelConfigError(
            message=f"Impossible de lire le fichier de panel YAML : {exc}",
            panel_path=str(path),
        ) from exc

    if not isinstance(data, dict) or "marker_weights" not in data:
        raise PanelConfigError(
            message=(
                f"Le fichier de panel '{path}' ne contient pas la clé 'marker_weights'. "
                "Structure attendue : marker_weights:\n  CD34: 3.0\n  CD117: 2.5\n  ..."
            ),
            panel_path=str(path),
        )

    raw = data["marker_weights"]
    if not isinstance(raw, dict) or not raw:
        raise PanelConfigError(
            message=(
                f"La clé 'marker_weights' dans '{path}' est vide ou invalide. "
                "Au moins un marqueur avec son poids doit être défini."
            ),
            panel_path=str(path),
        )

    weights: Dict[str, float] = {}
    for pattern, value in raw.items():
        try:
            weights[str(pattern)] = float(value)
        except (TypeError, ValueError) as exc:
            raise PanelConfigError(
                message=(
                    f"Poids invalide pour le marqueur '{pattern}' dans '{path}' : "
                    f"'{value}' ne peut pas être converti en float."
                ),
                panel_path=str(path),
            ) from exc

    _logger.info(
        "load_panel_weights : %d poids chargés depuis '%s'",
        len(weights),
        path,
    )
    return weights


def build_blast_weights(
    marker_names: List[str],
    custom_weights: Optional[Dict[str, float]] = None,
    panel_path: Optional[str | Path] = None,
) -> Dict[str, float]:
    """
    Construit le dictionnaire de poids pour le scoring blast ELN 2022 / Ogata.

    Chaque poids est assigné par correspondance de pattern sur le nom du marqueur.
    Les valeurs sont calibrées selon leur importance diagnostique dans la LAM
    (cf. module docstring pour le rationnel clinique complet).

    Hiérarchie des sources de poids (priorité décroissante) :
      1. ``panel_path`` : fichier YAML du panel (config/panels/aml_panel.yaml).
         Fournit les poids spécifiques à la maladie. Prioritaire sur tout.
      2. ``custom_weights`` : dict passé directement (depuis mrd_config.yaml).
         Écrase les poids du panel ou les valeurs par défaut.
      3. Valeurs par défaut internes : poids biologiques standards LAM.
         Utilisés uniquement si ni panel_path ni custom_weights ne sont fournis.

    Poids par défaut (LAM — utilisés si aucun panel n'est fourni) :
      +3.0  CD34      — progéniteur hématopoïétique (Ogata critère 1)
      +2.5  CD117     — c-Kit, progéniteur myéloïde (Ogata critère 2)
      +1.5  HLA-DR    — blaste myéloïde mature (ELN 2022 LAIP)
      +1.0  CD33      — engagement myéloïde (ELN 2022 LAIP)
      +0.5  CD13      — engagement myéloïde secondaire (ELN 2022 LAIP)
      −1.0  CD45      — CD45-dim discrimine les blastes (score Ogata)
      −1.5  CD19/CD3  — marqueurs lymphoïdes → anti-blaste (frein biologique)
      −1.0  SSC       — faible granularité cytoplasmique = morphologie blaste (Ogata)

    Args:
        marker_names: Liste des noms de marqueurs présents dans le panel.
        custom_weights: Dict optionnel {pattern: poids} chargé depuis mrd_config.yaml
            (clé marker_weights). Les clés sont matchées par pattern insensible à la
            casse sur le nom complet du marqueur (ex: "CD45" matche "CD45-PECY5").
            Écrase les poids du panel ou les valeurs par défaut.
        panel_path: Chemin optionnel vers un fichier YAML de panel
            (ex: "config/panels/aml_panel.yaml"). Si fourni, charge les poids
            depuis ce fichier en priorité. Lève PanelConfigError si absent.

    Returns:
        Dict {nom_marqueur: poids_float} — poids = 0.0 pour les marqueurs
        non reconnus (neutres, sans contribution au score).

    Raises:
        PanelConfigError: Si panel_path est fourni mais que le fichier est absent
            ou invalide.

    Note:
        La correspondance est insensible à la casse et tolère les suffixes
        courants (ex: "CD34-FITC", "CD34_BV421", "CD117/PE").
    """
    # ── Chargement des poids depuis le panel YAML (priorité maximale) ────────
    panel_weights: Optional[Dict[str, float]] = None
    if panel_path is not None:
        # Lève PanelConfigError si le fichier est absent ou invalide
        panel_weights = load_panel_weights(panel_path)

    weights: Dict[str, float] = {}

    for name in marker_names:
        upper = name.upper()

        # ── Résolution du poids : panel → custom_weights → valeurs par défaut ─
        resolved: Optional[float] = None

        # 1. Panel YAML (si fourni)
        if panel_weights:
            for pattern, value in panel_weights.items():
                if pattern.upper() in upper:
                    resolved = value
                    break

        # 2. custom_weights (override depuis mrd_config.yaml)
        if custom_weights:
            for pattern, value in custom_weights.items():
                if pattern.upper() in upper:
                    resolved = float(value)
                    break

        # 3. Valeurs par défaut internes (si aucune source externe n'a fourni de poids)
        if resolved is None:
            if "CD34" in upper:
                resolved = +3.0
            elif "CD117" in upper or "CKIT" in upper:
                resolved = +2.5
            elif "CD45" in upper:
                resolved = -1.0
            elif "HLAD" in upper or "HLA-DR" in upper:
                resolved = +1.5
            elif "CD33" in upper:
                resolved = +1.0
            elif "CD13" in upper:
                resolved = +0.5
            elif "CD19" in upper or ("CD3" in upper and "CD34" not in upper):
                resolved = -1.5
            elif "SSC" in upper:
                resolved = -1.0
            else:
                resolved = 0.0

        weights[name] = resolved

    return weights


def score_nodes_for_blasts(
    X_norm: np.ndarray,
    marker_names: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Calcule le score blast /10 pour chaque nœud SOM à partir de médianes z-scorées.

    ── Espace de représentation : z-score par rapport à la moelle normale ───────

    X_norm contient des **z-scores** calculés par rapport aux statistiques de la
    population de référence (moelle normale / NBM) :

        z[j] = (mediane_nœud[j] − median_ref[j]) / std_ref[j]

    Dans cet espace :
      • z ≈ 0   → le marqueur est au niveau médian de la moelle normale
      • z > +0.5 → surexpression au-dessus du NBM (seuil de sensibilité)
      • z < −0.5 → sous-expression en-dessous du NBM (CD45-dim, SSC-bas)

    ── Logique de contribution directionnelle (3 règles) ───────────────────────

    Seuil de déclenchement : ±0.5 SD (abaissé vs l'ancienne valeur de 1.0 SD
    pour capturer les blastes matures ayant perdu une partie de leur LAIP).

      1. Marqueurs blastiques à poids positif (CD34, CD117, HLA-DR, CD33, CD13) :
           → +points si z > +0.5 (surexpression dès 0.5 SD au-dessus du NBM).
           → Contribution = w × max(0, z − 0.5)
           → Interprétation ELN 2022 : CD34++ ou CD117++ = LAIP positif.

      2. Marqueurs de maturation à poids négatif (CD45, SSC) — récompense :
           → +points si z < −0.5 (sous-expression caractéristique des blastes).
           → Contribution = |w| × max(0, −z − 0.5)
           → Interprétation Ogata : CD45-dim / SSC-bas = zone « blasts gate ».

      3. Marqueurs de maturation à poids négatif (CD45, SSC) — PÉNALITÉ :
           → −points si z > 0 (surexpression = cellule mature normale).
           → Pénalité = |w| × max(0, z)
           → Élimine les faux positifs : monocytes HLA-DR+/CD33+ accumuleraient
             des points sur d'autres marqueurs sans être punis par leur CD45/SSC
             normal. La pénalité est nécessaire pour stopper ces faux positifs.

    ── Calibration attendue (arcsinh/5, référence NBM médiane+std) ─────────────

      Blaste typique  (CD34++, CD117+, CD45-dim, SSC-bas) → score ≈ 6–8 / 10
      Lymphocyte sain (CD45-bright, SSC-bas, CD34−)       → score ≈ 0–1 / 10
      Granulocyte sain (SSC-high, CD45+, CD34−)           → score ≈ 0   / 10

    ── Normalisation du score final ────────────────────────────────────────────

    Score brut clampé à 0 (pas de score négatif), puis normalisé par le maximum
    théorique (somme des |poids| non nuls) × 10, et clampé dans [0, 10].

    Args:
        X_norm: Matrice **z-scorée** [n_nodes, n_markers].
                Axe 0 = nœuds SOM, axe 1 = marqueurs.
                Produite par compute_reference_normalization() avec mode="zscore".
                z ≈ 0 = niveau NBM ; z > 0.5 = surexpression ; z < −0.5 = sous-expr.
        marker_names: Noms des marqueurs (colonnes de X_norm, dans le même ordre).
        weights: Poids pré-calculés (optionnel).
                 Si None, calculés automatiquement via build_blast_weights().

    Returns:
        np.ndarray de forme (n_nodes,), valeurs dans [0.0, 10.0].
        Score = 0.0 → aucune déviation blastique détectée (ou cellule mature pénalisée).
        Score ≈ 6–8 → signature blastique typique (CD34++ + CD117++ + CD45-dim).

    Example:
        >>> ref_med, ref_std, inv_cov = compute_reference_stats(X_nbm)
        >>> X_zscore = (node_medians - ref_med) / ref_std
        >>> scores = score_nodes_for_blasts(X_zscore, marker_names)
    """
    if weights is None:
        weights = build_blast_weights(marker_names)

    # Vecteur de poids aligné sur marker_names — (n_markers,)
    W = np.array([weights.get(m, 0.0) for m in marker_names], dtype=float)

    # Dénominateur = somme des poids absolus (pour normaliser en /10)
    max_theoretical = max(float(np.sum(np.abs(W[W != 0]))), 1e-6)

    # PERF-1 FIX : vectorisation complète — suppression de la boucle Python marker/marker.
    # Toutes les opérations sont réduites à des produits matriciels vectorisés O(N×M).
    #
    # X_norm : (n_nodes, n_markers)
    # W_pos  : poids positifs (CD34, CD117, HLA-DR…) — contribuent si z > +0.5
    # W_neg  : valeurs absolues des poids négatifs (CD45, SSC)
    #   → récompense si z < -0.5  (CD45-dim, SSC-bas = signature blaste)
    #   → pénalité  si z >  0.0  (CD45-bright, SSC-haut = cellule mature)

    W_pos = np.where(W > 0, W,  0.0)   # (n_markers,)
    W_neg = np.where(W < 0, -W, 0.0)   # valeurs absolues des poids négatifs

    # Contributions positives vectorisées : dot((z - 0.5)+, W_pos)
    contrib_pos = np.dot(np.maximum(0.0, X_norm - 0.5), W_pos)          # (n_nodes,)

    # Récompense négative : dot((-z - 0.5)+, W_neg)
    contrib_neg_reward = np.dot(np.maximum(0.0, -X_norm - 0.5), W_neg)  # (n_nodes,)

    # Pénalité : -dot(z+, W_neg)  — stoppe les monocytes/granulocytes
    contrib_neg_penalty = -np.dot(np.maximum(0.0, X_norm), W_neg)        # (n_nodes,)

    # Normalisation stricte : clamp à 0 avant normalisation /10
    scores_raw = np.maximum(0.0, contrib_pos + contrib_neg_reward + contrib_neg_penalty)
    scores_10 = np.clip(scores_raw / max_theoretical * 10.0, 0.0, 10.0)
    return scores_10


def categorize_blast_score(
    score: float,
    high_thresh: float = BLAST_HIGH_THRESHOLD,
    mod_thresh: float = BLAST_MODERATE_THRESHOLD,
    weak_thresh: float = BLAST_WEAK_THRESHOLD,
) -> str:
    """
    Classifie un score blast /10 en catégorie clinique ELN 2022 / Ogata.

    ── Correspondance avec les critères cliniques ───────────────────────────────

    BLAST_HIGH (≥ high_thresh, défaut 6.0) :
      Correspond à ≥2 anomalies phénotypiques majeures simultanées (ex: CD34++
      ET CD117++, ou CD34++ ET CD45-dim). Équivalent à un LAIP fort selon ELN 2022
      (Schuurhuis et al., Blood 2018, Table 2). Recommande une confirmation par
      morphologie ou biologie moléculaire.

    BLAST_MODERATE (≥ mod_thresh, défaut 2.0) :
      1 anomalie majeure ou combinaison d'anomalies mineures. Abaissé à 2.0 pour
      capturer les blastes matures ayant perdu le CD34 mais conservant CD45-dim
      (+2.0 pts) ou HLA-DR (+1.5 pts) — typique des leucémies massives avancées.
      Configurable via blast_phenotype_filter.moderate_threshold dans mrd_config.yaml.

    BLAST_WEAK (> weak_thresh, défaut 0.0) :
      Signal faible — population atypique sans seuil d'alarme clinique. Utile
      pour la traçabilité et la détection précoce sur séries longitudinales.

    NON_BLAST_UNK (= 0.0) :
      Aucune signature détectable. Nœud très probablement composé de cellules
      hématopoïétiques matures normales (granulocytes, lymphocytes, monocytes).

    ⚠  Rappel : ces seuils sont heuristiques. Calibrer via ROC avant usage
       diagnostique (cf. module docstring).

    Args:
        score: Score blast dans [0.0, 10.0] produit par score_nodes_for_blasts().
        high_thresh: Seuil BLAST_HIGH (défaut : constante globale BLAST_HIGH_THRESHOLD).
        mod_thresh: Seuil BLAST_MODERATE (défaut : constante globale BLAST_MODERATE_THRESHOLD).
        weak_thresh: Seuil BLAST_WEAK (défaut : constante globale BLAST_WEAK_THRESHOLD).

    Returns:
        Catégorie : "BLAST_HIGH" | "BLAST_MODERATE" | "BLAST_WEAK" | "NON_BLAST_UNK".
    """
    if score >= high_thresh:
        # ≥2 anomalies majeures — LAIP fort (ELN 2022)
        return "BLAST_HIGH"
    elif score >= mod_thresh:
        # 1 anomalie majeure ou combinaison mineures — phénotype intermédiaire
        return "BLAST_MODERATE"
    elif score > weak_thresh:
        # Signal léger — population atypique à surveiller
        return "BLAST_WEAK"
    # Score nul — aucune déviation phénotypique par rapport à la moelle normale
    return "NON_BLAST_UNK"


def build_blast_score_dataframe(
    node_ids: np.ndarray,
    X_norm: np.ndarray,
    marker_names: List[str],
    cell_counts_per_node: Optional[Dict[int, int]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Construit le DataFrame complet de scoring blast ELN 2022 / Ogata pour un
    ensemble de nœuds SOM.

    Wrapper de haut niveau combinant score_nodes_for_blasts() et
    categorize_blast_score(). Utilisé principalement pour l'export JSON,
    les rapports HTML/PDF et la visualisation des candidats blastiques.

    Le DataFrame résultant est trié par blast_score décroissant pour faciliter
    l'inspection manuelle des nœuds les plus suspects.

    Args:
        node_ids: IDs des nœuds SOM à scorer, shape (n_nodes,).
                  Doit correspondre en ordre aux lignes de X_norm.
        X_norm: Matrice normalisée [n_nodes, n_markers] dans l'espace de référence.
        marker_names: Noms des marqueurs (colonnes de X_norm).
        cell_counts_per_node: {node_id: n_cells} pour enrichir le rapport.
                               Permet d'évaluer la robustesse statistique du score
                               (un nœud avec 5 cellules et score élevé est moins
                               fiable qu'un nœud avec 500 cellules).
        weights: Poids pré-calculés (optionnel — auto-calculés si None).

    Returns:
        pd.DataFrame trié par blast_score décroissant, avec colonnes :
          - node_id, blast_score, blast_category
          - n_cells (si cell_counts_per_node fourni)
          - {marqueur}_M8 : valeur normalisée par marqueur (suffixe _M8 = espace
            de normalisation, où M8 = matrice de référence 8 populations)
    """
    if weights is None:
        weights = build_blast_weights(marker_names)

    scores = score_nodes_for_blasts(X_norm, marker_names, weights)
    categories = [categorize_blast_score(float(s)) for s in scores]

    records: Dict = {
        "node_id": node_ids,
        "blast_score": np.round(scores, 2),
        "blast_category": categories,
    }

    if cell_counts_per_node:
        records["n_cells"] = [cell_counts_per_node.get(int(nid), 0) for nid in node_ids]

    df = pd.DataFrame(records)

    # Ajouter les valeurs normalisées par marqueur (suffixe _M8 = espace référence)
    for j, m in enumerate(marker_names):
        df[f"{m}_M8"] = np.round(X_norm[:, j], 3)

    df = df.sort_values("blast_score", ascending=False).reset_index(drop=True)

    # Log du résumé par catégorie
    for cat in ["BLAST_HIGH", "BLAST_MODERATE", "BLAST_WEAK", "NON_BLAST_UNK"]:
        n = int((df["blast_category"] == cat).sum())
        if n > 0:
            _logger.info("  %s: %d nœud(s)", cat, n)

    return df


def compute_reference_stats(
    X_reference: np.ndarray,
    robust: bool = True,
    regularization: float = 1e-4,
    max_samples_for_stats: int = 500000,
    max_samples_for_covariance: int = 250000,
    max_samples_for_mincovdet: int = 60000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Calcule les statistiques de la population de référence (moelle normale / NBM)
    nécessaires au z-scoring et à la distance de Mahalanobis des nœuds SOM.

    ── Pourquoi le z-score plutôt que min-max ? ─────────────────────────────────

    La normalisation min-max [0, 1] place TOUTES les populations (y compris les
    blastes les plus extrêmes) dans l'intervalle [0, 1]. Dans cet espace,
    score_nodes_for_blasts() ne peut jamais trouver de valeurs > 1 ou < 0,
    donc le score est systématiquement 0 pour tous les nœuds.

    Le z-score par rapport à la médiane + IQR/1.35 de la moelle normale place
    les blastes À L'EXTÉRIEUR de l'espace de référence (z > +1 ou z < −1),
    permettant à la logique directionnelle de score_nodes_for_blasts() de
    détecter les déviations phénotypiques ELN 2022 / Ogata.

    ── Matrice de covariance inverse (Mahalanobis) ──────────────────────────────

    La 3e valeur retournée, inv_cov, est l'inverse de la matrice de covariance
    de la population NBM. Elle sert à score_nodes_mahalanobis() pour calculer :

        D²(x) = (x - μ_NBM)ᵀ × Σ_NBM⁻¹ × (x - μ_NBM)

    Stratégie de calcul :
      • robust=True  : tente MinCovDet (sklearn) pour une estimation robuste de la
                       covariance résistant aux outliers blastiques résiduels dans
                       le pool NBM. Fallback sur np.cov standard si sklearn absent.
      • robust=False : np.cov standard (plus rapide, moins robuste aux outliers).

    Régularisation de Tikhonov (ridge) : on ajoute λ×I à la matrice de covariance
    avant inversion, garantissant la non-singularité même si k ≈ n ou si des
    marqueurs sont colinéaires. λ = regularization (défaut 1e-4, inoffensif).

    Fallback hiérarchique si l'inversion échoue :
      1. scipy.linalg.pinv (pseudo-inverse de Moore-Penrose — gère les matrices
         singulières exactes, ex: marqueur constant dans le NBM).
      2. np.linalg.pinv (si scipy absent).
      3. Matrice diagonale 1/var(j) (distance euclidienne pondérée par la variance)
         si toutes les méthodes d'inversion échouent.

    ── Validation sur BLAST110 ───────────────────────────────────────────────────

    Sur BLAST110_100_P1 (arcsinh/5, NBM T1) :
      CD34 Cy55 : médiane_blast = 7.93, médiane_NBM = 4.72, std_NBM = 1.58
                  → z_blast = +2.07  (surexpression > 2σ) ✓
      CD45 KO   : médiane_blast = 6.82, médiane_NBM = 7.88, std_NBM = 1.22
                  → z_blast = −0.87  (CD45-dim modéré)
      SSC-A     : médiane_blast = 8.65, médiane_NBM = 10.63, std_NBM = 0.82
                  → z_blast = −2.42  (SSC-bas significatif) ✓

    Args:
        X_reference: Matrice des cellules/médianes de référence [n_ref, n_markers].
                     Typiquement les nœuds SOM issus des fichiers NBM (moelle normale).
                     Doit être dans le même espace transformé que X_unknown
                     (ex: arcsinh cofactor=5).
        robust: Si True (défaut), utilise la médiane + IQR/1.35 (pseudo-std robuste)
                pour center/scale, et MinCovDet pour la covariance si sklearn disponible.
                Si False, utilise la moyenne + std standard.
        regularization: Terme de régularisation λ ajouté à la diagonale de la matrice
                        de covariance avant inversion (défaut 1e-4). Augmenter à 1e-3
                        si le nombre de nœuds NBM est proche du nombre de marqueurs.
        max_samples_for_stats: Nombre maximum de lignes utilisées pour estimer
                centre/échelle robustes (médiane + IQR). Accélère les très
                grandes cohortes sans impacter significativement les médianes.
        max_samples_for_covariance: Nombre maximum de lignes utilisées pour
                l'estimation de covariance (Mahalanobis).
        max_samples_for_mincovdet: Seuil au-delà duquel MinCovDet est désactivé
                au profit de np.cov (beaucoup plus rapide).
        random_state: Seed du sous-échantillonnage aléatoire reproductible.

    Returns:
        Tuple (ref_center, ref_scale, inv_cov) :
          - ref_center : vecteur [n_markers] — médiane (ou moyenne) par marqueur.
          - ref_scale  : vecteur [n_markers] — IQR/1.35 (ou std), clampé à 0.01
                         pour éviter les divisions par zéro.
          - inv_cov    : matrice [n_markers, n_markers] — inverse de la covariance NBM,
                         ou None si X_reference contient moins de 2 lignes valides
                         (impossible de calculer une covariance sur un seul point).
    """
    X = np.asarray(X_reference, dtype=float)
    rng = np.random.default_rng(random_state)

    def _sample_rows(arr: np.ndarray, max_rows: int, purpose: str) -> np.ndarray:
        if max_rows > 0 and arr.shape[0] > max_rows:
            idx = rng.choice(arr.shape[0], size=max_rows, replace=False)
            _logger.info(
                "compute_reference_stats : sous-echantillonnage %s %d -> %d lignes.",
                purpose,
                arr.shape[0],
                max_rows,
            )
            return arr[idx]
        return arr

    X_stats = _sample_rows(X, int(max_samples_for_stats), "stats")

    # ── Centre et échelle (z-score 1D) ───────────────────────────────────────
    if robust:
        ref_center = np.nanmedian(X_stats, axis=0)
        q75 = np.nanpercentile(X_stats, 75, axis=0)
        q25 = np.nanpercentile(X_stats, 25, axis=0)
        ref_scale = (q75 - q25) / 1.35  # pseudo-std robuste (IQR/1.35 ≈ σ gaussien)
    else:
        ref_center = np.nanmean(X_stats, axis=0)
        ref_scale = np.nanstd(X_stats, axis=0)

    # Éviter div/0 : marqueurs constants ou absents
    ref_scale = np.where(np.isnan(ref_scale) | (ref_scale < 0.01), 0.01, ref_scale)
    ref_center = np.where(np.isnan(ref_center), 0.0, ref_center)

    # ── Matrice de covariance inverse (Mahalanobis) ───────────────────────────
    inv_cov: Optional[np.ndarray] = None

    # CR-5 FIX : traçabilité complète — on log la taille AVANT et APRÈS
    # le filtrage NaN pour identifier les datasets de mauvaise qualité.
    n_total_rows = X.shape[0]
    valid_mask = ~np.any(np.isnan(X), axis=1)
    n_nan_rows = int((~valid_mask).sum())
    X_valid = X[valid_mask]
    n_samples_total, n_features = X_valid.shape

    if n_nan_rows > 0:
        _logger.warning(
            "compute_reference_stats : %d/%d lignes contiennent des NaN "
            "— exclues du calcul de covariance.",
            n_nan_rows,
            n_total_rows,
        )

    if n_samples_total < 2:
        _logger.warning(
            "compute_reference_stats : seulement %d ligne(s) valide(s) dans X_reference "
            "— impossible de calculer la matrice de covariance. inv_cov = None.",
            n_samples_total,
        )
        return ref_center, ref_scale, None

    X_cov = _sample_rows(X_valid, int(max_samples_for_covariance), "covariance")
    n_samples_cov = X_cov.shape[0]

    try:
        # Stratégie 1 : MinCovDet (robuste aux outliers) si sklearn disponible
        if (
            robust
            and _HAS_SKLEARN
            and n_samples_cov >= max(5 * n_features, 20)
            and n_samples_cov <= max_samples_for_mincovdet
        ):
            # MinCovDet exige n_samples >> n_features (règle empirique : ≥5×)
            mcd = _MinCovDet(support_fraction=0.75, random_state=42)
            mcd.fit(X_cov)
            cov = mcd.covariance_
            _logger.debug(
                "compute_reference_stats : covariance robuste MinCovDet calculée "
                "(%d×%d, support_fraction=0.75).",
                n_features,
                n_features,
            )
        else:
            # Stratégie 2 : covariance empirique standard
            cov = np.cov(X_cov, rowvar=False)
            if robust and n_samples_cov > max_samples_for_mincovdet:
                _logger.info(
                    "compute_reference_stats : MinCovDet desactive pour grand volume "
                    "(%d lignes > seuil %d) — covariance empirique utilisee.",
                    n_samples_cov,
                    max_samples_for_mincovdet,
                )
            elif n_samples_cov < 5 * n_features and robust:
                _logger.debug(
                    "compute_reference_stats : MinCovDet ignoré (n_samples=%d < 5×n_feat=%d) "
                    "— covariance empirique standard utilisée.",
                    n_samples_cov,
                    5 * n_features,
                )

        # Régularisation de Tikhonov pour garantir la non-singularité
        cov_reg = cov + regularization * np.eye(n_features)

        # ── Vérification du conditionnement AVANT inversion ───────────────────
        # Un conditionnement > COND_THRESHOLD signifie que la matrice est
        # quasi-singulière : l'inversion amplifierait le bruit numérique et
        # produirait des distances de Mahalanobis non fiables cliniquement.
        _COND_THRESHOLD = 1e12
        cond_number = float(np.linalg.cond(cov_reg))
        if cond_number > _COND_THRESHOLD:
            raise ClinicalMathError(
                message=(
                    "La matrice de covariance NBM est mal conditionnée : "
                    "les marqueurs du panel sont probablement colinéaires ou "
                    "la population de référence contient trop peu de cellules. "
                    "Le scoring de Mahalanobis ne peut pas être effectué de manière fiable. "
                    "Vérifiez la qualité et la taille de l'échantillon NBM."
                ),
                condition_number=cond_number,
                details=(
                    f"n_samples={n_samples_cov}, n_features={n_features}, "
                    f"regularization={regularization}"
                ),
            )

        # Pseudo-inverse de Moore-Penrose — robuste aux matrices exactement singulières
        # (marqueur constant dans le NBM, rang déficient) sans fallback silencieux.
        if _HAS_SCIPY:
            inv_cov = _scipy_pinv(cov_reg)
        else:
            inv_cov = np.linalg.pinv(cov_reg)

        _logger.debug(
            "compute_reference_stats : inv_cov calculée — shape %s, cond=%.2e",
            inv_cov.shape,
            cond_number,
        )

    # ClinicalMathError se propage intentionnellement vers l'appelant (UI / pipeline).
    # Les autres exceptions restent des erreurs imprévues à logger.
    except ClinicalMathError:
        raise

    except Exception as exc:
        _logger.error(
            "compute_reference_stats : erreur inattendue lors du calcul de inv_cov : %s",
            exc,
        )
        raise ClinicalMathError(
            message=(
                "Erreur inattendue lors du calcul de la matrice de covariance NBM. "
                "Le scoring de Mahalanobis est impossible."
            ),
            details=str(exc),
        ) from exc

    return ref_center, ref_scale, inv_cov


def compute_reference_normalization(
    X_unknown: np.ndarray,
    X_reference: np.ndarray,
    robust: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Normalise X_unknown en z-scores par rapport à la population de référence (NBM).

    ── Changement vs version min-max ────────────────────────────────────────────

    L'ancienne normalisation min-max plaçait tout dans [0, 1], rendant
    score_nodes_for_blasts() aveugle (score = 0 systématique).

    Cette version produit des z-scores :
        z[j] = (X_unknown[j] − median_ref[j]) / scale_ref[j]

    Les blastes se retrouvent EN DEHORS de [−1, +1] sur les marqueurs
    discriminants (CD34 z ≈ +2, SSC z ≈ −2), déclenchant les contributions
    directionnelles de score_nodes_for_blasts().

    Args:
        X_unknown: Matrice des nœuds à scorer [n_unknown, n_markers].
        X_reference: Matrice de référence (cellules NBM ou médianes NBM)
                     [n_ref, n_markers], dans le même espace transformé.
        robust: Utiliser médiane+IQR/1.35 (True, défaut) ou moyenne+std (False).

    Returns:
        Tuple (X_zscore, ref_center, ref_scale, inv_cov) :
          - X_zscore   : z-scores [n_unknown, n_markers].
          - ref_center : vecteur des centres de référence par marqueur.
          - ref_scale  : vecteur des échelles de référence par marqueur.
          - inv_cov    : matrice inverse de covariance [n_markers, n_markers]
                         (None si X_reference trop petite pour calculer la covariance).
    """
    ref_center, ref_scale, inv_cov = compute_reference_stats(X_reference, robust=robust)
    X_zscore = (X_unknown - ref_center) / ref_scale
    return X_zscore, ref_center, ref_scale, inv_cov


# ─────────────────────────────────────────────────────────────────────────────
#  §10.4c — Distance de Mahalanobis + Score Hybride (Stratégie A)
#
#  Résout le paradoxe des LAM matures (P25/P57 : CD34−/CD117−, CD45 brillant)
#  sans réintroduire les faux positifs monocytaires (P105 + effet batch).
#
#  Principe :
#    La somme pondérée de Z-scores (score_nodes_for_blasts) projette l'espace
#    10D en un scalaire 1D — l'information de corrélation entre marqueurs est
#    détruite. Un blaste mature et un monocyte+batch partagent le même vecteur
#    de Z-scores individuels (CD34≈0, CD117≈0, CD45+brillant) mais occupent des
#    régions DIFFÉRENTES de l'espace 10D.
#
#    D²(x) = (x - μ_NBM)ᵀ × Σ_NBM⁻¹ × (x - μ_NBM) capture cette différence :
#      - Blaste mature : combinaison CD45+/HLA-DR+/CD33+/CD13+/FSC-SSC atypique
#        → très loin de l'hyper-ellipse NBM en 10D → D² élevé.
#      - Monocyte+batch : reste dans la "forme" du nuage NBM, juste décalé
#        → D² faible, rejeté.
#
#  Modulation par la pureté topologique (Porte 1) :
#    Le seuil de D² requis est abaissé quand le nœud est très pur (99.9%
#    cellules patient) — la pureté topologique EST une information biologique.
#    Inversement, un nœud mixte (60% patient) peut s'expliquer par l'effet
#    batch → on exige une distance biologique plus forte.
# ─────────────────────────────────────────────────────────────────────────────


def score_nodes_mahalanobis(
    node_medians_raw: np.ndarray,
    nbm_center: np.ndarray,
    nbm_inv_cov: np.ndarray,
    patho_purity: Optional[np.ndarray] = None,
    purity_modulation_power: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule la distance de Mahalanobis de chaque nœud SOM au nuage NBM,
    modulée optionnellement par la pureté topologique (Porte 1).

    ── Formule ──────────────────────────────────────────────────────────────────

        D²(i) = (x_i − μ_NBM)ᵀ × Σ_NBM⁻¹ × (x_i − μ_NBM)

    Contrairement à la distance euclidienne, Σ_NBM⁻¹ tient compte des
    corrélations entre marqueurs. Un nœud peut être "loin" en D² même si
    ses Z-scores individuels sont proches de 0, parce que la COMBINAISON
    de ses valeurs n'existe pas dans la moelle normale saine.

    ── Modulation par la pureté topologique ────────────────────────────────────

    Intuition clinique : si un nœud contient 99.9% de cellules patient (pureté
    élevée), c'est une forte preuve topologique de MRD. On peut se permettre
    d'accepter une distance biologique un peu moins extrême. Inversement, un
    nœud avec 60% de cellules patient peut simplement refléter un effet batch
    — la distance biologique doit alors être irréprochable.

    La modulation amplifie D² pour les nœuds purs et l'atténue pour les nœuds
    mixtes :

        D²_modulé = D² × purity^power

        power=0.5 (racine) : nœud 100% pur → ×1.0 ; 60% pur → ×0.77 ; 20% → ×0.45
        power=1.0 (linéaire): nœud 100% pur → ×1.0 ; 60% pur → ×0.60 ; 20% → ×0.20
        power=0.0           : pas de modulation (tous les nœuds traités également)

    ── Distribution de référence (calibration des seuils) ──────────────────────

    Sous H₀ (nœud issu du NBM), D² suit approximativement une loi χ²(k) où
    k = nombre de marqueurs. Valeurs de référence pour calibration :
        k=10 : χ²(10, p=0.95) ≈ 18.3  → d2_high_threshold ≈ 20
               χ²(10, p=0.80) ≈ 13.4  → d2_moderate_threshold ≈ 12

    ⚠  Ces seuils supposent la normalité multivariée — à calibrer sur cohorte
       locale si la distribution des nœuds NBM est non-gaussienne.

    Args:
        node_medians_raw: Médianes brutes [n_nodes, n_markers] dans l'espace
                          transformé (arcsinh/5). NE PAS utiliser les z-scores
                          ici — la distance de Mahalanobis intègre sa propre
                          normalisation via Σ_NBM⁻¹.
        nbm_center: Vecteur [n_markers] — centre du nuage NBM (de
                    compute_reference_stats, ref_center).
        nbm_inv_cov: Matrice [n_markers, n_markers] — inverse de la covariance
                     NBM (de compute_reference_stats, inv_cov).
        patho_purity: Tableau [n_nodes] de proportions [0.0, 1.0] — fraction de
                      cellules patient dans chaque nœud. None = pas de modulation.
        purity_modulation_power: Exposant de la modulation par pureté (0.5 par
                                  défaut). 0.0 désactive la modulation.

    Returns:
        Tuple (d2_raw, d2_modulated) :
          - d2_raw      : np.ndarray [n_nodes] — distances D² brutes (non modulées).
          - d2_modulated: np.ndarray [n_nodes] — distances D² après modulation
                          par la pureté (= d2_raw si patho_purity is None).
    """
    X = np.asarray(node_medians_raw, dtype=float)
    mu = np.asarray(nbm_center, dtype=float)
    VI = np.asarray(nbm_inv_cov, dtype=float)

    # Calcul vectorisé : D²(i) = diff_i @ VI @ diff_i
    # diff : [n_nodes, n_markers]
    diff = X - mu  # broadcast sur chaque nœud

    # Forme quadratique vectorisée : (diff @ VI) elementwise diff, sommé sur markers
    d2_raw = np.einsum("ij,jk,ik->i", diff, VI, diff)
    d2_raw = np.maximum(0.0, d2_raw)  # garantir la positivité (erreurs numériques)

    # Modulation par la pureté topologique
    if patho_purity is not None and purity_modulation_power > 0.0:
        purity_arr = np.clip(np.asarray(patho_purity, dtype=float), 0.0, 1.0)
        modulation = np.power(purity_arr, purity_modulation_power)
        d2_modulated = d2_raw * modulation
    else:
        d2_modulated = d2_raw.copy()

    return d2_raw, d2_modulated


def score_nodes_hybrid(
    node_medians_raw: np.ndarray,
    node_medians_zscore: np.ndarray,
    nbm_center: np.ndarray,
    nbm_inv_cov: np.ndarray,
    marker_names: List[str],
    patho_purity: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None,
    mahal_weight: float = 0.65,
    linear_weight: float = 0.35,
    d2_normalization: float = 20.0,
    purity_modulation_power: float = 0.5,
    mahal_boost_factor: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score blast hybride combinant la distance de Mahalanobis (géométrie 10D)
    et le score linéaire ELN 2022 / Ogata (connaissance clinique directionnelle).

    ── Philosophie ──────────────────────────────────────────────────────────────

    Les deux composantes sont complémentaires :

      Score de Mahalanobis (65% par défaut) :
        Répond à "ce nœud est-il LOIN du nuage NBM dans l'espace 10D ?"
        Capture les blastes matures (P25/P57) dont la COMBINAISON de marqueurs
        est anormale même si les Z-scores individuels semblent normaux.

      Score linéaire ELN 2022 / Ogata (35% par défaut) :
        Répond à "ce nœud ressemble-t-il à la signature clinique d'un blaste ?"
        Encode CD34+, CD117+, CD45-dim, SSC-bas — filtre les artefacts
        purement techniques qui s'éloigneraient du NBM dans des directions
        non-blastiques (ex: débris, agrégats).

      Score hybride = α × Score_Mahal_normalisé + (1−α) × Score_linéaire

    ── Calibration de mahal_weight ─────────────────────────────────────────────

      mahal_weight → 1.0 : cohorte avec beaucoup de LAM matures CD34−/CD117−
      mahal_weight → 0.5 : cohorte mixte, conserver l'apport ELN 2022
      mahal_weight → 0.0 : fallback pur sur le score linéaire (comportement historique)

    ── Normalisation de D² vers [0, 10] ────────────────────────────────────────

    D² est normalisé par d2_normalization (≈ χ²(k, p=0.95) pour k marqueurs) :
        Score_Mahal = clip(D²_modulé / d2_normalization × 10, 0, 10)

    Un nœud NBM typique a D² < 18.3 (χ²(10, 0.95)) → Score_Mahal < 10.
    Un blaste très atypique peut avoir D² >> 20 → Score_Mahal = 10 (clampé).

    Args:
        node_medians_raw: Médianes brutes [n_nodes, n_markers] (arcsinh/5).
                          Utilisées pour la composante Mahalanobis.
        node_medians_zscore: Médianes z-scorées [n_nodes, n_markers].
                              Utilisées pour la composante linéaire ELN 2022.
        nbm_center: Vecteur [n_markers] — centre NBM (de compute_reference_stats).
        nbm_inv_cov: Matrice [n_markers, n_markers] — inverse covariance NBM.
        marker_names: Noms des marqueurs (colonnes de node_medians_zscore).
        patho_purity: Pureté topologique [n_nodes] ∈ [0, 1] (Porte 1 output).
        weights: Poids du score linéaire (None = auto via build_blast_weights).
        mahal_weight: Poids de la composante Mahalanobis [0, 1].
        linear_weight: Poids de la composante linéaire [0, 1].
                       La somme mahal_weight + linear_weight n'a pas besoin d'être
                       exactement 1 — chaque composante est déjà dans [0, 10].
        d2_normalization: Valeur D² correspondant à un score de 10/10.
                          Défaut 20.0 ≈ χ²(10, p=0.95) + marge de sécurité.
        purity_modulation_power: Exposant de modulation pureté (0.5 = racine carrée).
        mahal_boost_factor: Facteur de boost appliqué au score linéaire quand D² brut
                             est très élevé (> d2_threshold_high). Amplifie la détection
                             des blastes atypiques géométriquement très anormaux (ex: P25).
                             Défaut : 1.5.

    Returns:
        Tuple (scores_hybrid, d2_raw) :
          - scores_hybrid : np.ndarray [n_nodes] dans [0.0, 10.0].
          - d2_raw        : np.ndarray [n_nodes] distances D² brutes (pour debug).
    """
    # Composante 1 : Mahalanobis modulé par la pureté
    d2_raw, d2_modulated = score_nodes_mahalanobis(
        node_medians_raw,
        nbm_center,
        nbm_inv_cov,
        patho_purity=patho_purity,
        purity_modulation_power=purity_modulation_power,
    )
    scores_mahal = np.clip(d2_modulated / max(d2_normalization, 1e-6) * 10.0, 0.0, 10.0)

    # Composante 2 : score linéaire ELN 2022 / Ogata
    if weights is None:
        weights = build_blast_weights(marker_names)
    scores_linear = score_nodes_for_blasts(
        np.asarray(node_medians_zscore, dtype=float),
        marker_names,
        weights,
    )

    # ── Boost géométrique ─────────────────────────────────────────────────────
    #
    # Bonus de rupture de corrélation : si D² est très élevé (> d2_normalization),
    # la géométrie est franchement anormale → on booste la composante linéaire.
    # Cela profite aux blastes atypiques (P25) dont le score CD33/HLA-DR est modeste
    # mais dont la position dans l'espace 10D est clairement hors du nuage NBM.
    boosted_linear = np.where(
        d2_raw > d2_normalization,
        scores_linear * mahal_boost_factor,
        scores_linear,
    )
    boosted_linear = np.clip(boosted_linear, 0.0, 10.0)

    # Combinaison pondérée (chaque composante est déjà dans [0, 10])
    scores_hybrid = mahal_weight * scores_mahal + linear_weight * boosted_linear

    return np.clip(scores_hybrid, 0.0, 10.0), d2_raw


# ─────────────────────────────────────────────────────────────────────────────
#  §10.4d — Traçabilité FCS source des cellules blast
# ─────────────────────────────────────────────────────────────────────────────


def trace_blast_cells_to_fcs_source(
    blast_candidates_df: pd.DataFrame,
    cell_data: "Any",  # anndata.AnnData
    source_priority_cols: Optional[List[str]] = None,
    condition_col: Optional[str] = "Condition",
    blast_categories_to_trace: Optional[List[str]] = None,
    alert_patho_threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Retrace les cellules des nœuds candidats blast vers leurs fichiers FCS sources.

    Utilisation clinique ELN 2022 : si >50% des cellules d'un nœud BLAST_HIGH
    proviennent d'un fichier "Patho" (diagnostic ou suivi), déclencher une
    ALERTE CLINIQUE.

    Cela permet de distinguer :
      - Un nœud BLAST_HIGH peuplé de cellules Saines → faux positif batch
      - Un nœud BLAST_HIGH peuplé de cellules Pathologiques → vrai signal MRD

    Args:
        blast_candidates_df: DataFrame de build_blast_score_dataframe (colonnes:
                              node_id, blast_score, blast_category, ...).
        cell_data: AnnData avec .obs['clustering'] (nœud SOM, 0-indexé) et
                   .obs[condition_col] (condition de la cellule).
        source_priority_cols: Colonnes .obs à inspecter pour retrouver le fichier
                               source (ex: ['File_Origin', 'filename']).
        condition_col: Colonne de condition dans cell_data.obs.
        blast_categories_to_trace: Catégories à inclure dans la traçabilité.
                                    Défaut: ['BLAST_HIGH', 'BLAST_MODERATE'].
        alert_patho_threshold: Fraction de cellules Pathologiques déclenchant
                                l'ALERTE CLINIQUE (défaut 0.50 = 50%).

    Returns:
        DataFrame avec colonnes :
          node_id, blast_score, blast_category,
          n_cells_total, n_cells_patho, n_cells_sain, pct_patho,
          source_files (str), clinical_alert (bool).
    """
    if blast_categories_to_trace is None:
        blast_categories_to_trace = ["BLAST_HIGH", "BLAST_MODERATE"]

    mask_trace = blast_candidates_df["blast_category"].isin(blast_categories_to_trace)
    df_trace = blast_candidates_df[mask_trace].copy()

    if df_trace.empty:
        _logger.info("Aucun nœud blast à tracer pour ces catégories.")
        return df_trace

    # ── Résoudre la colonne de clustering dans cell_data ─────────────────────
    obs_df = None
    try:
        obs_df = cell_data.obs.copy()
    except AttributeError:
        if isinstance(cell_data, pd.DataFrame):
            obs_df = cell_data.copy()

    if obs_df is None:
        _logger.warning("cell_data.obs inaccessible — traçabilité impossible.")
        df_trace["n_cells_total"] = 0
        df_trace["clinical_alert"] = False
        return df_trace

    clustering_col = None
    for candidate in ["clustering", "FlowSOM_cluster", "cluster", "node_id"]:
        if candidate in obs_df.columns:
            clustering_col = candidate
            break

    if clustering_col is None:
        _logger.warning("Colonne clustering introuvable dans cell_data.obs.")
        df_trace["n_cells_total"] = 0
        df_trace["clinical_alert"] = False
        return df_trace

    # ── Résoudre la colonne source de fichier ─────────────────────────────────
    source_col = None
    if source_priority_cols:
        for col in source_priority_cols:
            if col in obs_df.columns:
                source_col = col
                break
    if source_col is None:
        for candidate in [
            "File_Origin",
            "filename",
            "Filename",
            "sample_id",
            "SampleID",
        ]:
            if candidate in obs_df.columns:
                source_col = candidate
                break

    # ── PERF-4 FIX : Traçabilité vectorisée via groupby ──────────────────────
    # Remplace la boucle iterrows() + masque booléen O(N) par nœud par un
    # groupby unique O(N) sur tout le DataFrame — critique sur 500k+ cellules.

    # Normaliser l'ID de nœud une seule fois
    obs_df = obs_df.copy()
    obs_df["_node_int"] = obs_df[clustering_col].astype(int)

    # Pré-calculer les masques condition sur tout le DataFrame (vectorisé)
    _has_cond_col = bool(condition_col and condition_col in obs_df.columns)
    if _has_cond_col:
        cond_upper = obs_df[condition_col].astype(str).str.upper()
        obs_df["_is_patho"] = cond_upper.str.contains("PATHO|DIAG|DX|LAM|AML", na=False)
        obs_df["_is_sain"]  = cond_upper.str.contains("SAIN|NORMAL|NBM|HEALTHY", na=False)
    else:
        obs_df["_is_patho"] = False
        obs_df["_is_sain"]  = False

    # Groupby unique sur tous les nœuds
    grouped = obs_df.groupby("_node_int", sort=False)

    records: List[Dict] = []

    for _, row in df_trace.iterrows():
        node_id = int(row["node_id"])
        blast_score = float(row["blast_score"])
        blast_category = str(row["blast_category"])

        if node_id not in grouped.groups:
            records.append({
                "node_id": node_id,
                "blast_score": blast_score,
                "blast_category": blast_category,
                "n_cells_total": 0,
                "n_cells_patho": 0,
                "n_cells_sain": 0,
                "pct_patho": 0.0,
                "source_files": "",
                "clinical_alert": False,
            })
            continue

        cells_in_node = grouped.get_group(node_id)
        n_total = len(cells_in_node)
        n_patho = int(cells_in_node["_is_patho"].sum())
        n_sain  = int(cells_in_node["_is_sain"].sum())

        pct_patho = n_patho / max(n_total, 1)
        clinical_alert = (blast_category == "BLAST_HIGH") and (
            pct_patho >= alert_patho_threshold
        )

        if clinical_alert:
            _logger.warning(
                "ALERTE CLINIQUE — Nœud %d (%s) : %.1f%% cellules Pathologiques (%d/%d)",
                node_id,
                blast_category,
                100.0 * pct_patho,
                n_patho,
                n_total,
            )

        source_files_str = ""
        if source_col and source_col in cells_in_node.columns:
            src_counts = cells_in_node[source_col].value_counts()
            source_files_str = " | ".join(
                f"{fname}({cnt})" for fname, cnt in src_counts.items()
            )

        records.append({
            "node_id": node_id,
            "blast_score": blast_score,
            "blast_category": blast_category,
            "n_cells_total": n_total,
            "n_cells_patho": n_patho,
            "n_cells_sain": n_sain,
            "pct_patho": round(pct_patho * 100.0, 1),
            "source_files": source_files_str,
            "clinical_alert": clinical_alert,
        })

    result_df = (
        pd.DataFrame(records)
        .sort_values(["blast_category", "blast_score"], ascending=[True, False])
        .reset_index(drop=True)
    )

    n_alerts = int(result_df["clinical_alert"].sum())
    _logger.info(
        "Traçabilité terminée: %d nœuds tracés, %d ALERTE(S) CLINIQUE(S)",
        len(result_df),
        n_alerts,
    )
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
#  Utilitaire — vecteur de poids depuis colonnes + dict
# ─────────────────────────────────────────────────────────────────────────────


def build_weight_vector(
    cols: List[str],
    weights: Dict[str, float],
    default: float = 1.0,
) -> np.ndarray:
    """
    Construit un vecteur numpy de poids aligné sur une liste de colonnes.

    Utilisé pour les distances pondérées dans le mapping de populations :
    chaque marqueur reçoit un poids depuis le dict ``weights``, ou
    ``default`` s'il n'est pas présent.

    Args:
        cols: Liste ordonnée de noms de marqueurs.
        weights: Dict {marqueur: poids}.
        default: Poids par défaut si le marqueur est absent du dict (défaut 1.0).

    Returns:
        np.ndarray de shape (len(cols),) et dtype float64.

    Example::

        w = build_weight_vector(["CD34", "CD45", "SSC-A"],
                                {"CD34": 3.0, "CD45": -2.0},
                                default=0.5)
        # array([ 3. , -2. ,  0.5])
    """
    return np.array([weights.get(c, default) for c in cols], dtype=np.float64)


# Alias privé — compatibilité avec flowsom_pipeline.py (même logique que categorize_blast_score)
def _categorize_blast(score: float) -> str:
    """Alias interne de categorize_blast_score (mêmes seuils ELN 2022 / Ogata)."""
    return categorize_blast_score(score)
