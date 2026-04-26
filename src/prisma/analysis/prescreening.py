"""
prescreening.py — Pré-screening heuristique CD34+/CD45dim.

Calcule systématiquement le rapport CD34+/CD45dim sur les données brutes
(indépendamment des options de gating) pour :
  - Détecter une forte MRD ou régénération médullaire
  - Alerter l'utilisateur si le ratio est élevé (popup + rapport HTML)
  - Enregistrer les comptages CD34+ et CD34- pour traçabilité

Le gating CD34+/CD45dim est effectué par deux méthodes comparables :
  - GMM 2 composantes (scikit-learn GaussianMixture)
  - KDE 1D (scipy gaussian_kde) — méthode de RÉFÉRENCE

L'utilisateur peut choisir la méthode de référence via le paramètre
``cd34_cd45dim_density_method`` (défaut : "KDE").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_logger = logging.getLogger("analysis.prescreening")

# ── Seuils heuristiques ───────────────────────────────────────────────────────
# Rapport CD34+ / CD45dim > 5 % → forte MRD ou régénération
CD34_RATIO_HIGH_THRESHOLD = 0.05  # 5 %
# Rapport CD34+ / CD45dim > 15 % → très fort signal (régénération extrême ou MRD massive)
CD34_RATIO_VERY_HIGH_THRESHOLD = 0.15  # 15 %
# Nombre minimal de cellules CD45dim pour un calcul fiable
MIN_CD45DIM_CELLS = 100


@dataclass
class PrescreeningResult:
    """
    Résultat du pré-screening heuristique CD34+/CD45dim.

    Attributes:
        n_cd34_pos: Nombre de cellules CD34+ (méthode de référence)
        n_cd34_neg: Nombre de cellules CD34- (méthode de référence)
        n_cd45dim: Nombre de cellules CD45dim (dénominateur)
        ratio_cd34_cd45dim: Rapport CD34+ / CD45dim (0.0–1.0)
        ratio_pct: Rapport en pourcentage
        alert_level: "none" | "moderate" | "high"
        alert_message: Message d'alerte pour l'utilisateur
        method_used: Méthode ayant produit le résultat de référence ("GMM" | "KDE")
        gmm_n_cd34_pos: Comptage GMM (pour comparaison)
        gmm_ratio_pct: Rapport GMM en % (pour comparaison)
        kde_n_cd34_pos: Comptage KDE (pour comparaison)
        kde_ratio_pct: Rapport KDE en % (pour comparaison)
        cd34_threshold: Seuil CD34 appliqué (espace linéaire)
        cd45dim_threshold_low: Seuil bas CD45dim (espace linéaire)
        cd45dim_threshold_high: Seuil haut CD45dim (espace linéaire)
        warnings: Avertissements non bloquants
        laip_tracking_recommended: True si LAIP Tracking classique recommandé
        interpretation_warning: Texte affiché dans le rapport HTML
    """

    n_cd34_pos: int = 0
    n_cd34_neg: int = 0
    n_cd45dim: int = 0
    ratio_cd34_cd45dim: float = 0.0
    ratio_pct: float = 0.0
    alert_level: str = "none"
    alert_message: str = ""
    method_used: str = "KDE"
    gmm_n_cd34_pos: int = 0
    gmm_ratio_pct: float = 0.0
    kde_n_cd34_pos: int = 0
    kde_ratio_pct: float = 0.0
    cd34_threshold: float = 0.0
    cd45dim_threshold_low: float = 0.0
    cd45dim_threshold_high: float = 0.0
    warnings: List[str] = field(default_factory=list)
    laip_tracking_recommended: bool = False
    interpretation_warning: str = ""
    # Données KDE pour le graphique (non sérialisées en JSON)
    kde_x_grid: Optional[object] = None   # np.ndarray, espace log
    kde_density: Optional[object] = None  # np.ndarray, densité lissée
    kde_threshold_log: float = 0.0        # Seuil dans l'espace log (pour le plot)

    def to_dict(self) -> Dict:
        import dataclasses
        d = dataclasses.asdict(self)
        # Exclure les arrays numpy non-sérialisables
        d.pop("kde_x_grid", None)
        d.pop("kde_density", None)
        return d


def _find_marker_idx(var_names: List[str], candidates: List[str]) -> Optional[int]:
    """Trouve l'index d'un marqueur par sous-chaîne (insensible à la casse).

    Identique à PreGating.find_marker_index — recherche par sous-chaîne pour
    gérer les suffixes de fluorochrome (ex: "CD34-A", "CD34-PE", "CD34-FITC"…).
    """
    upper_vars = [v.upper() for v in var_names]
    for candidate in candidates:
        c_upper = candidate.upper()
        for i, name in enumerate(upper_vars):
            if c_upper in name:
                return i
    return None


def _gate_cd45dim(
    cd45: np.ndarray,
    valid: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """
    Identifie les cellules CD45dim par seuillage percentile sur les données brutes.

    CD45dim = cellules entre le 5e et 40e percentile de CD45 (zone intermédiaire
    entre CD45neg des érythroblastes et CD45bright des lymphocytes matures).
    Ces percentiles sont calibrés sur les données de moelle osseuse AML.

    Returns:
        (mask_cd45dim, seuil_bas, seuil_haut)
    """
    cd45_valid = cd45[valid]
    # Seuils percentile calibrés sur les données de moelle AML
    low = float(np.nanpercentile(cd45_valid, 5))
    high = float(np.nanpercentile(cd45_valid, 40))
    mask_dim = np.zeros(len(cd45), dtype=bool)
    mask_dim[valid] = (cd45_valid >= low) & (cd45_valid <= high)
    return mask_dim, low, high


def _gate_cd34_gmm(
    cd34: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, float, List[str]]:
    """
    Sépare CD34+ et CD34- par GMM 2 composantes (1D).

    Returns:
        (mask_cd34_pos, seuil_cd34, warnings)
    """
    warnings_out: List[str] = []
    n_cells = len(cd34)
    cd34_valid = cd34[valid_mask]

    if len(cd34_valid) < 100:
        warnings_out.append("GMM CD34: pas assez de cellules (<100)")
        return np.zeros(n_cells, dtype=bool), 0.0, warnings_out

    try:
        from sklearn.mixture import GaussianMixture

        # Sous-échantillonnage si trop de cellules
        max_samples = 50_000
        if len(cd34_valid) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(cd34_valid), max_samples, replace=False)
            cd34_fit = cd34_valid[idx].reshape(-1, 1)
        else:
            cd34_fit = cd34_valid.reshape(-1, 1)

        gmm = GaussianMixture(n_components=2, covariance_type="full", n_init=3, random_state=42)
        gmm.fit(cd34_fit)

        labels = gmm.predict(cd34_valid.reshape(-1, 1))
        means = gmm.means_.flatten()
        pos_component = int(np.argmax(means))

        # Seuil = point médian entre les deux moyennes GMM
        neg_component = 1 - pos_component
        threshold = float((means[pos_component] + means[neg_component]) / 2)

        mask_pos = np.zeros(n_cells, dtype=bool)
        mask_pos[valid_mask] = labels == pos_component

        _logger.info(
            "   [Prescreening-GMM] CD34: μ_neg=%.0f, μ_pos=%.0f → seuil=%.0f → %d CD34+",
            means[neg_component], means[pos_component], threshold, mask_pos.sum(),
        )
        return mask_pos, threshold, warnings_out

    except Exception as exc:
        warnings_out.append(f"GMM CD34 échoué: {exc}")
        _logger.warning("   [Prescreening-GMM] Échec: %s", exc)
        return np.zeros(n_cells, dtype=bool), 0.0, warnings_out


def gate_cd34_kde(
    cd34: np.ndarray,
    valid_mask: np.ndarray,
    n_grid: int = 512,
    sigma_smooth: float = 5.0,
    max_samples: int = 15_000,
    seuil_relatif_pied: float = 0.12,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, List[str]]:
    """
    Sépare CD34+ et CD34- par KDE 1D — pied droit du pic principal (CD34−).

    Stratégie : analogue à ``_kde1d_seuil_pied_pic`` pour CD45 mais en sens
    inverse.  CD45 cherche le pied GAUCHE de son pic principal ; CD34 cherche
    le pied DROIT du grand pic CD34− (là où la densité rechute avant de remonter
    pour le pic CD34+).

    Algorithme :
      1. Trouver le pic de densité le plus haut → pic principal CD34−.
      2. Scanner à droite depuis ce pic jusqu'à ce que la densité passe sous
         ``seuil_relatif_pied × densité_max`` (pied du pic).
      3. Ce point est le seuil séparant CD34− de CD34+.
      Fallback 1 — un seul pic ou seuil non atteint : minimum de densité dans la
        fenêtre [pic_principal ; 90e percentile des données].
      Fallback 2 — aucun pic : minimum dans la zone 30–80e pct de x_grid.

    Travaille directement en espace logicle/arcsinh (valeurs négatives OK —
    pas de re-transformation log).

    Args:
        cd34: Vecteur CD34 en espace logicle (déjà transformé)
        valid_mask: Masque booléen des cellules valides (isfinite)
        n_grid: Résolution de la grille KDE
        sigma_smooth: Sigma du lissage gaussien (en bins)
        max_samples: Sous-échantillonnage max pour la KDE
        seuil_relatif_pied: Fraction du max de densité définissant le «pied» (défaut 12 %)

    Returns:
        (mask_cd34_pos, threshold, x_grid, density_smooth, warnings)
        où x_grid et density_smooth sont en espace logicle (pour le plot)
    """
    warnings_out: List[str] = []
    n_cells = len(cd34)
    cd34_valid = cd34[valid_mask]

    if len(cd34_valid) < 100:
        warnings_out.append("KDE CD34: pas assez de cellules (<100)")
        return np.zeros(n_cells, dtype=bool), 0.0, np.array([]), np.array([]), warnings_out

    try:
        from scipy.stats import gaussian_kde
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks

        # Sous-échantillonnage
        if len(cd34_valid) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(cd34_valid), max_samples, replace=False)
            kde_data = cd34_valid[idx]
        else:
            kde_data = cd34_valid.copy()

        # KDE Silverman + lissage gaussien
        kde_obj = gaussian_kde(kde_data, bw_method="silverman")
        x_grid = np.linspace(float(cd34_valid.min()), float(cd34_valid.max()), n_grid)
        density = gaussian_filter1d(kde_obj(x_grid), sigma=sigma_smooth)

        density_max = float(density.max())
        seuil_abs = density_max * seuil_relatif_pied

        # ── Détection du pic principal ────────────────────────────────────────
        peak_prominence_min = density_max * 0.08   # 8 % du max
        peak_distance_min = n_grid // 8            # séparation minimale

        peaks, _ = find_peaks(
            density,
            prominence=peak_prominence_min,
            distance=peak_distance_min,
        )

        threshold = 0.0
        method_tag = "fallback"

        if len(peaks) >= 1:
            # Pic principal = pic le plus haut en densité (pic CD34−, toujours dominant)
            idx_main = int(peaks[np.argmax(density[peaks])])

            # ── Scan droite depuis le pic principal ───────────────────────────
            # Cherche le premier index où density < seuil_abs = pied du pic droit
            foot_idx = None
            for i in range(idx_main, len(density)):
                if density[i] < seuil_abs:
                    foot_idx = i
                    break

            if foot_idx is not None:
                # Pied du pic trouvé → c'est le seuil CD34
                threshold = float(x_grid[foot_idx])
                method_tag = f"pied_droit_pic_principal (seuil_rel={seuil_relatif_pied:.0%})"
                _logger.info(
                    "   [KDE-CD34] Pic principal: %.3f → pied droit à %.3f",
                    x_grid[idx_main], threshold,
                )
            else:
                # Le pied n'a jamais été atteint (pic très étalé ou données mono-pic)
                # Fallback : minimum de densité dans la zone [pic_principal ; p90]
                p90_val = float(np.percentile(cd34_valid, 90))
                zone_mask = (x_grid >= x_grid[idx_main]) & (x_grid <= p90_val)
                if zone_mask.any():
                    local_min_local = int(np.argmin(density[zone_mask]))
                    local_min_idx = int(np.where(zone_mask)[0][local_min_local])
                    threshold = float(x_grid[local_min_idx])
                    method_tag = "min_droit_pic_principal (pied non atteint)"
                else:
                    threshold = float(np.nanpercentile(cd34_valid, 85))
                    method_tag = "fallback_p85"
                warnings_out.append("KDE CD34: pied droit non atteint — fallback minimum local")
                _logger.warning("   [KDE-CD34] Pied non atteint → fallback: %.3f", threshold)
        else:
            # Aucun pic détecté — minimum dans zone 30–80e pct de x_grid
            p30 = float(np.percentile(x_grid, 30))
            p80 = float(np.percentile(x_grid, 80))
            zone = (x_grid >= p30) & (x_grid <= p80)
            if zone.any():
                local_min_idx = int(np.where(zone)[0][np.argmin(density[zone])])
                threshold = float(x_grid[local_min_idx])
            else:
                threshold = float(np.nanpercentile(cd34_valid, 85))
            method_tag = "fallback_zone_30_80pct (aucun pic)"
            warnings_out.append("KDE CD34: aucun pic détecté — fallback zone centrale")
            _logger.warning("   [KDE-CD34] Aucun pic → fallback: %.3f", threshold)

        mask_pos = np.zeros(n_cells, dtype=bool)
        mask_pos[valid_mask] = cd34_valid >= threshold

        _logger.info(
            "   [Prescreening-KDE] CD34 logicle: seuil=%.3f (%s) → %d CD34+",
            threshold, method_tag, int(mask_pos.sum()),
        )
        return mask_pos, threshold, x_grid, density, warnings_out

    except Exception as exc:
        warnings_out.append(f"KDE CD34 échoué: {exc} — fallback percentile 85")
        _logger.warning("   [Prescreening-KDE] Échec: %s — fallback percentile", exc)
        threshold = float(np.nanpercentile(cd34_valid, 85))
        mask_pos = np.zeros(n_cells, dtype=bool)
        mask_pos[valid_mask] = cd34_valid >= threshold
        return mask_pos, threshold, np.array([]), np.array([]), warnings_out


def compute_cd34_prescreening(
    X: np.ndarray,
    var_names: List[str],
    density_method: str = "KDE",
    cd34_ratio_high_threshold: float = CD34_RATIO_HIGH_THRESHOLD,
    cd34_ratio_very_high_threshold: float = CD34_RATIO_VERY_HIGH_THRESHOLD,
    min_cd45dim_cells: int = MIN_CD45DIM_CELLS,
) -> Optional[PrescreeningResult]:
    """
    Pré-screening heuristique CD34+/CD45dim — toujours exécuté.

    Calcule le rapport CD34+/CD45dim sur les données brutes (après gating
    débris/singlets/CD45, avant toute transformation ou clustering).
    Indépendant des paramètres de gating CD34 de l'utilisateur.

    Args:
        X: Matrice de données (n_cells × n_markers), déjà après gating viable
        var_names: Noms des marqueurs (colonnes de X)
        density_method: Méthode de référence ("KDE" | "GMM")
        cd34_ratio_high_threshold: Seuil de ratio élevé (défaut 5 %)
        cd34_ratio_very_high_threshold: Seuil très élevé (défaut 15 %)
        min_cd45dim_cells: Nombre minimal de cellules CD45dim pour un calcul fiable

    Returns:
        PrescreeningResult ou None si les marqueurs sont absents
    """
    _logger.info("=" * 50)
    _logger.info("PRÉ-SCREENING CD34+/CD45dim")
    _logger.info("=" * 50)

    n_cells = X.shape[0]
    warnings_list: List[str] = []

    # ── Recherche des marqueurs ────────────────────────────────────────────────
    cd34_idx = _find_marker_idx(var_names, ["CD34", "CD34-PE", "CD34-APC", "CD34-PECY7"])
    cd45_idx = _find_marker_idx(var_names, ["CD45", "CD45-PECY5", "CD45-PC5"])

    if cd34_idx is None:
        _logger.warning("[Prescreening] CD34 non trouvé dans les marqueurs — ignoré")
        return None
    if cd45_idx is None:
        _logger.warning("[Prescreening] CD45 non trouvé dans les marqueurs — ignoré")
        return None

    cd34 = X[:, cd34_idx].astype(np.float64)
    cd45 = X[:, cd45_idx].astype(np.float64)

    valid_cd34 = np.isfinite(cd34)
    valid_cd45 = np.isfinite(cd45)
    valid_both = valid_cd34 & valid_cd45

    if valid_both.sum() < 200:
        _logger.warning("[Prescreening] Pas assez de cellules valides (%d < 200)", valid_both.sum())
        return None

    # ── Gate CD45dim ──────────────────────────────────────────────────────────
    mask_cd45dim, cd45_low, cd45_high = _gate_cd45dim(cd45, valid_cd45)
    n_cd45dim = int(mask_cd45dim.sum())
    _logger.info(
        "   CD45dim (%.0f–%.0f): %d cellules (%.1f%%)",
        cd45_low, cd45_high, n_cd45dim, n_cd45dim / max(n_cells, 1) * 100,
    )

    if n_cd45dim < min_cd45dim_cells:
        warn = f"Seulement {n_cd45dim} cellules CD45dim (< {min_cd45dim_cells}) — résultat peu fiable"
        warnings_list.append(warn)
        _logger.warning("   [Prescreening] %s", warn)

    # ── Gate CD34 par les deux méthodes ───────────────────────────────────────
    # GMM
    mask_gmm, threshold_gmm, warns_gmm = _gate_cd34_gmm(cd34, valid_both)
    warnings_list.extend(warns_gmm)
    n_gmm_cd34_pos = int((mask_gmm & mask_cd45dim).sum())
    gmm_ratio = n_gmm_cd34_pos / max(n_cd45dim, 1)
    gmm_ratio_pct = gmm_ratio * 100

    # KDE (référence) — minimum local entre 2 pics
    mask_kde, threshold_kde, kde_x_grid, kde_density, warns_kde = gate_cd34_kde(cd34, valid_both)
    warnings_list.extend(warns_kde)
    n_kde_cd34_pos = int((mask_kde & mask_cd45dim).sum())
    kde_ratio = n_kde_cd34_pos / max(n_cd45dim, 1)
    kde_ratio_pct = kde_ratio * 100

    _logger.info(
        "   [GMM]  CD34+ dans CD45dim: %d / %d = %.1f%%",
        n_gmm_cd34_pos, n_cd45dim, gmm_ratio_pct,
    )
    _logger.info(
        "   [KDE]  CD34+ dans CD45dim: %d / %d = %.1f%%",
        n_kde_cd34_pos, n_cd45dim, kde_ratio_pct,
    )

    # ── Sélection méthode de référence ────────────────────────────────────────
    if density_method.upper() == "GMM":
        mask_ref = mask_gmm
        threshold_ref = threshold_gmm
        n_cd34_pos = n_gmm_cd34_pos
        ratio_pct = gmm_ratio_pct
        ratio = gmm_ratio
    else:
        mask_ref = mask_kde
        threshold_ref = threshold_kde
        n_cd34_pos = n_kde_cd34_pos
        ratio_pct = kde_ratio_pct
        ratio = kde_ratio

    # CD34- dans CD45dim
    n_cd34_neg = n_cd45dim - n_cd34_pos

    # ── Détermination du niveau d'alerte ─────────────────────────────────────
    if ratio >= cd34_ratio_very_high_threshold:
        alert_level = "high"
        alert_message = (
            f"ATTENTION : Rapport CD34+/CD45dim TRÈS ÉLEVÉ ({ratio_pct:.1f}%) !\n"
            f"Suggère une forte régénération médullaire ou une MRD massive.\n"
            f"→ Passez en LAIP Tracking classique ou vérifiez la morphologie.\n"
            f"→ Rapport CD34+/CD45dim élevé — attention pour l'interprétation de la MRD."
        )
        laip = True
        interpretation = (
            f"⚠️ Rapport CD34+/CD45dim élevé ({ratio_pct:.1f}%) — "
            "attention pour l'interprétation de la MRD. "
            "LAIP Tracking classique recommandé."
        )
    elif ratio >= cd34_ratio_high_threshold:
        alert_level = "moderate"
        alert_message = (
            f"Rapport CD34+/CD45dim élevé ({ratio_pct:.1f}%) détecté.\n"
            f"Possible régénération médullaire ou MRD significative.\n"
            f"→ Rapport CD34+/CD45dim élevé — attention pour l'interprétation de la MRD."
        )
        laip = True
        interpretation = (
            f"⚠️ Rapport CD34+/CD45dim modérément élevé ({ratio_pct:.1f}%) — "
            "interprétation MRD à contextualiser avec la morphologie."
        )
    else:
        alert_level = "none"
        alert_message = (
            f"Rapport CD34+/CD45dim normal ({ratio_pct:.1f}%).\n"
            f"Pas de signal de forte MRD ou régénération détecté."
        )
        laip = False
        interpretation = f"Rapport CD34+/CD45dim : {ratio_pct:.1f}% — dans les limites normales."

    _logger.info("   Niveau d'alerte: %s (ratio=%.1f%%)", alert_level, ratio_pct)
    if alert_level != "none":
        _logger.warning("   [Prescreening] %s", alert_message)

    # Le seuil KDE est déjà en espace logicle (x_grid en espace logicle)
    _kde_threshold_log = float(threshold_kde)  # même espace que x_grid du plot

    return PrescreeningResult(
        n_cd34_pos=n_cd34_pos,
        n_cd34_neg=n_cd34_neg,
        n_cd45dim=n_cd45dim,
        ratio_cd34_cd45dim=ratio,
        ratio_pct=ratio_pct,
        alert_level=alert_level,
        alert_message=alert_message,
        method_used=density_method.upper(),
        gmm_n_cd34_pos=n_gmm_cd34_pos,
        gmm_ratio_pct=gmm_ratio_pct,
        kde_n_cd34_pos=n_kde_cd34_pos,
        kde_ratio_pct=kde_ratio_pct,
        cd34_threshold=threshold_ref,
        cd45dim_threshold_low=cd45_low,
        cd45dim_threshold_high=cd45_high,
        warnings=warnings_list,
        laip_tracking_recommended=laip,
        interpretation_warning=interpretation,
        kde_x_grid=kde_x_grid if len(kde_x_grid) > 0 else None,
        kde_density=kde_density if len(kde_density) > 0 else None,
        kde_threshold_log=_kde_threshold_log,
    )
