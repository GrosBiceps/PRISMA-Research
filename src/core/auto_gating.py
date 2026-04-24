"""
auto_gating.py — Gating adaptatif par GMM et régression RANSAC (mode 'auto').

AutoGating utilise scikit-learn (GaussianMixture, RANSACRegressor) pour
détecter automatiquement les seuils de gating en modélisant les populations
naturelles dans les données, sans percentiles fixes.

Avantages vs PreGating (percentiles):
    - Si un échantillon a 10% de débris → la porte s'adapte automatiquement
    - Si un échantillon est propre → moins de perte de données
    - Pour les doublets: modélise la diagonale FSC-A/FSC-H statistiquement
    - Pour CD45+: trouve le creux bimodal entre CD45- et CD45+

Chaque méthode:
    - Retourne un masque numpy booléen
    - Enregistre un GateResult dans le registre global
    - Log structuré l'événement (audit JSON)
    - Gère les fallbacks (GMM → percentile si échec)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

_logger = logging.getLogger("core.auto_gating")
_rng = np.random.default_rng(seed=42)

from ..models.gate_result import (
    GateResult,
    gating_reports,
    gating_log_entries,
    log_gating_event,
)
from .gating import PreGating
from config.constants import (
    GMM_MAX_SAMPLES,
    RANSAC_R2_THRESHOLD,
    RANSAC_MAD_FACTOR,
)


# =============================================================================
# CLASSE AutoGating — Gating adaptatif par GMM/KDE
# =============================================================================
# Inspiré de CytoPy AutonomousGate (sans dépendance MongoDB)
# Utilise scikit-learn GaussianMixture pour trouver les "creux" réels
# entre les populations au lieu de couper à des percentiles fixes.
#
# Avantages vs PreGating (percentiles):
#   - Si un échantillon a 10% de débris → la porte s'adapte automatiquement
#   - Si un échantillon est propre → moins de perte de données
#   - Pour les doublets: modélise la diagonale FSC-A/FSC-H statistiquement
#   - Pour CD45+: trouve le creux bimodal entre CD45- et CD45+
#
# [V2 AMÉLIORATIONS]:
#   - safe_fit_gmm: sous-échantillonnage à 200k points max avant fit
#   - auto_gate_singlets: contrôle R² RANSAC + fallback ratio si R² < 0.85
#   - Toutes les fonctions retournent un GateResult structuré
#   - Scatter FSC-A vs FSC-H par fichier + tableau % singlets stockés
#   - Log structuré JSON pour audit automatique des runs
# =============================================================================
# GatingSession — conteneur de l'état mutable d'un run de gating.
#
# Remplace les anciens globals ransac_scatter_data / singlets_summary_per_file
# qui causaient des contaminations inter-patients en mode batch (les données du
# patient N restaient visibles pour le patient N+1 sans .clear() explicite).
#
# Usage recommandé :
#   session = GatingSession()
#   AutoGating.auto_gate_singlets(..., session=session)
#   scatter = session.ransac_scatter_data
#
# Rétrocompatibilité :
#   Les noms ransac_scatter_data et singlets_summary_per_file restent
#   exportés comme références vers l'instance de session par défaut
#   (_default_session). Le pipeline_executor peut toujours appeler .clear()
#   sur ces objets — ils se comportent comme des dict/list normaux.
# =============================================================================


class GatingSession:
    """
    Conteneur thread-local de l'état mutable d'un run de gating.

    Encapsule les données RANSAC par fichier et le résumé des singlets,
    évitant la contamination entre patients consécutifs en mode batch.
    """

    def __init__(self) -> None:
        self.ransac_scatter_data: Dict[str, Any] = {}
        self.singlets_summary_per_file: List[Dict[str, Any]] = []

    def clear(self) -> None:
        """Réinitialise la session avant un nouveau run patient."""
        self.ransac_scatter_data.clear()
        self.singlets_summary_per_file.clear()


# Instance par défaut — maintient la rétrocompatibilité avec les imports
# existants dans pipeline_executor.py qui font :
#   from src.core.auto_gating import ransac_scatter_data, ...
_default_session: GatingSession = GatingSession()
ransac_scatter_data: Dict[str, Any] = _default_session.ransac_scatter_data
singlets_summary_per_file: List[Dict[str, Any]] = _default_session.singlets_summary_per_file


class AutoGating:
    """
    Gating automatique adaptatif basé sur des modèles de mélange gaussien (GMM)
    et estimation de densité. Inspiré de CytoPy AutonomousGate.

    Chaque méthode utilise un GMM pour identifier les populations naturelles
    dans les données, au lieu de seuils fixes basés sur des percentiles.

    Dépendances: scikit-learn (GaussianMixture, StandardScaler)
    """

    @staticmethod
    def _subsample_for_gmm(data: np.ndarray, max_samples: int = None) -> np.ndarray:
        """
        Sous-échantillonne les données si elles dépassent max_samples.
        Améliore la convergence et évite les timeouts implicites sur gros jeux.

        Args:
            data: Données (n_samples, n_features)
            max_samples: Nombre max de points (défaut: GMM_MAX_SAMPLES)

        Returns:
            data_subsampled: Données sous-échantillonnées (ou originales si < max)
        """
        if max_samples is None:
            max_samples = GMM_MAX_SAMPLES
        max_samples = max(1, int(max_samples))
        if data.shape[0] > max_samples:
            idx = _rng.choice(data.shape[0], size=max_samples, replace=False)
            _logger.debug(
                "[GMM] Sous-échantillonnage: %d -> %d points",
                data.shape[0],
                max_samples,
            )
            return data[idx]
        return data

    @staticmethod
    def safe_fit_gmm(
        data: np.ndarray,
        n_components: int = 2,
        n_init: int = 1,
        max_retries: int = 3,
        random_state: int = 42,
        covariance_type: str = "full",
        max_iter: int = 100,
        subsample: bool = True,
    ) -> Any:
        """
        Wrapper robuste pour le fitting GMM avec gestion d'erreurs.

        Tente le fit plusieurs fois avec différentes initialisations.
        En cas d'échec total sur n_components > 1, fallback sur 1 composante.
        Vérifie la convergence et émet des warnings si nécessaire.

        [V3] Sous-échantillonnage automatique à 50k points max avant fit.

        Args:
            data: Données à fitter (n_samples, n_features) ou (n_samples, 1)
            n_components: Nombre de composantes GMM
            n_init: Nombre d'initialisations par tentative
            max_retries: Nombre max de tentatives avant fallback
            random_state: Seed pour reproductibilité
            covariance_type: Type de covariance ('full', 'diag', 'spherical', 'tied')
            max_iter: Nombre max d'itérations EM
            subsample: Si True, sous-échantillonne avant fit (défaut True)

        Returns:
            GaussianMixture fitté

        Raises:
            RuntimeError: Si le fit échoue après toutes les tentatives (y compris fallback)
        """
        from sklearn.mixture import GaussianMixture

        # Sous-échantillonnage pour convergence rapide sur gros jeux de données
        if subsample:
            data_fit = AutoGating._subsample_for_gmm(data)
        else:
            data_fit = data

        last_error = None
        for attempt in range(max_retries):
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    random_state=random_state + attempt,
                    n_init=n_init,
                    covariance_type=covariance_type,
                    max_iter=max_iter,
                )
                gmm.fit(data_fit)
                if not gmm.converged_:
                    warnings.warn(
                        f"GMM non-convergé (n={n_components}, tentative {attempt + 1}/{max_retries})"
                    )
                    log_gating_event(
                        "GMM",
                        f"n_components={n_components}",
                        "warning",
                        {"attempt": attempt + 1},
                        f"Non-convergé tentative {attempt + 1}/{max_retries}",
                    )
                    continue
                return gmm
            except Exception as e:
                last_error = e
                log_gating_event(
                    "GMM",
                    f"n_components={n_components}",
                    "error",
                    {"attempt": attempt + 1, "error": str(e)},
                )
                continue

        # Fallback: tenter avec 1 composante si n_components > 1
        if n_components > 1:
            warn_msg = f"GMM fallback unimodal après {max_retries} échecs (dernière erreur: {last_error})"
            warnings.warn(warn_msg)
            log_gating_event(
                "GMM",
                "fallback_unimodal",
                "fallback",
                {"original_n_components": n_components, "error": str(last_error)},
                warn_msg,
            )
            try:
                gmm = GaussianMixture(
                    n_components=1,
                    random_state=random_state,
                    n_init=1,
                    covariance_type=covariance_type,
                    max_iter=max_iter,
                )
                gmm.fit(data_fit)
                return gmm
            except Exception as e:
                raise RuntimeError(
                    f"GMM fit échoué après {max_retries} tentatives + fallback unimodal: {e}"
                )

        raise RuntimeError(
            f"GMM fit échoué après {max_retries} tentatives: {last_error}"
        )

    @staticmethod
    def auto_gate_debris(
        X: np.ndarray,
        var_names: List[str],
        n_components: int = 3,
        min_cluster_fraction: float = 0.02,
        gmm_max_samples: int = GMM_MAX_SAMPLES,
        covariance_type: str = "full",
        density_method: str = "GMM",
        export_plot: bool = False,
        plot_output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Gate débris adaptatif par GMM ou KDE 2D sur FSC-A / SSC-A.

        Méthode sélectionnable via ``density_method`` :
          - "GMM" (défaut) : Gaussian Mixture Model — paramétrable, rapide.
          - "KDE" : Kernel Density Estimation — non paramétrique, plus fidèle aux données.

        En mode GMM, ``covariance_type`` contrôle la forme des gaussiennes :
          - "full"     : forme libre par cluster (défaut, risque sur-exclusion)
          - "tied"     : même forme pour tous les clusters (plus robuste)
          - "diag"     : gaussiennes compactes (recommandé si sur-exclusion des viables)
          - "spherical": forme sphérique (plus contrainte)

        Args:
            X: Matrice des données (n_cells, n_markers)
            var_names: Noms des marqueurs
            n_components: Nombre max de composantes GMM à tester (2 ou 3)
            min_cluster_fraction: Fraction min d'événements pour inclure un cluster
            gmm_max_samples: Nombre max de cellules utilisées pour le fit/BIC GMM
            covariance_type: Type de covariance GMM ('full', 'tied', 'diag', 'spherical')
            density_method: 'GMM' ou 'KDE'
            export_plot: Si True, exporte un graphique des densités GMM
            plot_output_path: Chemin de sortie du graphique (PNG)

        Returns:
            Masque booléen (True = cellule viable, False = débris/saturé)
        """
        from sklearn.preprocessing import StandardScaler

        n_cells = X.shape[0]
        fsc_idx = PreGating.find_marker_index(var_names, ["FSC-A"])
        ssc_idx = PreGating.find_marker_index(var_names, ["SSC-A"])

        if fsc_idx is None or ssc_idx is None:
            _logger.warning("[!] FSC-A ou SSC-A non trouvé pour auto-gate débris")
            log_gating_event(
                "debris", "auto_gmm", "error", warning_msg="FSC-A ou SSC-A non trouvé"
            )
            return np.ones(n_cells, dtype=bool)

        fsc = X[:, fsc_idx].astype(np.float64)
        ssc = X[:, ssc_idx].astype(np.float64)

        # Filtrer les NaN/Inf
        valid = np.isfinite(fsc) & np.isfinite(ssc)
        data_2d = np.column_stack([fsc[valid], ssc[valid]])

        if valid.sum() < 200:
            _logger.warning("[!] Pas assez de données valides pour auto-gate débris")
            return np.ones(n_cells, dtype=bool)

        # ── Méthode KDE ────────────────────────────────────────────────────────
        if density_method.upper() == "KDE":
            return AutoGating._auto_gate_debris_kde(
                fsc, ssc, valid, n_cells, min_cluster_fraction
            )

        # ── Méthode GMM (défaut) ────────────────────────────────────────────────
        # Standardiser avant GMM pour meilleure convergence
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_2d)

        # Sélection automatique du nombre de composantes par BIC
        _n_comp_candidates = list({2, min(n_components, 3)})
        _n_comp_candidates.sort()
        best_bic = np.inf
        best_gmm = None
        for n_comp in _n_comp_candidates:
            try:
                gmm_test = AutoGating.safe_fit_gmm(
                    data_scaled,
                    n_components=n_comp,
                    covariance_type=covariance_type,
                    n_init=1,
                    max_iter=100,
                )
                bic = gmm_test.bic(
                    data_scaled
                    if data_scaled.shape[0] <= int(gmm_max_samples)
                    else AutoGating._subsample_for_gmm(
                        data_scaled, max_samples=int(gmm_max_samples)
                    )
                )
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm_test
            except RuntimeError as e:
                _logger.warning("[!] GMM %d composantes échoué: %s", n_comp, e)
                continue

        if best_gmm is None:
            _logger.warning(
                "[!] Aucun GMM n'a convergé, conservation de tous les événements"
            )
            log_gating_event(
                "debris",
                "auto_gmm",
                "fallback",
                warning_msg="Aucun GMM convergé, toutes cellules conservées",
            )
            return np.ones(n_cells, dtype=bool)

        labels = best_gmm.predict(data_scaled)
        n_comp = best_gmm.n_components

        # Statistiques par cluster (en espace original)
        cluster_sizes = np.bincount(labels, minlength=n_comp)
        cluster_fsc_means = np.array(
            [data_2d[labels == i, 0].mean() for i in range(n_comp)]
        )

        # Population principale = plus grand cluster
        main_cluster = np.argmax(cluster_sizes)

        # Inclure les clusters avec assez d'événements et un FSC raisonnable
        # (exclure les débris = FSC très bas)
        mask_valid = np.zeros(valid.sum(), dtype=bool)
        fsc_threshold = cluster_fsc_means[main_cluster] * 0.25

        for i in range(n_comp):
            fraction = cluster_sizes[i] / len(labels)
            if (
                fraction >= min_cluster_fraction
                and cluster_fsc_means[i] >= fsc_threshold
            ):
                mask_valid |= labels == i

        # Sécurité: si aucun cluster sélectionné, garder le principal
        if not mask_valid.any():
            mask_valid = labels == main_cluster

        mask = np.zeros(n_cells, dtype=bool)
        mask[valid] = mask_valid

        n_kept = mask.sum()
        _logger.info(
            "   [Auto-GMM] %d composantes détectées (BIC=%.0f)",
            best_gmm.n_components,
            best_bic,
        )
        for i in range(n_comp):
            status = "[OK]" if mask_valid[labels == i].any() else "[--]"
            _logger.info(
                "     %s Cluster %d: %d evt, FSC-A moy=%.0f",
                status,
                i,
                cluster_sizes[i],
                cluster_fsc_means[i],
            )
        _logger.info("   [Auto-GMM] Conservés: %d événements", n_kept)

        # Log structuré
        log_gating_event(
            "debris",
            "auto_gmm",
            "success",
            {
                "n_components": int(n_comp),
                "bic": float(best_bic),
                "n_kept": int(n_kept),
                "n_total": int(n_cells),
                "cluster_sizes": cluster_sizes.tolist(),
            },
        )

        # Export graphique GMM si demandé
        if export_plot:
            try:
                AutoGating._export_gmm_density_plot(
                    data_2d=data_2d,
                    labels=labels,
                    mask_valid=mask_valid,
                    cluster_fsc_means=cluster_fsc_means,
                    cluster_sizes=cluster_sizes,
                    n_comp=n_comp,
                    covariance_type=covariance_type,
                    output_path=plot_output_path,
                )
            except Exception as _pe:
                _logger.warning("Export graphique GMM échoué (non bloquant): %s", _pe)

        # Construire GateResult
        gate_result = GateResult(
            mask=mask,
            n_kept=int(n_kept),
            n_total=int(n_cells),
            method="auto_gmm_debris",
            gate_name="G1_debris",
            details={
                "n_components": int(n_comp),
                "bic": float(best_bic),
                "cluster_fsc_means": cluster_fsc_means.tolist(),
                "covariance_type": covariance_type,
            },
        )
        gating_reports.append(gate_result)

        return mask

    # ─────────────────────────────────────────────────────────────────────────
    # Méthodes internes
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _auto_gate_debris_kde(
        fsc: np.ndarray,
        ssc: np.ndarray,
        valid: np.ndarray,
        n_cells: int,
        min_cluster_fraction: float = 0.02,
    ) -> np.ndarray:
        """
        Gating débris par KDE (Kernel Density Estimation).

        Estime la densité conjointe FSC-A / SSC-A, puis identifie la région
        de haute densité comme étant les cellules viables.
        Méthode non paramétrique — plus fidèle mais non configurable.
        """
        try:
            from scipy.stats import gaussian_kde
        except ImportError:
            _logger.warning("[KDE] scipy non disponible — fallback GMM")
            return np.ones(n_cells, dtype=bool)

        fsc_v = fsc[valid]
        ssc_v = ssc[valid]

        # Sous-échantillonner pour performance KDE
        n_kde = min(len(fsc_v), GMM_MAX_SAMPLES)
        if len(fsc_v) > n_kde:
            idx = _rng.choice(len(fsc_v), size=n_kde, replace=False)
            fsc_kde = fsc_v[idx]
            ssc_kde = ssc_v[idx]
        else:
            fsc_kde = fsc_v
            ssc_kde = ssc_v

        # Estimer la densité
        kde = gaussian_kde(np.vstack([fsc_kde, ssc_kde]))
        density = kde(np.vstack([fsc_v, ssc_v]))

        # Seuil = médiane de la densité (conserver les cellules au-dessus)
        threshold = np.median(density) * 0.3
        mask_valid_kde = density >= threshold

        # Sécurité: si trop peu de cellules conservées, garder tout
        if mask_valid_kde.sum() < int(len(fsc_v) * min_cluster_fraction):
            mask_valid_kde = np.ones(len(fsc_v), dtype=bool)

        mask = np.zeros(n_cells, dtype=bool)
        mask[valid] = mask_valid_kde

        n_kept = mask.sum()
        _logger.info(
            "   [Auto-KDE] Conservés: %d / %d événements (%.1f%%)",
            n_kept,
            n_cells,
            n_kept / n_cells * 100,
        )
        log_gating_event(
            "debris",
            "auto_kde",
            "success",
            {"n_kept": int(n_kept), "n_total": int(n_cells)},
        )

        gate_result = GateResult(
            mask=mask,
            n_kept=int(n_kept),
            n_total=int(n_cells),
            method="auto_kde_debris",
            gate_name="G1_debris",
            details={"method": "KDE", "threshold": float(threshold)},
        )
        gating_reports.append(gate_result)

        return mask

    @staticmethod
    def _export_gmm_density_plot(
        data_2d: np.ndarray,
        labels: np.ndarray,
        mask_valid: np.ndarray,
        cluster_fsc_means: np.ndarray,
        cluster_sizes: np.ndarray,
        n_comp: int,
        covariance_type: str,
        output_path: Optional[str] = None,
    ) -> None:
        """
        Exporte un graphique de densité GMM montrant les populations identifiées.

        Chaque composante gaussienne est représentée par :
          - Un nuage de points coloré (cellules appartenant au cluster)
          - Une courbe de densité 1D (KDE sur FSC-A) par cluster
          - Un indicateur [CONSERVÉ] / [EXCLU] selon la logique de gating

        Args:
            data_2d: Données brutes FSC-A / SSC-A (n_valid, 2)
            labels: Assignation cluster GMM (n_valid,)
            mask_valid: Masque des clusters conservés (n_comp,)
            cluster_fsc_means: Moyenne FSC-A par cluster
            cluster_sizes: Taille de chaque cluster
            n_comp: Nombre de composantes
            covariance_type: Type de covariance utilisé
            output_path: Chemin de sauvegarde (PNG). Si None, génère un nom auto.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde as scipy_kde
        import os

        colors = ["#4cc9f0", "#f72585", "#7209b7", "#3a0ca3", "#4361ee"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#1e1e2e")
        for ax in axes:
            ax.set_facecolor("#1e1e2e")
            ax.tick_params(colors="#e2e8f0")
            ax.xaxis.label.set_color("#e2e8f0")
            ax.yaxis.label.set_color("#e2e8f0")
            ax.title.set_color("#e2e8f0")
            for spine in ax.spines.values():
                spine.set_edgecolor("#45475a")

        fsc_all = data_2d[:, 0]
        ssc_all = data_2d[:, 1]

        # Panneau gauche : scatter FSC-A vs SSC-A coloré par cluster
        ax_scatter = axes[0]
        n_display = min(len(fsc_all), 30_000)
        idx_disp = _rng.choice(len(fsc_all), size=n_display, replace=False)

        for i in range(n_comp):
            cluster_idx = np.where(labels == i)[0]
            cluster_disp = np.intersect1d(cluster_idx, idx_disp)
            kept = bool(mask_valid[labels == i].any())
            alpha = 0.6 if kept else 0.2
            label_txt = (
                f"Pop {i} ({'CONSERVÉE' if kept else 'EXCLUE'}) n={cluster_sizes[i]:,}"
            )
            ax_scatter.scatter(
                fsc_all[cluster_disp],
                ssc_all[cluster_disp],
                s=2,
                alpha=alpha,
                color=colors[i % len(colors)],
                label=label_txt,
                rasterized=True,
            )

        ax_scatter.set_xlabel("FSC-A")
        ax_scatter.set_ylabel("SSC-A")
        ax_scatter.set_title(f"Populations GMM — covariance: {covariance_type}")
        ax_scatter.legend(
            fontsize=7,
            facecolor="#313244",
            labelcolor="#e2e8f0",
            markerscale=4,
            framealpha=0.8,
        )

        # Panneau droit : densités 1D FSC-A par cluster
        ax_density = axes[1]
        fsc_range = np.linspace(fsc_all.min(), fsc_all.max(), 500)

        for i in range(n_comp):
            cluster_mask = labels == i
            fsc_cluster = fsc_all[cluster_mask]
            if len(fsc_cluster) < 10:
                continue
            kept = bool(mask_valid[cluster_mask].any())
            linestyle = "-" if kept else "--"
            alpha = 0.9 if kept else 0.4
            label_txt = f"Pop {i} ({'CONSERVÉE' if kept else 'EXCLUE'})"
            try:
                # Sous-échantillonnage strict : scipy_kde sur 3.8M pts → 68s
                _GMM_KDE_MAX = 10_000
                if len(fsc_cluster) > _GMM_KDE_MAX:
                    fsc_cluster_sub = _rng.choice(
                        fsc_cluster, size=_GMM_KDE_MAX, replace=False
                    )
                else:
                    fsc_cluster_sub = fsc_cluster
                kde_fn = scipy_kde(fsc_cluster_sub)
                density_vals = kde_fn(fsc_range)
                # Normaliser par taille relative du cluster
                weight = cluster_sizes[i] / cluster_sizes.sum()
                ax_density.fill_between(
                    fsc_range,
                    density_vals * weight,
                    alpha=0.2 if kept else 0.08,
                    color=colors[i % len(colors)],
                )
                ax_density.plot(
                    fsc_range,
                    density_vals * weight,
                    color=colors[i % len(colors)],
                    linewidth=2 if kept else 1,
                    linestyle=linestyle,
                    alpha=alpha,
                    label=label_txt,
                )
            except Exception:
                pass

        ax_density.set_xlabel("FSC-A")
        ax_density.set_ylabel("Densité (pondérée par taille du cluster)")
        ax_density.set_title("Densités GMM par population")
        ax_density.legend(
            fontsize=8, facecolor="#313244", labelcolor="#e2e8f0", framealpha=0.8
        )

        plt.suptitle(
            f"Gating débris — GMM ({n_comp} composantes, covariance={covariance_type})",
            color="#e2e8f0",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()

        if output_path is None:
            output_path = "gmm_debris_density.png"
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(
            output_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor()
        )
        plt.close(fig)
        _logger.info("[GMM] Graphique densités exporté: %s", output_path)

    @staticmethod
    def auto_gate_singlets(
        X: np.ndarray,
        var_names: List[str],
        file_origin: Optional[np.ndarray] = None,
        per_file: bool = True,
        r2_threshold: float = 0.85,
        mad_factor: float = RANSAC_MAD_FACTOR,
        session: Optional["GatingSession"] = None,
    ) -> np.ndarray:
        """
        Gate singlets adaptatif par régression linéaire robuste (RANSAC).

        Les singlets forment une diagonale sur le plot FSC-A vs FSC-H.
        Les doublets se situent au-dessus de cette diagonale (FSC-A augmente mais pas FSC-H).

        Méthode améliorée (V2):
        1. Pré-filtre viable (FSC/SSC 1-99%) pour exclure les outliers extrêmes
        2. Régression linéaire robuste RANSAC sur FSC-A vs FSC-H
        3. Contrôle qualité R² sur les inliers RANSAC
        4. Si R² < seuil (0.85): fallback vers gating ratio FSC-A/FSC-H simple
        5. Stockage des scatter data par fichier dans la session (plus de global mutable)

        Args:
            X: Matrice des données (n_cells, n_markers)
            var_names: Noms des marqueurs
            file_origin: Array contenant l'origine de chaque cellule (pour gating par fichier)
            per_file: Si True, applique le gating séparément par fichier
            r2_threshold: Seuil R² minimum (défaut 0.85). En dessous → fallback ratio
            session: GatingSession à utiliser pour stocker les résultats. Si None,
                     utilise _default_session (rétrocompatibilité).

        Returns:
            Masque booléen (True = singlet, False = doublet)
        """
        # Résoudre la session — préférer l'argument explicite, sinon session par défaut
        _session = session if session is not None else _default_session
        _ransac_scatter = _session.ransac_scatter_data
        _singlets_summary = _session.singlets_summary_per_file
        from sklearn.linear_model import RANSACRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        n_cells = X.shape[0]
        fsc_a_idx = PreGating.find_marker_index(var_names, ["FSC-A"])
        fsc_h_idx = PreGating.find_marker_index(var_names, ["FSC-H"])

        if fsc_a_idx is None or fsc_h_idx is None:
            _logger.warning("[!] FSC-A ou FSC-H non trouvé pour auto-gate singlets")
            return np.ones(n_cells, dtype=bool)

        fsc_a = X[:, fsc_a_idx].astype(np.float64)
        fsc_h = X[:, fsc_h_idx].astype(np.float64)

        # Pré-filtre viable (FSC/SSC 1-99%) pour réduire l'impact des outliers
        # extrêmes (blastes matures, granulocytes agrégés) sur la régression RANSAC
        viable = PreGating.gate_viable_cells(
            X, var_names, min_percentile=1.0, max_percentile=99.0
        )

        # Filtrer: valeurs valides avec FSC > seuil minimal + viabilité
        valid = (
            viable
            & np.isfinite(fsc_a)
            & np.isfinite(fsc_h)
            & (fsc_h > 100)
            & (fsc_a > 100)
        )

        if valid.sum() < 200:
            _logger.warning("[!] Pas assez de données valides pour auto-gate singlets")
            return np.ones(n_cells, dtype=bool)

        mask = np.zeros(n_cells, dtype=bool)

        # ─── Helper interne: fallback ratio FSC-A/FSC-H ───
        def _fallback_ratio_gating(
            fsc_a_local, fsc_h_local, ratio_min=0.6, ratio_max=1.5
        ):
            """Gating simple par ratio FSC-A/FSC-H (ancienne méthode)."""
            ratio = fsc_a_local.ravel() / np.maximum(fsc_h_local.ravel(), 1.0)
            return (ratio >= ratio_min) & (ratio <= ratio_max)

        # Gating par fichier si demandé et si file_origin fourni
        if per_file and file_origin is not None:
            unique_files = np.unique(file_origin[valid])
            _logger.info(
                "   [Auto-RANSAC] Gating par fichier (%d fichiers)", len(unique_files)
            )

            total_singlets = 0
            total_doublets = 0
            file_payloads: List[Dict[str, Any]] = []

            def _process_one_file(
                file_name: Any, file_indices: np.ndarray
            ) -> Dict[str, Any]:
                n_file = int(file_indices.size)
                payload: Dict[str, Any] = {
                    "file": str(file_name),
                    "indices": file_indices,
                    "method": "skip_too_few",
                    "r2": None,
                    "slope": None,
                    "intercept": None,
                    "event_name": None,
                    "event_status": None,
                    "event_details": None,
                    "event_warning": None,
                    "scatter": None,
                    "error": None,
                    "singlets": np.ones(n_file, dtype=bool),
                }

                if n_file < 50:
                    return payload

                fsc_a_file = fsc_a[file_indices].reshape(-1, 1)
                fsc_h_file = fsc_h[file_indices].reshape(-1, 1)

                try:
                    ransac = RANSACRegressor(
                        estimator=LinearRegression(),
                        min_samples=50,
                        residual_threshold=None,
                        random_state=42,
                        max_trials=100,
                    )
                    ransac.fit(fsc_h_file, fsc_a_file.ravel())

                    inlier_mask = ransac.inlier_mask_
                    r2_val = None
                    if inlier_mask is not None and inlier_mask.sum() > 50:
                        r2_val = r2_score(
                            fsc_a_file[inlier_mask].ravel(),
                            ransac.predict(fsc_h_file[inlier_mask]),
                        )

                    slope = float(ransac.estimator_.coef_[0])
                    intercept = float(ransac.estimator_.intercept_)
                    payload["r2"] = float(r2_val) if r2_val is not None else None
                    payload["slope"] = slope
                    payload["intercept"] = intercept

                    if r2_val is not None and r2_val < r2_threshold:
                        singlets_file = _fallback_ratio_gating(fsc_a_file, fsc_h_file)
                        payload["singlets"] = singlets_file
                        payload["method"] = "ratio_fallback"
                        payload["event_name"] = "ransac_fallback_ratio"
                        payload["event_status"] = "fallback"
                        payload["event_details"] = {
                            "file": str(file_name),
                            "r2": float(r2_val),
                        }
                        payload["event_warning"] = (
                            f"R² faible pour {file_name} (R²={r2_val:.2f} < {r2_threshold}), "
                            "fallback gating ratio"
                        )
                    else:
                        fsc_a_pred = ransac.predict(fsc_h_file)
                        residuals = fsc_a_file.ravel() - fsc_a_pred
                        median_residual = np.median(residuals)
                        mad = np.median(np.abs(residuals - median_residual))
                        threshold_upper = median_residual + mad_factor * mad
                        payload["singlets"] = residuals <= threshold_upper
                        payload["method"] = "ransac"
                        payload["event_name"] = "ransac"
                        payload["event_status"] = "success"

                    n_sample_pts = min(2000, n_file)
                    local_rng = np.random.default_rng(
                        (abs(hash(str(file_name))) + 42) % (2**32 - 1)
                    )
                    sample_idx = local_rng.choice(n_file, n_sample_pts, replace=False)
                    payload["scatter"] = {
                        "fsc_h": fsc_h_file[sample_idx].ravel().tolist(),
                        "fsc_a": fsc_a_file[sample_idx].ravel().tolist(),
                        "pred": ransac.predict(fsc_h_file[sample_idx]).tolist(),
                        "r2": payload["r2"],
                        "method": payload["method"],
                        "slope": slope,
                        "intercept": intercept,
                    }
                    return payload
                except Exception as e:
                    payload["method"] = "error_keep_all"
                    payload["error"] = str(e)
                    payload["event_name"] = "ransac"
                    payload["event_status"] = "error"
                    payload["event_details"] = {"file": str(file_name), "error": str(e)}
                    payload["event_warning"] = f"Échec RANSAC pour {file_name}: {e}"
                    return payload

            file_tasks = []
            for file_name in unique_files:
                file_indices = np.where((file_origin == file_name) & valid)[0]
                if file_indices.size > 0:
                    file_tasks.append((file_name, file_indices))

            try:
                from joblib import Parallel, delayed

                if len(file_tasks) > 1:
                    file_payloads = Parallel(n_jobs=-1, prefer="threads")(
                        delayed(_process_one_file)(fname, findices)
                        for fname, findices in file_tasks
                    )
                else:
                    file_payloads = [
                        _process_one_file(fname, findices)
                        for fname, findices in file_tasks
                    ]
            except Exception as _par_e:
                _logger.debug("Joblib indisponible pour RANSAC parallèle: %s", _par_e)
                file_payloads = [
                    _process_one_file(fname, findices) for fname, findices in file_tasks
                ]

            for payload in file_payloads:
                file_name = payload["file"]
                file_indices = payload["indices"]
                singlets_file = payload["singlets"]
                method = payload["method"]
                r2_val = payload["r2"]

                mask[file_indices] = singlets_file
                n_sing = int(singlets_file.sum())
                n_tot = int(len(singlets_file))
                n_doub = n_tot - n_sing
                total_singlets += n_sing
                total_doublets += n_doub

                if payload["event_name"] is not None:
                    log_gating_event(
                        "singlets",
                        payload["event_name"],
                        payload["event_status"],
                        payload["event_details"],
                        payload["event_warning"],
                    )

                if payload["scatter"] is not None:
                    _ransac_scatter[file_name] = payload["scatter"]

                if payload["error"] is not None:
                    _logger.warning(
                        "      [!] Échec RANSAC pour %s: %s",
                        file_name,
                        payload["error"],
                    )
                elif method == "ratio_fallback":
                    _logger.warning("      [!] %s", payload["event_warning"])

                file_short = (
                    file_name if len(file_name) <= 25 else file_name[:22] + "..."
                )
                if method == "ratio_fallback":
                    _logger.info(
                        "      • %s: %d singlets / %d (%.1f%%) - RATIO FALLBACK (R²=%.2f)",
                        file_short,
                        n_sing,
                        n_tot,
                        n_sing / max(n_tot, 1) * 100,
                        r2_val if r2_val is not None else float("nan"),
                    )
                elif method == "ransac":
                    r2_str = f", R²={r2_val:.3f}" if r2_val is not None else ""
                    _logger.info(
                        "      • %s: %d singlets / %d (%.1f%%) - y=%.3fx+%.0f%s",
                        file_short,
                        n_sing,
                        n_tot,
                        n_sing / max(n_tot, 1) * 100,
                        payload["slope"],
                        payload["intercept"],
                        r2_str,
                    )
                else:
                    _logger.info(
                        "      • %s: %d singlets / %d (%.1f%%) - %s",
                        file_short,
                        n_sing,
                        n_tot,
                        n_sing / max(n_tot, 1) * 100,
                        method,
                    )

                _singlets_summary.append(
                    {
                        "file": str(file_name),
                        "n_total": n_tot,
                        "n_singlets": n_sing,
                        "pct_singlets": round(n_sing / max(n_tot, 1) * 100, 1),
                        "method": method,
                        "r2": round(float(r2_val), 3) if r2_val is not None else None,
                    }
                )

            _logger.info(
                "   [Auto-RANSAC] Total: %d singlets, %d doublets exclus",
                total_singlets,
                total_doublets,
            )

            # Résumé tableau % singlets par fichier
            if _singlets_summary:
                header = (
                    f"\n   {'Fichier':<30} {'Méthode':<18} {'R²':>6} {'% Singlets':>12}"
                )
                _logger.info(header)
                for row in _singlets_summary:
                    r2_disp = f"{row['r2']:.3f}" if row["r2"] is not None else "N/A"
                    fname_short = (
                        row["file"]
                        if len(row["file"]) <= 30
                        else row["file"][:27] + "..."
                    )
                    _logger.info(
                        "   %-30s %-18s %6s %10.1f%%",
                        fname_short,
                        row["method"],
                        r2_disp,
                        row["pct_singlets"],
                    )

        else:
            # Gating global (ancien comportement)
            _logger.info("   [Auto-RANSAC] Gating global sur toutes les données")

            fsc_a_valid = fsc_a[valid].reshape(-1, 1)
            fsc_h_valid = fsc_h[valid].reshape(-1, 1)

            # Régression RANSAC
            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=100,
                residual_threshold=None,
                random_state=42,
                max_trials=100,
            )
            ransac.fit(fsc_h_valid, fsc_a_valid.ravel())

            # ─── CONTRÔLE QUALITÉ R² GLOBAL ───
            inlier_mask = ransac.inlier_mask_
            r2_val = None
            if inlier_mask is not None and inlier_mask.sum() > 50:
                r2_val = r2_score(
                    fsc_a_valid[inlier_mask].ravel(),
                    ransac.predict(fsc_h_valid[inlier_mask]),
                )
                if r2_val < r2_threshold:
                    warn_msg = f"R² faible global (R²={r2_val:.2f} < {r2_threshold}), fallback gating ratio"
                    _logger.warning("   [!] %s", warn_msg)
                    log_gating_event(
                        "singlets",
                        "ransac_fallback_ratio",
                        "fallback",
                        {"r2": float(r2_val)},
                        warn_msg,
                    )

                    singlets_mask = _fallback_ratio_gating(fsc_a_valid, fsc_h_valid)
                    mask[valid] = singlets_mask

                    n_singlets = mask.sum()
                    n_doublets = valid.sum() - n_singlets
                    _logger.info(
                        "   [RATIO FALLBACK] Singlets: %d (%.1f%%)",
                        n_singlets,
                        n_singlets / valid.sum() * 100,
                    )

                    gate_result = GateResult(
                        mask=mask,
                        n_kept=int(n_singlets),
                        n_total=int(n_cells),
                        method="ratio_fallback_global",
                        gate_name="G2_singlets",
                        details={"r2": float(r2_val)},
                        warnings=[warn_msg],
                    )
                    gating_reports.append(gate_result)
                    return mask

            # Prédiction et résidus
            fsc_a_pred = ransac.predict(fsc_h_valid)
            residuals = fsc_a_valid.ravel() - fsc_a_pred

            # Seuil adaptatif MAD
            median_residual = np.median(residuals)
            mad = np.median(np.abs(residuals - median_residual))
            threshold_upper = median_residual + mad_factor * mad

            # Masque singlets
            singlets_mask = residuals <= threshold_upper
            mask[valid] = singlets_mask

            n_singlets = mask.sum()
            n_doublets = valid.sum() - n_singlets
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_

            r2_str = f", R²={r2_val:.3f}" if r2_val is not None else ""
            _logger.info(
                "   [Auto-RANSAC] Droite: y = %.3fx + %.0f%s", slope, intercept, r2_str
            )
            _logger.info(
                "   [Auto-RANSAC] Seuil MAD: médiane + %.1f×MAD = %.0f",
                mad_factor,
                threshold_upper,
            )
            _logger.info(
                "   [Auto-RANSAC] Singlets: %d (%.1f%%)",
                n_singlets,
                n_singlets / valid.sum() * 100,
            )
            _logger.info(
                "   [Auto-RANSAC] Doublets rejetés: %d (%.1f%%)",
                n_doublets,
                n_doublets / valid.sum() * 100,
            )

        # GateResult structuré
        gate_result = GateResult(
            mask=mask,
            n_kept=int(mask.sum()),
            n_total=int(n_cells),
            method="ransac_singlets",
            gate_name="G2_singlets",
            details={
                "per_file": per_file,
                "n_files": len(_singlets_summary) if per_file else 1,
                "files_summary": list(_singlets_summary) if per_file else [],
            },
        )
        gating_reports.append(gate_result)

        return mask

    @staticmethod
    def _kde1d_seuil_pied_pic(
        x_data: np.ndarray,
        seuil_relatif: float = 0.05,
        finesse: float = 0.6,
        sigma_smooth: int = 10,
        n_grid: int = 1000,
        max_samples: int = 10000,
    ):
        """
        Détection du seuil CD45 par KDE 1D + pied du pic (méthode vallée robuste).

        Calcule la densité KDE sur les données log-normalisées, applique un lissage
        gaussien, puis recule depuis le maximum de densité jusqu'à tomber sous
        `seuil_relatif * max(densité)` — correspondant au pied du grand pic CD45+.

        Comme fallback si aucun pied n'est trouvé, applique Otsu 1D sur histogramme
        des données log-transformées pour maximiser la variance inter-classes.

        Args:
            x_data: Données 1D CD45 (déjà transformées logicle/arcsinh)
            seuil_relatif: Fraction du maximum de densité définissant le pied (défaut 0.05 = 5%)
            finesse: Facteur de finesse du bandwidth Silverman (0.3=très fin, 1.0=très lissé)
            sigma_smooth: Sigma du lissage gaussien supplémentaire sur la courbe KDE
            n_grid: Nombre de points de la grille d'évaluation KDE

        Returns:
            threshold: Seuil de séparation CD45- / CD45+ dans l'espace transformé
            method_used: Nom de la méthode utilisée ("kde_pied_pic" ou "otsu_log")
        """
        from scipy.stats import gaussian_kde
        from scipy.ndimage import gaussian_filter1d

        # Sous-échantillonnage strict : gaussian_kde est O(N²) — 3M pts → 3m33s
        _KDE_MAX = max(1000, int(max_samples))
        if len(x_data) > _KDE_MAX:
            x_sub = np.random.default_rng(42).choice(
                x_data, size=_KDE_MAX, replace=False
            )
        else:
            x_sub = x_data

        n = len(x_sub)
        std = np.std(x_sub)
        iqr = np.percentile(x_sub, 75) - np.percentile(x_sub, 25)
        bw = 0.9 * min(std, iqr / 1.34) * n ** (-1 / 5) * finesse
        if std < 1e-12:
            # Distribution dégénérée → fallback Otsu
            bw = 0.1
        kde = gaussian_kde(x_sub, bw_method=bw / (std if std > 1e-12 else 1.0))
        x_grid = np.linspace(x_data.min(), x_data.max(), n_grid)
        densite = gaussian_filter1d(kde(x_grid), sigma=sigma_smooth)

        seuil_abs = densite.max() * seuil_relatif
        idx_max = np.argmax(densite)

        # Recule depuis le pic max vers la gauche jusqu'à passer sous seuil_abs
        for i in range(idx_max, 0, -1):
            if densite[i] < seuil_abs:
                return float(x_grid[i]), "kde_pied_pic"

        # Fallback : Otsu 1D sur histogramme log-normalisé
        # Les données sont supposées déjà dans l'espace log (logicle/arcsinh) —
        # on normalise entre 0 et 1 pour un Otsu stable.
        x_norm = (x_data - x_data.min()) / (x_data.max() - x_data.min() + 1e-12)
        counts, bin_edges = np.histogram(x_norm, bins=512)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        total = counts.sum()
        best_var, best_t_norm = 0.0, bin_centers[len(bin_centers) // 2]
        for t in range(1, len(counts)):
            w0 = counts[:t].sum() / total
            w1 = counts[t:].sum() / total
            if w0 == 0 or w1 == 0:
                continue
            mu0 = (counts[:t] * bin_centers[:t]).sum() / (counts[:t].sum() + 1e-12)
            mu1 = (counts[t:] * bin_centers[t:]).sum() / (counts[t:].sum() + 1e-12)
            var_inter = w0 * w1 * (mu0 - mu1) ** 2
            if var_inter > best_var:
                best_var = var_inter
                best_t_norm = bin_centers[t]
        # Dénormaliser vers l'espace original
        threshold_otsu = best_t_norm * (x_data.max() - x_data.min()) + x_data.min()
        return float(threshold_otsu), "otsu_log_fallback"

    @staticmethod
    def auto_gate_cd45(
        X: np.ndarray,
        var_names: List[str],
        kde_seuil_relatif: float = 0.05,
        kde_finesse: float = 0.6,
        kde_sigma_smooth: int = 10,
        kde_n_grid: int = 1000,
        kde_max_samples: int = 10000,
        threshold_percentile: float = 5.0,
    ) -> np.ndarray:
        """
        Gate CD45+ par KDE 1D — méthode pied du pic (vallée robuste).

        Utilise la densité KDE 1D sur les données transformées (logicle/arcsinh).
        Recule depuis le maximum de densité jusqu'à trouver le pied du grand pic CD45+
        (seuil = seuil_relatif × max(densité)). Fallback automatique sur Otsu 1D
        (histogramme log-normalisé) si aucun pied n'est détecté.

        Méthode recommandée pour les données de moelle avec distribution asymétrique
        (CD45+ dominant, CD45- minoritaire), plus robuste que le GMM bimodal.

        Args:
            X: Matrice des données (transformées)
            var_names: Noms des marqueurs
            kde_seuil_relatif: Fraction du max densité pour détecter le pied (0.05 = 5%)
            kde_finesse: Facteur bandwidth Silverman (0.3=très fin → 1.0=très lissé)
            kde_sigma_smooth: Lissage gaussien supplémentaire sur la courbe KDE (sigma)
            kde_n_grid: Résolution de la grille KDE
            kde_max_samples: Nombre max d'événements utilisés pour estimer la KDE
            threshold_percentile: Percentile fallback ultime si KDE/Otsu échouent

        Returns:
            Masque booléen (True = CD45+, False = CD45-)
        """
        n_cells = X.shape[0]
        cd45_idx = PreGating.find_marker_index(
            var_names, ["CD45", "CD45-PECY5", "CD45-PC5"]
        )

        if cd45_idx is None:
            _logger.warning("[!] CD45 non trouvé pour auto-gate CD45+")
            return np.ones(n_cells, dtype=bool)

        cd45 = X[:, cd45_idx].astype(np.float64)
        valid = np.isfinite(cd45)

        if valid.sum() < 200:
            _logger.warning("[!] Pas assez de données valides pour auto-gate CD45+")
            return np.ones(n_cells, dtype=bool)

        cd45_valid = cd45[valid]

        # ── Transformation log pour le KDE ────────────────────────────────
        # Les données X sont en unités linéaires brutes (0–262144).
        # La méthode KDE pied du pic a été calibrée sur des données logicle.
        # On applique une transformation log10(1 + x) * (M / log10(1 + T))
        # identique au fallback logicle du notebook, puis on reconvertit
        # le seuil vers l'espace linéaire pour le masque.
        T, M = 262144.0, 4.5
        _scale = M / np.log10(1 + T)
        cd45_log = np.log10(1.0 + np.clip(cd45_valid, 0, None)) * _scale

        try:
            threshold_log, method_used = AutoGating._kde1d_seuil_pied_pic(
                cd45_log,
                seuil_relatif=kde_seuil_relatif,
                finesse=kde_finesse,
                sigma_smooth=kde_sigma_smooth,
                n_grid=kde_n_grid,
                max_samples=kde_max_samples,
            )
            # Reconvertir le seuil de l'espace log vers l'espace linéaire
            threshold = 10.0 ** (threshold_log / _scale) - 1.0
        except Exception as e:
            warn_msg = f"KDE CD45 échoué: {e} — fallback percentile"
            _logger.warning("   [!] %s", warn_msg)
            log_gating_event(
                "cd45",
                "kde_fallback_percentile",
                "fallback",
                {"error": str(e)},
                warn_msg,
            )
            threshold = np.nanpercentile(cd45_valid, threshold_percentile)
            threshold_log = np.log10(1.0 + max(threshold, 0)) * _scale
            method_used = "percentile_fallback"
            mask = np.zeros(n_cells, dtype=bool)
            mask[valid] = cd45_valid > threshold

            gate_result = GateResult(
                mask=mask,
                n_kept=int(mask.sum()),
                n_total=int(valid.sum()),
                method="kde_cd45_fallback_percentile",
                gate_name="G3_cd45",
                details={
                    "threshold_linear": float(threshold),
                    "threshold_log": float(threshold_log),
                    "fallback": True,
                },
                warnings=[warn_msg],
            )
            gating_reports.append(gate_result)
            return mask

        mask = np.zeros(n_cells, dtype=bool)
        mask[valid] = cd45_valid >= threshold
        n_pos = mask.sum()

        _logger.info(
            "   [KDE-CD45] Méthode: %s | seuil_relatif=%.3f | finesse=%.2f",
            method_used,
            kde_seuil_relatif,
            kde_finesse,
        )
        _logger.info(
            "   [KDE-CD45] Seuil logicle=%.4f → linéaire=%.0f  →  CD45+ : %d (%.1f%%)",
            threshold_log,
            threshold,
            n_pos,
            n_pos / valid.sum() * 100,
        )

        gate_result = GateResult(
            mask=mask,
            n_kept=int(n_pos),
            n_total=int(valid.sum()),
            method=f"kde_cd45_{method_used}",
            gate_name="G3_cd45",
            details={
                "threshold_linear": float(threshold),
                "threshold_log": float(threshold_log),
                "method_used": method_used,
                "kde_seuil_relatif": kde_seuil_relatif,
                "kde_finesse": kde_finesse,
                "kde_sigma_smooth": kde_sigma_smooth,
                "fallback": method_used != "kde_pied_pic",
            },
        )
        gating_reports.append(gate_result)
        log_gating_event(
            "cd45",
            f"kde_{method_used}",
            "success",
            {
                "threshold_linear": float(threshold),
                "threshold_log": float(threshold_log),
                "method_used": method_used,
                "n_pos": int(n_pos),
            },
        )

        return mask

    @staticmethod
    def auto_gate_cd34(
        X: np.ndarray,
        var_names: List[str],
        use_ssc_filter: bool = True,
        n_components: int = 2,
    ) -> np.ndarray:
        """
        Gate CD34+ blastes adaptatif par GMM.

        Identifie la population CD34 bright (blastes) par GMM au lieu d'un
        percentile fixe. Optionnel: combine avec SSC low (blastes = faible granularité).

        Args:
            X: Matrice des données
            var_names: Noms des marqueurs
            use_ssc_filter: Combiner avec filtre GMM SSC low
            n_components: Nombre de composantes GMM

        Returns:
            Masque booléen (True = blaste CD34+, False = autre)
        """
        n_cells = X.shape[0]
        cd34_idx = PreGating.find_marker_index(
            var_names, ["CD34", "CD34-PE", "CD34-APC", "CD34-PECY7"]
        )

        if cd34_idx is None:
            _logger.warning("[!] CD34 non trouvé pour auto-gate blastes")
            return np.ones(n_cells, dtype=bool)

        cd34 = X[:, cd34_idx].astype(np.float64)
        valid = np.isfinite(cd34)

        if valid.sum() < 200:
            _logger.warning("[!] Pas assez de données valides pour auto-gate CD34")
            return np.ones(n_cells, dtype=bool)

        # GMM pour séparer CD34- et CD34+
        try:
            gmm = AutoGating.safe_fit_gmm(
                cd34[valid].reshape(-1, 1), n_components=n_components, n_init=3
            )
        except RuntimeError as e:
            warn_msg = f"GMM CD34 échoué: {e} — conservation de toutes les cellules"
            _logger.warning("   [!] %s", warn_msg)
            log_gating_event("cd34", "gmm", "error", {"error": str(e)}, warn_msg)
            return np.ones(n_cells, dtype=bool)

        labels = gmm.predict(cd34[valid].reshape(-1, 1))
        means = gmm.means_.flatten()
        pos_component = np.argmax(means)

        mask_cd34 = np.zeros(n_cells, dtype=bool)
        mask_cd34[valid] = labels == pos_component

        n_cd34_pos = mask_cd34.sum()
        _logger.info(
            "   [Auto-GMM] CD34: μ=%s, CD34+ cluster = μ=%.0f",
            means.round(0),
            means[pos_component],
        )

        # Filtre SSC low optionnel (blastes = faible granularité)
        if use_ssc_filter:
            ssc_idx = PreGating.find_marker_index(var_names, ["SSC-A", "SSC-H", "SSC"])
            if ssc_idx is not None:
                ssc = X[:, ssc_idx].astype(np.float64)
                valid_ssc = np.isfinite(ssc)

                if valid_ssc.sum() >= 200:
                    try:
                        gmm_ssc = AutoGating.safe_fit_gmm(
                            ssc[valid_ssc].reshape(-1, 1), n_components=2, n_init=3
                        )
                    except RuntimeError as e:
                        _logger.warning(
                            "   [!] GMM SSC échoué: %s — filtre SSC ignoré", e
                        )
                        _logger.info("   [Auto-GMM] CD34+ blastes: %d", n_cd34_pos)
                        gate_result = GateResult(
                            mask=mask_cd34,
                            n_kept=int(n_cd34_pos),
                            n_total=int(valid.sum()),
                            method="gmm_cd34_no_ssc",
                            gate_name="G4_cd34",
                            details={"means": means.tolist(), "ssc_filter": False},
                            warnings=[f"GMM SSC échoué: {e}"],
                        )
                        gating_reports.append(gate_result)
                        return mask_cd34

                    labels_ssc = gmm_ssc.predict(ssc[valid_ssc].reshape(-1, 1))
                    ssc_means = gmm_ssc.means_.flatten()
                    low_ssc_component = np.argmin(ssc_means)

                    mask_ssc = np.zeros(n_cells, dtype=bool)
                    mask_ssc[valid_ssc] = labels_ssc == low_ssc_component

                    combined = mask_cd34 & mask_ssc
                    _logger.info(
                        "   [Auto-GMM] + Filtre SSC low (μ=%.0f): %d blastes purs",
                        ssc_means[low_ssc_component],
                        combined.sum(),
                    )

                    gate_result = GateResult(
                        mask=combined,
                        n_kept=int(combined.sum()),
                        n_total=int(valid.sum()),
                        method="gmm_cd34_ssc",
                        gate_name="G4_cd34",
                        details={
                            "cd34_means": means.tolist(),
                            "ssc_means": ssc_means.tolist(),
                            "ssc_filter": True,
                        },
                    )
                    gating_reports.append(gate_result)
                    log_gating_event(
                        "cd34",
                        "gmm+ssc",
                        "success",
                        {
                            "cd34_means": means.tolist(),
                            "ssc_means": ssc_means.tolist(),
                            "n_blastes": int(combined.sum()),
                        },
                    )
                    return combined

        _logger.info("   [Auto-GMM] CD34+ blastes: %d", n_cd34_pos)
        gate_result = GateResult(
            mask=mask_cd34,
            n_kept=int(n_cd34_pos),
            n_total=int(valid.sum()),
            method="gmm_cd34",
            gate_name="G4_cd34",
            details={"means": means.tolist()},
        )
        gating_reports.append(gate_result)
        return mask_cd34

    @staticmethod
    def auto_gate_cd34_cd45dim(
        X: np.ndarray,
        var_names: List[str],
        density_method: str = "KDE",
        n_components: int = 2,
    ) -> np.ndarray:
        """
        Gate CD34+ dans la population CD45dim — pré-screening MRD/régénération.

        Identifie les cellules CD34+ parmi les CD45dim par deux méthodes
        comparables (GMM et KDE) avec sélection de la méthode de référence.
        Contrairement à ``auto_gate_cd34``, cette fonction cible spécifiquement
        la population CD45dim (blastes + précurseurs) et force n_components=2.

        La méthode KDE est la référence par défaut (plus robuste sur les
        distributions asymétriques de moelle osseuse AML).

        Args:
            X: Matrice des données (n_cells, n_markers)
            var_names: Noms des marqueurs
            density_method: Méthode de référence — "KDE" (défaut) ou "GMM"
            n_components: Nombre de composantes GMM (forcé à 2)

        Returns:
            Masque booléen (True = CD34+ dans CD45dim, False = autre)
        """
        n_cells = X.shape[0]

        # ── Recherche des marqueurs ────────────────────────────────────────────
        cd34_idx = PreGating.find_marker_index(
            var_names, ["CD34", "CD34-PE", "CD34-APC", "CD34-PECY7"]
        )
        cd45_idx = PreGating.find_marker_index(
            var_names, ["CD45", "CD45-PECY5", "CD45-PC5"]
        )

        if cd34_idx is None:
            _logger.warning("[!] CD34 non trouvé pour auto-gate CD34/CD45dim")
            return np.zeros(n_cells, dtype=bool)
        if cd45_idx is None:
            _logger.warning("[!] CD45 non trouvé pour auto-gate CD34/CD45dim")
            return np.zeros(n_cells, dtype=bool)

        cd34 = X[:, cd34_idx].astype(np.float64)
        cd45 = X[:, cd45_idx].astype(np.float64)
        valid = np.isfinite(cd34) & np.isfinite(cd45)

        if valid.sum() < 200:
            _logger.warning("[!] Pas assez de données valides pour auto-gate CD34/CD45dim")
            return np.zeros(n_cells, dtype=bool)

        # ── Gate CD45dim (percentile 5–40) ────────────────────────────────────
        cd45_valid = cd45[valid]
        cd45_low = float(np.nanpercentile(cd45_valid, 5))
        cd45_high = float(np.nanpercentile(cd45_valid, 40))
        mask_cd45dim = np.zeros(n_cells, dtype=bool)
        mask_cd45dim[valid] = (cd45_valid >= cd45_low) & (cd45_valid <= cd45_high)
        n_cd45dim = int(mask_cd45dim.sum())

        if n_cd45dim < 100:
            _logger.warning(
                "[!] Seulement %d cellules CD45dim — gate CD34/CD45dim peu fiable", n_cd45dim
            )

        valid_in_dim = valid & mask_cd45dim
        if valid_in_dim.sum() < 50:
            return np.zeros(n_cells, dtype=bool)

        cd34_in_dim = cd34[valid_in_dim]

        # ── Méthode GMM (2 composantes, comme auto_gate_debris) ───────────────
        mask_gmm = np.zeros(n_cells, dtype=bool)
        try:
            gmm = AutoGating.safe_fit_gmm(
                cd34_in_dim.reshape(-1, 1),
                n_components=2,
                n_init=3,
            )
            labels_gmm = gmm.predict(cd34_in_dim.reshape(-1, 1))
            means_gmm = gmm.means_.flatten()
            pos_comp = int(np.argmax(means_gmm))
            mask_gmm[valid_in_dim] = labels_gmm == pos_comp
            _logger.info(
                "   [CD34/CD45dim-GMM] μ_neg=%.0f, μ_pos=%.0f → %d CD34+",
                means_gmm[1 - pos_comp], means_gmm[pos_comp], int(mask_gmm.sum()),
            )
        except RuntimeError as exc:
            _logger.warning("   [!] GMM CD34/CD45dim échoué: %s", exc)
            log_gating_event("cd34_cd45dim", "gmm", "error", {"error": str(exc)})

        # ── Méthode KDE (référence) — minimum local entre 2 pics CD34 ─────────
        mask_kde = np.zeros(n_cells, dtype=bool)
        threshold_kde = 0.0
        try:
            from src.analysis.prescreening import gate_cd34_kde

            # Reconstruit un masque valide sur l'ensemble (n_cells) pointant
            # uniquement les cellules CD45dim valides.
            _cd34_full = cd34.copy()
            _valid_full = np.zeros(n_cells, dtype=bool)
            _valid_full[valid_in_dim] = True

            _mask_kde_full, threshold_kde, _xg, _dens, _warns_kde = gate_cd34_kde(
                _cd34_full, _valid_full
            )
            for _w in _warns_kde:
                _logger.warning("   [!] KDE CD34/CD45dim: %s", _w)
                log_gating_event("cd34_cd45dim", "kde", "error", {"error": _w})
            mask_kde = _mask_kde_full
            _logger.info(
                "   [CD34/CD45dim-KDE] seuil_logicle=%.3f → %d CD34+",
                threshold_kde, int(mask_kde.sum()),
            )
        except Exception as exc:
            _logger.warning("   [!] KDE CD34/CD45dim échoué: %s — fallback percentile 85", exc)
            log_gating_event("cd34_cd45dim", "kde", "error", {"error": str(exc)})
            threshold_kde = float(np.nanpercentile(cd34_in_dim, 85))
            mask_kde[valid_in_dim] = cd34_in_dim >= threshold_kde

        # ── Sélection de la méthode de référence ─────────────────────────────
        if density_method.upper() == "GMM":
            mask_ref = mask_gmm
            method_label = "gmm_cd34_cd45dim"
        else:
            mask_ref = mask_kde
            method_label = "kde_cd34_cd45dim"

        n_pos = int(mask_ref.sum())
        _logger.info(
            "   [CD34/CD45dim] Méthode référence=%s | CD34+ dans CD45dim: %d/%d (%.1f%%)",
            density_method.upper(), n_pos, n_cd45dim, n_pos / max(n_cd45dim, 1) * 100,
        )

        gate_result = GateResult(
            mask=mask_ref,
            n_kept=n_pos,
            n_total=n_cd45dim,
            method=method_label,
            gate_name="G5_cd34_cd45dim",
            details={
                "density_method": density_method.upper(),
                "gmm_n_pos": int(mask_gmm.sum()),
                "kde_n_pos": int(mask_kde.sum()),
                "n_cd45dim": n_cd45dim,
                "cd45dim_range": [cd45_low, cd45_high],
            },
        )
        gating_reports.append(gate_result)
        log_gating_event(
            "cd34_cd45dim",
            method_label,
            "success",
            {
                "n_cd34_pos": n_pos,
                "n_cd45dim": n_cd45dim,
                "ratio_pct": round(n_pos / max(n_cd45dim, 1) * 100, 2),
                "density_method": density_method.upper(),
            },
        )
        return mask_ref
