# -*- coding: utf-8 -*-
"""
parameter_tabs.py — Dashboard paramétrique central PRISMA v2.0.

Classe principale : ParameterDashboard(QWidget)
  - Contient un QTabWidget avec 6 onglets thématiques
  - Chaque onglet est relié en temps réel aux dataclasses PipelineConfig
  - API simple : load(config) / save() / connect_live()

Onglets:
  1. Pré-processing   (transform, normalize, downsampling, markers)
  2. UMAP / t-SNE     (visualization, gpu)
  3. FlowSOM          (flowsom, auto_clustering, stratified_downsampling)
  4. Spectral / MRD   (pregate, mrd_config via _extra)
  5. Batch correction (data_integration / Harmony)
  6. Export           (export_mode, patho_fcs_export, batch)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Imports PRISMA internes
try:
    from flowsom_pipeline_pro.gui.widgets.config_binding import (
        CheckBinding,
        ComboBinding,
        ConfigBinder,
        DoubleSpinBinding,
        LineEditBinding,
        NClustBinding,
        SpinBinding,
    )
    from flowsom_pipeline_pro.gui.widgets.shared_forms import (
        DarkComboBox,
        FormGrid,
        ToggleSwitch,
        labeled_combo,
        labeled_dspin,
        labeled_spin,
        make_toggle,
        section_label,
    )
except ImportError:
    import sys, os
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from gui.widgets.config_binding import (  # type: ignore
        CheckBinding, ComboBinding, ConfigBinder, DoubleSpinBinding,
        LineEditBinding, NClustBinding, SpinBinding,
    )
    from gui.widgets.shared_forms import (  # type: ignore
        DarkComboBox, FormGrid, ToggleSwitch,
        labeled_combo, labeled_dspin, labeled_spin, make_toggle, section_label,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scroll_wrap(widget: QWidget) -> QScrollArea:
    """Entoure un widget d'un QScrollArea vertical."""
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sa.setWidget(widget)
    return sa


def _group(title: str, layout: QVBoxLayout | None = None) -> tuple[QGroupBox, QVBoxLayout]:
    """Crée un QGroupBox avec un QVBoxLayout interne."""
    g = QGroupBox(title)
    v = QVBoxLayout(g) if layout is None else layout
    v.setSpacing(8)
    v.setContentsMargins(10, 12, 10, 12)
    return g, v


# ─────────────────────────────────────────────────────────────────────────────
# ParameterDashboard
# ─────────────────────────────────────────────────────────────────────────────

class ParameterDashboard(QWidget):
    """
    Dashboard paramétrique central relié en temps réel à PipelineConfig.

    Signaux:
        config_changed() : émis à chaque modification d'un paramètre

    Usage:
        dashboard = ParameterDashboard(config)
        dashboard.connect_live()       # auto-save temps réel
        layout.addWidget(dashboard)

        # Recharger depuis une config YAML
        dashboard.load(new_config)

        # Lire la config courante (déjà synchronisée si connect_live)
        config = dashboard.config
    """

    config_changed = pyqtSignal()

    def __init__(self, config: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._binders: List[ConfigBinder] = []
        self._mrd_raw: Dict[str, Any] = {}  # stocke _extra["mrd_parameters"]

        self._tabs = QTabWidget()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._tabs)

        self._build_tab_preprocessing()
        self._build_tab_umap()
        self._build_tab_flowsom()
        self._build_tab_spectral_mrd()
        self._build_tab_batch_correction()
        self._build_tab_export()

        self.load(config)

    # ── API publique ──────────────────────────────────────────────────────────

    @property
    def config(self) -> Any:
        return self._config

    def load(self, config: Any) -> None:
        """Charge config dans tous les widgets (config → UI)."""
        self._config = config
        for binder in self._binders:
            binder.set_config(config)
        self._load_mrd_extra(config)

    def save(self) -> None:
        """Écrit les widgets dans la config (UI → config)."""
        for binder in self._binders:
            binder.save()
        self._save_mrd_extra()

    def connect_live(self) -> None:
        """Connecte tous les signaux pour mise à jour immédiate."""
        for binder in self._binders:
            binder.connect_live()
        # Réémet config_changed sur chaque changement
        for binder in self._binders:
            for b in binder._bindings:
                b.connect(self.config_changed.emit)

    # ─────────────────────────────────────────────────────────────────────────
    # Onglet 1 — Pré-processing
    # ─────────────────────────────────────────────────────────────────────────

    def _build_tab_preprocessing(self) -> None:
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setSpacing(16)
        vbox.setContentsMargins(20, 16, 20, 16)

        bindings: list = []

        # ── Transformation ────────────────────────────────────────────────────
        g_tr, v_tr = _group("Transformation")
        fg_tr = FormGrid()
        self.combo_transform = fg_tr.add_combo(
            "Méthode :", ["logicle", "arcsinh", "log10", "none"], "logicle", "combo_transform",
            "Transformation des intensités fluorescentes avant clustering."
        )
        self.spin_cofactor = fg_tr.add_dspin(
            "Cofacteur (arcsinh) :", 1.0, 500.0, 5.0, "spin_cofactor",
            step=1.0, decimals=1,
            tooltip="Cofacteur arcsinh. Ignoré si méthode ≠ arcsinh."
        )
        self.combo_normalize = fg_tr.add_combo(
            "Normalisation :", ["zscore", "minmax", "none"], "zscore", "combo_normalize",
            "Normalisation post-transformation."
        )
        v_tr.addWidget(fg_tr)
        vbox.addWidget(g_tr)

        bindings += [
            ComboBinding(self.combo_transform, "transform", "method"),
            DoubleSpinBinding(self.spin_cofactor, "transform", "cofactor"),
            ComboBinding(self.combo_normalize, "normalize", "method"),
        ]

        # ── Downsampling ──────────────────────────────────────────────────────
        g_ds, v_ds = _group("Downsampling")
        fg_ds = FormGrid()
        self.chk_downsampling = fg_ds.add_toggle(
            "Activer le downsampling", True, "chk_downsampling"
        )
        self.spin_max_cells = fg_ds.add_spin(
            "Max cellules / fichier :", 1000, 5_000_000, 50_000, "spin_max_cells",
            tooltip="Limite par fichier FCS avant concaténation.",
            suffix=" cel."
        )
        self.spin_max_cells_total = fg_ds.add_spin(
            "Max cellules total :", 10_000, 20_000_000, 1_000_000, "spin_max_cells_total",
            tooltip="Limite sur l'ensemble du pool d'entraînement.",
            suffix=" cel."
        )
        v_ds.addWidget(fg_ds)
        vbox.addWidget(g_ds)

        bindings += [
            CheckBinding(self.chk_downsampling, "downsampling", "enabled"),
            SpinBinding(self.spin_max_cells, "downsampling", "max_cells_per_file"),
            SpinBinding(self.spin_max_cells_total, "downsampling", "max_cells_total"),
        ]

        # ── Marqueurs ─────────────────────────────────────────────────────────
        g_mk, v_mk = _group("Marqueurs & Scatter")
        fg_mk = FormGrid()
        self.chk_exclude_scatter = fg_mk.add_toggle(
            "Exclure scatter (FSC/SSC)", True, "chk_exclude_scatter",
            "Exclut les colonnes FSC/SSC du clustering. Recommandé."
        )
        self.chk_keep_area_only = fg_mk.add_toggle(
            "Garder -A uniquement (exclure -H)", True, "chk_keep_area_only",
            "Supprime les colonnes Height quand Area existe. Réduit la colinéarité."
        )
        self.edit_exclude_cols = fg_mk.add_lineedit(
            "Colonnes supplémentaires à exclure :",
            "ex: Time, Width, Event_length",
            "edit_exclude_cols",
            "Colonnes à exclure du clustering, séparées par des virgules."
        )
        v_mk.addWidget(fg_mk)
        vbox.addWidget(g_mk)

        bindings += [
            CheckBinding(self.chk_exclude_scatter, "markers", "exclude_scatter"),
            CheckBinding(self.chk_keep_area_only, "markers", "keep_area_only"),
            LineEditBinding(self.edit_exclude_cols, "markers", "exclude_additional", splitter=","),
        ]

        vbox.addStretch()
        binder = ConfigBinder(self._config, bindings)
        self._binders.append(binder)
        self._tabs.addTab(_scroll_wrap(container), "Pré-processing")

    # ─────────────────────────────────────────────────────────────────────────
    # Onglet 2 — UMAP / t-SNE
    # ─────────────────────────────────────────────────────────────────────────

    def _build_tab_umap(self) -> None:
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setSpacing(16)
        vbox.setContentsMargins(20, 16, 20, 16)

        bindings: list = []

        # ── Visualisation ─────────────────────────────────────────────────────
        g_viz, v_viz = _group("Réduction dimensionnelle & Visualisation")
        fg_viz = FormGrid()
        self.chk_umap = fg_viz.add_toggle(
            "Calculer UMAP après clustering", False, "chk_umap",
            "Lance UMAP pour la visualisation 2D post-FlowSOM. Coûteux en temps."
        )
        self.chk_gpu = fg_viz.add_toggle(
            "GPU (CUDA) — cuML UMAP", True, "chk_gpu",
            "Utilise cuML (RAPIDS) si disponible. Accélération x10–x100 sur UMAP."
        )
        fg_viz.add_section("── Paramètres export figures ──")
        self.combo_plot_format = fg_viz.add_combo(
            "Format figures :", ["png", "pdf", "svg"], "png", "combo_plot_format",
            "Format de sauvegarde des figures matplotlib."
        )
        self.spin_dpi = fg_viz.add_spin(
            "DPI :", 72, 600, 150, "spin_dpi",
            tooltip="Résolution des figures exportées."
        )
        self.chk_save_plots = fg_viz.add_toggle(
            "Sauvegarder les figures", True, "chk_save_plots"
        )
        v_viz.addWidget(fg_viz)
        vbox.addWidget(g_viz)

        bindings += [
            CheckBinding(self.chk_umap, "visualization", "umap_enabled"),
            CheckBinding(self.chk_gpu, "gpu", "enabled"),
            ComboBinding(self.combo_plot_format, "visualization", "plot_format"),
            SpinBinding(self.spin_dpi, "visualization", "dpi"),
            CheckBinding(self.chk_save_plots, "visualization", "save_plots"),
        ]

        # ── Note t-SNE ────────────────────────────────────────────────────────
        g_tsne, v_tsne = _group("t-SNE (information)")
        note = QLabel(
            "t-SNE est calculé automatiquement sur les données post-FlowSOM si UMAP est désactivé.\n"
            "Les paramètres avancés t-SNE (perplexity, n_iter) sont configurables dans le YAML."
        )
        note.setWordWrap(True)
        note.setObjectName("subtitleLabel")
        v_tsne.addWidget(note)
        vbox.addWidget(g_tsne)

        vbox.addStretch()
        binder = ConfigBinder(self._config, bindings)
        self._binders.append(binder)
        self._tabs.addTab(_scroll_wrap(container), "UMAP / t-SNE")

    # ─────────────────────────────────────────────────────────────────────────
    # Onglet 3 — FlowSOM
    # ─────────────────────────────────────────────────────────────────────────

    def _build_tab_flowsom(self) -> None:
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setSpacing(16)
        vbox.setContentsMargins(20, 16, 20, 16)

        bindings: list = []

        # ── Grille SOM ────────────────────────────────────────────────────────
        g_som, v_som = _group("Grille SOM")
        fg_som = FormGrid()
        self.spin_xdim = fg_som.add_spin(
            "Dimension X (xdim) :", 3, 50, 10, "spin_xdim",
            tooltip="Largeur de la grille auto-organisatrice. Défaut : 10."
        )
        self.spin_ydim = fg_som.add_spin(
            "Dimension Y (ydim) :", 3, 50, 10, "spin_ydim",
            tooltip="Hauteur de la grille auto-organisatrice. Défaut : 10."
        )
        self.spin_metaclusters = fg_som.add_spin(
            "Métaclusters :", 2, 50, 8, "spin_metaclusters",
            tooltip="Nombre de métaclusters hiérarchiques. Défaut : 8."
        )
        self.spin_seed = fg_som.add_spin(
            "Seed :", 0, 99999, 42, "spin_seed",
            tooltip="Graine pour la reproductibilité."
        )
        v_som.addWidget(fg_som)
        vbox.addWidget(g_som)

        bindings += [
            SpinBinding(self.spin_xdim, "flowsom", "xdim"),
            SpinBinding(self.spin_ydim, "flowsom", "ydim"),
            SpinBinding(self.spin_metaclusters, "flowsom", "n_metaclusters"),
            SpinBinding(self.spin_seed, "flowsom", "seed"),
        ]

        # ── Apprentissage ─────────────────────────────────────────────────────
        g_lr, v_lr = _group("Paramètres d'apprentissage")
        fg_lr = FormGrid()
        self.spin_lr = fg_lr.add_dspin(
            "Learning rate :", 0.001, 1.0, 0.05, "spin_lr",
            step=0.01, decimals=3,
            tooltip="Taux d'apprentissage SOM. Défaut : 0.05."
        )
        self.spin_sigma = fg_lr.add_dspin(
            "Sigma voisinage :", 0.1, 10.0, 1.5, "spin_sigma",
            step=0.1, decimals=1,
            tooltip="Rayon de voisinage gaussien. Défaut : 1.5."
        )
        v_lr.addWidget(fg_lr)
        vbox.addWidget(g_lr)

        bindings += [
            DoubleSpinBinding(self.spin_lr, "flowsom", "learning_rate"),
            DoubleSpinBinding(self.spin_sigma, "flowsom", "sigma"),
        ]

        # ── Auto-clustering ───────────────────────────────────────────────────
        g_ac, v_ac = _group("Auto-sélection du nombre de clusters (bootstrap)")
        fg_ac = FormGrid()
        self.chk_auto_clustering = fg_ac.add_toggle(
            "Activer l'auto-sélection (bootstrap)", False, "chk_auto_clustering",
            "Détermine le nombre optimal de métaclusters par rééchantillonnage."
        )
        self.spin_min_clusters = fg_ac.add_spin(
            "Min clusters :", 2, 50, 4, "spin_min_clusters"
        )
        self.spin_max_clusters = fg_ac.add_spin(
            "Max clusters :", 3, 100, 20, "spin_max_clusters"
        )
        self.spin_n_bootstrap = fg_ac.add_spin(
            "N bootstrap :", 2, 100, 10, "spin_n_bootstrap",
            tooltip="Nombre de rééchantillonnages pour la stabilité."
        )
        self.spin_sample_bootstrap = fg_ac.add_spin(
            "Taille échantillon bootstrap :", 1000, 500_000, 20_000, "spin_sample_bootstrap",
            suffix=" cel."
        )
        v_ac.addWidget(fg_ac)
        vbox.addWidget(g_ac)

        bindings += [
            CheckBinding(self.chk_auto_clustering, "auto_clustering", "enabled"),
            SpinBinding(self.spin_min_clusters, "auto_clustering", "min_clusters"),
            SpinBinding(self.spin_max_clusters, "auto_clustering", "max_clusters"),
            SpinBinding(self.spin_n_bootstrap, "auto_clustering", "n_bootstrap"),
            SpinBinding(self.spin_sample_bootstrap, "auto_clustering", "sample_size_bootstrap"),
        ]

        # ── Déséquilibre Maîtrisé ─────────────────────────────────────────────
        g_sd, v_sd = _group("Déséquilibre Maîtrisé (rééquilibrage sain/patho)")
        fg_sd = FormGrid()
        self.chk_balance_conditions = fg_sd.add_toggle(
            "Rééquilibrage actif", False, "chk_balance_conditions",
            "Force un ratio n_sain/n_patho avant entraînement FlowSOM.\n"
            "Améliore la visibilité des clusters rares (<1%)."
        )
        self.spin_imbalance_ratio = fg_sd.add_dspin(
            "Ratio sain / patho :", 0.5, 10.0, 2.0, "spin_imbalance_ratio",
            step=0.5, decimals=1,
            tooltip="1.0 = équilibré. 2.0 = 2 sains pour 1 blaste."
        )
        self.chk_allow_oversampling = fg_sd.add_toggle(
            "Oversampling NBM si quota non atteint", False, "chk_allow_oversampling",
            "Rééchantillonnage avec remplacement pour atteindre le ratio. Introduit des doublons."
        )
        v_sd.addWidget(fg_sd)
        vbox.addWidget(g_sd)

        bindings += [
            CheckBinding(self.chk_balance_conditions, "stratified_downsampling", "balance_conditions"),
            DoubleSpinBinding(self.spin_imbalance_ratio, "stratified_downsampling", "imbalance_ratio"),
            CheckBinding(self.chk_allow_oversampling, "stratified_downsampling", "allow_oversampling"),
        ]

        vbox.addStretch()
        binder = ConfigBinder(self._config, bindings)
        self._binders.append(binder)
        self._tabs.addTab(_scroll_wrap(container), "FlowSOM")

    # ─────────────────────────────────────────────────────────────────────────
    # Onglet 4 — Spectral / MRD (pré-gating + paramètres MRD)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_tab_spectral_mrd(self) -> None:
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setSpacing(16)
        vbox.setContentsMargins(20, 16, 20, 16)

        bindings: list = []

        # ── Pré-gating ────────────────────────────────────────────────────────
        g_pg, v_pg = _group("Pré-gating automatique")
        fg_pg = FormGrid()
        self.chk_pregate = fg_pg.add_toggle(
            "Activer le pré-gating", True, "chk_pregate"
        )
        self.combo_gate_mode = fg_pg.add_combo(
            "Mode :", ["auto", "manual"], "auto", "combo_gate_mode",
            "auto : seuils calculés automatiquement (GMM/KDE)\nmanual : seuils fixes"
        )
        self.chk_viable = fg_pg.add_toggle("Gate Débris (FSC/SSC)", True, "chk_viable")
        self.chk_singlets = fg_pg.add_toggle("Gate Doublets (FSC-H/FSC-A)", True, "chk_singlets")
        self.chk_cd45 = fg_pg.add_toggle("Gate CD45 dim", False, "chk_cd45")
        self.chk_cd34 = fg_pg.add_toggle("Gate CD34+ blastes", False, "chk_cd34")
        self.chk_mode_blastes = fg_pg.add_toggle(
            "Gating CD45 asymétrique (patho seulement)", True, "chk_mode_blastes",
            "Applique le gating CD45 uniquement sur l'échantillon pathologique."
        )
        self.combo_cd45_autogating_mode = fg_pg.add_combo(
            "Dénominateur MRD :",
            ["none", "cd45", "cd45_dim"],
            "none", "combo_cd45_autogating_mode",
            "none     → MRD % = blastes / toutes cellules patho\n"
            "cd45     → MRD % = blastes / cellules patho CD45+\n"
            "cd45_dim → MRD % = blastes / cellules patho CD45+ (blastes dim inclus)"
        )
        v_pg.addWidget(fg_pg)
        vbox.addWidget(g_pg)

        bindings += [
            CheckBinding(self.chk_pregate, "pregate", "apply"),
            ComboBinding(self.combo_gate_mode, "pregate", "mode"),
            CheckBinding(self.chk_viable, "pregate", "viable"),
            CheckBinding(self.chk_singlets, "pregate", "singlets"),
            CheckBinding(self.chk_cd45, "pregate", "cd45"),
            CheckBinding(self.chk_cd34, "pregate", "cd34"),
            CheckBinding(self.chk_mode_blastes, "pregate", "mode_blastes_vs_normal"),
            ComboBinding(self.combo_cd45_autogating_mode, "pregate", "cd45_autogating_mode"),
        ]

        # ── Méthode densité ───────────────────────────────────────────────────
        g_dens, v_dens = _group("Méthode d'estimation de densité (tri initial)")
        fg_dens = FormGrid()
        self.combo_density_method = fg_dens.add_combo(
            "Méthode (viable) :", ["GMM", "KDE"], "GMM", "combo_density_method",
            "GMM (Gaussian Mixture Model) : robuste, recommandé\n"
            "KDE (Kernel Density Estimation) : léger, bon pour CD45"
        )
        self.spin_gmm_components = fg_dens.add_spin(
            "Composantes GMM :", 1, 10, 3, "spin_gmm_components",
            tooltip="Nombre de composantes gaussiennes (débris/transitoire/viables)."
        )
        self.combo_gmm_cov = fg_dens.add_combo(
            "Type covariance GMM :",
            ["full", "tied", "diag", "spherical"],
            "full", "combo_gmm_cov",
            "full : matrice complète (recommandé)\ntied : partagée\ndiag/spherical : simplifiée"
        )
        fg_dens.add_section("── Paramètres KDE CD45 ──")
        self.spin_kde_finesse = fg_dens.add_dspin(
            "Finesse bandwidth :", 0.1, 2.0, 0.6, "spin_kde_finesse",
            step=0.05, decimals=2,
            tooltip="< 1 = plus fin, > 1 = plus lissé. Défaut : 0.6"
        )
        self.spin_kde_sigma = fg_dens.add_spin(
            "Sigma lissage KDE :", 1, 50, 10, "spin_kde_sigma",
            tooltip="Lissage gaussien post-KDE (sigma en points de grille)."
        )
        self.spin_kde_seuil = fg_dens.add_dspin(
            "Seuil relatif CD45 :", 0.01, 0.5, 0.05, "spin_kde_seuil",
            step=0.01, decimals=3,
            tooltip="Fraction du pic max pour détecter le pied du pic CD45."
        )
        v_dens.addWidget(fg_dens)
        vbox.addWidget(g_dens)

        bindings += [
            ComboBinding(self.combo_density_method, "pregate", "density_method"),
            SpinBinding(self.spin_gmm_components, "pregate", "gmm_n_components_debris"),
            ComboBinding(self.combo_gmm_cov, "pregate", "gmm_covariance_type"),
            DoubleSpinBinding(self.spin_kde_finesse, "pregate", "kde_cd45_finesse"),
            SpinBinding(self.spin_kde_sigma, "pregate", "kde_cd45_sigma_smooth"),
            DoubleSpinBinding(self.spin_kde_seuil, "pregate", "kde_cd45_seuil_relatif"),
        ]

        # ── Paramètres MRD (stockés dans _extra/mrd_config) ──────────────────
        g_mrd, v_mrd = _group("Paramètres MRD (mrd_config.yaml)")
        fg_mrd = FormGrid()
        self.combo_mrd_method = fg_mrd.add_combo(
            "Méthode MRD :", ["all", "flo", "jf", "eln"], "all", "combo_mrd_method"
        )
        fg_mrd.add_section("── ELN ──")
        self.spin_eln_min_events = fg_mrd.add_spin(
            "Min events / nœud (LOQ) :", 1, 500, 50, "spin_eln_min_events",
            tooltip="ELN 2022 : minimum 17 événements par nœud."
        )
        self.spin_eln_positivity = fg_mrd.add_dspin(
            "Seuil positivité ELN (%) :", 0.01, 10.0, 0.1, "spin_eln_positivity",
            step=0.05, decimals=2
        )
        fg_mrd.add_section("── Méthode Flo ──")
        self.spin_flo_multiplier = fg_mrd.add_dspin(
            "Multiplicateur moelle normale :", 0.5, 20.0, 2.0, "spin_flo_multiplier",
            step=0.5, decimals=1
        )
        fg_mrd.add_section("── Méthode JF ──")
        self.spin_jf_max_normal = fg_mrd.add_dspin(
            "Max % moelle normale :", 0.01, 10.0, 0.1, "spin_jf_max_normal",
            step=0.05, decimals=2
        )
        self.spin_jf_min_patho = fg_mrd.add_dspin(
            "Min % cellules patho :", 0.1, 100.0, 10.0, "spin_jf_min_patho",
            step=1.0, decimals=1
        )
        fg_mrd.add_section("── Filtre Phénotypique ──")
        self.chk_blast_filter = fg_mrd.add_toggle(
            "Filtre hybride ELN 2022 (BLAST_HIGH/MODERATE)", False, "chk_blast_filter",
            "Porte biologique : nœud validé seulement si blast_score ELN 2022 activé."
        )
        v_mrd.addWidget(fg_mrd)
        vbox.addWidget(g_mrd)

        # MRD widgets — gérés séparément via _load_mrd_extra / _save_mrd_extra
        # (pas de ConfigBinder car ils pointent vers _extra dict, pas des dataclasses)

        vbox.addStretch()
        binder = ConfigBinder(self._config, bindings)
        self._binders.append(binder)
        self._tabs.addTab(_scroll_wrap(container), "Spectral / MRD")

    # ─────────────────────────────────────────────────────────────────────────
    # Onglet 5 — Batch correction (Harmony)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_tab_batch_correction(self) -> None:
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setSpacing(16)
        vbox.setContentsMargins(20, 16, 20, 16)

        bindings: list = []

        # ── Harmony ───────────────────────────────────────────────────────────
        g_h, v_h = _group("Harmony (correction d'effet batch inter-fichiers)")
        fg_h = FormGrid()
        self.chk_harmony = fg_h.add_toggle(
            "Activer Harmony (harmonypy)", True, "chk_harmony",
            "Aligne les espaces d'expression entre fichiers NBM de sessions différentes."
        )
        self.edit_harmony_markers = fg_h.add_lineedit(
            "Marqueurs à aligner (vide = tous) :",
            "ex: FSC-A, SSC-A  (vide = tous)",
            "edit_harmony_markers",
            "Séparés par virgules. Vide = tous les marqueurs du clustering."
        )
        fg_h.add_section("── Hyperparamètres Harmony ──")
        self.spin_harmony_sigma = fg_h.add_dspin(
            "Sigma :", 0.001, 1.0, 0.05, "spin_harmony_sigma",
            step=0.01, decimals=3,
            tooltip="Largeur de distribution. Petit = correction agressive. Défaut : 0.05."
        )
        self.spin_harmony_nclust = fg_h.add_spin(
            "nclust (0 = auto) :", 0, 200, 30, "spin_harmony_nclust",
            tooltip="0 = N/30 (lent sur grands datasets). Défaut : 30."
        )
        self.spin_harmony_max_iter = fg_h.add_spin(
            "Max itérations :", 1, 100, 10, "spin_harmony_max_iter",
            tooltip="Nombre max d'itérations Harmony. Défaut : 10."
        )
        self.spin_harmony_block = fg_h.add_dspin(
            "Block size :", 0.01, 1.0, 0.20, "spin_harmony_block",
            step=0.05, decimals=2,
            tooltip="Fraction de cellules par bloc. 0.20 = 5 blocs. Défaut : 0.20."
        )
        v_h.addWidget(fg_h)
        vbox.addWidget(g_h)

        bindings += [
            CheckBinding(self.chk_harmony, "data_integration", "enabled"),
            LineEditBinding(
                self.edit_harmony_markers,
                "data_integration", "harmony_params", "markers_to_align",
                splitter=","
            ),
            DoubleSpinBinding(self.spin_harmony_sigma, "data_integration", "harmony_params", "sigma"),
            NClustBinding(self.spin_harmony_nclust, "data_integration", "harmony_params", "nclust"),
            SpinBinding(self.spin_harmony_max_iter, "data_integration", "harmony_params", "max_iter"),
            DoubleSpinBinding(self.spin_harmony_block, "data_integration", "harmony_params", "block_size"),
        ]

        # ── Mapping populations ───────────────────────────────────────────────
        g_pm, v_pm = _group("Mapping populations (Ref MFI — ELN 2022)")
        fg_pm = FormGrid()
        self.chk_pop_mapping = fg_pm.add_toggle(
            "Activer le mapping populationnel", False, "chk_pop_mapping",
            "Associe chaque nœud SOM à une population via distance cosine sur MFI de référence."
        )
        self.combo_mapping_method = fg_pm.add_combo(
            "Méthode mapping :",
            ["cosine_prior", "cosine", "euclidean"],
            "cosine_prior", "combo_mapping_method",
            "cosine_prior : recommandé ELN 2022 (prior log10_cubed)"
        )
        self.spin_distance_percentile = fg_pm.add_spin(
            "Percentile distance :", 10, 99, 60, "spin_distance_percentile",
            tooltip="Seuil percentile pour classifier un nœud comme 'Unknown'."
        )
        v_pm.addWidget(fg_pm)
        vbox.addWidget(g_pm)

        bindings += [
            CheckBinding(self.chk_pop_mapping, "population_mapping", "enabled"),
            ComboBinding(self.combo_mapping_method, "population_mapping", "mapping_method"),
            SpinBinding(self.spin_distance_percentile, "population_mapping", "distance_percentile"),
        ]

        # ── Mode comparaison ──────────────────────────────────────────────────
        g_cmp, v_cmp = _group("Mode comparaison")
        fg_cmp = FormGrid()
        self.chk_compare = fg_cmp.add_toggle(
            "Mode comparaison Sain vs Patho", True, "chk_compare",
            "Entraîne FlowSOM sur le pool sain + patho et compare les fréquences."
        )
        v_cmp.addWidget(fg_cmp)
        vbox.addWidget(g_cmp)

        bindings += [
            CheckBinding(self.chk_compare, "analysis", "compare_mode"),
        ]

        vbox.addStretch()
        binder = ConfigBinder(self._config, bindings)
        self._binders.append(binder)
        self._tabs.addTab(_scroll_wrap(container), "Batch correction")

    # ─────────────────────────────────────────────────────────────────────────
    # Onglet 6 — Export
    # ─────────────────────────────────────────────────────────────────────────

    def _build_tab_export(self) -> None:
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setSpacing(16)
        vbox.setContentsMargins(20, 16, 20, 16)

        bindings: list = []

        # ── Mode export ───────────────────────────────────────────────────────
        g_em, v_em = _group("Mode d'export des résultats")
        fg_em = FormGrid()
        self.combo_export_mode = fg_em.add_combo(
            "Mode export :", ["standard", "compact"], "standard", "combo_export_mode",
            "standard : tous les fichiers (FCS, CSV, JSON, plots, rapports)\n"
            "compact  : essentiel uniquement (PDF, HTML, MRD JSON, FCS patho)"
        )
        self.chk_export_csv = fg_em.add_toggle(
            "Exporter les CSV", True, "chk_export_csv",
            "Génère les CSV complets (MFI, statistiques, clustering)."
        )
        self.chk_export_per_file_csv = fg_em.add_toggle(
            "CSV par fichier FCS source", True, "chk_export_per_file_csv",
            "Un CSV par fichier FCS en plus du CSV global."
        )
        v_em.addWidget(fg_em)
        vbox.addWidget(g_em)

        bindings += [
            ComboBinding(self.combo_export_mode, "export_mode", "mode"),
            CheckBinding(self.chk_export_csv, "export_mode", "export_csv"),
            CheckBinding(self.chk_export_per_file_csv, "export_mode", "export_per_file_csv"),
        ]

        # ── Export FCS patho ──────────────────────────────────────────────────
        g_fcs, v_fcs = _group("Export FCS pathologique + Is_MRD")
        fg_fcs = FormGrid()
        self.chk_patho_fcs_export = fg_fcs.add_toggle(
            "Générer FCS patho avec colonne Is_MRD", False, "chk_patho_fcs_export",
            "Exporte un FCS restreint aux cellules pathologiques avec Is_MRD=0/1."
        )
        self.combo_mrd_fcs_method = fg_fcs.add_combo(
            "Méthode Is_MRD pour export FCS :", ["flo", "jf"], "flo", "combo_mrd_fcs_method",
            "Méthode utilisée pour générer la colonne Is_MRD dans le FCS patho."
        )
        v_fcs.addWidget(fg_fcs)
        vbox.addWidget(g_fcs)

        bindings += [
            CheckBinding(self.chk_patho_fcs_export, "patho_fcs_export", "enabled"),
            ComboBinding(self.combo_mrd_fcs_method, "patho_fcs_export", "mrd_method"),
        ]

        # ── Mode Batch ────────────────────────────────────────────────────────
        g_bt, v_bt = _group("Mode Batch")
        fg_bt = FormGrid()
        self.chk_batch = fg_bt.add_toggle(
            "Mode Batch (tous les fichiers patho en séquence)", False, "chk_batch",
            "Traite tous les FCS du dossier patho un par un dans des runs séparés."
        )
        v_bt.addWidget(fg_bt)
        vbox.addWidget(g_bt)

        bindings += [
            CheckBinding(self.chk_batch, "batch", "enabled"),
        ]

        # ── Monitoring performance ─────────────────────────────────────────────
        g_perf, v_perf = _group("Monitoring de performance système")
        fg_perf = FormGrid()
        self.chk_perf_monitoring = fg_perf.add_toggle(
            "Activer le monitoring (CPU/GPU/RAM)", False, "chk_perf_monitoring",
            "Collecte les métriques système pendant le pipeline."
        )
        self.spin_perf_interval = fg_perf.add_dspin(
            "Intervalle de collecte (s) :", 0.1, 60.0, 1.0, "spin_perf_interval",
            step=0.5, decimals=1
        )
        v_perf.addWidget(fg_perf)
        vbox.addWidget(g_perf)

        bindings += [
            CheckBinding(self.chk_perf_monitoring, "performance_monitoring", "enabled"),
            DoubleSpinBinding(self.spin_perf_interval, "performance_monitoring", "interval_seconds"),
        ]

        vbox.addStretch()
        binder = ConfigBinder(self._config, bindings)
        self._binders.append(binder)
        self._tabs.addTab(_scroll_wrap(container), "Export")

    # ─────────────────────────────────────────────────────────────────────────
    # Gestion MRD extra (stocké dans _extra dict, pas dataclass)
    # ─────────────────────────────────────────────────────────────────────────

    def _load_mrd_extra(self, cfg: Any) -> None:
        """Charge les paramètres MRD depuis cfg._extra (ou mrd_raw si présent)."""
        mrd_raw = getattr(cfg, "_extra", {}).get("mrd_raw", {})
        params = mrd_raw.get("mrd_parameters", {})
        if not params:
            return

        method = params.get("method", "all")
        idx = self.combo_mrd_method.findText(method)
        if idx >= 0:
            self.combo_mrd_method.setCurrentIndex(idx)

        eln = params.get("eln_standards", {})
        self.spin_eln_min_events.setValue(int(eln.get("min_cluster_events", 50)))
        self.spin_eln_positivity.setValue(float(eln.get("clinical_positivity_pct", 0.1)))

        flo = params.get("method_flo", {})
        self.spin_flo_multiplier.setValue(float(flo.get("normal_marrow_multiplier", 2.0)))

        jf = params.get("method_jf", {})
        self.spin_jf_max_normal.setValue(float(jf.get("max_normal_marrow_pct", 0.1)))
        self.spin_jf_min_patho.setValue(float(jf.get("min_patho_cells_pct", 10.0)))

        bpf = params.get("blast_phenotype_filter", {})
        self.chk_blast_filter.setChecked(bool(bpf.get("enabled", False)))

    def _save_mrd_extra(self) -> None:
        """Écrit les paramètres MRD widgets dans cfg._extra["mrd_raw"]."""
        if self._config is None:
            return
        if not hasattr(self._config, "_extra"):
            return
        mrd_raw = self._config._extra.setdefault("mrd_raw", {})
        params = mrd_raw.setdefault("mrd_parameters", {})

        params["method"] = self.combo_mrd_method.currentText()
        params.setdefault("eln_standards", {})["min_cluster_events"] = (
            self.spin_eln_min_events.value()
        )
        params["eln_standards"]["clinical_positivity_pct"] = self.spin_eln_positivity.value()
        params.setdefault("method_flo", {})["normal_marrow_multiplier"] = (
            self.spin_flo_multiplier.value()
        )
        params.setdefault("method_jf", {})["max_normal_marrow_pct"] = (
            self.spin_jf_max_normal.value()
        )
        params["method_jf"]["min_patho_cells_pct"] = self.spin_jf_min_patho.value()
        params.setdefault("blast_phenotype_filter", {})["enabled"] = (
            self.chk_blast_filter.isChecked()
        )
