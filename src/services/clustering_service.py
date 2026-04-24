"""
clustering_service.py — Orchestration du clustering FlowSOM.

Chaîne de clustering complète:
  1. Sélection et filtrage des marqueurs (exclusion FSC/SSC si configuré)
  2. Empilement des échantillons en une matrice unique
  3. Sous-échantillonnage pour le clustering (ratio par condition si COMPARE_MODE)
  4. FlowSOM (GPU → CPU fallback)
  5. Auto-clustering (optionnel): recherche du k optimal
  6. Calcul de la matrice MFI médiane par métacluster
  7. Propagation des labels à toutes les cellules

Retourne un PipelineResult enrichi avec les métadonnées de clustering.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import anndata as ad

    _ANNDATA_AVAILABLE = True
except ImportError:
    _ANNDATA_AVAILABLE = False

try:
    import harmonypy

    _HARMONY_AVAILABLE = True
except (ImportError, OSError) as _harmony_import_error:
    # Capture aussi les erreurs OSError (DLL manquante, torch CUDA échoué, etc.)
    _HARMONY_AVAILABLE = False
    _logger_import = __import__("logging").getLogger("services.clustering")
    _error_msg = str(_harmony_import_error)
    if "torch" in _error_msg.lower() or "dll" in _error_msg.lower() or "1114" in _error_msg:
        _logger_import.warning(
            "torch/harmonypy non fonctionnel (DLL manquante, CUDA non disponible, etc.) — "
            "intégration Harmony désactivée. "
            "Erreur: %s. "
            "Vous pouvez réinstaller avec : pip install torch --index-url https://download.pytorch.org/whl/cpu",
            _error_msg,
        )
    else:
        _logger_import.warning(
            "harmonypy non installé ou incompatible — intégration Harmony désactivée. "
            "Erreur: %s. "
            "Installez-le avec : pip install harmonypy",
            _error_msg,
        )

from config.pipeline_config import PipelineConfig
from config.constants import SCATTER_PATTERNS
from src.models.sample import FlowSample
from src.models.pipeline_result import (
    PipelineResult,
    ClusteringMetrics,
)
from src.core.clustering import FlowSOMClusterer
from src.core.metaclustering import find_optimal_clusters
from src.utils.validators import (
    check_no_fsc_ssc_in_analysis_markers,
    check_nan,
)
from src.utils.logger import get_logger
from src.utils.class_balancer import equilibrer_pool_flowsom

_logger = get_logger("services.clustering")

# ── Cache diagnostique Harmony ────────────────────────────────────────────────
# Stocke les figures générées lors du dernier run_clustering pour injection
# dans les rapports HTML/PDF sans modifier la signature de run_clustering.
_HARMONY_DIAG: dict = {}  # {"plotly": go.Figure | None, "mpl": mpl.Figure | None}


def _build_harmony_diag(
    X_before: "np.ndarray",
    X_after: "np.ndarray",
    conditions: "np.ndarray",
    n_sample: int = 5_000,
) -> None:
    """
    Génère une visualisation PCA avant/après correction Harmony.

    Utilise un sous-échantillon aléatoire (max *n_sample* cellules) et une PCA
    à 2 composantes pour rester ultra-rapide (~0.2 s sur 50 k cellules).

    Résultats stockés dans _HARMONY_DIAG["plotly"] et _HARMONY_DIAG["mpl"].
    En cas d'erreur la fonction est silencieuse (non bloquante).
    """
    global _HARMONY_DIAG
    _HARMONY_DIAG = {}
    try:
        import numpy as _np
        from sklearn.decomposition import PCA as _PCA

        n_total = X_before.shape[0]
        rng = _np.random.default_rng(42)
        idx = rng.choice(n_total, size=min(n_sample, n_total), replace=False)

        Xb = X_before[idx].astype("float32")
        Xa = X_after[idx].astype("float32")
        conds = conditions[idx]
        unique_conds = sorted(set(conds.tolist()))

        # ── PCA commune (fit sur avant, transform avant + après) ──────────────
        pca = _PCA(n_components=2, random_state=42)
        pc_before = pca.fit_transform(Xb)
        pc_after = pca.transform(Xa)

        var_exp = pca.explained_variance_ratio_
        xlabel = f"PC1 ({var_exp[0] * 100:.1f}%)"
        ylabel = f"PC2 ({var_exp[1] * 100:.1f}%)"

        # ── Couleurs par condition ────────────────────────────────────────────
        _PALETTE = [
            "#667eea",
            "#f093fb",
            "#4facfe",
            "#43e97b",
            "#fa709a",
            "#fee140",
            "#a18cd1",
            "#ffecd2",
            "#89f7fe",
            "#ff9a9e",
        ]
        color_map = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(unique_conds)}
        colors_arr = _np.array([color_map[c] for c in conds])

        # ── Figure Plotly ─────────────────────────────────────────────────────
        try:
            import plotly.graph_objects as _go
            from plotly.subplots import make_subplots as _msp

            fig_p = _msp(
                rows=1,
                cols=2,
                subplot_titles=("Avant correction Harmony", "Après correction Harmony"),
                horizontal_spacing=0.08,
            )
            for cond in unique_conds:
                mask = conds == cond
                col_hex = color_map[cond]
                shared = dict(
                    mode="markers",
                    name=str(cond),
                    marker=dict(size=3, color=col_hex, opacity=0.55),
                    legendgroup=str(cond),
                )
                fig_p.add_trace(
                    _go.Scattergl(
                        x=pc_before[mask, 0].tolist(),
                        y=pc_before[mask, 1].tolist(),
                        showlegend=True,
                        **shared,
                    ),
                    row=1,
                    col=1,
                )
                fig_p.add_trace(
                    _go.Scattergl(
                        x=pc_after[mask, 0].tolist(),
                        y=pc_after[mask, 1].tolist(),
                        showlegend=False,
                        **shared,
                    ),
                    row=1,
                    col=2,
                )

            fig_p.update_xaxes(title_text=xlabel)
            fig_p.update_yaxes(title_text=ylabel)
            fig_p.update_layout(
                title_text=(
                    f"Correction Harmony — PCA ({min(n_sample, n_total):,} cellules, "
                    f"échantillon aléatoire)"
                ),
                height=480,
                template="plotly_white",
                legend_title_text="Condition",
            )
            _HARMONY_DIAG["plotly"] = fig_p
        except Exception as _pe:
            _logger.debug("Harmony diag Plotly: %s", _pe)
            _HARMONY_DIAG["plotly"] = None

        # ── Figure Matplotlib (fallback PDF) ─────────────────────────────────
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt

            fig_m, axes = _plt.subplots(1, 2, figsize=(12, 4.5), facecolor="#1e1e2e")
            for ax, (pc_data, title) in zip(
                axes,
                [
                    (pc_before, "Avant correction Harmony"),
                    (pc_after, "Après correction Harmony"),
                ],
            ):
                ax.set_facecolor("#1e1e2e")
                for cond in unique_conds:
                    mask = conds == cond
                    ax.scatter(
                        pc_data[mask, 0],
                        pc_data[mask, 1],
                        s=1.5,
                        alpha=0.5,
                        color=color_map[cond],
                        label=str(cond),
                        rasterized=True,
                    )
                ax.set_title(title, color="#e2e8f0", fontsize=10)
                ax.set_xlabel(xlabel, color="#e2e8f0", fontsize=8)
                ax.set_ylabel(ylabel, color="#e2e8f0", fontsize=8)
                ax.tick_params(colors="#e2e8f0")
                for sp in ax.spines.values():
                    sp.set_color("#45475a")

            handles, labels = axes[0].get_legend_handles_labels()
            fig_m.legend(
                handles,
                labels,
                title="Condition",
                loc="lower center",
                ncol=min(len(unique_conds), 5),
                fontsize=7,
                framealpha=0.2,
                labelcolor="#e2e8f0",
            )
            fig_m.suptitle(
                f"Correction Harmony — PCA ({min(n_sample, n_total):,} cellules)",
                color="#e2e8f0",
                fontsize=11,
            )
            fig_m.tight_layout(rect=[0, 0.08, 1, 1])
            _HARMONY_DIAG["mpl"] = fig_m
        except Exception as _me:
            _logger.debug("Harmony diag Matplotlib: %s", _me)
            _HARMONY_DIAG["mpl"] = None

    except Exception as _exc:
        _logger.debug("_build_harmony_diag échoué (non bloquant): %s", _exc)


def select_markers_for_clustering(
    var_names: List[str],
    config: PipelineConfig,
) -> List[str]:
    """
    Sélectionne les marqueurs à utiliser pour le clustering FlowSOM.

    Règles (dans l'ordre de priorité):
    1. whitelist dans config.markers.use (si non vide)
    2. Exclusion des patterns FSC/SSC/Time si config.flowsom.exclude_scatter=True
    3. Exclusion de la blacklist config.markers.exclude

    Args:
        var_names: Tous les marqueurs disponibles.
        config: Configuration du pipeline.

    Returns:
        Liste des marqueurs sélectionnés.
    """
    # Cas 1: whitelist explicite
    whitelist = getattr(config.markers, "use", [])
    if whitelist:
        available = [m for m in whitelist if m in var_names]
        missing = [m for m in whitelist if m not in var_names]
        if missing:
            _logger.warning("Marqueurs whitelist absents: %s", missing)
        return available

    # Cas 2: exclusion automatique
    markers = list(var_names)

    if getattr(config.markers, "exclude_scatter", True):
        markers = [m for m in markers if not any(p in m.upper() for p in SCATTER_PATTERNS)]

    # Cas 3: blacklist (correspondance par sous-chaîne, comme le monolithe)
    blacklist = getattr(config.markers, "exclude_additional", [])
    if blacklist:
        markers = [m for m in markers if not any(excl.upper() in m.upper() for excl in blacklist)]

    # Validation finale
    check_no_fsc_ssc_in_analysis_markers(markers)

    return markers


def stack_samples(
    samples: List[FlowSample],
    selected_markers: List[str],
    seed: int = 42,
    max_cells_total: Optional[int] = None,
    balance_conditions: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Empile plusieurs FlowSample en une matrice unique pour FlowSOM.

    Args:
        samples: Liste des échantillons prétraités.
        selected_markers: Marqueurs à inclure.
        seed: Graine pour le sous-échantillonnage d'équilibrage.
        max_cells_total: Limite totale de cellules (sous-échantillonnage global).
        balance_conditions: Si True, limite chaque condition au même nombre de cellules.

    Returns:
        Tuple (X_stacked [n_cells, n_markers], obs_metadata DataFrame).
    """
    all_X: List[np.ndarray] = []
    # PERF-2 FIX : accumulation de DataFrames vectorisés (np.full) au lieu d'une
    # boucle Python cellule par cellule — O(N) au lieu de O(N) appels Python.
    all_obs: List[pd.DataFrame] = []

    for sample in samples:
        X_s = sample.matrix
        var_s = sample.var_names

        # Sélectionner et réordonner les marqueurs.
        # Lookup O(1) par marqueur via dictionnaire — évite le double appel
        # O(N×M) de var_s.index() et le risque d'index incorrect si doublons.
        marker_to_idx = {m: i for i, m in enumerate(var_s)}
        missing = [m for m in selected_markers if m not in marker_to_idx]
        if missing:
            _logger.warning(
                "%s: marqueurs manquants %s — colonnes initialisées à 0",
                sample.name,
                missing,
            )

        X_sel = np.zeros((X_s.shape[0], len(selected_markers)), dtype=np.float64)
        for j, m in enumerate(selected_markers):
            if m in marker_to_idx:
                X_sel[:, j] = X_s[:, marker_to_idx[m]]

        all_X.append(X_sel)

        # PERF-2 : vectorisé — np.full évite la boucle Python sur chaque cellule
        n = X_s.shape[0]
        all_obs.append(
            pd.DataFrame(
                {
                    "condition": np.full(n, sample.condition, dtype=object),
                    "file_origin": np.full(n, sample.name, dtype=object),
                }
            )
        )

    if not all_X:
        raise ValueError("Aucun échantillon valide à empiler")

    X = np.vstack(all_X)
    obs = pd.concat(all_obs, ignore_index=True)

    # Vérification d'équilibre (info seulement — le déséquilibre Sain/Patho est voulu :
    # les NBM servent de référence calibrée et doivent dominer)
    if "condition" in obs.columns:
        conditions = obs["condition"].unique()
        if len(conditions) == 2:
            n1 = int((obs["condition"] == conditions[0]).sum())
            n2 = int((obs["condition"] == conditions[1]).sum())
            ratio = max(n1, n2) / max(min(n1, n2), 1)
            _logger.info(
                "Ratio cellules %s/%s : %.1f× (%d vs %d) — déséquilibre intentionnel NBM/Patho",
                conditions[0],
                conditions[1],
                ratio,
                n1,
                n2,
            )

    # Sous-échantillonnage global si nécessaire
    # CR-4 FIX : capturer n_before AVANT de modifier X pour que le log soit correct.
    if max_cells_total and X.shape[0] > max_cells_total:
        n_before = X.shape[0]
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_before, size=max_cells_total, replace=False)
        X = X[idx]
        obs = obs.iloc[idx].reset_index(drop=True)
        _logger.info(
            "Matrice sous-échantillonnée à %d/%d cellules",
            max_cells_total,
            n_before,  # taille originale, pas post-échantillonnage
        )

    # Contrôle NaN final
    n_nan = check_nan(X)
    if n_nan > 0:
        _logger.warning("%d NaN dans la matrice finale — remplacement par 0", n_nan)
        X = np.nan_to_num(X, nan=0.0)

    _logger.info(
        "Matrice FlowSOM: %d cellules × %d marqueurs",
        X.shape[0],
        X.shape[1],
    )
    return X, obs


def stack_raw_markers(
    samples: List[FlowSample],
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Empile les données brutes pré-transformation (raw_data) pour l'export FCS.

    Utilise ``sample.raw_data`` (valeurs linéaires, avant arcsinh/logicle) si
    disponible. Sinon, replie sur les données transformées (``sample.matrix``).
    Cette matrice est utilisée dans le fichier FCS final pour compatibilité
    Kaluza/FlowJo — les logiciels appliquent eux-mêmes leur transformation
    d'affichage à partir des valeurs brutes.

    Returns:
        Tuple (X_raw [n_cells, n_all_markers], raw_var_names, obs_df).
    """
    if not samples:
        raise ValueError("Aucun échantillon valide")

    has_raw = all(s.raw_data is not None for s in samples)
    if not has_raw:
        _logger.debug(
            "raw_data absent dans un ou plusieurs FlowSample — "
            "utilisation des données transformées pour l'export FCS "
            "(normal en mode batch/cache NBM)"
        )

    all_X: List[np.ndarray] = []
    all_conditions: List[np.ndarray] = []
    all_origins: List[np.ndarray] = []

    if has_raw:
        raw_var_names: List[str] = list(samples[0].raw_data.columns)
        for s in samples:
            X_s = s.raw_data.values
            if X_s.dtype != np.float32:
                X_s = X_s.astype(np.float32)
            n = X_s.shape[0]
            all_X.append(X_s)
            all_conditions.append(np.full(n, s.condition, dtype=object))
            all_origins.append(np.full(n, s.name, dtype=object))
    else:
        raw_var_names = list(samples[0].var_names)
        for s in samples:
            X_s = s.matrix
            if X_s.dtype != np.float32:
                X_s = X_s.astype(np.float32)
            n = X_s.shape[0]
            all_X.append(X_s)
            all_conditions.append(np.full(n, s.condition, dtype=object))
            all_origins.append(np.full(n, s.name, dtype=object))

    X = np.vstack(all_X)
    obs = pd.DataFrame(
        {
            "condition": np.concatenate(all_conditions),
            "file_origin": np.concatenate(all_origins),
        }
    )
    return X, raw_var_names, obs


def run_clustering(
    samples: List[FlowSample],
    config: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, FlowSOMClusterer, List[str], np.ndarray, pd.DataFrame, List]:
    """
    Exécute le clustering FlowSOM complet sur la liste d'échantillons.

    Args:
        samples: Échantillons prétraités.
        config: Configuration du pipeline.

    Returns:
        Tuple (metaclustering [n_cells], clustering [n_cells], clusterer, selected_markers).
    """
    if not samples:
        raise ValueError("Aucun échantillon à clusteriser")

    # Sélection des marqueurs depuis l'intersection de tous les panels.
    # Utiliser l'intersection garantit que selected_markers est présent dans
    # chaque sample, même si samples[0] n'a pas tous les marqueurs.
    _common_vars: List[str] = list(samples[0].var_names)
    for _s in samples[1:]:
        _s_set = set(_s.var_names)
        _common_vars = [m for m in _common_vars if m in _s_set]
    if len(_common_vars) < len(samples[0].var_names):
        _logger.warning(
            "Panels hétérogènes détectés : %d marqueurs communs sur %d dans samples[0]. "
            "Clustering basé sur l'intersection.",
            len(_common_vars),
            len(samples[0].var_names),
        )
    selected_markers = select_markers_for_clustering(_common_vars, config)
    _logger.info("Marqueurs pour FlowSOM (%d): %s", len(selected_markers), selected_markers)

    # ── Déséquilibre Maîtrisé (Stratified Downsampling) ──────────────────────
    # Rééquilibre le pool sain/patho AVANT le SOM pour rendre les clusters rares
    # (blastes LAIP) visibles malgré le déséquilibre extrême de classes.
    sd_cfg = getattr(config, "stratified_downsampling", None)
    if sd_cfg is not None and getattr(sd_cfg, "balance_conditions", False):
        _logger.info(
            "Déséquilibre Maîtrisé activé (ratio=%.1f×, seed=%d)...",
            sd_cfg.imbalance_ratio,
            sd_cfg.seed,
        )
        # Si nbm_ids est vide, on infère automatiquement les IDs NBM
        # depuis les échantillons dont la condition est "Healthy".
        nbm_ids = list(sd_cfg.nbm_ids)
        if not nbm_ids:
            _HEALTHY_CONDITIONS = {
                "Healthy",
                "Sain",
                "sain",
                "healthy",
                "Normal",
                "normal",
                "NBM",
            }
            nbm_ids = [s.name for s in samples if s.condition in _HEALTHY_CONDITIONS]
            _logger.info(
                "nbm_ids non spécifiés → auto-détection: %d fichiers NBM (conditions: %s)",
                len(nbm_ids),
                sorted({s.condition for s in samples if s.condition in _HEALTHY_CONDITIONS}),
            )

        # Ajouter un index local par fichier pour pouvoir retrouver les lignes
        # dans raw_data après l'équilibrage (ignore_index=True dans _to_flat_dataframe
        # efface les indices originaux).
        _IDX_COL = "_cell_idx"
        _orig_data_backup: dict = {}
        for _s in samples:
            _orig_data_backup[_s.name] = _s.data
            _s.data = _s.data.copy()
            _s.data[_IDX_COL] = np.arange(len(_s.data), dtype=np.int64)

        try:
            df_balanced = equilibrer_pool_flowsom(
                samples=samples,
                nbm_ids=nbm_ids,
                balance_conditions=True,
                imbalance_ratio=sd_cfg.imbalance_ratio,
                seed=sd_cfg.seed,
                allow_oversampling=getattr(sd_cfg, "allow_oversampling", False),
            )
        finally:
            # Restaurer les données originales dans tous les cas (exception ou non)
            # pour éviter que les samples restent pollués avec _cell_idx.
            for _s in samples:
                if _s.name in _orig_data_backup:
                    _s.data = _orig_data_backup[_s.name]

        # Reconstruire des FlowSample synthétiques à partir du DataFrame équilibré,
        # un par fichier source, pour conserver condition/file_origin par cellule.
        from src.models.sample import FlowSample as _FlowSample

        _meta_cols = {"condition", "file_origin", "class", _IDX_COL}
        _marker_cols = [c for c in df_balanced.columns if c not in _meta_cols]
        _balanced_samples: List[_FlowSample] = []
        for _fname, _grp in df_balanced.groupby("file_origin", sort=False):
            _cond = _grp["condition"].iloc[0] if "condition" in _grp.columns else "Unknown"
            # Sample original pour récupérer le raw_data si disponible
            _orig = next((s for s in samples if s.name == _fname), None)
            _df_markers = _grp[_marker_cols].reset_index(drop=True)

            # Propager raw_data en sélectionnant les lignes correspondantes
            _raw_data_balanced: Optional[pd.DataFrame] = None
            if _orig is not None and _orig.raw_data is not None and _IDX_COL in _grp.columns:
                _cell_indices = _grp[_IDX_COL].values.astype(int)
                _raw_data_balanced = _orig.raw_data.iloc[_cell_indices].reset_index(drop=True)

            _s = _FlowSample(
                name=str(_fname),
                path=_orig.path if _orig else "",
                condition=str(_cond),
                data=_df_markers,
                metadata=_orig.metadata if _orig else {},
                n_cells_raw=len(_df_markers),
                raw_data=_raw_data_balanced,
            )
            _balanced_samples.append(_s)
        samples = _balanced_samples
        _logger.info(
            "Pool équilibré: %d cellules injectées dans FlowSOM (%d fichiers sources).",
            len(df_balanced),
            len(samples),
        )
    # ── Fin Déséquilibre Maîtrisé ─────────────────────────────────────────────

    # Empilement des matrices
    flowsom_cfg = config.flowsom
    X, obs = stack_samples(
        samples,
        selected_markers,
        seed=config.flowsom.seed,
        max_cells_total=getattr(config.downsampling, "max_cells_for_clustering", None),
    )

    # Auto-clustering (trouver le k optimal)
    n_clusters = flowsom_cfg.n_metaclusters
    if getattr(config.auto_clustering, "enabled", False):
        _logger.info("Auto-clustering activé — recherche du k optimal...")
        n_clusters = find_optimal_clusters(
            X,
            min_clusters=getattr(config.auto_clustering, "min_clusters", 5),
            max_clusters=getattr(config.auto_clustering, "max_clusters", 35),
            n_bootstrap=getattr(config.auto_clustering, "n_bootstrap", 10),
            sample_size_bootstrap=getattr(config.auto_clustering, "sample_size_bootstrap", 20_000),
            min_stability_threshold=getattr(
                config.auto_clustering, "min_stability_threshold", 0.75
            ),
            weight_stability=getattr(config.auto_clustering, "weight_stability", 0.65),
            weight_silhouette=getattr(config.auto_clustering, "weight_silhouette", 0.35),
            xdim=flowsom_cfg.xdim,
            ydim=flowsom_cfg.ydim,
            seed=config.flowsom.seed,
            verbose=True,
        )
        _logger.info("k optimal trouvé: %d", n_clusters)
    elif n_clusters is None:
        n_clusters = 10  # défaut

    # FlowSOM
    clusterer = FlowSOMClusterer(
        xdim=flowsom_cfg.xdim,
        ydim=getattr(flowsom_cfg, "ydim", flowsom_cfg.xdim),
        n_metaclusters=n_clusters,
        seed=config.flowsom.seed,
        rlen=getattr(flowsom_cfg, "rlen", "auto"),
        use_gpu=getattr(config.gpu, "enabled", False),
    )

    # ── Intégration Harmony (correction d'effet batch) ────────────────────────
    # Aligne la distribution du patient sur celle de la NBM AVANT le SOM pour
    # éviter les clusters isolés générés par les variations de laser/réactifs.
    di_cfg = getattr(config, "data_integration", None)
    if (
        di_cfg is not None
        and getattr(di_cfg, "enabled", False)
        and getattr(di_cfg, "method", "harmony") == "harmony"
    ):
        if not _HARMONY_AVAILABLE:
            _logger.warning(
                "data_integration.enabled=True mais harmonypy est absent. "
                "Le pipeline continue avec les données brutes (pas d'alignement Harmony)."
            )
        elif "condition" not in obs.columns:
            _logger.warning(
                "Colonne 'condition' absente dans obs — Harmony ne peut pas identifier "
                "les batches. Le pipeline continue avec les données brutes."
            )
        else:
            import time as _time

            hp = getattr(di_cfg, "harmony_params", None)
            harmony_sigma = float(getattr(hp, "sigma", 0.05)) if hp else 0.05
            harmony_nclust = getattr(hp, "nclust", 30) if hp else 30
            harmony_block_size = float(getattr(hp, "block_size", 0.20)) if hp else 0.20
            harmony_max_iter = int(getattr(hp, "max_iter", 10)) if hp else 10
            harmony_max_iter_km = int(getattr(hp, "max_iter_kmeans", 10)) if hp else 10
            harmony_verbose = bool(getattr(hp, "verbose", False)) if hp else False
            markers_to_align_raw = list(getattr(hp, "markers_to_align", []) or []) if hp else []
            # nclust None = auto (N/30) — explicitement None si la config le dit
            if harmony_nclust is not None:
                harmony_nclust = int(harmony_nclust)

            # ── Harmony partiel (biology-first) ────────────────────────────
            # On corrige uniquement SSC pour préserver les marqueurs biologiques.
            def _marker_key(name: str) -> str:
                k = str(name).upper().strip().replace(" ", "")
                for suffix in ("-A", "-H", "-W", "_A", "_H", "_W"):
                    if k.endswith(suffix):
                        k = k[: -len(suffix)]
                        break
                return k

            selected_keys = [_marker_key(m) for m in selected_markers]
            align_indices = [i for i, m_key in enumerate(selected_keys) if m_key == "SSC"]
            align_markers = [selected_markers[i] for i in align_indices]

            if not align_indices:
                _logger.info(
                    "Harmony SSC: aucun canal SSC détecté dans selected_markers — "
                    "correction désactivée pour ce run."
                )

            if not align_indices:
                ho = None
                X_before_h = None
                X_harmony = None
            else:
                X_harmony = X[:, align_indices]
                # Copie pour le diagnostic avant/après uniquement si plusieurs
                # conditions sont présentes (sinon le diag n'a pas de sens).
                _n_conditions = len(obs["condition"].unique()) if "condition" in obs.columns else 1
                X_before_h = X_harmony.copy() if _n_conditions > 1 else None

                _logger.info(
                    "Harmony activé — %d cellules × %d marqueurs à aligner "
                    "(nclust=%s, sigma=%.3f, block_size=%.2f, max_iter=%d)...",
                    X.shape[0],
                    len(align_indices),
                    harmony_nclust,
                    harmony_sigma,
                    harmony_block_size,
                    harmony_max_iter,
                )
            _t0 = _time.perf_counter()

            # Sélection device avec fallback CPU si OOM GPU
            _device = None  # auto-détection (harmonypy choisit cuda si dispo)
            _meta_df = pd.DataFrame({"condition": obs["condition"].values})

            def _run_harmony_with_fallback(device_str):
                return harmonypy.run_harmony(
                    X_harmony,
                    _meta_df,
                    "condition",
                    sigma=harmony_sigma,
                    nclust=harmony_nclust,
                    block_size=harmony_block_size,
                    max_iter_harmony=harmony_max_iter,
                    max_iter_kmeans=harmony_max_iter_km,
                    verbose=harmony_verbose,
                    device=device_str,
                )

            if align_indices:
                try:
                    ho = _run_harmony_with_fallback(_device)
                except RuntimeError as _gpu_err:
                    _msg = str(_gpu_err).lower()
                    if "cuda" in _msg or "out of memory" in _msg or "device" in _msg:
                        _logger.warning("Harmony GPU erreur (%s) — reprise sur CPU.", _gpu_err)
                        try:
                            ho = _run_harmony_with_fallback("cpu")
                        except Exception as _cpu_err:
                            _logger.warning(
                                "Harmony CPU également échoué (%s) — données brutes conservées.",
                                _cpu_err,
                            )
                            ho = None
                    else:
                        _logger.warning(
                            "Harmony a échoué (%s) — données brutes conservées.",
                            _gpu_err,
                        )
                        ho = None
                except Exception as _exc:
                    _logger.warning("Harmony a échoué (%s) — données brutes conservées.", _exc)
                    ho = None

            if ho is not None:
                # Vérification robuste de l'orientation: harmonypy peut renvoyer
                # [n_markers, n_cells] selon la version et les wrappers.
                X_corr_raw = np.asarray(ho.Z_corr, dtype=np.float32)
                if X_corr_raw.shape == X_harmony.shape:
                    X_corrected_subset = X_corr_raw
                elif X_corr_raw.T.shape == X_harmony.shape:
                    X_corrected_subset = X_corr_raw.T
                else:
                    X_corrected_subset = None

                if X_corrected_subset is not None:
                    # ── Diagnostic avant/après (rapide, sous-échantillon) ─────
                    _build_harmony_diag(
                        X_before=X_before_h,
                        X_after=X_corrected_subset,
                        conditions=obs["condition"].values,
                    )

                    if len(align_indices) == X.shape[1]:
                        X = X_corrected_subset
                    else:
                        X_updated = X.copy()
                        X_updated[:, align_indices] = X_corrected_subset
                        X = X_updated

                    _logger.info(
                        "Harmony partiel terminé en %.1fs — %d marqueur(s) corrigé(s): %s",
                        _time.perf_counter() - _t0,
                        len(align_markers),
                        align_markers,
                    )
                else:
                    _logger.warning(
                        "Harmony a retourné une matrice de forme inattendue %s "
                        "(attendu %s) — données brutes conservées.",
                        X_corr_raw.shape,
                        X_harmony.shape,
                    )
    # ── Fin intégration Harmony ───────────────────────────────────────────────

    _logger.info(
        "FlowSOM: grille %d×%d, %d métaclusters...",
        clusterer.xdim,
        clusterer.ydim,
        n_clusters,
    )
    clusterer.fit(X, selected_markers)
    metaclustering = getattr(clusterer, "metacluster_assignments_", np.zeros(X.shape[0], dtype=int))
    clustering = getattr(clusterer, "node_assignments_", np.zeros(X.shape[0], dtype=int))

    return metaclustering, clustering, clusterer, selected_markers, X, obs, samples


def extract_date_from_filename(filename: str) -> Tuple[str, int]:
    """
    Extrait la date/timepoint depuis le nom de fichier FCS.

    Essaie successivement:
    1. DD-MM-YYYY ou DD/MM/YYYY
    2. YYYY-MM-DD ou YYYY/MM/DD
    3. DD-MM-YY ou DD/MM/YY
    4. DDMMYYYY (8 chiffres collés)

    Args:
        filename: Nom de fichier (sans chemin ou avec).

    Returns:
        Tuple (timepoint_str, timepoint_num) où timepoint_num est
        un entier YYYYMMDD (0 si non trouvé).
    """
    import re
    from pathlib import Path as _Path

    stem = _Path(filename).stem

    patterns = [
        r"(\d{2})[-/](\d{2})[-/](\d{4})",  # DD-MM-YYYY
        r"(\d{4})[-/](\d{2})[-/](\d{2})",  # YYYY-MM-DD
        r"(\d{2})[-/](\d{2})[-/](\d{2})",  # DD-MM-YY
        r"(\d{8})",  # DDMMYYYY ou YYYYMMDD
    ]

    for pattern in patterns:
        m = re.search(pattern, stem)
        if m is None:
            continue
        groups = m.groups()
        try:
            if len(groups) == 3:
                g1, g2, g3 = groups
                if len(g3) == 4:
                    # DD-MM-YYYY
                    tp_str = f"{g1}/{g2}/{g3}"
                    tp_num = int(f"{g3}{g2}{g1}")
                elif len(g1) == 4:
                    # YYYY-MM-DD
                    tp_str = f"{g3}/{g2}/{g1}"
                    tp_num = int(f"{g1}{g2}{g3}")
                else:
                    # DD-MM-YY → 20YY
                    year = int(g3) + (2000 if int(g3) < 50 else 1900)
                    tp_str = f"{g1}/{g2}/{year}"
                    tp_num = year * 10000 + int(g2) * 100 + int(g1)
            else:
                # 8 digits
                raw = groups[0]
                if len(raw) == 8:
                    # Heuristique: YYYY en tête si raw[0:4] > 1900
                    if 1900 < int(raw[:4]) < 2100:
                        tp_str = f"{raw[6:8]}/{raw[4:6]}/{raw[:4]}"
                        tp_num = int(raw)
                    else:
                        tp_str = f"{raw[:2]}/{raw[2:4]}/{raw[4:8]}"
                        tp_num = int(f"{raw[4:8]}{raw[2:4]}{raw[:2]}")
                else:
                    continue
            return tp_str, tp_num
        except (ValueError, IndexError):
            continue

    return "", 0


def build_cells_dataframe(
    X_raw: np.ndarray,
    var_names: List[str],
    obs: pd.DataFrame,
    metaclustering: np.ndarray,
    clustering: np.ndarray,
) -> pd.DataFrame:
    """
    Construit le DataFrame cellulaire complet (marqueurs + métadonnées + clustering).

    Ajoute automatiquement des colonnes Timepoint/Timepoint_Num extraites
    du nom de fichier FCS (DD/MM/YYYY et entier YYYYMMDD).

    Args:
        X_raw: Matrice des marqueurs (n_cells, n_markers).
        var_names: Noms des colonnes.
        obs: Métadonnées des cellules (condition, file_origin).
        metaclustering: Assignation métacluster (n_cells,).
        clustering: Assignation cluster SOM (n_cells,).

    Returns:
        DataFrame complet.
    """
    df = pd.DataFrame(X_raw, columns=var_names)
    condition_values = (
        obs["condition"].to_numpy(copy=False)
        if "condition" in obs.columns
        else np.full(X_raw.shape[0], "Unknown", dtype=object)
    )
    file_origin_values = (
        obs["file_origin"].to_numpy(copy=False)
        if "file_origin" in obs.columns
        else np.full(X_raw.shape[0], "", dtype=object)
    )
    df["condition"] = condition_values
    df["file_origin"] = file_origin_values
    df["FlowSOM_metacluster"] = metaclustering.astype(np.int32)
    df["FlowSOM_cluster"] = clustering.astype(np.int32)

    # Encodage numérique vectorisé (évite .map sur des millions de lignes)
    conditions_unique, cond_inverse = np.unique(condition_values, return_inverse=True)
    df["Condition_Num"] = (cond_inverse + 1).astype(np.float32)
    df["Condition"] = df["condition"]

    # Extraction timepoint par fichier unique (vectorisé via indices inverses)
    if file_origin_values.size > 0:
        unique_files, file_inverse = np.unique(file_origin_values, return_inverse=True)
        tp_str_unique = np.empty(unique_files.shape[0], dtype=object)
        tp_num_unique = np.empty(unique_files.shape[0], dtype=np.int64)
        for i, fname in enumerate(unique_files):
            tp_str, tp_num = extract_date_from_filename(str(fname))
            tp_str_unique[i] = tp_str
            tp_num_unique[i] = tp_num
        df["Timepoint"] = tp_str_unique[file_inverse]
        df["Timepoint_Num"] = tp_num_unique[file_inverse]

    return df
