"""
clustering.py — Orchestration du clustering FlowSOM.

Ce module encapsule l'entraînement du SOM, le métaclustering hiérarchique,
et l'auto-optimisation du nombre de clusters (stabilité + silhouette).

Hiérarchie d'estimateurs (ordre de priorité):
  1. GPUFlowSOMEstimator  (CuPy/CUDA, si gpu.enabled=True et CUDA dispo)
  2. fs.FlowSOM           (CPU, package flowsom officiel)

Usage:
    clusterer = FlowSOMClusterer(config.flowsom, config.gpu)
    result = clusterer.fit(X, used_markers)
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Assurer le multi-threading BLAS dès l'import du module (dev + exe).
# En mode exe le runtime_hook_debug.py les définit déjà, ce setdefault est inerte.
_N_CPU = str(os.cpu_count() or 1)
for _blas_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_blas_var, _N_CPU)

_logger = logging.getLogger("core.clustering")

# Imports conditionnels — wrappés pour graceful fallback
try:
    import flowsom as fs

    _FLOWSOM_AVAILABLE = True
except ImportError:
    _FLOWSOM_AVAILABLE = False
    warnings.warn("flowsom non installé: pip install flowsom")

def _resolve_gpu_module_path() -> str | None:
    """
    Résout le chemin du module GPU FlowSOMEstimator de manière portable.

    Stratégie (priorité décroissante) :
      1. Variable d'environnement FLOWSOM_GPU_PATH — portable, définie au lancement.
      2. Racine du package détectée via importlib.resources / __spec__ — stable
         en mode développement et après PyInstaller.

    Retourne None si aucun chemin valide n'est trouvé.
    """
    import os
    import sys

    # Priorité 1 : variable d'environnement explicite (portable, recommandé en prod)
    env_path = os.environ.get("FLOWSOM_GPU_PATH")
    if env_path:
        return env_path

    # Priorité 2 : remontée depuis __file__ du module courant.
    # FlowSomGpu peut se trouver N niveaux au-dessus de flowsom_pipeline_pro
    # (ex: Documents/FlowSom/FlowSomGpu/ alors que le package est dans
    #  Documents/FlowSom/Perplexity/flowsom_pipeline_pro/).
    # On remonte jusqu'à 6 niveaux depuis ce fichier pour le trouver.
    try:
        import os.path as osp
        current = osp.dirname(osp.abspath(__file__))
        for _ in range(6):
            if osp.isdir(osp.join(current, "FlowSomGpu")):
                return current
            current = osp.dirname(current)
    except Exception:
        pass

    # Priorité 3 : racine du package installé via __spec__
    try:
        import importlib.util
        spec = importlib.util.find_spec("flowsom_pipeline_pro")
        if spec is not None and spec.submodule_search_locations:
            pkg_root = str(list(spec.submodule_search_locations)[0])
            import os.path as osp
            candidate = osp.dirname(pkg_root)
            if osp.isdir(osp.join(candidate, "FlowSomGpu")):
                return candidate
    except Exception:
        pass

    return None


try:
    import sys
    import os

    _gpu_path = _resolve_gpu_module_path()
    if _gpu_path is not None and _gpu_path not in sys.path:
        sys.path.insert(0, _gpu_path)
    from FlowSomGpu.models import GPUFlowSOMEstimator

    _GPU_AVAILABLE = True
except ImportError as _gpu_import_err:
    _GPU_AVAILABLE = False
    GPUFlowSOMEstimator = None
    _logger.debug("GPUFlowSOMEstimator non disponible (ImportError): %s", _gpu_import_err)
except Exception as _gpu_err:
    _GPU_AVAILABLE = False
    GPUFlowSOMEstimator = None
    _logger.debug("GPUFlowSOMEstimator non disponible (pas de CUDA): %s", _gpu_err)


def compute_optimal_rlen(n_cells: int, rlen_setting: Any = "auto") -> int:
    """
    Calcule rlen optimal basé sur la taille du dataset.

    Formule littérature: rlen ∝ √N × 0.1, borné [10, 100].
    Exemples:
        10k   cellules → rlen ≈ 10
        100k  cellules → rlen ≈ 31
        500k  cellules → rlen ≈ 70
        1M    cellules → rlen ≈ 100

    Args:
        n_cells: Nombre de cellules dans le dataset.
        rlen_setting: 'auto' ou entier explicite.

    Returns:
        Valeur rlen calculée ou passée explicitement.
    """
    if isinstance(rlen_setting, int):
        return rlen_setting
    return max(10, min(100, int(np.sqrt(n_cells) * 0.1)))


def compute_optimal_grid(
    n_cells: int,
    xdim: int = 10,
    ydim: int = 10,
) -> Tuple[int, int]:
    """
    Ajuste la grille SOM si peu de cellules.

    < 50k cellules → 7×7 recommandé (éviter trop de nodes vides).

    Args:
        n_cells: Nombre de cellules.
        xdim: Dimension X configurée.
        ydim: Dimension Y configurée.

    Returns:
        Tuple (xdim_final, ydim_final).
    """
    if n_cells < 50_000 and xdim * ydim > 49:
        return 7, 7
    return xdim, ydim


class FlowSOMClusterer:
    """
    Orchestre l'entraînement FlowSOM et le métaclustering.

    Args:
        xdim: Dimension X de la grille SOM.
        ydim: Dimension Y de la grille SOM.
        n_metaclusters: Nombre de métaclusters.
        rlen: Itérations SOM ('auto' ou entier).
        seed: Graine aléatoire.
        use_gpu: Activer GPUFlowSOMEstimator si disponible.
        learning_rate: Taux d'apprentissage SOM.
        sigma: Sigma de voisinage SOM.
    """

    def __init__(
        self,
        xdim: int = 10,
        ydim: int = 10,
        n_metaclusters: int = 8,
        rlen: Any = "auto",
        seed: int = 42,
        use_gpu: bool = True,
        learning_rate: float = 0.05,
        sigma: float = 1.5,
    ) -> None:
        self.xdim = xdim
        self.ydim = ydim
        self.n_metaclusters = n_metaclusters
        self.rlen = rlen
        self.seed = seed
        self.use_gpu = use_gpu
        self.learning_rate = learning_rate
        self.sigma = sigma

        self._fsom_model: Optional[Any] = None
        self.node_assignments_: Optional[np.ndarray] = None
        self.metacluster_map_: Optional[np.ndarray] = None  # node→metacluster
        self.metacluster_assignments_: Optional[np.ndarray] = None  # cell→metacluster
        self.used_gpu_: bool = False
        self._mst_layout_: Optional[np.ndarray] = (
            None  # (n_nodes, 2) — coordonnées MST Kamada-Kawai
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        marker_names: Optional[List[str]] = None,
    ) -> "FlowSOMClusterer":
        """
        Entraîne le modèle FlowSOM sur les données X.

        Args:
            X: Matrice (n_cells, n_markers) — doit être transformée et normalisée.
            marker_names: Noms des marqueurs (pour logging).

        Returns:
            self (pour chaînage).

        Raises:
            ImportError: Si aucun backend FlowSOM n'est disponible.
            ValueError: Si X contient des NaN ou Inf.
        """
        # Validation critique (NaN/Inf font planter FlowSOM silencieusement)
        n_nan = np.isnan(X).sum()
        n_inf = np.isinf(X).sum()
        if n_nan > 0 or n_inf > 0:
            raise ValueError(
                f"X contient {n_nan} NaN et {n_inf} Inf. "
                "Nettoyer les données avant FlowSOM."
            )

        n_cells = X.shape[0]
        xdim, ydim = compute_optimal_grid(n_cells, self.xdim, self.ydim)
        rlen = compute_optimal_rlen(n_cells, self.rlen)

        # ── Tentative GPU ──────────────────────────────────────────────
        if self.use_gpu and _GPU_AVAILABLE and GPUFlowSOMEstimator is not None:
            try:
                self._fsom_model = GPUFlowSOMEstimator(
                    xdim=xdim,
                    ydim=ydim,
                    n_clusters=self.n_metaclusters,
                    learning_rate=self.learning_rate,
                    sigma=self.sigma,
                    seed=self.seed,
                )
                self._fsom_model.fit_predict(X)
                self.used_gpu_ = True
                _logger.info("[OK] FlowSOM GPU — grille %d×%d, rlen=%d", xdim, ydim, rlen)
                self._extract_assignments_gpu(X)
                return self
            except Exception as e:
                warnings.warn(f"GPU FlowSOM échoué ({e}), basculement CPU.")

        # ── CPU (flowsom officiel) ─────────────────────────────────────
        if not _FLOWSOM_AVAILABLE:
            raise ImportError(
                "Aucun backend FlowSOM disponible. Installer: pip install flowsom"
            )

        try:
            import anndata as ad

            adata = ad.AnnData(X)
            if marker_names:
                adata.var_names = marker_names[: X.shape[1]]

            self._fsom_model = fs.FlowSOM(
                adata,
                cols_to_use=list(range(X.shape[1])),
                xdim=xdim,
                ydim=ydim,
                n_clusters=self.n_metaclusters,
                rlen=rlen,
                seed=self.seed,
            )
            self.used_gpu_ = False
            _logger.info(
                "[OK] FlowSOM CPU — grille %d×%d, rlen=%d, k=%d",
                xdim, ydim, rlen, self.n_metaclusters,
            )
            self._extract_assignments_cpu(X)

        except Exception as e:
            raise RuntimeError(f"FlowSOM CPU échoué: {e}") from e

        return self

    def _extract_assignments_gpu(self, X: np.ndarray) -> None:
        """Extrait les assignations depuis le modèle GPU (BaseFlowSOMEstimator)."""
        try:
            # fit_predict() a déjà renseigné cluster_labels_ et labels_
            self.node_assignments_ = np.asarray(
                self._fsom_model.cluster_labels_, dtype=int
            )
            self.metacluster_assignments_ = np.asarray(
                self._fsom_model.labels_, dtype=int
            )
            if hasattr(self._fsom_model, "_y_codes"):
                self.metacluster_map_ = np.asarray(self._fsom_model._y_codes, dtype=int)
        except Exception as e:
            warnings.warn(f"Extraction assignations GPU échouée: {e}")

        # Calcul du layout MST sur le codebook GPU (identique à fs.FlowSOM.build_MST)
        try:
            codebook = np.asarray(self._fsom_model.codes, dtype=float)
            self._mst_layout_ = self._compute_mst_layout(codebook)
            _logger.info(
                "   [OK] Layout MST calculé sur codebook GPU (%d nodes)", codebook.shape[0]
            )
        except Exception as e:
            warnings.warn(f"Calcul layout MST GPU échoué: {e}")

    def _extract_assignments_cpu(self, X: np.ndarray) -> None:
        """Extrait les assignations depuis le modèle CPU (flowsom)."""
        try:
            fsom = self._fsom_model
            # fs.FlowSOM expose les données via get_cell_data() (AnnData)
            cell_data = fsom.get_cell_data()

            if "clustering" in cell_data.obs.columns:
                self.node_assignments_ = cell_data.obs["clustering"].values.astype(int)
            else:
                self.node_assignments_ = np.zeros(X.shape[0], dtype=int)
                warnings.warn(
                    "Impossible d'extraire node_assignments depuis flowsom CPU"
                )

            if "metaclustering" in cell_data.obs.columns:
                self.metacluster_assignments_ = cell_data.obs[
                    "metaclustering"
                ].values.astype(int)
            else:
                # Recalcul par AgglomerativeClustering sur le codebook
                self._recompute_metaclusters()

            # Extraire le metacluster_map_ (node → metacluster) depuis cluster_data
            try:
                cluster_data = fsom.get_cluster_data()
                if "metaclustering" in cluster_data.obs.columns:
                    self.metacluster_map_ = cluster_data.obs[
                        "metaclustering"
                    ].values.astype(int)
            except Exception as e:
                _logger.debug("metacluster_map_ non extrait depuis cluster_data: %s", e)

        except Exception as e:
            warnings.warn(f"Extraction assignations CPU échouée: {e}")

    def _compute_mst_layout(self, codebook: np.ndarray) -> np.ndarray:
        """
        Calcule le layout MST Kamada-Kawai sur le codebook SOM.

        Reproduit exactement fs.FlowSOM.build_MST() (saeyslab/flowsom):
          1. Distance euclidienne entre tous les nodes (cdist)
          2. Graphe complet pondéré (igraph Weighted_Adjacency)
          3. Arbre couvrant minimum (spanning_tree)
          4. Layout Kamada-Kawai (seed=grille, maxiter=50×N)

        Args:
            codebook: Matrice (n_nodes, n_markers) — centroïdes SOM.

        Returns:
            Array float (n_nodes, 2) — coordonnées x/y du layout MST.
        """
        from scipy.spatial.distance import cdist
        import igraph as ig

        adjacency = cdist(codebook, codebook, metric="euclidean")
        full_graph = ig.Graph.Weighted_Adjacency(
            adjacency, mode="undirected", loops=False
        )
        mst_graph = ig.Graph.spanning_tree(full_graph, weights=full_graph.es["weight"])
        mst_graph.es["weight"] = [
            w / np.mean(mst_graph.es["weight"]) for w in mst_graph.es["weight"]
        ]
        layout = mst_graph.layout_kamada_kawai(
            seed=mst_graph.layout_grid(),
            maxiter=50 * mst_graph.vcount(),
            kkconst=max(mst_graph.vcount(), 1),
        ).coords
        return np.array(layout, dtype=float)

    def _get_codebook(self) -> Optional[np.ndarray]:
        """
        Récupère le codebook SOM (centroïdes des nodes) depuis le modèle.

        Gère les deux backends:
          - GPU: _fsom_model.codes
          - CPU (saeyslab): _fsom_model.get_cluster_data().X

        Returns:
            Array (n_nodes, n_markers) ou None si indisponible.
        """
        if self._fsom_model is None:
            return None

        # GPU backend — attribut .codes direct
        if hasattr(self._fsom_model, "codes"):
            return np.asarray(self._fsom_model.codes, dtype=float)

        # CPU saeyslab — codebook dans get_cluster_data().X
        if hasattr(self._fsom_model, "get_cluster_data"):
            try:
                cluster_data = self._fsom_model.get_cluster_data()
                if cluster_data.X is not None:
                    return np.asarray(cluster_data.X, dtype=float)
            except Exception as e:
                _logger.debug("Codebook non extrait depuis get_cluster_data(): %s", e)

        return None

    def _recompute_metaclusters(self) -> None:
        """Recalcule les métaclusters par clustering hiérarchique sur le codebook SOM."""
        try:
            from sklearn.cluster import AgglomerativeClustering

            if self._fsom_model is None or self.node_assignments_ is None:
                return

            # Obtenir le codebook (centroïdes des nodes)
            codebook = self._get_codebook()
            if codebook is None:
                return

            n_nodes = codebook.shape[0]
            agg = AgglomerativeClustering(
                n_clusters=min(self.n_metaclusters, n_nodes),
                linkage="ward",
            )
            self.metacluster_map_ = agg.fit_predict(codebook).astype(int)
            self.metacluster_assignments_ = self.metacluster_map_[
                self.node_assignments_
            ]
        except Exception as e:
            warnings.warn(f"Recalcul métaclusters échoué: {e}")

    # ------------------------------------------------------------------
    # Propriétés de résultat
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        """Nombre total de nodes dans la grille SOM."""
        return self.xdim * self.ydim

    def get_mfi_matrix(
        self,
        X: np.ndarray,
        marker_names: List[str],
    ) -> "np.ndarray":
        """
        Calcule la matrice MFI (Mean Fluorescence Intensity) par métacluster.

        Args:
            X: Matrice de données transformées (n_cells, n_markers).
            marker_names: Noms des marqueurs.

        Returns:
            Matrice (n_metaclusters, n_markers).

        Raises:
            RuntimeError: Si les assignations ne sont pas disponibles.
        """
        if self.metacluster_assignments_ is None:
            raise RuntimeError("fit() doit être appelé avant get_mfi_matrix()")

        # PERF-3 FIX : vectorisation par tri + découpage (np.split).
        # Évite de créer un masque booléen O(N) pour chaque métacluster.
        # Complexité : O(N log N) pour le tri, O(N) pour np.split et les médianes.
        unique_mc = np.unique(self.metacluster_assignments_)
        sort_idx = np.argsort(self.metacluster_assignments_, kind="stable")
        X_sorted = X[sort_idx]
        labels_sorted = self.metacluster_assignments_[sort_idx]
        # Points de coupure entre groupes consécutifs de métaclusters différents
        split_pts = np.where(np.diff(labels_sorted))[0] + 1
        chunks = np.split(X_sorted, split_pts)

        mfi = np.zeros((len(unique_mc), X.shape[1]))
        for i, chunk in enumerate(chunks):
            mfi[i] = np.median(chunk, axis=0)  # Médiane = MFI robuste

        return mfi

    def summary(self) -> str:
        """Résumé du clustering."""
        backend = "GPU" if self.used_gpu_ else "CPU"
        n_cells = (
            len(self.node_assignments_) if self.node_assignments_ is not None else 0
        )
        return (
            f"FlowSOMClusterer({backend}, grid={self.xdim}×{self.ydim}, "
            f"k={self.n_metaclusters}, cells={n_cells:,})"
        )

    def get_grid_coords(self) -> np.ndarray:
        """
        Retourne les coordonnées de grille SOM (n_nodes, 2) pour chaque node.

        Pour le modèle CPU (fs.FlowSOM), utilise get_cluster_data().obsm["grid"]
        si disponible. Pour le GPU (ou en fallback), calcule le mapping
        grille régulière row-major: node i → (col=i % xdim, row=i // xdim).

        Returns:
            Array float (n_nodes, 2) — colonnes [x, y].
        """
        # CPU fs.FlowSOM — obsm["grid"] disponible
        if not self.used_gpu_ and self._fsom_model is not None:
            try:
                cluster_data = self._fsom_model.get_cluster_data()
                grid = cluster_data.obsm.get("grid", None)
                if grid is not None:
                    return np.asarray(grid, dtype=float)
            except Exception:
                pass

        # Fallback : grille régulière row-major (identique au comportement R FlowSOM)
        n_nodes = self.xdim * self.ydim
        coords = np.array(
            [(i % self.xdim + 1, i // self.xdim + 1) for i in range(n_nodes)],
            dtype=float,
        )
        return coords

    def get_layout_coords(self) -> np.ndarray:
        """
        Retourne les coordonnées MST Kamada-Kawai (n_nodes, 2) pour chaque node.

        Priorité:
          1. _mst_layout_ préalablement calculé (GPU ou CPU)
          2. obsm["layout"] du modèle CPU fs.FlowSOM
          3. Calcul à la demande depuis le codebook GPU
          4. Fallback grille régulière

        Returns:
            Array float (n_nodes, 2) — colonnes [x, y].
        """
        # Layout déjà calculé (GPU via _extract_assignments_gpu)
        if self._mst_layout_ is not None:
            return self._mst_layout_

        # CPU fs.FlowSOM — obsm["layout"] natif
        if not self.used_gpu_ and self._fsom_model is not None:
            try:
                cluster_data = self._fsom_model.get_cluster_data()
                layout = cluster_data.obsm.get("layout", None)
                if layout is not None:
                    self._mst_layout_ = np.asarray(layout, dtype=float)
                    return self._mst_layout_
            except Exception:
                pass

        # Calcul à la demande depuis le codebook (GPU ou CPU)
        codebook = self._get_codebook()
        if codebook is not None:
            try:
                self._mst_layout_ = self._compute_mst_layout(codebook)
                return self._mst_layout_
            except Exception as e:
                warnings.warn(f"Calcul layout MST à la demande échoué: {e}")

        # Fallback ultime : grille régulière
        return self.get_grid_coords()

    def get_node_sizes(self) -> np.ndarray:
        """
        Retourne le nombre de cellules par node SOM (n_nodes,).

        Returns:
            Array int (n_nodes,).
        """
        if self.node_assignments_ is None:
            return np.zeros(self.n_nodes, dtype=int)
        return np.bincount(
            self.node_assignments_.astype(int), minlength=self.n_nodes
        ).astype(float)



# ─────────────────────────────────────────────────────────────────────────────
#  Optimisation multi-critères du nombre de clusters (3 phases)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import AgglomerativeClustering

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from sklearn.metrics import adjusted_rand_score as _ari_score

    _ARI_AVAILABLE = True
except ImportError:
    _ARI_AVAILABLE = False


def _get_logger_clustering():
    """Logger local pour éviter la circularité d'import."""
    import logging

    return logging.getLogger("core.clustering.stability")


def _train_som_codebook(
    X: np.ndarray,
    xdim: int,
    ydim: int,
    n_metaclusters: int,
    rlen: int,
    seed: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Entraîne un SOM et retourne (codebook, node_assignments).

    Returns None si aucun backend disponible ou si l'entraînement échoue.
    """
    try:
        if _GPU_AVAILABLE and GPUFlowSOMEstimator is not None:
            est = GPUFlowSOMEstimator(
                xdim=xdim,
                ydim=ydim,
                n_clusters=n_metaclusters,
                seed=seed,
            )
            est.fit_predict(X)
            node_assignments = np.asarray(est.cluster_labels_, dtype=int)

            # Codebook = centroïdes des nodes
            if hasattr(est, "codes_"):
                codebook = np.asarray(est.codes_)
            elif hasattr(est, "_prototypes"):
                codebook = np.asarray(est._prototypes)
            else:
                # Recalcul par médiane si le modèle n'expose pas les codes
                n_nodes = xdim * ydim
                codebook = np.array(
                    [
                        np.median(X[node_assignments == n], axis=0)
                        if (node_assignments == n).sum() > 0
                        else np.zeros(X.shape[1])
                        for n in range(n_nodes)
                    ]
                )
            return codebook, node_assignments

        elif _FLOWSOM_AVAILABLE:
            import anndata as ad

            adata = ad.AnnData(X)
            fsom = fs.FlowSOM(
                adata,
                cols_to_use=list(range(X.shape[1])),
                xdim=xdim,
                ydim=ydim,
                n_clusters=n_metaclusters,
                rlen=rlen,
                seed=seed,
            )
            cell_data = fsom.get_cell_data()
            node_assignments = cell_data.obs["clustering"].values.astype(int)
            codebook = fsom.get_cluster_data().X
            if codebook is None:
                n_nodes = xdim * ydim
                codebook = np.array(
                    [
                        np.median(X[node_assignments == n], axis=0)
                        if (node_assignments == n).sum() > 0
                        else np.zeros(X.shape[1])
                        for n in range(n_nodes)
                    ]
                )
            return codebook, node_assignments

    except Exception as e:
        _get_logger_clustering().warning(
            "_train_som_codebook échoué (k=%d): %s", n_metaclusters, e
        )

    return None


def _metacluster_codebook(
    codebook: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Re-métaclustère le codebook SOM par AgglomerativeClustering.

    Beaucoup plus rapide que de ré-entraîner FlowSOM complet pour chaque k.
    Conforme au protocole ELN (raisonnement sur les nœuds, pas les cellules).
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError("sklearn requis: pip install scikit-learn")
    n_nodes = codebook.shape[0]
    k_eff = min(k, n_nodes)
    agg = AgglomerativeClustering(n_clusters=k_eff, linkage="ward")
    return agg.fit_predict(codebook).astype(int)


def phase1_silhouette_on_codebook(
    X: np.ndarray,
    xdim: int,
    ydim: int,
    rlen: int,
    seed: int,
    k_range: List[int],
) -> "pd.DataFrame":
    """
    Phase 1 — Score silhouette sur le codebook SOM pour chaque k candidat.

    L'astuce d'efficacité: on entraîne le SOM UNE SEULE FOIS avec k=max,
    puis on ré-métaclustère avec AgglomerativeClustering pour chaque k.
    Cela réduit le temps de O(|k_range| × T_SOM) à O(T_SOM + |k_range| × T_Agg).

    Le silhouette est calculé sur les nœuds du codebook (100–400 points),
    pas sur les cellules (10^5–10^6), ce qui est très rapide.

    Args:
        X: Matrice de données (n_cells, n_markers).
        xdim, ydim: Dimensions de la grille SOM.
        rlen: Itérations SOM.
        seed: Graine aléatoire.
        k_range: Liste des k à évaluer.

    Returns:
        DataFrame [k, silhouette_score] trié par k.
    """
    import pandas as pd

    if not _SKLEARN_AVAILABLE:
        raise ImportError("sklearn requis: pip install scikit-learn")

    _log = _get_logger_clustering()
    k_max = max(k_range)
    _log.info(
        "Phase 1: entraînement SOM (k_max=%d, grille %dx%d)...", k_max, xdim, ydim
    )

    result = _train_som_codebook(X, xdim, ydim, k_max, rlen, seed)
    if result is None:
        raise RuntimeError("Impossible d'entraîner le SOM — vérifier les backends.")

    codebook, _ = result
    _log.info("Phase 1: codebook SOM (%d nœuds, %d marqueurs)", *codebook.shape)

    rows = []
    for k in sorted(k_range):
        if k < 2:
            rows.append({"k": k, "silhouette_score": 0.0})
            continue
        labels = _metacluster_codebook(codebook, k)
        n_unique = len(np.unique(labels))
        if n_unique < 2 or n_unique >= len(codebook):
            rows.append({"k": k, "silhouette_score": 0.0})
            continue
        try:
            sil = silhouette_score(codebook, labels)
        except Exception:
            sil = 0.0
        rows.append({"k": k, "silhouette_score": float(sil)})
        _log.debug("  k=%d: silhouette=%.4f", k, sil)

    df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    _log.info(
        "Phase 1 terminée. Best k=%d (sil=%.4f)",
        df.loc[df["silhouette_score"].idxmax(), "k"],
        df["silhouette_score"].max(),
    )
    return df


def phase2_bootstrap_stability(
    X: np.ndarray,
    xdim: int,
    ydim: int,
    rlen: int,
    seed: int,
    candidates_k: List[int],
    n_bootstrap: int = 5,
    sample_size: Optional[int] = None,
) -> Dict[int, float]:
    """
    Phase 2 — Stabilité bootstrap (ARI) pour les k candidats.

    Pour chaque k candidat, entraîne n_bootstrap fois le SOM avec des graines
    différentes sur un sous-ensemble de données, calcule l'ARI entre chaque
    paire de runs. Un ARI moyen élevé = clustering stable.

    Args:
        X: Matrice (n_cells, n_markers).
        xdim, ydim: Dimensions grille SOM.
        rlen: Itérations SOM.
        seed: Graine de base (chaque run = seed + i).
        candidates_k: Liste des k à évaluer (≤5 pour rester rapide).
        n_bootstrap: Nombre de runs par k (5 = bon compromis vitesse/précision).
        sample_size: Taille de l'échantillon par run (None = X complet).

    Returns:
        Dict {k: mean_ari} — ARI moyen entre les n_bootstrap(n_bootstrap-1)/2 paires.
    """
    if not _ARI_AVAILABLE:
        raise ImportError("sklearn requis: pip install scikit-learn")

    _log = _get_logger_clustering()
    stability_results: Dict[int, float] = {}
    rng = np.random.default_rng(seed)

    for k in candidates_k:
        _log.info("Phase 2: stabilité bootstrap k=%d (%d runs)...", k, n_bootstrap)
        run_labels: List[np.ndarray] = []

        for i in range(n_bootstrap):
            run_seed = seed + 1000 * (k - 1) + i  # seeds déterministes distincts

            # Sous-échantillonnage optionnel pour accélérer (ELN: 20k cellules suffisent)
            X_run = X
            if sample_size is not None and sample_size < X.shape[0]:
                idx = rng.choice(X.shape[0], size=sample_size, replace=False)
                X_run = X[idx]

            result = _train_som_codebook(X_run, xdim, ydim, k, rlen, run_seed)
            if result is None:
                _log.warning("  Run %d/k=%d: SOM échoué, ignoré.", i, k)
                continue

            _, node_assignments = result
            # Convertir node_assignments → label par métacluster
            codebook_tmp, _ = result
            meta_labels = _metacluster_codebook(codebook_tmp, k)
            cell_meta_labels = meta_labels[node_assignments]
            run_labels.append(cell_meta_labels)

        # ARI pairwise entre tous les runs
        ari_scores = []
        for a in range(len(run_labels)):
            for b in range(a + 1, len(run_labels)):
                n_min = min(len(run_labels[a]), len(run_labels[b]))
                ari = _ari_score(run_labels[a][:n_min], run_labels[b][:n_min])
                ari_scores.append(ari)

        mean_ari = float(np.mean(ari_scores)) if ari_scores else 0.0
        stability_results[k] = mean_ari
        _log.info("  k=%d: ARI moyen=%.4f sur %d paires", k, mean_ari, len(ari_scores))

    return stability_results


def phase3_composite_selection(
    sil_df: "pd.DataFrame",
    stability_results: Dict[int, float],
    w_stability: float = 0.6,
    w_silhouette: float = 0.4,
    min_stability: float = 0.7,
) -> Tuple[int, "pd.DataFrame"]:
    """
    Phase 3 — Score composite stabilité + silhouette pour choisir le k final.

    Score = w_stability × ARI_normalisé + w_silhouette × Sil_normalisé
    Pénalité 0.7× sur les k dont la stabilité < min_stability.

    Args:
        sil_df: Résultat phase 1 — DataFrame [k, silhouette_score].
        stability_results: Résultat phase 2 — dict {k: mean_ari}.
        w_stability: Poids de la stabilité (default 0.6).
        w_silhouette: Poids de la silhouette (default 0.4).
        min_stability: Seuil minimal de stabilité pour éviter la pénalité.

    Returns:
        Tuple (best_k, scores_df) où scores_df = [k, sil, ari, composite_score].
    """
    import pandas as pd

    _log = _get_logger_clustering()

    # Aligner sur les k communs entre phase 1 et phase 2
    k_common = [k for k in stability_results if k in sil_df["k"].values]
    if not k_common:
        # Fallback: juste phase 1
        best_k = int(sil_df.loc[sil_df["silhouette_score"].idxmax(), "k"])
        _log.warning(
            "Phase 3: aucun k commun phase1/phase2 — fallback silhouette k=%d", best_k
        )
        return best_k, sil_df

    sil_map = dict(zip(sil_df["k"], sil_df["silhouette_score"]))

    ari_vals = np.array([stability_results[k] for k in k_common])
    sil_vals = np.array([sil_map[k] for k in k_common])

    # Normalisation min-max (robuste aux plages différentes)
    def _norm(arr: np.ndarray) -> np.ndarray:
        rng_ = arr.max() - arr.min()
        return (arr - arr.min()) / max(rng_, 1e-9)

    ari_norm = _norm(ari_vals)
    sil_norm = _norm(sil_vals)

    composite = w_stability * ari_norm + w_silhouette * sil_norm

    # Pénaliser les k instables
    for i, k in enumerate(k_common):
        if stability_results[k] < min_stability:
            composite[i] *= 0.7
            _log.debug(
                "  k=%d: pénalisé (ARI=%.3f < %.2f)",
                k,
                stability_results[k],
                min_stability,
            )

    best_idx = int(np.argmax(composite))
    best_k = k_common[best_idx]

    scores_df = (
        pd.DataFrame(
            {
                "k": k_common,
                "silhouette_score": sil_vals,
                "ari_stability": ari_vals,
                "composite_score": composite,
            }
        )
        .sort_values("k")
        .reset_index(drop=True)
    )

    _log.info(
        "Phase 3 terminée. Best k=%d — sil=%.4f, ARI=%.4f, composite=%.4f",
        best_k,
        sil_map[best_k],
        stability_results[best_k],
        float(composite[best_idx]),
    )
    return best_k, scores_df


def find_optimal_clusters_stability(
    X: np.ndarray,
    seed: int = 42,
    xdim: int = 10,
    ydim: int = 10,
    rlen: Any = "auto",
    k_range: Optional[List[int]] = None,
    n_bootstrap: int = 5,
    bootstrap_sample_size: Optional[int] = 20000,
    top_n_phase2: int = 3,
    w_stability: float = 0.6,
    w_silhouette: float = 0.4,
    min_stability: float = 0.7,
) -> Tuple[int, int, int, int]:
    """
    Optimisation en 3 phases du nombre optimal de métaclusters FlowSOM.

    Phase 1 (rapide) : Silhouette sur codebook SOM — sélectionne les top k.
    Phase 2 (coûteux) : Bootstrap ARI sur les `top_n_phase2` candidats.
    Phase 3 (rapide) : Score composite stabilité+silhouette pour le choix final.

    Protocole conforme ELN 2022: le SOM est entraîné sur l'ensemble, le
    silhouette est calculé sur les nœuds (pas les cellules) pour la robustesse.

    Args:
        X: Matrice (n_cells, n_markers) transformée+normalisée (sans NaN/Inf).
        seed: Graine aléatoire (tous les runs utilisent seed+offset).
        xdim, ydim: Dimensions de la grille SOM.
        rlen: Itérations SOM ("auto" ou entier).
        k_range: Liste des k à évaluer (default: 4..15).
        n_bootstrap: Runs bootstrap par k candidat en phase 2.
        bootstrap_sample_size: Cellules par run bootstrap (None = tout).
        top_n_phase2: Nombre de k candidats passés en phase 2.
        w_stability, w_silhouette: Poids du score composite (somme ≈ 1.0).
        min_stability: ARI minimal sous lequel le k subit une pénalité ×0.7.

    Returns:
        Tuple (best_k, rlen_final, xdim_final, ydim_final).
    """
    _log = _get_logger_clustering()

    if k_range is None:
        k_range = list(range(4, 16))  # défaut biologique pour la moelle osseuse

    n_cells = X.shape[0]
    rlen_final = compute_optimal_rlen(n_cells, rlen)
    xdim_final, ydim_final = compute_optimal_grid(n_cells, xdim, ydim)

    _log.info(
        "Optimisation clusters: %d cellules, grille %dx%d, rlen=%d, k_range=%s",
        n_cells,
        xdim_final,
        ydim_final,
        rlen_final,
        k_range,
    )

    # ── Phase 1: silhouette rapide sur codebook ───────────────────────────────
    sil_df = phase1_silhouette_on_codebook(
        X,
        xdim=xdim_final,
        ydim=ydim_final,
        rlen=rlen_final,
        seed=seed,
        k_range=k_range,
    )

    # Top k_range candidats pour la phase 2 (coûteuse)
    candidates_k = sil_df.nlargest(top_n_phase2, "silhouette_score")["k"].tolist()
    _log.info("Phase 2 candidats: %s", candidates_k)

    # ── Phase 2: bootstrap ARI sur les top candidats ─────────────────────────
    stability_results = phase2_bootstrap_stability(
        X,
        xdim=xdim_final,
        ydim=ydim_final,
        rlen=rlen_final,
        seed=seed,
        candidates_k=candidates_k,
        n_bootstrap=n_bootstrap,
        sample_size=bootstrap_sample_size,
    )

    # ── Phase 3: score composite pour le choix final ─────────────────────────
    best_k, scores_df = phase3_composite_selection(
        sil_df=sil_df,
        stability_results=stability_results,
        w_stability=w_stability,
        w_silhouette=w_silhouette,
        min_stability=min_stability,
    )

    _log.info(
        "Optimisation terminée: best_k=%d, grille=%dx%d, rlen=%d",
        best_k,
        xdim_final,
        ydim_final,
        rlen_final,
    )
    return best_k, rlen_final, xdim_final, ydim_final
