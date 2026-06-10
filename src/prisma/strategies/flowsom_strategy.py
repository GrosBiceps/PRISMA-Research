"""
prisma/strategies/flowsom_strategy.py — FlowSOM natif (saeyslab) + proxy sklearn.

Backend prioritaire : saeyslab flowsom (FlowSOM + ConsensusCluster).
Fallback automatique  : MiniBatchKMeans (SOM proxy) + ConsensusCluster ou Agglomerative.

Enregistré dans StrategyRegistry sous "flowsom".

API complète saeyslab exposée :
  FlowSOMParams.rlen, .mst, .alpha, .mad_allowed
  FlowSOMParams.consensus_H, .consensus_resample, .consensus_linkage
  Stockage : node_labels, metacluster_labels, codes SOM, MFI, counts, node_meta

Compatible avec Sample (src/models/sample.py) via add_cluster_labels().
Toutes les données utiles stockées dans sample.results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from prisma.core.registry import StrategyRegistry
from prisma.strategies.base import ClusterParams

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

try:
    import flowsom as _fs_lib
    _FLOWSOM_AVAILABLE = True
    logger.info("flowsom (saeyslab) disponible")
except ImportError:
    _FLOWSOM_AVAILABLE = False
    logger.warning("flowsom non installé — proxy sklearn actif")

try:
    from flowsom.models import ConsensusCluster as _ConsensusCluster
    _CONSENSUS_AVAILABLE = True
except Exception:
    _CONSENSUS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------

@dataclass
class FlowSOMParams(ClusterParams):
    """
    Paramètres complets FlowSOM.

    Tous les paramètres du constructeur saeyslab FlowSOM sont exposés.
    Les paramètres du ConsensusCluster (métaclustering) sont également inclus.
    """
    # — SOM grid
    xdim: int = 10
    ydim: int = 10
    # — Entraînement SOM
    rlen: int = 10
    mst: int = 1
    alpha: Tuple[float, float] = (0.05, 0.01)
    # — Sécurité
    mad_allowed: int = 4
    # — Sélection de marqueurs
    markers: Optional[List[str]] = None
    # — ConsensusCluster (métaclustering)
    n_metaclusters: int = 20
    consensus_H: int = 100
    consensus_resample: float = 0.9
    consensus_linkage: str = "average"
    # — Stockage dans Sample
    som_label: str = "flowsom_nodes"
    meta_label: str = "flowsom_meta"
    store_codes: bool = True
    store_mfi: bool = True
    compute_centers: bool = True


# ---------------------------------------------------------------------------
# FlowSOMResult — snapshot des résultats bruts
# ---------------------------------------------------------------------------

@dataclass
class FlowSOMResult:
    """
    Résultats bruts d'un run FlowSOM, avant stockage dans Sample.

    Attributes:
        node_labels: ndarray (n_cells,) — nœud SOM par cellule.
        metacluster_labels: ndarray (n_cells,) — métacluster par cellule.
        codes: ndarray (n_nodes, n_features) — codebook SOM. None si non dispo.
        node_meta: ndarray (n_nodes,) — quel métacluster pour chaque nœud.
        mfi: dict {metacluster_id: median_expression (list)} ou None.
        counts: dict {metacluster_id: n_cells}.
        backend: "saeyslab" | "proxy".
        fsom_object: objet FlowSOM saeyslab (None si proxy).
    """
    node_labels: np.ndarray
    metacluster_labels: np.ndarray
    codes: Optional[np.ndarray]
    node_meta: np.ndarray
    mfi: Optional[Dict[int, List[float]]]
    counts: Dict[int, int]
    backend: str
    fsom_object: Any = None


# ---------------------------------------------------------------------------
# FlowSOMStrategy
# ---------------------------------------------------------------------------

@StrategyRegistry.register_clustering("flowsom")
class FlowSOMStrategy:
    """
    Stratégie FlowSOM complète pour PRISMA Research.

    Implémente le Protocol ClusterStrategy (fit_predict) ET expose
    run(sample) pour intégration avec ResearchPipelineExecutor.

    Backends :
      1. saeyslab flowsom  — FlowSOM natif + ConsensusCluster
      2. proxy sklearn     — MiniBatchKMeans + ConsensusCluster (si dispo) ou Agglomerative

    Toutes les méthodes de l'API saeyslab sont wrappées et accessibles
    après un appel à fit_predict() ou run() via self.result_.
    """

    name: str = "flowsom"

    def __init__(self) -> None:
        self.result_: Optional[FlowSOMResult] = None
        self._params_: Optional[FlowSOMParams] = None

    # ------------------------------------------------------------------
    # Protocol ClusterStrategy
    # ------------------------------------------------------------------

    def fit_predict(self, data: np.ndarray, params: ClusterParams) -> np.ndarray:
        """
        Entraîne FlowSOM et retourne les labels de métaclusters (n_cells,).

        Compatible avec le Protocol ClusterStrategy de PRISMA.
        Stocke le résultat complet dans self.result_.

        Args:
            data: Matrice (n_cells × n_features), dtype float32/float64.
            params: FlowSOMParams ou ClusterParams générique.

        Returns:
            Labels métaclusters int32 (n_cells,).
        """
        p = self._coerce_params(params)
        self._params_ = p

        if _FLOWSOM_AVAILABLE:
            result = self._fit_saeyslab(data, p)
        else:
            result = self._fit_proxy(data, p)

        self.result_ = result
        return result.metacluster_labels

    # ------------------------------------------------------------------
    # Méthodes API saeyslab accessibles après fit_predict()
    # ------------------------------------------------------------------

    def get_node_labels(self) -> np.ndarray:
        """Labels nœuds SOM par cellule (n_cells,)."""
        self._check_fitted()
        return self.result_.node_labels

    def get_metacluster_labels(self) -> np.ndarray:
        """Labels métaclusters par cellule (n_cells,)."""
        self._check_fitted()
        return self.result_.metacluster_labels

    def get_codes(self) -> Optional[np.ndarray]:
        """Codebook SOM (n_nodes × n_features). None si non disponible."""
        self._check_fitted()
        return self.result_.codes

    def get_node_metacluster_mapping(self) -> np.ndarray:
        """Pour chaque nœud SOM : quel métacluster (n_nodes,)."""
        self._check_fitted()
        return self.result_.node_meta

    def get_mfi(self) -> Optional[Dict[int, List[float]]]:
        """MFI médian par métacluster. None si compute_centers=False."""
        self._check_fitted()
        return self.result_.mfi

    def get_counts(self) -> Dict[int, int]:
        """Nombre de cellules par métacluster."""
        self._check_fitted()
        return self.result_.counts

    def get_percentages(self) -> Dict[int, float]:
        """Pourcentage de cellules par métacluster."""
        self._check_fitted()
        counts = self.result_.counts
        total = sum(counts.values())
        if total == 0:
            return {k: 0.0 for k in counts}
        return {k: v / total * 100.0 for k, v in counts.items()}

    def get_fsom_object(self) -> Any:
        """Objet FlowSOM saeyslab brut (None si backend proxy)."""
        self._check_fitted()
        return self.result_.fsom_object

    def map_new_data(self, data: np.ndarray) -> np.ndarray:
        """
        Projette de nouvelles données sur le SOM existant.

        Backend saeyslab : FlowSOM.new_data().
        Backend proxy : map_data_to_codes() → node → métacluster.

        Returns:
            Labels métaclusters (n_new_cells,).
        """
        self._check_fitted()
        if self.result_.backend == "saeyslab" and self.result_.fsom_object is not None:
            import anndata
            fsom = self.result_.fsom_object
            n_feat = data.shape[1]
            cols = [f"f{i}" for i in range(n_feat)]
            adata = anndata.AnnData(X=data.astype(np.float32))
            adata.var_names = cols
            fsom_new = fsom.new_data(adata)
            return np.array(fsom_new.get_cell_data().obs["metaclustering"], dtype=np.int32)

        return self._map_proxy(data)

    def test_outliers(
        self,
        mad_allowed: Optional[int] = None,
        fsom_reference: Any = None,
    ) -> Optional[np.ndarray]:
        """
        Détecte les cellules outliers via MAD (backend saeyslab seulement).

        Returns:
            Masque booléen (n_cells,) True = outlier. None si backend proxy.
        """
        self._check_fitted()
        if self.result_.backend != "saeyslab" or self.result_.fsom_object is None:
            logger.warning("test_outliers: disponible uniquement avec le backend saeyslab.")
            return None
        p = self._params_
        mad = mad_allowed if mad_allowed is not None else (p.mad_allowed if p else 4)
        fsom = self.result_.fsom_object
        outlier_result = fsom.test_outliers(mad_allowed=mad, fsom_reference=fsom_reference)
        return outlier_result

    def get_cluster_percentages_positive(
        self,
        cutoffs: Dict[str, float],
    ) -> Optional[Any]:
        """
        Pourcentage de cellules positives par cluster (backend saeyslab).

        Args:
            cutoffs: Dict {marker_name: threshold}

        Returns:
            DataFrame ou None si backend proxy.
        """
        self._check_fitted()
        if self.result_.backend != "saeyslab" or self.result_.fsom_object is None:
            logger.warning("get_cluster_percentages_positive: backend saeyslab requis.")
            return None
        from flowsom.tl import get_cluster_percentages_positive
        return get_cluster_percentages_positive(self.result_.fsom_object, cutoffs)

    def get_metacluster_percentages_positive(
        self,
        cutoffs: Dict[str, float],
    ) -> Optional[Any]:
        """Pourcentage de cellules positives par métacluster (backend saeyslab)."""
        self._check_fitted()
        if self.result_.backend != "saeyslab" or self.result_.fsom_object is None:
            logger.warning("get_metacluster_percentages_positive: backend saeyslab requis.")
            return None
        from flowsom.tl import get_metacluster_percentages_positive
        return get_metacluster_percentages_positive(self.result_.fsom_object, cutoffs)

    def get_features(
        self,
        files: List[Any],
        level: List[str] = ("clusters", "metaclusters"),
        feat_type: List[str] = ("counts",),
        MFI: Optional[List[str]] = None,
        positive_cutoffs: Optional[Dict[str, float]] = None,
    ) -> Optional[Any]:
        """
        Extraction de features multi-niveaux (backend saeyslab).

        Args:
            files: Liste de fichiers/AnnData.
            level: ["clusters", "metaclusters"].
            feat_type: ["counts", "percentages", "MFI"].
            MFI: Marqueurs pour MFI.
            positive_cutoffs: Seuils de positivité.

        Returns:
            DataFrame ou None.
        """
        self._check_fitted()
        if self.result_.backend != "saeyslab" or self.result_.fsom_object is None:
            logger.warning("get_features: backend saeyslab requis.")
            return None
        from flowsom.tl import get_features
        return get_features(
            self.result_.fsom_object,
            files=files,
            level=list(level),
            type=list(feat_type),
            MFI=MFI,
            positive_cutoffs=positive_cutoffs,
        )

    # ------------------------------------------------------------------
    # Backends internes
    # ------------------------------------------------------------------

    def _fit_saeyslab(self, data: np.ndarray, p: FlowSOMParams) -> FlowSOMResult:
        import anndata
        import flowsom as fs

        n_cells, n_feat = data.shape
        logger.info(
            "[FlowSOM/saeyslab] %d × %d, SOM=%dx%d, méta=%d, rlen=%d, seed=%s",
            n_cells, n_feat, p.xdim, p.ydim, p.n_metaclusters, p.rlen, p.seed,
        )

        cols = p.markers if p.markers and len(p.markers) == n_feat else [f"f{i}" for i in range(n_feat)]
        adata = anndata.AnnData(X=data.astype(np.float32))
        adata.var_names = cols

        fsom = fs.FlowSOM(
            adata,
            cols_to_use=cols,
            xdim=p.xdim,
            ydim=p.ydim,
            n_clusters=p.n_metaclusters,
            rlen=p.rlen,
            mst=p.mst,
            alpha=p.alpha,
            seed=p.seed,
        )

        cell_data = fsom.get_cell_data()
        cluster_data = fsom.get_cluster_data()

        node_labels = np.array(cell_data.obs["clustering"], dtype=np.int32)
        metacluster_labels = np.array(cell_data.obs["metaclustering"], dtype=np.int32)

        codes = None
        if "codes" in cluster_data.obsm:
            codes = np.array(cluster_data.obsm["codes"], dtype=np.float32)

        node_meta_raw = cluster_data.obs.get("metaclustering", None)
        node_meta = np.array(node_meta_raw, dtype=np.int32) if node_meta_raw is not None else np.zeros(p.xdim * p.ydim, dtype=np.int32)

        mfi = self._compute_mfi(data, metacluster_labels) if p.compute_centers else None
        counts = self._compute_counts(metacluster_labels)

        logger.info(
            "[FlowSOM/saeyslab] %d nœuds SOM, %d métaclusters",
            p.xdim * p.ydim, p.n_metaclusters,
        )

        return FlowSOMResult(
            node_labels=node_labels,
            metacluster_labels=metacluster_labels,
            codes=codes,
            node_meta=node_meta,
            mfi=mfi,
            counts=counts,
            backend="saeyslab",
            fsom_object=fsom,
        )

    def _fit_proxy(self, data: np.ndarray, p: FlowSOMParams) -> FlowSOMResult:
        from sklearn.cluster import MiniBatchKMeans

        n_nodes = p.xdim * p.ydim
        n_cells, n_feat = data.shape
        n_meta = min(p.n_metaclusters, n_nodes)

        logger.info(
            "[FlowSOM/proxy] %d × %d, SOM=%dx%d (%d nœuds), méta=%d",
            n_cells, n_feat, p.xdim, p.ydim, n_nodes, n_meta,
        )

        # — Étape 1 : SOM via MiniBatchKMeans
        som = MiniBatchKMeans(
            n_clusters=n_nodes,
            random_state=p.seed,
            n_init=3,
            max_iter=300,
        )
        node_labels = som.fit_predict(data).astype(np.int32)
        codes = som.cluster_centers_.astype(np.float32)

        # — Étape 2 : Métaclustering sur les codes
        node_meta = self._metacluster_codes(codes, n_meta, p)
        metacluster_labels = node_meta[node_labels]

        mfi = self._compute_mfi(data, metacluster_labels) if p.compute_centers else None
        counts = self._compute_counts(metacluster_labels)

        logger.info(
            "[FlowSOM/proxy] %d nœuds → %d métaclusters",
            n_nodes, n_meta,
        )

        return FlowSOMResult(
            node_labels=node_labels,
            metacluster_labels=metacluster_labels,
            codes=codes,
            node_meta=node_meta,
            mfi=mfi,
            counts=counts,
            backend="proxy",
            fsom_object=None,
        )

    def _metacluster_codes(
        self,
        codes: np.ndarray,
        n_meta: int,
        p: FlowSOMParams,
    ) -> np.ndarray:
        """ConsensusCluster sur les codes SOM, fallback Agglomerative."""
        if _CONSENSUS_AVAILABLE:
            try:
                cc = _ConsensusCluster(
                    n_clusters=n_meta,
                    H=p.consensus_H,
                    resample_proportion=p.consensus_resample,
                    linkage=p.consensus_linkage,
                )
                logger.info("[FlowSOM/proxy] ConsensusCluster H=%d", p.consensus_H)
                return cc.fit_predict(codes).astype(np.int32)
            except Exception as exc:
                logger.warning("[FlowSOM/proxy] ConsensusCluster échoué (%s) — Agglomerative", exc)

        from sklearn.cluster import AgglomerativeClustering
        agg = AgglomerativeClustering(n_clusters=n_meta)
        return agg.fit_predict(codes).astype(np.int32)

    def _map_proxy(self, data: np.ndarray) -> np.ndarray:
        """Projection de nouvelles données sur le SOM proxy."""
        self._check_fitted()
        codes = self.result_.codes
        if codes is None:
            raise RuntimeError("Codes SOM non disponibles pour la projection.")
        # Distance euclidienne vers le nœud le plus proche
        diffs = data[:, np.newaxis, :] - codes[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        node_assignments = np.argmin(dists, axis=1).astype(np.int32)
        return self.result_.node_meta[node_assignments]

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_mfi(
        data: np.ndarray,
        metacluster_labels: np.ndarray,
    ) -> Dict[int, List[float]]:
        valid = metacluster_labels >= 0
        mfi = {}
        for m in np.unique(metacluster_labels[valid]):
            mask = (metacluster_labels == m) & valid
            mfi[int(m)] = np.median(data[mask], axis=0).tolist()
        return mfi

    @staticmethod
    def _compute_counts(metacluster_labels: np.ndarray) -> Dict[int, int]:
        valid = metacluster_labels[metacluster_labels >= 0]
        counts = {}
        for m in np.unique(valid):
            counts[int(m)] = int(np.sum(valid == m))
        return counts

    @staticmethod
    def _coerce_params(params: ClusterParams) -> FlowSOMParams:
        if isinstance(params, FlowSOMParams):
            return params
        return FlowSOMParams(
            n_clusters=params.n_clusters,
            seed=params.seed,
            n_jobs=params.n_jobs,
        )

    def _check_fitted(self) -> None:
        if self.result_ is None:
            raise RuntimeError(
                "FlowSOMStrategy non entraîné. Appelez fit_predict() d'abord."
            )
