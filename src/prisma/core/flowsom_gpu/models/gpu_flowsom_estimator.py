"""
gpu_flowsom_estimator.py — FlowSOMEstimator avec SOM accéléré GPU.

Remplacement direct de FlowSOMEstimator :
  - Utilise GPUSOMEstimator comme cluster_model par défaut
  - Le méta-clustering (ConsensusCluster) reste sur CPU :
    il ne traite que les n_nodes nœuds SOM (ex. 100 × 20 = 2000 flottants),
    pas les millions de cellules → aucun intérêt à le passer sur GPU.

Usage :
    from flowsom import FlowSOM
    from flowsom.models import GPUFlowSOMEstimator

    fsom = FlowSOM(
        adata,
        n_clusters=20,
        model=GPUFlowSOMEstimator,
        xdim=10, ydim=10, rlen=10,
        batch_size=20_000,  # ajuster selon VRAM disponible
    )
"""

from __future__ import annotations

from .base_flowsom_estimator import BaseFlowSOMEstimator
from .consensus_cluster import ConsensusCluster
from .gpu_som_estimator import GPUSOMEstimator


class GPUFlowSOMEstimator(BaseFlowSOMEstimator):
    """
    FlowSOM estimator utilisant GPUSOMEstimator pour l'étape SOM.

    Tous les kwargs sont distribués entre GPUSOMEstimator et ConsensusCluster
    via l'introspection des signatures (logique héritée de BaseFlowSOMEstimator).

    Paramètres supplémentaires acceptés (vs FlowSOMEstimator) :
        batch_size (int): taille des mini-batchs GPU (défaut 10_000).
            Augmenter pour de meilleures performances si la VRAM le permet.
            Réduire en cas d'erreur "CUDA out of memory".
    """

    def __init__(
        self,
        cluster_model: type = GPUSOMEstimator,
        metacluster_model: type = ConsensusCluster,
        **kwargs,
    ):
        super().__init__(
            cluster_model=cluster_model,
            metacluster_model=metacluster_model,
            **kwargs,
        )
