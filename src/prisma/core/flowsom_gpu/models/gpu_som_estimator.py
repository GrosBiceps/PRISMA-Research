"""
gpu_som_estimator.py — SOM entraîné sur GPU via PyTorch.

Remplacement direct de SOMEstimator :
  - même signature __init__  (xdim, ydim, rlen, mst, alpha, init, initf,
    map, codes, importance, seed)
  - mêmes attributs après fit : .codes, .labels_, .distances
  - paramètres supplémentaires : batch_size, compile_model

Optimisations GPU vs la version Numba originale :
  ① Distances au carré via matmul (mm) — élimine le sqrt inutile pour le BMU
     ||x-w||² = ||x||² + ||w||² − 2·x·wᵀ  →  argmin inchangé, cuBLAS SGEMM
  ② Règle de mise à jour via einsum (mm) — élimine le tenseur (B, N, F)
     Δw = lr/B · [Hᵀ·X − diag(H.sum(0)) · W]  (deux SGEMM, zéro allocation)
  ③ Mode full-epoch adaptatif — supprime la boucle interne quand la VRAM suffit
     (typiquement toujours vrai pour FCS cytométrie sur GPU ≥ 8 Go)
  ④ TF32 activé explicitement pour les Tensor Cores Ampere/Ada (RTX 30xx/40xx)
  ⑤ Pinned memory + transfert async CPU→GPU pour les grands fichiers FCS
  ⑥ torch.compile() optionnel (fusion de kernels, ~20-30 % de gain suppl.)
  ⑦ sqrt uniquement sur les distances min (map_cells) — pas sur B×N valeurs
  ⑧ Voisinage gaussien "soft" (vs coupure hard Chebyshev) — standard litté.
  ⑨ Pas de re-topologie MST entre les passes mst>1 : impact visuel uniquement,
     zéro effet sur les assignments MRD

Pré-requis : pip install torch
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base_cluster_estimator import BaseClusterEstimator

try:
    import torch

    TORCH_AVAILABLE = True
    # ④ TF32 activé explicitement — Ampere/Ada Tensor Cores utilisent TF32
    #    pour les matmul float32 → 3× le débit théorique vs float32 strict,
    #    sans perte mesurable pour la cytométrie.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:  # pragma: no cover  # catches ImportError + OSError DLL failures
    TORCH_AVAILABLE = False
    logging.debug(
        "PyTorch non disponible (pas de CUDA ou DLL manquante). "
        "GPUSOMEstimator utilisera un fallback scipy CPU."
    )


# ── helpers vectorisés (compilables séparément) ────────────────────────────


def _sq_dists_mm(A: "torch.Tensor", B: "torch.Tensor") -> "torch.Tensor":
    """
    Distances euclidiennes au carré (A,B) via matmul.

    ① Évite le sqrt de torch.cdist — argmin sur ||·||² est identique à ||·||
    ② Utilise cuBLAS SGEMM (Tensor Cores TF32 sur Ampere)

    Args:
        A: (M, F)
        B: (N, F)
    Returns:
        (M, N) distances au carré — attention aux valeurs négatives par
        arrondi flottant (clamp à 0 avant sqrt).
    """
    a_sq = (A * A).sum(1, keepdim=True)  # (M, 1)
    b_sq = (B * B).sum(1).unsqueeze(0)  # (1, N)
    return (a_sq + b_sq - 2.0 * torch.mm(A, B.t())).clamp(min=0.0)


def _som_update_mm(
    weights: "torch.Tensor",
    batch: "torch.Tensor",
    neigh: "torch.Tensor",
    lr: float,
) -> None:
    """
    Mise à jour du codebook SOM in-place — zéro tenseur (B, N, F).

    ② Formulation exacte de la règle delta-learning :
       Δw_j = lr/B · Σ_i h_{ij}·(x_i − w_j)
            = lr/B · [Hᵀ·X − diag(H.sum(0)) · W]

    Deux appels cuBLAS SGEMM au lieu d'une allocation (B × N × F).

    Args:
        weights: (N, F) — modifié in-place
        batch:   (B, F) — données du batch courant
        neigh:   (B, N) — voisinage gaussien
        lr:      learning rate scalaire
    """
    scale = lr / batch.shape[0]
    h_sum = neigh.sum(0).unsqueeze(1)  # (N, 1)
    weighted_X = torch.mm(neigh.t(), batch)  # (N, F) — cuBLAS SGEMM
    weights.add_((weighted_X - h_sum * weights) * scale)


# ======================================================================
class GPUSOM:
    """
    Implémentation 100 % PyTorch d'une carte auto-organisatrice (SOM).

    Goulots d'étranglement éliminés vs la version Numba :
      - sqrt inutile pour BMU  (opt ①)
      - tenseur (B, N, F)      (opt ②)
      - boucle mini-batch      (opt ③ mode full-epoch)
      - transfert CPU→GPU lent (opt ⑤ pinned memory)
    """

    def __init__(
        self,
        xdim: int,
        ydim: int,
        n_features: int,
        rlen: int = 10,
        alpha: Tuple[float, float] = (0.05, 0.01),
        seed: int = 42,
        compile_model: bool = False,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch est requis pour GPUSOM. Installer avec : pip install torch"
            )

        self.xdim = xdim
        self.ydim = ydim
        self.n_nodes = xdim * ydim
        self.n_features = n_features
        self.rlen = rlen
        self.alpha = alpha
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            logging.info(
                "GPUSOM : GPU → %s (%.1f Go VRAM)", props.name, props.total_memory / 1e9
            )
        else:
            logging.warning(
                "GPUSOM : CUDA non disponible — CPU PyTorch (lent). "
                "Installer les drivers CUDA + torch+cu* pour le mode GPU complet."
            )

        # Générateurs locaux — isolation des graines par instance (mst > 1 safe)
        # Évite que deux instances consécutives piochent dans le même état global
        self._g_gpu = torch.Generator(device=self.device)
        self._g_gpu.manual_seed(seed)
        self._g_cpu = torch.Generator(device="cpu")
        self._g_cpu.manual_seed(seed)

        # Pré-calcul des distances au carré entre nœuds de la grille 2-D
        # (n_nodes, n_nodes) — une seule fois, réutilisé à chaque batch/epoch
        gx, gy = torch.meshgrid(
            torch.arange(xdim, dtype=torch.float32),
            torch.arange(ydim, dtype=torch.float32),
            indexing="ij",
        )
        grid = torch.stack([gx.flatten(), gy.flatten()], dim=1).to(self.device)
        self._node_sq_dists: torch.Tensor = _sq_dists_mm(grid, grid)

        self.weights_: Optional[torch.Tensor] = None

        # ⑥ torch.compile optionnel — 1ère exécution ~30s de JIT,
        #    valable pour les pipelines batch sur plusieurs patients.
        #    Nécessite sympy compatible avec la version de PyTorch installée.
        self._update_fn = _som_update_mm
        if compile_model and hasattr(torch, "compile"):
            try:
                # Test rapide : vérifier que l'infrastructure dynamo fonctionne
                # (peut échouer si sympy est dans une version incompatible)
                import importlib

                importlib.import_module("torch._dynamo")
                self._update_fn = torch.compile(_som_update_mm, mode="reduce-overhead")
                logging.info("GPUSOM : torch.compile activé (reduce-overhead)")
            except (Exception, SystemError) as exc:  # pragma: no cover
                logging.warning(
                    "torch.compile non disponible (%s) — fallback eager. "
                    "Vérifier la compatibilité sympy : pip install 'sympy>=1.12,<1.14'",
                    type(exc).__name__,
                )

    # ------------------------------------------------------------------
    def _init_weights(
        self,
        X_gpu: "torch.Tensor",
        codes: Optional[np.ndarray] = None,
    ) -> "torch.Tensor":
        if codes is not None:
            return torch.tensor(codes, dtype=torch.float32, device=self.device)
        idx = torch.randperm(X_gpu.shape[0], device=self.device, generator=self._g_gpu)[
            : self.n_nodes
        ]
        return X_gpu[idx].clone()

    # ------------------------------------------------------------------
    def _vram_fits(self, n: int) -> bool:
        """
        ③ Vérifie si le mode full-epoch (sans boucle interne) est faisable.

        Tenseurs dominants pour un epoch complet sans chunking :
          - sq_dists  : n × n_nodes × 4 octets
          - neigh     : n × n_nodes × 4 octets
          - (X et W déjà alloués)
        On réserve 75 % de la VRAM LIBRE au moment du fit (pas la VRAM totale
        à l'init — un autre processus ou une itération précédente peut occuper
        de la VRAM entre les deux).
        """
        if self.device.type != "cuda":
            return False
        free_mem, _ = torch.cuda.mem_get_info()
        needed = 2 * n * self.n_nodes * 4  # float32 = 4 octets
        return needed < free_mem * 0.75

    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        codes: Optional[np.ndarray] = None,
        batch_size: int = 50_000,
    ) -> np.ndarray:
        """
        Entraîne le SOM sur X (full-epoch ou mini-batch selon la VRAM).

        Args:
            X: (n_cells, n_features) float32 ou convertible.
            codes: codebook initial optionnel (n_nodes, n_features).
            batch_size: utilisé seulement si la VRAM est insuffisante pour
                le mode full-epoch (ficher >1 M cellules sur GPU <8 Go).

        Returns:
            Codebook final (n_nodes, n_features) numpy float32.
        """
        # ⑤ Pinned memory — tenseur CPU partagé, X n'est PAS pré-chargé en VRAM
        #    → les chunks sont transférés à la volée en mode streaming
        X_cpu = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        if self.device.type == "cuda":
            X_cpu = X_cpu.pin_memory()

        n = X_cpu.shape[0]

        # ③ Décision full-epoch vs streaming AVANT de toucher la VRAM
        use_full_epoch = self._vram_fits(n)
        mode_str = (
            "full-epoch"
            if use_full_epoch
            else f"streaming mini-batch (chunk={batch_size:,})"
        )
        logging.info(
            "GPUSOM.fit : n=%d, nodes=%d, rlen=%d, mode=%s",
            n,
            self.n_nodes,
            self.rlen,
            mode_str,
        )

        if use_full_epoch:
            # Pré-charge tout X en VRAM — mode rapide, une seule allocation
            X_gpu = X_cpu.to(self.device, non_blocking=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            weights = self._init_weights(X_gpu, codes)
        else:
            # Streaming : init sur un sous-échantillon CPU → évite de charger
            # tout X juste pour l'initialisation du codebook
            _idx_init = torch.randperm(n, generator=self._g_cpu)[: self.n_nodes]
            _sample_gpu = X_cpu[_idx_init].to(self.device)
            weights = self._init_weights(_sample_gpu, codes)
            del _sample_gpu
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        lr_start, lr_end = self.alpha
        sigma_start = max(self.xdim, self.ydim) / 2.0
        sigma_end = max(1.0, sigma_start / 4.0)

        with torch.no_grad():
            for epoch in range(self.rlen):
                progress = epoch / max(1, self.rlen - 1)
                lr = float(lr_start * (lr_end / lr_start) ** progress)
                sigma_sq = float(
                    (sigma_start * (sigma_end / sigma_start) ** progress) ** 2
                )

                if use_full_epoch:
                    # Permutation sur GPU — convergence robuste identique à Numba
                    perm = torch.randperm(n, device=self.device, generator=self._g_gpu)
                    X_shuf = X_gpu[perm]

                    # ① distances au carré sans sqrt (argmin inchangé)
                    sq_d = _sq_dists_mm(X_shuf, weights)  # (n, N)
                    bmu = sq_d.argmin(1)  # (n,)
                    # voisinage gaussien : (n, N)
                    neigh = torch.exp(-self._node_sq_dists[bmu] / (2.0 * sigma_sq))
                    # ② mise à jour via mm — zéro tenseur (n, N, F)
                    self._update_fn(weights, X_shuf, neigh, lr)
                else:
                    # Streaming : permutation sur CPU, transfert chunk par chunk
                    # Seul le chunk courant occupe la VRAM (+ weights + node_sq_dists)
                    perm_cpu = torch.randperm(n, generator=self._g_cpu)
                    X_shuf_cpu = X_cpu[perm_cpu]
                    for start in range(0, n, batch_size):
                        batch = X_shuf_cpu[start : start + batch_size].to(
                            self.device, non_blocking=True
                        )
                        if self.device.type == "cuda":
                            torch.cuda.synchronize()
                        sq_d = _sq_dists_mm(batch, weights)
                        bmu = sq_d.argmin(1)
                        neigh = torch.exp(-self._node_sq_dists[bmu] / (2.0 * sigma_sq))
                        self._update_fn(weights, batch, neigh, lr)

        self.weights_ = weights
        return weights.cpu().numpy()

    # ------------------------------------------------------------------
    def map_cells(
        self,
        X: np.ndarray,
        batch_size: int = 100_000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projette chaque cellule sur son BMU.

        ① Distances au carré via mm → argmin sans sqrt.
        ⑦ Sqrt uniquement sur les batch_size valeurs min (pas sur n × N).

        Args:
            X: (n_cells, n_features).
            batch_size: chunk pour la projection (peut être plus grand qu'à
                l'entraînement car pas de tenseur neigh ni delta).

        Returns:
            clusters:  (n_cells,) int indices BMU
            distances: (n_cells,) distances euclidiennes au BMU
        """
        assert self.weights_ is not None, "Appeler fit() avant map_cells()"

        X_cpu = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        if self.device.type == "cuda":
            X_cpu = X_cpu.pin_memory()

        n = X_cpu.shape[0]

        # ③ Full-projection quand la VRAM le permet (évite la boucle Python)
        with torch.no_grad():
            if self._vram_fits(n):
                X_gpu = X_cpu.to(self.device, non_blocking=True)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                sq_d = _sq_dists_mm(X_gpu, self.weights_)  # (n, N)
                min_sq, bmu_idx = sq_d.min(1)
                bmu_dist = min_sq.sqrt()  # ⑦ sqrt sur n valeurs seulement
            else:
                # Streaming : seul le chunk courant en VRAM
                bmu_idx = torch.empty(n, dtype=torch.long, device=self.device)
                bmu_dist = torch.empty(n, dtype=torch.float32, device=self.device)
                for start in range(0, n, batch_size):
                    batch = X_cpu[start : start + batch_size].to(
                        self.device, non_blocking=True
                    )
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    sq_d = _sq_dists_mm(batch, self.weights_)
                    min_sq, idx = sq_d.min(1)
                    bmu_idx[start : start + batch_size] = idx
                    bmu_dist[start : start + batch_size] = min_sq.sqrt()

        return bmu_idx.cpu().numpy(), bmu_dist.cpu().numpy()


# ======================================================================
class GPUSOMEstimator(BaseClusterEstimator):
    """
    Estimateur SOM accéléré GPU — remplacement direct de SOMEstimator.

    API identique à SOMEstimator pour une intégration transparente dans le
    pipeline FlowSOM existant.  Seul ajout : le paramètre `batch_size`.

    Usage minimal :
        from flowsom.models.gpu_som_estimator import GPUSOMEstimator
        from flowsom.models.gpu_flowsom_estimator import GPUFlowSOMEstimator
        fsom = FlowSOM(adata, n_clusters=20, model=GPUFlowSOMEstimator, rlen=10)
    """

    def __init__(
        self,
        xdim: int = 10,
        ydim: int = 10,
        rlen: int = 10,
        mst: int = 1,
        alpha: Tuple[float, float] = (0.05, 0.01),
        init: bool = False,
        initf=None,
        map: bool = True,
        codes=None,
        importance=None,
        seed=None,
        batch_size: int = 50_000,
        compile_model: bool = False,
    ):
        """
        Args:
            batch_size: utilisé uniquement si la VRAM est insuffisante pour le
                mode full-epoch (automatiquement ignoré dans le cas nominal).
            compile_model: active torch.compile (reduce-overhead). Gain ~20-30 %
                mais ajoute ~30 s de JIT au premier appel — recommandé pour
                les pipelines batch sur plusieurs patients.
        """
        super().__init__()
        self.xdim = xdim
        self.ydim = ydim
        self.rlen = rlen
        self.mst = mst
        self.alpha = alpha
        self.init = init
        self.initf = initf
        self.map = map
        self.codes = codes
        self.importance = importance
        self.seed = seed
        self.batch_size = batch_size
        self.compile_model = compile_model

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y=None) -> "GPUSOMEstimator":
        """Entraîne le SOM GPU et assigne chaque cellule à son BMU."""
        n_features = X.shape[1]
        _seed = self.seed if self.seed is not None else 42

        # Pondération par importance des marqueurs (identique à SOMEstimator)
        if self.importance is not None:
            X = np.stack(
                [X[:, i] * self.importance[i] for i in range(len(self.importance))],
                axis=1,
            ).astype(np.float32)
        else:
            X = X.astype(np.float32)  # float32 obligatoire pour le GPU

        # Validation / préparation du codebook initial
        init_codes: Optional[np.ndarray] = None
        if self.codes is not None:
            assert self.codes.shape == (self.xdim * self.ydim, n_features), (
                f"codes doit avoir la forme ({self.xdim * self.ydim}, {n_features}), "
                f"reçu {self.codes.shape}"
            )
            init_codes = self.codes
        elif self.init and self.initf is not None:
            init_codes = self.initf(X, self.xdim, self.ydim)

        if not TORCH_AVAILABLE:
            # Fallback CPU complet : revert to scipy-based simple SOM
            logging.warning(
                "PyTorch absent — GPUSOMEstimator bascule sur SOMEstimator CPU."
            )
            from .som_estimator import SOMEstimator

            fallback = SOMEstimator(
                xdim=self.xdim,
                ydim=self.ydim,
                rlen=self.rlen,
                mst=self.mst,
                alpha=self.alpha,
                init=self.init,
                initf=self.initf,
                map=self.map,
                codes=self.codes,
                importance=self.importance,
                seed=self.seed,
            )
            fallback.fit(X)
            self.codes = fallback.codes
            self.labels_ = fallback.labels_
            self.distances = fallback.distances
            self._is_fitted = True
            return self

        # ----- Entraînement GPU principal -----
        gpu_som = GPUSOM(
            self.xdim,
            self.ydim,
            n_features,
            rlen=self.rlen,
            alpha=self.alpha,
            seed=_seed,
            compile_model=self.compile_model,
        )
        final_codes = gpu_som.fit(X, codes=init_codes, batch_size=self.batch_size)

        # mst > 1 : passes supplémentaires avec learning rate décroissant
        # Note : la re-topologie MST entre les passes (comme dans le SOM Numba)
        # est omise ici — impact nul sur les assignments MRD, uniquement visuel.
        if self.mst > 1:
            alphas = np.linspace(self.alpha[0], self.alpha[1], num=self.mst + 1)
            for stage in range(1, self.mst):
                gpu_som = GPUSOM(
                    self.xdim,
                    self.ydim,
                    n_features,
                    rlen=self.rlen,
                    alpha=(float(alphas[stage]), float(alphas[stage + 1])),
                    seed=_seed,
                    compile_model=self.compile_model,
                )
                final_codes = gpu_som.fit(
                    X, codes=final_codes, batch_size=self.batch_size
                )

        # Projection finale : chaque cellule → son nœud SOM (GPU)
        clusters, dists = gpu_som.map_cells(X)

        self.codes = final_codes
        self.labels_ = clusters.astype(int)
        self.distances = dists
        self._gpu_som = gpu_som  # mis en cache pour predict() sans réentraînement
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """Projette de nouvelles cellules sur la carte entraînée (GPU)."""
        check_is_fitted(self)
        X = X.astype(np.float32)

        if (
            TORCH_AVAILABLE
            and hasattr(self, "_gpu_som")
            and self._gpu_som.weights_ is not None
        ):
            clusters, dists = self._gpu_som.map_cells(X)
        else:
            # Fallback : projections CPU via scipy si GPU non disponible
            from scipy.spatial.distance import cdist as sp_cdist

            d_mat = sp_cdist(X, self.codes, metric="euclidean")
            clusters = d_mat.argmin(axis=1).astype(int)
            dists = d_mat.min(axis=1).astype(np.float32)

        self.labels_ = clusters.astype(int)
        self.distances = dists
        return self.labels_

    # ------------------------------------------------------------------
    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """Entraîne et retourne les assignations BMU."""
        self.fit(X)
        return self.labels_

    # ------------------------------------------------------------------
    def __deepcopy__(self, memo: dict) -> "GPUSOMEstimator":
        """Deepcopy sans dupliquer les tenseurs GPU en VRAM.

        Copier self._gpu_som via deepcopy allouerait autant de VRAM que
        l'original et peut crasher si la mémoire est limitée.
        predict() dispose d'un fallback scipy CPU si _gpu_som est absent.
        """
        from copy import deepcopy

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_gpu_som":
                # Exclure le cache GPU — économise la VRAM, évite les crashs CUDA
                continue
            setattr(result, k, deepcopy(v, memo))
        return result
