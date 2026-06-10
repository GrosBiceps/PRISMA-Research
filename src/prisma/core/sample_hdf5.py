"""
sample_hdf5.py — Modèle de données FCS out-of-core via HDF5.

Remplace le pd.DataFrame en mémoire de Sample par un fichier HDF5
avec compression LZF. Rien n'est chargé en RAM sauf les colonnes
explicitement demandées par l'utilisateur (lazy column-wise loading).

Workflow :
    1. Conversion FCS → HDF5 une seule fois via SampleHDF5.from_sample()
    2. Lecture lazy par SampleHDF5.load_columns() ou .load_for_plot()
    3. Compatible avec GatingMask pour filtrage sans chargement complet
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import h5py
import numpy as np

if TYPE_CHECKING:
    from prisma.core.models import Sample


CHUNK_SIZE = 65_536  # 64k rows par chunk — optimal I/O séquentiel


class SampleHDF5:
    """
    Données FCS stockées en HDF5 compressé (LZF).

    Chaque colonne est un dataset HDF5 indépendant → lecture column-wise
    sans charger les autres canaux. RAM = O(colonnes demandées × N_events).
    """

    __slots__ = ("_path", "_f", "columns", "n_events", "name", "condition",
                 "metadata", "sample_id")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_sample(cls, sample: "Sample", hdf5_path: str | Path) -> "SampleHDF5":
        """
        Convertit un Sample Pandas existant en fichier HDF5.
        Appeler une seule fois lors du premier chargement.
        """
        hdf5_path = Path(hdf5_path)
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(hdf5_path, "w") as f:
            for col in sample.data.columns:
                arr = sample.data[col].to_numpy(dtype=np.float32)
                n = len(arr)
                chunk = min(n, CHUNK_SIZE)
                f.create_dataset(
                    col,
                    data=arr,
                    chunks=(chunk,),
                    compression="lzf",   # LZF : ratio ~3x, décompression 2-3 Go/s
                )
            f.attrs["n_events"] = sample.n_cells
            f.attrs["columns"] = list(sample.data.columns)
            f.attrs["name"] = sample.name
            f.attrs["condition"] = sample.condition
            f.attrs["sample_id"] = sample.sample_id
            f.attrs["n_cells_raw"] = sample.n_cells_raw
            # Sérialise les métadonnées scalaires FCS
            for k, v in sample.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    f.attrs[f"meta_{k}"] = v

        return cls(hdf5_path)

    def __init__(self, hdf5_path: str | Path) -> None:
        self._path = Path(hdf5_path)
        # Mode 'r' : aucun lock exclusif → plusieurs workers peuvent lire en parallèle
        self._f = h5py.File(self._path, "r")
        self.columns: List[str] = list(self._f.attrs["columns"])
        self.n_events: int = int(self._f.attrs["n_events"])
        self.name: str = str(self._f.attrs.get("name", self._path.stem))
        self.condition: str = str(self._f.attrs.get("condition", ""))
        self.sample_id: str = str(self._f.attrs.get("sample_id", ""))
        self.metadata: dict = {
            k[5:]: v for k, v in self._f.attrs.items() if k.startswith("meta_")
        }

    # ------------------------------------------------------------------
    # Lecture lazy
    # ------------------------------------------------------------------

    def load_columns(
        self,
        cols: List[str],
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Charge uniquement les colonnes demandées.

        Args:
            cols: Noms des colonnes (marqueurs) à charger.
            mask: Masque booléen optionnel (GatingMask.unpack() ou ndarray bool).

        Returns:
            np.ndarray shape (N, len(cols)), dtype float32.
        """
        arrays: List[np.ndarray] = []
        for col in cols:
            if col not in self._f:
                raise KeyError(f"Colonne '{col}' absente du HDF5 ({self._path.name})")
            data: np.ndarray = self._f[col][:]           # lecture séquentielle cache-friendly
            if mask is not None:
                data = data[mask]
            arrays.append(data)
        if len(arrays) == 1:
            return arrays[0].reshape(-1, 1)
        return np.column_stack(arrays)                   # (N, len(cols)), float32

    def load_for_plot(
        self,
        x_col: str,
        y_col: str,
        mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shortcut scatter plot — charge exactement 2 colonnes."""
        mat = self.load_columns([x_col, y_col], mask)
        return mat[:, 0], mat[:, 1]

    def load_matrix(
        self,
        cols: Optional[List[str]] = None,
        mask: Optional[np.ndarray] = None,
        dtype: np.dtype = np.float32,
    ) -> np.ndarray:
        """
        Charge la matrice complète (ou sous-ensemble de colonnes).
        Utilisé par les stratégies UMAP/FlowSOM.
        """
        if cols is None:
            cols = self.columns
        mat = self.load_columns(cols, mask)
        return mat.astype(dtype, copy=False)

    # ------------------------------------------------------------------
    # Compat Sample
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        return self.n_events

    @property
    def markers(self) -> List[str]:
        return self.columns

    def summary(self) -> str:
        return (
            f"SampleHDF5({self.name!r}, condition={self.condition!r}, "
            f"cells={self.n_events:,}, markers={len(self.columns)}, "
            f"hdf5={self._path.name})"
        )

    def __repr__(self) -> str:
        return self.summary()

    def __del__(self) -> None:
        try:
            if hasattr(self, "_f") and self._f.id.valid:
                self._f.close()
        except Exception:
            pass
