"""
gating_mask.py — Masques de gating compressés bitwise (np.packbits).

GatingMask stocke N booléens en N/8 bytes (8x moins de RAM que bool array).
Opérations AND/OR/NOT directement sur les bits → microsecondes.

Compatibilité : remplace les np.ndarray bool dans PreGating et AutoGating.
"""

from __future__ import annotations

import numpy as np


class GatingMask:
    """
    Masque booléen compressé via np.packbits.

    8x moins de RAM qu'un np.ndarray bool standard.
    Opérations bitwise AND/OR/NOT en microsecondes même sur 10M cellules.
    """

    __slots__ = ("_packed", "_n")

    def __init__(self, bool_array: np.ndarray) -> None:
        arr = np.asarray(bool_array, dtype=bool)
        self._n: int = len(arr)
        self._packed: np.ndarray = np.packbits(arr)

    @classmethod
    def from_packed(cls, packed: np.ndarray, n: int) -> "GatingMask":
        obj = object.__new__(cls)
        obj._packed = packed
        obj._n = n
        return obj

    @classmethod
    def all_true(cls, n: int) -> "GatingMask":
        packed = np.full((n + 7) // 8, 0xFF, dtype=np.uint8)
        return cls.from_packed(packed, n)

    @classmethod
    def all_false(cls, n: int) -> "GatingMask":
        packed = np.zeros((n + 7) // 8, dtype=np.uint8)
        return cls.from_packed(packed, n)

    def unpack(self) -> np.ndarray:
        return np.unpackbits(self._packed)[:self._n].astype(bool)

    def __and__(self, other: "GatingMask") -> "GatingMask":
        return GatingMask.from_packed(self._packed & other._packed, self._n)

    def __or__(self, other: "GatingMask") -> "GatingMask":
        return GatingMask.from_packed(self._packed | other._packed, self._n)

    def __invert__(self) -> "GatingMask":
        return GatingMask.from_packed(~self._packed, self._n)

    def __len__(self) -> int:
        return self._n

    @property
    def count(self) -> int:
        return int(np.unpackbits(self._packed)[:self._n].sum())

    @property
    def frequency(self) -> float:
        return self.count / self._n if self._n > 0 else 0.0

    @property
    def nbytes(self) -> int:
        return self._packed.nbytes

    def __repr__(self) -> str:
        return f"GatingMask(n={self._n}, selected={self.count}, {self.frequency:.1%})"
