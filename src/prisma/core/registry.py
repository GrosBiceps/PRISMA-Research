"""
registry.py — Registre des stratégies et plugins PRISMA Research.

Permet l'enregistrement/découverte des stratégies de dim-réduction et clustering
sans couplage dur. Les stratégies s'enregistrent à l'import.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Registre singleton des stratégies disponibles."""

    _dimreduc: Dict[str, Type] = {}
    _clustering: Dict[str, Type] = {}

    @classmethod
    def register_dimreduc(cls, name: str):
        """Décorateur : enregistre une stratégie de réduction dimensionnelle."""
        def decorator(klass):
            cls._dimreduc[name] = klass
            logger.debug("DimReduc strategy registered: %s", name)
            return klass
        return decorator

    @classmethod
    def register_clustering(cls, name: str):
        """Décorateur : enregistre une stratégie de clustering."""
        def decorator(klass):
            cls._clustering[name] = klass
            logger.debug("Clustering strategy registered: %s", name)
            return klass
        return decorator

    @classmethod
    def get_dimreduc(cls, name: str) -> Optional[Type]:
        return cls._dimreduc.get(name)

    @classmethod
    def get_clustering(cls, name: str) -> Optional[Type]:
        return cls._clustering.get(name)

    @classmethod
    def list_dimreduc(cls) -> List[str]:
        return sorted(cls._dimreduc.keys())

    @classmethod
    def list_clustering(cls) -> List[str]:
        return sorted(cls._clustering.keys())
