"""
registry.py — Registre des stratégies et plugins PRISMA Research.

Permet l'enregistrement/découverte des stratégies de réduction dimensionnelle
et de clustering sans couplage dur. Les stratégies s'enregistrent à l'import.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, TypeVar

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RegistryError(ValueError):
    """Erreur de cohérence du registre de stratégies."""


class StrategyRegistry:
    """Registre singleton des stratégies disponibles."""

    _dimreduc: Dict[str, Type[Any]] = {}
    _clustering: Dict[str, Type[Any]] = {}

    @staticmethod
    def _normalize_name(name: str) -> str:
        normalized = name.strip().lower()
        if not normalized:
            raise RegistryError("Strategy name must be a non-empty string.")
        return normalized

    @classmethod
    def _register(
        cls,
        registry: Dict[str, Type[T]],
        kind: str,
        name: str,
        klass: Type[T],
        overwrite: bool,
    ) -> Type[T]:
        key = cls._normalize_name(name)
        previous = registry.get(key)
        if previous is not None and previous is not klass and not overwrite:
            raise RegistryError(
                f"{kind} strategy '{key}' already registered with {previous.__name__}."
            )
        registry[key] = klass
        logger.debug("%s strategy registered: %s -> %s", kind, key, klass.__name__)
        return klass

    @classmethod
    def _decorator_register(
        cls,
        registry: Dict[str, Type[T]],
        kind: str,
        name: str,
        overwrite: bool,
    ) -> Callable[[Type[T]], Type[T]]:
        def decorator(klass: Type[T]) -> Type[T]:
            cls._register(registry, kind, name, klass, overwrite=overwrite)
            return klass

        return decorator

    @classmethod
    def register_dimreduc(
        cls,
        name: str,
        klass: Optional[Type[Any]] = None,
        *,
        overwrite: bool = False,
    ) -> Callable[[Type[Any]], Type[Any]] | Type[Any]:
        """Décorateur : enregistre une stratégie de réduction dimensionnelle."""
        if klass is not None:
            return cls._register(cls._dimreduc, "DimReduc", name, klass, overwrite=overwrite)
        return cls._decorator_register(cls._dimreduc, "DimReduc", name, overwrite=overwrite)

    @classmethod
    def register_clustering(
        cls,
        name: str,
        klass: Optional[Type[Any]] = None,
        *,
        overwrite: bool = False,
    ) -> Callable[[Type[Any]], Type[Any]] | Type[Any]:
        """Décorateur : enregistre une stratégie de clustering."""
        if klass is not None:
            return cls._register(cls._clustering, "Clustering", name, klass, overwrite=overwrite)
        return cls._decorator_register(cls._clustering, "Clustering", name, overwrite=overwrite)

    @classmethod
    def get_dimreduc(cls, name: str) -> Optional[Type[Any]]:
        return cls._dimreduc.get(cls._normalize_name(name))

    @classmethod
    def get_clustering(cls, name: str) -> Optional[Type[Any]]:
        return cls._clustering.get(cls._normalize_name(name))

    @classmethod
    def list_dimreduc(cls) -> List[str]:
        return sorted(cls._dimreduc.keys())

    @classmethod
    def list_clustering(cls) -> List[str]:
        return sorted(cls._clustering.keys())

    @classmethod
    def create_dimreduc(cls, name: str, **init_kwargs: Any) -> Any:
        strategy_cls = cls.get_dimreduc(name)
        if strategy_cls is None:
            raise RegistryError(f"Unknown DimReduc strategy: '{name}'.")
        return strategy_cls(**init_kwargs)

    @classmethod
    def create_clustering(cls, name: str, **init_kwargs: Any) -> Any:
        strategy_cls = cls.get_clustering(name)
        if strategy_cls is None:
            raise RegistryError(f"Unknown clustering strategy: '{name}'.")
        return strategy_cls(**init_kwargs)

    @classmethod
    def clear(cls) -> None:
        """Utilitaire de test: vide entièrement le registre."""
        cls._dimreduc.clear()
        cls._clustering.clear()
