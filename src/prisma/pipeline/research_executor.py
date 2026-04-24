"""
research_executor.py — Exécuteur modulaire PRISMA basé sur StrategyRegistry.

Cette couche orchestre des stratégies de réduction dimensionnelle et de
clustering enregistrées dynamiquement, sans appels codés en dur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Sequence

import numpy as np

from prisma.core.registry import StrategyRegistry
from prisma.strategies.base import ClusterParams, DimReducParams


StepKind = Literal["dimreduc", "clustering"]


@dataclass(frozen=True)
class ResearchPipelineStep:
    """Décrit une étape du pipeline modulaire."""

    kind: StepKind
    strategy_name: str
    params: DimReducParams | ClusterParams
    output_name: str


@dataclass
class ResearchPipelineResult:
    """Résultats intermédiaires et final du pipeline."""

    outputs: Dict[str, np.ndarray] = field(default_factory=dict)
    final_output: np.ndarray | None = None


class ResearchPipelineExecutor:
    """Exécute une séquence de stratégies enregistrées dans StrategyRegistry."""

    def __init__(self, registry: type[StrategyRegistry] = StrategyRegistry) -> None:
        self._registry = registry

    def fit_transform(
        self,
        data: np.ndarray,
        strategy_name: str,
        params: DimReducParams,
    ) -> np.ndarray:
        strategy = self._registry.create_dimreduc(strategy_name)
        return strategy.fit_transform(data, params)

    def fit_predict(
        self,
        data: np.ndarray,
        strategy_name: str,
        params: ClusterParams,
    ) -> np.ndarray:
        strategy = self._registry.create_clustering(strategy_name)
        return strategy.fit_predict(data, params)

    def run(
        self,
        data: np.ndarray,
        steps: Sequence[ResearchPipelineStep],
    ) -> ResearchPipelineResult:
        current = np.asarray(data)
        outputs: Dict[str, np.ndarray] = {}

        for step in steps:
            if step.kind == "dimreduc":
                current = self.fit_transform(
                    current,
                    step.strategy_name,
                    self._coerce_dimreduc_params(step.params),
                )
            elif step.kind == "clustering":
                current = self.fit_predict(
                    current,
                    step.strategy_name,
                    self._coerce_cluster_params(step.params),
                )
            else:
                raise ValueError(f"Unknown pipeline step kind: {step.kind!r}")

            outputs[step.output_name] = np.asarray(current)

        return ResearchPipelineResult(outputs=outputs, final_output=current)

    @staticmethod
    def _coerce_dimreduc_params(params: DimReducParams | ClusterParams) -> DimReducParams:
        if isinstance(params, DimReducParams):
            return params
        return DimReducParams(seed=params.seed, n_jobs=params.n_jobs, extra=dict(params.extra))

    @staticmethod
    def _coerce_cluster_params(params: DimReducParams | ClusterParams) -> ClusterParams:
        if isinstance(params, ClusterParams):
            return params
        return ClusterParams(seed=params.seed, n_jobs=params.n_jobs, extra=dict(params.extra))
