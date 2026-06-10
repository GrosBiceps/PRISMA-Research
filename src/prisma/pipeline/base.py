"""
pipeline/base.py — Contrats Pipeline de PRISMA Research.

PipelineStep (Protocol) : interface que chaque étape doit respecter.
PipelineRunner : exécuteur séquentiel avec logging, snapshots et RunMetadata.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Protocol, runtime_checkable

from prisma.core.models import Experiment, RunMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contexte de run passé à chaque étape
# ---------------------------------------------------------------------------

@dataclass
class PipelineContext:
    """
    Contexte mutable partagé entre toutes les étapes du pipeline.

    Attributes:
        experiment: Expérience en cours (modifiable par les étapes).
        run_meta: Métadonnées du run courant.
        output_dir: Répertoire de sortie pour ce run.
        params: Paramètres globaux du run.
        step_results: Résultats intermédiaires indexés par nom d'étape.
        seed: Graine globale.
    """

    experiment: Experiment
    run_meta: RunMetadata
    output_dir: Path
    params: Dict[str, Any]
    step_results: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42


# ---------------------------------------------------------------------------
# Protocol PipelineStep
# ---------------------------------------------------------------------------

@runtime_checkable
class PipelineStep(Protocol):
    """
    Interface que toute étape de pipeline doit respecter.

    Une étape reçoit un PipelineContext, le modifie in-place (par ex.
    remplace experiment.samples), et retourne le même contexte.
    """

    name: str

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Exécute l'étape et retourne le contexte mis à jour."""
        ...

    def validate(self, ctx: PipelineContext) -> None:
        """
        Vérifie les préconditions avant exécution.
        Lève ValueError si le contexte est invalide pour cette étape.
        """
        ...


# ---------------------------------------------------------------------------
# PipelineRunner
# ---------------------------------------------------------------------------

class PipelineRunner:
    """
    Exécuteur séquentiel d'une liste de PipelineStep.

    - Logging structuré à chaque étape.
    - Snapshot optionnel du contexte après chaque étape.
    - Capture des erreurs avec mise à jour de RunMetadata.
    """

    def __init__(self, steps: List[PipelineStep], snapshot: bool = False) -> None:
        self._steps = steps
        self._snapshot = snapshot

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """
        Exécute toutes les étapes séquentiellement.

        Args:
            ctx: Contexte initial du pipeline.

        Returns:
            Contexte final après toutes les étapes.

        Raises:
            RuntimeError: Si une étape échoue.
        """
        logger.info(
            "Pipeline '%s' starting — %d steps, seed=%d",
            ctx.run_meta.pipeline_name,
            len(self._steps),
            ctx.seed,
        )

        for i, step in enumerate(self._steps, start=1):
            logger.info("[%d/%d] Step '%s' starting", i, len(self._steps), step.name)
            t0 = time.perf_counter()

            try:
                step.validate(ctx)
                ctx = step.run(ctx)
            except Exception as exc:
                logger.exception("Step '%s' failed: %s", step.name, exc)
                ctx.run_meta.finish(status="failed", error=str(exc))
                raise RuntimeError(f"Pipeline aborted at step '{step.name}'") from exc

            elapsed = time.perf_counter() - t0
            logger.info("[%d/%d] Step '%s' done in %.2fs", i, len(self._steps), step.name, elapsed)

            if self._snapshot:
                self._save_snapshot(ctx, step.name)

        ctx.run_meta.finish(status="success")
        meta_path = ctx.run_meta.save(ctx.output_dir)
        logger.info("Pipeline finished. Metadata saved: %s", meta_path)
        return ctx

    @staticmethod
    def _save_snapshot(ctx: PipelineContext, step_name: str) -> None:
        """Sauvegarde un résumé JSON léger après chaque étape."""
        import json
        snap = {
            "step": step_name,
            "n_samples": ctx.experiment.n_samples,
            "step_result_keys": list(ctx.step_results.keys()),
        }
        path = ctx.output_dir / f"snapshot_{step_name}.json"
        path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
