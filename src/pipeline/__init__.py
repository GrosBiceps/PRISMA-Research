"""
src/pipeline/__init__.py — Exports de la couche pipeline.
"""

from src.legacy_clinical.pipeline_executor_legacy import FlowSOMPipeline, run_flowsom_pipeline

__all__ = ["FlowSOMPipeline"]
