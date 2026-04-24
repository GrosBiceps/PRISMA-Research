"""
tests/conftest.py — Stubs des dépendances legacy pour les tests analysis/.

Injectés avant que pytest collecte les modules, donc avant que
src/analysis/__init__.py importe ses dépendances flowsom_pipeline_pro.
"""

import sys
import types
import logging


def _ensure_stub(name: str) -> types.ModuleType:
    """Crée un stub vide si le module n'est pas déjà dans sys.modules."""
    if name not in sys.modules:
        parent_name = name.rsplit(".", 1)[0] if "." in name else None
        m = types.ModuleType(name)
        sys.modules[name] = m
        # Attacher comme attribut du parent pour que from parent import child fonctionne
        if parent_name and parent_name in sys.modules:
            child_attr = name.rsplit(".", 1)[1]
            setattr(sys.modules[parent_name], child_attr, m)
    return sys.modules[name]


# Ordre important : parents avant enfants
_stubs_needed = [
    "flowsom_pipeline_pro",
    "flowsom_pipeline_pro.config",
    "flowsom_pipeline_pro.config.pipeline_config",
    "flowsom_pipeline_pro.config.constants",
    "flowsom_pipeline_pro.src",
    "flowsom_pipeline_pro.src.exceptions",
    "flowsom_pipeline_pro.src.utils",
    "flowsom_pipeline_pro.src.utils.logger",
    "flowsom_pipeline_pro.src.pipeline",
    "flowsom_pipeline_pro.src.pipeline.pipeline_executor",
    "flowsom_pipeline_pro.src.analysis",
    "flowsom_pipeline_pro.src.analysis.mrd_calculator",
    "flowsom_pipeline_pro.src.analysis.blast_detection",
]

for _name in _stubs_needed:
    _ensure_stub(_name)

# --- Exceptions attendues par blast_detection & mrd_calculator ---
_exc = sys.modules["flowsom_pipeline_pro.src.exceptions"]
if not hasattr(_exc, "ClinicalMathError"):
    class ClinicalMathError(Exception):
        pass
    class PanelConfigError(Exception):
        pass
    _exc.ClinicalMathError = ClinicalMathError   # type: ignore
    _exc.PanelConfigError = PanelConfigError     # type: ignore

# --- get_logger ---
_log_mod = sys.modules["flowsom_pipeline_pro.src.utils.logger"]
if not hasattr(_log_mod, "get_logger"):
    def _get_logger(name: str = __name__):
        return logging.getLogger(name)
    _log_mod.get_logger = _get_logger  # type: ignore

# --- Constants (statistics.py + autres) ---
_const = sys.modules["flowsom_pipeline_pro.config.constants"]
for _attr, _val in [
    ("MRD_POSITIVE_THRESHOLD", 0.01),
    ("MRD_NEGATIVE_THRESHOLD", 0.001),
    ("MRD_GREY_ZONE", 0.005),
    ("MRD_LOD", 0.0001),
    ("MRD_LOQ", 0.001),
    ("NBM_FREQ_MAX", 0.05),
    ("MRD_FOLD_CHANGE_THRESHOLD", 10.0),
]:
    if not hasattr(_const, _attr):
        setattr(_const, _attr, _val)

# --- PipelineConfig & FlowSOMPipeline sentinelles ---
_pipe_cfg = sys.modules["flowsom_pipeline_pro.config.pipeline_config"]
if not hasattr(_pipe_cfg, "PipelineConfig"):
    class _PipelineConfig:
        pass
    _pipe_cfg.PipelineConfig = _PipelineConfig  # type: ignore

_pipe_exec = sys.modules["flowsom_pipeline_pro.src.pipeline.pipeline_executor"]
if not hasattr(_pipe_exec, "FlowSOMPipeline"):
    class _FlowSOMPipeline:
        pass
    _pipe_exec.FlowSOMPipeline = _FlowSOMPipeline  # type: ignore
