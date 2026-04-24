"""
conftest.py racine — isole les tests prisma/ du package legacy.

Ce fichier empêche pytest de remonter jusqu'au __init__.py legacy
et injecte src/ dans sys.path pour les imports prisma.*.
"""

import sys
from pathlib import Path

# Injecte src/ en premier pour que `import prisma` trouve src/prisma/
_src = Path(__file__).parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# Stub flowsom_pipeline_pro pour éviter l'import du __init__.py legacy
# quand pytest collecte les tests depuis la racine.
import types as _types

if "flowsom_pipeline_pro" not in sys.modules:
    _stub = _types.ModuleType("flowsom_pipeline_pro")
    sys.modules["flowsom_pipeline_pro"] = _stub

    # Stubs avec attributs sentinelles pour satisfaire les from … import …
    _sub_names = [
        "flowsom_pipeline_pro.config",
        "flowsom_pipeline_pro.config.pipeline_config",
        "flowsom_pipeline_pro.src",
        "flowsom_pipeline_pro.src.pipeline",
        "flowsom_pipeline_pro.src.pipeline.pipeline_executor",
        "flowsom_pipeline_pro.src.utils",
        "flowsom_pipeline_pro.src.utils.logger",
    ]
    for _sub in _sub_names:
        _m = _types.ModuleType(_sub)
        sys.modules[_sub] = _m

    # Sentinelles de classe vides pour éviter ImportError
    class _PipelineConfig:
        pass

    class _FlowSOMPipeline:
        pass

    sys.modules["flowsom_pipeline_pro.config.pipeline_config"].PipelineConfig = _PipelineConfig  # type: ignore
    sys.modules["flowsom_pipeline_pro.src.pipeline.pipeline_executor"].FlowSOMPipeline = _FlowSOMPipeline  # type: ignore

    import logging as _logging

    def _get_logger(name: str = __name__):
        return _logging.getLogger(name)

    sys.modules["flowsom_pipeline_pro.src.utils.logger"].get_logger = _get_logger  # type: ignore
