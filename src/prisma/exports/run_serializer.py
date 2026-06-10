"""Sérialisation complète d'un run PRISMA (config, contexte, artefacts, versions)."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


def collect_runtime_versions(extra_modules: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """Collecte les versions logicielles pour audit/reproductibilité."""
    module_names = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "umap",
        "anndata",
        "seaborn",
        "matplotlib",
        "PySide6",
    ]
    if extra_modules:
        module_names.extend(extra_modules)

    versions: Dict[str, str] = {}
    for name in sorted(set(module_names)):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "not_installed"
    return versions


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def serialize_full_run(
    output_dir: Path | str,
    *,
    run_metadata: Any,
    config: Mapping[str, Any],
    run_context: Mapping[str, Any],
    modules_enabled: Iterable[str],
    seeds: Optional[Mapping[str, int]] = None,
    artifacts: Optional[Iterable[Path | str]] = None,
    extra_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    """Sérialise toutes les métadonnées d'un run dans deux JSON versionnés."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    artifact_list = [str(Path(p)) for p in (artifacts or [])]

    # Pourquoi: on enrichit le contexte de run pour auditabilité future SaMD.
    bundle: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "hostname": platform.node(),
        },
        "modules_enabled": sorted(set(modules_enabled)),
        "software_versions": collect_runtime_versions(),
        "seeds": dict(seeds or {}),
        "artifacts": artifact_list,
        "run_metadata": _to_jsonable(run_metadata),
        "run_context": _to_jsonable(dict(run_context)),
        "config": _to_jsonable(dict(config)),
    }

    if "global" not in bundle["seeds"] and hasattr(run_metadata, "seed"):
        bundle["seeds"]["global"] = int(run_metadata.seed)

    if extra_payload:
        bundle["extra"] = _to_jsonable(dict(extra_payload))

    if hasattr(run_metadata, "artifacts") and isinstance(run_metadata.artifacts, list):
        existing = set(str(p) for p in run_metadata.artifacts)
        existing.update(artifact_list)
        run_metadata.artifacts = sorted(existing)

    bundle_path = out / "run_bundle.json"
    config_path = out / "config_snapshot.json"

    bundle_path.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(_to_jsonable(dict(config)), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return {
        "bundle": bundle_path,
        "config": config_path,
    }
