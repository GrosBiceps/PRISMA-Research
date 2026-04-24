"""
utils/logging_config.py — Configuration logging structuré PRISMA Research.

Appel unique setup_logging() au démarrage de l'app ou du pipeline.
Format JSON-like pour parsing machine + format lisible en console.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    structured: bool = False,
) -> None:
    """
    Configure le logging global.

    Args:
        level: Niveau de log (logging.DEBUG, INFO, WARNING…).
        log_file: Si fourni, duplique les logs dans ce fichier.
        structured: Si True, format JSON-like (pour parsing automatisé).
    """
    fmt_console = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    fmt_structured = (
        '{"time":"%(asctime)s","level":"%(levelname)s",'
        '"logger":"%(name)s","msg":"%(message)s"}'
    )

    fmt = fmt_structured if structured else fmt_console
    datefmt = "%Y-%m-%dT%H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )

    # Taire les loggers verbeux des bibliothèques tierces
    for noisy in ("matplotlib", "numba", "umap", "PIL", "anndata"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("prisma").info("PRISMA logging initialized (level=%s)", logging.getLevelName(level))
