"""
utils/seed.py — Gestion déterministe de la graine globale.

set_global_seed() fixe numpy, random et torch (si dispo).
"""

from __future__ import annotations

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """
    Fixe la graine pour numpy, random, et torch si disponible.

    Args:
        seed: Entier ≥ 0.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    logger.debug("Global seed set: %d", seed)
