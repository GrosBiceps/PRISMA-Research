"""conftest.py — Ajoute src/ au sys.path pour les tests prisma/."""

import sys
from pathlib import Path

# Résout src/ depuis tests/prisma/ → ../../src/
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
