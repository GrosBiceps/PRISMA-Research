"""Auto-enregistrement des stratégies au premier import du module."""

from prisma.strategies import (  # noqa: F401
    umap_strategy,
    tsne_strategy,
    flowsom_strategy,
    spectral_strategy,
)
