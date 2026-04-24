"""Auto-enregistrement des stratégies au premier import du module."""

from prisma.strategies import (  # noqa: F401
    umap_strategy,
    tsne_strategy,
    phate_strategy,
    flowsom_strategy,
    flowsom_like_strategy,
    phenograph_strategy,
    hdbscan_strategy,
    parc_strategy,
    spade_strategy,
    spectral_strategy,
)
