"""
src/gui/viewer/plot_panel_model.py — Modèle de données d'un PlotWidgetPanel.

PlotPanelModel  : état pur (non-Qt) d'un panneau de visualisation.
RenderMode      : stratégie de rendu selon le volume d'événements.
BackgateRequest : requête de coloration par sous-populations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np


class RenderMode(Enum):
    SCATTER = auto()       # scatter classique < 10k events
    DENSITY = auto()       # histogram2d / pseudo-color ≥ 100k events
    HYBRID = auto()        # scatter sous-échantillonné + density fond
    BACKGATE = auto()      # parent gris + sous-populations colorées


@dataclass
class PlotPanelModel:
    """
    Source de vérité locale d'un panneau de visualisation.

    Ce dataclass ne contient aucun widget Qt. Il est lisible par le thread
    principal et par les workers d'analyse sans risque de race condition
    sur les attributs primitifs.

    Attributes
    ----------
    gate_node       : Tuple identifiant la population source, ex: ('Lymphocytes', 'root').
    x_channel       : Pnn du canal X affiché.
    y_channel       : Pnn du canal Y, ou '' pour histogramme 1D.
    transform_id    : ID de transformation FlowKit actif.
    comp_id         : ID de matrice de compensation, ou None.
    render_mode     : Stratégie de rendu courante.
    panel_id        : Identifiant unique du panneau (pour les refreshes ciblés).
    """

    gate_node: Tuple[str, ...]
    x_channel: str = ""
    y_channel: str = ""
    transform_id: Optional[str] = None
    comp_id: Optional[str] = None
    render_mode: RenderMode = RenderMode.SCATTER
    panel_id: str = ""
    density_coloring: bool = False  # coloration KDE scatter (mode SCATTER uniquement)

    # Seuils alignés avec PlotWidgetPanel (OpenGL gère 80k pts à 60fps)
    SCATTER_MAX: int = 80_000
    DENSITY_MIN: int = 500_000

    def resolve_render_mode(self, n_events: int, force_density_coloring: bool = False) -> RenderMode:
        """
        Retourne le RenderMode optimal.
        Si force_density_coloring=True (bouton densité coché), reste en SCATTER
        même au-delà de SCATTER_MAX — la coloration densité est appliquée via
        density_coloring=True dans set_data_2d(), pas via set_data_density().
        """
        if force_density_coloring:
            # Densité KDE scatter toujours en mode SCATTER (coloration dans set_data_2d)
            return RenderMode.SCATTER
        if n_events <= self.SCATTER_MAX:
            return RenderMode.SCATTER
        if n_events >= self.DENSITY_MIN:
            return RenderMode.DENSITY
        return RenderMode.HYBRID

    @property
    def is_2d(self) -> bool:
        return bool(self.x_channel and self.y_channel)

    @property
    def population_label(self) -> str:
        return str(self.gate_node[0]) if self.gate_node else "root"


@dataclass
class DensityCache:
    """
    Cache du calcul histogram2d pour éviter les recalculs coûteux au repaint.

    La clé d'invalidation est (x_channel, y_channel, sample_id, n_events).
    Tout changement dans ces 4 paramètres invalide le cache.
    """

    x_channel: str = ""
    y_channel: str = ""
    sample_id: str = ""
    n_events: int = 0
    histogram: Optional[np.ndarray] = field(default=None, repr=False)
    x_edges: Optional[np.ndarray] = field(default=None, repr=False)
    y_edges: Optional[np.ndarray] = field(default=None, repr=False)

    def is_valid_for(
        self,
        x_ch: str,
        y_ch: str,
        sid: str,
        n: int,
    ) -> bool:
        return (
            self.histogram is not None
            and self.x_channel == x_ch
            and self.y_channel == y_ch
            and self.sample_id == sid
            and self.n_events == n
        )

    def update(
        self,
        x_ch: str,
        y_ch: str,
        sid: str,
        n: int,
        histogram: np.ndarray,
        x_edges: np.ndarray,
        y_edges: np.ndarray,
    ) -> None:
        self.x_channel = x_ch
        self.y_channel = y_ch
        self.sample_id = sid
        self.n_events = n
        self.histogram = histogram
        self.x_edges = x_edges
        self.y_edges = y_edges

    def invalidate(self) -> None:
        self.histogram = None
        self.x_edges = None
        self.y_edges = None
        self.n_events = 0
