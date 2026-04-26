"""
src/gui/viewer/worksheet_layout.py — Gestion du layout configurable du WorksheetArea (Phase 5).

Encapsule la logique de persistance de la disposition des panels.
Utilisé par WorksheetArea et PrismaGatingWorkspace.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger("viewer.worksheet_layout")

_FORMAT_VERSION = "prisma_workspace_layout_v2"
_VALID_COLUMN_COUNTS = frozenset({1, 2, 3, 4})


class WorksheetLayoutError(ValueError):
    """Erreur de validation du layout."""


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------


@dataclass
class PanelState:
    """État sérialisable d'un PlotWidgetPanel."""

    panel_id: str
    gate_node: Tuple[str, ...]
    x_channel: str
    y_channel: str
    transform_id: Optional[str]
    comp_id: Optional[str]
    render_mode: str
    density_coloring: bool
    position: int  # ordre dans la grille (0-based)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "gate_node": list(self.gate_node),
            "x_channel": self.x_channel,
            "y_channel": self.y_channel,
            "transform_id": self.transform_id,
            "comp_id": self.comp_id,
            "render_mode": self.render_mode,
            "density_coloring": self.density_coloring,
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PanelState":
        return cls(
            panel_id=str(d.get("panel_id", uuid.uuid4().hex[:8])),
            gate_node=tuple(str(x) for x in (d.get("gate_node") or ["root"])),
            x_channel=str(d.get("x_channel", "")),
            y_channel=str(d.get("y_channel", "")),
            transform_id=d.get("transform_id"),
            comp_id=d.get("comp_id"),
            render_mode=str(d.get("render_mode", "SCATTER")),
            density_coloring=bool(d.get("density_coloring", False)),
            position=int(d.get("position", 0)),
        )


@dataclass
class WorkspaceLayoutState:
    """État complet sérialisable du workspace layout."""

    panel_column_count: int = 2
    active_panel_id: Optional[str] = None
    stats_dock_visible: bool = True
    panels: List[PanelState] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": _FORMAT_VERSION,
            "panel_column_count": self.panel_column_count,
            "active_panel_id": self.active_panel_id,
            "stats_dock_visible": self.stats_dock_visible,
            "panels": [p.to_dict() for p in self.panels],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkspaceLayoutState":
        panels = []
        for pd_raw in (d.get("panels") or []):
            try:
                panels.append(PanelState.from_dict(pd_raw))
            except Exception as exc:
                _logger.warning("PanelState ignoré lors du chargement: %s", exc)
        return cls(
            panel_column_count=int(d.get("panel_column_count", 2)),
            active_panel_id=d.get("active_panel_id"),
            stats_dock_visible=bool(d.get("stats_dock_visible", True)),
            panels=sorted(panels, key=lambda p: p.position),
        )


# ---------------------------------------------------------------------------
# Fonctions de persistance
# ---------------------------------------------------------------------------


def validate_column_count(count: int) -> int:
    """
    Valide et retourne le nombre de colonnes.
    Lève WorksheetLayoutError si invalide.
    """
    if count not in _VALID_COLUMN_COUNTS:
        raise WorksheetLayoutError(
            f"Nombre de colonnes invalide: {count}. Valeurs acceptées: {sorted(_VALID_COLUMN_COUNTS)}"
        )
    return count


def save_layout_state(state: WorkspaceLayoutState, path: "str | Path") -> None:
    """
    Sauvegarde l'état du layout dans un fichier JSON.
    Crée les répertoires parents si nécessaires.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest.write_text(
            json.dumps(state.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _logger.info("Layout sauvegardé: %s", dest)
    except OSError as exc:
        _logger.error("Impossible de sauvegarder le layout (%s): %s", dest, exc)
        raise


def load_layout_state(path: "str | Path") -> WorkspaceLayoutState:
    """
    Charge un état de layout depuis un fichier JSON.
    Tolérante : retourne un état par défaut si le fichier est invalide ou partiel.
    Ne lève jamais d'exception liée au contenu JSON.
    """
    src = Path(path)
    if not src.exists():
        _logger.warning("Fichier layout introuvable: %s. Utilisation des valeurs par défaut.", src)
        return WorkspaceLayoutState()

    try:
        raw = src.read_text(encoding="utf-8")
        d = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        _logger.warning("Layout JSON invalide (%s): %s. Valeurs par défaut.", src, exc)
        return WorkspaceLayoutState()

    fmt = str(d.get("format", ""))
    if fmt not in (_FORMAT_VERSION, "prisma_workspace_layout_v1"):
        _logger.warning("Format de layout inconnu: '%s'. Valeurs par défaut.", fmt)
        return WorkspaceLayoutState()

    try:
        state = WorkspaceLayoutState.from_dict(d)
    except Exception as exc:
        _logger.warning("Désérialisation layout échouée: %s. Valeurs par défaut.", exc)
        return WorkspaceLayoutState()

    # Validation du nombre de colonnes — fallback sur 2 si invalide
    try:
        validate_column_count(state.panel_column_count)
    except WorksheetLayoutError:
        _logger.warning(
            "Nombre de colonnes invalide dans le layout (%d). Fallback sur 2.",
            state.panel_column_count,
        )
        state.panel_column_count = 2

    _logger.info(
        "Layout chargé: %s — %d panels, %d colonnes",
        src,
        len(state.panels),
        state.panel_column_count,
    )
    return state
