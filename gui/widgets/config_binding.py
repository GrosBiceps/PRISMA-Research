# -*- coding: utf-8 -*-
"""
config_binding.py — Couche de binding bidirectionnel UI ↔ dataclass.

Chaque Binding relie un widget PyQt5 à un attribut d'un dataclass via:
  - init_from_config(cfg)  : charge les valeurs du dataclass dans les widgets
  - sync_to_config(cfg)    : écrit les valeurs des widgets dans le dataclass
  - connect_live(cfg)      : connecte les signaux pour mise à jour immédiate

Usage:
    bindings = [
        SpinBinding(widget.spin_xdim, "flowsom", "xdim"),
        CheckBinding(widget.chk_umap, "visualization", "umap_enabled"),
        ComboBinding(widget.combo_transform, "transform", "method"),
    ]
    binder = ConfigBinder(config, bindings)
    binder.load()          # config → widgets
    binder.save()          # widgets → config
    binder.connect_live()  # auto-save à chaque changement
"""

from __future__ import annotations

from dataclasses import fields as dc_fields
from typing import Any, Callable, List, Optional

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
    QWidget,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────────────────────

def _get_nested(cfg: Any, *attrs: str) -> Any:
    """Descend dans l'arborescence config via une suite d'attributs."""
    obj = cfg
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def _set_nested(cfg: Any, path: tuple[str, ...], value: Any) -> None:
    """Écrit value dans cfg.path[0].path[1]...path[-1]."""
    obj = cfg
    for attr in path[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, path[-1], value)


# ─────────────────────────────────────────────────────────────────────────────
# Binding de base
# ─────────────────────────────────────────────────────────────────────────────

class _BaseBinding:
    """Contrat minimal d'un binding UI ↔ dataclass."""

    def __init__(self, widget: QWidget, *config_path: str) -> None:
        self.widget = widget
        self.config_path: tuple[str, ...] = config_path

    def load(self, cfg: Any) -> None:
        """Copie la valeur du dataclass vers le widget."""
        raise NotImplementedError

    def save(self, cfg: Any) -> None:
        """Copie la valeur du widget vers le dataclass."""
        raise NotImplementedError

    def connect(self, callback: Callable[[], None]) -> None:
        """Connecte le signal de changement du widget au callback."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Bindings concrets
# ─────────────────────────────────────────────────────────────────────────────

class SpinBinding(_BaseBinding):
    """QSpinBox ↔ int."""

    def load(self, cfg: Any) -> None:
        val = _get_nested(cfg, *self.config_path)
        if isinstance(val, int):
            self.widget.setValue(val)

    def save(self, cfg: Any) -> None:
        _set_nested(cfg, self.config_path, int(self.widget.value()))

    def connect(self, callback: Callable[[], None]) -> None:
        self.widget.valueChanged.connect(lambda _: callback())


class DoubleSpinBinding(_BaseBinding):
    """QDoubleSpinBox ↔ float."""

    def load(self, cfg: Any) -> None:
        val = _get_nested(cfg, *self.config_path)
        if isinstance(val, (int, float)):
            self.widget.setValue(float(val))

    def save(self, cfg: Any) -> None:
        _set_nested(cfg, self.config_path, float(self.widget.value()))

    def connect(self, callback: Callable[[], None]) -> None:
        self.widget.valueChanged.connect(lambda _: callback())


class CheckBinding(_BaseBinding):
    """QCheckBox / ToggleSwitch ↔ bool."""

    def load(self, cfg: Any) -> None:
        val = _get_nested(cfg, *self.config_path)
        self.widget.setChecked(bool(val))

    def save(self, cfg: Any) -> None:
        _set_nested(cfg, self.config_path, bool(self.widget.isChecked()))

    def connect(self, callback: Callable[[], None]) -> None:
        # ToggleSwitch émet toggled(bool), QCheckBox également
        self.widget.toggled.connect(lambda _: callback())


class ComboBinding(_BaseBinding):
    """QComboBox ↔ str."""

    def load(self, cfg: Any) -> None:
        val = str(_get_nested(cfg, *self.config_path))
        idx = self.widget.findText(val)
        if idx >= 0:
            self.widget.setCurrentIndex(idx)

    def save(self, cfg: Any) -> None:
        _set_nested(cfg, self.config_path, self.widget.currentText())

    def connect(self, callback: Callable[[], None]) -> None:
        self.widget.currentIndexChanged.connect(lambda _: callback())


class LineEditBinding(_BaseBinding):
    """QLineEdit ↔ str."""

    def __init__(self, widget: QLineEdit, *config_path: str,
                 splitter: Optional[str] = None) -> None:
        super().__init__(widget, *config_path)
        # splitter: si défini, convertit str ↔ List[str] via split/join
        self.splitter = splitter

    def load(self, cfg: Any) -> None:
        val = _get_nested(cfg, *self.config_path)
        if self.splitter is not None and isinstance(val, list):
            self.widget.setText(self.splitter.join(v for v in val if v))
        else:
            self.widget.setText(str(val) if val is not None else "")

    def save(self, cfg: Any) -> None:
        text = self.widget.text().strip()
        if self.splitter is not None:
            value: Any = [s.strip() for s in text.split(self.splitter) if s.strip()]
        else:
            value = text
        _set_nested(cfg, self.config_path, value)

    def connect(self, callback: Callable[[], None]) -> None:
        self.widget.editingFinished.connect(callback)


class NClustBinding(_BaseBinding):
    """QSpinBox ↔ Optional[int]  (0 → None, sinon entier positif)."""

    def load(self, cfg: Any) -> None:
        val = _get_nested(cfg, *self.config_path)
        self.widget.setValue(int(val) if val is not None else 0)

    def save(self, cfg: Any) -> None:
        v = int(self.widget.value())
        _set_nested(cfg, self.config_path, v if v > 0 else None)

    def connect(self, callback: Callable[[], None]) -> None:
        self.widget.valueChanged.connect(lambda _: callback())


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrateur
# ─────────────────────────────────────────────────────────────────────────────

class ConfigBinder:
    """
    Orchestre une liste de bindings sur un même PipelineConfig.

    Usage:
        binder = ConfigBinder(config, [SpinBinding(...), CheckBinding(...)])
        binder.load()           # config → UI (initialisation)
        binder.save()           # UI → config (avant pipeline)
        binder.connect_live()   # connecte tous les signaux (temps réel)
    """

    def __init__(self, cfg: Any, bindings: List[_BaseBinding]) -> None:
        self._cfg = cfg
        self._bindings = bindings
        self._live = False

    @property
    def config(self) -> Any:
        return self._cfg

    def set_config(self, cfg: Any) -> None:
        """Remplace la config (ex: chargement depuis YAML) et recharge l'UI."""
        self._cfg = cfg
        self.load()

    def load(self) -> None:
        """Initialise tous les widgets depuis la config (sans déclencher de signaux)."""
        for b in self._bindings:
            try:
                b.load(self._cfg)
            except Exception:
                pass  # attribut absent dans config plus ancienne — ignoré silencieusement

    def save(self) -> None:
        """Écrit tous les widgets dans la config."""
        for b in self._bindings:
            try:
                b.save(self._cfg)
            except Exception:
                pass

    def connect_live(self) -> None:
        """Connecte les signaux de chaque widget → save immédiat à chaque changement."""
        if self._live:
            return
        self._live = True
        for b in self._bindings:
            b.connect(self.save)
