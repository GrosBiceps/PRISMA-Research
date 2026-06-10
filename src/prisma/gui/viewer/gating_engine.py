"""
src/gui/viewer/gating_engine.py — Moteur FlowKit pour PrismaGatingViewer.

Dual backend Session / Workspace :
  - flowkit.Session   : pipeline headless (FCS brut, GatingML, compensation)
  - flowkit.Workspace : fichiers FlowJo WSP avec groupes de samples

Extensions produit PRISMA (non natives FlowKit) :
  - Time gating sur canal Time
  - Calcul matrice compensation depuis single-stained controls
  - Import FACSDiva XML via adaptateur

Architecture :
  FlowKitBackend     — Union[fk.Session, fk.Workspace] avec méthodes communes
  PrismaFlowEngine   — façade publique
  TransformSpec      — configuration d'une transformation
  GateHierarchyNode  — nœud de la hiérarchie pour le QTreeView
  PrismaEngineError  — exceptions métier
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

try:
    from PyQt5.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, QMetaObject, Qt
    _QT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _QT_AVAILABLE = False
    QObject = object  # type: ignore[assignment,misc]
    QRunnable = object  # type: ignore[assignment,misc]
    QThreadPool = None  # type: ignore[assignment]

try:
    import flowkit as fk
    from flowkit import gates as fk_gates
    from flowkit import transforms as fk_transforms

    fkt = fk_transforms

    FLOWKIT_AVAILABLE = True
except Exception as exc:  # pragma: no cover - dépendance optionnelle
    fk = None  # type: ignore[assignment]
    fkt = None  # type: ignore[assignment]
    fk_gates = None  # type: ignore[assignment]
    fk_transforms = None  # type: ignore[assignment]
    FLOWKIT_AVAILABLE = False
    _FLOWKIT_IMPORT_ERROR = exc

if TYPE_CHECKING:
    import flowkit as fk_type
else:
    fk_type = Any

from src.prisma.utils.logger import get_logger

_logger = get_logger("viewer.gating_engine")

# Type alias pour le backend dual
FlowKitBackend = Union["fk_type.Session", "fk_type.Workspace", Any]


# ---------------------------------------------------------------------------
# Exceptions métier
# ---------------------------------------------------------------------------


class PrismaEngineError(Exception):
    """Erreur métier du moteur de gating."""


class SessionNotLoadedError(PrismaEngineError):
    """Aucun sample chargé dans la session."""


class SampleNotFoundError(PrismaEngineError):
    """Sample ID introuvable dans la session."""


class GateNotFoundError(PrismaEngineError):
    """Gate introuvable dans la stratégie."""


class GateNameConflictError(PrismaEngineError):
    """Nom de gate déjà utilisé avec un chemin différent."""


class IncompatibleAxesError(PrismaEngineError):
    """Canaux demandés absents du sample ou incompatibles."""


class GateParentMissingError(PrismaEngineError):
    """Gate parente référencée introuvable."""


class BooleanGateRefMissingError(PrismaEngineError):
    """Une gate référencée par une BooleanGate est absente."""


class _UnavailableFlowKitBackend:
    """Backend de secours quand FlowKit n'est pas installé."""

    def __init__(self, import_error: Exception) -> None:
        self._import_error = import_error

    def __getattr__(self, name: str) -> Any:
        raise PrismaEngineError(
            "FlowKit n'est pas installé dans cet environnement. "
            "Le workspace de gating peut s'ouvrir, mais les actions de gating sont indisponibles."
        ) from self._import_error


# ---------------------------------------------------------------------------
# Types de données
# ---------------------------------------------------------------------------


@dataclass
class TransformSpec:
    """
    Spécification d'une transformation pour un ou plusieurs canaux.

    Attributes:
        kind: 'logicle' | 'log' | 'linear' | 'hyperlog' | 'asinh'
        channel_ids: Canaux concernés. Vide → global (non natif FlowKit, utilisé
                     comme identifiant de transform dans add_transform).
        params: Paramètres spécifiques (cf. flowkit.transforms.*).
    """

    kind: str
    channel_ids: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    _DEFAULTS: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "logicle": {"param_t": 262144, "param_w": 0.5, "param_m": 4.5, "param_a": 0.0},
            "log": {"param_t": 262144, "param_m": 4.5},
            "linear": {"param_t": 262144, "param_a": 0.0},
            "hyperlog": {"param_t": 262144, "param_w": 0.5, "param_m": 4.5, "param_a": 0.0},
            "asinh": {"param_t": 262144, "param_m": 4.0, "param_a": 0.0},
        },
        init=False,
        repr=False,
    )

    def merged_params(self) -> Dict[str, Any]:
        return {**self._DEFAULTS.get(self.kind, {}), **self.params}


@dataclass
class GateHierarchyNode:
    """Nœud de hiérarchie de gating exporté vers QTreeView."""

    gate_name: str
    gate_path: Tuple[str, ...]
    gate_type: str = "unknown"
    children: List["GateHierarchyNode"] = field(default_factory=list)
    count: int = 0
    pct_parent: float = 0.0
    pct_grandparent: float = 0.0

    @property
    def display_path(self) -> str:
        return " / ".join(self.gate_path + (self.gate_name,))


# ---------------------------------------------------------------------------
# Moteur principal
# ---------------------------------------------------------------------------


class _AnalysisWorker(QRunnable if _QT_AVAILABLE else object):  # type: ignore[misc]
    """Worker QRunnable pour l'analyse FlowKit asynchrone."""

    class _Signals(QObject if _QT_AVAILABLE else object):  # type: ignore[misc]
        if _QT_AVAILABLE:
            finished = pyqtSignal(str)   # sample_id
            error = pyqtSignal(str, str) # sample_id, error_message

        def __init__(self) -> None:
            if _QT_AVAILABLE:
                super().__init__()

    def __init__(self, engine: "PrismaFlowEngine", sample_id: Optional[str]) -> None:
        if _QT_AVAILABLE:
            super().__init__()
            self.setAutoDelete(True)
        self._engine = engine
        self._sample_id = sample_id
        self.signals = _AnalysisWorker._Signals()

    def run(self) -> None:  # type: ignore[override]
        sid = self._sample_id or ""
        try:
            self._engine._run_analyze_sync(sample_id=self._sample_id)
            if _QT_AVAILABLE:
                self.signals.finished.emit(sid)
        except Exception as exc:
            _logger.warning("AnalysisWorker error: %s", exc)
            if _QT_AVAILABLE:
                self.signals.error.emit(sid, str(exc))


class PrismaFlowEngine(QObject if _QT_AVAILABLE else object):  # type: ignore[misc]
    """
    Façade FlowKit dual-backend (Session + Workspace) pour PrismaGatingViewer.

    Backend actif : self._backend est soit fk.Session soit fk.Workspace.
    L'API commune est utilisée directement ; les méthodes spécifiques à chaque
    backend sont appelées via _is_workspace() / _as_session() / _as_workspace().

    Signaux Qt :
      gatingStrategyChanged(gate_id, change_type) — "added"|"modified"|"removed"
      analysisCompleted(sample_id)
      analysisError(sample_id, error_message)

    Extensions PRISMA non natives FlowKit :
      - time_gate()              : filtrage sur canal Time
      - compute_spillover_matrix(): calcul spillover depuis single-stained
      - load_facsdiva_xml()      : import FACSDiva XML
    """

    if _QT_AVAILABLE:
        gatingStrategyChanged = pyqtSignal(str, str)  # gate_id, change_type
        analysisCompleted = pyqtSignal(str)            # sample_id
        analysisError = pyqtSignal(str, str)           # sample_id, error_message

    def __init__(self, parent=None) -> None:
        if _QT_AVAILABLE:
            try:
                super().__init__(parent)
            except TypeError:
                super().__init__()
        if FLOWKIT_AVAILABLE and fk is not None:
            self._backend = fk.Session()
        else:
            self._backend = _UnavailableFlowKitBackend(_FLOWKIT_IMPORT_ERROR)
        self.log = _logger
        self._active_sample_id: Optional[str] = None
        self._active_group: Optional[str] = None
        self._transform_specs: Dict[str, TransformSpec] = {}
        self._comp_matrix_ids: List[str] = []
        # Registre (gate_id → (x_channel, y_channel)) — source de vérité dimensionnelle
        self._gate_dimensions: Dict[str, Tuple[str, Optional[str]]] = {}
        # Cache membership : (gate_id, sample_id) → np.ndarray[bool]
        self._membership_cache: Dict[Tuple[str, str], np.ndarray] = {}
        # File d'attente analyse async
        self._analysis_pending: bool = False
        self._pending_sample_id: Optional[str] = None
        if FLOWKIT_AVAILABLE:
            _logger.info("PrismaFlowEngine initialisé (QObject)")
        else:
            _logger.warning(
                "PrismaFlowEngine initialisé en mode dégradé: FlowKit indisponible (%s)",
                _FLOWKIT_IMPORT_ERROR,
            )

    # ------------------------------------------------------------------
    # A. Accès backend
    # ------------------------------------------------------------------

    @property
    def backend(self) -> FlowKitBackend:
        return self._backend

    @property
    def session(self) -> fk.Session:
        """Accès direct si Session (raises si Workspace)."""
        if not FLOWKIT_AVAILABLE or fk is None:
            raise PrismaEngineError("FlowKit est indisponible: impossible d'accéder à une Session.")
        if not isinstance(self._backend, fk.Session):
            raise PrismaEngineError(
                "Le backend actuel est un Workspace. Utilisez .backend directement."
            )
        return self._backend

    def _is_workspace(self) -> bool:
        if not FLOWKIT_AVAILABLE or fk is None:
            return False
        return isinstance(self._backend, fk.Workspace)

    def _as_workspace(self) -> fk.Workspace:
        if not FLOWKIT_AVAILABLE or fk is None:
            raise PrismaEngineError("FlowKit est indisponible: impossible d'utiliser un Workspace.")
        if not isinstance(self._backend, fk.Workspace):
            raise PrismaEngineError("Backend n'est pas un Workspace.")
        return self._backend

    def _as_session(self) -> fk.Session:
        if not FLOWKIT_AVAILABLE or fk is None:
            raise PrismaEngineError("FlowKit est indisponible: impossible d'utiliser une Session.")
        if not isinstance(self._backend, fk.Session):
            raise PrismaEngineError("Backend n'est pas une Session.")
        return self._backend

    def _require_flowkit(self) -> None:
        if not FLOWKIT_AVAILABLE:
            raise PrismaEngineError("FlowKit n'est pas installé dans cet environnement.")

    @property
    def active_sample_id(self) -> Optional[str]:
        return self._active_sample_id

    @property
    def sample_id(self) -> Optional[str]:
        """Alias de compatibilité pour l'échantillon actif."""
        return self._active_sample_id

    @property
    def active_group(self) -> Optional[str]:
        return self._active_group

    # ------------------------------------------------------------------
    # API compatibilité dimensionnelle (source de vérité des gates)
    # ------------------------------------------------------------------

    def gate_dimensions(self, gate_id: str) -> Optional[Tuple[str, Optional[str]]]:
        """Retourne (x_channel, y_channel) de la gate, ou None si non enregistrée."""
        return self._gate_dimensions.get(gate_id)

    def is_gate_compatible_with(self, gate_id: str, x_ch: str, y_ch: Optional[str]) -> bool:
        """True si la gate est définie sur exactement ces dimensions."""
        dims = self._gate_dimensions.get(gate_id)
        if dims is None:
            return False
        return dims[0] == x_ch and dims[1] == (y_ch or None)

    def get_compatible_gate_ids(self, x_ch: str, y_ch: Optional[str]) -> List[str]:
        """Retourne tous les gate_ids compatibles avec les axes donnés."""
        return [
            gid for gid, dims in self._gate_dimensions.items()
            if dims[0] == x_ch and dims[1] == (y_ch or None)
        ]

    def _register_gate_dimensions(
        self,
        gate_id: str,
        x_ch: str,
        y_ch: Optional[str],
    ) -> None:
        self._gate_dimensions[gate_id] = (x_ch, y_ch or None)

    def _unregister_gate_dimensions(self, gate_id: str) -> None:
        self._gate_dimensions.pop(gate_id, None)

    def _invalidate_membership_cache(self, sample_id: Optional[str] = None) -> None:
        if sample_id:
            keys = [k for k in self._membership_cache if k[1] == sample_id]
        else:
            keys = list(self._membership_cache.keys())
        for k in keys:
            del self._membership_cache[k]

    def get_gate_membership_cached(
        self,
        gate_id: str,
        sample_id: str,
        gate_path: Optional[Tuple[str, ...]] = None,
    ) -> Optional[np.ndarray]:
        """Membership array avec cache. Invalider via _invalidate_membership_cache()."""
        key = (gate_id, sample_id)
        if key not in self._membership_cache:
            try:
                arr = self._backend.get_gate_membership(sample_id, gate_id, gate_path=gate_path)
                self._membership_cache[key] = np.asarray(arr, dtype=bool).flatten()
            except Exception as exc:
                _logger.debug("get_gate_membership_cached '%s': %s", gate_id, exc)
                return None
        return self._membership_cache[key]

    def get_gate_vertices(self, gate_id: str) -> Optional[List[Tuple[float, float]]]:
        """
        Retourne les vertices d'une gate depuis FlowKit.
        Supporte PolygonGate et RectangleGate. Retourne None sinon.
        """
        try:
            paths = self.find_gate_paths(gate_id)
            gate_path = paths[0] if paths else None
            gate_obj = self._backend.get_gate(gate_id, gate_path=gate_path)
            if gate_obj is None:
                return None
            cls_name = type(gate_obj).__name__
            if "Polygon" in cls_name:
                # FlowKit PolygonGate.vertices = list of (x, y)
                verts = getattr(gate_obj, "vertices", None)
                if verts is not None:
                    return [(float(v[0]), float(v[1])) for v in verts]
            if "Rectangle" in cls_name:
                dims = getattr(gate_obj, "dimensions", [])
                if len(dims) >= 2:
                    x_min = getattr(dims[0], "min", None)
                    x_max = getattr(dims[0], "max", None)
                    y_min = getattr(dims[1], "min", None)
                    y_max = getattr(dims[1], "max", None)
                    if None not in (x_min, x_max, y_min, y_max):
                        return [
                            (float(x_min), float(y_min)),
                            (float(x_max), float(y_min)),
                            (float(x_max), float(y_max)),
                            (float(x_min), float(y_max)),
                        ]
        except Exception as exc:
            _logger.debug("get_gate_vertices '%s': %s", gate_id, exc)
        return None

    def get_gate_vertices_in_transform_space(
        self, gate_id: str, transform_id: Optional[str]
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Retourne les vertices d'une gate dans l'espace transformé courant.

        Pour l'instant, retourne get_gate_vertices() directement : les vertices
        FlowKit sont déjà en coordonnées transformées lors de la création via
        create_polygon_gate_from_vertices avec transform_ref.

        Limitation : si transform_id diffère de celui utilisé lors de la création,
        les vertices retournés sont dans l'espace de création d'origine et non dans
        l'espace de la nouvelle transformation. Un recalcul serait nécessaire ici
        via la transformation inverse puis forward, ce qui n'est pas encore implémenté.

        Args:
            gate_id:      Identifiant de la gate.
            transform_id: ID de transformation courant (ignoré pour l'instant).

        Returns:
            Liste de tuples (x, y) ou None si la gate est introuvable.
        """
        return self.get_gate_vertices(gate_id)

    def get_population_df_sampled(
        self,
        gate_node: Tuple[str, ...],
        max_events: int = 50_000,
        sample_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        comp_matrix_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Comme get_population_df() mais avec sous-échantillonnage stratifié pour le rendu.

        Les calculs statistiques doivent utiliser get_population_df() (sans limitation).
        Sous-échantillonne via df.sample(min(max_events, len(df)), random_state=42).

        Args:
            gate_node:      Chemin vers la gate (ex: ('root',) ou ('Leuco', 'root')).
            max_events:     Nombre maximum d'événements dans le DataFrame retourné.
            sample_id:      Sample actif si None.
            transform_id:   ID de transformation.
            comp_matrix_id: ID de matrice de compensation.

        Returns:
            DataFrame sous-échantillonné prêt pour l'affichage.
        """
        df = self.get_population_df(
            gate_node,
            sample_id=sample_id,
            transform_id=transform_id,
            comp_matrix_id=comp_matrix_id,
        )
        n = len(df)
        if n <= max_events:
            return df
        return df.sample(max_events, random_state=42).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Analyse asynchrone
    # ------------------------------------------------------------------

    def analyze_async(self, sample_id: Optional[str] = None) -> None:
        """
        Lance analyze() dans un QThreadPool. Émet analysisCompleted quand terminé.

        File d'attente simple : si une analyse est déjà en cours, le sample_id
        est mémorisé et relancé à la fin de l'analyse courante.
        """
        if not _QT_AVAILABLE:
            self._run_analyze_sync(sample_id=sample_id)
            return

        if getattr(self, "_analysis_pending", False):
            # Une analyse tourne déjà — mémoriser la demande
            self._pending_sample_id: Optional[str] = sample_id
            _logger.debug(
                "analyze_async: analyse déjà en cours — sample '%s' mis en attente", sample_id
            )
            return

        self._analysis_pending = True
        self._pending_sample_id = None
        worker = _AnalysisWorker(self, sample_id)
        worker.signals.finished.connect(self._on_analysis_finished)
        worker.signals.error.connect(self._on_analysis_error)
        QThreadPool.globalInstance().start(worker)

    def _emit_strategy_changed(self, gate_id: str, change_type: str) -> None:
        """Émet gatingStrategyChanged de façon défensive (guard si QObject non init)."""
        if not _QT_AVAILABLE:
            return
        try:
            self.gatingStrategyChanged.emit(gate_id, change_type)
        except RuntimeError:
            pass

    def _on_analysis_finished(self, sample_id: str) -> None:
        self._analysis_pending = False
        self._invalidate_membership_cache(sample_id=sample_id or None)
        try:
            self.analysisCompleted.emit(sample_id)
        except RuntimeError:
            pass
        # Relancer si une analyse était en attente
        pending = getattr(self, "_pending_sample_id", None)
        if pending is not None:
            self._pending_sample_id = None
            _logger.debug("analyze_async: relancement analyse en attente pour '%s'", pending)
            self.analyze_async(pending)

    def _on_analysis_error(self, sample_id: str, error_msg: str) -> None:
        try:
            self.analysisError.emit(sample_id, error_msg)
        except RuntimeError:
            pass

    def _run_analyze_sync(self, sample_id: Optional[str] = None) -> None:
        """Analyse synchrone interne — appelé depuis le worker thread."""
        self.analyze(sample_id=sample_id, use_mp=False, verbose=False)

    def set_active_sample(self, sample_id: str) -> None:
        self._assert_sample_exists(sample_id)
        self._active_sample_id = sample_id

    def set_active_group(self, group_name: str) -> None:
        """Définit le groupe actif (Workspace uniquement)."""
        groups = self.get_sample_groups()
        if group_name not in groups:
            raise PrismaEngineError(f"Groupe inconnu : '{group_name}'. Disponibles : {groups}")
        self._active_group = group_name
        _logger.debug("Groupe actif : %s", group_name)

    def get_sample_ids(self, group_name: Optional[str] = None) -> List[str]:
        """
        Retourne les sample_ids.

        Args:
            group_name: Si fourni et backend=Workspace, filtre par groupe.
        """
        if group_name is not None and self._is_workspace():
            return list(self._as_workspace().get_sample_ids(group_name=group_name))
        return list(self._backend.get_sample_ids())

    def get_sample_channels(self, sample_id: Optional[str] = None) -> List[str]:
        """Retourne les noms de canaux (pnn normalisé depuis tuples FlowKit)."""
        sid = sample_id or self._require_active_sample()
        sample = self._backend.get_sample(sid)
        df = sample.as_dataframe(source="raw")
        return [col[0] if isinstance(col, tuple) else str(col) for col in df.columns]

    def get_channel_marker_mapping(self, sample_id: Optional[str] = None) -> Dict[str, str]:
        """
        Retourne le mapping Pnn -> PnS (fallback Pnn si PnS absent).

        Le mapping est utilisé par l'UI pour afficher "Marqueur (Canal)"
        tout en conservant le canal Pnn en userData.
        """
        sid = sample_id or self._require_active_sample()
        sample = self._backend.get_sample(sid)

        def _norm(values: Any) -> List[str]:
            if values is None:
                return []
            return [str(v).strip() for v in list(values)]

        pnn_labels = _norm(getattr(sample, "pnn_labels", None))
        pns_labels = _norm(getattr(sample, "pns_labels", None))

        mapping: Dict[str, str] = {}
        if pnn_labels:
            for i, pnn in enumerate(pnn_labels):
                if not pnn:
                    continue
                pns = pns_labels[i] if i < len(pns_labels) else ""
                mapping[pnn] = pns or pnn

        if not mapping:
            channels = self.get_sample_channels(sample_id=sid)
            for ch in channels:
                mapping[str(ch)] = str(ch)

        return mapping

    def get_active_sample(self) -> fk.Sample:
        sid = self._require_active_sample()
        return self._backend.get_sample(sid)

    # ------------------------------------------------------------------
    # B. Groupes de samples (Workspace)
    # ------------------------------------------------------------------

    def get_sample_groups(self) -> List[str]:
        """
        Retourne les groupes de samples FlowJo.

        Retourne ['All Samples'] si backend=Session (pas de groupes).
        """
        if self._is_workspace():
            return list(self._as_workspace().get_sample_groups())
        return ["All Samples"]

    def get_samples_in_group(self, group_name: str) -> List[fk.Sample]:
        """Retourne les objets Sample d'un groupe (Workspace uniquement)."""
        if not self._is_workspace():
            _logger.debug("get_samples_in_group : backend=Session, groupe ignoré")
            return [self._backend.get_sample(sid) for sid in self.get_sample_ids()]
        return list(self._as_workspace().get_samples(group_name=group_name))

    def get_workspace_keywords(self, sample_id: str) -> Dict[str, Any]:
        """Retourne les keywords FCS d'un sample (Workspace uniquement)."""
        if not self._is_workspace():
            return {}
        return dict(self._as_workspace().get_keywords(sample_id))

    def get_workspace_summary(self) -> str:
        """Résumé textuel du workspace FlowJo (Workspace uniquement)."""
        if not self._is_workspace():
            return "Backend: Session (pas de workspace FlowJo chargé)"
        return str(self._as_workspace().summary())

    def get_gating_strategy(self, sample_id: str) -> Any:
        """Retourne la GatingStrategy d'un sample (Workspace uniquement)."""
        return self._as_workspace().get_gating_strategy(sample_id)

    # ------------------------------------------------------------------
    # C. I/O
    # ------------------------------------------------------------------

    def load_fcs(self, fcs_path: Union[str, Path], make_active: bool = True) -> str:
        """Charge un FCS dans le backend Session avec sample_id sûr et logs explicites."""
        self._require_flowkit()
        fcs_path = Path(fcs_path)
        if not fcs_path.is_file():
            raise FileNotFoundError(f"FCS introuvable : {fcs_path}")

        if self._is_workspace():
            raise PrismaEngineError(
                "load_fcs() non disponible sur Workspace. "
                "Chargez le WSP avec load_wsp() qui inclut les FCS."
            )

        session = self._as_session()
        try:
            # ID stable/sûr: nom de fichier sans extension, caractères incompatibles normalisés.
            base_id = "".join(ch if (ch.isalnum() or ch in "_-.") else "_" for ch in fcs_path.stem)
            base_id = base_id.strip("._") or "sample"

            existing_ids = set(session.get_sample_ids())
            sample_id = base_id
            suffix = 1
            while sample_id in existing_ids:
                sample_id = f"{base_id}_{suffix}"
                suffix += 1

            try:
                sample = fk.Sample(str(fcs_path), sample_id=sample_id)
            except TypeError:
                # Compatibilité API FlowKit: certains builds n'acceptent pas sample_id au constructeur.
                sample = fk.Sample(str(fcs_path))
                try:
                    sample.id = sample_id
                except Exception:
                    sample_id = str(getattr(sample, "id", sample_id))

            if hasattr(session, "add_sample"):
                session.add_sample(sample)
            else:
                session.add_samples(sample)

            loaded_id = str(getattr(sample, "id", sample_id))
            if make_active:
                self._active_sample_id = loaded_id

            # Auto-compensation depuis métadonnées FCS (SPILL/SPILLOVER)
            # Enregistre la matrice dans la session ET l'applique au Sample natif
            try:
                meta = {}
                try:
                    meta = dict(sample.get_metadata())
                except Exception:
                    meta = dict(getattr(sample, "metadata", {}) or {})
                spill_key = next(
                    (
                        k
                        for k in meta
                        if str(k).strip().lower() in {"spill", "spillover", "$spill", "$spillover"}
                    ),
                    None,
                )
                if spill_key is not None:
                    spill_value = meta[spill_key]
                    # FlowKit Matrix(spill_data, detectors) — les détecteurs sont les canaux fluoro
                    fluoro_pnn = [sample.pnn_labels[i] for i in sample.fluoro_indices]
                    fluoro_pns = [sample.pns_labels[i] for i in sample.fluoro_indices]
                    matrix_id = f"spill_{loaded_id}"
                    matrix = fk.Matrix(spill_value, detectors=fluoro_pnn, fluorochromes=fluoro_pns)
                    # Enregistrer dans la session (pour get_comp_matrix() ultérieur)
                    session.add_comp_matrix(matrix_id, matrix)
                    if matrix_id not in self._comp_matrix_ids:
                        self._comp_matrix_ids.append(matrix_id)
                    self.log.info("Auto-compensation enregistrée : %s", matrix_id)
            except Exception as exc_comp:
                self.log.debug("Auto-compensation ignorée pour %s : %s", loaded_id, exc_comp)

            self.log.info("FCS chargé et prêt: %s -> %s", fcs_path.name, loaded_id)
            return loaded_id
        except Exception as exc:
            self.log.error(
                "Echec load_fcs('%s') dans PrismaFlowEngine: %s",
                str(fcs_path),
                exc,
                exc_info=True,
            )
            raise

    def load_fcs_batch(
        self,
        sources: Union[List[Union[str, Path]], Union[str, Path]],
        make_first_active: bool = True,
    ) -> List[str]:
        """Charge un dossier ou liste de FCS dans la Session."""
        self._require_flowkit()
        if self._is_workspace():
            raise PrismaEngineError("load_fcs_batch() non disponible sur Workspace.")

        sources_path = Path(str(sources)) if not isinstance(sources, list) else None
        if sources_path and sources_path.is_dir():
            files = sorted(sources_path.glob("*.fcs")) + sorted(sources_path.glob("*.FCS"))
        else:
            files = [Path(p) for p in (sources if isinstance(sources, list) else [])]

        if not files:
            raise FileNotFoundError("Aucun FCS trouvé dans la source fournie.")

        ids: List[str] = []
        for f in files:
            try:
                sid = self.load_fcs(f, make_active=False)
                ids.append(sid)
            except Exception as exc:
                _logger.warning("Échec chargement %s : %s", f.name, exc)

        if ids and make_first_active:
            self._active_sample_id = ids[0]
        _logger.info("%d FCS chargés", len(ids))
        return ids

    def load_wsp(
        self,
        wsp_path: Union[str, Path],
        fcs_dir: Optional[Union[str, Path]] = None,
    ) -> List[str]:
        """
        Charge un workspace FlowJo (.wsp) via fk.Workspace (API stable 1.3.0).

        Remplace le backend courant par le Workspace chargé.
        Toutes les méthodes communes (get_gate_events, analyze_samples, etc.)
        continuent de fonctionner — Workspace expose la même API.

        Args:
            wsp_path: Chemin .wsp.
            fcs_dir:  Dossier racine FCS si chemins WSP invalides.
        """
        self._require_flowkit()
        wsp_path = Path(wsp_path)
        if not wsp_path.is_file():
            raise FileNotFoundError(f"WSP introuvable : {wsp_path}")

        fcs_samples = None
        if fcs_dir is not None:
            fcs_dir = Path(fcs_dir)
            fcs_files = sorted(fcs_dir.glob("*.fcs")) + sorted(fcs_dir.glob("*.FCS"))
            if fcs_files:
                fcs_samples = [fk.Sample(str(f)) for f in fcs_files]

        try:
            workspace = fk.Workspace(
                str(wsp_path),
                fcs_samples=fcs_samples,
                find_fcs_files_from_wsp=fcs_samples is None,
            )
        except Exception as exc:
            raise PrismaEngineError(f"Échec ouverture WSP '{wsp_path.name}' : {exc}") from exc

        ids = list(workspace.get_sample_ids())
        if not ids:
            raise PrismaEngineError(f"WSP '{wsp_path.name}' : aucun sample.")

        self._backend = workspace
        self._active_sample_id = ids[0]
        self._active_group = None
        _logger.info("WSP chargé : %s → %d samples", wsp_path.name, len(ids))
        return ids

    def export_wsp(self, output_path: Union[str, Path], group_name: str = "PRISMA") -> None:
        """Exporte en workspace FlowJo (.wsp) — Session uniquement."""
        self._require_flowkit()
        if self._is_workspace():
            raise PrismaEngineError(
                "export_wsp() non disponible sur Workspace chargé. "
                "Utilisez archive_results() à la place."
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            self._as_session().export_wsp(fh, group_name=group_name)
        _logger.info("WSP exporté : %s", output_path)

    def archive_workspace_results(
        self,
        group_name: str,
        out_dir: Union[str, Path],
        output_prefix: Optional[str] = None,
    ) -> None:
        """Archive les résultats d'analyse d'un groupe (Workspace uniquement)."""
        self._require_flowkit()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._as_workspace().archive_results(
            group_name=group_name,
            out_dir=str(out_dir),
            output_prefix=output_prefix,
            overwrite=True,
        )
        _logger.info("Résultats archivés : %s → %s", group_name, out_dir)

    def load_gml(self, gml_path: Union[str, Path]) -> None:
        """
        Importe une stratégie GatingML 2.0 dans la Session courante.

        Stratégie : parse le XML → GatingStrategy → recrée la Session
        en réinjectant les samples déjà chargés.
        Compatible FlowKit ≥ 0.9 (API parse_gating_xml + Session(gating_strategy=...)).
        """
        self._require_flowkit()
        if self._is_workspace():
            raise PrismaEngineError("load_gml() non disponible sur Workspace.")
        gml_path = Path(gml_path)
        if not gml_path.is_file():
            raise FileNotFoundError(f"GatingML introuvable : {gml_path}")

        # Récupérer les samples existants avant de reconstruire la session
        try:
            existing_ids = list(self._backend.get_sample_ids())
        except Exception:
            existing_ids = []

        # Parser le GatingML → GatingStrategy
        try:
            strategy = fk.parse_gating_xml(str(gml_path))
        except AttributeError:
            # Fallback si parse_gating_xml absent (FlowKit très ancien)
            strategy = None

        if strategy is not None:
            # Reconstruire la Session avec la stratégie importée
            new_session = fk.Session(gating_strategy=strategy)
        else:
            # Dernier fallback : Session vide (perd la stratégie)
            _logger.warning("parse_gating_xml indisponible — stratégie GML non chargée")
            new_session = fk.Session()

        # Réinjecter les samples
        for sid in existing_ids:
            try:
                new_session.add_samples(self._backend.get_sample(sid))
            except Exception as e:
                _logger.debug("Réinjection sample '%s' ignorée : %s", sid, e)

        self._backend = new_session
        # Invalider le cache membership (stratégie changée)
        self._invalidate_membership_cache()
        _logger.info("GatingML importé : %s", gml_path.name)

    def export_gml(self, output_path: Union[str, Path], sample_id: Optional[str] = None) -> None:
        """
        Exporte la stratégie en GatingML 2.0 (Session uniquement).

        API FlowKit ≥ 0.9 : fk.export_gatingml(gating_strategy, file_handle).
        Fallback sur session.export_gml() si l'ancienne API existe.
        """
        self._require_flowkit()
        if self._is_workspace():
            raise PrismaEngineError("export_gml() non disponible sur Workspace.")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import io as _io

        session = self._as_session()

        # Tenter la nouvelle API en priorité : fk.export_gatingml(strategy, fh)
        try:
            strategy = session.gating_strategy
            buf = _io.BytesIO()
            fk.export_gatingml(strategy, buf, sample_id=sample_id)
            content = buf.getvalue()
        except (AttributeError, TypeError):
            # Fallback : ancienne API session.export_gml(fh) si elle existe
            try:
                buf = _io.BytesIO()
                session.export_gml(buf, sample_id=sample_id)
                content = buf.getvalue()
            except (AttributeError, TypeError):
                buf_text = _io.StringIO()
                session.export_gml(buf_text, sample_id=sample_id)
                content = buf_text.getvalue().encode("utf-8")

        with open(output_path, "wb") as fh:
            fh.write(content)
        _logger.info("GatingML exporté : %s", output_path)

    # ------------------------------------------------------------------
    # D. Compensation
    # ------------------------------------------------------------------

    def apply_spillover_matrix(self, sample_id: Optional[str] = None) -> Optional[str]:
        """
        Active la compensation depuis les métadonnées FCS (TEXT segment), si présente.

        Returns:
            matrix_id ajouté à la session, ou None si aucune matrice n'est trouvée.
        """
        sid = sample_id or self._require_active_sample()
        if self._is_workspace():
            raise PrismaEngineError(
                "apply_spillover_matrix() non disponible sur Workspace FlowJo chargé."
            )

        sample = self._backend.get_sample(sid)
        metadata: Dict[str, Any] = {}
        try:
            metadata = dict(sample.get_metadata())
        except Exception:
            metadata = dict(getattr(sample, "metadata", {}) or {})

        spill_value: Any = None
        if metadata:
            for key, value in metadata.items():
                if str(key).strip().lower() in {"spill", "spillover", "$spill", "$spillover"}:
                    spill_value = value
                    break

        if spill_value is None:
            self.log.warning("Aucune matrice spillover trouvée dans les métadonnées de %s", sid)
            return None

        try:
            matrix_id = f"spill_{sid}"
            fluoro_pnn = [sample.pnn_labels[i] for i in sample.fluoro_indices]
            fluoro_pns = [sample.pns_labels[i] for i in sample.fluoro_indices]
            matrix = fk.Matrix(spill_value, detectors=fluoro_pnn, fluorochromes=fluoro_pns)
            self._as_session().add_comp_matrix(matrix_id, matrix)
            if matrix_id not in self._comp_matrix_ids:
                self._comp_matrix_ids.append(matrix_id)
            self.log.info("Compensation activée depuis métadonnées FCS: %s", matrix_id)
            return matrix_id
        except Exception as exc:
            self.log.error(
                "Echec apply_spillover_matrix(sample_id=%s): %s", sid, exc, exc_info=True
            )
            raise

    def load_compensation_from_fcs_metadata(self, sample_id: Optional[str] = None) -> Optional[str]:
        """Lit la matrice spillover depuis les métadonnées FCS."""
        sid = sample_id or self._require_active_sample()
        sample = self._backend.get_sample(sid)
        meta = sample.get_metadata()

        spill_key = next(
            (k for k in meta if k.lower() in ("spill", "spillover", "$spill", "$spillover")),
            None,
        )
        if spill_key is None:
            _logger.warning("Pas de matrice spillover dans les métadonnées FCS de %s", sid)
            return None

        try:
            matrix_id = f"spill_{sid}"
            fluoro_pnn = [sample.pnn_labels[i] for i in sample.fluoro_indices]
            fluoro_pns = [sample.pns_labels[i] for i in sample.fluoro_indices]
            matrix = fk.Matrix(meta[spill_key], detectors=fluoro_pnn, fluorochromes=fluoro_pns)
            if not self._is_workspace():
                self._as_session().add_comp_matrix(matrix_id, matrix)
            if matrix_id not in self._comp_matrix_ids:
                self._comp_matrix_ids.append(matrix_id)
            _logger.info("Matrice de compensation FCS chargée : %s", matrix_id)
            return matrix_id
        except Exception as exc:
            _logger.error("Erreur lecture matrice FCS : %s", exc)
            return None

    def load_compensation_from_csv(
        self, csv_path: Union[str, Path], matrix_id: Optional[str] = None
    ) -> str:
        """Charge une matrice de compensation depuis un CSV externe."""
        if self._is_workspace():
            raise PrismaEngineError(
                "add_comp_matrix() non disponible sur Workspace. "
                "La compensation est définie dans le WSP FlowJo."
            )
        csv_path = Path(csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV compensation introuvable : {csv_path}")

        df = pd.read_csv(csv_path, index_col=0)
        spill_array = df.values.astype(np.float64)
        detectors = list(df.columns)
        fluorochromes = list(df.index)
        matrix_id = matrix_id or f"comp_{csv_path.stem}"
        matrix = fk.Matrix(spill_array, detectors=detectors, fluorochromes=fluorochromes)
        self._as_session().add_comp_matrix(matrix_id, matrix)
        self._comp_matrix_ids.append(matrix_id)
        _logger.info("Matrice compensation CSV chargée : %s", matrix_id)
        return matrix_id

    def compute_spillover_from_single_stained(
        self,
        controls: Dict[str, Union[str, Path]],
        detector_channel_map: Dict[str, str],
        matrix_id: str = "comp_single_stained",
    ) -> str:
        """
        Calcule une matrice de compensation depuis des contrôles single-stained.

        Extension PRISMA — non native FlowKit. Algorithme classique :
          Pour chaque fluorochrome F, le contrôle SS_F donne la colonne
          de spillover : MFI dans chaque détecteur / MFI dans le détecteur primaire.

        Args:
            controls: {fluorochrome_name: fcs_path} pour chaque SS control.
            detector_channel_map: {fluorochrome_name: primary_detector_channel}.
            matrix_id: Identifiant de la matrice produite.

        Returns:
            matrix_id enregistré dans la session.

        Raises:
            PrismaEngineError: si backend=Workspace.
            ValueError: si les canaux sont incompatibles.
        """
        if self._is_workspace():
            raise PrismaEngineError(
                "compute_spillover_from_single_stained() requiert un backend Session."
            )

        fluorochromes = list(controls.keys())
        detectors = [detector_channel_map[f] for f in fluorochromes]

        all_channels: Optional[List[str]] = None
        spill_cols: List[np.ndarray] = []

        for fluoro, fcs_path in controls.items():
            sample = fk.Sample(str(fcs_path))
            df = sample.as_dataframe(source="raw")
            # Normaliser les colonnes tuple
            cols = {(c[0] if isinstance(c, tuple) else str(c)): c for c in df.columns}
            if all_channels is None:
                all_channels = list(cols.keys())

            primary_det = detector_channel_map[fluoro]
            if primary_det not in cols:
                raise ValueError(
                    f"Canal primaire '{primary_det}' absent du FCS {fcs_path}. "
                    f"Canaux disponibles : {list(cols.keys())}"
                )

            # MFI (médiane) dans chaque détecteur
            mfi_primary = float(df[cols[primary_det]].median())
            if mfi_primary <= 0:
                _logger.warning(
                    "MFI primaire nul pour %s (%s) — colonne de spillover mise à 0",
                    fluoro,
                    primary_det,
                )
                spill_col = np.zeros(len(detectors), dtype=np.float64)
            else:
                spill_col = np.array(
                    [
                        float(df[cols[d]].median()) / mfi_primary if d in cols else 0.0
                        for d in detectors
                    ],
                    dtype=np.float64,
                )
                # Normaliser : diagonale = 1.0
                idx_primary = detectors.index(primary_det)
                if spill_col[idx_primary] > 0:
                    spill_col /= spill_col[idx_primary]

            spill_cols.append(spill_col)

        # Matrice n×n (fluorochromes × détecteurs)
        spill_matrix = np.column_stack(spill_cols).T  # shape (n_fluoro, n_det)

        matrix = fk.Matrix(
            spill_matrix,
            detectors=detectors,
            fluorochromes=fluorochromes,
        )
        self._as_session().add_comp_matrix(matrix_id, matrix)
        self._comp_matrix_ids.append(matrix_id)
        _logger.info(
            "Matrice compensation single-stained calculée : %s (%d×%d)",
            matrix_id,
            len(fluorochromes),
            len(detectors),
        )
        return matrix_id

    def get_comp_matrix_ids(self) -> List[str]:
        return list(self._comp_matrix_ids)

    # ------------------------------------------------------------------
    # E. Transformations
    # ------------------------------------------------------------------

    def add_transform(self, transform_id: str, spec: TransformSpec) -> None:
        """Ajoute une transformation dans la Session (non disponible Workspace)."""
        if self._is_workspace():
            raise PrismaEngineError(
                "add_transform() non disponible sur Workspace. "
                "Les transformations sont définies dans le WSP FlowJo."
            )
        p = spec.merged_params()
        kind = spec.kind.lower()
        transform: Any

        if kind == "logicle":
            transform = fk_transforms.LogicleTransform(
                param_t=p["param_t"],
                param_w=p["param_w"],
                param_m=p["param_m"],
                param_a=p["param_a"],
            )
        elif kind == "log":
            transform = fk_transforms.LogTransform(param_t=p["param_t"], param_m=p["param_m"])
        elif kind == "linear":
            transform = fk_transforms.LinearTransform(param_t=p["param_t"], param_a=p["param_a"])
        elif kind == "hyperlog":
            transform = fk_transforms.HyperlogTransform(
                param_t=p["param_t"],
                param_w=p["param_w"],
                param_m=p["param_m"],
                param_a=p["param_a"],
            )
        elif kind == "asinh":
            transform = fk_transforms.AsinhTransform(
                param_t=p["param_t"],
                param_m=p["param_m"],
                param_a=p["param_a"],
            )
        else:
            raise PrismaEngineError(f"Type de transformation inconnu : {kind!r}")

        self._as_session().add_transform(transform_id, transform)
        self._transform_specs[transform_id] = spec
        _logger.info("Transformation ajoutée : %s (%s)", transform_id, kind)

    def add_logicle_transform_to_all(
        self,
        transform_id: str = "logicle_default",
        param_t: float = 262144,
        param_w: float = 0.5,
        param_m: float = 4.5,
        param_a: float = 0.0,
    ) -> str:
        """Raccourci : ajoute une transformation Logicle globale."""
        spec = TransformSpec(
            kind="logicle",
            params={"param_t": param_t, "param_w": param_w, "param_m": param_m, "param_a": param_a},
        )
        self.add_transform(transform_id, spec)
        return transform_id

    def apply_logicle_transform(
        self,
        transform_id: str = "logicle_default",
        param_t: float = 262144,
        param_m: float = 4.5,
        param_w: float = 0.5,
        param_a: float = 0.0,
        sample_id: Optional[str] = None,
    ) -> str:
        """
        Crée/active une transformation Logicle pour la vue et le gating.

        Les canaux de diffusion/temps (FSC/SSC/TIME) sont exclus du scope
        de référence afin d'aligner l'usage cytométrie standard.
        """
        if self._is_workspace():
            raise PrismaEngineError(
                "apply_logicle_transform() non disponible sur Workspace FlowJo chargé."
            )

        sid = sample_id or self._require_active_sample()
        channels = self.get_sample_channels(sample_id=sid)
        fluorescence_channels = [
            ch
            for ch in channels
            if all(token not in ch.upper() for token in ("FSC", "SSC", "TIME"))
        ]

        if transform_id not in self._transform_specs:
            spec = TransformSpec(
                kind="logicle",
                channel_ids=fluorescence_channels,
                params={
                    "param_t": param_t,
                    "param_m": param_m,
                    "param_w": param_w,
                    "param_a": param_a,
                },
            )
            self.add_transform(transform_id, spec)
        else:
            self.log.debug("Transformation '%s' déjà active", transform_id)

        self.log.info(
            "Logicle active: id=%s, canaux_fluorescents=%d",
            transform_id,
            len(fluorescence_channels),
        )
        return transform_id

    def apply_transformation(self, transform_type: str = "Logicle") -> Optional[str]:
        """
        Instancie et applique dynamiquement la transformation sélectionnée.

        Correctif critique: applique explicitement la transformation au sample actif
        pour éviter l'erreur "Transformed events were requested but do not exist".
        """
        self._require_flowkit()
        if self._is_workspace():
            raise PrismaEngineError(
                "apply_transformation() non disponible sur Workspace FlowJo chargé."
            )

        sid = self.sample_id or self._require_active_sample()
        if not sid:
            return None

        transform_name = str(transform_type or "Logicle").strip().title()
        transform_id = f"transform_{transform_name.lower()}"

        # Paramètres de référence cytométrie
        params = {
            "param_t": 262144,
            "param_m": 4.5,
            "param_w": 0.5,
            "param_a": 0.0,
        }

        def _build_xform() -> Any:
            # Compatibilité API FlowKit: signatures différentes selon versions.
            if transform_name == "Logicle":
                for kwargs in (
                    {"t": 262144, "m": 4.5, "w": 0.5, "a": 0.0},
                    {"param_t": 262144, "param_m": 4.5, "param_w": 0.5, "param_a": 0.0},
                ):
                    try:
                        return fkt.LogicleTransform(**kwargs)
                    except TypeError:
                        continue
            elif transform_name == "Hyperlog":
                for kwargs in (
                    {"t": 262144, "m": 4.5, "w": 0.5, "a": 0.0},
                    {"param_t": 262144, "param_m": 4.5, "param_w": 0.5, "param_a": 0.0},
                ):
                    try:
                        return fkt.HyperlogTransform(**kwargs)
                    except TypeError:
                        continue
            elif transform_name == "Asinh":
                for kwargs in (
                    {"t": 262144, "m": 4.5, "a": 0.0},
                    {"param_t": 262144, "param_m": 4.5, "param_a": 0.0},
                ):
                    try:
                        return fkt.AsinhTransform(**kwargs)
                    except TypeError:
                        continue
            elif transform_name == "Log":
                for kwargs in (
                    {"t": 262144, "m": 4.5},
                    {"param_t": 262144, "param_m": 4.5},
                ):
                    try:
                        return fkt.LogTransform(**kwargs)
                    except TypeError:
                        continue
            elif transform_name == "Linear":
                for kwargs in (
                    {"a": 1.0, "t": 262144},
                    {"param_a": 1.0, "param_t": 262144},
                ):
                    try:
                        return fkt.LinearTransform(**kwargs)
                    except TypeError:
                        continue

            # Fallback robuste
            try:
                return fkt.LogicleTransform(t=262144, m=4.5, w=0.5, a=0.0)
            except TypeError:
                return fkt.LogicleTransform(
                    param_t=params["param_t"],
                    param_m=params["param_m"],
                    param_w=params["param_w"],
                    param_a=params["param_a"],
                )

        try:
            xform = _build_xform()
            session = self.session

            # 1) Enregistrement dans la session
            existing_transform = None
            try:
                existing_transform = session.get_transform(transform_id)
            except Exception:
                existing_transform = None

            if existing_transform is not None:
                xform = existing_transform
            else:
                try:
                    session.add_transform(transform_id, xform)
                except TypeError:
                    try:
                        session.add_transform(xform)
                    except KeyError:
                        # Déjà défini: réutiliser la transformation existante.
                        xform = session.get_transform(transform_id)
                except KeyError:
                    # Déjà défini: réutiliser la transformation existante.
                    xform = session.get_transform(transform_id)

            # 2) Application directe au sample actif (correctif cache xform)
            sample = session.get_sample(sid)
            try:
                sample.apply_transform(xform)
            except TypeError:
                sample.apply_transform(transform=xform)

            # Conserver la transformation active côté engine
            self._transform_specs[transform_id] = TransformSpec(
                kind=transform_name.lower(),
                params=params,
            )

            self.log.info(
                "Transformation %s appliquée avec succès au sample %s.",
                transform_name,
                sid,
            )
            return transform_id
        except Exception as exc:
            self.log.error(
                "Echec apply_transformation(type=%s, sample=%s): %s",
                transform_type,
                sid,
                exc,
                exc_info=True,
            )
            raise

    def get_transform_ids(self) -> List[str]:
        return list(self._transform_specs.keys())

    def get_transform_spec(self, transform_id: str) -> Optional[TransformSpec]:
        return self._transform_specs.get(transform_id)

    def get_transformed_dataframe(
        self,
        sample_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        comp_matrix_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retourne le DataFrame transformé/compensé d'un sample."""
        sid = sample_id or self._require_active_sample()
        sample = self._backend.get_sample(sid)
        if transform_id is None:
            return self._normalize_dataframe_channels(sample.as_dataframe(source="raw"))

        try:
            return self._normalize_dataframe_channels(sample.as_dataframe(source="xform"))
        except Exception:
            # Fallback: si cache xform absent, appliquer explicitement la transform puis relire.
            if not self._is_workspace():
                try:
                    transform = self.session.get_transform(transform_id)
                    try:
                        sample.apply_transform(transform)
                    except TypeError:
                        sample.apply_transform(transform=transform)
                    return self._normalize_dataframe_channels(sample.as_dataframe(source="xform"))
                except Exception:
                    pass
            return self._normalize_dataframe_channels(sample.as_dataframe(source="raw"))

    def _normalize_dataframe_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise les colonnes FlowKit vers des noms Pnn simples.

        FlowKit peut retourner des colonnes tuple (Pnn, PnS) ; l'UI manipule
        des canaux Pnn (`FL1-A`, `CD45-A`, etc.), donc on aplati ici.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df

        normalized_cols: List[str] = []
        has_tuple_cols = False

        for col in df.columns:
            if isinstance(col, tuple):
                has_tuple_cols = True
                # FlowKit retourne typiquement (Pnn, PnS) ; l'UI manipule les Pnn.
                pnn = str(col[0]).strip() if len(col) > 0 else ""
                if pnn:
                    normalized_cols.append(pnn)
                else:
                    fallback = ""
                    for part in col:
                        part_s = str(part).strip()
                        if part_s:
                            fallback = part_s
                            break
                    normalized_cols.append(fallback or str(col))
            else:
                normalized_cols.append(str(col))

        if not has_tuple_cols:
            return df

        out = df.copy()
        out.columns = normalized_cols
        return out

    # ------------------------------------------------------------------
    # F. Gating — création (Session uniquement)
    # ------------------------------------------------------------------

    def add_polygon_gate(
        self,
        gate_name: str,
        gate_path: Tuple[str, ...],
        x_channel: str,
        y_channel: str,
        vertices: List[Tuple[float, float]],
        comp_ref: str = "uncompensated",
        transform_ref: Optional[str] = None,
        sample_id: Optional[str] = None,
    ) -> None:
        """Ajoute une gate polygonale. sample_id → gate custom par sample."""
        self._require_session_for_gate_edit()
        self._validate_gate_name(gate_name, gate_path)
        self._validate_channels(x_channel, y_channel)

        # Si la gate existe déjà au même path, on la remplace explicitement
        # pour éviter GateTreeError: Gate already exists.
        self._replace_existing_gate_if_present(gate_name, gate_path, sample_id=sample_id)

        dim_x = self._build_dimension(x_channel, comp_ref=comp_ref, transform_ref=transform_ref)
        dim_y = self._build_dimension(y_channel, comp_ref=comp_ref, transform_ref=transform_ref)
        gate = fk_gates.PolygonGate(gate_name, [dim_x, dim_y], vertices)
        self._as_session().add_gate(gate, gate_path, sample_id=sample_id)
        self._register_gate_dimensions(gate_name, x_channel, y_channel)
        _logger.info("PolygonGate : %s @ %s (sample=%s)", gate_name, gate_path, sample_id)
        self._emit_strategy_changed(gate_name, "added")

    def add_rectangle_gate(
        self,
        gate_name: str,
        gate_path: Tuple[str, ...],
        x_channel: str,
        y_channel: Optional[str],
        x_min: Optional[float],
        x_max: Optional[float],
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        comp_ref: str = "uncompensated",
        transform_ref: Optional[str] = None,
        sample_id: Optional[str] = None,
    ) -> None:
        """
        Ajoute une gate rectangulaire (1D si y_channel=None, 2D sinon).

        Convention 1D depuis l'UI : y_channel='', y_min=y_max=0.0
        → détecte automatiquement la gate 1D.
        """
        self._require_session_for_gate_edit()
        self._validate_gate_name(gate_name, gate_path)

        is_1d = not y_channel or (y_min == 0.0 and y_max == 0.0 and not y_channel)

        # Remplacement idempotent de la gate existante sur le même chemin.
        self._replace_existing_gate_if_present(gate_name, gate_path, sample_id=sample_id)

        if is_1d:
            dim_x = self._build_dimension(
                x_channel,
                comp_ref=comp_ref,
                transform_ref=transform_ref,
                range_min=x_min,
                range_max=x_max,
            )
            gate = fk_gates.RectangleGate(gate_name, [dim_x])
        else:
            self._validate_channels(x_channel, y_channel)
            dim_x = self._build_dimension(
                x_channel,
                comp_ref=comp_ref,
                transform_ref=transform_ref,
                range_min=x_min,
                range_max=x_max,
            )
            dim_y = self._build_dimension(
                y_channel,
                comp_ref=comp_ref,
                transform_ref=transform_ref,
                range_min=y_min,
                range_max=y_max,
            )
            gate = fk_gates.RectangleGate(gate_name, [dim_x, dim_y])

        self._as_session().add_gate(gate, gate_path, sample_id=sample_id)
        if is_1d:
            self._register_gate_dimensions(gate_name, x_channel, None)
        else:
            self._register_gate_dimensions(gate_name, x_channel, y_channel)
        _logger.info("RectangleGate %s : %s @ %s", "1D" if is_1d else "2D", gate_name, gate_path)
        self._emit_strategy_changed(gate_name, "added")

    def add_ellipsoid_gate(
        self,
        gate_name: str,
        gate_path: Tuple[str, ...],
        x_channel: str,
        y_channel: str,
        coordinates: List[float],
        covariance_matrix: List[List[float]],
        distance_square: float,
        comp_ref: str = "uncompensated",
        transform_ref: Optional[str] = None,
        sample_id: Optional[str] = None,
    ) -> None:
        self._require_session_for_gate_edit()
        self._validate_gate_name(gate_name, gate_path)
        self._validate_channels(x_channel, y_channel)

        self._replace_existing_gate_if_present(gate_name, gate_path, sample_id=sample_id)

        dim_x = self._build_dimension(x_channel, comp_ref=comp_ref, transform_ref=transform_ref)
        dim_y = self._build_dimension(y_channel, comp_ref=comp_ref, transform_ref=transform_ref)
        gate = fk_gates.EllipsoidGate(
            gate_name,
            [dim_x, dim_y],
            coordinates=coordinates,
            covariance_matrix=np.array(covariance_matrix),
            distance_square=distance_square,
        )
        self._as_session().add_gate(gate, gate_path, sample_id=sample_id)
        _logger.info("EllipsoidGate : %s @ %s", gate_name, gate_path)

    def add_quadrant_gate(
        self,
        gate_name: str,
        gate_path: Tuple[str, ...],
        x_channel: str,
        y_channel: str,
        x_threshold: float,
        y_threshold: float,
        comp_ref: str = "uncompensated",
        transform_ref: Optional[str] = None,
        sample_id: Optional[str] = None,
    ) -> None:
        """Ajoute une QuadrantGate (4 quadrants nommés Q1_PosPos … Q4_PosNeg)."""
        self._require_session_for_gate_edit()
        self._validate_gate_name(gate_name, gate_path)
        self._validate_channels(x_channel, y_channel)

        self._replace_existing_gate_if_present(gate_name, gate_path, sample_id=sample_id)

        div_x_id = f"{gate_name}_div_x"
        div_y_id = f"{gate_name}_div_y"
        x_comp_ref, x_transform_ref = self._resolve_dimension_refs(
            x_channel, comp_ref, transform_ref
        )
        y_comp_ref, y_transform_ref = self._resolve_dimension_refs(
            y_channel, comp_ref, transform_ref
        )
        div_x = fk.QuadrantDivider(div_x_id, x_channel, x_comp_ref, [x_threshold], x_transform_ref)
        div_y = fk.QuadrantDivider(div_y_id, y_channel, y_comp_ref, [y_threshold], y_transform_ref)

        q_pp = fk_gates.Quadrant(
            f"{gate_name}_Q1_PosPos",
            [div_x_id, div_y_id],
            [(x_threshold, None), (y_threshold, None)],
        )
        q_np = fk_gates.Quadrant(
            f"{gate_name}_Q2_NegPos",
            [div_x_id, div_y_id],
            [(None, x_threshold), (y_threshold, None)],
        )
        q_nn = fk_gates.Quadrant(
            f"{gate_name}_Q3_NegNeg",
            [div_x_id, div_y_id],
            [(None, x_threshold), (None, y_threshold)],
        )
        q_pn = fk_gates.Quadrant(
            f"{gate_name}_Q4_PosNeg",
            [div_x_id, div_y_id],
            [(x_threshold, None), (None, y_threshold)],
        )

        gate = fk_gates.QuadrantGate(gate_name, [div_x, div_y], [q_pp, q_np, q_nn, q_pn])
        self._as_session().add_gate(gate, gate_path, sample_id=sample_id)
        self._register_gate_dimensions(gate_name, x_channel, y_channel)
        _logger.info("QuadrantGate : %s @ %s", gate_name, gate_path)
        self._emit_strategy_changed(gate_name, "added")

    def add_boolean_gate(
        self,
        gate_name: str,
        gate_path: Tuple[str, ...],
        bool_type: str,
        gate_refs: List[Dict[str, Any]],
        sample_id: Optional[str] = None,
    ) -> None:
        """Ajoute une BooleanGate. bool_type : 'and' | 'or' | 'not'."""
        self._require_session_for_gate_edit()
        self._validate_gate_name(gate_name, gate_path)

        for ref in gate_refs:
            ref_name = ref["gate_name"]
            ref_path = ref.get("gate_path", ("root",))
            try:
                self._backend.get_gate(ref_name, gate_path=ref_path)
            except Exception:
                raise BooleanGateRefMissingError(
                    f"Gate référencée introuvable : {ref_name} @ {ref_path}"
                )

        gate = fk_gates.BooleanGate(gate_name, bool_type=bool_type, gate_refs=gate_refs)
        self._as_session().add_gate(gate, gate_path, sample_id=sample_id)
        _logger.info("BooleanGate : %s (%s) @ %s", gate_name, bool_type, gate_path)

    # ------------------------------------------------------------------
    # G. Gating — lecture (Session + Workspace)
    # ------------------------------------------------------------------

    def remove_gate(
        self,
        gate_name: str,
        gate_path: Optional[Tuple[str, ...]] = None,
        keep_children: bool = False,
        sample_id: Optional[str] = None,
    ) -> None:
        self._require_session_for_gate_edit()
        self._require_gate_exists(gate_name, gate_path)
        self._as_session().remove_gate(
            gate_name,
            gate_path=gate_path,
            sample_id=sample_id,
            keep_children=keep_children,
        )
        self._unregister_gate_dimensions(gate_name)
        self._invalidate_membership_cache(sample_id=sample_id)
        _logger.info("Gate supprimée : %s (sample=%s)", gate_name, sample_id)
        self._emit_strategy_changed(gate_name, "removed")

    def rename_gate(
        self,
        gate_name: str,
        new_name: str,
        gate_path: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self._require_session_for_gate_edit()
        self._require_gate_exists(gate_name, gate_path)
        self._as_session().rename_gate(gate_name, new_name, gate_path=gate_path)
        _logger.info("Gate renommée : %s → %s", gate_name, new_name)

    def get_gate(
        self,
        gate_name: str,
        gate_path: Optional[Tuple[str, ...]] = None,
        sample_id: Optional[str] = None,
    ) -> Any:
        """Retourne l'objet FlowKit Gate (template ou custom si sample_id)."""
        return self._backend.get_gate(gate_name, gate_path=gate_path)

    def get_gate_ids(self) -> List[str]:
        """Retourne les gate_names (normalise les tuples FlowKit)."""
        raw = self._backend.get_gate_ids()
        return [str(item[0]) if isinstance(item, tuple) else str(item) for item in raw]

    def get_gate_id_path_pairs(self) -> List[Tuple[str, Tuple[str, ...]]]:
        """Retourne tous les (gate_name, gate_path) de la stratégie."""
        raw = self._backend.get_gate_ids()
        result: List[Tuple[str, Tuple[str, ...]]] = []
        for item in raw:
            if isinstance(item, tuple):
                result.append((str(item[0]), item[1]))
            else:
                result.append((str(item), ("root",)))
        return result

    def get_sample_gates(self, sample_id: str) -> List[Any]:
        """Retourne les gates custom spécifiques à un sample (Session uniquement)."""
        if self._is_workspace():
            _logger.debug("get_sample_gates : Workspace — retourne liste vide")
            return []
        return list(self._as_session().get_sample_gates(sample_id))

    def find_gate_paths(self, gate_name: str) -> List[Tuple[str, ...]]:
        return list(self._backend.find_matching_gate_paths(gate_name))

    def get_children(
        self, gate_name: str, gate_path: Optional[Tuple[str, ...]] = None
    ) -> List[str]:
        try:
            return list(self._backend.get_child_gate_ids(gate_name, gate_path=gate_path))
        except Exception:
            return []

    def get_gate_hierarchy_dict(self) -> Dict[str, Any]:
        """
        Retourne la hiérarchie complète sous forme de dict imbriqué FlowKit.

        Structure : {'name': 'root', 'children': [{'name': 'G1', 'gate': ..., ...}]}
        """
        return self._backend.get_gate_hierarchy(output="dict")

    # ------------------------------------------------------------------
    # H. Analyse
    # ------------------------------------------------------------------

    def analyze(
        self,
        sample_id: Optional[str] = None,
        group_name: Optional[str] = None,
        cache_events: bool = False,
        use_mp: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Lance l'analyse FlowKit.

        Args:
            sample_id:    Analyse un sample spécifique. None → tous.
            group_name:   Analyse un groupe (Workspace uniquement). Ignoré sur Session.
            cache_events: Garder les événements en cache (utile pour queries répétées).
            use_mp:       Multiprocessing. Désactivé par défaut (stabilité PyQt5).
            verbose:      Log détaillé FlowKit.
        """
        if group_name is not None and self._is_workspace():
            self._as_workspace().analyze_samples(
                group_name=group_name,
                sample_id=sample_id,
                cache_events=cache_events,
                use_mp=use_mp,
                verbose=verbose,
            )
            _logger.info("Analyse groupe '%s' terminée", group_name)
        elif self._is_workspace():
            self._as_workspace().analyze_samples(
                sample_id=sample_id,
                cache_events=cache_events,
                use_mp=use_mp,
                verbose=verbose,
            )
            _logger.info("Analyse Workspace terminée (sample=%s)", sample_id)
        else:
            self._as_session().analyze_samples(
                sample_id=sample_id,
                cache_events=cache_events,
                use_mp=use_mp,
                verbose=verbose,
            )
            _logger.info("Analyse Session terminée (sample=%s)", sample_id)

    def analyze_group(
        self,
        group_name: str,
        cache_events: bool = False,
        use_mp: bool = False,
    ) -> None:
        """Raccourci : analyse tout un groupe (Workspace uniquement)."""
        self.analyze(group_name=group_name, cache_events=cache_events, use_mp=use_mp)

    def get_gate_dataframe(
        self,
        gate_name: str,
        gate_path: Optional[Tuple[str, ...]] = None,
        sample_id: Optional[str] = None,
        matrix_id: Optional[str] = None,
        transform_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retourne le DataFrame des événements positifs pour une gate."""
        sid = sample_id or self._require_active_sample()
        matrix = (
            self._backend.get_comp_matrix(matrix_id)
            if matrix_id and not self._is_workspace()
            else None
        )
        transform = (
            self._backend.get_transform(transform_id)
            if transform_id and not self._is_workspace()
            else None
        )
        df = self._backend.get_gate_events(
            sid,
            gate_name=gate_name,
            gate_path=gate_path,
            matrix=matrix,
            transform=transform,
        )
        return self._normalize_dataframe_channels(df)

    def get_gate_membership(
        self,
        gate_name: str,
        gate_path: Optional[Tuple[str, ...]] = None,
        sample_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Retourne le masque booléen d'appartenance à une gate.

        Pour les QuadrantGates : FlowKit ne supporte pas get_gate_membership
        sur la QuadrantGate parente. On calcule le masque union des 4 sous-quadrants
        (tout événement appartenant à au moins un quadrant = dans la QuadrantGate).
        """
        sid = sample_id or self._require_active_sample()
        try:
            arr = self._backend.get_gate_membership(sid, gate_name, gate_path=gate_path)
            return np.asarray(arr, dtype=bool).flatten()
        except Exception as exc:
            # Fallback QuadrantGate : calculer depuis x_threshold / y_threshold
            try:
                paths = self.find_gate_paths(gate_name)
                gp = paths[0] if paths else gate_path
                gate_obj = self._backend.get_gate(gate_name, gate_path=gp)
                if gate_obj is not None and "Quadrant" in type(gate_obj).__name__:
                    return self._get_quadrant_gate_membership(gate_obj, sid)
            except Exception as inner:
                _logger.debug("QuadrantGate fallback membership échoué: %s", inner)
            raise exc

    def _get_quadrant_gate_membership(
        self,
        gate_obj: Any,
        sample_id: str,
    ) -> np.ndarray:
        """
        Calcule l'union des 4 quadrants d'une QuadrantGate.
        Retourne un masque booléen : True si l'événement est dans au moins un quadrant.
        """
        quad_ids = list(getattr(gate_obj, "quadrants", {}).keys())
        if not quad_ids:
            raise PrismaEngineError("QuadrantGate sans quadrants définis")

        full_df_size = None
        combined = None
        for qid in quad_ids:
            try:
                arr = self._backend.get_gate_membership(sample_id, qid)
                arr_bool = np.asarray(arr, dtype=bool).flatten()
                if combined is None:
                    combined = arr_bool.copy()
                    full_df_size = len(arr_bool)
                else:
                    combined |= arr_bool
            except Exception as exc:
                _logger.debug("Quadrant '%s' membership ignoré : %s", qid, exc)

        if combined is None:
            raise PrismaEngineError("Impossible d'obtenir le membership d'aucun sous-quadrant")
        return combined

    def get_quadrant_memberships(
        self,
        gate_name: str,
        gate_path: Optional[Tuple[str, ...]] = None,
        sample_id: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Retourne les 4 masques booléens distincts d'une QuadrantGate.

        Returns:
            {quadrant_id: np.ndarray[bool]} — un masque par quadrant (Q1/Q2/Q3/Q4).
        """
        sid = sample_id or self._require_active_sample()
        paths = self.find_gate_paths(gate_name)
        gp = paths[0] if paths else gate_path
        gate_obj = self._backend.get_gate(gate_name, gate_path=gp)
        if gate_obj is None or "Quadrant" not in type(gate_obj).__name__:
            raise PrismaEngineError(f"'{gate_name}' n'est pas une QuadrantGate")

        result: Dict[str, np.ndarray] = {}
        for qid in getattr(gate_obj, "quadrants", {}).keys():
            try:
                arr = self._backend.get_gate_membership(sid, qid)
                result[qid] = np.asarray(arr, dtype=bool).flatten()
            except Exception as exc:
                _logger.debug("Quadrant '%s' membership ignoré : %s", qid, exc)
        return result

    def get_analysis_report(self, group_name: Optional[str] = None) -> pd.DataFrame:
        """
        Retourne le rapport analytique FlowKit.

        Args:
            group_name: Filtre par groupe (Workspace uniquement).
        """
        if group_name is not None and self._is_workspace():
            return self._as_workspace().get_analysis_report(group_name=group_name)
        return self._backend.get_analysis_report()

    def get_raw_dataframe(
        self,
        sample_id: Optional[str] = None,
        gate_name: Optional[str] = None,
        gate_path: Optional[Tuple[str, ...]] = None,
        transform_id: Optional[str] = None,
        comp_matrix_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """DataFrame brut ou filtré par gate, transformé si demandé."""
        if gate_name is not None:
            return self.get_gate_dataframe(
                gate_name,
                gate_path=gate_path,
                sample_id=sample_id,
                matrix_id=comp_matrix_id,
                transform_id=transform_id,
            )
        return self.get_transformed_dataframe(
            sample_id=sample_id,
            transform_id=transform_id,
            comp_matrix_id=comp_matrix_id,
        )

    def get_population_df(
        self,
        gate_node: Tuple[str, ...],
        sample_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        comp_matrix_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Pipeline garanti comp→xform→filter (Kaluza-like).

        Convention gate_node:
            ('root',)               -> population globale (tous événements)
            (gate_name, *gate_path) -> événements positifs de la gate

        La compensation et la transformation sont appliquées manuellement sur
        le DataFrame brut du Sample, indépendamment des comp_ref/transformation_ref
        stockés dans les Dimensions des gates.  C'est la seule approche garantissant
        un rendu visuel correct quel que soit l'état de la GatingStrategy.
        """
        sid = sample_id or self._require_active_sample()
        full_df = self._build_comp_xform_df(sid, transform_id, comp_matrix_id)

        if not gate_node or str(gate_node[0]).lower() == "root":
            return full_df

        gate_name = str(gate_node[0])
        gate_path = tuple(gate_node[1:]) if len(gate_node) > 1 else None

        try:
            mask = self._backend.get_gate_membership(sid, gate_name, gate_path=gate_path)
            mask_flat = np.asarray(mask, dtype=bool).flatten()
            if mask_flat.shape[0] == full_df.shape[0]:
                filtered = full_df[mask_flat].reset_index(drop=True)
                print(f"[ENGINE] Gate '{gate_name}': {mask_flat.sum()}/{len(mask_flat)} events")
                return filtered
            _logger.warning(
                "Shape mismatch masque %d vs df %d pour gate '%s' — population complète",
                mask_flat.shape[0],
                full_df.shape[0],
                gate_name,
            )
        except Exception as exc:
            _logger.warning(
                "get_gate_membership échoué pour '%s': %s — population complète", gate_name, exc
            )

        return full_df

    def get_colored_population(
        self,
        parent_gate_node: Tuple[str, ...],
        x_pnn: str,
        y_pnn: Optional[str] = None,
        sample_id: Optional[str] = None,
        transform_id: Optional[str] = None,
        comp_matrix_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retourne le DataFrame de la population parente avec colonne 'Population_Color'.

        Chaque cellule reçoit la couleur HEX de sa population enfant la plus proche.
        Les cellules n'appartenant à aucun enfant reçoivent '#506070' (gris-bleu).

        Args:
            parent_gate_node: Tuple de chemin vers la gate parente (ex: ('Leuco',)).
            x_pnn:            Canal X (Pnn).
            y_pnn:            Canal Y (Pnn) — None OK.
            sample_id:        Sample actif si None.
            transform_id:     ID de transformation.
            comp_matrix_id:   ID de matrice de compensation.

        Returns:
            DataFrame avec colonne 'Population_Color' (str HEX).
        """
        _CHILD_COLORS = [
            "#5BAAFF",
            "#39FF8A",
            "#FF9B3D",
            "#FF3D6E",
            "#FFE032",
            "#7B52FF",
            "#7EC8E3",
            "#FF6BFF",
        ]
        _DEFAULT_COLOR = "#506070"

        sid = sample_id or self._require_active_sample()
        parent_df = self.get_population_df(
            parent_gate_node,
            sample_id=sid,
            transform_id=transform_id,
            comp_matrix_id=comp_matrix_id,
        )

        if parent_df.empty:
            return parent_df

        # Récupérer les enfants directs
        parent_name = str(parent_gate_node[0]) if parent_gate_node else None
        parent_path: Optional[Tuple[str, ...]] = (
            tuple(parent_gate_node[1:]) if len(parent_gate_node) > 1 else None
        )
        children: List[str] = []
        if parent_name and parent_name.lower() != "root":
            try:
                children = list(
                    self._backend.get_child_gate_ids(parent_name, gate_path=parent_path)
                )
            except Exception as exc:
                _logger.debug("get_colored_population: get_child_gate_ids échoué: %s", exc)

        # Construire le masque global pour retrouver les lignes du parent dans le df complet
        full_df = self._build_comp_xform_df(sid, transform_id, comp_matrix_id)
        if full_df.empty or full_df.shape[0] == 0:
            parent_df["Population_Color"] = _DEFAULT_COLOR
            return parent_df

        # Obtenir le masque du parent dans full_df
        if parent_name and parent_name.lower() != "root":
            try:
                parent_mask_full = np.asarray(
                    self._backend.get_gate_membership(sid, parent_name, gate_path=parent_path),
                    dtype=bool,
                ).flatten()
                if parent_mask_full.shape[0] != full_df.shape[0]:
                    parent_mask_full = np.ones(full_df.shape[0], dtype=bool)
            except Exception:
                parent_mask_full = np.ones(full_df.shape[0], dtype=bool)
        else:
            parent_mask_full = np.ones(full_df.shape[0], dtype=bool)

        # Tableau de couleurs indexé sur les lignes du parent_df
        n_parent = parent_mask_full.sum()
        colors = np.full(n_parent, _DEFAULT_COLOR, dtype=object)

        if children:
            parent_indices = np.where(parent_mask_full)[0]

            for i, child_name in enumerate(children):
                color = _CHILD_COLORS[i % len(_CHILD_COLORS)]
                try:
                    child_paths = self.find_gate_paths(child_name)
                    child_path = child_paths[0] if child_paths else None
                    child_mask_full = np.asarray(
                        self._backend.get_gate_membership(sid, child_name, gate_path=child_path),
                        dtype=bool,
                    ).flatten()
                    if child_mask_full.shape[0] != full_df.shape[0]:
                        continue
                    # Indices dans parent_df (position relative au sous-ensemble parent)
                    child_in_parent = child_mask_full[parent_indices]
                    colors[child_in_parent] = color
                except Exception as exc:
                    _logger.debug("get_colored_population: child '%s' ignoré: %s", child_name, exc)

        parent_df = parent_df.copy()
        parent_df["Population_Color"] = colors
        return parent_df

    def _build_comp_xform_df(
        self,
        sid: str,
        transform_id: Optional[str],
        comp_matrix_id: Optional[str],
    ) -> pd.DataFrame:
        """
        Pipeline natif FlowKit : comp → xform sur le Sample, puis as_dataframe.

        Utilise exclusivement les méthodes natives FlowKit :
          - sample.apply_compensation(matrix_obj) → peuple _comp_events
          - sample.apply_transform(xform_obj)     → peuple _transformed_events
          - sample.as_dataframe(source='xform', col_multi_index=False)

        Le Sample est modifié in-place (FlowKit le fait toujours ainsi).
        Priorité de la source retournée : xform → comp → raw.
        """
        self._require_flowkit()
        sample = self._backend.get_sample(sid)

        # --- Étape 1 : Compensation native FlowKit ---
        matrix_id = comp_matrix_id or (self._comp_matrix_ids[0] if self._comp_matrix_ids else None)
        comp_ok = False
        if matrix_id and not self._is_workspace():
            try:
                session = self._as_session()
                matrix_obj = session.get_comp_matrix(matrix_id)
                sample.apply_compensation(matrix_obj)
                comp_ok = True
                _logger.debug("[PIPELINE] Compensation OK: matrix_id=%s", matrix_id)
                print(f"[ENGINE] Compensation appliquée: {matrix_id}")
            except Exception as exc:
                _logger.warning("[PIPELINE] apply_compensation échoué (%s): %s", matrix_id, exc)
                print(f"[ENGINE] WARN compensation échouée ({matrix_id}): {exc}")
        else:
            print(
                f"[ENGINE] Pas de compensation (matrix_id={matrix_id}, is_workspace={self._is_workspace()})"
            )

        # --- Étape 2 : Transformation native FlowKit ---
        xform_id = transform_id or (next(iter(self._transform_specs), None))
        xform_ok = False
        if xform_id and not self._is_workspace():
            try:
                session = self._as_session()
                xform_obj = session.get_transform(xform_id)
                sample.apply_transform(xform_obj)
                xform_ok = True
                _logger.debug("[PIPELINE] Transform OK: xform_id=%s", xform_id)
                print(f"[ENGINE] Transformation appliquée: {xform_id}")
            except Exception as exc:
                _logger.warning("[PIPELINE] apply_transform échoué (%s): %s", xform_id, exc)
                print(f"[ENGINE] WARN transform échouée ({xform_id}): {exc}")
        else:
            print(
                f"[ENGINE] Pas de transformation (xform_id={xform_id}, is_workspace={self._is_workspace()})"
            )

        # --- Étape 3 : Lire dans le meilleur espace disponible ---
        for source in ("xform", "comp", "raw"):
            try:
                df = sample.as_dataframe(source=source, col_multi_index=False)
                if df is not None and not df.empty:
                    df.columns = [str(c).strip() for c in df.columns]
                    # Log diagnostique : min/max sur 2 premiers canaux fluoro
                    fluoro_cols = [
                        c
                        for c in df.columns
                        if all(t not in str(c).upper() for t in ("FSC", "SSC", "TIME"))
                    ]
                    if fluoro_cols:
                        col0 = fluoro_cols[0]
                        v = df[col0].dropna()
                        print(
                            f"[ENGINE] DataFrame source='{source}' OK — {len(df)} events, "
                            f"{len(df.columns)} canaux | {col0}: min={v.min():.4f} max={v.max():.4f}"
                        )
                    else:
                        print(f"[ENGINE] DataFrame source='{source}' OK — {len(df)} events")
                    return df
            except AttributeError:
                print(f"[ENGINE] source='{source}' non peuplé (AttributeError) → essai suivant")
                continue
            except Exception as exc:
                _logger.debug("as_dataframe(source='%s') échoué: %s", source, exc)
                print(f"[ENGINE] source='{source}' échoué: {exc}")
                continue

        _logger.error("_build_comp_xform_df: toutes sources épuisées pour %s", sid)
        print(f"[ENGINE] ERREUR: toutes sources épuisées pour sample {sid}")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # I. Hiérarchie pour le QTreeView
    # ------------------------------------------------------------------

    def build_hierarchy(self, sample_id: Optional[str] = None) -> List[GateHierarchyNode]:
        """
        Construit la liste de GateHierarchyNode racines depuis get_gate_hierarchy(dict).

        Priorité : get_gate_hierarchy(output='dict') — structure native FlowKit.
        QuadrantGate : enfants injectés depuis gate_obj.quadrants si absents.
        Counts : remplis depuis standardize_report() si analyze() a été appelé.
        """
        try:
            hierarchy_dict = self.get_gate_hierarchy_dict()
        except Exception as exc:
            _logger.warning("get_gate_hierarchy dict échoué : %s — fallback pairs", exc)
            return self._build_hierarchy_from_pairs(sample_id)

        if not hierarchy_dict or "children" not in hierarchy_dict:
            return []

        # Construire le dict de counts depuis le rapport
        count_map: Dict[str, int] = {}
        sid = sample_id or self._active_sample_id
        if sid:
            try:
                from src.prisma.exports.gating_exporter import GatingExporter

                report_raw = self.get_analysis_report()
                if report_raw is not None and not report_raw.empty:
                    report = GatingExporter.standardize_report(report_raw)
                    for _, row in report.iterrows():
                        if row.get("sample_id") is not None and str(row["sample_id"]) != str(sid):
                            continue
                        gname_r = str(row.get("gate_name") or "")
                        count_val = row.get("count")
                        if gname_r and count_val is not None:
                            count_map[gname_r] = int(count_val)
            except Exception:
                pass

        roots: List[GateHierarchyNode] = []
        for child_dict in hierarchy_dict.get("children", []):
            node = self._dict_to_node(child_dict, ("root",), count_map)
            roots.append(node)

        self._fill_pcts(roots, parent_count=None, grandparent_count=None)
        return roots

    def _dict_to_node(
        self,
        d: Dict[str, Any],
        parent_path: Tuple[str, ...],
        count_map: Dict[str, int],
    ) -> GateHierarchyNode:
        """Convertit un dict FlowKit de hiérarchie en GateHierarchyNode."""
        gate_name = str(d.get("name", ""))
        gate_obj = d.get("gate")
        gate_type = d.get("gate_type", "unknown").lower().replace("gate", "")
        gate_path = parent_path

        node = GateHierarchyNode(
            gate_name=gate_name,
            gate_path=gate_path,
            gate_type=gate_type,
            count=count_map.get(gate_name, 0),
        )

        current_path = parent_path + (gate_name,)

        # Enfants récursifs depuis le dict
        for child_dict in d.get("children", []):
            child_node = self._dict_to_node(child_dict, current_path, count_map)
            node.children.append(child_node)

        # CORRECTIF C3 : QuadrantGate — injecter Q1/Q2/Q3/Q4 si absents
        if gate_obj is not None and isinstance(gate_obj, fk_gates.QuadrantGate):
            existing_child_names = {c.gate_name for c in node.children}
            for quad_id in gate_obj.quadrants:
                if quad_id not in existing_child_names:
                    node.children.append(
                        GateHierarchyNode(
                            gate_name=quad_id,
                            gate_path=current_path,
                            gate_type="quadrant",
                            count=count_map.get(quad_id, 0),
                        )
                    )

        return node

    def _build_hierarchy_from_pairs(
        self, sample_id: Optional[str] = None
    ) -> List[GateHierarchyNode]:
        """Fallback : construit la hiérarchie depuis get_gate_id_path_pairs()."""
        pairs = self.get_gate_id_path_pairs()
        if not pairs:
            return []

        count_map: Dict[str, int] = {}
        sid = sample_id or self._active_sample_id
        if sid:
            try:
                from src.prisma.exports.gating_exporter import GatingExporter

                report_raw = self.get_analysis_report()
                if report_raw is not None and not report_raw.empty:
                    report = GatingExporter.standardize_report(report_raw)
                    for _, row in report.iterrows():
                        if row.get("sample_id") is not None and str(row["sample_id"]) != str(sid):
                            continue
                        gname_r = str(row.get("gate_name") or "")
                        count_val = row.get("count")
                        if gname_r and count_val is not None:
                            count_map[gname_r] = int(count_val)
            except Exception:
                pass

        nodes: Dict[str, GateHierarchyNode] = {}
        for gname, gpath in pairs:
            try:
                gate_obj = self._backend.get_gate(gname, gate_path=gpath)
                gate_type = type(gate_obj).__name__.lower().replace("gate", "")
            except Exception:
                gate_obj = None
                gate_type = "unknown"

            node = GateHierarchyNode(
                gate_name=gname,
                gate_path=gpath,
                gate_type=gate_type,
                count=count_map.get(gname, 0),
            )
            node_key = f"{gname}|{'|'.join(gpath)}"
            nodes[node_key] = node

            if gate_obj is not None and isinstance(gate_obj, fk_gates.QuadrantGate):
                child_path = gpath + (gname,)
                existing = {n for n, p in pairs if p == child_path}
                for quad_id in gate_obj.quadrants:
                    if quad_id not in existing:
                        qkey = f"{quad_id}|{'|'.join(child_path)}"
                        if qkey not in nodes:
                            nodes[qkey] = GateHierarchyNode(
                                gate_name=quad_id,
                                gate_path=child_path,
                                gate_type="quadrant",
                                count=count_map.get(quad_id, 0),
                            )

        roots: List[GateHierarchyNode] = []
        node_by_name: Dict[str, GateHierarchyNode] = {}
        for node in nodes.values():
            node_by_name[node.gate_name] = node

        for node in nodes.values():
            if len(node.gate_path) <= 1:
                roots.append(node)
            else:
                parent_node = node_by_name.get(node.gate_path[-1])
                if parent_node is not None:
                    parent_node.children.append(node)
                else:
                    roots.append(node)

        self._fill_pcts(roots, parent_count=None, grandparent_count=None)
        return roots

    def _fill_pcts(
        self,
        nodes: List[GateHierarchyNode],
        parent_count: Optional[int],
        grandparent_count: Optional[int],
    ) -> None:
        for node in nodes:
            if parent_count and parent_count > 0:
                node.pct_parent = 100.0 * node.count / parent_count
            if grandparent_count and grandparent_count > 0:
                node.pct_grandparent = 100.0 * node.count / grandparent_count
            self._fill_pcts(node.children, node.count, parent_count)

    # ------------------------------------------------------------------
    # J. Converters UI → FlowKit
    # ------------------------------------------------------------------

    def create_polygon_gate_from_vertices(
        self,
        gate_name: str,
        gate_path: Tuple[str, ...],
        x_channel: str,
        y_channel: str,
        vertices: List[Tuple[float, float]],
        comp_ref: str = "uncompensated",
        transform_ref: Optional[str] = None,
        sample_id: Optional[str] = None,
    ) -> None:
        if len(vertices) < 3:
            raise PrismaEngineError("Polygone : minimum 3 sommets requis.")
        self._validate_parent_path(gate_path)
        self.add_polygon_gate(
            gate_name,
            gate_path,
            x_channel,
            y_channel,
            vertices,
            comp_ref=comp_ref,
            transform_ref=transform_ref,
            sample_id=sample_id,
        )

    def create_rectangle_gate_from_bounds(
        self,
        gate_name: str,
        gate_path: Tuple[str, ...],
        x_channel: str,
        y_channel: Optional[str],
        x_min: Optional[float],
        x_max: Optional[float],
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        comp_ref: str = "uncompensated",
        transform_ref: Optional[str] = None,
        sample_id: Optional[str] = None,
    ) -> None:
        self._validate_parent_path(gate_path)
        self.add_rectangle_gate(
            gate_name,
            gate_path,
            x_channel,
            y_channel,
            x_min,
            x_max,
            y_min,
            y_max,
            comp_ref=comp_ref,
            transform_ref=transform_ref,
            sample_id=sample_id,
        )

    def create_quadrant_gate_from_thresholds(
        self,
        gate_name: str,
        gate_path: Tuple[str, ...],
        x_channel: str,
        y_channel: str,
        x_threshold: float,
        y_threshold: float,
        comp_ref: str = "uncompensated",
        transform_ref: Optional[str] = None,
        sample_id: Optional[str] = None,
    ) -> None:
        self._validate_parent_path(gate_path)
        self.add_quadrant_gate(
            gate_name,
            gate_path,
            x_channel,
            y_channel,
            x_threshold,
            y_threshold,
            comp_ref=comp_ref,
            transform_ref=transform_ref,
            sample_id=sample_id,
        )

    def create_boolean_gate(
        self,
        gate_name: str,
        parent_path: Tuple[str, ...],
        boolean_type: str,
        ref_gates: List[str],
    ) -> None:
        """
        Crée une BooleanGate FlowKit combinant des gates existantes.

        Args:
            gate_name:    Identifiant unique de la gate.
            parent_path:  Chemin de la gate parente (tuple, ex: ("root",)).
            boolean_type: 'and' | 'or' | 'not'  (FlowKit BooleanGate literals).
            ref_gates:    Liste de gate_name référencés (doivent exister dans la stratégie).

        Raises:
            PrismaEngineError: FlowKit absent, type invalide ou gate référencée manquante.
            BooleanGateRefMissingError: Une ref_gate est introuvable.
        """
        if not FLOWKIT_AVAILABLE:
            raise PrismaEngineError("FlowKit non disponible — impossible de créer une BooleanGate.")

        bool_type_clean = boolean_type.strip().lower()
        if bool_type_clean not in ("and", "or", "not"):
            raise PrismaEngineError(
                f"boolean_type invalide '{boolean_type}'. Valeurs acceptées : 'and', 'or', 'not'."
            )

        existing_ids = set(self.get_gate_ids())
        missing = [g for g in ref_gates if g not in existing_ids]
        if missing:
            raise BooleanGateRefMissingError(
                f"Gates référencées introuvables : {missing}. "
                f"Gates disponibles : {sorted(existing_ids)}"
            )

        try:
            # BooleanGate FlowKit : liste de GateReference (gate_name, gate_path)
            gate_refs = []
            for gid in ref_gates:
                paths = self.find_gate_paths(gid)
                gate_path = paths[0] if paths else None
                gate_refs.append(fk_gates.GateReference(gid, gate_path=gate_path))

            bool_gate = fk_gates.BooleanGate(
                gate_name=gate_name,
                bool_type=bool_type_clean,
                ref_gates=gate_refs,
            )
            self._backend.add_gate(bool_gate, gate_path=parent_path)
            _logger.info(
                "BooleanGate '%s' (%s) créée avec refs=%s", gate_name, bool_type_clean, ref_gates
            )
        except (BooleanGateRefMissingError, PrismaEngineError):
            raise
        except Exception as exc:
            raise PrismaEngineError(f"Création BooleanGate '{gate_name}' échouée : {exc}") from exc

    # ------------------------------------------------------------------
    # K. Extensions PRISMA
    # ------------------------------------------------------------------

    def time_gate(
        self,
        sample_id: Optional[str] = None,
        time_channel: str = "Time",
        method: str = "flowai",
        sigma: float = 3.0,
    ) -> np.ndarray:
        """
        Extension PRISMA — Time gating sur canal Time.

        Non natif FlowKit : détecte les événements aberrants dans le temps
        (fluctuations de débit, agrégats, perturbations laser).

        Méthodes disponibles :
          'sigma'  : rejette les événements dont le taux d'acquisition local
                     dépasse +/- sigma écarts-types de la médiane.
          'flowai' : utilise flowAI si disponible (pip install flowai).

        Args:
            sample_id:    Sample à analyser (None → actif).
            time_channel: Nom du canal temps (par défaut 'Time').
            method:       'sigma' | 'flowai'.
            sigma:        Seuil en écarts-types (méthode 'sigma').

        Returns:
            Masque booléen True = événement valide.
        """
        sid = sample_id or self._require_active_sample()
        sample = self._backend.get_sample(sid)
        df = sample.as_dataframe(source="raw")
        cols = {(c[0] if isinstance(c, tuple) else str(c)): c for c in df.columns}

        if time_channel not in cols:
            raise PrismaEngineError(
                f"Canal temps '{time_channel}' absent. Canaux : {list(cols.keys())}"
            )

        time_values = df[cols[time_channel]].to_numpy(dtype=np.float64)

        if method == "flowai":
            try:
                from flowai import quality_control

                mask = quality_control.time_flow_check(time_values)
                _logger.info("Time gate flowAI : %d/%d événements valides", mask.sum(), len(mask))
                return mask
            except ImportError:
                _logger.warning("flowAI non disponible — fallback méthode sigma")

        # Méthode sigma : taux d'acquisition local (événements/bin)
        n_bins = max(100, len(time_values) // 500)
        counts, edges = np.histogram(time_values, bins=n_bins)
        counts_f = counts.astype(np.float64)
        median_rate = float(np.median(counts_f[counts_f > 0]))
        std_rate = float(np.std(counts_f[counts_f > 0]))
        lo = median_rate - sigma * std_rate
        hi = median_rate + sigma * std_rate

        # Trouver les bins anormaux
        bad_bins = np.where((counts_f < lo) | (counts_f > hi))[0]
        mask = np.ones(len(time_values), dtype=bool)
        for b in bad_bins:
            in_bin = (time_values >= edges[b]) & (time_values < edges[b + 1])
            mask[in_bin] = False

        n_removed = int((~mask).sum())
        _logger.info(
            "Time gate sigma=%.1f : %d événements supprimés sur %d (%.1f%%)",
            sigma,
            n_removed,
            len(mask),
            100 * n_removed / max(len(mask), 1),
        )
        return mask

    def load_facsdiva_xml(
        self,
        xml_path: Union[str, Path],
        fcs_dir: Optional[Union[str, Path]] = None,
    ) -> List[str]:
        """
        Extension PRISMA — Import FACSDiva XML (non natif FlowKit).

        Parse le XML FACSDiva pour en extraire :
          - la liste des FCS associés,
          - les noms de portes (non traduits en FlowKit — structure XML variable).

        Les FCS sont chargés via load_fcs_batch(). La stratégie de gating
        FACSDiva n'est pas importée automatiquement (format propriétaire BD).

        Args:
            xml_path: Chemin du fichier XML FACSDiva (.xml ou .baf).
            fcs_dir:  Dossier des FCS. Si None, cherche à côté du XML.

        Returns:
            Liste des sample_ids FCS chargés.
        """
        xml_path = Path(xml_path)
        if not xml_path.is_file():
            raise FileNotFoundError(f"FACSDiva XML introuvable : {xml_path}")

        if self._is_workspace():
            raise PrismaEngineError(
                "load_facsdiva_xml() requiert un backend Session. "
                "Créez un nouvel engine avant d'importer."
            )

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as exc:
            raise PrismaEngineError(f"XML FACSDiva invalide : {exc}") from exc

        # Extraire les noms de FCS depuis le XML (attribut 'data_filename' ou 'fcs_filename')
        fcs_names: List[str] = []
        for elem in root.iter():
            for attr in ("data_filename", "fcs_filename", "filename"):
                fname = elem.get(attr, "")
                if fname.lower().endswith(".fcs"):
                    fcs_names.append(fname)

        fcs_names = list(dict.fromkeys(fcs_names))  # déduplique, préserve l'ordre

        if not fcs_names:
            _logger.warning("Aucun nom de FCS trouvé dans le XML FACSDiva : %s", xml_path.name)
            return []

        search_dir = Path(fcs_dir) if fcs_dir else xml_path.parent
        fcs_paths: List[Path] = []
        for name in fcs_names:
            candidate = search_dir / name
            if candidate.is_file():
                fcs_paths.append(candidate)
            else:
                # Recherche insensible à la casse (Windows)
                matches = list(search_dir.glob(f"*{Path(name).stem}*.fcs"))
                if matches:
                    fcs_paths.append(matches[0])
                else:
                    _logger.warning("FCS introuvable : %s (cherché dans %s)", name, search_dir)

        if not fcs_paths:
            raise PrismaEngineError(
                f"Aucun FCS correspondant trouvé pour le XML FACSDiva '{xml_path.name}'."
            )

        ids = self.load_fcs_batch(fcs_paths, make_first_active=True)
        _logger.info(
            "FACSDiva XML '%s' : %d FCS chargés sur %d trouvés",
            xml_path.name,
            len(ids),
            len(fcs_names),
        )
        return ids

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    def _require_active_sample(self) -> str:
        if not self._active_sample_id:
            ids = list(self._backend.get_sample_ids())
            if not ids:
                raise SessionNotLoadedError("Aucun sample chargé.")
            self._active_sample_id = ids[0]
        return self._active_sample_id

    def _assert_sample_exists(self, sample_id: str) -> None:
        if sample_id not in self._backend.get_sample_ids():
            raise SampleNotFoundError(f"Sample introuvable : {sample_id!r}")

    def _require_gate_exists(self, gate_name: str, gate_path: Optional[Tuple[str, ...]]) -> None:
        try:
            self._backend.get_gate(gate_name, gate_path=gate_path)
        except Exception:
            raise GateNotFoundError(f"Gate introuvable : {gate_name!r} @ {gate_path}")

    def _require_session_for_gate_edit(self) -> None:
        """Les modifications de gates (add/remove/rename) nécessitent Session."""
        if self._is_workspace():
            raise PrismaEngineError(
                "add/remove/rename_gate() non disponibles sur Workspace. "
                "Les gates Workspace sont en lecture seule via FlowKit. "
                "Exportez en Session (export_gml) pour éditer."
            )

    def _validate_gate_name(self, gate_name: str, gate_path: Tuple[str, ...]) -> None:
        existing_paths = self.find_gate_paths(gate_name)
        if existing_paths and gate_path not in existing_paths:
            raise GateNameConflictError(
                f"Nom '{gate_name}' déjà utilisé avec chemin différent : {existing_paths}"
            )

    def _validate_channels(self, *channels: str) -> None:
        if not self._active_sample_id:
            return
        available = set(self.get_sample_channels())
        for ch in channels:
            if ch and ch not in available:
                raise IncompatibleAxesError(
                    f"Canal '{ch}' absent du sample. Disponibles : {sorted(available)}"
                )

    def _validate_parent_path(self, gate_path: Tuple[str, ...]) -> None:
        if not gate_path or gate_path == ("root",):
            return
        parent_name = gate_path[-1]
        parent_ancestors = gate_path[:-1] or None
        try:
            self._backend.get_gate(parent_name, gate_path=parent_ancestors)
        except Exception:
            raise GateParentMissingError(
                f"Gate parente introuvable : {parent_name!r} @ {parent_ancestors}"
            )

    def _channel_is_untransformed(self, channel_name: str) -> bool:
        """
        Détecte les canaux classiquement non transformés (FSC/SSC/Time).

        Ces canaux doivent rester en coordonnées raw dans les Dimensions FlowKit,
        sinon les gates 2D mixtes (raw vs logicle) sélectionnent 0 événements.
        """
        ch = str(channel_name).upper()
        return any(token in ch for token in ("FSC", "SSC", "TIME"))

    def _resolve_dimension_refs(
        self,
        channel_name: str,
        comp_ref: str,
        transform_ref: Optional[str],
    ) -> Tuple[str, Optional[str]]:
        """Retourne (comp_ref, transform_ref) adapté à un canal donné."""
        if self._channel_is_untransformed(channel_name):
            return "uncompensated", None
        return comp_ref, transform_ref

    def _build_dimension(
        self,
        channel_name: str,
        comp_ref: str,
        transform_ref: Optional[str],
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
    ) -> Any:
        """Construit une fk.Dimension robuste pour canaux mixtes raw/transformés."""
        effective_comp_ref, effective_transform_ref = self._resolve_dimension_refs(
            channel_name,
            comp_ref,
            transform_ref,
        )
        return fk.Dimension(
            channel_name,
            compensation_ref=effective_comp_ref,
            transformation_ref=effective_transform_ref,
            range_min=range_min,
            range_max=range_max,
        )

    def _replace_existing_gate_if_present(
        self,
        gate_name: str,
        gate_path: Tuple[str, ...],
        sample_id: Optional[str] = None,
    ) -> None:
        """Supprime la gate existante au même path pour permettre un overwrite sûr."""
        session = self._as_session()
        try:
            existing_gate = session.get_gate(gate_name, gate_path=gate_path)
            if existing_gate is not None:
                session.remove_gate(
                    gate_name,
                    gate_path=gate_path,
                    sample_id=sample_id,
                    keep_children=False,
                )
                _logger.debug("Gate remplacée : %s @ %s", gate_name, gate_path)
        except Exception as exc:
            # Les erreurs "gate absente" sont attendues au premier ajout.
            if self._is_missing_gate_lookup_error(exc) or self._is_gate_tree_error(exc):
                return
            raise

    def _is_gate_tree_error(self, exc: Exception) -> bool:
        """Compatibilité FlowKit: détecte GateTreeError sans dépendre d'une API précise."""
        exc_mod = getattr(fk, "exceptions", None)
        gate_tree_err = getattr(exc_mod, "GateTreeError", None) if exc_mod is not None else None
        if gate_tree_err is not None and isinstance(exc, gate_tree_err):
            return True
        return exc.__class__.__name__ == "GateTreeError"

    def _is_missing_gate_lookup_error(self, exc: Exception) -> bool:
        """Détecte les erreurs FlowKit indiquant qu'une gate n'existe pas encore."""
        exc_mod = getattr(fk, "exceptions", None)
        gate_ref_err = getattr(exc_mod, "GateReferenceError", None) if exc_mod is not None else None
        if gate_ref_err is not None and isinstance(exc, gate_ref_err):
            return True

        exc_name = exc.__class__.__name__
        if exc_name == "GateReferenceError":
            return True

        msg = str(exc).lower()
        return "was not found in gating strategy" in msg or "not found" in msg
