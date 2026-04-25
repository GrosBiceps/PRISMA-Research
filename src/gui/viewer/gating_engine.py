"""
src/gui/viewer/gating_engine.py — Moteur FlowKit pour PrismaGatingViewer.

Encapsule flowkit.Session comme unique source de vérité pour :
  - import/export FCS, WSP, GatingML 2.0
  - compensation et transformations
  - hiérarchie de gating
  - analyse et extraction d'événements

Architecture :
  PrismaFlowEngine   — façade publique pour l'UI
  TransformSpec      — configuration d'une transformation par canal
  GateHierarchyNode  — nœud de la hiérarchie exporté vers la vue
  PrismaEngineError  — exceptions métier
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import flowkit as fk
from flowkit import gates as fk_gates
from flowkit import transforms as fk_transforms

from src.utils.logger import get_logger

_logger = get_logger("viewer.gating_engine")


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


# ---------------------------------------------------------------------------
# Types de données
# ---------------------------------------------------------------------------


@dataclass
class TransformSpec:
    """
    Spécification d'une transformation pour un ou plusieurs canaux.

    Attributes:
        kind: 'logicle' | 'log' | 'linear' | 'hyperlog' | 'asinh'
        channel_ids: Liste de canaux concernés. Si vide → appliqué à tous.
        params: Paramètres spécifiques au type (dépend de flowkit).
    """

    kind: str
    channel_ids: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    # Valeurs par défaut raisonnables pour cytométrie spectrale
    _DEFAULTS: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "logicle":  {"param_t": 262144, "param_w": 0.5, "param_m": 4.5, "param_a": 0.0},
        "log":      {"param_t": 262144, "param_m": 4.5},
        "linear":   {"param_t": 262144, "param_a": 0.0},
        "hyperlog": {"param_t": 262144, "param_w": 0.5, "param_m": 4.5, "param_a": 0.0},
        "asinh":    {"param_t": 262144, "param_m": 4.0, "param_a": 0.0},
    }, init=False, repr=False)

    def merged_params(self) -> Dict[str, Any]:
        """Retourne les paramètres fusionnés avec les défauts pour ce type."""
        defaults = self._DEFAULTS.get(self.kind, {})
        return {**defaults, **self.params}


@dataclass
class GateHierarchyNode:
    """
    Nœud de la hiérarchie de gating exporté vers la vue.

    Attributes:
        gate_name: Identifiant de la gate (gate_id FlowKit).
        gate_path: Tuple complet des ancêtres (gate_path FlowKit).
        gate_type: 'polygon' | 'rectangle' | 'ellipsoid' | 'quadrant' | 'boolean'.
        children:  Nœuds enfants.
        count:     Nombre d'événements positifs (rempli après analyze_samples).
        pct_parent: Pourcentage par rapport au parent.
        pct_grandparent: Pourcentage par rapport au grand-parent.
    """

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


class PrismaFlowEngine:
    """
    Façade FlowKit pour PrismaGatingViewer.

    Toute la logique métier de gating passe par cette classe.
    L'UI ne manipule jamais directement la Session FlowKit.

    Usage typique :
        engine = PrismaFlowEngine()
        engine.load_fcs("sample.fcs")
        engine.add_logicle_transform_to_all()
        engine.add_polygon_gate("Lymphocytes", ("root",), "FSC-A", "SSC-A",
                                [(100, 200), (300, 200), (300, 400), (100, 400)])
        engine.analyze()
        df = engine.get_gate_dataframe("Lymphocytes", ("root",), sample_id=None)
    """

    def __init__(self) -> None:
        self._session: fk.Session = fk.Session()
        self._active_sample_id: Optional[str] = None
        # transform_id → TransformSpec (pour introspection UI)
        self._transform_specs: Dict[str, TransformSpec] = {}
        # comp_matrix_id → label (pour introspection UI)
        self._comp_matrix_ids: List[str] = []
        _logger.info("PrismaFlowEngine initialisé")

    # ------------------------------------------------------------------
    # A. Accès session
    # ------------------------------------------------------------------

    @property
    def session(self) -> fk.Session:
        return self._session

    @property
    def active_sample_id(self) -> Optional[str]:
        return self._active_sample_id

    def set_active_sample(self, sample_id: str) -> None:
        self._assert_sample_exists(sample_id)
        self._active_sample_id = sample_id
        _logger.debug("Sample actif : %s", sample_id)

    def get_sample_ids(self) -> List[str]:
        return list(self._session.get_sample_ids())

    def get_active_sample(self) -> fk.Sample:
        sid = self._require_active_sample()
        return self._session.get_sample(sid)

    def get_sample_channels(self, sample_id: Optional[str] = None) -> List[str]:
        """
        Retourne les noms de canaux utilisables comme dimension_id FlowKit.

        FlowKit retourne les colonnes sous forme de tuples (pnn, pns).
        On extrait le pnn (premier élément) qui est le référent attendu
        par fk.Dimension() et les gates.
        """
        sid = sample_id or self._require_active_sample()
        sample = self._session.get_sample(sid)
        df = sample.as_dataframe(source="raw")
        cols = list(df.columns)
        # Normaliser : extraire pnn si tuple, sinon garder la chaîne
        result: List[str] = []
        for col in cols:
            if isinstance(col, tuple):
                result.append(col[0])  # pnn
            else:
                result.append(str(col))
        return result

    # ------------------------------------------------------------------
    # B. I/O
    # ------------------------------------------------------------------

    def load_fcs(self, fcs_path: Union[str, Path], make_active: bool = True) -> str:
        """
        Charge un fichier FCS dans la session.

        Returns:
            sample_id du sample chargé.
        """
        fcs_path = Path(fcs_path)
        if not fcs_path.is_file():
            raise FileNotFoundError(f"FCS introuvable : {fcs_path}")

        sample = fk.Sample(str(fcs_path))
        self._session.add_samples(sample)
        sample_id = sample.id
        _logger.info("FCS chargé : %s → sample_id=%s", fcs_path.name, sample_id)

        if make_active:
            self._active_sample_id = sample_id
        return sample_id

    def load_fcs_batch(
        self,
        sources: Union[List[Union[str, Path]], Union[str, Path]],
        make_first_active: bool = True,
    ) -> List[str]:
        """
        Charge un dossier entier ou une liste de fichiers FCS.

        Returns:
            Liste des sample_ids chargés.
        """
        sources_path = Path(str(sources)) if not isinstance(sources, list) else None
        if sources_path and sources_path.is_dir():
            files = sorted(sources_path.glob("*.fcs")) + sorted(
                sources_path.glob("*.FCS")
            )
        else:
            files = [Path(p) for p in (sources if isinstance(sources, list) else [])]

        if not files:
            raise FileNotFoundError("Aucun fichier FCS trouvé dans la source fournie.")

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

        fk.Session.from_wsp() n'est pas documentée en mode headless et peut
        disparaître. fk.Workspace est l'objet public stable.

        Args:
            wsp_path: Chemin du fichier .wsp.
            fcs_dir:  Dossier racine des FCS si introuvables via les chemins WSP.
                      Si None, FlowKit cherche dans le même dossier que le WSP.

        Returns:
            Liste des sample_ids chargés.

        Raises:
            FileNotFoundError: si le fichier WSP est absent.
            PrismaEngineError: si le workspace ne contient aucun sample.
        """
        wsp_path = Path(wsp_path)
        if not wsp_path.is_file():
            raise FileNotFoundError(f"WSP introuvable : {wsp_path}")

        fcs_samples = None
        if fcs_dir is not None:
            fcs_dir = Path(fcs_dir)
            fcs_files = sorted(fcs_dir.glob("*.fcs")) + sorted(fcs_dir.glob("*.FCS"))
            if fcs_files:
                fcs_samples = [fk.Sample(str(f)) for f in fcs_files]
                _logger.debug("FCS pré-chargés pour WSP : %d fichiers", len(fcs_samples))

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
            raise PrismaEngineError(
                f"Le workspace '{wsp_path.name}' ne contient aucun sample."
            )

        # Remplacer la session courante par le workspace
        # fk.Workspace expose la même API que fk.Session pour analyze/get_gate_events/etc.
        self._session = workspace  # type: ignore[assignment]
        self._active_sample_id = ids[0]
        _logger.info("WSP chargé : %s → %d samples", wsp_path.name, len(ids))
        return ids

    def export_wsp(self, output_path: Union[str, Path], group_name: str = "PRISMA") -> None:
        """Exporte la session courante en workspace FlowJo (.wsp)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            self._session.export_wsp(fh, group_name=group_name)
        _logger.info("WSP exporté : %s", output_path)

    def load_gml(self, gml_path: Union[str, Path]) -> None:
        """
        Importe une stratégie de gating depuis un fichier GatingML 2.0.

        La session doit déjà avoir des samples chargés pour que l'analyse
        soit possible après import.
        """
        gml_path = Path(gml_path)
        if not gml_path.is_file():
            raise FileNotFoundError(f"GatingML introuvable : {gml_path}")

        session = fk.Session.from_gml(str(gml_path))
        # Préserver les samples existants si possible
        existing_ids = list(self._session.get_sample_ids())
        if existing_ids:
            for sid in existing_ids:
                try:
                    session.add_samples(self._session.get_sample(sid))
                except Exception:
                    pass
        self._session = session
        _logger.info("GatingML importé : %s", gml_path.name)

    def export_gml(self, output_path: Union[str, Path], sample_id: Optional[str] = None) -> None:
        """Exporte la stratégie courante en GatingML 2.0."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            self._session.export_gml(fh, sample_id=sample_id)
        _logger.info("GatingML exporté : %s", output_path)

    # ------------------------------------------------------------------
    # C. Compensation
    # ------------------------------------------------------------------

    def load_compensation_from_fcs_metadata(
        self, sample_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Lit la matrice de compensation depuis les métadonnées FCS du sample.

        Returns:
            matrix_id si succès, None si absent.
        """
        sid = sample_id or self._require_active_sample()
        sample = self._session.get_sample(sid)
        meta = sample.get_metadata()

        spill_key = next(
            (k for k in meta if k.lower() in ("spill", "spillover", "$spill", "$spillover")),
            None,
        )
        if spill_key is None:
            _logger.warning("Pas de matrice spillover dans les métadonnées FCS du sample %s", sid)
            return None

        try:
            matrix_id = f"spill_{sid}"
            matrix = fk.Matrix(meta[spill_key])
            self._session.add_comp_matrix(matrix_id, matrix)
            self._comp_matrix_ids.append(matrix_id)
            _logger.info("Matrice de compensation chargée depuis FCS : %s", matrix_id)
            return matrix_id
        except Exception as exc:
            _logger.error("Erreur lecture matrice FCS : %s", exc)
            return None

    def load_compensation_from_csv(
        self, csv_path: Union[str, Path], matrix_id: Optional[str] = None
    ) -> str:
        """
        Charge une matrice de compensation depuis un CSV externe.

        Le CSV doit avoir la première ligne = noms des détecteurs,
        et la première colonne = noms des fluorochromes (ou répétés).

        Returns:
            matrix_id enregistré dans la session.
        """
        csv_path = Path(csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV compensation introuvable : {csv_path}")

        df = pd.read_csv(csv_path, index_col=0)
        spill_array = df.values.astype(np.float64)
        detectors = list(df.columns)
        fluorochromes = list(df.index)

        matrix_id = matrix_id or f"comp_{csv_path.stem}"
        matrix = fk.Matrix(spill_array, detectors=detectors, fluorochromes=fluorochromes)
        self._session.add_comp_matrix(matrix_id, matrix)
        self._comp_matrix_ids.append(matrix_id)
        _logger.info("Matrice de compensation CSV chargée : %s", matrix_id)
        return matrix_id

    def get_comp_matrix_ids(self) -> List[str]:
        return list(self._comp_matrix_ids)

    # ------------------------------------------------------------------
    # D. Transformations
    # ------------------------------------------------------------------

    def add_transform(self, transform_id: str, spec: TransformSpec) -> None:
        """
        Ajoute une transformation dans la session FlowKit.

        Args:
            transform_id: Identifiant unique (ex: 'logicle_global').
            spec: TransformSpec décrivant le type et les paramètres.
        """
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
            transform = fk_transforms.LogTransform(
                param_t=p["param_t"], param_m=p["param_m"]
            )
        elif kind == "linear":
            transform = fk_transforms.LinearTransform(
                param_t=p["param_t"], param_a=p["param_a"]
            )
        elif kind == "hyperlog":
            transform = fk_transforms.HyperlogTransform(
                param_t=p["param_t"],
                param_w=p["param_w"],
                param_m=p["param_m"],
                param_a=p["param_a"],
            )
        elif kind == "asinh":
            transform = fk_transforms.AsinhTransform(
                param_t=p["param_t"], param_m=p["param_m"], param_a=p["param_a"]
            )
        else:
            raise PrismaEngineError(f"Type de transformation inconnu : {kind!r}")

        self._session.add_transform(transform_id, transform)
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
        """
        Retourne le DataFrame transformé et/ou compensé d'un sample.

        Si transform_id est None, retourne les données brutes.
        """
        sid = sample_id or self._require_active_sample()
        sample = self._session.get_sample(sid)

        transform = self._session.get_transform(transform_id) if transform_id else None
        matrix = self._session.get_comp_matrix(comp_matrix_id) if comp_matrix_id else None

        source = "xform" if transform is not None else "raw"
        df = sample.as_dataframe(source=source)
        _logger.debug("DataFrame %s retourné : %d cellules × %d canaux", source, *df.shape)
        return df

    # ------------------------------------------------------------------
    # E. API de gating — création
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
    ) -> None:
        """
        Ajoute une gate polygonale dans la stratégie FlowKit.

        Args:
            gate_name:     Nom unique de la gate.
            gate_path:     Tuple des ancêtres (ex: ('root',) ou ('root', 'Lymphocytes')).
            x_channel:     Canal axe X (dimension_id FlowKit).
            y_channel:     Canal axe Y.
            vertices:      Liste de (x, y) en coordonnées données transformées.
            comp_ref:      Référence de compensation ('uncompensated' ou matrix_id).
            transform_ref: Référence de transformation (transform_id ou None).
        """
        self._validate_gate_name(gate_name, gate_path)
        self._validate_channels(x_channel, y_channel)

        dim_x = fk.Dimension(
            x_channel,
            compensation_ref=comp_ref,
            transformation_ref=transform_ref,
        )
        dim_y = fk.Dimension(
            y_channel,
            compensation_ref=comp_ref,
            transformation_ref=transform_ref,
        )
        gate = fk_gates.PolygonGate(gate_name, [dim_x, dim_y], vertices)
        self._session.add_gate(gate, gate_path)
        _logger.info("PolygonGate ajoutée : %s @ %s", gate_name, gate_path)

    def add_rectangle_gate(
        self,
        gate_name: str,
        gate_path: Tuple[str, ...],
        x_channel: str,
        y_channel: str,
        x_min: Optional[float],
        x_max: Optional[float],
        y_min: Optional[float],
        y_max: Optional[float],
        comp_ref: str = "uncompensated",
        transform_ref: Optional[str] = None,
    ) -> None:
        """
        Ajoute une gate rectangulaire.

        range_min/max None = pas de borne sur cet axe.
        """
        self._validate_gate_name(gate_name, gate_path)
        self._validate_channels(x_channel, y_channel)

        dim_x = fk.Dimension(
            x_channel,
            compensation_ref=comp_ref,
            transformation_ref=transform_ref,
            range_min=x_min,
            range_max=x_max,
        )
        dim_y = fk.Dimension(
            y_channel,
            compensation_ref=comp_ref,
            transformation_ref=transform_ref,
            range_min=y_min,
            range_max=y_max,
        )
        gate = fk_gates.RectangleGate(gate_name, [dim_x, dim_y])
        self._session.add_gate(gate, gate_path)
        _logger.info("RectangleGate ajoutée : %s @ %s", gate_name, gate_path)

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
    ) -> None:
        """Ajoute une gate ellipsoïde."""
        self._validate_gate_name(gate_name, gate_path)
        self._validate_channels(x_channel, y_channel)

        dim_x = fk.Dimension(x_channel, compensation_ref=comp_ref, transformation_ref=transform_ref)
        dim_y = fk.Dimension(y_channel, compensation_ref=comp_ref, transformation_ref=transform_ref)
        gate = fk_gates.EllipsoidGate(
            gate_name,
            [dim_x, dim_y],
            coordinates=coordinates,
            covariance_matrix=np.array(covariance_matrix),
            distance_square=distance_square,
        )
        self._session.add_gate(gate, gate_path)
        _logger.info("EllipsoidGate ajoutée : %s @ %s", gate_name, gate_path)

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
    ) -> None:
        """
        Ajoute une QuadrantGate (4 quadrants).

        Les quadrant_ids générés sont : Q1_PosPos, Q2_NegPos, Q3_NegNeg, Q4_PosNeg.
        """
        self._validate_gate_name(gate_name, gate_path)
        self._validate_channels(x_channel, y_channel)

        divider_x_id = f"{gate_name}_div_x"
        divider_y_id = f"{gate_name}_div_y"

        div_x = fk.QuadrantDivider(
            divider_id=divider_x_id,
            dimension_ref=x_channel,
            compensation_ref=comp_ref,
            values=[x_threshold],
            transformation_ref=transform_ref,
        )
        div_y = fk.QuadrantDivider(
            divider_id=divider_y_id,
            dimension_ref=y_channel,
            compensation_ref=comp_ref,
            values=[y_threshold],
            transformation_ref=transform_ref,
        )

        q_pp = fk_gates.Quadrant(
            f"{gate_name}_Q1_PosPos",
            divider_refs=[divider_x_id, divider_y_id],
            divider_ranges=[(x_threshold, None), (y_threshold, None)],
        )
        q_np = fk_gates.Quadrant(
            f"{gate_name}_Q2_NegPos",
            divider_refs=[divider_x_id, divider_y_id],
            divider_ranges=[(None, x_threshold), (y_threshold, None)],
        )
        q_nn = fk_gates.Quadrant(
            f"{gate_name}_Q3_NegNeg",
            divider_refs=[divider_x_id, divider_y_id],
            divider_ranges=[(None, x_threshold), (None, y_threshold)],
        )
        q_pn = fk_gates.Quadrant(
            f"{gate_name}_Q4_PosNeg",
            divider_refs=[divider_x_id, divider_y_id],
            divider_ranges=[(x_threshold, None), (None, y_threshold)],
        )

        gate = fk_gates.QuadrantGate(
            gate_name,
            dividers=[div_x, div_y],
            quadrants=[q_pp, q_np, q_nn, q_pn],
        )
        self._session.add_gate(gate, gate_path)
        _logger.info("QuadrantGate ajoutée : %s @ %s", gate_name, gate_path)

    def add_boolean_gate(
        self,
        gate_name: str,
        gate_path: Tuple[str, ...],
        bool_type: str,
        gate_refs: List[Dict[str, Any]],
    ) -> None:
        """
        Ajoute une BooleanGate.

        Args:
            bool_type: 'and' | 'or' | 'not'
            gate_refs: Liste de dicts {'gate_name': str, 'gate_path': tuple, 'complement': bool}
        """
        self._validate_gate_name(gate_name, gate_path)

        for ref in gate_refs:
            ref_name = ref["gate_name"]
            ref_path = ref.get("gate_path", ("root",))
            try:
                self._session.get_gate(ref_name, gate_path=ref_path)
            except Exception:
                raise BooleanGateRefMissingError(
                    f"Gate référencée introuvable : {ref_name} @ {ref_path}"
                )

        gate = fk_gates.BooleanGate(gate_name, bool_type=bool_type, gate_refs=gate_refs)
        self._session.add_gate(gate, gate_path)
        _logger.info("BooleanGate ajoutée : %s (%s) @ %s", gate_name, bool_type, gate_path)

    # ------------------------------------------------------------------
    # E. API de gating — modification / suppression
    # ------------------------------------------------------------------

    def remove_gate(
        self,
        gate_name: str,
        gate_path: Optional[Tuple[str, ...]] = None,
        keep_children: bool = False,
    ) -> None:
        """Supprime une gate de la stratégie."""
        self._require_gate_exists(gate_name, gate_path)
        self._session.remove_gate(gate_name, gate_path=gate_path, keep_children=keep_children)
        _logger.info("Gate supprimée : %s", gate_name)

    def rename_gate(
        self,
        gate_name: str,
        new_name: str,
        gate_path: Optional[Tuple[str, ...]] = None,
    ) -> None:
        """Renomme une gate existante."""
        self._require_gate_exists(gate_name, gate_path)
        self._session.rename_gate(gate_name, new_name, gate_path=gate_path)
        _logger.info("Gate renommée : %s → %s", gate_name, new_name)

    def get_gate(
        self, gate_name: str, gate_path: Optional[Tuple[str, ...]] = None
    ) -> Any:
        """Retourne l'objet FlowKit Gate."""
        self._require_gate_exists(gate_name, gate_path)
        return self._session.get_gate(gate_name, gate_path=gate_path)

    def get_gate_ids(self) -> List[str]:
        """
        Liste tous les gate_names de la stratégie.

        flowkit.Session.get_gate_ids() retourne des tuples (gate_name, gate_path).
        Cette méthode retourne uniquement les gate_names (strings).
        """
        raw = self._session.get_gate_ids()
        result: List[str] = []
        for item in raw:
            if isinstance(item, tuple):
                result.append(str(item[0]))
            else:
                result.append(str(item))
        return result

    def get_gate_id_path_pairs(self) -> List[Tuple[str, Tuple[str, ...]]]:
        """Retourne tous les (gate_name, gate_path) de la stratégie."""
        raw = self._session.get_gate_ids()
        result: List[Tuple[str, Tuple[str, ...]]] = []
        for item in raw:
            if isinstance(item, tuple):
                result.append((str(item[0]), item[1]))
            else:
                result.append((str(item), ("root",)))
        return result

    def find_gate_paths(self, gate_name: str) -> List[Tuple[str, ...]]:
        """Retourne tous les chemins correspondant à un nom de gate."""
        return list(self._session.find_matching_gate_paths(gate_name))

    def get_children(
        self,
        gate_name: str,
        gate_path: Optional[Tuple[str, ...]] = None,
    ) -> List[str]:
        """Retourne les gate_ids enfants directs."""
        try:
            return list(self._session.get_child_gate_ids(gate_name, gate_path=gate_path))
        except Exception:
            return []

    # ------------------------------------------------------------------
    # F. Analyse / extraction
    # ------------------------------------------------------------------

    def analyze(self, sample_id: Optional[str] = None) -> None:
        """Lance l'analyse FlowKit sur un sample ou tous les samples."""
        self._session.analyze_samples(sample_id=sample_id, use_mp=False)
        _logger.info("Analyse terminée : sample_id=%s", sample_id or "all")

    def get_gate_dataframe(
        self,
        gate_name: str,
        gate_path: Optional[Tuple[str, ...]] = None,
        sample_id: Optional[str] = None,
        matrix_id: Optional[str] = None,
        transform_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retourne le DataFrame des événements positifs pour une gate donnée.

        Args:
            gate_name:    Identifiant de la gate.
            gate_path:    Chemin complet des ancêtres.
            sample_id:    ID du sample (défaut = sample actif).
            matrix_id:    Matrice de compensation à appliquer.
            transform_id: Transformation à appliquer.

        Returns:
            DataFrame (N_gated × N_channels).
        """
        sid = sample_id or self._require_active_sample()
        matrix = self._session.get_comp_matrix(matrix_id) if matrix_id else None
        transform = self._session.get_transform(transform_id) if transform_id else None

        df = self._session.get_gate_events(
            sid,
            gate_name=gate_name,
            gate_path=gate_path,
            matrix=matrix,
            transform=transform,
        )
        return df

    def get_gate_membership(
        self,
        gate_name: str,
        gate_path: Optional[Tuple[str, ...]] = None,
        sample_id: Optional[str] = None,
    ) -> np.ndarray:
        """Retourne le masque booléen d'appartenance à une gate."""
        sid = sample_id or self._require_active_sample()
        return self._session.get_gate_membership(sid, gate_name, gate_path=gate_path)

    def get_analysis_report(self) -> pd.DataFrame:
        """Retourne le rapport analytique FlowKit sous forme de DataFrame."""
        return self._session.get_analysis_report()

    def get_raw_dataframe(
        self,
        sample_id: Optional[str] = None,
        gate_name: Optional[str] = None,
        gate_path: Optional[Tuple[str, ...]] = None,
        transform_id: Optional[str] = None,
        comp_matrix_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retourne le DataFrame brut ou filtré prêt pour affichage frontend.

        Si gate_name est None → tous les événements.
        Si transform_id ou comp_matrix_id sont fournis → appliqués.
        """
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

    # ------------------------------------------------------------------
    # G. Hiérarchie pour la vue
    # ------------------------------------------------------------------

    def build_hierarchy(
        self, sample_id: Optional[str] = None
    ) -> List[GateHierarchyNode]:
        """
        Construit une liste de GateHierarchyNode racines pour le QTreeView.

        Utilise get_gate_id_path_pairs() pour obtenir les (name, path) corrects
        depuis FlowKit (qui retourne des tuples).
        Les comptages sont remplis si analyze() a été appelé.
        """
        pairs = self.get_gate_id_path_pairs()
        if not pairs:
            return []

        # Créer tous les nœuds
        nodes: Dict[str, GateHierarchyNode] = {}
        for gname, gpath in pairs:
            try:
                gate_obj = self._session.get_gate(gname, gate_path=gpath)
                gate_type = type(gate_obj).__name__.lower().replace("gate", "")
            except Exception:
                gate_obj = None
                gate_type = "unknown"

            node = GateHierarchyNode(
                gate_name=gname,
                gate_path=gpath,
                gate_type=gate_type,
            )
            # Clé unique = (name, path) car deux gates peuvent avoir le même nom
            node_key = f"{gname}|{'|'.join(gpath)}"
            nodes[node_key] = node

            # CORRECTIF C3 : QuadrantGate — injecter explicitement les 4 quadrants
            # comme nœuds fils si FlowKit ne les a pas listés dans get_gate_ids().
            # FlowKit 1.3.0 les expose déjà via get_gate_ids() mais en cas d'absence
            # (WSP importé, version future), on les force manuellement.
            if gate_obj is not None and isinstance(gate_obj, fk.gates.QuadrantGate):
                child_path = gpath + (gname,)
                existing_child_names = {n for n, p in pairs if p == child_path}
                for quad_id, quad_obj in gate_obj.quadrants.items():
                    if quad_id in existing_child_names:
                        continue  # déjà présent via get_gate_ids()
                    quad_key = f"{quad_id}|{'|'.join(child_path)}"
                    if quad_key not in nodes:
                        nodes[quad_key] = GateHierarchyNode(
                            gate_name=quad_id,
                            gate_path=child_path,
                            gate_type="quadrant",
                        )

        # Remplir counts depuis le rapport d'analyse si disponible
        # Utilise standardize_report pour être robuste aux renommages FlowKit
        if sample_id or self._active_sample_id:
            sid = sample_id or self._active_sample_id
            try:
                from src.exports.gating_exporter import GatingExporter
                report_raw = self.get_analysis_report()
                if report_raw is not None and not report_raw.empty:
                    report = GatingExporter.standardize_report(report_raw)
                    for _, row in report.iterrows():
                        if row.get("sample_id") is not None and str(row["sample_id"]) != str(sid):
                            continue
                        gname_r = str(row.get("gate_name") or "")
                        count_val = row.get("count")
                        for node in nodes.values():
                            if node.gate_name == gname_r:
                                node.count = int(count_val) if count_val is not None else 0
                                break
            except Exception:
                pass

        # Construire l'arborescence parent/enfant
        # gate_path = ('root',) → racine ; gate_path = ('root', 'ParentGate') → enfant de ParentGate
        roots: List[GateHierarchyNode] = []
        node_by_name: Dict[str, GateHierarchyNode] = {}
        for node in nodes.values():
            node_by_name[node.gate_name] = node

        for node in nodes.values():
            if len(node.gate_path) <= 1:
                roots.append(node)
            else:
                parent_name = node.gate_path[-1]
                parent_node = node_by_name.get(parent_name)
                if parent_node is not None:
                    parent_node.children.append(node)
                else:
                    roots.append(node)

        # Calculer pct_parent / pct_grandparent
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
    # G. Conversion UI → FlowKit
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
    ) -> None:
        """
        Crée et enregistre une PolygonGate depuis des sommets canvas.

        Valide : axes, unicité du nom, path parent, nombre de sommets.
        """
        if len(vertices) < 3:
            raise PrismaEngineError("Un polygone requiert au minimum 3 sommets.")
        self._validate_parent_path(gate_path)
        self.add_polygon_gate(
            gate_name, gate_path, x_channel, y_channel, vertices,
            comp_ref=comp_ref, transform_ref=transform_ref,
        )

    def create_rectangle_gate_from_bounds(
        self,
        gate_name: str,
        gate_path: Tuple[str, ...],
        x_channel: str,
        y_channel: str,
        x_min: Optional[float],
        x_max: Optional[float],
        y_min: Optional[float],
        y_max: Optional[float],
        comp_ref: str = "uncompensated",
        transform_ref: Optional[str] = None,
    ) -> None:
        """Crée et enregistre une RectangleGate depuis des bornes canvas."""
        self._validate_parent_path(gate_path)
        self.add_rectangle_gate(
            gate_name, gate_path, x_channel, y_channel,
            x_min, x_max, y_min, y_max,
            comp_ref=comp_ref, transform_ref=transform_ref,
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
    ) -> None:
        """Crée et enregistre une QuadrantGate depuis un point croisé canvas."""
        self._validate_parent_path(gate_path)
        self.add_quadrant_gate(
            gate_name, gate_path, x_channel, y_channel,
            x_threshold, y_threshold,
            comp_ref=comp_ref, transform_ref=transform_ref,
        )

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    def _require_active_sample(self) -> str:
        if not self._active_sample_id:
            ids = list(self._session.get_sample_ids())
            if not ids:
                raise SessionNotLoadedError("Aucun sample chargé dans la session.")
            self._active_sample_id = ids[0]
        return self._active_sample_id

    def _assert_sample_exists(self, sample_id: str) -> None:
        if sample_id not in self._session.get_sample_ids():
            raise SampleNotFoundError(f"Sample introuvable : {sample_id!r}")

    def _require_gate_exists(
        self, gate_name: str, gate_path: Optional[Tuple[str, ...]]
    ) -> None:
        try:
            self._session.get_gate(gate_name, gate_path=gate_path)
        except Exception:
            raise GateNotFoundError(
                f"Gate introuvable : {gate_name!r} @ {gate_path}"
            )

    def _validate_gate_name(
        self, gate_name: str, gate_path: Tuple[str, ...]
    ) -> None:
        existing_paths = self.find_gate_paths(gate_name)
        if existing_paths and gate_path not in existing_paths:
            raise GateNameConflictError(
                f"Nom de gate '{gate_name}' déjà utilisé avec un chemin différent : {existing_paths}"
            )

    def _validate_channels(self, *channels: str) -> None:
        if not self._active_sample_id:
            return
        available = set(self.get_sample_channels())
        for ch in channels:
            if ch not in available:
                raise IncompatibleAxesError(
                    f"Canal '{ch}' absent du sample. Disponibles : {sorted(available)}"
                )

    def _validate_parent_path(self, gate_path: Tuple[str, ...]) -> None:
        if not gate_path or gate_path == ("root",):
            return
        parent_name = gate_path[-1]
        parent_ancestors = gate_path[:-1] or None
        try:
            self._session.get_gate(parent_name, gate_path=parent_ancestors)
        except Exception:
            raise GateParentMissingError(
                f"Gate parente introuvable : {parent_name!r} @ {parent_ancestors}"
            )
