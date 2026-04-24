"""
marker_harmonizer.py — Harmonisation des noms de marqueurs FCS inter-fichiers.

Résout l'instabilité de nomenclature $PnS observée lors de la construction
d'une matrice de référence NBM (Normal Bone Marrow) :
  - "CD34 Cy55"  →  "CD34"
  - "CD13 BV421" →  "CD13"
  - "CD45 KO"    →  "CD45"

La logique de matching utilise un regex word-boundary pour éviter les
faux positifs (ex: ne pas matcher "CD33" quand on cherche "CD3").
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Optional

import pandas as pd

from src.utils.logger import get_logger

_logger = get_logger("utils.marker_harmonizer")


def _extract_channel_suffix(name: str) -> Optional[str]:
    """Return cytometry channel suffix if present (A/H), else None."""
    m = re.search(r"-(A|H)\b", name, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).upper()


def harmonize_marker_names(
    raw_columns: List[str],
    target_markers: List[str],
) -> Dict[str, str]:
    """
    Construit un dictionnaire de renommage {colonne_brute → marqueur_cible}.

    Logique de matching (par ordre de priorité) :
    1. Correspondance exacte  → conservée sans log.
    2. Regex word-boundary    → "CD13 BV421" matche "CD13" mais pas "CD133".
       Pattern : r'(?i)^<marker>\\b'  (le marqueur cible doit apparaître
       en début de chaîne, suivi d'un non-word char ou fin de chaîne).

    Args:
        raw_columns:    Liste des noms de colonnes tels que lus dans le FCS.
        target_markers: Liste des noms canoniques attendus (ex: ["CD34", "CD13"]).

    Returns:
        Dictionnaire {raw_col: canonical_name} pour toutes les colonnes qui
        ont pu être mappées.  Les colonnes sans correspondance ne sont PAS
        incluses (pas de renommage implicite).

    Raises:
        ValueError: Si deux colonnes brutes pointent vers le même marqueur cible
                    (ambiguïté non résoluble automatiquement).
    """
    rename_map: Dict[str, str] = {}
    # Vérifier les doublons de cibles
    target_hit: Dict[str, str] = {}  # canonical → raw_col ayant matchée

    for raw_col in raw_columns:
        matched_target: Optional[str] = None

        # Passe 1 : correspondance exacte (case-sensitive)
        if raw_col in target_markers:
            matched_target = raw_col
            # Aucun log : pas de transformation

        # Passe 2 : matching regex
        if matched_target is None:
            for target in target_markers:
                # Si la cible encode explicitement le canal (-A/-H), on impose
                # le même suffixe dans la colonne brute.
                if target.upper().endswith("-A") or target.upper().endswith("-H"):
                    target_base = target[:-2]
                    target_suffix = target[-1].upper()
                    pattern = re.compile(
                        r"^" + re.escape(target_base) + r"(?:\s|$|[^a-zA-Z0-9])",
                        re.IGNORECASE,
                    )
                    if (
                        pattern.match(raw_col)
                        and _extract_channel_suffix(raw_col) == target_suffix
                    ):
                        matched_target = target
                        break  # Premier match wins ; l'ordre de target_markers compte
                else:
                    # Word boundary après le nom du marqueur pour éviter CD3 → CD33
                    pattern = re.compile(
                        r"^" + re.escape(target) + r"(?:\s|$|[^a-zA-Z0-9])",
                        re.IGNORECASE,
                    )
                    if pattern.match(raw_col):
                        matched_target = target
                        break  # Premier match wins ; l'ordre de target_markers compte

        # Si la colonne brute porte un suffixe -A/-H, préférer une cible qui
        # porte le même suffixe lorsqu'elle existe dans la liste des cibles.
        if matched_target is not None:
            raw_suffix = _extract_channel_suffix(raw_col)
            if raw_suffix and not (
                matched_target.upper().endswith("-A")
                or matched_target.upper().endswith("-H")
            ):
                candidate = f"{matched_target}-{raw_suffix}"
                if candidate in target_markers:
                    matched_target = candidate

        if matched_target is None:
            _logger.debug(
                "Harmonisation : aucune cible trouvée pour la colonne '%s' — ignorée.",
                raw_col,
            )
            continue

        # Détecter les conflits (deux raw → même target)
        if matched_target in target_hit and target_hit[matched_target] != raw_col:
            raise ValueError(
                f"Conflit d'harmonisation : les colonnes '{raw_col}' et "
                f"'{target_hit[matched_target]}' correspondent toutes deux "
                f"au marqueur cible '{matched_target}'. "
                "Vérifiez votre panel ou déduplicez manuellement."
            )

        target_hit[matched_target] = raw_col
        rename_map[raw_col] = matched_target

        # Log détaillé uniquement en debug pour éviter le spam en mode batch
        if raw_col != matched_target:
            _logger.debug(
                "Harmonisation : La colonne '%s' a été identifiée et renommée "
                "en '%s' par regex.",
                raw_col,
                matched_target,
            )

    n_renamed = sum(1 for src, dst in rename_map.items() if src != dst)
    n_exact = len(rename_map) - n_renamed
    _logger.info(
        "Harmonisation terminée : %d colonne(s) renommée(s), "
        "%d déjà conformes, %d colonne(s) brute(s) sans correspondance.",
        n_renamed,
        n_exact,
        len(raw_columns) - len(rename_map),
    )

    return rename_map


def apply_harmonization(
    df: pd.DataFrame,
    target_markers: List[str],
) -> pd.DataFrame:
    """
    Applique l'harmonisation directement sur un DataFrame Pandas.

    Seules les colonnes présentes dans target_markers (après mapping) sont
    conservées dans l'ordre défini par target_markers.  Les colonnes
    techniques non-marqueurs (FSC, SSC, Time…) sont ignorées.

    Args:
        df:             DataFrame dont les colonnes correspondent aux $PnS du FCS.
        target_markers: Liste canonique des marqueurs à conserver.

    Returns:
        Nouveau DataFrame avec les colonnes renommées et réordonnées selon
        target_markers.  Les colonnes absentes du panel sont remplies de NaN.
    """
    rename_map = harmonize_marker_names(list(df.columns), target_markers)

    # Renommer les colonnes
    df_renamed = df.rename(columns=rename_map)

    # Sélectionner et réordonner selon target_markers (NaN si absent)
    available = [m for m in target_markers if m in df_renamed.columns]
    missing = [m for m in target_markers if m not in df_renamed.columns]

    if missing:
        _logger.warning(
            "Marqueurs cibles absents du fichier : %s",
            ", ".join(missing),
        )

    return df_renamed[available]


# ---------------------------------------------------------------------------
# Exemple d'utilisation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Marqueurs canoniques attendus dans la matrice NBM
    TARGET = ["CD45", "CD34", "CD13", "CD3", "CD19", "CD117"]

    # Simulation de colonnes $PnS lues depuis différents exports FCS
    raw_cols_file_A = [
        "FSC-A",
        "SSC-A",
        "CD45 KO",
        "CD34 Cy55",
        "CD13 BV421",
        "CD3 FITC",
        "CD19 PE",
        "CD117 APC",
        "Time",
    ]
    raw_cols_file_B = [
        "FSC-A",
        "SSC-A",
        "CD45",
        "CD34",
        "CD13",
        "CD3",
        "CD19",
        "CD117",
        "Time",
    ]

    print("\n--- Fichier A (avec fluorochromes) ---")
    mapping_A = harmonize_marker_names(raw_cols_file_A, TARGET)
    print("Mapping :", mapping_A)

    print("\n--- Fichier B (noms courts) ---")
    mapping_B = harmonize_marker_names(raw_cols_file_B, TARGET)
    print("Mapping :", mapping_B)

    # Application sur un DataFrame fictif
    import numpy as np

    rng = np.random.default_rng(42)
    df_A = pd.DataFrame(
        rng.random((5, len(raw_cols_file_A))),
        columns=raw_cols_file_A,
    )

    print("\n--- DataFrame avant harmonisation ---")
    print(df_A.columns.tolist())

    df_harmonized = apply_harmonization(df_A, TARGET)

    print("\n--- DataFrame après harmonisation ---")
    print(df_harmonized.columns.tolist())
    print(df_harmonized.head())
