"""
population_mapping.py â€” Assignation de populations aux nÅ“uds FlowSOM.

Sections implÃ©mentÃ©es (fidÃ¨le au monolithe flowsom_pipeline.py) :
  Â§10.1b  apply_cyto_transform_matrix, _logicle_matrix_pytometry
  Â§10.2   extract_node_centroids_from_fcs
  Â§10.3   normalize_col_name, filter_area_columns, build_direct_mapping_a_only,
          load_population_csv_transformed
  Â§10.3b  _welford_update, compute_pop_stats_from_csv
  Â§10.4   normalize_matrix, map_populations_to_nodes_v3
  Â§10.4b  M1â€“M12 complets, map_populations_to_nodes_v5,
          _apply_bayesian_prior, _mahalanobis_distance_batch,
          _proportional_stratified_pool, _knn_vote,
          compute_unknown_threshold, assign_with_auto_threshold,
          _otsu_threshold_1d, _mad_threshold_1d

RÃ¨gles critiques :
  - Seules les colonnes -A (Area) entrent dans le mapping, jamais -H.
  - La transformation est appliquÃ©e AVANT le calcul des centroÃ¯des :
    mean(transform(x)) â‰  transform(mean(x)).
  - Le cache Parquet est indexÃ© par (pop_name + _transform_tag) pour
    Ã©viter tout mÃ©lange entre espaces de transformation.
  - Les IDs de nÅ“uds dans le FCS final sont 1-indexÃ©s : soustraire 1.
  - M12 (cosine + log10Â³ prior + hard limit) est la mÃ©thode recommandÃ©e ELN 2022.
"""

from __future__ import annotations

import hashlib
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.spatial.distance import cdist, cosine as cosine_dist

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    warnings.warn("scipy requis pour map_populations_to_nodes: pip install scipy")

try:
    from sklearn.preprocessing import RobustScaler

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from sklearn.neighbors import KNeighborsClassifier

    _KNN_AVAILABLE = True
except ImportError:
    _KNN_AVAILABLE = False

try:
    import pytometry  # type: ignore

    _PYTOMETRY_AVAILABLE = True
except ImportError:
    _PYTOMETRY_AVAILABLE = False

from src.utils.logger import get_logger

_logger = get_logger("analysis.population_mapping")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Constantes
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Canaux Ã  exclure absolument du mapping (ne sont jamais des marqueurs -A utiles)
_ALWAYS_EXCLUDE: frozenset = frozenset(
    {
        "FSC-H",
        "SSC-H",
        "FSC-Width",
        "TIME",
        "Time",
        "Event_length",
        "Event_count",
        "Viability",
        "LIVE_DEAD",
        "Ghost",
    }
)
# Colonnes mÃ©ta FlowSOM jamais Ã  passer dans le mapping
_FLOWSOM_META_COLS: frozenset = frozenset(
    {
        "FlowSOM_cluster",
        "FlowSOM_metacluster",
        "xGrid",
        "yGrid",
        "xNodes",
        "yNodes",
        "size",
        "Condition",
        "Condition_Num",
        "File_Origin",
    }
)
# PrÃ©fixes scatter (FSC/SSC) et time
_SCATTER_PREFIXES = ("FSC", "SSC", "TIME", "WIDTH", "AREA", "HEIGHT")
_NON_A_SUFFIXES_TO_EXCLUDE = ("-H", "-W", "-Width", "-Height")

# Couleurs standard population cytomÃ©trique
POPULATION_COLORS: Dict[str, str] = {
    "Granulo": "#e26f1a",
    "Granulocytes": "#e26f1a",
    "HÃ©matogone 34+": "#9467bd",
    "Hematogones19+": "#2ca02c",
    "Ly T_NK": "#17becf",
    "Lymphos B": "#1f77b4",
    "Lymphos": "#aec7e8",
    "Plasmo": "#d62728",
    "Unknown": "#7f7f7f",
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Â§10.3 â€” Utilitaires de normalisation des noms de colonnes
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def normalize_col_name(name: str) -> str:
    """
    Normalise un nom de colonne pour la comparaison CSV â†” FCS.

    OpÃ©rations : minuscules, suppression des accents (NFDâ†’ASCII),
    compression des espaces, conservation des traits d'union.
    Exemples : "CD45-PerCP-Cy5.5-A" â†’ "cd45-percp-cy5.5-a"
              "Granulocytes " â†’ "granulocytes"

    Args:
        name: Nom de colonne brut.

    Returns:
        Nom normalisÃ©.
    """
    # DÃ©composer les accents (NFD) puis filtrer uniquement ASCII
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_str.strip().lower().replace("  ", " ").replace(" ", " ")


def filter_area_columns(columns: List[str]) -> List[str]:
    """
    Filtre les colonnes pour ne garder que les canaux -A (fluorescence Area)
    et exclure FSC-H, SSC-H, FSC-Width, TIME, EVENT_COUNT, et mÃ©ta FlowSOM.

    Le mapping NE DOIT OPÃ‰RER QUE sur les canaux -A pour que les profils
    soient comparables entre nÅ“uds SOM et CSV de rÃ©fÃ©rence.

    Args:
        columns: Liste de noms de colonnes.

    Returns:
        Colonnes filtrÃ©es.
    """
    _all_exclude = _ALWAYS_EXCLUDE | _FLOWSOM_META_COLS
    filtered: List[str] = []
    for c in columns:
        if c in _all_exclude:
            continue
        cu = c.upper()
        # Exclure colonnes mÃ©ta FlowSOM (insensible casse)
        if any(cu == ex.upper() for ex in _FLOWSOM_META_COLS):
            continue
        # Exclure les suffixes non -A quand le -A correspondant existe
        if any(cu.endswith(s.upper()) for s in _NON_A_SUFFIXES_TO_EXCLUDE):
            prefix = None
            for s in _NON_A_SUFFIXES_TO_EXCLUDE:
                if cu.endswith(s.upper()):
                    prefix = cu[: -len(s)]
                    break
            if prefix and any(col.upper() == prefix + "-A" for col in columns):
                continue
        filtered.append(c)
    return filtered


def build_direct_mapping_a_only(
    csv_cols_a: List[str],
    fcs_cols_a: List[str],
    verbose: bool = False,
) -> Dict[str, str]:
    """
    Construit une correspondance 1:1 entre colonnes CSV et colonnes FCS,
    en comparant les noms normalisÃ©s (minuscule, sans accents).

    Seules les colonnes -A sont autorisÃ©es (au sens de filter_area_columns).

    Args:
        csv_cols_a: Colonnes CSV de rÃ©fÃ©rence (dÃ©jÃ  filtrÃ©es -A).
        fcs_cols_a: Colonnes du FCS final (dÃ©jÃ  filtrÃ©es -A).
        verbose: Afficher les correspondances trouvÃ©es/manquantes.

    Returns:
        Dict {csv_col: fcs_col} pour toutes les correspondances trouvÃ©es.
    """
    norm_fcs = {normalize_col_name(c): c for c in fcs_cols_a}
    mapping: Dict[str, str] = {}

    for csv_col in csv_cols_a:
        norm = normalize_col_name(csv_col)
        if norm in norm_fcs:
            mapping[csv_col] = norm_fcs[norm]
            if verbose:
                _logger.debug("  CSV '%s' â†’ FCS '%s'", csv_col, norm_fcs[norm])
        else:
            if verbose:
                _logger.debug(
                    "  [!] CSV '%s' (norm='%s') sans correspondance FCS", csv_col, norm
                )

    return mapping


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Â§10.1b â€” Transformation cytomÃ©trique (appliquÃ©e aux CSV de rÃ©fÃ©rence)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _logicle_matrix_pytometry(
    X: np.ndarray,
    col_names: List[str],
) -> np.ndarray:
    """
    Applique la transformation Logicle via pytometry sur une matrice numpy.

    CrÃ©e un AnnData temporaire, appelle pytometry.pp.normalization,
    puis retourne la matrice transformÃ©e. Fallback sur arcsinh(x/5)
    si pytometry n'est pas disponible.

    Args:
        X: Matrice (n_cells, n_channels) en valeurs brutes.
        col_names: Noms des canaux (colonnes de X).

    Returns:
        Matrice transformÃ©e (mÃªme shape).
    """
    if not _PYTOMETRY_AVAILABLE:
        _logger.debug("pytometry absent â€” fallback arcsinh(x/5) pour Logicle")
        return np.arcsinh(X / 5.0)

    try:
        import anndata as ad  # type: ignore

        adata = ad.AnnData(X=X.astype(np.float32), var=pd.DataFrame(index=col_names))
        pytometry.pp.normalization(adata)
        return adata.X.astype(np.float64)
    except Exception as exc:
        _logger.warning(
            "Logicle pytometry Ã©chouÃ© (%s) â€” fallback arcsinh(x/5)", exc
        )
        return np.arcsinh(X / 5.0)


def apply_cyto_transform_matrix(
    X: np.ndarray,
    col_names: List[str],
    apply_to_scatter: bool = False,
    transform_type: str = "arcsinh",
    cofactor: float = 5.0,
) -> np.ndarray:
    """
    Applique la transformation cytomÃ©trique colonne par colonne sur une matrice.

    Rationnel : la transformation doit Ãªtre appliquÃ©e CELLULE PAR CELLULE
    avant le calcul des centroÃ¯des :
      mean(transform(x)) â‰  transform(mean(x))

    Les canaux FSC-A et SSC-A sont traitÃ©s en linÃ©aire par dÃ©faut
    (apply_to_scatter=False, standard ELN cytomÃ©trie).

    Args:
        X: Matrice (n_cells, n_channels) brute.
        col_names: Noms des canaux.
        apply_to_scatter: Appliquer la transformation aux FSC-A/SSC-A.
        transform_type: "arcsinh" | "logicle" | "log10" | "none".
        cofactor: Cofacteur pour arcsinh (dÃ©faut 5.0).

    Returns:
        Matrice transformÃ©e (float64, mÃªme shape).
    """
    X_out = X.astype(np.float64)

    if transform_type == "none":
        return X_out

    _scatter_cols = frozenset(
        i
        for i, c in enumerate(col_names)
        if any(c.upper().startswith(p) for p in ("FSC", "SSC", "TIME"))
    )

    for j, col in enumerate(col_names):
        if j in _scatter_cols and not apply_to_scatter:
            continue  # conserver en linÃ©aire

        col_data = X_out[:, j]

        if transform_type == "arcsinh":
            X_out[:, j] = np.arcsinh(col_data / cofactor)
        elif transform_type == "logicle":
            # La logicle nÃ©cessite t.p. pytometry; appliquÃ© colonne entiÃ¨re ici
            X_out[:, j] = _logicle_matrix_pytometry(
                col_data.reshape(-1, 1), [col]
            ).ravel()
        elif transform_type == "log10":
            X_out[:, j] = np.log10(np.maximum(col_data, 1.0))

    return X_out


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Â§10.2 â€” Extraction des centroÃ¯des SOM depuis le FCS final exportÃ©
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def extract_node_centroids_from_fcs(
    fcs_path: Path,
    transform_type: str = "arcsinh",
    cofactor: float = 5.0,
    apply_to_scatter: bool = False,
    col_cluster: str = "FlowSOM_cluster",
    col_xgrid: str = "xGrid",
    col_ygrid: str = "yGrid",
    col_xnodes: Optional[str] = "xNodes",
    col_ynodes: Optional[str] = "yNodes",
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Lit le FCS final exportÃ© par le pipeline et calcule, pour chaque nÅ“ud SOM,
    le centroÃ¯de (MFI moyen) dans l'espace transformÃ© sur les canaux -A.

    OpÃ©rations :
      1. Lecture via flowsom.io.read_FCS (fallback flowkit).
      2. Filtre colonnes -A uniquement (filter_area_columns).
      3. Applique apply_cyto_transform_matrix sur chaque cellule.
      4. Calcule np.add.at pour la moyenne par nÅ“ud.

    IDs de nÅ“uds dans le FCS : 1-indexÃ©s â†’ soustraire 1 pour 0-indexÃ©.

    Args:
        fcs_path: Chemin du FCS avec colonnes FlowSOM_cluster, xGrid, yGrid, etc.
        transform_type: "arcsinh" | "logicle" | "log10" | "none".
        cofactor: Cofacteur arcsinh.
        apply_to_scatter: Transformer FSC-A/SSC-A.
        col_cluster: Nom de la colonne nÅ“ud SOM (1-indexÃ©e dans le FCS).
        col_xgrid: Colonne xGrid.
        col_ygrid: Colonne yGrid.
        col_xnodes: Colonne xNodes (layout MST, peut Ãªtre None).
        col_ynodes: Colonne yNodes (layout MST, peut Ãªtre None).

    Returns:
        Tuple de 5 Ã©lÃ©ments:
          - node_mfi_df   : DataFrame (n_nodes, n_markers_A) â€” centroÃ¯des transformÃ©s
          - node_coords_df: DataFrame (n_nodes, [xGrid,yGrid,xNodes,yNodes])
          - node_counts   : np.ndarray (n_nodes,) â€” nb cellules par nÅ“ud
          - mc_per_node   : np.ndarray (n_nodes,) â€” mÃ©tacluster dominant par nÅ“ud
          - node_ids_raw  : np.ndarray (n_nodes,) â€” array des indices 0-based
    """
    # â”€â”€ 1. Lecture du FCS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    df_fcs: Optional[pd.DataFrame] = None

    try:
        import flowsom as fs  # type: ignore

        _ff = fs.io.read_FCS(str(fcs_path))
        df_fcs = pd.DataFrame(_ff.events, columns=_ff.channels)
        _logger.info("FCS lu via flowsom.io: %d cellules, %d canaux", *df_fcs.shape)
    except Exception as exc:
        _logger.debug("flowsom.io.read_FCS Ã©chouÃ© (%s), tentative flowkit", exc)

    if df_fcs is None:
        try:
            import flowkit as fk  # type: ignore

            sample = fk.Sample(str(fcs_path))
            df_fcs = sample.as_dataframe(source="raw")
            # flowkit retourne un MultiIndex (channel_id, label) -> aplatir en strings
            if len(df_fcs.columns) > 0 and isinstance(df_fcs.columns[0], tuple):
                df_fcs.columns = [c[0] for c in df_fcs.columns]
            _logger.info("FCS lu via flowkit: %d cellules, %d canaux", *df_fcs.shape)
        except Exception as exc2:
            raise RuntimeError(f"Impossible de lire {fcs_path}: {exc2}") from exc2

    all_cols = list(df_fcs.columns)

    # â”€â”€ 2. VÃ©rification de la colonne nÅ“ud â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # fcswrite supprime les underscores : FlowSOM_cluster -> FlowSOMcluster
    if col_cluster not in all_cols:
        alt = col_cluster.replace("_", "")
        if alt in all_cols:
            col_cluster = alt
        else:
            raise ValueError(
                f"Colonne '{col_cluster}' absente du FCS. "
                f"Colonnes disponibles: {all_cols[:20]}..."
            )

    # IDs nÅ“uds (1-indexÃ©s dans le FCS â†’ 0-indexÃ©s ici)
    node_ids_1indexed = df_fcs[col_cluster].to_numpy(dtype=np.int32)
    n_nodes = int(node_ids_1indexed.max())
    node_ids_0indexed = node_ids_1indexed - 1  # 0-based

    # â”€â”€ 3. Colonnes de coordonnÃ©es â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    coord_cols_available = [
        c
        for c in [col_xgrid, col_ygrid, col_xnodes, col_ynodes]
        if c is not None and c in all_cols
    ]
    # Extraire les coordonnÃ©es par nÅ“ud (premiÃ¨re occurrence suffit â€” constante par nÅ“ud)
    node_coords_dict: Dict[str, np.ndarray] = {}
    for cc in coord_cols_available:
        vals = np.full(n_nodes, np.nan, dtype=np.float64)
        # Utilise np.add.at + count pour la moyenne (robuste aux multi-cellules)
        np.add.at(vals, node_ids_0indexed, df_fcs[cc].to_numpy(dtype=np.float64))
        counts_tmp = np.bincount(node_ids_0indexed, minlength=n_nodes)
        counts_tmp = np.where(counts_tmp == 0, 1, counts_tmp)
        node_coords_dict[cc] = vals / counts_tmp

    node_coords_df = pd.DataFrame(node_coords_dict, index=np.arange(n_nodes))

    # â”€â”€ 4. Extraction de la colonne FlowSOM_metacluster (si prÃ©sente) â”€â”€â”€â”€â”€â”€â”€â”€â”€
    mc_col = "FlowSOM_metacluster"
    if mc_col in all_cols:
        mc_vals = np.zeros(n_nodes, dtype=np.int32)
        np.add.at(mc_vals, node_ids_0indexed, df_fcs[mc_col].to_numpy(dtype=np.int32))
        mc_counts = np.bincount(node_ids_0indexed, minlength=n_nodes)
        mc_counts = np.where(mc_counts == 0, 1, mc_counts)
        mc_per_node = (mc_vals / mc_counts).round().astype(np.int32)
    else:
        mc_per_node = np.zeros(n_nodes, dtype=np.int32)

    # â”€â”€ 5. SÃ©lection stricte colonnes -A pour le mapping â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _meta_cols = set(coord_cols_available) | {col_cluster, mc_col}
    candidate_cols = [c for c in all_cols if c not in _meta_cols]
    marker_cols_a = filter_area_columns(candidate_cols)

    if not marker_cols_a:
        _logger.warning("Aucune colonne -A dans le FCS : vÃ©rifier les noms de canaux.")
        marker_cols_a = candidate_cols[:10]

    _logger.info("  Canaux -A retenus pour le mapping: %d", len(marker_cols_a))

    # â”€â”€ 6. Transformation + calcul des centroÃ¯des â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    X_raw = df_fcs[marker_cols_a].to_numpy(dtype=np.float64)
    X_transformed = apply_cyto_transform_matrix(
        X_raw,
        col_names=marker_cols_a,
        apply_to_scatter=apply_to_scatter,
        transform_type=transform_type,
        cofactor=cofactor,
    )

    # Accumulation par nÅ“ud (mean(transform(x)))
    node_mfi_acc = np.zeros((n_nodes, len(marker_cols_a)), dtype=np.float64)
    np.add.at(node_mfi_acc, node_ids_0indexed, X_transformed)
    node_counts = np.bincount(node_ids_0indexed, minlength=n_nodes)
    node_counts_safe = np.where(node_counts == 0, 1, node_counts)
    node_mfi = node_mfi_acc / node_counts_safe[:, np.newaxis]

    node_mfi_df = pd.DataFrame(
        node_mfi, columns=marker_cols_a, index=np.arange(n_nodes)
    )

    _logger.info(
        "CentroÃ¯des SOM calculÃ©s: %d nÅ“uds Ã— %d marqueurs "
        "(transform=%s, apply_scatter=%s)",
        n_nodes,
        len(marker_cols_a),
        transform_type,
        apply_to_scatter,
    )

    return (
        node_mfi_df,
        node_coords_df,
        node_counts.astype(np.float32),
        mc_per_node,
        np.arange(n_nodes),
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Â§10.3 â€” Chargement CSV de rÃ©fÃ©rence MFI + cache Parquet
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _parquet_cache_path(
    csv_path: Path,
    cache_dir: Path,
    transform_tag: str,
) -> Path:
    """Construit le chemin du fichier Parquet de cache."""
    key = hashlib.md5(f"{csv_path.stem}|{transform_tag}".encode()).hexdigest()[:12]
    return cache_dir / f"ref_{csv_path.stem}_{transform_tag[:24]}_{key}.parquet"


def load_population_csv_transformed(
    csv_path: Path,
    cache_dir: Optional[Path],
    area_cols_to_keep: List[str],
    transform_type: str = "arcsinh",
    cofactor: float = 5.0,
    apply_to_scatter: bool = False,
    force_reload: bool = False,
    chunk_size: int = 50_000,
    sep: str = ";",
    decimal: str = ",",
) -> pd.Series:
    """
    Charge un CSV de rÃ©fÃ©rence MFI, applique la transformation cytomÃ©trique
    par chunks, et retourne la MFI moyenne dans l'espace transformÃ©.

    Cache Parquet : la serie transformÃ©e est sauvegardÃ©e avec une clÃ© unique
    (stem_csv + transform_tag) pour Ã©viter tout recalcul inutile.

    Args:
        csv_path: Chemin du fichier CSV de rÃ©fÃ©rence (ex: Granulocytes.csv).
        cache_dir: RÃ©pertoire Parquet (None = pas de cache).
        area_cols_to_keep: Colonnes Ã  conserver aprÃ¨s lecture (FCS-aligned -A).
        transform_type: "arcsinh" | "logicle" | "none".
        cofactor: Cofacteur arcsinh.
        apply_to_scatter: Transformer FSC-A/SSC-A.
        force_reload: Ignorer le cache existant.
        chunk_size: Taille des chunks de lecture CSV.
        sep: SÃ©parateur CSV (dÃ©faut ";").
        decimal: SÃ©parateur dÃ©cimal CSV (dÃ©faut ",").

    Returns:
        pd.Series index=area_cols_to_keep, values=MFI moyenne transformÃ©e.
    """
    transform_tag = (
        f"{transform_type}_cof{cofactor}"
        f"{'_scatter' if apply_to_scatter else '_noscatter'}"
    )

    # â”€â”€ Cache Parquet â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if cache_dir is not None and not force_reload:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        pq_path = _parquet_cache_path(csv_path, cache_dir, transform_tag)

        if pq_path.exists():
            df_cached = pd.read_parquet(pq_path)
            # Aligner sur les colonnes demandÃ©es
            cols_ok = [c for c in area_cols_to_keep if c in df_cached.columns]
            if cols_ok:
                _logger.debug("Cache Parquet: %s", pq_path.name)
                return df_cached.iloc[0][cols_ok].rename(csv_path.stem)

    # â”€â”€ Lecture du CSV par chunks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    total_cells = 0
    mfi_acc = np.zeros(len(area_cols_to_keep), dtype=np.float64)

    try:
        for chunk in pd.read_csv(
            csv_path,
            sep=sep,
            decimal=decimal,
            chunksize=chunk_size,
            low_memory=False,
            encoding="utf-8-sig",
        ):
            # Normaliser les noms de colonnes CSV et trouver ceux qui matchent
            norm_chunk = {normalize_col_name(c): c for c in chunk.columns}
            cols_csv_in_chunk: List[str] = []
            reindex_from_chunk: List[str] = []
            for fcs_col in area_cols_to_keep:
                nc = normalize_col_name(fcs_col)
                if nc in norm_chunk:
                    cols_csv_in_chunk.append(fcs_col)
                    reindex_from_chunk.append(norm_chunk[nc])

            if not reindex_from_chunk:
                continue

            X_chunk = chunk[reindex_from_chunk].to_numpy(dtype=np.float64)
            # Supprimer les NaN avant transformation (ne doivent pas exister)
            nan_mask = np.isnan(X_chunk).any(axis=1)
            if nan_mask.any():
                X_chunk = X_chunk[~nan_mask]

            if len(X_chunk) == 0:
                continue

            X_tr = apply_cyto_transform_matrix(
                X_chunk,
                col_names=reindex_from_chunk,
                apply_to_scatter=apply_to_scatter,
                transform_type=transform_type,
                cofactor=cofactor,
            )

            # Accumuler uniquement sur les colonnes qui matchent
            for k, fcs_col in enumerate(cols_csv_in_chunk):
                idx = area_cols_to_keep.index(fcs_col)
                chunk_col_idx = reindex_from_chunk.index(
                    [
                        norm_chunk[normalize_col_name(fcs_col)]
                        for _ in [None]
                        if normalize_col_name(fcs_col) in norm_chunk
                    ][0]
                )
                mfi_acc[idx] += X_tr[:, chunk_col_idx].sum()

            total_cells += len(X_chunk)

    except UnicodeDecodeError:
        # Fallback encodage latin-1
        for chunk in pd.read_csv(
            csv_path,
            sep=sep,
            decimal=decimal,
            chunksize=chunk_size,
            low_memory=False,
            encoding="latin-1",
        ):
            norm_chunk = {normalize_col_name(c): c for c in chunk.columns}
            cols_csv_in_chunk = []
            reindex_from_chunk = []
            for fcs_col in area_cols_to_keep:
                nc = normalize_col_name(fcs_col)
                if nc in norm_chunk:
                    cols_csv_in_chunk.append(fcs_col)
                    reindex_from_chunk.append(norm_chunk[nc])
            if not reindex_from_chunk:
                continue
            X_chunk = chunk[reindex_from_chunk].dropna().to_numpy(dtype=np.float64)
            if len(X_chunk) == 0:
                continue
            X_tr = apply_cyto_transform_matrix(
                X_chunk,
                col_names=reindex_from_chunk,
                apply_to_scatter=apply_to_scatter,
                transform_type=transform_type,
                cofactor=cofactor,
            )
            for k, fcs_col in enumerate(cols_csv_in_chunk):
                idx = area_cols_to_keep.index(fcs_col)
                ci = reindex_from_chunk.index(norm_chunk[normalize_col_name(fcs_col)])
                mfi_acc[idx] += X_tr[:, ci].sum()
            total_cells += len(X_chunk)

    if total_cells == 0:
        _logger.warning("[!] CSV vide ou aucune colonne commune: %s", csv_path.name)
        return pd.Series(
            np.zeros(len(area_cols_to_keep)),
            index=area_cols_to_keep,
            name=csv_path.stem,
        )

    mfi_mean = mfi_acc / total_cells
    result = pd.Series(mfi_mean, index=area_cols_to_keep, name=csv_path.stem)
    _logger.info(
        "CSV '%s' chargÃ©: %d cellules, transform=%s",
        csv_path.stem,
        total_cells,
        transform_type,
    )

    # â”€â”€ Sauvegarder le cache Parquet â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if cache_dir is not None:
        try:
            df_save = result.to_frame().T
            df_save.to_parquet(pq_path, engine="pyarrow")
        except Exception as exc:
            _logger.debug("Cache Parquet non sauvegardÃ©: %s", exc)

    return result


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Â§10.3b â€” Statistiques biologiques de rÃ©fÃ©rence (Welford + rÃ©servoir KNN)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _welford_update(
    n: int,
    mean: np.ndarray,
    M2: np.ndarray,
    batch: np.ndarray,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Mise Ã  jour incrÃ©mentale de la moyenne et de la somme des carrÃ©s Ã©cartÃ©s
    (algorithme de Welford â€” one-pass, numÃ©riquement stable).

    Args:
        n: Nombre de points dÃ©jÃ  traitÃ©s.
        mean: Vecteur moyenne courant (n_features,).
        M2: Vecteur somme des carrÃ©s Ã©cartÃ©s M2[j] = Î£(x_j - Î¼_j)Â² courant.
        batch: Nouveau batch (n_batch, n_features).

    Returns:
        (n_new, mean_new, M2_new)
    """
    for x in batch:
        n += 1
        delta = x - mean
        mean = mean + delta / n
        delta2 = x - mean
        M2 = M2 + delta * delta2
    return n, mean, M2


def compute_pop_stats_from_csv(
    csv_path: Path,
    cols: List[str],
    transform_type: str = "arcsinh",
    cofactor: float = 5.0,
    apply_to_scatter: bool = False,
    knn_sample_size: int = 2_000,
    chunk_size: int = 50_000,
    reg_alpha: float = 1e-4,
    sep: str = ";",
    decimal: str = ",",
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Calcule les statistiques biologiques de rÃ©fÃ©rence pour une population :
      - Covariance Welford (one-pass, memory-efficient pour >500k cellules).
      - Reservoir sampling pour l'Ã©chantillon KNN.

    Utilisation :
      - `n_cells` â†’ prior bayÃ©sien (populations frÃ©quentes vs rares).
      - `cov` â†’ distance de Mahalanobis (M10).
      - `knn_sample` â†’ vote KNN densitÃ© (M11).

    Args:
        csv_path: Chemin du CSV de rÃ©fÃ©rence.
        cols: Colonnes Ã  utiliser (FCS-aligned, -A uniquement).
        transform_type: Transformation cytomÃ©trique.
        cofactor: Cofacteur arcsinh.
        apply_to_scatter: Transformer FSC-A/SSC-A.
        knn_sample_size: Taille de l'Ã©chantillon rÃ©servoir pour KNN.
        chunk_size: Taille des chunks de lecture.
        reg_alpha: RÃ©gularisation de la covariance (Ã©vite matrice singuliÃ¨re).
        sep: SÃ©parateur CSV.
        decimal: SÃ©parateur dÃ©cimal.

    Returns:
        Tuple (n_cells, covariance_matrix, knn_sample_array).
          - n_cells: int â€” nombre total de cellules dans le CSV.
          - covariance_matrix: np.ndarray (n_cols, n_cols) rÃ©gularisÃ©e.
          - knn_sample: np.ndarray (min(knn_sample_size, n_cells), n_cols).
    """
    n_cols = len(cols)
    n = 0
    mean = np.zeros(n_cols, dtype=np.float64)
    M2 = np.zeros(n_cols, dtype=np.float64)

    # RÃ©servoir de taille knn_sample_size
    reservoir: np.ndarray = np.empty((knn_sample_size, n_cols), dtype=np.float64)
    reservoir_count = 0

    rng = np.random.default_rng(42)

    try:
        reader = pd.read_csv(
            csv_path,
            sep=sep,
            decimal=decimal,
            chunksize=chunk_size,
            low_memory=False,
            encoding="utf-8-sig",
        )
    except Exception:
        reader = pd.read_csv(
            csv_path,
            sep=sep,
            decimal=decimal,
            chunksize=chunk_size,
            low_memory=False,
            encoding="latin-1",
        )

    norm_col_map = None

    for chunk in reader:
        if norm_col_map is None:
            nc_chunk = {normalize_col_name(c): c for c in chunk.columns}
            col_map_idx: List[int] = []
            actual_csv_cols: List[str] = []
            for i, c in enumerate(cols):
                nc = normalize_col_name(c)
                if nc in nc_chunk:
                    col_map_idx.append(i)
                    actual_csv_cols.append(nc_chunk[nc])
            norm_col_map = (col_map_idx, actual_csv_cols)

        col_map_idx, actual_csv_cols = norm_col_map
        if not actual_csv_cols:
            continue

        sub = chunk[actual_csv_cols].dropna()
        if len(sub) == 0:
            continue

        X_chunk = sub.to_numpy(dtype=np.float64)
        X_tr = apply_cyto_transform_matrix(
            X_chunk,
            col_names=actual_csv_cols,
            apply_to_scatter=apply_to_scatter,
            transform_type=transform_type,
            cofactor=cofactor,
        )

        # RÃ©aligner sur le vecteur complet (n_cols)
        X_full = np.zeros((len(X_tr), n_cols), dtype=np.float64)
        for local_i, global_i in enumerate(col_map_idx):
            X_full[:, global_i] = X_tr[:, local_i]

        # Mise Ã  jour Welford
        n, mean, M2 = _welford_update(n, mean, M2, X_full)

        # Mise Ã  jour rÃ©servoir (reservoir sampling Algorithm R)
        for row in X_full:
            if reservoir_count < knn_sample_size:
                reservoir[reservoir_count] = row
                reservoir_count += 1
            else:
                j = int(rng.integers(0, reservoir_count + 1))
                if j < knn_sample_size:
                    reservoir[j] = row

    if n == 0:
        _logger.warning("[!] CSV vide pour compute_pop_stats: %s", csv_path.name)
        cov = np.eye(n_cols) * reg_alpha
        return 0, cov, np.empty((0, n_cols))

    variance = M2 / max(n - 1, 1)
    # Covariance approximÃ©e (diagonale Welford)
    cov = np.diag(variance)
    # RÃ©gularisation ridge pour Ã©viter la singularitÃ©
    cov += np.eye(n_cols) * reg_alpha * max(variance.max(), 1.0)

    knn_sample = reservoir[:reservoir_count].copy()
    _logger.info(
        "Stats '%s': n=%d | cov_diag_mean=%.3f | knn_sample=%d",
        csv_path.stem,
        n,
        float(variance.mean()),
        reservoir_count,
    )
    return n, cov, knn_sample


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Â§10.4 â€” Normalisation + V3 (Euclidean)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def normalize_matrix(
    X: np.ndarray,
    method: str = "range",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalise une matrice de MFI par colonne.

    Args:
        X: Matrice (n_nodes, n_markers).
        method: "range" (min-max), "zscore" (mean/std), ou "none".

    Returns:
        Tuple (X_normalized, scale_params).
    """
    if method == "none":
        return X.copy(), {}

    if method == "range":
        col_min = X.min(axis=0)
        col_max = X.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1.0
        X_norm = (X - col_min) / col_range
        return X_norm, {"min": col_min, "max": col_max, "range": col_range}

    if method == "zscore":
        col_mean = X.mean(axis=0)
        col_std = X.std(axis=0)
        col_std[col_std == 0] = 1.0
        X_norm = (X - col_mean) / col_std
        return X_norm, {"mean": col_mean, "std": col_std}

    raise ValueError(
        f"MÃ©thode de normalisation inconnue: {method!r}. "
        "Utiliser 'range', 'zscore' ou 'none'."
    )


def map_populations_to_nodes_v3(
    node_mfi_raw: pd.DataFrame,
    pop_mfi_ref: pd.DataFrame,
    include_scatter: bool = True,
    distance_percentile: int = 60,
    normalization_method: str = "range",
) -> pd.DataFrame:
    """
    Assigne une population Ã  chaque nÅ“ud SOM par distance euclidienne minimale (M1).

    OpÃ¨re dans l'espace MFI normalisÃ©. Le seuil "Unknown" est le
    distance_percentile de la distribution des distances minimales.

    Args:
        node_mfi_raw: DataFrame [n_nodes Ã— n_markers] â€” MFI par nÅ“ud SOM.
        pop_mfi_ref: DataFrame [n_populations Ã— n_markers] â€” MFI de rÃ©fÃ©rence.
        include_scatter: Inclure FSC-A/SSC-A dans le calcul de distance.
        distance_percentile: Percentile du seuil Unknown.
        normalization_method: "range", "zscore", ou "none".

    Returns:
        DataFrame: [node_id, best_pop, best_dist, threshold, assigned_pop].
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy requis: pip install scipy")

    cols_node = set(node_mfi_raw.columns)
    cols_ref = set(pop_mfi_ref.columns)
    common = sorted(cols_node & cols_ref)
    if not common:
        raise ValueError(
            "Aucune colonne commune node_mfi_raw / pop_mfi_ref. Aligner en amont."
        )

    if include_scatter:
        cols_for_dist = common
    else:
        cols_for_dist = [
            c
            for c in common
            if not any(c.upper().startswith(p) for p in _SCATTER_PREFIXES)
        ]

    _logger.info(
        "V3 mapping: %d nÅ“uds Ã— %d pops Ã— %d marqueurs",
        len(node_mfi_raw),
        len(pop_mfi_ref),
        len(cols_for_dist),
    )

    X_nodes = node_mfi_raw[cols_for_dist].to_numpy(dtype=np.float64)
    X_pops = pop_mfi_ref[cols_for_dist].to_numpy(dtype=np.float64)

    if normalization_method != "none":
        X_nodes_norm, scale = normalize_matrix(X_nodes, method=normalization_method)
        if normalization_method == "range":
            X_pops_norm = np.clip((X_pops - scale["min"]) / scale["range"], 0.0, None)
        elif normalization_method == "zscore":
            X_pops_norm = (X_pops - scale["mean"]) / scale["std"]
        else:
            X_pops_norm = X_pops
    else:
        X_nodes_norm = X_nodes
        X_pops_norm = X_pops

    dist_matrix = cdist(X_nodes_norm, X_pops_norm, metric="euclidean")
    best_idx = np.argmin(dist_matrix, axis=1)
    best_dist = dist_matrix[np.arange(len(X_nodes_norm)), best_idx]
    threshold = float(np.percentile(best_dist, distance_percentile))

    pop_names = list(pop_mfi_ref.index)
    assigned = np.where(
        best_dist <= threshold,
        np.array([pop_names[i] for i in best_idx]),
        "Unknown",
    )

    result = pd.DataFrame(
        {
            "node_id": np.arange(len(X_nodes_norm)),
            "best_pop": [pop_names[i] for i in best_idx],
            "best_dist": np.round(best_dist, 4),
            "threshold": round(threshold, 4),
            "assigned_pop": assigned,
        }
    )

    n_tot = len(result)
    n_unk = int((result["assigned_pop"] == "Unknown").sum())
    _logger.info(
        "V3 terminÃ©: %d assignÃ©s (%d%%), %d Unknown (%d%%)",
        n_tot - n_unk,
        round(100 * (n_tot - n_unk) / n_tot),
        n_unk,
        round(100 * n_unk / n_tot),
    )
    return result


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Â§10.4b â€” Fonctions utilitaires pour le benchmark M1â€“M12
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def arcsinh_transform(
    X: np.ndarray,
    cofactor: float = 5.0,
) -> np.ndarray:
    """arcsinh(X / cofactor) â€” appliquÃ© sur toute la matrice."""
    return np.arcsinh(X / cofactor)


def robust_scale(
    X_nodes: np.ndarray,
    X_pops: np.ndarray,
    q_low: float = 0.10,
    q_high: float = 0.90,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalisation robuste P10-P90 calculÃ©e sur les nÅ“uds, appliquÃ©e aux deux.

    Args:
        X_nodes: Matrice nÅ“uds (n_nodes, n_markers).
        X_pops: Matrice populations (n_pops, n_markers).
        q_low: Percentile bas (dÃ©faut 0.10).
        q_high: Percentile haut (dÃ©faut 0.90).

    Returns:
        (X_nodes_scaled, X_pops_scaled)
    """
    p_lo = np.percentile(X_nodes, q_low * 100, axis=0)
    p_hi = np.percentile(X_nodes, q_high * 100, axis=0)
    rng = np.where(p_hi - p_lo == 0, 1.0, p_hi - p_lo)
    return (X_nodes - p_lo) / rng, (X_pops - p_lo) / rng


def _otsu_threshold_1d(values: np.ndarray, n_bins: int = 64) -> float:
    """
    Seuil bimodal d'Otsu 1D â€” maximise la variance inter-classes.

    Utile pour sÃ©parer les nÅ“uds "assignÃ©s" des "Unknown" quand la
    distribution des distances minimales est bimodale (cas courant
    en cytomÃ©trie quand le panel ne couvre pas toutes les populations).

    Args:
        values: Vecteur de distances ou scores (1D).
        n_bins: RÃ©solution de l'histogramme.

    Returns:
        Seuil optimal (float).
    """
    counts, bin_edges = np.histogram(values, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = counts.sum()
    if total == 0:
        return float(np.median(values))

    w0, w1 = 0.0, 1.0
    mu0, mu1 = 0.0, float(np.average(bin_centers, weights=counts))
    best_var = 0.0
    best_thresh = bin_centers[0]

    for i in range(len(counts)):
        if counts[i] == 0:
            continue
        w0 += counts[i] / total
        w1 = max(1.0 - w0, 1e-9)
        mu0 = (mu0 * (sum(counts[:i]) or 1) + bin_centers[i] * counts[i]) / max(
            sum(counts[: i + 1]), 1
        )
        w0_sum = sum(counts[: i + 1])
        if w0_sum > 0:
            mu0 = float(np.average(bin_centers[: i + 1], weights=counts[: i + 1]))
        if total - w0_sum > 0:
            mu1 = (
                float(np.average(bin_centers[i + 1 :], weights=counts[i + 1 :]))
                if i < len(counts) - 1
                else mu0
            )
        else:
            mu1 = mu0

        var = w0 * (1 - w0) * (mu0 - mu1) ** 2
        if var > best_var:
            best_var = var
            best_thresh = bin_centers[i]

    return float(best_thresh)


def _mad_threshold_1d(values: np.ndarray, k: float = 2.0) -> float:
    """
    Seuil basÃ© sur la MAD (Median Absolute Deviation) : mÃ©diane + k Ã— MAD.

    Args:
        values: Vecteur de distances.
        k: Multiplicateur (dÃ©faut 2.0).

    Returns:
        Seuil (float).
    """
    med = float(np.median(values))
    mad = (
        float(np.median(np.abs(values - med))) * 1.4826
    )  # facteur normalisation gaussienne
    return med + k * mad


def compute_unknown_threshold(
    best_dist: np.ndarray,
    mode: str = "auto_otsu",
    percentile: float = 70.0,
) -> float:
    """
    Calcule le seuil au-delÃ  duquel un nÅ“ud est classÃ© "Unknown".

    Args:
        best_dist: Vecteur des distances minimales par nÅ“ud.
        mode: "auto_otsu" | "percentile" | "mean_std" | "median_iqr" | "mad".
        percentile: Percentile (pour mode="percentile").

    Returns:
        Seuil (float).
    """
    finite = best_dist[np.isfinite(best_dist)]
    if len(finite) == 0:
        return np.inf

    if mode == "auto_otsu":
        return _otsu_threshold_1d(finite)
    elif mode == "percentile":
        return float(np.percentile(finite, percentile))
    elif mode == "mean_std":
        return float(finite.mean() + finite.std())
    elif mode == "median_iqr":
        q1, q3 = np.percentile(finite, [25.0, 75.0])
        return float(q3 + 1.5 * (q3 - q1))
    elif mode == "mad":
        return _mad_threshold_1d(finite)
    else:
        return float(np.percentile(finite, percentile))


def assign_with_auto_threshold(
    dist_matrix: np.ndarray,
    pop_names: List[str],
    threshold_mode: str = "auto_otsu",
    percentile: float = 70.0,
    min_assigned_frac: float = 0.30,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Assigne chaque nÅ“ud Ã  la population la plus proche et calcule le seuil Unknown.

    Si la fraction assignÃ©e est infÃ©rieure Ã  min_assigned_frac, le seuil est
    rehaussÃ© par le percentile pour garantir au moins 30% d'assignation.

    Returns:
        Tuple (best_idx, best_dist, threshold, assigned_labels).
    """
    best_idx = np.argmin(dist_matrix, axis=1)
    best_dist = dist_matrix[np.arange(len(dist_matrix)), best_idx]

    finite_mask = np.isfinite(best_dist)
    if not finite_mask.any():
        _logger.warning("Tous les nÅ“uds ont une distance infinie.")
        return (
            best_idx,
            best_dist,
            np.inf,
            np.full(len(dist_matrix), "Unknown", dtype=object),
        )

    threshold = compute_unknown_threshold(
        best_dist[finite_mask], mode=threshold_mode, percentile=percentile
    )

    assigned = np.where(
        finite_mask & (best_dist <= threshold),
        np.array([pop_names[i] for i in best_idx], dtype=object),
        "Unknown",
    )

    # Garantir un minimum d'assignation (sÃ©curitÃ© clinique)
    frac_assigned = (assigned != "Unknown").mean()
    if frac_assigned < min_assigned_frac:
        alt_thresh = float(np.percentile(best_dist[finite_mask], percentile))
        _logger.debug(
            "Seuil rehaussÃ© (%.1f%% < %.0f%% min): %.3f â†’ %.3f",
            100 * frac_assigned,
            100 * min_assigned_frac,
            threshold,
            alt_thresh,
        )
        threshold = alt_thresh
        assigned = np.where(
            finite_mask & (best_dist <= threshold),
            np.array([pop_names[i] for i in best_idx], dtype=object),
            "Unknown",
        )

    return best_idx, best_dist, threshold, assigned


def _apply_bayesian_prior(
    dist_matrix: np.ndarray,
    pop_names: List[str],
    cell_counts: Any,
    prior_mode: str = "log10_cubed",
    node_sizes: Optional[np.ndarray] = None,
    hard_limit_factor: float = 5.0,
    n_nodes_total: Optional[int] = None,
) -> np.ndarray:
    """
    Ajuste la matrice de distances par un prior bayÃ©sien de prÃ©valence.

    Les populations frÃ©quentes (Granulocytes ~2.2M) sont rendues plus
    attractives que les rares (Plasmocytes ~2000). Avec log10_cubed,
    le ratio d'attractivitÃ© est â‰ˆ7Ã— entre ces deux extrÃªmes â€” cohÃ©rent
    avec la biologie de la moelle osseuse.

    Hard limit ELN : si la taille d'un nÅ“ud dÃ©passe ce que la population
    peut biologiquement contenir (cell_count/total Ã— n_cells Ã— factor),
    la distance est mise Ã  âˆž (rejet absolu).

    Args:
        dist_matrix: Matrice (n_nodes, n_populations).
        pop_names: Noms des populations dans l'ordre des colonnes.
        cell_counts: dict {pop_name: n_cells} ou array de tailles.
        prior_mode: "log10_cubed" | "log10" | "linear" | "none".
        node_sizes: Taille (n_cells) de chaque nÅ“ud SOM â€” pour le hard limit.
        hard_limit_factor: Facteur multiplicatif du max biologique attendu.
        n_nodes_total: Nombre de nÅ“uds (utilisÃ© pour la part thÃ©orique).

    Returns:
        Matrice D ajustÃ©e (copie).
    """
    D = dist_matrix.copy()

    if isinstance(cell_counts, dict):
        counts = np.array([cell_counts.get(p, 1) for p in pop_names], dtype=np.float64)
    else:
        counts = np.asarray(cell_counts, dtype=np.float64)
        if len(counts) < len(pop_names):
            counts = np.ones(len(pop_names), dtype=np.float64)

    total_cells = float(max(counts.sum(), 1))

    for j, (_pname, _n) in enumerate(zip(pop_names, counts)):
        if prior_mode == "log10_cubed":
            w = np.log10(max(_n, 10)) ** 3
        elif prior_mode == "log10":
            w = np.log10(max(_n, 10))
        elif prior_mode == "linear":
            w = max(_n, 1) / max(total_cells / max(len(pop_names), 1), 1)
        else:
            w = 1.0

        D[:, j] /= max(w, 1e-9)

        # Hard limit : nÅ“ud sur-reprÃ©sentÃ© â†’ distance infinie vers cette pop
        if node_sizes is not None and n_nodes_total is not None:
            expected_max = (
                (_n / total_cells) * float(node_sizes.sum()) * hard_limit_factor
            )
            D[node_sizes > expected_max, j] = np.inf

    return D


def _mahalanobis_distance_batch(
    X_nodes: np.ndarray,
    X_pops_mean: np.ndarray,
    pop_names: List[str],
    cov_matrices: Dict[str, np.ndarray],
    cols: List[str],
) -> np.ndarray:
    """
    Calcule la distance de Mahalanobis de chaque nÅ“ud vers chaque population.

    D_M(x, Î¼) = sqrt((x - Î¼)áµ€ Î£â»Â¹ (x - Î¼))

    La matrice de covariance Î£ est inversÃ©e une seule fois par population.

    Args:
        X_nodes: Matrice nÅ“uds (n_nodes, n_features).
        X_pops_mean: Matrice des moyennes de rÃ©fÃ©rence (n_pops, n_features).
        pop_names: Noms des populations.
        cov_matrices: dict {pop_name: covariance_matrix (n_features, n_features)}.
        cols: Noms des colonnes (pour aligner les covariances).

    Returns:
        Matrice (n_nodes, n_pops) des distances Mahalanobis.
    """
    n_nodes, n_feat = X_nodes.shape
    n_pops = len(pop_names)
    D = np.full((n_nodes, n_pops), np.inf, dtype=np.float64)

    for j, pop in enumerate(pop_names):
        mu = X_pops_mean[j]
        cov = cov_matrices.get(pop)
        if cov is None:
            # Fallback euclidienne si covariance absente
            diff = X_nodes - mu
            D[:, j] = np.sqrt((diff**2).sum(axis=1))
            continue

        # SÃ©lectionner la sous-matrice de covariance pour les colonnes disponibles
        n_c = min(cov.shape[0], n_feat)
        cov_sub = cov[:n_c, :n_c].copy()

        try:
            from scipy.linalg import inv as _inv, LinAlgError

            inv_cov = _inv(cov_sub)
        except Exception:
            inv_cov = np.diag(1.0 / np.diag(cov_sub).clip(min=1e-9))

        diff = X_nodes[:, :n_c] - mu[:n_c]
        # D_MÂ² = diag( diff @ inv_cov @ diff.T )
        mid = diff @ inv_cov  # (n_nodes, n_c)
        d_sq = np.einsum("ij,ij->i", mid, diff)  # (n_nodes,)
        D[:, j] = np.sqrt(np.maximum(d_sq, 0.0))

    return D


def _proportional_stratified_pool(
    knn_samples: Dict[str, np.ndarray],
    pop_names: List[str],
    cell_counts: Dict[str, int],
    total_points: int = 15_000,
    min_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit un pool KNN stratifiÃ© proportionnellement aux effectifs biologiques.

    Chaque population reÃ§oit au minimum min_points points et au maximum
    sa quote-part proportionnelle de total_points.

    Args:
        knn_samples: dict {pop_name: array (n_sample, n_features)}.
        pop_names: Liste des populations dans l'ordre du mapping.
        cell_counts: dict {pop_name: n_cells}.
        total_points: Total de points dans le pool.
        min_points: Minimum par population.

    Returns:
        (X_pool, y_pool) oÃ¹ y_pool est l'index de la population (int).
    """
    total_cells = sum(cell_counts.get(p, 1) for p in pop_names)
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    rng = np.random.default_rng(42)

    for j, pop in enumerate(pop_names):
        sample = knn_samples.get(pop)
        if sample is None or len(sample) == 0:
            continue
        quota = max(
            min_points,
            int(total_points * cell_counts.get(pop, 1) / max(total_cells, 1)),
        )
        n_avail = len(sample)
        if n_avail <= quota:
            X_parts.append(sample)
        else:
            idx = rng.choice(n_avail, quota, replace=False)
            X_parts.append(sample[idx])
        y_parts.append(np.full(len(X_parts[-1]), j, dtype=np.int32))

    if not X_parts:
        return np.empty((0, 0)), np.empty(0, dtype=np.int32)

    return np.vstack(X_parts), np.concatenate(y_parts)


def _knn_vote(
    X_nodes: np.ndarray,
    pop_names: List[str],
    knn_samples: Dict[str, np.ndarray],
    cell_counts: Dict[str, int],
    k: int = 15,
    total_knn_points: int = 15_000,
    min_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vote KNN densitÃ© sur un pool stratifiÃ© par prÃ©valence biologique.

    Args:
        X_nodes: Matrice nÅ“uds normalisÃ©s (n_nodes, n_features).
        pop_names: Noms des populations.
        knn_samples: dict {pop_name: array rÃ©servoir}.
        cell_counts: dict {pop_name: n_cells} pour la stratification.
        k: Nombre de voisins.
        total_knn_points: Taille du pool stratifiÃ©.
        min_points: Minimum de points par population.

    Returns:
        (best_idx, dist_approx) â€” mÃªme signature que les autres mÃ©thodes.
    """
    if not _KNN_AVAILABLE:
        _logger.warning("sklearn absent â€” KNN vote non disponible")
        # Fallback euclidean
        from scipy.spatial.distance import cdist as _cdist

        pop_means = np.array(
            [
                knn_samples[p].mean(axis=0)
                if p in knn_samples and len(knn_samples[p]) > 0
                else np.zeros(X_nodes.shape[1])
                for p in pop_names
            ]
        )
        D = _cdist(X_nodes, pop_means, metric="euclidean")
        return np.argmin(D, axis=1), D.min(axis=1)

    X_pool, y_pool = _proportional_stratified_pool(
        knn_samples,
        pop_names,
        cell_counts,
        total_points=total_knn_points,
        min_points=min_points,
    )

    if len(X_pool) == 0:
        return np.zeros(len(X_nodes), dtype=np.int32), np.ones(len(X_nodes))

    knn = KNeighborsClassifier(
        n_neighbors=min(k, len(X_pool)), metric="euclidean", n_jobs=-1
    )
    knn.fit(X_pool, y_pool)

    pred_classes = knn.predict(X_nodes)
    # Distance approchÃ©e : distance au plus proche voisin
    dist_approx, _ = knn.kneighbors(X_nodes, n_neighbors=1)

    return pred_classes, dist_approx.ravel()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Â§10.4b â€” map_populations_to_nodes_v5 â€” 12 mÃ©thodes complÃ¨tes
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def map_populations_to_nodes_v5(
    node_mfi_raw: pd.DataFrame,
    pop_mfi_ref: pd.DataFrame,
    node_sizes: Optional[np.ndarray] = None,
    cell_counts: Optional[Any] = None,
    pop_cov_matrices: Optional[Dict[str, np.ndarray]] = None,
    pop_knn_samples: Optional[Dict[str, np.ndarray]] = None,
    method: str = "cosine_prior",
    include_scatter: bool = False,
    threshold_mode: str = "auto_otsu",
    threshold_percentile: float = 70.0,
    normalization_method: str = "range",
    hard_limit_factor: float = 5.0,
    prior_mode: str = "log10_cubed",
    transform_method: str = "none",
    arcsinh_cofactor: float = 5.0,
    data_already_transformed: bool = False,
    cache_dir: Optional[Path] = None,
    csv_source_path: Optional[str] = None,
    filter_area_only: bool = True,
    run_benchmark: bool = False,
    knn_k: int = 15,
    total_knn_points: int = 15_000,
) -> pd.DataFrame:
    """
    Algorithme V5 â€” mapping populations â†’ nÅ“uds SOM.

    12 mÃ©thodes disponibles (benchmark ou mÃ©thode unique) :
      M1  : Euclidean + range norm
      M2  : Manhattan + range norm
      M3  : Cosine (clip nÃ©gatifs)
      M4  : Euclidean robust scale P10-P90
      M5  : Euclidean + arcsinh(x/5) + range
      M6  : Euclidean weighted scatter Ã—0.3
      M7  : Vote M2+M3+M8
      M8  : Euclidean + ref-frame normalization (nÅ“uds scaler, puis ref)
      M9  : M8 + Bayesian Prior log10
      M10 : Mahalanobis (requiert pop_cov_matrices)
      M11 : KNN densitÃ© proportionnel (requiert pop_knn_samples + cell_counts)
      M12 : Cosine + log10Â³ prior + hard limit [RECOMMANDÃ‰ ELN 2022]

    Args:
        node_mfi_raw: DataFrame [n_nodes Ã— n_markers] â€” MFI par nÅ“ud SOM.
        pop_mfi_ref: DataFrame [n_populations Ã— n_markers] â€” MFI de rÃ©fÃ©rence.
        node_sizes: Taille (n_cells) de chaque nÅ“ud â€” requis pour hard limit.
        cell_counts: dict {pop: n_cells} â€” requis pour prior M9/M11/M12.
        pop_cov_matrices: dict {pop: cov (n_f,n_f)} â€” requis pour M10.
        pop_knn_samples: dict {pop: array (n_sample,n_f)} â€” requis pour M11.
        method: MÃ©thode parmi M1â€“M12 ou alias ("cosine_prior"=M12).
        include_scatter: Inclure FSC/SSC dans la distance.
        threshold_mode: "auto_otsu" | "percentile" | "mean_std" | "mad".
        threshold_percentile: Percentile pour le seuil Unknown.
        normalization_method: "range" | "zscore" | "none".
        hard_limit_factor: Facteur multiplicatif du max biologique.
        prior_mode: "log10_cubed" | "log10" | "linear".
        transform_method: Transformation des MFI de rÃ©fÃ©rence avant mapping.
        arcsinh_cofactor: Cofacteur arcsinh pour la transformation de rÃ©fÃ©rence.
        data_already_transformed: True si pop_mfi_ref est dÃ©jÃ  transformÃ©.
        filter_area_only: True = n'utilise que les canaux -A (jamais -H).
        run_benchmark: True = exÃ©cuter tous les M1â€“M12 et choisir le meilleur.
        knn_k: Nombre de voisins pour M11.
        total_knn_points: Taille du pool KNN stratifiÃ©.

    Returns:
        DataFrame: [node_id, best_pop, best_dist, threshold, assigned_pop, method].
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy requis: pip install scipy")

    # â”€â”€ Alias de mÃ©thode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _METHOD_MAP = {
        "cosine_prior": "M12",
        "euclidean_prior": "M9",
        "euclidean": "M1",
        "cosine": "M3",
        "mahalanobis": "M10",
        "knn": "M11",
    }
    method_id = _METHOD_MAP.get(method, method)

    # â”€â”€ Transformation des MFI de rÃ©fÃ©rence si nÃ©cessaire â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if transform_method != "none" and not data_already_transformed:
        _logger.info("Transformation des MFI de rÃ©fÃ©rence: %s", transform_method)
        pop_mfi_ref_tr = pop_mfi_ref.copy()
        numeric_cols = pop_mfi_ref_tr.select_dtypes(include=[np.number]).columns
        pop_mfi_ref_tr[numeric_cols] = apply_cyto_transform_matrix(
            pop_mfi_ref_tr[numeric_cols].to_numpy(dtype=np.float64),
            col_names=list(numeric_cols),
            transform_type=transform_method,
            cofactor=arcsinh_cofactor,
        )
    else:
        pop_mfi_ref_tr = pop_mfi_ref

    # â”€â”€ SÃ©lection des colonnes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cols_common = [c for c in node_mfi_raw.columns if c in pop_mfi_ref_tr.columns]
    if not cols_common:
        raise ValueError(
            "Aucune colonne commune node_mfi_raw / pop_mfi_ref. "
            "Aligner les marqueurs en amont."
        )

    if not include_scatter:
        cols_common = [
            c
            for c in cols_common
            if not any(c.upper().startswith(p) for p in _SCATTER_PREFIXES)
        ]

    if filter_area_only:
        cols_common = filter_area_columns(cols_common)

    if not cols_common:
        _logger.warning(
            "Aucun marqueur aprÃ¨s filtrage â†’ utilisation colonnes brutes"
        )
        cols_common = list(node_mfi_raw.columns)[: min(5, len(node_mfi_raw.columns))]

    pop_names = list(pop_mfi_ref_tr.index)
    n_nodes = len(node_mfi_raw)

    X_nodes_raw = node_mfi_raw[cols_common].to_numpy(dtype=np.float64)
    X_pops_raw = pop_mfi_ref_tr[cols_common].to_numpy(dtype=np.float64)

    _logger.info(
        "V5 [%s]: %d nÅ“uds Ã— %d pops Ã— %d marqueurs",
        method_id,
        n_nodes,
        len(pop_names),
        len(cols_common),
    )

    # â”€â”€ Normalisation par dÃ©faut (range) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    X_nodes_norm, scale = normalize_matrix(X_nodes_raw, method=normalization_method)
    if normalization_method == "range":
        X_pops_norm = np.clip((X_pops_raw - scale["min"]) / scale["range"], 0.0, None)
    elif normalization_method == "zscore":
        X_pops_norm = (X_pops_raw - scale["mean"]) / scale["std"]
    else:
        X_pops_norm = X_pops_raw

    # â”€â”€ Espace clipÃ© â‰¥0 pour distance cosine â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    X_nodes_pos = np.clip(X_nodes_norm, 0.0, None)
    X_pops_pos = np.clip(X_pops_norm, 0.0, None)

    # â”€â”€ Helper : matrice de distance pour une mÃ©thode donnÃ©e â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _compute_dist(mid: str) -> np.ndarray:
        if mid == "M1":
            return cdist(X_nodes_norm, X_pops_norm, metric="euclidean")

        elif mid == "M2":
            return cdist(X_nodes_norm, X_pops_norm, metric="cityblock")

        elif mid == "M3":
            return cdist(X_nodes_pos, X_pops_pos, metric="cosine")

        elif mid == "M4":
            _xn, _xp = robust_scale(X_nodes_raw, X_pops_raw)
            return cdist(_xn, _xp, metric="euclidean")

        elif mid == "M5":
            _xn5 = arcsinh_transform(X_nodes_raw)
            _xp5 = arcsinh_transform(X_pops_raw)
            _xn5n, _sc5 = normalize_matrix(_xn5)
            _xp5n = np.clip((_xp5 - _sc5["min"]) / _sc5["range"], 0.0, None)
            return cdist(_xn5n, _xp5n, metric="euclidean")

        elif mid == "M6":
            # PondÃ©ration des canaux : scatter Ã— 0.3, fluorescence Ã— 1.0
            weights = np.array(
                [
                    0.3 if any(c.upper().startswith(p) for p in ("FSC", "SSC")) else 1.0
                    for c in cols_common
                ]
            )
            _xnw = X_nodes_norm * weights
            _xpw = X_pops_norm * weights
            return cdist(_xnw, _xpw, metric="euclidean")

        elif mid == "M7":
            # Vote M2 + M3 + M8
            _d2 = _compute_dist("M2")
            _d3 = _compute_dist("M3")
            _d8 = _compute_dist("M8")

            # Rang-normaliser chaque matrice puis voter
            def _rank_norm(D):
                ranks = D.argsort(axis=1).argsort(axis=1).astype(np.float64)
                return ranks / max(ranks.max(), 1.0)

            return _rank_norm(_d2) + _rank_norm(_d3) + _rank_norm(_d8)

        elif mid == "M8":
            # Ref-frame : normaliser par les paramÃ¨tres de la rÃ©fÃ©rence
            _ref_min = X_pops_raw.min(axis=0)
            _ref_max = X_pops_raw.max(axis=0)
            _ref_rng = np.where(_ref_max - _ref_min == 0, 1.0, _ref_max - _ref_min)
            _xn8 = (X_nodes_raw - _ref_min) / _ref_rng
            _xp8 = (X_pops_raw - _ref_min) / _ref_rng
            return cdist(_xn8, _xp8, metric="euclidean")

        elif mid == "M9":
            _d8 = _compute_dist("M8")
            if cell_counts is not None:
                _d8 = _apply_bayesian_prior(
                    _d8,
                    pop_names,
                    cell_counts,
                    prior_mode="log10",
                    node_sizes=node_sizes,
                    hard_limit_factor=hard_limit_factor,
                    n_nodes_total=n_nodes,
                )
            return _d8

        elif mid == "M10":
            if pop_cov_matrices is None:
                _logger.debug(
                    "M10 Mahalanobis: pop_cov_matrices absent â†’ fallback M8"
                )
                return _compute_dist("M8")
            return _mahalanobis_distance_batch(
                X_nodes_norm, X_pops_norm, pop_names, pop_cov_matrices, cols_common
            )

        elif mid == "M11":
            if pop_knn_samples is None or cell_counts is None:
                _logger.debug("M11 KNN: samples/counts absents â†’ fallback M3")
                return _compute_dist("M3")
            _pred, _dist = _knn_vote(
                X_nodes_norm,
                pop_names,
                pop_knn_samples,
                cell_counts,
                k=knn_k,
                total_knn_points=total_knn_points,
            )
            # Convertir prÃ©dictions en matrice de distance fictive
            D_knn = np.full((n_nodes, len(pop_names)), 1.0)
            D_knn[np.arange(n_nodes), _pred] = _dist
            return D_knn

        elif mid == "M12":
            # MÃ‰THODE RECOMMANDÃ‰E ELN 2022
            # Cosine (insensible aux diffÃ©rences d'intensitÃ© absolue)
            # + prior log10Â³ (Granulos >> Plasmos)
            # + hard limit (rejet nÅ“uds sur-reprÃ©sentÃ©s)
            _d3 = cdist(X_nodes_pos, X_pops_pos, metric="cosine")
            if cell_counts is not None:
                _d3 = _apply_bayesian_prior(
                    _d3,
                    pop_names,
                    cell_counts,
                    prior_mode="log10_cubed",
                    node_sizes=node_sizes,
                    hard_limit_factor=hard_limit_factor,
                    n_nodes_total=n_nodes,
                )
            return _d3

        else:
            _logger.warning("MÃ©thode inconnue '%s' â†’ fallback M12", mid)
            return _compute_dist("M12")

    # â”€â”€ Mode benchmark (M1â€“M12) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if run_benchmark:
        methods_to_run = [f"M{i}" for i in range(1, 13)]
        best_result: Optional[pd.DataFrame] = None
        best_score = -1.0
        best_method_name = "M12"

        for mid in methods_to_run:
            try:
                dist_m = _compute_dist(mid)
                bi, bd, thr, assigned = assign_with_auto_threshold(
                    dist_m,
                    pop_names,
                    threshold_mode=threshold_mode,
                    percentile=threshold_percentile,
                )
                # Score de diversitÃ© : nombre de populations distinctes assignÃ©es
                n_distinct = len(set(assigned) - {"Unknown"})
                frac_assigned = (assigned != "Unknown").mean()
                score = n_distinct * frac_assigned  # favorise diversitÃ© + couverture

                _logger.debug(
                    "  %s: %d pops distinctes, %.1f%% assignÃ©s â†’ score=%.2f",
                    mid,
                    n_distinct,
                    100 * frac_assigned,
                    score,
                )

                df_m = pd.DataFrame(
                    {
                        "node_id": np.arange(n_nodes),
                        "best_pop": [pop_names[i] for i in bi],
                        "best_dist": np.round(bd, 6),
                        "threshold": round(thr, 6),
                        "assigned_pop": assigned,
                        "method": mid,
                    }
                )
                if "metacluster" in node_mfi_raw.columns:
                    df_m["metacluster"] = node_mfi_raw["metacluster"].values

                if score > best_score:
                    best_score = score
                    best_result = df_m
                    best_method_name = mid

            except Exception as exc:
                _logger.debug("  %s Ã©chouÃ©: %s", mid, exc)

        result = best_result if best_result is not None else pd.DataFrame()
        _logger.info(
            "Benchmark terminÃ© â€” meilleure mÃ©thode: %s (score=%.2f)",
            best_method_name,
            best_score,
        )
        return result

    # â”€â”€ Mode mÃ©thode unique â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    dist_matrix = _compute_dist(method_id)
    best_idx, best_dist, threshold, assigned = assign_with_auto_threshold(
        dist_matrix,
        pop_names,
        threshold_mode=threshold_mode,
        percentile=threshold_percentile,
    )

    result = pd.DataFrame(
        {
            "node_id": np.arange(n_nodes),
            "best_pop": [pop_names[i] for i in best_idx],
            "best_dist": np.round(best_dist, 6),
            "threshold": round(threshold, 6),
            "assigned_pop": assigned,
            "method": method_id,
        }
    )

    n_tot = len(result)
    n_unk = int((result["assigned_pop"] == "Unknown").sum())
    _logger.info(
        "V5 [%s] terminÃ©: %d assignÃ©s (%d%%), %d Unknown (%d%%)",
        method_id,
        n_tot - n_unk,
        round(100 * (n_tot - n_unk) / max(n_tot, 1)),
        n_unk,
        round(100 * n_unk / max(n_tot, 1)),
    )
    return result


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Helpers complÃ©mentaires pour le pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def map_nodes_to_metaclusters(
    mapping_df: pd.DataFrame,
    metaclustering_per_node: np.ndarray,
) -> pd.DataFrame:
    """
    Enrichit le mapping nÅ“udsâ†’populations avec l'assignation de mÃ©tacluster.

    Args:
        mapping_df: RÃ©sultat de map_populations_to_nodes_v5.
        metaclustering_per_node: Vecteur (n_nodes,) avec l'id de mÃ©tacluster.

    Returns:
        mapping_df enrichi avec colonne "metacluster".
    """
    result = mapping_df.copy()
    result["metacluster"] = np.array(
        [
            metaclustering_per_node[nid] if nid < len(metaclustering_per_node) else -1
            for nid in result["node_id"]
        ]
    )
    return result


def get_population_summary(mapping_df: pd.DataFrame) -> pd.DataFrame:
    """RÃ©sumÃ© de la distribution des nÅ“uds par population assignÃ©e."""
    counts = mapping_df["assigned_pop"].value_counts().reset_index()
    counts.columns = ["population", "n_nodes"]
    counts["pct_nodes"] = (counts["n_nodes"] / len(mapping_df) * 100).round(2)
    return counts


def build_population_color_map(
    populations: List[str],
    custom_colors: Optional[Dict[str, str]] = None,
    default_color: str = "#7f7f7f",
) -> Dict[str, str]:
    """
    Construit un dictionnaire population â†’ couleur pour la visualisation.

    Args:
        populations: Liste des populations.
        custom_colors: Couleurs personnalisÃ©es Ã  fusionner.
        default_color: Couleur pour les populations non listÃ©es.

    Returns:
        Dict {population: hex_color}.
    """
    colors = {**POPULATION_COLORS}
    if custom_colors:
        colors.update(custom_colors)
    return {pop: colors.get(pop, default_color) for pop in populations}
