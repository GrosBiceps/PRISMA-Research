"""
cli/main.py — Point d'entrée de la CLI FlowSOM Pipeline Pro.

Orchestre :
  1. Parsing des arguments (argparse via parsers.py)
  2. Configuration du logging
  3. Construction du PipelineConfig (YAML + CLI args en surcharge)
  4. Exécution du pipeline via FlowSOMPipeline.execute()
  5. Affichage du résumé et gestion des codes de sortie
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> None:
    """
    Point d'entrée principal de la CLI.

    Charge la configuration, exécute le pipeline et affiche le résumé.
    Sort avec le code 0 en cas de succès, 1 en cas d'échec.
    """
    from flowsom_pipeline_pro.cli.parsers import (
        build_argument_parser,
        detect_config_file,
    )
    from flowsom_pipeline_pro.src.pipeline.pipeline_executor import FlowSOMPipeline

    parser = build_argument_parser()
    args = parser.parse_args()

    # ── Configuration du logging ──────────────────────────────────────────────
    if args.quiet:
        log_level = logging.WARNING
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("flowsom_pipeline_pro")

    # ── Banner ────────────────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("  FlowSOM Analysis Pipeline Pro — v1.0.0")
    logger.info("=" * 72)

    # ── Résolution du fichier YAML ────────────────────────────────────────────
    # Priorité : --config CLI > auto-détection dans le répertoire courant
    config_path = args.config
    if config_path is None:
        config_path = detect_config_file(script_dir=Path(__file__).parent)
        if config_path:
            logger.info(f"Configuration auto-détectée : {config_path}")
        else:
            logger.info(
                "Aucun fichier YAML détecté — utilisation des valeurs par défaut."
            )
    else:
        logger.info(f"Configuration chargée depuis : {config_path}")

    # ── Construction du PipelineConfig ────────────────────────────────────────
    try:
        config = _build_config(args, config_path, logger)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    # ── Validation des chemins obligatoires ───────────────────────────────────
    if not config.paths.healthy_folder:
        logger.error(
            "Aucun dossier sain (NBM) spécifié.\n"
            "  → Utilisez : --healthy-folder 'chemin/vers/NBM'\n"
            "  → Ou renseignez 'paths.healthy_folder' dans le fichier YAML."
        )
        sys.exit(1)

    if not Path(config.paths.healthy_folder).exists():
        logger.error(f"Le dossier sain n'existe pas : {config.paths.healthy_folder}")
        sys.exit(1)

    if (
        config.analysis.compare_mode
        and config.paths.patho_folder
        and not Path(config.paths.patho_folder).exists()
    ):
        logger.error(
            f"Le dossier pathologique n'existe pas : {config.paths.patho_folder}"
        )
        sys.exit(1)

    # ── Résumé de la configuration ────────────────────────────────────────────
    logger.info("Configuration active :")
    logger.info(f"  Dossier sain          : {config.paths.healthy_folder}")
    if config.analysis.compare_mode and config.paths.patho_folder:
        logger.info(f"  Dossier pathologique  : {config.paths.patho_folder}")
    logger.info(f"  Mode comparaison      : {config.analysis.compare_mode}")
    logger.info(
        f"  Pre-gating            : viable={config.pregate.viable}, "
        f"singlets={config.pregate.singlets}, "
        f"CD45={config.pregate.cd45}, CD34={config.pregate.cd34}"
    )
    logger.info(
        f"  FlowSOM               : grille {config.flowsom.xdim}×{config.flowsom.ydim}, "
        f"{config.flowsom.n_metaclusters} métaclusters, "
        f"{config.flowsom.n_iterations} itérations"
    )
    logger.info(
        f"  Transformation        : {config.transform.method} "
        f"(cofacteur={config.transform.cofactor})"
    )
    logger.info(f"  Normalisation         : {config.normalize.method}")
    logger.info(f"  GPU                   : {config.gpu.enabled}")
    logger.info(f"  Dossier de sortie     : {config.paths.output_dir}")

    # ── Exécution ─────────────────────────────────────────────────────────────
    logger.info("\nDémarrage de l'analyse…")
    pipeline = FlowSOMPipeline(config=config)

    try:
        result = pipeline.execute()
    except Exception as exc:
        logger.error(
            f"Erreur inattendue lors de l'exécution : {exc}", exc_info=args.verbose
        )
        sys.exit(1)

    # ── Résultat ──────────────────────────────────────────────────────────────
    print(result.summary())

    if not result.success:
        logger.error("Le pipeline a échoué. Consultez le résumé ci-dessus.")
        sys.exit(1)

    logger.info(f"Résultats disponibles dans : {config.paths.output_dir}")
    # Code de sortie 0 implicite


# ── Helpers internes ──────────────────────────────────────────────────────────


def _build_config(args, config_path, logger):
    """Construit le PipelineConfig, puis applique les surcharges CLI."""
    from flowsom_pipeline_pro.config.pipeline_config import PipelineConfig

    cfg = PipelineConfig.from_args(args, yaml_path=config_path)

    # Surcharges spécifiques non prises en charge par from_args
    if getattr(args, "pregate_viable", None) is not None:
        cfg.pregate.viable = args.pregate_viable
    if getattr(args, "pregate_singlets", None) is not None:
        cfg.pregate.singlets = args.pregate_singlets
    if getattr(args, "pregate_cd45", None) is not None:
        cfg.pregate.cd45 = args.pregate_cd45
    if getattr(args, "pregate_cd34", None) is not None:
        cfg.pregate.cd34 = args.pregate_cd34
    if getattr(args, "pregate_mode", None):
        cfg.pregate.mode = args.pregate_mode

    if getattr(args, "auto_clustering", None) is not None:
        cfg.auto_clustering.enabled = args.auto_clustering
    if getattr(args, "min_clusters", None) is not None:
        cfg.auto_clustering.min_clusters = args.min_clusters
    if getattr(args, "max_clusters", None) is not None:
        cfg.auto_clustering.max_clusters = args.max_clusters

    return cfg


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
#  _load_yaml_config — Mapper YAML → dict plat de clés argparse
# ─────────────────────────────────────────────────────────────────────────────


def _load_yaml_config(config_path: str) -> dict:
    """
    Charge un fichier YAML et retourne un dict plat de clés argparse.

    Toutes les sections du YAML de configuration (paths, analysis, pregate,
    pregate_advanced, flowsom, auto_clustering, transform, normalize, markers,
    downsampling, visualization, gpu) sont aplaties en un dict dont les clés
    correspondent aux noms des arguments argparse du parser CLI.

    Args:
        config_path: Chemin vers le fichier YAML de configuration.

    Returns:
        Dict {cli_key: valeur} — vide si PyYAML absent ou fichier illisible.
    """
    try:
        import yaml
    except ImportError:
        logging.getLogger("flowsom_pipeline_pro.cli").warning(
            "PyYAML non installé — config YAML ignorée. (pip install pyyaml)"
        )
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg: dict = {}

    # ── paths ─────────────────────────────────────────────────────────────────
    paths = raw.get("paths", {})
    if paths.get("healthy_folder"):
        cfg["healthy_folder"] = paths["healthy_folder"]
    if paths.get("patho_folder"):
        cfg["patho_folder"] = paths["patho_folder"]
    if paths.get("output_dir"):
        cfg["output"] = paths["output_dir"]

    # ── analysis ─────────────────────────────────────────────────────────────
    analysis = raw.get("analysis", {})
    if "compare_mode" in analysis:
        cfg["compare_mode"] = analysis["compare_mode"]

    # ── pregate ──────────────────────────────────────────────────────────────
    pregate = raw.get("pregate", {})
    if "apply" in pregate:
        cfg["apply_pregating"] = pregate["apply"]
    if "mode" in pregate:
        cfg["gating_mode"] = pregate["mode"]
    if "mode_blastes_vs_normal" in pregate:
        cfg["mode_blastes_vs_normal"] = pregate["mode_blastes_vs_normal"]
    for k_yaml, k_cfg in (
        ("viable", "pregate_viable"),
        ("singlets", "pregate_singlets"),
        ("cd45", "pregate_cd45"),
        ("cd34", "pregate_cd34"),
    ):
        if k_yaml in pregate:
            cfg[k_cfg] = pregate[k_yaml]

    # ── pregate_advanced ─────────────────────────────────────────────────────
    pg_adv = raw.get("pregate_advanced", {})
    for k_yaml, k_cfg in (
        ("debris_min_percentile", "debris_min_percentile"),
        ("debris_max_percentile", "debris_max_percentile"),
        ("doublets_ratio_min", "ratio_min"),
        ("doublets_ratio_max", "ratio_max"),
        ("cd45_threshold_percentile", "cd45_threshold_percentile"),
        ("cd34_threshold_percentile", "cd34_threshold_percentile"),
        ("cd34_use_ssc_filter", "use_ssc_filter_for_blasts"),
        ("cd34_ssc_max_percentile", "ssc_max_percentile_blasts"),
    ):
        if k_yaml in pg_adv:
            cfg[k_cfg] = pg_adv[k_yaml]

    # ── flowsom ───────────────────────────────────────────────────────────────
    fs = raw.get("flowsom", {})
    for k in (
        "xdim",
        "ydim",
        "n_metaclusters",
        "learning_rate",
        "sigma",
        "n_iterations",
        "seed",
        "rlen",
    ):
        if k in fs:
            cfg[k] = fs[k]

    # ── auto_clustering ───────────────────────────────────────────────────────
    ac = raw.get("auto_clustering", {})
    if "enabled" in ac:
        cfg["auto_cluster"] = ac["enabled"]
    for k_yaml, k_cfg in (
        ("min_clusters", "min_clusters_auto"),
        ("max_clusters", "max_clusters_auto"),
        ("n_bootstrap", "n_bootstrap"),
        ("sample_size_bootstrap", "sample_size_bootstrap"),
        ("min_stability_threshold", "min_stability_threshold"),
        ("weight_stability", "w_stability"),
        ("weight_silhouette", "w_silhouette"),
    ):
        if k_yaml in ac:
            cfg[k_cfg] = ac[k_yaml]

    # ── transform ─────────────────────────────────────────────────────────────
    transform = raw.get("transform", {})
    if "method" in transform:
        cfg["transform"] = transform["method"]
    if "cofactor" in transform:
        cfg["cofactor"] = transform["cofactor"]
    if "apply_to_scatter" in transform:
        cfg["apply_to_scatter"] = transform["apply_to_scatter"]

    # ── normalize ─────────────────────────────────────────────────────────────
    normalize = raw.get("normalize", {})
    if "method" in normalize:
        cfg["normalize"] = normalize["method"]

    # ── markers ───────────────────────────────────────────────────────────────
    mk = raw.get("markers", {})
    if "exclude_scatter" in mk:
        cfg["exclude_scatter"] = mk["exclude_scatter"]
    if "exclude_additional" in mk:
        cfg["exclude_additional_markers"] = mk["exclude_additional"]

    # ── downsampling ──────────────────────────────────────────────────────────
    ds = raw.get("downsampling", {})
    if "enabled" in ds:
        cfg["downsample"] = ds["enabled"]
    if "max_cells_per_file" in ds:
        cfg["max_cells_per_file"] = ds["max_cells_per_file"]
    if "max_cells_total" in ds:
        cfg["max_cells_total"] = ds["max_cells_total"]

    # ── visualization ─────────────────────────────────────────────────────────
    viz = raw.get("visualization", {})
    if "save_plots" in viz:
        cfg["save_plots"] = viz["save_plots"]
    if "plot_format" in viz:
        cfg["plot_format"] = viz["plot_format"]
    if "dpi" in viz:
        cfg["dpi"] = viz["dpi"]

    # ── gpu ───────────────────────────────────────────────────────────────────
    gpu = raw.get("gpu", {})
    if "enabled" in gpu:
        cfg["use_gpu"] = gpu["enabled"]

    return cfg


def _get(args: "argparse.Namespace", yaml_cfg: dict, attr: str, default=None):
    """
    Résolution de priorité CLI → YAML → défaut.

    Retourne la valeur CLI si non-None, sinon la valeur YAML, sinon ``default``.

    Args:
        args: Namespace argparse.
        yaml_cfg: Dict plat retourné par _load_yaml_config().
        attr: Nom de l'attribut CLI / clé YAML.
        default: Valeur par défaut si absent des deux sources.

    Returns:
        Valeur résolue selon la priorité CLI > YAML > default.
    """
    cli_val = getattr(args, attr, None)
    if cli_val is not None:
        return cli_val
    return yaml_cfg.get(attr, default)
