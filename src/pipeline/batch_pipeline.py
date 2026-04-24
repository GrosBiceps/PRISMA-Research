"""
batch_pipeline.py — Orchestrateur du mode batch avec cache DATA NBM.

Objectif scientifique :
    Pour chaque moelle pathologique, entraîner une nouvelle grille FlowSOM
    de zéro sur [NBM subsamplé + Patho]. Chaque SOM est unique et s'adapte
    à la topologie des blastes du patient courant. Le subsampling du NBM est
    le levier principal pour équilibrer les deux conditions.

Optimisation :
    Le preprocessing du pool NBM (gating, arcsinh, z-score…) est coûteux
    (~30-60 s pour 20 fichiers FCS). Il est effectué une seule fois, puis le
    DataFrame résultant est mis en cache disque (Parquet). À chaque itération,
    le NBM est chargé depuis ce cache en <1 s.

Workflow :
    ┌─ Initialisation (une seule fois) ──────────────────────────────────────┐
    │  1. Charger + preprocesser les FCS NBM.                                │
    │  2. Sauvegarder le DataFrame NBM dans le cache Parquet.                │
    └────────────────────────────────────────────────────────────────────────┘
    ┌─ Boucle (pour chaque moelle pathologique) ─────────────────────────────┐
    │  1. Charger le DataFrame NBM depuis le cache (RAM).                    │
    │  2. Subsampler le NBM (ratio configurable ou n fixe).                  │
    │  3. Charger + preprocesser le FCS pathologique courant.                │
    │  4. Construire des FlowSamples combinés [NBM subsamplé + Patho].       │
    │  5. Injecter dans le checkpoint gating de FlowSOMPipeline.             │
    │  6. Appeler FlowSOMPipeline.execute() → SOM entraîné de zéro (fit).   │
    │  7. Libérer les variables intermédiaires (gc.collect()).               │
    └────────────────────────────────────────────────────────────────────────┘

Le SOM n'est PAS mis en cache — chaque grille est entraînée sur des données
fraîchement combinées pour forcer la diversité topologique.

Usage (GUI via BatchWorker) :
    batch = BatchPipeline(config)
    summary = batch.execute(progress_callback=cb)

Usage (CLI) :
    batch = BatchPipeline(config)
    summary = batch.execute()
"""

from __future__ import annotations

import copy
import gc
import threading
import time
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from config.pipeline_config import PipelineConfig
from src.io.fcs_reader import get_fcs_files, load_as_flow_samples
from src.models.sample import FlowSample
from src.models.pipeline_result import PipelineResult
from src.services.preprocessing_service import preprocess_combined
from src.services.clustering_service import select_markers_for_clustering
from src.pipeline.pipeline_executor import (
    FlowSOMPipeline,
    PipelineCheckpointManager,
)
from src.pipeline.nbm_cache_manager import NBMCacheManager
from src.utils.logger import GatingLogger, get_logger
from src.models.gate_result import gating_reports, gating_log_entries

_logger = get_logger("pipeline.batch")

# Callback de progression : (index courant, total, nom du fichier)
ProgressCallback = Callable[[int, int, str], None]

# ARCH-4 FIX : verrou protégeant les listes globales gating_reports / gating_log_entries
# contre les data races entre le thread GUI (lecture) et le worker batch (clear + écriture).
_GATING_STATE_LOCK: threading.Lock = threading.Lock()


class BatchPipeline:
    """
    Traitement batch de N moelles pathologiques contre un pool NBM mis en cache.

    Le SOM est ré-entraîné de zéro à chaque itération sur [NBM subsamplé + Patho].

    Args:
        config: Configuration du pipeline (batch.enabled doit être True).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Point d'entrée
    # ------------------------------------------------------------------

    def execute(
        self,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """
        Lance le pipeline batch complet.

        Args:
            progress_callback: Optionnel — appelé avant chaque moelle avec
                               (index, total, filename). Utilisé par le GUI.

        Returns:
            Dict {"results": [(stem, PipelineResult), …]} compatible avec
            le format attendu par BatchWorker du GUI.
        """
        t_start = time.time()
        config = self.config

        # ── Découverte des fichiers patho ─────────────────────────────────────
        patho_folder = Path(getattr(config.paths, "patho_folder", "") or "")
        if not patho_folder.exists():
            _logger.error("Batch: dossier patho introuvable: %s", patho_folder)
            return {"results": []}

        patho_files = sorted(get_fcs_files(patho_folder))
        if not patho_files:
            _logger.warning("Batch: aucun FCS dans %s", patho_folder)
            return {"results": []}

        n_total = len(patho_files)
        _logger.info("=" * 60)
        _logger.info("MODE BATCH — %d moelle(s) pathologique(s)", n_total)
        _logger.info("=" * 60)

        # ── Initialisation du cache NBM ───────────────────────────────────────
        nbm_files = self._discover_nbm_files()
        if not nbm_files:
            _logger.error("Batch: aucun FCS NBM dans %s", config.paths.healthy_folder)
            return {"results": []}

        cache_dir = Path(config.paths.output_dir) / "nbm_cache"
        cache_mgr = NBMCacheManager(
            config=config,
            cache_dir=cache_dir,
            nbm_file_paths=[str(p) for p in nbm_files],
        )
        _logger.info(cache_mgr.summary())

        df_nbm, selected_markers = self._get_or_build_nbm_cache(
            cache_mgr, nbm_files
        )
        if df_nbm is None:
            _logger.error("Batch: impossible d'initialiser le cache NBM.")
            return {"results": []}

        # ── Boucle batch ──────────────────────────────────────────────────────
        results: List[Tuple[str, PipelineResult]] = []

        for idx, patho_file in enumerate(patho_files):
            stem = Path(patho_file).stem

            if progress_callback is not None:
                progress_callback(idx, n_total, Path(patho_file).name)

            _logger.info("")
            _logger.info("─── Batch %d/%d : %s ───", idx + 1, n_total, stem)
            t_iter = time.time()

            try:
                result = self._process_one(
                    patho_file=patho_file,
                    df_nbm=df_nbm,
                    selected_markers=selected_markers,
                )
            except Exception as exc:
                _logger.error(
                    "Erreur non bloquante pour %s: %s", stem, exc, exc_info=True
                )
                result = PipelineResult.failure(
                    error=f"Batch error [{stem}]: {exc}",
                    config=self.config,
                )

            elapsed = time.time() - t_iter
            _logger.info(
                "[%s] %s — %.1fs",
                "OK" if result.success else "ÉCHEC",
                stem,
                elapsed,
            )
            results.append((stem, result))

            # Sauvegarde incrémentale de l'Excel après chaque patient
            self._generate_synthesis_excel(results)
            _logger.info(
                "Batch [%d/%d] Excel intermédiaire mis à jour (%d patient(s))",
                idx + 1, n_total, len(results),
            )

            # Libération mémoire explicite entre itérations
            gc.collect()

        # ── Résumé ────────────────────────────────────────────────────────────
        total = time.time() - t_start
        n_ok = sum(1 for _, r in results if r.success)
        _logger.info("")
        _logger.info("=" * 60)
        _logger.info(
            "BATCH TERMINÉ : %d/%d fichier(s) réussis en %.1fs (moy. %.1fs/patient)",
            n_ok, n_total, total, total / max(n_total, 1),
        )
        _logger.info("=" * 60)

        excel_path = self._generate_synthesis_excel(results)
        if excel_path:
            _logger.info("Excel de synthèse : %s", excel_path)

        return {"results": results, "excel": excel_path}

    # ------------------------------------------------------------------
    # Cache NBM
    # ------------------------------------------------------------------

    def _discover_nbm_files(self) -> List[Path]:
        """Retourne la liste triée des FCS dans le dossier NBM."""
        folder = Path(getattr(self.config.paths, "healthy_folder", "") or "")
        if not folder.exists():
            return []
        return sorted(get_fcs_files(folder))

    def _get_or_build_nbm_cache(
        self,
        cache_mgr: NBMCacheManager,
        nbm_files: List[Path],
    ) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """
        Charge le cache NBM s'il existe, sinon le construit.

        Returns:
            (df_nbm, selected_markers) ou (None, []) en cas d'échec.
        """
        # Tentative de chargement
        if cache_mgr.has_cache():
            _logger.info("Cache DATA NBM présent — chargement...")
            result = cache_mgr.load()
            if result is not None:
                return result
            _logger.warning("Cache corrompu — recalcul...")

        # Recalcul
        _logger.info(
            "Preprocessing du pool NBM (%d fichiers)...", len(nbm_files)
        )
        nbm_config = self._make_nbm_only_config()

        try:
            raw_samples = load_as_flow_samples(
                [str(p) for p in nbm_files],
                condition="Sain",
            )
        except Exception as exc:
            _logger.error("Chargement FCS NBM échoué: %s", exc)
            return None, []

        if not raw_samples:
            _logger.error("Aucun échantillon NBM chargé.")
            return None, []

        _logger.info("  %d fichier(s) NBM chargé(s)", len(raw_samples))

        gating_logger = GatingLogger()
        # ARCH-4 FIX : lock pour éviter le data race GUI thread ↔ batch worker
        with _GATING_STATE_LOCK:
            gating_reports.clear()
            gating_log_entries.clear()

        try:
            processed_nbm, _, _ = preprocess_combined(
                raw_samples,
                nbm_config,
                gating_logger=gating_logger,
                gating_plot_dir=None,
            )
        except Exception as exc:
            _logger.error("Preprocessing NBM échoué: %s", exc)
            return None, []

        if not processed_nbm:
            _logger.error("Aucun échantillon NBM valide après preprocessing.")
            return None, []

        # Déterminer les marqueurs pour le SOM
        all_markers = processed_nbm[0].markers if processed_nbm else []
        selected_markers = select_markers_for_clustering(all_markers, self.config)
        _logger.info(
            "  %d marqueurs sélectionnés: %s", len(selected_markers), selected_markers
        )

        # Sauvegarde + rechargement pour vérification
        cache_mgr.save(processed_nbm, selected_markers)

        # Libérer les FlowSamples bruts — le DataFrame cache suffit
        del raw_samples, processed_nbm
        gc.collect()

        result = cache_mgr.load()
        if result is None:
            _logger.error("Échec vérification post-save du cache NBM.")
            return None, []

        return result

    # ------------------------------------------------------------------
    # Traitement d'une moelle pathologique
    # ------------------------------------------------------------------

    def _process_one(
        self,
        patho_file: Path,
        df_nbm: pd.DataFrame,
        selected_markers: List[str],
    ) -> PipelineResult:
        """
        Traite une moelle pathologique en ré-entraînant le SOM de zéro.

        Étapes :
          1. Subsampler le NBM (depuis le cache RAM).
          2. Preprocesser le FCS pathologique.
          3. Reconstituer des FlowSamples combinés.
          4. Injecter dans le checkpoint gating de FlowSOMPipeline.
          5. Appeler execute() → SOM fit de zéro sur [NBM subsamplé + Patho].

        Args:
            patho_file:       Chemin du FCS pathologique.
            df_nbm:           DataFrame NBM pré-processé (depuis cache).
            selected_markers: Marqueurs retenus pour le SOM.

        Returns:
            PipelineResult complet.
        """
        patho_file = Path(patho_file)

        # ── Config pour ce run (single file) ─────────────────────────────────
        run_config = copy.deepcopy(self.config)
        run_config.paths.patho_single_file = str(patho_file)
        run_config.batch.enabled = False  # éviter récursion

        # ── Config patho-only : gate G1 débris désactivée ────────────────────
        # En mode batch, les NBM du cache sont déjà post-gating+transformation.
        # Le GMM débris ne peut pas être recalibré sur un seul FCS patho isolé
        # (seuil biaisé). On désactive G1 et G2 sur le patho seul : le gating
        # a déjà été fait sur les NBM lors de la construction du cache. Le patho
        # est directement transformé avec les mêmes paramètres que les NBM.
        patho_config = copy.deepcopy(run_config)
        patho_config.pregate.viable = False    # pas de gate G1 débris
        patho_config.pregate.singlets = False  # pas de gate G2 singlets

        gating_logger = GatingLogger()
        # ARCH-4 FIX : lock pour éviter le data race GUI thread ↔ batch worker
        with _GATING_STATE_LOCK:
            gating_reports.clear()
            gating_log_entries.clear()

        # ── 1. Chargement + preprocessing du FCS pathologique (sans gating) ──
        try:
            raw_patho = load_as_flow_samples(
                [str(patho_file)],
                condition="Pathologique",
            )
        except Exception as exc:
            return PipelineResult.failure(
                error=f"Chargement FCS patho échoué: {exc}",
                config=run_config,
            )

        if not raw_patho:
            return PipelineResult.failure(
                error="Aucun échantillon pathologique chargé.",
                config=run_config,
            )

        try:
            processed_patho, gating_figs, gating_ckpt_data = preprocess_combined(
                raw_patho,
                patho_config,   # G1/G2 désactivés
                gating_logger=gating_logger,
                gating_plot_dir=None,
            )
        except Exception as exc:
            return PipelineResult.failure(
                error=f"Preprocessing patho échoué: {exc}",
                config=run_config,
            )

        del raw_patho
        gc.collect()

        if not processed_patho:
            return PipelineResult.failure(
                error="Aucun échantillon valide après preprocessing patho.",
                config=run_config,
            )

        n_patho_cells = sum(s.n_cells for s in processed_patho)
        _logger.info("  Patho preprocessé (sans gating G1/G2): %d cellules", n_patho_cells)

        # ── 2. Subsampling du NBM avec le vrai n_patho ────────────────────────
        df_nbm_sub = self._subsample_nbm(df_nbm, patho_file, n_patho=n_patho_cells)
        _logger.info(
            "  NBM subsamplé: %d cellules (depuis %d)",
            len(df_nbm_sub), len(df_nbm),
        )

        # Reconstituer des FlowSamples NBM depuis le DataFrame subsamplé
        # (données déjà post-gating+transformation depuis le cache)
        nbm_samples = self._df_to_flow_samples(df_nbm_sub, condition="Sain")
        del df_nbm_sub
        gc.collect()

        # ── 3. Combinaison NBM post-cache + Patho post-transformation ─────────
        combined_samples = nbm_samples + processed_patho
        n_combined = sum(s.n_cells for s in combined_samples)
        _logger.info(
            "  Combiné: %d cellules totales (%d NBM + %d patho)",
            n_combined,
            sum(s.n_cells for s in nbm_samples),
            n_patho_cells,
        )

        del nbm_samples, processed_patho
        gc.collect()

        # ── 4. Injection dans le checkpoint gating ────────────────────────────
        # FlowSOMPipeline détecte le checkpoint et saute le preprocessing,
        # passant directement au SOM fit (de zéro) sur combined_samples.
        pipeline = FlowSOMPipeline(run_config)
        ckpt_mgr = PipelineCheckpointManager(run_config)

        input_files = pipeline._resolve_input_fcs_files()
        gating_key = pipeline._build_gating_cache_key(ckpt_mgr, input_files)

        # Invalider le checkpoint SOM pour forcer un nouveau fit à chaque itération
        # (le SOM ne doit PAS être réutilisé entre patients)
        som_key = pipeline._build_som_cache_key(ckpt_mgr, gating_key)
        _invalidate_som_checkpoint(ckpt_mgr, som_key)

        # Sauvegarder le checkpoint gating avec les samples combinés
        ckpt_mgr.save(
            gating_key,
            {
                "processed_samples": combined_samples,
                "gating_payload": gating_ckpt_data,
                "gating_events": [e.to_dict() for e in gating_logger.events],
                "n_total_raw": int(n_combined),
            },
        )

        del combined_samples
        gc.collect()

        # ── 5. Exécution du pipeline standard (SOM fit de zéro) ──────────────
        _logger.info("  Entraînement SOM de zéro sur données combinées...")
        result = pipeline.execute()

        return result

    # ------------------------------------------------------------------
    # Subsampling du NBM
    # ------------------------------------------------------------------

    def _subsample_nbm(
        self,
        df_nbm: pd.DataFrame,
        patho_file: Path,
        n_patho: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Subsampling du pool NBM proportionnellement au vrai n_patho.

        Stratégie (par ordre de priorité) :
          1. Si stratified_downsampling.balance_conditions=True :
             cible n_nbm = ratio × n_patho (vrai nombre de cellules patho
             après preprocessing, réparti équitablement par fichier NBM).
          2. Si downsampling.max_cells_total est défini, on borne le NBM
             à cette valeur.
          3. Sinon : retourne le DataFrame NBM complet (pas de subsampling).

        Le subsampling est proportionnel par fichier source (_sample_id) pour
        garantir une représentation équitable de chaque donneur NBM.

        Args:
            df_nbm:      DataFrame NBM complet (depuis cache).
            patho_file:  Chemin du FCS patho courant (non utilisé directement).
            n_patho:     Nombre réel de cellules patho après preprocessing.
                         Si None, fallback sur max_cells_per_file.

        Returns:
            DataFrame NBM subsamplé (ou intact si pas de subsampling).
        """
        cfg = self.config
        seed = int(cfg.flowsom.seed)
        # Copie défensive : df_nbm est partagé entre toutes les itérations batch.
        # Sans copie, un groupby ou sample() concurrent (parallélisation externe)
        # pourrait interférer avec les opérations d'indexation.
        df_nbm = df_nbm.copy()
        n_nbm_full = len(df_nbm)

        # Priorité 1 : équilibrage conditionnel via stratified_downsampling
        strat = getattr(cfg, "stratified_downsampling", None)
        if strat and getattr(strat, "balance_conditions", False):
            ratio = float(getattr(strat, "imbalance_ratio", 2.0))
            # Utiliser le vrai n_patho si disponible, sinon fallback
            if n_patho is not None and n_patho > 0:
                n_patho_ref = n_patho
            else:
                n_patho_ref = int(getattr(cfg.downsampling, "max_cells_per_file", 50_000))
                _logger.warning(
                    "  _subsample_nbm: n_patho non fourni — fallback max_cells_per_file=%d",
                    n_patho_ref,
                )
            n_target_nbm = int(ratio * n_patho_ref)

            if n_target_nbm >= n_nbm_full:
                # NBM déjà assez petit ou pas besoin de réduire
                _logger.info(
                    "  Subsampling NBM: cible=%d >= pool=%d — pas de réduction.",
                    n_target_nbm, n_nbm_full,
                )
                return df_nbm

            # Subsampling proportionnel par fichier source
            if "_sample_id" in df_nbm.columns:
                rng = np.random.default_rng(seed)
                groups = df_nbm.groupby("_sample_id", sort=False)
                n_files = groups.ngroups
                quota_per_file = max(1, n_target_nbm // n_files)
                allow_over = getattr(strat, "allow_oversampling", False)

                chunks = []
                for name, grp in groups:
                    n_dispo = len(grp)
                    rseed = int(rng.integers(0, 2**31))
                    if n_dispo >= quota_per_file:
                        chunks.append(grp.sample(n=quota_per_file, random_state=rseed))
                    elif allow_over:
                        chunks.append(grp.sample(n=quota_per_file, replace=True, random_state=rseed))
                    else:
                        chunks.append(grp)  # prendre tout ce qui est disponible

                df_sub = pd.concat(chunks, ignore_index=True)
                _logger.info(
                    "  Subsampling NBM (ratio=%.1f×, n_patho=%d): "
                    "%d → %d cellules (%d fichiers, quota=%d/fichier)",
                    ratio, n_patho_ref, n_nbm_full, len(df_sub), n_files, quota_per_file,
                )
                return df_sub
            else:
                # Fallback : tirage global sans stratification
                _logger.info(
                    "  Subsampling NBM (ratio=%.1f×): %d → %d cellules (global)",
                    ratio, n_nbm_full, n_target_nbm,
                )
                return df_nbm.sample(n=n_target_nbm, random_state=seed).reset_index(drop=True)

        # Priorité 2 : borne globale sur max_cells_total
        ds = getattr(cfg, "downsampling", None)
        if ds and getattr(ds, "enabled", False):
            max_total = int(getattr(ds, "max_cells_total", 0))
            if max_total > 0 and n_nbm_full > max_total:
                _logger.info(
                    "  Subsampling NBM (max_cells_total=%d): %d → %d cellules",
                    max_total, n_nbm_full, max_total,
                )
                return df_nbm.sample(n=max_total, random_state=seed).reset_index(drop=True)

        # Pas de subsampling
        return df_nbm

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_nbm_only_config(self) -> PipelineConfig:
        """Config sans dossier pathologique pour le preprocessing NBM seul.

        Le downsampling par fichier est désactivé : le cache NBM doit stocker
        les données complètes post-gating pour que _subsample_nbm puisse tirer
        le bon quota par patient (n_patho variable selon la moelle pathologique).
        """
        cfg = copy.deepcopy(self.config)
        cfg.paths.patho_folder = ""
        cfg.paths.patho_single_file = None
        cfg.analysis.compare_mode = False
        # Désactiver le downsampling par fichier pour le cache NBM complet
        cfg.downsampling.max_cells_per_file = 0
        return cfg

    @staticmethod
    def _df_to_flow_samples(
        df: pd.DataFrame,
        condition: str,
    ) -> List[FlowSample]:
        """
        Reconstruit des FlowSamples depuis un DataFrame (issu du cache NBM).

        Le DataFrame doit contenir une colonne '_sample_id' et des colonnes
        de marqueurs. Chaque groupe par '_sample_id' devient un FlowSample.

        Args:
            df:        DataFrame avec colonne '_sample_id'.
            condition: Label de condition à assigner ('healthy', …).

        Returns:
            Liste de FlowSamples.
        """
        _META_COLS = {"_sample_id", "_condition", "condition", "file_origin", "class", "_cell_idx"}
        marker_cols = [c for c in df.columns if c not in _META_COLS]
        samples: List[FlowSample] = []

        if "_sample_id" not in df.columns:
            # Fallback : un seul sample sans id
            samples.append(FlowSample(
                name="nbm_cached",
                path="",
                condition=condition,
                data=df[marker_cols].reset_index(drop=True),
                n_cells_raw=len(df),
            ))
            return samples

        for sample_id, group in df.groupby("_sample_id", sort=False):
            data = group[marker_cols].reset_index(drop=True)
            samples.append(FlowSample(
                name=str(sample_id),
                path="",
                condition=condition,
                data=data,
                n_cells_raw=len(data),
            ))

        return samples

    # ------------------------------------------------------------------
    # Export Excel de synthèse cohorte
    # ------------------------------------------------------------------

    def _generate_synthesis_excel(
        self, results: List[Tuple[str, "PipelineResult"]]
    ) -> Optional[str]:
        """Génère (ou met à jour) le fichier synthèse_cohorte.xlsx."""
        try:
            from src.io.csv_exporter import (
                extract_date_from_filename as _edf,
            )

            rows = []
            for stem, result in results:
                _fcs_date_str = getattr(result, "patho_date", None)
                if not _fcs_date_str:
                    _fcs_dt = _edf(stem)
                    _fcs_date_str = _fcs_dt.strftime("%Y-%m-%d") if _fcs_dt else ""

                row: Dict[str, object] = {
                    "Fichier FCS": stem,
                    "Date FCS": _fcs_date_str,
                    "Statut": "OK" if (result is not None and result.success) else "ERREUR",
                }

                if result is None or not result.success:
                    err = (
                        result.warnings[0]
                        if (result and result.warnings)
                        else "Erreur inconnue"
                    )
                    row["Erreur"] = err
                    rows.append(row)
                    continue

                row["Cellules totales"] = result.n_cells
                row["Durée (s)"] = round(result.elapsed_seconds, 1)

                mrd = result.mrd_result
                if mrd is not None:
                    n_total = mrd.total_cells or 1
                    row["Cellules patho"] = mrd.total_cells_patho
                    row["Cellules sain"] = mrd.total_cells_sain
                    row["% patho"] = round(mrd.total_cells_patho / n_total * 100, 2)
                    row["MRD % (Flo)"] = round(mrd.mrd_pct_flo, 4)
                    row["MRD % (JF)"] = round(mrd.mrd_pct_jf, 4)
                    row["MRD % (ELN)"] = round(mrd.mrd_pct_eln, 4)
                    row["Cellules MRD (Flo)"] = mrd.mrd_cells_flo
                    row["Cellules MRD (JF)"] = mrd.mrd_cells_jf
                    row["Cellules MRD (ELN)"] = mrd.mrd_cells_eln
                    row["Nœuds MRD (Flo)"] = mrd.n_nodes_mrd_flo
                    row["Nœuds MRD (JF)"] = mrd.n_nodes_mrd_jf
                    row["Nœuds MRD (ELN)"] = mrd.n_nodes_mrd_eln
                    row["ELN positif"] = "Oui" if mrd.eln_positive else "Non"
                    row["ELN bas niveau"] = "Oui" if mrd.eln_low_level else "Non"
                else:
                    if result.data is not None and "condition" in result.data.columns:
                        cond_counts = result.data["condition"].value_counts()
                        for cond, n_c in cond_counts.items():
                            row[f"Cellules {cond}"] = int(n_c)

                html = result.output_files.get("html_report", "")
                row["Rapport HTML"] = str(html) if html else ""

                rows.append(row)

            if not rows:
                return None

            df_synthesis = pd.DataFrame(rows)
            output_dir = Path(self.config.paths.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            excel_path = output_dir / "synthèse_cohorte.xlsx"

            with pd.ExcelWriter(str(excel_path), engine="openpyxl") as writer:
                df_synthesis.to_excel(writer, index=False, sheet_name="Synthèse")
                ws = writer.sheets["Synthèse"]
                for col in ws.columns:
                    max_len = max(
                        (len(str(cell.value)) for cell in col if cell.value is not None),
                        default=10,
                    )
                    ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

            _logger.info("Excel de synthèse: %s", excel_path)
            return str(excel_path)

        except Exception as exc:
            _logger.warning("Génération Excel de synthèse échouée: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Helper module-level
# ---------------------------------------------------------------------------

def _invalidate_som_checkpoint(
    ckpt_mgr: PipelineCheckpointManager,
    som_key: str,
) -> None:
    """
    Supprime le checkpoint SOM sur disque pour forcer un nouveau fit.

    À chaque itération batch, le SOM doit être ré-entraîné sur les
    données fraîches [NBM subsamplé + Patho courant]. Sans cette
    suppression, FlowSOMPipeline réutiliserait le SOM du patient précédent.
    """
    if not ckpt_mgr.enabled:
        return
    som_path = ckpt_mgr.base_dir / f"{som_key}.joblib"
    if som_path.exists():
        som_path.unlink(missing_ok=True)
        _logger.debug("Checkpoint SOM invalidé: %s", som_path.name)
