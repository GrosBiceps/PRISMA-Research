"""performance_monitor.py
Collecte les métriques système (CPU, RAM, Disque, Réseau, GPU optionnel)
en arrière-plan pendant l'exécution de la pipeline, puis exporte un
dashboard HTML Plotly auto-contenu à la fin — style Prometheus/Grafana.

Dépendances :
  psutil  >= 5.9   (obligatoire — pip install psutil)
  plotly  >= 5.0   (déjà présent dans le projet)
  GPUtil  >= 1.4   (optionnel — pip install gputil — pour le GPU NVIDIA)

Usage :
    monitor = PerformanceMonitor(interval=1.0, include_gpu=True)
    monitor.start()
    monitor.mark_phase("Chargement FCS")
    # ... exécution pipeline ...
    monitor.mark_phase("Prétraitement")
    # ... etc ...
    monitor.stop()
    monitor.export_dashboard(output_path)
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger("monitoring.performance_monitor")

# ─── Dépendances optionnelles ─────────────────────────────────────────────────
_PSUTIL = False
try:
    import psutil
    _PSUTIL = True
except ImportError:
    _logger.warning("psutil non installé — monitoring désactivé (pip install psutil)")

_PLOTLY = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY = True
except ImportError:
    pass

_GPUTIL = False
try:
    import GPUtil  # type: ignore
    _GPUTIL = True
except ImportError:
    pass

# ─── Palette de phases (Catppuccin Mocha) ─────────────────────────────────────
# (couleur bordure, couleur remplissage semi-transparent)
_PHASE_PALETTE: List[Tuple[str, str]] = [
    ("#89b4fa", "rgba(137,180,250,0.13)"),   # blue
    ("#a6e3a1", "rgba(166,227,161,0.13)"),   # green
    ("#fab387", "rgba(250,179,135,0.13)"),   # peach
    ("#cba6f7", "rgba(203,166,247,0.13)"),   # mauve
    ("#89dceb", "rgba(137,220,235,0.13)"),   # sky
    ("#f9e2af", "rgba(249,226,175,0.13)"),   # yellow
    ("#f38ba8", "rgba(243,139,168,0.13)"),   # red
    ("#94e2d5", "rgba(148,226,213,0.13)"),   # teal
    ("#b4befe", "rgba(180,190,254,0.13)"),   # lavender
    ("#eba0ac", "rgba(235,160,172,0.13)"),   # maroon
]

# ─── Thème sombre Grafana-like ────────────────────────────────────────────────
_BG      = "#0f0e17"
_PANEL   = "#1a1b2e"
_GRID    = "#2d2d4e"
_TEXT    = "#e2e8f0"
_MUTED   = "#64748b"
_C_CPU   = "#f97316"   # orange  — CPU
_C_RAM   = "#38bdf8"   # sky     — RAM
_C_READ  = "#4ade80"   # green   — Disk read
_C_WRITE = "#f87171"   # red     — Disk write
_C_SENT  = "#facc15"   # yellow  — Net sent
_C_RECV  = "#a78bfa"   # purple  — Net recv
_C_GPU   = "#fb7185"   # rose    — GPU
_C_GMEM  = "#34d399"   # emerald — GPU memory


# ═══════════════════════════════════════════════════════════════════════════════
#  Classe principale
# ═══════════════════════════════════════════════════════════════════════════════

class PerformanceMonitor:
    """
    Collecte les métriques système en arrière-plan et exporte un dashboard HTML.

    Attributes:
        interval: Intervalle de collecte en secondes (défaut: 1.0).
        include_gpu: Active la collecte GPU via GPUtil (si disponible).
    """

    def __init__(self, interval: float = 1.0, include_gpu: bool = True) -> None:
        self.interval   = max(0.2, float(interval))
        self.include_gpu = include_gpu and _GPUTIL

        self._records: List[Dict[str, Any]] = []
        self._phases:  List[Dict[str, Any]] = []   # {name, t_start, t_end, color, fill}
        self._lock       = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0: float  = 0.0
        self._running    = False

        # Compteurs I/O de référence
        self._prev_disk: Any = None
        self._prev_net:  Any = None
        self._prev_ts:   float = 0.0

    # ── API publique ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Démarre la collecte en arrière-plan."""
        if not _PSUTIL:
            _logger.warning("psutil absent — PerformanceMonitor inactif")
            return
        if self._running:
            return

        self._t0        = time.monotonic()
        self._prev_ts   = self._t0
        self._running   = True
        self._stop_event.clear()

        # Initialisation des compteurs delta
        try:
            self._prev_disk = psutil.disk_io_counters()
            self._prev_net  = psutil.net_io_counters()
        except Exception:
            pass

        self._thread = threading.Thread(
            target=self._collect_loop,
            name="PerfMonitorThread",
            daemon=True,
        )
        self._thread.start()
        _logger.debug("PerformanceMonitor démarré (interval=%.1fs)", self.interval)

    def stop(self) -> None:
        """Arrête la collecte. Non-bloquant si le thread n'est pas actif."""
        if not self._running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 3 + 1)
        self._running = False
        _logger.debug("PerformanceMonitor arrêté — %d points collectés",
                      len(self._records))

    def mark_phase(self, name: str) -> None:
        """
        Marque le début d'une nouvelle phase de pipeline.
        La phase précédente est automatiquement fermée.
        """
        t_now = time.monotonic() - self._t0
        idx   = len(self._phases)
        color, fill = _PHASE_PALETTE[idx % len(_PHASE_PALETTE)]

        with self._lock:
            # Fermer la phase précédente
            if self._phases:
                self._phases[-1]["t_end"] = t_now
            self._phases.append({
                "name":    name,
                "t_start": t_now,
                "t_end":   None,
                "color":   color,
                "fill":    fill,
            })
        _logger.debug("Phase marquée : %s (t=%.1fs)", name, t_now)

    def export_dashboard(self, output_path: "Path | str") -> Optional[str]:
        """
        Exporte le dashboard HTML Plotly auto-contenu.

        Args:
            output_path: Chemin de sortie (.html).

        Returns:
            Chemin du fichier généré, ou None si erreur.
        """
        if not _PLOTLY:
            _logger.warning("plotly absent — export dashboard impossible")
            return None
        if not self._records:
            _logger.warning("Aucune donnée collectée — dashboard ignoré")
            return None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            fig = self._build_figure()
            html = fig.to_html(
                full_html=True,
                include_plotlyjs="cdn",
                config={"displaylogo": False, "scrollZoom": True},
            )
            output_path.write_text(html, encoding="utf-8")
            _logger.info("Dashboard performance : %s (%.0f Ko)",
                         output_path, len(html) / 1024)
            return str(output_path)
        except Exception as exc:
            _logger.error("Erreur export dashboard : %s", exc, exc_info=True)
            return None

    # ── Thread de collecte ────────────────────────────────────────────────────

    def _collect_loop(self) -> None:
        """Boucle de collecte (thread daemon)."""
        while not self._stop_event.is_set():
            t_wake = time.monotonic()
            try:
                snap = self._snapshot(t_wake)
                with self._lock:
                    self._records.append(snap)
            except Exception as exc:
                _logger.debug("Erreur snapshot: %s", exc)
            elapsed_wake = time.monotonic() - t_wake
            sleep_time   = max(0.0, self.interval - elapsed_wake)
            self._stop_event.wait(sleep_time)

    def _snapshot(self, t_wake: float) -> Dict[str, Any]:
        """Collecte un point de mesure et retourne un dict."""
        elapsed = t_wake - self._t0
        snap: Dict[str, Any] = {"elapsed_s": elapsed}

        # CPU
        snap["cpu_pct"]  = psutil.cpu_percent(interval=None)
        snap["cpu_cores"] = psutil.cpu_percent(interval=None, percpu=True)

        # RAM
        vm = psutil.virtual_memory()
        snap["ram_pct"]     = vm.percent
        snap["ram_used_gb"] = vm.used / 1_073_741_824

        # Disk I/O — taux (MB/s) depuis dernier point
        disk_read_mb = disk_write_mb = 0.0
        try:
            curr_disk = psutil.disk_io_counters()
            dt = t_wake - self._prev_ts
            if self._prev_disk is not None and dt > 0:
                disk_read_mb  = (curr_disk.read_bytes  - self._prev_disk.read_bytes)  / dt / 1e6
                disk_write_mb = (curr_disk.write_bytes - self._prev_disk.write_bytes) / dt / 1e6
            self._prev_disk = curr_disk
        except Exception:
            pass
        snap["disk_read_mb"]  = max(0.0, disk_read_mb)
        snap["disk_write_mb"] = max(0.0, disk_write_mb)

        # Network I/O — taux (MB/s)
        net_sent_mb = net_recv_mb = 0.0
        try:
            curr_net = psutil.net_io_counters()
            dt = t_wake - self._prev_ts
            if self._prev_net is not None and dt > 0:
                net_sent_mb = (curr_net.bytes_sent - self._prev_net.bytes_sent) / dt / 1e6
                net_recv_mb = (curr_net.bytes_recv - self._prev_net.bytes_recv) / dt / 1e6
            self._prev_net = curr_net
        except Exception:
            pass
        snap["net_sent_mb"] = max(0.0, net_sent_mb)
        snap["net_recv_mb"] = max(0.0, net_recv_mb)

        # GPU (optionnel)
        snap["gpu_pct"]      = None
        snap["gpu_mem_pct"]  = None
        snap["gpu_temp"]     = None
        if self.include_gpu:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    g = gpus[0]
                    snap["gpu_pct"]     = round(g.load * 100, 1)
                    snap["gpu_mem_pct"] = round(g.memoryUtil * 100, 1)
                    snap["gpu_temp"]    = g.temperature
            except Exception:
                pass

        self._prev_ts = t_wake
        return snap

    # ── Construction du dashboard ─────────────────────────────────────────────

    @staticmethod
    def _fill_opaque(rgba_str: str, new_alpha: float = 0.30) -> str:
        """Remplace l'alpha d'une couleur rgba(r,g,b,a) par new_alpha."""
        import re
        m = re.match(r"rgba\((\d+),(\d+),(\d+),[\d.]+\)", rgba_str)
        if m:
            return f"rgba({m.group(1)},{m.group(2)},{m.group(3)},{new_alpha})"
        return rgba_str

    def _build_figure(self) -> "go.Figure":
        """Construit la figure Plotly multi-panneaux avec timeline des phases."""
        with self._lock:
            records = list(self._records)
            phases  = [dict(p) for p in self._phases]

        if not records:
            return go.Figure()

        # Clore la dernière phase avec le dernier timestamp
        t_total = records[-1]["elapsed_s"]
        if phases and phases[-1]["t_end"] is None:
            phases[-1]["t_end"] = t_total

        ts           = [r["elapsed_s"]    for r in records]
        cpu_pct      = [r["cpu_pct"]      for r in records]
        ram_pct      = [r["ram_pct"]      for r in records]
        ram_gb       = [r["ram_used_gb"]  for r in records]
        disk_r       = [r["disk_read_mb"] for r in records]
        disk_w       = [r["disk_write_mb"]for r in records]
        net_s        = [r["net_sent_mb"]  for r in records]
        net_r        = [r["net_recv_mb"]  for r in records]
        gpu_pct_vals = [r.get("gpu_pct")  for r in records]
        has_gpu      = any(v is not None for v in gpu_pct_vals)

        # n_rows = métriques + 1 ligne timeline
        n_metric_rows = 5 if has_gpu else 4
        n_rows        = n_metric_rows + 1          # +1 pour la timeline
        tl_row        = n_rows                     # dernière ligne = timeline

        # Hauteurs : métriques relativement hautes, timeline fine
        if not has_gpu:
            row_h = [0.255, 0.195, 0.185, 0.185, 0.10]
        else:
            row_h = [0.205, 0.165, 0.16, 0.16, 0.16, 0.10]

        metric_subtitles = [
            "CPU (%) — utilisation globale",
            "RAM (%) — mémoire utilisée",
            "Disque I/O (MB/s) — lecture & écriture",
            "Réseau I/O (MB/s) — envoi & réception",
        ]
        if has_gpu:
            metric_subtitles.append("GPU (%) — charge & mémoire")
        subtitles = metric_subtitles + ["Chronologie des phases"]

        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes=True,
            row_heights=row_h,
            vertical_spacing=0.035,
            subplot_titles=subtitles,
        )

        # ── Phase bands sur toutes les lignes métriques ───────────────────────
        # add_vrect avec row="all" cible tous les sous-graphes y compris timeline.
        # On utilise une opacité plus visible (0.22) et un liseré plus épais.
        for ph in phases:
            t0 = ph["t_start"]
            t1 = ph.get("t_end") or (ph["t_start"] + 0.5)
            fill_visible = self._fill_opaque(ph["fill"], 0.22)
            for row in range(1, n_metric_rows + 1):
                fig.add_vrect(
                    x0=t0, x1=t1,
                    fillcolor=fill_visible,
                    line_color=ph["color"],
                    line_width=1.2,
                    layer="below",
                    row=row, col=1,
                )

        # ── Traces fantômes pour la légende des phases ────────────────────────
        for ph in phases:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                name=ph["name"],
                marker=dict(color=ph["color"], symbol="square", size=11),
                legendgroup="phases",
                legendgrouptitle=dict(text="Phases pipeline"),
                showlegend=True,
            ), row=1, col=1)

        # ── Row 1 : CPU ───────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=ts, y=cpu_pct,
            name="CPU %", mode="lines",
            line=dict(color=_C_CPU, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(249,115,22,0.15)",
            hovertemplate="t=%{x:.1f}s<br>CPU=%{y:.1f}%<extra></extra>",
            showlegend=True,
            legendgroup="metrics",
            legendgrouptitle=dict(text="Métriques"),
        ), row=1, col=1)
        fig.add_hline(y=100, line_dash="dot", line_color=_MUTED,
                      line_width=0.8, row=1, col=1)

        # ── Row 2 : RAM ───────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=ts, y=ram_pct,
            name="RAM %", mode="lines",
            line=dict(color=_C_RAM, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.15)",
            customdata=list(zip(ram_gb)),
            hovertemplate=(
                "t=%{x:.1f}s<br>RAM=%{y:.1f}%"
                " (%{customdata[0]:.1f} Go)<extra></extra>"
            ),
            showlegend=True, legendgroup="metrics",
        ), row=2, col=1)
        fig.add_hline(y=100, line_dash="dot", line_color=_MUTED,
                      line_width=0.8, row=2, col=1)

        # ── Row 3 : Disk I/O ─────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=ts, y=disk_r,
            name="Disk lecture", mode="lines",
            line=dict(color=_C_READ, width=1.5),
            fill="tozeroy", fillcolor="rgba(74,222,128,0.15)",
            hovertemplate="t=%{x:.1f}s<br>Lecture=%{y:.2f} MB/s<extra></extra>",
            showlegend=True, legendgroup="metrics",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=ts, y=disk_w,
            name="Disk écriture", mode="lines",
            line=dict(color=_C_WRITE, width=1.5),
            fill="tozeroy", fillcolor="rgba(248,113,113,0.15)",
            hovertemplate="t=%{x:.1f}s<br>Écriture=%{y:.2f} MB/s<extra></extra>",
            showlegend=True, legendgroup="metrics",
        ), row=3, col=1)

        # ── Row 4 : Network I/O ───────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=ts, y=net_s,
            name="Réseau envoi", mode="lines",
            line=dict(color=_C_SENT, width=1.5),
            fill="tozeroy", fillcolor="rgba(250,204,21,0.15)",
            hovertemplate="t=%{x:.1f}s<br>Envoi=%{y:.3f} MB/s<extra></extra>",
            showlegend=True, legendgroup="metrics",
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=ts, y=net_r,
            name="Réseau réception", mode="lines",
            line=dict(color=_C_RECV, width=1.5),
            fill="tozeroy", fillcolor="rgba(167,139,250,0.15)",
            hovertemplate="t=%{x:.1f}s<br>Réception=%{y:.3f} MB/s<extra></extra>",
            showlegend=True, legendgroup="metrics",
        ), row=4, col=1)

        # ── Row 5 : GPU (optionnel) ───────────────────────────────────────────
        if has_gpu:
            gpu_clean = [v if v is not None else 0.0 for v in gpu_pct_vals]
            gpu_mem   = [r.get("gpu_mem_pct") or 0.0 for r in records]
            fig.add_trace(go.Scatter(
                x=ts, y=gpu_clean,
                name="GPU %", mode="lines",
                line=dict(color=_C_GPU, width=1.8),
                fill="tozeroy", fillcolor="rgba(251,113,133,0.15)",
                hovertemplate="t=%{x:.1f}s<br>GPU=%{y:.1f}%<extra></extra>",
                showlegend=True, legendgroup="metrics",
            ), row=5, col=1)
            fig.add_trace(go.Scatter(
                x=ts, y=gpu_mem,
                name="GPU mémoire %", mode="lines",
                line=dict(color=_C_GMEM, width=1.5),
                fill="tozeroy", fillcolor="rgba(52,211,153,0.15)",
                hovertemplate="t=%{x:.1f}s<br>VRAM=%{y:.1f}%<extra></extra>",
                showlegend=True, legendgroup="metrics",
            ), row=5, col=1)
            fig.add_hline(y=100, line_dash="dot", line_color=_MUTED,
                          line_width=0.8, row=5, col=1)

        # ── Ligne timeline (Gantt) ─────────────────────────────────────────────
        # Chaque phase = un rectangle coloré rempli + texte centré.
        # Y-range fixé à [0, 1] — on dessine les blocs entre y=0.05 et y=0.95.
        Y_LO, Y_HI = 0.05, 0.95
        for i, ph in enumerate(phases):
            t0 = ph["t_start"]
            t1 = ph.get("t_end") or (ph["t_start"] + 0.5)
            dur = t1 - t0
            color = ph["color"]
            fill_tl = self._fill_opaque(ph["fill"], 0.55)  # bien opaque sur timeline
            mid_t = (t0 + t1) / 2

            # Rectangle rempli
            fig.add_trace(go.Scatter(
                x=[t0, t0, t1, t1, t0],
                y=[Y_LO, Y_HI, Y_HI, Y_LO, Y_LO],
                fill="toself",
                fillcolor=fill_tl,
                line=dict(color=color, width=1.5),
                mode="lines",
                showlegend=False,
                hovertemplate=(
                    f"<b>{ph['name']}</b><br>"
                    f"Début : {t0:.1f} s<br>"
                    f"Fin : {t1:.1f} s<br>"
                    f"Durée : {dur:.1f} s"
                    "<extra></extra>"
                ),
            ), row=tl_row, col=1)

            # Texte centré dans le bloc — tronquer si la phase est trop étroite
            name_short = ph["name"]
            if t_total > 0 and (dur / t_total) < 0.07:
                name_short = str(i + 1)  # numéro d'index si trop étroit
            fig.add_trace(go.Scatter(
                x=[mid_t],
                y=[0.5],
                mode="text",
                text=[f"<b>{name_short}</b>"],
                textfont=dict(size=9, color=color),
                textposition="middle center",
                showlegend=False,
                hoverinfo="skip",
            ), row=tl_row, col=1)

        # Axe Y timeline : caché (pas de ticks, pas de grille)
        fig.update_yaxes(
            range=[0, 1],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            row=tl_row, col=1,
        )

        # ── Axe X partagé ────────────────────────────────────────────────────
        fig.update_xaxes(
            title_text="Temps (secondes)", row=tl_row, col=1,
            showgrid=True, gridcolor=_GRID, gridwidth=0.5,
            color=_TEXT, tickcolor=_TEXT,
        )
        for row in range(1, tl_row):
            fig.update_xaxes(
                showgrid=True, gridcolor=_GRID, gridwidth=0.5,
                row=row, col=1,
            )

        # ── Axes Y des métriques ──────────────────────────────────────────────
        y_ranges = {1: [0, 105], 2: [0, 105]}
        for row in range(1, n_metric_rows + 1):
            fig.update_yaxes(
                range=y_ranges.get(row),
                showgrid=True, gridcolor=_GRID, gridwidth=0.5,
                color=_TEXT, tickcolor=_TEXT,
                zerolinecolor=_GRID,
                row=row, col=1,
            )

        # ── Statistiques de synthèse dans le titre ────────────────────────────
        duration = t_total
        peak_cpu = max(cpu_pct) if cpu_pct else 0.0
        peak_ram = max(ram_pct) if ram_pct else 0.0
        n_phases = len(phases)
        now_str  = datetime.now().strftime("%d/%m/%Y %H:%M")

        title_text = (
            "<b>FlowSOM Pipeline Pro — Diagnostic de Performance</b>"
            f"<br><span style='font-size:12px;color:{_MUTED}'>"
            f"{now_str}  ·  Durée : {duration:.1f} s  ·  "
            f"Peak CPU : {peak_cpu:.1f} %  ·  Peak RAM : {peak_ram:.1f} %  ·  "
            f"{n_phases} phases  ·  {len(records)} points collectés"
            f"</span>"
        )

        # ── Mise en page globale ──────────────────────────────────────────────
        fig.update_layout(
            title=dict(text=title_text, x=0.01, xanchor="left",
                       font=dict(size=15, color=_TEXT)),
            height=240 + 155 * n_metric_rows,
            paper_bgcolor=_BG,
            plot_bgcolor=_PANEL,
            font=dict(color=_TEXT, family="Inter, Helvetica, sans-serif", size=11),
            legend=dict(
                bgcolor=_PANEL,
                bordercolor=_GRID,
                borderwidth=1,
                font=dict(size=10),
                groupclick="toggleitem",
                x=1.01, y=1.0,
                xanchor="left",
            ),
            margin=dict(l=60, r=200, t=90, b=50),
            hovermode="x unified",
        )

        # Titres des sous-graphes en couleur
        for ann in fig.layout.annotations:
            ann.font.color = _MUTED
            ann.font.size  = 10

        return fig
