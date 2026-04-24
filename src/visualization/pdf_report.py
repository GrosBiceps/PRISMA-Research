"""pdf_report.py
Générateur de rapport PDF A4 pour FlowSOM Pipeline Pro.
Reproduit fidèlement le contenu du rapport HTML :
  • couverture, paramètres, KPI, tableaux, marqueurs, métaclusters
  • toutes les figures matplotlib (PNG inline)
  • toutes les figures Plotly → PNG statique via kaleido
  • figures larges (SOM combined, heatmap…) en pages paysage automatiquement

Dépendances :
  reportlab >= 4.0   (layout PDF)
  kaleido   >= 0.1   (Plotly → PNG)
"""
from __future__ import annotations

import io
import logging
import struct
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger("visualization.pdf_report")

# ─── Dépendances optionnelles ─────────────────────────────────────────────────
_RL = False
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.pagesizes import landscape as _rl_landscape
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        HRFlowable,
        Image,
        KeepTogether,
        NextPageTemplate,
        PageBreak,
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.platypus import SimpleDocTemplate
    _RL = True
except ImportError:
    _logger.warning("reportlab non disponible — export PDF désactivé (pip install reportlab)")

_PLOTLY = False
try:
    import plotly.io as _pio
    _PLOTLY = True
except ImportError:
    pass

_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    pass

# ─── Constantes de mise en page (pts) ────────────────────────────────────────
# Définies dès l'import pour éviter les NameError même si reportlab est absent
_AW: float = 595.28
_AH: float = 841.89
_ML = _MR = 56.69   # 2 cm
_MT = 70.87          # 2.5 cm
_MB = 51.02          # 1.8 cm
_PW = _AW - _ML - _MR   # ≈ 481 pt — largeur portrait
_PH = _AH - _MT - _MB   # ≈ 720 pt — hauteur portrait
_LW = _AH - _ML - _MR   # ≈ 728 pt — largeur paysage
_LH = _AW - _MT - _MB   # ≈ 468 pt — hauteur paysage
_HDR_H = 22.0            # hauteur du bandeau en-tête
_FTR_H = 18.0            # hauteur du bandeau pied de page
_MAX_FIG_H_P = _PH - _HDR_H - _FTR_H - 20   # ≈ 660 pt
_MAX_FIG_H_L = _LH - _HDR_H - _FTR_H - 20   # ≈ 408 pt

if _RL:
    _AW, _AH = A4
    _ML = _MR = 2.0 * cm
    _MT = 2.5 * cm
    _MB = 1.8 * cm
    _PW = _AW - _ML - _MR
    _PH = _AH - _MT - _MB
    _LW = _AH - _ML - _MR
    _LH = _AW - _MT - _MB
    _MAX_FIG_H_P = _PH - _HDR_H - _FTR_H - 20
    _MAX_FIG_H_L = _LH - _HDR_H - _FTR_H - 20

# ─── Palette ──────────────────────────────────────────────────────────────────
if _RL:
    _CP   = colors.HexColor("#667eea")   # primary purple
    _CA   = colors.HexColor("#764ba2")   # accent deep purple
    _CW   = colors.white
    _CBG  = colors.HexColor("#f0f4ff")   # section bg léger
    _CTX  = colors.HexColor("#1e293b")   # texte principal
    _CMU  = colors.HexColor("#64748b")   # texte atténué
    _CTH  = colors.HexColor("#dde4f8")   # fond entête tableau
    _CR1  = colors.HexColor("#f8fafc")   # ligne impaire tableau
    _CR2  = colors.white                 # ligne paire tableau
    _CDV  = colors.HexColor("#cbd5e1")   # séparateur

# ─── Figures considérées comme « larges » → page paysage ─────────────────────
_WIDE_KEYS = frozenset({
    "fig_som_combined", "fig_mrd_summary", "fig_grid_mc",
    "fig_heatmap", "fig_heatmap_clinical", "fig_radar", "fig_cluster_radar",
    "fig_cluster_radar_jf", "fig_cluster_radar_flo",
    "fig_blast_mrd_classification",
})

# Ordre de rendu identique au rapport HTML
_FIG_ORDER = [
    # QC Gating — matplotlib
    "fig_overview", "fig_gate_debris", "fig_gate_singlets",
    "fig_gate_cd45", "fig_gate_cd34", "fig_kde_debris", "fig_kde_cd45", "fig_cd45_count",
    # Correction batch — Harmony diagnostic (Plotly ou Matplotlib)
    "fig_harmony_diag",
    # Analyse — matplotlib
    "fig_heatmap", "fig_comp", "fig_umap", "fig_mst_static",
    "fig_star_chart", "fig_som_grid_static", "fig_barplots", "fig_heatmap_clinical",
    # Plotly → PNG statique
    "fig_sankey", "fig_mst", "fig_grid_mc", "fig_radar", "fig_cluster_radar",
    "fig_cluster_radar_jf", "fig_cluster_radar_flo",
    "fig_patho_pct", "fig_cells_pct", "fig_patho_pct_som", "fig_cells_pct_som",
    "fig_som_combined", "fig_mrd_summary",
    # Classification phénotypique blast — toujours après la MRD
    "fig_blast_mrd_classification",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers bas niveau
# ═══════════════════════════════════════════════════════════════════════════════

def _png_size(data: bytes) -> Tuple[int, int]:
    """Lit largeur × hauteur depuis l'en-tête PNG sans PIL."""
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        w = struct.unpack(">I", data[16:20])[0]
        h = struct.unpack(">I", data[20:24])[0]
        return w, h
    return 800, 600  # fallback


def _mpl_to_png(fig: Any, dpi: int = 100) -> Optional[bytes]:
    """Rend une figure matplotlib en PNG (fond blanc)."""
    if not _MPL or fig is None:
        return None
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi,
                    bbox_inches="tight", facecolor="white", edgecolor="none")
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        _logger.debug("mpl_to_png: %s", exc)
        return None


def _plotly_to_png(fig: Any, wide: bool = False) -> Optional[bytes]:
    """Rend une figure Plotly en PNG via kaleido (scope persistant)."""
    if not _PLOTLY or fig is None:
        return None
    try:
        # Réutilise le scope kaleido déjà démarré si disponible
        try:
            from src.utils.kaleido_scope import ensure_kaleido_scope
            ensure_kaleido_scope()
        except Exception:
            pass
        w = 2200 if wide else 1400
        h = 700  if wide else 800
        return _pio.to_image(fig, format="png", width=w, height=h, scale=1.0)
    except Exception as exc:
        _logger.debug("plotly_to_png: %s", exc)
        return None


def _rl_image(png_bytes: bytes, max_w: float, max_h: float) -> Optional[Any]:
    """Crée un Image ReportLab mis à l'échelle dans la zone disponible."""
    if not _RL or not png_bytes:
        return None
    try:
        pw, ph = _png_size(png_bytes)
        scale = min(max_w / pw, max_h / ph)
        img = Image(io.BytesIO(png_bytes),
                    width=pw * scale, height=ph * scale)
        img.hAlign = "CENTER"
        return img
    except Exception as exc:
        _logger.debug("rl_image: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Styles
# ═══════════════════════════════════════════════════════════════════════════════

def _build_styles() -> dict:
    from reportlab.lib.styles import ParagraphStyle
    def _ps(name: str, **kw) -> ParagraphStyle:
        return ParagraphStyle(name=name, **kw)

    return {
        "title": _ps("pdfTitle",
            fontName="Helvetica-Bold", fontSize=28,
            textColor=_CW, alignment=TA_CENTER, spaceAfter=8),
        "subtitle": _ps("pdfSubtitle",
            fontName="Helvetica", fontSize=13,
            textColor=colors.HexColor("#c7d2fe"), alignment=TA_CENTER, spaceAfter=4),
        "ts": _ps("pdfTs",
            fontName="Helvetica-Oblique", fontSize=10,
            textColor=colors.HexColor("#a5b4fc"), alignment=TA_CENTER),
        "section": _ps("pdfSection",
            fontName="Helvetica-Bold", fontSize=12,
            textColor=_CP, spaceBefore=4, spaceAfter=4),
        "body": _ps("pdfBody",
            fontName="Helvetica", fontSize=9,
            textColor=_CTX, spaceAfter=4, leading=14),
        "small": _ps("pdfSmall",
            fontName="Helvetica", fontSize=8,
            textColor=_CMU, spaceAfter=3, leading=12),
        "caption": _ps("pdfCaption",
            fontName="Helvetica-Oblique", fontSize=8,
            textColor=_CMU, alignment=TA_CENTER, spaceBefore=4, spaceAfter=8),
        "th": _ps("pdfTh",
            fontName="Helvetica-Bold", fontSize=8.5, textColor=_CTX),
        "td": _ps("pdfTd",
            fontName="Helvetica", fontSize=8.5, textColor=_CTX),
        "kpi_val": _ps("pdfKpiVal",
            fontName="Helvetica-Bold", fontSize=20,
            textColor=_CP, alignment=TA_CENTER),
        "kpi_lbl": _ps("pdfKpiLbl",
            fontName="Helvetica", fontSize=8,
            textColor=_CMU, alignment=TA_CENTER),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Page callbacks (en-tête + pied de page)
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_header_footer(canvas: Any, doc: Any, page_w: float, page_h: float) -> None:
    canvas.saveState()
    # ── Bandeau en-tête ──────────────────────────────────────────────────────
    canvas.setStrokeColor(_CP)
    canvas.setLineWidth(1.8)
    canvas.line(_ML, page_h - _MT + 8, page_w - _MR, page_h - _MT + 8)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(_CP)
    canvas.drawString(_ML, page_h - _MT + 10, "FlowSOM Pipeline Pro")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(_CMU)
    canvas.drawRightString(page_w - _MR, page_h - _MT + 10, f"Page {doc.page}")
    # ── Pied de page ─────────────────────────────────────────────────────────
    canvas.setStrokeColor(_CDV)
    canvas.setLineWidth(0.5)
    canvas.line(_ML, _MB - 4, page_w - _MR, _MB - 4)
    canvas.setFont("Helvetica-Oblique", 7)
    canvas.setFillColor(_CMU)
    canvas.drawCentredString(page_w / 2, _MB - 14,
                             "Rapport généré automatiquement — FlowSOM Pipeline Pro")
    canvas.restoreState()


def _cb_portrait(canvas: Any, doc: Any) -> None:
    _draw_header_footer(canvas, doc, _AW, _AH)


def _cb_landscape(canvas: Any, doc: Any) -> None:
    # En paysage, A4 est retournée : largeur = _AH, hauteur = _AW
    _draw_header_footer(canvas, doc, _AH, _AW)


# ═══════════════════════════════════════════════════════════════════════════════
#  Composants UI réutilisables
# ═══════════════════════════════════════════════════════════════════════════════

def _section_header(title: str, num: int, styles: dict) -> Any:
    """Entête de section avec bordure gauche colorée."""
    tbl = Table(
        [[Paragraph(f"<b>{num}. {title}</b>", styles["section"])]],
        colWidths=[_PW],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), _CBG),
        ("LINEBEFORE",   (0, 0), (0, 0),   4, _CP),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS",(0, 0),(-1, -1), [_CBG]),
    ]))
    return tbl


def _kv_table(rows: List[Tuple[str, str]], styles: dict,
              col_w: Tuple[float, float] = (160, None)) -> Any:
    """Table deux colonnes clé → valeur."""
    w2 = _PW - col_w[0]
    data = [[Paragraph(str(k), styles["th"]),
             Paragraph(str(v), styles["td"])]
            for k, v in rows]
    if not data:
        return Spacer(1, 4)
    tbl = Table(data, colWidths=[col_w[0], w2])
    style = [
        ("BACKGROUND",   (0, 0), (0, -1), _CTH),
        ("GRID",         (0, 0), (-1, -1), 0.5, _CDV),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i in range(len(data)):
        bg = _CR1 if i % 2 == 0 else _CR2
        style.append(("BACKGROUND", (1, i), (1, i), bg))
    tbl.setStyle(TableStyle(style))
    return tbl


def _data_table(headers: List[str], rows: List[List[str]],
                styles: dict, col_widths: Optional[List[float]] = None) -> Any:
    """Table générique avec entête colorée et lignes alternées."""
    if not rows:
        return Paragraph("(aucune donnée)", styles["small"])
    n = len(headers)
    if col_widths is None:
        col_widths = [_PW / n] * n
    head_row = [Paragraph(h, styles["th"]) for h in headers]
    body_rows = [[Paragraph(str(c), styles["td"]) for c in row] for row in rows]
    tbl = Table([head_row] + body_rows, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND",    (0, 0), (-1, 0),  _CTH),
        ("GRID",          (0, 0), (-1, -1), 0.5, _CDV),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i, _ in enumerate(body_rows):
        bg = _CR1 if i % 2 == 0 else _CR2
        style.append(("BACKGROUND", (0, i + 1), (-1, i + 1), bg))
    tbl.setStyle(TableStyle(style))
    return tbl


# ═══════════════════════════════════════════════════════════════════════════════
#  Sections
# ═══════════════════════════════════════════════════════════════════════════════

def _cover_page(styles: dict, timestamp: str, n_cells: int = 0,
                n_files: int = 0) -> List[Any]:
    """Page de couverture avec bloc header coloré pleine largeur."""
    cover_data = [[
        Paragraph("FlowSOM Pipeline Pro", styles["title"]),
        Paragraph("Rapport d'Analyse — Cytométrie en Flux", styles["subtitle"]),
        Paragraph(f"Généré le {timestamp}", styles["ts"]),
    ]]
    cover_tbl = Table(
        [[Paragraph("FlowSOM Pipeline Pro", styles["title"])],
         [Paragraph("Rapport d'Analyse — Cytométrie en Flux", styles["subtitle"])],
         [Paragraph(f"Généré le {timestamp}", styles["ts"])]],
        colWidths=[_PW],
    )
    cover_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), colors.HexColor("#1e1e3f")),
        ("TOPPADDING",   (0, 0), (-1, -1), 22),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 22),
        ("LEFTPADDING",  (0, 0), (-1, -1), 24),
        ("RIGHTPADDING", (0, 0), (-1, -1), 24),
        ("LINEBELOW",    (0, -1), (-1, -1), 3, _CA),
    ]))
    items: List[Any] = [Spacer(1, 60), cover_tbl, Spacer(1, 30)]

    # Bloc résumé sous la couverture
    if n_cells or n_files:
        kv = []
        if n_cells:
            kv.append(("Cellules analysées", f"{n_cells:,}"))
        if n_files:
            kv.append(("Fichiers FCS", str(n_files)))
        items.append(_kv_table(kv, styles))

    items += [
        Spacer(1, 20),
        HRFlowable(width=_PW, color=_CDV, thickness=0.5),
        Spacer(1, 8),
        Paragraph(
            "Ce rapport reproduit fidèlement l'ensemble des analyses et "
            "visualisations du rapport HTML interactif.",
            styles["small"],
        ),
    ]
    return items


def _section_mrd_curation(
    styles: dict,
    curated_pct: float,
    algo_gauges: List[Dict[str, Any]],
    curated_cells: Optional[int],
    n_curated_nodes: Optional[int],
    num: int = 0,
) -> List[Any]:
    """
    Encadré « MRD Validée par l'Expert » en tête d'affiche.

    Affiché uniquement quand curated_pct est renseigné.
    La valeur algorithmique brute reste affichée en-dessous pour la traçabilité.
    """
    if not _RL:
        return []

    _green  = colors.HexColor("#166534")   # texte vert foncé
    _green_bg = colors.HexColor("#f0fdf4") # fond vert très clair
    _green_bd = colors.HexColor("#86efac") # bordure vert clair
    _accent   = colors.HexColor("#15803d") # accent ligne gauche

    items: List[Any] = []

    # ── Titre de section ─────────────────────────────────────────────────────
    if num:
        items.append(_section_header("Résultat MRD Validé par l'Expert", num, styles))
        items.append(Spacer(1, 8))

    from reportlab.lib.styles import ParagraphStyle as _PS

    _pct_style = _PS("mrdCuratedPct",
        fontName="Helvetica-Bold", fontSize=28,
        textColor=_accent, alignment=TA_CENTER,
    )
    _label_style = _PS("mrdCuratedLabel",
        fontName="Helvetica-Bold", fontSize=9,
        textColor=_green, alignment=TA_CENTER, spaceAfter=2,
    )
    _sub_style = _PS("mrdCuratedSub",
        fontName="Helvetica", fontSize=8,
        textColor=colors.HexColor("#4b5563"), alignment=TA_CENTER,
    )
    _trace_style = _PS("mrdAlgoTrace",
        fontName="Helvetica-Oblique", fontSize=8,
        textColor=colors.HexColor("#6b7280"), spaceAfter=2,
    )

    # ── Valeur curée (grande) ─────────────────────────────────────────────────
    curated_data = [
        [Paragraph("✓ MRD VALIDÉE PAR L'EXPERT", _label_style)],
        [Paragraph(f"{curated_pct:.4f} %", _pct_style)],
    ]
    sub_parts = []
    if curated_cells is not None:
        sub_parts.append(f"{curated_cells:,} cellules MRD")
    if n_curated_nodes is not None:
        sub_parts.append(f"{n_curated_nodes} nœud(s) validé(s)")
    if sub_parts:
        curated_data.append([Paragraph("  ·  ".join(sub_parts), _sub_style)])

    curated_tbl = Table(curated_data, colWidths=[_PW])
    curated_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), _green_bg),
        ("BOX",           (0, 0), (-1, -1), 1.5, _green_bd),
        ("LINEBEFORE",    (0, 0), (0, -1),  5,   _accent),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 16),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    items.append(curated_tbl)
    items.append(Spacer(1, 8))

    # ── Valeurs algorithmiques brutes (traçabilité) ───────────────────────────
    if algo_gauges:
        items.append(Paragraph(
            "<b>Valeurs algorithmiques brutes (référence / traçabilité)</b>",
            _trace_style,
        ))
        algo_rows = [
            [g.get("method", "?"), f"{g.get('pct', 0.0):.4f} %",
             f"{g.get('n_cells', 0):,}", "POSITIF" if g.get("positive") else "négatif"]
            for g in algo_gauges
        ]
        items.append(_data_table(
            ["Méthode", "MRD Algo (%)", "Cellules", "Statut"],
            algo_rows, styles,
            col_widths=[100, 130, 130, 121],
        ))
        items.append(Spacer(1, 6))

    return items


def _section_params(styles: dict, params: Dict[str, Any],
                    num: int = 1) -> List[Any]:
    items: List[Any] = [_section_header("Paramètres de l'Analyse", num, styles),
                        Spacer(1, 6)]
    rows = [(str(k), str(v)) for k, v in params.items()]
    items.append(_kv_table(rows, styles))
    items.append(Spacer(1, 10))
    return items


def _section_summary(styles: dict, stats: Dict[str, Any],
                     condition_data: Optional[List[Dict]], files_data: Optional[List[Dict]],
                     num: int = 2) -> List[Any]:
    items: List[Any] = [_section_header("Résumé des Données", num, styles),
                        Spacer(1, 6)]
    n_cells   = stats.get("n_cells", 0)
    n_markers = stats.get("n_markers", 0)
    n_files   = stats.get("n_files", 0)
    n_clusters= stats.get("n_clusters", 0)

    # KPI : 4 cellules dans un tableau
    kpi_vals = [
        (f"{n_cells:,}", "Cellules totales"),
        (str(n_markers),  "Marqueurs"),
        (str(n_files),    "Fichiers"),
        (str(n_clusters), "Métaclusters"),
    ]
    kpi_data = [[Paragraph(v, styles["kpi_val"]) for v, _ in kpi_vals],
                [Paragraph(l, styles["kpi_lbl"]) for _, l in kpi_vals]]
    w_kpi = _PW / 4
    kpi_tbl = Table(kpi_data, colWidths=[w_kpi] * 4)
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), _CBG),
        ("BOX",          (0, 0), (-1, -1), 0.8, _CDV),
        ("INNERGRID",    (0, 0), (-1, -1), 0.5, _CDV),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
    ]))
    items += [kpi_tbl, Spacer(1, 10)]

    # Répartition par condition
    if condition_data:
        items.append(Paragraph("<b>Répartition par condition</b>", styles["body"]))
        rows = [[d.get("condition", ""), f"{d.get('n_cells', 0):,}",
                 f"{d.get('pct', 0):.1f} %"] for d in condition_data]
        items.append(_data_table(["Condition", "Cellules", "%"], rows, styles,
                                  col_widths=[200, 150, 131]))
        items.append(Spacer(1, 8))

    # Répartition par fichier
    if files_data:
        items.append(Paragraph("<b>Répartition par fichier</b>", styles["body"]))
        rows = [[d.get("file", ""), f"{d.get('n_cells', 0):,}"] for d in files_data]
        items.append(_data_table(["Fichier FCS", "Cellules"], rows, styles,
                                  col_widths=[360, 121]))
        items.append(Spacer(1, 8))

    return items


def _section_markers(styles: dict, markers: List[str], num: int = 3) -> List[Any]:
    items: List[Any] = [_section_header("Marqueurs Utilisés", num, styles),
                        Spacer(1, 6)]
    if markers:
        text = "  •  ".join(markers)
        items.append(Paragraph(text, styles["body"]))
    else:
        items.append(Paragraph("(aucun marqueur)", styles["small"]))
    items.append(Spacer(1, 10))
    return items


def _section_metaclusters(styles: dict, mc_table: List[Dict[str, Any]],
                           num: int = 4) -> List[Any]:
    items: List[Any] = [_section_header("Résumé des Métaclusters", num, styles),
                        Spacer(1, 6)]
    if not mc_table:
        items.append(Paragraph("(aucune donnée)", styles["small"]))
        return items
    headers = ["Métacluster", "Cellules", "%", "Top 3 marqueurs"]
    rows: List[List[str]] = []
    for r in mc_table:
        mc  = str(r.get("metacluster", ""))
        n   = f"{r.get('n_cells', 0):,}"
        pct = f"{r.get('pct', 0):.1f} %"
        top = ", ".join(str(m) for m in r.get("top_markers", [])[:3])
        rows.append([mc, n, pct, top])
    items.append(_data_table(headers, rows, styles,
                              col_widths=[90, 80, 70, 241]))
    items.append(Spacer(1, 10))
    return items


def _section_figures(styles: dict,
                     plotly_figs: Dict[str, Any],
                     mpl_figs: Dict[str, Any],
                     figure_labels: Dict[str, str],
                     num: int = 5,
                     dpi_mpl: int = 100) -> List[Any]:
    """Génère les flowables pour toutes les figures dans l'ordre HTML."""
    items: List[Any] = [_section_header("Visualisations", num, styles)]

    # Construire un dict unifié : clé → (source, objet_figure)
    all_figs: Dict[str, Tuple[str, Any]] = {}
    for k, f in mpl_figs.items():
        all_figs[k] = ("mpl", f)
    for k, f in plotly_figs.items():
        if k not in all_figs:  # mpl a priorité si même clé (rare)
            all_figs[k] = ("plotly", f)

    rendered_keys: List[str] = []
    # Ordre prédéfini en premier, puis les clés restantes
    for key in _FIG_ORDER:
        if key in all_figs:
            rendered_keys.append(key)
    for key in all_figs:
        if key not in rendered_keys:
            rendered_keys.append(key)

    kaleido_missing = not _PLOTLY  # si plotly absent, on ne peut rien faire
    if not kaleido_missing:
        try:
            _pio.to_image  # basic check
        except AttributeError:
            kaleido_missing = True

    for key in rendered_keys:
        source, fig = all_figs[key]
        label = figure_labels.get(key, key)
        wide  = key in _WIDE_KEYS

        # ── Conversion → PNG bytes ───────────────────────────────────────────
        if source == "mpl":
            png = _mpl_to_png(fig, dpi=dpi_mpl)
        else:
            if kaleido_missing:
                items += [
                    Spacer(1, 6),
                    Paragraph(
                        f"⚠ Figure interactive « {label} » — "
                        "kaleido requis pour l'export PDF "
                        "(pip install kaleido).",
                        styles["small"],
                    ),
                ]
                continue
            png = _plotly_to_png(fig, wide=wide)

        if not png:
            items += [
                Spacer(1, 4),
                Paragraph(f"(figure non disponible : {label})", styles["small"]),
            ]
            continue

        # ── Mise en page ─────────────────────────────────────────────────────
        if wide:
            # Page paysage dédiée
            max_w, max_h = _LW, _MAX_FIG_H_L
        else:
            max_w, max_h = _PW, _MAX_FIG_H_P

        img = _rl_image(png, max_w, max_h)
        if img is None:
            continue

        caption = Paragraph(label, styles["caption"])

        if wide:
            items += [
                NextPageTemplate("landscape"),
                PageBreak(),
                img,
                caption,
                NextPageTemplate("portrait"),
                PageBreak(),
            ]
        else:
            items += [
                Spacer(1, 8),
                KeepTogether([img, caption]),
            ]

    return items


def _section_exports(styles: dict, export_paths: Dict[str, str],
                     num: int = 6) -> List[Any]:
    items: List[Any] = [_section_header("Fichiers Exportés", num, styles),
                        Spacer(1, 6)]
    rows = [[k, str(v)] for k, v in export_paths.items()]
    items.append(_data_table(["Type", "Chemin"], rows, styles,
                              col_widths=[130, 351]))
    return items


def _section_ransac(styles: dict, ransac_summary: Dict[str, Any],
                    num: int = 7) -> List[Any]:
    """Section résumé des modèles RANSAC (singlets) avec coefficients."""
    items: List[Any] = [_section_header("Résumé des Modèles RANSAC (Singlets)", num, styles),
                        Spacer(1, 4)]
    if _RL:
        items.append(Paragraph(
            "Régression linéaire robuste FSC-H → FSC-A par fichier. "
            "R² mesure la qualité de la corrélation (objectif > 0.85).",
            styles["small"],
        ))
        items.append(Spacer(1, 6))

    rows = []
    for fname, rdata in ransac_summary.items():
        r2_val = rdata.get("r2", float("nan"))
        slope_val = rdata.get("slope", float("nan"))
        intercept_val = rdata.get("intercept", float("nan"))
        pct_val = rdata.get("pct_singlets", rdata.get("pct", None))
        r2_str = f"{r2_val:.4f}" if isinstance(r2_val, float) and r2_val == r2_val else "N/A"
        slope_str = f"{slope_val:.4f}" if isinstance(slope_val, float) and slope_val == slope_val else "N/A"
        intercept_str = f"{intercept_val:.4f}" if isinstance(intercept_val, float) and intercept_val == intercept_val else "N/A"
        pct_str = f"{pct_val:.1f}%" if pct_val is not None else "N/A"
        rows.append([str(fname), slope_str, intercept_str, r2_str, pct_str])

    items.append(_data_table(
        ["Fichier", "Pente (slope)", "Intercept", "R² (corrélation)", "% Singlets"],
        rows,
        styles,
        col_widths=[190, 70, 70, 80, 71],
    ))
    return items


# ═══════════════════════════════════════════════════════════════════════════════
#  Point d'entrée principal
# ═══════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(
    output_path: "Path | str",
    *,
    plotly_figures: Optional[Dict[str, Any]] = None,
    matplotlib_figures: Optional[Dict[str, Any]] = None,
    figure_labels: Optional[Dict[str, str]] = None,
    analysis_params: Optional[Dict[str, Any]] = None,
    summary_stats: Optional[Dict[str, Any]] = None,
    metacluster_table: Optional[List[Dict[str, Any]]] = None,
    markers: Optional[List[str]] = None,
    condition_data: Optional[List[Dict[str, Any]]] = None,
    files_data: Optional[List[Dict[str, Any]]] = None,
    export_paths: Optional[Dict[str, str]] = None,
    timestamp: str = "",
    patho_info: Optional[Dict[str, str]] = None,
    dpi_mpl: int = 100,
    ransac_summary: Optional[Dict[str, Any]] = None,
    # ── Curation humaine (optionnelle) ──────────────────────────────────
    curated_mrd_percent: Optional[float] = None,
    curated_mrd_cells:   Optional[int]   = None,
    curated_nodes:       Optional[List[Dict[str, Any]]] = None,
    algo_gauges:         Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Génère un rapport PDF A4 complet reproduisant le rapport HTML.

    Dépendances :
        reportlab >= 4.0  (obligatoire)
        kaleido   >= 0.1  (pour les figures Plotly — facultatif mais recommandé)

    Returns:
        Chemin du fichier PDF si succès, None sinon.
    """
    if not _RL:
        _logger.error("reportlab non installé — impossible de générer le PDF. "
                      "Installez-le : pip install reportlab")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not timestamp:
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")

    plotly_figures    = plotly_figures    or {}
    matplotlib_figures= matplotlib_figures or {}
    figure_labels     = figure_labels     or {}
    analysis_params   = analysis_params   or {}
    summary_stats     = summary_stats     or {}
    metacluster_table = metacluster_table or []
    markers           = markers           or []
    condition_data    = condition_data    or []
    files_data        = files_data        or []
    export_paths      = export_paths      or {}

    n_cells = summary_stats.get("n_cells", 0)
    n_files = summary_stats.get("n_files", 0)

    styles = _build_styles()

    # ── Story (séquence de flowables) ────────────────────────────────────────
    story: List[Any] = []
    story.extend(_cover_page(styles, timestamp, n_cells, n_files))
    story.append(PageBreak())

    # ── Encadré moelle pathologique (en haut, avant les paramètres) ──────────
    if patho_info and _RL:
        from reportlab.lib.styles import ParagraphStyle as _PStyle
        _pname = patho_info.get("name", "")
        _pdate = patho_info.get("date", "Date inconnue")
        _warning_color = colors.HexColor("#f59e0b")
        _warning_bg    = colors.HexColor("#fff9e6")
        _warning_text  = colors.HexColor("#78350f")
        _warning_label = colors.HexColor("#92400e")
        patho_banner_data = [
            [Paragraph("&#9888; MOELLE PATHOLOGIQUE ANALYSÉE", _PStyle(
                "pdfPBTitle",
                fontName="Helvetica-Bold", fontSize=9,
                textColor=_warning_label, spaceAfter=4,
            ))],
            [Paragraph(f"<b>{_pname}</b>", _PStyle(
                "pdfPBName",
                fontName="Helvetica-Bold", fontSize=11,
                textColor=_warning_text, spaceAfter=3,
            ))],
            [Paragraph(f"Date du prélèvement : <b>{_pdate}</b>", _PStyle(
                "pdfPBDate",
                fontName="Helvetica", fontSize=10,
                textColor=_warning_text,
            ))],
        ]
        patho_tbl = Table(patho_banner_data, colWidths=[_PW])
        patho_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), _warning_bg),
            ("BOX",           (0, 0), (-1, -1), 1.5, _warning_color),
            ("LINEBEFORE",    (0, 0), (0, -1),  5,   colors.HexColor("#d97706")),
            ("LEFTPADDING",   (0, 0), (-1, -1), 14),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story += [patho_tbl, Spacer(1, 16)]

    # ── Encadré MRD Validée par l'Expert (affiché si curation disponible) ────
    if curated_mrd_percent is not None and _RL:
        n_nodes_curated = len(curated_nodes) if curated_nodes else None
        story.extend(_section_mrd_curation(
            styles,
            curated_pct=curated_mrd_percent,
            algo_gauges=algo_gauges or [],
            curated_cells=curated_mrd_cells,
            n_curated_nodes=n_nodes_curated,
            num=0,   # pas de numéro : affiché comme encadré avant §1
        ))
        story.append(Spacer(1, 8))

    story.extend(_section_params(styles, analysis_params, num=1))
    story.extend(_section_summary(styles, summary_stats,
                                   condition_data, files_data, num=2))
    story.extend(_section_markers(styles, markers, num=3))
    story.extend(_section_metaclusters(styles, metacluster_table, num=4))
    story.extend(_section_figures(styles, plotly_figures,
                                   matplotlib_figures, figure_labels, num=5,
                                   dpi_mpl=dpi_mpl))
    if export_paths:
        story.extend(_section_exports(styles, export_paths, num=6))
    if ransac_summary:
        story.extend(_section_ransac(styles, ransac_summary, num=7))

    # ── Document avec templates portrait + paysage ────────────────────────────
    doc = BaseDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=_ML, rightMargin=_MR,
        topMargin=_MT, bottomMargin=_MB,
        title="FlowSOM Pipeline Pro — Rapport d'Analyse",
        author="FlowSOM Pipeline Pro",
    )

    portrait_frame = Frame(
        _ML, _MB, _PW, _PH,
        id="portrait_frame", showBoundary=0,
    )
    # Paysage : l'A4 est retournée — x0 et y0 bougent aussi
    landscape_frame = Frame(
        _MB, _ML, _LW, _LH,
        id="landscape_frame", showBoundary=0,
    )

    doc.addPageTemplates([
        PageTemplate(id="portrait",  frames=[portrait_frame],
                     pagesize=A4,
                     onPage=_cb_portrait),
        PageTemplate(id="landscape", frames=[landscape_frame],
                     pagesize=_rl_landscape(A4),
                     onPage=_cb_landscape),
    ])

    try:
        doc.build(story)
    except Exception as exc:
        _logger.error("Erreur construction PDF : %s", exc, exc_info=True)
        return None

    size_mb = output_path.stat().st_size / 1_048_576
    _logger.info("Rapport PDF généré : %s (%.1f Mo)", output_path, size_mb)
    return str(output_path)
