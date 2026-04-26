# src/visualization — Scope

Figures, reports, and visual summaries only. No analytical logic here.

## Contents
- `flowsom_plots.py` — SOM grid, heatmaps, MST
- `gating_plots.py` — scatter plots with gate overlays
- `mrd_radar.py` — interactive Z-score radar (Plotly), key clinical output
- `html_report.py` — self-contained HTML with embedded Plotly.js
- `pdf_report.py` — clinical PDF (ReportLab), audit-compliant
- `population_viz.py` — population distribution plots
- `plot_helpers.py` — shared plotting utilities

## Rules
- No scientific computation — plots consume prepared data, never compute core results.
- Prefer reusable plotting helpers over duplicated figure code.
- Radar plot Z-scores are pre-computed by `src/analysis/` — do not recompute here.
- PDF reports must be scan-safe and reproducible (no random layout shifts).
- HTML reports must be self-contained (no external CDN dependencies in clinical output).
- Color palettes and thresholds come from `config/constants.py`, never hardcoded.
