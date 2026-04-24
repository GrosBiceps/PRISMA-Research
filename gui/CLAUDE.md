# gui/ — Scope

PyQt5 graphical interface. Thin orchestration layer only.

## Contents
- `main_window.py` — FlowSomAnalyzerPro (5-step wizard, ~5500 lines)
- `styles.py` — Catppuccin Mocha dark theme stylesheet
- `workers.py` — QThread workers (pipeline, plotting)
- `dialogs/pipeline_dashboard.py` — progress dialog + real-time logs
- `dialogs/expert_focus_dialog.py` — expert node curation interface
- `widgets/mrd_gauge.py` — animated circular MRD gauge
- `widgets/mrd_node_table.py` — interactive node table for curation
- `adapters/mrd_adapter.py` — MRD results → GUI bridge

## Rules
- No scientific or analytical logic in this folder.
- UI orchestrates calls to io/, analysis/, and visualization/ modules.
- All long computations run in QThread workers — never block the main thread.
- Use pyqtSignal for worker → window communication (progress, results, errors).
- Theme: background #1e1e2e, accent #6366f1, text #e2e8f0 — use `styles.py` constants.
- Every new feature must be gated by a flag in `config/default_config.yaml`.
- Expert curation must produce an audit trail (GatingLogger) for clinical compliance.
- If a feature needs heavy transformation, move it to core/ or analysis/, not here.
