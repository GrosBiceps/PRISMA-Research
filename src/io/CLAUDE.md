# src/io — Scope

Reading, validation, normalization, and export of all file formats.

## Contents
- `fcs_reader.py` — FCS discovery and loading via flowio
- `fcs_writer.py` — FCS export with clustering columns (Kaluza-compatible)
- `csv_exporter.py` — cell-level data, cluster statistics, MFI matrices
- `json_exporter.py` — metadata, gating logs, config snapshots
- `cluster_distribution_exporter.py`
- `patho_fcs_exporter.py` — pathological cells with Is_MRD flag

## Rules
- Be conservative with input assumptions — validate explicitly.
- Check schema, column names, and missing values on every FCS load.
- Never hide parsing errors silently — raise with a descriptive message.
- Compensation matrix ($SPILL) must be detected and applied here, before any analysis.
- File-format-specific logic stays isolated from analysis logic.
- No scientific computation here — only I/O and validation.
- FCS export must preserve original channel names and add new columns without renaming existing ones.
