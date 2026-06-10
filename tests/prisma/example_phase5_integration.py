"""
tests/prisma/example_phase5_integration.py
Exemple minimal d'intégration Phase 5 — PrismaGatingWorkspace avec StatisticsDock
et layout configurable.

Usage (dans un vrai contexte Qt) :
    python -m tests.prisma.example_phase5_integration
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QMainWindow


def run_example(fcs_folder: str) -> None:
    """
    Démonstration Phase 5 :
    1. Création du workspace
    2. Chargement d'un dossier FCS
    3. Ajout de 2 panels supplémentaires
    4. Changement 2 → 3 colonnes
    5. Sauvegarde JSON
    6. Rechargement JSON
    """
    from prisma.gui.viewer.gating_workspace import PrismaGatingWorkspace

    app = QApplication.instance() or QApplication(sys.argv)

    win = QMainWindow()
    workspace = PrismaGatingWorkspace()
    win.setCentralWidget(workspace)
    win.resize(1400, 900)
    win.show()

    # 1. Charger les FCS
    print(f"Chargement FCS depuis: {fcs_folder}")
    workspace.set_input_source_fcs(fcs_folder)

    # 2. Ajouter des panels (population root)
    root_node = ("root",)
    workspace._create_panel_for_gate_node(root_node)
    workspace._create_panel_for_gate_node(root_node)
    print(f"Panels: {len(workspace._worksheet.panels())}")

    # 3. Changer le nombre de colonnes : 2 → 3
    workspace.set_panel_columns(3)
    print("Layout: 3 colonnes")

    # 4. Sauvegarder
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        save_path = f.name

    saved = workspace.save_workspace_state(save_path)
    print(f"Sauvegardé: {saved}")

    # 5. Recharger
    workspace2 = PrismaGatingWorkspace()
    win2 = QMainWindow()
    win2.setCentralWidget(workspace2)
    win2.resize(1400, 900)
    win2.show()
    populations = workspace2.load_workspace_state(saved)
    print(f"Rechargé: {len(populations)} populations, colonnes={workspace2._worksheet._COLS}")

    app.exec_()


if __name__ == "__main__":
    fcs_folder = sys.argv[1] if len(sys.argv) > 1 else "."
    run_example(fcs_folder)
