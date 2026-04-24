"""gui/ — Interface graphique PyQt5 pour FlowSOM Analysis Pipeline Pro."""

__all__ = ["FlowSomAnalyzerPro", "launch_gui"]


def launch_gui():
    """Point d'entrée pour lancer l'application GUI."""
    import sys
    from PyQt5.QtWidgets import QApplication
    from gui.main_window import FlowSomAnalyzerPro

    app = QApplication(sys.argv)
    window = FlowSomAnalyzerPro()
    window.show()
    sys.exit(app.exec_())
