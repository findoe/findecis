from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from financial_app.config import DEFAULT_THEME, ERROR_TITLE
from financial_app.styles import build_stylesheet, set_current_theme
from financial_app.ui.main_window import FinancialAnalysisWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    set_current_theme(DEFAULT_THEME)
    app.setStyleSheet(build_stylesheet(DEFAULT_THEME))

    try:
        window = FinancialAnalysisWindow()
    except Exception as error:
        QMessageBox.critical(None, ERROR_TITLE, str(error))
        return 1

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
