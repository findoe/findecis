from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from financial_app.config import ERROR_TITLE
from financial_app.styles import build_stylesheet
from financial_app.ui.main_window import FinancialAnalysisWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(build_stylesheet())

    try:
        window = FinancialAnalysisWindow()
    except Exception as error:
        QMessageBox.critical(None, ERROR_TITLE, str(error))
        return 1

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
