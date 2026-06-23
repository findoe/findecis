from __future__ import annotations

import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QMessageBox

from src.config import DEFAULT_THEME, ERROR_TITLE, FONT_FAMILY
from src.styles import build_stylesheet, set_current_theme
from src.ui.main_window import FinancialAnalysisWindow



def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont(FONT_FAMILY, 10))
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