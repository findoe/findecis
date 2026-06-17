from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from src.config import DEFAULT_THEME, ERROR_TITLE
from src.styles import build_stylesheet, set_current_theme
from src.ui.main_window import FinancialAnalysisWindow


#Создание QApplication и главного окна
def main() -> int:
    #Инициализация Qt-приложения
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    set_current_theme(DEFAULT_THEME)
    app.setStyleSheet(build_stylesheet(DEFAULT_THEME))

    #Создание главного окна с обработкой ошибок загрузки данных и модели
    try:
        window = FinancialAnalysisWindow()
    except Exception as error:
        QMessageBox.critical(None, ERROR_TITLE, str(error))
        return 1

    window.show()
    return app.exec()


#Запуск файла напрямую
if __name__ == "__main__":
    raise SystemExit(main())