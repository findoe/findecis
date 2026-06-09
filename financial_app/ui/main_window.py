from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from financial_app.config import (
    ALL_INDUSTRIES_VALUE,
    APP_TITLE,
    CATEGORIES,
    DATA_PATH,
    ERROR_TITLE,
    FIELD_LABELS,
    FONT_FAMILY,
    MODEL_PATH,
    MUTED_TEXT_COLOR,
    SCALER_PATH,
    SEARCH_CONTROL_WIDTH,
    SEARCH_PANEL_WIDTH,
)
from financial_app.repository import CompanyDataRepository
from financial_app.services import FinancialModelService
from financial_app.ui.result_dialog import ResultDialog
from financial_app.validation import (
    InputValidationError,
    validate_inn,
    validate_numeric,
    validate_year,
)


class FinancialAnalysisWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.repository = CompanyDataRepository(DATA_PATH)
        self.model_service = FinancialModelService(MODEL_PATH, SCALER_PATH)

        self.entries: list[QLineEdit] = []
        self.inn_entry: QLineEdit | None = None
        self.year_entry: QLineEdit | None = None
        self.industry_combo: QComboBox | None = None
        self.status_label: QLabel | None = None

        self.setWindowTitle(APP_TITLE)
        self.resize(1220, 640)
        self.setMinimumSize(1050, 560)
        self._build_ui()

    def _build_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(14)
        main_layout.addLayout(left_layout, stretch=1)

        self._build_tabs(left_layout)
        self._build_analyze_button(left_layout)
        self._build_search_panel(main_layout)

    def _build_tabs(self, parent_layout: QVBoxLayout) -> None:
        tabs = QTabWidget()
        label_index = 0

        for category, fields_count in CATEGORIES:
            tab = QWidget()
            tab_layout = QGridLayout(tab)
            tab_layout.setContentsMargins(16, 14, 16, 14)
            tab_layout.setHorizontalSpacing(14)
            tab_layout.setVerticalSpacing(9)
            tab_layout.setColumnStretch(0, 1)
            tab_layout.setColumnStretch(1, 0)

            for row_index in range(fields_count):
                label = QLabel(f"{label_index + 1}. {FIELD_LABELS[label_index]}")
                label.setFont(QFont(FONT_FAMILY, 13, QFont.Weight.Bold))

                entry = QLineEdit()
                entry.setPlaceholderText(f"x{label_index + 1}")
                entry.setFixedWidth(150)
                entry.setFont(QFont(FONT_FAMILY, 13))

                tab_layout.addWidget(label, row_index, 0)
                tab_layout.addWidget(entry, row_index, 1)

                self.entries.append(entry)
                label_index += 1

            tabs.addTab(tab, category)

        parent_layout.addWidget(tabs, stretch=1)

    def _build_analyze_button(self, parent_layout: QVBoxLayout) -> None:
        analyze_button = QPushButton("Выполнить анализ")
        analyze_button.setFixedSize(190, 42)
        analyze_button.setFont(QFont(FONT_FAMILY, 14, QFont.Weight.Bold))
        analyze_button.clicked.connect(self.predict)
        parent_layout.addWidget(analyze_button, alignment=Qt.AlignmentFlag.AlignHCenter)

    def _build_search_panel(self, parent_layout: QHBoxLayout) -> None:
        search_frame = QFrame()
        search_frame.setObjectName("SearchPanel")
        search_frame.setFixedWidth(SEARCH_PANEL_WIDTH)

        layout = QVBoxLayout(search_frame)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(8)

        title = QLabel("Поиск предприятия")
        title.setFont(QFont(FONT_FAMILY, 16, QFont.Weight.Bold))
        layout.addWidget(title)

        hint = QLabel("Загрузите данные из базы или заполните показатели вручную.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {MUTED_TEXT_COLOR};")
        layout.addWidget(hint)

        layout.addSpacing(8)

        layout.addWidget(self._create_field_label("Поиск по ИНН:"))
        self.inn_entry = QLineEdit()
        self.inn_entry.setPlaceholderText("10 цифр")
        self.inn_entry.setFixedWidth(SEARCH_CONTROL_WIDTH)
        layout.addWidget(self.inn_entry)

        layout.addWidget(self._create_field_label("Поиск по году:"))
        self.year_entry = QLineEdit()
        self.year_entry.setPlaceholderText("например, 2009")
        self.year_entry.setFixedWidth(SEARCH_CONTROL_WIDTH)
        layout.addWidget(self.year_entry)

        layout.addWidget(self._create_field_label("Отрасль:"))
        self.industry_combo = QComboBox()
        self.industry_combo.addItems([ALL_INDUSTRIES_VALUE] + self.repository.industries)
        self.industry_combo.setFixedWidth(SEARCH_CONTROL_WIDTH)
        layout.addWidget(self.industry_combo)

        search_button = QPushButton("Найти")
        search_button.clicked.connect(self.search)
        layout.addWidget(search_button)

        random_button = QPushButton("Случайно")
        random_button.clicked.connect(self.random_company)
        layout.addWidget(random_button)

        clear_button = QPushButton("Очистить")
        clear_button.setObjectName("SecondaryButton")
        clear_button.clicked.connect(self.clear_form)
        layout.addWidget(clear_button)

        layout.addStretch(1)

        self.status_label = QLabel("Готово к работе")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {MUTED_TEXT_COLOR};")
        layout.addWidget(self.status_label)

        parent_layout.addWidget(search_frame)

    @staticmethod
    def _create_field_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setFont(QFont(FONT_FAMILY, 12, QFont.Weight.Bold))
        return label

    def predict(self) -> None:
        try:
            feature_values = self._collect_feature_values_from_entries()
            regression_values, probability = self.model_service.predict(feature_values)
            dialog = ResultDialog(regression_values, probability, self)
            dialog.exec()
        except InputValidationError as error:
            self._show_error(str(error))
        except Exception as error:
            self._show_error(str(error))

    def search(self) -> None:
        inn = self._get_inn_value()
        year = self._get_year_value()

        if not validate_inn(inn):
            self._show_error("ИНН должен содержать 10 цифр!")
            return

        if year and not validate_year(year):
            self._show_error("Год должен содержать 4 цифры!")
            return

        row = self.repository.find_by_inn_and_year(inn, year)
        if row is None:
            self._show_error("Данные не найдены!")
            return

        self._fill_company_data(row)
        self._set_status(
            f"Данные загружены: ИНН {self.repository.get_inn(row)}, "
            f"{self.repository.get_year(row)} год, {self.repository.get_industry(row)}."
        )

    def random_company(self) -> None:
        row = self.repository.get_random_by_industry(self._get_selected_industry())

        if row is None:
            self._show_error("В выбранной отрасли нет данных!")
            return

        self._fill_company_data(row)
        self._set_status(
            f"Случайный пример: ИНН {self.repository.get_inn(row)}, "
            f"{self.repository.get_year(row)} год, {self.repository.get_industry(row)}."
        )

    def clear_form(self) -> None:
        for entry in self.entries:
            entry.clear()

        self._require_inn_entry().clear()
        self._require_year_entry().clear()
        self._require_industry_combo().setCurrentText(ALL_INDUSTRIES_VALUE)
        self._set_status("Форма очищена")

    def _collect_feature_values_from_entries(self) -> list[float]:
        values = []

        for index, entry in enumerate(self.entries, start=1):
            value = entry.text().strip().replace(",", ".")

            if not value or not validate_numeric(value):
                entry.setFocus()
                raise InputValidationError(f"Заполните поле x{index} корректным числом!")

            values.append(float(value))

        return values

    def _fill_company_data(self, row: pd.Series) -> None:
        self._require_inn_entry().setText(self.repository.get_inn(row))
        self._require_year_entry().setText(self.repository.get_year(row))
        self._require_industry_combo().setCurrentText(self.repository.get_industry(row))
        self._fill_feature_entries(row)

    def _fill_feature_entries(self, row: pd.Series) -> None:
        for entry, value in zip(self.entries, self.repository.get_feature_values(row)):
            entry.setText(str(value))

    def _get_inn_value(self) -> str:
        return self._require_inn_entry().text().strip()

    def _get_year_value(self) -> str:
        return self._require_year_entry().text().strip()

    def _get_selected_industry(self) -> str:
        return self._require_industry_combo().currentText()

    def _set_status(self, text: str) -> None:
        if self.status_label is not None:
            self.status_label.setText(text)

    def _show_error(self, text: str) -> None:
        QMessageBox.critical(self, ERROR_TITLE, text)

    def _require_inn_entry(self) -> QLineEdit:
        if self.inn_entry is None:
            raise RuntimeError("Поле ИНН не создано")
        return self.inn_entry

    def _require_year_entry(self) -> QLineEdit:
        if self.year_entry is None:
            raise RuntimeError("Поле года не создано")
        return self.year_entry

    def _require_industry_combo(self) -> QComboBox:
        if self.industry_combo is None:
            raise RuntimeError("Список отраслей не создан")
        return self.industry_combo
