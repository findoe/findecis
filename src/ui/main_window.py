from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QButtonGroup,
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

from src.config import (
    ALL_INDUSTRIES_VALUE,
    APP_TITLE,
    CATEGORIES,
    DATA_PATH,
    ERROR_TITLE,
    DEFAULT_THEME,
    FIELD_LABELS,
    FONT_FAMILY,
    MODEL_PATH,
    SCALER_PATH,
    SEARCH_CONTROL_WIDTH,
    SEARCH_PANEL_WIDTH,
    THEME_DARK,
    THEME_DISPLAY_NAMES,
    THEME_LIGHT,
    WINDOW_HEIGHT,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
    WINDOW_WIDTH,
)
from src.repository import CompanyDataRepository
from src.services import FinancialModelService
from src.report_data import HistoryPoint
from src.styles import build_stylesheet, set_current_theme
from src.ui.result_dialog import ResultDialog
from src.validation import (
    InputValidationError,
    validate_inn,
    validate_year,
)



NUMERIC_INPUT_MODE = "numeric"
TEXT_INPUT_MODE = "text"

VALUE_DISPLAY_MAP = {
    0.0: "Низкая",
    0.25: "Ниже среднего",
    0.5: "Умеренная",
    0.75: "Выше среднего",
    1.0: "Высокая",
}

VALUE_OPTIONS = tuple(VALUE_DISPLAY_MAP.keys())


#Главное окно приложения
class FinancialAnalysisWindow(QMainWindow):
    #Инициализация репозитория, модели и элементов окна
    def __init__(self) -> None:
        super().__init__()
        self.repository = CompanyDataRepository(DATA_PATH)
        self.model_service = FinancialModelService(MODEL_PATH, SCALER_PATH)

        self.entries: list[QComboBox] = []
        self.input_mode = NUMERIC_INPUT_MODE
        self.mode_button_groups: list[QButtonGroup] = []
        self.numeric_mode_buttons: list[QPushButton] = []
        self.text_mode_buttons: list[QPushButton] = []
        self.inn_entry: QLineEdit | None = None
        self.year_entry: QLineEdit | None = None
        self.industry_combo: QComboBox | None = None
        self.status_label: QLabel | None = None
        self.theme_combo: QComboBox | None = None

        self.setWindowTitle(APP_TITLE)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self._build_ui()

    #Сборка основной структуры интерфейса
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

    #Создание вкладок с 35 полями ввода
    def _build_tabs(self, parent_layout: QVBoxLayout) -> None:
        tabs = QTabWidget()
        label_index = 0

        for category, fields_count in CATEGORIES:
            tab = QWidget()
            tab_layout = QGridLayout(tab)
            tab_layout.setContentsMargins(16, 10, 16, 14)
            tab_layout.setHorizontalSpacing(14)
            tab_layout.setVerticalSpacing(10)
            tab_layout.setColumnStretch(0, 1)
            tab_layout.setColumnStretch(1, 0)
            tab_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

            tab_layout.addWidget(
                self._create_input_mode_header(),
                0,
                1,
                alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            )

            block_label = QLabel("Блок:")
            block_label.setFont(QFont(FONT_FAMILY, 13, QFont.Weight.Bold))
            tab_layout.addWidget(block_label, 1, 0, alignment=Qt.AlignmentFlag.AlignLeft)

            expressiveness_label = QLabel("Выраженность:")
            expressiveness_label.setFont(QFont(FONT_FAMILY, 13, QFont.Weight.Bold))
            tab_layout.addWidget(expressiveness_label, 1, 1, alignment=Qt.AlignmentFlag.AlignLeft)

            field_row_height = max(58, min(88, 520 // max(fields_count, 1)))

            for row_index in range(fields_count):
                grid_row = row_index + 2
                tab_layout.setRowMinimumHeight(grid_row, field_row_height)

                label = QLabel(f"{label_index + 1}. {FIELD_LABELS[label_index]}")
                label.setFont(QFont(FONT_FAMILY, 15, QFont.Weight.Bold))

                entry = QComboBox()
                entry.setFixedWidth(150)
                entry.setFont(QFont(FONT_FAMILY, 14))
                self._populate_feature_combo(entry)
                entry.setCurrentIndex(-1)

                tab_layout.addWidget(label, grid_row, 0)
                tab_layout.addWidget(entry, grid_row, 1)

                self.entries.append(entry)
                label_index += 1

            tabs.addTab(tab, category)

        parent_layout.addWidget(tabs, stretch=1)

    #Переключатель формата значений внутри сетки
    def _create_input_mode_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("ModeHeader")
        header.setFixedWidth(150)

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        mode_label = QLabel("Вид:")
        mode_label.setFont(QFont(FONT_FAMILY, 13, QFont.Weight.Bold))
        header_layout.addWidget(mode_label)

        mode_switch = QFrame()
        mode_switch.setObjectName("ModeSwitch")
        mode_switch.setFixedSize(108, 32)

        mode_layout = QHBoxLayout(mode_switch)
        mode_layout.setContentsMargins(2, 2, 2, 2)
        mode_layout.setSpacing(2)

        numeric_button = QPushButton("0.0")
        text_button = QPushButton("Низкое")

        for button in (numeric_button, text_button):
            button.setObjectName("SegmentButton")
            button.setCheckable(True)
            button.setFixedHeight(28)
            button.setFont(QFont(FONT_FAMILY, 12, QFont.Weight.Bold))

        numeric_button.setChecked(self.input_mode == NUMERIC_INPUT_MODE)
        text_button.setChecked(self.input_mode == TEXT_INPUT_MODE)

        button_group = QButtonGroup(self)
        button_group.setExclusive(True)
        button_group.addButton(numeric_button)
        button_group.addButton(text_button)

        numeric_button.toggled.connect(
            lambda checked: self._change_input_mode(NUMERIC_INPUT_MODE) if checked else None
        )
        text_button.toggled.connect(
            lambda checked: self._change_input_mode(TEXT_INPUT_MODE) if checked else None
        )

        mode_layout.addWidget(numeric_button)
        mode_layout.addWidget(text_button)
        header_layout.addWidget(mode_switch)

        self.mode_button_groups.append(button_group)
        self.numeric_mode_buttons.append(numeric_button)
        self.text_mode_buttons.append(text_button)

        return header

    #Заполнение списка значениями в текущем формате
    def _populate_feature_combo(self, combo: QComboBox) -> None:
        combo.clear()

        for value in VALUE_OPTIONS:
            if self.input_mode == NUMERIC_INPUT_MODE:
                combo.addItem(f"{value:.2f}" if value in (0.25, 0.75) else f"{value:.1f}", value)
            else:
                combo.addItem(VALUE_DISPLAY_MAP[value], value)

    #Смена формата выбора значений во всех выпадающих списках
    def _change_input_mode(self, mode: str) -> None:
        if mode == self.input_mode:
            return

        selected_values = [entry.currentData() for entry in self.entries]
        self.input_mode = mode

        for numeric_button in self.numeric_mode_buttons:
            numeric_button.blockSignals(True)
            numeric_button.setChecked(mode == NUMERIC_INPUT_MODE)
            numeric_button.blockSignals(False)

        for text_button in self.text_mode_buttons:
            text_button.blockSignals(True)
            text_button.setChecked(mode == TEXT_INPUT_MODE)
            text_button.blockSignals(False)

        for entry, value in zip(self.entries, selected_values):
            self._populate_feature_combo(entry)
            self._set_feature_combo_value(entry, value)

    #Установка значения выпадающего списка по числовому эквиваленту
    @staticmethod
    def _set_feature_combo_value(combo: QComboBox, value: float | None) -> None:
        if value is None:
            combo.setCurrentIndex(-1)
            return

        numeric_value = float(value)

        for index in range(combo.count()):
            option_value = combo.itemData(index)
            if option_value is not None and abs(float(option_value) - numeric_value) < 0.000001:
                combo.setCurrentIndex(index)
                return

        combo.setCurrentIndex(-1)

    #Кнопка запуска анализа
    def _build_analyze_button(self, parent_layout: QVBoxLayout) -> None:
        analyze_button = QPushButton("Выполнить анализ")
        analyze_button.setFixedSize(205, 46)
        analyze_button.setFont(QFont(FONT_FAMILY, 15, QFont.Weight.Bold))
        analyze_button.clicked.connect(self.predict)
        parent_layout.addWidget(analyze_button, alignment=Qt.AlignmentFlag.AlignHCenter)

    #Кнопка панели поиска и настроек
    def _build_search_panel(self, parent_layout: QHBoxLayout) -> None:
        search_frame = QFrame()
        search_frame.setObjectName("SearchPanel")
        search_frame.setFixedWidth(SEARCH_PANEL_WIDTH)

        layout = QVBoxLayout(search_frame)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(8)

        title = QLabel("Поиск предприятия")
        title.setFont(QFont(FONT_FAMILY, 17, QFont.Weight.Bold))
        layout.addWidget(title)

        hint = QLabel("Загрузите данные или заполните показатели вручную.")
        hint.setObjectName("MutedLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addSpacing(8)

        layout.addWidget(self._create_field_label("Тема:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItem(THEME_DISPLAY_NAMES[THEME_DARK], THEME_DARK)
        self.theme_combo.addItem(THEME_DISPLAY_NAMES[THEME_LIGHT], THEME_LIGHT)
        self.theme_combo.setCurrentIndex(0 if DEFAULT_THEME == THEME_DARK else 1)
        self.theme_combo.setFixedWidth(SEARCH_CONTROL_WIDTH)
        self.theme_combo.currentIndexChanged.connect(self.change_theme)
        layout.addWidget(self.theme_combo)

        layout.addSpacing(8)

        layout.addWidget(self._create_field_label("Поиск по ИНН:"))
        self.inn_entry = QLineEdit()
        self.inn_entry.setPlaceholderText("10 цифр")
        self.inn_entry.setFixedWidth(SEARCH_CONTROL_WIDTH)
        layout.addWidget(self.inn_entry)

        layout.addWidget(self._create_field_label("Поиск по году:"))
        self.year_entry = QLineEdit()
        self.year_entry.setPlaceholderText("Например, 2009")
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

        self.status_label = QLabel("Готово к работе!")
        self.status_label.setObjectName("MutedLabel")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        parent_layout.addWidget(search_frame)

    #Подпись для поля ввода
    @staticmethod
    def _create_field_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setFont(QFont(FONT_FAMILY, 13, QFont.Weight.Bold))
        return label

    #Смена светлой и темной тем
    def change_theme(self) -> None:
        theme_name = self._require_theme_combo().currentData()
        set_current_theme(theme_name)

        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(build_stylesheet(theme_name))

    #Сбор значений, запуск модели и открытие окна результата
    def predict(self) -> None:
        try:
            feature_values = self._collect_feature_values_from_entries()
            regression_values, probability = self.model_service.predict(feature_values)
            dialog = ResultDialog(
                regression_values,
                probability,
                context=self._build_analysis_context(),
                history_points=self._build_history_points(),
                parent=self,
            )
            dialog.exec()
        except InputValidationError as error:
            self._show_error(str(error))
        except Exception as error:
            self._show_error(str(error))


    #Формирование исторических точек для PDF-отчета
    def _build_history_points(self) -> list[HistoryPoint]:
        inn = self._get_inn_value()

        if not validate_inn(inn):
            return []

        history = self.repository.get_company_history(inn)
        if history.empty or len(history) < 2:
            return []

        feature_rows = [
            self.repository.get_feature_values(row)
            for _, row in history.iterrows()
        ]
        regression_rows, probabilities = self.model_service.predict_many(feature_rows)

        points: list[HistoryPoint] = []
        for index, (_, row) in enumerate(history.iterrows()):
            points.append(
                HistoryPoint(
                    year=self.repository.get_year(row),
                    industry=self.repository.get_industry(row),
                    probability=float(probabilities[index]),
                    regression_values=[float(value) for value in regression_rows[index]],
                )
            )

        return points

    #Подготовка контекста анализа для отчета агента
    def _build_analysis_context(self) -> dict[str, str]:
        inn = self._get_inn_value()
        year = self._get_year_value()
        industry = self._get_selected_industry()

        context = {
            "ИНН": inn if inn else "не указан",
            "Год": year if year else "не указан",
            "Отрасль": industry if industry else "не указана",
        }

        if not inn and not year and industry == ALL_INDUSTRIES_VALUE:
            return {"Источник": "ручной ввод"}

        return context

    #Поиск предприятия по ИНН и году
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

    #Выбор случайного предприятия с учетом выбранной отрасли
    def random_company(self) -> None:
        row = self.repository.get_random_by_industry(self._get_selected_industry())

        if row is None:
            self._show_error("В выбранной отрасли нет данных!")
            return

        self._fill_company_data(row)
        self._set_status(
            f"Случайно выбрано: ИНН {self.repository.get_inn(row)}, "
            f"{self.repository.get_year(row)} год, {self.repository.get_industry(row)}."
        )

    #Очистка формы ввода
    def clear_form(self) -> None:
        for entry in self.entries:
            entry.setCurrentIndex(-1)

        self._require_inn_entry().clear()
        self._require_year_entry().clear()
        self._require_industry_combo().setCurrentText(ALL_INDUSTRIES_VALUE)
        self._set_status("Форма очищена!")

    #Проверка и сбор 35 входных значений
    def _collect_feature_values_from_entries(self) -> list[float]:
        values = []

        for index, entry in enumerate(self.entries, start=1):
            value = entry.currentData()

            if value is None:
                entry.setFocus()
                raise InputValidationError(f"Выберите значение x{index}!")

            values.append(float(value))

        return values

    #Заполнение формы данными найденной компании
    def _fill_company_data(self, row: pd.Series) -> None:
        self._require_inn_entry().setText(self.repository.get_inn(row))
        self._require_year_entry().setText(self.repository.get_year(row))
        self._require_industry_combo().setCurrentText(self.repository.get_industry(row))
        self._fill_feature_entries(row)

    #Заполнение полей x1-x35
    def _fill_feature_entries(self, row: pd.Series) -> None:
        for entry, value in zip(self.entries, self.repository.get_feature_values(row)):
            self._set_feature_combo_value(entry, value)

    #Получение ИНН из поля поиска
    def _get_inn_value(self) -> str:
        return self._require_inn_entry().text().strip()

    #Получение года из поля поиска
    def _get_year_value(self) -> str:
        return self._require_year_entry().text().strip()

    #Получение выбранной отрасли
    def _get_selected_industry(self) -> str:
        return self._require_industry_combo().currentText()

    #Вывод статуса в правой панели
    def _set_status(self, text: str) -> None:
        if self.status_label is not None:
            self.status_label.setText(text)

    #Показ сообщения об ошибке
    def _show_error(self, text: str) -> None:
        QMessageBox.critical(self, ERROR_TITLE, text)

    #Проверка, что поле ИНН создано
    def _require_inn_entry(self) -> QLineEdit:
        if self.inn_entry is None:
            raise RuntimeError("Поле ИНН не создано!")
        return self.inn_entry

    #Проверка, что поле года создано
    def _require_year_entry(self) -> QLineEdit:
        if self.year_entry is None:
            raise RuntimeError("Поле года не создано!")
        return self.year_entry

    #Проверка, что список отраслей создан
    def _require_industry_combo(self) -> QComboBox:
        if self.industry_combo is None:
            raise RuntimeError("Список отраслей не создан!")
        return self.industry_combo

    #Проверка, что список тем создан
    def _require_theme_combo(self) -> QComboBox:
        if self.theme_combo is None:
            raise RuntimeError("Список тем не создан!")
        return self.theme_combo