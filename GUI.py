from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


APP_TITLE = "Анализ финансового состояния предприятия"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_artifacts" / "best_model.keras"
SCALER_PATH = BASE_DIR / "model_artifacts" / "scaler.pkl"
DATA_PATH = BASE_DIR / "data_new.csv"
DATA_DELIMITER = ";"

INPUT_COUNT = 35
RISK_THRESHOLD = 0.5
ALL_INDUSTRIES_VALUE = "Все"

FONT_FAMILY = "Bahnschrift"
TEXT_COLOR = "white"
PLACEHOLDER_COLOR = "#9a9a9a"
ERROR_TITLE = "Ошибка!"

DARK_BG = "#2B2B2B"
DARK_PANEL = "#303030"
DARK_TAB = "#343638"
DARK_TAB_HOVER = "#3A3A3A"
INPUT_BG = "#242424"
BORDER_COLOR = "#565B5E"
BUTTON_COLOR = "#3B8ED0"
BUTTON_HOVER_COLOR = "#36719F"
OPTION_BUTTON_COLOR = "#565B5E"

SEARCH_PANEL_WIDTH = 260
SEARCH_CONTROL_WIDTH = 220

PREDICTION_LABELS = [
    "Reg", "Kredit", "Teh", "Market", "Staff", "Psich",
    "Ability", "Turn", "Finn", "Z25", "Z35",
]

CATEGORIES = [
    ("Регион. и отрасл. специфика", 5),
    ("Кредитная история", 4),
    ("Техн. оснащенность", 3),
    ("Рыночный потенциал", 5),
    ("Кадровое обеспечение", 7),
    ("Платежеспособность", 6),
    ("Рентабельность", 5),
]

FIELD_LABELS = [
    "Уровень конкуренции", "Концентрация рисков", "Доля отрасли", "Ситуация в регионе",
    "Макро-риски", "Кредитная история", "Кредиты", "Залог",
    "Срок работы", "Оборудование 1", "Оборудование 2", "Загрузка",
    "План", "Цены", "Инновации", "Поставщики",
    "Покупатели", "Кадры", "Качество кадров", "Квалификация",
    "Рост квалификации", "Фин. менеджмент", "Текучесть",
    "Кейтц", "Ликвидность", "Оборотные средства", "Дебиторка",
    "Кредиторка", "Соотношение", "Готовая продукция",
    "Финансовая активность", "Независимость", "Запасы",
    "Рентабельность", "Продажи",
]

INDUSTRY_COLUMN_CANDIDATES = ("Отрасль экономики", "Отрасль", "industry")
INN_COLUMN_CANDIDATES = ("ИНН", "inn")
YEAR_COLUMN_CANDIDATES = ("Год", "year")


def validate_numeric(value: str) -> bool:
    try:
        float(value.replace(",", "."))
        return True
    except ValueError:
        return False


def to_float(value: str) -> float:
    return float(value.replace(",", "."))


def validate_inn(inn: str) -> bool:
    return inn.isdigit() and len(inn) == 10


def validate_year(year: str) -> bool:
    return year.isdigit() and len(year) == 4


class InputValidationError(ValueError):
    pass


def resolve_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
    fallback_index: int,
) -> str:
    """Находит столбец по имени, а если имени нет — использует позицию."""
    normalized_columns = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized_columns:
            return normalized_columns[key]

    for column in dataframe.columns:
        column_name = str(column).strip().lower()
        for candidate in candidates:
            if candidate.strip().lower() in column_name:
                return column

    if fallback_index >= len(dataframe.columns):
        raise ValueError(f"В данных нет столбца с индексом {fallback_index}")

    return dataframe.columns[fallback_index]


@dataclass(frozen=True)
class DataColumns:
    industry: str
    inn: str
    year: str


class CompanyDataRepository:
    def __init__(self, data_path: Path, delimiter: str = DATA_DELIMITER) -> None:
        if not data_path.exists():
            raise FileNotFoundError(f"Файл данных не найден: {data_path}")

        self.data = pd.read_csv(data_path, delimiter=delimiter)
        self.columns = self._resolve_columns()
        self._normalize_key_columns()
        self._validate_feature_columns()

    def _resolve_columns(self) -> DataColumns:
        return DataColumns(
            industry=resolve_column(self.data, INDUSTRY_COLUMN_CANDIDATES, fallback_index=0),
            inn=resolve_column(self.data, INN_COLUMN_CANDIDATES, fallback_index=1),
            year=resolve_column(self.data, YEAR_COLUMN_CANDIDATES, fallback_index=2),
        )

    def _normalize_key_columns(self) -> None:
        for column in (self.columns.industry, self.columns.inn, self.columns.year):
            self.data[column] = self.data[column].astype(str).str.strip()

    def _validate_feature_columns(self) -> None:
        missing_columns = [column for column in self.feature_columns if column not in self.data.columns]
        if missing_columns:
            raise ValueError("В данных отсутствуют столбцы: " + ", ".join(missing_columns))

    @property
    def feature_columns(self) -> list[str]:
        return [f"x{i}" for i in range(1, INPUT_COUNT + 1)]

    @property
    def industries(self) -> list[str]:
        values = self.data[self.columns.industry].dropna().unique().tolist()
        return sorted(
            str(value).strip()
            for value in values
            if str(value).strip() and str(value).strip().lower() != "nan"
        )

    def find_by_inn_and_year(self, inn: str, year: str = "") -> pd.Series | None:
        filtered = self.data[self.data[self.columns.inn] == inn]

        if year:
            filtered = filtered[filtered[self.columns.year] == year]

        if filtered.empty:
            return None

        return filtered.iloc[0]

    def get_random_by_industry(self, industry: str) -> pd.Series | None:
        filtered = self.data

        if industry != ALL_INDUSTRIES_VALUE:
            filtered = filtered[filtered[self.columns.industry] == industry]

        if filtered.empty:
            return None

        return filtered.sample(1).iloc[0]

    def get_inn(self, row: pd.Series) -> str:
        return str(row[self.columns.inn])

    def get_year(self, row: pd.Series) -> str:
        return str(row[self.columns.year])

    def get_feature_values(self, row: pd.Series) -> list[float]:
        return [float(row[column]) for column in self.feature_columns]


class FinancialModelService:
    def __init__(self, model_path: Path, scaler_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Файл scaler не найден: {scaler_path}")

        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, feature_values: list[float]) -> tuple[np.ndarray, float]:
        input_data = np.array(feature_values).reshape(1, -1)
        input_data_scaled = self.scaler.transform(input_data)

        regression_output, classification_output = self.model.predict(input_data_scaled, verbose=0)
        probability = float(classification_output[0][0])

        return regression_output[0], probability



def build_app_stylesheet() -> str:
    return f"""
        QWidget {{
            background-color: {DARK_BG};
            color: {TEXT_COLOR};
            font-family: {FONT_FAMILY};
        }}

        QMainWindow {{
            background-color: {DARK_BG};
        }}

        QFrame#SearchPanel,
        QFrame#ResultPanel {{
            background-color: {DARK_PANEL};
            border: 1px solid {BORDER_COLOR};
            border-radius: 8px;
        }}

        QLabel {{
            color: {TEXT_COLOR};
            background-color: transparent;
        }}

        QLineEdit {{
            background-color: {INPUT_BG};
            color: {TEXT_COLOR};
            border: 1px solid {BORDER_COLOR};
            border-radius: 6px;
            padding: 6px;
            selection-background-color: {BUTTON_COLOR};
        }}

        QLineEdit:focus {{
            border: 1px solid {BUTTON_COLOR};
        }}

        QLineEdit::placeholder {{
            color: {PLACEHOLDER_COLOR};
        }}

        QPushButton {{
            background-color: {BUTTON_COLOR};
            color: {TEXT_COLOR};
            border: none;
            border-radius: 6px;
            padding: 8px 12px;
            font-weight: bold;
        }}

        QPushButton:hover {{
            background-color: {BUTTON_HOVER_COLOR};
        }}

        QPushButton:pressed {{
            background-color: #2F5F88;
        }}

        QComboBox {{
            background-color: {DARK_TAB};
            color: {TEXT_COLOR};
            border: 1px solid {BORDER_COLOR};
            border-radius: 6px;
            padding: 6px;
        }}

        QComboBox:hover {{
            border: 1px solid {BUTTON_COLOR};
        }}

        QComboBox::drop-down {{
            width: 28px;
            border-left: 1px solid {BORDER_COLOR};
            background-color: {OPTION_BUTTON_COLOR};
            border-top-right-radius: 6px;
            border-bottom-right-radius: 6px;
        }}

        QComboBox QAbstractItemView {{
            background-color: {DARK_BG};
            color: {TEXT_COLOR};
            border: 1px solid {BORDER_COLOR};
            selection-background-color: {DARK_TAB_HOVER};
            selection-color: {TEXT_COLOR};
            outline: 0;
        }}

        QTabWidget::pane {{
            border: 1px solid {BORDER_COLOR};
            background-color: {DARK_BG};
            top: -1px;
        }}

        QTabBar::tab {{
            background-color: {DARK_BG};
            color: {TEXT_COLOR};
            padding: 7px 10px;
            border: 1px solid {DARK_BG};
            border-bottom: none;
        }}

        QTabBar::tab:selected {{
            background-color: {DARK_TAB};
            border: 1px solid {BORDER_COLOR};
            border-bottom: 1px solid {DARK_TAB};
        }}

        QTabBar::tab:hover {{
            background-color: {DARK_TAB_HOVER};
        }}

        QScrollArea {{
            border: none;
            background-color: {DARK_BG};
        }}

        QScrollBar:vertical {{
            background-color: {DARK_BG};
            width: 12px;
            margin: 0;
        }}

        QScrollBar::handle:vertical {{
            background-color: {BORDER_COLOR};
            border-radius: 6px;
            min-height: 24px;
        }}

        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{
            height: 0;
        }}

        QMessageBox {{
            background-color: {DARK_BG};
        }}
    """



class FinancialAnalysisApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        try:
            self.repository = CompanyDataRepository(DATA_PATH)
            self.model_service = FinancialModelService(MODEL_PATH, SCALER_PATH)
        except Exception as error:
            QMessageBox.critical(None, ERROR_TITLE, str(error))
            raise

        self.entries: list[QLineEdit] = []
        self.inn_entry: QLineEdit | None = None
        self.year_entry: QLineEdit | None = None
        self.industry_combo: QComboBox | None = None
        self.result_windows: list[QDialog] = []

        self.setWindowTitle(APP_TITLE)
        self.resize(1120, 650)
        self.setMinimumSize(980, 560)

        self._build_ui()

    def _build_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)

        self._build_tabs(content_layout)
        self._build_search_panel(content_layout)

        main_layout.addLayout(content_layout, stretch=1)
        self._build_analyze_button(main_layout)

    def _build_search_panel(self, parent_layout: QHBoxLayout) -> None:
        search_frame = QFrame()
        search_frame.setObjectName("SearchPanel")
        search_frame.setFixedWidth(SEARCH_PANEL_WIDTH)
        search_frame.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        search_layout = QVBoxLayout(search_frame)
        search_layout.setContentsMargins(16, 16, 16, 16)
        search_layout.setSpacing(8)
        search_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        inn_label = QLabel("Поиск по ИНН:")
        inn_label.setFont(QFont(FONT_FAMILY, 15))
        search_layout.addWidget(inn_label)

        self.inn_entry = QLineEdit()
        self.inn_entry.setFont(QFont(FONT_FAMILY, 13))
        self.inn_entry.setFixedWidth(SEARCH_CONTROL_WIDTH)
        self.inn_entry.setPlaceholderText("10 цифр")
        search_layout.addWidget(self.inn_entry)

        year_label = QLabel("Поиск по году:")
        year_label.setFont(QFont(FONT_FAMILY, 15))
        search_layout.addWidget(year_label)

        self.year_entry = QLineEdit()
        self.year_entry.setFont(QFont(FONT_FAMILY, 13))
        self.year_entry.setFixedWidth(SEARCH_CONTROL_WIDTH)
        self.year_entry.setPlaceholderText("Например: 2024")
        search_layout.addWidget(self.year_entry)

        industry_label = QLabel("Отрасль:")
        industry_label.setFont(QFont(FONT_FAMILY, 15))
        search_layout.addWidget(industry_label)

        self.industry_combo = QComboBox()
        self.industry_combo.setFont(QFont(FONT_FAMILY, 13))
        self.industry_combo.setFixedWidth(SEARCH_CONTROL_WIDTH)
        self.industry_combo.addItems([ALL_INDUSTRIES_VALUE] + self.repository.industries)
        search_layout.addWidget(self.industry_combo)

        search_button = QPushButton("Готово")
        search_button.setFont(QFont(FONT_FAMILY, 13, QFont.Weight.Bold))
        search_button.setFixedWidth(SEARCH_CONTROL_WIDTH)
        search_button.clicked.connect(self.search)
        search_layout.addWidget(search_button)

        random_button = QPushButton("Случайно")
        random_button.setFont(QFont(FONT_FAMILY, 13, QFont.Weight.Bold))
        random_button.setFixedWidth(SEARCH_CONTROL_WIDTH)
        random_button.clicked.connect(self.random_inn)
        search_layout.addWidget(random_button)

        search_layout.addStretch(1)
        parent_layout.addWidget(search_frame)

    def _build_tabs(self, parent_layout: QHBoxLayout) -> None:
        tab_control = QTabWidget()
        tab_control.setDocumentMode(True)
        tab_control.setMovable(False)

        label_index = 0
        for category, fields_count in CATEGORIES:
            tab_page = QWidget()
            tab_layout = QVBoxLayout(tab_page)
            tab_layout.setContentsMargins(10, 10, 10, 10)
            tab_layout.setSpacing(6)
            tab_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

            for _ in range(fields_count):
                label = QLabel(f"{label_index + 1}. {FIELD_LABELS[label_index]}")
                label.setFont(QFont(FONT_FAMILY, 13))
                tab_layout.addWidget(label)

                entry = QLineEdit()
                entry.setFont(QFont(FONT_FAMILY, 13))
                entry.setPlaceholderText(f"x{label_index + 1}")
                entry.setMinimumWidth(220)
                entry.setMaximumWidth(420)
                tab_layout.addWidget(entry)

                self.entries.append(entry)
                label_index += 1

            tab_layout.addStretch(1)

            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(tab_page)
            tab_control.addTab(scroll_area, category)

        parent_layout.addWidget(tab_control, stretch=1)

    def _build_analyze_button(self, parent_layout: QVBoxLayout) -> None:
        analyze_button = QPushButton("Анализ")
        analyze_button.setFont(QFont(FONT_FAMILY, 16, QFont.Weight.Bold))
        analyze_button.setFixedWidth(180)
        analyze_button.clicked.connect(self.predict)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(analyze_button)
        button_row.addStretch(1)

        parent_layout.addLayout(button_row)

    def predict(self) -> None:
        try:
            feature_values = self._collect_feature_values_from_entries()
            regression_values, probability = self.model_service.predict(feature_values)
            self._show_result_window(regression_values, probability)
        except InputValidationError as error:
            QMessageBox.critical(self, ERROR_TITLE, str(error))
        except Exception as error:
            QMessageBox.critical(self, "Ошибка", str(error))

    def search(self) -> None:
        inn = self._get_inn_value()
        year = self._get_year_value()

        if not validate_inn(inn):
            QMessageBox.critical(self, ERROR_TITLE, "ИНН должен содержать 10 цифр!")
            return

        if year and not validate_year(year):
            QMessageBox.critical(self, ERROR_TITLE, "Год должен содержать 4 цифры!")
            return

        row = self.repository.find_by_inn_and_year(inn, year)
        if row is None:
            QMessageBox.critical(self, ERROR_TITLE, "Данные не найдены!")
            return

        self._fill_feature_entries(row)

    def random_inn(self) -> None:
        industry = self._require_industry_combo().currentText()
        row = self.repository.get_random_by_industry(industry)

        if row is None:
            QMessageBox.critical(self, ERROR_TITLE, "В выбранной отрасли нет данных!")
            return

        self._set_entry_value(self._require_inn_entry(), self.repository.get_inn(row))
        self._set_entry_value(self._require_year_entry(), self.repository.get_year(row))
        self._fill_feature_entries(row)

    def _collect_feature_values_from_entries(self) -> list[float]:
        values: list[float] = []

        for entry in self.entries:
            value = entry.text().strip()

            if not value or value.startswith("x") or not validate_numeric(value):
                raise InputValidationError("Заполните все поля корректно!")

            values.append(to_float(value))

        return values

    def _fill_feature_entries(self, row: pd.Series) -> None:
        for entry, value in zip(self.entries, self.repository.get_feature_values(row)):
            self._set_entry_value(entry, str(value))

    @staticmethod
    def _set_entry_value(entry: QLineEdit, value: str) -> None:
        entry.setText(value)

    def _show_result_window(self, regression_values: np.ndarray, probability: float) -> None:
        result_window = QDialog(self)
        result_window.setWindowTitle("Результаты")
        result_window.setMinimumWidth(520)
        result_window.setModal(False)

        dialog_layout = QVBoxLayout(result_window)
        dialog_layout.setContentsMargins(20, 20, 20, 20)
        dialog_layout.setSpacing(14)

        results_frame = QFrame()
        results_frame.setObjectName("ResultPanel")
        results_layout = QGridLayout(results_frame)
        results_layout.setContentsMargins(18, 18, 18, 18)
        results_layout.setHorizontalSpacing(20)
        results_layout.setVerticalSpacing(8)

        for index, value in enumerate(regression_values):
            name_label = QLabel(f"{PREDICTION_LABELS[index]}:")
            name_label.setFont(QFont(FONT_FAMILY, 15))
            results_layout.addWidget(name_label, index, 0, alignment=Qt.AlignmentFlag.AlignLeft)

            value_label = QLabel(f"{value:.3f}")
            value_label.setFont(QFont(FONT_FAMILY, 17, QFont.Weight.Bold))
            results_layout.addWidget(value_label, index, 1, alignment=Qt.AlignmentFlag.AlignLeft)

        dialog_layout.addWidget(results_frame)

        risk_text = "высокий риск банкротства" if probability > RISK_THRESHOLD else "низкий риск банкротства"
        risk_color = "#ff5555" if probability > RISK_THRESHOLD else "#57d957"

        risk_label = QLabel(f"Вероятность банкротства: {probability:.4f} ({risk_text})")
        risk_label.setFont(QFont(FONT_FAMILY, 17, QFont.Weight.Bold))
        risk_label.setStyleSheet(f"color: {risk_color}; background-color: transparent;")
        dialog_layout.addWidget(risk_label)

        self.result_windows.append(result_window)
        result_window.finished.connect(
            lambda _result, window=result_window: self.result_windows.remove(window)
            if window in self.result_windows
            else None
        )
        result_window.show()

    def _get_inn_value(self) -> str:
        return self._require_inn_entry().text().strip()

    def _get_year_value(self) -> str:
        return self._require_year_entry().text().strip()

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
            raise RuntimeError("Поле отрасли не создано")
        return self.industry_combo


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(build_app_stylesheet())

    window = FinancialAnalysisApp()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
