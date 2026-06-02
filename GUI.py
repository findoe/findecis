from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

APP_TITLE = "Анализ финансового состояния предприятия"
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model_artifacts" / "best_model.keras"
SCALER_PATH = BASE_DIR / "model_artifacts" / "scaler.pkl"
DATA_PATH = BASE_DIR / "data_new.csv"

INPUT_COUNT = 35
RISK_THRESHOLD = 0.5
ALL_INDUSTRIES = "Все"

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


def find_column(data: pd.DataFrame, names: tuple[str, ...], fallback_index: int) -> str:
    normalized = {str(column).strip().lower(): column for column in data.columns}

    for name in names:
        key = name.strip().lower()
        if key in normalized:
            return normalized[key]

    for column in data.columns:
        column_name = str(column).strip().lower()
        if any(name.strip().lower() in column_name for name in names):
            return column

    return data.columns[fallback_index]


def is_number(value: str) -> bool:
    try:
        float(value.replace(",", "."))
        return True
    except ValueError:
        return False


class FinancialAnalysisApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.feature_columns = [f"x{i}" for i in range(1, INPUT_COUNT + 1)]
        self.entries: list[QLineEdit] = []

        try:
            self.data = pd.read_csv(DATA_PATH, delimiter=";")
            self.industry_column = find_column(self.data, ("Отрасль экономики", "Отрасль", "industry"), 0)
            self.inn_column = find_column(self.data, ("ИНН", "inn"), 1)
            self.year_column = find_column(self.data, ("Год", "year"), 2)

            for column in (self.industry_column, self.inn_column, self.year_column):
                self.data[column] = self.data[column].astype(str).str.strip()

            missing = [column for column in self.feature_columns if column not in self.data.columns]
            if missing:
                raise ValueError("В данных отсутствуют столбцы: " + ", ".join(missing))

            self.model = load_model(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
        except Exception as error:
            QMessageBox.critical(None, "Ошибка!", str(error))
            raise

        self.setWindowTitle(APP_TITLE)
        self.resize(1080, 640)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        layout.addWidget(self._build_tabs(), stretch=1)
        layout.addWidget(self._build_side_panel())

        self.setStyleSheet("""
            QWidget {
                background-color: #2B2B2B;
                color: white;
                font-family: Bahnschrift;
                font-size: 13px;
            }
            QFrame#SidePanel {
                background-color: #303030;
                border: 1px solid #565B5E;
                border-radius: 8px;
            }
            QLineEdit, QComboBox {
                background-color: #242424;
                color: white;
                border: 1px solid #565B5E;
                border-radius: 6px;
                padding: 6px;
            }
            QPushButton {
                background-color: #3B8ED0;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #36719F;
            }
            QTabWidget::pane {
                border: 1px solid #565B5E;
            }
            QTabBar::tab {
                padding: 7px 10px;
            }
        """)

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        index = 0

        for category, count in CATEGORIES:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

            for _ in range(count):
                label = QLabel(f"{index + 1}. {FIELD_LABELS[index]}")
                entry = QLineEdit()
                entry.setPlaceholderText(f"x{index + 1}")

                page_layout.addWidget(label)
                page_layout.addWidget(entry)
                self.entries.append(entry)
                index += 1

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            tabs.addTab(scroll, category)

        return tabs

    def _build_side_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setFixedWidth(300)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.inn_entry = QLineEdit()
        self.inn_entry.setPlaceholderText("10 цифр")

        self.year_entry = QLineEdit()
        self.year_entry.setPlaceholderText("Например: 2024")

        self.industry_combo = QComboBox()
        self.industry_combo.addItems([ALL_INDUSTRIES] + self._industries())

        search_button = QPushButton("Готово")
        search_button.clicked.connect(self.search)

        random_button = QPushButton("Случайно")
        random_button.clicked.connect(self.random_company)

        analyze_button = QPushButton("Анализ")
        analyze_button.clicked.connect(self.predict)

        self.result_label = QLabel("Результаты анализа появятся здесь.")
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignTop)

        widgets = [
            QLabel("Поиск по ИНН:"), self.inn_entry,
            QLabel("Поиск по году:"), self.year_entry,
            QLabel("Отрасль:"), self.industry_combo,
            search_button, random_button, analyze_button,
            QLabel("Результаты:"), self.result_label,
        ]

        for widget in widgets:
            layout.addWidget(widget)

        return panel

    def _industries(self) -> list[str]:
        values = self.data[self.industry_column].dropna().unique()
        return sorted(str(value).strip() for value in values if str(value).strip())

    def search(self) -> None:
        inn = self.inn_entry.text().strip()
        year = self.year_entry.text().strip()

        if not inn.isdigit() or len(inn) != 10:
            self._show_error("ИНН должен содержать 10 цифр!")
            return

        if year and (not year.isdigit() or len(year) != 4):
            self._show_error("Год должен содержать 4 цифры!")
            return

        rows = self.data[self.data[self.inn_column] == inn]
        if year:
            rows = rows[rows[self.year_column] == year]

        if rows.empty:
            self._show_error("Данные не найдены!")
            return

        self._fill_entries(rows.iloc[0])

    def random_company(self) -> None:
        industry = self.industry_combo.currentText()
        rows = self.data

        if industry != ALL_INDUSTRIES:
            rows = rows[rows[self.industry_column] == industry]

        if rows.empty:
            self._show_error("В выбранной отрасли нет данных!")
            return

        row = rows.sample(1).iloc[0]
        self.inn_entry.setText(str(row[self.inn_column]))
        self.year_entry.setText(str(row[self.year_column]))
        self._fill_entries(row)

    def predict(self) -> None:
        try:
            values = self._entry_values()
            scaled_values = self.scaler.transform(np.array(values).reshape(1, -1))

            regression_output, classification_output = self.model.predict(scaled_values, verbose=0)
            regression_values = regression_output[0]
            probability = float(classification_output[0][0])

            self._show_result(regression_values, probability)
        except ValueError as error:
            self._show_error(str(error))
        except Exception as error:
            self._show_error(str(error))

    def _entry_values(self) -> list[float]:
        values = []

        for entry in self.entries:
            text = entry.text().strip()

            if not text or not is_number(text):
                raise ValueError("Заполните все поля корректно!")

            values.append(float(text.replace(",", ".")))

        return values

    def _fill_entries(self, row: pd.Series) -> None:
        for entry, column in zip(self.entries, self.feature_columns):
            entry.setText(str(row[column]))

    def _show_result(self, regression_values: np.ndarray, probability: float) -> None:
        risk = "высокий риск банкротства" if probability > RISK_THRESHOLD else "низкий риск банкротства"
        lines = [f"{name}: {value:.3f}" for name, value in zip(PREDICTION_LABELS, regression_values)]
        lines += ["", f"Вероятность банкротства: {probability:.4f}", f"Итог: {risk}"]
        self.result_label.setText("\n".join(lines))

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка!", message)


def main() -> int:
    app = QApplication(sys.argv)
    window = FinancialAnalysisApp()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
