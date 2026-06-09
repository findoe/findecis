from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFrame,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from financial_app.config import (
    DARK_BORDER,
    DARK_FIELD,
    DANGER,
    FONT_FAMILY,
    MUTED_TEXT_COLOR,
    PREDICTION_FULL_LABELS,
    RISK_THRESHOLD,
    SUCCESS,
)


class ResultDialog(QDialog):
    def __init__(self, regression_values: np.ndarray, probability: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Результаты")
        self.resize(760, 610)
        self.setModal(False)
        self._build_ui(regression_values, probability)

    def _build_ui(self, regression_values: np.ndarray, probability: float) -> None:
        risk_text, risk_color = self._risk_data(probability)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(22, 22, 22, 22)
        root_layout.setSpacing(16)

        title_label = QLabel("Результат анализа")
        title_label.setFont(QFont(FONT_FAMILY, 22, QFont.Weight.Bold))
        root_layout.addWidget(title_label)

        risk_card = QFrame()
        risk_card.setObjectName("RiskCard")
        risk_layout = QVBoxLayout(risk_card)
        risk_layout.setContentsMargins(18, 16, 18, 16)
        risk_layout.setSpacing(10)

        risk_label = QLabel(f"Уровень риска: {risk_text}")
        risk_label.setFont(QFont(FONT_FAMILY, 20, QFont.Weight.Bold))
        risk_label.setStyleSheet(f"color: {risk_color};")
        risk_layout.addWidget(risk_label)

        probability_label = QLabel(f"Вероятность банкротства: {probability * 100:.2f}%")
        probability_label.setFont(QFont(FONT_FAMILY, 16, QFont.Weight.Bold))
        risk_layout.addWidget(probability_label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(round(probability * 100))
        progress_bar.setTextVisible(False)
        progress_bar.setFixedHeight(14)
        progress_bar.setStyleSheet(self._progress_bar_style(risk_color))
        risk_layout.addWidget(progress_bar)

        interpretation = QLabel(
            "Модель оценивает финансовое состояние предприятия на основе 35 входных "
            "показателей и формирует 11 регрессионных оценок, а также вероятность банкротства."
        )
        interpretation.setWordWrap(True)
        interpretation.setStyleSheet(f"color: {MUTED_TEXT_COLOR};")
        risk_layout.addWidget(interpretation)

        root_layout.addWidget(risk_card)

        table = QTableWidget(len(PREDICTION_FULL_LABELS), 2)
        table.setHorizontalHeaderLabels(["Показатель", "Значение"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)

        for row_index, value in enumerate(regression_values):
            name_item = QTableWidgetItem(PREDICTION_FULL_LABELS[row_index])
            value_item = QTableWidgetItem(f"{value:.3f}")
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row_index, 0, name_item)
            table.setItem(row_index, 1, value_item)

        root_layout.addWidget(table)

        close_button = QPushButton("Закрыть")
        close_button.setFixedWidth(140)
        close_button.clicked.connect(self.close)
        root_layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)

    @staticmethod
    def _risk_data(probability: float) -> tuple[str, str]:
        if probability > RISK_THRESHOLD:
            return "высокий", DANGER
        return "низкий", SUCCESS

    @staticmethod
    def _progress_bar_style(color: str) -> str:
        return f"""
            QProgressBar {{
                background-color: {DARK_FIELD};
                border: 1px solid {DARK_BORDER};
                border-radius: 7px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 7px;
            }}
        """
