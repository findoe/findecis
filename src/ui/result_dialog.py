from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.agent import AgentReport, build_agent_report
from src.config import (
    FONT_FAMILY,
    PREDICTION_FULL_LABELS,
)
from src.styles import get_theme



#Окно с результатами анализа
class ResultDialog(QDialog):
    #Инициализация окна результата и отчета агента
    def __init__(
        self,
        regression_values: np.ndarray,
        probability: float,
        context: dict[str, str] | None = None,
        history_points: list | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.regression_values = regression_values
        self.agent_report = build_agent_report(regression_values, probability, context)
        self.context = context or {}
        self.history_points = history_points or []

        self.setWindowTitle("Результаты")
        self.resize(1180, 860)
        self.setMinimumSize(1080, 760)
        self.setModal(False)
        self._build_ui(regression_values, probability)

    #Сборка интерфейса окна результата
    def _build_ui(self, regression_values: np.ndarray, probability: float) -> None:
        risk_color = self._risk_color(self.agent_report.risk_level)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(22, 22, 22, 22)
        root_layout.setSpacing(14)

        title_label = QLabel("Результат анализа")
        title_label.setFont(QFont(FONT_FAMILY, 26, QFont.Weight.Bold))
        title_label.setStyleSheet("font-size: 26px; font-weight: 700;")
        root_layout.addWidget(title_label)

        context_label = QLabel(self._context_text())
        context_label.setObjectName("MutedLabel")
        context_label.setFont(QFont(FONT_FAMILY, 15))
        context_label.setStyleSheet("font-size: 15px;")
        context_label.setWordWrap(True)
        root_layout.addWidget(context_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        root_layout.addWidget(scroll_area, stretch=1)

        scroll_content = QWidget()
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)
        scroll_area.setWidget(scroll_content)

        content_layout.addWidget(self._build_risk_card(probability, risk_color))
        content_layout.addWidget(self._build_agent_card())
        content_layout.addWidget(self._build_prediction_table(regression_values))

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch(1)

        save_button = QPushButton("Сохранить PDF")
        save_button.setFont(QFont(FONT_FAMILY, 15, QFont.Weight.Bold))
        save_button.setFixedSize(190, 44)
        save_button.clicked.connect(self.save_report)
        buttons_layout.addWidget(save_button)

        close_button = QPushButton("Закрыть")
        close_button.setFont(QFont(FONT_FAMILY, 15, QFont.Weight.Bold))
        close_button.setFixedSize(170, 44)
        close_button.clicked.connect(self.close)
        buttons_layout.addWidget(close_button)

        root_layout.addLayout(buttons_layout)

    #Карточка уровня риска и вероятности банкротства
    def _build_risk_card(self, probability: float, risk_color: str) -> QFrame:
        risk_card = QFrame()
        risk_card.setObjectName("RiskCard")
        risk_layout = QVBoxLayout(risk_card)
        risk_layout.setContentsMargins(18, 16, 18, 16)
        risk_layout.setSpacing(10)

        risk_label = QLabel(f"Уровень риска: {self.agent_report.risk_level}")
        risk_label.setFont(QFont(FONT_FAMILY, 28, QFont.Weight.Bold))
        risk_label.setStyleSheet(f"color: {risk_color}; font-size: 28px; font-weight: 700;")
        risk_layout.addWidget(risk_label)

        probability_label = QLabel(f"Вероятность банкротства: {probability * 100:.2f}%")
        probability_label.setFont(QFont(FONT_FAMILY, 22, QFont.Weight.Bold))
        probability_label.setStyleSheet("font-size: 22px; font-weight: 700;")
        risk_layout.addWidget(probability_label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(round(probability * 100))
        progress_bar.setTextVisible(False)
        progress_bar.setFixedHeight(18)
        progress_bar.setStyleSheet(self._progress_bar_style(risk_color))
        risk_layout.addWidget(progress_bar)

        summary = QLabel(self.agent_report.risk_summary)
        summary.setObjectName("MutedLabel")
        summary.setFont(QFont(FONT_FAMILY, 16))
        summary.setStyleSheet("font-size: 16px;")
        summary.setWordWrap(True)
        risk_layout.addWidget(summary)

        action = QLabel(self.agent_report.risk_action)
        action.setFont(QFont(FONT_FAMILY, 16, QFont.Weight.Bold))
        action.setStyleSheet("font-size: 16px; font-weight: 700;")
        action.setWordWrap(True)
        risk_layout.addWidget(action)

        return risk_card

    #Карточка интерпретации финансового агента
    def _build_agent_card(self) -> QFrame:
        agent_card = QFrame()
        agent_card.setObjectName("AgentCard")
        layout = QVBoxLayout(agent_card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        title = QLabel("Интерпретация агента")
        title.setFont(QFont(FONT_FAMILY, 21, QFont.Weight.Bold))
        title.setStyleSheet("font-size: 21px; font-weight: 700;")
        layout.addWidget(title)

        layout.addWidget(self._section_label("Слабые блоки"))
        if self.agent_report.weak_blocks:
            for block in self.agent_report.weak_blocks:
                layout.addWidget(self._text_line(f"• {block.name}: {block.value:.3f} — {block.recommendation}."))
        else:
            layout.addWidget(self._text_line("• Критичных слабых блоков по установленному порогу не выявлено."))

        layout.addWidget(self._section_label("Сильные блоки"))
        if self.agent_report.strong_blocks:
            for block in self.agent_report.strong_blocks:
                layout.addWidget(self._text_line(f"• {block.name}: {block.value:.3f}."))
        else:
            layout.addWidget(self._text_line("• Явно выраженных сильных блоков по установленному порогу не выявлено."))

        layout.addWidget(self._section_label("Рекомендации"))
        for recommendation in self.agent_report.recommendations:
            layout.addWidget(self._text_line(f"• {recommendation}"))

        return agent_card

    #Таблица регрессионных прогнозов
    def _build_prediction_table(self, regression_values: np.ndarray) -> QTableWidget:
        table = QTableWidget(len(PREDICTION_FULL_LABELS), 2)
        table.setHorizontalHeaderLabels(["Показатель", "Значение"])
        table.setFont(QFont(FONT_FAMILY, 16))
        table.horizontalHeader().setFont(QFont(FONT_FAMILY, 16, QFont.Weight.Bold))
        table.verticalHeader().setDefaultSectionSize(38)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.setMinimumHeight(460)

        for row_index, value in enumerate(regression_values):
            name_item = QTableWidgetItem(PREDICTION_FULL_LABELS[row_index])
            name_item.setFont(QFont(FONT_FAMILY, 16))
            value_item = QTableWidgetItem(f"{value:.3f}")
            value_item.setFont(QFont(FONT_FAMILY, 16, QFont.Weight.Bold))
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row_index, 0, name_item)
            table.setItem(row_index, 1, value_item)

        return table

    #Сохранение PDF-отчета с графиками
    def save_report(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить PDF-отчет",
            "otchet_finansovogo_agenta.pdf",
            "PDF-файл (*.pdf)",
        )

        if not file_path:
            return

        try:
            from src.pdf_report import create_pdf_report

            saved_path = create_pdf_report(
                file_path=file_path,
                agent_report=self.agent_report,
                context=self.context,
                current_regression_values=self.regression_values,
                history_points=self.history_points,
            )
        except ImportError as error:
            QMessageBox.critical(
                self,
                "Ошибка!",
                f"Не удалось создать PDF-отчет: не установлена библиотека {error.name}. "
                "Установите зависимости из requirements.txt.",
            )
            return
        except OSError as error:
            QMessageBox.critical(self, "Ошибка!", f"Не удалось сохранить PDF-отчет: {error}")
            return
        except Exception as error:
            QMessageBox.critical(self, "Ошибка!", f"Не удалось создать PDF-отчет: {error}")
            return

        QMessageBox.information(self, "Готово", f"PDF-отчет сохранен:\n{saved_path}")

    #Формирование строки с контекстом анализа
    def _context_text(self) -> str:
        values = [f"{key}: {value}" for key, value in self.context.items() if value]
        if not values:
            return "Контекст анализа: ручной ввод."
        return "Контекст анализа: " + "; ".join(values) + "."

    #Создание заголовка раздела
    @staticmethod
    def _section_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setFont(QFont(FONT_FAMILY, 17, QFont.Weight.Bold))
        label.setStyleSheet("font-size: 17px; font-weight: 700;")
        return label

    #Создание текстовой строки с переносами
    @staticmethod
    def _text_line(text: str) -> QLabel:
        label = QLabel(text)
        label.setFont(QFont(FONT_FAMILY, 15))
        label.setStyleSheet("font-size: 15px;")
        label.setWordWrap(True)
        return label

    #Выбор цвета по уровню риска
    @staticmethod
    def _risk_color(risk_level: str) -> str:
        theme = get_theme()
        if risk_level == "высокий":
            return theme.risk_danger
        if risk_level == "средний":
            return theme.risk_warning
        return theme.risk_success

    #QSS-стиль прогресс-бара риска
    @staticmethod
    def _progress_bar_style(color: str) -> str:
        theme = get_theme()
        return f"""
            QProgressBar {{
                background-color: {theme.progress_track};
                border: 1px solid {theme.border};
                border-radius: 7px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 7px;
            }}
        """