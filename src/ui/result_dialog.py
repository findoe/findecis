from __future__ import annotations

from html import escape

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from src.agent import IndicatorAdvice, build_agent_report
from src.config import FONT_FAMILY, PREDICTION_FULL_LABELS
from src.styles import get_theme


class RoundedProgressBar(QWidget):
    def __init__(self, value: float, color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value = 0.0
        self._color = color
        self._track = QFrame(self)
        self._fill = QFrame(self._track)
        self.setFixedHeight(18)
        self._apply_styles()
        self.set_value(value)

    def set_value(self, value: float) -> None:
        self._value = max(0.0, min(1.0, float(value)))
        self._update_fill_geometry()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._track.setGeometry(0, 0, self.width(), self.height())
        self._update_fill_geometry()

    def _apply_styles(self) -> None:
        theme = get_theme()
        radius = max(1, self.height() // 2)
        self._track.setStyleSheet(
            f"background-color: {theme.progress_track}; border: 1px solid {theme.border}; border-radius: {radius}px;"
        )
        self._fill.setStyleSheet(
            f"background-color: {self._color}; border: none; border-radius: {radius - 1}px;"
        )

    def _update_fill_geometry(self) -> None:
        inner_margin = 1
        track_width = max(0, self.width() - inner_margin * 2)
        track_height = max(0, self.height() - inner_margin * 2)
        if track_width <= 0 or track_height <= 0 or self._value <= 0:
            self._fill.setGeometry(0, 0, 0, 0)
            return

        fill_width = int(track_width * self._value)
        fill_width = max(track_height, fill_width)
        fill_width = min(track_width, fill_width)
        self._fill.setGeometry(inner_margin, inner_margin, fill_width, track_height)


class ResultDialog(QDialog):
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

    def _build_ui(self, regression_values: np.ndarray, probability: float) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(22, 22, 22, 22)
        root_layout.setSpacing(14)

        title_label = QLabel("Результат анализа")
        title_label.setFont(QFont(FONT_FAMILY, 24, QFont.Weight.Bold))
        title_label.setStyleSheet("font-size: 24px; font-weight: 800;")
        root_layout.addWidget(title_label)
        root_layout.addWidget(self._build_context_card())

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        root_layout.addWidget(scroll_area, stretch=1)

        scroll_content = QWidget()
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)
        scroll_area.setWidget(scroll_content)

        risk_color = self._risk_color(self.agent_report.risk_level)
        content_layout.addWidget(self._build_risk_card(probability, risk_color))
        content_layout.addWidget(self._build_agent_card())
        content_layout.addWidget(self._build_prediction_card(regression_values))

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

    def _build_context_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("InfoBlock")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        title_label = QLabel("Контекст анализа")
        title_label.setFont(QFont(FONT_FAMILY, 16, QFont.Weight.Bold))
        title_label.setStyleSheet("font-size: 16px; font-weight: 800;")
        layout.addWidget(title_label)

        for line in self._context_lines():
            label = QLabel(line)
            label.setObjectName("MutedLabel")
            label.setFont(QFont(FONT_FAMILY, 15, QFont.Weight.DemiBold))
            label.setStyleSheet("font-size: 16px;")
            layout.addWidget(label)

        return card

    def _build_risk_card(self, probability: float, risk_color: str) -> QFrame:
        theme = get_theme()
        risk_card = QFrame()
        risk_card.setObjectName("RiskCard")
        risk_layout = QVBoxLayout(risk_card)
        risk_layout.setContentsMargins(18, 16, 18, 16)
        risk_layout.setSpacing(12)

        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(12)
        metrics_layout.addWidget(self._metric_card("Уровень риска", self.agent_report.risk_level.upper(), risk_color))
        metrics_layout.addWidget(self._metric_card("Вероятность банкротства", f"{probability * 100:.2f}%", risk_color))
        metrics_layout.addWidget(self._metric_card("Слабых блоков", str(len(self.agent_report.weak_blocks)), theme.risk_danger))
        risk_layout.addLayout(metrics_layout)

        progress_bar = RoundedProgressBar(probability, risk_color)
        risk_layout.addWidget(progress_bar)

        summary = QLabel(self._format_summary_text(self.agent_report.risk_summary))
        summary.setObjectName("MutedLabel")
        summary.setFont(QFont(FONT_FAMILY, 16))
        summary.setStyleSheet("font-size: 16px;")
        summary.setWordWrap(True)
        risk_layout.addWidget(summary)

        action = QLabel(self.agent_report.risk_action)
        action.setObjectName("ActionBox")
        action.setFont(QFont(FONT_FAMILY, 16, QFont.Weight.Bold))
        action.setStyleSheet("font-size: 16px; font-weight: 700;")
        action.setWordWrap(True)
        risk_layout.addWidget(action)

        return risk_card

    def _metric_card(self, title: str, value: str, color: str) -> QFrame:
        card = QFrame()
        card.setObjectName("MetricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setObjectName("MutedLabel")
        title_label.setFont(QFont(FONT_FAMILY, 15, QFont.Weight.Bold))
        title_label.setStyleSheet("font-size: 15px; font-weight: 700;")
        layout.addWidget(title_label)

        value_label = QLabel(value)
        value_label.setFont(QFont(FONT_FAMILY, 19, QFont.Weight.Bold))
        value_label.setStyleSheet(f"color: {color}; font-size: 19px; font-weight: 800;")
        layout.addWidget(value_label)

        return card

    def _build_agent_card(self) -> QFrame:
        theme = get_theme()
        agent_card = QFrame()
        agent_card.setObjectName("AgentCard")
        layout = QVBoxLayout(agent_card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel("Интерпретация агента")
        title.setFont(QFont(FONT_FAMILY, 20, QFont.Weight.Bold))
        title.setStyleSheet("font-size: 20px; font-weight: 800;")
        layout.addWidget(title)

        layout.addWidget(
            self._build_indicator_block(
                "Слабые блоки",
                self.agent_report.weak_blocks,
                empty_text="Критичных слабых блоков по установленному порогу не выявлено.",
                title_color=theme.risk_danger,
                accent_color=theme.risk_danger,
                show_recommendation=True,
            )
        )
        layout.addWidget(
            self._build_indicator_block(
                "Сильные блоки",
                self.agent_report.strong_blocks,
                empty_text="Явно выраженных сильных блоков по установленному порогу не выявлено.",
                title_color=theme.risk_success,
                accent_color=theme.risk_success,
                show_recommendation=False,
            )
        )
        layout.addWidget(self._build_text_block("Рекомендации", self.agent_report.recommendations))

        return agent_card

    def _build_indicator_block(
        self,
        title: str,
        blocks: list[IndicatorAdvice],
        empty_text: str,
        title_color: str,
        accent_color: str,
        show_recommendation: bool,
    ) -> QFrame:
        block = QFrame()
        block.setObjectName("InfoBlock")
        layout = QVBoxLayout(block)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(7)

        title_label = QLabel(title)
        title_label.setFont(QFont(FONT_FAMILY, 16, QFont.Weight.Bold))
        title_label.setStyleSheet(
            f"font-size: 16px; font-weight: 800; color: {title_color}; "
            f"background-color: {get_theme().background}; border-radius: 8px; padding: 8px 10px;"
        )
        layout.addWidget(title_label)

        if not blocks:
            layout.addWidget(self._text_line(empty_text))
            return block

        for advice in blocks:
            prefix = f"{advice.name}: {advice.value:.3f}"
            suffix = f" — {advice.recommendation}." if show_recommendation else "."
            layout.addWidget(self._rich_text_line(prefix, suffix, accent_color))

        return block

    def _build_text_block(self, title: str, lines: list[str]) -> QFrame:
        block = QFrame()
        block.setObjectName("InfoBlock")
        layout = QVBoxLayout(block)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(7)

        title_label = QLabel(title)
        title_label.setFont(QFont(FONT_FAMILY, 16, QFont.Weight.Bold))
        title_label.setStyleSheet(
            f"font-size: 16px; font-weight: 800; color: {get_theme().text}; "
            f"background-color: {get_theme().background}; border-radius: 8px; padding: 8px 10px;"
        )
        layout.addWidget(title_label)

        for text in lines:
            layout.addWidget(self._text_line(text))

        return block

    def _build_prediction_card(self, regression_values: np.ndarray) -> QFrame:
        card = QFrame()
        card.setObjectName("PredictionCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel("Прогнозные показатели")
        title.setFont(QFont(FONT_FAMILY, 20, QFont.Weight.Bold))
        title.setStyleSheet("font-size: 20px; font-weight: 800;")
        layout.addWidget(title)
        layout.addWidget(self._build_prediction_table(regression_values))

        return card

    def _build_prediction_table(self, regression_values: np.ndarray) -> QTableWidget:
        theme = get_theme()
        table = QTableWidget(len(PREDICTION_FULL_LABELS), 2)
        table.setHorizontalHeaderLabels(["Показатель", "Значение"])
        table.setFont(QFont(FONT_FAMILY, 12))
        table.horizontalHeader().setFont(QFont(FONT_FAMILY, 12, QFont.Weight.Bold))
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.setMinimumHeight(470)
        table.setWordWrap(True)

        for row_index, value in enumerate(regression_values):
            name_item = QTableWidgetItem(PREDICTION_FULL_LABELS[row_index])
            name_item.setFont(QFont(FONT_FAMILY, 11, QFont.Weight.Bold))
            name_item.setForeground(QBrush(QColor(theme.text)))
            name_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row_index, 0, name_item)

            value_item = QTableWidgetItem(f"{value:.3f}")
            value_item.setFont(QFont(FONT_FAMILY, 15, QFont.Weight.Bold))
            value_item.setForeground(QBrush(QColor(self._score_color(float(value)))))
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row_index, 1, value_item)
            table.setRowHeight(row_index, 44)

        return table

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

    def _context_lines(self) -> list[str]:
        lines = [f"{key}: {value}" for key, value in self.context.items() if value]
        return lines or ["Источник: ручной ввод"]

    @staticmethod
    def _format_summary_text(text: str) -> str:
        marker = "Наиболее слабые направления:"
        if marker in text:
            return text.replace(f" {marker}", f"\n{marker}")
        return text


    @staticmethod
    def _text_line(text: str) -> QLabel:
        label = QLabel(text)
        label.setFont(QFont(FONT_FAMILY, 16))
        label.setStyleSheet("font-size: 16px;")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _rich_text_line(prefix: str, suffix: str, color: str) -> QLabel:
        label = QLabel(
            f"<span style='font-weight:800; color:{color};'>{escape(prefix)}</span>{escape(suffix)}"
        )
        label.setFont(QFont(FONT_FAMILY, 16))
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setStyleSheet("font-size: 16px;")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _risk_color(risk_level: str) -> str:
        theme = get_theme()
        if risk_level == "высокий":
            return theme.risk_danger
        if risk_level == "средний":
            return theme.risk_warning
        return theme.risk_success

    @staticmethod
    def _score_color(value: float) -> str:
        theme = get_theme()
        if value < 0.4:
            return theme.risk_danger
        if value < 0.7:
            return theme.risk_warning
        return theme.risk_success