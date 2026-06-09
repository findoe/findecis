from __future__ import annotations

from financial_app.config import (
    BLUE,
    BLUE_HOVER,
    DARK_BG,
    DARK_BORDER,
    DARK_CARD,
    DARK_FIELD,
    DARK_PANEL,
    FONT_FAMILY,
    TEXT_COLOR,
)


def build_stylesheet() -> str:
    return f"""
        QWidget {{
            background-color: {DARK_BG};
            color: {TEXT_COLOR};
            font-family: {FONT_FAMILY};
            font-size: 13px;
        }}

        QMainWindow {{
            background-color: {DARK_BG};
        }}

        QFrame#SearchPanel,
        QFrame#RiskCard {{
            background-color: {DARK_PANEL};
            border: 1px solid #2F2F2F;
            border-radius: 10px;
        }}

        QLabel {{
            background: transparent;
        }}

        QLineEdit,
        QComboBox {{
            background-color: {DARK_FIELD};
            color: {TEXT_COLOR};
            border: 1px solid {DARK_BORDER};
            border-radius: 5px;
            padding: 6px 8px;
            min-height: 20px;
            selection-background-color: {BLUE};
        }}

        QLineEdit:focus,
        QComboBox:focus {{
            border: 1px solid {BLUE_HOVER};
        }}

        QComboBox::drop-down {{
            border: 0;
            width: 26px;
        }}

        QComboBox QAbstractItemView {{
            background-color: {DARK_FIELD};
            color: {TEXT_COLOR};
            border: 1px solid {DARK_BORDER};
            selection-background-color: {BLUE};
        }}

        QPushButton {{
            background-color: {BLUE};
            color: {TEXT_COLOR};
            border: none;
            border-radius: 6px;
            padding: 8px 12px;
            font-weight: 700;
            min-height: 22px;
        }}

        QPushButton:hover {{
            background-color: {BLUE_HOVER};
        }}

        QPushButton:pressed {{
            background-color: #195987;
        }}

        QPushButton#SecondaryButton {{
            background-color: {DARK_FIELD};
            border: 1px solid {DARK_BORDER};
        }}

        QPushButton#SecondaryButton:hover {{
            background-color: #3E4143;
        }}

        QTabWidget::pane {{
            border: 1px solid {DARK_BORDER};
            background-color: {DARK_CARD};
        }}

        QTabBar::tab {{
            background-color: {DARK_BG};
            color: {TEXT_COLOR};
            padding: 7px 10px;
            border: none;
            font-weight: 700;
        }}

        QTabBar::tab:selected {{
            background-color: {DARK_CARD};
        }}

        QTabBar::tab:hover {{
            background-color: #3A3A3A;
        }}

        QTableWidget {{
            background-color: {DARK_CARD};
            alternate-background-color: #303030;
            color: {TEXT_COLOR};
            gridline-color: #3C3C3C;
            border: 1px solid {DARK_BORDER};
            border-radius: 6px;
        }}

        QHeaderView::section {{
            background-color: {DARK_FIELD};
            color: {TEXT_COLOR};
            border: none;
            padding: 7px;
            font-weight: 700;
        }}

        QMessageBox {{
            background-color: {DARK_BG};
        }}
    """
