from __future__ import annotations

from dataclasses import dataclass

from financial_app.config import (
    DEFAULT_THEME,
    FONT_FAMILY,
    THEME_DARK,
    THEME_LIGHT,
)


@dataclass(frozen=True)
class ThemeColors:
    background: str
    panel: str
    card: str
    field: str
    border: str
    text: str
    muted_text: str
    placeholder_text: str
    primary: str
    primary_hover: str
    primary_pressed: str
    button_text: str
    field_text: str
    secondary_button_bg: str
    secondary_button_hover: str
    secondary_button_text: str
    selection: str
    selection_text: str
    table_alt: str
    table_grid: str
    tab_bg: str
    tab_selected: str
    tab_accent: str
    risk_danger: str
    risk_success: str
    risk_warning: str
    progress_track: str


# Тема сделана по присланным референсам:
# - светлая: кремовая база + teal для кнопок и верхних вкладок;
# - темная: cool dark + единый teal для кнопок, вкладок и кнопки «Случайно»;
# - «Очистить» в обеих темах: #DA4167.
THEMES: dict[str, ThemeColors] = {
    THEME_LIGHT: ThemeColors(
        background="#FAF8ED",
        panel="#F1E8D7",
        card="#FFFCF4",
        field="#F8F7EF",
        border="#D7D0C0",
        text="#1C2D35",
        muted_text="#6D6A5E",
        placeholder_text="#8D8A7E",
        primary="#008B84",
        primary_hover="#00766F",
        primary_pressed="#00665F",
        button_text="#FFFFFF",
        field_text="#1C2D35",
        secondary_button_bg="#DA4167",
        secondary_button_hover="#C7375B",
        secondary_button_text="#FFFFFF",
        selection="#B7DED7",
        selection_text="#1C2D35",
        table_alt="#F0EEE3",
        table_grid="#DCD6C6",
        tab_bg="#008B84",
        tab_selected="#00766F",
        tab_accent="#F2C56B",
        risk_danger="#DA4167",
        risk_success="#008B84",
        risk_warning="#D49B34",
        progress_track="#E8E2D3",
    ),
    THEME_DARK: ThemeColors(
        background="#0B1117",
        panel="#141D24",
        card="#0B1117",
        field="#19242B",
        border="#263942",
        text="#F2C56B",
        muted_text="#D8C690",
        placeholder_text="#C8B981",
        primary="#12A884",
        primary_hover="#0F9778",
        primary_pressed="#0B7E64",
        button_text="#FFF3C4",
        field_text="#F2C56B",
        secondary_button_bg="#DA4167",
        secondary_button_hover="#C7375B",
        secondary_button_text="#FFFFFF",
        selection="#2E806E",
        selection_text="#FFFFFF",
        table_alt="#101922",
        table_grid="#263942",
        tab_bg="#12A884",
        tab_selected="#0F9778",
        tab_accent="#F2C56B",
        risk_danger="#DA4167",
        risk_success="#12A884",
        risk_warning="#F2C56B",
        progress_track="#19242B",
    ),
}

_current_theme_name = DEFAULT_THEME


def normalize_theme(theme_name: str | None) -> str:
    if theme_name in THEMES:
        return theme_name
    return DEFAULT_THEME


def set_current_theme(theme_name: str | None) -> None:
    global _current_theme_name
    _current_theme_name = normalize_theme(theme_name)


def get_current_theme_name() -> str:
    return _current_theme_name


def get_theme(theme_name: str | None = None) -> ThemeColors:
    return THEMES[normalize_theme(theme_name or _current_theme_name)]


def build_stylesheet(theme_name: str | None = None) -> str:
    theme = get_theme(theme_name)

    return f"""
        QWidget {{
            background-color: {theme.background};
            color: {theme.text};
            font-family: {FONT_FAMILY};
            font-size: 14px;
        }}

        QMainWindow,
        QDialog {{
            background-color: {theme.background};
        }}

        QFrame#SearchPanel,
        QFrame#RiskCard {{
            background-color: {theme.panel};
            border: 1px solid {theme.border};
            border-radius: 10px;
        }}

        QLabel {{
            background: transparent;
        }}

        QLabel#MutedLabel {{
            color: {theme.muted_text};
        }}

        QLineEdit,
        QComboBox {{
            background-color: {theme.field};
            color: {theme.field_text};
            border: 1px solid {theme.border};
            border-radius: 6px;
            padding: 7px 9px;
            min-height: 23px;
            selection-background-color: {theme.selection};
            selection-color: {theme.selection_text};
        }}

        QLineEdit:focus,
        QComboBox:focus {{
            border: 1px solid {theme.primary};
        }}

        QLineEdit::placeholder {{
            color: {theme.placeholder_text};
        }}

        QComboBox::drop-down {{
            border: 0;
            width: 26px;
        }}

        QComboBox QAbstractItemView {{
            background-color: {theme.field};
            color: {theme.field_text};
            border: 1px solid {theme.border};
            selection-background-color: {theme.selection};
            selection-color: {theme.selection_text};
            outline: 0;
        }}

        QPushButton {{
            background-color: {theme.primary};
            color: {theme.button_text};
            border: none;
            border-radius: 7px;
            padding: 9px 13px;
            font-weight: 700;
            min-height: 24px;
        }}

        QPushButton:hover {{
            background-color: {theme.primary_hover};
            color: {theme.button_text};
        }}

        QPushButton:pressed {{
            background-color: {theme.primary_pressed};
            color: {theme.button_text};
        }}

        QPushButton#SecondaryButton {{
            background-color: {theme.secondary_button_bg};
            color: {theme.secondary_button_text};
            border: none;
        }}

        QPushButton#SecondaryButton:hover {{
            background-color: {theme.secondary_button_hover};
            color: {theme.secondary_button_text};
        }}

        QPushButton#SecondaryButton:pressed {{
            background-color: {theme.secondary_button_hover};
            color: {theme.secondary_button_text};
        }}

        QTabWidget::pane {{
            border: 1px solid {theme.border};
            background-color: {theme.card};
            border-radius: 6px;
        }}

        QTabBar::tab {{
            background-color: {theme.tab_bg};
            color: {theme.button_text};
            padding: 8px 11px;
            border: none;
            font-weight: 700;
        }}

        QTabBar::tab:selected {{
            background-color: {theme.tab_selected};
            color: {theme.button_text};
            border-top: 2px solid {theme.tab_accent};
        }}

        QTabBar::tab:hover {{
            background-color: {theme.primary_hover};
            color: {theme.button_text};
        }}

        QTableWidget {{
            background-color: {theme.card};
            alternate-background-color: {theme.table_alt};
            color: {theme.text};
            gridline-color: {theme.table_grid};
            border: 1px solid {theme.border};
            border-radius: 6px;
        }}

        QTableWidget::item:selected {{
            background-color: {theme.selection};
            color: {theme.selection_text};
        }}

        QHeaderView::section {{
            background-color: {theme.panel};
            color: {theme.text};
            border: none;
            padding: 7px;
            font-weight: 700;
        }}

        QProgressBar {{
            background-color: {theme.progress_track};
            border: 1px solid {theme.border};
            border-radius: 7px;
        }}

        QMessageBox {{
            background-color: {theme.background};
        }}
    """
