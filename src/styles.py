from __future__ import annotations

from dataclasses import dataclass

from src.config import DEFAULT_THEME, FONT_FAMILY, THEME_DARK, THEME_LIGHT



@dataclass(frozen=True)
class ThemeColors:
    background: str
    panel: str
    card: str
    elevated: str
    field: str
    border: str
    soft_border: str
    text: str
    muted_text: str
    placeholder_text: str
    primary: str
    primary_hover: str
    primary_pressed: str
    accent: str
    accent_alt: str
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
    subtle: str


THEMES: dict[str, ThemeColors] = {
    THEME_LIGHT: ThemeColors(
        background="#F5F7FB",
        panel="#FFFFFF",
        card="#FFFFFF",
        elevated="#F8FAFC",
        field="#F8FAFC",
        border="#D8DEE9",
        soft_border="#E8EDF5",
        text="#111827",
        muted_text="#64748B",
        placeholder_text="#94A3B8",
        primary="#2563EB",
        primary_hover="#1D4ED8",
        primary_pressed="#1E40AF",
        accent="#06B6D4",
        accent_alt="#7C3AED",
        button_text="#FFFFFF",
        field_text="#111827",
        secondary_button_bg="#EC4899",
        secondary_button_hover="#DB2777",
        secondary_button_text="#FFFFFF",
        selection="#DBEAFE",
        selection_text="#111827",
        table_alt="#F8FAFC",
        table_grid="#E5EAF2",
        tab_bg="#EEF2FF",
        tab_selected="#2563EB",
        tab_accent="#06B6D4",
        risk_danger="#E11D48",
        risk_success="#0F766E",
        risk_warning="#D97706",
        progress_track="#E8EDF5",
        subtle="#F1F5F9",
    ),
    THEME_DARK: ThemeColors(
        background="#080D19",
        panel="#111827",
        card="#151D33",
        elevated="#1B2540",
        field="#101827",
        border="#2A3554",
        soft_border="#202B49",
        text="#F8FAFC",
        muted_text="#A8B3CF",
        placeholder_text="#70809F",
        primary="#7C3AED",
        primary_hover="#8B5CF6",
        primary_pressed="#6D28D9",
        accent="#22D3EE",
        accent_alt="#F472B6",
        button_text="#FFFFFF",
        field_text="#F8FAFC",
        secondary_button_bg="#EC4899",
        secondary_button_hover="#DB2777",
        secondary_button_text="#FFFFFF",
        selection="#312E81",
        selection_text="#FFFFFF",
        table_alt="#101827",
        table_grid="#2A3554",
        tab_bg="#101827",
        tab_selected="#7C3AED",
        tab_accent="#22D3EE",
        risk_danger="#F43F5E",
        risk_success="#2DD4BF",
        risk_warning="#FBBF24",
        progress_track="#0D1424",
        subtle="#0E1628",
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
    return f'''
        QWidget {{
            background-color: {theme.background};
            color: {theme.text};
            font-family: "{FONT_FAMILY}";
            font-size: 14px;
        }}

        QMainWindow,
        QDialog {{
            background-color: {theme.background};
        }}

        QFrame#AppHeader {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {theme.primary_pressed}, stop:0.55 {theme.primary}, stop:1 {theme.accent});
            border: 1px solid {theme.soft_border};
            border-radius: 18px;
        }}

        QLabel#HeaderTitle {{
            color: #FFFFFF;
            font-size: 24px;
            font-weight: 800;
        }}

        QLabel#HeaderSubtitle {{
            color: rgba(255, 255, 255, 190);
            font-size: 13px;
        }}

        QLabel#HeaderLinkLabel {{
            color: #FFFFFF;
            font-size: 18px;
            font-weight: 800;
        }}

        QPushButton#HeaderLinkButton {{
            background-color: rgba(255, 255, 255, 36);
            color: #FFFFFF;
            border: 1px solid rgba(255, 255, 255, 70);
            border-radius: 15px;
            padding: 6px 14px;
        }}

        QPushButton#HeaderLinkButton:hover {{
            background-color: rgba(255, 255, 255, 56);
        }}

        QPushButton#HeaderLinkButton:pressed {{
            background-color: rgba(255, 255, 255, 74);
        }}

        QFrame#HeaderBadge {{
            background-color: rgba(255, 255, 255, 36);
            border: 1px solid rgba(255, 255, 255, 70);
            border-radius: 15px;
        }}

        QLabel#HeaderBadgeText {{
            color: #FFFFFF;
            font-weight: 700;
        }}

        QFrame#SearchPanel,
        QFrame#RiskCard,
        QFrame#AgentCard,
        QFrame#InfoBlock,
        QFrame#PredictionCard,
        QFrame#MetricCard,
        QFrame#StatusCard {{
            background-color: {theme.card};
            border: 1px solid {theme.soft_border};
            border-radius: 16px;
        }}

        QFrame#InfoBlock,
        QFrame#MetricCard,
        QFrame#StatusCard {{
            background-color: {theme.elevated};
        }}

        QLabel {{
            background: transparent;
        }}

        QLabel#MutedLabel {{
            color: {theme.muted_text};
        }}

        QLabel#ActionBox {{
            background-color: {theme.elevated};
            border: 1px solid {theme.soft_border};
            border-radius: 12px;
            padding: 12px;
        }}

        QLineEdit,
        QComboBox {{
            background-color: {theme.field};
            color: {theme.field_text};
            border: 1px solid {theme.border};
            border-radius: 10px;
            padding: 8px 10px;
            min-height: 25px;
            selection-background-color: {theme.selection};
            selection-color: {theme.selection_text};
        }}

        QLineEdit:focus,
        QComboBox:focus {{
            border: 1px solid {theme.accent};
        }}

        QLineEdit::placeholder {{
            color: {theme.placeholder_text};
        }}

        QComboBox::drop-down {{
            border: 0;
            width: 28px;
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
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {theme.primary}, stop:1 {theme.accent_alt});
            color: {theme.button_text};
            border: none;
            border-radius: 11px;
            padding: 10px 14px;
            font-weight: 800;
            min-height: 25px;
        }}

        QPushButton:hover {{
            background-color: {theme.primary_hover};
            color: {theme.button_text};
        }}

        QPushButton:pressed {{
            background-color: {theme.primary_pressed};
            color: {theme.button_text};
        }}

        QFrame#ModeSwitch {{
            background-color: {theme.subtle};
            border: 1px solid {theme.border};
            border-radius: 10px;
        }}

        QFrame#ModeSwitch QPushButton#SegmentButton {{
            background: transparent;
            color: {theme.muted_text};
            border: none;
            border-radius: 8px;
            padding: 0;
            min-height: 0;
        }}

        QFrame#ModeSwitch QPushButton#SegmentButton:hover {{
            background-color: {theme.elevated};
            color: {theme.text};
        }}

        QFrame#ModeSwitch QPushButton#SegmentButton:checked {{
            background-color: {theme.primary};
            color: {theme.button_text};
        }}

        QPushButton#SecondaryButton {{
            background: {theme.secondary_button_bg};
            color: {theme.secondary_button_text};
            border: none;
        }}

        QPushButton#SecondaryButton:hover,
        QPushButton#SecondaryButton:pressed {{
            background-color: {theme.secondary_button_hover};
            color: {theme.secondary_button_text};
        }}

        QTabWidget::pane {{
            border: 1px solid {theme.soft_border};
            background-color: {theme.card};
            border-radius: 16px;
            top: -1px;
        }}

        QTabBar::tab {{
            background-color: {theme.tab_bg};
            color: {theme.muted_text};
            padding: 10px 13px;
            border: 1px solid {theme.soft_border};
            border-bottom: none;
            border-top-left-radius: 11px;
            border-top-right-radius: 11px;
            font-weight: 800;
            margin-right: 3px;
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


        QTabBar::scroller {{
            width: 72px;
        }}

        QTabBar QToolButton {{
            background-color: {theme.elevated};
            color: {theme.text};
            border: 1px solid {theme.soft_border};
            border-radius: 10px;
            min-width: 28px;
            max-width: 28px;
            min-height: 28px;
            max-height: 28px;
            padding: 0;
            margin: 0 2px;
        }}

        QTabBar QToolButton:hover {{
            background-color: {theme.primary};
            color: {theme.button_text};
            border: 1px solid {theme.primary_hover};
        }}

        QTabBar QToolButton:pressed {{
            background-color: {theme.primary_pressed};
            color: {theme.button_text};
            border: 1px solid {theme.primary_pressed};
        }}

        QTabBar QToolButton:disabled {{
            background-color: {theme.subtle};
            color: {theme.placeholder_text};
            border: 1px solid {theme.soft_border};
        }}

        QTableWidget {{
            background-color: {theme.card};
            alternate-background-color: {theme.table_alt};
            color: {theme.text};
            gridline-color: {theme.table_grid};
            border: 1px solid {theme.soft_border};
            border-radius: 12px;
        }}

        QTableWidget::item {{
            padding: 6px;
            border: none;
        }}

        QTableWidget::item:selected {{
            background-color: {theme.selection};
            color: {theme.selection_text};
        }}

        QHeaderView::section {{
            background-color: {theme.elevated};
            color: {theme.text};
            border: none;
            padding: 9px;
            font-weight: 800;
        }}

        QProgressBar {{
            background-color: {theme.progress_track};
            border: 1px solid {theme.border};
            border-radius: 9px;
        }}

        QScrollArea {{
            border: none;
            background: transparent;
        }}

        QScrollBar:vertical {{
            background-color: {theme.background};
            width: 10px;
            margin: 0;
        }}

        QScrollBar::handle:vertical {{
            background-color: {theme.border};
            border-radius: 5px;
            min-height: 30px;
        }}

        QScrollBar::handle:vertical:hover {{
            background-color: {theme.primary};
        }}

        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{
            height: 0;
        }}

        QMessageBox {{
            background-color: {theme.background};
        }}
    '''
