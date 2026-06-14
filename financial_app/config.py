from __future__ import annotations

from pathlib import Path

# =========================
# НАСТРОЙКИ ПРИЛОЖЕНИЯ
# =========================
APP_TITLE = "Анализ финансового состояния предприятия"

MODEL_PATH = Path("model_artifacts/best_model.keras")
SCALER_PATH = Path("model_artifacts/scaler.pkl")
DATA_PATH = Path("data_new.csv")
DATA_DELIMITER = ";"

INPUT_COUNT = 35
RISK_THRESHOLD = 0.5
ALL_INDUSTRIES_VALUE = "Все"

FONT_FAMILY = "Bahnschrift"
ERROR_TITLE = "Ошибка!"

# =========================
# РАЗМЕРЫ ОКНА
# =========================
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 680
WINDOW_MIN_WIDTH = 1180
WINDOW_MIN_HEIGHT = 580
SEARCH_PANEL_WIDTH = 280
SEARCH_CONTROL_WIDTH = 235

# =========================
# ЦВЕТОВАЯ ПАЛИТРА
# =========================
CHARCOAL_BLUE = "#264653"
VERDIGRIS = "#2a9d8f"
TUSCAN_SUN = "#e9c46a"
SANDY_BROWN = "#f4a261"
BURNT_PEACH = "#e76f51"

THEME_LIGHT = "light"
THEME_DARK = "dark"
DEFAULT_THEME = THEME_DARK

THEME_DISPLAY_NAMES = {
    THEME_LIGHT: "Светлая",
    THEME_DARK: "Темная",
}

PREDICTION_LABELS = [
    "Reg", "Kredit", "Teh", "Market", "Staff", "Psich",
    "Ability", "Turn", "Finn", "Z25", "Z35",
]

PREDICTION_FULL_LABELS = [
    "Региональная и отраслевая специфика (Reg)",
    "Кредитоспособность (Kredit)",
    "Техническая оснащенность (Teh)",
    "Рыночный потенциал (Market)",
    "Кадровое обеспечение (Staff)",
    "Морально-психологический климат (Psich)",
    "Ликвидность (Ability)",
    "Оборачиваемость (Turn)",
    "Финансовая устойчивость (Finn)",
    "Интегральный показатель Z25",
    "Интегральный показатель Z35",
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
