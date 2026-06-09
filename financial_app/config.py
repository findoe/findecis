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

DARK_BG = "#202020"
DARK_PANEL = "#252525"
DARK_CARD = "#2B2B2B"
DARK_FIELD = "#343638"
DARK_BORDER = "#565B5E"
TEXT_COLOR = "#FFFFFF"
MUTED_TEXT_COLOR = "#A7A7A7"
BLUE = "#1F6AA5"
BLUE_HOVER = "#2A7AB8"
DANGER = "#FF3333"
SUCCESS = "#30C060"
WARNING = "#F4B942"

SEARCH_PANEL_WIDTH = 260
SEARCH_CONTROL_WIDTH = 220

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
