from __future__ import annotations

from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_TITLE = "Анализ финансового состояния предприятия"

MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_model.keras"
SCALER_PATH = PROJECT_ROOT / "artifacts" / "scaler.pkl"
RISK_CALIBRATOR_PATH = PROJECT_ROOT / "artifacts" / "risk_calibrator.json"
DATA_PATH = PROJECT_ROOT / "data" / "data_processed.csv"
DATA_DELIMITER = ";"

#Количество входных показателей, которые подаются в модель
INPUT_COUNT = 35
RISK_THRESHOLD = 0.5
ALL_INDUSTRIES_VALUE = "Все"

#Базовые параметры интерфейса
FONT_FAMILY = "Bahnschrift"
ERROR_TITLE = "Ошибка!"

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 680
WINDOW_MIN_WIDTH = 1180
WINDOW_MIN_HEIGHT = 580
SEARCH_PANEL_WIDTH = 280
SEARCH_CONTROL_WIDTH = 235

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

#Краткие названия прогнозируемых показателей
PREDICTION_LABELS = [
    "Reg", "Kredit", "Teh", "Market", "Staff", "Psich",
    "Ability", "Turn", "Finn", "Z25", "Z35",
]


#Веса резервной эвристики для расчета вероятности банкротства по 11 прогнозным показателям
#Основной расчет выполняется через risk_calibrator.json; эти веса используются только если файл недоступен.
BANKRUPTCY_RISK_WEIGHTS = [
    0.05,  # Reg
    0.12,  # Kredit
    0.06,  # Teh
    0.08,  # Market
    0.06,  # Staff
    0.05,  # Psich
    0.16,  # Ability
    0.10,  # Turn
    0.17,  # Finn
    0.06,  # Z25
    0.09,  # Z35
]

#Полные названия прогнозируемых показателей для окна результатов
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

#Группы вкладок и количество полей в каждой вкладке
CATEGORIES = [
    ("Регион. и отрасл. специфика", 5),
    ("Кредитная история", 4),
    ("Техн. оснащенность", 3),
    ("Рыночный потенциал", 5),
    ("Кадровое обеспечение", 7),
    ("Платежеспособность", 6),
    ("Рентабельность", 5),
]

#Подписи для 35 входных показателей
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

#Возможные названия служебных столбцов в CSV-файле
INDUSTRY_COLUMN_CANDIDATES = ("Отрасль экономики", "Отрасль", "industry")
INN_COLUMN_CANDIDATES = ("ИНН", "inn")
YEAR_COLUMN_CANDIDATES = ("Год", "year")