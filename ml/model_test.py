from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from keras.models import load_model

import sys

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.risk_probability import BankruptcyRiskEstimator

# =========================
#       ТЕСТИРОВАНИЕ
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_model.keras"
SCALER_PATH = PROJECT_ROOT / "artifacts" / "scaler.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "data_processed.csv"


#Загрузка модели и скейлера
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

#Загрузка данных
data = pd.read_csv(DATA_PATH, sep=";")

#Выбор случайной компании из датасета
row = data.sample(1, random_state=None)

#Мета-информация выбранной компании
inn = row["ИНН"].values[0]
year = row["Год"].values[0]
industry = row["Отрасль экономики"].values[0]

#Признаки x1 - x35
FEATURES = [f"x{i}" for i in range(1, 36)]
X = row[FEATURES].values

#Масштабирование входных признаков
X_scaled = scaler.transform(X)

#Предсказание регрессионных показателей и вероятности банкротства
#Вероятность считается по 11 регрессионным показателям, а не по бинарному выходу модели.
reg, _cls = model.predict(X_scaled, verbose=0)

risk_estimator = BankruptcyRiskEstimator()
prob = risk_estimator.predict(reg[0])


#Вывод общей инфо
print("\nРЕЗУЛЬТАТ:")
print("ИНН:", inn)
print("Год:", year)
print("Отрасль:", industry)

print("\nВероятность банкротства:", round(prob, 6))


#Названия регрессионных выходов
REG_NAMES = [
    "reg", "kredit", "teh", "market", "staff", "psich",
    "ability", "turn", "finn", "Z25", "Z35"
]


#Вывод регрессионных прогнозов
print("\nРегрессия (по переменным):")
for name, value in zip(REG_NAMES, reg[0]):
    print(f"{name:10s}: {value:.4f}")

print("\nКЛАСС РИСКА:")


#Интерпретация вероятности
if prob < 0.3:
    print("🟢 Низкий риск")
elif prob < 0.6:
    print("🟡 Средний риск")
else:
    print("🔴 Высокий риск банкротства")