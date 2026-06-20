from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from keras.models import load_model


#Сервис для работы с обученной нейросетевой моделью
class FinancialModelService:
    #Загрузка модели Keras и сохраненного StandardScaler
    def __init__(self, model_path: Path, scaler_path: Path) -> None:
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)

    #Предсказание регрессии и вероятности банкротства
    def predict(self, feature_values: list[float]) -> tuple[np.ndarray, float]:
        input_data = np.array(feature_values).reshape(1, -1)
        input_data_scaled = self.scaler.transform(input_data)

        regression_output, classification_output = self.model.predict(input_data_scaled, verbose=0)
        probability = float(classification_output[0][0])

        return regression_output[0], probability
    #Массовое предсказание для истории предприятия по годам
    def predict_many(self, feature_rows: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        if not feature_rows:
            return np.array([]), np.array([])

        input_data = np.array(feature_rows)
        input_data_scaled = self.scaler.transform(input_data)

        regression_output, classification_output = self.model.predict(input_data_scaled, verbose=0)
        probabilities = classification_output.reshape(-1)

        return regression_output, probabilities
