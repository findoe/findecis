from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from keras.models import load_model

from src.risk_probability import BankruptcyRiskEstimator



class FinancialModelService:
    def __init__(self, model_path: Path, scaler_path: Path) -> None:
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.risk_estimator = BankruptcyRiskEstimator()

    def predict(self, feature_values: list[float]) -> tuple[np.ndarray, float]:
        regression_output = self._predict_regression(np.array(feature_values).reshape(1, -1))
        regression_values = regression_output[0]
        return regression_values, self.risk_estimator.predict(regression_values)

    def predict_many(self, feature_rows: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        if not feature_rows:
            return np.array([]), np.array([])

        regression_output = self._predict_regression(np.array(feature_rows))
        probabilities = self.risk_estimator.predict_many(regression_output)
        return regression_output, probabilities

    def _predict_regression(self, input_data: np.ndarray) -> np.ndarray:
        input_data_scaled = self.scaler.transform(input_data)
        regression_output, *_ = self.model.predict(input_data_scaled, verbose=0)
        return regression_output