from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from src.config import BANKRUPTCY_RISK_WEIGHTS, RISK_CALIBRATOR_PATH


#Расчет вероятности банкротства по 11 уже предсказанным регрессионным показателям.
#Классификационный выход нейросети здесь намеренно не используется, чтобы результат
#не превращался в жесткие 0%/100%.
class BankruptcyRiskEstimator:
    def __init__(self, calibrator_path: Path = RISK_CALIBRATOR_PATH) -> None:
        self.calibrator = self._load_calibrator(calibrator_path)
        self.fallback_weights = self._normalize_weights(BANKRUPTCY_RISK_WEIGHTS)

    #Вероятность для одного предприятия
    def predict(self, regression_values: Sequence[float]) -> float:
        values = self._prepare_values(regression_values)

        if self.calibrator is not None:
            probability = self._predict_with_calibrator(values)
        else:
            probability = self._predict_with_fallback(values)

        return float(np.clip(probability, 0.0, 1.0))

    #Вероятности для истории предприятия по годам
    def predict_many(self, regression_rows: Sequence[Sequence[float]]) -> np.ndarray:
        rows = np.asarray(regression_rows, dtype=float)
        if rows.size == 0:
            return np.array([])
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)

        return np.array([self.predict(row) for row in rows], dtype=float)

    #Загрузка переносимого JSON-калибратора
    @staticmethod
    def _load_calibrator(path: Path) -> dict | None:
        try:
            if not path.exists():
                return None
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError):
            return None

    #Приведение выходов регрессии к диапазону 0..1
    @staticmethod
    def _prepare_values(regression_values: Sequence[float]) -> np.ndarray:
        values = np.asarray(regression_values, dtype=float).reshape(-1)
        if values.size != 11:
            raise ValueError("Для расчета вероятности банкротства нужно 11 прогнозных показателей.")
        return np.clip(values, 0.0, 1.0)

    #Логистическая калибровка по 11 показателям: reg, kredit, ..., Z35
    def _predict_with_calibrator(self, values: np.ndarray) -> float:
        assert self.calibrator is not None

        mean = np.asarray(self.calibrator["scaler_mean"], dtype=float)
        scale = np.asarray(self.calibrator["scaler_scale"], dtype=float)
        coefficients = np.asarray(self.calibrator["coefficients"], dtype=float)
        intercept = float(self.calibrator["intercept"])

        scale = np.where(scale == 0, 1.0, scale)
        standardized = (values - mean) / scale
        logit = intercept + float(np.dot(coefficients, standardized))

        return 1.0 / (1.0 + np.exp(-logit))

    #Резервный расчет: чем ниже интегральная оценка устойчивости, тем выше риск
    def _predict_with_fallback(self, values: np.ndarray) -> float:
        stability_score = float(np.dot(values, self.fallback_weights))
        return 1.0 - stability_score

    #Нормализация весов на случай изменения списка
    @staticmethod
    def _normalize_weights(weights: Sequence[float]) -> np.ndarray:
        values = np.asarray(weights, dtype=float)
        if values.size != 11 or np.isclose(values.sum(), 0.0):
            return np.ones(11, dtype=float) / 11.0
        return values / values.sum()
