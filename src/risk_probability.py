from __future__ import annotations

from typing import Sequence

import numpy as np

from src.config import BANKRUPTCY_RISK_WEIGHTS



RISK_CALIBRATOR = {
    "scaler_mean": [
        0.3907902058987201,
        0.43530884808013354,
        0.2573734001112966,
        0.30342237061769617,
        0.3134390651085142,
        0.4237618252643294,
        0.07178631051752922,
        0.15776293823038398,
        0.18503060656649972,
        0.13800779076238176,
        0.17125765164162493,
    ],
    "scaler_scale": [
        0.1868746537777366,
        0.2599357267127756,
        0.3535256891426107,
        0.21328505217219304,
        0.17176608025006765,
        0.26642522034010824,
        0.1753279199534881,
        0.22037834244163948,
        0.22853276557806665,
        0.15383081687686387,
        0.1584355190297337,
    ],
    "coefficients": [
        0.22955986632683653,
        -0.3898349661536141,
        -0.7962083841279248,
        -0.8065552538712585,
        0.02179765239310442,
        0.05325592936299122,
        0.006268816959128612,
        -0.27843686749385943,
        -0.08178829386930249,
        0.0031622402158565793,
        -0.4671266625044499,
    ],
    "intercept": 1.453846587226355,
    "base_probability": 0.05,
    "logistic_weight": 0.45,
    "deficit_weight": 0.50,
    "deficit_threshold": 0.60,
    "deficit_power": 1.30,
    "deficit_min_probability": 0.08,
    "deficit_max_probability": 0.96,
}


class BankruptcyRiskEstimator:
    def __init__(self) -> None:
        self.calibrator = RISK_CALIBRATOR
        self.fallback_weights = self._normalize_weights(BANKRUPTCY_RISK_WEIGHTS)

    def predict(self, regression_values: Sequence[float]) -> float:
        values = self._prepare_values(regression_values)
        probability = self._predict_with_calibrator(values)
        return float(np.clip(probability, 0.0, 1.0))

    def predict_many(self, regression_rows: Sequence[Sequence[float]]) -> np.ndarray:
        rows = np.asarray(regression_rows, dtype=float)
        if rows.size == 0:
            return np.array([])
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)

        return np.array([self.predict(row) for row in rows], dtype=float)

    @staticmethod
    def _prepare_values(regression_values: Sequence[float]) -> np.ndarray:
        values = np.asarray(regression_values, dtype=float).reshape(-1)
        if values.size != 11:
            raise ValueError("Для расчета вероятности банкротства нужно 11 прогнозных показателей.")
        return np.clip(values, 0.0, 1.0)

    def _predict_with_calibrator(self, values: np.ndarray) -> float:
        mean = np.asarray(self.calibrator["scaler_mean"], dtype=float)
        scale = np.asarray(self.calibrator["scaler_scale"], dtype=float)
        coefficients = np.asarray(self.calibrator["coefficients"], dtype=float)
        intercept = float(self.calibrator["intercept"])

        scale = np.where(scale == 0, 1.0, scale)
        standardized = (values - mean) / scale
        logit = intercept + float(np.dot(coefficients, standardized))
        logit = float(np.clip(logit, -35.0, 35.0))
        logistic_probability = 1.0 / (1.0 + np.exp(-logit))

        base_probability = float(self.calibrator["base_probability"])
        logistic_weight = float(self.calibrator["logistic_weight"])
        deficit_weight = float(self.calibrator["deficit_weight"])
        deficit_probability = self._deficit_probability(values)

        return base_probability + logistic_weight * logistic_probability + deficit_weight * deficit_probability

    def _deficit_probability(self, values: np.ndarray) -> float:
        threshold = max(float(self.calibrator["deficit_threshold"]), 0.01)
        power = float(self.calibrator["deficit_power"])
        minimum = float(self.calibrator["deficit_min_probability"])
        maximum = float(self.calibrator["deficit_max_probability"])

        deficit = np.clip((threshold - values) / threshold, 0.0, 1.0)
        deficit_score = float(np.dot(deficit, self.fallback_weights))

        return minimum + (maximum - minimum) * (deficit_score ** power)

    @staticmethod
    def _normalize_weights(weights: Sequence[float]) -> np.ndarray:
        values = np.asarray(weights, dtype=float)
        if values.size != 11 or np.isclose(values.sum(), 0.0):
            return np.ones(11, dtype=float) / 11.0
        return values / values.sum()
