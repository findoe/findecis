from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HistoryPoint:
    year: str
    probability: float
    regression_values: list[float]
    industry: str = ""
