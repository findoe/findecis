from __future__ import annotations

from dataclasses import dataclass


#Одна историческая точка предприятия для PDF-отчета
@dataclass(frozen=True)
class HistoryPoint:
    year: str
    probability: float
    regression_values: list[float]
    industry: str = ""
