from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import (
    ALL_INDUSTRIES_VALUE,
    DATA_DELIMITER,
    INDUSTRY_COLUMN_CANDIDATES,
    INN_COLUMN_CANDIDATES,
    INPUT_COUNT,
    YEAR_COLUMN_CANDIDATES,
)



@dataclass(frozen=True)
class DataColumns:
    industry: str
    inn: str
    year: str


def resolve_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
    fallback_index: int,
) -> str:
    normalized_columns = {str(column).strip().lower(): column for column in dataframe.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized_columns:
            return normalized_columns[key]

    for column in dataframe.columns:
        column_name = str(column).strip().lower()
        if any(candidate.strip().lower() in column_name for candidate in candidates):
            return column

    if fallback_index >= len(dataframe.columns):
        raise ValueError(f"В данных нет столбца с индексом {fallback_index}")

    return dataframe.columns[fallback_index]


class CompanyDataRepository:
    def __init__(self, data_path: Path, delimiter: str = DATA_DELIMITER) -> None:
        self.data = pd.read_csv(data_path, delimiter=delimiter)
        self.columns = self._resolve_columns()
        self._feature_columns = [f"x{i}" for i in range(1, INPUT_COUNT + 1)]
        self._normalize_key_columns()
        self._validate_feature_columns()

    @property
    def feature_columns(self) -> list[str]:
        return self._feature_columns

    @property
    def industries(self) -> list[str]:
        values = self.data[self.columns.industry].dropna().unique().tolist()
        return sorted(
            str(value)
            for value in values
            if str(value).strip() and str(value).strip().lower() != "nan"
        )

    def find_by_inn_and_year(self, inn: str, year: str = "") -> pd.Series | None:
        filtered = self.data[self.data[self.columns.inn] == inn]
        if year:
            filtered = filtered[filtered[self.columns.year] == year]
        return None if filtered.empty else filtered.iloc[0]

    def get_company_history(self, inn: str) -> pd.DataFrame:
        history = self.data[self.data[self.columns.inn] == inn].copy()
        if history.empty:
            return history

        history["_year_sort"] = pd.to_numeric(history[self.columns.year], errors="coerce")
        return history.sort_values(["_year_sort", self.columns.year], kind="stable").drop(columns=["_year_sort"])

    def get_random_by_industry(self, industry: str) -> pd.Series | None:
        filtered = self.data
        if industry != ALL_INDUSTRIES_VALUE:
            filtered = filtered[filtered[self.columns.industry] == industry]
        return None if filtered.empty else filtered.sample(1).iloc[0]

    def get_inn(self, row: pd.Series) -> str:
        return str(row[self.columns.inn])

    def get_year(self, row: pd.Series) -> str:
        return str(row[self.columns.year])

    def get_industry(self, row: pd.Series) -> str:
        return str(row[self.columns.industry])

    def get_feature_values(self, row: pd.Series) -> list[float]:
        return [float(row[column]) for column in self.feature_columns]

    def _resolve_columns(self) -> DataColumns:
        return DataColumns(
            industry=resolve_column(self.data, INDUSTRY_COLUMN_CANDIDATES, fallback_index=0),
            inn=resolve_column(self.data, INN_COLUMN_CANDIDATES, fallback_index=1),
            year=resolve_column(self.data, YEAR_COLUMN_CANDIDATES, fallback_index=2),
        )

    def _normalize_key_columns(self) -> None:
        for column in (self.columns.industry, self.columns.inn, self.columns.year):
            self.data[column] = self.data[column].astype(str)

    def _validate_feature_columns(self) -> None:
        missing_columns = [column for column in self.feature_columns if column not in self.data.columns]
        if missing_columns:
            raise ValueError("В данных отсутствуют столбцы: " + ", ".join(missing_columns))