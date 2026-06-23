from __future__ import annotations


class InputValidationError(ValueError):
    pass


def validate_inn(inn: str) -> bool:
    return inn.isdigit() and len(inn) == 10


def validate_year(year: str) -> bool:
    return year.isdigit() and len(year) == 4