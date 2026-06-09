from __future__ import annotations


def validate_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def validate_inn(inn: str) -> bool:
    return inn.isdigit() and len(inn) == 10


def validate_year(year: str) -> bool:
    return year.isdigit() and len(year) == 4


class InputValidationError(ValueError):
    pass
