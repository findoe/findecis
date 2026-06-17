from __future__ import annotations


#Проверка числового значения
def validate_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


#Проверка формата ИНН
def validate_inn(inn: str) -> bool:
    return inn.isdigit() and len(inn) == 10


#Проверка формата года
def validate_year(year: str) -> bool:
    return year.isdigit() and len(year) == 4


#Ошибка проверки пользовательского ввода
class InputValidationError(ValueError):
    pass