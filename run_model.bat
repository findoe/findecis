@echo off
chcp 65001 > nul
setlocal

echo.
echo =========================================
echo   Подготовка обучения финансовой модели
echo =========================================
echo.

rem Переход в корень
cd /d "%~dp0"

rem Путь к папке с результатами обучения
set "ARTIFACTS=%~dp0artifacts"

echo Проверяется структура проекта...

rem Проверка папки artifacts
if not exist "%ARTIFACTS%" (
    echo.
    echo Ошибка: папка artifacts не найдена.
    echo Ожидаемый путь: "%ARTIFACTS%"
    echo Проверьте, что bat-файл находится в корне проекта.
    pause
    exit /b 1
)

echo Структура проекта найдена.
echo.
echo Удаляются старые файлы обучения из папки artifacts...

rem Удаление всех файлов внутри artifacts и ее подпапок
del /s /q "%ARTIFACTS%\*" > nul 2>&1

echo Предыдущие файлы обучения успешно удалены.
echo.

set "PYTHON_CMD=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_CMD=%~dp0.venv\Scripts\python.exe"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON_CMD=%~dp0..\.venv\Scripts\python.exe"

echo Используемый интерпретатор Python:
echo "%PYTHON_CMD%"
echo.
echo Начинается обучение модели. Пожалуйста, дождитесь завершения процесса...
echo.

rem Запуск обучения модели
"%PYTHON_CMD%" "%~dp0ml\model.py"

if errorlevel 1 (
    echo.
    echo Обучение модели завершилось с ошибкой!
    echo Проверьте сообщения выше.
    pause
    exit /b 1
)

echo.
echo ======================================
echo   Обучение модели успешно завершено!
echo ======================================
echo.
pause