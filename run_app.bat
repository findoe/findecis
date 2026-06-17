@echo off
chcp 65001 > nul
setlocal

echo.
echo =============================
echo       Запуск приложения
echo =============================
echo.

rem Переход в корневую папку
cd /d "%~dp0"
set "PROJECT_ROOT=%CD%\"

set "PYTHON_CMD=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_CMD=%~dp0.venv\Scripts\python.exe"
if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON_CMD=%~dp0..\.venv\Scripts\python.exe"

rem Проверка Python
"%PYTHON_CMD%" --version > nul 2>&1
if errorlevel 1 (
    echo Ошибка: Python не найден или не запускается.
    echo Проверьте виртуальное окружение или Python.
    echo.
    pause
    exit /b 1
)

echo.
echo Приложение запускается. Дождитесь открытия окна...
echo.

"%PYTHON_CMD%" -m src.run

if errorlevel 1 (
    echo.
    echo Приложение завершилось с ошибкой.
    echo Проверьте сообщения выше.
    echo.
    pause
    exit /b 1
)

echo.
echo Приложение завершило работу.
pause