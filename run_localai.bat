@echo off
title LocalAI_V3

cd /d "%~dp0"

echo.
echo Starting LocalAI_V3...
echo.

if not exist ".\.venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found.
    echo Please run setup_windows.ps1 first.
    echo.
    pause
    exit /b 1
)

".\.venv\Scripts\python.exe" ".\main.py"

echo.
echo LocalAI_V3 closed.
pause
