@echo off
title LocalAI_V3

cd /d "%~dp0"

echo.
echo Starting LocalAI_V3...
echo.

if not exist ".\.venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found.
    echo.
    echo Please run setup first:
    echo powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
    echo.
    pause
    exit /b 1
)

echo [INFO] Checking required Python packages...
".\.venv\Scripts\python.exe" -c "import PySide6, torch, numpy" >nul 2>nul

if errorlevel 1 (
    echo.
    echo [ERROR] Required Python packages are missing.
    echo.
    echo This usually means you used:
    echo powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -SkipInstall
    echo.
    echo Please install dependencies first:
    echo powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
    echo.
    echo After setup finishes, run this file again:
    echo .\run_localai.bat
    echo.
    pause
    exit /b 1
)

if not exist ".\models\google_gemma-4-E4B-it-Q5_K_M.gguf" (
    echo.
    echo [WARN] Main Gemma model file is missing.
    echo Expected:
    echo models\google_gemma-4-E4B-it-Q5_K_M.gguf
    echo.
    echo Please read:
    echo docs\MODEL_SETUP.md
    echo.
)

if not exist ".\data\reference.wav" (
    echo.
    echo [WARN] Voice reference file is missing.
    echo Expected:
    echo data\reference.wav
    echo.
    echo XTTS voice output may not work until this file is added.
    echo.
)

echo [INFO] Launching app...
echo.

".\.venv\Scripts\python.exe" ".\main.py"

echo.
echo LocalAI_V3 closed.
pause
