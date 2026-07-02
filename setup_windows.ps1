param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "LocalAI_V3 Windows Setup" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan
Write-Host ""

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

Write-Host "[INFO] Project root: $RootDir"

if (!(Test-Path ".\.venv")) {
    Write-Host "[INFO] Creating virtual environment..."
    python -m venv .venv
} else {
    Write-Host "[OK] Virtual environment already exists." -ForegroundColor Green
}

$VenvPython = ".\.venv\Scripts\python.exe"

if (!(Test-Path $VenvPython)) {
    Write-Host "[ERROR] Venv Python not found." -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Venv Python found." -ForegroundColor Green

Write-Host "[INFO] Upgrading pip..."
& $VenvPython -m pip install --upgrade pip

if (!$SkipInstall) {
    if (Test-Path ".\requirements.txt") {
        Write-Host "[INFO] Installing dependencies..."
        & $VenvPython -m pip install -r requirements.txt
    } else {
        Write-Host "[WARN] requirements.txt not found." -ForegroundColor Yellow
    }
} else {
    Write-Host "[INFO] SkipInstall enabled. Dependency installation skipped." -ForegroundColor Yellow
}

if (Test-Path ".\tools\check_system.py") {
    Write-Host ""
    Write-Host "[INFO] Running system check..."
    & $VenvPython .\tools\check_system.py
} else {
    Write-Host "[WARN] tools/check_system.py not found." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Setup finished." -ForegroundColor Green
Write-Host ""
Write-Host "Next:"
Write-Host "1. Put model files in models/"
Write-Host "2. Put Qwen-VL files in models/qwen_vl/"
Write-Host "3. Put voice sample at data/reference.wav"
Write-Host "4. Start app with run_localai.bat"
Write-Host ""
