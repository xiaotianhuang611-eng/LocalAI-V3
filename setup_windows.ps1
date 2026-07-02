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

function Test-CommandExists {
    param([string]$Command)
    return $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

function Find-Python {
    if (Test-CommandExists "py") {
        try {
            py -3.11 --version *> $null
            if ($LASTEXITCODE -eq 0) {
                return "py -3.11"
            }
        } catch {}

        try {
            py -3.10 --version *> $null
            if ($LASTEXITCODE -eq 0) {
                return "py -3.10"
            }
        } catch {}
    }

    if (Test-CommandExists "python") {
        try {
            python --version *> $null
            if ($LASTEXITCODE -eq 0) {
                return "python"
            }
        } catch {}
    }

    return $null
}

function Ensure-Directory {
    param([string]$Path)

    if (!(Test-Path $Path)) {
        New-Item -ItemType Directory -Force $Path | Out-Null
        Write-Host "[INFO] Created directory: $Path"
    } else {
        Write-Host "[OK] Directory exists: $Path" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "=== Runtime Directories ===" -ForegroundColor Cyan

Ensure-Directory ".\models"
Ensure-Directory ".\models\qwen_vl"
Ensure-Directory ".\data"
Ensure-Directory ".\data\knowledge"
Ensure-Directory ".\data\temp"
Ensure-Directory ".\data\rag"
Ensure-Directory ".\data\memory"

Write-Host ""
Write-Host "=== Python Environment ===" -ForegroundColor Cyan

$PythonCommand = Find-Python

if ($null -eq $PythonCommand) {
    Write-Host "[ERROR] Python was not found." -ForegroundColor Red
    Write-Host "Please install Python 3.10 or 3.11 first."
    Write-Host "Recommended: Python 3.11 for Windows."
    exit 1
}

Write-Host "[OK] Python command found: $PythonCommand" -ForegroundColor Green

if (!(Test-Path ".\.venv")) {
    Write-Host "[INFO] Creating virtual environment: .venv"
    Invoke-Expression "$PythonCommand -m venv .venv"
} else {
    Write-Host "[OK] Virtual environment already exists: .venv" -ForegroundColor Green
}

$VenvPython = ".\.venv\Scripts\python.exe"

if (!(Test-Path $VenvPython)) {
    Write-Host "[ERROR] Virtual environment Python not found: $VenvPython" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Venv Python found: $VenvPython" -ForegroundColor Green

Write-Host "[INFO] Upgrading pip..."
& $VenvPython -m pip install --upgrade pip

if (!$SkipInstall) {
    if (Test-Path ".\requirements.txt") {
        Write-Host ""
        Write-Host "=== Installing Dependencies ===" -ForegroundColor Cyan
        Write-Host "[INFO] Installing Python dependencies from requirements.txt..."
        & $VenvPython -m pip install -r requirements.txt
    } else {
        Write-Host "[WARN] requirements.txt not found. Skipping dependency installation." -ForegroundColor Yellow
    }
} else {
    Write-Host "[INFO] SkipInstall enabled. Dependency installation skipped." -ForegroundColor Yellow
}

if (Test-Path ".\tools\check_system.py") {
    Write-Host ""
    Write-Host "=== System Check ===" -ForegroundColor Cyan
    & $VenvPython .\tools\check_system.py
} else {
    Write-Host "[WARN] tools/check_system.py not found. System check skipped." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Setup finished." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Put main model here:"
Write-Host "   models\google_gemma-4-E4B-it-Q5_K_M.gguf"
Write-Host ""
Write-Host "2. Put Qwen-VL model files here:"
Write-Host "   models\qwen_vl\Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
Write-Host "   models\qwen_vl\mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
Write-Host ""
Write-Host "3. Put voice sample here:"
Write-Host "   data\reference.wav"
Write-Host ""
Write-Host "4. Build RAG index:"
Write-Host "   .\.venv\Scripts\python.exe .\tools\build_rag_index.py"
Write-Host ""
Write-Host "5. Start LocalAI_V3:"
Write-Host "   .\run_localai.bat"
Write-Host ""
