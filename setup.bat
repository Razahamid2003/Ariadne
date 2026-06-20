@echo off
REM ============================================================================
REM  Ariadne RAG - One-time setup: create venv and install dependencies.
REM  Run this ONCE before going air-gapped (needs internet for pip).
REM ============================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo   ARIADNE - Setup
echo   ===============
echo.

python --version >nul 2>nul
if errorlevel 1 (
  echo   [ERROR] Python not found on PATH.
  echo   Install Python 3.11+ from python.org and tick "Add Python to PATH".
  echo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo   Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 (
    echo   [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
  )
)

echo   Installing dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo   [ERROR] Dependency installation failed. Check your internet connection.
  pause
  exit /b 1
)

echo.
echo   Setup complete.
echo   Next steps:
echo     1. Make sure Ollama is running:   ollama serve
echo     2. Pull the model:                ollama pull llama3.1:8b
echo     3. Index documents:               ingest.bat
echo     4. Launch Ariadne:                start.bat
echo.
pause
endlocal
