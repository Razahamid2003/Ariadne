@echo off
REM ============================================================================
REM  Ariadne RAG - Start the server (local only, 127.0.0.1:8080).
REM  Opens the browser automatically. Press Ctrl+C in this window to stop.
REM ============================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
set "RAGS_CONFIG_PATH=config\client.yaml"

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  echo   [ERROR] Virtual environment not found. Run setup.bat first.
  pause
  exit /b 1
)

echo.
echo   ARIADNE - Follow the thread
echo   ===========================
echo   Local address:  http://127.0.0.1:8080
echo   Press Ctrl+C to stop.
echo.

start "" /b cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8080"

"!PY!" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8080
if errorlevel 1 (
  echo.
  echo   [ERROR] Ariadne failed to start. Common causes:
  echo     - Ollama not running:    ollama serve
  echo     - Port 8080 in use:      close the other program using it
  echo     - Dependencies missing:  run setup.bat
  echo.
  pause
)
endlocal
