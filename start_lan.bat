@echo off
REM ============================================================================
REM  Ariadne RAG - Start on the local network (LAN mode, 0.0.0.0:8080).
REM  Other devices on the same network can reach Ariadne at this machine's IP.
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
echo   ARIADNE - LAN mode
echo   ==================
echo   This machine:   http://127.0.0.1:8080
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
  set "IP=%%a"
  set "IP=!IP: =!"
  echo   Other devices:  http://!IP!:8080
)
echo.
echo   If other devices cannot connect, allow Python through Windows Firewall
echo   for Private networks, or add an inbound TCP rule for port 8080.
echo   Press Ctrl+C to stop.
echo.

"!PY!" -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8080
if errorlevel 1 (
  echo.
  echo   [ERROR] Ariadne failed to start. Run setup.bat, then ensure Ollama is running.
  echo.
  pause
)
endlocal
