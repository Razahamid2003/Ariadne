@echo off
REM ============================================================================
REM  Ariadne RAG - Stop any server running on port 8080.
REM ============================================================================
setlocal enabledelayedexpansion
echo.
echo   Stopping Ariadne (port 8080)...
set "FOUND="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8080" ^| findstr "LISTENING"') do (
  set "FOUND=1"
  echo   Stopping PID %%p
  taskkill /PID %%p /F >nul 2>nul
)
if not defined FOUND (
  echo   No Ariadne server found running on port 8080.
) else (
  echo   Stopped.
)
echo.
pause
endlocal
