@echo off
REM ============================================================================
REM  Ariadne RAG - Ingest documents and build keyword + vector indexes.
REM  Four explicit phases so any failure is visible (not hidden in one command).
REM  First ingestion classifies each document with the LLM -- several minutes
REM  is normal on a large corpus. Ensure Ollama is running before starting.
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
echo   ARIADNE - Building the index
echo   ============================
echo   Ensure Ollama is running (ollama serve) and model is pulled.
echo.

echo   [1/4] Cleaning previous data and indexes...
"!PY!" scripts\manage_rag.py clean --processed --metadata --vector
if errorlevel 1 ( echo   [ERROR] Clean failed. & pause & exit /b 1 )

echo.
echo   [2/4] Ingesting + classifying documents (can take several minutes)...
"!PY!" scripts\manage_rag.py ingest --force
if errorlevel 1 (
  echo   [ERROR] Ingestion failed.
  echo   Most likely cause: Ollama is not running. Start it with: ollama serve
  pause
  exit /b 1
)

echo.
echo   [3/4] Building keyword index...
"!PY!" scripts\manage_rag.py build-keyword-index
if errorlevel 1 ( echo   [ERROR] Keyword index build failed. & pause & exit /b 1 )

echo.
echo   [4/4] Building vector index (embeds all chunks, may take a few minutes)...
"!PY!" scripts\manage_rag.py build-index
if errorlevel 1 (
  echo   [ERROR] Vector index build failed.
  echo   Check that the sentence-transformers model is accessible.
  pause
  exit /b 1
)

echo.
echo   ============================
echo   Done. All indexes built.
echo.
"!PY!" scripts\manage_rag.py status
echo.
echo   Run start.bat to launch Ariadne.
echo.
pause
endlocal
