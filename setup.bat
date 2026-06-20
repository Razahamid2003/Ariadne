@echo off
REM ============================================================================
REM  Ariadne RAG - One-time setup: create venv and install dependencies.
REM  Automatically detects your GPU and installs the right PyTorch build:
REM    - NVIDIA Blackwell GPU (RTX 50-series, sm_12x) -> CUDA 12.8 build
REM    - Other NVIDIA GPU (RTX 40-series and older)   -> CUDA 12.1 build
REM    - No NVIDIA GPU detected                        -> CPU-only build
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

echo   Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet

REM == Detect NVIDIA GPU =====================================================
echo   Detecting GPU...
set "TORCH_INDEX="
set "GPU_LABEL=CPU-only (no NVIDIA GPU detected)"

nvidia-smi >nul 2>nul
if not errorlevel 1 (
  for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2^>nul') do set "COMPUTE_CAP=%%g"
  if defined COMPUTE_CAP (
    set "CC_MAJOR=!COMPUTE_CAP:~0,2!"
    if "!CC_MAJOR!"=="12" (
      set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
      set "GPU_LABEL=NVIDIA Blackwell GPU (sm_!COMPUTE_CAP!) - installing CUDA 12.8 build for RTX 50-series"
    ) else (
      set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
      set "GPU_LABEL=NVIDIA GPU (sm_!COMPUTE_CAP!) - installing CUDA 12.1 build"
    )
  ) else (
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
    set "GPU_LABEL=NVIDIA GPU detected - installing CUDA 12.1 build"
  )
)
echo   !GPU_LABEL!
echo.

REM == Core dependencies (no torch) =========================================
echo   Installing core dependencies...
.venv\Scripts\python.exe -m pip install --quiet fastapi "uvicorn[standard]" pydantic pydantic-settings PyYAML httpx python-dotenv openpyxl pypdf python-docx python-pptx PyMuPDF pytesseract Pillow
if errorlevel 1 ( echo   [ERROR] Core dependency installation failed. & pause & exit /b 1 )

REM == sentence-transformers without torch (avoids pulling CPU torch) ========
echo   Installing sentence-transformers...
.venv\Scripts\python.exe -m pip install --quiet --no-deps sentence-transformers
.venv\Scripts\python.exe -m pip install --quiet transformers huggingface-hub tokenizers safetensors tqdm scipy scikit-learn
if errorlevel 1 ( echo   [ERROR] sentence-transformers installation failed. & pause & exit /b 1 )

REM == PyTorch - correct build for this GPU ==================================
if defined TORCH_INDEX (
  echo   Installing PyTorch with CUDA support (this may take a few minutes - ~2.5 GB)...
  .venv\Scripts\python.exe -m pip install --quiet torch torchvision torchaudio --index-url !TORCH_INDEX!
) else (
  echo   Installing PyTorch CPU build...
  .venv\Scripts\python.exe -m pip install --quiet torch
)
if errorlevel 1 ( echo   [ERROR] PyTorch installation failed. & pause & exit /b 1 )

REM == Verify ==============================================================
echo.
echo   Verifying installation...
.venv\Scripts\python.exe -c "import torch; cuda=torch.cuda.is_available(); print('  PyTorch:', torch.__version__); print('  CUDA available:', cuda); print('  GPU:', torch.cuda.get_device_name(0) if cuda else 'none (CPU mode)')"

echo.
echo   Setup complete.
echo   Next steps:
echo     1. Make sure Ollama is running:   ollama serve
echo     2. Pull the model:                ollama pull llama3.1:8b
echo     3. Index your documents:          ingest.bat
echo     4. Launch Ariadne:                start.bat
echo.
pause
endlocal
