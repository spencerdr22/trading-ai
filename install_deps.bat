@echo off
REM ============================================================
REM  install_deps.bat  —  Install missing packages into .venv
REM
REM  The .venv already has: pandas 3.x, numpy 2.x, streamlit 1.56,
REM  sqlalchemy 2.x, requests, dotenv.
REM
REM  This script adds the MISSING packages only, without touching
REM  what is already installed.
REM ============================================================

cd /d "C:\Users\spenc\Documents\trading-ai"

REM Use the venv Python directly — no activate needed
set VENV_PY=.venv\Scripts\python.exe
set VENV_PIP=.venv\Scripts\pip.exe

echo ============================================================
echo   Trading-AI Dependency Installer
echo   Using: %VENV_PY%
echo ============================================================
echo.

REM Confirm we are using the venv python
%VENV_PY% --version
echo.

REM ── Upgrade pip first ────────────────────────────────────────────────────────
echo [0] Upgrading pip...
%VENV_PIP% install --upgrade pip --quiet
echo     Done.
echo.

REM ── Group 1: Core ML (scikit-learn) ─────────────────────────────────────────
echo [1] Installing scikit-learn...
%VENV_PIP% install "scikit-learn>=1.4.0" --quiet
if %errorlevel% neq 0 (
    echo     WARNING: scikit-learn install had issues. Trying without version pin...
    %VENV_PIP% install scikit-learn --quiet
)
echo     Done.

REM ── Group 2: Scheduling ──────────────────────────────────────────────────────
echo [2] Installing apscheduler...
%VENV_PIP% install "apscheduler>=3.10.0" --quiet
echo     Done.

REM ── Group 3: Async / HTTP ────────────────────────────────────────────────────
echo [3] Installing aiohttp + psutil...
%VENV_PIP% install "aiohttp>=3.9.0" "psutil>=5.9.0" --quiet
echo     Done.

REM ── Group 4: LLM ─────────────────────────────────────────────────────────────
echo [4] Installing ollama client...
%VENV_PIP% install "ollama>=0.1.0" --quiet
echo     Done.

REM ── Group 5: Optimisation ────────────────────────────────────────────────────
echo [5] Installing optuna...
%VENV_PIP% install "optuna>=3.3.0" --quiet
echo     Done.

REM ── Group 6: Misc utilities ──────────────────────────────────────────────────
echo [6] Installing pyyaml + pytz + matplotlib...
%VENV_PIP% install "pyyaml>=6.0.0" "pytz>=2024.1" "matplotlib>=3.8.0" --quiet
echo     Done.

REM ── Group 7: Dashboard ───────────────────────────────────────────────────────
echo [7] Ensuring streamlit-autorefresh is present...
%VENV_PIP% install "streamlit-autorefresh>=1.0.0" --quiet
echo     Done.

REM ── Group 8: PyTorch (CUDA 12.1 for RTX 4070 Super) ────────────────────────
echo [8] Checking PyTorch...
%VENV_PY% -c "import torch; print('    torch ' + torch.__version__ + ' CUDA=' + str(torch.cuda.is_available()))" 2>nul
if %errorlevel% == 0 (
    echo     Already installed.
    goto torch_done
)
echo     torch not found. Installing CUDA 12.1 build for RTX 4070 Super...
echo     This downloads ~2GB - takes 3-10 minutes on a fast connection.
%VENV_PIP% install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
REM Check if torch now imports correctly after install
%VENV_PY% -c "import torch" 2>nul
if %errorlevel% == 0 (
    echo     CUDA torch installed successfully.
    goto torch_done
)
echo     CUDA build did not import cleanly. Trying CPU build as fallback...
%VENV_PIP% install torch --index-url https://download.pytorch.org/whl/cpu --quiet
:torch_done
echo.

REM ── Verify all critical imports ───────────────────────────────────────────────
echo ============================================================
echo   Verifying imports...
echo ============================================================
%VENV_PY% -c "import sklearn; print('  scikit-learn', sklearn.__version__)"
%VENV_PY% -c "import torch; print('  torch', torch.__version__)"
%VENV_PY% -c "import ollama; print('  ollama OK')"
%VENV_PY% -c "import aiohttp; print('  aiohttp', aiohttp.__version__)"
%VENV_PY% -c "import apscheduler; print('  apscheduler OK')"
%VENV_PY% -c "import pytz; print('  pytz OK')"
%VENV_PY% -c "import optuna; print('  optuna', optuna.__version__)"
%VENV_PY% -c "import streamlit; print('  streamlit', streamlit.__version__)"
%VENV_PY% -c "import streamlit_autorefresh; print('  streamlit_autorefresh OK')"
echo.

echo ============================================================
echo   Done. Run pre_flight.py to confirm all checks pass:
echo     .venv\Scripts\python.exe pre_flight.py
echo ============================================================
pause
