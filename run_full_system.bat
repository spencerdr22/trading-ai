@echo off
REM ============================================================
REM  run_full_system.bat  —  Start EVERYTHING for trading
REM
REM  Startup order:
REM   1. AI Orchestrator  (port 8000) — LLM priority routing (optional)
REM   2. Ollama server    (port 11434) — sentiment model backend
REM      NOTE: arbiter removed — AI Orchestrator handles routing
REM   3. Streamlit dashboard (port 8501) — monitoring UI
REM   4. Trading scheduler — waits for 9:30 AM ET, trades until 3:55 PM ET
REM ============================================================

cd /d "C:\Users\spenc\Documents\trading-ai"

set VENV_PY=.venv\Scripts\python.exe
set VENV_ST=.venv\Scripts\streamlit.exe

echo ============================================================
echo   Trading-AI Full System Launcher
echo   %date% %time%
echo ============================================================
echo.

REM ── Step 1: AI Orchestrator ──────────────────────────────────────────────────
echo [1/4] Checking AI Orchestrator on port 8000...
curl -s --max-time 3 http://localhost:8000/health >nul 2>&1
if %errorlevel% == 0 (
    echo        Already running. OK.
    goto step2
)
echo        Not running. Starting via Docker...
start "AI-Orchestrator" /MIN cmd /c "cd /d C:\Users\spenc\Documents\ai-orchestrator && docker compose -f docker\docker-compose.yml up --remove-orphans"
timeout /t 10 /nobreak >nul
curl -s --max-time 3 http://localhost:8000/health >nul 2>&1
if %errorlevel% == 0 (
    echo        Orchestrator started on port 8000.
) else (
    echo        Orchestrator not ready. Trading will use direct Ollama (no impact on trades).
)

:step2
echo.

REM ── Step 2: Ollama with GPU settings ─────────────────────────────────────────
echo [2/4] Starting Ollama with GPU optimisation...
taskkill /F /IM ollama.exe >nul 2>&1
taskkill /F /IM ollama_llama_server.exe >nul 2>&1
timeout /t 2 /nobreak >nul

set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_GPU_OVERHEAD=0
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_KEEP_ALIVE=10m
set OLLAMA_MAX_LOADED_MODELS=1

start "Ollama-Server" /MIN ollama serve
timeout /t 6 /nobreak >nul

curl -s --max-time 5 http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Ollama did not start on port 11434.
    echo        Install from https://ollama.com then re-run this script.
    pause
    exit /b 1
)
echo        Ollama running. FLASH_ATTENTION=1  NUM_PARALLEL=1  KEEP_ALIVE=10m
echo.

REM ── Step 3: Dashboard ─────────────────────────────────────────────────────────
echo [3/4] Starting dashboard on http://localhost:8501...
start "Trading-Dashboard" /MIN %VENV_ST% run app\monitor\dashboard.py --server.port 8501 --server.headless true --browser.gatherUsageStats false --server.fileWatcherType none
timeout /t 4 /nobreak >nul
echo        Dashboard starting. Open http://localhost:8501 in your browser.
echo.

REM ── Step 4: Trading scheduler ─────────────────────────────────────────────────
echo ============================================================
echo [4/4] Trading Scheduler Starting
echo.
echo   Symbol   : MES via SPY proxy on Alpaca paper
echo   Start    : 9:30 AM ET
echo   Stop     : 3:55 PM ET  (positions closed automatically)
echo   Dashboard: http://localhost:8501
echo   LLM:       http://localhost:8000  (Orchestrator)
echo.
echo   Press CTRL+C to stop at any time.
echo ============================================================
echo.

%VENV_PY% start_trading.py

echo.
echo ============================================================
echo   Trading session ended. All positions should be flat.
echo   Check http://localhost:8501 for today's results.
echo ============================================================
pause
