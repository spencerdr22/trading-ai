@echo off
REM ============================================================
REM  start_trading.bat — Trading-AI launcher with Ollama arbiter
REM
REM  Startup sequence:
REM   1. Kill existing Ollama processes
REM   2. Start Ollama with GPU-optimised settings
REM   3. Start the Ollama arbiter (priority router port 11435)
REM   4. Launch trading scheduler
REM
REM  The arbiter ensures trading inference always takes priority
REM  over any other project (Faceless-AI etc.) using the same Ollama.
REM ============================================================

cd /d "C:\Users\spenc\Documents\trading-ai"

REM ── Step 1: Kill existing Ollama processes ────────────────────────────────
echo Stopping any existing Ollama processes...
taskkill /F /IM ollama.exe >nul 2>&1
taskkill /F /IM ollama_llama_server.exe >nul 2>&1
timeout /t 2 /nobreak >nul

REM Kill any existing arbiter on port 11435
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":11435 "') do (
    taskkill /F /PID %%a >nul 2>&1
)
echo Done.
echo.

REM ── Step 2: Set Ollama GPU environment ───────────────────────────────────
echo Setting Ollama environment...
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_GPU_OVERHEAD=0
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_KEEP_ALIVE=10m
set OLLAMA_MAX_LOADED_MODELS=1

echo   FLASH_ATTENTION=%OLLAMA_FLASH_ATTENTION%
echo   GPU_OVERHEAD=%OLLAMA_GPU_OVERHEAD%
echo   NUM_PARALLEL=%OLLAMA_NUM_PARALLEL%
echo   KEEP_ALIVE=%OLLAMA_KEEP_ALIVE%
echo.

echo Starting Ollama server...
start /B ollama serve
echo Waiting for Ollama to initialise...
timeout /t 6 /nobreak >nul

curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Ollama did not start on port 11434.
    pause
    exit /b 1
)
echo Ollama running on port 11434.
echo.

REM ── Step 3: Start the priority arbiter ───────────────────────────────────
echo Starting Ollama priority arbiter on port 11435...
start "Ollama Arbiter" /MIN "C:\Users\spenc\Documents\trading-ai\.venv\Scripts\python.exe" "C:\Users\spenc\Documents\ollama_arbiter.py"
timeout /t 3 /nobreak >nul

curl -s http://localhost:11435/api/tags >nul 2>&1
if %errorlevel% == 0 (
    echo Arbiter running on port 11435.
    echo Trading requests will be prioritised over Faceless-AI.
) else (
    echo Arbiter starting in background ^(may take a few more seconds^).
)
echo.

REM ── Step 4: Launch trading scheduler ─────────────────────────────────────
echo  Starting MES Paper Trading Scheduler...
echo  Waiting until 9:30 AM ET then trading begins.
echo  Press CTRL+C to stop at any time.
echo.

call "C:\Users\spenc\Documents\trading-ai\.venv\Scripts\activate.bat"
python start_trading.py

echo.
echo  Scheduler exited.
pause
