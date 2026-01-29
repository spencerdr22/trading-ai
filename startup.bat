@echo off
setlocal EnableDelayedExpansion

REM Set colors (optional - for better visibility)
color 0A

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║          Trading-AI System - Full Startup                 ║
echo ║                                                            ║
echo ║  Hardware: Ryzen 7 7800X3D + RTX 4070 Super + 32GB DDR5  ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM ============================================================
REM STEP 1: Check Ollama Service
REM ============================================================
echo [1/5] Checking Ollama service...
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Ollama is not running - starting now...
    start "Ollama Server" /MIN ollama serve
    
    REM Wait for Ollama to start (max 10 seconds)
    set /a timeout=0
    :wait_ollama
    timeout /t 1 /nobreak >nul
    ollama list >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Ollama started successfully
        goto ollama_ready
    )
    set /a timeout+=1
    if !timeout! lss 10 goto wait_ollama
    
    echo ❌ ERROR: Ollama failed to start after 10 seconds
    echo    Please start it manually: ollama serve
    pause
    exit /b 1
) else (
    echo ✅ Ollama is already running
)
:ollama_ready
echo.

REM ============================================================
REM STEP 2: Verify Model is Available
REM ============================================================
echo [2/5] Verifying Qwen3 model...
ollama list | findstr "qwen3-30b-a3b" >nul
if %errorlevel% neq 0 (
    echo ⚠️  Model not found!
    echo.
    echo    The model needs to be pulled (this is a one-time download, ~16GB)
    echo    This will take 10-30 minutes depending on your internet speed.
    echo.
    set /p pull_model="    Pull model now? (y/n): "
    if /i "!pull_model!"=="y" (
        echo.
        echo    Pulling qwen3-30b-a3b:q4_K_M...
        ollama pull qwen3-30b-a3b:q4_K_M
        if %errorlevel% neq 0 (
            echo ❌ Model pull failed
            pause
            exit /b 1
        )
        echo ✅ Model pulled successfully
    ) else (
        echo    Skipping model pull - LLM features will not work
    )
) else (
    echo ✅ Model qwen3-30b-a3b:q4_K_M is available
)
echo.

REM ============================================================
REM STEP 3: Check GPU Status
REM ============================================================
echo [3/5] Checking GPU status...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  WARNING: nvidia-smi not found - GPU may not be available
    echo    LLM inference will fall back to CPU (much slower)
) else (
    echo ✅ GPU detected - checking VRAM...
    for /f "tokens=9" %%a in ('nvidia-smi --query-gpu^=memory.free --format^=csv^,noheader^,nounits') do (
        set vram_free=%%a
    )
    if !vram_free! lss 12000 (
        echo ⚠️  WARNING: Low VRAM available (!vram_free!MB free, need 12GB+)
        echo    Close other GPU applications for best performance
    ) else (
        echo ✅ GPU ready - !vram_free!MB VRAM available
    )
)
echo.

REM ============================================================
REM STEP 4: Activate Virtual Environment
REM ============================================================
echo [4/5] Activating Python virtual environment...
if not exist ".venv\Scripts\activate.bat" (
    echo ❌ ERROR: Virtual environment not found at .venv\
    echo    Run: python -m venv .venv
    pause
    exit /b 1
)

call .venv\Scripts\activate
if %errorlevel% neq 0 (
    echo ❌ ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✅ Virtual environment activated
echo.

REM ============================================================
REM STEP 5: Verify LLM Integration
REM ============================================================
echo [5/5] Verifying LLM integration...
python verify_llm_install.py >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  LLM verification had warnings - running detailed check...
    echo.
    python verify_llm_install.py
    echo.
    set /p continue_anyway="Continue anyway? (y/n): "
    if /i not "!continue_anyway!"=="y" (
        pause
        exit /b 1
    )
) else (
    echo ✅ LLM integration verified
)
echo.

REM ============================================================
REM System Ready - Show Menu
REM ============================================================
echo ╔════════════════════════════════════════════════════════════╗
echo ║                   🚀 SYSTEM READY 🚀                       ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

:menu
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo                       MAIN MENU
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo   1. 📊 Run Forward/Paper Trading (ES)
echo   2. 📈 Run Backtest (ES)
echo   3. 🧪 Test GPU Scheduler
echo   4. 🤖 Chat with Qwen3 (Interactive)
echo   5. 📋 Check System Status
echo   6. 🔄 Run Multi-Backtest Analysis
echo   7. 📺 Open Dashboard (Streamlit)
echo   8. 🚪 Exit
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" goto forward_trading
if "%choice%"=="2" goto backtest
if "%choice%"=="3" goto test_gpu
if "%choice%"=="4" goto chat_llm
if "%choice%"=="5" goto system_status
if "%choice%"=="6" goto multi_backtest
if "%choice%"=="7" goto dashboard
if "%choice%"=="8" goto exit_script

echo Invalid choice. Please try again.
echo.
goto menu

:forward_trading
echo.
echo Starting forward/paper trading on ES...
echo Press Ctrl+C to stop
echo.
python -m app.main --mode forward --symbol ES
echo.
pause
goto menu

:backtest
echo.
echo Running backtest on ES...
python -m app.main --mode backtest --symbol ES
echo.
pause
goto menu

:test_gpu
echo.
echo Testing GPU scheduler...
python -m app.llm.test_scheduling both
echo.
pause
goto menu

:chat_llm
echo.
echo Starting interactive chat with Qwen3...
echo Type 'exit' to return to menu
echo.
ollama run qwen3-30b-a3b:q4_K_M
echo.
pause
goto menu

:system_status
echo.
echo ═══════════════════════════════════════════════════════════
echo                     SYSTEM STATUS
echo ═══════════════════════════════════════════════════════════
echo.
echo [GPU Status]
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv,noheader
echo.
echo [Ollama Models]
ollama list
echo.
echo [Python Environment]
python --version
echo.
echo [LLM Integration]
python -c "from app.llm.system_config import system_config; import json; print(json.dumps(system_config.get_config_summary(), indent=2))"
echo.
echo [GPU Scheduler Metrics]
python -c "from app.llm.gpu_scheduler import gpu_scheduler; import json; print(json.dumps(gpu_scheduler.get_metrics(), indent=2))"
echo.
pause
goto menu

:multi_backtest
echo.
echo Running multi-parameter backtest analysis...
echo This will take several minutes...
echo.
python -m app.analysis.multi_backtest
echo.
pause
goto menu

:dashboard
echo.
echo Starting Streamlit dashboard...
echo Dashboard will open in your browser
echo Press Ctrl+C in this window to stop the dashboard
echo.
streamlit run app/monitor/dashboard.py -- --symbol ES
echo.
pause
goto menu

:exit_script
echo.
echo ═══════════════════════════════════════════════════════════
echo Thank you for using Trading-AI!
echo.
echo Remember:
echo   - Ollama server is still running (will stop on system shutdown)
echo   - To stop Ollama manually: taskkill /f /im ollama.exe
echo.
echo ═══════════════════════════════════════════════════════════
echo.
pause
exit /b 0
