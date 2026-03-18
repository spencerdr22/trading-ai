@echo off
echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║          Quick Start: Ollama + LLM Testing                ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Check if Ollama is running
echo [1/2] Checking Ollama status...
.\ollama.bat list >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Ollama not running - starting now...
    start "Ollama Server" /MIN .\ollama.bat serve
    
    REM Wait for Ollama to start
    timeout /t 3 /nobreak >nul
    
    REM Verify it started
    .\ollama.bat list >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ Failed to start Ollama
        echo    Please start manually: ollama serve
        pause
        exit /b 1
    )
    echo ✅ Ollama started
) else (
    echo ✅ Ollama already running
)
echo.

REM Activate environment
echo [2/2] Activating Python environment...
call .venv\Scripts\activate
if %errorlevel% neq 0 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✅ Environment ready
echo.

echo ════════════════════════════════════════════════════════════
echo System Ready! You can now run:
echo.
echo   • Test news APIs:      python -m app.llm.test_news_feeds api
echo   • Test full pipeline:  python -m app.llm.test_news_feeds full
echo   • Run trading:         python -m app.main --mode forward --symbol ES
echo ════════════════════════════════════════════════════════════
echo.

cmd /k
