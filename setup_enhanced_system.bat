@echo off
echo ============================================================
echo Enhanced Trading System - Setup and Test
echo ============================================================
echo.

REM Activate virtual environment
if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo WARNING: No .venv found. Install dependencies first!
    echo Run: python -m venv .venv
    echo Then: .venv\Scripts\activate
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo Checking if yfinance is installed...
python -c "import yfinance" 2>nul
if %errorlevel% neq 0 (
    echo yfinance not found. Installing...
    pip install yfinance
) else (
    echo yfinance already installed.
)

echo.
echo ============================================================
echo Running quick system test...
echo ============================================================
echo.

python test_enhanced_system.py

echo.
echo ============================================================
echo Test complete!
echo ============================================================
echo.
echo What to do next:
echo.
echo 1. If test passed: Train with real data
echo    Command: python -m app.ml.enhanced_training_pipeline --symbol ES --days 30
echo.
echo 2. If test failed: Check .env file has API keys
echo    Need: ALPACA_API_KEY and ALPACA_SECRET_KEY
echo    Get free keys at: https://alpaca.markets
echo.
echo 3. Read REAL_DATA_SETUP.md for full guide
echo.
echo ============================================================
pause
