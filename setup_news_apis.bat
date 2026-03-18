@echo off
echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║           News API Setup - Interactive Guide              ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

echo This script will help you set up your news API keys.
echo.

echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo Step 1: Get Your API Keys
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo Open these links to get your free API keys:
echo.
echo [1] Alpaca (best for futures):
echo     https://alpaca.markets/
echo     Sign up → Paper Trading → Generate API Keys
echo.
echo [2] Finnhub (good for macro news):
echo     https://finnhub.io/
echo     Get free API key → Copy key
echo.
echo [3] NewsAPI (backup source):
echo     https://newsapi.org/
echo     Get API Key → Copy key
echo.
pause
echo.

echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo Step 2: Enter Your Keys
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo NOTE: You need at least ONE API to continue.
echo       Press Enter to skip any API you don't have yet.
echo.

set /p alpaca_key="Enter Alpaca API Key ID (or press Enter to skip): "
set /p alpaca_secret="Enter Alpaca Secret Key (or press Enter to skip): "
set /p finnhub_key="Enter Finnhub API Key (or press Enter to skip): "
set /p newsapi_key="Enter NewsAPI Key (or press Enter to skip): "

echo.
echo Updating .env file...

REM Backup .env
copy .env .env.backup >nul 2>&1

REM Update keys in .env
powershell -Command "(Get-Content .env) -replace 'ALPACA_API_KEY=.*', 'ALPACA_API_KEY=%alpaca_key%' | Set-Content .env"
powershell -Command "(Get-Content .env) -replace 'ALPACA_SECRET_KEY=.*', 'ALPACA_SECRET_KEY=%alpaca_secret%' | Set-Content .env"
powershell -Command "(Get-Content .env) -replace 'FINNHUB_API_KEY=.*', 'FINNHUB_API_KEY=%finnhub_key%' | Set-Content .env"
powershell -Command "(Get-Content .env) -replace 'NEWSAPI_KEY=.*', 'NEWSAPI_KEY=%newsapi_key%' | Set-Content .env"

echo ✅ Keys saved to .env file
echo.

echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo Step 3: Test Connection
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

call .venv\Scripts\activate
python -m app.llm.test_news_feeds api

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo Setup Complete!
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo Next steps:
echo   1. Test full pipeline: python -m app.llm.test_news_feeds full
echo   2. See docs/NEWS_API_SETUP.md for troubleshooting
echo.
pause
