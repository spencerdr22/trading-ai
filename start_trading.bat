@echo off
REM ============================================================
REM  MES Paper Trading — Auto Launch
REM  Run this once to register the Windows Task Scheduler job,
REM  OR just double-click start_trading.bat to launch manually.
REM ============================================================

cd /d "C:\Users\spenc\Documents\trading-ai"

echo.
echo  Starting MES Paper Trading Scheduler...
echo  This window will wait until 9:30 AM ET then begin trading.
echo  Close this window (or press CTRL+C) to stop at any time.
echo.

call .venv\Scripts\activate.bat
python start_trading.py

echo.
echo  Scheduler exited.
pause
