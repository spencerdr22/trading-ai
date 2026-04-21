@echo off
cd /d "C:\Users\spenc\Documents\trading-ai"

echo Starting Trading-AI Dashboard...
echo.
echo   URL: http://localhost:8501
echo   Press CTRL+C to stop.
echo.

.venv\Scripts\streamlit.exe run app\monitor\dashboard.py --server.port 8501 --server.headless false --browser.gatherUsageStats false --server.fileWatcherType none

pause
