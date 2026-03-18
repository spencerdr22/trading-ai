@echo off
echo.
echo ════════════════════════════════════════════════════════════
echo Starting Ollama Server (Visible Window)
echo ════════════════════════════════════════════════════════════
echo.

REM Try to find Ollama
set OLLAMA_PATH=

if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    set OLLAMA_PATH=%LOCALAPPDATA%\Programs\Ollama\ollama.exe
) else if exist "%ProgramFiles%\Ollama\ollama.exe" (
    set OLLAMA_PATH=%ProgramFiles%\Ollama\ollama.exe
) else (
    echo ❌ ERROR: Cannot find ollama.exe
    echo.
    echo Searched locations:
    echo   - %LOCALAPPDATA%\Programs\Ollama\ollama.exe
    echo   - %ProgramFiles%\Ollama\ollama.exe
    echo.
    echo Please install Ollama from: https://ollama.com/download
    pause
    exit /b 1
)

echo ✅ Found Ollama at: %OLLAMA_PATH%
echo.
echo Starting server...
echo.
echo ⚠️  IMPORTANT: Keep this window open!
echo    Close it when you're done testing.
echo.
echo ════════════════════════════════════════════════════════════
echo.

"%OLLAMA_PATH%" serve
