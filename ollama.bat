@echo off
REM Ollama wrapper - finds and runs Ollama from common install locations

REM Try default installation path
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" %*
    exit /b %errorlevel%
)

REM Try Program Files
if exist "%ProgramFiles%\Ollama\ollama.exe" (
    "%ProgramFiles%\Ollama\ollama.exe" %*
    exit /b %errorlevel%
)

REM Try user profile
if exist "%USERPROFILE%\AppData\Local\Ollama\ollama.exe" (
    "%USERPROFILE%\AppData\Local\Ollama\ollama.exe" %*
    exit /b %errorlevel%
)

echo ERROR: Ollama not found in common installation locations
echo.
echo Please install Ollama from: https://ollama.com/download
echo.
echo Or add Ollama to your PATH manually
exit /b 1
