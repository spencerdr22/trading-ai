@echo off
echo Starting Ollama server...
start /B ollama serve

timeout /t 3 /nobreak >nul

echo Verifying Ollama is running...
ollama list

echo.
echo Ollama is ready!
echo Model: qwen3-30b-a3b:q4_K_M
echo.
pause
