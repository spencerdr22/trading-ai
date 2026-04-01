@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  start_ollama.bat — Launch Ollama optimised for RTX 4070 Super
REM
REM  Qwen3-30B-A3B-Q4_K_M is 18GB. RTX 4070 Super has ~9.3GB free VRAM.
REM  Ollama will load ~24/49 layers on GPU and ~25/49 on CPU.
REM  This is the physical limit — no env variable can exceed it.
REM  OLLAMA_FLASH_ATTENTION=1 and OLLAMA_GPU_OVERHEAD=0 give the best
REM  possible split within the VRAM constraint.
REM ─────────────────────────────────────────────────────────────────────────

echo Stopping any existing Ollama processes...
taskkill /F /IM ollama.exe >nul 2>&1
taskkill /F /IM ollama_llama_server.exe >nul 2>&1
timeout /t 2 /nobreak >nul
echo Done.
echo.

echo Setting Ollama environment...
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_GPU_OVERHEAD=0
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_KEEP_ALIVE=10m
set OLLAMA_MAX_LOADED_MODELS=1

echo   OLLAMA_FLASH_ATTENTION   = %OLLAMA_FLASH_ATTENTION%
echo   OLLAMA_GPU_OVERHEAD      = %OLLAMA_GPU_OVERHEAD%
echo   OLLAMA_NUM_PARALLEL      = %OLLAMA_NUM_PARALLEL%
echo   OLLAMA_KEEP_ALIVE        = %OLLAMA_KEEP_ALIVE%
echo   OLLAMA_MAX_LOADED_MODELS = %OLLAMA_MAX_LOADED_MODELS%
echo.

echo Starting Ollama server...
start /B ollama serve
timeout /t 5 /nobreak >nul

ollama list
echo.
echo Layer split will be ~24/49 GPU — correct for 18GB model on 12GB VRAM.
echo.
pause
