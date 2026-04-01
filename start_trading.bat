@echo off
REM ============================================================
REM  MES Paper Trading — Auto Launch
REM
REM  LAYER SPLIT REALITY FOR Qwen3-30B-A3B-Q4_K_M + RTX 4070 Super:
REM  - Model size: 18GB total across 49 layers (~367MB/layer)
REM  - VRAM available: ~9.3GB free after driver overhead
REM  - Maximum GPU layers: 9300 / 367 = ~25 layers
REM  - Ollama auto-calculates this correctly: 24/49 on GPU, 25/49 on CPU
REM  - There is NO environment variable in Ollama 0.18.3 that can force
REM    more layers onto GPU than VRAM physically allows.
REM
REM  WHAT WE CAN CONTROL:
REM  - OLLAMA_FLASH_ATTENTION=1  (saves ~15% VRAM during forward pass)
REM  - OLLAMA_GPU_OVERHEAD=0     (zero out the safety buffer Ollama reserves)
REM  - OLLAMA_NUM_PARALLEL=1     (no parallel slots competing for VRAM)
REM  - OLLAMA_MAX_LOADED_MODELS=1 (gemma3 can't steal VRAM)
REM  - OLLAMA_KEEP_ALIVE=10m     (no reload between headline batches)
REM
REM  EXPECTED PROFILE: ~30-40% GPU, ~50-60% CPU — this is correct for
REM  a 18GB model on a 12GB card. The CPU handles the overflow layers.
REM  Per-headline inference: 13-30 seconds is normal at this split.
REM ============================================================

cd /d "C:\Users\spenc\Documents\trading-ai"

REM ── Step 1: Kill any existing Ollama processes ────────────────────────────
echo Stopping any existing Ollama processes...
taskkill /F /IM ollama.exe >nul 2>&1
taskkill /F /IM ollama_llama_server.exe >nul 2>&1
timeout /t 2 /nobreak >nul
echo Done.
echo.

REM ── Step 2: Set GPU environment variables ────────────────────────────────
echo Setting Ollama environment...

REM FlashAttention-2: saves ~15% VRAM during forward pass (confirmed working)
set OLLAMA_FLASH_ATTENTION=1

REM Zero out the safety overhead buffer to maximise layers on GPU
set OLLAMA_GPU_OVERHEAD=0

REM Single inference slot: no parallel requests competing for VRAM
set OLLAMA_NUM_PARALLEL=1

REM Keep model hot between headline batches (avoids 5-30s reload penalty)
set OLLAMA_KEEP_ALIVE=10m

REM Only one model loaded: prevents gemma3 from sharing VRAM with qwen3
set OLLAMA_MAX_LOADED_MODELS=1

echo   OLLAMA_FLASH_ATTENTION = %OLLAMA_FLASH_ATTENTION%
echo   OLLAMA_GPU_OVERHEAD    = %OLLAMA_GPU_OVERHEAD%
echo   OLLAMA_NUM_PARALLEL    = %OLLAMA_NUM_PARALLEL%
echo   OLLAMA_KEEP_ALIVE      = %OLLAMA_KEEP_ALIVE%
echo   OLLAMA_MAX_LOADED_MODELS = %OLLAMA_MAX_LOADED_MODELS%
echo.

REM ── Step 3: Start Ollama ──────────────────────────────────────────────────
echo Starting Ollama server...
start /B ollama serve

echo Waiting for Ollama to initialise...
timeout /t 6 /nobreak >nul

REM ── Step 4: Verify ───────────────────────────────────────────────────────
echo Verifying Ollama is running...
ollama list
echo.
echo Expected layer split: ~24/49 on GPU (this is correct for 18GB model on 12GB VRAM)
echo Expected inference profile: 30-40pct GPU, 50-60pct CPU — this is normal.
echo.

REM ── Step 5: Launch trading scheduler ─────────────────────────────────────
echo  Starting MES Paper Trading Scheduler...
echo  Waiting until 9:30 AM ET then trading begins.
echo  Press CTRL+C to stop at any time.
echo.

call "C:\Users\spenc\Documents\trading-ai\.venv\Scripts\activate.bat"
python start_trading.py

echo.
echo  Scheduler exited.
pause
