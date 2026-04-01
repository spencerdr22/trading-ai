"""
Package: app.llm
Description: LLM-powered market intelligence using Qwen3 via Ollama.

Model selection rationale:
  qwen3:8b (5.2GB) is used instead of qwen3:30b-a3b-q4_K_M (18GB) because:
  - Fits entirely in RTX 4070 Super VRAM (12GB) with 4GB to spare
  - Full GPU inference: 2-5 seconds per headline vs 13-55 seconds for 30B
  - Sentiment classification accuracy is near-identical for financial headlines
  - The extra parameters in 30B help with complex reasoning/code, not
    single-sentence sentiment labels
  - A 20-headline batch completes in <2 minutes instead of 10-15 minutes

  To revert to 30B: change DEFAULT_MODEL and FALLBACK_MODEL below,
  then restart via start_trading.bat.

GPU profile with qwen3:8b on RTX 4070 Super:
  - All 36 layers on GPU (full VRAM fit)
  - Expected: 70-90% GPU, 5-15% CPU during inference
  - VRAM used: ~5.2GB model + ~0.5GB KV cache = ~5.7GB of 12GB
"""

import os
import ollama
from ..monitor.logger import get_logger

__version__ = "1.0.0"
logger = get_logger(__name__)

DEFAULT_MODEL  = "qwen3:8b"
FALLBACK_MODEL = "qwen3:14b"

# Expected GPU% range for full-GPU inference with qwen3:8b
_GPU_UTIL_MIN = 60.0
_GPU_UTIL_MAX = 95.0


def verify_ollama_gpu() -> bool:
    """
    Verify Ollama is running and a usable model is available.
    Logs the environment variables Ollama actually reads.
    """
    try:
        models_response = ollama.list()

        if hasattr(models_response, "models"):
            available_models = [m.model for m in models_response.models]
        elif isinstance(models_response, dict):
            available_models = [
                m.get("name", m.get("model", ""))
                for m in models_response.get("models", [])
            ]
        else:
            available_models = [str(m) for m in models_response]

        logger.info("Ollama available models: %s", available_models)

        if DEFAULT_MODEL not in available_models:
            if FALLBACK_MODEL in available_models:
                logger.warning(
                    "Model %s not found — using fallback %s. "
                    "Pull preferred model with: ollama pull %s",
                    DEFAULT_MODEL, FALLBACK_MODEL, DEFAULT_MODEL,
                )
            else:
                logger.warning(
                    "Neither %s nor %s found. Run: ollama pull %s",
                    DEFAULT_MODEL, FALLBACK_MODEL, DEFAULT_MODEL,
                )
                return False

        # Log the variables Ollama 0.18.3 actually reads
        flash_attn = os.environ.get("OLLAMA_FLASH_ATTENTION", "NOT SET")
        overhead   = os.environ.get("OLLAMA_GPU_OVERHEAD",    "NOT SET")
        parallel   = os.environ.get("OLLAMA_NUM_PARALLEL",    "NOT SET")
        keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE",      "NOT SET")

        logger.info(
            "Ollama env — FLASH_ATTENTION=%s  GPU_OVERHEAD=%s  "
            "NUM_PARALLEL=%s  KEEP_ALIVE=%s",
            flash_attn, overhead, parallel, keep_alive,
        )

        if flash_attn == "NOT SET":
            logger.warning(
                "OLLAMA_FLASH_ATTENTION not set. "
                "Restart via start_trading.bat to apply it."
            )

        logger.info(
            "Model: %s — expected full GPU inference "
            "(5.2GB fits in 12GB VRAM with 4GB headroom). "
            "Expected profile: 70-90pct GPU, 5-15pct CPU.",
            DEFAULT_MODEL,
        )

        test_model = DEFAULT_MODEL if DEFAULT_MODEL in available_models else FALLBACK_MODEL
        ollama.chat(
            model    = test_model,
            messages = [{"role": "user", "content": "Test"}],
        )
        logger.info("Ollama verification passed. Using model: %s", test_model)
        return True

    except Exception as e:
        logger.error("Ollama verification failed: %s", e)
        logger.info("Ensure Ollama is running via start_trading.bat")
        return False


def check_gpu_utilisation() -> dict:
    """
    Sample GPU utilisation.
    With qwen3:8b fully on GPU, expect 70-90% during inference.
    Warns if GPU is below 50% (suggests model not fully loaded on GPU).
    """
    result = {"gpu_pct": None, "cpu_pct": None, "within_range": None}
    try:
        import psutil
        result["cpu_pct"] = psutil.cpu_percent(interval=0.5)

        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
            result["gpu_pct"]      = util.gpu
            result["within_range"] = _GPU_UTIL_MIN <= util.gpu <= _GPU_UTIL_MAX

            if util.gpu < 20.0:
                logger.warning(
                    "GPU utilisation very low (%.0f%%). "
                    "Model may not be loaded. Check Ollama is running.",
                    util.gpu,
                )
            elif util.gpu < 50.0:
                logger.warning(
                    "GPU utilisation lower than expected (%.0f%%) for qwen3:8b. "
                    "Model may be partially on CPU — check Ollama loaded correctly.",
                    util.gpu,
                )
            else:
                logger.info(
                    "GPU utilisation: %.0f%% GPU / %.0f%% CPU (full GPU inference)",
                    util.gpu, result["cpu_pct"],
                )
        except ImportError:
            logger.debug("pynvml not available — skipping GPU check.")
        except Exception as e:
            logger.debug("GPU check failed: %s", e)

    except Exception as e:
        logger.debug("CPU check failed: %s", e)

    return result


def init_llm() -> bool:
    ok = verify_ollama_gpu()
    if ok:
        check_gpu_utilisation()
    return ok


__all__ = [
    "init_llm",
    "verify_ollama_gpu",
    "check_gpu_utilisation",
    "DEFAULT_MODEL",
    "FALLBACK_MODEL",
]
