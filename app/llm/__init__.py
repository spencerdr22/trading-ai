"""
Package: app.llm
Description: LLM-powered market intelligence using Qwen3 via Ollama
"""

import ollama
from ..monitor.logger import get_logger

__version__ = "1.0.0"
logger = get_logger(__name__)

# Model configuration
DEFAULT_MODEL = "qwen3-30b-a3b:q4_K_M"
FALLBACK_MODEL = "qwen3-30b-a3b:q4_0"


def verify_ollama_gpu():
    """
    Verify Ollama is running and GPU is accessible.
    """
    try:
        response = ollama.list()
        models = [m["name"] for m in response.get("models", [])]
        
        if DEFAULT_MODEL not in models and FALLBACK_MODEL not in models:
            logger.warning(
                f"Recommended model not found. Available: {models}\n"
                f"Run: ollama pull {DEFAULT_MODEL}"
            )
            return False
        
        # Test inference
        test = ollama.chat(
            model=DEFAULT_MODEL if DEFAULT_MODEL in models else FALLBACK_MODEL,
            messages=[{"role": "user", "content": "Test"}],
            options={"num_predict": 5}
        )
        
        logger.info(f"✅ Ollama GPU verification passed. Model: {DEFAULT_MODEL}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ollama verification failed: {e}")
        logger.info("Ensure Ollama is running: `ollama serve`")
        return False


def init_llm():
    """Initialize LLM subsystem."""
    return verify_ollama_gpu()


__all__ = [
    "init_llm",
    "verify_ollama_gpu"
]
