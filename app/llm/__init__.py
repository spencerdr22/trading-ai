"""
Package: app.llm
Description: LLM-powered market intelligence using Qwen3 via Ollama
"""

import ollama
from ..monitor.logger import get_logger

__version__ = "1.0.0"
logger = get_logger(__name__)

# Model configuration - Use 30B for best quality
DEFAULT_MODEL = "qwen3:30b-a3b-q4_K_M"
FALLBACK_MODEL = "qwen3:14b"


def verify_ollama_gpu():
    """
    Verify Ollama is running and GPU is accessible.
    """
    try:
        models_response = ollama.list()
        
        # FIX: Handle both dict and ModelResponse object
        if hasattr(models_response, 'models'):
            # ModelResponse object
            available_models = [m.model for m in models_response.models]
        elif isinstance(models_response, dict):
            # Dict response
            model_list = models_response.get("models", [])
            available_models = [m.get("name", m.get("model", "")) for m in model_list]
        else:
            # Unknown format - try to iterate
            available_models = [str(m) for m in models_response]
        
        logger.info(f"Available Ollama models: {available_models}")
        
        if DEFAULT_MODEL not in available_models and FALLBACK_MODEL not in available_models:
            logger.warning(
                f"Recommended model not found. Available: {available_models}\n"
                f"Run: ollama pull {DEFAULT_MODEL}"
            )
            return False
        
        # Test inference
        test_model = DEFAULT_MODEL if DEFAULT_MODEL in available_models else FALLBACK_MODEL
        test = ollama.chat(
            model=test_model,
            messages=[{"role": "user", "content": "Test"}]
        )
        
        logger.info(f"✅ Ollama GPU verification passed. Using model: {test_model}")
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
    "verify_ollama_gpu",
    "DEFAULT_MODEL",
    "FALLBACK_MODEL"
]
