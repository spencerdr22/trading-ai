"""
Test GPU scheduling and adaptive configuration.
"""

import time
import threading
from .gpu_scheduler import gpu_scheduler
from .system_config import system_config
from ..monitor.logger import get_logger

logger = get_logger(__name__)


def simulate_trading_inference():
    """Simulate trading model using GPU."""
    for i in range(10):
        with gpu_scheduler.trading_inference(timeout=2.0) as acquired:
            if acquired:
                logger.info(f"[TRADING] Inference {i+1}/10 started")
                time.sleep(0.1)  # Simulate 100ms inference
                logger.info(f"[TRADING] Inference {i+1}/10 complete")
        
        time.sleep(0.2)  # 200ms between calls


def simulate_llm_analysis():
    """Simulate LLM sentiment analysis."""
    for i in range(5):
        with gpu_scheduler.analysis_inference(timeout=0.0) as acquired:
            if acquired:
                logger.info(f"[LLM] Analysis {i+1}/5 started (GPU)")
                time.sleep(2.0)  # Simulate 2s LLM inference
                logger.info(f"[LLM] Analysis {i+1}/5 complete")
            else:
                logger.warning(f"[LLM] Analysis {i+1}/5 DEFERRED (GPU busy)")
        
        time.sleep(1.0)


def test_concurrent_access():
    """Test concurrent GPU access from trading + LLM."""
    logger.info("="*60)
    logger.info("TESTING CONCURRENT GPU ACCESS")
    logger.info("="*60)
    
    # Start both tasks concurrently
    trading_thread = threading.Thread(target=simulate_trading_inference)
    llm_thread = threading.Thread(target=simulate_llm_analysis)
    
    trading_thread.start()
    time.sleep(0.5)  # Stagger start
    llm_thread.start()
    
    trading_thread.join()
    llm_thread.join()
    
    # Print metrics
    logger.info("="*60)
    logger.info("GPU SCHEDULER METRICS")
    logger.info("="*60)
    
    metrics = gpu_scheduler.get_metrics()
    for key, val in metrics.items():
        logger.info(f"{key:.<30} {val}")


def test_adaptive_config():
    """Test adaptive configuration based on market hours."""
    logger.info("="*60)
    logger.info("TESTING ADAPTIVE CONFIGURATION")
    logger.info("="*60)
    
    config_summary = system_config.get_config_summary()
    
    for key, val in config_summary.items():
        logger.info(f"{key:.<30} {val}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m app.llm.test_scheduling [gpu|config|both]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "gpu":
        test_concurrent_access()
    elif mode == "config":
        test_adaptive_config()
    elif mode == "both":
        test_adaptive_config()
        print("\n")
        test_concurrent_access()
