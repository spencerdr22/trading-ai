"""
Real-time monitoring for GPU scheduler and system resources.
"""

from datetime import datetime
from .gpu_scheduler import gpu_scheduler
from .system_config import system_config
from ..monitor.logger import get_logger

logger = get_logger(__name__)


def log_system_status():
    """
    Log current system status (for APScheduler integration).
    """
    # GPU scheduler metrics
    gpu_metrics = gpu_scheduler.get_metrics()
    
    # System config
    config_summary = system_config.get_config_summary()
    
    logger.info(
        f"📊 SYSTEM STATUS | "
        f"Session: {config_summary['market_session']} | "
        f"GPU: {config_summary['gpu_mode']} | "
        f"CPU: {config_summary['cpu_load']}% | "
        f"RAM: {config_summary['memory_usage']['percent']:.1f}% | "
        f"Trading calls: {gpu_metrics['trading_calls']} | "
        f"Analysis deferred: {gpu_metrics['analysis_deferred']} | "
        f"Avg wait: {gpu_metrics['avg_wait_time_ms']:.2f}ms"
    )


def get_performance_report() -> dict:
    """
    Generate performance report for dashboard.
    
    Returns:
        Dict with comprehensive system metrics
    """
    return {
        "gpu_scheduler": gpu_scheduler.get_metrics(),
        "system_config": system_config.get_config_summary(),
        "timestamp": datetime.utcnow().isoformat()
    }
