"""
Module: system_config.py
Author: Trading-AI System
Description:
    Adaptive system configuration based on:
    - Market hours (trading sessions)
    - CPU/RAM/GPU utilization
    - Hardware capabilities (7800X3D + RTX 4070 Super)
    
    Dynamically adjusts worker counts and resource allocation.
"""

import os
import psutil
from datetime import datetime, time as dt_time
from typing import Dict, Literal
from dataclasses import dataclass
import pytz

from ..monitor.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# MARKET HOURS CONFIGURATION
# ============================================================

# US Futures Market Hours (Sunday 6PM ET - Friday 5PM ET)
FUTURES_OPEN_SUNDAY = dt_time(18, 0)  # 6:00 PM
FUTURES_CLOSE_FRIDAY = dt_time(17, 0)  # 5:00 PM

# US Equity Market Hours
EQUITY_OPEN = dt_time(9, 30)   # 9:30 AM
EQUITY_CLOSE = dt_time(16, 0)  # 4:00 PM


@dataclass
class ResourceLimits:
    """Resource allocation limits."""
    news_api_workers: int
    llm_batch_workers: int
    db_pool_size: int
    max_headlines_buffer: int
    llm_cache_size_mb: int
    feature_buffer_size: int


class SystemConfig:
    """
    Adaptive system configuration manager.
    
    Features:
    - Market hours detection
    - Dynamic resource allocation
    - Hardware-optimized settings
    - CPU/GPU load monitoring
    """
    
    def __init__(
        self,
        timezone: str = "America/New_York",
        cpu_threshold: float = 70.0,
        memory_threshold: float = 80.0
    ):
        self.tz = pytz.timezone(timezone)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        
        # Hardware detection
        self._cpu_count = psutil.cpu_count(logical=True)
        self._total_ram_gb = psutil.virtual_memory().total / (1024**3)
        
        logger.info(
            f"SystemConfig initialized: "
            f"CPUs={self._cpu_count}, "
            f"RAM={self._total_ram_gb:.1f}GB, "
            f"Timezone={timezone}"
        )
    
    # --------------------------------------------------------
    # MARKET HOURS DETECTION
    # --------------------------------------------------------
    
    def is_market_hours(
        self,
        market: Literal["futures", "equity"] = "futures"
    ) -> bool:
        """
        Check if market is currently open.
        
        Args:
            market: "futures" (ES/NQ 24/5) or "equity" (RTH only)
        
        Returns:
            True if market is actively trading
        """
        now = datetime.now(self.tz)
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        current_time = now.time()
        
        if market == "futures":
            # Futures trade Sunday 6PM - Friday 5PM ET
            
            # Friday after 5PM - closed
            if weekday == 4 and current_time >= FUTURES_CLOSE_FRIDAY:
                return False
            
            # Saturday - closed
            if weekday == 5:
                return False
            
            # Sunday before 6PM - closed
            if weekday == 6 and current_time < FUTURES_OPEN_SUNDAY:
                return False
            
            # All other times - open
            return True
        
        elif market == "equity":
            # Equity RTH: Mon-Fri 9:30 AM - 4:00 PM ET
            
            # Weekend
            if weekday >= 5:
                return False
            
            # Check time window
            return EQUITY_OPEN <= current_time <= EQUITY_CLOSE
        
        else:
            raise ValueError(f"Unknown market type: {market}")
    
    def get_market_session(self) -> str:
        """
        Determine current market session.
        
        Returns:
            "pre_market", "market_hours", "after_hours", "closed"
        """
        now = datetime.now(self.tz)
        weekday = now.weekday()
        current_time = now.time()
        
        # Weekend
        if weekday == 5 or (weekday == 6 and current_time < FUTURES_OPEN_SUNDAY):
            return "closed"
        
        # Weekday - check session
        if weekday < 5:  # Monday-Friday
            if current_time < EQUITY_OPEN:
                return "pre_market"
            elif EQUITY_OPEN <= current_time <= EQUITY_CLOSE:
                return "market_hours"
            else:
                return "after_hours"
        
        # Sunday evening - futures open
        return "after_hours"
    
    # --------------------------------------------------------
    # SYSTEM RESOURCE MONITORING
    # --------------------------------------------------------
    
    def get_cpu_load(self) -> float:
        """Get current CPU utilization (%)."""
        return psutil.cpu_percent(interval=0.5)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics.
        
        Returns:
            Dict with total, available, used, percent
        """
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_gb": mem.used / (1024**3),
            "percent": mem.percent
        }
    
    def is_system_stressed(self) -> bool:
        """
        Check if system is under high load.
        
        Returns:
            True if CPU or memory exceeds thresholds
        """
        cpu_load = self.get_cpu_load()
        mem_usage = self.get_memory_usage()
        
        return (
            cpu_load > self.cpu_threshold or
            mem_usage["percent"] > self.memory_threshold
        )
    
    # --------------------------------------------------------
    # ADAPTIVE RESOURCE ALLOCATION
    # --------------------------------------------------------
    
    def get_resource_limits(self) -> ResourceLimits:
        """
        Calculate optimal resource limits based on current state.
        
        Returns:
            ResourceLimits with adaptive worker counts
        """
        session = self.get_market_session()
        cpu_load = self.get_cpu_load()
        mem_usage = self.get_memory_usage()
        
        # Base configuration for 7800X3D (8C/16T)
        if session == "market_hours":
            # CONSERVATIVE - prioritize trading
            limits = ResourceLimits(
                news_api_workers=8,       # Reduce API calls
                llm_batch_workers=0,      # NO GPU usage during market hours
                db_pool_size=5,
                max_headlines_buffer=1000,
                llm_cache_size_mb=512,
                feature_buffer_size=2000
            )
        
        elif session == "after_hours":
            # MODERATE - balance analysis and trading prep
            limits = ResourceLimits(
                news_api_workers=12,
                llm_batch_workers=6,      # GPU available
                db_pool_size=8,
                max_headlines_buffer=5000,
                llm_cache_size_mb=1536,
                feature_buffer_size=5000
            )
        
        else:  # pre_market or closed
            # AGGRESSIVE - full analysis mode
            limits = ResourceLimits(
                news_api_workers=16,
                llm_batch_workers=8,      # Full GPU utilization
                db_pool_size=10,
                max_headlines_buffer=10000,
                llm_cache_size_mb=2048,
                feature_buffer_size=10000
            )
        
        # Adjust for system stress
        if cpu_load > 75:
            logger.warning(f"High CPU load ({cpu_load:.1f}%) - reducing workers")
            limits.news_api_workers = max(4, limits.news_api_workers // 2)
            limits.llm_batch_workers = max(2, limits.llm_batch_workers // 2)
        
        if mem_usage["available_gb"] < 4.0:
            logger.warning(
                f"Low memory ({mem_usage['available_gb']:.1f}GB free) - "
                f"reducing buffers"
            )
            limits.max_headlines_buffer = min(
                limits.max_headlines_buffer,
                5000
            )
            limits.llm_cache_size_mb = min(limits.llm_cache_size_mb, 1024)
        
        return limits
    
    def get_gpu_mode(self) -> Literal["trading", "analysis", "training"]:
        """
        Determine GPU usage priority.
        
        Returns:
            "trading" - GPU reserved for trading models
            "analysis" - GPU available for sentiment analysis
            "training" - GPU available for model retraining
        """
        session = self.get_market_session()
        
        if session == "market_hours":
            return "trading"
        elif session in ("pre_market", "after_hours"):
            return "analysis"
        else:  # closed
            return "training"
    
    def should_defer_analysis(self) -> bool:
        """
        Check if LLM analysis should be deferred.
        
        Returns:
            True if analysis should wait
        """
        return (
            self.is_market_hours(market="futures") or
            self.is_system_stressed()
        )
    
    # --------------------------------------------------------
    # CONFIGURATION PRESETS
    # --------------------------------------------------------
    
    def get_config_summary(self) -> Dict:
        """
        Get current configuration summary for logging.
        
        Returns:
            Dict with all relevant config parameters
        """
        limits = self.get_resource_limits()
        
        return {
            "market_session": self.get_market_session(),
            "gpu_mode": self.get_gpu_mode(),
            "cpu_load": round(self.get_cpu_load(), 1),
            "memory_usage": self.get_memory_usage(),
            "resource_limits": {
                "news_workers": limits.news_api_workers,
                "llm_workers": limits.llm_batch_workers,
                "db_pool": limits.db_pool_size,
                "headline_buffer": limits.max_headlines_buffer
            },
            "defer_analysis": self.should_defer_analysis()
        }


# ============================================================
# GLOBAL SINGLETON
# ============================================================

_system_config_instance: SystemConfig = None # type: ignore


def get_system_config() -> SystemConfig:
    """Get global system config instance (singleton)."""
    global _system_config_instance
    
    if _system_config_instance is None:
        _system_config_instance = SystemConfig()
    
    return _system_config_instance


# Convenience export
system_config = get_system_config()
