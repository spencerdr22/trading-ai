"""
Module: system_config.py
Author: Trading-AI System
Description:
    Adaptive system configuration based on:
    - Market hours (trading sessions)
    - CPU/RAM/GPU utilization
    - Hardware capabilities (7800X3D + RTX 4070 Super 12GB VRAM)

    GPU is the PRIMARY compute resource.
    LLM sentiment (qwen3:8b via Ollama) runs on GPU at ALL times during market hours.
    CPU handles API workers, DB, and lightweight tasks only.
"""

import os
import psutil
from datetime import datetime, time as dt_time
from typing import Dict, Literal, Optional
from dataclasses import dataclass, field
import pytz

from ..monitor.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# HARDWARE CONSTANTS — RTX 4070 Super + 7800X3D
# ============================================================

GPU_VRAM_TOTAL_GB = 12.0          # RTX 4070 Super
GPU_VRAM_LLM_RESERVED_GB = 5.5   # qwen3:8b needs ~5.2GB — keep headroom
GPU_VRAM_TRADING_RESERVED_GB = 1.0  # PyTorch inference (RF is CPU, but LSTM if used)
GPU_VRAM_SAFETY_MARGIN_GB = 1.0   # Don't cut it close

CPU_CORE_COUNT = 16               # 7800X3D logical (8C/16T)
RAM_TOTAL_GB = 32.0               # DDR5-6400

# ============================================================
# MARKET HOURS CONFIGURATION
# ============================================================

FUTURES_OPEN_SUNDAY = dt_time(18, 0)   # 6:00 PM ET
FUTURES_CLOSE_FRIDAY = dt_time(17, 0)  # 5:00 PM ET
EQUITY_OPEN = dt_time(9, 30)
EQUITY_CLOSE = dt_time(16, 0)


@dataclass
class GPUStatus:
    """Live GPU state snapshot."""
    available: bool = False
    vram_total_gb: float = GPU_VRAM_TOTAL_GB
    vram_used_gb: float = 0.0
    vram_free_gb: float = GPU_VRAM_TOTAL_GB
    utilization_pct: float = 0.0
    llm_can_run: bool = True
    pytorch_can_run: bool = True
    temperature_c: Optional[float] = None


@dataclass
class ResourceLimits:
    """Resource allocation limits — GPU-first."""
    news_api_workers: int
    llm_batch_workers: int      # LLM workers ON GPU (Ollama qwen3:8b)
    db_pool_size: int
    max_headlines_buffer: int
    llm_cache_size_mb: int
    feature_buffer_size: int
    gpu_llm_enabled: bool = True
    gpu_pytorch_enabled: bool = True


class SystemConfig:
    """
    GPU-first adaptive system configuration manager.

    Design principle:
        - GPU runs LLM sentiment (qwen3:8b via Ollama) at ALL times during trading.
        - CPU handles news API workers, DB connections, feature computation.
        - GPU utilization is monitored live; LLM is throttled only if VRAM is critically low.
        - should_defer_analysis() returns False during market hours — we WANT sentiment live.
    """

    def __init__(
        self,
        timezone: str = "America/New_York",
        cpu_threshold: float = 85.0,
        memory_threshold: float = 85.0,
        vram_low_threshold_gb: float = 2.0,
    ):
        self.tz = pytz.timezone(timezone)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.vram_low_threshold_gb = vram_low_threshold_gb

        self._cpu_count = psutil.cpu_count(logical=True)
        self._total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)

        # Try importing pynvml for live GPU stats
        self._nvml_available = self._init_nvml()

        logger.info(
            f"SystemConfig initialized: "
            f"CPUs={self._cpu_count}, "
            f"RAM={self._total_ram_gb:.1f}GB, "
            f"GPU_NVML={'YES' if self._nvml_available else 'NO (fallback mode)'}, "
            f"Timezone={timezone}"
        )

    # --------------------------------------------------------
    # GPU MONITORING
    # --------------------------------------------------------

    def _init_nvml(self) -> bool:
        """Try to initialise pynvml for live VRAM/utilisation queries."""
        try:
            import pynvml
            pynvml.nvmlInit()
            logger.info("pynvml initialised — live GPU monitoring active.")
            return True
        except Exception:
            logger.warning(
                "pynvml not available — GPU stats will use PyTorch fallback. "
                "Run: pip install pynvml --break-system-packages"
            )
            return False

    def get_gpu_status(self) -> GPUStatus:
        """
        Query live GPU state.

        Priority order:
          1. pynvml (most accurate, reads NVML directly)
          2. torch.cuda (available if PyTorch is loaded)
          3. Assume GPU is healthy with defaults (safest fallback)
        """
        status = GPUStatus(available=True)

        # --- pynvml path ---
        if self._nvml_available:
            try:
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)

                status.vram_total_gb = mem_info.total / (1024 ** 3)
                status.vram_used_gb = mem_info.used / (1024 ** 3)
                status.vram_free_gb = mem_info.free / (1024 ** 3)
                status.utilization_pct = float(util_info.gpu)

                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    status.temperature_c = float(temp)
                except Exception:
                    pass

                status.available = True
                status.llm_can_run = status.vram_free_gb >= self.vram_low_threshold_gb
                status.pytorch_can_run = status.vram_free_gb >= 1.0
                return status

            except Exception as e:
                logger.warning(f"pynvml query failed: {e} — falling back to torch")

        # --- PyTorch path ---
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                total = props.total_memory / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                free = total - reserved

                status.vram_total_gb = total
                status.vram_used_gb = allocated
                status.vram_free_gb = free
                status.utilization_pct = 0.0  # torch can't read GPU util
                status.available = True
                status.llm_can_run = free >= self.vram_low_threshold_gb
                status.pytorch_can_run = free >= 1.0
                return status
        except Exception as e:
            logger.warning(f"torch.cuda query failed: {e} — using defaults")

        # --- Safe fallback: assume GPU is fine ---
        # This prevents the system from incorrectly disabling LLM
        # when monitoring is simply unavailable.
        status.available = True
        status.vram_free_gb = GPU_VRAM_TOTAL_GB - GPU_VRAM_LLM_RESERVED_GB
        status.llm_can_run = True
        status.pytorch_can_run = True
        return status

    def is_gpu_vram_stressed(self) -> bool:
        """True only if VRAM is critically low (< vram_low_threshold_gb free)."""
        gpu = self.get_gpu_status()
        return gpu.available and gpu.vram_free_gb < self.vram_low_threshold_gb

    # --------------------------------------------------------
    # MARKET HOURS DETECTION
    # --------------------------------------------------------

    def is_market_hours(
        self,
        market: Literal["futures", "equity"] = "futures"
    ) -> bool:
        now = datetime.now(self.tz)
        weekday = now.weekday()
        current_time = now.time()

        if market == "futures":
            if weekday == 4 and current_time >= FUTURES_CLOSE_FRIDAY:
                return False
            if weekday == 5:
                return False
            if weekday == 6 and current_time < FUTURES_OPEN_SUNDAY:
                return False
            return True

        elif market == "equity":
            if weekday >= 5:
                return False
            return EQUITY_OPEN <= current_time <= EQUITY_CLOSE

        raise ValueError(f"Unknown market type: {market}")

    def get_market_session(self) -> str:
        now = datetime.now(self.tz)
        weekday = now.weekday()
        current_time = now.time()

        if weekday == 5 or (weekday == 6 and current_time < FUTURES_OPEN_SUNDAY):
            return "closed"

        if weekday < 5:
            if current_time < EQUITY_OPEN:
                return "pre_market"
            elif EQUITY_OPEN <= current_time <= EQUITY_CLOSE:
                return "market_hours"
            else:
                return "after_hours"

        return "after_hours"  # Sunday evening (futures open)

    # --------------------------------------------------------
    # SYSTEM RESOURCE MONITORING
    # --------------------------------------------------------

    def get_cpu_load(self) -> float:
        return psutil.cpu_percent(interval=0.5)

    def get_memory_usage(self) -> Dict[str, float]:
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024 ** 3),
            "available_gb": mem.available / (1024 ** 3),
            "used_gb": mem.used / (1024 ** 3),
            "percent": mem.percent,
        }

    def is_system_stressed(self) -> bool:
        """CPU or RAM over threshold — does NOT include GPU in this check."""
        cpu_load = self.get_cpu_load()
        mem_usage = self.get_memory_usage()
        return (
            cpu_load > self.cpu_threshold
            or mem_usage["percent"] > self.memory_threshold
        )

    # --------------------------------------------------------
    # GPU MODE
    # --------------------------------------------------------

    def get_gpu_mode(self) -> Literal["trading", "analysis", "training"]:
        """
        GPU priority label.

        During market_hours: GPU runs LLM sentiment AND trading inference.
        During after_hours / pre_market: GPU emphasises sentiment + analysis.
        During closed: GPU available for full model retraining.

        NOTE: In practice qwen3:8b (5.2GB) fits alongside PyTorch inference
        on the RTX 4070 Super's 12GB, so no session requires GPU exclusivity.
        """
        session = self.get_market_session()
        if session == "market_hours":
            return "trading"
        elif session in ("pre_market", "after_hours"):
            return "analysis"
        return "training"

    # --------------------------------------------------------
    # ADAPTIVE RESOURCE ALLOCATION
    # --------------------------------------------------------

    def get_resource_limits(self) -> ResourceLimits:
        """
        GPU-first resource allocation.

        Key decisions:
        - llm_batch_workers is NEVER 0 during market hours.
          The LLM (qwen3:8b) runs on GPU and must process headlines continuously.
        - CPU workers scale with session and CPU load.
        - VRAM stress is the only reason to reduce llm_batch_workers.
        """
        session = self.get_market_session()
        cpu_load = self.get_cpu_load()
        mem_usage = self.get_memory_usage()
        gpu = self.get_gpu_status()

        if session == "market_hours":
            # GPU: LLM running, PyTorch inference as needed
            # CPU: Moderate workers — don't saturate cores during active trading
            limits = ResourceLimits(
                news_api_workers=10,
                llm_batch_workers=4,       # GPU — qwen3:8b processes headlines live
                db_pool_size=6,
                max_headlines_buffer=2000,
                llm_cache_size_mb=768,
                feature_buffer_size=3000,
                gpu_llm_enabled=True,
                gpu_pytorch_enabled=True,
            )

        elif session == "after_hours":
            # GPU: Full LLM + analysis batch processing
            limits = ResourceLimits(
                news_api_workers=14,
                llm_batch_workers=6,
                db_pool_size=8,
                max_headlines_buffer=5000,
                llm_cache_size_mb=1536,
                feature_buffer_size=5000,
                gpu_llm_enabled=True,
                gpu_pytorch_enabled=True,
            )

        else:  # pre_market or closed
            # GPU: Full utilisation — batch analysis + retraining possible
            limits = ResourceLimits(
                news_api_workers=16,
                llm_batch_workers=8,
                db_pool_size=10,
                max_headlines_buffer=10000,
                llm_cache_size_mb=2048,
                feature_buffer_size=10000,
                gpu_llm_enabled=True,
                gpu_pytorch_enabled=True,
            )

        # --- CPU stress adjustment (CPU only, not GPU) ---
        if cpu_load > 80:
            logger.warning(f"High CPU load ({cpu_load:.1f}%) - reducing CPU-side workers")
            limits.news_api_workers = max(4, limits.news_api_workers // 2)
            # Do NOT reduce llm_batch_workers here — those run on GPU, not CPU

        # --- RAM stress adjustment ---
        if mem_usage["available_gb"] < 4.0:
            logger.warning(
                f"Low RAM ({mem_usage['available_gb']:.1f}GB free) - reducing buffers"
            )
            limits.max_headlines_buffer = min(limits.max_headlines_buffer, 3000)
            limits.llm_cache_size_mb = min(limits.llm_cache_size_mb, 512)

        # --- VRAM stress adjustment (only real GPU throttle) ---
        if not gpu.llm_can_run:
            logger.warning(
                f"VRAM critically low ({gpu.vram_free_gb:.1f}GB free) - "
                f"reducing LLM workers"
            )
            limits.llm_batch_workers = max(1, limits.llm_batch_workers // 2)
            limits.gpu_llm_enabled = gpu.vram_free_gb > 0.5

        if not gpu.pytorch_can_run:
            logger.warning("VRAM too low for PyTorch — disabling GPU inference")
            limits.gpu_pytorch_enabled = False

        return limits

    # --------------------------------------------------------
    # DEFERRED ANALYSIS — GPU-FIRST POLICY
    # --------------------------------------------------------

    def should_defer_analysis(self) -> bool:
        """
        Whether to defer LLM sentiment analysis.

        GPU-first policy: analysis is NEVER deferred purely because futures
        are open. We want live sentiment during market hours.

        Deferral only happens if VRAM is critically low.
        """
        if self.is_gpu_vram_stressed():
            logger.warning("Deferring LLM analysis due to low VRAM.")
            return True
        return False

    # --------------------------------------------------------
    # CONFIG SUMMARY
    # --------------------------------------------------------

    def get_config_summary(self) -> Dict:
        limits = self.get_resource_limits()
        gpu = self.get_gpu_status()

        return {
            "market_session": self.get_market_session(),
            "gpu_mode": self.get_gpu_mode(),
            "cpu_load": round(self.get_cpu_load(), 1),
            "memory_usage": self.get_memory_usage(),
            "gpu_status": {
                "available": gpu.available,
                "vram_total_gb": round(gpu.vram_total_gb, 2),
                "vram_used_gb": round(gpu.vram_used_gb, 2),
                "vram_free_gb": round(gpu.vram_free_gb, 2),
                "utilization_pct": gpu.utilization_pct,
                "temperature_c": gpu.temperature_c,
                "llm_can_run": gpu.llm_can_run,
                "pytorch_can_run": gpu.pytorch_can_run,
            },
            "resource_limits": {
                "news_workers": limits.news_api_workers,
                "llm_workers": limits.llm_batch_workers,
                "db_pool": limits.db_pool_size,
                "headline_buffer": limits.max_headlines_buffer,
                "gpu_llm_enabled": limits.gpu_llm_enabled,
                "gpu_pytorch_enabled": limits.gpu_pytorch_enabled,
            },
            "defer_analysis": self.should_defer_analysis(),
        }


# ============================================================
# GLOBAL SINGLETON
# ============================================================

_system_config_instance: Optional[SystemConfig] = None


def get_system_config() -> SystemConfig:
    """Get global system config instance (singleton)."""
    global _system_config_instance
    if _system_config_instance is None:
        _system_config_instance = SystemConfig()
    return _system_config_instance


system_config = get_system_config()
