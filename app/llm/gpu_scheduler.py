"""
Module: gpu_scheduler.py
Author: Trading-AI System
Description:
    GPU resource scheduler to prevent contention between:
    - Trading models (LSTM/RL) - HIGH PRIORITY
    - LLM sentiment analysis (Qwen3) - LOW PRIORITY
    
    Ensures trading decisions are never delayed by background analysis.
"""

import threading
import time
from contextlib import contextmanager
from enum import Enum
from typing import Optional
from ..monitor.logger import get_logger

logger = get_logger(__name__)


class GPUTask(Enum):
    """GPU task priority levels."""
    TRADING = 1      # Highest priority - never block
    ANALYSIS = 2     # Lower priority - can be deferred
    TRAINING = 3     # Lowest priority - background only


class GPUScheduler:
    """
    Thread-safe GPU resource scheduler.
    
    Features:
    - Priority-based access control
    - Non-blocking acquisition for low-priority tasks
    - Metrics tracking (wait times, conflicts)
    - Automatic fallback to CPU for deferred tasks
    """
    
    def __init__(self):
        self._lock = threading.RLock()  # Reentrant lock
        self._current_task: Optional[GPUTask] = None
        self._task_start_time: Optional[float] = None
        
        # Metrics
        self._trading_calls = 0
        self._analysis_calls = 0
        self._analysis_deferred = 0
        self._conflicts = 0
        self._total_wait_time = 0.0
        
        logger.info("✅ GPU Scheduler initialized")
    
    @contextmanager
    def trading_inference(self, timeout: float = 2.0):
        """
        Acquire GPU for trading model inference (HIGHEST PRIORITY).
        
        Args:
            timeout: Maximum wait time in seconds
        
        Yields:
            True if GPU acquired, False if timeout
        
        Raises:
            RuntimeError: If GPU cannot be acquired within timeout
        """
        start_time = time.time()
        acquired = False
        
        try:
            acquired = self._lock.acquire(blocking=True, timeout=timeout)
            
            if not acquired:
                elapsed = time.time() - start_time
                logger.error(
                    f"⚠️ GPU TIMEOUT for trading inference after {elapsed:.2f}s"
                )
                raise RuntimeError(
                    "GPU scheduler timeout - trading model blocked by background task"
                )
            
            # Track metrics
            wait_time = time.time() - start_time
            self._trading_calls += 1
            self._total_wait_time += wait_time
            
            if wait_time > 0.1:  # Log if we had to wait
                logger.warning(
                    f"⏱️ Trading inference waited {wait_time*1000:.1f}ms for GPU"
                )
            
            # Mark GPU as busy with trading task
            self._current_task = GPUTask.TRADING
            self._task_start_time = time.time()
            
            yield True
        
        finally:
            if acquired:
                self._current_task = None
                self._task_start_time = None
                self._lock.release()
    
    @contextmanager
    def analysis_inference(self, timeout: float = 0.0):
        """
        Acquire GPU for LLM sentiment analysis (LOWER PRIORITY).
        
        Args:
            timeout: Wait time (default 0 = non-blocking)
        
        Yields:
            True if GPU acquired, False if busy (should fallback to CPU)
        """
        start_time = time.time()
        acquired = False
        
        try:
            # Non-blocking by default - don't interfere with trading
            acquired = self._lock.acquire(blocking=(timeout > 0), timeout=timeout)
            
            if not acquired:
                self._analysis_deferred += 1
                
                if self._current_task == GPUTask.TRADING:
                    self._conflicts += 1
                    logger.debug(
                        "⏭️ LLM analysis deferred - GPU busy with trading model"
                    )
                
                # Return False to signal fallback to CPU
                yield False
                return
            
            # GPU acquired
            self._analysis_calls += 1
            self._current_task = GPUTask.ANALYSIS
            self._task_start_time = time.time()
            
            yield True
        
        finally:
            if acquired:
                self._current_task = None
                self._task_start_time = None
                self._lock.release()
    
    @contextmanager
    def training_inference(self, timeout: float = 60.0):
        """
        Acquire GPU for model training (BACKGROUND ONLY).
        
        Args:
            timeout: Maximum wait time (default 60s)
        
        Yields:
            True if GPU acquired within timeout
        """
        start_time = time.time()
        acquired = False
        
        try:
            acquired = self._lock.acquire(blocking=True, timeout=timeout)
            
            if not acquired:
                logger.warning(
                    f"⏭️ Training deferred - GPU busy for {timeout:.0f}s"
                )
                yield False
                return
            
            wait_time = time.time() - start_time
            if wait_time > 5.0:
                logger.info(
                    f"⏱️ Training waited {wait_time:.1f}s for GPU availability"
                )
            
            self._current_task = GPUTask.TRAINING
            self._task_start_time = time.time()
            
            yield True
        
        finally:
            if acquired:
                self._current_task = None
                self._task_start_time = None
                self._lock.release()
    
    def get_metrics(self) -> dict:
        """
        Get GPU scheduler performance metrics.
        
        Returns:
            Dict with usage statistics
        """
        return {
            "trading_calls": self._trading_calls,
            "analysis_calls": self._analysis_calls,
            "analysis_deferred": self._analysis_deferred,
            "conflicts": self._conflicts,
            "avg_wait_time_ms": (
                (self._total_wait_time / self._trading_calls * 1000)
                if self._trading_calls > 0 else 0
            ),
            "deferral_rate": (
                self._analysis_deferred / 
                (self._analysis_calls + self._analysis_deferred)
                if (self._analysis_calls + self._analysis_deferred) > 0 
                else 0
            ),
            "current_task": self._current_task.name if self._current_task else "IDLE"
        }
    
    def reset_metrics(self):
        """Reset all metrics counters."""
        self._trading_calls = 0
        self._analysis_calls = 0
        self._analysis_deferred = 0
        self._conflicts = 0
        self._total_wait_time = 0.0
        logger.info("📊 GPU scheduler metrics reset")
    
    def is_gpu_available(self) -> bool:
        """
        Check if GPU is currently available (non-blocking).
        
        Returns:
            True if GPU is idle
        """
        return self._current_task is None
    
    def force_release(self):
        """
        Emergency GPU release (use only for cleanup/shutdown).
        """
        try:
            if self._lock.locked():
                self._lock.release()
                logger.warning("⚠️ GPU scheduler force-released")
        except RuntimeError:
            pass  # Lock wasn't held


# ============================================================
# GLOBAL SINGLETON
# ============================================================

_gpu_scheduler_instance: Optional[GPUScheduler] = None


def get_gpu_scheduler() -> GPUScheduler:
    """
    Get global GPU scheduler instance (thread-safe singleton).
    """
    global _gpu_scheduler_instance
    
    if _gpu_scheduler_instance is None:
        _gpu_scheduler_instance = GPUScheduler()
    
    return _gpu_scheduler_instance


# Convenience export
gpu_scheduler = get_gpu_scheduler()
