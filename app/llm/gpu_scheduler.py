"""
gpu_scheduler.py — Priority-based GPU resource scheduler.

Prevents contention between trading inference (HIGH) and LLM
sentiment analysis (LOW).  All emoji removed for Windows cp1252
compatibility.
"""

import threading
import time
from contextlib import contextmanager
from enum import Enum
from typing import Optional

from ..monitor.logger import get_logger

logger = get_logger(__name__)


class GPUTask(Enum):
    TRADING  = 1
    ANALYSIS = 2
    TRAINING = 3


class GPUScheduler:
    """Thread-safe GPU resource scheduler."""

    def __init__(self):
        self._lock             = threading.RLock()
        self._current_task: Optional[GPUTask] = None
        self._task_start_time: Optional[float] = None

        self._trading_calls      = 0
        self._analysis_calls     = 0
        self._analysis_deferred  = 0
        self._conflicts          = 0
        self._total_wait_time    = 0.0

        logger.info("GPU Scheduler initialized.")

    # ------------------------------------------------------------------
    @contextmanager
    def trading_inference(self, timeout: float = 2.0):
        """Acquire GPU for trading — highest priority, blocks up to timeout."""
        start    = time.time()
        acquired = False
        try:
            acquired = self._lock.acquire(blocking=True, timeout=timeout)
            if not acquired:
                elapsed = time.time() - start
                logger.error("GPU TIMEOUT for trading inference after %.2fs", elapsed)
                raise RuntimeError("GPU scheduler timeout — trading model blocked.")

            wait = time.time() - start
            self._trading_calls   += 1
            self._total_wait_time += wait
            if wait > 0.1:
                logger.warning("Trading inference waited %.1fms for GPU.", wait * 1000)

            self._current_task    = GPUTask.TRADING
            self._task_start_time = time.time()
            yield True
        finally:
            if acquired:
                self._current_task    = None
                self._task_start_time = None
                self._lock.release()

    # ------------------------------------------------------------------
    @contextmanager
    def analysis_inference(self, timeout: float = 0.0):
        """Acquire GPU for LLM analysis — non-blocking by default."""
        acquired = False
        try:
            if timeout > 0:
                acquired = self._lock.acquire(blocking=True, timeout=timeout)
            else:
                acquired = self._lock.acquire(blocking=False)

            if not acquired:
                self._analysis_deferred += 1
                if self._current_task == GPUTask.TRADING:
                    self._conflicts += 1
                    logger.debug("LLM analysis deferred — GPU busy with trading model.")
                yield False
                return

            self._analysis_calls  += 1
            self._current_task    = GPUTask.ANALYSIS
            self._task_start_time = time.time()
            yield True
        finally:
            if acquired:
                self._current_task    = None
                self._task_start_time = None
                self._lock.release()

    # ------------------------------------------------------------------
    @contextmanager
    def training_inference(self, timeout: float = 60.0):
        """Acquire GPU for background training — lowest priority."""
        start    = time.time()
        acquired = False
        try:
            acquired = self._lock.acquire(blocking=True, timeout=timeout)
            if not acquired:
                logger.warning("Training deferred — GPU busy for %.0fs.", timeout)
                yield False
                return

            wait = time.time() - start
            if wait > 5.0:
                logger.info("Training waited %.1fs for GPU availability.", wait)

            self._current_task    = GPUTask.TRAINING
            self._task_start_time = time.time()
            yield True
        finally:
            if acquired:
                self._current_task    = None
                self._task_start_time = None
                self._lock.release()

    # ------------------------------------------------------------------
    def get_metrics(self) -> dict:
        total_analysis = self._analysis_calls + self._analysis_deferred
        return {
            "trading_calls":      self._trading_calls,
            "analysis_calls":     self._analysis_calls,
            "analysis_deferred":  self._analysis_deferred,
            "conflicts":          self._conflicts,
            "avg_wait_time_ms":   (
                self._total_wait_time / self._trading_calls * 1000
                if self._trading_calls > 0 else 0.0
            ),
            "deferral_rate":      (
                self._analysis_deferred / total_analysis
                if total_analysis > 0 else 0.0
            ),
            "current_task": self._current_task.name if self._current_task else "IDLE",
        }

    def reset_metrics(self):
        self._trading_calls = self._analysis_calls = 0
        self._analysis_deferred = self._conflicts  = 0
        self._total_wait_time   = 0.0
        logger.info("GPU scheduler metrics reset.")

    def is_gpu_available(self) -> bool:
        return self._current_task is None

    def force_release(self):
        """Emergency release — use only during shutdown."""
        try:
            if self._lock.locked():
                self._lock.release()
                logger.warning("GPU scheduler force-released.")
        except RuntimeError:
            pass


# ── Global singleton ──────────────────────────────────────────────────────────

_instance: Optional[GPUScheduler] = None


def get_gpu_scheduler() -> GPUScheduler:
    global _instance
    if _instance is None:
        _instance = GPUScheduler()
    return _instance


gpu_scheduler = get_gpu_scheduler()
