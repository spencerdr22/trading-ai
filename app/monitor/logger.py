"""
Unified logging configuration for Trading-AI System.
Compatible with pytest, Windows UTF-8, and file rotation.

FIX: LOG_DIR now uses an absolute path anchored to this file's location
so log files always land in the correct place regardless of the working
directory at the time get_logger() is first called.
"""

import logging
import sys
import io
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from logging.handlers import RotatingFileHandler

# Absolute path: <project_root>/data/logs
# __file__ is app/monitor/logger.py  →  go up two levels to project root
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_DIR = os.path.join(_PROJECT_ROOT, "data", "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a UTF-8-safe logger that writes to both console and a
    rotating file at data/logs/<name_with_dots_replaced>.log.

    Calling get_logger with the same name twice returns the cached
    logger (handlers are not duplicated).
    """
    is_pytest = any("pytest" in arg for arg in sys.argv)

    if not is_pytest:
        try:
            if not isinstance(sys.stdout, io.TextIOWrapper):
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer, encoding="utf-8", errors="replace"
                )
            if not isinstance(sys.stderr, io.TextIOWrapper):
                sys.stderr = io.TextIOWrapper(
                    sys.stderr.buffer, encoding="utf-8", errors="replace"
                )
        except Exception:
            pass

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File handler — absolute path so it never gets lost
    log_filename = name.replace(".", "_") + ".log"
    log_path = os.path.join(LOG_DIR, log_filename)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger
