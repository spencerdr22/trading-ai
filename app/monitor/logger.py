"""
Unified logging configuration for Trading-AI System.
Compatible with pytest, Windows UTF-8, and file rotation.
"""

import logging
import sys
import io
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join("data", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "app.log")
    handler = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=3, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 🧩 Fix: Disable propagation to stdout
    root_logger.addHandler(handler)
    root_logger.propagate = False

def get_logger(name: str) -> logging.Logger:
    """
    Returns a UTF-8 safe logger that outputs to both console and rotating file.
    Automatically detects pytest and avoids stdout modification.
    """
    # Detect pytest context
    is_pytest = any("pytest" in arg for arg in sys.argv)

    # Avoid modifying stdout if pytest manages capture
    if not is_pytest:
        try:
            if not isinstance(sys.stdout, io.TextIOWrapper):
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            if not isinstance(sys.stderr, io.TextIOWrapper):
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console output (pytest-safe)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Rotating file handler (persistent logs)
    log_file = os.path.join(LOG_DIR, f"{name.replace('.', '_')}.log")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger
