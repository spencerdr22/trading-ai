"""
Database package initializer for Trading-AI.
Exposes engine and session helpers for easy import.

get_engine() and get_session() are lazy — the engine is only
created on first call, not at import time.
"""

from app.db.init import get_engine, get_session

__all__ = ["get_engine", "get_session"]
