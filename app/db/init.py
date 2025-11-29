"""
File: app/db/init.py
Description:
    Initializes and manages the SQLAlchemy engine and sessions
    for the Trading-AI system. Supports PostgreSQL and SQLite.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.monitor.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# Database Configuration
# ============================================================

def get_database_url() -> str:
    """
    Fetches DATABASE_URL from environment or defaults to local SQLite.
    """
    db_url = os.getenv(
        "DATABASE_URL",
        "sqlite:///data/trading_ai.db"
    )
    if db_url.startswith("postgres"):
        logger.info(f"DATABASE_URL: {db_url}")
    else:
        logger.info("DATABASE_URL not set. Defaulting to SQLite at data/trading_ai.db")
    return db_url


# ============================================================
# Engine and Session Initialization
# ============================================================

_engine = None
_SessionFactory = None


def get_engine():
    """
    Returns the SQLAlchemy engine singleton.
    """
    global _engine
    if _engine is None:
        db_url = get_database_url()
        connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
        _engine = create_engine(db_url, echo=False, future=True, connect_args=connect_args)
        logger.info("Database engine initialized.")
    return _engine


def get_session():
    """
    Returns a new session instance for interacting with the database.
    """
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    return _SessionFactory()


# ============================================================
# CLI Entry (optional)
# ============================================================

if __name__ == "__main__":
    engine = get_engine()
    logger.info(f"Engine ready → {engine}")
