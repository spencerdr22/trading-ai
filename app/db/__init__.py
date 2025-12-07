"""
Database package initializer for Trading-AI.
Exposes SQLAlchemy engine and session helpers for easy import.
"""

from app.db.init import get_engine, get_session
from sqlalchemy.orm import sessionmaker

# Initialize the shared engine and session factory
engine = get_engine()
SessionLocal = sessionmaker(bind=engine)

__all__ = ["engine", "get_engine", "get_session", "SessionLocal"]
