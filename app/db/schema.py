"""
File: app/db/schema.py
Description:
    Defines SQLAlchemy ORM models for all Trading-AI database tables.
"""

import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# ============================================================
# Market Data
# ============================================================

class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    symbol = Column(String, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)


# ============================================================
# Trades
# ============================================================

class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    symbol = Column(String, nullable=False)
    side = Column(String)  # BUY or SELL
    quantity = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float)
    pnl = Column(Float)
    status = Column(String, default="closed")

    metrics = relationship("TradeMetric", back_populates="trade")


# ============================================================
# Trade Metrics
# ============================================================

class TradeMetric(Base):
    __tablename__ = "trade_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Integer, ForeignKey("trades.id"))
    pnl = Column(Float)
    reward = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    trade = relationship("Trade", back_populates="metrics")


# ============================================================
# Global Metrics
# ============================================================

class Metric(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    value = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


# ============================================================
# Positions
# ============================================================

class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String)
    side = Column(String)
    quantity = Column(Float)
    entry_price = Column(Float)
    current_price = Column(Float)
    unrealized_pnl = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


# ============================================================
# Model Registry (for reference)
# ============================================================

class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String)
    model_type = Column(String)
    version = Column(String)
    accuracy = Column(Float)
    reward_score = Column(Float)
    file_path = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    meta = Column(JSON)
