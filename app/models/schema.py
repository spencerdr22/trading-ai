"""
app/models/schema.py — Canonical ORM schema for Trading-AI.

All tables are defined here.  app/db/schema.py re-exports from this
module for backward compatibility.
"""

from typing import Optional
from sqlalchemy import String, DateTime, Float, Integer, JSON, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ── Market data ───────────────────────────────────────────────────────────────

class MarketData(Base):
    __tablename__ = "market_data"

    id:        Mapped[int]   = mapped_column(primary_key=True)
    symbol:    Mapped[str]   = mapped_column(String, index=True)
    timestamp: Mapped[DateTime] = mapped_column(DateTime(timezone=True), index=True)
    open:      Mapped[float] = mapped_column(Float)
    high:      Mapped[float] = mapped_column(Float)
    low:       Mapped[float] = mapped_column(Float)
    close:     Mapped[float] = mapped_column(Float)
    volume:    Mapped[float] = mapped_column(Float)


# ── Trades ────────────────────────────────────────────────────────────────────

class Trade(Base):
    """
    Full trade record.

    Columns cover both the backtester (entry_price, exit_price, quantity)
    and the paper executor (price, size, commission, slippage).
    Optional columns are nullable so either path can write without error.
    """
    __tablename__ = "trades"

    id:          Mapped[int]            = mapped_column(primary_key=True)
    symbol:      Mapped[str]            = mapped_column(String, index=True)
    timestamp:   Mapped[DateTime]       = mapped_column(DateTime(timezone=True), index=True)
    side:        Mapped[str]            = mapped_column(String)           # BUY / SELL
    # Backtester columns
    quantity:    Mapped[Optional[float]]= mapped_column(Float,   nullable=True)
    entry_price: Mapped[Optional[float]]= mapped_column(Float,   nullable=True)
    exit_price:  Mapped[Optional[float]]= mapped_column(Float,   nullable=True)
    # Paper executor columns
    price:       Mapped[Optional[float]]= mapped_column(Float,   nullable=True)
    size:        Mapped[Optional[int]]  = mapped_column(Integer, nullable=True)
    commission:  Mapped[Optional[float]]= mapped_column(Float,   nullable=True)
    slippage:    Mapped[Optional[float]]= mapped_column(Float,   nullable=True)
    # Common
    pnl:         Mapped[float]          = mapped_column(Float,   default=0.0)
    status:      Mapped[str]            = mapped_column(String,  default="FILLED")
    meta:        Mapped[Optional[dict]] = mapped_column(JSON,    nullable=True)


# ── Positions ─────────────────────────────────────────────────────────────────

class Position(Base):
    __tablename__ = "positions"

    id:        Mapped[int]            = mapped_column(primary_key=True)
    symbol:    Mapped[str]            = mapped_column(String)
    size:      Mapped[int]            = mapped_column(Integer)
    avg_price: Mapped[float]          = mapped_column(Float)
    opened_at: Mapped[DateTime]       = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)
    pnl:       Mapped[float]          = mapped_column(Float, default=0.0)


# ── Metrics ───────────────────────────────────────────────────────────────────

class Metric(Base):
    __tablename__ = "metrics"

    id:        Mapped[int]            = mapped_column(primary_key=True)
    name:      Mapped[str]            = mapped_column(String)
    value:     Mapped[float]          = mapped_column(Float)
    timestamp: Mapped[DateTime]       = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    meta:      Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


# ── Trade metrics (forward/paper loop) ───────────────────────────────────────

class TradeMetric(Base):
    __tablename__ = "trade_metrics"

    id:        Mapped[int]            = mapped_column(primary_key=True)
    symbol:    Mapped[str]            = mapped_column(String, index=True)
    timestamp: Mapped[DateTime]       = mapped_column(DateTime(timezone=True), index=True)
    side:      Mapped[Optional[dict]] = mapped_column(JSON,   nullable=True)
    pnl:       Mapped[float]          = mapped_column(Float,  default=0.0)
    status:    Mapped[str]            = mapped_column(String, default="FILLED")


# ── Strategy parameters ───────────────────────────────────────────────────────

class StrategyParam(Base):
    __tablename__ = "strategy_params"

    id:         Mapped[int]      = mapped_column(primary_key=True)
    name:       Mapped[str]      = mapped_column(String)
    value:      Mapped[float]    = mapped_column(Float)
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    reason:     Mapped[str]      = mapped_column(String)
