from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, DateTime, Float, Integer, JSON, func
from typing import Optional

class Base(DeclarativeBase):
    pass

class MarketData(Base):
    __tablename__ = "market_data"
    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[DateTime] = mapped_column(DateTime(timezone=True), index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)

class Trade(Base):
    __tablename__ = "trades"
    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[DateTime] = mapped_column(DateTime(timezone=True), index=True)
    side: Mapped[str] = mapped_column(String)
    price: Mapped[float] = mapped_column(Float)
    size: Mapped[int] = mapped_column(Integer)
    pnl: Mapped[float] = mapped_column(Float)
    commission: Mapped[float] = mapped_column(Float)
    slippage: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String, default="FILLED")
    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

class Position(Base):
    __tablename__ = "positions"
    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String)
    size: Mapped[int] = mapped_column(Integer)
    avg_price: Mapped[float] = mapped_column(Float)
    opened_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)
    pnl: Mapped[float] = mapped_column(Float, default=0.0)

class Metric(Base):
    __tablename__ = "metrics"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    value: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

class TradeMetric(Base):
    __tablename__ = "trade_metrics"
    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[DateTime] = mapped_column(DateTime(timezone=True), index=True)
    # store 'side' as JSON to accept dicts like {"side":"LONG", "confidence":0.8}
    side: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    pnl: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String, default="FILLED")

class StrategyParam(Base):
    __tablename__ = "strategy_params"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    value: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    reason: Mapped[str] = mapped_column(String)
