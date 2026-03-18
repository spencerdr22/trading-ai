"""
app/db/schema.py — Legacy schema compatibility shim.

All ORM table definitions live in app/models/schema.py.
This module re-exports them so that any code using
`from app.db.schema import X` continues to work unchanged.
"""

# Re-export everything from the canonical schema
from app.models.schema import (   # noqa: F401
    Base,
    MarketData,
    Trade,
    Position,
    Metric,
    TradeMetric,
    StrategyParam,
)
