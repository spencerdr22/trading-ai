"""
test_db_persistence.py — DB write/read tests using context-manager session.
"""

import pytest
from datetime import datetime, timezone

from app.db.init import get_engine, get_session
from app.models.schema import Base, StrategyParam, TradeMetric


def test_persist_strategy_params():
    """StrategyParam rows must survive a round-trip through the DB."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)

    with get_session() as s:
        s.add(StrategyParam(name="threshold_up", value=0.7, reason="test"))
        s.commit()

    with get_session() as s:
        count = s.query(StrategyParam).filter_by(name="threshold_up").count()
        assert count >= 1, "StrategyParam row not found after insert"


def test_persist_trade_metric():
    """TradeMetric rows must survive a round-trip through the DB."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)

    with get_session() as s:
        s.add(TradeMetric(
            symbol    = "MES",
            timestamp = datetime.now(timezone.utc),
            side      = {"side": "BUY", "confidence": 0.8},
            pnl       = 5.0,
            status    = "FILLED",
        ))
        s.commit()

    with get_session() as s:
        row = s.query(TradeMetric).filter_by(symbol="MES").first()
        assert row is not None, "TradeMetric row not found after insert"
        assert row.pnl == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
