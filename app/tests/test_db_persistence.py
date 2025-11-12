from ..db import get_session, engine
from ..models.schema import Base, StrategyParam

def test_persist_strategy_params():
    Base.metadata.create_all(bind=engine)
    with get_session() as s:
        s.add(StrategyParam(name="threshold_up", value=0.7, reason="test"))
        s.commit()
        count = s.query(StrategyParam).count()
        assert count >= 1
