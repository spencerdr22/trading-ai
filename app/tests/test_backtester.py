from ..backtest.backtester import Backtester
from ..data.loader import load_sample
from ..config import load_config

def test_backtester_runs():
    cfg = load_config()
    df = load_sample()
    bt = Backtester(cfg)
    res = bt.run(df.append([df]*30, ignore_index=True))  # ensure enough bars
    assert "win_rate" in res
    assert isinstance(res["trades"], list)
    assert len(res["equity_curve"]) == len(res["trades"]) + 1
