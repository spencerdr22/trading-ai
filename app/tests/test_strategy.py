from app.strategy.adaption import Adaptor

def test_adaptor_improves_or_changes():
    a = Adaptor()
    fake_trades = [{"pnl": 1}, {"pnl": -2}, {"pnl": 3}, {"pnl": -1}]
    before = (a.threshold_up, a.stop_loss)
    out = a.update(fake_trades)
    assert "threshold_up" in out and "stop_loss" in out
    assert isinstance(out["win_rate"], float)
