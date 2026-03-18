"""
test_backtester.py — Backtester integration test.
"""

import pytest
import pandas as pd
import numpy as np

from app.backtest.backtester import Backtester
from app.data.loader import load_sample
from app.config import load_config


def _make_sufficient_df(base_df: pd.DataFrame, min_rows: int = 300) -> pd.DataFrame:
    """Repeat base_df until we have at least min_rows rows."""
    repeats = (min_rows // len(base_df)) + 1
    df = pd.concat([base_df] * repeats, ignore_index=True)
    # Re-stamp timestamps so they're monotonically increasing
    df["timestamp"] = pd.date_range("2025-01-01", periods=len(df), freq="min")
    return df.iloc[:min_rows].copy()


def test_backtester_runs():
    cfg = load_config()
    df  = load_sample()
    df  = _make_sufficient_df(df, min_rows=300)

    bt  = Backtester(cfg)
    res = bt.run(df)

    assert "win_rate"     in res, "Missing win_rate in results"
    assert "trades"       in res, "Missing trades in results"
    assert "equity_curve" in res, "Missing equity_curve in results"
    assert isinstance(res["trades"],       list)
    assert isinstance(res["equity_curve"], list)
    assert len(res["equity_curve"]) == len(res["trades"]) + 1


def test_backtester_win_rate_range():
    cfg = load_config()
    df  = _make_sufficient_df(load_sample(), 300)
    res = Backtester(cfg).run(df)
    assert 0.0 <= res["win_rate"] <= 1.0


def test_backtester_insufficient_data():
    cfg = load_config()
    df  = load_sample().iloc[:50]   # too few rows
    with pytest.raises(ValueError, match="Not enough data"):
        Backtester(cfg).run(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
