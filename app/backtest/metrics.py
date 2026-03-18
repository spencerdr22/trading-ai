"""
metrics.py — Backtest performance metric helpers.
"""

import numpy as np
import pandas as pd


def compute_win_rate(trades: list) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return wins / len(trades)


def equity_curve_from_trades(
    trades: list,
    initial_capital: float = 10_000.0,
) -> list:
    """Build a cumulative equity list from a list of trade dicts."""
    eq = [initial_capital]
    for t in trades:
        pnl        = t.get("pnl", 0.0)
        commission = t.get("commission", 0.0)
        eq.append(eq[-1] + pnl - commission)
    return eq


def max_drawdown(equity_curve: list) -> float:
    """
    Return the maximum percentage drawdown (as a negative float, e.g. -0.15 = -15%).
    Handles zero and near-zero equity values safely.
    """
    arr  = np.array(equity_curve, dtype=float)
    if len(arr) < 2:
        return 0.0
    high = np.maximum.accumulate(arr)
    # Avoid division by zero — if high is 0, treat drawdown as 0
    safe_high = np.where(np.abs(high) < 1e-9, 1.0, high)
    dd        = (arr - high) / safe_high
    return float(dd.min())


def sharpe_minute(returns, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio from 1-minute return series."""
    arr  = np.asarray(returns, dtype=float)
    mean = np.mean(arr)
    std  = np.std(arr, ddof=1)
    if std == 0:
        return 0.0
    # ~390 trading minutes per day
    return float((mean - risk_free) / std * np.sqrt(390))


def sortino_minute(returns, risk_free: float = 0.0) -> float:
    """
    Annualised Sortino ratio from 1-minute return series.
    Only downside deviation is penalised.
    """
    arr      = np.asarray(returns, dtype=float)
    mean     = np.mean(arr)
    downside = arr[arr < risk_free]

    if len(downside) == 0:
        return float("inf")

    down_std = np.std(downside, ddof=1)
    if down_std == 0:
        return 0.0

    return float((mean - risk_free) / down_std * np.sqrt(390))
