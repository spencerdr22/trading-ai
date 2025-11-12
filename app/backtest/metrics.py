import numpy as np
import pandas as pd

def compute_win_rate(trades):
    if len(trades) == 0:
        return 0.0
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return wins / len(trades)

def equity_curve_from_trades(trades, initial_capital=10000.0):
    eq = [initial_capital]
    for t in trades:
        pnl = t.get("pnl", 0) - t.get("commission", 0)
        eq.append(eq[-1] + pnl)
    return eq

def max_drawdown(equity_curve):
    arr = np.array(equity_curve)
    high = np.maximum.accumulate(arr)
    dd = (arr - high) / high
    return float(dd.min())

def sharpe_minute(returns, risk_free=0.0):
    # returns is series of minute returns
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0
    # scale to daily-ish (approx 390 minutes per trading day)
    return (mean - risk_free) / std * np.sqrt(390)
