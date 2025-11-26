"""
Module: reward.py
Description:
    Computes reward signals for the adaptive reinforcement-learning system.
    Combines raw profitability metrics with risk-adjusted performance factors,
    including Sharpe-like and Sortino-like ratios, volatility penalties, and
    tail-risk weighting.

Used by:
    - ReinforcementLearner (learner.py)
    - Nightly retraining pipeline
"""

import numpy as np


def _safe(val, eps=1e-9):
    """Prevent division by zero in risk metrics."""
    return val if abs(val) > eps else eps


def compute_sharpe(pnl_series):
    """
    Compute a Sharpe-like ratio using the mean and std dev of PnL returns.
    """
    returns = np.array(pnl_series)
    if len(returns) < 2:
        return 0.0

    mean_r = np.mean(returns)
    std_r = np.std(returns)
    return float(mean_r / _safe(std_r))


def compute_sortino(pnl_series):
    """
    Compute Sortino ratio using downside deviation only.
    """
    returns = np.array(pnl_series)
    if len(returns) < 2:
        return 0.0

    downside = np.std([r for r in returns if r < 0]) or 1e-9
    mean_r = np.mean(returns)
    return float(mean_r / downside)


def compute_drawdown(pnl_series):
    """
    Compute max drawdown from cumulative PnL series.
    """
    arr = np.array(pnl_series)
    if arr.size < 2:
        return 0.0

    cummax = np.maximum.accumulate(arr)
    drawdowns = (arr - cummax)
    max_dd = min(drawdowns)  # negative value
    return float(abs(max_dd))


def compute_reward(
    pnl_series,
    win_rate,
    leverage_factor=1.0,
    pnl_weight=0.5,
    sharpe_weight=0.2,
    sortino_weight=0.2,
    dd_penalty_weight=0.3,
    win_rate_weight=0.4
):
    """
    Compute the final reinforcement-learning reward signal.

    Inputs:
        pnl_series   — list/array of per-trade PnL values
        win_rate     — recent win percentage [0,1]
        leverage_factor — promotes lower leverage if volatile
        *_weight     — controls contribution of each metric

    Returns:
        A single float reward usable by the RL policy gradient update.
    """

    pnl_series = np.array(pnl_series, dtype=float)
    total_pnl = float(np.sum(pnl_series))

    # Risk-adjusted factors
    sharpe = compute_sharpe(pnl_series)
    sortino = compute_sortino(pnl_series)
    drawdown = compute_drawdown(pnl_series)

    # Weighted scoring
    reward = (
        (total_pnl * pnl_weight)
        + (sharpe * sharpe_weight)
        + (sortino * sortino_weight)
        + (win_rate * win_rate_weight)
        - (drawdown * dd_penalty_weight)
        - (abs(leverage_factor - 1.0) * 0.1)   # discourage excessive leverage
    )

    # Final clipping to keep gradient stable
    reward = float(np.clip(reward, -10.0, 10.0))

    return reward


# Convenience API for nightly batch processing
def compute_batch_reward(trade_pnls, win_rate):
    """
    Computes a reward for the entire trading session.
    """
    return compute_reward(
        pnl_series=trade_pnls,
        win_rate=win_rate,
        pnl_weight=0.5,
        sharpe_weight=0.2,
        sortino_weight=0.2,
        dd_penalty_weight=0.3,
        win_rate_weight=0.4,
    )
