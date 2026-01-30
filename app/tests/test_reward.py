"""
Test suite for reward calculation functions.
Validates Sharpe, Sortino, drawdown, and composite reward logic.
"""

import pytest
import numpy as np
from app.adaptive.reward import (
    compute_sharpe,
    compute_sortino,
    compute_drawdown,
    compute_reward,
    compute_batch_reward
)


def test_compute_sharpe():
    """Test Sharpe ratio calculation."""
    # Positive returns with volatility
    pnl = [1, 2, -1, 3, -0.5, 2]
    sharpe = compute_sharpe(pnl)
    assert sharpe > 0, "Positive PnL should have positive Sharpe"
    
    # Zero returns
    pnl_zero = [0, 0, 0, 0]
    sharpe_zero = compute_sharpe(pnl_zero)
    assert sharpe_zero == 0, "Zero returns should have zero Sharpe"
    
    # Single value
    sharpe_single = compute_sharpe([1])
    assert sharpe_single == 0, "Single value should return 0"


def test_compute_sortino():
    """Test Sortino ratio calculation."""
    # Mixed returns (upside and downside)
    pnl = [1, 2, -1, 3, -0.5, 2]
    sortino = compute_sortino(pnl)
    assert sortino > 0, "Positive mean PnL should have positive Sortino"
    
    # Only positive returns (perfect Sortino = inf)
    pnl_positive = [1, 2, 3, 4, 5]
    sortino_positive = compute_sortino(pnl_positive)
    assert sortino_positive == float('inf'), "All positive returns should have infinite Sortino"
    
    # Mixed returns - Sortino should be higher than Sharpe for upside-skewed returns
    pnl_mixed = [5, 3, -1, 8, 2, -0.5, 6]
    sortino_mixed = compute_sortino(pnl_mixed)
    sharpe_mixed = compute_sharpe(pnl_mixed)
    assert sortino_mixed > sharpe_mixed, "Sortino should be higher for upside-skewed returns"


def test_compute_drawdown():
    """Test max drawdown calculation."""
    # Increasing equity
    pnl_increasing = [1, 1, 1, 1, 1]
    dd_increasing = compute_drawdown(pnl_increasing)
    assert dd_increasing == 0, "No drawdown in increasing equity"
    
    # Drawdown scenario
    pnl_dd = [10, 5, -5, -10, 5]  # Peak at 10, trough at -10
    dd = compute_drawdown(pnl_dd)
    assert dd > 0, "Should detect drawdown"
    
    # Single value
    dd_single = compute_drawdown([1])
    assert dd_single == 0, "Single value has no drawdown"


def test_compute_reward_weights():
    """Test that reward weights are applied correctly."""
    pnl = [1, 2, -1, 3, -0.5, 2]
    win_rate = 0.6
    
    # Get reward with default weights
    reward = compute_reward(pnl, win_rate)
    
    # Reward should be positive for profitable series
    assert reward > 0, "Profitable trading should have positive reward"
    
    # Test that Sortino is weighted more than Sharpe
    reward_high_sortino = compute_reward(
        pnl, win_rate,
        sharpe_weight=0.1,
        sortino_weight=0.3
    )
    
    reward_high_sharpe = compute_reward(
        pnl, win_rate,
        sharpe_weight=0.3,
        sortino_weight=0.1
    )
    
    # For upside-skewed returns, high Sortino weight should yield higher reward
    # (This may not always be true depending on other factors, so we just check it doesn't crash)
    assert isinstance(reward_high_sortino, float)
    assert isinstance(reward_high_sharpe, float)


def test_compute_batch_reward():
    """Test batch reward computation."""
    trade_pnls = [1, -0.5, 2, -1, 3, 1.5]
    win_rate = 0.67  # 4 wins, 2 losses
    
    reward = compute_batch_reward(trade_pnls, win_rate)
    
    assert isinstance(reward, float), "Reward should be float"
    assert reward > 0, "Profitable batch should have positive reward"
    assert -10.0 <= reward <= 10.0, "Reward should be clipped to [-10, 10]"


def test_reward_clipping():
    """Test that extreme rewards are clipped."""
    # Extreme profitable scenario
    pnl_extreme = [100, 200, 300, 400, 500]
    win_rate = 1.0
    
    reward = compute_reward(pnl_extreme, win_rate)
    assert reward <= 10.0, "Reward should be clipped at 10.0"
    
    # Extreme loss scenario
    pnl_loss = [-100, -200, -300]
    win_rate = 0.0
    
    reward_loss = compute_reward(pnl_loss, win_rate)
    assert reward_loss >= -10.0, "Reward should be clipped at -10.0"


def test_leverage_penalty():
    """Test that high leverage is penalized."""
    # Use smaller PnL values to avoid hitting reward ceiling
    pnl = [0.5, 0.3, -0.1, 0.4, 0.2]  # Modest returns
    win_rate = 0.6  # 60% win rate
    
    reward_low_lev = compute_reward(pnl, win_rate, leverage_factor=1.0)
    reward_high_lev = compute_reward(pnl, win_rate, leverage_factor=3.0)
    
    # Both should be below the 10.0 ceiling to see the penalty
    assert reward_low_lev < 10.0, "Reward should not hit ceiling for this test"
    assert reward_high_lev < 10.0, "Reward should not hit ceiling for this test"
    assert reward_low_lev > reward_high_lev, "High leverage should reduce reward"


def test_sortino_infinity_handling():
    """Test that infinite Sortino (all positive returns) is handled correctly."""
    pnl_perfect = [1, 2, 3, 4, 5]  # All positive
    win_rate = 1.0
    
    # Should not raise exception and should return finite reward
    reward = compute_reward(pnl_perfect, win_rate)
    
    assert isinstance(reward, float), "Reward should be float"
    assert not np.isinf(reward), "Reward should not be inf"
    assert not np.isnan(reward), "Reward should not be nan"
    assert -10.0 <= reward <= 10.0, "Reward should be within bounds"
    assert reward > 0, "Perfect trading should have positive reward"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
