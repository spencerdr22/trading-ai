"""
test_integration.py — Integration tests for walk-forward and Monte Carlo modules.
"""

import pytest
import numpy as np
import pandas as pd

from app.validation.walk_forward import walk_forward_split
from app.validation.monte_carlo import bootstrap_trades, monte_carlo_analysis


def test_walk_forward_split_anchored():
    splits = walk_forward_split(n_samples=1000, n_splits=5, mode="anchored")
    assert len(splits) == 5
    for train, test in splits:
        # No overlap between train and test indices
        assert len(set(train) & set(test)) == 0


def test_walk_forward_split_rolling():
    splits = walk_forward_split(n_samples=1000, n_splits=5, mode="rolling")
    assert len(splits) > 0
    for train, test in splits:
        assert len(set(train) & set(test)) == 0


def test_monte_carlo_bootstrap_shape():
    np.random.seed(42)
    trades_df = pd.DataFrame({"R": np.random.randn(100)})
    sequences = bootstrap_trades(trades_df, n_sequences=10, seed=42)

    assert len(sequences) == 10
    for seq in sequences:
        assert len(seq) == 100
        assert "R" in seq.columns
        assert "cumulative_R" in seq.columns


def test_monte_carlo_bootstrap_block():
    np.random.seed(0)
    trades_df = pd.DataFrame({"R": np.random.randn(50)})
    sequences = bootstrap_trades(trades_df, n_sequences=5, block_size=10, seed=0)
    assert len(sequences) == 5


def test_monte_carlo_analysis_returns_dict():
    np.random.seed(1)
    trades_df = pd.DataFrame({"R": np.random.randn(100)})
    results   = monte_carlo_analysis(trades_df, n_sequences=20)

    assert "summary_metrics"    in results
    assert "equity_curves"      in results
    assert "stability_assessment" in results
    assert results["n_sequences"] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
