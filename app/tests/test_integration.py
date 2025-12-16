"""
Integration tests for ml_project_scaffold features.
"""

import pytest
import pandas as pd
import numpy as np
from app.validation.walk_forward import walk_forward_split
from app.validation.monte_carlo import bootstrap_trades


def test_walk_forward_split():
    """Test walk-forward split generation."""
    splits = walk_forward_split(n_samples=1000, n_splits=5, mode='anchored')
    
    assert len(splits) == 5
    for train, test in splits:
        assert len(set(train) & set(test)) == 0


def test_monte_carlo_bootstrap():
    """Test Monte Carlo bootstrap sampling."""
    np.random.seed(42)
    trades_df = pd.DataFrame({'R': np.random.randn(100)})
    
    sequences = bootstrap_trades(trades_df, n_sequences=10, seed=42)
    
    assert len(sequences) == 10
    assert all(len(seq) == 100 for seq in sequences)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
