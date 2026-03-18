"""
test_model.py — Trainer training and prediction tests.
"""

import pytest
import pandas as pd
import numpy as np

from app.ml.trainer import Trainer
from app.data.loader import load_sample


def _make_df(rows: int = 300) -> pd.DataFrame:
    np.random.seed(7)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=rows, freq="min"),
        "open":      np.random.rand(rows) * 10 + 5700,
        "high":      np.random.rand(rows) * 10 + 5710,
        "low":       np.random.rand(rows) * 10 + 5690,
        "close":     np.random.rand(rows) * 10 + 5700,
        "volume":    np.random.randint(100, 500, rows),
    })


def test_trainer_train_predict():
    """Trainer must return a fitted model that can predict."""
    df       = _make_df(300)
    trainer  = Trainer()
    model    = trainer.train(df)

    assert model is not None, "train() returned None"
    assert hasattr(model, "predict"), "Model has no predict() method"
    assert hasattr(model, "predict_proba"), "Model has no predict_proba() method"


def test_trainer_with_load_sample():
    """Trainer must work with the standard load_sample() data source."""
    df = load_sample()
    # Ensure we have enough rows
    if len(df) < 200:
        repeats = (200 // len(df)) + 1
        df = pd.concat([df] * repeats, ignore_index=True)
        df["timestamp"] = pd.date_range("2025-01-01", periods=len(df), freq="min")

    trainer = Trainer()
    model   = trainer.train(df)
    assert model is not None


def test_trainer_predict_proba_shape():
    """predict_proba output must be 2-D with 2 columns (binary classifier)."""
    df      = _make_df(300)
    trainer = Trainer()
    model   = trainer.train(df)

    from app.ml.features import make_features
    feat = make_features(df)
    X    = feat.drop(
        columns=["timestamp", "open", "high", "low", "close", "volume"],
        errors="ignore",
    ).select_dtypes(include=[float, int])

    proba = model.predict_proba(X.iloc[:5])
    assert proba.shape[1] == 2, f"Expected 2 probability columns, got {proba.shape[1]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
