# File: tests/test_trainer.py
import os
import pandas as pd
import numpy as np
import joblib
import json
import pytest
from app.ml.trainer import Trainer

@pytest.fixture
def sample_data():
    """Create synthetic OHLCV-like DataFrame."""
    np.random.seed(42)
    rows = 300
    return pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=rows, freq="T"),
        "open": np.random.rand(rows) * 100,
        "high": np.random.rand(rows) * 100,
        "low": np.random.rand(rows) * 100,
        "close": np.random.rand(rows) * 100,
        "volume": np.random.randint(100, 1000, rows)
    })

@pytest.fixture
def trainer(tmp_path):
    """Initialize Trainer with a temporary directory."""
    model_path = tmp_path / "model.pkl"
    return Trainer(model_path=str(model_path))

def test_train_and_save(sample_data, trainer):
    """Ensure training produces a valid model and metadata file."""
    model = trainer.train(sample_data)
    assert model is not None, "Model should not be None after training"

    # Check model file
    assert os.path.exists(trainer.model_path), "Model file not found after training"
    assert os.path.getsize(trainer.model_path) > 0, "Model file appears empty"

    # Check metadata
    meta_file = os.path.splitext(trainer.model_path)[0] + "_meta.json"
    assert os.path.exists(meta_file), "Metadata JSON not found"
    with open(meta_file, "r") as f:
        meta = json.load(f)
    assert "accuracy" in meta
    assert meta["model_type"] == "RandomForestClassifier"

def test_load_model(sample_data, trainer):
    """Trainer should load an existing model correctly."""
    trainer.train(sample_data)
    loaded = trainer.load()
    assert loaded is not None
    assert hasattr(loaded, "predict"), "Loaded object is not a valid sklearn model"

def test_evaluate_accuracy(sample_data, trainer):
    """Ensure evaluation produces a valid accuracy score."""
    trainer.train(sample_data)
    acc = trainer.evaluate(sample_data)
    assert 0 <= acc <= 1, "Accuracy should be within valid bounds"

def test_empty_dataframe(trainer):
    """Trainer should handle empty DataFrame gracefully."""
    df = pd.DataFrame()
    result = trainer.train(df)
    assert result is None, "Empty DataFrame should not produce a model"

def test_corrupted_model_load(tmp_path):
    """Trainer should handle model load errors gracefully."""
    fake_model_path = tmp_path / "corrupt.pkl"
    fake_model_path.write_bytes(b"not a valid joblib file")

    trainer = Trainer(model_path=str(fake_model_path))
    model = trainer.load()
    assert model is None, "Corrupt model load should return None without raising"