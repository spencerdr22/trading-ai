"""
test_trainer.py — Unit tests for the Trainer class.

Verifies training, evaluation, persistence, and ModelHub integration.
Uses the canonical Trainer from app.ml.trainer (not a local duplicate).
"""

import os
import logging
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from app.ml.trainer import Trainer
from app.adaptive.model_hub import ModelHub

logger = logging.getLogger(__name__)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(rows: int = 300) -> pd.DataFrame:
    np.random.seed(0)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=rows, freq="min"),
        "open":      np.random.rand(rows) * 10 + 5700,
        "high":      np.random.rand(rows) * 10 + 5710,
        "low":       np.random.rand(rows) * 10 + 5690,
        "close":     np.random.rand(rows) * 10 + 5700,
        "volume":    np.random.randint(100, 500, rows),
    })


MODEL_PATH = "data/models/test_trainer_model.pkl"


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_trainer_returns_model():
    """Trainer.train() must return a fitted sklearn model."""
    trainer = Trainer(model_path=MODEL_PATH)
    model   = trainer.train(_make_df())
    assert model is not None, "train() returned None"
    assert hasattr(model, "predict"), "Returned object has no predict()"


def test_trainer_saves_file():
    """Model file must exist on disk after training."""
    trainer = Trainer(model_path=MODEL_PATH)
    trainer.train(_make_df())
    assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"


def test_trainer_evaluate():
    """Trainer.evaluate() must return a float accuracy in [0, 1]."""
    trainer = Trainer(model_path=MODEL_PATH)
    trainer.train(_make_df())
    acc = trainer.evaluate(_make_df())
    assert 0.0 <= acc <= 1.0, f"Unexpected accuracy: {acc}"


def test_trainer_load():
    """Trainer.load() must return the persisted model."""
    trainer = Trainer(model_path=MODEL_PATH)
    trainer.train(_make_df())

    loader = Trainer(model_path=MODEL_PATH)
    model  = loader.load()
    assert model is not None, "load() returned None"
    assert hasattr(model, "predict"), "Loaded object has no predict()"


def test_trainer_empty_df():
    """Trainer.train() on an empty DataFrame must return None safely."""
    trainer = Trainer(model_path=MODEL_PATH)
    result  = trainer.train(pd.DataFrame())
    assert result is None, "Expected None for empty DataFrame"


def test_trainer_modelhub_integration():
    """ModelHub must contain a record after training."""
    trainer = Trainer(model_path=MODEL_PATH)
    trainer.train(_make_df())

    hub      = ModelHub()
    versions = hub.list_versions("supervised_rf")
    # Accept either the canonical name or any name containing 'rf'
    all_names = []
    try:
        from sqlalchemy.orm import Session
        from app.db.init import get_engine
        from app.adaptive.model_hub import ModelRegistry
        session = Session(get_engine())
        all_names = [r.model_name for r in session.query(ModelRegistry).all()]
        session.close()
    except Exception:
        pass

    rf_entries = [n for n in all_names if "rf" in n.lower() or "supervised" in n.lower()]
    assert len(rf_entries) > 0 or len(versions) > 0, \
        "No RF model found in ModelHub after training"


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
