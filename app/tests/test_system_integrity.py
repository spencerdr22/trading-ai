"""
File: tests/test_system_integrity.py
Purpose:
    Automated verification of Trading-AI system integrity:
    - Database schema creation
    - Model training & persistence
    - Adaptive ModelHub functionality
    - Core module imports

Usage:
    pytest -v tests/test_system_integrity.py
    OR
    python tests/test_system_integrity.py
"""

import os
import pandas as pd
import numpy as np
import traceback
from pathlib import Path

# Core system imports
from app.db.init import get_engine, get_session
from app.db.schema import Base, MarketData
from app.ml.trainer import Trainer
from app.adaptive.model_hub import ModelHub
from app.monitor.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# TEST 1 — Database Connectivity
# ============================================================

def test_database_connection():
    """Ensure the database engine connects and tables are creatable."""
    try:
        engine = get_engine()
        Base.metadata.create_all(engine)
        logger.info("✅ Database connection OK and tables verified.")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        traceback.print_exc()
        raise


# ============================================================
# TEST 2 — Mock Data Insertion
# ============================================================

def test_mock_data_insertion():
    """Insert and fetch mock market data to verify ORM operation."""
    try:
        session = get_session()
        md = MarketData(symbol="ES", open=100, high=105, low=99, close=103, volume=5000)
        session.add(md)
        session.commit()
        result = session.query(MarketData).filter_by(symbol="ES").first()
        assert result is not None
        logger.info("✅ Mock data insertion verified.")
    except Exception as e:
        logger.error(f"❌ ORM test failed: {e}")
        traceback.print_exc()
        raise
    finally:
        session.close()


# ============================================================
# TEST 3 — Model Training (Trainer)
# ============================================================

def test_model_training(tmp_path: Path=Path("data/models/test_model.pkl")):
    """Train a simple RandomForest model on synthetic data."""
    try:
        # Create fake OHLCV data
        np.random.seed(42)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=200, freq="min"),
            "open": np.random.rand(200) * 100,
            "high": np.random.rand(200) * 100,
            "low": np.random.rand(200) * 100,
            "close": np.random.rand(200) * 100,
            "volume": np.random.rand(200) * 1000
        })

        trainer = Trainer(model_path=str(tmp_path))
        model = trainer.train(df)

        assert model is not None
        assert os.path.exists(tmp_path)
        logger.info("✅ Model training and persistence test passed.")
    except Exception as e:
        logger.error(f"❌ Model training test failed: {e}")
        traceback.print_exc()
        raise


# ============================================================
# TEST 4 — ModelHub Save/Load
# ============================================================

def test_modelhub_operations():
    """Verify saving and loading models via ModelHub."""
    try:
        mh = ModelHub()
        dummy_model = {"param": 123}
        path = mh.save_model(dummy_model, "test_model", "RandomForest", metrics={"accuracy": 0.95})
        assert path and os.path.exists(path)

        loaded = mh.load_model("test_model")
        assert loaded is not None
        logger.info("✅ ModelHub save/load cycle complete.")
    except Exception as e:
        logger.error(f"❌ ModelHub test failed: {e}")
        traceback.print_exc()
        raise


# ============================================================
# TEST 5 — Module Import Sanity
# ============================================================

def test_module_imports():
    """Ensure critical modules import cleanly."""
    try:
        import app.strategy.engine
        # import app.visual.dashboard  # TODO: Uncomment once dashboard module is implemented
        import app.monitor.logger
        logger.info("✅ Core modules import without error.")
    except Exception as e:
        logger.error(f"❌ Module import test failed: {e}")
        traceback.print_exc()
        raise


# ============================================================
# ENTRYPOINT FOR DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    print("🚀 Running system integrity checks...\n")
    for test_func in [
        test_database_connection,
        test_mock_data_insertion,
        test_model_training,
        test_modelhub_operations,
        test_module_imports,
    ]:
        print(f"Running {test_func.__name__} ...")
        test_func()
        print("OK\n")

    print("✅ All tests completed successfully.")
